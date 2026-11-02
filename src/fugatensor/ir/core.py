"""Structured Fuga SSA, isolated regions and verification.

This module describes tensor programs. It does not allocate tensors, launch a
kernel, import ASK IR objects, or compile an external DSL.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import re
import struct
from types import MappingProxyType

from .types import ScalarType, TensorType, Type, as_type, constant_value

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)
_TERMINATORS = {"return", "yield"}


@dataclass(frozen=True)
class _StructuralKey:
    value: tuple
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, '_hash', hash(self.value))

    def __hash__(self):
        return self._hash


def _identifier(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("Fuga symbols and argument names must be ASCII identifiers")
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("IR attribute names must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (bool, int, float, str) or isinstance(value, (ScalarType, TensorType, ExternalSpec, CallSpec)):
        return value
    raise TypeError(f"unsupported Fuga attribute: {type(value).__name__}")


def _effects(values):
    if isinstance(values, str):
        raise TypeError("effects must be a sequence of names, not one string")
    values = tuple(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("effects must be nonempty string names; () means pure")
    return tuple(sorted(set(values)))


@dataclass(frozen=True, eq=False)
class ExternalSpec:
    compiler: str
    source: str
    entry: str
    signature: Mapping
    source_bindings: Mapping
    constants: Mapping = field(default_factory=dict)
    options: Mapping = field(default_factory=dict)
    target: Mapping = field(default_factory=dict)
    grid: tuple[int, int, int] = (1, 1, 1)
    shape_rule: str = "same_as"
    shape_operand: int = 0
    effects: tuple[str, ...] = ("unknown",)

    def __post_init__(self):
        _identifier(self.compiler)
        _identifier(self.entry)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("external source must be a nonempty string")
        for field_ in ("signature", "source_bindings", "constants", "options", "target"):
            value = getattr(self, field_)
            if not isinstance(value, Mapping):
                raise TypeError(f"external {field_} must be a mapping")
            object.__setattr__(self, field_, _freeze(value))
        if set(self.signature) != set(self.source_bindings):
            raise ValueError("external signature and source bindings must name the same parameters")
        if any(not isinstance(value, str) for value in self.signature.values()):
            raise TypeError("external signature types must be strings")
        if any(type(value) not in (str, bool, int, float)
               or (type(value) is float and not math.isfinite(value)) for value in self.source_bindings.values()):
            raise TypeError("external source bindings require port names or finite scalar values")
        if self.shape_rule not in ("same_as", "explicit"):
            raise ValueError("external shape_rule must be same_as or explicit")
        if type(self.shape_operand) is not int or self.shape_operand < 0:
            raise ValueError("external shape_operand must be a nonnegative integer")
        grid = tuple(self.grid)
        if len(grid) != 3 or any(type(n) is not int or not 0 < n < 2**32 for n in grid):
            raise ValueError("external grid must contain three positive 32-bit integers")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "effects", _effects(self.effects))


@dataclass(frozen=True, eq=False)
class CallSpec:
    """A custom differentiation boundary around an isolated forward graph.

    Executable Python is only a tracing callback, never serialized as tensor
    computation. The printable descriptor contains the forward program and a
    stable rule identity. Saved residuals are ordinary hidden SSA results.
    """
    name: str
    forward: Module
    output_count: int
    backward_rule: object = field(default=None, repr=False)
    backward_name: str | None = None
    definition_key: str = ""
    input_types: tuple = field(init=False)
    result_types: tuple = field(init=False)
    effects: tuple = field(init=False)
    _forward_key: _StructuralKey = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("custom calls require a nonempty name")
        if not isinstance(self.forward, Module) or len(self.forward.functions) != 1:
            raise TypeError("custom calls require one isolated Fuga forward function")
        self.forward.verify()
        function = self.forward.functions[0]
        if type(self.output_count) is not int or not 0 < self.output_count <= len(function.outputs):
            raise ValueError("custom call output_count must name its visible forward results")
        if self.backward_rule is not None and not callable(self.backward_rule):
            raise TypeError("custom backward rule must be a tracing callback")
        if self.backward_rule is not None and (not isinstance(self.backward_name, str) or not self.backward_name):
            raise ValueError("custom backward rules require a stable printable name")
        if not isinstance(self.definition_key, str):
            raise TypeError("custom call definition_key must be a string")
        object.__setattr__(self, "input_types", tuple(value.type for value in function.inputs.values()))
        object.__setattr__(self, "result_types", tuple(value.type for value in function.outputs))
        object.__setattr__(self, "effects", tuple(sorted({effect for op in function.operations for effect in op.effects})))
        object.__setattr__(self, "_forward_key", self.forward._structural_key())

    @property
    def saved_count(self):
        return len(self.result_types) - self.output_count

    @property
    def has_backward(self):
        return self.backward_rule is not None

    def verify(self):
        if self.forward._structural_key() != self._forward_key:
            raise ValueError("custom call forward graph changed after its definition was captured")

    def build_backward(self, target_graph, operands, forward_results, cotangents, *, engine=None):
        if not self.has_backward:
            raise ValueError(f"custom function {self.name} has no backward rule")
        self.verify()
        return self.backward_rule(target_graph, operands, forward_results, cotangents, engine=engine)


@dataclass(frozen=True, eq=False)
class Value:
    name: str
    type: Type
    value_id: str
    owner: Block
    producer: Operation | None = None
    index: int = 0

    def __post_init__(self):
        as_type(self.type)
        if not isinstance(self.name, str) or not self.name.startswith("%"):
            raise ValueError("SSA value names must start with %")
        if not isinstance(self.value_id, str) or not self.value_id or type(self.index) is not int or self.index < 0:
            raise ValueError("SSA values require a nonempty identity and nonnegative result index")

    @property
    def origin_id(self):
        if self.producer is None:
            return self.value_id
        return f"{self.producer.origin_id}.result{self.index}"


@dataclass(frozen=True, eq=False)
class Operation:
    opcode: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attrs: Mapping
    regions: tuple[Region, ...]
    effects: tuple[str, ...]
    id: str
    owner: Block
    origin_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "attrs", _freeze(self.attrs))
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "effects", _effects(self.effects))
        origin = self.id if self.origin_id is None else self.origin_id
        if not isinstance(self.id, str) or not self.id or not isinstance(origin, str) or not origin:
            raise ValueError("Fuga operations require nonempty identities and origins")
        object.__setattr__(self, "origin_id", origin)


@dataclass(frozen=True, eq=False)
class Region:
    blocks: tuple[Block, ...]
    parent: Function | Operation | None = None

    def __post_init__(self):
        object.__setattr__(self, "blocks", tuple(self.blocks))


class _Sealable:
    __slots__ = ()

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise ValueError("a cached Fuga graph template is immutable")
        object.__setattr__(self, name, value)


class Block(_Sealable):
    """An ordered SSA scope; mutation goes through its builder until terminated."""
    __slots__ = ("arguments", "region", "_operations", "_available", "_terminated", "_sealed")

    def __init__(self):
        self.arguments = ()
        self.region = None
        self._operations = []
        self._available = set()
        self._terminated = False

    @property
    def operations(self):
        return tuple(self._operations)

    @property
    def terminator(self):
        return self._operations[-1] if self._terminated else None


def _values(value):
    if isinstance(value, Value):
        return (value,)
    return tuple(value)


def _unpack(values):
    return values[0] if len(values) == 1 else tuple(values)


class BlockBuilder:
    def __init__(self, function, block):
        self.function, self.block = function, block

    @property
    def operations(self):
        return self.block.operations

    @property
    def arguments(self):
        return self.block.arguments

    @property
    def module(self):
        return self.function.module

    def _check_values(self, values):
        if self.block._terminated:
            raise ValueError("cannot append operations after a block terminator")
        if any(not isinstance(value, Value) or value not in self.block._available for value in values):
            raise ValueError("operands must be preceding values from this block; pass region captures as block arguments")

    def operation(self, opcode, operands=(), result_types=(), *, attrs=None, regions=(), effects=(),
                  id=None, origin_id=None):
        """Append a structured operation; verify() checks its semantic contract."""
        operands, regions = tuple(operands), tuple(regions)
        self._check_values(operands)
        result_types = tuple(as_type(type_) for type_ in result_types)
        if not isinstance(opcode, str) or not opcode:
            raise ValueError("operation opcode must be a nonempty string")
        sequence = self.function._next_operation()
        if id is None:
            while f"{self.function.name}.op{sequence}" in self.function._ids:
                sequence = self.function._next_operation()
            id = f"{self.function.name}.op{sequence}"
        if not isinstance(id, str) or not id:
            raise ValueError("operation id must be a nonempty string")
        if id in self.function._ids:
            raise ValueError(f"duplicate Fuga operation id: {id}")
        if any(not isinstance(region, Region) or region.parent is not None for region in regions):
            raise ValueError("each region must be unattached and owned by exactly one operation")
        op = Operation(opcode, operands, (), _freeze(attrs or {}), regions, _effects(effects),
                       id, self.block, origin_id)
        results = tuple(Value(self.function._reserve_name(f"%v{sequence}_{index}"), type_,
                              f"{id}.result{index}", self.block, op, index)
                        for index, type_ in enumerate(result_types))
        object.__setattr__(op, "results", results)
        for region in regions:
            object.__setattr__(region, "parent", op)
        self.function._ids.add(id)
        self.block._operations.append(op)
        self.block._available.update(results)
        if opcode in _TERMINATORS:
            self.block._terminated = True
        return op

    def constant(self, value, *, dtype="i32", id=None):
        from .primitives import infer_result_types
        type_, = infer_result_types("constant", attrs={"value": value}, dtype=dtype)
        return self.operation("constant", (), (type_,), attrs={"value": constant_value(value, type_.dtype)}, id=id).results[0]

    def splat(self, value, shape, *, id=None):
        from .primitives import infer_result_types
        self._check_values((value,))
        result, = infer_result_types("splat", (value.type,), {"shape": shape})
        return self.operation("splat", (value,), (result,), attrs={"shape": result.shape}, id=id).results[0]

    def cast(self, value, *, dtype, id=None):
        from .primitives import infer_result_types
        self._check_values((value,))
        types = infer_result_types("cast", (value.type,), dtype=dtype)
        return self.operation("cast", (value,), types, id=id).results[0]

    def _binary(self, opcode, lhs, rhs, *, id=None):
        from .primitives import infer_result_types
        self._check_values((lhs, rhs))
        attrs = {"overflow": "wrap"} if lhs.type.dtype in ("i32", "i64") else {}
        types = infer_result_types(opcode, (lhs.type, rhs.type), attrs)
        return self.operation(opcode, (lhs, rhs), types, attrs=attrs, id=id).results[0]

    def add(self, lhs, rhs, *, id=None):
        return self._binary("add", lhs, rhs, id=id)

    def mul(self, lhs, rhs, *, id=None):
        return self._binary("mul", lhs, rhs, id=id)

    def external(self, *operands, spec, result_types=None, id=None):
        self._check_values(operands)
        if not isinstance(spec, ExternalSpec):
            raise TypeError("external requires a Fuga ExternalSpec")
        if result_types is None:
            if spec.shape_rule != "same_as" or spec.shape_operand >= len(operands):
                raise ValueError("external requires explicit result_types or a valid same_as operand")
            result_types = (operands[spec.shape_operand].type,)
        return _unpack(self.operation("external", operands, result_types,
                                     attrs={"spec": spec}, effects=spec.effects, id=id).results)

    def call(self, *operands, spec, id=None):
        self._check_values(operands)
        if not isinstance(spec, CallSpec):
            raise TypeError("call requires a Fuga CallSpec")
        spec.verify()
        if tuple(value.type for value in operands) != spec.input_types:
            raise TypeError("custom call operands disagree with its forward signature")
        return _unpack(self.operation("call", operands, spec.result_types,
                                     attrs={"spec": spec}, effects=spec.effects, id=id).results)

    def _region(self, types):
        block = Block()
        region = Region((block,))
        block.region = region
        sequence = self.function._next_region()
        block.arguments = tuple(Value(self.function._reserve_name(f"%r{sequence}_arg{index}"), as_type(type_),
                                      f"{self.function.name}.region{sequence}.arg{index}", block, None, index)
                                for index, type_ in enumerate(types))
        block._available.update(block.arguments)
        return region, BlockBuilder(self.function, block)

    @staticmethod
    def _finish_region(builder, callback):
        result = callback(builder, *builder.arguments)
        if not builder.block._terminated:
            if result is None:
                raise ValueError("region callback must return values or call yield_")
            builder.yield_(*_values(result))
        elif result is not None:
            raise ValueError("region callback cannot both call yield_ and return values")
        return builder.block.terminator.operands

    def if_(self, condition, operands, then_fn, else_fn, *, id=None):
        operands = tuple(operands)
        self._check_values((condition, *operands))
        if not _is_predicate(condition.type):
            raise TypeError("if predicate must be scalar bool or rank-zero bool tensor")
        regions, yields = [], []
        for callback in (then_fn, else_fn):
            region, builder = self._region(value.type for value in operands)
            outputs = self._finish_region(builder, callback)
            regions.append(region)
            yields.append(tuple(value.type for value in outputs))
        if yields[0] != yields[1]:
            raise TypeError("if branches must yield exactly the same result types")
        return _unpack(self.operation("if", (condition, *operands), yields[0], regions=regions,
                                     effects=_region_effects(regions), id=id).results)

    def for_(self, lower, upper, step, initial_values, body_fn, *, id=None):
        if any(type(value) is not int or not -(2**63) <= value < 2**63 for value in (lower, upper, step)) or step == 0:
            raise ValueError("for bounds must be signed i64 integers and step must be nonzero")
        initial = tuple(initial_values)
        self._check_values(initial)
        types = tuple(value.type for value in initial)
        region, builder = self._region((ScalarType("i64"), *types))
        outputs = self._finish_region(builder, body_fn)
        if tuple(value.type for value in outputs) != types:
            raise TypeError("for body must yield the loop-carried value types")
        return _unpack(self.operation("for", initial, types,
                                     attrs={"lower": lower, "upper": upper, "step": step},
                                     regions=(region,), effects=_region_effects((region,)), id=id).results)

    def while_(self, initial_values, condition_fn, body_fn, *, id=None):
        initial = tuple(initial_values)
        self._check_values(initial)
        types = tuple(value.type for value in initial)
        regions = []
        for callback in (condition_fn, body_fn):
            region, builder = self._region(types)
            outputs = self._finish_region(builder, callback)
            if not regions:
                if len(outputs) != 1 or not _is_predicate(outputs[0].type):
                    raise TypeError("while condition must yield one bool predicate")
            elif tuple(value.type for value in outputs) != types:
                raise TypeError("while body must yield the loop-carried value types")
            regions.append(region)
        return _unpack(self.operation("while", initial, types, regions=regions,
                                     effects=_region_effects(regions), id=id).results)

    def yield_(self, *values):
        if isinstance(self, Function):
            raise ValueError("function body requires return_, not yield_")
        self.operation("yield", values)


class Function(_Sealable, BlockBuilder):
    def __init__(self, module, name, inputs, results):
        self._module = module
        self.name = _identifier(name)
        self._counter = self._region_counter = 0
        self._ids = set()
        self._names = set()
        self.result_types = None if results is None else tuple(as_type(type_) for type_ in results)
        block = Block()
        self.body = Region((block,), self)
        block.region = self.body
        block.arguments = tuple(Value(self._reserve_name(f"%{_identifier(name_)}"), as_type(type_),
                                      f"{name}.arg{index}", block, None, index)
                                for index, (name_, type_) in enumerate(inputs.items()))
        block._available.update(block.arguments)
        self.inputs = MappingProxyType(dict(zip(inputs, block.arguments)))
        super().__init__(self, block)

    @property
    def module(self):
        return self._module

    @property
    def outputs(self):
        if not self.block._terminated or self.block.terminator.opcode != "return":
            raise ValueError("Fuga function has no return")
        return self.block.terminator.operands

    def _next_operation(self):
        result = self._counter
        self._counter += 1
        return result

    def _reserve_name(self, candidate):
        while candidate in self._names:
            candidate += "_"
        self._names.add(candidate)
        return candidate

    def _next_region(self):
        result = self._region_counter
        self._region_counter += 1
        return result

    def return_(self, *values):
        self._check_values(values)
        types = tuple(value.type for value in values)
        if self.result_types is not None and self.result_types != types:
            raise TypeError("return types must match the Fuga function declaration")
        self.operation("return", values)
        self.result_types = types


class Module(_Sealable):
    def __init__(self):
        self._functions = []

    @property
    def functions(self):
        return tuple(self._functions)

    def graph(self, name, *, inputs, results=None):
        if getattr(self, "_sealed", False):
            raise ValueError("a cached Fuga graph template is immutable")
        if not isinstance(inputs, Mapping):
            raise TypeError("Fuga graph inputs must map argument names to types")
        if any(function.name == name for function in self._functions):
            raise ValueError(f"duplicate Fuga function symbol: {name}")
        function = Function(self, name, inputs, results)
        self._functions.append(function)
        return function

    def verify(self):
        if getattr(self, "_sealed", False):
            return self
        if len({function.name for function in self.functions}) != len(self.functions):
            raise ValueError("duplicate Fuga function symbol")
        for function in self.functions:
            if function.module is not self or function.body.parent is not function:
                raise ValueError("function/module ownership mismatch")
            if tuple(function.inputs.values()) != function.block.arguments:
                raise ValueError("function input declarations disagree with block arguments")
            _verify_region(function.body, "return", function.result_types, set(), set(), set(), set())
        return self

    def _seal(self):
        """Verify and freeze an internal execution template exactly once."""
        if getattr(self, "_sealed", False):
            return self
        self.verify()

        def seal_block(block):
            for operation in block.operations:
                if operation.opcode == "call":
                    operation.attrs["spec"].forward._seal()
                for region in operation.regions:
                    for child in region.blocks:
                        seal_block(child)
            block._operations = tuple(block._operations)
            block._available = frozenset(block._available)
            block._sealed = True

        for function in self.functions:
            seal_block(function.block)
            function._ids = frozenset(function._ids)
            function._names = frozenset(function._names)
            function._sealed = True
        self._functions = tuple(self._functions)
        self._key = self._structural_key()
        self._sealed = True
        return self

    def _structural_key(self):
        """Exact immutable structure, with scalar types and float bits preserved."""
        if getattr(self, "_sealed", False):
            return self._key

        def attribute(value):
            if isinstance(value, Mapping):
                return 'mapping', tuple(sorted((key, attribute(item)) for key, item in value.items()))
            if isinstance(value, tuple):
                return 'tuple', tuple(attribute(item) for item in value)
            if isinstance(value, ExternalSpec):
                return 'external', tuple((name, attribute(getattr(value, name))) for name in (
                    'compiler', 'source', 'entry', 'signature', 'source_bindings', 'constants',
                    'options', 'target', 'grid', 'shape_rule', 'shape_operand', 'effects'))
            if isinstance(value, CallSpec):
                return ('call', value.name, value.forward._structural_key(), value.output_count,
                        value.backward_name, value.definition_key)
            if type(value) is float:
                return 'float', struct.pack('!d', value)
            return type(value).__name__, value

        def value_key(value):
            return value.name, value.value_id, value.type

        def block_key(block):
            indices = {value: index for index, value in enumerate(block.arguments)}
            operations = []
            for operation in block.operations:
                operands = tuple(indices[value] for value in operation.operands)
                for value in operation.results:
                    indices[value] = len(indices)
                regions = tuple(tuple(block_key(child) for child in region.blocks) for region in operation.regions)
                operations.append((operation.opcode, operation.id, operation.origin_id, operands,
                                   tuple(value_key(value) for value in operation.results),
                                   attribute(operation.attrs), operation.effects, regions))
            return tuple(value_key(value) for value in block.arguments), tuple(operations)

        self.verify()
        return _StructuralKey(tuple((function.name, tuple(function.inputs), function.result_types,
                                     block_key(function.block)) for function in self.functions))

    def to_text(self):
        if getattr(self, "_sealed", False) and hasattr(self, '_text'):
            return self._text
        self.verify()
        lines = ["fuga.module {"]
        for function in self.functions:
            args = ", ".join(f"{value.name}: {value.type}" for value in function.block.arguments)
            results = ", ".join(str(type_) for type_ in function.result_types)
            lines.append(f"  fuga.func @{function.name}({args}) -> ({results}) {{")
            _print_block(function.block, lines, 4)
            lines.append("  }")
        lines.append("}")
        text = "\n".join(lines) + "\n"
        if getattr(self, "_sealed", False):
            object.__setattr__(self, '_text', text)
        return text

    def __str__(self):
        return self.to_text()


def _is_predicate(type_):
    return type_ == ScalarType("bool") or type_ == TensorType((), "bool")


def _region_effects(regions):
    return tuple(sorted({effect for region in regions for block in region.blocks
                         for operation in block.operations for effect in operation.effects}))


def _verify_region(region, terminator, result_types, seen_values, seen_ids, seen_names, seen_value_ids):
    if len(region.blocks) != 1:
        raise ValueError("Fuga structured regions currently require exactly one block")
    block = region.blocks[0]
    if block.region is not region:
        raise ValueError("region/block ownership mismatch")
    available = set()
    for index, value in enumerate(block.arguments):
        as_type(value.type)
        if (value.owner is not block or value.producer is not None or value.index != index or value in seen_values
                or value.name in seen_names or value.value_id in seen_value_ids):
            raise ValueError("invalid or reused Fuga block argument")
        available.add(value)
        seen_values.add(value)
        seen_names.add(value.name)
        seen_value_ids.add(value.value_id)
    if not block.operations or block.operations[-1].opcode != terminator:
        raise ValueError(f"Fuga block requires a final {terminator}")
    for position, op in enumerate(block.operations):
        if op.owner is not block or op.id in seen_ids or not isinstance(op.origin_id, str) or not op.origin_id:
            raise ValueError("operation ownership mismatch or duplicate operation id")
        seen_ids.add(op.id)
        if any(value not in available for value in op.operands):
            raise ValueError("operand does not dominate its use; region captures must be explicit arguments")
        if op.opcode in _TERMINATORS and position != len(block.operations) - 1:
            raise ValueError("operation follows a terminator")
        for index, value in enumerate(op.results):
            as_type(value.type)
            if (value.owner is not block or value.producer is not op or value.index != index or value in seen_values
                    or value.name in seen_names or value.value_id in seen_value_ids):
                raise ValueError("invalid or multiply-defined SSA result")
            seen_values.add(value)
            seen_names.add(value.name)
            seen_value_ids.add(value.value_id)
        _verify_operation(op, seen_values, seen_ids, seen_names, seen_value_ids)
        available.update(op.results)
    if result_types is None or tuple(value.type for value in block.operations[-1].operands) != tuple(result_types):
        raise TypeError("Fuga region terminator does not match its result types")


def _verify_operation(op, seen_values, seen_ids, seen_names, seen_value_ids):
    operands, results, attrs = op.operands, op.results, op.attrs
    kinds = {"constant", "splat", "cast", "add", "mul", "external", "call", "if", "for", "while", "return", "yield"}
    if op.opcode not in kinds:
        raise ValueError(f"unknown Fuga operation: {op.opcode}")
    if op.opcode not in ("if", "for", "while") and op.regions:
        raise ValueError(f"{op.opcode} cannot own regions")
    if op.opcode in ("constant", "splat", "cast", "add", "mul") and len(results) != 1:
        raise TypeError(f"{op.opcode} requires one result")
    if op.opcode not in ("external", "call", "if", "for", "while") and op.effects:
        raise ValueError(f"{op.opcode} must be pure")
    if op.opcode == "constant":
        from .primitives import validate_primitive
        validate_primitive(op)
        if operands or not isinstance(results[0].type, ScalarType) or set(attrs) != {"value"}:
            raise TypeError("constant requires a scalar result, a value, and no operands")
        canonical = constant_value(attrs["value"], results[0].type.dtype)
        if isinstance(canonical, float):
            raw = attrs["value"]
            equal = (type(raw) is float and math.isnan(raw)) if math.isnan(canonical) else raw == canonical
            if not equal or (canonical == 0 and math.copysign(1.0, raw) != math.copysign(1.0, canonical)):
                raise ValueError("Fuga floating constants must already be rounded to their declared dtype")
    elif op.opcode in ("splat", "cast", "add", "mul"):
        from .primitives import validate_primitive
        validate_primitive(op)
    elif op.opcode == "external":
        spec = attrs.get("spec")
        if set(attrs) != {"spec"} or not isinstance(spec, ExternalSpec) or not results:
            raise TypeError("external requires a structured spec and at least one result")
        if op.effects != spec.effects:
            raise ValueError("external effect declaration differs from its spec")
        pointers = [value for value in spec.source_bindings.values() if isinstance(value, str)]
        bindings = set(pointers)
        expected = {f"input:{index}" for index in range(len(operands))} | {f"output:{index}" for index in range(len(results))}
        if bindings != expected or len(bindings) != len(pointers):
            raise ValueError("external source bindings must cover every input and output exactly once")
        if spec.shape_rule == "same_as":
            if spec.shape_operand >= len(operands):
                raise ValueError("external shape operand is out of range")
            source = operands[spec.shape_operand].type
            if any(type(value.type) is not type(source) or
                   (isinstance(source, TensorType) and value.type.shape != source.shape) for value in results):
                raise TypeError("external same_as result shapes disagree with the selected operand")
    elif op.opcode == "call":
        spec = attrs.get("spec")
        if set(attrs) != {"spec"} or not isinstance(spec, CallSpec):
            raise TypeError("call requires a structured custom function spec")
        spec.verify()
        if tuple(value.type for value in operands) != spec.input_types or tuple(value.type for value in results) != spec.result_types:
            raise TypeError("custom call input or result types disagree with its forward signature")
        if op.effects != spec.effects:
            raise ValueError("custom call effects disagree with its forward graph")
    elif op.opcode in _TERMINATORS:
        if results or attrs or op.regions:
            raise TypeError("terminators cannot have results, attributes, or regions")
    else:
        if op.effects != _region_effects(op.regions):
            raise ValueError("control-flow effects must include exactly the nested region effects")
        result_types = tuple(value.type for value in results)
        if op.opcode == "if":
            if len(op.regions) != 2 or not operands or not _is_predicate(operands[0].type) or attrs:
                raise TypeError("if requires one bool predicate and two regions")
            region_types = [tuple(value.type for value in operands[1:])] * 2
            yield_types = [result_types] * 2
        elif op.opcode == "for":
            if len(op.regions) != 1 or set(attrs) != {"lower", "upper", "step"} or any(
                    type(attrs[name]) is not int or not -(2**63) <= attrs[name] < 2**63 for name in attrs) or attrs["step"] == 0:
                raise TypeError("for requires one region and signed i64 static bounds with nonzero step")
            if tuple(value.type for value in operands) != result_types:
                raise TypeError("for operands and results must preserve carried types")
            region_types = [(ScalarType("i64"), *result_types)]
            yield_types = [result_types]
        else:
            if len(op.regions) != 2 or attrs or tuple(value.type for value in operands) != result_types:
                raise TypeError("while requires condition/body regions and invariant carried types")
            region_types = [result_types] * 2
            predicate_results = op.regions[0].blocks[0].operations[-1].operands if op.regions[0].blocks and op.regions[0].blocks[0].operations else ()
            if len(predicate_results) != 1 or not _is_predicate(predicate_results[0].type):
                raise TypeError("while condition must yield one bool predicate")
            yield_types = [(predicate_results[0].type,), result_types]
        for region, arguments, yielded in zip(op.regions, region_types, yield_types):
            if region.parent is not op or len(region.blocks) != 1 or tuple(value.type for value in region.blocks[0].arguments) != arguments:
                raise TypeError("control-flow region ownership or argument types disagree")
            _verify_region(region, "yield", yielded, seen_values, seen_ids, seen_names, seen_value_ids)


def _format_attr(value):
    if isinstance(value, ExternalSpec):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
    elif isinstance(value, CallSpec):
        value = {"name": value.name, "definition_key": value.definition_key,
                 "input_types": value.input_types, "result_types": value.result_types,
                 "output_count": value.output_count, "saved_count": value.saved_count,
                 "backward": value.backward_name, "forward": value.forward.to_text()}
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{name if _IDENTIFIER.fullmatch(name) else json.dumps(name, ensure_ascii=False)}={_format_attr(value[name])}"
                               for name in sorted(value)) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_format_attr(item) for item in value) + "]"
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (ScalarType, TensorType)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _print_block(block, lines, indentation):
    indent = " " * indentation
    for op in block.operations:
        prefix = ", ".join(value.name for value in op.results)
        prefix = prefix + " = " if prefix else ""
        operands = ", ".join(value.name for value in op.operands)
        attrs = " " + _format_attr(op.attrs) if op.attrs else ""
        effects = " effects=" + _format_attr(op.effects) if op.effects else ""
        origin = " origin=" + _format_attr(op.origin_id) if op.origin_id != op.id else ""
        result_types = " -> (" + ", ".join(str(value.type) for value in op.results) + ")" if op.results else ""
        lines.append(f"{indent}{prefix}fuga.{op.opcode}({operands}){attrs}{result_types}{effects}{origin}")
        for index, region in enumerate(op.regions):
            inner = region.blocks[0]
            arguments = ", ".join(f"{value.name}: {value.type}" for value in inner.arguments)
            lines.append(f"{indent}region {index}({arguments}) {{")
            _print_block(inner, lines, indentation + 2)
            lines.append(indent + "}")

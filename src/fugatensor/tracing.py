"""Trace tensor expressions into device-independent Fuga SSA."""
from __future__ import annotations

import math
from . import _native
from .ir import Module, TensorType, Value
from .ir.core import BlockBuilder, _effects, _freeze
from .ir.types import as_type
from .dtypes import DType
from .devices import device as Device, resolve_device

from .tensor import normalize_dtype, normalize_shape
from ._state import active_trace


class TraceTensor(_native.TraceHandle):
    """A Fuga SSA value used only while an operation is being traced."""
    __slots__ = ()

    def __init__(self, context: TraceContext, value):
        value = context._import(value)
        if not isinstance(value, _native.TraceValue) or value.arena is not context._arena:
            raise ValueError("traced value must belong to its TraceContext graph")
        if not isinstance(value.type, TensorType):
            raise TypeError("TraceTensor requires a Fuga tensor value")
        super().__init__(context, value.value_id)

    @property
    def device(self) -> Device:
        return resolve_device(self.context.engine.device if self.context.engine is not None else "cpu")

    def astype(self, dtype: DType | str) -> TraceTensor:
        value = self.context.graph.cast(self.value, dtype=normalize_dtype(dtype))
        return self.context.wrap(value)

    def reshape(self, *shape) -> TraceTensor:
        from ._shape import infer_reshape
        shape = infer_reshape(shape, math.prod(self.shape))
        return self.context.wrap(self.context.graph.reshape(self.value, shape))

    def detach(self) -> TraceTensor:
        return self.context.wrap(self.context.graph.stop_gradient(self.value))

    def _operand(self, other):
        if isinstance(other, TraceTensor):
            if other.context is not self.context:
                raise ValueError("traced operands must belong to the same graph")
            if other.shape != self.shape:
                raise ValueError("tensor operands require equal shapes; use broadcast_to explicitly")
            if other.dtype != self.dtype:
                raise TypeError("tensor operands require equal dtypes; use astype explicitly")
            return other.value
        if type(other) not in (bool, int, float):
            raise TypeError("traced expressions accept traced tensors or Python scalars; pass tensor captures as operation arguments")
        constant = self.context.graph.constant(other, dtype=self.dtype)
        return self.context.graph.splat(constant, self.shape)

    def _binary(self, operation, other, *, reverse=False):
        if self.dtype == "bool" and not isinstance(other, TraceTensor):
            raise TypeError("bool arithmetic is not defined; use astype to select a numeric type")
        other_value = self._operand(other)
        lhs, rhs = (other_value, self.value) if reverse else (self.value, other_value)
        value = getattr(self.context.graph, operation)(lhs, rhs)
        return self.context.wrap(value)

    def __add__(self, other):
        return self._binary("add", other)

    def __radd__(self, other):
        return self._binary("add", other, reverse=True)

    def __mul__(self, other):
        return self._binary("mul", other)

    def __rmul__(self, other):
        return self._binary("mul", other, reverse=True)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def square(self):
        return self * self

    def __bool__(self):
        if self.shape != ():
            raise TypeError('control-flow conditions require a scalar Tensor with shape=()')
        raise TypeError('data-dependent control flow inside an isolated fun or VJP trace is unsupported')

    def tolist(self):
        raise TypeError("tensor values cannot be read while tracing an isolated operation")

    def __truediv__(self, other):
        from .ops import divide
        return divide(self, other)

    def __rtruediv__(self, other):
        from .ops import divide
        return divide(other, self)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .ops import matmul
        return matmul(other, self)

    def __eq__(self, other):
        from .ops import compare
        return compare(self, other, 'eq')

    def __ne__(self, other):
        from .ops import compare
        return compare(self, other, 'ne')

    def __lt__(self, other):
        from .ops import compare
        return compare(self, other, 'lt')

    def __le__(self, other):
        from .ops import compare
        return compare(self, other, 'le')

    def __gt__(self, other):
        from .ops import compare
        return compare(self, other, 'gt')

    def __ge__(self, other):
        from .ops import compare
        return compare(self, other, 'ge')

    def sum(self, axis=None, *, keepdims=False, dtype=None):
        from .ops import sum
        return sum(self, axis, keepdims=keepdims, dtype=dtype)

    def amax(self, axis=None, *, keepdims=False):
        from .ops import amax
        return amax(self, axis, keepdims=keepdims)

    def mean(self, axis=None, *, keepdims=False, dtype=None):
        from .ops import mean
        return mean(self, axis, keepdims=keepdims, dtype=dtype)

    def transpose(self, permutation=None):
        from .ops import transpose
        return transpose(self, permutation)

    @property
    def T(self):
        return self.transpose()

    def broadcast_to(self, shape):
        from .ops import broadcast_to
        return broadcast_to(self, shape)

    def __repr__(self):
        return f"TraceTensor(shape={self.shape!r}, dtype={self.dtype!r}, value={self.value.name!r})"


class _TraceBuilder(BlockBuilder):
    """Use shared abstract rules, recording their results in the native arena."""

    def __init__(self, context, inputs):
        self.context = context
        self.inputs = inputs
        self.result_types = None
        self._ids = set()

    def _check_values(self, values):
        if self.context._arena.terminated:
            raise ValueError("cannot append operations after a block terminator")
        if any(not isinstance(value, _native.TraceValue) or value.arena is not self.context._arena
               for value in values):
            raise ValueError("operands must be preceding values from this block; pass region captures as block arguments")

    def operation(self, opcode, operands=(), result_types=(), *, attrs=None, regions=(), effects=(),
                  id=None, origin_id=None):
        operands = tuple(operands)
        self._check_values(operands)
        result_types = tuple(as_type(type_) for type_ in result_types)
        if not isinstance(opcode, str) or not opcode:
            raise ValueError("operation opcode must be a nonempty string")
        if regions:
            raise ValueError("isolated Tensor traces contain flat operations")
        if id is not None:
            if not isinstance(id, str) or not id:
                raise ValueError("operation id must be a nonempty string")
            if id in self._ids:
                raise ValueError(f"duplicate Fuga operation id: {id}")
            self._ids.add(id)
        return self.context._arena.append(opcode, operands, result_types, _freeze(attrs or {}),
                                          _effects(effects), id, origin_id)

    def return_(self, *values):
        self._check_values(values)
        self.result_types = tuple(value.type for value in values)
        self.operation('return', values)

    @property
    def operations(self):
        self.context._export()
        return self.context._target.operations

    @property
    def module(self):
        return self.context.module


class TraceContext:
    """Own one Fuga graph and infer its result types from the Python return."""
    def __init__(self, input_specs, *, engine=None, name="fuga_main"):
        self.engine = engine
        self._arena = _native.TraceArena()
        self._module = self._target = None
        self._name = name
        self._imported = []
        self._imports = {}
        self._exported = ()
        self._emitted = 0
        self._finished = False
        self.input_specs = tuple((normalize_shape(shape), normalize_dtype(dtype))
                                 for shape, dtype in input_specs)
        self.graph = _TraceBuilder(self, {
            f"arg{index}": _native.TraceValue(self._arena, self._arena.input(TensorType(shape, dtype)))
            for index, (shape, dtype) in enumerate(self.input_specs)})
        self.inputs = tuple(TraceTensor(self, value) for value in self.graph.inputs.values())
        self.output_specs = None
        self.output_is_tuple = None
        self.external_count = 0
        parent = active_trace.get()
        self.dependencies = parent.dependencies if parent is not None else {}

    @classmethod
    def from_graph(cls, graph, *, engine=None):
        """Attach tracing glue to an existing Fuga builder during transforms."""
        context = cls.__new__(cls)
        parent = active_trace.get()
        context.engine = engine if engine is not None else (parent.engine if parent is not None else None)
        context._arena = _native.TraceArena()
        context._module = graph.module
        context._target = graph
        context._name = graph.function.name
        context._imported = []
        context._imports = {}
        context._exported = ()
        context._emitted = 0
        context._finished = False
        context.graph = _TraceBuilder(context, {name: context._import(value) for name, value in graph.inputs.items()})
        context.input_specs = tuple((value.type.shape, value.type.dtype)
                                    for value in graph.inputs.values() if isinstance(value.type, TensorType))
        context.inputs = tuple(TraceTensor(context, value) for value in context.graph.inputs.values()
                               if isinstance(value.type, TensorType))
        context.output_specs = context.output_is_tuple = None
        context._finished = False
        context.external_count = sum(op.opcode == 'external' for op in graph.operations)
        context.dependencies = parent.dependencies if parent is not None else {}
        return context

    def _import(self, value):
        if isinstance(value, Value):
            if self._target is None or value.owner is not self._target.block:
                raise ValueError("traced value must belong to its TraceContext graph")
            if value not in self._imports:
                self._imports[value] = _native.TraceValue(self._arena, self._arena.input(value.type))
                self._imported.append(value)
            return self._imports[value]
        return value

    def _export(self):
        if self._target is None:
            self._module = Module()
            self._target = self._module.graph(self._name, inputs={
                name: value.type for name, value in self.graph.inputs.items()})
            self._imported = list(self._target.inputs.values())
        self._exported = self._arena.emit(self._target, tuple(self._imported), self._exported, self._emitted)
        self._emitted = self._arena.node_count

    def export_values(self, values):
        """Append a VJP trace to its existing Python IR builder once."""
        values = tuple(values)
        self._export()
        return tuple(self._exported[value.value.value_id] for value in values)

    @property
    def module(self):
        if not self._finished or self._module is None:
            self._export()
        return self._module

    def depend_on(self, definition):
        self.dependencies[definition] = definition._definition_key()

    def wrap(self, value: Value) -> TraceTensor:
        if self._finished:
            raise ValueError("cannot extend a completed trace")
        return TraceTensor(self, value)

    def finish(self, result) -> Module:
        if self._finished:
            raise ValueError("a trace can only be finished once")
        is_tuple = isinstance(result, tuple)
        outputs = result if is_tuple else (result,)
        if not outputs or any(not isinstance(value, TraceTensor) for value in outputs):
            raise TypeError("an operation trace must return a traced Tensor or a nonempty tuple of traced Tensors")
        if any(value.context is not self for value in outputs):
            raise ValueError("operation outputs must belong to the current trace")
        self.graph.return_(*(value.value for value in outputs))
        self.output_specs = tuple((value.shape, value.dtype) for value in outputs)
        self.output_is_tuple = is_tuple
        self._finished = True
        return self.module.verify()

"""Join pending Tensor expressions into one ASK compilation at a value boundary."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import _native
from .ir import Module, TensorType, project_outputs, prune, specialize
from .ir.transforms import _clone_operation
from .tensor import Tensor


Expression = _native.Expression
Result = _native.Result
Alias = _native.Alias


@dataclass(frozen=True)
class Program:
    executable: object
    fuga_text: str


def defer_module(module, inputs, *, engine, dependencies=()):
    """Create stable result handles with immutable expression provenance."""
    engine._check_open()
    inputs = tuple(inputs)
    if not isinstance(module, Module) or len(module.functions) != 1:
        raise TypeError('deferred execution requires one Fuga function')
    function = module.functions[0]
    if len(function.inputs) != len(inputs):
        raise ValueError('deferred graph input count differs from tensor bindings')
    for spec, value in zip(function.inputs.values(), inputs):
        engine._check_tensor(value)
        if (spec.type.shape, spec.type.dtype) != (value.shape, value.dtype):
            raise ValueError('deferred graph input type differs from tensor binding')
    return _template(module).bind(inputs, engine, tuple(dependencies))


def _source(tensor):
    return _native.source_tensor(tensor)


@dataclass(frozen=True)
class _Slice:
    module: Module
    indices: tuple
    inputs: tuple
    definitions: frozenset
    key: object


@dataclass(frozen=True)
class _Template:
    module: Module
    definitions: frozenset
    output_types: tuple
    slices: dict = field(default_factory=dict, compare=False)
    dependencies: dict = field(default_factory=dict, compare=False)

    def bind(self, inputs, engine, dependencies):
        """Instantiate a sealed template after its input signature has matched."""
        return _native.bind_pending(engine, Tensor, self, inputs, dependencies)

    def project(self, indices):
        selected = self.slices.get(indices)
        if selected is None:
            selected = self.slices.setdefault(indices, _project(self.module, indices))
        return selected

    def live_dependencies(self, selected, dependencies):
        key = selected.indices, dependencies
        if key not in self.dependencies:
            from .function import fun, _fingerprint
            kept = []
            for definition, version in dependencies:
                # Expanded derivative graphs retain dependencies even when no
                # explicit call remains. Only discard proven dead call ports.
                if isinstance(definition, fun):
                    identity = _fingerprint(version)
                    if identity in self.definitions and identity not in selected.definitions:
                        continue
                kept.append((definition, version))
            self.dependencies[key] = tuple(kept)
        return self.dependencies[key]


def _template(module):
    result = getattr(module, '_execution_template', None)
    if result is None:
        module._seal()
        result = _Template(module, frozenset(_call_definitions(module)),
                           tuple(value.type for value in module.functions[0].outputs))
        object.__setattr__(module, '_execution_template', result)
    return result


def _call_definitions(module):
    """Definition identities represented by custom boundaries in this module."""
    result = set()
    stack = [module]
    while stack:
        current = stack.pop()
        for function in current.functions:
            for operation in function.operations:
                if operation.opcode == 'call':
                    spec = operation.attrs['spec']
                    result.add(spec.definition_key)
                    stack.append(spec.forward)
    return result


def _project(module, indices):
    # This is a numerical copy only. Expression.module retains
    # its original CallSpecs and custom VJP rules for later differentiation.
    projected = project_outputs(module, indices)
    function = projected.functions[0]
    needed = set(function.outputs)
    definitions = set()
    for operation in reversed(function.operations):
        if operation.opcode == 'return':
            continue
        selected = tuple(index for index, value in enumerate(operation.results) if value in needed)
        if not selected and not operation.effects:
            continue
        if operation.opcode == 'call':
            spec = operation.attrs['spec']
            child = _project(spec.forward, selected or tuple(range(len(operation.results))))
            definitions.add(spec.definition_key)
            definitions.update(child.definitions)
            needed.update(operation.operands[index] for index in child.inputs)
        else:
            needed.update(operation.operands)
    numerical = prune(specialize(projected))
    function = numerical.functions[0]
    used = {value for operation in function.operations for value in operation.operands}
    inputs = tuple(index for index, value in enumerate(function.inputs.values()) if value in used)
    numerical._seal()
    return _Slice(numerical, tuple(indices), inputs, frozenset(definitions), numerical._structural_key())


def _compose(roots, collected):
    leaves, expressions, slices = collected
    module = Module()
    graph = module.graph('fuga_main', inputs={
        f'arg{index}': TensorType(value.shape, value.dtype)
        for index, value in enumerate(leaves)})
    values = {value._identity: arg for value, arg in zip(leaves, graph.inputs.values())}
    results = {}

    def value_for(tensor):
        source = _source(tensor)
        if source._storage is not None:
            return values[source._identity]
        pending = source._pending
        return results[pending.expression][pending.index]

    for expression in expressions:
        selected = slices[expression]
        function = selected.module.functions[0]
        arguments = tuple(function.inputs.values())
        mapping = {arguments[index]: value_for(expression.inputs[index]) for index in selected.inputs}
        for operation in function.operations:
            if operation.opcode == 'return':
                results[expression] = dict(zip(selected.indices, (mapping[value] for value in operation.operands)))
                break
            copied = _clone_operation(graph, operation,
                                      tuple(mapping[value] for value in operation.operands))
            mapping.update(zip(operation.results, copied.results))
    graph.return_(*(value_for(value) for value in roots))
    return module.verify()


def materialize(engine, tensors):
    tensors = tuple(tensors)
    for value in tensors:
        engine._check_tensor(value)
    roots = _native.pending_roots(tensors)
    if roots:
        inputs, key = _native.collect_pending(roots)
        program = engine._graph_cache.get(key)
        if program is None:
            module = _compose(roots, _native.describe_pending(roots))
            fuga_text = module.to_text()
            program = Program(engine._compile(module), fuga_text)
            engine._graph_cache[key] = program
        else:
            engine._stats['cache_hits'] += 1
        outputs = engine._execute(program.executable, inputs)
        # Publish only after the whole execution has succeeded. Histories stay
        # attached to the original handles, independently of numerical buffers.
        _native.publish_pending(roots, outputs)
        executable = program.executable
        engine.last_ir = engine.last_ask_ir = executable.source
        engine.last_fuga_ir = program.fuga_text
        engine.last_report, engine.last_artifact_dir = executable.report, executable.artifact_dir
        engine.last_execution_plan = executable.execution_plan
    for value in tensors:
        if value._storage is None:
            _native.publish_pending((value,), (_source(value)._storage,))
    return tensors

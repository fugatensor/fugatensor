"""Join pending Tensor expressions into one ASK compilation at a value boundary."""
from __future__ import annotations

from dataclasses import dataclass, field

from .ir import Module, project_outputs, prune, specialize
from .ir.transforms import _clone_operation
from .tensor import Tensor
from .tracing import TraceContext


@dataclass(frozen=True, eq=False)
class Expression:
    module: Module
    inputs: tuple
    dependencies: tuple


@dataclass(frozen=True)
class Result:
    expression: Expression
    index: int


@dataclass(frozen=True)
class Alias:
    source: Tensor


@dataclass(frozen=True)
class Program:
    executable: object
    fuga_text: str


def defer_module(module, inputs, *, engine, dependencies=()):
    """Create stable result handles, capturing gradient mode before execution."""
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
    _template(module)
    expression = Expression(module, inputs, tuple(dependencies))
    outputs = tuple(Tensor._from_pending(engine, value.type.shape, value.type.dtype,
                                         Result(expression, index))
                    for index, value in enumerate(function.outputs))
    from .autograd import record_execution
    return record_execution(module, inputs, outputs, engine)


def _source(tensor):
    while tensor._storage is None and isinstance(tensor._pending, Alias):
        tensor = tensor._pending.source
    return tensor


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
    slices: dict = field(default_factory=dict, compare=False)
    dependencies: dict = field(default_factory=dict, compare=False)

    def project(self, indices):
        if indices not in self.slices:
            self.slices[indices] = _project(self.module, indices)
        return self.slices[indices]

    def live_dependencies(self, selected, dependencies):
        key = selected.indices, dependencies
        if key not in self.dependencies:
            from .function import fun, _fingerprint
            kept = []
            for definition, version in dependencies:
                # Expanded backward graphs retain dependencies even when no
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
        result = _Template(module, frozenset(_call_definitions(module)))
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
    # This is a numerical copy only. Expression.module and AD histories keep
    # their original CallSpecs, residual ports and custom backward callbacks.
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


def _carrier(roots):
    """Find storage for constant-only ABI inputs without evaluating dead code."""
    preferred = roots[0].shape, roots[0].dtype
    fallback = None
    stack, seen = list(reversed(roots)), set()
    while stack:
        source = _source(stack.pop())
        if id(source) in seen:
            continue
        seen.add(id(source))
        if source._storage is not None:
            if (source.shape, source.dtype) == preferred:
                return source
            if fallback is None:
                fallback = source
        else:
            stack.extend(reversed(source._pending.expression.inputs))
    if fallback is None:
        raise ValueError('a constant-only deferred graph needs a materialized input carrier')
    return fallback


def _collect(roots):
    # First propagate requested result ports to a fixed point. A shared
    # expression may gain another requested output after its first visit.
    demands, slices = {}, {}
    stack = list(reversed(roots))
    while stack:
        source = _source(stack.pop())
        if source._storage is not None:
            continue
        result = source._pending
        expression = result.expression
        required = demands.setdefault(expression, set())
        if result.index in required:
            continue
        required.add(result.index)
        selected = _template(expression.module).project(tuple(sorted(required)))
        slices[expression] = selected
        stack.extend(expression.inputs[index] for index in reversed(selected.inputs))

    # Order only the selected numerical dependencies. Explicit stacks keep
    # long Python loops independent of Python's recursion limit.
    leaves, expressions = [], []
    seen_leaves, seen_expressions = set(), set()
    stack = [(False, tensor) for tensor in reversed(roots)]
    while stack:
        finish, item = stack.pop()
        if finish:
            if item not in seen_expressions:
                expressions.append(item)
                seen_expressions.add(item)
            continue
        tensor = _source(item)
        if tensor._storage is not None:
            if id(tensor) not in seen_leaves:
                leaves.append(tensor)
                seen_leaves.add(id(tensor))
            continue
        expression = tensor._pending.expression
        if expression not in seen_expressions:
            stack.append((True, expression))
            stack.extend((False, expression.inputs[index]) for index in reversed(slices[expression].inputs))
    if not leaves:
        leaves.append(_carrier(roots))
    return leaves, expressions, slices


def _dependencies(expressions, slices):
    dependencies = {}
    for expression in expressions:
        selected = slices[expression]
        template = _template(expression.module)
        for definition, version in template.live_dependencies(selected, expression.dependencies):
            if definition in dependencies and dependencies[definition] != version:
                raise RuntimeError('a function definition changed after this Tensor was constructed')
            dependencies[definition] = version
    if any(definition._definition_key() != version for definition, version in dependencies.items()):
        raise RuntimeError('a function definition changed after this Tensor was constructed')
    return tuple(dependencies.items())


def _structure_key(roots, collected, dependencies):
    """Flat topology with canonical leaf indices; values and addresses are not keys."""
    leaves, expressions, slices = collected
    leaf_indices = {id(value): index for index, value in enumerate(leaves)}
    expression_indices = {value: index for index, value in enumerate(expressions)}

    def binding(tensor):
        tensor = _source(tensor)
        if tensor._storage is not None:
            return 'input', leaf_indices[id(tensor)]
        pending = tensor._pending
        return 'result', expression_indices[pending.expression], pending.index

    nodes = tuple((slices[expression].key,
                   tuple(binding(expression.inputs[index]) for index in slices[expression].inputs))
                  for expression in expressions)
    return (tuple((value.shape, value.dtype) for value in leaves), nodes,
            tuple(binding(value) for value in roots), frozenset(dependencies))


def _compose(roots, engine, collected=None):
    leaves, expressions, slices = _collect(roots) if collected is None else collected
    context = TraceContext([(value.shape, value.dtype) for value in leaves], engine=engine)
    values = {id(value): arg.value for value, arg in zip(leaves, context.inputs)}
    results = {}

    def value_for(tensor):
        source = _source(tensor)
        if source._storage is not None:
            return values[id(source)]
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
            copied = _clone_operation(context.graph, operation,
                                      tuple(mapping[value] for value in operation.operands))
            mapping.update(zip(operation.results, copied.results))
    context.finish(tuple(context.wrap(value_for(value)) for value in roots))
    return context


def materialize(engine, tensors):
    tensors = tuple(tensors)
    for value in tensors:
        engine._check_tensor(value)
    roots, seen = [], set()
    for value in tensors:
        source = _source(value)
        if source._storage is None and id(source) not in seen:
            roots.append(source)
            seen.add(id(source))
    if roots:
        collected = _collect(roots)
        inputs, expressions, slices = collected
        dependencies = _dependencies(expressions, slices)
        key = _structure_key(roots, collected, dependencies)
        program = engine._graph_cache.get(key)
        if program is None:
            context = _compose(roots, engine, collected)
            fuga_text = context.module.to_text()
            program = Program(engine._compile(context), fuga_text)
            engine._graph_cache[key] = program
        else:
            engine._stats['cache_hits'] += 1
        outputs = engine._execute(program.executable, inputs)
        if len(outputs) != len(roots):
            raise RuntimeError('ASK returned an unexpected number of graph results')
        # Publish only after the whole execution has succeeded. Histories stay
        # attached to the original handles, independently of numerical buffers.
        for root, output in zip(roots, outputs):
            root._storage = output._storage
            root._pending = None
        executable = program.executable
        engine.last_ir = engine.last_ask_ir = executable.source
        engine.last_fuga_ir = program.fuga_text
        engine.last_report, engine.last_artifact_dir = executable.report, executable.artifact_dir
    for value in tensors:
        if value._storage is None:
            value._storage = _source(value)._storage
            value._pending = None
    return tensors

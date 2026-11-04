"""Functional reverse-mode differentiation of persistent Tensor expressions."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import weakref

from ._state import active_trace, require_not_tracing
from .ir import FLOAT_DTYPES, project_outputs, vjp
from .ir.transforms import derivative_inputs

_cache = weakref.WeakKeyDictionary()
_dependency_cache = weakref.WeakKeyDictionary()
_cache_lock = threading.RLock()


@dataclass(frozen=True)
class _Plan:
    derivative: object
    dependencies: tuple


def _needed_inputs(module, output_index):
    with _cache_lock:
        entries = _dependency_cache.setdefault(module, {})
        if output_index not in entries:
            entries[output_index] = derivative_inputs(module, (output_index,))
        return entries[output_index]


def _validate_definitions(module):
    """Validate frozen custom rules on the selected graph, including cache hits."""
    from ._lazy import _template
    live = _template(module).project(tuple(range(len(module.functions[0].outputs)))).definitions
    for function in module.functions:
        used = {value for operation in function.operations for value in operation.operands}
        for operation in function.operations:
            if operation.opcode != "call":
                continue
            selected = tuple(index for index, value in enumerate(operation.results) if value in used)
            if not selected:
                continue
            spec = operation.attrs["spec"]
            if spec.definition_key not in live:
                continue
            spec.verify()
            validate = getattr(spec.vjp_rule, "validate_definition", None)
            if validate is not None:
                validate()
            _validate_definitions(project_outputs(spec.forward, selected))


def _plan(expression, output_indices, input_indices, engine):
    from ._lazy import _template
    template = _template(expression.module)
    source_dependencies = template.live_dependencies(template.project(output_indices), expression.dependencies)
    if any(definition._definition_key() != version for definition, version in source_dependencies):
        raise ValueError("function definition changed after tracing; retrace the forward call")
    key = input_indices, output_indices
    with _cache_lock:
        by_module = _cache.setdefault(engine, weakref.WeakKeyDictionary())
        plans = by_module.setdefault(expression.module, {})
        if key not in plans:
            projected = project_outputs(expression.module, output_indices)
            _validate_definitions(projected)
            from .tracing import TraceContext
            context = TraceContext.from_graph(projected.functions[0], engine=engine)
            token = active_trace.set(context)
            try:
                derivative = vjp(projected, argnums=input_indices)
            finally:
                active_trace.reset(token)
            plans[key] = _Plan(derivative, (*source_dependencies, *context.dependencies.items()))
        plan = plans[key]
        # A derivative can prune all calls; validate its captured definitions too.
        _validate_definitions(project_outputs(expression.module, output_indices))
        _validate_definitions(plan.derivative)
        if any(definition._definition_key() != version for definition, version in plan.dependencies):
            raise ValueError("custom VJP definition changed after tracing; retrace the forward call")
        return plan


def _key(tensor):
    origin = tensor._origin
    return origin if origin is not None else tensor._identity


def _tensors(value, name):
    from .tensor import Tensor
    if isinstance(value, Tensor):
        return (value,), True
    if not isinstance(value, (tuple, list)) or not value:
        raise TypeError(f"grad {name} must be a Tensor or a nonempty tuple/list of Tensors")
    if not all(isinstance(item, Tensor) for item in value):
        raise TypeError(f"grad {name} must contain only Tensors")
    return tuple(value), False


def _seeds(outputs, seed, single, engine):
    from .ops import ones_like
    if seed is None:
        if any(value.shape != () for value in outputs):
            raise ValueError("grad requires an explicit seed for nonscalar outputs")
        return tuple(ones_like(value) for value in outputs)
    values, seed_single = _tensors(seed, "seed")
    if single != seed_single or len(values) != len(outputs):
        raise TypeError("grad seed structure must match outputs")
    for value, output in zip(values, outputs):
        engine._check_tensor(value)
        if value.shape != output.shape or value.dtype != output.dtype:
            raise TypeError("grad seed shape and dtype must match its output")
    return values


def _active_graph(outputs, inputs):
    """Find target-dependent edges without recursion or mutable tape state."""
    targets = {_key(value) for value in inputs}
    active, visiting = {}, set()
    pending = [(value, False) for value in reversed(outputs)]
    while pending:
        value, finish = pending.pop()
        key = _key(value)
        if key in active:
            continue
        origin = value._origin
        parents = (() if origin is None else
                   tuple(origin.expression.inputs[index]
                         for index in _needed_inputs(origin.expression.module, origin.index)))
        if finish:
            visiting.remove(key)
            active[key] = key in targets or any(active[_key(parent)] for parent in parents)
            continue
        if key in visiting:
            raise RuntimeError("Tensor expressions contain a cycle")
        visiting.add(key)
        pending.append((value, True))
        pending.extend((parent, False) for parent in reversed(parents))

    ports, edges = {}, {}
    reached = set()
    pending = list(outputs)
    while pending:
        value = pending.pop()
        key = _key(value)
        if key in reached or not active[key]:
            continue
        reached.add(key)
        origin = value._origin
        if origin is None:
            continue
        expression = origin.expression
        selected = tuple(index for index in _needed_inputs(expression.module, origin.index)
                         if active[_key(expression.inputs[index])])
        if not selected:
            continue
        ports.setdefault(expression, set()).add(origin.index)
        edges.setdefault(expression, set()).update(selected)
        pending.extend(expression.inputs[index] for index in selected)

    # All result ports are discovered before sorting expression dependencies:
    # a later output of a shared expression can expose another predecessor.
    complete, visiting, expressions = set(), set(), []
    pending = [(expression, False) for expression in ports]
    while pending:
        expression, finish = pending.pop()
        if expression in complete:
            continue
        if finish:
            visiting.remove(expression)
            complete.add(expression)
            expressions.append(expression)
            continue
        if expression in visiting:
            raise RuntimeError("Tensor expressions contain a cycle")
        visiting.add(expression)
        pending.append((expression, True))
        for index in sorted(edges[expression], reverse=True):
            origin = expression.inputs[index]._origin
            if origin is not None and origin.expression in ports:
                pending.append((origin.expression, False))
    return expressions, ports, edges


def grad(outputs, inputs, *, seed=None):
    """Return ordinary lazy Tensors representing a vector-Jacobian product.

    Scalar outputs default to unit seeds. Nonscalar outputs require matching
    explicit seeds. A Tensor input returns one Tensor; a tuple/list returns a
    tuple in input order. Disconnected inputs produce zeros. The transformation
    preserves source expressions, so its results can be differentiated again.
    """
    require_not_tracing()
    outputs, single_output = _tensors(outputs, "outputs")
    inputs, single_input = _tensors(inputs, "inputs")
    engine = outputs[0].engine
    for value in (*outputs, *inputs):
        engine._check_tensor(value)
        if value.dtype not in FLOAT_DTYPES:
            raise TypeError("grad outputs and inputs must have floating dtype")
    seeds = _seeds(outputs, seed, single_output, engine)
    expressions, ports, edges = _active_graph(outputs, inputs)
    plans = {expression: _plan(expression, tuple(sorted(ports[expression])),
                                tuple(sorted(edges[expression])), engine)
             for expression in expressions}
    from ._lazy import Result, defer_module
    from .ops import add, zeros_like
    adjoints = {}

    def contribute(key, value):
        adjoints[key] = add(adjoints[key], value) if key in adjoints else value

    for output, value in zip(outputs, seeds):
        contribute(_key(output), value)
    for expression in reversed(expressions):
        plan = plans[expression]
        bindings = (*expression.inputs,
                    *(adjoints[Result(expression, index)] for index in sorted(ports[expression])))
        gradients = defer_module(plan.derivative, bindings, engine=engine,
                                 dependencies=plan.dependencies)
        for index, value in zip(sorted(edges[expression]), gradients):
            contribute(_key(expression.inputs[index]), value)
    result = tuple(adjoints[_key(value)] if _key(value) in adjoints else zeros_like(value)
                   for value in inputs)
    return result[0] if single_input else result


def stop_gradient(tensor):
    """Return a numerical alias with no derivative edge to its source."""
    from .tensor import Tensor
    from .tracing import TraceTensor
    if not isinstance(tensor, (Tensor, TraceTensor)):
        raise TypeError("stop_gradient expects a Tensor")
    return tensor.detach()

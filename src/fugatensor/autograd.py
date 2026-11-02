"""Record graph history when tensors are created; ASK executes derivative graphs."""
from __future__ import annotations

from contextlib import ContextDecorator, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import threading
import weakref

from ._state import active_trace, require_not_tracing
from .ir import FLOAT_DTYPES, Module, project_outputs, prune, specialize, vjp

_enabled = ContextVar("fugatensor_grad_enabled", default=True)
_cache = weakref.WeakKeyDictionary()
_dependency_cache = weakref.WeakKeyDictionary()
_cache_lock = threading.RLock()


class no_grad(ContextDecorator):
    """Construct computations without attaching backward history to their outputs."""
    def __init__(self):
        self._tokens = ContextVar(f"fugatensor_no_grad_{id(self)}", default=())

    def __enter__(self):
        require_not_tracing()
        token = _enabled.set(False)
        self._tokens.set((*self._tokens.get(), token))
        return self

    def __exit__(self, *_):
        tokens = self._tokens.get()
        _enabled.reset(tokens[-1])
        self._tokens.set(tokens[:-1])


def is_grad_enabled():
    return _enabled.get()


@dataclass(eq=False)
class _Node:
    module: Module
    inputs: tuple
    output_ids: tuple
    output_specs: tuple
    selected: tuple
    engine: object
    released: bool = False


@dataclass
class _Plan:
    module: Module
    input_indices: tuple
    output_indices: tuple
    derivative: Module
    dependencies: tuple = ()
    higher_derivative: Module = None


def record_execution(module, inputs, outputs, engine):
    """Attach one history node when a graph's stable output handles are created.

    The caller must use the returned tuple: an identity/duplicate return may
    need a new Tensor handle so an input leaf never acquires its own history.
    Capture grad-enabled state now, even if numerical execution happens later.
    Materialization must preserve these handles and must not record them again.
    """
    from .tensor import Tensor
    inputs, outputs = tuple(inputs), tuple(outputs)
    if not _enabled.get():
        return outputs
    selected = tuple(index for index, value in enumerate(inputs)
                     if value.requires_grad and value.dtype in FLOAT_DTYPES)
    if not selected or not any(value.dtype in FLOAT_DTYPES for value in outputs):
        return outputs
    if not isinstance(module, Module) or len(module.functions) != 1:
        raise TypeError("backward history requires one Fuga function")
    source = module.functions[0]
    if len(source.inputs) != len(inputs) or len(source.outputs) != len(outputs):
        raise ValueError("Fuga graph does not match its tensor bindings")
    seen = {id(value) for value in inputs}
    result = []
    for value in outputs:
        if id(value) in seen:
            value = Tensor(engine, value._storage)
        seen.add(id(value))
        result.append(value)
    outputs = tuple(result)
    node = _Node(module, inputs, tuple(id(value) for value in outputs),
                 tuple((value.shape, value.dtype) for value in outputs), selected, engine)
    for index, value in enumerate(outputs):
        if value.dtype in FLOAT_DTYPES and set(selected).intersection(_needed_inputs(module, index)):
            value._requires_grad = True
            value._history, value._history_index = node, index
    return outputs


def _needed_inputs(module, output_index):
    """Use the native liveness pass to exclude unrelated runtime histories."""
    with _cache_lock:
        entries = _dependency_cache.setdefault(module, {})
        if output_index not in entries:
            function = project_outputs(module, (output_index,)).functions[0]
            used = {value for op in function.operations for value in op.operands}
            entries[output_index] = tuple(index for index, value in enumerate(function.inputs.values()) if value in used)
        return entries[output_index]


def _validate_definitions(module):
    """Check custom rules even when their derivative graph is cached."""
    def visit(operations):
        for operation in operations:
            if operation.opcode == "call":
                spec = operation.attrs["spec"]
                spec.verify()
                validate = getattr(spec.backward_rule, "validate_definition", None)
                if validate is not None:
                    validate()
                _validate_definitions(spec.forward)
            for region in operation.regions:
                for block in region.blocks:
                    visit(block.operations)
    for function in module.functions:
        visit(function.operations)


def _plan(node, output_indices, input_indices, *, create_graph=False):
    _validate_definitions(node.module)
    key = input_indices, output_indices
    with _cache_lock:
        by_module = _cache.setdefault(node.engine, weakref.WeakKeyDictionary())
        plans = by_module.setdefault(node.module, {})
        if key not in plans:
            projected = project_outputs(node.module, output_indices)
            if create_graph:
                from .ir.primitives import validate_composable
                validate_composable(projected, phase="forward")
            from .tracing import TraceContext
            context = TraceContext.from_graph(projected.functions[0], engine=node.engine)
            token = active_trace.set(context)
            try:
                derivative = vjp(projected, argnums=input_indices)
            finally:
                active_trace.reset(token)
            plans[key] = _Plan(projected, input_indices, output_indices, derivative,
                               tuple(context.dependencies.items()))
        plan = plans[key]
        _validate_definitions(plan.derivative)
        if any(definition._definition_key() != version for definition, version in plan.dependencies):
            raise ValueError("custom backward definition changed after its graph was traced; retrace the forward call")
        if create_graph and plan.higher_derivative is None:
            from .ir.primitives import validate_composable
            validate_composable(plan.module, phase="forward")
            names = tuple(op.attrs["spec"].name for function in plan.module.functions
                          for op in function.operations if op.opcode == "call")
            phase = "backward" + (" of " + ", ".join(names) if names else "")
            validate_composable(plan.derivative, phase=phase)
            higher = prune(specialize(plan.derivative, expand_calls=False))
            validate_composable(higher, phase="composed backward")
            plan.higher_derivative = higher
        return plan


def _add(left, right):
    from .ops import add
    return right if left is None else add(left, right)


def _seed(tensor, gradient, *, create_graph=False):
    from .tensor import Tensor
    if gradient is None:
        if tensor.shape != ():
            raise ValueError("backward requires an explicit gradient for a nonscalar Tensor")
        return tensor.engine.tensor(1.0, dtype=tensor.dtype)
    if not isinstance(gradient, Tensor):
        raise TypeError("backward gradient must be a Tensor")
    if gradient.engine is not tensor.engine:
        raise ValueError("backward gradient must belong to the same Engine")
    if gradient.shape != tensor.shape or gradient.dtype != tensor.dtype:
        raise TypeError("backward gradient shape and dtype must match the output Tensor")
    tensor.engine._check_tensor(gradient)
    return gradient if create_graph else gradient.detach()


@contextmanager
def _recording(enabled):
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


def backward(tensor, gradient=None, *, retain_graph=None, create_graph=False):
    """Validate reverse graphs, then execute their compiled gradient updates."""
    require_not_tracing()
    if (retain_graph is not None and type(retain_graph) is not bool) or type(create_graph) is not bool:
        raise TypeError("retain_graph must be bool or None; create_graph must be bool")
    if retain_graph is None:
        retain_graph = create_graph
    tensor.engine._check_tensor(tensor)
    if not tensor.requires_grad or tensor.dtype not in FLOAT_DTYPES:
        raise ValueError("backward requires a floating Tensor with requires_grad=True")
    # Validate the seed before building any backward graph or changing .grad.
    if gradient is not None:
        from .tensor import Tensor
        if not isinstance(gradient, Tensor):
            raise TypeError("backward gradient must be a Tensor")
        if gradient.engine is not tensor.engine:
            raise ValueError("backward gradient must belong to the same Engine")
        if gradient.shape != tensor.shape or gradient.dtype != tensor.dtype:
            raise TypeError("backward gradient shape and dtype must match the output Tensor")
        tensor.engine._check_tensor(gradient)
    elif tensor.shape != ():
        raise ValueError("backward requires an explicit gradient for a nonscalar Tensor")

    ports, active_inputs = {}, {}
    leaves = {}
    pending = [tensor]
    while pending:
        value = pending.pop()
        if not value.requires_grad:
            continue
        node = value._history
        if node is None:
            leaves[id(value)] = value
            continue
        if node.released:
            raise RuntimeError("backward graph has been freed; use retain_graph=True on the first backward call")
        node.engine._check_open()
        output_index = value._history_index
        if output_index in ports.setdefault(node, set()):
            continue
        ports[node].add(output_index)
        required = set(node.selected).intersection(_needed_inputs(node.module, output_index))
        new_inputs = required.difference(active_inputs.setdefault(node, set()))
        active_inputs[node].update(required)
        pending.extend(node.inputs[index] for index in sorted(new_inputs, reverse=True))
    # Build every VJP before any reverse compilation/execution. Missing rules
    # cannot leave a partially accumulated leaf gradient.
    # Discovering a second output can introduce an extra predecessor after the
    # first visit. Sort again over the complete set of reached dependency edges.
    complete, visiting, ordered = set(), set(), []
    pending = [(tensor._history, False)] if tensor._history is not None else []
    while pending:
        node, finish = pending.pop()
        if node in complete:
            continue
        if finish:
            visiting.remove(node)
            complete.add(node)
            ordered.append(node)
            continue
        if node in visiting:
            raise RuntimeError('backward history contains a cycle')
        visiting.add(node)
        pending.append((node, True))
        for index in sorted(active_inputs[node], reverse=True):
            parent = node.inputs[index]._history
            if parent is not None:
                pending.append((parent, False))
    plans = {node: _plan(node, tuple(sorted(ports[node])), tuple(sorted(active_inputs[node])),
                         create_graph=create_graph) for node in ordered}
    from ._lazy import defer_module
    with _recording(create_graph):
        adjoints = {id(tensor): _seed(tensor, gradient, create_graph=create_graph)}
        for node in reversed(ordered):
            plan = plans[node]
            seeds = [adjoints[node.output_ids[index]] for index in plan.output_indices]
            bindings = (*node.inputs, *seeds)
            gradients = defer_module(plan.higher_derivative if create_graph else plan.derivative,
                                     bindings, engine=node.engine, dependencies=plan.dependencies)
            if len(gradients) != len(plan.input_indices):
                raise RuntimeError("backward graph returned an invalid gradient count")
            for index, value in zip(plan.input_indices, gradients):
                original = node.inputs[index]
                adjoints[id(original)] = _add(adjoints.get(id(original)), value)
        # Accumulation is part of the same lazy graph. Force every update before
        # committing any .grad so compilation/execution failures stay atomic.
        updates = [(leaf, _add(leaf._grad, adjoints[id_]))
                   for id_, leaf in leaves.items() if id_ in adjoints]
        if updates:
            tensor.engine.evaluate(*(value for _, value in updates))
        for leaf, value in updates:
            leaf._grad = value
    if not retain_graph:
        for node in ordered:
            node.released = True
            node.inputs = ()
            node.module = None


def zero_grad(tensor, *, set_to_none=True):
    require_not_tracing()
    if type(set_to_none) is not bool:
        raise TypeError("set_to_none must be bool")
    if set_to_none:
        tensor._grad = None
    elif tensor._grad is not None:
        from .ops import zeros_like
        with no_grad():
            tensor._grad = zeros_like(tensor._grad)

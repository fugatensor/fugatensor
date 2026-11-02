"""Custom derivatives preserve a function boundary in Fuga IR until AD runs."""
from __future__ import annotations

from contextvars import ContextVar
import hashlib
import inspect
import marshal
from types import MemberDescriptorType

from ._state import active_trace
from .ir import FLOAT_DTYPES
from .ir.core import CallSpec
from ._jit import JITFunction, _static_key, jit
from .tracing import TraceContext, TraceTensor

_active_functions = ContextVar("fugatensor_active_functions", default=())


class FunctionContext:
    """Save explicit SSA residuals in forward and read them in backward."""

    __slots__ = ("_trace", "_saved", "_backward", "_closed")

    def __init__(self, trace, *, saved=None):
        self._trace = trace
        self._saved = () if saved is None else tuple(saved)
        self._backward = saved is not None
        self._closed = False

    def save_for_backward(self, *values):
        if self._backward or self._closed:
            raise RuntimeError("save_for_backward is only available during forward tracing")
        if any(not isinstance(value, TraceTensor) or value.context is not self._trace for value in values):
            raise TypeError("save_for_backward requires tensors from this forward trace")
        # As in the return statement, the last explicit call defines the tuple.
        self._saved = tuple(values)

    @property
    def saved_tensors(self):
        if not self._backward:
            raise RuntimeError("saved_tensors is only available during backward tracing")
        return self._saved


def _method(cls, name):
    descriptor = inspect.getattr_static(cls, name, None)
    if isinstance(descriptor, JITFunction):
        return descriptor.function, descriptor
    if descriptor is None:
        return None, None
    if not inspect.isfunction(descriptor):
        raise TypeError(f"fun.{name} must be an instance method")
    # Keep one internal wrapper per method definition. Repeated lookups must
    # retain its revision identity so queued graphs and derivative plans can
    # detect method replacement and explicit cache invalidation.
    wrapper = jit(descriptor)
    setattr(cls, name, wrapper)
    return descriptor, wrapper


def _configuration_value(value):
    try:
        return _static_key(value)
    except TypeError as error:
        raise TypeError("fun state must contain only immutable DType/device/bool/int/float/str/None values or tuples; "
                        "pass Tensor state as explicit function arguments") from error


def _function_key(function, wrapper):
    if function is None:
        return None
    return (function, function.__code__,
            tuple(_configuration_value(value) for value in (function.__defaults__ or ())),
            tuple(sorted((name, _configuration_value(value))
                         for name, value in (function.__kwdefaults__ or {}).items())),
            wrapper._revision if wrapper is not None else 0)


def _fingerprint(key):
    """The text descriptor records code/configuration, never object reprs."""
    digest = hashlib.sha256()
    configuration, forward, backward = key
    digest.update(repr(configuration).encode())
    for function in (forward, backward):
        if function is None:
            digest.update(b"no-rule")
        else:
            body, code, defaults, kwdefaults, revision = function
            digest.update(f"{body.__module__}.{body.__qualname__}".encode())
            digest.update(marshal.dumps(code))
            digest.update(repr((defaults, kwdefaults, revision)).encode())
    return digest.hexdigest()


def _outputs(result, context, where):
    values = result if isinstance(result, tuple) else (result,)
    if not values or any(not isinstance(value, TraceTensor) for value in values):
        raise TypeError(f"{where} must return a Tensor or a nonempty tuple of Tensors; None is unsupported")
    if any(value.context is not context for value in values):
        raise ValueError(f"{where} results must belong to its current trace")
    return values


class fun:
    """Subclass with a plain forward and an optional explicit backward rule.

    ``backward(self, ctx, dy)`` receives a Tensor or tuple matching forward's
    return structure. It returns a tuple of gradients for floating Tensor
    arguments only, in input order, retaining each input's shape and dtype.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        forward, wrapper = _method(cls, "forward")
        if forward is None:
            raise TypeError("fun subclasses must define forward(self, ctx, ...)")
        parameters = tuple(inspect.signature(forward).parameters.values())
        if len(parameters) < 2 or any(parameter.kind not in
                (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for parameter in parameters[:2]):
            raise TypeError("fun.forward must begin with self and ctx parameters")
        cls.__signature__ = inspect.signature(forward).replace(parameters=parameters[2:])
        backward, backward_wrapper = _method(cls, "backward")
        if backward is not None:
            backward_parameters = tuple(inspect.signature(backward).parameters.values())
            if len(backward_parameters) != 3 or any(parameter.kind not in
                    (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for parameter in backward_parameters):
                raise TypeError("fun.backward must have the signature backward(self, ctx, dy)")

    def _definition_key(self):
        state = {}
        for cls in reversed(type(self).__mro__):
            if cls in (object, fun):
                continue
            for name, value in vars(cls).items():
                if name.startswith("__") or name.startswith("_fuga_"):
                    continue
                if isinstance(value, (JITFunction, property, staticmethod, classmethod)) or callable(value):
                    continue
                if isinstance(value, MemberDescriptorType):
                    if hasattr(self, name):
                        state[name] = getattr(self, name)
                else:
                    state[name] = value
        state.update((name, value) for name, value in vars(self).items() if not name.startswith("_fuga_"))
        configuration = tuple(sorted((name, _configuration_value(value)) for name, value in state.items()))
        forward, forward_wrapper = _method(type(self), "forward")
        if forward is None:
            raise TypeError("fun.forward must remain an instance method")
        backward, backward_wrapper = _method(type(self), "backward")
        return configuration, _function_key(forward, forward_wrapper), _function_key(backward, backward_wrapper)

    def __call__(self, *args, **kwargs):
        if active_trace.get() is not None:
            return self._trace(*args, **kwargs)
        self._definition_key()
        compiled = getattr(self, "_fuga_compiled", None)
        if compiled is None:
            def invoke(*args, **kwargs):
                return self._trace(*args, **kwargs)
            invoke.__signature__ = self.__signature__
            invoke.__name__ = type(self).__name__
            # A custom forward may register external nodes whose static
            # callbacks observe current Python configuration. Capture a fresh
            # CallSpec per invocation; executable caching belongs to Engine.
            compiled = jit(invoke, cache=False)
            object.__setattr__(self, "_fuga_compiled", compiled)
        return compiled(*args, **kwargs)

    def clear_cache(self):
        """Invalidate recorded calls after changing untracked Python captures."""
        compiled = getattr(self, "_fuga_compiled", None)
        if compiled is not None:
            compiled.clear_cache()
        _, wrapper = _method(type(self), "forward")
        wrapper.clear_cache()

    def _trace(self, *args, **kwargs):
        outer = active_trace.get()
        if outer is None:
            raise RuntimeError("fun graph construction requires an active trace")
        definition = self._definition_key()
        outer.depend_on(self)
        if self in _active_functions.get():
            raise ValueError("recursive fun.forward calls are unsupported")
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        tensors = []
        for value in bound.arguments.values():
            if isinstance(value, TraceTensor):
                if value.context is not outer:
                    raise ValueError("fun arguments must belong to the current trace")
                tensors.append(value)
            else:
                _static_key(value)
        if not tensors:
            raise TypeError("a fun call requires at least one Tensor argument")
        context = TraceContext([(value.shape, value.dtype) for value in tensors], engine=outer.engine,
                               name="fun_forward")
        ctx = FunctionContext(context)
        replacements = iter(context.inputs)
        for name, value in tuple(bound.arguments.items()):
            if isinstance(value, TraceTensor):
                bound.arguments[name] = next(replacements)
        forward, _ = _method(type(self), "forward")
        token = active_trace.set(context)
        recursion = _active_functions.set((*_active_functions.get(), self))
        try:
            result = forward(self, ctx, *bound.args, **bound.kwargs)
            visible = _outputs(result, context, "fun.forward")
            module = context.finish((*visible, *ctx._saved))
        finally:
            ctx._closed = True
            _active_functions.reset(recursion)
            active_trace.reset(token)
        if self._definition_key() != definition:
            raise ValueError("fun.forward must not mutate function state while tracing")
        backward, _ = _method(type(self), "backward")
        output_count = len(visible)
        # Preserve tuple structure even when it has a single visible result.
        output_is_tuple = isinstance(result, tuple)
        input_types = tuple(value.value.type for value in tensors)
        name = f"{type(self).__module__}.{type(self).__qualname__}"

        def validate_definition():
            if self._definition_key() != definition:
                raise ValueError("custom function definition changed after its forward graph was traced; retrace the call")

        def build_backward(target_graph, operands, forward_results, cotangents, *, engine=None):
            validate_definition()
            backward_context = TraceContext.from_graph(target_graph, engine=engine)
            backward_context.depend_on(self)
            saved = tuple(backward_context.wrap(value) for value in forward_results[output_count:])
            backward_ctx = FunctionContext(backward_context, saved=saved)
            seeds = tuple(backward_context.wrap(value) for value in cotangents)
            token = active_trace.set(backward_context)
            try:
                gradients = backward(self, backward_ctx, seeds if output_is_tuple else seeds[0])
                if not isinstance(gradients, tuple):
                    raise TypeError("fun.backward must return a tuple of gradients, including for one input; None is unsupported")
                values = _outputs(gradients, backward_context, "fun.backward")
            finally:
                backward_ctx._closed = True
                active_trace.reset(token)
            expected = tuple(type_ for type_ in input_types if type_.dtype in FLOAT_DTYPES)
            if len(values) != len(expected):
                raise ValueError(f"fun.backward must return {len(expected)} gradients for floating Tensor inputs; "
                                 f"received {len(values)}")
            for index, (value, type_) in enumerate(zip(values, expected)):
                if value.value.type != type_:
                    raise TypeError(f"fun.backward gradient {index} shape and dtype must match input {type_}")
            if self._definition_key() != definition:
                raise ValueError("fun.backward must not mutate function state while tracing")
            return tuple(value.value for value in values)

        build_backward.validate_definition = validate_definition
        spec = CallSpec(name=name, forward=module, output_count=output_count,
                        backward_rule=build_backward if backward is not None else None,
                        backward_name=f"{name}.backward" if backward is not None else None,
                        definition_key=_fingerprint(definition))
        results = outer.graph.call(*(value.value for value in tensors), spec=spec)
        results = results if isinstance(results, tuple) else (results,)
        outputs = tuple(outer.wrap(value) for value in results[:output_count])
        return outputs if output_is_tuple else outputs[0]

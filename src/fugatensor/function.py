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
    configuration, forward, vjp = key
    digest.update(repr(configuration).encode())
    for function in (forward, vjp):
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
    """Subclass with a forward and an optional explicit vector-Jacobian product.

    ``forward(self, *args)`` returns a Tensor or tuple of Tensors.
    ``vjp(self, primals, outputs, cotangents)`` receives tuples and returns
    gradients for floating Tensor inputs, in input order and with matching
    shapes/dtypes. Integer/bool outputs have None cotangents. Without a custom
    VJP the forward graph supplies the derivative.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        forward, wrapper = _method(cls, "forward")
        if forward is None:
            raise TypeError("fun subclasses must define forward(self, ...)")
        parameters = tuple(inspect.signature(forward).parameters.values())
        if not parameters or any(parameter.kind not in
                (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for parameter in parameters[:1]):
            raise TypeError("fun.forward must begin with a self parameter")
        cls.__signature__ = inspect.signature(forward).replace(parameters=parameters[1:])
        vjp, vjp_wrapper = _method(cls, "vjp")
        if vjp is not None:
            vjp_parameters = tuple(inspect.signature(vjp).parameters.values())
            if len(vjp_parameters) != 4 or any(parameter.kind not in
                    (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for parameter in vjp_parameters):
                raise TypeError("fun.vjp must have the signature vjp(self, primals, outputs, cotangents)")

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
        vjp, vjp_wrapper = _method(type(self), "vjp")
        return configuration, _function_key(forward, forward_wrapper), _function_key(vjp, vjp_wrapper)

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
        replacements = iter(context.inputs)
        for name, value in tuple(bound.arguments.items()):
            if isinstance(value, TraceTensor):
                bound.arguments[name] = next(replacements)
        forward, _ = _method(type(self), "forward")
        token = active_trace.set(context)
        recursion = _active_functions.set((*_active_functions.get(), self))
        try:
            result = forward(self, *bound.args, **bound.kwargs)
            visible = _outputs(result, context, "fun.forward")
            module = context.finish(visible)
        finally:
            _active_functions.reset(recursion)
            active_trace.reset(token)
        if self._definition_key() != definition:
            raise ValueError("fun.forward must not mutate function state while tracing")
        vjp, _ = _method(type(self), "vjp")
        # Preserve tuple structure even when it has a single visible result.
        output_is_tuple = isinstance(result, tuple)
        input_types = tuple(value.value.type for value in tensors)
        name = f"{type(self).__module__}.{type(self).__qualname__}"

        def validate_definition():
            if self._definition_key() != definition:
                raise ValueError("custom function definition changed after its forward graph was traced; retrace the call")

        def build_vjp(target_graph, operands, forward_results, cotangents, *, engine=None):
            validate_definition()
            vjp_context = TraceContext.from_graph(target_graph, engine=engine)
            vjp_context.depend_on(self)
            inputs = tuple(vjp_context.wrap(value) for value in operands)
            outputs = tuple(vjp_context.wrap(value) for value in forward_results)
            seeds = tuple(None if value is None else vjp_context.wrap(value) for value in cotangents)
            token = active_trace.set(vjp_context)
            try:
                gradients = vjp(self, inputs, outputs, seeds)
                if not isinstance(gradients, tuple):
                    raise TypeError("fun.vjp must return a tuple of gradients, including for one input; None is unsupported")
                values = _outputs(gradients, vjp_context, "fun.vjp")
            finally:
                active_trace.reset(token)
            expected = tuple(type_ for type_ in input_types if type_.dtype in FLOAT_DTYPES)
            if len(values) != len(expected):
                raise ValueError(f"fun.vjp must return {len(expected)} gradients for floating Tensor inputs; "
                                 f"received {len(values)}")
            for index, (value, type_) in enumerate(zip(values, expected)):
                if value.value.type != type_:
                    raise TypeError(f"fun.vjp gradient {index} shape and dtype must match input {type_}")
            if self._definition_key() != definition:
                raise ValueError("fun.vjp must not mutate function state while tracing")
            return vjp_context.export_values(values)

        build_vjp.validate_definition = validate_definition
        spec = CallSpec(name=name, forward=module,
                        vjp_rule=build_vjp if vjp is not None else None,
                        vjp_name=f"{name}.vjp" if vjp is not None else None,
                        definition_key=_fingerprint(definition))
        results = outer.graph.call(*(value.value for value in tensors), spec=spec)
        results = results if isinstance(results, tuple) else (results,)
        outputs = tuple(outer.wrap(value) for value in results)
        return outputs if output_is_tuple else outputs[0]

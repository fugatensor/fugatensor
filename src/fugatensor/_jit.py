"""Private tracing templates for primitive operations and custom functions."""
from __future__ import annotations

from dataclasses import dataclass
from functools import update_wrapper
import inspect
import struct
import threading
import weakref
from types import MethodType

from ._state import active_trace
from .dtypes import DType
from .devices import device as Device
from .tensor import Tensor
from .tracing import TraceContext, TraceTensor


@dataclass(frozen=True)
class _Traced:
    template: object
    is_tuple: bool
    dependencies: tuple


def _static_key(value):
    if isinstance(value, DType):
        return 'dtype', value.value
    if isinstance(value, Device):
        return 'device', value.type, value.index
    if isinstance(value, tuple):
        return 'tuple', tuple(_static_key(item) for item in value)
    if type(value) in (bool, int, str, type(None)):
        return type(value).__name__, value
    if type(value) is float:
        return 'float', struct.pack('!d', value)
    raise TypeError('operation arguments must be Tensor objects or static device/DType/bool/int/float/str/None values or tuples')


class JITFunction:
    def __init__(self, function, *, cache=True):
        if not callable(function):
            raise TypeError('a tracing template requires a Python function')
        self.function = function
        self._signature = inspect.signature(function)
        if any(parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
               for parameter in self._signature.parameters.values()):
            raise TypeError('tracing templates need explicit parameters')
        self._names = tuple(self._signature.parameters)
        self._positional_count = sum(parameter.kind != parameter.KEYWORD_ONLY
                                     for parameter in self._signature.parameters.values())
        self._cache = weakref.WeakKeyDictionary()
        self._cache_enabled = cache
        self._lock = threading.RLock()
        self._revision = 0
        update_wrapper(self, function, updated=())

    def __get__(self, instance, owner=None):
        return self if instance is None else MethodType(self, instance)

    def _definition_key(self):
        function = getattr(self.function, '__func__', self.function)
        definition = getattr(function, '_definition_key', None)
        return self._revision, getattr(function, '__code__', None), definition() if definition else None

    def __call__(self, *args, **kwargs):
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return self._call(tuple(bound.arguments.values()))

    def _invoke(self, values):
        count = self._positional_count
        return self.function(*values[:count], **dict(zip(self._names[count:], values[count:])))

    def _call(self, values):
        outer = active_trace.get()
        if outer is not None:
            for value in values:
                if isinstance(value, Tensor):
                    raise TypeError('real tensor captures are unsupported while tracing; pass tensors as operation arguments')
                if isinstance(value, TraceTensor):
                    if value.context is not outer:
                        raise ValueError('operation arguments must belong to the current trace')
                else:
                    _static_key(value)
            outer.depend_on(self)
            return self._invoke(values)
        if any(isinstance(value, TraceTensor) for value in values):
            raise TypeError('a traced Tensor cannot escape its active trace')
        inputs = tuple(value for value in values if isinstance(value, Tensor))
        if not inputs:
            raise TypeError('an operation requires at least one Tensor argument')
        engine = inputs[0].engine
        for value in inputs:
            engine._check_tensor(value)
        signature = tuple(('tensor', value.shape, value.dtype) if isinstance(value, Tensor)
                           else ('static', _static_key(value)) for value in values)
        with self._lock:
            record = self._cache.get(engine, {}).get(signature) if self._cache_enabled else None
            if record is not None and any(definition._definition_key() != version
                                          for definition, version in record.dependencies):
                record = None
            if record is None:
                context = TraceContext([(value.shape, value.dtype) for value in inputs], engine=engine)
                context.depend_on(self)
                replacements = iter(context.inputs)
                traced = tuple(next(replacements) if isinstance(value, Tensor) else value for value in values)
                token = active_trace.set(context)
                try:
                    context.finish(self._invoke(traced))
                finally:
                    active_trace.reset(token)
                from ._lazy import _template
                record = _Traced(_template(context.module), context.output_is_tuple,
                                 tuple(context.dependencies.items()))
                if self._cache_enabled:
                    self._cache.setdefault(engine, {})[signature] = record
                else:
                    # Configuration callbacks still run on every invocation.
                    # Intern their resulting graph, not the Python call itself.
                    key = ('snapshot', signature, context.output_is_tuple,
                           context.module._structural_key(), frozenset(record.dependencies))
                    record = self._cache.setdefault(engine, {}).setdefault(key, record)
                engine._register_jit(self)
            outputs = record.template.bind(inputs, engine, record.dependencies)
            return outputs if record.is_tuple else outputs[0]

    def _drop_engine(self, engine):
        with self._lock:
            self._cache.pop(engine, None)

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self._revision += 1


def jit(function=None, *, cache=True):
    if isinstance(function, JITFunction):
        return function
    if function is None:
        return lambda function: JITFunction(function, cache=cache)
    return JITFunction(function, cache=cache)


class _PrimitiveFunction(JITFunction):
    """Internal primitive templates with a prepared positional binding layout."""

    def __init__(self, function):
        super().__init__(function)
        self._keyword_defaults = tuple(parameter.default for parameter in self._signature.parameters.values()
                                       if parameter.kind == parameter.KEYWORD_ONLY)
        if any(value is inspect.Parameter.empty for value in self._keyword_defaults):
            raise TypeError('primitive keyword-only arguments require defaults')

    def __call__(self, *args, **kwargs):
        if not kwargs and len(args) == self._positional_count:
            return self._call(args + self._keyword_defaults)
        return super().__call__(*args, **kwargs)


def primitive(function):
    return _PrimitiveFunction(function)

"""Trace tensor operations once per static signature, then execute ASK code."""
from __future__ import annotations

from functools import update_wrapper
import inspect
import struct
import threading
import weakref

from ._state import active_trace

from .tensor import Tensor
from .tracing import TraceContext, TraceTensor


def _static_key(value):
    if type(value) in (bool, int, str, type(None)):
        return type(value).__name__, value
    if type(value) is float:
        return 'float', struct.pack('!d', value)
    raise TypeError('JIT arguments must be Tensor objects or static bool/int/float/str/None values')


class JITFunction:
    def __init__(self, function):
        if not callable(function):
            raise TypeError('jit expects a Python function')
        self.function = function
        self._signature = inspect.signature(function)
        if any(parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
               for parameter in self._signature.parameters.values()):
            raise TypeError('JIT functions need explicit parameters; *args and **kwargs are unsupported')
        self._cache = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()
        self._compilations = self._hits = 0
        self.last_ir = self.last_report = self.last_artifact_dir = None
        self.last_fuga_ir = self.last_ask_ir = None
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        if active_trace.get() is not None:
            raise TypeError('nested @jit calls are unsupported; use undecorated tensor helpers')
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if any(isinstance(value, TraceTensor) for value in bound.arguments.values()):
            raise TypeError('nested @jit calls are not supported; call an undecorated tensor helper while tracing')
        inputs = [value for value in bound.arguments.values() if isinstance(value, Tensor)]
        if not inputs:
            raise TypeError('a JIT call requires at least one Tensor argument')
        engine = inputs[0].engine
        for value in inputs:
            engine._check_tensor(value)
        spec = tuple((name, ('tensor', value.shape, value.dtype) if isinstance(value, Tensor)
                      else ('static', _static_key(value))) for name, value in bound.arguments.items())
        with self._lock:
            record = self._cache.get(engine, {}).get(spec)
            if record is None:
                context = TraceContext([(value.shape, value.dtype) for value in inputs], engine=engine)
                replacements = iter(context.inputs)
                traced = self._signature.bind(*args, **kwargs)
                traced.apply_defaults()
                for name, value in traced.arguments.items():
                    if isinstance(value, Tensor):
                        traced.arguments[name] = next(replacements)
                token = active_trace.set(context)
                try:
                    result = self.function(*traced.args, **traced.kwargs)
                    context.finish(result)
                finally:
                    active_trace.reset(token)
                executable = engine._compile(context)
                record = (executable, context.output_is_tuple, executable.source, context.module.to_text())
                engine._register_jit(self)
                self._cache.setdefault(engine, {})[spec] = record
                self._compilations += 1
            else:
                self._hits += 1
            executable, is_tuple, source, fuga_source = record
            self.last_ir, self.last_report = source, executable.report
            self.last_ask_ir, self.last_fuga_ir = source, fuga_source
            self.last_artifact_dir = executable.artifact_dir
            outputs = engine._execute(executable, inputs)
            return outputs if is_tuple else outputs[0]

    def cache_info(self):
        with self._lock:
            return {'compilations': self._compilations, 'cache_hits': self._hits,
                    'specializations': sum(len(bucket) for bucket in self._cache.values())}

    def _drop_engine(self, engine):
        with self._lock:
            self._cache.pop(engine, None)

    def clear_cache(self):
        """Drop compiled specializations after changing captured Python constants."""
        with self._lock:
            self._cache.clear()
            self._compilations = self._hits = 0
            self.last_ir = self.last_report = self.last_artifact_dir = None
            self.last_fuga_ir = self.last_ask_ir = None


def jit(function=None):
    if function is None:
        return JITFunction
    return JITFunction(function)

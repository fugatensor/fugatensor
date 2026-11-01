"""Trace tensor operations once per static signature, then execute ASK code."""
from __future__ import annotations

from functools import update_wrapper
import inspect
import struct
import threading
import weakref
from types import MethodType
from dataclasses import dataclass, field

from ._state import active_trace

from .tensor import Tensor
from .tracing import TraceContext, TraceTensor
from .ir import Module


@dataclass
class _Compiled:
    executable: object
    module: object
    is_tuple: bool
    dependencies: tuple
    fuga_text: str


@dataclass
class _Dynamic:
    segments: dict = field(default_factory=dict)
    dependencies: tuple = ()


def _valid(record):
    return all(definition._definition_key() == version for definition, version in record.dependencies)


class _Run:
    """One Python invocation, continuing the same frame across graph breaks."""
    def __init__(self, function, engine, inputs, plan):
        self.function, self.engine = function, engine
        self.inputs, self.plan = tuple(inputs), plan
        self.broke = False
        self.last_record = None

    def execute(self, context, result):
        context.finish(result)
        fuga_text = context.module.to_text()
        dependencies = tuple(context.dependencies.items())
        # Identical IR text can carry different custom backward closures.
        # Keep their Python definition identities and versions in the key too.
        key = fuga_text, frozenset(dependencies)
        record = self.plan.segments.get(key)
        token = active_trace.set(None)
        try:
            if record is None or not _valid(record):
                executable = self.engine._compile(context)
                record = _Compiled(executable, context.module, context.output_is_tuple,
                                   dependencies, fuga_text)
                self.plan.segments[key] = record
                self.function._compilations += 1
            else:
                self.function._hits += 1
            outputs = self.engine._execute(record.executable, self.inputs)
            from .autograd import record_execution
            outputs = record_execution(record.module, self.inputs, outputs, self.engine)
        finally:
            active_trace.reset(token)
        self.last_record = record
        self.function._inspect(record)
        return outputs

    def break_graph(self, predicate):
        self.broke = True
        context = predicate.context
        handles = list(context._handles.values())
        by_value = {}
        for handle in handles:
            by_value.setdefault(handle.value, []).append(handle)
        arguments = tuple(context.graph.inputs.values())
        argument_values = set(arguments)
        order = [*arguments, *(value for op in context.graph.operations for value in op.results)]
        carried = [value for value in order if value in by_value]
        computed = [value for value in carried if value not in argument_values]
        concrete = dict(zip(arguments, self.inputs))
        if computed:
            outputs = self.execute(context, tuple(by_value[value][0] for value in computed))
            concrete.update(zip(computed, outputs))
        elif any(op.effects for op in context.graph.operations):
            raise TypeError('graph breaks require an executable pure tensor boundary')
        token = active_trace.set(None)
        try:
            condition = bool(concrete[predicate.value])
        finally:
            active_trace.reset(token)
        self.inputs = tuple(concrete[value] for value in carried)
        # Rebind every live Python alias to fresh block arguments. Old SSA
        # objects remain immutable in cached programs and the backward tape.
        context.module = Module()
        context.graph = context.module.graph('fuga_main', inputs={
            f'arg{index}': value.type for index, value in enumerate(carried)})
        context.input_specs = tuple((value.type.shape, value.type.dtype) for value in carried)
        context.inputs = ()  # do not keep dead loop temporaries alive ourselves
        context.output_specs = context.output_is_tuple = None
        context._finished = False
        context.external_count = 0
        for value, replacement in zip(carried, context.graph.inputs.values()):
            for handle in by_value[value]:
                handle.value = replacement
        return condition


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
        self._revision = 0
        self.last_ir = self.last_report = self.last_artifact_dir = None
        self.last_fuga_ir = self.last_ask_ir = None
        self.last_segments = []
        update_wrapper(self, function, updated=())

    def __get__(self, instance, owner=None):
        return self if instance is None else MethodType(self, instance)

    def _definition_key(self):
        function = getattr(self.function, '__func__', self.function)
        definition = getattr(function, '_definition_key', None)
        return self._revision, getattr(function, '__code__', None), definition() if definition else None

    def _inspect(self, record):
        executable = record.executable
        self.last_ir = self.last_ask_ir = executable.source
        self.last_fuga_ir = record.fuga_text
        self.last_report, self.last_artifact_dir = executable.report, executable.artifact_dir
        self.last_segments.append({'fuga_ir': self.last_fuga_ir, 'ask_ir': self.last_ask_ir,
                                   'report': executable.report, 'artifact_dir': executable.artifact_dir})

    def __call__(self, *args, **kwargs):
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        outer = active_trace.get()
        if outer is not None:
            for value in bound.arguments.values():
                if isinstance(value, Tensor):
                    raise TypeError('nested JIT cannot use real tensor captures while tracing; pass tensors as JIT arguments')
                if isinstance(value, TraceTensor):
                    if value.context is not outer:
                        raise ValueError('nested JIT arguments must belong to the current trace')
                else:
                    _static_key(value)
            outer.depend_on(self)
            return self.function(*bound.args, **bound.kwargs)
        if any(isinstance(value, TraceTensor) for value in bound.arguments.values()):
            raise TypeError('a traced Tensor cannot escape its active trace')
        inputs = [value for value in bound.arguments.values() if isinstance(value, Tensor)]
        if not inputs:
            raise TypeError('a JIT call requires at least one Tensor argument')
        engine = inputs[0].engine
        for value in inputs:
            engine._check_tensor(value)
        spec = tuple((name, ('tensor', value.shape, value.dtype) if isinstance(value, Tensor)
                      else ('static', _static_key(value))) for name, value in bound.arguments.items())
        with self._lock:
            self.last_segments = []
            record = self._cache.get(engine, {}).get(spec)
            if record is not None and not _valid(record):
                record = None
            if isinstance(record, _Compiled):
                self._hits += 1
                self._inspect(record)
                outputs = engine._execute(record.executable, inputs)
                from .autograd import record_execution
                outputs = record_execution(record.module, inputs, outputs, engine)
                return outputs if record.is_tuple else outputs[0]
            else:
                plan = record if isinstance(record, _Dynamic) else _Dynamic()
                context = TraceContext([(value.shape, value.dtype) for value in inputs], engine=engine)
                context.depend_on(self)
                run = _Run(self, engine, inputs, plan)
                context._runtime = run
                replacements = iter(context.inputs)
                traced = self._signature.bind(*args, **kwargs)
                traced.apply_defaults()
                for name, value in traced.arguments.items():
                    if isinstance(value, Tensor):
                        traced.arguments[name] = next(replacements)
                token = active_trace.set(context)
                try:
                    result = self.function(*traced.args, **traced.kwargs)
                    outputs = run.execute(context, result)
                finally:
                    active_trace.reset(token)
                if run.broke:
                    dependencies = dict(plan.dependencies)
                    dependencies.update(context.dependencies)
                    plan.dependencies = tuple(dependencies.items())
                    record = plan
                else:
                    record = run.last_record
                engine._register_jit(self)
                self._cache.setdefault(engine, {})[spec] = record
                return outputs if context.output_is_tuple else outputs[0]

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
            self._revision += 1
            self._compilations = self._hits = 0
            self.last_ir = self.last_report = self.last_artifact_dir = None
            self.last_fuga_ir = self.last_ask_ir = None
            self.last_segments = []


def jit(function=None):
    if isinstance(function, JITFunction):
        return function
    if function is None:
        return JITFunction
    return JITFunction(function)

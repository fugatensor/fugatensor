"""Resource ownership and compilation through the installed ASK engine."""
from __future__ import annotations

import atexit
import contextvars
import json
from pathlib import Path
import os
import tempfile
import threading
import weakref

from fugatensor._ask import Compiler, ExecutionSession

from ._state import require_not_tracing
from ._config import runtime_defaults
from .dtypes import fp32
from .devices import normalize_device as _device_name, resolve_device
from .backends.host import HostBackend, HostStorage
from .tensor import Tensor, normalize_data, normalize_dtype
from .lowering import lower_to_ask
from .ir import TensorType, prune, specialize
from ._buffers import contiguous_bytes

_current = contextvars.ContextVar('fugatensor_engine', default=None)
_default_device = contextvars.ContextVar('fugatensor_default_device', default='cpu')
_defaults = {}
_default_lock = threading.Lock()


def _environment_options(device, *, include_compilers=True, use_config=True):
    options = runtime_defaults(device) if use_config and not os.environ.get('ASK_OPT') else {}
    if not include_compilers:
        options.pop('compilers', None)
    for name, variable in (('ask_opt', 'ASK_OPT'), ('cache_dir', 'FUGATENSOR_CACHE_DIR'),
                           ('llc', 'FUGATENSOR_LLC'), ('ptxas', 'FUGATENSOR_PTXAS'),
                           ('llvm_link', 'FUGATENSOR_LLVM_LINK'), ('libdevice', 'FUGATENSOR_LIBDEVICE')):
        if os.environ.get(variable):
            options[name] = os.environ[variable]
    if device != 'cpu' and include_compilers:
        from .triton import environment_from_env
        environment = environment_from_env(use_installed=(
            use_config and not options.get('ask_opt') and 'triton' not in options.get('compilers', {})))
        if environment is not None:
            options.setdefault('compilers', {})['triton'] = environment
    return options


def set_default_device(device):
    """Select the device for future tensors in this context, without initializing it.

    Explicit devices and Engine contexts take precedence. Existing tensors and
    engines retain their device and lifetime.
    """
    require_not_tracing()
    _default_device.set(_device_name(device))


def get_default_device():
    """Return the resolved device object, including any active Engine context."""
    active = _current.get()
    return resolve_device(active.device if active is not None else _default_device.get())


def current_engine(device=None):
    active = _current.get()
    selected = (_device_name(device) if device is not None
                else active.device if active is not None else _default_device.get())
    if active is not None and active.device == selected:
        active._check_open()
        return active
    with _default_lock:
        engine = _defaults.get(selected)
        if engine is None or engine.closed:
            engine = Engine(device=selected, **_environment_options(selected))
            _defaults[selected] = engine
        return engine


def configure(device='cpu', **options):
    """Configure a device once; tensor(..., device=...) then uses that engine.

    Replacing a configuration closes its old default engine and its tensors.
    Without explicit configuration the same paths can be provided by environment.
    This does not change the device selected by set_default_device().
    """
    require_not_tracing()
    device = _device_name(device)
    settings = _environment_options(device, include_compilers='compilers' not in options,
                                    use_config=options.get('ask_opt') is None)
    settings.update(options)
    engine = Engine(device=device, **settings)
    with _default_lock:
        previous = _defaults.get(device)
        _defaults[device] = engine
    if previous is not None:
        previous.close()
    return engine


def shutdown(device=None):
    """Close default engines without changing the default device selection.

    Also called at normal process exit. Explicit Engines own their lifetime.
    """
    require_not_tracing()
    with _default_lock:
        if device is None:
            engines = list(_defaults.values())
            _defaults.clear()
        else:
            engine = _defaults.pop(_device_name(device), None)
            engines = [engine] if engine is not None else []
    for engine in engines:
        engine.close()


atexit.register(shutdown)


def synchronize(device=None):
    """Wait for submitted work on the selected device without evaluating pending graphs."""
    require_not_tracing()
    current_engine(device).synchronize()


class Engine:
    """Own an ASK compiler and, on CUDA, one resident execution session."""

    def __init__(self, device='cpu', *, ask_opt=None, cache_dir=None,
                 compilers=None, llc=None, ptxas=None, llvm_link=None, libdevice=None, cc='cc'):
        device = _device_name(device)
        # An explicit compiler path selects a separate installation. Do not
        # attach this environment's saved workers or SDK to that installation.
        if ask_opt is None:
            defaults = _environment_options(device, include_compilers=compilers is None)
            ask_opt = defaults.get('ask_opt')
            llc = llc if llc is not None else defaults.get('llc')
            ptxas = ptxas if ptxas is not None else defaults.get('ptxas')
            llvm_link = llvm_link if llvm_link is not None else defaults.get('llvm_link')
            libdevice = libdevice if libdevice is not None else defaults.get('libdevice')
            cache_dir = cache_dir if cache_dir is not None else defaults.get('cache_dir')
            compilers = compilers if compilers is not None else defaults.get('compilers')
        ordinal = None if device == 'cpu' else (0 if device == 'cuda' else int(device[5:]))
        self.device = device
        self._ordinal = ordinal
        self.compiler = Compiler(executable=ask_opt)
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else base / 'fugatensor'
        self.cache_dir = self.cache_dir.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compilers = dict(compilers or {})
        self.llc, self.ptxas = llc, ptxas
        self.llvm_link, self.libdevice = llvm_link, libdevice
        self._host = HostBackend(self.compiler, self.cache_dir / 'host', cc=cc) if ordinal is None else None
        self._session = None
        self._closed = False
        self._lock = threading.RLock()
        self._operations = {}
        self._graph_cache = {}
        self._jit_functions = weakref.WeakSet()
        self._tokens = contextvars.ContextVar(f'fugatensor_context_{id(self)}', default=())
        self._stats = {'compilations': 0, 'executions': 0, 'cache_hits': 0}
        self.last_ir = self.last_ask_ir = self.last_fuga_ir = None
        self.last_report = self.last_artifact_dir = None
        self.last_execution_plan = None

    @property
    def closed(self):
        return self._closed

    def _check_open(self):
        if self._closed:
            raise RuntimeError('FugaTensor Engine is closed')

    @property
    def session(self):
        with self._lock:
            self._check_open()
            if self._ordinal is None:
                raise ValueError('CPU engine has no CUDA session')
            if self._session is None:
                self._session = ExecutionSession(device=self._ordinal)
            return self._session

    @property
    def capability(self):
        return self.session.capability

    @property
    def stats(self):
        with self._lock:
            result = dict(self._stats)
            if self._session is not None:
                result['runtime'] = self._session.stats
            return result

    def tensor(self, values, *, dtype=fp32, shape=None):
        require_not_tracing()
        flat, shape = normalize_data(values, shape=shape)
        dtype = normalize_dtype(dtype)
        with self._lock:
            self._check_open()
            storage = (HostStorage.from_flat(flat, shape=shape, dtype=dtype)
                       if self._ordinal is None else self.session.from_host(flat, shape=shape, dtype=dtype))
            return Tensor(self, storage)

    def to_flat(self, tensor):
        require_not_tracing()
        with self._lock:
            self._check_tensor(tensor)
            return tensor.storage.to_flat() if self._ordinal is None else tensor.storage.to_host()

    def from_buffer(self, data, *, dtype, shape, copy=True):
        """Import contiguous elements or raw bytes without Python element conversion."""
        require_not_tracing()
        with self._lock:
            self._check_open()
            if self._ordinal is None:
                storage = HostStorage.from_buffer(data, shape=shape, dtype=normalize_dtype(dtype), copy=copy)
            else:
                if type(copy) is not bool:
                    raise TypeError('copy must be bool')
                if not copy:
                    raise ValueError('CUDA from_buffer requires copy=True for host-to-device transfer')
                tensor = TensorType(tuple(shape), dtype)
                source = contiguous_bytes(data, shape=tensor.shape, dtype=tensor.dtype)
                storage = self.session.tensor(tensor.shape, tensor.dtype).copy_from_buffer(source)
            return Tensor(self, storage)

    def _check_tensor(self, tensor):
        self._check_open()
        if not isinstance(tensor, Tensor):
            raise ValueError('tensor belongs to a different FugaTensor Engine')
        if tensor.engine.device != self.device:
            raise RuntimeError('expected all tensors on the same device, got '
                               f'{resolve_device(self.device)} and {resolve_device(tensor.engine.device)}')
        if tensor.engine is not self:
            raise ValueError('tensor belongs to a different FugaTensor Engine')

    def elementwise(self, operation, operands, *, dtype=None):
        # Primitive templates create deferred nodes; evaluation fuses the DAG.
        require_not_tracing()
        from ._jit import primitive
        key = (operation, dtype)
        with self._lock:
            if key not in self._operations:
                if operation == 'add':
                    function = lambda lhs, rhs: lhs + rhs
                elif operation == 'mul':
                    function = lambda lhs, rhs: lhs * rhs
                elif operation == 'cast':
                    function = lambda value: value.astype(dtype)
                else:
                    raise ValueError(f'unsupported operation: {operation}')
                self._operations[key] = primitive(function)
            function = self._operations[key]
        return function(*operands)

    def evaluate(self, *tensors):
        """Compile and materialize the requested Tensor graphs together."""
        require_not_tracing()
        from ._lazy import materialize
        with self._lock:
            self._check_open()
            return materialize(self, tensors)

    def cache_info(self):
        with self._lock:
            return {'compilations': self._stats['compilations'],
                    'cache_hits': self._stats['cache_hits'],
                    'specializations': len(self._graph_cache)}

    def clear_cache(self):
        """Drop compiled graphs; already materialized tensors remain valid."""
        with self._lock:
            self._check_open()
            self._graph_cache.clear()
            self.last_ir = self.last_ask_ir = self.last_fuga_ir = None
            self.last_report = self.last_artifact_dir = None
            self.last_execution_plan = None

    def _compile(self, context):
        with self._lock:
            self._check_open()
            target = {'backend': 'host'}
            if self._ordinal is not None:
                major, minor = self.session.capability
                target = {'backend': 'cuda', 'arch': major * 10 + minor}
            source_map = {}
            flat = prune(specialize(context.module))
            # Autograd supplies saved inputs and cotangents. Remove unused
            # physical parameters while retaining the Python call signature.
            from .ir import Module
            from .ir.transforms import _clone_operation
            function = flat.functions[0]
            arguments = tuple(function.inputs.items())
            used = {value for op in function.operations for value in op.operands}
            input_indices = tuple(index for index, (_, value) in enumerate(arguments) if value in used)
            if not input_indices and arguments:
                # Keep one carrier for constant-only kernels and existing ABI.
                output_type = function.outputs[0].type
                input_indices = (next((index for index, (_, value) in enumerate(arguments)
                                       if value.type == output_type), 0),)
            if len(input_indices) != len(arguments):
                projected = Module()
                graph = projected.graph(function.name,
                    inputs={arguments[index][0]: arguments[index][1].type for index in input_indices},
                    results=[value.type for value in function.outputs])
                values = {arguments[index][1]: graph.inputs[arguments[index][0]] for index in input_indices}
                for op in function.operations:
                    if op.opcode == 'return':
                        graph.return_(*(values[value] for value in op.operands))
                    else:
                        copied = _clone_operation(graph, op, tuple(values[value] for value in op.operands))
                        values.update(zip(op.results, copied.results))
                flat = projected.verify()
            external_count = sum(op.opcode == 'external' for function in flat.functions for op in function.operations)
            module = lower_to_ask(flat, target=target, source_map=source_map)
            source_map['input_bindings'] = [
                {'argument': index, 'compiled_argument': compiled_index,
                 'origin_value_id': arguments[index][1].origin_id}
                for compiled_index, index in enumerate(input_indices)]
            origins = {arguments[index][1].name: arguments[index][1].origin_id for index in input_indices}
            for row in source_map['values']:
                if row['fuga_value'] in origins:
                    row['origin_value_id'] = origins[row['fuga_value']]
            source = module.to_mlir()
            if self._ordinal is None:
                if external_count:
                    raise NotImplementedError('Triton operations require a CUDA Engine')
                executable = self._host.compile(source)
            else:
                artifacts = self.cache_dir / 'artifacts'
                artifacts.mkdir(exist_ok=True)
                directory = Path(tempfile.mkdtemp(prefix='cuda-', dir=artifacts))
                options = {'target': target}
                kernel = self.compiler.compile_cuda(source, compilers=self.compilers,
                    llc=self.llc, ptxas=self.ptxas, llvm_link=self.llvm_link, libdevice=self.libdevice, cache_dir=self.cache_dir / 'ask', output_dir=directory,
                    **options)
                executable = _CUDAExecutable(kernel, self.session, directory, source)
            (executable.artifact_dir / 'fuga.ir').write_text(context.module.to_text(), encoding='utf-8')
            (executable.artifact_dir / 'frontend.mlir').write_text(source, encoding='utf-8')
            (executable.artifact_dir / 'fuga-to-ask.json').write_text(
                json.dumps(source_map, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            self._stats['compilations'] += 1
            return _InputProjection(executable, input_indices, len(arguments))

    def _execute(self, executable, inputs):
        with self._lock:
            for value in inputs:
                self._check_tensor(value)
            self._check_open()
            outputs = executable.run([value.storage for value in inputs])
            self._stats['executions'] += 1
            return tuple(Tensor(self, value) for value in outputs)

    def synchronize(self):
        with self._lock:
            self._check_open()
            if self._session is not None:
                self._session.synchronize()

    def _register_jit(self, function):
        with self._lock:
            self._check_open()
            self._jit_functions.add(function)

    def close(self):
        # Callers acquire the JIT lock before this lock. Evict after releasing
        # it, so a concurrent call cannot deadlock against Engine.close().
        with self._lock:
            if self._closed:
                return
            if self._session is not None:
                self._session.close()
            self._closed = True
            functions = tuple(self._jit_functions)
            self._jit_functions.clear()
            self._operations.clear()
            self._graph_cache.clear()
            self.last_execution_plan = None
        for function in functions:
            function._drop_engine(self)

    def __enter__(self):
        self._check_open()
        token = _current.set(self)
        self._tokens.set((*self._tokens.get(), token))
        return self

    def __exit__(self, *_):
        tokens = self._tokens.get()
        _current.reset(tokens[-1])
        self._tokens.set(tokens[:-1])
        if len(tokens) == 1:
            self.close()


class _CUDAExecutable:
    def __init__(self, kernel, session, directory, source):
        self.kernel = kernel
        self.prepared = kernel.prepare(session, graph=True)
        self.source, self.report, self.artifact_dir = source, kernel.report, directory
        self.execution_plan = kernel.execution_plan

    def run(self, inputs):
        return self.prepared.run(inputs)


class _InputProjection:
    """Bind the original Fuga signature to the live compiled input ports."""
    def __init__(self, executable, indices, arity):
        self.executable, self.indices, self.arity = executable, indices, arity
        self.source, self.report = executable.source, executable.report
        self.artifact_dir, self.execution_plan = executable.artifact_dir, executable.execution_plan

    def run(self, inputs):
        if len(inputs) != self.arity:
            raise ValueError('input count differs from the original Fuga graph')
        return self.executable.run([inputs[index] for index in self.indices])


def evaluate(*tensors):
    """Materialize tensors together, returning the same handles."""
    if not tensors or any(not isinstance(value, Tensor) for value in tensors):
        raise TypeError('evaluate requires at least one Tensor')
    results = tensors[0].engine.evaluate(*tensors)
    return results[0] if len(results) == 1 else results


def _select_engine(engine, device):
    require_not_tracing()
    selected = engine if engine is not None else current_engine(device)
    if device is not None and _device_name(device) != selected.device:
        raise ValueError('device differs from the selected Engine; create an Engine for that device')
    return selected


def tensor(data, *, dtype=fp32, engine=None, device=None):
    selected = _select_engine(engine, device)
    return selected.tensor(data, dtype=dtype)


def from_buffer(data, *, dtype, shape, copy=True, engine=None, device=None):
    selected = _select_engine(engine, device)
    return selected.from_buffer(data, dtype=dtype, shape=shape, copy=copy)

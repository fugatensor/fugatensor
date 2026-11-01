"""Resource ownership and compilation through the installed ASK engine."""
from __future__ import annotations

import contextvars
import json
from pathlib import Path
import os
import tempfile
import threading
import weakref

from ask import Compiler, ExecutionSession

from ._state import require_not_tracing
from ._config import runtime_defaults
from .backends.host import HostBackend, HostStorage
from .tensor import Tensor, normalize_data, normalize_dtype
from .lowering import lower_to_ask
from .ir import prune, specialize

_current = contextvars.ContextVar('fugatensor_engine', default=None)
_defaults = {}
_default_lock = threading.Lock()


def _device_name(device):
    if device in ('cpu', 'cuda'):
        return device
    if isinstance(device, str) and device.startswith('cuda:') and device[5:].isdigit():
        ordinal = int(device[5:])
        return 'cuda' if ordinal == 0 else f'cuda:{ordinal}'
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<ordinal>'")


def _environment_options(device, *, include_compilers=True, use_config=True):
    options = runtime_defaults(device) if use_config and not os.environ.get('ASK_OPT') else {}
    if not include_compilers:
        options.pop('compilers', None)
    for name, variable in (('ask_opt', 'ASK_OPT'), ('cache_dir', 'FUGATENSOR_CACHE_DIR'),
                           ('llc', 'FUGATENSOR_LLC'), ('ptxas', 'FUGATENSOR_PTXAS')):
        if os.environ.get(variable):
            options[name] = os.environ[variable]
    if device != 'cpu' and include_compilers:
        from .triton import environment_from_env
        environment = environment_from_env()
        if environment is not None:
            options.setdefault('compilers', {})['triton'] = environment
    return options


def current_engine(device=None):
    active = _current.get()
    selected = _device_name(device) if device is not None else (active.device if active else 'cpu')
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
    """Wait for and close default engines. Explicit Engine contexts own themselves."""
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


class Engine:
    """Own an ASK compiler and, on CUDA, one resident execution session."""

    def __init__(self, device='cpu', *, ask_opt=None, cache_dir=None,
                 compilers=None, llc=None, ptxas=None, cc='cc'):
        device = _device_name(device)
        # An explicit compiler path selects a separate installation. Do not
        # attach this environment's saved workers or SDK to that installation.
        if ask_opt is None:
            defaults = _environment_options(device, include_compilers=compilers is None)
            ask_opt = defaults.get('ask_opt')
            llc = llc if llc is not None else defaults.get('llc')
            ptxas = ptxas if ptxas is not None else defaults.get('ptxas')
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
        self._host = HostBackend(self.compiler, self.cache_dir / 'host', cc=cc) if ordinal is None else None
        self._session = None
        self._closed = False
        self._lock = threading.RLock()
        self._operations = {}
        self._jit_functions = weakref.WeakSet()
        self._tokens = contextvars.ContextVar(f'fugatensor_context_{id(self)}', default=())
        self._stats = {'compilations': 0, 'executions': 0}

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

    def tensor(self, values, *, dtype='f32', shape=None, requires_grad=False):
        require_not_tracing()
        flat, shape = normalize_data(values, shape=shape)
        dtype = normalize_dtype(dtype)
        with self._lock:
            self._check_open()
            storage = (HostStorage.from_flat(flat, shape=shape, dtype=dtype)
                       if self._ordinal is None else self.session.from_host(flat, shape=shape, dtype=dtype))
            return Tensor(self, storage, requires_grad=requires_grad)

    def to_flat(self, tensor):
        require_not_tracing()
        with self._lock:
            self._check_tensor(tensor)
            return tensor.storage.to_flat() if self._ordinal is None else tensor.storage.to_host()

    def _check_tensor(self, tensor):
        self._check_open()
        if not isinstance(tensor, Tensor) or tensor.engine is not self:
            raise ValueError('tensor belongs to a different FugaTensor Engine')

    def elementwise(self, operation, operands, *, dtype=None):
        # Eager syntax is a tiny JIT function too. @jit groups a whole expression
        # so its intermediate tensors need not cross the Python execution API.
        require_not_tracing()
        from .jit import jit
        for value in operands:
            if isinstance(value, Tensor):
                self._check_tensor(value)
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
                self._operations[key] = jit(function)
            function = self._operations[key]
        return function(*operands)

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
                executable = self._host.compile(module)
            else:
                if external_count:
                    if len(input_indices) != 1 or len(context.output_specs) != 1 or external_count != 1:
                        raise NotImplementedError('ASK external CUDA fusion currently requires one tensor input, '
                                                  'one output and one external operation')
                    if context.output_specs[0] != context.input_specs[input_indices[0]]:
                        raise NotImplementedError('ASK external CUDA fusion requires equal input/output '
                                                  'shape and storage dtype')
                if self.llc is None or self.ptxas is None:
                    raise ValueError('CUDA compilation requires explicit llc and ptxas paths')
                artifacts = self.cache_dir / 'artifacts'
                artifacts.mkdir(exist_ok=True)
                directory = Path(tempfile.mkdtemp(prefix='cuda-', dir=artifacts))
                options = {}
                if not external_count:
                    options['target'] = target
                kernel = self.compiler.compile_cuda(module, compilers=self.compilers,
                    llc=self.llc, ptxas=self.ptxas, cache_dir=self.cache_dir / 'ask', output_dir=directory,
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
        self.prepared = kernel.prepare(session)
        self.session = session
        self.source, self.report, self.artifact_dir = source, kernel.report, directory

    def run(self, inputs):
        if len(inputs) != len(self.prepared.inputs):
            raise ValueError('CUDA input count differs from the compiled graph')
        outputs = [self.session.tensor(port['shape'], port['element_type'])
                   for port in self.prepared.outputs]
        self.prepared.submit_many(inputs, outputs)
        return outputs


class _InputProjection:
    """Bind the original Fuga signature to the live compiled input ports."""
    def __init__(self, executable, indices, arity):
        self.executable, self.indices, self.arity = executable, indices, arity

    def __getattr__(self, name):
        return getattr(self.executable, name)

    def run(self, inputs):
        if len(inputs) != self.arity:
            raise ValueError('input count differs from the original Fuga graph')
        return self.executable.run([inputs[index] for index in self.indices])


def tensor(values, *, dtype='f32', shape=None, engine=None, device=None, requires_grad=False):
    require_not_tracing()
    selected = engine if engine is not None else current_engine(device)
    if device is not None and _device_name(device) != selected.device:
        raise ValueError('device differs from the selected Engine; create an Engine for that device')
    return selected.tensor(values, dtype=dtype, shape=shape, requires_grad=requires_grad)

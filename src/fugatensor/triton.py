"""Record user-written Triton implementations in device-independent Fuga IR."""
from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import textwrap
import symtable
import types

from fugatensor._ask import CompilerEnvironment

from ._jit import jit
from .dtypes import DType
from .ir import ExternalSpec, TensorType
from .tensor import Tensor, normalize_dtype, normalize_shape
from .tracing import TraceTensor

_SIGNATURE = {'bool': '*i1', 'i32': '*i32', 'i64': '*i64', 'f32': '*fp32', 'f16': '*fp16', 'bf16': '*bf16'}
_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z', re.ASCII)


def _identifier(value, field):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f'{field} must be an ASCII identifier')
    return value


def _constant(value):
    """Serialize constexpr data without importing the selected compiler."""
    if isinstance(value, DType):
        names = {'bool': 'int1', 'i32': 'int32', 'i64': 'int64',
                 'f32': 'float32', 'f16': 'float16', 'bf16': 'bfloat16'}
        return {'__triton_dtype__': names[value]}
    if type(value).__module__.startswith('triton.language') and type(value).__name__ == 'constexpr':
        return _constant(value.value)
    if type(value).__module__.startswith('triton.language') and type(value).__name__ == 'dtype':
        return {'__triton_dtype__': value.name}
    if value is None or type(value) in (bool, str):
        return value
    if type(value) is int and -(2**63) <= value < 2**64:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_constant(item) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _constant(item) for key, item in value.items()}
    raise TypeError('constexpr data must be finite JSON values or a FugaTensor/Triton dtype')


def _constants(values):
    if not isinstance(values, Mapping):
        raise TypeError('constants must be a mapping or a callback returning a mapping')
    return {_identifier(name, 'constant name'): _constant(value) for name, value in values.items()}


def _scalars(values):
    if not isinstance(values, Mapping):
        raise TypeError('scalars must map parameter names to (ABI type, value) pairs')
    result = {}
    for name, pair in values.items():
        _identifier(name, 'scalar name')
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError('scalar bindings require (ABI type, value) pairs')
        dtype, value = pair
        if isinstance(dtype, DType):
            dtype = _SIGNATURE[dtype][1:]
        if not isinstance(dtype, str) or type(value) not in (bool, int, float) or (type(value) is float and not math.isfinite(value)):
            raise TypeError("scalar bindings require an ABI type and a finite numeric value")
        result[name] = (dtype, value)
    return result


def _decorator_kind(node, aliases):
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr
    return aliases.get(target.id, target.id) if isinstance(target, ast.Name) else ''


def _launch_grid(value):
    if type(value) is int:
        value = (value,)
    try:
        result = tuple(value)
    except TypeError as error:
        raise TypeError('grid must contain one to three positive integer dimensions') from error
    if not 1 <= len(result) <= 3 or any(type(n) is not int or not 0 < n < 2**31 for n in result):
        raise ValueError('grid must contain one to three positive signed 32-bit integer dimensions')
    return result + (1,) * (3 - len(result))


@dataclass(frozen=True)
class TensorSpec:
    """Static specialization metadata; it never contains tensor storage or data."""
    shape: tuple[int, ...]
    dtype: DType
    device: str

    def __post_init__(self):
        object.__setattr__(self, 'dtype', DType(normalize_dtype(self.dtype)))

    @property
    def numel(self):
        return math.prod(self.shape)


class TritonKernel:
    """A snapshot of user source plus explicit ASK parameter and launch bindings.

    Registration does not import or execute the source. FugaTensor's Triton
    worker compiles it later for independent ASK scheduling. Re-register after changing source,
    then call the new registration to record new expressions.
    """
    def __init__(self, source, *, entry, input=None, inputs=None, output=None, outputs=None,
                 grid, constants=None, scalars=None, options=None, config=None,
                 source_path=None, output_shape=None, output_dtype=None):
        if not isinstance(source, str) or not source.strip():
            raise TypeError('Triton source must be a nonempty string')
        self._entry = _identifier(entry, 'entry')
        if (input is None) == (inputs is None):
            raise ValueError('provide exactly one of input or inputs')
        self._single_input = input is not None
        inputs = (input,) if inputs is None else inputs
        if isinstance(inputs, str):
            raise TypeError('inputs must be a sequence of source parameter names')
        self._inputs = tuple(_identifier(name, 'input parameter') for name in inputs)
        if not self._inputs or len(set(self._inputs)) != len(self._inputs):
            raise ValueError('inputs must contain distinct source parameter names')
        if (output is None) == (outputs is None):
            raise ValueError('provide exactly one of output or outputs')
        self._single_output = output is not None
        if outputs is None:
            outputs = {output: {'shape': output_shape, 'dtype': output_dtype}}
        elif output_shape is not None or output_dtype is not None:
            raise ValueError('use per-output shape/dtype metadata with outputs')
        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError('outputs must map parameter names to shape/dtype metadata')
        self._outputs = {}
        for name, metadata in outputs.items():
            _identifier(name, 'output parameter')
            if name in self._inputs:
                raise ValueError('input and output must bind different source parameters')
            if not isinstance(metadata, Mapping) or set(metadata) - {'shape', 'dtype'}:
                raise TypeError('each output accepts shape and dtype metadata')
            shape, dtype = metadata.get('shape'), metadata.get('dtype')
            self._outputs[name] = (shape if callable(shape) or shape is None else normalize_shape(shape),
                                   dtype if callable(dtype) or dtype is None else normalize_dtype(dtype))
        tree = ast.parse(source, filename=str(source_path or '<triton-source>'))
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry]
        if len(matches) != 1:
            raise ValueError('source must define the requested entry exactly once')
        args = matches[0].args
        if args.vararg is not None or args.kwarg is not None:
            raise ValueError('Triton entry requires explicit parameter names')
        parameters = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        if not set((*self._inputs, *self._outputs)) <= parameters:
            raise ValueError('input/output binding names must exist in the source entry parameters')
        self._parameter_names = parameters - {*self._inputs, *self._outputs}
        defaults = dict(zip([arg.arg for arg in (*args.posonlyargs, *args.args)][-len(args.defaults):],
                            args.defaults)) if args.defaults else {}
        defaults.update({arg.arg: value for arg, value in zip(args.kwonlyargs, args.kw_defaults) if value is not None})
        self._default_names = set(defaults)
        constexpr_names = {'constexpr'}
        for item in tree.body:
            if isinstance(item, ast.ImportFrom) and item.module in ('triton.language', 'triton.language.core'):
                constexpr_names.update(alias.asname or alias.name for alias in item.names if alias.name == 'constexpr')
        self._default_scalars = {}
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            annotation = (arg.annotation.value if isinstance(arg.annotation, ast.Constant) and isinstance(arg.annotation.value, str)
                          else ast.unparse(arg.annotation) if arg.annotation else '')
            if arg.arg not in defaults or annotation in constexpr_names or annotation.endswith('.constexpr'):
                continue
            try:
                value = ast.literal_eval(defaults[arg.arg])
            except (ValueError, TypeError):
                continue
            dtype = 'i1' if type(value) is bool else ('i32' if -(2**31) <= value < 2**31 else 'i64') if type(value) is int else 'fp32'
            if type(value) in (bool, int, float):
                self._default_scalars[arg.arg] = (dtype, value)
        # Heuristics and selected autotune configs fill constexpr values in the worker.
        decorator_aliases = {alias.asname or alias.name: alias.name for node in tree.body
                             if isinstance(node, ast.ImportFrom) and node.module == 'triton' for alias in node.names}
        self._decorated = any(_decorator_kind(node, decorator_aliases) in ('heuristics', 'autotune')
                              for node in matches[0].decorator_list)
        self._source = source
        self._source_path = Path(source_path) if source_path is not None else None
        self._constants = constants if callable(constants) else _constants(constants or {})
        self._scalars = scalars if callable(scalars) else _scalars(scalars or {})
        if not callable(self._constants) and not callable(self._scalars):
            self._check_bindings(self._constants, self._scalars)
        self._grid = grid if callable(grid) else _launch_grid(grid)
        self._options = _constants(options or {})
        if config is not None:
            if type(config) is not int or config < 0:
                raise ValueError('config must be a nonnegative autotune configuration index')
            self._options['autotune_config'] = config
        # Callback results are frozen into each ExternalSpec as the pending
        # expression is built. Do not cache a graph by shape/dtype here: a
        # constants/grid callback may intentionally capture updated Python
        # configuration between calls. ASK still caches compiled artifacts.
        def invoke(*values):
            return self._trace(*values)
        invoke.__signature__ = inspect.Signature([
            inspect.Parameter(f'value{index}', inspect.Parameter.POSITIONAL_ONLY)
            for index in range(len(self._inputs))])
        self._record = jit(invoke, cache=False)

    @property
    def source(self):
        return self._source

    @property
    def source_path(self):
        return self._source_path

    @property
    def entry(self):
        return self._entry

    def _check_bindings(self, constants, scalars):
        names = set(constants) | set(scalars)
        missing = self._parameter_names - names - self._default_names
        extra = names - self._parameter_names
        if extra or (missing and not self._decorated) or set(constants) & set(scalars):
            raise ValueError(f'parameter bindings disagree with source; missing={sorted(missing)}, extra={sorted(extra)}')

    def __call__(self, *values):
        if len(values) != len(self._inputs):
            raise TypeError(f'a registered Triton kernel expects {len(self._inputs)} tensors')
        if all(isinstance(value, Tensor) for value in values):
            return self._record(*values)
        return self._trace(*values)

    def _trace(self, *values):
        if any(not isinstance(value, TraceTensor) for value in values):
            raise TypeError('a registered Triton kernel expects FugaTensor tensors from the current trace')
        context = values[0].context
        if any(value.context is not context for value in values):
            raise ValueError('kernel operands must belong to the same trace')
        specs = tuple(TensorSpec(tuple(value.shape), value.dtype, value.device) for value in values)
        spec = specs[0] if self._single_input else specs
        if any(value.numel < 1 for value in specs):
            raise ValueError('ASK Triton kernels require nonempty tensors')
        result_types = []
        for shape, dtype in self._outputs.values():
            shape = shape(spec) if callable(shape) else shape
            dtype = dtype(spec) if callable(dtype) else dtype
            result = TensorType(specs[0].shape if shape is None else normalize_shape(shape),
                                specs[0].dtype if dtype is None else normalize_dtype(dtype))
            if math.prod(result.shape) < 1:
                raise ValueError('ASK Triton kernels require nonempty outputs')
            result_types.append(result)
        constants = _constants(self._constants(spec) if callable(self._constants) else self._constants)
        scalars = _scalars({**self._default_scalars, **(self._scalars(spec) if callable(self._scalars) else self._scalars)})
        self._check_bindings(constants, scalars)
        grid = _launch_grid(self._grid(spec) if callable(self._grid) else self._grid)
        count = getattr(context, 'external_count', 0)
        symbol = f'fuga_triton_external_{count}'
        implementation = ExternalSpec(compiler='triton',
            source=self._source, entry=self._entry,
            signature={**{name: _SIGNATURE[value.dtype] for name, value in zip(self._inputs, values)},
                       **{name: _SIGNATURE[value.dtype] for name, value in zip(self._outputs, result_types)},
                       **{name: value[0] for name, value in scalars.items()}},
            source_bindings={**{name: f'input:{index}' for index, name in enumerate(self._inputs)},
                             **{name: f'output:{index}' for index, name in enumerate(self._outputs)},
                             **{name: value[1] for name, value in scalars.items()}}, constants=constants,
            options=dict(self._options), target={'backend': 'cuda', 'warp_size': 32},
            grid=grid, shape_rule='explicit', shape_operand=0, effects=())
        result = context.graph.external(*(value.value for value in values), spec=implementation,
                                        result_types=result_types, id=symbol)
        context.external_count = count + 1
        wrapped = tuple(context.wrap(value) for value in (result if isinstance(result, tuple) else (result,)))
        return wrapped[0] if self._single_output else wrapped


def kernel(source_path, **kwargs):
    """Snapshot a UTF-8 Python source file without importing its Triton module."""
    path = Path(source_path).expanduser().resolve()
    return TritonKernel(path.read_bytes().decode('utf-8'), source_path=path, **kwargs)


def from_source(source, **kwargs):
    """Register trusted source with explicit tensor and scalar bindings."""
    return TritonKernel(source, **kwargs)


def from_jit(function, **kwargs):
    """Snapshot a loaded Triton JIT function, referenced helpers and global values.

    Importing Triton remains the caller's choice. The resulting registration
    contains source, not a live compiler object, and runs in the selected worker.
    """
    definitions, seen, active = [], {}, set()

    def emit(name, value):
        if name in seen:
            if seen[name] is not value:
                raise ValueError(f'conflicting source dependency name: {name}')
            return
        seen[name] = value
        if id(value) in active:
            raise ValueError('recursive source dependencies are unsupported')
        if isinstance(value, types.ModuleType):
            definitions.append(f'import {value.__name__} as {name}')
            return
        if (inspect.isbuiltin(value) or inspect.isclass(value)) and getattr(value, '__module__', '').startswith(('triton.', 'math')):
            definitions.append(f'from {value.__module__} import {value.__name__} as {name}')
            return
        base, compiled, wrappers = value, None, []
        while not inspect.isfunction(base) and hasattr(base, 'fn'):
            if hasattr(base, 'src'):
                compiled = base
            else:
                wrappers.append(type(base).__name__)
            base = base.fn
        if inspect.isfunction(base):
            if base.__module__.startswith(('triton.', 'math')):
                definitions.append(f'from {base.__module__} import {base.__name__} as {name}')
                return
            active.add(id(value))
            if compiled is None:
                node = ast.parse(textwrap.dedent(inspect.getsource(base))).body[0]
                environment = {**base.__globals__, **inspect.getclosurevars(base).nonlocals}
            else:
                node = ast.parse(compiled.src).body[0]
                original = ast.parse(textwrap.dedent(''.join(compiled.raw_src))).body[0]
                environment = compiled.get_capture_scope()
                aliases = {key: getattr(item, '__name__', key) for key, item in environment.items()}
                decorators = [item for item in original.decorator_list
                              if _decorator_kind(item, aliases) in ('autotune', 'heuristics')]
                if len(decorators) != len(wrappers):
                    raise ValueError('programmatic tuning wrappers require an explicit source snapshot')
                if compiled._repr is not None or compiled.launch_metadata is not None or compiled.pre_run_hooks:
                    raise ValueError('custom JIT repr, launch metadata and pre-run hooks cannot be snapshotted')
                settings = {key: getattr(compiled, key) for key in
                            ('version', 'do_not_specialize', 'do_not_specialize_on_alignment', 'debug', 'noinline')}
                node.decorator_list = decorators + [ast.parse('triton.jit(' + ', '.join(
                    key + '=' + repr(item) for key, item in settings.items() if item is not None) + ')', mode='eval').body]
                environment = {**environment, 'triton': __import__('triton')}

            symbols = symtable.symtable(ast.unparse(node), '<snapshot>', 'exec')
            def global_names(table):
                names = {item.get_name() for item in table.get_symbols()
                         if item.is_referenced() and (table.get_type() == 'module' or item.is_global() or item.is_free())}
                for child in table.get_children():
                    names.update(global_names(child))
                return names
            loaded = global_names(symbols)
            for dependency in sorted(loaded & environment.keys() - {base.__name__}):
                emit(dependency, environment[dependency])
            node.name = name
            definitions.append(ast.unparse(node))
            active.remove(id(value))
            return
        if type(value).__module__.startswith('triton.') and hasattr(value, 'all_kwargs'):
            if value.pre_hook is not None or value.ir_override is not None:
                raise ValueError('autotune config hooks and IR overrides are unsupported')
            definitions.append(f'{name} = triton.Config({value.kwargs!r}, num_warps={value.num_warps!r}, '
                               f'num_stages={value.num_stages!r}, num_ctas={value.num_ctas!r}, maxnreg={value.maxnreg!r})')
            return
        # Preserve literal globals and dtype objects as executable source data.
        def literal(item):
            if type(item).__module__.startswith('triton.language') and type(item).__name__ == 'constexpr':
                return 'triton.language.constexpr(' + literal(item.value) + ')'
            encoded = _constant(item)
            if isinstance(encoded, dict) and set(encoded) == {'__triton_dtype__'}:
                return 'triton.language.' + _identifier(encoded['__triton_dtype__'], 'dtype')
            if isinstance(item, list):
                return '[' + ', '.join(literal(part) for part in item) + ']'
            if isinstance(item, tuple):
                return '(' + ', '.join(literal(part) for part in item) + (',)' if item else ')')
            if isinstance(item, Mapping):
                return '{' + ', '.join(repr(key) + ': ' + literal(part) for key, part in item.items()) + '}'
            return repr(item)
        if isinstance(value, (tuple, list)) and any(hasattr(item, 'all_kwargs') for item in value):
            names = []
            for index, item in enumerate(value):
                alias = f'_fuga_{name}_{index}'
                emit(alias, item)
                names.append(alias)
            definitions.append(f'{name} = [{", ".join(names)}]')
        else:
            definitions.append(f'{name} = {literal(value)}')

    base = function
    while not inspect.isfunction(base) and hasattr(base, 'fn'):
        base = base.fn
    if not inspect.isfunction(base) or not hasattr(function, 'arg_names'):
        raise TypeError('from_jit expects a loaded Triton JIT function or its tuning wrapper')
    emit(base.__name__, function)
    return TritonKernel('import triton\nimport triton.language\n' + '\n\n'.join(definitions) + '\n',
                        entry=base.__name__, **kwargs)

def environment(*, python, worker=None):
    """Select a compiler interpreter and this package's standalone Triton worker."""
    selected = Path(__file__).with_name('_triton_worker.py') if worker is None else worker
    return CompilerEnvironment(python=python, worker=selected)


def environment_from_env():
    """Optional setup convenience; importing this module never imports Triton."""
    python = os.environ.get('FUGATENSOR_TRITON_PYTHON')
    worker = os.environ.get('FUGATENSOR_TRITON_WORKER')
    if python is None and worker is None:
        return None
    if not python:
        raise ValueError('set FUGATENSOR_TRITON_PYTHON when selecting a Triton worker')
    return environment(python=python, worker=worker)

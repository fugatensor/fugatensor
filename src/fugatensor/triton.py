"""Register user-written Triton source as an ASK external operation."""
from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from types import MappingProxyType

from ask import CompilerEnvironment

from .jit import jit
from .tensor import Tensor
from .tracing import TraceTensor

_SIGNATURE = {'bool': '*i1', 'i32': '*i32', 'i64': '*i64', 'f32': '*fp32', 'f16': '*fp16', 'bf16': '*bf16'}
_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z', re.ASCII)


def _identifier(value, field):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f'{field} must be an ASCII identifier')
    return value


def _integer_constants(values):
    if not isinstance(values, Mapping):
        raise TypeError('constants must be a mapping or a callback returning a mapping')
    result = deepcopy(dict(values))
    for name, value in result.items():
        _identifier(name, 'constant name')
        if type(value) is not int or not -(2**63) <= value < 2**63:
            raise TypeError('ASK currently accepts signed 64-bit integer constexpr constants')
    return result


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
    dtype: str
    device: str

    @property
    def numel(self):
        return math.prod(self.shape)


class TritonKernel:
    """A snapshot of user source plus explicit ASK parameter and launch bindings.

    Registration does not import or execute the source. ASK's selected Triton
    adapter validates and compiles it later. Re-register after changing source,
    and retrace an enclosing JIT function if it captured the old registration.
    """
    def __init__(self, source, *, entry, input, output, grid, constants=None,
                 options=None, source_path=None):
        if not isinstance(source, str) or not source.strip():
            raise TypeError('Triton source must be a nonempty string')
        self._entry = _identifier(entry, 'entry')
        self._input = _identifier(input, 'input parameter')
        self._output = _identifier(output, 'output parameter')
        if input == output:
            raise ValueError('input and output must bind different source parameters')
        tree = ast.parse(source, filename=str(source_path or '<triton-source>'))
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry]
        if len(matches) != 1:
            raise ValueError('source must define the requested entry exactly once')
        function = matches[0]
        args = function.args
        if args.vararg is not None or args.kwarg is not None:
            raise ValueError('Triton entry requires explicit parameter names')
        parameters = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        if input not in parameters or output not in parameters:
            raise ValueError('input/output binding names must exist in the source entry parameters')
        self._constant_names = frozenset(parameters - {input, output})
        self._source = source
        self._source_path = Path(source_path) if source_path is not None else None
        if callable(constants):
            self._constants = constants
        else:
            frozen = _integer_constants({} if constants is None else constants)
            self._check_constant_names(frozen)
            self._constants = MappingProxyType(frozen)
        self._grid = grid if callable(grid) else _launch_grid(grid)
        if options is not None and not isinstance(options, Mapping):
            raise TypeError('options must be a mapping of fixed compiler options')
        selected = {'num_warps': 4}
        selected.update(deepcopy(dict(options or {})))
        for name, value in selected.items():
            _identifier(name, 'option name')
            if type(value) not in (str, int, bool):
                raise TypeError('ASK fixed options must contain strings, integers or booleans')
            if type(value) is int and not -(2**63) <= value < 2**63:
                raise ValueError('ASK integer options must fit in signed 64 bits')
        self._options = MappingProxyType(selected)

        # Each registration owns its own wrapper/cache. Its body is only a graph
        # call; the user source is handed to ASK unchanged, never synthesized.
        def invoke(value):
            return self._trace(value)
        self._eager = jit(invoke)

    @property
    def source(self):
        return self._source

    @property
    def source_path(self):
        return self._source_path

    @property
    def entry(self):
        return self._entry

    def _check_constant_names(self, constants):
        if set(constants) != self._constant_names:
            missing = sorted(self._constant_names - set(constants))
            extra = sorted(set(constants) - self._constant_names)
            raise ValueError(f'constexpr bindings must cover the other source parameters; missing={missing}, extra={extra}')

    def __call__(self, value):
        if isinstance(value, Tensor):
            return self._eager(value)
        return self._trace(value)

    def _trace(self, value):
        if not isinstance(value, TraceTensor):
            raise TypeError('a registered Triton kernel expects one FugaTensor tensor')
        context = value.context
        engine = context.engine
        if engine is None or engine.device == 'cpu':
            raise NotImplementedError('Triton operations require a CUDA Engine')
        spec = TensorSpec(tuple(value.shape), value.dtype, engine.device)
        if spec.numel < 1:
            raise ValueError('ASK Triton fusion requires nonempty tensors')
        constants = _integer_constants(self._constants(spec) if callable(self._constants) else self._constants)
        self._check_constant_names(constants)
        grid = _launch_grid(self._grid(spec) if callable(self._grid) else self._grid)
        major, minor = engine.capability
        count = getattr(context, 'external_count', 0)
        symbol = f'fuga_triton_external_{count}'
        semantic = context.module.semantics(symbol + '_semantics', inputs=[value.value.type],
            results=[value.value.type], shape_rule='same_as', shape_operand=0, effects='none')
        implementation = context.module.implementation(symbol + '_implementation', compiler='triton',
            source=self._source, entry=self._entry,
            signature={self._input: _SIGNATURE[value.dtype], self._output: _SIGNATURE[value.dtype]},
            source_bindings={self._input: 'input:0', self._output: 'output:0'}, constants=constants,
            options=dict(self._options), target={'backend': 'cuda', 'arch': major * 10 + minor, 'warp_size': 32},
            grid=grid)
        result = context.graph.external(value.value, semantics=semantic, implementation=implementation,
                                        id=symbol)
        context.external_count = count + 1
        return context.wrap(result)

    def cache_info(self):
        """Statistics for direct op(tensor) calls; enclosing @jit owns its cache."""
        return self._eager.cache_info()

    def clear_cache(self):
        self._eager.clear_cache()


def kernel(source_path, *, entry, input, output, grid, constants=None, options=None):
    """Snapshot a UTF-8 Python source file without importing its Triton module."""
    path = Path(source_path).expanduser().resolve()
    source = path.read_bytes().decode('utf-8')
    return TritonKernel(source, entry=entry, input=input, output=output, grid=grid,
                        constants=constants, options=options, source_path=path)


def from_source(source, *, entry, input, output, grid, constants=None, options=None):
    """Register an explicit user-written source string using the same bindings."""
    return TritonKernel(source, entry=entry, input=input, output=output, grid=grid,
                        constants=constants, options=options)


def environment_from_env():
    """Optional setup convenience; importing this module never imports Triton."""
    python = os.environ.get('FUGATENSOR_TRITON_PYTHON')
    worker = os.environ.get('FUGATENSOR_TRITON_WORKER')
    if python is None and worker is None:
        return None
    if not python or not worker:
        raise ValueError('set both FUGATENSOR_TRITON_PYTHON and FUGATENSOR_TRITON_WORKER')
    return CompilerEnvironment(python=python, worker=worker)

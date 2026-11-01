"""Record user-written Triton implementations in device-independent Fuga IR."""
from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from types import MappingProxyType

from ask import CompilerEnvironment

from ._jit import jit
from .ir import ExternalSpec, TensorType
from .tensor import Tensor, normalize_dtype, normalize_shape
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
    then call the new registration to record new expressions.
    """
    def __init__(self, source, *, entry, input=None, inputs=None, output, grid, constants=None,
                 options=None, source_path=None, output_shape=None, output_dtype=None):
        if not isinstance(source, str) or not source.strip():
            raise TypeError('Triton source must be a nonempty string')
        self._entry = _identifier(entry, 'entry')
        if (input is None) == (inputs is None):
            raise ValueError('provide exactly one of input or inputs')
        self._single_input = input is not None
        if inputs is None:
            inputs = (input,)
        if isinstance(inputs, str):
            raise TypeError('inputs must be a sequence of source parameter names')
        self._inputs = tuple(_identifier(name, 'input parameter') for name in inputs)
        if not self._inputs or len(set(self._inputs)) != len(self._inputs):
            raise ValueError('inputs must contain distinct source parameter names')
        self._output = _identifier(output, 'output parameter')
        if output in self._inputs:
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
        if not set((*self._inputs, output)) <= parameters:
            raise ValueError('input/output binding names must exist in the source entry parameters')
        self._constant_names = frozenset(parameters - {*self._inputs, output})
        self._output_shape = output_shape if callable(output_shape) or output_shape is None else normalize_shape(output_shape)
        self._output_dtype = output_dtype if callable(output_dtype) or output_dtype is None else normalize_dtype(output_dtype)
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

    def _check_constant_names(self, constants):
        if set(constants) != self._constant_names:
            missing = sorted(self._constant_names - set(constants))
            extra = sorted(set(constants) - self._constant_names)
            raise ValueError(f'constexpr bindings must cover the other source parameters; missing={missing}, extra={extra}')

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
        shape = self._output_shape(spec) if callable(self._output_shape) else self._output_shape
        dtype = self._output_dtype(spec) if callable(self._output_dtype) else self._output_dtype
        result_type = TensorType(specs[0].shape if shape is None else normalize_shape(shape),
                                 specs[0].dtype if dtype is None else normalize_dtype(dtype))
        if math.prod(result_type.shape) < 1:
            raise ValueError('ASK Triton kernels require nonempty outputs')
        constants = _integer_constants(self._constants(spec) if callable(self._constants) else self._constants)
        self._check_constant_names(constants)
        grid = _launch_grid(self._grid(spec) if callable(self._grid) else self._grid)
        count = getattr(context, 'external_count', 0)
        symbol = f'fuga_triton_external_{count}'
        implementation = ExternalSpec(compiler='triton',
            source=self._source, entry=self._entry,
            signature={**{name: _SIGNATURE[value.dtype] for name, value in zip(self._inputs, values)},
                       self._output: _SIGNATURE[result_type.dtype]},
            source_bindings={**{name: f'input:{index}' for index, name in enumerate(self._inputs)},
                             self._output: 'output:0'}, constants=constants,
            options=dict(self._options), target={'backend': 'cuda', 'warp_size': 32},
            grid=grid, shape_rule='same_as' if shape is None else 'explicit', shape_operand=0, effects=())
        result = context.graph.external(*(value.value for value in values), spec=implementation,
                                        result_types=(result_type,), id=symbol)
        context.external_count = count + 1
        return context.wrap(result)

def kernel(source_path, *, entry, input=None, inputs=None, output, grid, constants=None, options=None,
           output_shape=None, output_dtype=None):
    """Snapshot a UTF-8 Python source file without importing its Triton module."""
    path = Path(source_path).expanduser().resolve()
    source = path.read_bytes().decode('utf-8')
    return TritonKernel(source, entry=entry, input=input, inputs=inputs, output=output, grid=grid,
                        constants=constants, options=options, source_path=path,
                        output_shape=output_shape, output_dtype=output_dtype)


def from_source(source, *, entry, input=None, inputs=None, output, grid, constants=None, options=None,
                output_shape=None, output_dtype=None):
    """Register an explicit user-written source string using the same bindings."""
    return TritonKernel(source, entry=entry, input=input, inputs=inputs, output=output, grid=grid,
                        constants=constants, options=options,
                        output_shape=output_shape, output_dtype=output_dtype)


def environment_from_env():
    """Optional setup convenience; importing this module never imports Triton."""
    python = os.environ.get('FUGATENSOR_TRITON_PYTHON')
    worker = os.environ.get('FUGATENSOR_TRITON_WORKER')
    if python is None and worker is None:
        return None
    if not python or not worker:
        raise ValueError('set both FUGATENSOR_TRITON_PYTHON and FUGATENSOR_TRITON_WORKER')
    return CompilerEnvironment(python=python, worker=worker)

"""Independent dense-index references and generated tensor-operation cases."""
from itertools import combinations, permutations, product
import math
import tempfile

import fugatensor as ft
from fugatensor._state import active_trace
from fugatensor.backends.host import HostStorage
from fugatensor.tracing import TraceContext
from tests.support.selection import select

DTYPES = {name: getattr(ft, name) for name in ('fp32', 'fp16', 'bf16', 'int32', 'int64', 'bool')}
SHAPES = ((), (4,), (2, 3), (1, 2, 1), (0, 3), (2, 0))
MATRIX_SHAPES = ((2, 3, 4), (1, 1, 0), (0, 3, 2), (2, 0, 3), (3, 2, 1))
BATCHES = ((), (2,), (1, 2), (0,))


def dtypes(floating=False):
    selected = select('FUGATENSOR_TEST_DTYPES', DTYPES)
    return tuple(name for name in selected if not floating or name in ('fp32', 'fp16', 'bf16'))


def indices(shape):
    return product(*(range(size) for size in shape))


def offset(index, shape):
    result = 0
    for coordinate, size in zip(index, shape):
        result = result * size + coordinate
    return result


def values(shape, dtype='fp32', phase=0):
    return [bool((index + phase) % 2) if dtype == 'bool' else (index + phase) % 5 - 2
            for index in range(math.prod(shape))]


def flatten(value):
    if isinstance(value, list):
        return [item for child in value for item in flatten(child)]
    return [value]


def transpose_reference(data, shape, permutation):
    target = tuple(shape[axis] for axis in permutation)
    inverse = tuple(permutation.index(axis) for axis in range(len(shape)))
    return [data[offset(tuple(index[axis] for axis in inverse), shape)] for index in indices(target)]


def reduction_reference(data, shape, axes, maximum=False):
    remaining = tuple(axis for axis in range(len(shape)) if axis not in axes)
    target = tuple(shape[axis] for axis in remaining)
    results = []
    for output in indices(target):
        entries = []
        for reduced in indices(tuple(shape[axis] for axis in axes)):
            index = dict(zip(remaining, output)) | dict(zip(axes, reduced))
            entries.append(data[offset(tuple(index[axis] for axis in range(len(shape))), shape)])
        results.append(max(entries) if maximum else sum(entries))
    return target, results


def geometry_matrix():
    for dtype, shape in product(dtypes(), SHAPES):
        yield dtype, shape, tuple(permutations(range(len(shape)))), tuple(
            axes for length in range(len(shape) + 1) for axes in combinations(range(len(shape)), length))


def matmul_matrix():
    for dtype, batch, (m, n, k), ta, tb in product(dtypes(True), BATCHES, MATRIX_SHAPES, (False, True), (False, True)):
        yield dtype, (*batch, k, m) if ta else (*batch, m, k), (*batch, n, k) if tb else (*batch, k, n), ta, tb


def host_workspace_matrix():
    types = tuple(name for name in dtypes() if name not in ('bool', 'bf16'))
    return product(types, (0, 1, 5), (0, 3), (2, 5), (False, True), (False, True))


def host_mixed_workspace_matrix():
    types = tuple(name for name in dtypes() if name != 'bf16')
    return product(types, types, (False, True))


def materialized_chain_reference(data, shape, storage_types):
    rows, columns = shape
    current, captured = data, None
    for stage, dtype in enumerate(storage_types):
        cast = bool if dtype == 'bool' else int if dtype in ('int32', 'int64') else float
        numeric = int if dtype == 'bool' else cast
        reduced = [cast(sum(numeric(value) for value in current[row * columns:(row + 1) * columns]))
                   for row in range(rows)]
        if stage == 0:
            captured = sum(numeric(value) for value in reduced)
        current = [value for value in reduced for _ in range(columns)]
    return current, captured


def matmul_reference(left, right, ashape, bshape, ta=False, tb=False):
    m, k = ashape[-2:][::-1] if ta else ashape[-2:]
    _, n = bshape[-2:][::-1] if tb else bshape[-2:]
    result = []
    for batch in indices(ashape[:-2]):
        for row, column in product(range(m), range(n)):
            result.append(sum(left[offset((*batch, inner, row) if ta else (*batch, row, inner), ashape)] *
                              right[offset((*batch, column, inner) if tb else (*batch, inner, column), bshape)]
                              for inner in range(k)))
    return (*ashape[:-2], m, n), result


def trace(function, specs):
    context = TraceContext([(shape, DTYPES.get(dtype, dtype)) for shape, dtype in specs])
    token = active_trace.set(context)
    try:
        return context.finish(function(*context.inputs))
    finally:
        active_trace.reset(token)


def execute_ir(module, arguments, engine, *, graph=True):
    specs = [value.type for value in module.functions[0].inputs.values()]
    target = {'backend': 'cuda'} if engine.device != 'cpu' else {'backend': 'host'}
    if target['backend'] == 'cuda':
        capability = engine.session.capability
        target['arch'] = 10 * capability[0] + capability[1]
    lowered = ft.lower_to_ask(module, target=target)
    if engine.device == 'cpu':
        compiled = engine._host.compile(lowered)
        inputs = [HostStorage.from_flat(data, shape=spec.shape, dtype=spec.dtype) for spec, data in zip(specs, arguments)]
        return [value.to_flat() for value in compiled.run(inputs)]
    with tempfile.TemporaryDirectory(prefix='fuga-basis-', dir=engine.cache_dir) as directory:
        compiled = engine.compiler.compile_cuda(lowered, target=target, cache_dir=engine.cache_dir / 'ir-matrix',
            output_dir=directory, llc=engine.llc, ptxas=engine.ptxas, llvm_link=engine.llvm_link, libdevice=engine.libdevice)
        prepared = compiled.prepare(engine.session, graph=graph)
        inputs = [engine.session.from_host(data, shape=spec.shape, dtype=spec.dtype) for spec, data in zip(specs, arguments)]
        return [value.to_host() for value in prepared.run(inputs)]

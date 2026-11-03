"""Static compiler partition fixtures for compilation orchestration tests."""
from types import SimpleNamespace


def owned_partition(source):
    plan = {'schema_version': 1, 'kind': 'ask.cuda.execution_plan',
            'inputs': ['x'], 'outputs': ['y'],
            'buffers': [{'id': name, 'shape': [10], 'element_type': 'i32'} for name in ('x', 'y')],
            'steps': [{'id': 'step0', 'kind': 'owned', 'function': 'step0',
                       'inputs': ['x'], 'outputs': ['y'], 'depends_on': []}],
            'scheduling': {'policy': 'sequential', 'stream': 'session'}}
    return SimpleNamespace(external_requests=[], shape_snapshot={}, execution_plan=plan,
                           step_sources={'step0': source})


def matmul_plan(dtype='f32', *, batch=(), m=2, n=3, k=4, transpose_a=False, transpose_b=False):
    import math
    left = [*batch, k, m] if transpose_a else [*batch, m, k]
    right = [*batch, n, k] if transpose_b else [*batch, k, n]
    return {'schema_version': 1, 'kind': 'ask.cuda.execution_plan',
            'inputs': ['a', 'b'], 'outputs': ['c'],
            'buffers': [{'id': name, 'shape': shape, 'element_type': dtype}
                        for name, shape in (('a', left), ('b', right), ('c', [*batch, m, n]))],
            'steps': [{'id': 'mm', 'kind': 'library', 'function': 'mm',
                       'inputs': ['a', 'b'], 'outputs': ['c'], 'depends_on': [],
                       'library': 'cublaslt', 'matmul': {'element_type': dtype, 'm': m, 'n': n, 'k': k,
                           'batch_count': math.prod(batch), 'transpose_a': transpose_a, 'transpose_b': transpose_b}}],
            'scheduling': {'policy': 'sequential', 'stream': 'session'}}

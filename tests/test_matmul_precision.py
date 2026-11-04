"""CUDA library precision on non-dyadic inputs, cancellation, and exceptional values."""
from itertools import product
import math

import pytest
import fugatensor as ft
from tests.support.tensor_primitives import DTYPES, flatten
from tests.support.matmul import assert_matmul_values, reference_and_scale


@pytest.mark.cuda
@pytest.mark.shared_engine
@pytest.mark.usefixtures('engine_context')
class TestCUDAMatmulPrecision:
    device = 'cuda'

    @pytest.mark.parametrize('name,pattern', tuple(product(('fp32', 'fp16', 'bf16'), ('rounded_inputs', 'cancellation'))))
    def test_library_accuracy_and_shared_result(self, name, pattern):
        m, n, k = 3, 17, 33
        if pattern == 'rounded_inputs':
            av = [((i * 13) % 31 - 15) * 0.13 + 2**-14 for i in range(m * k)]
            bv = [((i * 7) % 29 - 14) * 0.17 - 2**-13 for i in range(k * n)]
        else:
            av = [((1.0003, 0.00017, -1.0003)[i % 3]) for i in range(m * k)]
            bv = [1 + (i % n) * 0.0003 for i in range(k * n)]
        a = self.engine.tensor(av, shape=(m, k), dtype=DTYPES[name])
        b = self.engine.tensor(bv, shape=(k, n), dtype=DTYPES[name])
        # Reference the actual input dtype, using high precision for products and reduction.
        shape, expected, scale = reference_and_scale(flatten(a.tolist()), flatten(b.tolist()), a.shape, b.shape)
        result = a @ b
        consumed = result * 2
        ft.evaluate(result, consumed)
        assert result.shape == shape and result.dtype is DTYPES[name]
        assert_matmul_values(flatten(result.tolist()), expected, scale, k=k, dtype=name, tf32=name == 'fp32')
        assert_matmul_values(flatten(consumed.tolist()), [x * 2 for x in expected], [x * 2 for x in scale],
                             k=k + 1, dtype=name, tf32=name == 'fp32')
        assert sum(step['kind'] == 'library' for step in self.engine.last_execution_plan['steps']) == 1

    @pytest.mark.parametrize('name', ('fp32', 'fp16', 'bf16'))
    def test_nan_infinity_and_zero_products(self, name):
        left = [[math.inf, 1, -math.inf], [math.inf, 1, 2],
                [-math.inf, 0, 0], [math.nan, 1, 0], [-0.0, 0.0, -0.0]]
        right = [[1, 0, -1]] * 3
        a = self.engine.tensor(left, dtype=DTYPES[name])
        b = self.engine.tensor(right, dtype=DTYPES[name])
        shape, expected, scale = reference_and_scale(flatten(left), flatten(right), a.shape, b.shape)
        result = a @ b
        assert result.shape == shape and result.dtype is DTYPES[name]
        assert_matmul_values(flatten(result.tolist()), expected, scale, k=3, dtype=name, tf32=name == 'fp32')

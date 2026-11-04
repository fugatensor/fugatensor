"""CPU matmul geometry, BLAS f32 accuracy, and strict f16 accumulation."""
from itertools import product
import math

import pytest
import fugatensor as ft
from tests.support.tensor_primitives import DTYPES, flatten
from tests.support.matmul import host_cases, case_id, reference_and_scale, assert_matmul_values


def pytest_generate_tests(metafunc):
    if 'matmul_case' in metafunc.fixturenames:
        metafunc.parametrize('matmul_case', host_cases(exhaustive=metafunc.config.getoption('--exhaustive')),
                             ids=case_id)


@pytest.mark.shared_engine
@pytest.mark.usefixtures('engine_context')
class TestHostMatmul:
    def test_tile_layout_dtype_and_shared_output_matrix(self, matmul_case):
        name, batch, m, n, k, ta, tb, shared = matmul_case
        ashape = (*batch, k, m) if ta else (*batch, m, k)
        bshape = (*batch, n, k) if tb else (*batch, k, n)
        av = [((i * 7 + i // ashape[-1]) % 9 - 4) / 4 for i in range(math.prod(ashape))]
        bv = [((i * 3 + i // bshape[-1]) % 7 - 3) / 4 for i in range(math.prod(bshape))]
        a = self.engine.tensor(av, shape=ashape, dtype=DTYPES[name])
        b = self.engine.tensor(bv, shape=bshape, dtype=DTYPES[name])
        actual = ft.matmul(a + 0.25 if shared else a, b, transpose_a=ta, transpose_b=tb)
        target_shape, expected, scale = reference_and_scale(
            [value + 0.25 for value in av] if shared else av, bv, ashape, bshape, ta, tb)
        outputs = (actual, actual * 2 + 1) if shared else (actual,)
        ft.evaluate(*outputs)
        assert actual.shape == target_shape
        assert actual.dtype is DTYPES[name]
        assert_matmul_values(flatten(actual.tolist()), expected, scale, k=k, dtype=name)
        if shared:
            assert_matmul_values(flatten(outputs[1].tolist()), [value * 2 + 1 for value in expected],
                                 [value * 2 + 1 for value in scale], k=k + 1, dtype=name)

    @pytest.mark.parametrize('columns,operands', tuple(product((15, 16, 17), (
        ([-1, 1 + 2**-23], [1, 1 - 2**-23]),
        ([2**24, 1, -(2**24)], [1, 1, 1]),
    ))), ids=lambda value: str(value))
    def test_f32_blas_error_bound_under_cancellation(self, columns, operands):
        # BLAS may contract or reorder; cancellation is measured against the sum of magnitudes.
        left, right = operands
        a = self.engine.tensor([left])
        b = self.engine.tensor([[value] * columns for value in right])
        _, expected, scale = reference_and_scale(left, flatten(b.tolist()), a.shape, b.shape)
        assert_matmul_values(flatten((a @ b).tolist()), expected, scale, k=len(left), dtype='fp32')

    @pytest.mark.parametrize('columns', (15, 16, 17))
    def test_f16_keeps_ordered_f32_accumulation(self, columns):
        a = self.engine.tensor([[65504, 2**-10, -65504]], dtype=ft.fp16)
        b = self.engine.tensor([[1] * columns] * 3, dtype=ft.fp16)
        assert (a @ b).tolist() == [[0.0] * columns]

    @pytest.mark.parametrize('name,rows,columns', tuple(product(('fp32', 'fp16'), (3, 4, 5), (15, 16, 17))))
    def test_nonfinite_values_across_row_and_column_tails(self, name, rows, columns):
        patterns = ((math.inf, 1, -math.inf), (math.inf, 1, 2),
                    (-math.inf, 0, 0), (math.nan, 1, 0))
        left = [list(patterns[row % len(patterns)]) for row in range(rows)]
        right = [[int(column % 2 == 0) for column in range(columns)] for _ in range(3)]
        a, b = (self.engine.tensor(value, dtype=DTYPES[name]) for value in (left, right))
        _, expected, scale = reference_and_scale(flatten(left), flatten(right), (rows, 3), (3, columns))
        assert_matmul_values(flatten((a @ b).tolist()), expected, scale, k=3, dtype=name)

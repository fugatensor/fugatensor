"""Host matmul tile edges, shared results, and ordered floating-point arithmetic."""
from itertools import product
import math
import unittest

import fugatensor as ft
from tests._helpers import EngineTestCase
from tests.support.tensor_primitives import DTYPES, flatten, matmul_matrix, matmul_reference


class HostMatmulTests(EngineTestCase):
    def test_tile_layout_dtype_and_shared_output_matrix(self):
        shapes = tuple(product((3, 4, 5, 8), (15, 16, 17, 31, 32, 33), (3, 17)))
        matrix = matmul_matrix(shapes=shapes, batches=((), (2,)), types=('fp32', 'fp16'))
        for (name, ashape, bshape, ta, tb), shared in product(matrix, (False, True)):
            with self.subTest(dtype=name, a=ashape, b=bshape, ta=ta, tb=tb, shared=shared):
                av = [((i * 7 + i // ashape[-1]) % 9 - 4) / 4 for i in range(math.prod(ashape))]
                bv = [((i * 3 + i // bshape[-1]) % 7 - 3) / 4 for i in range(math.prod(bshape))]
                a = self.engine.tensor(av, shape=ashape, dtype=DTYPES[name])
                b = self.engine.tensor(bv, shape=bshape, dtype=DTYPES[name])
                actual = ft.matmul(a + 0.25 if shared else a, b, transpose_a=ta, transpose_b=tb)
                target_shape, expected = matmul_reference(
                    [value + 0.25 for value in av] if shared else av, bv, ashape, bshape, ta, tb)
                outputs = (actual, actual * 2 + 1) if shared else (actual,)
                ft.evaluate(*outputs)
                self.assertEqual(actual.shape, target_shape)
                self.assertEqual(actual.dtype, DTYPES[name])
                self.assertEqual(flatten(actual.tolist()), expected)
                if shared:
                    self.assertEqual(flatten(outputs[1].tolist()), [value * 2 + 1 for value in expected])

    def test_f32_product_rounding_and_k_order(self):
        # Product rounding and cancellation distinguish ordered mul/add from
        # contraction or a reassociated reduction. Broadcast across tile edges.
        operands = (
            ([-1, 1 + 2**-23], [1, 1 - 2**-23]),
            ([2**24, 1, -(2**24)], [1, 1, 1]),
        )
        for columns, (left, right) in product((15, 16, 17), operands):
            with self.subTest(columns=columns, k=len(left)):
                a = self.engine.tensor([left])
                b = self.engine.tensor([[value] * columns for value in right])
                self.assertEqual((a @ b).tolist(), [[0.0] * columns])

    def test_nonfinite_values_across_row_and_column_tails(self):
        patterns = ((math.inf, 1, -math.inf), (math.inf, 1, 2),
                    (-math.inf, 0, 0), (math.nan, 1, 0))
        for name, rows, columns in product(('fp32', 'fp16'), (3, 4, 5), (15, 16, 17)):
            with self.subTest(dtype=name, rows=rows, columns=columns):
                left = [list(patterns[row % len(patterns)]) for row in range(rows)]
                right = [[int(column % 2 == 0) for column in range(columns)] for _ in range(3)]
                a, b = (self.engine.tensor(value, dtype=DTYPES[name]) for value in (left, right))
                actual = flatten((a @ b).tolist())
                _, expected = matmul_reference(flatten(left), flatten(right), (rows, 3), (3, columns))
                for value, target in zip(actual, expected):
                    if math.isnan(target):
                        self.assertTrue(math.isnan(value))
                    else:
                        self.assertEqual(value, target)


if __name__ == '__main__':
    unittest.main()

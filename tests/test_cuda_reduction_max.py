"""Public CUDA maximum matrices for warp tails, layouts, and IEEE values."""
from itertools import product
import math
import unittest

import fugatensor as ft
from tests._helpers import EngineTestCase, requires_cuda
from tests.support.reduction_max import (AXES, COMPOSITIONS, LAYOUTS, maximum_dtypes,
    maximum_reference, maximum_values, reduction_matrix, storage_layout)
from tests.support.tensor_primitives import DTYPES, flatten


@requires_cuda
class CUDAMaximumTests(EngineTestCase):
    device = 'cuda'

    def assert_maximum(self, result, name, shape, expected):
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, DTYPES[name])
        actual = flatten(result.tolist())
        self.assertEqual(len(actual), len(expected))
        for index, (value, target) in enumerate(zip(actual, expected)):
            with self.subTest(element=index):
                if isinstance(target, float) and math.isnan(target):
                    self.assertTrue(math.isnan(value))
                else:
                    self.assertEqual(value, target)
                    if name in ('fp32', 'fp16', 'bf16') and target == 0:
                        self.assertEqual(math.copysign(1, value), math.copysign(1, target))

    def input_with_layout(self, name, shape, permutation):
        data = maximum_values(shape, name)
        physical, physical_shape, inverse = storage_layout(data, shape, permutation)
        value = self.engine.tensor(physical, shape=physical_shape, dtype=DTYPES[name])
        return data, value if permutation == (0, 1, 2) else ft.transpose(value, inverse)

    def test_dtype_shape_axes_layout_and_fusion_matrix(self):
        for name, shape, permutation in reduction_matrix():
            if name == 'bf16' and not self.supports_bf16:
                continue
            with self.subTest(dtype=name, shape=shape, layout=permutation):
                data, x = self.input_with_layout(name, shape, permutation)
                pending = []
                for axes in AXES:
                    if any(shape[axis] == 0 for axis in axes):
                        with self.assertRaisesRegex(ValueError, 'nonempty'):
                            ft.amax(x, axis=axes)
                        continue
                    for composition in COMPOSITIONS:
                        if composition == 'plain':
                            result = ft.amax(x, axis=axes)
                        else:
                            result = -ft.amax(ft.where(x < 0, ft.zeros_like(x), x), axis=axes)
                        target_shape, expected = maximum_reference(data, shape, axes, composition)
                        pending.append((result, axes, composition, target_shape, expected))
                ft.evaluate(*(item[0] for item in pending))
                for result, axes, composition, target_shape, expected in pending:
                    with self.subTest(axes=axes, composition=composition):
                        self.assert_maximum(result, name, target_shape, expected)

    def test_nested_max_and_expanded_intermediate_matrix(self):
        for name, width, permutation in product(maximum_dtypes(), (33, 65), LAYOUTS):
            if name == 'bf16' and not self.supports_bf16:
                continue
            shape = (7, 2, width)
            with self.subTest(dtype=name, shape=shape, layout=permutation):
                data, x = self.input_with_layout(name, shape, permutation)
                nested = ft.amax(ft.amax(x, axis=1), axis=1)
                first = ft.amax(x, axis=2, keepdims=True)
                expanded = ft.amax(ft.broadcast_to(first, shape), axis=(1, 2))
                rows = ft.amax(x, axis=2)
                ft.evaluate(nested, expanded, rows)
                target_shape, expected = maximum_reference(data, shape, (1, 2))
                for result in (nested, expanded):
                    self.assert_maximum(result, name, target_shape, expected)
                target_shape, expected = maximum_reference(data, shape, (2,))
                self.assert_maximum(rows, name, target_shape, expected)


if __name__ == '__main__':
    unittest.main()

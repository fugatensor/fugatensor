"""Public CUDA maximum matrices for warp tails, layouts, and IEEE values."""

import pytest
from itertools import product
import math

import fugatensor as ft
from tests._helpers import requires_cuda
from tests.support.reduction_max import (AXES, COMPOSITIONS, LAYOUTS, maximum_dtypes,
    maximum_reference, maximum_values, reduction_matrix, storage_layout)
from tests.support.tensor_primitives import DTYPES, flatten
from tests.support.covering import pairwise


def pytest_generate_tests(metafunc):
    exhaustive = metafunc.config.getoption('--exhaustive')
    if 'maximum_case' in metafunc.fixturenames:
        metafunc.parametrize('maximum_case', reduction_matrix(exhaustive=exhaustive),
                             ids=lambda row: f'{row[0]}-{row[1]}-layout{row[2]}')
    if 'nested_case' in metafunc.fixturenames:
        metafunc.parametrize('nested_case', pairwise((maximum_dtypes(), (33, 65), LAYOUTS), exhaustive=exhaustive),
                             ids=lambda row: f'{row[0]}-width{row[1]}-layout{row[2]}')


@requires_cuda
@pytest.mark.shared_engine
@pytest.mark.usefixtures("engine_context")
class TestCUDAMaximum:
    device = 'cuda'

    def assert_maximum(self, result, name, shape, expected):
        assert (result.shape) == (shape)
        assert (result.dtype) == (DTYPES[name])
        actual = flatten(result.tolist())
        assert (len(actual)) == (len(expected))
        for index, (value, target) in enumerate(zip(actual, expected)):
            if isinstance(target, float) and math.isnan(target):
                assert (math.isnan(value))
            else:
                assert (value) == (target)
                if name in ('fp32', 'fp16', 'bf16') and target == 0:
                    assert (math.copysign(1, value)) == (math.copysign(1, target))

    def input_with_layout(self, name, shape, permutation):
        data = maximum_values(shape, name)
        physical, physical_shape, inverse = storage_layout(data, shape, permutation)
        value = self.engine.tensor(physical, shape=physical_shape, dtype=DTYPES[name])
        return data, value if permutation == (0, 1, 2) else ft.transpose(value, inverse)

    def test_dtype_shape_axes_layout_and_fusion_matrix(self, maximum_case):
        name, shape, permutation = maximum_case
        if name == 'bf16' and not self.supports_bf16:
            pytest.skip('bf16 numerical execution requires sm80+')
        data, x = self.input_with_layout(name, shape, permutation)
        pending = []
        for axes in AXES:
            if any(shape[axis] == 0 for axis in axes):
                with pytest.raises(ValueError, match='nonempty'):
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
            self.assert_maximum(result, name, target_shape, expected)

    def test_nested_max_and_expanded_intermediate_matrix(self, nested_case):
        name, width, permutation = nested_case
        if name == 'bf16' and not self.supports_bf16:
            pytest.skip('bf16 numerical execution requires sm80+')
        shape = (7, 2, width)
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

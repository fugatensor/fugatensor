"""Static argument normalization preserves numerical graph cache contracts."""

import pytest
from itertools import product

import fugatensor as ft
from tests.support.primitive_binding import (
    axis_forms, broadcast_parameters, invalid_axes, reduction_parameters, shape_forms,
)
from tests.support.tensor_primitives import DTYPES, flatten, reduction_reference, values


@pytest.mark.usefixtures("engine_context")
class TestPrimitiveBinding:
    def test_equivalent_reduction_parameters_reuse_numerical_program(self):
        for dtype, shape, maximum, keepdims, axes in reduction_parameters():
            operation = ft.amax if maximum else ft.sum
            compilations = None
            for phase, (representation, argument) in enumerate(axis_forms(axes, len(shape))):
                data = values(shape, dtype, phase)
                source = self.engine.tensor(data, shape=shape, dtype=DTYPES[dtype])
                result = operation(source, argument, keepdims=keepdims)
                target, expected = reduction_reference(data, shape, axes, maximum)
                if keepdims:
                    target = tuple(1 if axis in axes else size for axis, size in enumerate(shape))
                assert (result.shape) == (target)
                assert (result.dtype) is (DTYPES[dtype])
                assert (flatten(result.tolist())) == (expected)
                if compilations is not None:
                    assert (self.engine.stats['compilations']) == (compilations)
                compilations = self.engine.stats['compilations']

    def test_automatic_broadcast_matches_explicit_dimensions(self):
        for dtype, shape in broadcast_parameters():
            target = (2, *shape)
            data = values(shape, dtype)
            source = self.engine.tensor(data, shape=shape, dtype=DTYPES[dtype])
            explicit = ft.broadcast_in_dim(source, target, tuple(range(1, len(target))))
            assert (flatten(explicit.tolist())) == (data * 2)
            compilations = self.engine.stats['compilations']
            for phase, (representation, argument) in enumerate(shape_forms(target), 1):
                data = values(shape, dtype, phase)
                source = self.engine.tensor(data, shape=shape, dtype=DTYPES[dtype])
                result = ft.broadcast_to(source, argument)
                assert (result.shape) == (target)
                assert (result.dtype) is (DTYPES[dtype])
                assert (flatten(result.tolist())) == (data * 2)
                assert (self.engine.stats['compilations']) == (compilations)

    def test_invalid_parameters_after_warm_binding_keep_error_contracts(self):
        source = self.engine.tensor(values((2, 3)), shape=(2, 3))
        for operation in (ft.sum, ft.amax):
            operation(source, 1)
            for axis, exception, message in invalid_axes(len(source.shape)):
                with pytest.raises(exception, match=message):
                    operation(source, axis)
        for operation, keepdims in product((ft.sum, ft.amax), (0, 1, None)):
            with pytest.raises(TypeError, match='keepdims'):
                operation(source, 1, keepdims=keepdims)
        ft.broadcast_to(source, (2, 3))
        for target in ((3,), (2, 4)):
            with pytest.raises(ValueError):
                ft.broadcast_to(source, target)

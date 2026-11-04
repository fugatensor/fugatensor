"""Device storage contracts, raw byte transport, and leaf factory matrices."""

import pytest
from array import array
from itertools import product
import math
from unittest import mock

import fugatensor as ft
from fugatensor._ask.cuda import CUDAError
from fugatensor.backends.host import HostStorage
from tests._helpers import requires_cuda
from tests.support.storage import CPU_DTYPES, SHAPES, create, creation_matrix, expected


@pytest.mark.usefixtures("engine_context")
class TestHostTensorStorage:
    def test_cpu_creation_matrix_and_bf16_rejection(self):
        before = self.engine.stats['compilations']
        for dtype, shape, name in creation_matrix((*CPU_DTYPES, ft.bf16)):
            if dtype is ft.bf16:
                with pytest.raises(TypeError, match='CPU tensors support only bool/i32/i64/f32/f16'):
                    create(name, shape, dtype)
                continue
            value = create(name, shape, dtype)
            assert (value.shape) == (shape)
            assert (value.dtype) is (dtype)
            if name != 'empty':
                assert (self.engine.to_flat(value)) == (expected(name, shape, dtype))
        for shape in SHAPES:
            with pytest.raises(TypeError, match='CPU tensors support only bool/i32/i64/f32/f16'):
                ft.from_buffer(bytearray(2 * math.prod(shape)), shape=shape, dtype=ft.bf16, copy=False)
        assert (self.engine.stats['compilations']) == (before)


@requires_cuda
@pytest.mark.usefixtures("engine_context")
class TestCUDATensorStorage:
    device = 'cuda'

    def test_cuda_creation_uses_device_storage_and_bf16_arithmetic(self):
        before = self.engine.stats['compilations']
        with mock.patch.object(HostStorage, '__init__', side_effect=AssertionError('CUDA created CPU storage')):
            for dtype, shape, name in creation_matrix((*CPU_DTYPES, ft.bf16)):
                if dtype is ft.bf16 and not self.supports_bf16:
                    with pytest.raises(CUDAError, match='bf16.*support'):
                        create(name, shape, dtype)
                    continue
                value = create(name, shape, dtype)
                assert (value.shape) == (shape)
                assert (value.dtype) is (dtype)
                assert (value.device.type) == ('cuda')
                if name != 'empty':
                    assert (self.engine.to_flat(value)) == (expected(name, shape, dtype))
        assert (self.engine.stats['compilations']) == (before)
        if not self.supports_bf16:
            return
        for shape in SHAPES:
            value = ft.full(shape, 2, dtype=ft.bf16)
            assert (self.engine.to_flat(ft.square(value))) == ([4] * math.prod(shape))

    def test_bf16_raw_buffer_preserves_bits_and_copy_contract(self):
        bits = b'\xc1\x7f\x00\x80'  # NaN payload and negative zero.
        if not self.supports_bf16:
            with pytest.raises(CUDAError, match='bf16.*support'):
                ft.from_buffer(bits, shape=(2,), dtype=ft.bf16)
            return
        for shape, exporter in product(((2,), (1, 2)), (bytes, bytearray, memoryview)):
            value = ft.from_buffer(exporter(bits), shape=shape, dtype=ft.bf16)
            assert (bytes(value.to_buffer())) == (bits)
        for copy in (False, 1, None):
            error, message = (ValueError, 'copy=True') if copy is False else (TypeError, 'copy must be bool')
            with pytest.raises(error, match=message):
                ft.from_buffer(bits, shape=(2,), dtype=ft.bf16, copy=copy)
        with pytest.raises(TypeError, match='format'):
            ft.from_buffer(array('H', [0, 0]), shape=(2,), dtype=ft.bf16)
        with pytest.raises(ValueError, match='C-contiguous'):
            ft.from_buffer(memoryview(bytes(8))[::2], shape=(2,), dtype=ft.bf16)
        with pytest.raises(ValueError, match='byte count'):
            ft.from_buffer(bytes(3), shape=(2,), dtype=ft.bf16)

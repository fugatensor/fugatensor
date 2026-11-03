"""Device storage contracts, raw byte transport, and leaf factory matrices."""
from array import array
from itertools import product
import math
import unittest
from unittest import mock

import fugatensor as ft
from fugatensor._ask.cuda import CUDAError
from fugatensor.backends.host import HostStorage
from tests._helpers import EngineTestCase, requires_cuda
from tests.support.storage import CPU_DTYPES, SHAPES, create, creation_matrix, expected


class HostTensorStorageTests(EngineTestCase):
    def test_cpu_creation_matrix_and_bf16_rejection(self):
        before = self.engine.stats['compilations']
        for dtype, shape, name in creation_matrix((*CPU_DTYPES, ft.bf16)):
            with self.subTest(dtype=dtype, shape=shape, factory=name):
                if dtype is ft.bf16:
                    with self.assertRaisesRegex(TypeError, 'CPU tensors support only bool/i32/i64/f32/f16'):
                        create(name, shape, dtype)
                    continue
                value = create(name, shape, dtype)
                self.assertEqual(value.shape, shape)
                self.assertIs(value.dtype, dtype)
                if name != 'empty':
                    self.assertEqual(self.engine.to_flat(value), expected(name, shape, dtype))
        for shape in SHAPES:
            with self.subTest(shape=shape, factory='borrow_buffer'):
                with self.assertRaisesRegex(TypeError, 'CPU tensors support only bool/i32/i64/f32/f16'):
                    ft.from_buffer(bytearray(2 * math.prod(shape)), shape=shape, dtype=ft.bf16, copy=False)
        self.assertEqual(self.engine.stats['compilations'], before)


@requires_cuda
class CUDATensorStorageTests(EngineTestCase):
    device = 'cuda'

    def test_cuda_creation_uses_device_storage_and_bf16_arithmetic(self):
        before = self.engine.stats['compilations']
        with mock.patch.object(HostStorage, '__init__', side_effect=AssertionError('CUDA created CPU storage')):
            for dtype, shape, name in creation_matrix((*CPU_DTYPES, ft.bf16)):
                with self.subTest(dtype=dtype, shape=shape, factory=name):
                    if dtype is ft.bf16 and not self.supports_bf16:
                        with self.assertRaisesRegex(CUDAError, 'bf16.*support'):
                            create(name, shape, dtype)
                        continue
                    value = create(name, shape, dtype)
                    self.assertEqual(value.shape, shape)
                    self.assertIs(value.dtype, dtype)
                    self.assertEqual(value.device.type, 'cuda')
                    if name != 'empty':
                        self.assertEqual(self.engine.to_flat(value), expected(name, shape, dtype))
        self.assertEqual(self.engine.stats['compilations'], before)
        if not self.supports_bf16:
            return
        for shape in SHAPES:
            with self.subTest(shape=shape, operation='square'):
                value = ft.full(shape, 2, dtype=ft.bf16)
                self.assertEqual(self.engine.to_flat(ft.square(value)), [4] * math.prod(shape))

    def test_bf16_raw_buffer_preserves_bits_and_copy_contract(self):
        bits = b'\xc1\x7f\x00\x80'  # NaN payload and negative zero.
        if not self.supports_bf16:
            with self.assertRaisesRegex(CUDAError, 'bf16.*support'):
                ft.from_buffer(bits, shape=(2,), dtype=ft.bf16)
            return
        for shape, exporter in product(((2,), (1, 2)), (bytes, bytearray, memoryview)):
            with self.subTest(shape=shape, exporter=exporter):
                value = ft.from_buffer(exporter(bits), shape=shape, dtype=ft.bf16)
                self.assertEqual(bytes(value.to_buffer()), bits)
        for copy in (False, 1, None):
            with self.subTest(copy=copy):
                error, message = (ValueError, 'copy=True') if copy is False else (TypeError, 'copy must be bool')
                with self.assertRaisesRegex(error, message):
                    ft.from_buffer(bits, shape=(2,), dtype=ft.bf16, copy=copy)
        with self.assertRaisesRegex(TypeError, 'format'):
            ft.from_buffer(array('H', [0, 0]), shape=(2,), dtype=ft.bf16)
        with self.assertRaisesRegex(ValueError, 'C-contiguous'):
            ft.from_buffer(memoryview(bytes(8))[::2], shape=(2,), dtype=ft.bf16)
        with self.assertRaisesRegex(ValueError, 'byte count'):
            ft.from_buffer(bytes(3), shape=(2,), dtype=ft.bf16)


if __name__ == '__main__':
    unittest.main()

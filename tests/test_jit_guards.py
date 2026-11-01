"""Regression checks for tracing isolation, specialization and engine lifetime."""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import fugatensor as ft


class _MetadataSession:
    """Only tensor allocation metadata, for launch rejection before GPU work."""
    capability = (8, 6)
    stats = {}

    def __init__(self, device):
        self.ordinal = device
        self.closed = False

    def from_host(self, values, *, shape, dtype):
        data = list(values)
        return SimpleNamespace(shape=shape, dtype=dtype, to_host=lambda: list(data))

    def close(self):
        self.closed = True


class JITGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = os.environ.get('ASK_OPT')
        if not cls.driver:
            raise unittest.SkipTest('set ASK_OPT to run real ASK host JIT guard tests')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='fugatensor-guards-')
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(ask_opt=self.driver, cache_dir=self.directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)

    def test_real_tensor_reads_creation_and_eager_are_forbidden_only_during_trace(self):
        captured = ft.tensor([3, 4], dtype='i32')
        x = ft.tensor([1, 2], dtype='i32')

        @ft.jit
        def valid(value):
            return value + 1

        actions = {
            'captured_read': lambda: captured.tolist(),
            'captured_storage_read': lambda: captured.storage.to_flat(),
            'create_tensor': lambda: ft.tensor([7, 8], dtype='i32'),
            'captured_eager': lambda: captured + 2,
        }
        for name, action in actions.items():
            with self.subTest(action=name):
                @ft.jit
                def invalid(value):
                    action()
                    return value + 1

                before = self.engine.stats['compilations']
                with self.assertRaisesRegex(TypeError, 'while tracing'):
                    invalid(x)
                self.assertEqual(invalid.cache_info()['specializations'], 0)
                self.assertEqual(self.engine.stats['compilations'], before)
                # A failed trace must restore the context before the next call.
                self.assertEqual(captured.tolist(), [3, 4])
                self.assertEqual(valid(x).tolist(), [2, 3])
        self.assertEqual(valid.cache_info()['compilations'], 1)
        self.assertEqual(valid.cache_info()['cache_hits'], len(actions) - 1)

    def test_nested_jit_cannot_execute_on_a_captured_real_tensor(self):
        captured = ft.tensor([3, 4], dtype='i32')

        @ft.jit
        def inner(value):
            return value * 2

        @ft.jit
        def outer(value):
            inner(captured)
            return value + 1

        x = ft.tensor([1, 2], dtype='i32')
        with self.assertRaisesRegex(TypeError, 'nested'):
            outer(x)
        self.assertEqual(inner.cache_info()['compilations'], 0)
        self.assertEqual(outer.cache_info()['compilations'], 0)
        self.assertEqual(inner(x).tolist(), [2, 4])

    def test_static_arguments_specialize_but_tensor_values_do_not(self):
        traces = 0

        @ft.jit
        def compute(value, factor=2, *, mode='multiply'):
            nonlocal traces
            traces += 1
            return value * factor if mode == 'multiply' else value + factor

        x = ft.tensor([1, 2], dtype='i32')
        changed = ft.tensor([5, 6], dtype='i32')
        self.assertEqual(compute(x).tolist(), [2, 4])
        self.assertEqual(compute(changed, factor=2, mode='multiply').tolist(), [10, 12])
        self.assertEqual(compute(x, 3).tolist(), [3, 6])
        self.assertEqual(compute(x, factor=3, mode='add').tolist(), [4, 5])
        self.assertEqual(traces, 3)
        self.assertEqual(compute.cache_info(), {'compilations': 3, 'cache_hits': 1, 'specializations': 3})

    def test_signed_zero_static_constants_have_distinct_specializations(self):
        @ft.jit
        def scale(value, factor):
            return value * factor

        x = ft.tensor([1.0], dtype='f32')
        positive = scale(x, 0.0).tolist()[0]
        negative = scale(x, -0.0).tolist()[0]
        self.assertEqual(math.copysign(1.0, positive), 1.0)
        self.assertEqual(math.copysign(1.0, negative), -1.0)
        self.assertEqual(scale.cache_info()['specializations'], 2)

    def test_close_evicts_only_that_engine_and_does_not_retain_it(self):
        @ft.jit
        def increment(value):
            return value + 1

        baseline = ft.tensor([1], dtype='i32')
        self.assertEqual(increment(baseline).tolist(), [2])
        another = ft.Engine(ask_opt=self.driver, cache_dir=Path(self.directory.name) / 'another')
        x = another.tensor([3], dtype='i32')
        result = increment(x)
        self.assertEqual(result.tolist(), [4])
        self.assertEqual(increment.cache_info()['specializations'], 2)
        reference = weakref.ref(another)
        another.close()
        self.assertEqual(increment.cache_info()['specializations'], 1)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            increment(x)
        del result, x, another
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(increment(baseline).tolist(), [2])

    def test_jit_weak_cache_does_not_keep_an_unclosed_cpu_engine_alive(self):
        @ft.jit
        def increment(value):
            return value + 1

        another = ft.Engine(ask_opt=self.driver, cache_dir=Path(self.directory.name) / 'temporary')
        x = another.tensor([3], dtype='i32')
        result = increment(x)
        self.assertEqual(result.tolist(), [4])
        reference = weakref.ref(another)
        del result, x, another
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(increment.cache_info()['specializations'], 0)

    def test_cuda_zero_alias_and_storage_type_rejection_before_compilation(self):
        # This is an API/metadata negative test. Real GPU arithmetic has separate
        # integration coverage; no fake kernel result is counted as execution.
        with mock.patch('fugatensor.engine.ExecutionSession', _MetadataSession):
            with ft.Engine('cuda:0', ask_opt=self.driver,
                           cache_dir=Path(self.directory.name) / 'cuda') as engine:
                x = ft.tensor([1, 2], dtype='f32', device='cuda:0')
                self.assertIs(x.engine, engine)
                self.assertEqual(x.device, 'cuda')
                self.assertEqual(x.tolist(), [1, 2])
                self.assertEqual(engine.session.ordinal, 0)
                with self.assertRaisesRegex(ValueError, 'device differs'):
                    ft.tensor([1], engine=engine, device='cuda:1')

                user_kernel = ft.triton.kernel(
                    Path(__file__).parent / 'fixtures' / 'user_identity.py',
                    entry='identity_kernel', input='inp', output='out',
                    constants=lambda spec: {'N': spec.numel, 'BLOCK': 128},
                    grid=lambda spec: ((spec.numel + 127) // 128, 1, 1),
                )

                @ft.jit
                def unsupported(value):
                    return user_kernel(value).astype('f16')

                with mock.patch.object(engine.compiler, 'compile_cuda') as compile_cuda:
                    with self.assertRaisesRegex(NotImplementedError, 'equal input/output shape and storage dtype'):
                        unsupported(x)
                    compile_cuda.assert_not_called()
                self.assertEqual(unsupported.cache_info()['specializations'], 0)
                self.assertEqual(engine.stats['compilations'], 0)


if __name__ == '__main__':
    unittest.main()

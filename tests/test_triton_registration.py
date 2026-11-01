"""User source is registered verbatim; only its ASK bindings are constructed."""
from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from ask import Compiler
import fugatensor as ft
from fugatensor.tracing import TraceContext


AFFINE = '''import triton
import triton.language as tl

# This is the user's kernel body, not a framework template.
@triton.jit
def affine(in_data, out_data, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(in_data + offsets, mask=offsets < N, other=0)
    answer = value * 2 + 1
    tl.store(out_data + offsets, answer, mask=offsets < N)
'''

NEGATE = '''import triton
import triton.language as tl

@triton.jit
def negate(destination, source, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    loaded = tl.load(source + offsets, mask=offsets < N, other=0)
    tl.store(destination + offsets, -loaded, mask=offsets < N)
'''


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler(executable=os.environ.get('ASK_OPT'))

    def context(self, shape=(2, 5), dtype='f32', device='cuda'):
        # This engine supplies metadata only. Native ASK preparation verifies the
        # emitted graph, but these tests do not pretend to execute Triton on GPU.
        return TraceContext([(shape, dtype)], engine=SimpleNamespace(device=device, capability=(8, 6)))

    def registration(self, **changes):
        options = dict(entry='affine', input='in_data', output='out_data',
                       constants=lambda spec: {'N': spec.numel, 'BLOCK': 128},
                       grid=lambda spec: ((spec.numel + 127) // 128,))
        options.update(changes)
        return ft.triton.from_source(AFFINE, **options)

    def prepare(self, op, context=None):
        context = self.context() if context is None else context
        module = context.finish(op(context.inputs[0]))
        return self.compiler.prepare(module).report['external_requests'][0]

    def test_source_file_is_not_imported_and_remains_a_verbatim_snapshot(self):
        source = AFFINE.replace('\n', '\r\n')
        original_import = builtins.__import__

        def reject_triton(name, *args, **kwargs):
            if name == 'triton' or name.startswith('triton.'):
                raise AssertionError('registration imported the user compiler')
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'user_kernel.py'
            path.write_bytes(source.encode('utf-8'))
            with mock.patch('builtins.__import__', side_effect=reject_triton):
                op = ft.triton.kernel(path, entry='affine', input='in_data', output='out_data',
                                      constants={'N': 10, 'BLOCK': 128}, grid=(1,))
            path.write_text(NEGATE)
            self.assertEqual(op.source, source)
            self.assertEqual(op.source_path, path)
            self.assertEqual(self.prepare(op)['source'], source)

    def test_callbacks_receive_only_frozen_static_metadata(self):
        received = []

        def constants(spec):
            received.append(spec)
            with self.assertRaises(FrozenInstanceError):
                spec.dtype = 'i32'
            self.assertFalse(hasattr(spec, 'storage'))
            self.assertFalse(hasattr(spec, 'tolist'))
            return {'N': spec.numel, 'BLOCK': 128}

        request = self.prepare(self.registration(constants=constants), self.context((3, 129), 'f16'))
        self.assertEqual(received, [ft.triton.TensorSpec((3, 129), 'f16', 'cuda')])
        self.assertEqual(request['constants'], {'N': 387, 'BLOCK': 128})
        self.assertEqual(request['grid'], [4, 1, 1])
        self.assertEqual(request['target'], {'backend': 'cuda', 'arch': 86, 'warp_size': 32})
        self.assertEqual(request['signature'], {'in_data': '*fp16', 'out_data': '*fp16'})

    def test_options_and_constant_dicts_are_copied_at_registration(self):
        constants = {'N': 10, 'BLOCK': 128}
        options = {'num_warps': 8, 'enable_fp_fusion': False}
        op = self.registration(constants=constants, options=options)
        constants['N'] = 999
        options['num_warps'] = 4
        options['enable_fp_fusion'] = True
        request = self.prepare(op)
        self.assertEqual(request['constants']['N'], 10)
        self.assertEqual(request['fixed_options'], {'num_warps': 8, 'enable_fp_fusion': False})

    def test_two_user_kernels_have_distinct_sources_and_explicit_bindings(self):
        first = self.registration()
        second = ft.triton.from_source(NEGATE, entry='negate', input='source', output='destination',
                                       constants=lambda spec: {'N': spec.numel, 'BLOCK': 128}, grid=(1, 1))
        context = self.context()
        module = context.finish(second(first(context.inputs[0])))
        requests = self.compiler.prepare(module).report['external_requests']
        self.assertEqual([request['entry'] for request in requests], ['affine', 'negate'])
        self.assertEqual([request['source'] for request in requests], [AFFINE, NEGATE])
        self.assertEqual(requests[1]['source_bindings'], {'source': 'input:0', 'destination': 'output:0'})
        self.assertEqual(requests[1]['grid'], [1, 1, 1])
        self.assertEqual(context.external_count, 2)

    def test_all_storage_types_change_only_binding_metadata(self):
        for dtype, signature in [('bool', '*i1'), ('i32', '*i32'), ('i64', '*i64'),
                                 ('f32', '*fp32'), ('f16', '*fp16'), ('bf16', '*bf16')]:
            with self.subTest(dtype=dtype):
                request = self.prepare(self.registration(), self.context(dtype=dtype))
                self.assertEqual(request['source'], AFFINE)
                self.assertEqual(request['signature'], {'in_data': signature, 'out_data': signature})

    def test_missing_or_conflicting_source_bindings_fail_before_compilation(self):
        for changes in ({'entry': 'missing'}, {'input': 'missing'}, {'output': 'in_data'},
                        {'constants': {'BLOCK': 128}},
                        {'constants': {'N': 10, 'BLOCK': 128, 'in_data': 0}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.registration(**changes)
        with self.assertRaisesRegex(TypeError, 'input'):
            ft.triton.from_source(AFFINE, entry='affine', output='out_data', grid=(1,))
        op = self.registration(constants=lambda spec: {'BLOCK': 128})
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.prepare(op)

    def test_bad_launch_metadata_and_unsupported_device_are_explicit(self):
        for grid in ((), (0,), (True,), (1, 1, 1, 1), (2**31,)):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                self.registration(grid=grid)
        with self.assertRaisesRegex(TypeError, 'integer constexpr'):
            self.registration(constants={'N': 10.0, 'BLOCK': 128})
        with self.assertRaisesRegex(NotImplementedError, 'CUDA Engine'):
            self.prepare(self.registration(), self.context(device='cpu'))
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.prepare(self.registration(), self.context(shape=(0,)))


if __name__ == '__main__':
    unittest.main()

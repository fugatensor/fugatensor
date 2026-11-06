"""Public attention API contracts and generated CUDA value/gradient coverage."""
from __future__ import annotations

import pytest

import builtins
from itertools import product
import math
from types import SimpleNamespace
from unittest import mock

import fugatensor as ft
from fugatensor._state import active_trace
from fugatensor.tracing import TraceContext
from tests._helpers import requires_cuda
from tests.support.attention import flatten, input_data, reference, reference_backward, attention_cases
from tests.support.selection import select

attention = ft.nn.functional.scaled_dot_product_attention
DTYPES = {name: getattr(ft, name) for name in ('bool', 'int32', 'int64', 'fp32', 'fp16', 'bf16')}
# Shape descriptors are independent of precision, causality and composition.
SHAPES = {
    'tail': (1, 1, 35, 41, 32, 64),
    'wide': (1, 1, 7, 8, 64, 32),
    'batched': (2, 2, 3, 5, 128, 128),
}
COMPOSITIONS = {'plain': (1.0, 1.0), 'scaled': (0.5, 2.0)}


def pytest_generate_tests(metafunc):
    if 'attention_case' not in metafunc.fixturenames:
        return
    dtype_names = tuple(name for name in select('FUGATENSOR_TEST_DTYPES', DTYPES, default=('fp16', 'bf16'))
                        if name in ('fp16', 'bf16'))
    shape_names = select('FUGATENSOR_TEST_ATTENTION_SHAPES', SHAPES)
    causal_names = select('FUGATENSOR_TEST_ATTENTION_CAUSAL', ('false', 'true'))
    compositions = select('FUGATENSOR_TEST_ATTENTION_COMPOSITIONS', COMPOSITIONS)
    rows = attention_cases(dtype_names, shape_names, causal_names, compositions,
                           exhaustive=metafunc.config.getoption('--exhaustive'))
    metafunc.parametrize('attention_case', rows, ids=lambda row: '-'.join(row))


class TestAttentionContract:
    def context(self, shapes=None, dtypes=None, *, device='cuda', capability=(8, 0)):
        shapes = shapes or ((1, 1, 3, 32), (1, 1, 5, 32), (1, 1, 5, 64))
        dtypes = dtypes or (ft.fp16,) * 3
        return TraceContext(list(zip(shapes, dtypes)),
                            engine=SimpleNamespace(device=device, capability=capability))

    def test_public_contract_rejects_invalid_inputs_before_kernel_registration(self):
        valid = self.context().inputs
        with mock.patch('fugatensor.nn.functional.attention', side_effect=AssertionError('registered invalid input')):
            for value in (None, 1, 'true'):
                with pytest.raises(TypeError, match='is_causal'):
                    attention(*valid, is_causal=value)
            with pytest.raises(TypeError, match='three Tensor'):
                attention(valid[0], valid[1], None)
            changes = (
                ({'device': 'cpu'}, ValueError, 'CUDA'),
                ({'capability': (7, 5)}, ValueError, 'capability'),
                ({'dtypes': (ft.fp32,) * 3}, TypeError, 'matching'),
                ({'dtypes': (ft.fp16, ft.bf16, ft.fp16)}, TypeError, 'matching'),
                ({'shapes': ((1, 3, 32), (1, 1, 5, 32), (1, 1, 5, 64))}, ValueError, 'batch, head'),
                ({'shapes': ((1, 1, 3, 32), (2, 1, 5, 32), (1, 1, 5, 64))}, ValueError, 'disagree'),
                ({'shapes': ((1, 1, 3, 32), (1, 1, 5, 32), (1, 1, 4, 64))}, ValueError, 'disagree'),
                ({'shapes': ((1, 1, 3, 16), (1, 1, 5, 16), (1, 1, 5, 64))}, ValueError, 'feature'),
                ({'shapes': ((1, 1, 0, 32), (1, 1, 5, 32), (1, 1, 5, 64))}, ValueError, 'positive'),
                ({'shapes': ((1, 1, 3, 32), (1, 1, 0, 32), (1, 1, 0, 64))}, ValueError, 'positive'),
            )
            for configuration, error, message in changes:
                with pytest.raises(error, match=message):
                    attention(*self.context(**configuration).inputs)
            with pytest.raises(RuntimeError, match='same device'):
                attention(valid[0], self.context(device='cuda:1').inputs[1], valid[2])

    def test_public_api_traces_without_importing_triton(self, cleanup):
        context = self.context()
        original_import = builtins.__import__

        def block_triton(name, *args, **kwargs):
            if name == 'triton' or name.startswith('triton.'):
                raise AssertionError('attention must not import Triton into the host')
            return original_import(name, *args, **kwargs)

        token = active_trace.set(context)
        cleanup.callback(active_trace.reset, token)
        class AttentionBlock(ft.fun):
            def forward(self, query, key, value):
                return attention(query, key, value, is_causal=True)

        with mock.patch('builtins.__import__', side_effect=block_triton):
            output = AttentionBlock()(*context.inputs)
        assert (output.shape) == ((1, 1, 3, 64))
        assert (output.dtype) is (ft.fp16)
        context.finish(output).verify()


@requires_cuda
@pytest.mark.shared_engine
@pytest.mark.usefixtures("engine_context")
class TestAttentionCUDAExecution:
    device = 'cuda'
    enable_triton = False

    def assert_values(self, actual, expected, *, dtype, backward=False):
        assert (len(actual)) == (len(expected))
        tolerance = (0.02 if dtype is ft.fp16 else 0.04) if backward else (0.003 if dtype is ft.fp16 else 0.02)
        assert (all((math.isfinite(item) for item in actual)))
        for index, (value, target) in enumerate(zip(actual, expected)):
            assert (abs(value - target)) <= (tolerance + tolerance * abs(target)), f'element {index}: {value} != {target}'

    def test_generated_forward_backward_matrix(self, attention_case):
        dtype_name, shape_name, causal_name, combination = attention_case
        self.run_case(dtype=DTYPES[dtype_name], dimensions=SHAPES[shape_name],
                      causal=causal_name == 'true', composition=combination)

    def run_case(self, *, dtype, dimensions, causal, composition):
        batch, heads, m, n, d, dv = dimensions
        shapes = ((batch, heads, m, d), (batch, heads, n, d), (batch, heads, n, dv))
        inputs = tuple(ft.tensor(input_data(shape, period=period, scale=scale), dtype=dtype)
                       for shape, period, scale in zip(shapes, (23, 19, 17), (0.03125, 0.03125, 0.0625)))
        q, k, v = inputs
        q_scale, output_scale = COMPOSITIONS[composition]
        shifted = q if q_scale == 1 else q * q_scale
        output = attention(shifted, k, v, is_causal=causal)
        if output_scale != 1:
            output = output * output_scale
        actual = flatten(output.tolist())
        forward_steps = [step['kind'] for step in self.engine.last_execution_plan['steps']]
        assert (forward_steps) == (['external'] if composition == 'plain' else ['owned', 'external', 'owned'])
        assert (output.shape) == ((batch, heads, m, dv))
        assert (output.dtype) is (dtype)
        values = tuple(flatten(item.tolist()) for item in (shifted, k, v))
        parameters = dict(batch=batch, heads=heads, m=m, n=n, d=d, dv=dv, causal=causal)
        expected = [output_scale * value for value in reference(*values, **parameters)]
        self.assert_values(actual, expected, dtype=dtype)
        seed = ft.tensor(input_data(output.shape, period=13, scale=0.0625), dtype=dtype)
        gradients = ft.grad(output, inputs, seed=seed)
        ft.evaluate(*gradients)
        backward_steps = [step['kind'] for step in self.engine.last_execution_plan['steps']]
        assert (backward_steps.count('external')) == (3)
        expected_gradients = reference_backward(*values,
            [output_scale * value for value in flatten(seed.tolist())], **parameters)
        expected_gradients = ([q_scale * value for value in expected_gradients[0]], *expected_gradients[1:])
        for name, tensor, derivative, expected in zip(('query', 'key', 'value'), inputs, gradients, expected_gradients):
            assert (derivative.shape) == (tensor.shape)
            assert (derivative.dtype) is (dtype)
            actual = flatten(derivative.tolist())
            self.assert_values(actual, expected, dtype=dtype, backward=True)
            assert (max((abs(value - target) for value, target in zip(actual, expected)))) < (0.02)

    def test_independent_engines_on_same_device_are_rejected(self):
        q, k, v = (ft.ones(1, 1, 2, 32, dtype=ft.fp16) for _ in range(3))
        with ft.Engine('cuda', compilers=self.engine.compilers) as other:
            foreign = ft.ones(1, 1, 2, 32, dtype=ft.fp16)
            with mock.patch.object(self.engine, '_compile', side_effect=AssertionError('compiled invalid inputs')):
                with pytest.raises(ValueError, match='Engine|engine'):
                    attention(q, foreign, v)

    def test_higher_derivatives_fail_before_compile(self):
        q, k, v = (ft.ones(1, 1, 2, 32, dtype=ft.fp16) for _ in range(3))
        output = attention(q, k, v)
        seed = ft.ones(1, 1, 2, 32, dtype=ft.fp16)
        first = ft.grad(output, q, seed=seed)
        with mock.patch.object(self.engine, '_compile', side_effect=AssertionError('compiled before validation')):
            with pytest.raises(ValueError, match='external|derivative|VJP'):
                ft.grad(first, q, seed=ft.ones_like(first))

    def test_varlen_packed_thd_forward_backward(self):
        seqlens = [16, 32, 24]
        b = len(seqlens)
        h = 4
        d = 64
        total_tokens = sum(seqlens)
        max_s = max(seqlens)

        cu_seqlens = [0]
        for s in seqlens:
            cu_seqlens.append(cu_seqlens[-1] + s)

        q_val = [[[0.01 * ((i + j + k) % 13) for k in range(d)] for j in range(h)] for i in range(total_tokens)]
        k_val = [[[0.01 * ((i + j + k) % 17) for k in range(d)] for j in range(h)] for i in range(total_tokens)]
        v_val = [[[0.01 * ((i + j + k) % 19) for k in range(d)] for j in range(h)] for i in range(total_tokens)]

        q = ft.tensor(q_val, dtype=ft.fp16, device='cuda')
        k = ft.tensor(k_val, dtype=ft.fp16, device='cuda')
        v = ft.tensor(v_val, dtype=ft.fp16, device='cuda')
        cu_q = ft.tensor(cu_seqlens, dtype=ft.int32, device='cuda')
        cu_kv = ft.tensor(cu_seqlens, dtype=ft.int32, device='cuda')

        varlen_attention = ft.nn.functional.scaled_dot_product_attention_varlen
        out = varlen_attention(q, k, v, cu_q, cu_kv, max_s, max_s, is_causal=True)
        ft.evaluate(out)

        assert out.shape == (total_tokens, h, d)
        assert out.dtype is ft.fp16

        # Check against sliced reference
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            seq_len = end - start
            sub_q = [q_val[idx] for idx in range(start, end)]
            sub_k = [k_val[idx] for idx in range(start, end)]
            sub_v = [v_val[idx] for idx in range(start, end)]
            q_trans = [[[sub_q[s][head][dim] for dim in range(d)] for s in range(seq_len)] for head in range(h)]
            k_trans = [[[sub_k[s][head][dim] for dim in range(d)] for s in range(seq_len)] for head in range(h)]
            v_trans = [[[sub_v[s][head][dim] for dim in range(d)] for s in range(seq_len)] for head in range(h)]
            ref_slice = reference(flatten([q_trans]), flatten([k_trans]), flatten([v_trans]),
                                  batch=1, heads=h, m=seq_len, n=seq_len, d=d, dv=d, causal=True)
            actual_sub = out.tolist()[start:end]
            actual_trans = flatten([[[actual_sub[s][head][dim] for dim in range(d)] for s in range(seq_len)] for head in range(h)])
            self.assert_values(actual_trans, ref_slice, dtype=ft.fp16)

        # Backward
        seed = ft.ones_like(out)
        dq, dk, dv = ft.grad(out, (q, k, v), seed=seed)
        ft.evaluate(dq, dk, dv)
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert dq.dtype is ft.fp16

"""Validate FlashAttention-2 forward and backward through ASK execution plans.

Run: .venv/bin/python examples/flash_attention2.py [--dtype bf16] [--causal] [--backward]
Contiguous tensors only; no dropout, bias, variable length or performance claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import fugatensor as ft

KERNEL_PATH = Path(__file__).with_name('kernels') / 'flash_attention2.py'
BLOCK = 32


def _flat(value):
    if isinstance(value, (list, tuple)):
        return [item for part in value for item in _flat(part)]
    return [value]


def _dimensions(specs):
    q, k, v = specs[:3]
    if any(len(spec.shape) != 4 for spec in (q, k, v)):
        raise ValueError('attention inputs require [batch, head, sequence, feature]')
    b, h, m, d = q.shape
    bk, hk, n, dk = k.shape
    bv, hv, nv, dv = v.shape
    if (b, h, d) != (bk, hk, dk) or (b, h, n) != (bv, hv, nv):
        raise ValueError('attention batch, head, key and value dimensions disagree')
    if d not in (32, 64, 128) or dv not in (32, 64, 128):
        raise ValueError('validated attention feature sizes are 32, 64 or 128')
    if q.dtype not in ('f16', 'bf16') or any(spec.dtype != q.dtype for spec in (k, v)):
        raise TypeError('attention requires matching f16 or bf16 inputs')
    return b, h, m, n, d, dv


def _stats_dimensions(specs):
    q, k, o, do = specs
    b, h, m, d = q.shape
    bk, hk, n, dk = k.shape
    bo, ho, mo, dv = o.shape
    if (b, h, d) != (bk, hk, dk) or (b, h, m) != (bo, ho, mo):
        raise ValueError('attention stats batch, head or sequence dimensions disagree')
    if (b, h, m, dv) != do.shape:
        raise ValueError('attention output and output gradient shapes disagree')
    if q.dtype not in ('f16', 'bf16') or any(spec.dtype != q.dtype for spec in (k, o, do)):
        raise TypeError('attention requires matching f16 or bf16 inputs')
    if d not in (32, 64, 128) or dv not in (32, 64, 128):
        raise ValueError('validated attention feature sizes are 32, 64 or 128')
    return b, h, m, n, d, dv


def _make_kernels(causal):
    def constants(specs):
        _, _, m, n, d, dv = _dimensions(specs)
        return {'M': m, 'N': n, 'D': d, 'DV': dv, 'CAUSAL': int(causal),
                'BLOCK_M': BLOCK, 'BLOCK_N': BLOCK}

    def forward_shape(specs):
        b, h, m, _, _, dv = _dimensions(specs)
        return b, h, m, dv

    def query_grid(specs):
        b, h, m, _, _, _ = _dimensions(specs)
        return ((m + BLOCK - 1) // BLOCK, b * h)

    def key_grid(specs):
        b, h, _, n, _, _ = _dimensions(specs)
        return ((n + BLOCK - 1) // BLOCK, b * h)

    def stats_constants(specs):
        b, h, m, n, d, dv = _stats_dimensions(specs)
        return {'M': m, 'N': n, 'D': d, 'DV': dv, 'CAUSAL': int(causal),
                'BLOCK_M': BLOCK, 'BLOCK_N': BLOCK}

    def stats_shape(specs):
        b, h, m, _, _, _ = _stats_dimensions(specs)
        return b, h, m, 2

    def stats_grid(specs):
        b, h, m, _, _, _ = _stats_dimensions(specs)
        return ((m + BLOCK - 1) // BLOCK, b * h)

    def dq_shape(specs):
        return specs[0].shape

    def dk_shape(specs):
        return specs[1].shape

    def dv_shape(specs):
        return specs[2].shape

    def register(entry, inputs, output, constant_fn, grid_fn, shape_fn, dtype=None):
        return ft.triton.kernel(
            KERNEL_PATH, entry=entry, inputs=inputs, output=output,
            constants=constant_fn, grid=grid_fn, output_shape=shape_fn,
            output_dtype=dtype, options={'num_warps': 4, 'num_stages': 2})

    return (
        register('flash_attention2', ('Q', 'K', 'V'), 'O', constants, query_grid, forward_shape),
        register('flash_attention2_bwd_stats', ('Q', 'K', 'O', 'dO'), 'Stats',
                 stats_constants, stats_grid, stats_shape, 'f32'),
        register('flash_attention2_bwd_dq', ('Q', 'K', 'V', 'dO', 'Stats'), 'dQ',
                 constants, query_grid, dq_shape),
        register('flash_attention2_bwd_dk', ('Q', 'K', 'V', 'dO', 'Stats'), 'dK',
                 constants, key_grid, dk_shape),
        register('flash_attention2_bwd_dv', ('Q', 'K', 'V', 'dO', 'Stats'), 'dV',
                 constants, key_grid, dv_shape),
    )


class FlashAttention2(ft.fun):
    """Custom forward/backward boundary around four independent Triton kernels."""
    def __init__(self, *, causal=False):
        if type(causal) is not bool:
            raise TypeError('causal must be a bool')
        self.causal = causal
        (self._fuga_forward, self._fuga_stats, self._fuga_dq, self._fuga_dk, self._fuga_dv) = _make_kernels(causal)

    def forward(self, ctx, q, k, v):
        output = self._fuga_forward(q, k, v)
        ctx.save_for_backward(q, k, v, output)
        return output

    def backward(self, ctx, do):
        q, k, v, output = ctx.saved_tensors
        stats = self._fuga_stats(q, k, output, do)
        return (self._fuga_dq(q, k, v, do, stats),
                self._fuga_dk(q, k, v, do, stats),
                self._fuga_dv(q, k, v, do, stats))


_ATTENTION = {}


def attention(*, causal=False):
    """Return the cached custom operator for one causal configuration."""
    operation = _ATTENTION.get(causal)
    if operation is None:
        operation = _ATTENTION[causal] = FlashAttention2(causal=causal)
    return operation


def sample_inputs(*, batch, heads, m, n, d, dv, seed=42, scale=1.0):
    import random
    rng = random.Random(seed)
    return tuple([rng.uniform(-scale, scale) for _ in range(count)]
                 for count in (batch * heads * m * d, batch * heads * n * d,
                               batch * heads * n * dv))


def reference(q, k, v, *, batch, heads, m, n, d, dv, causal):
    """Small FP64 host oracle over already quantized, flattened values."""
    output = []
    scale = d ** -0.5
    for bh in range(batch * heads):
        for row in range(m):
            keys = min(n, row + 1) if causal else n
            scores = [math.fsum(q[(bh * m + row) * d + dim] *
                                k[(bh * n + col) * d + dim] for dim in range(d)) * scale
                      for col in range(keys)]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = math.fsum(weights)
            probabilities = [weight / denominator for weight in weights]
            output.extend(math.fsum(probability * v[(bh * n + col) * dv + dim]
                                    for col, probability in enumerate(probabilities))
                          for dim in range(dv))
    return output


def reference_backward(q, k, v, do, *, batch, heads, m, n, d, dv, causal):
    """Analytic softmax-attention gradient for the same quantized values."""
    dq = [0.0] * (batch * heads * m * d)
    dk = [0.0] * (batch * heads * n * d)
    dvalue = [0.0] * (batch * heads * n * dv)
    scale = d ** -0.5
    for bh in range(batch * heads):
        for row in range(m):
            keys = min(n, row + 1) if causal else n
            if keys == 0:
                continue
            scores = [math.fsum(q[(bh * m + row) * d + dim] *
                                k[(bh * n + col) * d + dim] for dim in range(d)) * scale
                      for col in range(keys)]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = math.fsum(weights)
            probabilities = [weight / denominator for weight in weights]
            output = [math.fsum(probability * v[(bh * n + col) * dv + dim]
                                for col, probability in enumerate(probabilities))
                      for dim in range(dv)]
            delta = math.fsum(do[(bh * m + row) * dv + dim] * output[dim] for dim in range(dv))
            dp = [math.fsum(do[(bh * m + row) * dv + dim] * v[(bh * n + col) * dv + dim]
                            for dim in range(dv)) for col in range(keys)]
            ds = [probabilities[col] * (dp[col] - delta) for col in range(keys)]
            for dim in range(d):
                dq[(bh * m + row) * d + dim] = scale * math.fsum(
                    ds[col] * k[(bh * n + col) * d + dim] for col in range(keys))
            for col in range(keys):
                for dim in range(d):
                    dk[(bh * n + col) * d + dim] += scale * ds[col] * q[(bh * m + row) * d + dim]
                for dim in range(dv):
                    dvalue[(bh * n + col) * dv + dim] += probabilities[col] * do[(bh * m + row) * dv + dim]
    return dq, dk, dvalue


def _compare(actual, expected, dtype, label, *, gradient=False):
    if gradient:
        atol, rtol = (0.02, 0.02) if dtype == 'f16' else (0.04, 0.04)
    else:
        atol, rtol = (0.003, 0.003) if dtype == 'f16' else (0.02, 0.02)
    if len(actual) != len(expected):
        raise AssertionError(f'{label} length differs from the reference')
    error = max(abs(value - target) for value, target in zip(actual, expected))
    if any(not math.isfinite(value) or abs(value - target) > atol + rtol * abs(target)
           for value, target in zip(actual, expected)):
        raise AssertionError(f'{label} differs from the reference; max_abs_error={error}, '
                             f'atol={atol}, rtol={rtol}')
    return error


def validate(*, dtype='f16', causal=False, backward=False, m=35, n=41, d=32, dv=64):
    operation = attention(causal=causal)
    with ft.Engine('cuda') as engine:
        def make(count, period, scale, shape):
            return ft.tensor([((i * 7) % period - period // 2) * scale for i in range(count)],
                             shape=shape, dtype=dtype, requires_grad=backward)
        q = make(m * d, 23, 0.03125, (1, 1, m, d))
        k = make(n * d, 19, 0.03125, (1, 1, n, d))
        v = make(n * dv, 17, 0.0625, (1, 1, n, dv))
        shifted = q * 0.5
        output = operation(shifted, k, v) * 2
        # Reads happen only after the complete graph has been submitted.
        actual = engine.to_flat(output)
        forward_report = engine.last_report
        forward_steps = [step['kind'] for step in forward_report['execution_plan']['steps']]
        expected = [2.0 * value for value in reference(
            _flat(shifted.tolist()), _flat(k.tolist()), _flat(v.tolist()), causal=causal,
            batch=1, heads=1, m=m, n=n, d=d, dv=dv)]
        forward_error = _compare(actual, expected, dtype, 'attention forward')
        receipt = {'dtype': dtype, 'causal': causal, 'q_shape': q.shape, 'k_shape': k.shape,
                   'v_shape': v.shape, 'output_shape': output.shape,
                   'max_abs_error': forward_error, 'steps': forward_steps}
        if backward:
            seed_values = [((index * 5) % 13 - 6) * 0.0625 for index in range(len(actual))]
            seed = ft.tensor(seed_values, shape=output.shape, dtype=dtype)
            output.backward(seed)
            if q.grad is None or k.grad is None or v.grad is None:
                raise AssertionError('attention backward did not populate all input gradients')
            dq, dk, dv = reference_backward(
                _flat(shifted.tolist()), _flat(k.tolist()), _flat(v.tolist()),
                [2.0 * value for value in _flat(seed.tolist())], causal=causal,
                batch=1, heads=1, m=m, n=n, d=d, dv=dv)
            dq = [0.5 * value for value in dq]
            backward_report = engine.last_report
            backward_steps = [step['kind'] for step in backward_report['execution_plan']['steps']]
            receipt.update({
                'backward_plan_steps': backward_steps,
                'dq_error': _compare(_flat(q.grad.tolist()), dq, dtype, 'attention dQ', gradient=True),
                'dk_error': _compare(_flat(k.grad.tolist()), dk, dtype, 'attention dK', gradient=True),
                'dv_error': _compare(_flat(v.grad.tolist()), dv, dtype, 'attention dV', gradient=True),
            })
        receipt['verified'] = True
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=('f16', 'bf16'), default='f16')
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate(dtype=args.dtype, causal=args.causal, backward=args.backward), indent=2))


if __name__ == '__main__':
    main()

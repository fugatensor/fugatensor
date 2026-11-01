"""Validate a FlashAttention-2-style forward through ASK's multi-kernel plan.

Run: .venv/bin/python examples/flash_attention2.py [--dtype bf16] [--causal]
Contiguous tensors only; no backward, dropout, bias or performance claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import fugatensor as ft


def attention(*, causal=False):
    def dimensions(specs):
        q, k, v = specs
        if any(len(spec.shape) != 4 for spec in specs):
            raise ValueError('attention inputs require [batch, head, sequence, feature]')
        b, h, m, d = q.shape
        bk, hk, n, dk = k.shape
        bv, hv, nv, dv = v.shape
        if (b, h, d) != (bk, hk, dk) or (b, h, n) != (bv, hv, nv):
            raise ValueError('attention batch, head, key and value dimensions disagree')
        if d not in (32, 64, 128) or dv not in (32, 64, 128):
            raise ValueError('validated attention feature sizes are 32, 64 or 128')
        if q.dtype not in ('f16', 'bf16') or any(spec.dtype != q.dtype for spec in specs):
            raise TypeError('attention requires matching f16 or bf16 inputs')
        return b, h, m, n, d, dv

    def constants(specs):
        _, _, m, n, d, dv = dimensions(specs)
        return {'M': m, 'N': n, 'D': d, 'DV': dv, 'CAUSAL': int(causal),
                'BLOCK_M': 32, 'BLOCK_N': 32}

    def shape(specs):
        b, h, m, _, _, dv = dimensions(specs)
        return b, h, m, dv

    def grid(specs):
        b, h, m, _, _, _ = dimensions(specs)
        return (m + 31) // 32, b * h

    return ft.triton.kernel(Path(__file__).with_name('kernels') / 'flash_attention2.py',
        entry='flash_attention2', inputs=('Q', 'K', 'V'), output='O',
        constants=constants, grid=grid, output_shape=shape,
        options={'num_warps': 4, 'num_stages': 2})


def reference(q, k, v, *, causal=False):
    """Small CPU test oracle; never used by the FugaTensor execution path."""
    b, h, m, d = len(q), len(q[0]), len(q[0][0]), len(q[0][0][0])
    n, dv = len(k[0][0]), len(v[0][0][0])
    result = []
    for batch in range(b):
        for head in range(h):
            for row in range(m):
                scores = [sum(q[batch][head][row][t] * k[batch][head][col][t]
                              for t in range(d)) / math.sqrt(d)
                          for col in range(n) if not causal or col <= row]
                maximum = max(scores)
                weights = [math.exp(score - maximum) for score in scores]
                denominator = sum(weights)
                for feature in range(dv):
                    result.append(2 * sum(weight * v[batch][head][col][feature]
                                           for col, weight in enumerate(weights)) / denominator)
    return result


def validate(*, dtype='f16', causal=False, m=35, n=41, d=32, dv=64):
    op = attention(causal=causal)
    with ft.Engine('cuda') as engine:
        def make(count, period, scale, shape):
            return ft.tensor([((i * 7) % period - period // 2) * scale for i in range(count)],
                             shape=shape, dtype=dtype)
        q = make(m * d, 23, 0.03125, (1, 1, m, d))
        k = make(n * d, 19, 0.03125, (1, 1, n, d))
        v = make(n * dv, 17, 0.0625, (1, 1, n, dv))
        shifted = q * 0.5
        output = op(shifted, k, v) * 2
        # Reads happen only after the complete graph has been submitted.
        actual = engine.to_flat(output)
        report = engine.last_report
        artifact_dir = engine.last_artifact_dir
        expected = reference(shifted.tolist(), k.tolist(), v.tolist(), causal=causal)
        error = max(abs(a - e) for a, e in zip(actual, expected))
        tolerance = 0.015 if dtype == 'bf16' else 0.003
        if error > tolerance:
            raise AssertionError((dtype, causal, error, tolerance))
        plan = report.get('execution_plan')
        if not plan or len(plan['steps']) < 3:
            raise AssertionError('expected owned -> external attention -> owned execution')
        return {'dtype': dtype, 'causal': causal, 'q_shape': q.shape, 'k_shape': k.shape,
                'v_shape': v.shape, 'output_shape': output.shape, 'max_abs_error': error,
                'steps': [step['kind'] for step in plan['steps']],
                'artifact_dir': str(artifact_dir), 'verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=('f16', 'bf16'), default='f16')
    parser.add_argument('--causal', action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate(dtype=args.dtype, causal=args.causal), indent=2))


if __name__ == '__main__':
    main()

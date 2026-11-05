"""FlashAttention implementation; Triton source is loaded only by the worker."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..dtypes import fp16, bf16, fp32
from ..function import fun
from ..triton import kernel

KERNEL_PATH = Path(__file__).with_name('_kernels') / 'flash_attention2.py'
BLOCK = 32


def _block_size(m, n):
    if m >= 64 and n >= 64:
        return 64
    return 32


def _dimensions(specs):
    q, k, v = specs[:3]
    if any(len(spec.shape) != 4 for spec in (q, k, v)):
        raise ValueError('attention inputs require [batch, head, sequence, feature]')
    b, h, m, d = q.shape
    bk, hk, n, dk = k.shape
    bv, hv, nv, dv = v.shape
    if (b, h, d) != (bk, hk, dk) or (b, h, n) != (bv, hv, nv):
        raise ValueError('attention batch, head, key and value dimensions disagree')
    if min(b, h, m, n, d, dv) <= 0:
        raise ValueError('attention dimensions must be positive')
    if d not in (32, 64, 128) or dv not in (32, 64, 128):
        raise ValueError('supported attention feature sizes are 32, 64 or 128')
    if q.dtype not in (fp16, bf16) or any(spec.dtype != q.dtype for spec in (k, v)):
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
    if q.dtype not in (fp16, bf16) or any(spec.dtype != q.dtype for spec in (k, o, do)):
        raise TypeError('attention requires matching f16 or bf16 inputs')
    if d not in (32, 64, 128) or dv not in (32, 64, 128):
        raise ValueError('supported attention feature sizes are 32, 64 or 128')
    return b, h, m, n, d, dv


def _make_kernels(causal):
    def constants(specs):
        _, _, m, n, d, dv = _dimensions(specs)
        block = _block_size(m, n)
        return {'M': m, 'N': n, 'D': d, 'DV': dv, 'CAUSAL': int(causal),
                'BLOCK_M': block, 'BLOCK_N': block}

    def forward_shape(specs):
        b, h, m, _, _, dv = _dimensions(specs)
        return b, h, m, dv

    def query_grid(specs):
        b, h, m, n, _, _ = _dimensions(specs)
        block = _block_size(m, n)
        return ((m + block - 1) // block, b * h)

    def key_grid(specs):
        b, h, m, n, _, _ = _dimensions(specs)
        block = _block_size(m, n)
        return ((n + block - 1) // block, b * h)

    def stats_constants(specs):
        b, h, m, n, d, dv = _stats_dimensions(specs)
        block = _block_size(m, n)
        return {'M': m, 'N': n, 'D': d, 'DV': dv, 'CAUSAL': int(causal),
                'BLOCK_M': block, 'BLOCK_N': block}

    def stats_shape(specs):
        b, h, m, _, _, _ = _stats_dimensions(specs)
        return b, h, m, 2

    def stats_grid(specs):
        b, h, m, n, _, _ = _stats_dimensions(specs)
        block = _block_size(m, n)
        return ((m + block - 1) // block, b * h)

    def dq_shape(specs):
        return specs[0].shape

    def dk_shape(specs):
        return specs[1].shape

    def dv_shape(specs):
        return specs[2].shape

    def register(entry, inputs, output, constant_fn, grid_fn, shape_fn, dtype=None):
        return kernel(
            KERNEL_PATH, entry=entry, inputs=inputs, output=output,
            constants=constant_fn, grid=grid_fn, output_shape=shape_fn,
            output_dtype=dtype, options={'num_warps': 4, 'num_stages': 2})

    return (
        register('flash_attention2', ('Q', 'K', 'V'), 'O', constants, query_grid, forward_shape),
        register('flash_attention2_bwd_stats', ('Q', 'K', 'O', 'dO'), 'Stats',
                 stats_constants, stats_grid, stats_shape, fp32),
        register('flash_attention2_bwd_dq', ('Q', 'K', 'V', 'dO', 'Stats'), 'dQ',
                 constants, query_grid, dq_shape),
        register('flash_attention2_bwd_dk', ('Q', 'K', 'V', 'dO', 'Stats'), 'dK',
                 constants, key_grid, dk_shape),
        register('flash_attention2_bwd_dv', ('Q', 'K', 'V', 'dO', 'Stats'), 'dV',
                 constants, key_grid, dv_shape),
    )


class _FlashAttention(fun):
    """One forward kernel and four kernels for the first derivative."""
    def __init__(self, *, causal=False):
        if type(causal) is not bool:
            raise TypeError('causal must be a bool')
        self.causal = causal
        (self._fuga_forward, self._fuga_stats, self._fuga_dq, self._fuga_dk, self._fuga_dv) = _make_kernels(causal)

    def forward(self, q, k, v):
        return self._fuga_forward(q, k, v)

    def vjp(self, primals, outputs, cotangents):
        q, k, v = primals
        output, = outputs
        do, = cotangents
        stats = self._fuga_stats(q, k, output, do)
        return (self._fuga_dq(q, k, v, do, stats),
                self._fuga_dk(q, k, v, do, stats),
                self._fuga_dv(q, k, v, do, stats))


@lru_cache(maxsize=2)
def attention(causal):
    return _FlashAttention(causal=causal)

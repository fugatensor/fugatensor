"""Dense FlashAttention-2-style forward, pointer ABI for Ampere and newer.

Adapted from the online-softmax forward algorithm in Triton's fused attention
example (OpenAI kernel team), pinned to Triton 3.7.1 commit
f797708c0626e5f9840ca5b0a98790e2c7cb09ad:
https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/tutorials/06-fused-attention.py
Algorithm: https://arxiv.org/abs/2307.08691v1 (Tri Dao, FlashAttention-2).
Upstream license: LICENSE.triton in this directory (MIT).

One program owns BLOCK_M query rows and streams BLOCK_N key/value rows.
The accumulator, online maximum and normalization sum use FP32; both dot
products use Tensor Cores with FP16/BF16 operands. No quadratic score buffer,
autotuning, dropout, bias, variable lengths or performance claim.
Contiguous Q[B,H,M,D], K[B,H,N,D], V[B,H,N,DV] -> O[B,H,M,DV].
CAUSAL uses top-left alignment: key position <= query position, including M!=N.
"""
import triton
import triton.language as tl


@triton.jit
def flash_attention2(Q, K, V, O, M: tl.constexpr, N: tl.constexpr,
                     D: tl.constexpr, DV: tl.constexpr, CAUSAL: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, D)
    value_dims = tl.arange(0, DV)
    q = tl.load(Q + batch_head * M * D + rows[:, None] * D + dims[None, :],
                mask=rows[:, None] < M, other=0)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DV), tl.float32)
    stop = N
    if CAUSAL:
        stop = tl.minimum(N, (query_block + 1) * BLOCK_M)
    for start in range(0, stop, BLOCK_N):
        positions = start + cols
        k = tl.load(K + batch_head * N * D + positions[None, :] * D + dims[:, None],
                    mask=positions[None, :] < N, other=0)
        scores = tl.dot(q, k) * (D ** -0.5) * 1.4426950408889634
        allowed = positions[None, :] < N
        if CAUSAL:
            allowed = allowed & (rows[:, None] >= positions[None, :])
        scores = tl.where(allowed, scores, float('-inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        probabilities = tl.exp2(scores - new_maximum[:, None])
        correction = tl.exp2(maximum - new_maximum)
        denominator = denominator * correction + tl.sum(probabilities, 1)
        accumulator = accumulator * correction[:, None]
        v = tl.load(V + batch_head * N * DV + positions[:, None] * DV + value_dims[None, :],
                    mask=positions[:, None] < N, other=0)
        accumulator = tl.dot(probabilities.to(q.dtype), v, accumulator)
        maximum = new_maximum
    result = accumulator / denominator[:, None]
    tl.store(O + batch_head * M * DV + rows[:, None] * DV + value_dims[None, :],
             result.to(O.dtype.element_ty), mask=rows[:, None] < M)

# Backward kernels. They use the same top-left causal alignment as the forward
# kernel. Stats holds [delta, log2sumexp] per query row so that dQ, dK and dV can
# each run as one independent kernel with one tensor output.
@triton.jit
def flash_attention2_bwd_stats(Q, K, O, dO, Stats, M: tl.constexpr, N: tl.constexpr,
                               D: tl.constexpr, DV: tl.constexpr, CAUSAL: tl.constexpr,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, D)
    value_dims = tl.arange(0, DV)
    q = tl.load(Q + batch_head * M * D + rows[:, None] * D + dims[None, :],
                mask=rows[:, None] < M, other=0)
    do = tl.load(dO + batch_head * M * DV + rows[:, None] * DV + value_dims[None, :],
                 mask=rows[:, None] < M, other=0)
    o = tl.load(O + batch_head * M * DV + rows[:, None] * DV + value_dims[None, :],
                mask=rows[:, None] < M, other=0)
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
    stop = N
    if CAUSAL:
        stop = tl.minimum(N, (query_block + 1) * BLOCK_M)
    row_max = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    for start in range(0, stop, BLOCK_N):
        positions = start + cols
        k = tl.load(K + batch_head * N * D + positions[None, :] * D + dims[:, None],
                    mask=positions[None, :] < N, other=0)
        scores = tl.dot(q, k) * (D ** -0.5) * 1.4426950408889634
        allowed = (positions[None, :] < N) & (rows[:, None] < M)
        if CAUSAL:
            allowed = allowed & (rows[:, None] >= positions[None, :])
        scores = tl.where(allowed, scores, float('-inf'))
        new_maximum = tl.maximum(row_max, tl.max(scores, 1))
        row_sum = row_sum * tl.exp2(row_max - new_maximum) + tl.sum(
            tl.exp2(scores - new_maximum[:, None]), 1)
        row_max = new_maximum
    lse = row_max + tl.log2(row_sum)
    base = (batch_head * M + rows) * 2
    tl.store(Stats + base, delta, mask=rows < M)
    tl.store(Stats + base + 1, lse, mask=rows < M)


@triton.jit
def flash_attention2_bwd_dq(Q, K, V, dO, Stats, dQ, M: tl.constexpr, N: tl.constexpr,
                            D: tl.constexpr, DV: tl.constexpr, CAUSAL: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, D)
    value_dims = tl.arange(0, DV)
    q = tl.load(Q + batch_head * M * D + rows[:, None] * D + dims[None, :],
                mask=rows[:, None] < M, other=0)
    do = tl.load(dO + batch_head * M * DV + rows[:, None] * DV + value_dims[None, :],
                 mask=rows[:, None] < M, other=0)
    base = (batch_head * M + rows) * 2
    delta = tl.load(Stats + base, mask=rows < M, other=0)
    lse = tl.load(Stats + base + 1, mask=rows < M, other=0)
    dq = tl.zeros((BLOCK_M, D), tl.float32)
    stop = N
    if CAUSAL:
        stop = tl.minimum(N, (query_block + 1) * BLOCK_M)
    for start in range(0, stop, BLOCK_N):
        positions = start + cols
        k = tl.load(K + batch_head * N * D + positions[:, None] * D + dims[None, :],
                    mask=positions[:, None] < N, other=0)
        v = tl.load(V + batch_head * N * DV + positions[:, None] * DV + value_dims[None, :],
                    mask=positions[:, None] < N, other=0)
        scores = tl.dot(q, tl.trans(k)) * (D ** -0.5) * 1.4426950408889634
        allowed = (positions[None, :] < N) & (rows[:, None] < M)
        if CAUSAL:
            allowed = allowed & (rows[:, None] >= positions[None, :])
        probabilities = tl.where(allowed, tl.exp2(scores - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (probabilities * (dp - delta[:, None]) * 1.4426950408889634).to(q.dtype)
        dq += tl.dot(ds, k)
    dq *= D ** -0.5
    tl.store(dQ + batch_head * M * D + rows[:, None] * D + dims[None, :],
             dq.to(dQ.dtype.element_ty), mask=rows[:, None] < M)


@triton.jit
def flash_attention2_bwd_dk(Q, K, V, dO, Stats, dK, M: tl.constexpr, N: tl.constexpr,
                            D: tl.constexpr, DV: tl.constexpr, CAUSAL: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    positions = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, D)
    value_dims = tl.arange(0, DV)
    k = tl.load(K + batch_head * N * D + positions[:, None] * D + dims[None, :],
                mask=positions[:, None] < N, other=0)
    v = tl.load(V + batch_head * N * DV + positions[:, None] * DV + value_dims[None, :],
                mask=positions[:, None] < N, other=0)
    dk = tl.zeros((BLOCK_N, D), tl.float32)
    for start in range(0, M, BLOCK_M):
        query_rows = start + rows
        q = tl.load(Q + batch_head * M * D + query_rows[:, None] * D + dims[None, :],
                    mask=query_rows[:, None] < M, other=0)
        do = tl.load(dO + batch_head * M * DV + query_rows[:, None] * DV + value_dims[None, :],
                     mask=query_rows[:, None] < M, other=0)
        base = (batch_head * M + query_rows) * 2
        delta = tl.load(Stats + base, mask=query_rows < M, other=0)
        lse = tl.load(Stats + base + 1, mask=query_rows < M, other=0)
        scores = tl.dot(q, tl.trans(k)) * (D ** -0.5) * 1.4426950408889634
        allowed = (positions[None, :] < N) & (query_rows[:, None] < M)
        if CAUSAL:
            allowed = allowed & (query_rows[:, None] >= positions[None, :])
        probabilities = tl.where(allowed, tl.exp2(scores - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (probabilities * (dp - delta[:, None]) * 1.4426950408889634).to(k.dtype)
        dk += tl.dot(tl.trans(ds), q)
    dk *= D ** -0.5
    tl.store(dK + batch_head * N * D + positions[:, None] * D + dims[None, :],
             dk.to(dK.dtype.element_ty), mask=positions[:, None] < N)


@triton.jit
def flash_attention2_bwd_dv(Q, K, V, dO, Stats, dV, M: tl.constexpr, N: tl.constexpr,
                            D: tl.constexpr, DV: tl.constexpr, CAUSAL: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    positions = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, D)
    value_dims = tl.arange(0, DV)
    k = tl.load(K + batch_head * N * D + positions[:, None] * D + dims[None, :],
                mask=positions[:, None] < N, other=0)
    v = tl.load(V + batch_head * N * DV + positions[:, None] * DV + value_dims[None, :],
                mask=positions[:, None] < N, other=0)
    dv = tl.zeros((BLOCK_N, DV), tl.float32)
    for start in range(0, M, BLOCK_M):
        query_rows = start + rows
        q = tl.load(Q + batch_head * M * D + query_rows[:, None] * D + dims[None, :],
                    mask=query_rows[:, None] < M, other=0)
        do = tl.load(dO + batch_head * M * DV + query_rows[:, None] * DV + value_dims[None, :],
                     mask=query_rows[:, None] < M, other=0)
        base = (batch_head * M + query_rows) * 2
        lse = tl.load(Stats + base + 1, mask=query_rows < M, other=0)
        scores = tl.dot(q, tl.trans(k)) * (D ** -0.5) * 1.4426950408889634
        allowed = (positions[None, :] < N) & (query_rows[:, None] < M)
        if CAUSAL:
            allowed = allowed & (query_rows[:, None] >= positions[None, :])
        probabilities = tl.where(allowed, tl.exp2(scores - lse[:, None]), 0.0)
        dv += tl.dot(tl.trans(probabilities).to(k.dtype), do)
    # Keep the value pointer in the exported ABI and dtype without changing dV.
    dv = tl.where(positions[:, None] < N, dv, v.to(tl.float32))
    tl.store(dV + batch_head * N * DV + positions[:, None] * DV + value_dims[None, :],
             dv.to(V.dtype.element_ty), mask=positions[:, None] < N)

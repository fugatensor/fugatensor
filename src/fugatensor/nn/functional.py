"""Neural network operations executed by ASK and cuDNN."""
from __future__ import annotations

import math

from .. import ops as _ops
from ..autograd import stop_gradient
from ..creation import arange
from ..dtypes import fp16, bf16, fp32
from ..tensor import Tensor
from ..tracing import TraceTensor
from ._attention import _dimensions, _varlen_dimensions, attention, attention_varlen

__all__ = ['scaled_dot_product_attention', 'scaled_dot_product_attention_varlen',
           'relu', 'leaky_relu', 'sigmoid', 'silu', 'gelu',
           'softmax', 'log_softmax', 'layer_norm', 'rms_norm',
           'embedding', 'cross_entropy']


def scaled_dot_product_attention(query, key, value, *, is_causal=False, scale=None):
    """Compute softmax(query @ key.T / sqrt(D)) @ value with cuDNN 9 SDPA.

    Inputs are contiguous CUDA tensors shaped [batch, heads, sequence, feature]
    with matching fp16 or bf16 dtype. Query/key and value feature widths may
    differ; both must be 32, 64 or 128. All dimensions must be positive.
    CUDA capability 8.0 or newer is required.
    The result has shape [batch, heads, query_sequence, value_feature].

    ``is_causal=True`` enforces lower-triangular causal masking.
    Supports composition and first-order VJP.
    """
    if type(is_causal) is not bool:
        raise TypeError('is_causal must be a bool')
    inputs = (query, key, value)
    if any(not isinstance(item, (Tensor, TraceTensor)) for item in inputs):
        raise TypeError('scaled_dot_product_attention requires three Tensor inputs')
    if any(item.device != query.device for item in inputs[1:]):
        raise RuntimeError('expected all attention tensors on the same device')
    _dimensions(inputs)
    if query.device.type != 'cuda':
        raise ValueError('scaled_dot_product_attention requires CUDA tensors')
    engine = query.engine if isinstance(query, Tensor) else query.context.engine
    if isinstance(query, Tensor):
        for item in inputs:
            engine._check_tensor(item)
    if engine.capability[0] < 8:
        raise ValueError('scaled_dot_product_attention requires CUDA capability 8.0 or newer')
    return attention(is_causal, scale=scale)(query, key, value)


def scaled_dot_product_attention_varlen(query, key, value, cu_seqlens_q, cu_seqlens_kv,
                                        max_seqlen_q, max_seqlen_kv, *, is_causal=False, scale=None):
    """Compute variable-sequence-length SDPA with cuDNN 9.

    Packed inputs have shape [total_tokens, heads, feature] and matching fp16/bf16 dtype.
    ``cu_seqlens_q`` and ``cu_seqlens_kv`` are 1D int32 tensors shaped [batch + 1].
    ``max_seqlen_q`` and ``max_seqlen_kv`` are maximum sequence length bounds.
    """
    if type(is_causal) is not bool:
        raise TypeError('is_causal must be a bool')
    if type(max_seqlen_q) is not int or max_seqlen_q <= 0:
        raise ValueError('max_seqlen_q must be a positive integer')
    if type(max_seqlen_kv) is not int or max_seqlen_kv <= 0:
        raise ValueError('max_seqlen_kv must be a positive integer')
    inputs = (query, key, value, cu_seqlens_q, cu_seqlens_kv)
    if any(not isinstance(item, (Tensor, TraceTensor)) for item in inputs):
        raise TypeError('scaled_dot_product_attention_varlen requires Tensor inputs')
    if any(item.device != query.device for item in inputs[1:]):
        raise RuntimeError('expected all attention tensors on the same device')
    _varlen_dimensions(inputs)
    if query.device.type != 'cuda':
        raise ValueError('scaled_dot_product_attention_varlen requires CUDA tensors')
    engine = query.engine if isinstance(query, Tensor) else query.context.engine
    if isinstance(query, Tensor):
        for item in inputs:
            engine._check_tensor(item)
    if engine.capability[0] < 8:
        raise ValueError('scaled_dot_product_attention_varlen requires CUDA capability 8.0 or newer')
    return attention_varlen(is_causal, scale=scale, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv)(
        query, key, value, cu_seqlens_q, cu_seqlens_kv)


def _norm_axes(x, normalized_shape, axis):
    rank = len(x.shape)
    if normalized_shape is not None and axis is not None:
        raise ValueError('specify either normalized_shape or axis, not both')
    if axis is not None:
        axes = _ops._axes(axis, rank)
        norm_shape = tuple(x.shape[a] for a in axes)
        return axes, norm_shape
    if normalized_shape is None:
        if rank == 0:
            raise ValueError('normalization requires at least 1D input when normalized_shape is omitted')
        norm_shape = (x.shape[-1],)
    elif isinstance(normalized_shape, int):
        norm_shape = (normalized_shape,)
    else:
        norm_shape = tuple(normalized_shape)
    if not norm_shape or len(norm_shape) > rank or x.shape[-len(norm_shape):] != norm_shape:
        raise ValueError(f'normalized_shape {norm_shape} does not match trailing shape {x.shape}')
    axes = tuple(range(rank - len(norm_shape), rank))
    return axes, norm_shape


def _apply_affine(x, normalized_shape, weight=None, bias=None):
    target_shape = x.shape
    if weight is not None:
        w = weight.astype(x.dtype) if weight.dtype != x.dtype else weight
        x = x * _ops.broadcast_to(w, target_shape)
    if bias is not None:
        b = bias.astype(x.dtype) if bias.dtype != x.dtype else bias
        x = x + _ops.broadcast_to(b, target_shape)
    return x


def relu(x):
    """Apply elementwise rectified linear unit max(x, 0)."""
    return _ops.maximum(x, 0)


def leaky_relu(x, negative_slope=0.01):
    """Apply elementwise leaky ReLU."""
    return _ops.where(x > 0, x, x * negative_slope)


def sigmoid(x):
    """Apply overflow-safe logistic sigmoid 1 / (1 + exp(-x))."""
    pos = x >= 0
    neg_x = _ops.where(pos, -x, x)
    z = _ops.exp(neg_x)
    denom = z + 1
    return _ops.where(pos, 1 / denom, z / denom)


def silu(x):
    """Apply elementwise SiLU / Swish activation x * sigmoid(x)."""
    return x * sigmoid(x)


def gelu(x, *, approximate='none'):
    """Apply Gaussian Error Linear Unit using exact erf or tanh approximation."""
    if approximate == 'none':
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return 0.5 * x * (1 + _ops.erf(x * inv_sqrt2))
    if approximate == 'tanh':
        sqrt_2_pi = math.sqrt(2.0 / math.pi)
        inner = sqrt_2_pi * (x + 0.044715 * (x * x * x))
        return 0.5 * x * (1 + _ops.tanh(inner))
    raise ValueError("gelu approximate must be 'none' or 'tanh'")


def softmax(x, axis=-1):
    """Compute numerically stable softmax along ``axis`` (upcasting f16/bf16 statistics to fp32)."""
    orig_dtype = x.dtype
    work = x.astype(fp32) if orig_dtype in (fp16, bf16) else x
    m = stop_gradient(_ops.amax(work, axis=axis, keepdims=True))
    shifted = work - _ops.broadcast_to(m, work.shape)
    ex = _ops.exp(shifted)
    denom = _ops.broadcast_to(_ops.sum(ex, axis=axis, keepdims=True), work.shape)
    out = ex / denom
    return out.astype(orig_dtype) if work is not x else out


def log_softmax(x, axis=-1):
    """Compute numerically stable log-softmax along ``axis``."""
    orig_dtype = x.dtype
    work = x.astype(fp32) if orig_dtype in (fp16, bf16) else x
    m = stop_gradient(_ops.amax(work, axis=axis, keepdims=True))
    shifted = work - _ops.broadcast_to(m, work.shape)
    lse = _ops.log(_ops.sum(_ops.exp(shifted), axis=axis, keepdims=True))
    out = shifted - _ops.broadcast_to(lse, work.shape)
    return out.astype(orig_dtype) if work is not x else out


def layer_norm(x, normalized_shape=None, weight=None, bias=None, *, eps=1e-5, axis=None):
    """Apply Layer Normalization over trailing ``normalized_shape`` dimensions or ``axis``."""
    axes, norm_shape = _norm_axes(x, normalized_shape, axis)
    orig_dtype = x.dtype
    work = x.astype(fp32) if orig_dtype in (fp16, bf16) else x
    mean = _ops.broadcast_to(_ops.mean(work, axis=axes, keepdims=True), work.shape)
    diff = work - mean
    var = _ops.mean(diff * diff, axis=axes, keepdims=True)
    inv_std = _ops.broadcast_to(1 / _ops.sqrt(var + eps), work.shape)
    normed = diff * inv_std
    out = normed.astype(orig_dtype) if work is not x else normed
    return _apply_affine(out, norm_shape, weight=weight, bias=bias)


def rms_norm(x, normalized_shape=None, weight=None, *, eps=1e-6, axis=None):
    """Apply Root Mean Square Normalization over trailing ``normalized_shape`` dimensions or ``axis``."""
    axes, norm_shape = _norm_axes(x, normalized_shape, axis)
    orig_dtype = x.dtype
    work = x.astype(fp32) if orig_dtype in (fp16, bf16) else x
    ms = _ops.mean(work * work, axis=axes, keepdims=True)
    inv_rms = _ops.broadcast_to(1 / _ops.sqrt(ms + eps), work.shape)
    out = (work * inv_rms)
    out = out.astype(orig_dtype) if work is not x else out
    return _apply_affine(out, norm_shape, weight=weight, bias=None)


def embedding(input, weight):
    """Look up rows of ``weight`` indexed by integer tensor ``input``."""
    return _ops.gather(weight, input, axis=0)


def cross_entropy(logits, targets, *, axis=-1, reduction='mean'):
    """Compute cross-entropy loss from unnormalized ``logits`` and integer ``targets``."""
    if reduction not in ('mean', 'sum', 'none'):
        raise ValueError("cross_entropy reduction must be 'mean', 'sum' or 'none'")
    rank = len(logits.shape)
    if rank == 0:
        raise ValueError('cross_entropy logits must have at least 1 dimension')
    ax = axis + rank if axis < 0 else axis
    if not 0 <= ax < rank:
        raise ValueError('cross_entropy axis out of range')
    lsm = log_softmax(logits, axis=ax)
    if ax != rank - 1:
        perm = tuple(i for i in range(rank) if i != ax) + (ax,)
        lsm = _ops.transpose(lsm, perm)
    batch_shape = lsm.shape[:-1]
    if targets.shape != batch_shape:
        raise ValueError(f'cross_entropy targets shape {targets.shape} must match logits batch shape {batch_shape}')
    classes = lsm.shape[-1]
    n = math.prod(batch_shape)
    if n == 0:
        if reduction == 'none':
            return _ops.zeros_like(targets.astype(logits.dtype))
        return lsm.reshape((0,)).sum() if reduction == 'sum' else lsm.reshape((0,)).mean()
    flat_lsm = lsm.reshape((-1,))
    flat_targets = targets.reshape((n,))
    row_offsets = arange(n, dtype=targets.dtype, device=logits.device) * classes
    valid = (flat_targets >= 0) & (flat_targets < classes)
    safe_idx = _ops.where(valid, row_offsets + flat_targets, -1)
    nll = -_ops.gather(flat_lsm, safe_idx, axis=0)
    if batch_shape:
        nll = nll.reshape(batch_shape)
    else:
        nll = nll.reshape(())
    if reduction == 'none':
        return nll
    if reduction == 'sum':
        return nll.sum()
    return nll.mean()

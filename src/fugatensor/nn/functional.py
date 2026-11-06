"""Neural network operations executed by ASK and cuDNN."""
from __future__ import annotations

from ..tensor import Tensor
from ..tracing import TraceTensor
from ._attention import _dimensions, _varlen_dimensions, attention, attention_varlen

__all__ = ['scaled_dot_product_attention', 'scaled_dot_product_attention_varlen']


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

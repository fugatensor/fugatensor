"""Neural network operations executed by ASK."""
from __future__ import annotations

from ..tensor import Tensor
from ..tracing import TraceTensor
from ._attention import _dimensions, attention

__all__ = ['scaled_dot_product_attention']


def scaled_dot_product_attention(query, key, value, *, is_causal=False):
    """Compute softmax(query @ key.T / sqrt(D)) @ value with FlashAttention.

    Inputs are contiguous CUDA tensors shaped [batch, heads, sequence, feature]
    with matching fp16 or bf16 dtype. Query/key and value feature widths may
    differ; both must be 32, 64 or 128. All dimensions must be positive.
    CUDA capability 8.0 or newer and the configured Triton worker are required.
    The result has shape [batch, heads, query_sequence, value_feature].

    ``is_causal=True`` allows key_index <= query_index (top-left alignment).
    Supports ordinary composition and first-order backward. Masks, dropout,
    grouped-query attention, custom scaling and higher derivatives are not
    supported by this interface.
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
    return attention(is_causal)(query, key, value)

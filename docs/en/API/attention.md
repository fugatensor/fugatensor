Back to API Index: [Categories](README.md#categories) · Related Modules: [Basic Operations](ops.md) · [Autograd](autograd.md)

# Attention Operators

<a id="sdpa"></a>

## Scaled Dot-Product Attention

### `ft.nn.functional.scaled_dot_product_attention(query, key, value, *, is_causal=False, scale=None)`
Executes forward and first-order backward passes using native cuDNN 9 SDPA graphs:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **Requirements and Constraints**:
  - **Device**: CUDA only (Compute Capability >= 8.0);
  - **Data Type**: `ft.fp16` and `ft.bf16` ($Q, K, V$ dtypes must match);
  - **Input Shape**: 4D tensor `(batch, heads, seq_len, dim)`, where head counts for $Q, K, V$ match, and head dimensions for $Q$ and $K$ match;
  - **Head Dimension Limit**: Supported head dimensions are 32, 64, and 128.
- **Parameters**:
  - `query`: Query tensor $Q$;
  - `key`: Key tensor $K$;
  - `value`: Value tensor $V$;
  - `is_causal`: If `True`, applies a lower-triangular causal mask;
  - `scale`: Optional floating-point scale factor; defaults to $1 / \sqrt{d_k}$.
- **Returns**: Unevaluated tensor of shape `(batch, heads, seq_len_q, dim_v)`.
- **Autograd**: Supports first-order backward derivatives ($dQ, dK, dV$); active higher-order derivatives are rejected statically.

```python
import fugatensor as ft

q = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")
k = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")
v = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")

out = ft.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
ft.evaluate(out)
print(out.shape)  # (2, 4, 128, 64)

# First-order gradient computation
seed = ft.ones_like(out)
dq, dk, dv = ft.grad(out, (q, k, v), seed=seed)
ft.evaluate(dq, dk, dv)
print(dq.shape)  # (2, 4, 128, 64)
```

<a id="sdpa-varlen"></a>

## Variable-Sequence-Length SDPA

### `ft.nn.functional.scaled_dot_product_attention_varlen(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *, is_causal=False, scale=None)`
Computes packed batch (Packed THD) attention across variable-length sequences, avoiding computation overhead from padding invalid tokens.

- **Requirements and Constraints**:
  - **Device**: CUDA only (Compute Capability >= 8.0);
  - **Data Type**: `ft.fp16` and `ft.bf16`;
  - **Input Shape**: 3D packed tensor `(total_tokens, heads, dim)`;
  - **Sequence Boundaries**: `cu_seqlens_q` and `cu_seqlens_kv` are 1D `ft.int32` tensors of shape `(batch + 1,)`, starting with 0 and ending with `total_tokens`;
  - `max_seqlen_q` and `max_seqlen_kv` are positive Python integers representing the maximum upper bound of individual sequence lengths.
- **Parameters**:
  - `query`: Packed query tensor $Q$ of shape `(total_tokens_q, heads, dim)`;
  - `key`: Packed key tensor $K$ of shape `(total_tokens_kv, heads, dim)`;
  - `value`: Packed value tensor $V$ of shape `(total_tokens_kv, heads, dim_v)`;
  - `cu_seqlens_q`: Cumulative sequence length prefix-sum tensor for queries;
  - `cu_seqlens_kv`: Cumulative sequence length prefix-sum tensor for keys/values;
  - `max_seqlen_q`: Maximum sequence length for queries;
  - `max_seqlen_kv`: Maximum sequence length for keys/values;
  - `is_causal`: If `True`, applies causal masking independently to each sequence;
  - `scale`: Optional floating-point scale factor; defaults to $1 / \sqrt{d_k}$.
- **Returns**: Unevaluated tensor of shape `(total_tokens_q, heads, dim_v)`.
- **Autograd**: Supports first-order backward derivative computation, returning a gradient tuple `(dq, dk, dv)` for floating-point inputs.

```python
import fugatensor as ft

# Batch contains 2 sequences with lengths 64 and 128; total token count is 192
seqlens = [64, 128]
cu_seqlens = ft.tensor([0, 64, 192], dtype=ft.int32, device="cuda")

q = ft.ones(192, 4, 64, dtype=ft.fp16, device="cuda")
k = ft.ones(192, 4, 64, dtype=ft.fp16, device="cuda")
v = ft.ones(192, 4, 64, dtype=ft.fp16, device="cuda")

out = ft.nn.functional.scaled_dot_product_attention_varlen(
    q, k, v, cu_seqlens, cu_seqlens, max_seqlen_q=128, max_seqlen_kv=128, is_causal=True
)
ft.evaluate(out)
print(out.shape)  # (192, 4, 64)

# First-order backward gradients
seed = ft.ones_like(out)
dq, dk, dv = ft.grad(out, (q, k, v), seed=seed)
ft.evaluate(dq, dk, dv)
print(dq.shape)  # (192, 4, 64)
```

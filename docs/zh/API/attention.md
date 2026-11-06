返回 API 目录：[分类目录](README.md#categories) · 关联模块：[基础运算](ops.md) · [自动微分](autograd.md)

# 注意力算子

<a id="sdpa"></a>

## 缩放点积注意力 (Scaled Dot-Product Attention)

### `ft.nn.functional.scaled_dot_product_attention(query, key, value, *, is_causal=False, scale=None)`
基于 cuDNN 9 SDPA 原生图执行前向与一阶反向：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **约束与必要条件**：
  - **设备**：仅支持 CUDA（Compute Capability >= 8.0）；
  - **精度**：`ft.fp16` 与 `ft.bf16`（$Q, K, V$ 精度须一致）；
  - **输入形状**：4D 张量 `(batch, heads, seq_len, dim)`，其中 $Q, K, V$ 头数一致，$Q$ 与 $K$ 特征维度一致；
  - **头维度限制**：特征维度支持 32、64、128。
- **参数**：
  - `query`：查询张量 $Q$；
  - `key`：键张量 $K$；
  - `value`：值张量 $V$；
  - `is_causal`：若为 `True`，应用下三角因果掩码；
  - `scale`：可选浮点缩放因子，缺省为 $1 / \sqrt{d_k}$。
- **返回值**：形状为 `(batch, heads, seq_len_q, dim_v)` 的待计算张量。
- **自动微分**：支持一阶反向导数（$dQ, dK, dV$），活跃高阶导数将被静态拒绝。

```python
import fugatensor as ft

q = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")
k = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")
v = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")

out = ft.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
ft.evaluate(out)
print(out.shape)  # (2, 4, 128, 64)

# 一阶梯度计算
seed = ft.ones_like(out)
dq, dk, dv = ft.grad(out, (q, k, v), seed=seed)
ft.evaluate(dq, dk, dv)
print(dq.shape)  # (2, 4, 128, 64)
```

<a id="sdpa-varlen"></a>

## 变长注意力 (Variable-Sequence-Length SDPA)

### `ft.nn.functional.scaled_dot_product_attention_varlen(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *, is_causal=False, scale=None)`
计算变长序列组成的紧凑批次（Packed THD）注意力，避免填充无效 token 的计算开销。

- **约束与必要条件**：
  - **设备**：仅支持 CUDA（Compute Capability >= 8.0）；
  - **精度**：`ft.fp16` 与 `ft.bf16`；
  - **输入形状**：3D 紧凑张量 `(total_tokens, heads, dim)`；
  - **序列界限**：`cu_seqlens_q` 与 `cu_seqlens_kv` 为 1D `ft.int32` 张量，形状为 `(batch + 1,)`，首项为 0，末项为 `total_tokens`；
  - `max_seqlen_q` 与 `max_seqlen_kv` 为 Python 正整数，表示各序列长度的最大上界。
- **参数**：
  - `query`：紧凑查询张量 $Q$，形状 `(total_tokens_q, heads, dim)`；
  - `key`：紧凑键张量 $K$，形状 `(total_tokens_kv, heads, dim)`；
  - `value`：紧凑值张量 $V$，形状 `(total_tokens_kv, heads, dim_v)`；
  - `cu_seqlens_q`：累积序列长度前缀和张量；
  - `cu_seqlens_kv`：键/值累积序列长度前缀和张量；
  - `max_seqlen_q`：查询序列最大长度；
  - `max_seqlen_kv`：键/值序列最大长度；
  - `is_causal`：若为 `True`，对各序列独立应用因果掩码；
  - `scale`：可选浮点缩放因子，缺省为 $1 / \sqrt{d_k}$。
- **返回值**：形状为 `(total_tokens_q, heads, dim_v)` 的待计算张量。
- **自动微分**：支持一阶反向导数计算，返回浮点输入 `(dq, dk, dv)` 的梯度元组。

```python
import fugatensor as ft

# 批次内包含 2 条序列，长度分别为 64 与 128，总 token 数为 192
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

# 一阶反向梯度
seed = ft.ones_like(out)
dq, dk, dv = ft.grad(out, (q, k, v), seed=seed)
ft.evaluate(dq, dk, dv)
print(dq.shape)  # (192, 4, 64)
```

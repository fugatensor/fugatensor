返回 API 目录：[分类目录](README.md#categories) · 关联模块：[基础运算](ops.md) · [自动微分](autograd.md)

# 注意力算子

<a id="sdpa"></a>

## 缩放点积注意力 (Scaled Dot-Product Attention)

### `ft.nn.functional.scaled_dot_product_attention(query, key, value, *, is_causal=False, scale=None)`
高效计算多头或批量注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **约束与支持条件**：
  - **设备**：仅支持 CUDA；
  - **精度**：`ft.fp16` 与 `ft.bf16`；
  - **输入形状**：4D 张量 `(batch, heads, seq_len, dim)`，其中 $Q, K, V$ 的头数和特征维度须匹配；
  - **头维度限制**：`dim <= 128`，要求能被 16 整除。
- **参数**：
  - `query`：查询张量 $Q$；
  - `key`：键张量 $K$；
  - `value`：值张量 $V$；
  - `is_causal`：若为 `True`，应用下三角因果掩码（防止位置看见未来 token）；
  - `scale`：缩放因子，缺省为 $1 / \sqrt{d_k}$。
- **自动微分**：支持完整的一阶反向导数计算（自动推导 $dQ, dK, dV$）。

```python
import fugatensor as ft

# 仅在具备 CUDA 环境时执行
# q = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")
# k = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")
# v = ft.ones(2, 4, 128, 64, dtype=ft.fp16, device="cuda")

# out = ft.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
# print("Output shape:", out.shape)  # (2, 4, 128, 64)

# 一阶梯度计算
# dq, dk, dv = ft.grad(out.sum(), (q, k, v))
# print("dQ shape:", dq.shape)       # (2, 4, 128, 64)
```

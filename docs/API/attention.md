返回 API：[分类目录](README.md) · 相关 API：[张量与数据交换](张量与数据交换.md) · [自动微分](自动微分与自定义函数.md) · 环境准备：[Triton 配置](../安装.md#configure-triton)

# attention

`ft.nn.functional.scaled_dot_product_attention` 计算缩放点积注意力。这是fugatensor内置的一个基于Triton的kernel示例
<a id="scaled-dot-product-attention"></a>

## 1. 参数与结果

```text
ft.nn.functional.scaled_dot_product_attention(query, key, value, *, is_causal=False)
```

| 参数 | 约束 |
| --- | --- |
| `query` | 形状 `[B, H, M, D]` 的 Tensor |
| `key` | 形状 `[B, H, N, D]` 的 Tensor |
| `value` | 形状 `[B, H, N, DV]` 的 Tensor |
| `is_causal=False` | Python bool；为 `True` 时，查询行 `i` 只读取键列 `j <= i` |

三个输入必须是同一设备、同一引擎上的连续 CUDA 张量，GPU 要求 SM80 或更新架构，类型相同且为 `ft.fp16` 或 `ft.bf16`。各维须大于零，`D`、`DV` 各自取 32、64 或 128；`M` 与 `N` 可以不同，也不必是分块大小的整数倍。

返回形状 `[B, H, M, DV]` 的待计算 Tensor，类型和设备与输入一致。计算式为 `softmax(Q @ Kᵀ / sqrt(D)) @ V`；因果掩码在非方阵上按左上角对齐。反向传播计算 Q、K、V 的一阶梯度。

<a id="example"></a>

## 2. 前向与反向

先完成 [CUDA 与 Triton 环境配置](../安装.md#configure-triton)，再直接调用：

```python
import fugatensor as ft

q = ft.ones(1, 2, 16, 32, dtype=ft.fp16, device="cuda")
k = ft.ones(1, 2, 24, 32, dtype=ft.fp16, device="cuda")
v = ft.ones(1, 2, 24, 64, dtype=ft.fp16, device="cuda")

output = ft.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
dq, dk, dv = ft.grad(output, (q, k, v), seed=ft.ones_like(output))
ft.evaluate(output, dq, dk, dv)
print(output.shape)  # (1, 2, 16, 64)
print(dq.shape, dk.shape, dv.shape)
# (1, 2, 16, 32) (1, 2, 24, 32) (1, 2, 24, 64)
```

调用 `ft.grad(output, (q, k, v), seed=...)` 时传入同形状、同类型的余切种子，返回三个待计算的梯度 Tensor。前后运算可写为 `scaled_dot_product_attention(q * 0.5, k, v) * 2`，一起参与链式求导。

<a id="support-boundaries"></a>

## 3. 当前边界

使用前安装 `fugatensor[all]` 或配置独立 Triton worker。输入须满足上文的 CUDA、形状和类型要求，创建时选择 `dtype=ft.fp16` 或 `dtype=ft.bf16`。

当前没有自定义 scale、dropout、加性 bias、外部 attention mask、分组查询注意力、变长序列或非连续 stride 参数。手写 VJP 支持一阶微分；继续对梯度调用 `ft.grad` 时，活跃的反向外部 kernel 缺少微分规则，会被拒绝。

实现与执行计划见[ASK 桥接与多后端执行](../architecture/ASK桥接与多后端执行.md#cuda-and-triton)；精度、因果模式和形状的数值验证见[测试组织](../测试.md#attention-matrix)。

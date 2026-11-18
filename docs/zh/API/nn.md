返回分类目录：[分类目录](README.md#categories) · 关联模块：[基础算子](ops.md) · [注意力算子](attention.md) · [自动微分](autograd.md)

# 神经网络函数式算子（`ft.nn.functional`）

`ft.nn.functional` 基于 Fuga IR 原语直接实现常用激活函数、归一化、词嵌入查找和损失函数，与计算图融合、CPU/CUDA 编译及 `ft.grad` 高阶自动微分无缝组合，不依赖任何 Python 数值计算回退。cuDNN FlashAttention 见[注意力算子](attention.md)。

<a id="activations"></a>

## 激活函数

| 函数 | 公式 / 行为 | 说明 |
| --- | --- | --- |
| `ft.nn.functional.relu(x)` | $\max(x, 0)$ | 支持整数与浮点张量 |
| `ft.nn.functional.leaky_relu(x, negative_slope=0.01)` | `x > 0` 时取 `x`，否则 `negative_slope * x` | 支持浮点张量 |
| `ft.nn.functional.sigmoid(x)` | $1 / (1 + e^{-x})$ | 采用符号分段实现避免大数值指数溢出 |
| `ft.nn.functional.silu(x)` | $x \cdot \sigma(x)$ | Swish / SiLU 激活 |
| `ft.nn.functional.gelu(x, *, approximate='none')` | `'none'`: $\frac{1}{2} x (1 + \text{erf}(x / \sqrt{2}))$；`'tanh'`: tanh 近似 | 支持 `ft.grad` 求导 |

```python
import fugatensor as ft
import fugatensor.nn.functional as F

x = ft.tensor([-1.0, 0.0, 2.0])
print(F.relu(x).tolist())     # [0.0, 0.0, 2.0]
print(F.sigmoid(x).tolist())  # [0.2689..., 0.5, 0.8807...]
```

<a id="softmax-and-norms"></a>

## Softmax 与归一化

- `ft.nn.functional.softmax(x, axis=-1)` / `ft.nn.functional.log_softmax(x, axis=-1)`：
  沿 `axis` 执行减最大值的数值稳定 softmax 与 log-softmax。`ft.fp16` / `ft.bf16` 在 `ft.fp32` 累加统计量后再转回原精度；最大值平移动作通过 `stop_gradient` 断开微分边以保持一阶/高阶导数稳定性。
- `ft.nn.functional.layer_norm(x, normalized_shape=None, weight=None, bias=None, *, eps=1e-5, axis=None)`：
  对末尾 `normalized_shape` 维度（默认最后一维）或显式 `axis` 做层归一化，低精度输入使用 `ft.fp32` 累加并支持可选仿射 `weight`/`bias`。
- `ft.nn.functional.rms_norm(x, normalized_shape=None, weight=None, *, eps=1e-6, axis=None)`：
  在 `normalized_shape` 或 `axis` 上执行均方根归一化，可选应用 `weight` 缩放。

```python
import fugatensor as ft
import fugatensor.nn.functional as F

x = ft.tensor([[1.0, 2.0, 3.0]])
w = ft.ones(3)
print(F.softmax(x, axis=-1).tolist())
print(F.rms_norm(x, weight=w).tolist())
```

<a id="embedding-and-loss"></a>

## 词嵌入查找与损失函数

- `ft.nn.functional.embedding(input, weight)`：
  沿 `axis=0` 按整数下标 `input` 从 `weight` 查找对应行。前向通过 `gather` 降级执行，反向梯度经 `scatter_add` 精确累加重复下标对应行。
- `ft.nn.functional.cross_entropy(logits, targets, *, axis=-1, reduction='mean')`：
  由未归一化 `logits` 与整型类别标签 `targets`（形状匹配移除 `axis` 后的 `logits` 批次维）计算负对数似然，`reduction` 可选 `'mean'`、`'sum'` 或 `'none'`。

```python
import fugatensor as ft
import fugatensor.nn.functional as F

table = ft.tensor([[0.1, 0.2], [0.3, 0.4], [-0.5, 0.6]])
tokens = ft.tensor([0, 2, 0], dtype=ft.int32)
emb = F.embedding(tokens, table)
grad_table = ft.grad(emb.sum(), table)
print(grad_table.tolist())  # [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]
```

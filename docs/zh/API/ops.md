返回 API 目录：[分类目录](README.md#categories) · 关联模块：[张量构建](tensor.md) · [自动微分](autograd.md) · [设备执行](engine.md)

# 基础算术与代数运算

FugaTensor 运算返回待计算的表达式节点。算子在 JIT 编译时自动融合为单个计算内核。

<a id="elementwise"></a>

## 逐元素算术

支持原生 Python 运算符及对应具名函数：

| 算子符号 | 对应函数 | 功能说明 |
| --- | --- | --- |
| `a + b` | `ft.add(a, b)` | 加法（支持标量与张量相加） |
| `a - b` | `ft.subtract(a, b)` | 减法 |
| `-a` | `ft.negative(a)` | 取负 |
| `a * b` | `ft.multiply(a, b)` | 逐元素相乘 |
| `a / b` | `ft.divide(a, b)` | 逐元素除法 |
| `a.square()` | `ft.square(a)` | 元素平方（`a * a`） |
| - | `ft.exp(a)` | 自然指数（$e^x$） |
| - | `ft.log(a)` | 自然对数（$\ln(x)$） |
| - | `ft.sqrt(a)` | 平方根（$\sqrt{x}$） |

```python
import fugatensor as ft

x = ft.tensor([1.0, 4.0, 9.0])
y = ft.sqrt(x) + 1.0
print(y.tolist())  # [2.0, 3.0, 4.0]
```

<a id="comparison"></a>

## 逻辑比较与条件选择

### 比较运算符
支持 `==`, `!=`, `<`, `<=`, `>`, `>=` 以及 `ft.compare(a, b, op)`，返回 `ft.bool` 张量。

```python
a = ft.tensor([1, 4, 3])
b = ft.tensor([2, 2, 3])
mask = a > b
print(mask.tolist(), mask.dtype)  # [False, True, False] bool
```

### `ft.where(condition, true_branch, false_branch)`
按布尔掩码条件逐元素选择值。当 `condition` 为真时取 `true_branch`，否则取 `false_branch`。

```python
cond = ft.tensor([True, False, True])
x = ft.tensor([10.0, 20.0, 30.0])
y = ft.tensor([-1.0, -2.0, -3.0])
out = ft.where(cond, x, y)
print(out.tolist())  # [10.0, -2.0, 30.0]
```

<a id="geometry"></a>

## 广播与几何重排

### `ft.broadcast_to(x, shape)`
将张量广播到目标形状，遵循标准 NumPy 广播兼容规则。

```python
x = ft.tensor([1.0, 2.0])  # shape (2,)
b = ft.broadcast_to(x, (2, 2))
print(b.tolist())  # [[1.0, 2.0], [1.0, 2.0]]
```

### `ft.broadcast_in_dim(x, shape, broadcast_dimensions)`
低层精确广播原语，指定输入各轴映射到输出目标维度的显式索引。

```python
x = ft.tensor([10.0, 20.0])
# 将 1D 张量的维度 0 映射到目标 (2, 3) 的第 0 轴
b = ft.broadcast_in_dim(x, (2, 3), (0,))
print(b.tolist())  # [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]
```

### `ft.transpose(x, permutation=None)` / `x.T`
转置张量维度。若不传 `permutation`，默认翻转所有维度。矩阵提供 `.T` 简写属性。

```python
m = ft.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(m.T.shape)     # (3, 2)
print(m.T.tolist())  # [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```

<a id="reductions"></a>

## 统计归约

归约算子沿指定轴折叠张量：

### `ft.sum(x, axis=None, keepdims=False, dtype=None)` / `x.sum(...)`
沿指定轴求和。`axis=None` 时对全张量元素归约。

```python
x = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.sum().item())                 # 10.0
print(x.sum(axis=0).tolist())          # [4.0, 6.0]
print(x.sum(axis=1, keepdims=True).shape) # (2, 1)
```

### `ft.mean(x, axis=None, keepdims=False, dtype=None)` / `x.mean(...)`
沿指定轴计算均值。

```python
x = ft.tensor([[2.0, 4.0], [6.0, 8.0]])
print(x.mean().item())        # 5.0
print(x.mean(axis=1).tolist()) # [3.0, 7.0]
```

### `ft.amax(x, axis=None, keepdims=False)` / `x.amax(...)`
沿指定轴计算最大值。

```python
x = ft.tensor([1.5, 9.2, 3.4])
print(x.amax().item())  # 9.2
```

<a id="matmul"></a>

## 矩阵乘法

### `a @ b` / `ft.matmul(a, b, *, transpose_a=False, transpose_b=False)`
执行二维矩阵乘法或批量矩阵乘法。

- **后端机制**：
  - CPU 走 ASK 优化的 BLAS 降级与缓存；
  - CUDA 走 cuBLASLt 高性能引擎，支持 FP32（TF32 加速）与 FP16 Tensor Core，自动复用经过调优的静态描述与 scratch 槽位。
- **参数**：
  - `transpose_a`：是否在矩阵乘前转置矩阵 A。
  - `transpose_b`：是否在矩阵乘前转置矩阵 B。

```python
import fugatensor as ft

a = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
b = ft.tensor([[5.0, 6.0], [7.0, 8.0]])
c = a @ b
print(c.tolist())  # [[19.0, 22.0], [43.0, 50.0]]
```

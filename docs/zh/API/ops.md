返回 API 目录：[分类目录](README.md#categories) · 关联模块：[张量构建](tensor.md) · [神经网络算子](nn.md) · [自动微分](autograd.md) · [设备执行](engine.md)

# 算术运算、切片索引与归约

FugaTensor 运算返回待计算的表达式节点。兼容的逐元素、几何与归约操作在 JIT 编译期间自动融合；`scatter_add` 独立划分为原子更新步骤。

<a id="elementwise"></a>

## 逐元素数学、位运算与逻辑运算

支持原生 Python 运算符及对应具名函数：

| 算子符号 | 对应函数 | 功能说明 |
| --- | --- | --- |
| `a + b`, `a - b`, `-a` | `ft.add`, `ft.subtract`, `ft.negative` | 加法、减法与取负（支持标量） |
| `a * b`, `a / b` | `ft.multiply`, `ft.divide` | 逐元素相乘与浮点真除法 |
| `a // b`, `a % b` | `ft.floordiv`, `ft.mod` | 向下取整除法与取余（整数与浮点） |
| `a ** b` | `ft.pow(a, b)` | 逐元素幂运算（`x ** y`） |
| `abs(a)`, `a.square()` | `ft.abs(a)`, `ft.square(a)` | 绝对值与平方 |
| - | `ft.maximum(a, b)`, `ft.minimum(a, b)`, `ft.clamp(x, min=None, max=None)` | 逐元素极值与区间裁剪（`ft.clip` 别名） |
| - | `ft.exp`, `ft.log`, `ft.sqrt`, `ft.sin`, `ft.cos`, `ft.tanh`, `ft.erf` | 浮点初等数学函数 |
| - | `ft.floor`, `ft.ceil`, `ft.round` | 浮点向下取整、向上取整与偶舍入 |
| `~a`, `a & b`, `a \| b`, `a ^ b` | `ft.bitwise_not`, `bitwise_and`, `bitwise_or`, `bitwise_xor` | `ft.bool`、`ft.int32`、`ft.int64` 按位逻辑 |
| `a << b`, `a >> b` | `ft.shift_left`, `ft.shift_right` | `ft.int32`、`ft.int64` 上的确定性位移 |
| - | `ft.logical_not`, `logical_and`, `logical_or`, `logical_xor` | 任意精度张量的布尔真值逻辑运算 |

```python
import fugatensor as ft

x = ft.tensor([1.0, 4.0, 9.0])
y = ft.sin(ft.sqrt(x)).clamp(min=-0.5, max=0.9)
print(y.shape)  # (3,)
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

<a id="slicing-and-indexing"></a>

## 切片、填充、拼接与索引

- **索引与切片（`x[...]`、`ft.slice`）**：支持整数（降维）、切片（含正负步长的 `start:stop:step`）、`None`（增维）、`...`（`Ellipsis`）以及至多一个整型张量下标（通过 `gather` 降级）。动态形状的布尔掩码索引不在静态图范围内。
- **填充（`ft.pad(x, pad_width, constant_value=0, *, interior=None)`）**：用标量或可微 0-D `Tensor` 填补边界及可选内部膨胀间距 `interior`。
- **组合与拆分**：`ft.concatenate(tensors, axis=0)`（`ft.cat`）、`ft.stack(tensors, axis=0)`、`ft.split(tensor, split_size_or_sections, axis=0)`、`ft.chunk(tensor, chunks, axis=0)`。
- **形状工具**：`ft.squeeze(x, axis=None)`、`ft.unsqueeze(x, axis)`、`ft.flatten(x, start_dim=0, end_dim=-1)`。
- **掩码与独热编码**：`ft.tril(x, diagonal=0)`、`ft.triu(x, diagonal=0)`、`ft.one_hot(indices, num_classes, *, dtype=ft.fp32)`。
- **聚集与散布累加**：
  - `ft.gather(params, indices, axis=0)`（`ft.take`）：沿 `axis` 按下标读取；越界下标返回零；VJP 通过 `scatter_add` 回传 `params` 梯度。
  - `ft.scatter_add(base, indices, updates, axis=0)`：以 `base` 初始化输出并沿 `axis` 原子累加 `updates`（跳过越界下标）；支持对 `base` 与 `updates` 求导。

```python
x = ft.arange(10).reshape(2, 5)
print(x[::-1, 1::2].tolist())  # [[6, 8], [1, 3]]
base = ft.zeros(3, 2)
idx = ft.tensor([2, 0, 2], dtype=ft.int32)
upd = ft.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(ft.scatter_add(base, idx, upd, axis=0).tolist())  # [[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]]
```

<a id="reductions"></a>

## 统计归约

归约算子沿指定轴折叠张量：

- **`ft.sum` / `ft.mean` / `ft.prod`**：求和、均值与乘积（支持归约前 `dtype=` 转换；空张量 `sum`/`prod` 分别返回 `0`/`1`）。
- **`ft.amax` / `ft.amin`**：非空轴上的最大值与最小值；浮点传播 `NaN`。
- **`ft.any` / `ft.all`**：沿 `axis` 的布尔归约（空域分别返回 `False` / `True`）。
- **`ft.argmax` / `ft.argmin`**：返回非空轴上极值（或浮点首个 `NaN`）首次出现位置的 `ft.int64` 索引。

```python
x = ft.tensor([[2.0, 4.0], [6.0, 8.0]])
print(x.sum(axis=0).tolist())        # [8.0, 12.0]
print(x.prod(axis=1).tolist())       # [8.0, 48.0]
print(x.argmax(axis=1).tolist())     # [1, 1]
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

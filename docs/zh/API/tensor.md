返回 API 目录：[分类目录](README.md#categories) · 关联模块：[基础运算](ops.md) · [设备与执行](engine.md) · [自动微分](autograd.md)

# 张量创建与数据交换

<a id="dtypes"></a>

## 数据类型系统

FugaTensor 提供单例类型对象，可通过 `x.dtype is ft.fp32` 进行类型检查：

| 类型对象 | 位宽与格式 | 说明 |
| --- | --- | --- |
| `ft.bool` | 8-bit 布尔值 | 条件掩码、比较结果 |
| `ft.int32` | 32-bit 有符号整数 | 通用整数计算、索引 |
| `ft.int64` | 64-bit 有符号整数 | 大规模计数、长整型 |
| `ft.fp32` | 32-bit 单精度浮点 | 框架默认精度 |
| `ft.fp16` | 16-bit 半精度浮点 | 低精度计算、模型加速 |
| `ft.bf16` | 16-bit bfloat16 浮点 | 仅支持 CUDA 设备，要求 Compute Capability >= 8.0 |

```python
import fugatensor as ft

x = ft.tensor([1, 2, 3], dtype=ft.int32)
print(x.dtype)          # int32
print(x.dtype is ft.int32)  # True
```

<a id="factories"></a><a id="tensor-create"></a><a id="size-factories"></a>

## 张量构建工厂

### `ft.tensor(data, *, dtype=ft.fp32, device=None, engine=None)`
从嵌套 Python 列表、标量或已有序列构建具名张量。

- **参数**：
  - `data`：标量（bool/int/float）或规则嵌套序列。
  - `dtype`：目标精度，默认 `ft.fp32`。
  - `device`：目标设备，支持 `"cpu"`、`"cuda"`、`"cuda:0"` 或 `ft.device` 实例。缺省时使用当前上下文默认设备。
- **返回**：`ft.Tensor` 实例。

```python
import fugatensor as ft

a = ft.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
print("Matrix:", a.tolist())  # [[1.0, 2.0], [3.0, 4.0]]
print("Shape:", a.shape)      # (2, 2)
```

### `ft.zeros(*size, dtype=ft.fp32, device=None)`
创建指定形状的全零张量。尺寸支持变长参数展开或元组输入。

```python
z = ft.zeros(2, 3, dtype=ft.fp16)
print(z.tolist())  # [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```

### `ft.ones(*size, dtype=ft.fp32, device=None)`
创建指定形状的全一张量。

```python
o = ft.ones((3, 2))
print(o.tolist())  # [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
```

### `ft.empty(*size, dtype=ft.fp32, device=None)`
仅分配底层存储内存，不保证内容初值。适合作为预分配写入缓冲区。

```python
e = ft.empty(2, 2)
print(e.shape)  # (2, 2)
```

### `ft.full(size, fill_value, *, dtype=ft.fp32, device=None)`
创建指定形状并以指定常数填充的张量。

```python
f = ft.full((2, 2), 3.14)
print(f.tolist())  # [[3.14, 3.14], [3.14, 3.14]]
```

### `ft.zeros_like(x)` / `ft.ones_like(x)` / `ft.full_like(x, fill_value, *, dtype=None, device=None)`
创建与输入张量 `x` 形状（及默认精度、设备）一致的全零、全一或指定常数填充张量。`full_like` 同样支持在 `ft.fun` 追踪体内使用。

```python
x = ft.tensor([1, 2, 3], dtype=ft.int32)
z = ft.zeros_like(x)
f = ft.full_like(x, 7)
print(z.tolist(), f.tolist())  # [0, 0, 0] [7, 7, 7]
```

<a id="range-factories"></a>

## 坐标、序列与对角工厂

零输入创建算子通过 ASK `iota` 图直接在目标设备生成数据，同样可在自定义追踪函数内组合使用，无需先回传主机：

| 函数 | 说明 |
| --- | --- |
| `ft.arange(start, stop=None, step=1, *, dtype=None, device=None)` | 一维半开等差序列；起止与步长全为整数时默认 `ft.int64`，否则默认 `ft.fp32`。 |
| `ft.linspace(start, stop, steps, *, dtype=ft.fp32, device=None)` | 在 `[start, stop]` 上均匀采样 `steps` 个元素；`steps > 1` 时严格锚定端点。 |
| `ft.eye(n, m=None, *, dtype=ft.fp32, device=None)` | 主对角线为 `1`、其余元素为 `0` 的 `(n, m)` 二维矩阵。 |
| `ft.iota(shape, *, dimension=0, dtype=ft.fp32, device=None)` | 按指定 `shape` 生成沿 `dimension` 维递增坐标 `0, 1, ...` 的数值张量。 |

```python
import fugatensor as ft

print(ft.arange(0, 8, 2).tolist())              # [0, 2, 4, 6]
print(ft.linspace(-1.0, 1.0, 3).tolist())       # [-1.0, 0.0, 1.0]
print(ft.eye(2, 3, dtype=ft.int32).tolist())    # [[1, 0, 0], [0, 1, 0]]
print(ft.iota((2, 3), dimension=1).tolist())    # [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
```

<a id="interop"></a>

## 内存互操作与缓冲交换

### `ft.from_buffer(data, *, dtype, shape, copy=True, device=None, engine=None)`
从支持 Python Buffer 协议的对象（如 `bytes`、`bytearray`、`memoryview`）直接构建张量。

```python
import array
raw = array.array('f', [1.0, 2.0, 3.0, 4.0])
t = ft.from_buffer(raw, dtype=ft.fp32, shape=(2, 2))
print(t.tolist())  # [[1.0, 2.0], [3.0, 4.0]]
```

### CUDA Array Interface
CUDA 张量原生暴露 `__cuda_array_interface__`（版本 3），支持与 PyTorch、CuPy、Numba 进行零拷贝显存互通：

```python
import fugatensor as ft
import torch

a_ft = ft.ones(2, 2, device="cuda")
# 通过 __cuda_array_interface__ 零拷贝导出至 PyTorch
a_pt = torch.as_tensor(a_ft, device="cuda")
print(a_pt)  # tensor([[1., 1.], [1., 1.]], device='cuda:0')
```

<a id="manipulation"></a><a id="reshape"></a>

## 形状变换与属性读取

### `ft.reshape(tensor, *shape)` / `tensor.reshape(*shape)`
返回具有相同元素但新维度的张量。支持单一推断维度 `-1`。

- **约束**：元素总数必须严格守恒；保持连续元素次序；内部通过 Fuga IR 延迟原语表示，反向微分保持对应形状。

```python
x = ft.tensor([1, 2, 3, 4, 5, 6])
y = x.reshape(2, -1)
print(y.shape)     # (2, 3)
print(y.tolist())  # [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```

### `tensor.astype(dtype)` / `ft.cast(x, dtype)`
将张量转换为指定精度。

```python
x = ft.tensor([1.5, 2.7], dtype=ft.fp32)
y = x.astype(ft.int32)
print(y.tolist())  # [1, 2]
```

### 属性与导出方法

| 属性 / 方法 | 返回类型 | 说明 |
| --- | --- | --- |
| `x.shape` | `tuple[int, ...]` | 张量各维度长度 |
| `x.dtype` | `DType` | 元素类型单例 |
| `x.device` | `str` | 张量所在设备名称（如 `"cpu"` 或 `"cuda:0"`） |
| `x.ndim` | `int` | 张量维度数（`len(x.shape)`） |
| `x.numel` | `int` | 张量元素总数（`prod(x.shape)`） |
| `x.tolist()` | `list` / 标量 | 触发求值并将数值转换为原生嵌套 Python 对象 |
| `x.item()` | 标量 | 仅限单元素张量，转换为 Python `bool`、`int` 或 `float` |

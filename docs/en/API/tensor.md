Back to API Index: [Categories](README.md#categories) · Related Modules: [Basic Operations](ops.md) · [Device & Execution](engine.md) · [Autograd](autograd.md)

# Tensor Creation and Data Exchange

<a id="dtypes"></a>

## Data Type System

FugaTensor provides singleton type objects that support identity checking via `x.dtype is ft.fp32`:

| Type Object | Width & Format | Description |
| --- | --- | --- |
| `ft.bool` | 8-bit boolean | Conditional masks, comparison results |
| `ft.int32` | 32-bit signed integer | General integer arithmetic, indexing |
| `ft.int64` | 64-bit signed integer | Large counters, 64-bit integers |
| `ft.fp32` | 32-bit single-precision float | Default framework precision |
| `ft.fp16` | 16-bit half-precision float | Low-precision computation, model acceleration |
| `ft.bf16` | 16-bit bfloat16 float | CUDA only; requires Compute Capability >= 8.0 |

```python
import fugatensor as ft

x = ft.tensor([1, 2, 3], dtype=ft.int32)
print(x.dtype)          # int32
print(x.dtype is ft.int32)  # True
```

<a id="factories"></a><a id="tensor-create"></a><a id="size-factories"></a>

## Tensor Creation Factories

### `ft.tensor(data, *, dtype=ft.fp32, device=None, engine=None)`
Constructs a tensor from nested Python lists, scalars, or existing sequences.

- **Parameters**:
  - `data`: Scalar (bool/int/float) or regularly nested sequence.
  - `dtype`: Target precision; defaults to `ft.fp32`.
  - `device`: Target device, supporting `"cpu"`, `"cuda"`, `"cuda:0"`, or `ft.device` instances. Defaults to the current context's default device.
- **Returns**: An `ft.Tensor` instance.

```python
import fugatensor as ft

a = ft.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
print("Matrix:", a.tolist())  # [[1.0, 2.0], [3.0, 4.0]]
print("Shape:", a.shape)      # (2, 2)
```

### `ft.zeros(*size, dtype=ft.fp32, device=None)`
Creates an all-zeros tensor of the specified shape. Dimensions can be passed as unpacked arguments or a tuple.

```python
z = ft.zeros(2, 3, dtype=ft.fp16)
print(z.tolist())  # [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```

### `ft.ones(*size, dtype=ft.fp32, device=None)`
Creates an all-ones tensor of the specified shape.

```python
o = ft.ones((3, 2))
print(o.tolist())  # [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
```

### `ft.empty(*size, dtype=ft.fp32, device=None)`
Allocates underlying storage memory without initializing contents. Suitable as a pre-allocated write buffer.

```python
e = ft.empty(2, 2)
print(e.shape)  # (2, 2)
```

### `ft.full(size, fill_value, *, dtype=ft.fp32, device=None)`
Creates a tensor of the specified shape filled with a specified constant value.

```python
f = ft.full((2, 2), 3.14)
print(f.tolist())  # [[3.14, 3.14], [3.14, 3.14]]
```

### `ft.zeros_like(x)` / `ft.ones_like(x)` / `ft.full_like(x, fill_value, *, dtype=None, device=None)`
Creates an all-zeros, all-ones, or constant-filled tensor matching `x`'s shape (and by default `dtype` and `device`). `full_like` also works inside traced `ft.fun` bodies.

```python
x = ft.tensor([1, 2, 3], dtype=ft.int32)
z = ft.zeros_like(x)
f = ft.full_like(x, 7)
print(z.tolist(), f.tolist())  # [0, 0, 0] [7, 7, 7]
```

<a id="range-factories"></a>

## Coordinate, Sequence, and Diagonal Factories

Zero-input creation routines execute purely on the target device via ASK `iota` graphs (and also compose inside traced functions without host materialization):

| Function | Description |
| --- | --- |
| `ft.arange(start, stop=None, step=1, *, dtype=None, device=None)` | 1-D half-open arithmetic sequence; defaults to `ft.int64` when all inputs are integers, else `ft.fp32`. |
| `ft.linspace(start, stop, steps, *, dtype=ft.fp32, device=None)` | `steps` evenly spaced floating or integer samples over `[start, stop]`, pinning the endpoint when `steps > 1`. |
| `ft.eye(n, m=None, *, dtype=ft.fp32, device=None)` | 2-D `(n, m)` matrix with `1`s on the main diagonal and `0`s elsewhere. |
| `ft.iota(shape, *, dimension=0, dtype=ft.fp32, device=None)` | Tensor of `shape` filled with element coordinates `0, 1, ...` along `dimension` (numeric dtypes). |

```python
import fugatensor as ft

print(ft.arange(0, 8, 2).tolist())              # [0, 2, 4, 6]
print(ft.linspace(-1.0, 1.0, 3).tolist())       # [-1.0, 0.0, 1.0]
print(ft.eye(2, 3, dtype=ft.int32).tolist())    # [[1, 0, 0], [0, 1, 0]]
print(ft.iota((2, 3), dimension=1).tolist())    # [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
```

<a id="interop"></a>

## Memory Interoperability and Buffer Exchange

### `ft.from_buffer(data, *, dtype, shape, copy=True, device=None, engine=None)`
Constructs a tensor directly from an object supporting the Python Buffer Protocol (such as `bytes`, `bytearray`, `memoryview`).

```python
import array
raw = array.array('f', [1.0, 2.0, 3.0, 4.0])
t = ft.from_buffer(raw, dtype=ft.fp32, shape=(2, 2))
print(t.tolist())  # [[1.0, 2.0], [3.0, 4.0]]
```

### CUDA Array Interface
CUDA tensors natively expose `__cuda_array_interface__` (version 3), enabling zero-copy VRAM interoperation with PyTorch, CuPy, and Numba:

```python
import fugatensor as ft
import torch

a_ft = ft.ones(2, 2, device="cuda")
# Zero-copy export to PyTorch via __cuda_array_interface__
a_pt = torch.as_tensor(a_ft, device="cuda")
print(a_pt)  # tensor([[1., 1.], [1., 1.]], device='cuda:0')
```

<a id="manipulation"></a><a id="reshape"></a>

## Shape Manipulation and Attribute Access

### `ft.reshape(tensor, *shape)` / `tensor.reshape(*shape)`
Returns a tensor with the same elements under a new shape. Supports a single inferred dimension `-1`.

- **Constraints**: Total element count must be strictly conserved; contiguous element order is preserved; represented internally via Fuga IR lazy primitives, preserving corresponding shapes during reverse-mode differentiation.

```python
x = ft.tensor([1, 2, 3, 4, 5, 6])
y = x.reshape(2, -1)
print(y.shape)     # (2, 3)
print(y.tolist())  # [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```

### `tensor.astype(dtype)` / `ft.cast(x, dtype)`
Casts the tensor to the specified precision.

```python
x = ft.tensor([1.5, 2.7], dtype=ft.fp32)
y = x.astype(ft.int32)
print(y.tolist())  # [1, 2]
```

### Attributes and Export Methods

| Attribute / Method | Return Type | Description |
| --- | --- | --- |
| `x.shape` | `tuple[int, ...]` | Dimension lengths of the tensor |
| `x.dtype` | `DType` | Element data type singleton |
| `x.device` | `str` | Name of the device housing the tensor (e.g. `"cpu"` or `"cuda:0"`) |
| `x.ndim` | `int` | Number of dimensions (`len(x.shape)`) |
| `x.numel` | `int` | Total element count (`prod(x.shape)`) |
| `x.tolist()` | `list` / scalar | Triggers evaluation and converts values to native nested Python objects |
| `x.item()` | scalar | Converts single-element tensor to a Python `bool`, `int`, or `float` |

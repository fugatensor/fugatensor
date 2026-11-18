Back to API Index: [Categories](README.md#categories) · Related Modules: [Tensor Creation](tensor.md) · [Neural Network Ops](nn.md) · [Autograd](autograd.md) · [Device & Execution](engine.md)

# Operations, Indexing, and Reductions

FugaTensor operations return unevaluated expression nodes. Compatible elementwise, geometric, and reduction chains fuse during JIT compilation; `scatter_add` isolates into an atomic update step.

<a id="elementwise"></a>

## Element-Wise Math, Bitwise, and Logical Ops

Supports native Python operators and corresponding named functions:

| Operator | Function | Description |
| --- | --- | --- |
| `a + b`, `a - b`, `-a` | `ft.add`, `ft.subtract`, `ft.negative` | Addition, subtraction, negation (supports scalars) |
| `a * b`, `a / b` | `ft.multiply`, `ft.divide` | Element-wise multiply and true division |
| `a // b`, `a % b` | `ft.floordiv`, `ft.mod` | Floored division and remainder (integers & floats) |
| `a ** b` | `ft.pow(a, b)` | Element-wise power (`x ** y`) |
| `abs(a)`, `a.square()` | `ft.abs(a)`, `ft.square(a)` | Absolute value and square |
| - | `ft.maximum(a, b)`, `ft.minimum(a, b)`, `ft.clamp(x, min=None, max=None)` | Element-wise extrema and clamping (`ft.clip` alias) |
| - | `ft.exp`, `ft.log`, `ft.sqrt`, `ft.sin`, `ft.cos`, `ft.tanh`, `ft.erf` | Floating-point elementary math functions |
| - | `ft.floor`, `ft.ceil`, `ft.round` | Floating-point floor, ceiling, and round-half-to-even |
| `~a`, `a & b`, `a \| b`, `a ^ b` | `ft.bitwise_not`, `bitwise_and`, `bitwise_or`, `bitwise_xor` | Bitwise logic on `ft.bool`, `ft.int32`, `ft.int64` |
| `a << b`, `a >> b` | `ft.shift_left`, `ft.shift_right` | Deterministic shifts on `ft.int32`, `ft.int64` |
| - | `ft.logical_not`, `logical_and`, `logical_or`, `logical_xor` | Boolean truth-value logic across any dtype |

```python
import fugatensor as ft

x = ft.tensor([1.0, 4.0, 9.0])
y = ft.sin(ft.sqrt(x)).clamp(min=-0.5, max=0.9)
print(y.shape)  # (3,)
```

<a id="comparison"></a>

## Logical Comparisons and Conditional Selection

### Comparison Operators
Supports `==`, `!=`, `<`, `<=`, `>`, `>=`, and `ft.compare(a, b, op)`, returning an `ft.bool` tensor.

```python
a = ft.tensor([1, 4, 3])
b = ft.tensor([2, 2, 3])
mask = a > b
print(mask.tolist(), mask.dtype)  # [False, True, False] bool
```

### `ft.where(condition, true_branch, false_branch)`
Performs element-wise selection based on a boolean mask condition. Takes values from `true_branch` where `condition` is True, and `false_branch` otherwise.

```python
cond = ft.tensor([True, False, True])
x = ft.tensor([10.0, 20.0, 30.0])
y = ft.tensor([-1.0, -2.0, -3.0])
out = ft.where(cond, x, y)
print(out.tolist())  # [10.0, -2.0, 30.0]
```

<a id="geometry"></a>

## Broadcasting and Geometric Rearrangement

### `ft.broadcast_to(x, shape)`
Broadcasts a tensor to a target shape following standard NumPy broadcasting compatibility rules.

```python
x = ft.tensor([1.0, 2.0])  # shape (2,)
b = ft.broadcast_to(x, (2, 2))
print(b.tolist())  # [[1.0, 2.0], [1.0, 2.0]]
```

### `ft.broadcast_in_dim(x, shape, broadcast_dimensions)`
Low-level exact broadcasting primitive that specifies the explicit output dimension indices corresponding to each input axis.

```python
x = ft.tensor([10.0, 20.0])
# Map dimension 0 of the 1D tensor to axis 0 of the target shape (2, 3)
b = ft.broadcast_in_dim(x, (2, 3), (0,))
print(b.tolist())  # [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]
```

### `ft.transpose(x, permutation=None)` / `x.T`
Transposes tensor dimensions. If `permutation` is omitted, reverses all dimensions by default. 2D matrices provide a `.T` shorthand property.

```python
m = ft.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(m.T.shape)     # (3, 2)
print(m.T.tolist())  # [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```

<a id="slicing-and-indexing"></a>

## Slicing, Padding, Concatenation, and Indexing

- **Indexing and Slicing (`x[...]`, `ft.slice`)**: Supports integers (squeezing dimensions), slices (`start:stop:step` with positive or negative steps), `None` (`newaxis`), `...` (`Ellipsis`), and up to one integer `Tensor` index (lowered via `gather`). Boolean mask indexing requires static shapes and is intentionally omitted.
- **Padding (`ft.pad(x, pad_width, constant_value=0, *, interior=None)`)**: Pads borders and optional `interior` gaps using a scalar or differentiable 0-D `Tensor` fill value.
- **Combining & Splitting**: `ft.concatenate(tensors, axis=0)` (`ft.cat`), `ft.stack(tensors, axis=0)`, `ft.split(tensor, split_size_or_sections, axis=0)`, `ft.chunk(tensor, chunks, axis=0)`.
- **Shape Utilities**: `ft.squeeze(x, axis=None)`, `ft.unsqueeze(x, axis)`, `ft.flatten(x, start_dim=0, end_dim=-1)`.
- **Masks & One-Hot**: `ft.tril(x, diagonal=0)`, `ft.triu(x, diagonal=0)`, `ft.one_hot(indices, num_classes, *, dtype=ft.fp32)`.
- **Gather & Scatter-Add**:
  - `ft.gather(params, indices, axis=0)` (`ft.take`): Indexed read along `axis`; out-of-bounds indices return zeros; VJP accumulates gradients into `params` via `scatter_add`.
  - `ft.scatter_add(base, indices, updates, axis=0)`: Initializes output with `base` and atomically accumulates `updates` along `axis` (skipping out-of-bounds indices); differentiable with respect to `base` and `updates`.

```python
x = ft.arange(10).reshape(2, 5)
print(x[::-1, 1::2].tolist())  # [[6, 8], [1, 3]]
base = ft.zeros(3, 2)
idx = ft.tensor([2, 0, 2], dtype=ft.int32)
upd = ft.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(ft.scatter_add(base, idx, upd, axis=0).tolist())  # [[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]]
```

<a id="reductions"></a>

## Reductions

Reduction operators collapse tensors along specified axes:

- **`ft.sum` / `ft.mean` / `ft.prod`**: Arithmetic sum, mean, and product (`dtype=` cast supported before reduction; empty `sum`/`prod` yield `0`/`1`).
- **`ft.amax` / `ft.amin`**: Maximum and minimum along non-empty domains; floats propagate `NaN`.
- **`ft.any` / `ft.all`**: Boolean reductions along `axis` (empty domains yield `False` / `True`).
- **`ft.argmax` / `ft.argmin`**: Returns `ft.int64` first-occurrence indices of extrema (or first `NaN` for floats) along non-empty axes.

```python
x = ft.tensor([[2.0, 4.0], [6.0, 8.0]])
print(x.sum(axis=0).tolist())        # [8.0, 12.0]
print(x.prod(axis=1).tolist())       # [8.0, 48.0]
print(x.argmax(axis=1).tolist())     # [1, 1]
```

<a id="matmul"></a>

## Matrix Multiplication

### `a @ b` / `ft.matmul(a, b, *, transpose_a=False, transpose_b=False)`
Performs 2D matrix multiplication or batched matrix multiplication.

- **Backend Mechanism**:
  - CPU uses ASK-optimized BLAS lowering and caching;
  - CUDA uses the high-performance cuBLASLt engine, supporting FP32 (TF32 acceleration) and FP16 Tensor Cores, automatically reusing tuned static descriptors and scratch slots.
- **Parameters**:
  - `transpose_a`: Whether to transpose matrix A before multiplication.
  - `transpose_b`: Whether to transpose matrix B before multiplication.

```python
import fugatensor as ft

a = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
b = ft.tensor([[5.0, 6.0], [7.0, 8.0]])
c = a @ b
print(c.tolist())  # [[19.0, 22.0], [43.0, 50.0]]
```

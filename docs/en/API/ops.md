Back to API Index: [Categories](README.md#categories) · Related Modules: [Tensor Creation](tensor.md) · [Autograd](autograd.md) · [Device & Execution](engine.md)

# Basic Arithmetic and Algebraic Operations

FugaTensor operations return unevaluated expression nodes. Operators are automatically fused into a single compute kernel during JIT compilation.

<a id="elementwise"></a>

## Element-Wise Arithmetic

Supports native Python operators and corresponding named functions:

| Operator | Function | Description |
| --- | --- | --- |
| `a + b` | `ft.add(a, b)` | Addition (supports scalar + tensor) |
| `a - b` | `ft.subtract(a, b)` | Subtraction |
| `-a` | `ft.negative(a)` | Negation |
| `a * b` | `ft.multiply(a, b)` | Element-wise multiplication |
| `a / b` | `ft.divide(a, b)` | Element-wise division |
| `a.square()` | `ft.square(a)` | Element-wise square (`a * a`) |
| - | `ft.exp(a)` | Natural exponential ($e^x$) |
| - | `ft.log(a)` | Natural logarithm ($\ln(x)$) |
| - | `ft.sqrt(a)` | Square root ($\sqrt{x}$) |

```python
import fugatensor as ft

x = ft.tensor([1.0, 4.0, 9.0])
y = ft.sqrt(x) + 1.0
print(y.tolist())  # [2.0, 3.0, 4.0]
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

<a id="reductions"></a>

## Reductions

Reduction operators collapse tensors along specified axes:

### `ft.sum(x, axis=None, keepdims=False, dtype=None)` / `x.sum(...)`
Sums elements along specified axes. If `axis=None`, reduces all elements across the entire tensor.

```python
x = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.sum().item())                 # 10.0
print(x.sum(axis=0).tolist())          # [4.0, 6.0]
print(x.sum(axis=1, keepdims=True).shape) # (2, 1)
```

### `ft.mean(x, axis=None, keepdims=False, dtype=None)` / `x.mean(...)`
Computes the mean along specified axes.

```python
x = ft.tensor([[2.0, 4.0], [6.0, 8.0]])
print(x.mean().item())        # 5.0
print(x.mean(axis=1).tolist()) # [3.0, 7.0]
```

### `ft.amax(x, axis=None, keepdims=False)` / `x.amax(...)`
Computes the maximum value along specified axes.

```python
x = ft.tensor([1.5, 9.2, 3.4])
print(x.amax().item())  # 9.2
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

Back to API Index: [Categories](README.md#categories) · Related Modules: [Basic Operations](ops.md) · [IR & Lowering](ir.md) · Architecture: [Autograd Architecture](../architecture/autograd.md)

# Autograd and Custom Operators

FugaTensor uses functional automatic differentiation. Autograd operates as a directed graph transformation on Fuga IR expressions and returns unevaluated derivative tensors, supporting arbitrary composition, nesting, and higher-order derivatives.

<a id="grad"></a>

## Autograd Interface

### `ft.grad(outputs, inputs, *, seed=None)`
Computes gradients of outputs with respect to specified inputs.

- **Parameters**:
  - `outputs`: Target scalar tensor or a sequence of tensors (tuple/list).
  - `inputs`: Independent variable tensor or a sequence of tensors.
  - `seed`: Initial cotangent seed for backpropagation. If `outputs` is a scalar, the seed defaults to 1.0; if non-scalar, an explicit seed matching in structure, shape, and dtype must be provided.
- **Returns**: Gradient tensor matching the structure of `inputs` (returns a gradient tuple if `inputs` is a sequence). Inputs not connected to the outputs return zero gradients of matching shape.

```python
import fugatensor as ft

# Scalar function differentiation: y = x^2 + 3x
x = ft.tensor([2.0, 3.0])
y = (x * x + 3.0 * x).sum()

dx = ft.grad(y, x)
print(dx.tolist())  # [7.0, 9.0] (corresponds to 2x + 3)

# Higher-order differentiation
d2x = ft.grad(dx.sum(), x)
print(d2x.tolist())  # [2.0, 2.0]
```

<a id="gradient-control"></a>

## Gradient Control

### `ft.stop_gradient(x)` / `x.detach()`
Disconnects the autograd dependency chain of the tensor while preserving forward numerical dependencies. When propagating derivatives through the computation graph, backpropagation truncates at this boundary without passing gradients back to inputs.

```python
x = ft.tensor([2.0, 4.0])
y = ft.stop_gradient(x * 3.0)
z = (y * x).sum()

dx = ft.grad(z, x)
# When differentiating with respect to x, y is treated as a constant: d/dx (y * x) = y = [6.0, 12.0]
print(dx.tolist())  # [6.0, 12.0]
```

<a id="custom-fun"></a>

## Custom Differentiable Functions: `ft.fun`

`ft.fun` encapsulates custom logic into a differentiable composite operator with isolated tracing boundaries, allowing explicit backward propagation defined via pure `vjp` rules:

### Contract Specifications
1. Subclass `ft.fun` and implement `forward(self, ...)`;
2. (Optional) Implement the pure function `vjp(self, primals, outputs, cotangents)`:
   - `primals`: A tuple of all tensor inputs to the forward pass;
   - `outputs`: A tuple of all public outputs from the forward pass;
   - `cotangents`: A tuple of cotangent tensors aligned with `outputs`;
   - Return value: A tuple of gradients aligned with floating-point tensor inputs in `primals`.

```python
import fugatensor as ft

class CustomSquare(ft.fun):
    def forward(self, x):
        return x * x

    def vjp(self, primals, outputs, cotangents):
        (x,) = primals
        (dy,) = cotangents
        # Custom backward rule: dy * 2 * x
        return (dy * 2.0 * x,)

square_op = CustomSquare()
a = ft.tensor([3.0, 5.0])
b = square_op(a)
print("Forward:", b.tolist())  # [9.0, 25.0]

da = ft.grad(b.sum(), a)
print("Gradient:", da.tolist())  # [6.0, 10.0]
```

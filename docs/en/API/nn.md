Back to API Index: [Categories](README.md#categories) · Related Modules: [Basic Operations](ops.md) · [Attention Operators](attention.md) · [Autograd](autograd.md)

# Neural Network Functional Operators (`ft.nn.functional`)

`ft.nn.functional` implements common activations, normalizations, embedding lookup, and loss functions directly on Fuga IR primitives, so they participate in fusion, CPU/CUDA compilation, and higher-order `ft.grad` automatic differentiation without Python numerical fallbacks. For cuDNN FlashAttention, see [Attention Operators](attention.md).

<a id="activations"></a>

## Activations

| Function | Formula / Behavior | Notes |
| --- | --- | --- |
| `ft.nn.functional.relu(x)` | $\max(x, 0)$ | Supports integer and floating-point tensors |
| `ft.nn.functional.leaky_relu(x, negative_slope=0.01)` | `x` where `x > 0` else `negative_slope * x` | Supports floating-point tensors |
| `ft.nn.functional.sigmoid(x)` | $1 / (1 + e^{-x})$ | Sign-split formulation avoids overflow on large magnitudes |
| `ft.nn.functional.silu(x)` | $x \cdot \sigma(x)$ | Swish / SiLU activation |
| `ft.nn.functional.gelu(x, *, approximate='none')` | `'none'`: $\frac{1}{2} x (1 + \text{erf}(x / \sqrt{2}))$; `'tanh'`: tanh approximation | Differentiable under `ft.grad` |

```python
import fugatensor as ft
import fugatensor.nn.functional as F

x = ft.tensor([-1.0, 0.0, 2.0])
print(F.relu(x).tolist())     # [0.0, 0.0, 2.0]
print(F.sigmoid(x).tolist())  # [0.2689..., 0.5, 0.8807...]
```

<a id="softmax-and-norms"></a>

## Softmax and Normalizations

- `ft.nn.functional.softmax(x, axis=-1)` / `ft.nn.functional.log_softmax(x, axis=-1)`:
  Max-shifted numerically stable softmax and log-softmax along `axis`. `ft.fp16` and `ft.bf16` statistics accumulate in `ft.fp32` and cast back to the input precision; max stabilization uses `stop_gradient` so first- and higher-order gradients remain well-behaved.
- `ft.nn.functional.layer_norm(x, normalized_shape=None, weight=None, bias=None, *, eps=1e-5, axis=None)`:
  Normalizes across `normalized_shape` trailing axes (defaulting to the last axis) or explicit `axis`, accumulating statistics in `ft.fp32` for low-precision inputs and applying optional affine `weight`/`bias`.
- `ft.nn.functional.rms_norm(x, normalized_shape=None, weight=None, *, eps=1e-6, axis=None)`:
  Root Mean Square Normalization over `normalized_shape` or `axis` with optional scaling `weight`.

```python
import fugatensor as ft
import fugatensor.nn.functional as F

x = ft.tensor([[1.0, 2.0, 3.0]])
w = ft.ones(3)
print(F.softmax(x, axis=-1).tolist())
print(F.rms_norm(x, weight=w).tolist())
```

<a id="embedding-and-loss"></a>

## Embeddings and Losses

- `ft.nn.functional.embedding(input, weight)`:
  Gathers rows from 2-D or ND `weight` along `axis=0` at integer indices `input`. Forward lookup compiles to `gather`; reverse gradients accumulate cleanly into duplicated weights via `scatter_add`.
- `ft.nn.functional.cross_entropy(logits, targets, *, axis=-1, reduction='mean')`:
  Computes negative log-likelihood from unnormalized `logits` and integer class `targets` (`logits` batch shape with `axis` removed). `reduction` may be `'mean'`, `'sum'`, or `'none'`.

```python
import fugatensor as ft
import fugatensor.nn.functional as F

table = ft.tensor([[0.1, 0.2], [0.3, 0.4], [-0.5, 0.6]])
tokens = ft.tensor([0, 2, 0], dtype=ft.int32)
emb = F.embedding(tokens, table)
grad_table = ft.grad(emb.sum(), table)
print(grad_table.tolist())  # [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]
```

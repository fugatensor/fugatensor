# FugaTensor API

Back to Documentation Index: [Documentation Index](../README.md) · Setup: [Installation & Setup](../install.md) · System Architecture: [Architecture Overview](../architecture/README.md)

FugaTensor defaults to CPU and `ft.fp32`. Tensor operations record unevaluated nodes in independent Fuga IR, which are optimized via automatic differentiation and C++ liveness analysis before being handed over to the backend for JIT compilation and execution.

<a id="quick-start"></a>

## Quick Start

```python
import fugatensor as ft

# Create tensors
x = ft.tensor([1.0, 2.0, 3.0])
y = (x + 1.0) * 2.0

# Lazy evaluation: inspecting metadata does not trigger execution
print("Shape:", y.shape)  # (3,)
print("Dtype:", y.dtype)  # fp32

# Reading values or explicit evaluation triggers compilation and execution
print("Result:", y.tolist())  # [4.0, 6.0, 8.0]
```

<a id="categories"></a>

## API by Category

| Module | Description | Core APIs |
| --- | --- | --- |
| [Tensor Creation & Data Exchange](tensor.md) | Tensor construction, coordinate/identity factories, type system, and memory interoperability | `tensor`, `zeros`, `ones`, `empty`, `full`, `full_like`, `arange`, `linspace`, `eye`, `iota`, `from_buffer`, `reshape`, `astype` |
| [Basic Arithmetic & Algebraic Ops](ops.md) | Element-wise math, logic/bitwise, slicing/padding/concat, indexing/scatter, reductions, and matmul | `add`, `sin`, `where`, `slice`, `pad`, `cat`, `gather`, `scatter_add`, `sum`, `amax`, `amin`, `prod`, `argmax`, `matmul` |
| [Autograd & Custom Operators](autograd.md) | Functional differentiation, gradient cut-off, and explicit VJP operators | `grad`, `stop_gradient`, `detach`, `fun` |
| [Device Management & Execution](engine.md) | Device switching, default device settings, stream synchronization, and engine lifecycle | `device`, `set_default_device`, `evaluate`, `synchronize`, `Engine` |
| [Neural Network Operators](nn.md) | Common activation, normalization, embedding lookup, and cross-entropy loss functions | `nn.functional.relu`, `gelu`, `softmax`, `layer_norm`, `rms_norm`, `embedding`, `cross_entropy` |
| [Attention Operators](attention.md) | High-performance built-in attention operators, causal masking, and variable-length SDPA | `nn.functional.scaled_dot_product_attention`, `nn.functional.scaled_dot_product_attention_varlen` |
| [Triton Kernel Integration](triton.md) | External Triton custom kernel binding and configuration callbacks | `triton.kernel`, `triton.from_source`, `triton.from_jit` |
| [Weight Persistence](weights.md) | Model parameter PyTree serialization and content-addressed verification | `save_weights`, `load_weights`, `weights.manifest` |
| [IR & Lowering](ir.md) | Fuga IR primitive construction, control flow, and ASK compiler bridge | `ir.Module`, `lower_to_ask` |

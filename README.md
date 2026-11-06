<img src="docs/assets/fugatensor-logo-inverse.svg" alt="FugaTensor logo" width="100%">

---

<p align="center">
  <b><a href="README.md">English</a></b> | <b><a href="README.zh.md">简体中文</a></b>
</p>

<p align="center">
  <a href="docs/en/install.md#requirements"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab" alt="Python 3.10+"></a>
  <a href="docs/en/API/engine.md#default-device"><img src="https://img.shields.io/badge/backend-CPU%20%7C%20CUDA-76b900" alt="CPU / CUDA"></a>
  <a href="docs/en/install.md#installation-variants"><img src="https://img.shields.io/badge/Triton-optional-8b5cf6" alt="Triton optional"></a>
</p>

FugaTensor is a tensor computing framework based on [ASK](../ask/README.md). Tensor operations are initially recorded into Fuga IR; reading results, evaluating scalar conditions, or calling `ft.evaluate` triggers JIT compilation and execution on CPU or CUDA.

```python
import fugatensor as ft

def arithmetic(x, y):
    return (x + y) * 2

x = ft.tensor([[1, 2], [3, 4]])
y = ft.tensor([[10, 20], [30, 40]])
print(arithmetic(x, y).tolist())
# [[22.0, 44.0], [66.0, 88.0]]
```

Defaults to CPU and `ft.fp32`. Other precisions are specified via type objects such as `dtype=ft.fp16`, and devices via `ft.device("cuda", 0)` or `"cuda:0"`. Call `ft.set_default_device(...)` to change the default device for subsequent tensor creations.

Size-based creation and reshaping can be expressed directly:

```python
x = ft.ones(2, 2)
y = ft.tensor([1, 2, 3, 4]).reshape(2, 2)
print((x + y).reshape(-1).tolist())  # [2.0, 3.0, 4.0, 5.0]
```

`tensor` infers shapes from nested data; `zeros`, `ones`, and `empty` accept dimensions, while `full(size, value)` populates values. `reshape` preserves element order and autograd graph connections, allowing `-1` for inferred dimensions. See [Tensor Creation & Shapes](docs/en/API/tensor.md#size-factories).

For API reference, see [API](docs/en/API/README.md). For architecture and design, see [Documentation Index](docs/en/README.md), [Architecture Overview](docs/en/architecture/README.md), and [Fuga IR Overview](docs/en/ir/README.md).

<a id="setup"></a>

## Standalone Build and Installation

Install the ASK C++ SDK first, then build FugaTensor by providing `CMAKE_PREFIX_PATH`. The resulting wheel packages the ASK compiler and runtime shared libraries. The native extension consumes compilation artifacts and diagnostics via [ASK named stage results](docs/en/architecture/backend.md#compiler-results).

```bash
CMAKE_PREFIX_PATH=/absolute/path/to/ask-sdk python3 -m pip wheel --no-deps . --wheel-dir dist
```

`fugatensor[pure]` installs base dependencies; `fugatensor[all]` additionally installs Triton. Both share the same wheel; installing `fugatensor` directly defaults to `pure`. See [Installation Variants](docs/en/install.md#installation-variants) for commands and environment requirements.

CPU JIT uses `cc` and libm; FP32 matrix multiplication links against OpenBLAS provided by the ASK SDK. Native CUDA operations compile via `llc` / `ptxas`; exp and log link CUDA libdevice, while matrix multiplication uses cuBLASLt with TF32 enabled for FP32. See [Installation Guide](docs/en/install.md) for toolchain configuration and setup.

## Basic Usage

### 1. Functional Autograd and Custom VJP

```python
def loss(x):
    return ft.square(x) + 3 * x

x = ft.tensor(2.0)
y = loss(x)
first = ft.grad(y, x)
print(first.tolist())           # 7.0
print(ft.grad(first, x).tolist()) # 2.0
```

`ft.grad` returns unevaluated gradient tensors; higher-order derivatives can be computed after reading results. Non-scalar outputs require cotangent seeds matching shape and dtype via `seed=`. `ft.stop_gradient(x)` or `x.detach()` explicitly cuts autograd paths and can also be used in `ft.fun` forward passes and VJPs.

Custom functions inherit from `ft.fun`. Standard forward passes differentiate automatically, or can define pure VJP rules:

```python
class CustomSquare(ft.fun):
    def forward(self, x):
        return ft.square(x)

    def vjp(self, primals, outputs, cotangents):
        (x,) = primals
        (dy,) = cotangents
        return (2 * x * dy,)
```

All three VJP arguments are tuples, and the return value must be a tuple of gradients corresponding to floating-point tensor inputs. Gradient graphs compose into higher-order derivatives provided active paths define valid derivative rules. See [Autograd API](docs/en/API/autograd.md) and [IR Transformations](docs/en/architecture/autograd.md).

### 2. Integrating User Triton Kernels

Use `ft.triton.from_source(source, ...)` to register source strings, `ft.triton.kernel(path, ...)` to load kernel files, or `ft.triton.from_jit(fn, ...)` to wrap existing JIT functions. The returned operators compose seamlessly with native tensor operations.

Kernel source snapshots are captured upon registration. The worker compiles them into cubins, and ASK schedules them according to graph dependencies. For scalars, multi-outputs, and compilation configs, see [Triton Integration](docs/en/API/triton.md#example); for environment setup, see [Triton Configuration](docs/en/install.md#configure-triton).

The built-in attention operator is accessible via `ft.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`, supporting CUDA FP16/BF16 forward and first-order backward passes. See [Attention API](docs/en/API/attention.md) for shape constraints.

### 3. Inspecting IR, Trace Artifacts, and Cache Statistics

```python
x = ft.tensor([1, 2, 3])
y = (x + 1) * 2
ft.evaluate(y)                   # Explicit execution; supports multiple outputs
engine = x.engine                # Inspect the underlying execution engine
print(engine.cache_info())       # Graph cache entries, compilation counts, hit stats
print(engine.last_fuga_ir)        # Last executed Fuga IR
print(engine.last_ask_ir)         # MLIR text submitted to ASK
print(engine.last_report)        # Compiler diagnostics and cache breakdown
print(engine.last_execution_plan) # CUDA partition plan; None on CPU
print(engine.last_artifact_dir)  # fuga.ir, frontend.mlir, and fuga-to-ask.json
```

`ft.evaluate(a, b)` merges and evaluates dependency graphs across multiple outputs, returning the original tensors. `shape`, `dtype`, `device`, and `repr` inspect metadata. Python `if`/`while` conditions on 0-D tensors evaluate truth values on-demand before continuing graph tracing along the taken path; data-dependent control flow inside `ft.fun` raises an error.

The engine queries caches by graph template and connectivity; cache hits reuse compiled artifacts while normal Python logic executes unchanged. Contiguous memory can be imported with `ft.from_buffer(...)` and exported with `tensor.to_buffer()`. See [Buffer Interface & Lifetime Rules](docs/en/architecture/tensor_model.md#buffer-io).

## Current Feature Support

| Category | Current Behavior |
| --- | --- |
| **Data Types** | Defaults to `ft.fp32`; supports `ft.bool`, `ft.int32`, `ft.int64`, `ft.fp32`, `ft.fp16`, `ft.bf16` via `.astype(...)`; BF16 tensors require CUDA |
| **Tensor Creation & Ops** | `tensor`, `zeros`, `ones`, `empty`, `full`; basic arithmetic, exp/log/sqrt, comparisons, explicit broadcasting, transposition, reductions, matmul (static shapes, including 0-D and empty tensors) |
| **Control Flow** | Fuga IR static `if` selection and bounded `for` unrolling; Python `if`/`while` on 0-D tensors evaluates on demand |
| **Autograd** | `ft.grad(outputs, inputs, seed=...)` lazy differentiation, `detach()` / `stop_gradient()`, `ft.fun` pure VJP rules, gradient graph composition, and higher-order derivatives |
| **Attention** | `nn.functional.scaled_dot_product_attention`, CUDA FP16/BF16, optional causal masking, FlashAttention forward and first-order backward |
| **Weight Export** | Versioned PyTree weight directories, nested dict/list/tuple, 6 base dtypes, content-addressed deduplication, atomic serialization, integrity validation |
| **Backends** | CPU (`.o` → `.so` → `ctypes`), native CUDA, Triton cubin, multi-kernel scheduling, multi-port buffers, and by-value scalar arguments |

Differentiation for external kernels is provided via pure VJP rules; higher-order derivatives compose along active graph paths. Built-in FlashAttention provides first-order derivatives.

## Testing

Numerical tests generate representative combinations covering operators, precisions, and boundary conditions. State and lifetime behaviors are validated via dedicated contract tests. `--exhaustive` expands the parameter search space. See [Testing Matrix](docs/en/testing.md).

```bash
# CPU JIT, C++ native graph analysis, Fuga IR / JVP / VJP, and lowering tests
.venv/bin/python -m pytest -q

# Enable native CUDA physical GPU tests
.venv/bin/python -m pytest -q --cuda

# Enable native CUDA and Triton GPU scheduling & backward tests
.venv/bin/python -m pytest -q --cuda --triton
```

## Documentation

- [Installation and Standalone Build](docs/en/install.md)
- [Documentation Index](docs/en/README.md)
- [API](docs/en/API/README.md): Tensors, data exchange, arithmetic ops, devices, autograd, custom functions, Triton integration, weights, IR lowering, and attention.
- **System Architecture (`docs/en/architecture/`)**: [Architecture Overview](docs/en/architecture/README.md) · [Tensor Model & Tracing](docs/en/architecture/tensor_model.md) · [Native Graph Analysis & C++ Boundary](docs/en/architecture/native.md) · [Autograd & Custom Operators](docs/en/architecture/autograd.md) · [ASK Bridge & Multi-Backend Execution](docs/en/architecture/backend.md) · [Weight Format & Export](docs/en/architecture/weights.md)
- **Intermediate Representation (`docs/en/ir/`)**: [Fuga IR Overview](docs/en/ir/README.md) · [Object Model & Verification](docs/en/ir/model.md) · [Control Flow & Composable Transforms](docs/en/ir/transforms.md) · [Lowering to ASK](docs/en/ir/lowering.md)

## License

This project is licensed under the [Apache-2.0](LICENSE) License.

[Multi-Kernel Plan Contract](../ask/docs/en/partitioning.md) · [Attention API](docs/en/API/attention.md) · [Weight Format & Export](docs/en/architecture/weights.md)

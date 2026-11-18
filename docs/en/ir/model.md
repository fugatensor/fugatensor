# Object Model and Verification

Back to IR Overview: [Object Model and Verification](README.md#ir-object-model) · Related topics: [Control Flow and Composable Transformations](transforms.md) · [Lowering to ASK](lowering.md) · [Native Graph Analysis and C++ Boundary](../architecture/native.md)

<a id="types-and-ssa"></a>

## 1. Types and SSA Values (`ir/types.py` / `ir/core.py`)

- `ScalarType(dtype)` and `TensorType(shape, dtype)`:
  - `dtype` accepts `ft.bool`, `ft.int32`, `ft.int64`, `ft.fp32`, `ft.fp16`, `ft.bf16`, internally normalized to `("bool", "i32", "i64", "f32", "f16", "bf16")` encodings; the set of differentiable floating-point types is defined as `FLOAT_DTYPES = frozenset(("f16", "bf16", "f32"))`.
  - Scalars are printed as `f32`, `i32`, etc.; 0-D tensors `TensorType((), "f32")` are printed as `tensor<f32>`, with both types represented distinctly. Multidimensional or empty tensors like `(2, 3)`, `(0, 4)` are printed as `tensor<2x3xf32>`, `tensor<0x4xf32>`.
  - Tensor dimensions allow non-negative integers or unknown dimensions `None`, and must be fully concretized prior to lowering to ASK. Devices are chosen by the execution engine.
- SSA `Value`:
  - Each `Value` possesses a unique `value_id`, a short print name, a `ScalarType` or `TensorType`, an owning `Block` (`owner`), and a defining operation `producer` (`producer` is `None` for block arguments).
  - Each `Value` is defined exactly once, and `value_id` serves as the identifier across graph transformations and cross-layer mappings.

<a id="ops-and-regions"></a>

## 2. Operations, Basic Blocks, Regions, and Module Hierarchy

```text
Module (functions)
 └── Function (name, body, inputs, result_types)
      └── Region (blocks, parent)
           └── Block (arguments, operations)
                └── Operation (opcode, operands, results, attrs, regions, effects, id, origin_id)
```

List of currently supported `Operation.opcode`:

| Category | Opcode | Semantics and Attributes / Regions |
| --- | --- | --- |
| **Constants & Coordinate Factories** | `constant` | No operands, produces a `ScalarType` value, `attrs["value"]` rounded to declared precision; floating-point text printing uses `float.hex()` to maintain exact precision. |
| | `splat` | Splats a scalar operand to the tensor shape specified by `attrs["shape"]`. |
| | `iota` | No operands; produces a numeric tensor of `attrs["shape"]` / `attrs["dtype"]` holding coordinates along `attrs["dimension"]`. |
| **Basic Elementwise & Bitwise** | `add`, `mul`, `pow`, `floordiv`, `mod` | Binary arithmetic on matching numeric scalar/tensor operands; `pow`/`floordiv`/`mod` accept integer and floating-point types. |
| | `div`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `erf`, `floor`, `ceil`, `round` | Floating-point elementwise operations preserving type and shape. |
| | `abs`, `maximum`, `minimum` | `abs` accepts numeric operands; `maximum`/`minimum` accept all six types. |
| | `bitwise_not`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `shift_left`, `shift_right` | Bitwise logic on `bool`/`i32`/`i64`, and deterministic bit shifts on `i32`/`i64`. |
| | `compare`, `select` | Comparison specified by `predicate`, returning `bool` of matching shape; `select` condition is `bool`, both value operands share identical type. |
| | `cast` | Casts to `dtype` declared in result, preserving original scalar or tensor shape. |
| **Shape, Slicing & Indexing** | `reshape` | One tensor input, `attrs["shape"]` provides concrete target shape; preserves dtype, total element count, and contiguous element order, supporting 0-D and empty tensors. |
| | `broadcast_in_dim`, `transpose` | Broadcast stores `shape/dimensions`, transpose stores `permutation`. |
| | `slice`, `pad`, `concatenate` | `slice` stores `starts/stops/steps`; `pad` takes `(value, pad_value)` (scalar or 0-D tensor) and `pad_low/pad_high/interior`; `concatenate` joins >= 1 same-dtype operands along `axis`. |
| | `gather`, `scatter_add` | `gather` indexes `params` by `indices` along `axis`; `scatter_add` initializes from `base` and accumulates `updates` at `indices` along `axis`. |
| **Reductions** | `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_prod`, `reduce_any`, `reduce_all` | `axes` are strictly ascending and unique axes (`reduce_max/min` require non-empty domains; `any/all` take `bool`; full reduction yields `ScalarType`). |
| | `argmax`, `argmin` | Reduces non-empty single `axis` into `i64` first-extremum indices (`ScalarType` when rank == 1). |
| **Matrix Multiplication** | `matmul` | Batch-aligned matrix multiplication, `transpose_a/b` controls transposition of the trailing two dimensions; fp32 accumulation. |
| **Differentiation Boundary** | `stop_gradient` | One scalar or tensor input, forward returns value, type, and shape verbatim; JVP tangents and VJP cotangents halt propagation, and numerical lowering directly maps the input value. |
| **Functions & External Ops** | `call` | `attrs["spec"]` stores `CallSpec` and forward subgraph (e.g. [`ft.fun`](../architecture/autograd.md#custom-fun)), results are public forward outputs, differentiation rules receive primals, forward outputs, and cotangents. |
| | `external` | Calls third-party compiler operators, `attrs["spec"]` stores `ExternalSpec` (compiler ID, source code text, port bindings, constants, grid, and effects list). |
| **Structured Control Flow** | `if` | Accepts scalar or 0-D tensor `bool` condition, contains `then` and `else` single-block `Region`s; result type lists yielded by both branches must match. |
| | `for` | Static interval counter loop (`attrs` contains `lower`, `upper`, `step`), loop body block arguments are `i64` scalar induction variable followed by carried states. |
| | `while` | Contains condition and body `Region`s: condition region yields a boolean, body region yields carried states for the next iteration. |
| **Terminators** | `return`, `yield` | Serve as final terminating instructions for top-level functions and nested regions, respectively. |

`Operation.origin_id` inherits the source node ID when cloning, unrolling loops, or generating new nodes during differentiation, ultimately recorded into [`fuga-to-ask.json`](lowering.md#external-and-sourcemap).

<a id="module-verifier"></a>

## 3. Structural Verifier (`ir/core.py`)

`Module.verify()` validates the following rules, raising `ValueError` or `TypeError` upon violation:

1. **Dominance and Scoping**: Operands must be entry arguments of the current block or results defined prior to use. Regions receive external values via block arguments.
2. **Block Structure and Terminators**: Each region contains a single non-empty block; `return` / `yield` can and must reside only at the end of the block, with operand types matching function `result_types` or the result contract of the parent control flow instruction one by one.
3. **Operator Type Constraints**: `add`/`mul` reject `bool` or mismatched types, `splat` requires a scalar input, `reshape` requires input and output to be concrete tensors of matching dtype and element count; conditions require scalar or 0-D `bool`, `for` requires `step != 0` with unchanged carried types, and `call` verifies the forward signature in `CallSpec`.

Cached templates freeze after verification and store structural keys. Appending further operations or modifying functions/blocks raises an error to ensure structural keys correspond to actual graph contents.

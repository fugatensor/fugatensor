Back to API Index: [Categories](README.md#categories) · IR Model: [Object Model & Verification](../ir/model.md) · Control Flow & Transforms: [Composable Transforms](../ir/transforms.md) · Lowering: [Lowering to ASK](../ir/lowering.md)

# IR and Lowering

Fuga IR is a backend-independent, purely static SSA intermediate representation. Graph construction, graph transformation, and lowering/compilation are cleanly decoupled. Standard users generate IR automatically through high-level Tensor lazy tracing, while advanced developers and operator implementers can directly manipulate Fuga IR to construct and transform computation graphs.

---

<a id="ir-types-and-builders"></a>

## Type System and Graph Construction

| API | Description |
| --- | --- |
| `ft.ir.ScalarType(dtype)` | Scalar type. `dtype` accepts `ft.bool`, `ft.int32`, `ft.int64`, `ft.fp32`, `ft.fp16`, `ft.bf16`. |
| `ft.ir.TensorType(shape, dtype)` | Tensor type. `shape` is a tuple of dimensions; `None` is permitted as a placeholder, but static lowering requires concrete dimensions. |
| `ft.ir.Type` | Type alias for `ScalarType | TensorType`. |
| `ft.ir.DTYPES`, `ft.ir.FLOAT_DTYPES` | Tuple of supported IR data type encodings (internally normalized to `"i32"`, `"f32"`, etc.) and set of differentiable floating-point encodings. |
| `ft.ir.Module()` | Creates an empty Fuga IR module container. |
| `module.graph(name, *, inputs, results=None)` | Creates and returns a `Function` builder within the module. `inputs` is an ordered dict (variable name $\to$ `ft.ir.Type`); `results` optionally specifies the list of return types. |
| `graph.inputs[name]` | Retrieves the named input SSA `Value`. |
| `graph.constant(value, *, dtype=..., id=None)` | Constructs a scalar constant SSA value. |
| `graph.splat(value, shape, *, id=None)` | Broadcasts a scalar into a tensor SSA value of the given shape. |
| `graph.add(a, b)` / `sub` / `mul` / `div` | Basic element-wise binary operations for SSA values of matching shape and type. |
| `graph.exp(x)` / `log` / `sqrt` | Elementary mathematical functions. |
| `graph.cast(value, *, dtype, id=None)` | Explicit type casting. |
| `graph.reshape(value, shape, *, id=None)` | Reshapes a tensor while preserving the total number of elements and their ordering. |
| `graph.transpose(value, permutation, *, id=None)` | Permutes tensor dimensions. |
| `graph.broadcast_in_dim(x, shape, dimensions)` | Explicit dimension alignment and broadcast expansion. |
| `graph.reduce_sum(x, axes)` / `reduce_max` | Reduces along specified axes. Full reduction produces a `ScalarType`. |
| `graph.matmul(a, b, transpose_a=False, transpose_b=False)` | Matrix multiplication for 2D and higher dimensions. |
| `graph.stop_gradient(value, *, id=None)` | Truncates gradient propagation; maps directly to the forward input during lowering to ASK. |
| `graph.return_(*values)` | Terminates function construction and specifies the list of return SSA values. |
| `module.verify()` | Structural validity verification; raises `ValueError` if invalid, returns `self` on success. |
| `module.to_text()` | Emits standard Fuga IR text. |

### Graph Construction Example

```python
import fugatensor as ft

# 1. Create module and graph function
module = ft.ir.Module()
graph = module.graph("elementwise_square", inputs={"x": ft.ir.TensorType((3,), ft.fp32)})
x = graph.inputs["x"]

# 2. Build SSA operations
sq = graph.mul(x, x)
graph.return_(sq)

# 3. Verify and print IR text
module.verify()
print(module.to_text())
```

**Output:**

```text
fuga.module {
  fuga.func @elementwise_square(%x: tensor<3xf32>) -> (tensor<3xf32>) {
    %v0_0 = fuga.mul(%x, %x) -> (tensor<3xf32>)
    fuga.return(%v0_0)
  }
}
```

---

<a id="control-flow-and-external"></a>

## Control Flow and External Operators

Fuga IR natively supports conditional branching and loop regions:

| API | Region Signature & Description |
| --- | --- |
| `graph.if_(condition, operands, then_fn, else_fn, *, id=None)` | Conditional branching region. Both `then_fn` and `else_fn` have signature `fn(builder, *operands)`; return types from both branches must match strictly. |
| `graph.for_(lower, upper, step, initial_values, body_fn, *, id=None)` | Counted loop. `body_fn(builder, index, *carried)`; `index` is an `int64` SSA scalar; the loop body returns the updated sequence of `carried` SSA values. |
| `graph.while_(initial_values, condition_fn, body_fn, *, id=None)` | Conditional loop. `condition_fn(builder, *carried)` returns a boolean SSA scalar; `body_fn` returns the updated loop carried values. |
| `ft.ir.ExternalSpec(...)` | External kernel specification metadata declaring function name, target platform, input/output constraints, and pass-by-value scalar arguments. |
| `graph.external(*operands, spec=..., result_types=None, id=None)` | Inserts an external node (such as a Triton kernel or cuBLAS native call). |

> [!NOTE]
> When lowering to ASK for execution, the static simplifier `ft.ir.specialize` unrolls static conditional branches and fixed-length loops ahead of time. Dynamic `while` loops require full loop execution support from the underlying runtime.

---

<a id="ir-transforms"></a>

## Graph Transformations and Automatic Differentiation

Fuga IR provides pure algebraic graph analysis and differentiation transformations, where inputs and outputs are always closed Fuga IR modules:

| Transform Function | Description |
| --- | --- |
| `ft.ir.specialize(module, *, max_unroll=10000, expand_calls=True)` | Constant folding and static control flow unrolling, flattening the module into a single flat graph. |
| `ft.ir.prune(module)` | Dead code elimination and unused node pruning for flat graphs, returning a minimized module. |
| `ft.ir.project_outputs(module, indices, *, name_suffix="", max_unroll=10000)` | Retains only public outputs at specified indices, pruning all other dependency paths. |
| `ft.ir.analyze_function(function, selected=())` | Static liveness and topological sort analysis, returning topological operator ordering and reverse dependency indices (accelerated via C++ `_native`). |
| `ft.ir.jvp(module, *, argnums=None, max_unroll=10000)` | Forward-mode tangent automatic differentiation. Inputs include original arguments and tangents; outputs include primal results and tangent derivatives. |
| `ft.ir.vjp(module, *, argnums=None, max_unroll=10000)` | Reverse-mode cotangent automatic differentiation. Inputs include original arguments and output cotangent seeds; outputs gradients with respect to target inputs. |

### VJP Differentiation Transformation Example

```python
import fugatensor as ft

# Define forward module
module = ft.ir.Module()
graph = module.graph("square", inputs={"x": ft.ir.TensorType((3,), ft.fp32)})
x = graph.inputs["x"]
graph.return_(graph.mul(x, x))

# Perform reverse-mode algebraic differentiation
vjp_module = ft.ir.vjp(module)
print(vjp_module.to_text())
```

**Output:**

```text
fuga.module {
  fuga.func @square_vjp(%x: tensor<3xf32>, %seed0: tensor<3xf32>) -> (tensor<3xf32>) {
    %v0_0 = fuga.mul(%seed0, %x) -> (tensor<3xf32>)
    %v1_0 = fuga.mul(%seed0, %x) -> (tensor<3xf32>)
    %v2_0 = fuga.add(%v0_0, %v1_0) -> (tensor<3xf32>)
    fuga.return(%v2_0)
  }
}
```

---

<a id="lowering-to-ask"></a>

## Lowering to ASK

```python
lowered = ft.lower_to_ask(module, *, target=None, max_unroll=10000, source_map=None)
```

- **Parameters**:
  - `module` (`ft.ir.Module`): Valid and initialized Fuga IR module.
  - `target` (`dict | None`): Target compilation description, e.g. `{"backend": "host"}` or `{"backend": "cuda", "arch": 86}`.
  - `max_unroll` (`int`): Upper limit for static loop unrolling depth; default is 10000.
  - `source_map` (`dict | None`): Optional dictionary populated with node-by-node provenance mapping from Fuga IR Values to ASK SSA identifiers.
- **Returns**:
  - `LoweringResult`: Contains ASK MLIR text and metadata. Call `.to_mlir()` to export standard MLIR source code for submission to the ASK C++ Compiler.

### Lowering Example

```python
import fugatensor as ft

module = ft.ir.Module()
graph = module.graph("square", inputs={"x": ft.ir.TensorType((3,), ft.fp32)})
x = graph.inputs["x"]
graph.return_(graph.mul(x, x))

lowered = ft.lower_to_ask(module, target={"backend": "host"})
print(lowered.to_mlir())
```

**Output:**

```text
module attributes {ask.schema_version = 1 : i64} {
  func.func @square(%x: tensor<3xf32> {ask.value_id = "square.arg0"}) -> (tensor<3xf32>) {
    %ask_v0 = "ask.mul"(%x, %x) {ask.id = "square.op0", ask.value_id = "square.op0.result0"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    func.return %ask_v0 : tensor<3xf32>
  }
}
```

# Control Flow and Composable Transformations

Back to IR Overview: [Structured Control Flow and Composable Differentiation](README.md#ir-control-transforms) · Related topics: [Object Model and Verification](model.md) · [Lowering to ASK](lowering.md) · [Automatic Differentiation and Custom Operators](../architecture/autograd.md)

<a id="structured-control-flow"></a>

## 1. Two Control Flow Paths

| Mechanism | Generation Method | Compilation & Differentiation Behavior |
| --- | --- | --- |
| **IR-Level Structured Regions** (`fuga.if` / `fuga.for` / `fuga.while`) | Constructed via `Function` / `BlockBuilder` | Saved as nested `Region`s. `if` with constant conditions and `for` with static integer ranges can be simplified and differentiated, then lowered to single-block ASK graphs via [`lower_to_ask`](lowering.md#static-simplification). Data-dependent IR `if` and all IR `while` currently cannot be lowered. |
| **Python Scalar Condition Boundaries** ([On-Demand Execution](../architecture/tensor_model.md#graph-break)) | Executing `if cond:` / `while cond:` on 0-D Tensors in standard Python | Compiles and executes the dependency graph upon reading the condition truth value; Python continues building Tensor expressions according to actual branches and iteration counts; expression provenance across stages remains connected. Isolated tracing of custom `ft.fun` does not support this type of data-dependent control flow. |

<a id="primitive-registry"></a>

## 2. Primitive Rules Table (`ir/primitives.py`)

Basic operators declare three rules (`PrimitiveRule`) in the `PRIMITIVE_RULES` registry:

1. **`abstract_eval(input_types, attrs)`**: Infers output type tuple from input types and semantic attributes, validating arguments; supports scalars and tensors. Tracing, verification, and transformation builders share identical inference logic.
2. **`jvp(builder, operation, primals, tangents)`**: Forward tangent propagation rule. Receives primals `primals` and tangents `tangents` (`None` for non-differentiable ports), generating tangent instructions:
   - `constant` → `None`;
   - `splat(x)` → Generates `splat(dx, shape)` if `dx` exists;
   - `add(a, b)` → `da + db` (returns the other side directly when one side is `None`);
   - `mul(a, b)` → `da * b + a * db`;
   - `cast(x, to=T)` → Generates `cast(dx, to=T)` when both original value and target are differentiable floats (`FLOAT_DTYPES`); raises an error if the differentiation path crosses non-floating-point types.
   - `reshape(x, shape)` → Changes `dx` to output shape.
   - `stop_gradient(x)` → Tangents halt propagation; output tangent is zero.
3. **`vjp(builder, operation, primals, cotangent, active)`**: Reverse cotangent propagation rule, where `active` indicates inputs requiring gradients:
   - `add(a, b)` → `(dout, dout)`;
   - `mul(a, b)` → `(dout * b, dout * a)`;
   - `splat(x)` → Constant coefficient branches skip differentiation; cotangent of active scalars is fully reduced back to ScalarType via `reduce_sum`;
   - `cast(x, to=T)` → Generates `cast(dout, to=x.dtype)` between differentiable floats; raises an error if the differentiation path crosses integers or booleans;
   - `reshape(x, shape)` → Restores `dout` to `x.shape`.
   - `stop_gradient(x)` → Does not propagate cotangents to inputs.

`ir/tensor_rules.py` provides typing and derivative rules for elementwise math (`sin`/`cos`/`tanh`/`erf`/`abs`/`pow`/`floordiv`/`mod`/`maximum`/`minimum`/`floor`/`ceil`/`round`), bitwise/shift ops (`bitwise_*`, `shift_*`), geometry/indexing (`slice`, `pad`, `concatenate`, `gather`, `scatter_add`, `iota`), broadcast/transpose, reductions (`reduce_sum`/`max`/`min`/`prod`/`any`/`all`, `argmax`/`argmin`), and matrix multiplication, uniformly registered into `PRIMITIVE_RULES`:

- **Mixed / Non-Differentiable Inputs**: Operations accepting both integer and float types (or index arguments alongside floating data, e.g. `gather`, `scatter_add`) set `floating_values=False` in `PrimitiveRule`, allowing active value ports to differentiate while severing index or integer/bool edges (`argmax`, `argmin`, `floor`, `ceil`, `round`, `bitwise_*`, `shift_*`, `reduce_any/all`, `iota`).
- **Extrema & Product AD**: `reduce_max`, `reduce_min`, `maximum`, and `minimum` distribute gradients evenly across ties and preserve `NaN`; `reduce_prod` uses zero-safe exclusive products (`0/1/>=2` zero counts) so first- and higher-order derivatives remain NaN-free at zeros.
- **Geometry & Indexing Adjoints**: `slice` backward pads with negative-step reversal (mapping `-1` stop bounds properly); `pad` differentiates wrt both `value` (sliced back) and `pad_value` (masked `reduce_sum`); `concatenate` slices cotangents per operand; `gather` VJP is `scatter_add` onto zeros, while `gather` JVP gathers tangents; `scatter_add` VJP returns cotangent to `base` and gathers from `indices` for `updates` (with matching JVP).

Full reduction produces ScalarType; public Tensor APIs convert results to 0-D Tensors using existing splat. VJP of splat (and 0-D `pad` values) restores exact primal types via `_restore`, maintaining differentiation closure between `ScalarType` and `TensorType(())`. `_known_scalar` inside `specialize` folds compile-time integer, bitwise, shift, `abs`, `maximum`/`minimum`, and `compare` scalar chains for static `fuga.if` selection.

Instructions generated by these rules still belong to the same set of basic primitives, and derivative `Module`s can continue to be passed to `jvp`, `vjp`, `prune`, or `lower_to_ask`. Numerical dependencies and differentiation dependencies of `stop_gradient` are handled separately: values are still computed following standard dependencies, while active differentiation paths are severed here; thus, external operators connected to derivative results solely through such nodes do not require JVP/VJP rules.

<a id="jvp-and-vjp"></a>

## 3. `ir.jvp` and `ir.vjp` Graph-Level Transformations (`ir/jvp.py` / `ir/differentiate.py`)

Both transformations return a new `Module`:

- **Forward-mode `jvp(module, argnums=...)`**:
  New function inputs are `(*primals, *selected_tangents)`, and outputs are `(*primal_outputs, *output_tangents)`. Static regions are first simplified, then active subgraphs are identified using C++ [`analyze_function`](../architecture/native.md#native-analyze), generating primals and tangents in topological order, and finally calling `prune`. Standard `fuga.call` automatically JVPs via forward subgraphs; invocations providing explicit VJPs currently reject active JVPs. Active bare external operators lacking JVP rules raise an error.
- **Reverse-mode `vjp(module, argnums=...)`**:
  Generates a new function whose inputs are `(*primals, *output_cotangent_seeds)` and outputs correspond to gradients of differentiable inputs selected by `argnums`:
  1. Select static `if` branches, unroll static `for` loops, converting intermediate values across iterations into SSA values.
  2. Call C++ `analyze_function` to obtain the reverse traversal order; the source graph must have no side effects, and any node with non-empty `effects` causes an error.
  3. Replay the forward pass, accumulating cotangents in reverse topological order. Gradients of shared values are accumulated via `fuga.add`; `fuga.call` uses forward primitive rules or explicit VJP, with rules receiving primals, public outputs, and cotangents; differentiable inputs without gradient contributions are padded with zeros.
  4. Call `prune` to eliminate dead nodes in the backward graph. For example, the derivative graph of `f(x) = 3 * x` retains only constants and gradient seeds.

`tests/test_ir_autodiff.py` validates consistency of compilation and execution results on ASK between `vjp(jvp(f))` and `jvp(vjp(f))`.

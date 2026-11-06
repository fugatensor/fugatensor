# Lowering to ASK

Back to IR Overview: [Lowering to ASK IR and Cross-Layer Provenance](README.md#ir-lowering-boundary) · Related topics: [Object Model and Verification](model.md) · [Control Flow and Composable Transformations](transforms.md) · [ASK Bridge and Multi-Backend Execution](../architecture/backend.md) · [ASK IR and Numerical Semantics](../../../../ask/docs/en/ir.md)

[`lower_to_ask`](../../../src/fugatensor/lowering.py) receives a Fuga `Module` and target device, returning an internal `_ask.Module`. Its `to_mlir()` method produces text submitted to ASK C++, and the `source_map` argument can receive value-by-value mappings.

The lowered graph is submitted to ASK by the internal Compiler bridge layer. Preparation results separately store `ir`, `shape_snapshot`, `external_requests`, and `diagnostics`; partition results further provide `execution_plan` and `step_sources`, while codegen results provide `abi` or `cuda_artifact` depending on the target. Graph translation, compilation contracts, and diagnostics are passed via their respective fields; for field usages, see [Compilation Stage Results](../architecture/backend.md#compiler-results).

API: [IR and Lowering](../API/ir.md)

<a id="static-simplification"></a>

## 1. Static Control Flow Simplification and Dynamic Region Interception

Prior to translating into a single-basic-block ASK graph, the lowerer calls `ir.specialize(module, max_unroll=10000)` to expand static control flow on a new module:

- **Inline `fuga.call`**: Expands invoked forward subgraphs into the main block, preserving corresponding result mappings; dead results are eliminated by subsequent pruning.
- **Static `fuga.if` Selection**: Conditions must resolve to compile-time `bool`s; only the selected branch is cloned.
- **Bounded `fuga.for` Unrolling**: Unrolls loops according to `lower`, `upper`, `step`, with all unrolled iterations sharing the `max_unroll` budget; exceeding the budget raises an error, and cloned instructions preserve `origin_id` while acquiring non-conflicting local IDs.
- **Dynamic Region Rejection**: Data-dependent `fuga.if` and all `fuga.while` raise `StaticControlFlowError`. Standard Python dynamic branching executes piecewise by reading [0-D scalars](../architecture/tensor_model.md#graph-break).

`specialize` calls `Module.verify()` before and after expansion. Lowering requires all shapes to be fully concrete and operations implemented on the chosen backend, raising an error otherwise.

<a id="type-and-op-mapping"></a>

## 2. Type and Basic Operation Mapping

The internal module `fugatensor._ask` generates ASK MLIR according to the following rules:

| Fuga IR | ASK Description or Operation | Notes |
| --- | --- | --- |
| `ScalarType(dtype)` | Internal `ScalarType(dtype)` | Corresponds to MLIR scalar types; `bool` encodes as `i1`. |
| `TensorType(shape, dtype)` | Internal `TensorType(shape, dtype)` | Preserves static shapes; 0-D tensors `()` and scalar types are represented separately, and empty tensors preserve dimensions. |
| `fuga.constant` | `graph.constant(value, dtype=...)` | Encodes floating-point constants according to [single-step RNE rules](../../../../ask/docs/en/ir.md#types-and-precision). |
| `fuga.splat` | `graph.splat(scalar, shape)` | Splats a scalar into a tensor of specified shape, including 0-D tensors. |
| `fuga.add` / `fuga.mul` | `graph.add(lhs, rhs)` / `graph.mul(lhs, rhs)` | Requires identical shapes and types. |
| `div`, `exp/log/sqrt`, `compare/select`, `broadcast_in_dim/transpose`, `reduce_sum/reduce_max`, `matmul` | Identically named `ask` operations | Obtains result types from unified primitive type rules, with bridge layer preserving static attributes. |
| `fuga.cast` | `graph.cast(operand, dtype=...)` | Generates `ask.cast`, preserving original scalar or tensor shape. |
| `fuga.reshape` | `graph.reshape(operand, shape)` | Generates `ask.reshape` and target `shape`; preserves dtype, element count, and contiguous element order. |
| `fuga.stop_gradient` | Input ASK value | Differentiation boundaries are handled in the Fuga IR phase; numerical lowering directly reuses input mappings, preserving original values. |
| `fuga.return` | `graph.return_(...)` → `func.return` | Supports multiple outputs as well as heterogeneous shapes and types. |

<a id="external-and-sourcemap"></a>

## 3. External Implementation Binding and Cross-Layer Provenance

The `ExternalSpec` of `fuga.external` is converted into `ask.semantics`, `ask.implementation`, and an `ask.external` referencing them; see [ASK External Declarations](../../../../ask/docs/en/ir.md#external-declaration) for fields.

- Targets must be compatible with implementation declarations, and CUDA implementations require concrete integer architecture numbers. The selected device supplies Triton's `arch` here.
- `same_as` shape rules are preserved, and Fuga's `explicit` rule converts to ASK's `fixed`; each output type is determined by verified Fuga types.
- Values before and after reshape retain their respective real shapes, and port shapes in external requests match Tensors actually received by that node.
- Tensor arguments bind via `input:N` / `output:N`, standard scalars preserve declared launch values, and compile-time configurations retain recursive JSON. External operations with non-empty `effects` currently cannot be lowered for execution.
- Each external node is retained as an independent kernel, while owned regions are left to ASK for optimization. Worker invocations and device memory allocation occur in subsequent compilation and execution phases.

`source_map` has the structure `{"schema_version": 1, "values": [...]}`. Each value records function name, Fuga name and `value_id`, `origin_value_id`, ASK name and `value_id`. Multi-function inputs use function name prefixes to prevent identity collisions within ASK modules.

`Engine` supplements `input_bindings`, recording indices before and after input pruning, then writes the mapping to `fuga-to-ask.json`, enabling correlation queries between traced values and provenance records in ASK diagnostics.

# Fuga IR Overview

Navigation: [Documentation Index](../README.md) · API Usage: [IR and Lowering API](../API/ir.md) · System Architecture: [Architecture Overview](../architecture/README.md) · Underlying Engine: [ASK IR and Numerical Semantics](../../../../ask/docs/en/ir.md)

Fuga IR (`src/fugatensor/ir/`) represents tensor programs using types, SSA values, operations, and nested regions. Pending execution graphs and isolated trace nodes of Tensors are stored in native arenas, from which these IR objects are exported upon a compilation miss or when tracing concludes, for use in analysis, differentiation, and lowering.

<a id="ir-design"></a>

## 1. Design Responsibilities and Device Independence

Fuga IR preserves three categories of information prior to lowering to the [ASK Static Computation Graph](../../../../ask/docs/en/ir.md):

- Structured control flow: `fuga.if`, `fuga.for`, `fuga.while` are represented using nested `Region`s and block arguments.
- Custom differentiation: [`ft.fun`](../architecture/autograd.md#custom-fun) is traced into `fuga.call`; the VJP stage uses primitive rules from the forward subgraph or explicit `vjp` rules, with primals and forward public outputs passed as SSA values.
- Device information: Types and SSA values store shapes and dtypes, while specific GPU architectures for external Triton operators are bound at lowering time.

```text
Tensor native node pool (cpp/lazy.cpp) / Isolated trace node pool (cpp/trace.cpp)
  -> Fuga IR Module (device-independent, with Regions and fuga.call)
  -> Framework-level transforms: C++ liveness analysis & pruning (transforms.py) / JVP & VJP differentiation (jvp.py, differentiate.py)
  -> Lowering bridge (lowering.py): static if selection, bounded for unrolling, arch binding, and fuga-to-ask.json provenance
  -> ASK IR Module -> ASK optimization, code generation, and resident execution
```

<a id="ir-object-model"></a>

## 2. Object Model and Verification

Fuga IR consists of `ScalarType`, `TensorType`, `Value`, `Operation`, `Block`, `Region`, `Function`, and `Module`. `Module.verify()` checks dominance relationships, types, and region terminators. Verified cached templates are frozen; the engine queries cache using templates and connection topologies, composing graphs upon misses and saving debug text via `module.to_text()`.

Engineering details: [Object Model and Verification](model.md) · [Native Graph Analysis and C++ Boundary](../architecture/native.md)

<a id="ir-control-transforms"></a>

## 3. Structured Control Flow and Composable Differentiation (JVP / VJP)

`ir/primitives.py` centrally defines abstract typing rules (`abstract_eval`), forward tangent rules (`jvp`), and reverse cotangent rules (`vjp`) for basic primitives:

- Arithmetic, type casting, and shape transformations use `constant`, `splat`, `add`, `mul`, `div`, `cast`, `reshape`, `exp/log/sqrt`, `compare/select`, `broadcast_in_dim/transpose`, `reduce_sum/reduce_max`, and `matmul`; `stop_gradient` retains numerical values while halting differentiation propagation. `subtract`, `negative`, and `square` are composed and expanded from basic primitives during tracing;
- `ir.jvp(module)` and `ir.vjp(module)` generate new Fuga IR `Module`s. Within the scope of existing rules, `vjp(vjp(module))`, `jvp(vjp(module))`, and `vjp(jvp(module))` can be nested. `ft.grad` uses VJP to generate new derivative expressions, which retain expression provenance after evaluation.

Engineering details: [Control Flow and Composable Transformations](transforms.md) · [Functional Automatic Differentiation and Higher-Order Derivatives](../architecture/autograd.md#higher-order)

<a id="ir-lowering-boundary"></a>

## 4. Lowering to ASK IR and Cross-Layer Provenance

`lower_to_ask` selects compile-time known `fuga.if` branches, unrolls bounded static `fuga.for` loops, and maps Fuga `Value`s to ASK `Value`s, producing `fuga-to-ask.json`. Data-dependent IR `if` and all IR `while` operations currently cannot be lowered and raise an exception.

Engineering details: [Lowering to ASK](lowering.md) · [ASK Bridge and Multi-Backend Execution](../architecture/backend.md)

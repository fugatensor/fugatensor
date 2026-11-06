# Native Graph Analysis and C++ Boundary

Back to Architecture Overview: [Independent Fuga IR and Native C++ Graph Analysis](README.md#arch-ir-native) · Related topics: [Tensor Model and Tracing Mechanism](tensor_model.md) · [Automatic Differentiation and Custom Operators](autograd.md) · [Fuga IR Object Model](../ir/model.md)

Both automatic differentiation and input pruning require computing reachable scopes, active nodes, and reverse traversal order in SSA graphs. On-demand evaluation also collects dependencies of pending Tensors and builds compilation cache keys. These analyses are performed by the C++17 extension `fugatensor._native`, while the Python layer builds and transforms Fuga IR.

<a id="python-cpp-boundary"></a>

## 1. Module Responsibilities of Python and C++

| Layer | Code Location | Responsibilities |
| --- | --- | --- |
| **Python Framework & IR Layer** | `src/fugatensor/*.py`, `src/fugatensor/ir/*.py` | `Tensor` / `ft.grad` / `ft.evaluate` / `ft.fun` / `ft.triton` interfaces, automatic Tensor graph construction, on-demand execution and isolated tracing, Fuga IR object structures, JVP/VJP differentiation rule expansion, lowering to ASK IR, and `Engine` backend orchestration. |
| **FugaTensor C++ Extension** | `cpp/graph.*`, `cpp/lazy.*`, `cpp/trace.cpp`, `cpp/primitive.cpp`, `cpp/compiler.cpp`, `cpp/ask_runtime.cpp` (`fugatensor._native`) | SSA liveness and reverse topological analysis; primitive argument binding, arena storage for trace nodes and pending nodes; output requirement propagation, dependency collection, definition invalidation validation, and structural key generation; binding ASK named compilation results and `PreparedProgram`, dispatching execution via native plans. |
| **ASK C++ Compilation & Runtime** | Shared libraries `ASK::Compiler` / `ASK::Runtime` from ASK C++ SDK | `ask` dialect validation and optimization, CPU / owned CUDA native LLVM IR code generation, graph partitioning, native execution plans, buffer and event reuse, CUDA Graph replay; FugaTensor's private `_ask/` handles graph translation, worker invocations, artifact validation, and disk caching. |

`ir/transforms.py` calls `_native` for graph analysis; compilation and execution bindings also invoke ASK shared libraries via this extension. For the build workflow, see [Installation Guide](../install.md#build-fugatensor).

`cpp/compiler.cpp` binds ASK `PrepareResult`, `PartitionResult`, `CodegenResult`; the internal Python Compiler decodes versioned JSON fields into named results for the corresponding stage, with compilation inputs sourced from independent contract fields. Full CLI reports undergo the same decomposition at the bridge entry point. For field usages, see [Compilation Stage Results](backend.md#compiler-results). Execution binding is handled by `cpp/ask_runtime.cpp` invoking `PreparedProgram.run` / `submit`, with buffers, scheduling, and in-flight resources managed by ASK.

CUDA port descriptions are stored as native `CUDAProgramPorts` during preparation. Each submission freezes the input sequence, checks port types, extracts device buffers, and wraps outputs in a single native entry point; input aliases and duplicate outputs reuse the same storage handle. Python retains the session lock, shutdown state, and ownership management of external memory; dynamic external descriptions and producer streams are validated before submission.

<a id="native-analyze"></a>

## 2. Reachability and Reverse Boundary Analysis (`cpp/graph.cpp`)

`fugatensor._native.analyze(...)` receives the function's input count, topologically ordered operands and result counts, side-effect flags for each node, output values, and indices of inputs requiring differentiation.

The C++ layer validates value numbering and topological order, with `bindings.cpp` simultaneously rejecting Python `bool`s masquerading as integer IDs. It then computes three sets of results in linear time $O(V + E)$:

1. `active_values`: Starting from the selected inputs, marks values affected along `operand → result`.
2. `reverse_operations`: Backtracks from affected outputs to generate the reverse topological order of required nodes. Nodes that do not flow into target outputs or do not depend on selected inputs are skipped during differentiation; multiple-return nodes appear only once.
3. `live_operations`: Backtracks dependencies from outputs and side-effecting nodes to generate the forward keep-alive sequence for DCE. Active numerical coefficients and side-effecting nodes can be retained here without participating in reverse differentiation.

<a id="transforms-reuse"></a>

## 3. Graph Pruning and Output Projection (`ir/transforms.py`)

`ir/transforms.py` encapsulates three shared functions based on `_native.analyze`:

- `analyze_function(function, selected=())`: Encodes SSA edges of a flat function for `_native.analyze`, returning the operation sequence, value list ordered by ID, and analysis results.
- `prune(module)`: Reconstructs functions according to `live_operations` from the analysis results, eliminating side-effect-free dead nodes.
- `project_outputs(module, indices, ...)`: First simplifies static control flow, then selects outputs and calls `prune` to eliminate unrelated branches.

Graphs containing control flow regions are first expanded via `specialize` before performing flat graph analysis.

`ir/differentiate.py`, `ir/jvp.py`, `autograd.py`, and [input pruning in `engine.py`](backend.md#engine-lifecycle) share these functions.

<a id="lazy-collection"></a>

## 4. Dependency Collection for Pending Tensors (`cpp/lazy.cpp`)

`_lazy.materialize` passes the root Tensors of the current request all at once to `_native.collect_pending`. The native code first merges the requirements of multiple-output expressions, then determines active inputs, the expression topological order, and shared leaf IDs. Materialized Tensors serve as numerical inputs; aliases are resolved along their original provenance. Graphs producing only constants select storage carriers required by the ABI from the original input chain.

Each Engine holds a `RuntimeArena`. Value records, expression records, and input edges are stored in contiguous arrays that grow on demand and reuse capacity, interconnected by `uint32_t` IDs; shape types, templates, definition dependencies, and storage resources reside in separate tables. The Python `Tensor` class retains operator methods, while the native thin shell holds the Engine and value ID. Expression provenance is also stored in native nodes, with autograd reading views on demand.

When a node is created, an incremental digest is computed from template fingerprints and ordered input fingerprints. `PlanKey.__hash__` reads the saved digest; key equality comparison normalizes inputs, connectivity topology, output ordering, and definition identities. Materialization that alters numerical input boundaries increments the arena version, refreshing corresponding digests during collection. Multi-output projections establish exact keys based on actual current requirements.

Template projections and associated definitions are prepared and cached by Python upon first use. Hot collection traverses native indices, retrieves the current storage inputs, validates live definitions, and returns a tuple of keep-alive storages alongside the structural key. Backend execution receives these storages directly and publishes returned storages back to the original Tensors. Only on cache misses does `describe_pending` generate leaf Tensors and expression views for Python to stitch Fuga IR.

<a id="native-tracing"></a>

## 5. Native Tracing and Argument Binding

`cpp/primitive.cpp` encodes exact argument signatures for built-in primitives: Tensors use type IDs stable within the same Engine, static arguments are encoded by type and value, and floating-point values preserve bit patterns. Upon a template hit, it directly invokes the native pending-node binding entry point. Standard Python functions and custom function callbacks still execute per call; definition versions of primitive templates are validated before reuse.

`cpp/trace.cpp` uses pre-allocated contiguous containers to store trace nodes, value IDs, input edges, and attribute atoms. Node records are PODs, the type table holds shared immutable Fuga types, and `CallSpec` / `ExternalSpec` reside in separate Python object tables. `TraceTensor` holds only the trace context and value ID.

The trace builder reuses primitive construction methods and abstract typing rules from `ir/core.py`, writing inferred results into the arena. `TraceContext.finish` exports Python IR once and verifies it; the export entry point for custom VJPs appends newly added operations to the existing differentiation graph. Public Python IR, autograd rules, and ASK lowering continue sharing identical semantics. For invocation patterns, see [Internal Tracing](tensor_model.md#trace-context).

<a id="gil-and-concurrency"></a>

## 6. Concurrency and Ownership

`analyze(const Graph&)` in `cpp/graph.cpp` uses local state and read-only inputs for each invocation. After parameter conversion in the binding layer, the GIL is released via `nb::gil_scoped_release`, then re-acquired after analysis to return results. For multithreading coverage, see `tests/test_native_graph.py`.

The Engine's evaluation lock protects dependency collection, compilation caching, and result publication. Native nodes are kept alive by reference counts from thin shells, input edges, and differentiation views; zero-reference nodes are released iteratively, returning slots to the pool. Materialized outputs continue retaining differentiation provenance. The template registry uses weak ownership, with active expressions keeping required templates alive; cache keys retain static descriptions required for exact comparison.

The native extension exposes Python references for templates, definitions, and Engines to CPython's GC, enabling the reclamation of ownership cycles formed by user callbacks. The node pool itself is reclaimed via native reference counts, with capacity reused based on the peak concurrent live footprint.

Primitive template caches are grouped by weak references to Engines, with `Engine.close` and primitive `clear_cache` cleaning up corresponding records. Native trace handles, the arena's Python object table, and template caches register GC traversal and clearing, integrating cross-language references into Python cyclic garbage collection.

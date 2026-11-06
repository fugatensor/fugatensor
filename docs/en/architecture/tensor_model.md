# Tensor Model and Tracing Mechanism

Back to Architecture Overview: [Tensor Model, Automatic Graph Construction, and Execution Boundary](README.md#arch-tensor-trace) · Related topics: [Native Graph Analysis and C++ Boundary](native.md) · [Automatic Differentiation and Custom Operators](autograd.md) · [Fuga IR Object Model](../ir/model.md)

API: [Tensor and Data Exchange](../API/tensor.md) · [Basic Operations](../API/ops.md) · [Device and Execution](../API/engine.md)

<a id="tensor-model"></a>

## 1. `Tensor` Storage and Pending Execution State (`tensor.py` / `ops.py`)

`ft.tensor(data, *, dtype=ft.fp32, engine=None, device=None)` infers shapes from regular nested data, handling rectangular verification and byte encoding/decoding; size-based creation uses [`zeros`, `ones`, `empty`, `full`](../API/tensor.md#size-factories). CPU inputs for bool, i32, i64, f32, and f16 use compact `ctypes` buffers; CUDA additionally supports bf16, where inputs are directly encoded as bytes and uploaded to device storage managed by the [ASK `ExecutionSession`](../../../../ask/docs/en/runtime.md#execution-session). The six type objects correspond to [ASK's 6 foundational types](../../../../ask/docs/en/ir.md#types-and-precision); floating-point expressions can be differentiated with respect to specified inputs via `ft.grad`.

Tensor operations initially return pending Tensors, recording shape, dtype, Engine, and input dependencies. For example, `y = (x + 1).square()` records addition and square, with subtraction, negation, and square composed from primitive addition and multiplication operators.

Tensor, TraceTensor, and Triton's TensorSpec return `DType` singletons and device objects, while internal IR and backend storage use `f32`, `f16`, etc. The framework reuses default Engines per device.

[`reshape`](../API/tensor.md#reshape) generates lazily executed `fuga.reshape`, preserving input dependencies and expression provenance. Lowered `ask.reshape` and external kernel metadata all use the target shape. JVP changes the tangent to the output shape, while VJP restores the cotangent to the input shape. Output storage is allocated by the execution plan, and the interface does not guarantee shared storage with the input.

`shape`, `dtype`, `device`, and `repr` read metadata only. `.tolist()`, `.storage`, and 0-D scalar truth-value evaluations trigger computation; CUDA `.tolist()` also synchronizes and downloads results. `ft.evaluate(y)` returns the original Tensor; `ft.evaluate(a, b)` merges shared dependencies and dispatches multiple outputs together.

<a id="buffer-io"></a>

When contiguous memory already exists, use `ft.from_buffer(data, dtype=..., shape=..., device=None, copy=True)` or `engine.from_buffer(...)` for bulk import, avoiding elementwise conversion into Python numeric objects. Inputs must implement the buffer protocol and be C-contiguous; typed buffers must match dtype, element width, and native endianness, while `bytes`, `bytearray`, and raw byte views interpret raw bit patterns according to the explicit dtype. Byte counts must match the shape; raw byte imports preserve NaN payloads and negative zeros, and bf16 tensors require CUDA. Generic buffer format validation is located in `_buffers.py`; CPU creates or borrows `HostStorage`, while CUDA directly uploads to device storage after validation.

```python
from array import array
import fugatensor as ft

source = array("f", [1.0, 2.0, 3.0])
x = ft.from_buffer(source, shape=(3,), dtype=ft.fp32, device="cpu")
raw = (x + 1).to_buffer()  # Read-only memoryview containing contiguous raw bytes
```

By default, both import and export copy data. CPU `from_buffer(..., copy=False)` borrows writable and aligned storage, keeping the original object alive; subsequent writes to the original object modify the Tensor contents, and callers must prevent concurrent modification during execution. CPU `.to_buffer(copy=False)` returns a read-only view of the keep-alive storage; read-only restrictions apply to that view and cannot prevent other owners from mutating the same memory. CUDA import and export require host-device transfers and only accept `copy=True`; `.to_buffer()` will complete the pending graph and await download first.

<a id="jit-specialization"></a>

## 2. Automatic Graph Fusion and Engine Caching (`_lazy.py` / `engine.py`)

Standard Python functions execute normally on every call. Function call boundaries do not trigger tensor execution, and expressions generated across functions can still be merged:

```python
def shift(x):
    return x + 1

def scale(x):
    return shift(x) * 2

y = scale(x)       # Records add and mul
ft.evaluate(y)     # JIT compiles and executes after merging dependency graph
```

After internal tracing completes, expression graphs freeze into immutable templates, and caches index numerical projections obtained via output selection. Expression provenance preserves `fuga.call` boundaries for custom differentiation; numerical projections inline subfunctions and prune unused inputs.

Basic primitives fix positional arguments and default value layouts upon registration. Full positional calls directly enter C++ argument binding, while keyword calls first apply standard Python signature rules. The native template cache encodes Tensor type IDs and static values in argument order; upon a hit, pending nodes are appended directly to the Engine's arena. Engine ownership is checked at invocation boundaries, and template input types are matched against exact signatures. Caches use weak references to distinguish Engines, freeing corresponding records when an Engine closes or primitive caches are cleared.

Range checking for reduction axes, negative axis normalization, and automatic broadcasting dimensions are handled during initial template tracing. Subsequent calls look up templates by Tensor type and static arguments; equivalent axis or broadcast expressions reuse identical execution programs in the numerical graph cache.

Custom functions and external operator callbacks execute on every invocation. Identical trace results reuse templates, while changes in configuration or function definitions yield new templates.

At execution time, the [C++ dependency collector](native.md#lazy-collection) backtracks from outputs to inputs with existing storage, composing structural keys from template projections, expression connectivity, shared leaves, output order, and associated definition identities. Multi-output requirement merging, topological traversal, definition aggregation, and key construction are performed in a single native call. Keys incorporate input shapes, types, and constants in the graph, leaving runtime element values and memory addresses to be bound at execution time. On cache misses, Fuga IR is stitched, text is saved, and compilation occurs; on hits, current inputs are bound and executed directly. Every invocation validates function definitions in the active graph.

Storage objects are passed directly between dependency collection, backend dispatch, and result publication. Input pruning is fixed during compilation preparation; programs retaining all inputs directly use backend execution entry points. Upon successful backend dispatch, returned storages are written to requested Tensors while expression provenance remains connected, enabling subsequent `ft.grad` to trace back derivatives. Subsequent reads reuse results; intermediate Tensors retain storage on demand. For caching and diagnostics interfaces, see [ASK Bridge and Multi-Backend Execution](backend.md#cache-hierarchy).

<a id="trace-context"></a>

## 3. Internal Tracing and Custom Function Isolation (`_jit.py` / `tracing.py`)

Built-in operators and `ft.fun` use `TraceContext` / `TraceTensor` to construct Fuga IR. `forward(self, ...)` of custom functions traces into forward subfunctions; optional `vjp(self, primals, outputs, cotangents)` constructs cotangent expressions during differentiation, while standard forwards use primitive differentiation rules.

`TraceTensor` is a native handle holding the trace context and a 32-bit value ID. Nodes, edges, and attributes are stored in the C++ `TraceArena`, with type tables sharing immutable Fuga type objects. Shape inference and typing rules reuse Python IR's abstract evaluation functions; upon trace completion, a Python `Module` is exported once for validation, autograd, and lowering. Custom VJPs import existing SSA values as arena inputs, appending new nodes all at once to the derivative graph upon return. For implementation boundaries, see [Native Tracing and Argument Binding](native.md#native-tracing).

`detach()` and `ft.stop_gradient(...)` on traced Tensors generate `fuga.stop_gradient`, freezing local differentiation dependencies in forward and VJP passes; numerical values continue to be provided by the original expressions. Runtime `ft.grad` is invoked outside isolated tracing.

During tracing, `.shape`, `.dtype`, and `.device` are static metadata accessible for Python branching. `x + 2.5` generates a constant based on `x.dtype`, then fills it to `x.shape` via `fuga.splat`. Binary operations between two Tensors require identical shapes, dtypes, and devices.

Custom functions cannot capture external real Tensors into tracing, nor can they read tensor data or create real Tensors within methods. Runtime tensor state must be passed as explicit arguments. For pure VJP arguments, method signatures, and definition change checks, see [Custom Functions](autograd.md#custom-fun).

<a id="graph-break"></a>

## 4. Python Scalar Conditions Forming Execution Boundaries

Standard Python `if cond:` or `while cond:` requires condition Tensors to be 0-D, raising an error otherwise. Reading the condition executes its dependency graph, after which Python selects the branch or continues the loop; other expressions remain pending.

Branches and loops execute on each run based on actual data, and gradients propagate along the graph formed in that run. Expressions before and after condition evaluations are connected via Tensor expression provenance.

`forward` / `vjp` of `ft.fun` are traced into subfunctions and can only use static metadata to determine branches. Control flow dependent on tensor numerical values should reside in outer Python code.

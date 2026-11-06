# ASK Bridge and Multi-Backend Execution

Back to Architecture Overview: [Lowering to ASK, Multi-Backend Execution, and Multi-Level Caching](README.md#arch-ask-backends) · Related topics: [Tensor Model and Tracing Mechanism](tensor_model.md) · [Lowering Fuga IR to ASK](../ir/lowering.md) · [ASK Resident Runtime and Multi-Level Caching](../../../../ask/docs/en/runtime.md)

API: [Device and Execution](../API/engine.md) · [Triton Operator Integration](../API/triton.md) · [attention](../API/attention.md)

When reading standard results, gradient results, or calling `ft.evaluate`, `Engine` merges the required expressions into Fuga IR, prunes inputs, lowers them to ASK IR, and then invokes CPU shared libraries or GPU sessions for execution.

<a id="engine-lifecycle"></a>

## 1. `Engine` Lifecycle and Pruning of Unused Input Ports (`engine.py`)

`Engine` connects to the [ASK compiler and runtime](../../../../ask/docs/en/build.md#python-sdk), managing the following state:

- **Engines and Devices**: Default engines are created and reused per device, and shut down upon process exit or `ft.shutdown()`. `ft.set_default_device` uses a `ContextVar` to store the default device for subsequent tensors, while existing tensors retain their original device and engine; explicit `device=` takes precedence. An isolated `Engine`'s context can temporarily override the default selection. After an engine shuts down, reading or executing operations on its tensors raises an error; cross-device operations raise `RuntimeError`, and cross-engine operations on the same device raise `ValueError`.
- **Pruning Unused Physical Input Ports**:
  For example, when a backward function signature is `(x, dy)` but the computation is only `dx = dy * 2`, `Engine._compile` uses [`prune`](native.md#transforms-reuse) to identify actual inputs and compiles physical ports only for `dy`; at invocation time, `_InputProjection` filters actual arguments by index.
- **Compilation Directory Artifacts**: Each compilation writes `fuga.ir` (Fuga IR text), `frontend.mlir` (ASK MLIR), and `fuga-to-ask.json` (see [Lowering Provenance Map](../ir/lowering.md#external-and-sourcemap)) into the artifact directory, accessible via `engine.last_artifact_dir`, `engine.last_fuga_ir`, and `engine.last_ask_ir`.

<a id="compiler-results"></a>

### Compilation Stage Results

The internal `fugatensor._ask.Compiler` invokes the [ASK `ask::compiler` API](../../../../ask/docs/en/api.md#compiler-api) via `_native`, returning the following types by stage:

| Stage and Result | Compilation Contract Fields |
| --- | --- |
| `prepare` / `optimize` → `PrepareResult` | `ir`, `shape_snapshot`, `external_requests` |
| `partition` → `PartitionResult` | The above fields, plus `execution_plan`, `step_sources` |
| `emit_host_llvm` / `emit_host_object` / `emit_cuda_llvm` → `CodegenResult` | Preparation stage fields, plus `target`, `output_path`, `abi`, `cuda_artifact` |

All three result types use an independent `diagnostics` dictionary to store stage, provenance, and optimization statistics. Versioned contracts are decoded into dict/list in the bridge layer; `step_sources` is a mapping from step ID to standalone ASK MLIR module text. Compilation orchestration reads named contract fields, using `shape_snapshot` in cache keys. For Host results, `abi` is populated and `cuda_artifact` is `None`; for CUDA LLVM results, the reverse holds. Host object mode returns the object file path and empty `ir`; LLVM mode returns text and empty `output_path`.

When `ask_opt` is explicitly selected, the CLI still outputs a full JSON report. The bridge entry point splits and validates the report into the same stage results, then enters the shared compilation pipeline. Result field validation covers the matching of stage, target, step source code, and corresponding ABI/artifacts; external floating-point configurations preserve lexical types, negative zero, and 64-bit integer precision.

<a id="host-backend"></a>

## 2. CPU Backend: Object File Generation and `ctypes` Invocation (`backends/host.py`)

The CPU execution path (`HostBackend`) invokes [ASK Host Shared Loop Lowering](../../../../ask/docs/en/codegen.md#host-codegen):

```text
Fuga Module -> lower_to_ask -> ASK::Compiler native code generation (graph.o + ABI contract)
            -> cc links dependencies per ABI (graph.so) -> ctypes invokes function pointer per ABI
```

- **Compilation Entry**: Defaults to calling ASK C++ `ask::compiler::codegen` via `_native`; the `ask_opt` parameter can select the CLI path.
- **ABI Binding**: Reads shapes, dtypes, and byte counts of entry points and ports from `CodegenResult.abi` and writes them separately to `abi.json`; `report.json` stores compilation diagnostics. Subsequent calls validate inputs, allocate `HostStorage` outputs, and invoke the loaded function pointer.
- **Library Dependencies**: `link_libraries` in the ABI declares required library names, and `host_link_libraries` in compiler capabilities provides corresponding installed library paths; HostBackend links against them and configures search paths accordingly. fp32 matrix multiplication calls single-threaded OpenBLAS bundled with the ASK SDK, with libraries and symbols using ASK prefixes.
- **Reduction Reuse**: Scratch spaces are managed by ASK-generated functions; allocations and lifetimes are documented in [Host Codegen](../../../../ask/docs/en/codegen.md#host-codegen).

<a id="cuda-and-triton"></a>

## 3. Owned CUDA and User Triton Operator Execution (`engine.py` / `triton.py`)

The GPU engine creates an [ASK `ExecutionSession`](../../../../ask/docs/en/runtime.md#execution-session) when uploading a Tensor for the first time or accessing `engine.session`. When compiling graphs or querying `engine.capability`, it queries the target architecture, after which ASK partitions the graph into owned, external, and library call steps:

1. **Pure Owned Computation Graph → ASK Owned CUDA Lowering**:
   `Engine` passes the ASK graph and `arch` to the [ASK CUDA Backend](../../../../ask/docs/en/codegen.md#owned-cuda) to generate a cubin. Once the session loads it, it prepares a C++ `PreparedProgram`, and the engine calls the bridge layer `prepared.run(inputs)` to dispatch onto a CUDA stream. Supports multi-input, multi-output with heterogeneous shapes and types, 0-D, and empty tensors.
2. **Mixed Graphs Containing `ft.triton.kernel` → Dependency Scheduling of Owned and External Kernels**:
   - Register `@triton.jit` code via `ft.triton.kernel("kernels.py", entry=..., input=... or inputs=..., output=..., output_shape=..., output_dtype=..., constants=..., grid=..., options=...)` or `ft.triton.from_source(...)`.
   - Registration captures UTF-8 source code and configuration snapshots; subsequent file modifications require re-registration.
   - Each invocation executes `constants`, `scalars`, and `grid` callbacks, passing `TensorSpec(shape, dtype, device, numel)`. `shape` uses reshaped dimensions, while `dtype` and `device` use public objects. Returned configurations freeze with the node, and closure changes only affect subsequent calls. During lowering, the target GPU's `arch` is bound, generating `ask.implementation` and `ask.external`.
   - [`_triton_worker.py`](../../../src/fugatensor/_triton_worker.py) invokes Triton within the selected interpreter, returning `asm["cubin"]` and launch metadata, integrating into the [worker interface](../../../../ask/docs/en/artifacts.md#worker-interface) and [execution plan](../../../../ask/docs/en/partitioning.md#program-runtime).

Matrix multiplication generates `library="cublaslt"` steps, where the plan stores dtype, M/N/K, batch_count, and transpose flags; the runtime selects the algorithm and manages scratch space during the preparation phase.

ASK partitions graphs in MLIR, fuses owned regions, and schedules external nodes and matrix multiplications separately into [Execution Plan JSON v1](../../../../ask/docs/en/partitioning.md#plan-v1). The bridge layer obtains the plan from `PartitionResult.execution_plan`, compiles each step via `step_sources`, and routes workers via `external_requests`. The `execution_plan` field of GPU compilation artifacts independently stores partition plans, while `report` preserves diagnostics and caching information. When reduction results are reused via broadcast, they materialize at the boundary to avoid repeating the entire reduction for each output. Elementwise operations and index transformations within a partition remain fused; derivative graphs follow the same rules. Steps transfer data via device buffers; standard streams launch in sequence, while Graph schedules according to actual dependencies. Invalid graphs, corrupted artifacts, and compilation failures raise errors.

When installing all, the engine discovers Triton from the current environment's installation metadata and launches workers using that interpreter. Isolated environments are selected via `ft.triton.environment(python=...)`. The worker verifies the version, strips constexprs in JIT argument order, fills in scratch parameters for the CUDA ABI, and reads dynamic shared memory size. Global scratch, tensor descriptors, clusters, cooperative, or PDL launches are currently unsupported. Versions and ABIs are recorded in [`triton.dependencies.lock.json`](../../../src/fugatensor/triton.dependencies.lock.json).

`kernel` / `from_source` accept trusted Python source code; the worker executes module top-level code, permitting standard global constants, helper functions, default parameters, and package imports; dependency packages must be installed in the worker environment. `from_jit(existing_kernel, ...)` receives JIT functions already loaded by the caller, recursively saving referenced Python/JIT helpers and global constants while preserving imported modules; modifying original function files or global values after registration does not alter the snapshot. If dependencies in different modules bind different objects using the same name, an explicit error is raised. Snapshots use source code and JIT options stored in the JIT object; programmatic autotuning wrappers, custom reprs, launch metadata, or pre-run hooks that cannot be reconstructed require the caller to provide explicit source code.

Cache probing records the Triton compiler fingerprint, `ptxas`, `libdevice`, runtime code, and digests of loaded dependency files. The compilation environment returned by `get_cache_invalidating_env_vars()` is stored as `producer.environment`; changes in dependencies or floating-point fusion flags invalidate the cache. The worker uses an empty temporary Python cache directory on each invocation to prevent interference from stale `.pyc` files. Custom `extern_libs`, free-form `ptxas` options, and IR/target overrides are rejected.

`required_element_types` records port storage types. Triton and `ptxas` verify architectural requirements for internal instructions, while ASK checks compatibility between artifacts and physical devices.

Tensor ports and scalars are converted into native `PlanStep`s during `PreparedCUDAProgram` construction. When cubin/PTX is loaded, complete `parameter_types` are submitted, and ASK checks type widths against the parameter layout reported by the driver. After preparation, Python passes only input/output buffers on each call, with the native plan responsible for address binding and launching. Parameter schemas, artifact digests, and file contents are validated during preparation.

For syntax regarding multiple inputs, multiple outputs, configuration callbacks, by-value scalars, and constexprs, see [Triton Integration](../API/triton.md#registration-arguments). Each external node retains an independent launch step; physical outputs are determined by kernel contracts, and all physical outputs are allocated even if the frontend consumes only a subset of results.

`ft.nn.functional.scaled_dot_product_attention` uses in-tree FlashAttention source code and `ft.fun.vjp` gradient rules. The forward pass is a single external node, which can fuse with preceding and succeeding owned computations. VJP retrieves the forward public output `O` from `outputs`, constructing four backward kernels: statistics, dQ, dK, and dV. The current derivative IR replays the forward pass to compute `O`, resulting in five external steps for gradient computation.

<a id="native-plan-replay"></a>

### Native Plans and CUDA Graph

GPU JIT uses CUDA Graph for both single-kernel and mixed plans. The initial submission creates a graph executable, whose execution slots are reused upon completion while updating changed input/output addresses. In-flight calls each use a separate set of intermediate buffers and graph instances, returning a completion event per submission.

ASK derives node dependencies from read/write ports during preparation, statically checking the access order of each read and shared slot. Independent branches retain concurrency opportunities, each holding scratch spaces still in use; port interfaces submitted by Fuga remain consistent. CPU materialized scratch spaces are also reused by ASK at compile time based on actual read lifetimes; layouts are documented in [Host Codegen](../../../../ask/docs/en/codegen.md#host-codegen).

The C++ plan retains kernels and in-flight buffers. Fuga's C++ port objects store shape/dtype during preparation, validate actual tensors at submission, establish external stream dependencies, and retain ownership; the ASK native plan validates device placement, byte counts, write permissions, and aliases before dispatching all steps in a single submission. Underlying CUDA events can be recycled after completion, while user-held Events remain completed.

In the bridge layer, `prepared.run(inputs)` allocates outputs via the native plan, asynchronously returning a list of device tensors; duplicate outputs and input aliases retain identical tensor handles. `prepared.submit_many(inputs, outputs)` uses caller-provided outputs and returns a completion event. Both entry points uniformly validate ports, establish external stream dependencies, and retain ownership; completion events for automatic outputs are retained by the session, and external producers must synchronize the session before reusing input storages.

ASK provides both standard stream submission and Graph replay. FugaTensor JIT defaults to the latter.

For comparisons of latency and memory between dependency scheduling and CPU scratch space reuse, see [ASK Verification Report](../../../../ask/docs/en/workspace_report.md#fuga-profile).

<a id="cache-hierarchy"></a>

## 4. Division of Labor between Graph Compilation and ASK Caching

| Level | Location | Cache Key | Work Skipped on Hit |
| --- | --- | --- | --- |
| **1. Engine Graph Compilation Cache** | In-memory per FugaTensor `Engine` | Output projection of immutable template, DAG connectivity and shared leaf relationships, output ordering, relevant definition identities | Graph stitching, full IR text generation, Fuga IR lowering, and backend compilation; standard Python logic still executes every time |
| **2. Owned GPU LLVM IR Cache** | `Engine.cache_dir/ask/entries` on disk | ASK IR source digest, shape snapshot, target and tool fingerprints | Owned LLVM lowering |
| **3. Owned GPU `cubin` Cache** | `Engine.cache_dir/ask/executables/entries` on disk | LLVM IR digest, artifact contract, codegen parameters, and tool/environment fingerprints | Conditional libdevice linking, `llc` PTX generation, and `ptxas` assembly |
| **4. External Compilation Artifact Cache** | `Engine.cache_dir/ask/external/entries` on disk | Full request, worker, compiler version, and environment fingerprints | Triton subprocess compilation; `cubin` stored directly in cache |
| **5. Session Loaded Modules** | Within ASK `ExecutionSession` | `cubin` bytes and entry name | `cuModuleLoadDataEx` within the same GPU session |

`engine.cache_info()` returns cumulative `compilations`, `cache_hits`, and the current number of `specializations`. `engine.clear_cache()` clears in-memory compiled graphs and recent diagnostic information while preserving cumulative counts, computed Tensors, and disk caches. Recently submitted graphs and reports can be inspected via `engine.last_fuga_ir`, `last_ask_ir`, `last_report`, `last_artifact_dir`; CUDA partition plans are viewed via `engine.last_execution_plan`, which is `None` if no partition plan was produced, no execution has occurred, cache was cleared, or the engine was shut down.

For template freezing and structural key rules, see [Automatic Graph Fusion and Engine Caching](tensor_model.md#jit-specialization). For bulk import/export and borrowing lifetimes of CPU/CUDA contiguous memory, see [Buffer Interface](tensor_model.md#buffer-io).

When owned CUDA LLVM IR contains `__nv_` math functions, compilation orchestration links libdevice using `llvm-link --only-needed` before generating PTX. The cache key records llvm-link and libdevice file digests; standard elementwise graphs only use llc and ptxas.

The ASK IR text digest participates in disk cache keys; changes in text generate distinct cache entries. For verification and invalidation criteria, see [ASK Two-Tier Persistent Cache](../../../../ask/docs/en/runtime.md#two-tier-cache).

<a id="bootstrap-and-config"></a>

## 5. Post-Installation Configuration (`_config.py`)

FugaTensor links against `ASK::Compiler` and `ASK::Runtime` from the ASK SDK at build time, and the wheel bundles corresponding shared libraries; see [Installation Guide](../install.md). CPU execution requires `cc` to link object files, owned CUDA code generation requires `llc` / `ptxas`, and external kernel binaries are provided by workers.

`Engine` and `_config.py` read configuration according to the following precedence:

1. **Explicit Arguments**: `ft.configure(device=..., ...)` or `ft.Engine(...)` arguments;
2. **Environment Variables**: `ASK_OPT`, `FUGATENSOR_CACHE_DIR`, `FUGATENSOR_LLC`, `FUGATENSOR_PTXAS`, `FUGATENSOR_LLVM_LINK`, `FUGATENSOR_LIBDEVICE`, `FUGATENSOR_TRITON_{PYTHON,WORKER}` (and `FUGATENSOR_CONFIG` specifying a config file path);
3. **Optional Environment Config File**: `sys.prefix/share/fugatensor/engine.json` in the current Python environment, defaulting to empty configuration if nonexistent;
4. **Current Environment Auto-Discovery**: When no Triton worker is configured, if Triton is installed in the current interpreter, the local environment and in-tree worker are used. Explicitly passing `compilers={}` disables compiler discovery.

`cc` is configured via `ft.Engine(cc=...)` or `ft.configure(cc=...)`. Specifying `ask_opt` via code or environment variable isolates SDK and worker paths in the default configuration file.

Development installation uses `CMAKE_PREFIX_PATH=SDK_DIR pip install -e '.[pure]'` (or `'.[all]'` to also install Triton). For installation details, see [Installation Guide](../install.md).

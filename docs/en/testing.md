Back to Documentation Index: [Documentation Index](README.md) · Related API: [API](API/README.md) · Environment Setup: [Installation and Configuration](install.md) · Backend Verification: [ASK Testing System](../../../ask/docs/en/testing.md)

# Test Organization and Coverage Matrix

Tests are located in `tests/`, invoking public APIs to verify numerical results, gradients, and exceptions. Input generation and numerical references reside in test support modules.

Baseline performance between Fuga and Torch can be compared via standalone benchmark scripts, generating workload matrices and checking numerical values outside timing loops.

<a id="running"></a>

## Execution Scope

Install test dependencies in the development environment: `python -m pip install -e '.[pure,test]'`; when using Triton, select `'.[all,test]'`. Tests use the installed native extension.

```bash
# CPU, Fuga IR, native graph analysis, and API contracts
.venv/bin/python -m pytest -q

# Add physical CUDA owned computations; --triton enables CUDA and external workers simultaneously
.venv/bin/python -m pytest --cuda -q
.venv/bin/python -m pytest --triton -q

# Expand full combinations of the numerical matrix
.venv/bin/python -m pytest --triton --exhaustive -q
```

Fixtures in `tests/conftest.py` manage Engines, contexts, and temporary caches. The numerical parameter matrix reuses Engines per class, while state, invalidation, and lifetime cases create independent Engines. Explicitly selected CUDA/Triton tests require available drivers, tools, and workers, failing if missing. Unselected hardware test cases are recorded as deselected; skipped cases must be logged individually.

Default regression runs generative representative combinations and targeted boundaries. `--exhaustive` expands all axes; pairwise coverage validates value combinations for each factor pair, without claiming coverage of all three-factor and higher-order interactions. Numerical error, materialization order, lifecycles, concurrency, definition invalidation, and true hash collisions use dedicated behavioral assertions.

Backend protocols, SDKs, and underlying numerical tests are selected independently, configured via `ASK_SOURCE_ROOT`, `ASK_OPT`, `ASK_LLVM_TOOLS_DIR`, and `ASK_PTXAS`:

```bash
.venv/bin/python -m pytest tests/backend -q
.venv/bin/python -m pytest tests/backend --cuda -q
```

`ASK_SOURCE_ROOT` allows integration tests to read ASK fixtures and dependency locks. ASK native tests are still executed via ASK's CTest.

Wheel installation tests create separate virtual environments for pure and all, installing dependencies via pip, and require explicitly specifying the wheel:

```bash
.venv/bin/python -m pytest tests/packaging --wheel /absolute/path/to/fugatensor.whl --triton -q
```

Omit `--triton` when verifying CPU installation only. Default test discovery does not enter `tests/backend/` and `tests/packaging/`. Rounding and saturating casts share independent references in `tests/backend/support/pointwise.py`.

<a id="operation-matrix"></a>

## Basic API Combination Matrix

`OPERATIONS` in [`tests/support/ops_matrix.py`](../../tests/support/ops_matrix.py) registers invocation styles, independent reference expressions, and applicable dtypes for each API. `plan_matrix()` automatically generates sequential compositions, crossed with dtype, shape, and differentiation order. By default, single operators are checked across all shapes, and all consecutive two-layer combinations are checked on matrix shapes; `--exhaustive` expands combinations across all shapes. By default, 666 dtype/shape/operator configurations are generated, with 2,350 configurations in the full set; common inputs consist of two tensors, each layer consuming the previous layer's result and the second input, while single-input operators consume only the previous layer's result.

[`tests/test_ops_matrix.py`](../../tests/test_ops_matrix.py) checks shape, dtype, forward values, first-order partial derivatives, and second-order partial derivatives including mixed terms, while also covering constant paths and gradients of unused inputs. Reference results are calculated using bivariate polynomials and exact fractional differentiation. Combinations of identical precision and shape are evaluated in batches to reduce compilation and dispatch overhead. Shape axes cover scalars, vectors, matrices, singleton dimensions, and empty tensors; reshape automatically generates round-trip transformations using the current shape.

| Environment Variable | Options and Default Behavior |
| --- | --- |
| `FUGATENSOR_TEST_OPS` | Comma-separated registered names; defaults to all: `add,subtract,multiply,negative,square,reshape,cast,zeros_like,ones_like` |
| `FUGATENSOR_TEST_DEPTHS` | Subset of `1,2,3`; defaults to `1,2` |
| `FUGATENSOR_TEST_DTYPES` | Subset of `fp32,fp16,bf16,int32,int64,bool`; defaults to all, filtered by operator's supported types |
| `FUGATENSOR_TEST_SHAPES` | Subset of `scalar,vector,matrix,singleton,empty`; defaults to all |
| `FUGATENSOR_TEST_ORDERS` | Subset of `0,1,2,3`; defaults to `0,1,2`; integers and bool check forward pass only |

To reproduce only multiplication, square, and their two-layer combinations, verifying fp16 forward values and first-order derivatives:

```bash
FUGATENSOR_TEST_OPS=multiply,square \
FUGATENSOR_TEST_DEPTHS=2 \
FUGATENSOR_TEST_DTYPES=fp16 \
FUGATENSOR_TEST_ORDERS=0,1 \
.venv/bin/python -m pytest tests/test_ops_matrix.py::TestHostOperationMatrix -q
```

When switching to GPU, select `TestCUDAOperationMatrix` and add `--cuda`. Filter values must use registered names; empty values and unknown names raise errors immediately to prevent missed testing due to typos.

When adding new basic operators, add invocation and reference expressions in `OPERATIONS` to integrate into the combination matrix. Current references cover polynomial operations as well as shape and type transformations preserving numerical values; non-polynomial operators must provide supplemental reference methods. Shared branches, multiple outputs, repeated differentiation, resource lifecycles, caching, and exceptions are covered by specialized tests.

The storage matrix for CUDA Array Interface is generated by `tests/backend/support/storage_matrix.py`, crossing five CAI types, five shapes, and three stride descriptions (implicit contiguous, explicit contiguous, and strided). Assertions check parsing results of valid descriptions and rejection of non-contiguous storage.

The GPU parameter ABI probe in `tests/backend/support/scalar_abi.py` generates standalone PTX, covering scalars of various widths, aligned pointers, and null; the same probe crosses PTX/cubin and stream/Graph launches, checking exact bit patterns, frozen constants, and public port address updates.

<a id="attention-matrix"></a>

## Attention Matrix

[`tests/test_attention.py`](../../tests/test_attention.py) calls the [attention API](API/attention.md), generating 6 pairwise direct invocation configurations from fp16/bf16, three shape categories, and causal modes, then adding tail-block, causal-mode, and pre/post-scaling combinations for each precision, totaling 8 configurations. `--exhaustive` expands to all 24 configurations. Each configuration checks forward values, Q/K/V first-order gradients, shape, precision, and execution plan; numerical references and backward formulas are documented in [`tests/support/attention.py`](../../tests/support/attention.py).

| Environment Variable | Options and Default Behavior |
| --- | --- |
| `FUGATENSOR_TEST_DTYPES` | Shared with basic matrix; attention defaults to `fp16,bf16`, executing only types supported by this API |
| `FUGATENSOR_TEST_ATTENTION_SHAPES` | Defaults to `tail,wide,batched`, covering chunk tail, unequal feature width, and multi-batch/multi-head respectively |
| `FUGATENSOR_TEST_ATTENTION_CAUSAL` | Defaults to `false,true` |
| `FUGATENSOR_TEST_ATTENTION_COMPOSITIONS` | Defaults to `plain,scaled`; `scaled` adds query input scaling and output scaling simultaneously |

To check a single attention configuration:

```bash
FUGATENSOR_TEST_DTYPES=fp16 \
FUGATENSOR_TEST_ATTENTION_SHAPES=tail \
FUGATENSOR_TEST_ATTENTION_CAUSAL=true \
FUGATENSOR_TEST_ATTENTION_COMPOSITIONS=scaled \
.venv/bin/python -m pytest tests/test_attention.py::TestAttentionCUDAExecution --triton -q
```

Input contracts are also checked by CPU-runnable `TestAttentionContract`, covering unsupported devices, types, dimensions, zero dimensions, and causal parameters; GPU tests additionally verify that higher-order differentiation lacking rules is rejected during derivative graph construction.

<a id="tensor-basis-matrix"></a>

## Tensor Primitives and Matrix Multiplication Matrix

`tests/support/tensor_primitives.py` generates dtype × shape × permutation/axes × device combinations, using independent row-major index references to verify broadcast, transpose, comparison, select, and reduction. `tests/test_tensor_primitives.py` simultaneously checks elementary functions, JVP, VJP, second-order derivatives, and fused region counts.

Matrix multiplication generates pairwise representative combinations from floating-point dtypes, batch shapes, M/N/K, and transpose flags, with targeted additions of empty M/N/K, zero batches, and all transpose flags for each dtype. Defaults to 51 cases on CPU and 67 on CUDA; `--exhaustive` expands to 160 and 240 cases, respectively. Independent index references verify numerical values, and the gradient matrix covers transpose, batch dimensions, and crossed second-order derivatives. CUDA checks library call steps, elementwise fusion on both sides of matrix multiplication, and both standard stream and Graph execution.

[`tests/test_host_matmul.py`](../../tests/test_host_matmul.py) generates 26 cases from 1,536 combinations of CPU FP32/FP16, batch shapes, row/col tile boundaries, K, transpose, and shared outputs, covering all 204 factor pair values. An additional 6 FP32 cancellation error cases, 3 FP16 accumulation order cases, and 18 NaN/Inf row/col tail boundary cases make a total of 53 cases. `tests/test_case_selection.py` independently validates pairwise coverage; `--exhaustive` expands to 1,563 cases.

FP32 CPU matmul is validated against OpenBLAS's f32 accumulation error bounds, with FP16 preserving step-by-K f32 accumulation order. CUDA FP32 uses TF32 tolerances, where independent high-precision dot product references calculate error bounds combining input rounding, reduction length, and absolute product sum; non-exact TF32 inputs, cancellations, and NaN/Inf are checked specifically by [`tests/test_matmul_precision.py`](../../tests/test_matmul_precision.py). Inputs are sampled according to actual storage types first, preventing input encoding errors from being misclassified as multiplication errors.

```bash
.venv/bin/python -m pytest tests/test_tensor_primitives.py --cuda -q
```

`FUGATENSOR_TEST_DTYPES` simultaneously controls the dtype axis of this matrix; the CPU numerical matrix selects five types based on `host_supported_element_types`, with bf16 independently asserted to be rejected at compile time; CUDA executes bf16 numerical verification.

The CPU scratch space access matrix crosses 4 reducible dtypes, 3 row counts, 2 column counts, 2 reduction extension depths, transpose layout, and intermediate value public outputs, generating 27 pairwise combinations and targeted lifetime interactions by default (192 cases in full), covering empty tensors and single-element SSA. An additional 5 × 5 type cast and scalar early-capture combinations (50 cases in total) verify cross-type slots and SSA keep-alives. Independent Python numerical references verify full user API paths. The GPU native branch matrix crosses branch counts, chain depths, and stream/Graph modes, checking numerical values, retained outputs, variable-address replays, dependency edge counts, and internal scratch capacity; ASK's native matrix supplements coverage of static illegal plans and cuBLASLt scratch.

The CUDA `amax` matrix in [`tests/test_cuda_reduction_max.py`](../../tests/test_cuda_reduction_max.py) crosses five numerical dtypes, ten shapes, and two physical layouts, generating 50 pairwise input combinations by default (100 in full); each group generates all seven non-empty reduction axis combinations from rank, checking direct, pre-, and post-elementwise combinations, totaling 580 valid outputs and 60 empty reduction domain rejections by default (1,160 and 120 in full). Reduction widths cover 0, 1, 31, 32, 33, 63, 64, 65, including full tensor reduction to scalar and outputs with empty preserved dimensions. Floating-point values check NaN, positive/negative infinity, and zero signs elementwise; integers cover extreme values and i64 upper 32-bit variations. An additional 10 default (20 full) groups cover nested reductions, reduction expansions, and public outputs of differing shapes. Inputs and independent IEEE maximum references reside in [`tests/support/reduction_max.py`](../../tests/support/reduction_max.py).

```bash
.venv/bin/python -m pytest tests/test_cuda_reduction_max.py --cuda -q
```

<a id="lazy-schedule-matrix"></a>

## Lazy Graph and Cache Binding Matrix

[`tests/test_lazy_schedule.py`](../../tests/test_lazy_schedule.py) generates 112 cases across fp32/int32, scalar/matrix, shared/independent inputs, direct/detach outputs, and non-empty subsets of three return values. Each case re-evaluates with swapped input data, verifying numerical values, output types, cache hits, actual execution counts, unrequested outputs, and retained results. It also covers output ordering, multiple returns of the same value, returning inputs directly, constant graph input carriers, derivative provenance after materialization, and arithmetic chains and alias chains exceeding Python recursion depth.

Graphs and numerical references reside in [`tests/support/lazy_graphs.py`](../../tests/support/lazy_graphs.py). Definition changes, invalidated branches, cold/hot call arguments, and result publication upon compilation failure are verified by `tests/test_jit.py` and `tests/test_autodiff.py`.

```bash
.venv/bin/python -m pytest tests/test_lazy_schedule.py -q
```

<a id="native-arena-matrix"></a>

## Native Node Pool and Structural Identity

[`tests/test_native_arena.py`](../../tests/test_native_arena.py) generates 96 cases from dtype × shape × layout × input sharing × output ordering × materialization point, verifying numerical values, types, shapes, and cache hits after changing data. Inputs and independent references reside in [`tests/support/arena_graphs.py`](../../tests/support/arena_graphs.py).

State cases verify storage boundaries when downstream constructs graphs first and parent nodes materialize later, using value changes in borrowed inputs to distinguish saved results from recomputation. Identical hashes for integer constants `-1`, `-2` form real collisions, validating that different graphs still use their respective compilation results. Lifetime cases verify differentiation after intermediate Python thin shells are released, reclamation after closing Engines, and stable slot capacity with older outputs retaining provenance during repeated graph construction.

[`tests/test_arena_ownership.py`](../../tests/test_arena_ownership.py) crosses automatic/explicit VJP definitions with retaining/releasing pending roots, verifying reclamation of temporary function instances and template references while Engine is alive, and covering concurrent cold graph construction on the same Engine.

[`tests/test_primitive_binding.py`](../../tests/test_primitive_binding.py) generates 32 configuration groups from dtype, shape, reduction kind, keepdims, and axis sets, deriving positive/negative integers, tuple, list, generator, and all-axis representations for a total of 208 invocations. Swapping input data verifies that equivalent arguments reuse numerical programs; default broadcasting crossed with explicit dimension shape representations validates 12 automatic broadcasting calls, along with checking invalid argument errors after warm-up.

```bash
.venv/bin/python -m pytest tests/test_native_arena.py tests/test_arena_ownership.py tests/test_primitive_binding.py -q
```

<a id="storage-matrix"></a>

## Tensor Creation and Storage Matrix

`tests/support/storage.py` generates combinations by dtype × shape × creation entry point; `tests/test_tensor_storage.py` verifies CPU five dtypes, CPU bf16 creation rejection, CUDA six dtypes, raw byte fidelity, and copy contracts. GPU creation directly uses device storage, and the creation matrix concurrently verifies that compilation counts remain unchanged. BF16 computation and weight round-trips execute on CUDA devices supporting that precision.

```bash
.venv/bin/python -m pytest tests/test_tensor_storage.py tests/test_weights.py --cuda -q
```

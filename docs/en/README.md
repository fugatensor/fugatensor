# FugaTensor Documentation Index

Browse documentation by API, architecture, and IR. For underlying compilation and execution, see [ASK Documentation](../../../ask/docs/en/README.md).

For environment setup, see [Installation and Build](install.md).

## API (`docs/API/`)

[API Directory](API/README.md) provides signatures, arguments, return values, and usage:

- [Tensor and Data Exchange](API/tensor.md) · [Basic Operations](API/ops.md)
- [Device and Execution](API/engine.md) · [Automatic Differentiation and Custom Functions](API/autograd.md)
- [Triton Operator Integration](API/triton.md) · [Weight Reading and Writing](API/weights.md)
- [IR and Lowering](API/ir.md) · [attention](API/attention.md)

For running tests and extending test cases, see [Test Organization and Coverage Matrix](testing.md).

## System Architecture (`docs/architecture/`)

| Document | Description | Relevant Code |
| --- | --- | --- |
| [Architecture Overview](architecture/README.md) | Division of responsibilities between FugaTensor and ASK, end-to-end invocation flow, and subsystem entry points | Global architecture |
| [Tensor Model and Tracing Mechanism](architecture/tensor_model.md) | CPU/CUDA tensor storage and buffer interface, immutable templates and structural cache, on-demand JIT, isolated tracing, and 0-D scalar conditions | `tensor.py`, `ops.py`, `_lazy.py`, `_jit.py`, `tracing.py` |
| [Native Graph Analysis and C++ Boundary](architecture/native.md) | Native node pool, tracing and primitive binding, SSA analysis, structural keys, ownership, and concurrency | `cpp/graph.*`, `cpp/lazy.*`, `cpp/trace.cpp`, `cpp/primitive.cpp`, `ir/transforms.py`, `_lazy.py` |
| [Automatic Differentiation and Custom Operators](architecture/autograd.md) | `ft.grad` pure IR transformation, expression provenance, `ft.fun` pure VJP, higher-order derivatives, and rule validation | `autograd.py`, `function.py`, `ir/differentiate.py`, `ir/jvp.py` |
| [ASK Bridge and Multi-Backend Execution](architecture/backend.md) | `Engine` lifecycle, unused input port pruning, CPU `HostBackend`, native Graph dispatch, owned CUDA and isolated Triton scheduling, compilation caching, and configuration | `engine.py`, `backends/`, `cpp/compiler.cpp`, `cpp/ask_runtime.cpp`, `_ask/`, `triton.py`, `_triton_worker.py`, `_config.py` |
| [Weight Format and Export](architecture/weights.md) | Versioned PyTree weight directories, manifest/array identity, atomic export, integrity validation, and architectural extension boundaries | `weights.py`, `schemas/weights-manifest-v1.schema.json` |

## Fuga IR Design and Transformations (`docs/ir/`)

| Document | Description | Relevant Code |
| --- | --- | --- |
| [Fuga IR Overview](ir/README.md) | Responsibilities of independent framework-level IR, device-independent representations, and flow from tracing to lowering | `src/fugatensor/ir/` |
| [Object Model and Verification](ir/model.md) | `ScalarType`, `TensorType`, SSA `Value`, `Operation`, `Block`, `Region`, `Function`, `Module`, and structural verifier | `ir/types.py`, `ir/core.py` |
| [Control Flow and Composable Transformations](ir/transforms.md) | `if`/`for`/`while` region semantics, primitive rules table (`abstract_eval` / `jvp` / `vjp`), and `jvp`/`vjp` graph transformations | `ir/primitives.py`, `ir/jvp.py`, `ir/differentiate.py` |
| [Lowering to ASK](ir/lowering.md) | Static branch selection and bounded loop unrolling, type and operator mapping, external Triton architecture binding, and `fuga-to-ask.json` provenance | `lowering.py` |

## Underlying Compilation Engine Documentation (`ask/docs/`)

- [ASK Installation](../../../ask/docs/en/install.md)
- [ASK Documentation Index](../../../ask/docs/en/README.md) · [ASK Core Concepts](../../../ask/docs/en/concepts.md)
- [Graph Partitioning and Multi-Kernel Execution Plan](../../../ask/docs/en/partitioning.md) · [Multi-Kernel Plan JSON Contract](../../../ask/schemas/execution-plan-v1.schema.json)
- [External Compiler Integration Specification](../../../ask/docs/en/compilers.md) · [Worker Request and Artifact Contracts](../../../ask/schemas/)
- [ASK IR and Numerical Semantics](../../../ask/docs/en/ir.md) · [External Artifacts and Execution Interface](../../../ask/docs/en/artifacts.md)
- [Resident Runtime and Multi-Level Caching](../../../ask/docs/en/runtime.md) · [FugaTensor Collaboration and Evolution Roadmap](../../../ask/docs/en/fugatensor.md)

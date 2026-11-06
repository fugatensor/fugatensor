# FugaTensor 文档目录

按 API、架构和 IR 查找文档。底层编译与执行见 [ASK 文档](../../ask/docs/README.md)。

环境准备见[安装与构建](install.md)。

## API（`docs/API/`）

[API 分类目录](API/README.md)提供签名、参数、返回值和用法：

- [张量与数据交换](API/tensor.md) · [基础运算](API/ops.md)
- [设备与执行](API/engine.md) · [自动微分与自定义函数](API/autograd.md)
- [Triton 算子接入](API/triton.md) · [权重读写](API/weights.md)
- [IR 与降级](API/ir.md) · [attention](API/attention.md)

测试运行和用例扩展见[测试组织与覆盖矩阵](testing.md)。

## 系统架构（`docs/architecture/`）

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [架构总览](architecture/README.md) | FugaTensor 与 ASK 的职责划分、端到端调用流程与子系统入口 | 全局架构 |
| [张量模型与追踪机制](architecture/tensor_model.md) | CPU/CUDA 张量存储与 buffer 接口、不可变模板与结构缓存、按需 JIT、隔离追踪与 0-D 标量条件 | `tensor.py`、`ops.py`、`_lazy.py`、`_jit.py`、`tracing.py` |
| [原生图分析与 C++ 边界](architecture/native.md) | 原生节点池、追踪与原语绑定、SSA 分析、结构键、所有权和并发 | `cpp/graph.*`、`cpp/lazy.*`、`cpp/trace.cpp`、`cpp/primitive.cpp`、`ir/transforms.py`、`_lazy.py` |
| [自动微分与自定义算子](architecture/autograd.md) | `ft.grad` 纯 IR 变换、表达式来源、`ft.fun` 纯 VJP、高阶求导与规则校验 | `autograd.py`、`function.py`、`ir/differentiate.py`、`ir/jvp.py` |
| [ASK 桥接与多后端执行](architecture/backend.md) | `Engine` 生命周期、无用输入端口裁剪、CPU `HostBackend`、原生 Graph 提交、自有 CUDA 与独立 Triton 调度、编译缓存与配置 | `engine.py`、`backends/`、`cpp/compiler.cpp`、`cpp/ask_runtime.cpp`、`_ask/`、`triton.py`、`_triton_worker.py`、`_config.py` |
| [权重格式与导出](architecture/weights.md) | 版本化 PyTree 权重目录、manifest/数组身份、原子导出、完整性校验与架构扩展边界 | `weights.py`、`schemas/weights-manifest-v1.schema.json` |

## Fuga IR 设计与变换（`docs/ir/`）

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [Fuga IR 总览](ir/README.md) | 独立框架级 IR 的职责、设备无关表示及从追踪到降级的流程 | `src/fugatensor/ir/` |
| [对象模型与验证](ir/model.md) | `ScalarType`、`TensorType`、SSA `Value`、`Operation`、`Block`、`Region`、`Function`、`Module` 及结构化校验器 | `ir/types.py`、`ir/core.py` |
| [控制流与可组合变换](ir/transforms.md) | `if`/`for`/`while` 区域语义、原语规则表（`abstract_eval` / `jvp` / `vjp`）及 `jvp`/`vjp` 图变换 | `ir/primitives.py`、`ir/jvp.py`、`ir/differentiate.py` |
| [降级到 ASK](ir/lowering.md) | 静态分支选择与有界循环展开、类型与算子映射、外部 Triton 架构绑定及 `fuga-to-ask.json` 溯源 | `lowering.py` |

## 底层编译引擎文档（`ask/docs/`）

- [ASK 安装](../../ask/docs/install.md)
- [ASK 文档目录](../../ask/docs/README.md) · [ASK 核心思想](../../ask/docs/concepts.md)
- [图切分与多 Kernel 执行计划](../../ask/docs/partitioning.md) · [多 kernel 计划 JSON 契约](../../ask/schemas/execution-plan-v1.schema.json)
- [外部编译器适配接入规范](../../ask/docs/compilers.md) · [worker 请求与 artifact 契约](../../ask/schemas/)
- [ASK IR 与数值语义](../../ask/docs/ir.md) · [外部产物与执行接口](../../ask/docs/artifacts.md)
- [常驻运行时与多级缓存](../../ask/docs/runtime.md) · [FugaTensor 协同与演进路线](../../ask/docs/fugatensor.md)

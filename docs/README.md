# FugaTensor 文档目录

按 API、架构和 IR 查找文档。底层编译与执行见 [ASK 文档](../../ask/docs/README.md)。

环境准备见[安装与构建](安装.md)。

## API（`docs/API/`）

[API 分类目录](API/README.md)提供签名、参数、返回值和用法：

- [张量与数据交换](API/张量与数据交换.md) · [基础运算](API/基础运算.md)
- [设备与执行](API/设备与执行.md) · [自动微分与自定义函数](API/自动微分与自定义函数.md)
- [Triton 算子接入](API/Triton算子接入.md) · [权重读写](API/权重读写.md)
- [IR 与降级](API/IR与降级.md) · [attention](API/attention.md)

测试运行和用例扩展见[测试组织与覆盖矩阵](测试.md)。

## 系统架构（`docs/architecture/`）

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [架构总览](architecture/README.md) | FugaTensor 与 ASK 的职责划分、端到端调用流程与子系统入口 | 全局架构 |
| [张量模型与追踪机制](architecture/张量模型与追踪机制.md) | CPU/CUDA 张量存储与 buffer 接口、不可变模板与结构缓存、按需 JIT、隔离追踪与 0-D 标量条件 | `tensor.py`、`ops.py`、`_lazy.py`、`_jit.py`、`tracing.py` |
| [原生图分析与 C++ 边界](architecture/原生图分析与C++边界.md) | Python 与 C++ `_native` 扩展的分工、可达性与反向边界分析、`ir/transforms.py` 剪枝投影与无状态并发 | `cpp/graph.h`、`cpp/graph.cpp`、`cpp/bindings.cpp`、`ir/transforms.py` |
| [自动微分与自定义算子](architecture/自动微分与自定义算子.md) | 运行时 `GradHistory` 反向图调度、`ft.fun` 自定义前向/反向、JVP/VJP 高阶求导与反向预检 | `autograd.py`、`function.py`、`ir/differentiate.py`、`ir/jvp.py` |
| [ASK 桥接与多后端执行](architecture/ASK桥接与多后端执行.md) | `Engine` 生命周期、无用输入端口裁剪、CPU `HostBackend`、原生 Graph 提交、自有 CUDA 与独立 Triton 调度、编译缓存与配置 | `engine.py`、`backends/`、`cpp/runtime.cpp`、`_ask/`、`triton.py`、`_triton_worker.py`、`_config.py`、`tools/bootstrap.py` |
| [权重格式与导出](architecture/权重格式与导出.md) | 版本化 PyTree 权重目录、manifest/数组身份、原子导出、完整性校验与架构扩展边界 | `weights.py`、`schemas/weights-manifest-v1.schema.json` |

## Fuga IR 设计与变换（`docs/ir/`）

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [Fuga IR 总览](ir/README.md) | 独立框架级 IR 的职责、设备无关表示及从追踪到降级的流程 | `src/fugatensor/ir/` |
| [对象模型与验证](ir/对象模型与验证.md) | `ScalarType`、`TensorType`、SSA `Value`、`Operation`、`Block`、`Region`、`Function`、`Module` 及结构化校验器 | `ir/types.py`、`ir/core.py` |
| [控制流与可组合变换](ir/控制流与可组合变换.md) | `if`/`for`/`while` 区域语义、原语规则表（`abstract_eval` / `jvp` / `vjp`）及 `jvp`/`vjp` 图变换 | `ir/primitives.py`、`ir/jvp.py`、`ir/differentiate.py` |
| [降级到 ASK](ir/降级到ASK.md) | 静态分支选择与有界循环展开、类型与算子映射、外部 Triton 架构绑定及 `fuga-to-ask.json` 溯源 | `lowering.py` |

## 底层编译引擎文档（`ask/docs/`）

- [ASK 安装](../../ask/docs/安装.md)
- [ASK 文档目录](../../ask/docs/README.md) · [ASK 核心思想](../../ask/docs/核心思想.md)
- [图切分与多 Kernel 执行计划](../../ask/docs/图切分与多Kernel执行计划.md) · [多 kernel 计划 JSON 契约](../../ask/schemas/execution-plan-v1.schema.json)
- [外部编译器适配接入规范](../../ask/docs/外部编译器适配接入规范.md) · [worker 请求与 artifact 契约](../../ask/schemas/)
- [ASK IR 与数值语义](../../ask/docs/ASK-IR与数值语义.md) · [外部产物与执行接口](../../ask/docs/外部产物与执行接口.md)
- [常驻运行时与多级缓存](../../ask/docs/常驻运行时与多级缓存.md) · [FugaTensor 协同与演进路线](../../ask/docs/FugaTensor协同与演进路线.md)

# FugaTensor 文档目录

FugaTensor 构建在 [ASK](../../ask/docs/README.md) 编译与运行时引擎之上：前端记录独立的 Fuga IR，结合 C++ 原生图分析完成控制流化简与前向/反向微分，再将静态计算图降级为 ASK IR 交由 CPU 或 CUDA 执行。

## 系统架构（`docs/architecture/`）

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [架构总览](architecture/README.md) | FugaTensor 与 ASK 的职责划分、端到端调用流程与子系统入口 | 全局架构 |
| [01. 张量模型与追踪机制](architecture/01-张量模型与追踪机制.md) | CPU/CUDA 张量存储、`@ft.jit` 特化键、`TraceContext` 追踪与 0-D 标量动态图中断（Graph Break） | `tensor.py`、`ops.py`、`jit.py`、`tracing.py` |
| [02. 原生图分析与 C++ 边界](architecture/02-原生图分析与C++边界.md) | Python 与 C++ `_native` 扩展的分工、可达性与反向边界分析、`ir/transforms.py` 剪枝投影与无状态并发 | `cpp/graph.h`、`cpp/graph.cpp`、`cpp/bindings.cpp`、`ir/transforms.py` |
| [03. 自动微分与自定义算子](architecture/03-自动微分与自定义算子.md) | 运行时 `GradHistory` 反向图调度、`ft.fun` 自定义前向/反向、JVP/VJP 高阶求导与反向预检 | `autograd.py`、`function.py`、`ir/differentiate.py`、`ir/jvp.py` |
| [04. ASK 桥接与多后端执行](architecture/04-ASK桥接与多后端执行.md) | `Engine` 生命周期、无用输入端口裁剪、CPU `HostBackend`、自有 CUDA 与 Triton 融合执行、五层缓存与 `bootstrap.py` 配置 | `engine.py`、`backends/`、`triton.py`、`_config.py`、`tools/bootstrap.py` |

## Fuga IR 设计与变换（`docs/ir/`）

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [Fuga IR 总览](ir/README.md) | 独立框架级 IR 的职责、设备无关表示及从追踪到降级的流程 | `src/fugatensor/ir/` |
| [01. 对象模型与验证](ir/01-对象模型与验证.md) | `TensorType`、SSA `Value`、`Operation`、`Block`、`Region`、`Function`、`Module` 及结构化校验器 | `ir/types.py`、`ir/core.py`、`ir/verify.py`、`ir/printer.py` |
| [02. 控制流与可组合变换](ir/02-控制流与可组合变换.md) | `if`/`for`/`while` 区域语义、原语规则表（`abstract_eval` / `jvp` / `vjp`）及 `jvp`/`vjp` 图变换 | `ir/primitives.py`、`ir/jvp.py`、`ir/differentiate.py` |
| [03. 降级到 ASK](ir/03-降级到ASK.md) | 静态分支选择与有界循环展开、类型与算子映射、外部 Triton 架构绑定及 `fuga-to-ask.json` 溯源 | `lowering.py` |

## 底层编译引擎文档（`ask/docs/`）

- [ASK 文档目录](../../ask/docs/README.md) · [ASK 核心思想](../../ask/docs/核心思想.md)
- [ASK IR 与数值语义](../../ask/docs/01-ASK-IR与数值语义.md) · [外部编译器适配与边界内联](../../ask/docs/04-外部编译器适配与边界内联.md)
- [常驻运行时与多级缓存](../../ask/docs/05-常驻运行时与多级缓存.md) · [FugaTensor 协同与演进路线](../../ask/docs/07-FugaTensor协同与演进路线.md)

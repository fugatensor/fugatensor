# FugaTensor 文档目录

FugaTensor 构建在 [ASK](../../ask/docs/README.md) 编译与运行时引擎之上：前端记录独立的 Fuga IR，结合 C++ 原生图分析完成控制流化简与前向/反向微分，再将静态计算图降级为 ASK IR 交由 CPU 或 CUDA 执行。

安装入口：[安装与独立构建](安装.md)。先安装 ASK C++ SDK，再构建 FugaTensor wheel；wheel 携带所需共享库，运行时只有 FugaTensor Python 包。消费 SDK 不需要 ASK 源码或 LLVM 开发环境。

## 用户 API（`docs/用户API/`）

[用户 API 分类目录与快速示例](用户API/README.md)按接口类型说明签名、参数、返回值与支持边界：

- [张量与数据交换](用户API/01-张量与数据交换.md) · [基础运算](用户API/02-基础运算.md)
- [引擎与执行](用户API/03-引擎与执行.md) · [自动微分与自定义函数](用户API/04-自动微分与自定义函数.md)
- [Triton 算子接入](用户API/05-Triton算子接入.md) · [权重读写](用户API/06-权重读写.md)
- [IR 与降级](用户API/07-IR与降级.md)

## 系统架构（`docs/architecture/`）

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [架构总览](architecture/README.md) | FugaTensor 与 ASK 的职责划分、端到端调用流程与子系统入口 | 全局架构 |
| [01. 张量模型与追踪机制](architecture/01-张量模型与追踪机制.md) | CPU/CUDA 张量存储与 buffer 接口、不可变模板与结构缓存、按需 JIT、隔离追踪与 0-D 标量条件 | `tensor.py`、`ops.py`、`_lazy.py`、`_jit.py`、`tracing.py` |
| [02. 原生图分析与 C++ 边界](architecture/02-原生图分析与C++边界.md) | Python 与 C++ `_native` 扩展的分工、可达性与反向边界分析、`ir/transforms.py` 剪枝投影与无状态并发 | `cpp/graph.h`、`cpp/graph.cpp`、`cpp/bindings.cpp`、`ir/transforms.py` |
| [03. 自动微分与自定义算子](architecture/03-自动微分与自定义算子.md) | 运行时 `GradHistory` 反向图调度、`ft.fun` 自定义前向/反向、JVP/VJP 高阶求导与反向预检 | `autograd.py`、`function.py`、`ir/differentiate.py`、`ir/jvp.py` |
| [04. ASK 桥接与多后端执行](architecture/04-ASK桥接与多后端执行.md) | `Engine` 生命周期、无用输入端口裁剪、CPU `HostBackend`、原生 Graph 提交、自有 CUDA 与独立 Triton 调度、编译缓存与配置 | `engine.py`、`backends/`、`cpp/runtime.cpp`、`_ask/`、`triton.py`、`_triton_worker.py`、`_config.py`、`tools/bootstrap.py` |
| [05. 权重格式与导出](architecture/05-权重格式与导出.md) | 版本化 PyTree 权重目录、manifest/数组身份、原子导出、完整性校验与架构扩展边界 | `weights.py`、`schemas/weights-manifest-v1.schema.json` |

## Fuga IR 设计与变换（`docs/ir/`）

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [Fuga IR 总览](ir/README.md) | 独立框架级 IR 的职责、设备无关表示及从追踪到降级的流程 | `src/fugatensor/ir/` |
| [01. 对象模型与验证](ir/01-对象模型与验证.md) | `ScalarType`、`TensorType`、SSA `Value`、`Operation`、`Block`、`Region`、`Function`、`Module` 及结构化校验器 | `ir/types.py`、`ir/core.py` |
| [02. 控制流与可组合变换](ir/02-控制流与可组合变换.md) | `if`/`for`/`while` 区域语义、原语规则表（`abstract_eval` / `jvp` / `vjp`）及 `jvp`/`vjp` 图变换 | `ir/primitives.py`、`ir/jvp.py`、`ir/differentiate.py` |
| [03. 降级到 ASK](ir/03-降级到ASK.md) | 静态分支选择与有界循环展开、类型与算子映射、外部 Triton 架构绑定及 `fuga-to-ask.json` 溯源 | `lowering.py` |

## 底层编译引擎文档（`ask/docs/`）

以下相对链接供源码并列检出时浏览，不表示必须存在同级 ASK 目录；独立构建和包安装不依赖这些文档路径。

- [ASK 安装](../../ask/docs/安装.md)
- [ASK 文档目录](../../ask/docs/README.md) · [ASK 核心思想](../../ask/docs/核心思想.md)
- [图切分与多 Kernel 执行计划](../../ask/docs/08-图切分与多Kernel执行计划.md) · [多 kernel 计划 JSON 契约](../../ask/schemas/execution-plan-v1.schema.json)
- [外部编译器适配接入规范](../../ask/docs/09-外部编译器适配接入规范.md) · [worker 请求与 artifact 契约](../../ask/schemas/)
- [ASK IR 与数值语义](../../ask/docs/01-ASK-IR与数值语义.md) · [外部产物与执行接口](../../ask/docs/04-外部产物与执行接口.md)
- [常驻运行时与多级缓存](../../ask/docs/05-常驻运行时与多级缓存.md) · [FugaTensor 协同与演进路线](../../ask/docs/07-FugaTensor协同与演进路线.md)

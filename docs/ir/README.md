# Fuga IR

[返回项目说明](../../README.md) · [执行流程](../architecture.md) · [ASK 中的框架边界](../../../ask/docs/18-FugaTensor接入示例.md)

Fuga IR 是 Fuga 的统一计算表示。它保存值的类型和用途、分支的两个区域、循环携带的状态，以及反向图需要的正向值。基本类型和操作不包含 CPU/CUDA 放置或 GPU 架构；相同基础函数、shape 和 dtype 在 CPU/CUDA 上产生相同的 Fuga 图。执行时，再结合设备选择把能交给 ASK 的静态图降成 ASK IR。

```text
Python 张量表达式
  -> Fuga IR
  -> 框架变换：静态控制流处理、backward 所需的内部 VJP
  -> ASK IR
  -> ASK 优化与 CPU/CUDA lowering
  -> 设备执行
```

| 文档 | 回答的问题 |
|---|---|
| [对象模型与验证](01-对象模型与验证.md) | Value、Operation、Block、Region 怎样连接，怎样发现错误的图 |
| [控制流与自动微分](02-控制流与自动微分.md) | 怎样保留分支和循环，怎样生成基础反向图，哪些部分还不能执行 |
| [降到 ASK](03-降到ASK.md) | 怎样把独立的 Fuga 图变成 ASK 图，怎样保留外部源码和检查两层产物 |
| [JIT、backward 与 fun 类](04-JIT与自定义梯度提案.md) | 张量历史、叶子梯度、显式输出梯度、fun 自定义反向、Python 图中断与 C++ 分工 |
| [原子算子与可组合变换](05-原子算子与可组合变换.md) | 可执行的抽象求值、VJP/JVP 规则怎样共同生成可继续变换的基础图 |

当前支持常量、splat、加法、乘法、cast、外部调用、fun 调用、返回及结构化 region。减法、取负和平方用基础操作组成，保留求导路径。Tensor.backward 根据真实执行历史生成反向计算，create_graph 可为已知基础规则继续建立高阶图；不透明第三方运算会提前拒绝。C++ 核心负责依赖与反向路径分析。Python 零维 Tensor 条件通过图中断选择实际片段，反向沿实际路径连接；直接提交的 IR region 仍只能静态处理。

跨设备统一不表示所有外部实现都能在任意设备执行。Triton 外部调用可以保存 CUDA 实现描述，但具体 GPU 架构在降到 ASK 时才绑定；如果选择的设备没有对应实现，就给出明确错误。Fuga 的逻辑值不会因此变成另一种设备专用 Value。

`module.to_text()` 输出可读的 Fuga 文本，供检查和测试使用。它不是 ASK MLIR，也没有声明成稳定的文本导入或跨版本存档格式。编译与变换的输入始终是结构化对象。

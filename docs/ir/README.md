# Fuga IR 总览

目录导航：[文档首页](../README.md) · 系统架构：[架构总览](../architecture/README.md) · 底层引擎：[ASK IR 与数值语义](../../../ask/docs/01-ASK-IR与数值语义.md)

Fuga IR（`src/fugatensor/ir/`）是 FugaTensor 内部独立的中间表示。它既不包装 `ask.Value`，也不靠拼接 MLIR 字符串来混日子，而是用自包含的类型、SSA 值、操作和嵌套区域对象承载框架层的所有图分析与图变换。

<a id="ir-design"></a>

## 1. 为什么在 ASK IR 之上还要有一层 Fuga IR？

底层 [ASK IR](../../../ask/docs/01-ASK-IR与数值语义.md) 面向的是**已经准备就绪、可以直接生成原生代码的单基本块静态计算图**；而在用户写下的 Python 函数抵达那个阶段之前，框架层还有很多“静态后端不该操心”的事情要处理：

- **保留未展开的结构化控制流**：显式构造或中间生成的 `fuga.if`、`fuga.for`、`fuga.while` 需要携带嵌套 `Region` 与块参数，在静态化简或微分时按结构处理；
- **保留自定义求导边界**：[`ft.fun`](../architecture/03-自动微分与自定义算子.md#custom-fun) 在追踪期必须保持为一个独立的 `fuga.call` 边界节点（连同 `saved` 中间张量输出端口），直到 `vjp` 阶段才替换为用户的 `backward` 子图；
- **设备无关性（Device Neutrality）**：在 CPU 和 CUDA 上追踪同一个函数、相同 `(shape, dtype)` 的输入，产出的 Fuga IR 一模一样——Fuga 的类型与 SSA 值不包含设备放置或显卡 `sm_xx` 架构号，外部 Triton 算子的物理架构号延迟到降级阶段才绑定。

```text
Python 追踪 (tracing.py)
  -> Fuga IR Module (设备无关、含 Region 与 fuga.call)
  -> 框架级变换：C++ 活跃性分析与剪枝 (transforms.py) / JVP & VJP 求导 (jvp.py, differentiate.py)
  -> 降级桥接 (lowering.py)：静态 if 选择、有界 for 展开、架构绑定与 fuga-to-ask.json 溯源
  -> ASK IR Module -> ASK 优化、代码生成与常驻执行
```

<a id="ir-object-model"></a>

## 2. 结构化对象模型与严格验证

Fuga IR 由 `TensorType`、SSA `Value`、`Operation`、`Block`、`Region`、`Function` 和 `Module` 七类核心结构组成，所有变换都在内存对象图上直接进行；`module.to_text()` 只用于生成人类可读的调试与快照文本（如编译目录中的 `fuga.ir`），不把文本反序列化当成编译通道。

任何由追踪器、手动 `FunctionBuilder` 或自动微分产出的 `Module`，在进入变换和降级前都会经过 `verify_module` 的支配关系、类型匹配与区域终结符检查，确保错误在源头暴露。

工程细节：[01. 对象模型与验证](01-对象模型与验证.md) · [原生图分析与 C++ 边界](../architecture/02-原生图分析与C++边界.md)

<a id="ir-control-transforms"></a>

## 3. 结构化控制流与可组合微分（JVP / VJP）

Fuga IR 把每个基础算子的**抽象类型推导（`abstract_eval`）**、**前向切线规则（`jvp`）**与**反向余切规则（`vjp`）**统一收敛在 `ir/primitives.py` 的原语注册表中：

- 算术与转换建立在五种最小闭合基础算子（`constant`、`splat`、`add`、`mul`、`cast`）之上，`subtract`、`negative`、`square` 等高层运算在追踪期直接由它们组合而成；
- `ir.jvp(module)` 与 `ir.vjp(module)` 输出的依然是合法的 Fuga IR `Module`，因此 `vjp(vjp(module))`、`jvp(vjp(module))` 或 `vjp(jvp(module))` 可以像搭积木一样任意嵌套组合，直接支撑运行时 `create_graph=True` 的二阶、三阶导数。

工程细节：[02. 控制流与可组合变换](02-控制流与可组合变换.md) · [运行时自动微分与高阶求导](../architecture/03-自动微分与自定义算子.md#higher-order)

<a id="ir-lowering-boundary"></a>

## 4. 降级到 ASK IR 与跨层溯源

`lower_to_ask` 是 FugaTensor 与 ASK 之间的唯一翻译关口：它先消解静态已知的 `fuga.if` 并展开有界静态 `fuga.for`（对无法静态消解的动态 `if`/`while` 明确报错拒绝，绝不拿运行时偶然数据冒充编译期常量），随后将每个 Fuga `Value` 一一映射为 ASK `Value`，并输出完整的 `fuga-to-ask.json` 映射表供调试核对。

工程细节：[03. 降级到 ASK](03-降级到ASK.md) · [ASK 桥接与多后端执行](../architecture/04-ASK桥接与多后端执行.md)

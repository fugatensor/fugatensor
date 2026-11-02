# Fuga IR 总览

目录导航：[文档目录](../README.md) · 接口用法：[IR 与降级 API](../API/07-IR与降级.md) · 系统架构：[架构总览](../architecture/README.md) · 底层引擎：[ASK IR 与数值语义](../../../ask/docs/01-ASK-IR与数值语义.md)

Fuga IR（`src/fugatensor/ir/`）是 FugaTensor 内部的中间表示，使用独立的类型、SSA 值、操作和嵌套区域对象承载框架层的图分析与变换，不依赖后端桥接层的值对象或 MLIR 字符串。

<a id="ir-design"></a>

## 1. 设计职责与设备无关性

底层 [ASK IR](../../../ask/docs/01-ASK-IR与数值语义.md) 面向可直接生成目标代码的单基本块静态计算图；在降级到 ASK 之前，Fuga IR 负责在框架层处理三类信息：

- **结构化控制流**：通过嵌套 `Region` 与块参数保留未展开的 `fuga.if`、`fuga.for`、`fuga.while`，供静态化简或微分按结构处理；
- **自定义求导边界**：[`ft.fun`](../architecture/03-自动微分与自定义算子.md#custom-fun) 在追踪期保留为带 `saved` 中间值输出端口的 `fuga.call` 节点，到 `vjp` 阶段再替换为对应的 `backward` 子图；
- **设备无关表示**：Fuga 的类型与 SSA 值不包含设备放置或显卡 `sm_xx` 架构号；同一函数、相同 `(shape, dtype)` 的输入在 CPU 和 CUDA 上生成相同的 Fuga IR，外部 Triton 算子的物理架构号延迟到降级阶段绑定。

```text
Tensor 自动组图 (_lazy.py) / 自定义函数隔离追踪 (tracing.py)
  -> Fuga IR Module (设备无关、含 Region 与 fuga.call)
  -> 框架级变换：C++ 活跃性分析与剪枝 (transforms.py) / JVP & VJP 求导 (jvp.py, differentiate.py)
  -> 降级桥接 (lowering.py)：静态 if 选择、有界 for 展开、架构绑定与 fuga-to-ask.json 溯源
  -> ASK IR Module -> ASK 优化、代码生成与常驻执行
```

<a id="ir-object-model"></a>

## 2. 对象模型与验证

Fuga IR 由 `ScalarType`、`TensorType`、SSA `Value`、`Operation`、`Block`、`Region`、`Function` 和 `Module` 组成，所有图变换直接在内存对象结构上执行。`Module.verify()` 检查支配关系、类型匹配与区域终结符，内部缓存模板在校验后冻结。引擎按模板结构与连接关系查询缓存，未命中才组合整图并调用 `module.to_text()` 保存调试文本，命中不重新打印完整 IR。

工程细节：[01. 对象模型与验证](01-对象模型与验证.md) · [原生图分析与 C++ 边界](../architecture/02-原生图分析与C++边界.md)

<a id="ir-control-transforms"></a>

## 3. 结构化控制流与可组合微分（JVP / VJP）

`ir/primitives.py` 集中定义各基础算子的抽象类型推导（`abstract_eval`）、前向切线规则（`jvp`）与反向余切规则（`vjp`）：

- 算术、类型转换和形状变换建立在 `constant`、`splat`、`add`、`mul`、`cast`、`reshape` 六种基础算子之上，`subtract`、`negative`、`square` 在追踪期由它们组合展开；
- `ir.jvp(module)` 与 `ir.vjp(module)` 输出合法的 Fuga IR `Module`，支持 `vjp(vjp(module))`、`jvp(vjp(module))`、`vjp(jvp(module))` 嵌套组合，支撑运行时 `create_graph=True` 的高阶求导。

工程细节：[02. 控制流与可组合变换](02-控制流与可组合变换.md) · [运行时自动微分与高阶求导](../architecture/03-自动微分与自定义算子.md#higher-order)

<a id="ir-lowering-boundary"></a>

## 4. 降级到 ASK IR 与跨层溯源

`lower_to_ask` 先消解编译期已知的静态 `fuga.if` 并展开有界静态 `fuga.for`（对未消解的动态 `if`/`while` 区域报错拒绝），随后将每个 Fuga `Value` 映射为 ASK `Value`，并生成 `fuga-to-ask.json` 映射表。

工程细节：[03. 降级到 ASK](03-降级到ASK.md) · [ASK 桥接与多后端执行](../architecture/04-ASK桥接与多后端执行.md)

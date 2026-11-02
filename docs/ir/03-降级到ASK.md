# 03. 降级到 ASK

返回 IR 总览：[降级到 ASK IR 与跨层溯源](README.md#ir-lowering-boundary) · 相关细节：[对象模型与验证](01-对象模型与验证.md) · [控制流与可组合变换](02-控制流与可组合变换.md) · [ASK 桥接与多后端执行](../architecture/04-ASK桥接与多后端执行.md) · [ASK IR 与数值语义](../../../ask/docs/01-ASK-IR与数值语义.md)

[`lower_to_ask`](../../src/fugatensor/lowering.py) 接收 Fuga `Module` 与目标设备信息，返回 FugaTensor 内部 `_ask.Module` 图描述；调用其 `to_mlir()` 可得到提交给 ASK C++ 编译入口的文本。可选的 `source_map` 字典接收逐值溯源记录。Fuga IR 本身不依赖这些后端对象，ASK 也不提供独立 Python 包。

<a id="static-simplification"></a>

## 1. 静态控制流化简与动态区域拦截

向单基本块 ASK 图翻译前，降级器调用 `ir.specialize(module, max_unroll=10000)`，在新模块上展开静态控制流：

- **内联 `fuga.call`**：将调用的前向子图展开到主块，保留相应结果映射；死结果由后续剪枝消除。
- **静态 `fuga.if` 选择**：条件必须能确定为编译期 `bool`，只克隆选中的分支。
- **有界 `fuga.for` 展开**：按 `lower`、`upper`、`step` 展开循环，所有展开的迭代共用 `max_unroll` 预算；超限直接报错，克隆指令保留 `origin_id` 并获得不冲突的局部 ID。
- **动态区域拒绝**：依赖运行时输入的 `fuga.if` 或任何 `fuga.while` 抛出 `StaticControlFlowError`，不猜测分支。普通 Python 动态分支由前端 [0-D 标量执行边界](../architecture/01-张量模型与追踪机制.md#graph-break)分段处理。

`specialize` 在展开前后调用 `Module.verify()`。未具体化的符号形状不能进入 ASK 静态降级；后端或操作不受支持时也直接报错。

<a id="type-and-op-mapping"></a>

## 2. 类型与基础操作映射

内部图桥接层逐条生成真正的 ASK MLIR。下表中的类型和构建器均位于 `fugatensor._ask`，不是另一套用户 API：

| Fuga IR | ASK 描述或操作 | 说明 |
| --- | --- | --- |
| `ScalarType(dtype)` | 内部 `ScalarType(dtype)` | 对应 MLIR 标量类型；`bool` 编码为 `i1`。 |
| `TensorType(shape, dtype)` | 内部 `TensorType(shape, dtype)` | 保留静态形状；0-D 张量 `()` 与标量类型分别表示，空张量也保留维度。 |
| `fuga.constant` | `graph.constant(value, dtype=...)` | 按[单步 RNE 规则](../../../ask/docs/01-ASK-IR与数值语义.md#types-and-precision)编码浮点常量。 |
| `fuga.splat` | `graph.splat(scalar, shape)` | 显式将标量填充为指定形状张量，包括 0-D 张量。 |
| `fuga.add` / `fuga.mul` | `graph.add(lhs, rhs)` / `graph.mul(lhs, rhs)` | 同形状、同类型运算，无隐式广播或提升。 |
| `fuga.cast` | `graph.cast(operand, dtype=...)` | 生成显式 `ask.cast`，保留原标量或张量形状。 |
| `fuga.return` | `graph.return_(...)` → `func.return` | 支持多输出及异构形状、类型。 |

<a id="external-and-sourcemap"></a>

## 3. 外部实现绑定与跨层溯源

图中包含 `fuga.external` 时，降级器根据 `ExternalSpec` 生成 `ask.semantics`、`ask.implementation` 和调用它们的 `ask.external`，完整契约见 [ASK 外部声明](../../../ask/docs/01-ASK-IR与数值语义.md#external-declaration)。

- 目标与实现声明必须兼容，CUDA 实现需具体整数架构号。Triton 的 `arch` 在这里由所选设备补齐；通用降级器不识别特定 DSL。
- `same_as` 形状规则保留，Fuga 的 `explicit` 规则转换为 ASK 的 `fixed`；每个输出类型由已验证的 Fuga 类型确定。
- 张量参数通过 `input:N` / `output:N` 绑定，普通标量保存声明的启动值，编译期配置保留递归 JSON。`effects` 非空的外部操作当前不能降级执行。
- 每个外部节点保留为独立 kernel，自有区域继续由 ASK 优化。降级本身不启动 worker，也不分配设备存储。

`source_map` 的结构为 `{"schema_version": 1, "values": [...]}`。每个值记录函数名、Fuga 名称与 `value_id`、`origin_value_id`、ASK 名称与 `value_id`。多函数输入使用函数名前缀，避免 ASK 模块内的身份冲突。

`Engine` 还补充 `input_bindings`，记录原始输入序号与裁剪后的物理输入序号，然后将映射写入 `fuga-to-ask.json`。这些记录把 Fuga 追踪值与 ASK 准备、分区报告连接起来；Fuga 的 IR 结构与微分规则仍由框架维护。

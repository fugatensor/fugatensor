# 03. 降级到 ASK

返回 IR 总览：[降级到 ASK IR 与跨层溯源](README.md#ir-lowering-boundary) · 相关细节：[对象模型与验证](01-对象模型与验证.md) · [控制流与可组合变换](02-控制流与可组合变换.md) · [ASK 桥接与多后端执行](../architecture/04-ASK桥接与多后端执行.md) · [ASK IR 与数值语义](../../../ask/docs/01-ASK-IR与数值语义.md)

`src/fugatensor/lowering.py` 中的 `lower_to_ask` 接收通过校验的 Fuga `Module` 与目标设备信息，返回新的 `ask.Module` 及逐值溯源结果 `LoweringResult`。

<a id="static-simplification"></a>

## 1. 静态控制流化简与动态区域拦截

向单基本块 ASK 图翻译前，降级器先执行静态控制流化简（`simplify_static_control_flow`）：

- **内联前向 `fuga.call`**：前向执行路径中若包含 `ft.fun` 调用，将其前向子图的公开输出内联进主块（丢弃仅反向使用的 `saved` 输出端口）。
- **静态 `fuga.if` 选择**：若条件来自编译期已知的 `bool` 常量，仅内联命中的 `then` 或 `else` 区域，丢弃未命中分支。
- **有界静态 `fuga.for` 展开**：按 `range(start, stop, step)`（默认上限 `max_unroll_steps = 256`）将循环体展开为线性 SSA 指令序列，克隆指令保留原指令的 `origin_id`。
- **未决动态区域拒绝**：若化简后仍存在依赖运行时输入的 `fuga.if` 或任何 `fuga.while`，直接抛出 `LoweringError`；运行时动态分支由前端 [0-D 标量 Graph Break](../architecture/01-张量模型与追踪机制.md#graph-break) 分段处理。

<a id="type-and-op-mapping"></a>

## 2. 类型与基础操作映射

静态化简并经 `verify_module` 复核后，调用 `ask.Module` 构建器逐条映射：

| Fuga IR | ASK IR | 说明 |
| --- | --- | --- |
| `TensorType((), dtype)` | `ask.ScalarType(dtype)` | 0-D 标量对应 ASK 标量类型（`bool`/`i32`/`i64`/`f32`/`f16`/`bf16`）。 |
| `TensorType(shape, dtype)` | `ask.TensorType(dtype, shape)` | 非零维张量（含空张量）对应 ASK 静态形状张量。 |
| `fuga.constant` | `builder.constant(value, type)` | 由 ASK 前端按[单步 RNE 规则](../../../ask/docs/01-ASK-IR与数值语义.md#types-and-precision)编码字面量。 |
| `fuga.splat` | `builder.splat(scalar, shape)` | 目标为 0-D 标量时直接复用原标量 `ask.Value`。 |
| `fuga.add` / `fuga.mul` | `builder.add(lhs, rhs)` / `builder.mul(lhs, rhs)` | 同 shape、同 dtype 逐元素运算。 |
| `fuga.cast` | `builder.cast(operand, target_type)` | 生成显式 `ask.cast`。 |
| `fuga.return` | `func.return` | 支持单输出或异构 shape / dtype 多输出。 |

若目标后端不支持或缺少架构参数，转换过程直接抛出异常，不返回不完整的 `ask.Module`。

<a id="external-and-sourcemap"></a>

## 3. 外部 Triton 实现绑定与 `fuga-to-ask.json` 溯源

图中包含 [`ft.triton.kernel`](../architecture/04-ASK桥接与多后端执行.md#cuda-and-triton) 生成的 `fuga.external` 时，降级器在此阶段绑定具体 GPU 架构：

1. 检查目标后端为 `"cuda"` 且已提供整数架构号 `arch`（如 `86`）；提交给 CPU 引擎时直接报错。
2. 在 `ask.Module` 顶层生成一对声明（见 [ASK 外部算子语义与实现分离](../../../ask/docs/01-ASK-IR与数值语义.md#external-declaration)）：
   - `ask.semantics`：声明逻辑输入/输出端口、形状规则（`same_as` 或 `explicit`）与副作用列表；多输入算子按操作数顺序声明端口，显式输出形状由 Fuga `TensorType` 进入 ASK `port_shapes`。
   - `ask.implementation`：填入用户 `@triton.jit` 源码快照、`entry`、`signature`、`source_bindings`、`constants`、`options`、`grid` 以及 `target={"backend": "cuda", "arch": arch, "warp_size": 32}`。
3. 在函数体中生成引用上述两个符号的 `ask.external` 指令。ASK 按[执行计划 v1](../../../ask/docs/08-图切分与多Kernel执行计划.md#plan-v1)将外部节点独立调度；降级器不把多输入或复杂算子改写成逐元素图。

降级器同时记录每个 Fuga `Value` 和 `Operation` 的跨层对应关系，由 `Engine` 写入编译产物目录下的 `fuga-to-ask.json`：

- **`value_map`**：记录每个 `fuga_value_id`（含 shape、dtype）对应的 `ask_value_name`（如 `%arg0`、`%2`）；
- **`op_origins`**：记录每条生成的 ASK 指令对应的 Fuga `origin_id`（包括 `fuga.for` 展开后生成的各轮指令）。

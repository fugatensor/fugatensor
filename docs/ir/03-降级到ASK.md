# 03. 降级到 ASK

返回 IR 总览：[降级到 ASK IR 与跨层溯源](README.md#ir-lowering-boundary) · 相关细节：[对象模型与验证](01-对象模型与验证.md) · [控制流与可组合变换](02-控制流与可组合变换.md) · [ASK 桥接与多后端执行](../architecture/04-ASK桥接与多后端执行.md) · [ASK IR 与数值语义](../../../ask/docs/01-ASK-IR与数值语义.md)

`src/fugatensor/lowering.py` 中的 `lower_to_ask` 是整个框架从“独立 Fuga IR”跨入“底层 [ASK IR](../../../ask/docs/01-ASK-IR与数值语义.md)”的唯一桥梁。它接收通过校验的 Fuga `Module`（以及目标后端/GPU 架构信息），返回全新的 `ask.Module` 与完整的逐值溯源结果 `LoweringResult`。

<a id="static-simplification"></a>

## 1. 静态控制流消解与未决分支拦截

在向单基本块 ASK 图翻译之前，降级器会先做一轮确定性的静态控制流化简（`simplify_static_control_flow`）：

- **内联无自定义反向残留的 `fuga.call`**：前向执行路径中若包含 `ft.fun` 调用，在此将其前向子图公开输出内联展开进主块（丢弃仅反向需要的额外 `saved` 输出端口）。
- **静态 `fuga.if` 选择**：若条件操作数来自编译期已知的 `bool` 常量，直接把命中的 `then` 或 `else` 区域内联进来，彻底丢弃未命中分支（连同未命中分支里可能包含的非法或异构设备外部算子也不会误留）。
- **有界静态 `fuga.for` 展开**：按确定的 `range(start, stop, step)`（默认展开上限 `max_unroll_steps = 256` 防止指数级代码膨胀）把循环体逐轮展开为线性 SSA 数据流，每轮产出的克隆指令保留原指令的 `origin_id` 以便溯源。
- **动态未决控制流明确拒绝**：如果化简后仍然残留依赖运行时输入的 `fuga.if` 或任何 `fuga.while`，降级器立即抛出 `LoweringError`——运行时动态分支应当走前端的 [0-D 标量 Graph Break](../architecture/01-张量模型与追踪机制.md#graph-break) 分段机制，绝不允许在降级层偷取某次输入把动态图伪装成静态图。

<a id="type-and-op-mapping"></a>

## 2. 类型与基础操作的一一映射

静态化简并经 `verify_module` 复核后，降级器调用 `ask.Module` 构建器逐条翻译：

| Fuga IR 概念 | 降级后的 ASK IR 结构 | 关键细节 |
| --- | --- | --- |
| `TensorType((), dtype)` | `ask.ScalarType(dtype)` | 0-D 标量直接对应 ASK 标量类型（`bool`/`i32`/`i64`/`f32`/`f16`/`bf16`）。 |
| `TensorType(shape, dtype)` | `ask.TensorType(dtype, shape)` | 非零维张量（含 0 元素空张量）一一对应 ASK 静态形状张量。 |
| `fuga.constant` | `builder.constant(value, type)` | 交由 ASK 前端按[一步到位 RNE 规则](../../../ask/docs/01-ASK-IR与数值语义.md#types-and-precision)编码字面量。 |
| `fuga.splat` | `builder.splat(scalar, shape)` | 若目标是 0-D 标量则直接复用原标量 `ask.Value`。 |
| `fuga.add` / `fuga.mul` | `builder.add(lhs, rhs)` / `builder.mul(lhs, rhs)` | 保持严格的同 shape、同 dtype 逐元素语义。 |
| `fuga.cast` | `builder.cast(operand, target_type)` | 生成显式 `ask.cast`，遵循 ASK 饱和浮点转整数与 IEEE-754 精度转换语义。 |
| `fuga.return` | `func.return` | 支持单输出或多输出（异构 shape / dtype 均可）。 |

整个翻译过程具有**原子性**：如果遇到不支持的目标后端或架构缺失，异常直接在构建期间抛出，绝不向后端提交残缺的半张 `ask.Module`。

<a id="external-and-sourcemap"></a>

## 3. 外部 Triton 实现绑定与 `fuga-to-ask.json` 跨层溯源

当图中包含由 [`ft.triton.kernel`](../architecture/04-ASK桥接与多后端执行.md#cuda-and-triton) 生成的 `fuga.external` 时，降级器在此刻完成**延迟硬件绑定**：

1. 检查目标后端必须为 `"cuda"` 且当前 GPU 会话已提供具体整数架构号 `arch`（例如 `86`）；若把含 CUDA Triton 算子的图提交给 CPU 引擎，当场报出清晰错误。
2. 在 `ask.Module` 顶层先发射一对配套声明（详见 [ASK 外部算子语义与实现分离](../../../ask/docs/01-ASK-IR与数值语义.md#external-declaration)）：
   - `ask.semantics`：声明逻辑输入/输出端口、形状规则（`same_as` 或 `explicit`）与副作用列表；
   - `ask.implementation`：原封不动填入用户 `@triton.jit` 源码快照、`entry`、`signature`、`source_bindings`、`constants`、`options`、`grid` 以及刚刚绑定的 `target={"backend": "cuda", "arch": arch, "warp_size": 32}`。
3. 在函数体对应位置发射引用上述两个符号的 `ask.external` 调用。

与此同时，降级器会为每一个 Fuga `Value` 和每一条 `Operation` 记录精确的跨层映射（由 `Engine` 落盘为编译产物目录下的 `fuga-to-ask.json`）：

- **`value_map`**：记录每个 `fuga_value_id`（含 shape、dtype）映射到了哪个 `ask_value_name`（如 `%arg0`、`%2`）；
- **`op_origins`**：记录每条生成的 ASK 指令源自哪一个 Fuga `origin_id`——即便一条 `fuga.for` 循环体指令被展开成了 8 条 ASK 指令，通过 `origin_id` 也能瞬间追回原始 Fuga 节点，彻底打通两仓联合调试链路。

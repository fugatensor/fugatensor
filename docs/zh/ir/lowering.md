# 降级到 ASK

返回 IR 总览：[降级到 ASK IR 与跨层溯源](README.md#ir-lowering-boundary) · 相关细节：[对象模型与验证](model.md) · [控制流与可组合变换](transforms.md) · [ASK 桥接与多后端执行](../architecture/backend.md) · [ASK IR 与数值语义](../../../../ask/docs/zh/ir.md)

[`lower_to_ask`](../../../src/fugatensor/lowering.py) 接收 Fuga `Module` 和目标设备，返回内部 `_ask.Module`。其 `to_mlir()` 方法生成提交给 ASK C++ 的文本，`source_map` 参数可接收逐值映射。

降级后的图由内部 Compiler 桥接层提交给 ASK。准备结果分别保存 `ir`、`shape_snapshot`、`external_requests` 和 `diagnostics`；分区结果再提供 `execution_plan` 与 `step_sources`，代码生成结果按目标提供 `abi` 或 `cuda_artifact`。图转换、编译契约与诊断各自通过对应字段传递，字段用途见[编译阶段结果](../architecture/backend.md#compiler-results)。

API：[IR 与降级](../API/ir.md)

<a id="static-simplification"></a>

## 1. 静态控制流化简与动态区域拦截

向单基本块 ASK 图翻译前，降级器调用 `ir.specialize(module, max_unroll=10000)`，在新模块上展开静态控制流：

- **内联 `fuga.call`**：将调用的前向子图展开到主块，保留相应结果映射；死结果由后续剪枝消除。
- **静态 `fuga.if` 选择**：条件必须能确定为编译期 `bool`，只克隆选中的分支。
- **有界 `fuga.for` 展开**：按 `lower`、`upper`、`step` 展开循环，所有展开的迭代共用 `max_unroll` 预算；超限直接报错，克隆指令保留 `origin_id` 并获得不冲突的局部 ID。
- **动态区域拒绝**：数据依赖的 `fuga.if` 和所有 `fuga.while` 抛出 `StaticControlFlowError`。普通 Python 动态分支通过读取 [0-D 标量](../architecture/tensor_model.md#graph-break)分段执行。

`specialize` 在展开前后调用 `Module.verify()`。降级要求形状全部具体，操作在所选后端有实现，否则报错。

<a id="type-and-op-mapping"></a>

## 2. 类型与基础操作映射

内部模块 `fugatensor._ask` 按以下规则生成 ASK MLIR：

| Fuga IR | ASK 描述或操作 | 说明 |
| --- | --- | --- |
| `ScalarType(dtype)` | 内部 `ScalarType(dtype)` | 对应 MLIR 标量类型；`bool` 编码为 `i1`。 |
| `TensorType(shape, dtype)` | 内部 `TensorType(shape, dtype)` | 保留静态形状；0-D 张量 `()` 与标量类型分别表示，空张量也保留维度。 |
| `fuga.constant` | `graph.constant(value, dtype=...)` | 按[单步 RNE 规则](../../../../ask/docs/zh/ir.md#types-and-precision)编码浮点常量。 |
| `fuga.splat` | `graph.splat(scalar, shape)` | 将标量填充为指定形状张量，包括 0-D 张量。 |
| `fuga.add` / `fuga.mul` | `graph.add(lhs, rhs)` / `graph.mul(lhs, rhs)` | 要求形状、类型相同。 |
| `div`、`exp/log/sqrt`、`compare/select`、`broadcast_in_dim/transpose`、`reduce_sum/reduce_max`、`matmul` | 同名 `ask` 操作 | 从统一原语类型规则取得结果类型，桥接层保留静态属性。 |
| `fuga.cast` | `graph.cast(operand, dtype=...)` | 生成 `ask.cast`，保留原标量或张量形状。 |
| `fuga.reshape` | `graph.reshape(operand, shape)` | 生成 `ask.reshape` 和目标 `shape`；保留 dtype、元素数和连续元素次序。 |
| `fuga.stop_gradient` | 输入的 ASK 值 | 微分边界在 Fuga IR 阶段处理；数值降级直接复用输入映射，保持原值。 |
| `fuga.return` | `graph.return_(...)` → `func.return` | 支持多输出及异构形状、类型。 |

<a id="external-and-sourcemap"></a>

## 3. 外部实现绑定与跨层溯源

`fuga.external` 的 `ExternalSpec` 会转成 `ask.semantics`、`ask.implementation` 和引用它们的 `ask.external`，字段见 [ASK 外部声明](../../../../ask/docs/zh/ir.md#external-declaration)。

- 目标须与实现声明兼容，CUDA 实现要求具体整数架构号。所选设备在此补齐 Triton 的 `arch`。
- `same_as` 形状规则保留，Fuga 的 `explicit` 规则转换为 ASK 的 `fixed`；每个输出类型由已验证的 Fuga 类型确定。
- reshape 前后的值各自保留真实形状，外部请求中的端口形状与该节点实际接收的 Tensor 一致。
- 张量参数通过 `input:N` / `output:N` 绑定，普通标量保存声明的启动值，编译期配置保留递归 JSON。`effects` 非空的外部操作当前不能降级执行。
- 每个外部节点保留为独立 kernel，自有区域交给 ASK 优化。worker 调用和设备存储分配在后续编译、执行阶段完成。

`source_map` 的结构为 `{"schema_version": 1, "values": [...]}`。每个值记录函数名、Fuga 名称与 `value_id`、`origin_value_id`、ASK 名称与 `value_id`。多函数输入使用函数名前缀，避免 ASK 模块内的身份冲突。

`Engine` 补充 `input_bindings`，记录输入裁剪前后的序号，再将映射写入 `fuga-to-ask.json`，供追踪值与 ASK 诊断中的溯源记录关联查询。

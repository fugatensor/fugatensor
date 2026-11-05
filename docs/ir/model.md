# 对象模型与验证

返回 IR 总览：[对象模型与验证](README.md#ir-object-model) · 相关细节：[控制流与可组合变换](transforms.md) · [降级到 ASK](lowering.md) · [原生图分析与 C++ 边界](../architecture/native.md)

<a id="types-and-ssa"></a>

## 1. 类型与 SSA 值（`ir/types.py` / `ir/core.py`）

- `ScalarType(dtype)` 与 `TensorType(shape, dtype)`：
  - `dtype` 接受 `ft.bool`、`ft.int32`、`ft.int64`、`ft.fp32`、`ft.fp16`、`ft.bf16`，内部规范化为 `("bool", "i32", "i64", "f32", "f16", "bf16")` 编码；可微浮点集合定义为 `FLOAT_DTYPES = frozenset(("f16", "bf16", "f32"))`。
  - 标量打印为 `f32`、`i32` 等；0-D 张量 `TensorType((), "f32")` 打印为 `tensor<f32>`，两种类型分别表示。多维或空张量如 `(2, 3)`、`(0, 4)` 打印为 `tensor<2x3xf32>`、`tensor<0x4xf32>`。
  - 张量维度允许非负整数或未知维度 `None`，降级到 ASK 前须全部具体化。设备由执行引擎选择。
- SSA `Value`：
  - 每个 `Value` 具有唯一 `value_id`、打印短名、`ScalarType` 或 `TensorType`、所属 `Block`（`owner`）及定义算子 `producer`（块参数的 `producer` 为 `None`）。
  - 每个 `Value` 只定义一次，图变换与跨层映射均以 `value_id` 作为标识。

<a id="ops-and-regions"></a>

## 2. 操作、基本块、区域与模块层次

```text
Module (functions)
 └── Function (name, body, inputs, result_types)
      └── Region (blocks, parent)
           └── Block (arguments, operations)
                └── Operation (opcode, operands, results, attrs, regions, effects, id, origin_id)
```

当前支持的 `Operation.opcode` 列表：

| 类别 | 操作名 | 语义与属性 / 区域 |
| --- | --- | --- |
| **常量与广播** | `constant` | 无操作数，产生 `ScalarType` 值，`attrs["value"]` 已按声明精度舍入；浮点文本打印使用十六进制 `float.hex()` 保持精确值。 |
| | `splat` | 将标量操作数填充到 `attrs["shape"]` 指定的张量形状。 |
| **基础逐元素** | `add`、`mul` | 接收两个类型完全一致且非 `bool` 的标量或张量，产生同类型结果。 |
| | `div`、`exp`、`log`、`sqrt` | 浮点逐元素运算，输入和结果保持相同类型。 |
| | `compare`、`select` | 比较由 `predicate` 指定，返回同形状 bool；选择的条件为 bool、两个值类型相同。 |
| | `cast` | 转换为结果声明的 `dtype`，保留原标量或张量形状。 |
| **形状变换** | `reshape` | 一个张量输入，`attrs["shape"]` 给出具体目标形状；保持 dtype、元素总数和连续元素次序，支持 0-D 与空张量。 |
| | `broadcast_in_dim`、`transpose` | 广播保存 `shape/dimensions`，转置保存 `permutation`。 |
| **归约** | `reduce_sum`、`reduce_max` | `axes` 为递增且唯一的轴；全归约产生 ScalarType，部分归约产生 TensorType。rank > 0 且 axes 为空时保持 TensorType。 |
| **矩阵乘法** | `matmul` | 同批维度矩阵乘法，`transpose_a/b` 控制末两维转置；fp32 累加。 |
| **微分边界** | `stop_gradient` | 一个标量或张量输入，前向原样返回数值、类型和形状；JVP 切线与 VJP 余切停止传播，数值降级时直接映射输入值。 |
| **函数与外部算子** | `call` | `attrs["spec"]` 保存 `CallSpec` 及前向子图（如 [`ft.fun`](../architecture/autograd.md#custom-fun)），结果为公开前向输出，求导规则接收输入原值、前向输出与余切。 |
| | `external` | 调用第三方编译器算子，`attrs["spec"]` 保存 `ExternalSpec`（编译器 ID、源码原文、端口绑定、常量、grid 与副作用列表）。 |
| **结构化控制流** | `if` | 接收标量或 0-D 张量 `bool` 条件，含 `then` 与 `else` 两个单块 `Region`，两分支 `yield` 的结果类型列表必须一致。 |
| | `for` | 静态区间计数循环（`attrs` 含 `lower`、`upper`、`step`），循环体块参数依次为 `i64` 标量归纳变量与携带状态。 |
| | `while` | 含条件和循环体两个 `Region`：条件区域 `yield` 一个布尔值，循环体 `yield` 下一轮携带状态。 |
| **终结符** | `return`、`yield` | 分别作为顶层函数与嵌套区域的末尾终结指令。 |

`Operation.origin_id` 在克隆、循环展开或微分生成新节点时继承源节点 ID，最终记录到 [`fuga-to-ask.json`](lowering.md#external-and-sourcemap)。

<a id="module-verifier"></a>

## 3. 结构化校验器（`ir/core.py`）

`Module.verify()` 检查以下规则，违规时抛出 `ValueError` 或 `TypeError`：

1. **支配与作用域**：操作数必须是当前块的入口参数或此前定义的结果。区域通过块参数接收外部值。
2. **块结构与终结符**：每个区域包含单个非空块，`return` / `yield` 只能且必须位于块末尾，操作数类型与函数 `result_types` 或父控制流指令的结果约定逐一匹配。
3. **算子类型约束**：`add`/`mul` 拒绝 `bool` 或不一致类型，`splat` 要求标量输入，`reshape` 要求输入输出为具体形状、同 dtype、同元素数的张量；条件要求标量或 0-D `bool`，`for` 要求 `step != 0` 且携带类型不变，`call` 核对 `CallSpec` 的前向签名。

缓存模板校验后冻结，并保存结构键。再次追加操作或修改函数、块会报错，以保证结构键对应实际图内容。

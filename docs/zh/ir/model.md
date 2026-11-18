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
| **常量与坐标工厂** | `constant` | 无操作数，产生 `ScalarType` 值，`attrs["value"]` 已按声明精度舍入；浮点文本打印使用十六进制 `float.hex()` 保持精确值。 |
| | `splat` | 将标量操作数填充到 `attrs["shape"]` 指定的张量形状。 |
| | `iota` | 无操作数；生成具有 `attrs["shape"]` / `attrs["dtype"]`、沿 `attrs["dimension"]` 填充坐标下标的数值张量。 |
| **基础逐元素与位运算** | `add`、`mul`、`pow`、`floordiv`、`mod` | 接收同形状、同数值类型的标量或张量；`pow` / `floordiv` / `mod` 同时接受整数和浮点类型。 |
| | `div`、`exp`、`log`、`sqrt`、`sin`、`cos`、`tanh`、`erf`、`floor`、`ceil`、`round` | 浮点逐元素运算，输入与结果保持相同类型和形状。 |
| | `abs`、`maximum`、`minimum` | `abs` 接受数值操作数；`maximum`/`minimum` 支持全部六种类型。 |
| | `bitwise_not`、`bitwise_and`、`bitwise_or`、`bitwise_xor`、`shift_left`、`shift_right` | `bool`/`i32`/`i64` 按位逻辑运算与 `i32`/`i64` 上的确定性位移。 |
| | `compare`、`select` | 比较由 `predicate` 指定，返回同形状 bool；选择的条件为 bool、两个值类型相同。 |
| | `cast` | 转换为结果声明的 `dtype`，保留原标量或张量形状。 |
| **形状、切片与索引** | `reshape` | 一个张量输入，`attrs["shape"]` 给出具体目标形状；保持 dtype、元素总数和连续元素次序，支持 0-D 与空张量。 |
| | `broadcast_in_dim`、`transpose` | 广播保存 `shape/dimensions`，转置保存 `permutation`。 |
| | `slice`、`pad`、`concatenate` | `slice` 保存 `starts/stops/steps`；`pad` 接收 `(value, pad_value)`（标量或 0-D 张量）与 `pad_low/pad_high/interior`；`concatenate` 沿 `axis` 拼接 >= 1 个同类型操作数。 |
| | `gather`、`scatter_add` | `gather` 沿 `axis` 用 `indices` 读取 `params`；`scatter_add` 由 `base` 初始化并沿 `axis` 在 `indices` 处累加 `updates`。 |
| **归约** | `reduce_sum`、`reduce_max`、`reduce_min`、`reduce_prod`、`reduce_any`、`reduce_all` | `axes` 为递增且唯一的轴（`reduce_max/min` 要求非空域；`any/all` 接收 `bool`；全归约产生 `ScalarType`）。 |
| | `argmax`、`argmin` | 沿非空单 `axis` 归约出 `i64` 首次极值索引（一维输入产生 `ScalarType`）。 |
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

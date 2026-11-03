返回API：[分类目录](README.md#categories) · 相关接口：[自动微分](自动微分与自定义函数.md) · 设计说明：[Fuga IR 总览](../ir/README.md)

# IR 与降级

这些接口用于手写计算图和图变换。构图与实际 Tensor 求值分开，执行入口见[设备与执行](设备与执行.md#execution)。

<a id="ir-builders"></a>

## 类型和构图

| 接口 | 用法 |
| --- | --- |
| `ft.ir.ScalarType(dtype)` | 标量类型；dtype 可传 `ft.int32`、`ft.fp32` 等六种类型对象。 |
| `ft.ir.TensorType(shape, dtype)` | 张量类型；IR 中可用 `None` 表示未知维，但执行降级要求具体形状。`shape=()` 仍是张量类型。 |
| `ft.ir.Type` | `ScalarType \| TensorType` 类型别名，用于接口类型标注。 |
| `ft.ir.DTYPES`、`FLOAT_DTYPES` | IR 基础类型编码 tuple 与可微浮点编码集合；内部使用 `i32`、`f32` 等字符串。 |
| `ft.ir.Module()` | 创建模块。 |
| `module.graph(name, *, inputs, results=None)` | 创建并返回 `Function`；`inputs` 是有序的名称→类型映射，`results` 可显式指定结果类型列表。 |
| `graph.inputs[name]` | 获取输入 SSA `Value`。 |
| `graph.constant(value, *, dtype=..., id=None)` | 构造标量常量，dtype 可省略，默认 int32。 |
| `graph.splat(value, shape, *, id=None)` | 将标量扩展为张量。 |
| `graph.add(a, b, *, id=None)`、`graph.mul(a, b, *, id=None)` | 类型和形状匹配的基础运算。 |
| `graph.cast(value, *, dtype, id=None)` | 显式类型转换。 |
| `graph.reshape(value, shape, *, id=None)` | 保持张量 dtype 和元素次序，按静态形状重排相同数量的元素。 |
| `graph.return_(*values)` | 结束函数并指定返回值。 |
| `module.verify()` | 校验模块，成功返回自身，非法图报错。 |
| `module.to_text()` | 输出 Fuga IR 文本。 |

```python
import fugatensor as ft

module = ft.ir.Module()
graph = module.graph("square", inputs={"x": ft.ir.TensorType((3,), ft.fp32)})
x = graph.inputs["x"]
graph.return_(graph.mul(x, x))
module.verify()
print(module.to_text())
```

IR 内部用字符串编码 dtype，构建器接受 `ft.DType` 类型对象。图由 `Value`、`Operation`、`Block`、`BlockBuilder`、`Region`、`Function` 组成。

| 控制流接口（均可传 `id=None`） | 回调签名与结果 |
| --- | --- |
| `graph.if_(condition, operands, then_fn, else_fn)` | 两个分支均为 `callback(builder, *operands)`，返回类型须相同；条件为标量 bool 或 0-D bool 张量。 |
| `graph.for_(lower, upper, step, initial_values, body_fn)` | `body_fn(builder, index, *carried)`；边界与非零步长为 Python i64 范围整数，`index` 为 i64 SSA 值，返回值类型与初始值一致。 |
| `graph.while_(initial_values, condition_fn, body_fn)` | 两个回调均为 `callback(builder, *carried)`；条件返回一个 bool 值，循环体返回与初始值同类型的值。 |

回调通过传入的 builder 构图，返回 Value 或 Value 序列；也可显式调用 `builder.yield_(...)` 并返回 `None`，两种结束方式不能混用。区域使用传入的块参数，不能直接捕获外层 SSA 值。构图允许动态控制流，但 ASK 降级当前只接受静态 `if` 和可展开的 `for`，不接受 `while`。区域规则见[对象模型](../ir/对象模型与验证.md#ops-and-regions)，执行限制见[静态化简](../ir/降级到ASK.md#static-simplification)。

用 `ft.ir.ExternalSpec` 和 `graph.external(*operands, spec=..., result_types=None, id=None)` 构造外部节点。Triton kernel 的便捷入口见[算子注册](Triton算子接入.md)。

<a id="ir-transforms"></a>

## 图变换

| 接口 | 返回值与参数含义 |
| --- | --- |
| `ft.ir.specialize(module, *, max_unroll=10000, expand_calls=True)` | 返回化简静态控制流的新 Module；展开上限用于限制静态循环。 |
| `ft.ir.prune(module)` | 对平坦图删除死代码，返回新 Module；含控制流区域时须先 `specialize`。 |
| `ft.ir.project_outputs(module, indices, *, name_suffix="", max_unroll=10000)` | 静态化简后，从首个函数按结果索引选择输出并裁剪无关图；返回只含该函数的新 Module，通常用于单函数模块。 |
| `ft.ir.analyze_function(function, selected=())` | 分析平坦函数，返回 `(operations, numbered_values, analysis)`；不递归分析控制流区域。`selected` 为选定输入索引，`analysis` 含活跃值与正反向算子调度索引。 |
| `ft.ir.jvp(module, *, argnums=None, max_unroll=10000)` | 新 Module 接收原始输入及选定输入的切线，输出原始结果及对应结果切线。 |
| `ft.ir.vjp(module, *, argnums=None, max_unroll=10000)` | 新 Module 接收原始输入及每个结果的余切种子，输出选定输入的梯度。 |

`argnums` 接受单个输入索引或无重复索引 list/tuple，默认选择全部浮点输入。JVP/VJP 当前要求单函数、浮点输出及具体的被微分张量形状；缺少微分规则的活跃外部节点会报错。变换规则见[可组合微分](../ir/控制流与可组合变换.md#jvp-and-vjp)。

<a id="lowering"></a>

## 降级到 ASK

```text
ft.lower_to_ask(module, *, target=None, max_unroll=10000, source_map=None)
```

接收 Fuga `Module`，返回内部 ASK 图桥接对象；调用其 `to_mlir()` 可以取得提交给 ASK C++ 的文本。`source_map` 若为 dict，会填入逐值溯源信息；`target` 可为 `{"backend": "host"}` 或 `{"backend": "cuda", "arch": 86}`。外部实现固定的目标与请求目标必须兼容。

```python
import fugatensor as ft

module = ft.ir.Module()
graph = module.graph("identity", inputs={"x": ft.ir.TensorType((2,), ft.fp32)})
graph.return_(graph.inputs["x"])
mapping = {}
lowered = ft.lower_to_ask(module, target={"backend": "host"}, source_map=mapping)
print(lowered.to_mlir())
```

图的编译和执行使用 [Tensor 求值入口](设备与执行.md#execution)；直接编译 ASK IR 可用 [ASK C++ 编译接口](../../../ask/docs/用户API.md#compiler-api)。该接口分别返回准备、分区和代码生成结果，形状快照、执行计划、ABI 与产物通过具名字段读取；诊断用于查看阶段、溯源和优化统计。框架内部的对应关系见[编译阶段结果](../architecture/ASK桥接与多后端执行.md#compiler-results)。

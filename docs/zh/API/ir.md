返回分类目录：[API 索引](README.md#categories) · IR 模型：[对象模型与验证](../ir/model.md) · 控制流与微分：[可组合变换](../ir/transforms.md) · 后端降级：[降级到 ASK](../ir/lowering.md)

# IR 与降级

Fuga IR 是独立于后端的纯静态 SSA 中间表示。图构建、图变换与降级编译相互解耦。常规用户通过高层 Tensor 延迟追踪自动生成 IR，高级开发人员与算子扩展者可直接操作 Fuga IR 构造与变换计算图。

---

<a id="ir-types-and-builders"></a>

## 类型系统与图构建

| 接口 | 说明 |
| --- | --- |
| `ft.ir.ScalarType(dtype)` | 标量类型。dtype 接收 `ft.bool`、`ft.int32`、`ft.int64`、`ft.fp32`、`ft.fp16`、`ft.bf16`。 |
| `ft.ir.TensorType(shape, dtype)` | 张量类型。`shape` 为维度 tuple，允许 `None` 占位，静态降级时须为确定尺寸。 |
| `ft.ir.Type` | `ScalarType | TensorType` 的类型别名。 |
| `ft.ir.DTYPES`、`ft.ir.FLOAT_DTYPES`、`ft.ir.INT_DTYPES` | IR 支持的数据类型编码 tuple（内部归一化为 `"i32"`、`"f32"` 等）、浮点可微编码集合以及有符号整数编码集合（`("i32", "i64")`）。 |
| `ft.ir.Module()` | 创建空的 Fuga IR 模块容器。 |
| `module.graph(name, *, inputs, results=None)` | 在模块内创建并返回 `Function` 构建器。`inputs` 为有序 dict（变量名 $\to$ `ft.ir.Type`），`results` 可选指定返回值类型列表。 |
| `graph.inputs[name]` | 获取命名输入 SSA `Value`。 |
| `graph.constant(value, *, dtype=..., id=None)` | 构造标量常量 SSA 值。 |
| `graph.splat(value, shape, *, id=None)` | 将标量广播为指定 shape 的张量 SSA 值。 |
| `graph.iota(shape, dimension, *, dtype='f32', id=None)` | 生成数值张量沿 `dimension` 轴的坐标索引。 |
| `graph.add` / `sub` / `mul` / `div` / `floordiv` / `mod` / `pow` / `maximum` / `minimum` | 逐元素二元算术与极值操作。 |
| `graph.exp` / `log` / `sqrt` / `sin` / `cos` / `tanh` / `erf` / `abs` / `floor` / `ceil` / `round` | 初等数学与取整 SSA 操作。 |
| `graph.bitwise_not` / `bitwise_and` / `bitwise_or` / `bitwise_xor` / `shift_left` / `shift_right` | 整数/布尔按位运算与位移操作。 |
| `graph.cast(value, *, dtype, id=None)` | 显式类型转换。 |
| `graph.reshape` / `transpose` / `broadcast_in_dim` / `slice` / `pad` / `concatenate` | 布局重排、切片、填充与变长拼接原语。 |
| `graph.gather(params, indices, axis)` / `scatter_add(base, indices, updates, axis)` | 沿 `axis` 聚集查找与原子散布累加原语。 |
| `graph.reduce_sum` / `reduce_max` / `reduce_min` / `reduce_prod` / `reduce_any` / `reduce_all` / `argmax` / `argmin` | 归约与极值位置查找；全标量归约产生 `ScalarType`。 |
| `graph.matmul(a, b, transpose_a=False, transpose_b=False)` | 二维及以上矩阵乘法。 |
| `graph.stop_gradient(value, *, id=None)` | 截断梯度流动标记；降级到 ASK 时直接映射为前向输入。 |
| `graph.return_(*values)` | 终止函数构建并指定返回值 SSA 列表。 |
| `module.verify()` | 结构合法性校验；非法时抛出 `ValueError`，成功返回 `self`。 |
| `module.to_text()` | 导出标准的 Fuga IR 文本。 |

### 构图示例

```python
import fugatensor as ft

# 1. 创建模块与计算图函数
module = ft.ir.Module()
graph = module.graph("elementwise_square", inputs={"x": ft.ir.TensorType((3,), ft.fp32)})
x = graph.inputs["x"]

# 2. 构建 SSA 算子
sq = graph.mul(x, x)
graph.return_(sq)

# 3. 校验并打印 IR 文本
module.verify()
print(module.to_text())
```

**输出：**

```text
fuga.module {
  fuga.func @elementwise_square(%x: tensor<3xf32>) -> (tensor<3xf32>) {
    %v0_0 = fuga.mul(%x, %x) -> (tensor<3xf32>)
    fuga.return(%v0_0)
  }
}
```

---

<a id="control-flow-and-external"></a>

## 控制流与外部算子

Fuga IR 原生支持条件分支与循环区域：

| 接口 | 区域签名与说明 |
| --- | --- |
| `graph.if_(condition, operands, then_fn, else_fn, *, id=None)` | 分支区域。`then_fn` 与 `else_fn` 均为 `fn(builder, *operands)`，两分支返回的值类型必须严格一致。 |
| `graph.for_(lower, upper, step, initial_values, body_fn, *, id=None)` | 计步循环。`body_fn(builder, index, *carried)`；`index` 为 `int64` SSA 标量，循环体返回更新后的 `carried` SSA 序列。 |
| `graph.while_(initial_values, condition_fn, body_fn, *, id=None)` | 条件循环。`condition_fn(builder, *carried)` 返回布尔 SSA 标量，`body_fn` 返回循环体更新值。 |
| `ft.ir.ExternalSpec(...)` | 外部 kernel 规格元数据，声明函数名、目标平台、输入输出约束与按值标量参数。 |
| `graph.external(*operands, spec=..., result_types=None, id=None)` | 插入外部节点（如 Triton kernel 或 cuBLAS 原生调用）。 |

> [!NOTE]
> 降级到 ASK 执行时，静态化简器 `ft.ir.specialize` 会预先展开静态条件分支和定长循环。动态 `while` 需等待底层运行时完整循环支持。

---

<a id="ir-transforms"></a>

## 图变换与自动微分

Fuga IR 提供纯代数级别的图分析与微分变换，变换输入输出均为闭合的 Fuga IR 模块：

| 变换函数 | 说明 |
| --- | --- |
| `ft.ir.specialize(module, *, max_unroll=10000, expand_calls=True)` | 常量折叠与静态控制流展开，将模块化简为平坦图。 |
| `ft.ir.prune(module)` | 平坦图死代码消除与无用节点裁剪，返回最小化模块。 |
| `ft.ir.project_outputs(module, indices, *, name_suffix="", max_unroll=10000)` | 仅保留指定索引的公开输出，自动裁剪其余依赖路径。 |
| `ft.ir.analyze_function(function, selected=())` | 静态活跃性与拓扑排序分析，返回算子拓扑序与反向依赖索引（基于 C++ `_native` 加速）。 |
| `ft.ir.jvp(module, *, argnums=None, max_unroll=10000)` | 前向切线自动微分。输入包含原参数与切线，输出原值与切线产物。 |
| `ft.ir.vjp(module, *, argnums=None, max_unroll=10000)` | 反向余切自动微分。输入包含原参数与输出余切种子（cotangents），输出目标输入的梯度。 |

### VJP 微分变换示例

```python
import fugatensor as ft

# 定义前向模块
module = ft.ir.Module()
graph = module.graph("square", inputs={"x": ft.ir.TensorType((3,), ft.fp32)})
x = graph.inputs["x"]
graph.return_(graph.mul(x, x))

# 执行反向代数微分变换
vjp_module = ft.ir.vjp(module)
print(vjp_module.to_text())
```

**输出：**

```text
fuga.module {
  fuga.func @square_vjp(%x: tensor<3xf32>, %seed0: tensor<3xf32>) -> (tensor<3xf32>) {
    %v0_0 = fuga.mul(%seed0, %x) -> (tensor<3xf32>)
    %v1_0 = fuga.mul(%seed0, %x) -> (tensor<3xf32>)
    %v2_0 = fuga.add(%v0_0, %v1_0) -> (tensor<3xf32>)
    fuga.return(%v2_0)
  }
}
```

---

<a id="lowering-to-ask"></a>

## 降级到 ASK

```python
lowered = ft.lower_to_ask(module, *, target=None, max_unroll=10000, source_map=None)
```

- **参数**：
  - `module` (`ft.ir.Module`)：合法且已初始化的 Fuga IR 模块。
  - `target` (`dict | None`)：编译目标描述，如 `{"backend": "host"}` 或 `{"backend": "cuda", "arch": 86}`。
  - `max_unroll` (`int`)：静态循环展开深度上限，默认 10000。
  - `source_map` (`dict | None`)：可选传入字典，填充 Fuga IR Value 到 ASK SSA 标识的逐节点溯源映射。
- **返回值**：
  - `LoweringResult`：包含 ASK MLIR 文本及元数据。通过 `.to_mlir()` 可导出标准 MLIR 源码并提交至 ASK C++ Compiler。

### 降级示例

```python
import fugatensor as ft

module = ft.ir.Module()
graph = module.graph("square", inputs={"x": ft.ir.TensorType((3,), ft.fp32)})
x = graph.inputs["x"]
graph.return_(graph.mul(x, x))

lowered = ft.lower_to_ask(module, target={"backend": "host"})
print(lowered.to_mlir())
```

**输出：**

```text
module attributes {ask.schema_version = 1 : i64} {
  func.func @square(%x: tensor<3xf32> {ask.value_id = "square.arg0"}) -> (tensor<3xf32>) {
    %ask_v0 = "ask.mul"(%x, %x) {ask.id = "square.op0", ask.value_id = "square.op0.result0"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    func.return %ask_v0 : tensor<3xf32>
  }
}
```

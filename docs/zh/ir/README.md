# Fuga IR 总览

目录导航：[文档目录](../README.md) · 接口用法：[IR 与降级 API](../API/ir.md) · 系统架构：[架构总览](../architecture/README.md) · 底层引擎：[ASK IR 与数值语义](../../../../ask/docs/zh/ir.md)

Fuga IR（`src/fugatensor/ir/`）用类型、SSA 值、操作和嵌套区域表示张量程序。Tensor 的待计算图与隔离追踪节点保存在原生 arena，编译未命中或追踪结束时导出这些 IR 对象，供分析、求导和降级使用。

<a id="ir-design"></a>

## 1. 设计职责与设备无关性

Fuga IR 在降级到 [ASK 静态计算图](../../../../ask/docs/zh/ir.md) 前保留三类信息：

- 结构化控制流：`fuga.if`、`fuga.for`、`fuga.while` 用嵌套 `Region` 和块参数表示。
- 自定义求导：[`ft.fun`](../architecture/autograd.md#custom-fun) 追踪为 `fuga.call`，VJP 阶段使用前向子图的原语规则或显式 `vjp` 规则，原值与前向公开输出作为 SSA 值传入。
- 设备信息：类型和 SSA 值保存形状与 dtype，外部 Triton 算子的具体 GPU 架构在降级时绑定。

```text
Tensor 原生节点池 (cpp/lazy.cpp) / 隔离追踪节点池 (cpp/trace.cpp)
  -> Fuga IR Module (设备无关、含 Region 与 fuga.call)
  -> 框架级变换：C++ 活跃性分析与剪枝 (transforms.py) / JVP & VJP 求导 (jvp.py, differentiate.py)
  -> 降级桥接 (lowering.py)：静态 if 选择、有界 for 展开、架构绑定与 fuga-to-ask.json 溯源
  -> ASK IR Module -> ASK 优化、代码生成与常驻执行
```

<a id="ir-object-model"></a>

## 2. 对象模型与验证

Fuga IR 由 `ScalarType`、`TensorType`、`Value`、`Operation`、`Block`、`Region`、`Function` 和 `Module` 组成。`Module.verify()` 检查支配关系、类型和区域终结符。校验后的缓存模板会冻结；引擎按模板及连接关系查缓存，未命中时组合图，并用 `module.to_text()` 保存调试文本。

工程细节：[对象模型与验证](model.md) · [原生图分析与 C++ 边界](../architecture/native.md)

<a id="ir-control-transforms"></a>

## 3. 结构化控制流与可组合微分（JVP / VJP）

`ir/primitives.py` 集中定义各基础算子的抽象类型推导（`abstract_eval`）、前向切线规则（`jvp`）与反向余切规则（`vjp`）：

- 算术、类型转换和形状变换使用 `constant`、`splat`、`add`、`mul`、`div`、`cast`、`reshape`、`exp/log/sqrt`、`compare/select`、`broadcast_in_dim/transpose`、`reduce_sum/reduce_max` 与 `matmul`；`stop_gradient` 保留数值并停止微分传播。`subtract`、`negative`、`square` 在追踪期由基础算子组合展开；
- `ir.jvp(module)` 与 `ir.vjp(module)` 生成新的 Fuga IR `Module`。在已有规则范围内，可以嵌套 `vjp(vjp(module))`、`jvp(vjp(module))`、`vjp(jvp(module))`，`ft.grad` 使用 VJP 生成新的导数表达式，导数求值后仍保留来源。

工程细节：[控制流与可组合变换](transforms.md) · [函数式自动微分与高阶求导](../architecture/autograd.md#higher-order)

<a id="ir-lowering-boundary"></a>

## 4. 降级到 ASK IR 与跨层溯源

`lower_to_ask` 选择编译期已知的 `fuga.if` 分支、展开有界静态 `fuga.for`，再将 Fuga `Value` 映射为 ASK `Value`，生成 `fuga-to-ask.json`。数据依赖的 IR `if` 和所有 IR `while` 当前无法降级，会抛出异常。

工程细节：[降级到 ASK](lowering.md) · [ASK 桥接与多后端执行](../architecture/backend.md)

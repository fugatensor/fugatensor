# FugaTensor 用户 API

返回文档目录：[文档目录](../README.md) · 环境准备：[安装与配置](../安装.md) · 实现原理：[架构总览](../architecture/README.md)

本文档介绍 `import fugatensor as ft` 后可用的接口。普通 Python 函数直接组合 Tensor 运算，无需 JIT 装饰器；读取结果、反向传播或 `ft.evaluate(...)` 才触发所需计算。示例默认使用已安装的 FugaTensor 和 CPU 引擎；CUDA、Triton 示例另有环境要求。

<a id="categories"></a>

## 按类型查找

| 分类 | 主要接口 |
| --- | --- |
| [01. 张量与数据交换](01-张量与数据交换.md) | `tensor`、`from_buffer`、`Tensor` 属性、`tolist`、`to_buffer` |
| [02. 基础运算](02-基础运算.md) | 加减乘、负号、`square`、`cast`、`zeros_like`、`ones_like` |
| [03. 引擎与执行](03-引擎与执行.md) | `Engine`、`configure`、`shutdown`、`evaluate`、同步、缓存和诊断 |
| [04. 自动微分与自定义函数](04-自动微分与自定义函数.md) | `backward`、`grad`、`zero_grad`、`detach`、`no_grad`、`fun` |
| [05. Triton 算子接入](05-Triton算子接入.md) | `triton.kernel`、`from_source`、`from_jit`、配置回调与编译环境 |
| [06. 权重读写](06-权重读写.md) | `save_weights`、`load_weights`、`weights` 的清单与元数据接口 |
| [07. IR 与降级](07-IR与降级.md) | `ir.Module`、类型与构图、`jvp` / `vjp`、`lower_to_ask` |

<a id="quick-start"></a>

## 最小用法

```python
import fugatensor as ft

with ft.Engine("cpu") as engine:
    x = ft.tensor([1.0, 2.0, 3.0])
    y = (x + 1) * 2
    assert y.shape == (3,)       # 读取元信息不执行图
    assert y.tolist() == [4.0, 6.0, 8.0]
```

引擎关闭后不能继续读取它的 Tensor。需要跨多次调用复用引擎时，使用 [`configure`](03-引擎与执行.md#engine-lifecycle)，在完成后调用 `shutdown`。

<a id="api-scope"></a>

## 接口范围

运行时 Tensor 使用具体形状，支持 `bool`、`i32`、`i64`、`f32`、`f16`、`bf16`，`fp16` 是 `f16` 的别名。类型提升、不同形状张量的广播、索引、归约与矩阵乘法尚未实现；各类接口的具体约束见对应页面。

`fugatensor._ask`、`_native`、`_jit` 等下划线模块属于内部实现，不需要用户直接导入。ASK 本身只提供 [C++ SDK 与 CLI](../../../ask/docs/用户API.md)，FugaTensor 是 Python 用户入口。图级扩展接口见第 07 篇，普通张量计算从前六篇开始即可。

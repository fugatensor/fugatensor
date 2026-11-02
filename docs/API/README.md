# FugaTensor API

返回文档目录：[文档目录](../README.md) · 环境准备：[安装与配置](../安装.md) · 实现原理：[架构总览](../architecture/README.md)

FugaTensor 默认使用 CPU 和 `ft.fp32`。张量运算先组图，读取结果、反向传播或调用 `ft.evaluate(...)` 时执行。CUDA 和 Triton 的环境配置见[安装说明](../安装.md)。

<a id="categories"></a>

## 按类型查找

| 分类 | 主要接口 |
| --- | --- |
| [张量与数据交换](张量与数据交换.md) | `tensor`、`zeros`、`ones`、`empty`、`full`、`reshape`、buffer 读写 |
| [基础运算](基础运算.md) | 加减乘、负号、`square`、`cast`、`zeros_like`、`ones_like` |
| [设备与执行](设备与执行.md) | `device`、默认设备、`evaluate`、`synchronize`、可选配置与高级引擎管理 |
| [自动微分与自定义函数](自动微分与自定义函数.md) | `backward`、`grad`、`zero_grad`、`detach`、`no_grad`、`fun` |
| [Triton 算子接入](Triton算子接入.md) | `triton.kernel`、`from_source`、`from_jit`、配置回调与编译环境 |
| [权重读写](权重读写.md) | `save_weights`、`load_weights`、`weights` 的清单与元数据接口 |
| [IR 与降级](IR与降级.md) | `ir.Module`、类型与构图、`jvp` / `vjp`、`lower_to_ask` |
| [attention](attention.md) | `nn.functional.scaled_dot_product_attention`、因果掩码与一阶反向 |

<a id="quick-start"></a>

## 最小用法

```python
import fugatensor as ft

x = ft.tensor([1, 2, 3])
y = (x + 1) * 2
print(y.shape)     # (3,)，读取元信息不执行图
print(y.tolist())  # [4.0, 6.0, 8.0]
```

用 `dtype=ft.fp16` 选择创建精度，用 `x.astype(ft.fp16)` 转换类型。使用 GPU 时传 `device="cuda"`，或设置 [`ft.set_default_device("cuda")`](设备与执行.md#default-device)。

<a id="api-scope"></a>

## 接口范围

运行时 Tensor 使用具体形状，类型对象为 `ft.bool`、`ft.int32`、`ft.int64`、`ft.fp32`、`ft.fp16`、`ft.bf16`。类型提升、不同形状张量的广播、索引、归约与矩阵乘法尚未实现；各类接口的具体约束见对应页面。

手写图和图变换见 [IR 与降级](IR与降级.md)，后端接口见 [ASK C++ SDK 与 CLI](../../../ask/docs/用户API.md)。

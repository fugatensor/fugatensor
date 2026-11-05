# FugaTensor API

返回文档目录：[文档目录](../README.md) · 环境准备：[安装与配置](../install.md) · 系统架构：[架构总览](../architecture/README.md)

FugaTensor 默认使用 CPU 和 `ft.fp32`。张量运算先生成独立 Fuga IR 待计算节点，通过自动微分与 C++ 活跃性分析优化，最终交由后端编译执行。

<a id="quick-start"></a>

## 快速上手

```python
import fugatensor as ft

# 创建张量
x = ft.tensor([1.0, 2.0, 3.0])
y = (x + 1.0) * 2.0

# 延迟求值：查看元信息不触发实际执行
print("Shape:", y.shape)  # (3,)
print("Dtype:", y.dtype)  # fp32

# 读取数值或显式求值时触发编译执行
print("Result:", y.tolist())  # [4.0, 6.0, 8.0]
```

<a id="categories"></a>

## 按功能分类查找

| 模块 | 说明 | 核心接口 |
| --- | --- | --- |
| [张量创建与数据交换](tensor.md) | 张量构建、尺寸工厂、类型系统与内存互操作 | `tensor`, `zeros`, `ones`, `empty`, `full`, `from_buffer`, `reshape`, `astype` |
| [基础算术与代数运算](ops.md) | 逐元素运算、逻辑比较、几何变换、归约与矩阵乘法 | `add`, `multiply`, `where`, `transpose`, `sum`, `mean`, `amax`, `matmul` |
| [自动微分与自定义算子](autograd.md) | 函数式求导、梯度路径阻断与显式 VJP 算子 | `grad`, `stop_gradient`, `detach`, `fun` |
| [设备管理与执行调度](engine.md) | 设备切换、默认设备配置、流同步与引擎生命周期 | `device`, `set_default_device`, `evaluate`, `synchronize`, `Engine` |
| [注意力算子](attention.md) | 高性能内置注意力算子与因果掩码 | `nn.functional.scaled_dot_product_attention` |
| [Triton 算子接入](triton.md) | 外部 Triton 自定义 Kernel 绑定与配置回调 | `triton.kernel`, `triton.from_source`, `triton.from_jit` |
| [权重持久化](weights.md) | 模型参数 PyTree 序列化与内容寻址校验 | `save_weights`, `load_weights`, `weights.manifest` |
| [IR 与后端降级](ir.md) | Fuga IR 原语构建、控制流与 ASK 编译桥接 | `ir.Module`, `lower_to_ask` |

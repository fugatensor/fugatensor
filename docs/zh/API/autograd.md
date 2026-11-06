返回 API 目录：[分类目录](README.md#categories) · 关联模块：[基础运算](ops.md) · [IR 与降级](ir.md) · 架构原理：[自动微分架构](../architecture/autograd.md)

# 自动微分与自定义算子

FugaTensor 采用函数式自动微分。自动微分是对 Fuga IR 表达式的有向图变换，返回待计算的导数张量，能够自由组合、多次嵌套与高阶求导。

<a id="grad"></a>

## 自动微分接口

### `ft.grad(outputs, inputs, *, seed=None)`
计算输出关于指定输入的梯度。

- **参数**：
  - `outputs`：目标标量张量，或张量序列（tuple/list）。
  - `inputs`：自变量张量，或张量序列。
  - `seed`：反向传播的起始梯度（余切种子）。若 `outputs` 为标量，默认种子为 1.0；若为非标量，必须显式传入结构与尺寸匹配的 seed。
- **返回**：与 `inputs` 结构对应的梯度张量（若 `inputs` 为序列则返回梯度元组）。未连接到输出的输入返回匹配形状的零梯度。

```python
import fugatensor as ft

# 标量函数求导：y = x^2 + 3x
x = ft.tensor([2.0, 3.0])
y = (x * x + 3.0 * x).sum()

dx = ft.grad(y, x)
print(dx.tolist())  # [7.0, 9.0] (对应 2x + 3)

# 高阶求导
d2x = ft.grad(dx.sum(), x)
print(d2x.tolist())  # [2.0, 2.0]
```

<a id="gradient-control"></a>

## 梯度控制

### `ft.stop_gradient(x)` / `x.detach()`
断开张量的自动微分依赖链，同时保留前向数值依赖。求导沿计算图传播时在此处截断，不向输入反传梯度。

```python
x = ft.tensor([2.0, 4.0])
y = ft.stop_gradient(x * 3.0)
z = (y * x).sum()

dx = ft.grad(z, x)
# 对 x 求导时，y 被视作常量：d/dx (y * x) = y = [6.0, 12.0]
print(dx.tolist())  # [6.0, 12.0]
```

<a id="custom-fun"></a>

## 自定义可微函数 `ft.fun`

`ft.fun` 允许将一段逻辑封装为具有隔离追踪边界的可微复合算子，并可通过 `vjp` 定义纯规则的反向传播：

### 接口规范
1. 继承 `ft.fun` 并实现 `forward(self, ...)`；
2. （可选）实现纯函数 `vjp(self, primals, outputs, cotangents)`：
   - `primals`：前向全部张量输入的 tuple；
   - `outputs`：前向全部公开输出的 tuple；
   - `cotangents`：与 outputs 对齐的余切张量 tuple；
   - 返回值：与 `primals` 中浮点张量输入对齐的梯度 tuple。

```python
import fugatensor as ft

class CustomSquare(ft.fun):
    def forward(self, x):
        return x * x

    def vjp(self, primals, outputs, cotangents):
        (x,) = primals
        (dy,) = cotangents
        # 自定义反向规则：dy * 2 * x
        return (dy * 2.0 * x,)

square_op = CustomSquare()
a = ft.tensor([3.0, 5.0])
b = square_op(a)
print("Forward:", b.tolist())  # [9.0, 25.0]

da = ft.grad(b.sum(), a)
print("Gradient:", da.tolist())  # [6.0, 10.0]
```

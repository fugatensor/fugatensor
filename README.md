# FugaTensor

FugaTensor 是基于 [ASK](../ask/README.md) 编译与运行时引擎的张量计算框架。张量运算自动记录独立的 Fuga IR；读取结果、判断标量条件、反向传播或显式调用 `ft.evaluate` 时，才合并所需计算并 JIT 编译，交由 CPU 或 CUDA 执行。

```python
import fugatensor as ft

def arithmetic(x, y):
    return (x + y) * 2

x = ft.tensor([[1, 2], [3, 4]])
y = ft.tensor([[10, 20], [30, 40]])
print(arithmetic(x, y).tolist())
# [[22.0, 44.0], [66.0, 88.0]]
```

默认使用 CPU 和 `ft.fp32`，无需创建引擎或使用 `with`，资源在正常进程退出时自动清理。需要其他精度时传 `dtype=ft.fp16` 等类型对象；设备可用 `ft.device("cuda", 0)` 或字符串 `"cuda:0"` 指定，也可用 `ft.set_default_device(...)` 统一选择后续张量的默认设备。

按尺寸创建张量和改变形状可以直接写：

```python
x = ft.ones(2, 2)
y = ft.tensor([1, 2, 3, 4]).reshape(2, 2)
print((x + y).reshape(-1).tolist())  # [2.0, 3.0, 4.0, 5.0]
```

`tensor` 从嵌套数据推断形状；`zeros`、`ones`、`empty` 接受多个尺寸或一个 tuple/list，`full(size, value)` 指定填充值。`reshape` 保留元素次序并参与延迟计算和自动微分，可用一个 `-1` 推断尺寸。完整规则见[张量创建与形状](docs/API/张量与数据交换.md#size-factories)。

- **自动组图与 JIT 执行**：`x + 2`、`x.square()`、`x.astype(ft.fp16)` 等运算先返回待计算 Tensor，跨算子和普通函数调用保留融合机会；执行边界合并依赖图，并按图结构和类型复用编译结果。所有张量算术都由 ASK 执行。
- **CPU 与自有 CUDA 后端**：将张量指定为 `device="cuda"` 即可通过 ASK 自有 CUDA 后端生成并异步执行 GPU kernel，无需依赖 Triton。
- **自动微分与高阶导数**：支持 `requires_grad=True`、`y.backward()`、`ft.fun` 自定义前向/反向、跨 0-D 标量 `if`/`while` 动态分支的梯度传播，以及 `create_graph=True` 二阶/三阶导数。
- **用户 Triton 算子调度**：通过 `ft.triton.kernel` / `from_jit` 接入用户源码或已有 JIT 函数，支持标量参数、多输出与常用 constexpr，由 FugaTensor worker 编译为 `cubin`，ASK 将其作为独立步骤与自有计算一起调度。

接口用法见 [API](docs/API/README.md)。设计细节见 [文档目录](docs/README.md)、[架构总览](docs/architecture/README.md) 与 [Fuga IR 总览](docs/ir/README.md)。

<a id="setup"></a>

## 独立构建与安装

ASK 独立构建 C++ SDK，FugaTensor 是唯一的 Python 包。构建 FugaTensor 时通过 `CMAKE_PREFIX_PATH` 指定安装好的 ASK SDK；wheel 携带编译器和运行时共享库，运行时无需安装 ASK Python 包，也无需保留 SDK 或源码树。

```bash
CMAKE_PREFIX_PATH=/absolute/path/to/ask-sdk python3 -m pip wheel --no-deps . --wheel-dir dist
```

安装选项为 `fugatensor[pure]` 和 `fugatensor[all]`：pure 不安装 Triton，all 自动安装受支持的 Triton，并默认使用当前 Python 环境的 worker。两者共用同一 wheel；直接安装 `fugatensor` 等同 pure。具体命令和环境要求见[安装选项](docs/安装.md#installation-variants)。

构建需要 C++17、Python 3.10+ 开发环境、CMake 3.28+ 和 nanobind；消费 ASK 的 `Compiler` / `Runtime` 不需要 LLVM SDK。CPU JIT 仍需要 `cc` 链接器，自有 CUDA 代码生成需配置 `llc` / `ptxas`。独立 Triton worker 的配置见[安装文档](docs/安装.md#configure-triton)。

安装、配置和开发引导命令见[安装文档](docs/安装.md)。跨仓文档链接仅供源码浏览。

## 基本用法

### 1. 自动微分、`ft.fun` 自定义反向与高阶求导

```python
def loss(x):
    return ft.square(x) + 3 * x

x = ft.tensor(2.0, requires_grad=True)
loss(x).backward()
print(x.grad.tolist())  # 7.0
```

- 非标量张量调用 `.backward(dy)` 时须显式传入同 shape、同 dtype 的梯度种子，不做隐式求和。
- 自定义反向通过继承 `ft.fun` 实现，直接定义普通 `forward` 与可选 `backward` 方法，通过 `ctx.save_for_backward(...)` 与 `ctx.saved_tensors` 传递中间张量：

```python
class CustomSquare(ft.fun):
    def forward(self, ctx, x):
        ctx.save_for_backward(x)
        return ft.square(x)

    def backward(self, ctx, dy):
        (x,) = ctx.saved_tensors
        return (2 * x * dy,)
```

- 传入 `y.backward(create_graph=True)` 会记录梯度的计算图，先保存 `first = x.grad`，再调用 `x.zero_grad()` 与 `first.backward()` 可继续求高阶导数；求导路径包含无原语规则的不透明外部算子时，会在反向执行前报错拒绝。见 [自动微分与自定义算子](docs/architecture/自动微分与自定义算子.md)。

### 2. 接入用户 Triton 算子

使用 `ft.triton.from_source(source, ...)` 注册源码字符串，或通过 `ft.triton.kernel(path, ...)` 读取自己的 kernel 文件；已有 JIT 对象可通过 `ft.triton.from_jit(fn, ...)` 注册。配置张量端口、启动网格与参数后，返回的算子可以直接与普通张量运算组合。

注册时保存源码快照，不在宿主进程导入 Triton。worker 将源码编译成 `cubin`，ASK 在同一 stream 上调度外部 kernel 与前后自有计算，自有区域仍可融合。`scalars=` 绑定按值参数，`outputs=` 声明多个输出；autotune 只选择显式配置。完整用法见[标量与双输出示例](docs/API/Triton算子接入.md#example)，环境准备见 [Triton 配置](docs/安装.md#configure-triton)。

内置注意力直接调用 `ft.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`，无需注册 kernel。该接口使用 FlashAttention 实现，支持 CUDA 上 `ft.fp16` / `ft.bf16` 的前向与一阶反向；形状与支持边界见[attention API](docs/API/attention.md)。

### 3. 查看 IR、溯源产物与缓存统计

```python
x = ft.tensor([1, 2, 3])
y = (x + 1) * 2
ft.evaluate(y)                   # 显式执行；多个输出可一起传入
engine = x.engine                # 需要诊断时查看框架管理的引擎
print(engine.cache_info())       # 图缓存条目、编译次数与内存命中统计
print(engine.last_fuga_ir)        # 最近执行的 Fuga IR
print(engine.last_ask_ir)         # 提交给 ASK 的 MLIR 文本
print(engine.last_report)        # ASK 编译报告
print(engine.last_artifact_dir)  # fuga.ir、frontend.mlir 与 fuga-to-ask.json
```

`ft.evaluate(a, b)` 合并两个待计算输出的依赖图，并返回原 Tensor；已计算结果不会重复执行。`shape`、`dtype`、`device` 与 `repr` 只读取元信息。普通 Python 的 `if`/`while` 在读取 0-D Tensor 真值时执行条件所需的图，再按实际条件继续组图；`ft.fun` 内部的隔离追踪仍不支持数据依赖控制流。

引擎在拼接完整图之前，用不可变模板及其连接关系查询缓存；重复图跳过组合、IR 文本生成和编译，普通 Python 函数仍照常执行。已有连续数组可用 `ft.from_buffer(data, shape=..., dtype=...)` 批量导入，用 `tensor.to_buffer()` 导出原始字节；CPU 支持显式借用存储，详见 [buffer 接口与寿命规则](docs/architecture/张量模型与追踪机制.md#buffer-io)。

## 当前支持范围

| 项目 | 当前行为 |
| --- | --- |
| **数据类型** | 默认 `ft.fp32`；支持 `ft.bool`、`ft.int32`、`ft.int64`、`ft.fp32`、`ft.fp16`、`ft.bf16`，用 `.astype(...)` 显式转换 |
| **张量创建与基础运算** | `tensor`、`zeros`、`ones`、`empty`、`full`；加减乘、负号、`square`、`cast`、`reshape`、`zeros_like`、`ones_like`（静态形状，含 0-D 标量与空张量，无隐式广播） |
| **控制流** | Fuga IR 静态 `if` 选择与有界 `for` 展开；普通 Python 支持 0-D Tensor 的 `if`/`while`，条件读取触发按需执行 |
| **自动微分** | `requires_grad`、`.backward()`、`.grad` 累加、`detach()`、`no_grad()`、`ft.fun` 自定义前反向、`create_graph=True` 高阶求导 |
| **注意力** | `nn.functional.scaled_dot_product_attention`，CUDA fp16/bf16、可选因果掩码、FlashAttention 前向及一阶反向 |
| **权重导出** | 版本化 PyTree 权重目录，支持嵌套 dict/list/tuple、六种基础 dtype、内容寻址去重、原子导出与完整性校验 |
| **执行后端** | CPU（`.o` → `.so` → `ctypes`）、自有 CUDA（多输入/异构多输出，无需 Triton）、独立 Triton `cubin` 与多 kernel 调度、多输入/多输出静态外部算子、按值标量 |

暂未实现隐式广播、矩阵乘法与归约算子；含不透明 Triton 算子的路径支持一阶手写反向，不支持二阶及以上高阶微分。

## 测试

示例写在文档中，测试放在 `tests/` 并使用断言。数值覆盖由算子、组合深度、精度和求导阶数生成；筛选参数与扩展方式见[测试矩阵](docs/测试.md)。

```bash
# CPU 编译、C++ 原生图分析、Fuga IR / JVP / VJP 与降级测试
.venv/bin/python -m unittest -v

# 启用物理 GPU 自有 CUDA 测试
FUGATENSOR_TEST_CUDA=1 .venv/bin/python -m unittest -v

# 同时启用自有 CUDA 与用户 Triton GPU 调度及反向测试
FUGATENSOR_TEST_CUDA=1 FUGATENSOR_TEST_TRITON=1 .venv/bin/python -m unittest -v
```

## 文档

- [安装与独立构建](docs/安装.md)
- [文档目录](docs/README.md)
- [API](docs/API/README.md)：张量与数据交换、基础运算、设备与执行、自动微分与自定义函数、Triton 接入、权重读写、IR 与降级、attention。
- **系统架构（`docs/architecture/`）**：[架构总览](docs/architecture/README.md) · [张量模型与追踪机制](docs/architecture/张量模型与追踪机制.md) · [原生图分析与 C++ 边界](docs/architecture/原生图分析与C++边界.md) · [自动微分与自定义算子](docs/architecture/自动微分与自定义算子.md) · [ASK 桥接与多后端执行](docs/architecture/ASK桥接与多后端执行.md) · [权重格式与导出](docs/architecture/权重格式与导出.md)
- **中间表示（`docs/ir/`）**：[Fuga IR 总览](docs/ir/README.md) · [对象模型与验证](docs/ir/对象模型与验证.md) · [控制流与可组合变换](docs/ir/控制流与可组合变换.md) · [降级到 ASK](docs/ir/降级到ASK.md)

[多 kernel 计划契约](../ask/docs/图切分与多Kernel执行计划.md) · [attention API](docs/API/attention.md) · [权重格式与导出](docs/architecture/权重格式与导出.md)

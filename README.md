# FugaTensor

FugaTensor 是基于 [ASK](../ask/README.md) 编译与运行时引擎的张量计算框架。Python 追踪先记录独立的 Fuga IR，在框架层完成控制流化简与前向/反向微分，再降级为 ASK IR 交由 CPU 或 CUDA 执行。

```python
import fugatensor as ft

@ft.jit
def arithmetic(x, y):
    return (x + y) * 2

x = ft.tensor([[1, 2], [3, 4]], dtype="f32")
y = ft.tensor([[10, 20], [30, 40]], dtype="f32")
print(arithmetic(x, y).tolist())
# [[22.0, 44.0], [66.0, 88.0]]
```

- **JIT 编译执行**：首次调用追踪生成设备无关的 Fuga IR 并编译为原生代码，相同 `(shape, dtype)` 的张量更换数值直接复用可执行代码；`x + 2`、`x.square()`、`x.astype("f16")` 等运算同样通过 `@ft.jit` 执行。
- **CPU 与自有 CUDA 后端**：将张量指定为 `device="cuda"` 即可通过 ASK 自有 CUDA 后端生成并异步执行 GPU kernel，无需依赖 Triton。
- **自动微分与高阶导数**：支持 `requires_grad=True`、`y.backward()`、`ft.fun` 自定义前向/反向、跨 0-D 标量 `if`/`while` 动态分支的梯度传播，以及 `create_graph=True` 二阶/三阶导数。
- **用户 Triton 算子融合**：通过 `ft.triton.kernel` 接入用户编写的 `@triton.jit` 源码，由 ASK 完成 `ASK → Triton → ASK` 原生 LLVM IR 值边界融合。

设计细节见 [文档目录](docs/README.md)、[架构总览](docs/architecture/README.md) 与 [Fuga IR 总览](docs/ir/README.md)。

<a id="setup"></a>

## 构建与环境引导

环境要求：Python 3.11+、`pip`、C/C++17 工具链（默认读取同级目录 `../ask`；GPU 执行另需 NVIDIA 驱动与 CUDA `ptxas`）。

在仓库根目录执行 `bootstrap.py` 完成依赖拉取、ASK 编译安装、FugaTensor（含 C++ `_native` 图分析扩展）可编辑安装，并将工具路径写入 `.venv/share/fugatensor/engine.json`：

```bash
# 默认启用 ASK_THIRD_PARTY_COMPILERS=all（含 Triton 3.7.1 适配器）
python3 tools/bootstrap.py

# 运行示例
.venv/bin/python examples/jit_demo.py --device both
.venv/bin/python examples/mixed_triton.py
.venv/bin/python examples/autodiff.py --device both --triton
```

常用构建参数：

```bash
# 仅修改 FugaTensor 自身 Python / C++ 代码时跳过 ASK 重编
python3 tools/bootstrap.py --skip-ask

# 仅启用自有 CPU + 自有 CUDA 后端，不安装 Triton 依赖
python3 tools/bootstrap.py --compilers none
```

引导完成后直接使用 `.venv/bin/python` 即可自动读取已保存的工具路径。如需在代码中指定缓存目录或工具路径，可调用 `ft.configure(device="cpu", cache_dir="...")` 或使用 `with ft.Engine(...)`，见 [ASK 桥接与多后端执行](docs/architecture/04-ASK桥接与多后端执行.md#bootstrap-and-config)。

## 基本用法

### 1. 自动微分、`ft.fun` 自定义反向与高阶求导

```python
@ft.jit
def loss(x):
    return ft.square(x) + 3 * x

x = ft.tensor(2.0, requires_grad=True)
loss(x).backward()
print(x.grad.tolist())  # 7.0
```

- 非标量张量调用 `.backward(dy)` 时须显式传入同 shape、同 dtype 的梯度种子，不做隐式求和。
- 自定义反向通过继承 `ft.fun` 实现，`forward` 与 `backward` 均须添加 `@ft.jit`，通过 `ctx.save_for_backward(...)` 与 `ctx.saved_tensors` 传递中间张量：

```python
class CustomSquare(ft.fun):
    @ft.jit
    def forward(self, ctx, x):
        ctx.save_for_backward(x)
        return ft.square(x)

    @ft.jit
    def backward(self, ctx, dy):
        (x,) = ctx.saved_tensors
        return (2 * x * dy,)
```

- 传入 `y.backward(create_graph=True)` 会记录梯度的计算图，先调用 `x.zero_grad()` 再对 `x.grad` 调用 `.backward()` 可继续求高阶导数；求导路径包含无原语规则的不透明外部算子时，会在反向执行前报错拒绝。见 [自动微分与自定义算子](docs/architecture/03-自动微分与自定义算子.md)。

### 2. 接入用户 Triton 算子

编写标准 Triton kernel（如 [examples/kernels.py](examples/kernels.py)）并注册输入输出参数与启动配置：

```python
scale = ft.triton.kernel(
    "examples/kernels.py",
    entry="scale_kernel",
    input="inp",
    output="out",
    constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
    grid=lambda spec: ((spec.numel + 127) // 128, 1, 1),
    options={"num_warps": 4},
)

@ft.jit
def mixed(x):
    return scale(x + 1) + 3

x = ft.tensor([[1, 2], [3, 4]], dtype="f32", device="cuda")
print(mixed(x).tolist())  # [[7.0, 9.0], [11.0, 13.0]]
```

注册时保存源码快照，不在宿主进程导入 Triton；ASK 编译时将前后的 `+ 1` 与 `+ 3` 内联进 `scale_kernel` 的原生 LLVM IR 生成单条 `cubin`。

### 3. 查看 IR、溯源产物与缓存统计

```python
print(arithmetic.cache_info())       # 特化数、编译次数与内存命中统计
print(arithmetic.last_fuga_ir)       # Fuga IR 文本
print(arithmetic.last_ask_ir)        # 提交给 ASK 的 MLIR 文本
print(arithmetic.last_artifact_dir)  # 包含 fuga.ir、frontend.mlir 与 fuga-to-ask.json
```

含 0-D 标量 `if`/`while` 图中断的函数可通过 `fn.last_segments` 查看本次执行各片段的 IR 与编译报告。

## 当前支持范围

| 项目 | 当前行为 |
| --- | --- |
| **数据类型** | `bool`、`i32`、`i64`、`f32`、`f16`（别名 `fp16`）、`bf16`；显式 `.astype(...)` 混合精度转换 |
| **基础运算** | `add`、`subtract`、`multiply`、`negative`、`square`、`cast`、`zeros_like`、`ones_like`（静态形状，含 0-D 标量与空张量，无隐式广播） |
| **控制流** | Fuga IR 静态 `if` 选择与有界 `for` 展开；普通 `@ft.jit` 函数支持 0-D 标量 `if`/`while` 运行时图中断 |
| **自动微分** | `requires_grad`、`.backward()`、`.grad` 累加、`detach()`、`no_grad()`、`ft.fun` 自定义前反向、`create_graph=True` 高阶求导 |
| **执行后端** | CPU（`.o` → `.so` → `ctypes`）、自有 CUDA（多输入/异构多输出，无需 Triton）、Triton 单输入单输出逐元素融合 |

暂未实现隐式广播、矩阵乘法与归约算子；含不透明 Triton 算子的路径支持一阶手写反向，不支持二阶及以上高阶微分。

## 测试

```bash
# CPU 编译、C++ 原生图分析、Fuga IR / JVP / VJP 与降级测试
.venv/bin/python -m unittest -v

# 启用物理 GPU 自有 CUDA 测试
FUGATENSOR_TEST_CUDA=1 .venv/bin/python -m unittest -v

# 同时启用自有 CUDA 与用户 Triton GPU 融合及反向测试
FUGATENSOR_TEST_CUDA=1 FUGATENSOR_TEST_TRITON=1 .venv/bin/python -m unittest -v
```

## 文档

- [文档目录](docs/README.md)
- **系统架构（`docs/architecture/`）**：[架构总览](docs/architecture/README.md) · [01. 张量模型与追踪机制](docs/architecture/01-张量模型与追踪机制.md) · [02. 原生图分析与 C++ 边界](docs/architecture/02-原生图分析与C++边界.md) · [03. 自动微分与自定义算子](docs/architecture/03-自动微分与自定义算子.md) · [04. ASK 桥接与多后端执行](docs/architecture/04-ASK桥接与多后端执行.md)
- **中间表示（`docs/ir/`）**：[Fuga IR 总览](docs/ir/README.md) · [01. 对象模型与验证](docs/ir/01-对象模型与验证.md) · [02. 控制流与可组合变换](docs/ir/02-控制流与可组合变换.md) · [03. 降级到 ASK](docs/ir/03-降级到ASK.md)

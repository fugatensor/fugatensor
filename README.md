# FugaTensor

FugaTensor 是基于 [ASK](../ask/README.md) 编译与运行时引擎的张量计算框架。张量运算自动记录独立的 Fuga IR；读取结果、判断标量条件、反向传播或显式调用 `ft.evaluate` 时，才合并所需计算并 JIT 编译，交由 CPU 或 CUDA 执行。

```python
import fugatensor as ft

def arithmetic(x, y):
    return (x + y) * 2

x = ft.tensor([[1, 2], [3, 4]], dtype="f32")
y = ft.tensor([[10, 20], [30, 40]], dtype="f32")
print(arithmetic(x, y).tolist())
# [[22.0, 44.0], [66.0, 88.0]]
```

- **自动组图与 JIT 执行**：`x + 2`、`x.square()`、`x.astype("f16")` 等运算先返回待计算 Tensor，跨算子和普通函数调用保留融合机会；执行边界合并依赖图，并按图结构和类型复用编译结果。所有张量算术都由 ASK 执行。
- **CPU 与自有 CUDA 后端**：将张量指定为 `device="cuda"` 即可通过 ASK 自有 CUDA 后端生成并异步执行 GPU kernel，无需依赖 Triton。
- **自动微分与高阶导数**：支持 `requires_grad=True`、`y.backward()`、`ft.fun` 自定义前向/反向、跨 0-D 标量 `if`/`while` 动态分支的梯度传播，以及 `create_graph=True` 二阶/三阶导数。
- **用户 Triton 算子调度**：通过 `ft.triton.kernel` 接入用户编写的 `@triton.jit` 源码，由 FugaTensor worker 编译为 `cubin`，ASK 将其作为独立步骤与自有计算一起调度。

设计细节见 [文档目录](docs/README.md)、[架构总览](docs/architecture/README.md) 与 [Fuga IR 总览](docs/ir/README.md)。

<a id="setup"></a>

## 独立构建与安装

ASK 是必需的运行依赖，负责实际图编译与算术执行；两个项目分别构建，再安装到同一运行环境。当前 FugaTensor 包要求 Python 3.10+、`ask-compiler==0.1.0`。运行不要求 ASK 源码树，也不要求两个仓库位于同级目录。

1. **构建 ASK**：在 ASK 自身的构建环境中准备其锁定 LLVM/MLIR SDK，生成兼容的 `ask-compiler` wheel；已有该产物时可复用。
2. **构建 FugaTensor**：在独立构建环境中执行 `python3 -m pip wheel --no-deps /absolute/path/to/fugatensor --wheel-dir /absolute/path/to/fuga-wheel-output`。该步骤需要 C++17 工具链、Python 开发头文件与 CMake 3.28+，不需要安装 ASK 或 LLVM SDK。
3. **安装并运行**：准备与 Python ABI、系统及架构匹配的两个 wheel，在运行环境中安装：

```bash
python3 -m venv .venv
# 填入两个实际产物的完整文件名
ask_wheel='/absolute/path/to/ask_compiler-0.1.0-<python>-<abi>-<platform>.whl'
fuga_wheel='/absolute/path/to/fugatensor-0.1.0-<python>-<abi>-<platform>.whl'
.venv/bin/python -m pip install --no-index "$ask_wheel" "$fuga_wheel"
.venv/bin/python -m pip check
```

CPU JIT 还需要 `cc` 链接器，运行时使用 ASK 包内的原生扩展和伴生 CLI，无需 LLVM SDK 或 `engine.json`。自有 CUDA 编译另需配置 `llc`、`ptxas`，执行需要 NVIDIA 驱动；用户 Triton 算子通过 FugaTensor 自带 worker 接入，编译环境需安装 Triton 3.7.1。

完整命令、源码目录之外的 CPU 验收、GPU 配置与可选开发引导见[安装文档](docs/安装.md)。这里从本地产物安装，不假定项目已发布到 PyPI。文档中的 `../ask/...` 跨仓链接只供并列检出源码时浏览，不构成构建或运行路径要求。

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

- 传入 `y.backward(create_graph=True)` 会记录梯度的计算图，先保存 `first = x.grad`，再调用 `x.zero_grad()` 与 `first.backward()` 可继续求高阶导数；求导路径包含无原语规则的不透明外部算子时，会在反向执行前报错拒绝。见 [自动微分与自定义算子](docs/architecture/03-自动微分与自定义算子.md)。

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
def mixed(x):
    return scale(x + 1) + 3

x = ft.tensor([[1, 2], [3, 4]], dtype="f32", device="cuda")
print(mixed(x).tolist())  # [[7.0, 9.0], [11.0, 13.0]]
```

注册时保存源码快照，不在宿主进程导入 Triton。使用 `ft.triton.environment(python="/path/to/triton-env/bin/python")` 配置编译环境，worker 默认随 FugaTensor 包提供。上述图编译成 `owned(+1) → external(scale) → owned(+3)` 三个步骤，同一 stream 顺序执行，中间张量保存在显存；ASK 自有计算区域仍可融合。安装步骤见 [Triton 配置](docs/安装.md#configure-triton)。

### 3. 查看 IR、溯源产物与缓存统计

```python
with ft.Engine(device="cpu") as engine:
    x = ft.tensor([1, 2, 3])
    y = (x + 1) * 2
    ft.evaluate(y)                  # 显式执行；多个输出可一起传入
    print(engine.cache_info())     # 图缓存条目、编译次数与内存命中统计
    print(engine.last_fuga_ir)      # 最近执行的 Fuga IR
    print(engine.last_ask_ir)       # 提交给 ASK 的 MLIR 文本
    print(engine.last_report)      # ASK 编译报告
    print(engine.last_artifact_dir) # fuga.ir、frontend.mlir 与 fuga-to-ask.json
```

`ft.evaluate(a, b)` 合并两个待计算输出的依赖图，并返回原 Tensor；已计算结果不会重复执行。`shape`、`dtype`、`device` 与 `repr` 只读取元信息。普通 Python 的 `if`/`while` 在读取 0-D Tensor 真值时执行条件所需的图，再按实际条件继续组图；`ft.fun` 内部的隔离追踪仍不支持数据依赖控制流。

## 当前支持范围

| 项目 | 当前行为 |
| --- | --- |
| **数据类型** | `bool`、`i32`、`i64`、`f32`、`f16`（别名 `fp16`）、`bf16`；显式 `.astype(...)` 混合精度转换 |
| **基础运算** | `add`、`subtract`、`multiply`、`negative`、`square`、`cast`、`zeros_like`、`ones_like`（静态形状，含 0-D 标量与空张量，无隐式广播） |
| **控制流** | Fuga IR 静态 `if` 选择与有界 `for` 展开；普通 Python 支持 0-D Tensor 的 `if`/`while`，条件读取触发按需执行 |
| **自动微分** | `requires_grad`、`.backward()`、`.grad` 累加、`detach()`、`no_grad()`、`ft.fun` 自定义前反向、`create_graph=True` 高阶求导 |
| **权重导出** | 版本化 PyTree 权重目录，支持嵌套 dict/list/tuple、六种基础 dtype、内容寻址去重、原子导出与完整性校验 |
| **执行后端** | CPU（`.o` → `.so` → `ctypes`）、自有 CUDA（多输入/异构多输出，无需 Triton）、独立 Triton `cubin` 与多 kernel 调度、多输入静态外部算子；FlashAttention-2 前向/反向多 kernel 示例 |

暂未实现隐式广播、矩阵乘法与归约算子；含不透明 Triton 算子的路径支持一阶手写反向，不支持二阶及以上高阶微分。

## 测试

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
- **系统架构（`docs/architecture/`）**：[架构总览](docs/architecture/README.md) · [01. 张量模型与追踪机制](docs/architecture/01-张量模型与追踪机制.md) · [02. 原生图分析与 C++ 边界](docs/architecture/02-原生图分析与C++边界.md) · [03. 自动微分与自定义算子](docs/architecture/03-自动微分与自定义算子.md) · [04. ASK 桥接与多后端执行](docs/architecture/04-ASK桥接与多后端执行.md) · [05. 权重格式与导出](docs/architecture/05-权重格式与导出.md)
- **中间表示（`docs/ir/`）**：[Fuga IR 总览](docs/ir/README.md) · [01. 对象模型与验证](docs/ir/01-对象模型与验证.md) · [02. 控制流与可组合变换](docs/ir/02-控制流与可组合变换.md) · [03. 降级到 ASK](docs/ir/03-降级到ASK.md)

[多 kernel 计划契约](../ask/docs/08-图切分与多Kernel执行计划.md) · [Flash Attention 2 前向/反向示例](examples/flash_attention2.py) · [权重格式与导出](docs/architecture/05-权重格式与导出.md)

FlashAttention-2 前向是单个 `external` attention 步骤加自有前后计算；反向为统计量、dQ、dK、dV 四个单输出外部 kernel，当前执行计划还会重算一次前向 attention 以取得保存的 `O`，共调度五个外部步骤。

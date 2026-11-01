# FugaTensor

创建张量，把基础计算写成 Python 函数，交给 ASK 做 JIT：

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

第一次调用时记录 Fuga IR，完成框架层变换后降成 ASK IR，再由 ASK 编译。再次传入相同形状和类型的张量时复用代码；输入值可以变化。CPU 和 CUDA 上都能直接写 `x + 2`、`x * y`、`x.astype("f16")`，这些操作同样经过 JIT。用 `@ft.jit` 可以把一段表达式一起交给 ASK。

## Fuga 有自己的 IR

Python 追踪保存的是 Fuga 的类型、SSA 值、操作和 region。分支与循环可以在这一层保留结构，自动微分也在这一层生成反向图；提交给 ASK 前，再选择静态分支、展开有界循环并生成 ASK 图。Fuga IR 的节点没有包装 `ask.Value`，也没有把 ASK 文本字符串当作操作对象。

```python
print(arithmetic.last_fuga_ir)  # Fuga 图：fuga.add、fuga.mul 等
print(arithmetic.last_ask_ir)   # 降给 ASK 的静态图
```

`last_ir` 暂时保留原来的 ASK 文本含义。每个编译目录同时保存 `fuga.ir`、`frontend.mlir` 和逐值对应关系 `fuga-to-ask.json`，便于核对是哪一层改变了图。对象模型、控制流、基础 VJP 和 ASK 边界见 [Fuga IR 文档](docs/ir/README.md)。

fun 类定义正向与反向，backward 沿实际执行的计算传播梯度，jit 保留嵌套函数体并编译。接口和当前边界见 [JIT、backward 与 fun 类](docs/ir/04-JIT与自定义梯度提案.md)。

## 基础自动微分

创建张量时开启梯度，执行计算后调用 backward：

```python
@ft.jit
def loss(x):
    return ft.square(x) + 3 * x

x = ft.tensor(2.0, requires_grad=True)
y = loss(x)
y.backward()
print(x.grad.tolist())  # 7.0
```

张量输出必须指定输出梯度种子，不会隐式求和：

```python
x = ft.tensor([1, 2, 3], requires_grad=True)
dy = ft.tensor([1, 2, -1])
ft.square(x).backward(dy)
print(x.grad.tolist())  # [2.0, 8.0, -6.0]
```

参与计算的函数统一使用 `@ft.jit`。自己提供反向时，定义 fun 类；`forward` 和提供的 `backward` 都必须带 `@ft.jit`。`backward` 返回与浮点输入顺序对应的梯度 tuple：

```python
class Square(ft.fun):
    @ft.jit
    def forward(self, ctx, x):
        ctx.save_for_backward(x)
        return ft.square(x)

    @ft.jit
    def backward(self, ctx, dy):
        (x,) = ctx.saved_tensors
        return (2 * x * dy,)

square = Square()
x = ft.tensor(3.0, requires_grad=True)
square(x).backward()
print(x.grad.tolist())  # 6.0
```

梯度累加在叶子 `.grad` 中；`x.zero_grad()` 清理梯度，`x.detach()` 断开历史，`with ft.no_grad():` 可包住不需要记录梯度的调用。普通 backward 后释放图；`retain_graph=True` 保留历史，`create_graph=True` 还记录梯度的计算图，默认同时保留所需历史。

对已保存的 `first = x.grad` 再调用 backward，可以求下一阶；先 `x.zero_grad()`，避免把不同阶结果累加在一起。高阶计算必须由已知 Fuga 基础运算组成，纯 fun 的手写反向也可参与；若需要的不透明 Triton 正向或反向没有可组合规则，会在任何反向执行前报错。

裸 external 位于必经反向路径且缺少规则时，在反向编译和执行前报错。用户可以把 Triton 调用放在 fun 的正向或反向中；框架检查梯度接口，公式仍由用户负责。可运行 [autodiff.py](examples/autodiff.py) 展示基础计算、fun 类及手写 Triton 反向。

内置 `ft.add`、`ft.multiply`、`ft.subtract`、`ft.negative`、`ft.square`、`ft.cast`、`ft.zeros_like` 和 `ft.ones_like` 也都声明为 JIT 函数。`x - y`、`-x`、`x.square()` 使用同样的可见计算；减法、取负和平方展开为已有的加乘，不引入隐藏的 Python 数值计算。

## 同一段代码在 GPU 上运行

只需选择 `device="cuda"`，上面的函数不需要改变：

```python
x = ft.tensor([1, 2, 3], device="cuda")
y = ft.tensor([10, 20, 30], device="cuda")
print(arithmetic(x, y).tolist())  # [22.0, 44.0, 66.0]
```

这条路径使用 ASK 自己生成的 GPU kernel，`ASK_THIRD_PARTY_COMPILERS=none` 构建即可。ASK 把张量表达式降成包含线程索引、越界检查和访存的 LLVM IR，再用 LLVM `llc` 与 CUDA `ptxas` 生成可执行代码。它不需要 Triton，也不会插入 external。环境[配置一次](#setup)后，直接写 `x + 1` 或 `x.astype("f32")` 也在 CUDA 上执行。

## 把自己写的 Triton 算子放进图

先写普通的 Triton 代码，例如 [examples/kernels.py](examples/kernels.py)：

```python
import triton
import triton.language as tl

@triton.jit
def scale_kernel(inp, out, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    loaded = tl.load(inp + index, mask=index < N, other=0)
    tl.store(out + index, loaded * 2, mask=index < N)
```

再注册这个文件，说明哪个参数是输入、哪个是输出，以及特化常量和启动配置。在 FugaTensor 根目录运行：

```python
import fugatensor as ft

scale = ft.triton.kernel(
    "examples/kernels.py",
    entry="scale_kernel", input="inp", output="out",
    constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
    grid=lambda spec: ((spec.numel + 127) // 128, 1, 1),
    options={"num_warps": 4},
)

@ft.jit
def mixed(x):
    return scale(x + 1) + 3

x = ft.tensor([[1, 2], [3, 4]], dtype="f32", device="cuda")
print(mixed(x).tolist())
# [[7.0, 9.0], [11.0, 13.0]]
```

CUDA 环境只需在程序外或开头[配置一次](#setup)。注册后也可以直接调用 `scale(x)`。`spec` 只提供 `shape`、`dtype`、`device`、`numel`，没有张量数据。已有源码字符串时，用 `ft.triton.from_source(source_text, ...)`，其余参数相同；当前不接收进程内的 Triton JIT 对象，每份 DSL 文件或文本只支持一个 kernel 入口。

`scale` 的计算来自你写的文件。`scale(x + 1)` 先生成 `fuga.external`，其中保存源码、端口绑定和固定编译配置；降级时再原样提交为 ASK 实现记录及 `ask.external`。ASK 的 Triton 适配器负责编译、追踪与边界检查，再把前后的 ASK add 内联到原生 LLVM IR 中，生成自己的 cubin。FugaTensor 不替你生成 kernel body，也不直接调用 Triton launcher。

注册时读取文件，不在框架进程中 import 它。之后修改源文件不会改变已有的 `scale` 对象；需要重新注册。如果外层 `mixed` 已经编译，还要调用 `mixed.clear_cache()` 后重新追踪。

CUDA 创建张量时上传数据，JIT 调用提交计算，`tolist()` 等待并下载结果。热调用复用 Session 和模块，并为返回张量分配输出存储。可以直接看[入门测试](tests/test_quickstart.py)和[可运行示例](examples/jit_demo.py)，基础例子检查 ASK 独立执行；Triton 例子额外检查这份手写源码的接入。

<a id="setup"></a>

## 环境只需配置一次

在 FugaTensor 根目录运行：

```bash
python3 tools/bootstrap.py
.venv/bin/python examples/jit_demo.py --device both
.venv/bin/python examples/mixed_triton.py
.venv/bin/python examples/autodiff.py --device both --triton
```

初始化默认使用同级 `../ask`，以 `ASK_THIRD_PARTY_COMPILERS=all` 构建并安装 ASK，再把 FugaTensor 以可编辑方式装入 `.venv`。当前 `all` 包含 Triton 适配器；对应编译器包按 ASK 的依赖锁文件安装。LLVM/MLIR 缺失时使用 ASK 已有的锁定下载流程，已有 SDK 会复用。初始化脚本需要 Python 3.11+、pip 和系统 C/C++ 工具链，会补齐缺少或过旧的 CMake/Ninja；GPU 示例需要 CUDA 驱动与硬件。也可 `source .venv/bin/activate` 后使用普通 `python` 命令。

Fuga 自己也会构建必要的 nanobind C++ 扩展，负责 SSA 依赖和反向路径分析。Python 保留接口、追踪和求导规则展开；实际 Tensor 计算继续交给 ASK。这次迁移先覆盖图分析，后续可继续移动核心变换。

工具路径保存到 `.venv/share/fugatensor/engine.json`，运行时自动读取。通常不需要每次填写 `ASK_OPT`、`llc`、`ptxas` 或 worker 路径。`mixed_triton.py` 读取用户编写的 [kernels.py](examples/kernels.py)，把普通 Fuga 运算与 Triton 算子一起交给 ASK 做原生 LLVM 融合，并检查结果 `[7.0, 9.0, 11.0]`。

只需要基础 CPU/CUDA 算子时，可显式选择 `python3 tools/bootstrap.py --compilers none`；这仍包含 ASK 自有 GPU lowering。`--compilers triton` 选择单个适配器。其他目录和工具可通过 `--ask-root`、`--venv`、`--llvm-root`、`--ptxas`、`--jobs` 指定；已有编译器环境可用 `--compiler-python triton=/path/to/python` 复用。`all` 是 FugaTensor 初始化的默认值，ASK 自身的全局 CMake 默认值仍为 `none`。

下面的手动配置用于已有 ASK 构建或开发调试。Python 必须能加载该 ASK 构建的原生扩展；CPU JIT 还需要 C 链接器，默认 `cc`。

假设 ASK 与 FugaTensor 两个目录同级，在 FugaTensor 根目录运行 CPU 示例：

```bash
export ASK_ROOT="$(realpath ../ask)"
export PYTHONPATH="$PWD/src:$ASK_ROOT/build/core/python:$ASK_ROOT/python"
export ASK_OPT="$ASK_ROOT/build/core/bin/ask-opt"
python3 examples/jit_demo.py --device cpu --json .cache/cpu-demo.json
```

运行基础 GPU 图仍使用上面的 `core`（`none`）构建，只需补上代码生成工具：

```bash
export FUGATENSOR_LLC="$ASK_ROOT/.deps/llvm-1f126a6d-ubuntu-x64-1/bin/llc"
export FUGATENSOR_PTXAS=/path/to/cuda/bin/ptxas
python3 examples/jit_demo.py --device both --json .cache/cpu-cuda-demo.json
```

只有要运行用户写的 Triton 算子时，才切换到启用 Triton 适配器的构建并配置它的编译环境：

```bash
export PYTHONPATH="$PWD/src:$ASK_ROOT/build/triton/python:$ASK_ROOT/python"
export ASK_OPT="$ASK_ROOT/build/triton/bin/ask-opt"
export FUGATENSOR_TRITON_PYTHON=/path/to/triton-env/bin/python
export FUGATENSOR_TRITON_WORKER="$ASK_ROOT/adapters/triton/worker.py"
python3 examples/jit_demo.py --device both --triton --json .cache/triton-demo.json
```

运行时配置按显式调用参数、环境变量、保存的配置文件这个优先级取值。指定 `ask_opt` 或 `ASK_OPT` 表示切换 ASK 安装，此时不会读取当前环境保存的 SDK 和 worker 配置，所需工具应随这套安装一起提供。`FUGATENSOR_CONFIG` 可指定另一份配置文件。首次使用对应设备时建立默认运行时；也可以在程序开头调用 `ft.configure(device="cpu", ...)` 或 `ft.configure(device="cuda", ...)` 显式设置 `ask_opt`、`cache_dir`、`compilers`、`llc`、`ptxas`、`cc`。例如：

```python
ft.configure(device="cpu", cache_dir=".cache/my-project")
```

配置与张量计算分开：不需要给每个 `ft.tensor` 传编译器路径。需要自己管理运行时隔离时，仍可使用 `with ft.Engine(...)`。默认设备是 CPU；`device="cuda"` 或 `device="cuda:1"` 选择对应 GPU。用完默认运行时后可调用 `ft.shutdown()`；关闭后，旧张量不能继续计算或下载。

如果 ASK 已安装到当前 Python 环境，可以安装 FugaTensor：

```bash
python3 -m pip install --no-deps -e .
```

这里 `--no-deps` 保留已经准备好的 ASK，不会补齐缺失的引擎。示例的工具路径是显式输入，库不会把某台开发机的路径当成默认环境。

## 看生成的图和缓存

```python
print(arithmetic.cache_info())
print(arithmetic.last_fuga_ir)
print(arithmetic.last_ask_ir)
print(arithmetic.last_artifact_dir)
```

`cache_info()` 记录 FugaTensor 的特化次数和内存命中。CPU 首次编译生成 `.o` 和 `.so`，随后调用原生函数；目前没有 CPU 持久代码缓存。GPU 还复用 ASK 自己的 LLVM IR 与可执行产物缓存，细节保留在 `last_report`。

形状、dtype、运行时和 Python 静态参数决定特化。只换 Tensor 数据仍命中原特化。修改闭包或全局常量后，应调用 `function.clear_cache()`，让下一次调用重新追踪。

## 当前支持范围

| 项目 | 当前行为 |
|---|---|
| 创建张量 | Python 标量或矩形 list/tuple；支持显式 shape；默认 `f32` |
| 类型 | `bool`、`i32`、`i64`、`f32`、`f16`、`bf16`；`fp16` 是 `f16` 的别名 |
| 基础计算 | 加、减、乘、取负、平方、显式 cast，以及 zeros_like/ones_like；同形状同类型张量，或张量与 Python 常量 |
| Fuga IR | 独立的类型、SSA 值、operation、block 和 region；可验证并打印 |
| IR 控制流 | 显式 if/for/while region；ASK 入口目前只接受可静态选择的 if 和可有界展开的 for |
| Python 控制流 | 零维 Tensor 的 if/while 触发图中断，由 Python 选择并调度静态片段 |
| 自动微分 | `requires_grad`、backward、叶子梯度累加、fun 自定义反向；create_graph 支持可组合基础运算的高阶反向 |
| 组合调用 | 嵌套 JIT 与真实执行片段的反向连接；fun 的 forward/backward 都需 `@ft.jit` |
| CPU JIT | 多输入、多输出；静态形状，包括零维与空张量 |
| 自有 CUDA JIT | 多输入、多输出；静态形状，包括零维与空张量；每个输出可有自己的形状和 dtype |
| Triton 融合 | 删除无用参数后，一个实际输入、一个同形状同存储类型输出、一个 external |
| 混合精度 | 显式写 `astype("f32")` 计算后再转回，支持低精度存储与 f32 中间计算 |
| Python 参数 | Tensor 或按值特化的 `bool/int/float/str/None`；返回 Tensor 或 Tensor tuple |

自定义 Triton 源码仍需满足 ASK 适配器能够证明的逐元素范围；注册 API 接收源码，不代表任意 Triton kernel 都能融合。当前 external 后的计算只能使用 external 结果与常量，尚不能绕过它再次引用原始输入。反向会删除无用的物理输入，因此只依赖 dy 的手写 Triton backward 可正常接收运行时梯度；确实需要多个实际输入时仍受现有 external 限制。

当前没有张量广播、矩阵乘法和归约；不透明第三方运算不支持高阶反向。普通 JIT 中，零维 Tensor 可作为 Python 条件；向量条件仍拒绝，也未增加张量比较运算。fun 的隔离 forward/backward 内部仍要求静态追踪，不能对运行时 Tensor 条件进行图中断。bool 可以创建和转换，不能直接加乘。CPU bf16 编译及 CUDA bf16 执行由 ASK 检查实际硬件，不支持时明确报错。

自有 CUDA 路径和 Triton 融合路径共用 ASK 的设备张量、异步 Session、LLVM IR 缓存与 cubin 缓存。选择 CUDA 就在 CUDA 上计算，编译失败会报告错误。一个图目前最多有 64 个输入输出指针参数；加乘的两侧仍要求同形状同类型。独立输出可以有不同形状，不能据此进行张量广播。

没有图中断的 JIT 只在追踪时执行 Python；有运行时条件的函数每次执行 Python 控制逻辑，并复用已编译片段。需要的张量应作为参数传入；不能任意读取真实 Tensor 数据或创建张量来固化第一次输入。嵌套 JIT 继续记录计算，图中断按实际执行片段连接反向历史。用 `last_segments` 查看本次各片段的 IR 和报告；执行分工见[实现说明](docs/architecture.md)。

## 测试

```bash
# 在上面配置的环境中，运行真实 CPU 编译和数值验证。
python3 -m unittest discover -s tests -v

# 验证 none 构建的真实 GPU 执行，不设置 Triton 环境。
FUGATENSOR_TEST_CUDA=1 FUGATENSOR_TEST_NONE=1 python3 -m unittest discover -s tests -v

# 切换到 triton 构建后，额外验证用户手写 kernel 的融合。
FUGATENSOR_TEST_CUDA=1 FUGATENSOR_TEST_TRITON=1 python3 -m unittest discover -s tests -v
```

未要求 GPU 检查时，相关测试显示 skipped。显式要求后，缺失环境或编译失败会导致测试失败。测试复用同一表达式检查多个 dtype，以及更换输入值后的 JIT 命中。依赖、构建产物和缓存不提交到 Git。

`tests/test_jit_integration.py` 的 `BasicJITGraphs` 只定义一套图，CPU 与自有 CUDA 共享这些测试：六种 dtype、多输入输出、不同输出类型与形状、零维、空张量、尾部越界，以及 fp16/bf16 转 f32 计算再转回。硬件支持 bf16 就执行，不支持才验证明确拒绝。GPU 另测清除内存特化后，LLVM IR 与可执行产物缓存都能命中；用户 Triton 内核保留独立的可选回归测试。

`tests/test_autodiff.py` 复用 CPU/CUDA 图验证叶子梯度、多输出、共享输入累加、混合精度、历史释放、no_grad/detach 和手写 Triton 反向。`tests/test_higher_order.py` 检查可组合基础运算的二阶、三阶及不透明外部算子的提前拒绝；图中断、原语规则和内部 JVP 另有独立测试。

本次全量验证日志为 `build/final-composable-all-tests.log` 和 `build/final-composable-none-tests.log`。`build/final-composable-demo-installed.log` 记录从 `/tmp` 使用 Python `-I`、不设置工具路径环境变量的示例，涵盖 CPU/CUDA、运行时 if/while、三阶梯度和 Triton 一阶反向。这些本地输出不纳入 Git。

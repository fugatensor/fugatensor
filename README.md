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

第一次调用时记录 ASK 图并编译。再次传入相同形状和类型的张量时复用代码；输入值可以变化。CPU 和 CUDA 上都能直接写 `x + 2`、`x * y`、`x.astype("f16")`，这些操作同样经过 JIT。用 `@ft.jit` 可以把一段表达式一起交给 ASK。

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

`scale` 的计算来自你写的文件。FugaTensor 将源码快照原样放进 ASK 实现记录，`scale(x + 1)` 生成一个 `ask.external`。ASK 的 Triton 适配器负责编译、追踪与边界检查，再把前后的 ASK add 内联到原生 LLVM IR 中，生成自己的 cubin。FugaTensor 不替你生成 kernel body，也不直接调用 Triton launcher。

注册时读取文件，不在框架进程中 import 它。之后修改源文件不会改变已有的 `scale` 对象；需要重新注册。如果外层 `mixed` 已经编译，还要调用 `mixed.clear_cache()` 后重新追踪。

CUDA 创建张量时上传数据，JIT 调用提交计算，`tolist()` 等待并下载结果。热调用复用 Session 和模块，并为返回张量分配输出存储。可以直接看[入门测试](tests/test_quickstart.py)和[可运行示例](examples/jit_demo.py)，基础例子检查 ASK 独立执行；Triton 例子额外检查这份手写源码的接入。

<a id="setup"></a>

## 环境只需配置一次

FugaTensor 使用已经安装或构建好的 ASK。Python 必须能加载该 ASK 构建的原生扩展；CPU JIT 还需要 C 链接器，默认 `cc`。

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

库读取这些环境变量，首次使用对应设备时建立默认运行时。也可以在程序开头调用 `ft.configure(device="cpu", ...)` 或 `ft.configure(device="cuda", ...)` 显式设置 `ask_opt`、`cache_dir`、`compilers`、`llc`、`ptxas`、`cc`。例如：

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
print(arithmetic.last_ir)
print(arithmetic.last_artifact_dir)
```

`cache_info()` 记录 FugaTensor 的特化次数和内存命中。CPU 首次编译生成 `.o` 和 `.so`，随后调用原生函数；目前没有 CPU 持久代码缓存。GPU 还复用 ASK 自己的 LLVM IR 与可执行产物缓存，细节保留在 `last_report`。

形状、dtype、运行时和 Python 静态参数决定特化。只换 Tensor 数据仍命中原特化。修改闭包或全局常量后，应调用 `function.clear_cache()`，让下一次调用重新追踪。

## 当前支持范围

| 项目 | 当前行为 |
|---|---|
| 创建张量 | Python 标量或矩形 list/tuple；支持显式 shape；默认 `f32` |
| 类型 | `bool`、`i32`、`i64`、`f32`、`f16`、`bf16`；`fp16` 是 `f16` 的别名 |
| 基础计算 | 加法、乘法、显式 `astype`；同形状同类型张量，或张量与 Python 常量 |
| CPU JIT | 多输入、多输出；静态形状，包括零维与空张量 |
| 自有 CUDA JIT | 多输入、多输出；静态形状，包括零维与空张量；每个输出可有自己的形状和 dtype |
| Triton 融合 | 一个输入、一个同形状同存储类型输出、一个 Triton external |
| 混合精度 | 显式写 `astype("f32")` 计算后再转回，支持低精度存储与 f32 中间计算 |
| Python 参数 | Tensor 或按值特化的 `bool/int/float/str/None`；返回 Tensor 或 Tensor tuple |

自定义 Triton 源码仍需满足 ASK 适配器能够证明的逐元素范围；注册 API 接收源码，不代表任意 Triton kernel 都能融合。

当前没有张量广播、自动微分、矩阵乘法、归约和数据驱动的 Python 控制流。bool 可以创建和转换，不能直接加乘。CPU bf16 编译及 CUDA bf16 执行由 ASK 检查实际硬件，不支持时明确报错。

自有 CUDA 路径和 Triton 融合路径共用 ASK 的设备张量、异步 Session、LLVM IR 缓存与 cubin 缓存。选择 CUDA 就在 CUDA 上计算，编译失败会报告错误。一个图目前最多有 64 个输入输出指针参数；加乘的两侧仍要求同形状同类型。独立输出可以有不同形状，不能据此进行张量广播。

JIT 函数只在追踪时执行 Python，不要依赖其中的副作用在每次调用时发生。追踪时不能读取真实 Tensor 数据、创建实际张量或触发即时计算；需要的张量应作为函数参数传入。当前不支持嵌套 `@jit`，普通张量表达式辅助函数可以在追踪期间调用。执行分工见[实现说明](docs/architecture.md)。

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

本轮在 RTX 3080 上验证：`none` 构建执行 56 项测试全部通过，另有 7 项 Triton 测试未启用；切换到 Triton 构建并启用外部算子测试后，63 项全部通过。基础 CPU/CUDA 示例与可选 Triton 示例也已执行并核对结果。本地记录位于 `build/owned-cuda-*.log` 和对应 JSON，不纳入 Git。

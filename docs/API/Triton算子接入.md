返回 API：[分类目录](README.md) · 相关 API：[设备与执行](设备与执行.md) · [自动微分与自定义函数](自动微分与自定义函数.md)

# Triton 算子接入

`ft.triton` 将 Triton kernel 接入张量计算图。调用返回待计算 Tensor，求值时由 worker 编译为 cubin，再交给 ASK 执行。版本与环境配置见[安装说明](../安装.md#configure-triton)。

<a id="registration"></a>

## 1. 注册入口与返回值

| API | 输入与行为 |
| --- | --- |
| `ft.triton.kernel(source_path, **kwargs)` | 按 UTF-8 读取源码文件并保存快照，不导入或执行该模块；`kwargs` 必须指定 `entry`。 |
| `ft.triton.from_source(source, **kwargs)` | 注册非空源码字符串；`kwargs` 必须指定 `entry`。 |
| `ft.triton.from_jit(function, **kwargs)` | 保存已加载的 Triton JIT 函数及依赖，入口名取原函数名。 |

三者都返回可调用的 `TritonKernel`。注册对象提供只读属性 `source`（源码字符串）、`source_path`（`pathlib.Path` 或 `None`）、`entry`（入口名）。修改原文件不会改变已有注册，更新代码后应重新注册。

调用形式为 `operation(x)` 或 `operation(x, y, ...)`，位置顺序对应 `input` / `inputs` 的声明。使用 `output=` 时返回一个 Tensor；使用 `outputs=` 时按字典顺序返回 tuple，即使只有一个输出也如此。所有输入须属于同一个引擎，实际执行要求 CUDA。

<a id="registration-arguments"></a>

## 2. 注册参数

这些参数传给上述注册函数；表中 `entry` 只用于源码入口。

| 参数 | 默认值与约束 |
| --- | --- |
| `entry` | 必填的源码顶层函数名；源码须恰好定义一次该函数，执行时必须是 Triton JIT 函数。 |
| `input=None` / `inputs=None` | 必须二选一。`input="X"` 绑定一个输入；`inputs=("X", "Y")` 按序绑定多个输入，名字须非空、互不重复。 |
| `output=None` / `outputs=None` | 必须二选一。`output="Y"` 绑定一个输出；`outputs={"Y": {...}, "Z": {...}}` 为每个输出声明元信息。输入与输出参数名不能重叠。 |
| `output_shape=None`、`output_dtype=None` | 仅用于 `output=`。可传静态值或回调；dtype 使用 `ft.fp32` 等类型对象，`None` 沿用首个输入的形状或类型。 |
| `outputs` 中的 `shape`、`dtype` | 每项只能包含这两个字段，各字段可省略、设为 `None`、静态值或回调；dtype 使用类型对象，默认沿用首个输入。不能再同时传 `output_shape` / `output_dtype`。 |
| `grid` | 必填，整数、长度 1–3 的整数序列或回调。各维须满足 `0 < n < 2**31`，不足三维时补 1。 |
| `constants=None` | `constexpr` 名字到值的字典，或返回该字典的回调；默认空字典。 |
| `scalars=None` | 普通标量参数名到 `(ABI 类型, 值)` 的字典，或返回该字典的回调；默认空字典。 |
| `options=None` | 固定编译选项字典，例如 `{"num_warps": 4}`；默认空字典，未指定项使用所选编译器版本的默认值。 |
| `config=None` | `@triton.autotune` 配置列表的非负整数索引。带 autotune 的入口须选择有效索引，按该配置编译。 |
| `source_path=None` | `from_source` / `from_jit` 可选的源码来源记录；`kernel` 自动记录解析后的文件路径。记录路径不会自动添加到 worker 的模块搜索路径。 |

参数名须为 ASCII 标识符且存在于入口签名中；入口不能使用 `*args` 或 `**kwargs`。所有非默认参数都须由张量端口、`constants`、`scalars` 或受支持的装饰器提供，不允许同名同时出现在 `constants` 和 `scalars`。输出由框架分配独立存储。

<a id="metadata-callbacks"></a>

## 3. 配置回调、constexpr 与启动标量

使用 `input=` 时，回调接收一个 `TensorSpec`；使用 `inputs=` 时，回调接收按声明顺序排列的 `TensorSpec` tuple，包括 `inputs=("X",)` 的情况。`TensorSpec` 是不可变元信息：`shape` 为整数 tuple，`dtype` 为类型对象（如 `ft.fp32`），`device` 为实际设备对象（如 `ft.device("cuda", 0)`），`numel` 为元素总数；没有张量数据或设备指针。输入经过 `reshape` 时，回调收到改变后的形状。

`constants`、`scalars`、`grid` 和输出 shape/dtype 回调在每次算子调用时执行，结果固定在本次表达式中。修改闭包配置只影响后续调用，并可能生成新的编译缓存项。

- `constants` 接受 `None`、bool、字符串、有限浮点数、整数（`-2**63 <= n < 2**64`）、列表/tuple 和字符串键的递归字典。列表和 tuple 在 worker 中都变为 tuple。dtype 直接传 `ft.fp16` 等类型对象，worker 会转换为对应的 Triton dtype；也接受已导入的 `tl.float16`。kernel 能如何使用这些值仍由 Triton constexpr 语义决定。
- `scalars` 可写为 `{"N": (ft.int32, n), "scale": (ft.fp32, 2.0)}`。六种类型对象会映射到对应启动 ABI；更细的标量类型可使用 ABI 字符串 `i1`、`i8/i16/i32/i64`、`u8/u16/u32/u64`、`fp16/bf16/fp32/fp64`。`ft.bool` / `i1` 要求 Python bool；整数须在对应范围内；浮点值须有限。这些值按声明类型传给 kernel，须由 Python 提供。
- 普通参数的 bool/int/float 字面量默认值可自动推断为 `i1`、`i32` 或 `i64`、`fp32`。更复杂的默认表达式应显式绑定 `scalars`。constexpr 默认值由 worker 从函数签名补齐。
- `@triton.heuristics` 在 worker 中求值，可读取标量、constexpr 和张量静态元信息。它接收 Triton 的张量元信息对象，支持 `shape`、`dtype`、`ndim`、`numel()`、`stride()`。

<a id="compiler-environment"></a>

## 4. 选择编译环境

安装 `fugatensor[all]` 后，CUDA 引擎默认使用当前 Python 环境的 Triton worker；安装方法见[可选依赖](../安装.md#installation-variants)。使用独立环境时，`ft.triton.environment(*, python, worker=None)` 返回供 `ft.configure(compilers=...)` 使用的环境描述；`python` 指定安装了受支持 Triton 稳定版的解释器，`worker=None` 使用 FugaTensor 自带 worker。以下示例从 `TRITON_PYTHON` 读取独立解释器路径，运行前需设置该环境变量。

```python
import os
import fugatensor as ft

environment = ft.triton.environment(python=os.environ["TRITON_PYTHON"])
ft.configure("cuda", compilers={"triton": environment})
```

编译环境在程序开始时配置。创建 CUDA 张量时传 `device="cuda"`，或用 `ft.set_default_device("cuda")` 设置默认设备。

也可通过 `FUGATENSOR_TRITON_PYTHON`、`FUGATENSOR_TRITON_WORKER` 配置。`ft.triton.environment_from_env(*, use_installed=False)` 默认在两项均未设置时返回 `None`；设为 `True` 时还会检查当前环境的 Triton 安装元数据。Python 变量为空，或仅设置 worker 时抛出 `ValueError`。CUDA 引擎按[配置优先级](设备与执行.md#configuration)选择环境。

图中含 FugaTensor 自有 CUDA 算术时，配置 [`llc` / `ptxas`](../安装.md#configure-cuda)。worker 会执行源码的模块顶层语句，源码须可信，导入的包须安装在所选环境中。

<a id="example"></a>

## 5. 自包含示例

安装 all 后，在兼容的 NVIDIA GPU 上注册一个返回两个输出的 kernel：

```python
import fugatensor as ft

source = """
import triton
import triton.language as tl

@triton.jit
def affine_pair(X, Y, Z, N, scale, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, mask=offsets < N, other=0)
    tl.store(Y + offsets, values * scale, mask=offsets < N)
    tl.store(Z + offsets, values + 1, mask=offsets < N)
"""
operation = ft.triton.from_source(
    source,
    entry="affine_pair",
    input="X",
    outputs={"Y": {}, "Z": {}},
    scalars=lambda spec: {"N": (ft.int32, spec.numel), "scale": (ft.fp32, 2.0)},
    constants={"BLOCK": 128},
    grid=lambda spec: ((spec.numel + 127) // 128,),
    options={"num_warps": 4},
)
x = ft.tensor([1.0, 2.0, 3.0], device="cuda")
y, z = operation(x)
ft.evaluate(y, z)
print(y.tolist())  # [2.0, 4.0, 6.0]
print(z.tolist())  # [2.0, 3.0, 4.0]
```

<a id="support-boundaries"></a>

## 6. 当前支持边界

- 张量端口使用六种基础 dtype 和静态连续存储。输入、输出都必须非空；0-D 张量有一个元素，可以使用。普通启动参数是扁平指针或标量，不支持运行时 tuple 参数及 TMA tensor descriptor。
- 支持编译元数据声明的动态共享内存。需要非零 global/profile scratch、tensor memory、cluster（`num_ctas != 1`）、协作式启动、PDL 或插桩启动的产物会报错拒绝。
- `options` 限于所选受支持版本的 CUDA 编译选项；禁止覆盖 `arch`、`backend_name`、`ir_override`、`extern_libs` 和 `ptx_options`。
- autotune 仅选择已声明配置；带输入重置/恢复、自定义 pre/post hook、配置 pre-hook 或 IR override 的用法不受支持。显式 `constants` / `options` 可覆盖所选配置，heuristics 随后求值。
- `from_jit` 可保存引用的 Python/JIT helper、全局常量和模块导入；依赖同名冲突或无法重建的程序式调优包装会报错。自定义 JIT repr、launch metadata 和 pre-run hook 也不能快照。
- Triton 算子的梯度通过 [`ft.fun`](自动微分与自定义函数.md) 定义反向；含不透明外部算子的反向路径目前只支持一阶微分。各编译器产物分别作为独立 kernel 调度。

编译与调度过程见[自有 CUDA 与用户 Triton 算子执行](../architecture/ASK桥接与多后端执行.md#cuda-and-triton)，内置 FlashAttention 的调用与梯度用法见[attention API](attention.md)。

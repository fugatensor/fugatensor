# 从 FugaTensor 张量到 ASK 执行

[返回使用说明](../README.md) · [Fuga IR](ir/README.md)

FugaTensor 先把用户表达的计算变成自己的 Fuga IR。控制流和自动微分变换在这一层处理；可以交付执行的静态图再降成 ASK IR。ASK 继续负责目标相关优化、LLVM 降级和设备执行。两层都检查自己接收的图，Fuga 的验证不会替代 ASK 的验证。下面按一次调用的顺序说明。

## 1. Tensor 保存内存和元信息

`ft.tensor(data, dtype=..., shape=..., device=...)` 先选择该设备的默认运行时；没有指定设备时，采用当前 `with Engine` 上下文，或默认 CPU。环境变量与 `ft.configure` 负责一次性设置工具路径，张量计算不需要反复传它们。然后检查 Python 数据是否为矩形，并把它展平成一个连续序列。省略 shape 时从嵌套结构得出形状；显式 shape 的元素总数必须与数据一致。类型明确给出，不从第一项猜测。

CPU Tensor 持有 `HostStorage`，其 ctypes 缓冲区按照 ASK dtype 编码。f16/bf16 也使用实际的 16 位存储，不保存成一份 Python float 列表。编码与解码复用 ASK 的数值转换函数；这一步是数据搬运，不执行张量加乘。

CUDA Tensor 持有 ASK 的 `DeviceTensor`。分配、上传、事件和下载由 ASK Session 管理。`Tensor` 自身只保留 engine 和 storage；打印其描述不读取设备数据，`tolist()` 才取得结果。

## 2. 第一次 JIT 调用记录 Fuga 图

`@ft.jit` 保存 Python 函数。调用时先根据参数建立特化键：张量记录所属 Engine、shape 和 dtype，Python 标量记录类型和值。Tensor 的内存地址和数据不进入键，所以更换数据可以复用代码。

第一次遇到某个特化时，用追踪值替换 Tensor 参数，执行一次 Python 函数。`TraceTensor.value` 是 Fuga 自己的 SSA 值。加法映射到 `fuga.add`，乘法映射到 `fuga.mul`，`astype` 映射到 `fuga.cast`。Python 常量变成标量常量，并按张量形状 splat。返回值决定图的输出列表。

此时 `x` 只有类型和形状，没有输入元素。对它调用 `tolist()`、判断真假或使用数据驱动的控制流都会失败。追踪上下文也禁止读取闭包里的真实 Tensor 数据，以及创建真实 Tensor 或触发即时执行；否则 Python 可能把第一次输入的数据固化进后续所有调用。需要的数据必须通过 JIT 参数进入图。可以根据 `x.dtype` 这类特化时已知的元信息选择表达式，因为不同 dtype 本来就有不同的特化。

追踪产物是独立的 `fugatensor.ir.Module`：操作保存 operands、results、属性和嵌套 region，能够被验证和变换。它不依赖 ASK 的 `Value`、类型或字符串构图器。显式 IR builder 可以创建分支和循环；这不表示 Python `if tensor` 会自动转换成 region，Python 追踪仍按上一段的规则拒绝读取张量数据。

`lower_to_ask` 是两层之间的边界。它先处理可静态确定的控制流，再逐个映射操作、类型和外部实现，创建新的 `ask.Module`。动态条件、while 和没有具体形状的图在这一步明确拒绝，不能取第一次运行的数据来把它们偷偷固定下来。控制流和 VJP 的实现范围见 [Fuga IR 变换](ir/02-控制流与自动微分.md)。

## 3. CPU 首次编译与执行

[host.py](../src/fugatensor/backends/host.py) 的编译路径为：

```text
Fuga Module
  -> lower_to_ask：静态控制流处理与类型/操作映射
  -> ASK Module
  -> ask-opt --optimize --emit-object
  -> graph.o + report.json
  -> cc -shared
  -> graph.so
  -> ctypes 加载 ABI 报告中的实际函数
```

它在调用发生时编译，因此属于 JIT；当前没有使用 LLVM ORC，也没有把编译器嵌进 FugaTensor。ASK 可以用 nanobind 做前端准备；本机目标文件的生成使用配套 CLI。两者的能力报告先核对，防止混用两个构建。

不能根据图的名字猜入口 ABI。后端读取 ASK 报告中的函数符号、C 调用约定、输入输出参数位置、shape、dtype、每元素宽度和对齐要求。当前只接受张量边界。执行前再次检查实际输入与已编译端口相符，再为每个输出分配独立缓冲区。

`HostExecutable` 持有动态库句柄，保证函数指针使用期间库仍然存活。热调用只校验张量并调用该指针，不再启动编译器、链接器或执行 Python 算子循环。每个编译结果写入新的目录，目前 CPU 没有跨进程持久代码缓存。

## 4. ASK 自己生成 CUDA kernel

没有 external 的图直接交给 ASK 的自有 CUDA lowering。FugaTensor 从当前 Session 查询实际 GPU 架构，并把它与 ASK Module 一起传给 `Compiler.compile_cuda`。`compilers` 可以为空；`none` 构建包含这条路径。

```text
Fuga Module
  -> lower_to_ask
  -> ASK Module
  -> ASK 验证、形状推导与优化
  -> 自有 CUDA lowering：索引、掩码、加载、标量表达式、存储
  -> 原生 GPU LLVM IR
  -> llc -> PTX -> ptxas -> cubin
  -> ASK Session 提交设备张量
```

LLVM IR 中的线程按元素编号工作，每次访问都有对应输出元素数量的边界检查。加乘和 cast 计算中间值直接留在 kernel 内，不需要 FugaTensor 创建中间设备张量。不同输出可以使用不同形状和存储类型；每个算子的输入约束仍由 ASK 检查。

ASK 返回的 `port_map` 描述全部逻辑输入输出。FugaTensor 按输出端口逐一分配设备张量，再调用 `PreparedCUDAKernel.submit_many(inputs, outputs)`。张量数量、shape、dtype、字节数、设备归属和不允许的存储重叠由 ASK 执行层验证。所有参数都绑定到实际缓冲区，不按“只有一个输入输出”的旧假设启动 kernel。

首次编译后，热调用只分配输出并异步提交。下载结果时等待；连续计算使用同一个 Session 的 stream 顺序。这里没有 Triton DSL、第三方 worker 或 CPU 回退。

## 5. 用户源码成为一个 external

用户自己写 Triton kernel，然后用 `ft.triton.kernel(path, entry=..., input=..., output=..., constants=..., grid=..., options=...)` 注册。文件入口读取完整 UTF-8 文本；`ft.triton.from_source(text, ...)` 则接收现成字符串。注册不 import 该模块、不执行 DSL、不创建 GPU，也不接受框架进程中的 `@triton.jit` 对象。

返回的算子对象保存源码快照、源入口、参数绑定与固定配置。它不会根据一个算子名字拼出另一份计算代码。现有字典配置在注册时复制；之后改原文件或原字典，不会修改已经注册的算子。需要采用新代码时重新注册；外层 JIT 已有特化时还要 `clear_cache()` 再追踪。

`constants` 和 `grid` 可直接给值，也可接受 `TensorSpec` 回调。回调看到的只有 shape、dtype、device 和 numel，不能读取输入元素。它们在特化期间求值，结果写入 ASK 实现声明。回调依赖的其他 Python 状态也应保持固定；修改这类状态需要重新注册或清除相应 JIT 缓存。

算子对象接受真实 Tensor 时经过自己的 JIT；接受追踪值时生成 `fuga.external`。它保存独立的外部实现描述，降到 ASK 时才创建 `ask.semantics`、`ask.implementation` 和 `ask.external`。用户源码保持原样。当前接口声明一个输入、一个同形状同存储类型输出和无外部效果，用户 kernel 应符合这个契约。源参数名用 `input` / `output` 明确映射到逻辑端口；实际 LLVM 参数仍由 ASK 根据第三方导出结果验证。

例如用户文件中写的是 `loaded * 2`，应用代码写：

```python
@ft.jit
def mixed(x):
    return scale(x + 1) + 3
```

得到 Fuga add → external → Fuga add，经 `lower_to_ask` 映射为对应的 ASK 图。FugaTensor 调用 ASK 的 `Compiler.compile_cuda`，按 compiler ID 提供环境。ASK 的 Triton 适配器通过公开编译接口得到 LLVM IR，并在请求融合时处理追踪标记、布局与掩码证明和数值模式。FugaTensor 不改 kernel body、不在这里实现另一套 Triton 分析，也不调用 `kernel[grid](...)` 直接启动原算子。

ASK 返回规范化产物和自己的 cubin。FugaTensor 将其准备到常驻 Session，提交设备张量执行。提交遵守同一 stream 的顺序，结果下载时等待。相同图、dtype 和形状可以复用已准备的 executable。

能注册源码不代表任意 Triton 程序都能融合。当前 ASK 路径要求单输入、单输出、单 external，以及适配器能够证明的逐元素边界。共享内存、复杂布局、无法识别的数据流等情况由 ASK 报错；FugaTensor 不忽略失败或悄悄直接启动原 Triton 二进制。这些限制仅属于 external 融合，不限制前一节的自有 CUDA 图。

## 6. 哪一层缓存什么

| 缓存 | 归属 | 命中后跳过什么 |
|---|---|---|
| 函数特化与已准备的 executable | FugaTensor 内存 | Python 追踪及完整编译流程 |
| GPU LLVM IR | ASK 持久缓存 | 自有 CUDA 降级，或第三方导出与 LLVM 融合 |
| PTX / cubin | ASK 持久缓存 | LLVM 到 PTX 和 ptxas |
| 已加载 CUDA module | ASK Session | 同一 Session 内的重复加载 |

Python 静态参数值变化时会生成新特化。Tensor 内容变化时仍命中旧特化，实际输入缓冲区在调用时绑定。Engine 不同也不复用同一个 executable，避免把模块、stream 或设备张量跨运行时混用。特化缓存按 Engine 分组，用弱引用保存其归属；运行时关闭时主动移除对应特化。CUDA executable 仍然保持实际使用的 ASK Session，直到相关对象释放。

`function.cache_info()` 查看 FugaTensor 特化次数和命中；`last_fuga_ir` 查看变换前的 Fuga 图，`last_ask_ir` 查看提交 ASK 的静态图，`last_ir` 保留后者的兼容含义。编译目录同时写入 `fuga.ir`、`frontend.mlir` 和逐值映射 `fuga-to-ask.json`，循环展开得到的多个值保留原始来源。`last_report` 保留 ASK 的报告，其中 GPU 路径另有 IR 与可执行缓存记录。不要把内存 JIT 命中等同于某一次磁盘缓存命中：内存命中时根本没有再次请求磁盘缓存。

## 7. 这版的边界

已有 IR 层基础 VJP，尚无完整的 `Tensor.backward()`、梯度 tape、张量广播、视图、别名输出或跨设备复制。动态 region 能保存在 Fuga IR 中，当前执行桥接仍只接收能转为静态图的部分。CPU 输出使用新缓冲区，CUDA 输出同样由当前 Engine 分配。退出 `with Engine(...)`、调用 `engine.close()`，或者用 `ft.shutdown()` 关闭默认运行时后，使用已经关闭的 Engine 或混用两个 Engine 的张量会报错。

常量和张量都显式声明 dtype。低精度输入若需要 f32 中间计算，代码应写出 cast，再在返回之前转回所需存储类型。f16/bf16 的创建、CPU 编译和 GPU 执行是不同步骤：能编码一份数据不表示当前 CPU/GPU 能执行该类型，设备能力检查由 ASK 完成。

Python 函数中的副作用只在追踪时发生。闭包和全局对象不会被自动序列化或持续监视；会改变计算的值应写成静态参数。修改了捕获常量时，应调用 `function.clear_cache()` 后重新追踪。图构造失败或编译失败不登记成功特化，下一次调用仍会重新尝试。

当前注册协议的 constexpr 常量限于有符号 64 位整数。源码文本仍可包含浮点字面量；若需要浮点 constexpr 参数或更复杂的参数封送，需要先扩展 ASK 的请求协议。

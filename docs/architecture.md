# 从 FugaTensor 张量到 ASK 执行

[返回使用说明](../README.md) · [Fuga IR](ir/README.md)

FugaTensor 先把用户表达的计算变成自己的 Fuga IR。控制流和自动微分变换在这一层处理；可以交付执行的静态图再降成 ASK IR。ASK 继续负责目标相关优化、LLVM 降级和设备执行。两层都检查自己接收的图，Fuga 的验证不会替代 ASK 的验证。下面按一次调用的顺序说明。

## 1. Tensor 保存内存和元信息

`ft.tensor(data, dtype=..., shape=..., device=...)` 先选择该设备的默认运行时；没有指定设备时，采用当前 `with Engine` 上下文，或默认 CPU。环境变量与 `ft.configure` 负责一次性设置工具路径，张量计算不需要反复传它们。然后检查 Python 数据是否为矩形，并把它展平成一个连续序列。省略 shape 时从嵌套结构得出形状；显式 shape 的元素总数必须与数据一致。类型明确给出，不从第一项猜测。

CPU Tensor 持有 `HostStorage`，其 ctypes 缓冲区按照 ASK dtype 编码。f16/bf16 也使用实际的 16 位存储，不保存成一份 Python float 列表。编码与解码复用 ASK 的数值转换函数；这一步是数据搬运，不执行张量加乘。

CUDA Tensor 持有 ASK 的 `DeviceTensor`。分配、上传、事件和下载由 ASK Session 管理。Tensor 还保存 requires_grad、叶子梯度及需要的执行历史；打印描述不读取设备数据，`tolist()` 才取得结果。

## 2. 第一次 JIT 调用记录 Fuga 图

`@ft.jit` 保存 Python 函数。调用时先根据参数建立特化键：张量记录所属 Engine、shape 和 dtype，Python 标量记录类型和值。Tensor 的内存地址和数据不进入键，所以更换数据可以复用代码。

第一次遇到某个特化时，用追踪值替换 Tensor 参数，执行一次 Python 函数。`TraceTensor.value` 是 Fuga 自己的 SSA 值。加法映射到 `fuga.add`，乘法映射到 `fuga.mul`，`astype` 映射到 `fuga.cast`。Python 常量变成标量常量，并按张量形状 splat。返回值决定图的输出列表。

追踪值通常只有类型和形状，不能任意调用 `tolist()` 读取数据，也不能把闭包里的真实 Tensor 隐式带入计算。数据必须作为参数传入。shape、dtype 等静态元信息可以直接用于 Python 选择。

零维 Tensor 的 Python 条件有专门路径：先执行已经记录的前缀，读取标量真假，再在同一个调用帧中继续追踪实际分支。while 逐轮执行同样的流程。CUDA 条件读取会同步设备；向量条件明确拒绝，不隐式 any/all。包含这种图中断的函数每次运行 Python 控制逻辑，复用的是已编译片段。

追踪产物是独立的 `fugatensor.ir.Module`：操作保存 operands、results、属性和嵌套 region，能够被验证和变换。它不依赖 ASK 的 `Value`、类型或字符串构图器。显式 IR builder 的 if/for/while region 与 Python 图中断是两条路径：前者保存 IR 结构，后者由 Python 调度本次实际执行的静态片段。

`lower_to_ask` 是两层之间的边界。它先处理可静态确定的控制流，再逐个映射操作、类型和外部实现，创建新的 `ask.Module`。动态条件、while 和没有具体形状的图在这一步明确拒绝，不能取第一次运行的数据来把它们偷偷固定下来。控制流和 VJP 的实现范围见 [Fuga IR 变换](ir/02-控制流与自动微分.md)。

真实执行的片段通过 Tensor 历史相连。`output.backward()` 先检查全部反向计划，再用内部 IR VJP 生成静态计算并交给 ASK，最终把贡献累加到 requires_grad 叶子的 `.grad`。fun 类的 forward 和提供的 backward 都必须声明 `@ft.jit`；`fuga.call` 保留求导边界，选定规则后才展开。普通嵌套 JIT 继续记录外层图，图中断后的真实结果继续携带历史。接口和生命周期见 [fun 与 backward](ir/04-JIT与自定义梯度提案.md)。

fun 的隔离 forward/backward 暂不支持运行时 Tensor 条件中断，遇到时在执行前拒绝。`no_grad` 应包住真实 JIT 调用，不能放进待追踪的计算函数体；real Tensor 的 detach 可在外层使用，TraceTensor 暂无 detach。

Fuga 的必要 C++ 扩展 `_native` 负责 SSA 拓扑、所选输入的依赖传播、反向路径与活跃性分析。Python 暂时保留 IR 对象、追踪和反向规则生成；原生分析结果决定哪些操作需要处理。Tensor 算术与设备代码仍全部由 ASK 执行。这是核心逻辑的第一步迁移，没有宣称整个编译过程已经在 C++ 中，也没有未经测量的加速结论。

执行桥还会删除完全无用的物理输入，记录逻辑参数到实际端口的映射。调用方仍传原来的参数列表，执行包装器只选择底层代码需要的存储。例如反向只需要 dy 时，不把未使用的原始 x 强塞进 Triton 融合入口。

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
| 单图函数特化与已准备的 executable | FugaTensor 内存 | Python 追踪及完整编译流程 |
| 动态控制流的已编译片段 | FugaTensor 内存 | 该片段编译；Python 控制逻辑仍执行 |
| GPU LLVM IR | ASK 持久缓存 | 自有 CUDA 降级，或第三方导出与 LLVM 融合 |
| PTX / cubin | ASK 持久缓存 | LLVM 到 PTX 和 ptxas |
| 已加载 CUDA module | ASK Session | 同一 Session 内的重复加载 |

Python 静态参数值变化时会生成新特化。Tensor 内容变化时仍命中旧特化，实际输入缓冲区在调用时绑定。Engine 不同也不复用同一个 executable，避免把模块、stream 或设备张量跨运行时混用。特化缓存按 Engine 分组，用弱引用保存其归属；运行时关闭时主动移除对应特化。CUDA executable 仍然保持实际使用的 ASK Session，直到相关对象释放。

动态片段不能只按前向文本缓存：文本相同的两个 fun 可能有不同 backward 闭包。片段键还包含当前定义的身份与版本，保留各自的反向语义；普通可变闭包状态仍需要调用者显式 clear_cache。

`function.cache_info()` 查看 FugaTensor 特化次数和命中；`last_fuga_ir` 查看变换前的 Fuga 图，`last_ask_ir` 查看提交 ASK 的静态图，`last_ir` 保留后者的兼容含义。编译目录同时写入 `fuga.ir`、`frontend.mlir` 和逐值映射 `fuga-to-ask.json`，循环展开得到的多个值保留原始来源。`last_report` 保留 ASK 的报告，其中 GPU 路径另有 IR 与可执行缓存记录。不要把内存 JIT 命中等同于某一次磁盘缓存命中：内存命中时根本没有再次请求磁盘缓存。

包含图中断时，用 `last_segments` 逐段查看本次执行的 `fuga_ir`、`ask_ir`、`report` 和 `artifact_dir`。单个产物字段不能代表全部控制流；反向沿实际执行片段连接，而不沿某次首次追踪固定的分支。

## 7. 这版的边界

已有 backward、叶子梯度累加、fun 自定义反向、历史释放与保留、no_grad 和外层 detach。create_graph 把梯度继续表达为可组合的 Fuga 基础操作，支持对它再反传；不透明第三方正向或反向会在高阶请求的反向执行前拒绝。归约、张量广播、通用视图和跨设备复制仍未实现。动态 Python 控制流由运行时调度，ASK 每次仍只接收静态片段。CPU/CUDA 输出均由 Engine 分配；关闭 Engine 后不能继续计算或反向，跨 Engine 混用张量也会报错。

常量和张量都显式声明 dtype。低精度输入若需要 f32 中间计算，代码应写出 cast，再在返回之前转回所需存储类型。f16/bf16 的创建、CPU 编译和 GPU 执行是不同步骤：能编码一份数据不表示当前 CPU/GPU 能执行该类型，设备能力检查由 ASK 完成。

单图函数的 Python 代码只在追踪时运行；动态函数每次运行控制逻辑，因此不要依赖两条路径的 Python 副作用次数相同。闭包和全局对象不会自动持续监视；会改变计算的值应写成静态参数，修改捕获常量后显式 clear_cache。图构造或编译失败不登记成功特化。

当前注册协议的 constexpr 常量限于有符号 64 位整数。源码文本仍可包含浮点字面量；若需要浮点 constexpr 参数或更复杂的参数封送，需要先扩展 ASK 的请求协议。

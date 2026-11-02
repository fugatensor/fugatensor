# 04. ASK 桥接与多后端执行

返回架构总览：[降级到 ASK、多后端执行与多级缓存](README.md#arch-ask-backends) · 相关细节：[张量模型与追踪机制](01-张量模型与追踪机制.md) · [Fuga IR 降级到 ASK](../ir/03-降级到ASK.md) · [ASK 常驻运行时与多级缓存](../../../ask/docs/05-常驻运行时与多级缓存.md)

API：[设备与执行](../API/03-设备与执行.md) · [Triton 算子接入](../API/05-Triton算子接入.md) · [注意力](../API/08-注意力.md)

读取结果或反向传播触发待计算表达式合并为 Fuga IR `Module` 后，`Engine` 负责裁剪未使用的物理输入端口、降级为 ASK IR、写出跨层溯源产物，并调用 CPU 动态库或 GPU 常驻会话执行。

<a id="engine-lifecycle"></a>

## 1. `Engine` 生命周期与无用输入端口裁剪（`engine.py`）

`Engine` 管理 FugaTensor 与 [ASK 编译器及运行时](../../../ask/docs/02-前端接入与构建配置.md#python-sdk) 的连接状态：

- **延迟初始化与上下文隔离**：普通张量创建默认使用 CPU，按设备惰性创建并复用引擎；正常进程退出或调用 `ft.shutdown()` 时关闭。设备参数接受 `ft.device` 对象或字符串，未指定 CUDA 编号时解析为 0。`ft.set_default_device` 使用 `ContextVar` 保存后续创建的默认设备，切换不关闭引擎或移动已有张量；显式 `device=` 优先。需要独立资源时仍可创建 `Engine`，其上下文临时覆盖默认设备。Tensor 绑定创建它的引擎，关闭后读取或运算会报错；不同设备算术报 `RuntimeError`，同设备但不同独立引擎报 `ValueError`。`ft.synchronize` 只等待已提交的任务，不会求值待计算表达式。
- **无用物理输入端口裁剪**：
  反向传播或复合函数中常出现签名包含 `(x, dy)` 但图内仅消费 `dy` 的情况（如线性缩放算子的反向 `dx = dy * 2`）。`Engine._compile` 在降级前先调用 [`ir/transforms.py` 的 `prune`](02-原生图分析与C++边界.md#transforms-reuse) 找出实际被使用的输入参数，仅保留活跃输入端口编译为底层可执行文件，外层由 `_InputProjection` 在调用时按索引筛选实参。这样手写 Triton 反向不会因签名中的未用参数而被前端拒绝。
- **编译目录产物**：每次编译在产物目录写出 `fuga.ir`（Fuga IR 文本）、`frontend.mlir`（ASK MLIR）和 `fuga-to-ask.json`（见 [降级溯源表](../ir/03-降级到ASK.md#external-and-sourcemap)），可通过 `engine.last_artifact_dir`、`engine.last_fuga_ir`、`engine.last_ask_ir` 查看。

<a id="host-backend"></a>

## 2. CPU 后端：目标文件生成与 `ctypes` 调用（`backends/host.py`）

CPU 执行路径（`HostBackend`）调用 [ASK Host 共享循环降级](../../../ask/docs/03-图优化与自有代码生成.md#host-codegen)：

```text
Fuga Module -> lower_to_ask -> ASK::Compiler 原生代码生成 (graph.o + ABI报告)
            -> cc -shared (graph.so) -> ctypes 按 ABI 报告调用函数指针
```

- **进程内编译**：默认通过 FugaTensor `_native` 调用 ASK C++ `codegenJSON`，生成 `.o` 时不启动 `ask-opt`。显式指定 `ask_opt` 才走兼容 CLI 路径。
- **按报告绑定 ABI**：从 `report.json` 读取入口符号名及 `inputs` / `outputs` 的 shape、dtype 和字节数；重复调用时仅校验输入缓冲区、分配输出 `HostStorage` 并调用驻留内存的 C 函数指针。

<a id="cuda-and-triton"></a>

## 3. 自有 CUDA 与用户 Triton 算子执行（`engine.py` / `triton.py`）

GPU `Engine` 初始化时只保存配置。首次上传 Tensor 或访问 `engine.session` 时才创建常驻 [ASK `ExecutionSession`](../../../ask/docs/05-常驻运行时与多级缓存.md#execution-session)；编译图或访问 `engine.capability` 时查询设备能力，确定如 `sm_86` 的目标架构。编译时按图中是否含外部算子分派：

1. **纯自有计算图 → ASK 自有 CUDA 降级**：
   不依赖 Triton。`Engine` 将内部 ASK 图描述与 `arch` 交给私有编译编排，由 [ASK 自有 CUDA 后端](../../../ask/docs/03-图优化与自有代码生成.md#owned-cuda) 生成 `cubin`，加载到会话后准备 C++ `PreparedProgram`，通过 FugaTensor `_native.run_program` 直接提交到同一 CUDA stream。支持多输入、异构形状与类型的多输出、0-D 标量与空张量。
2. **含 `ft.triton.kernel` 的混合图 → 自有与外部 kernel 顺序调度**：
   - 通过 `ft.triton.kernel("kernels.py", entry=..., input=... 或 inputs=..., output=..., output_shape=..., output_dtype=..., constants=..., grid=..., options=...)` 或 `ft.triton.from_source(...)` 注册 `@triton.jit` 代码。
   - **源码快照**：注册时按 UTF-8 读取源码文本并复制配置字典，不在宿主进程导入用户模块或 Triton；后续修改原文件不影响已注册的算子对象。
   - **配置快照与延迟架构绑定**：每次调用算子时执行 `constants`、`scalars` 与 `grid` 回调，仅向其提供不含张量数据的 `TensorSpec(shape, dtype, device, numel)`；其中 dtype 和 device 为公开类型、设备对象，shape 保留 reshape 后的真实维度。返回配置在节点创建时复制、冻结；后续修改回调闭包只影响新的调用，包括算子位于普通 `ft.fun.forward` 中的情况。生成的 `fuga.external` 不含物理架构号；`lower_to_ask` 面向具体 GPU `Engine` 降级时再填入该显卡的 `arch` 生成 `ask.implementation` 与 `ask.external`。
   - 本包 [`_triton_worker.py`](../../src/fugatensor/_triton_worker.py) 在选定的解释器中调用 Triton，直接返回 `asm["cubin"]` 与启动元数据。通过 [ASK 通用 worker 接口](../../../ask/docs/04-外部产物与执行接口.md#worker-interface) 接入[执行计划调度](../../../ask/docs/08-图切分与多Kernel执行计划.md#program-runtime)，不经过 LLVM 导出或跨编译器融合。

ASK 在高层 MLIR 中切分静态图，按[执行计划 JSON v1](../../../ask/docs/08-图切分与多Kernel执行计划.md#plan-v1)保留自有计算区域的融合，每个外部节点成为独立步骤。各步骤通过设备中间缓冲区传递数据，在同一会话 stream 上顺序提交。用户仍直接写普通函数；`ft.evaluate(...)` 触发求值，所有计算均编译后交给 ASK 执行。非法图、坏导出及编译失败直接报错。

`ft.triton.environment(python=...)` 返回 ASK 通用 `CompilerEnvironment`，worker 默认取本包路径，也可显式指定。宿主不导入 Triton；编译子进程要求锁定的 Triton 3.7.1，按公开 JIT 参数顺序去掉 constexpr，再绑定该版本 CUDA ABI 末尾的两个空 scratch 指针。动态共享内存来自编译元数据；需要全局 scratch、tensor descriptor、cluster、协作式或 PDL 启动的产物明确报错。版本与 ABI 记录见 [`triton.dependencies.lock.json`](../../src/fugatensor/triton.dependencies.lock.json)。

`kernel` / `from_source` 接收可信 Python 源码，worker 会执行模块顶层代码，允许正常的全局常量、辅助函数、默认参数与包导入；依赖包须安装在 worker 环境中。`from_jit(existing_kernel, ...)` 接收调用方已经加载的 JIT 函数，递归保存引用的 Python/JIT helper 和全局常量，保留导入模块；注册后修改原函数文件或全局值不改变快照。不同模块依赖若使用同一名字绑定不同对象，会明确报错。快照使用 JIT 对象保存的源码和 jit 选项；无法重建的程序式调优包装、自定义 repr、launch metadata 或 pre-run hook 需要调用方提供显式源码。

缓存探测复用 Triton 官方编译器指纹，记录实际 `ptxas`、`libdevice`、运行时代码以及执行源码时已加载的模块文件摘要，因此跨模块或动态导入的依赖变化会使 外部产物缓存失效。每次 worker 使用空的临时 Python 缓存目录，不读取旧 `.pyc`，避免源码与时间戳相同长度变化时仍执行旧代码。产物缓存未命中时重新执行 Triton 编译。不接受自定义 `extern_libs`、自由形式 `ptxas` 参数或 IR/目标覆盖。

worker 的 `required_element_types` 记录端口存储类型，不反向解析内部 LLVM/PTX 来枚举计算类型。内部指令能否用于目标架构由 Triton 与 `ptxas` 编译校验，ASK 再验证实际设备与产物目标架构的兼容性。

多输入外部算子使用 `inputs=("Q", "K", "V")` 绑定源码参数，并通过 `output_shape`、`output_dtype` 的静态值或回调声明输出；这些回调与 `constants` / `grid` 一样仅接收 TensorSpec 元数据。使用 `inputs=` 时回调接收按输入顺序排列的 TensorSpec tuple，原 `input=` 单输入接口仍接收单个 TensorSpec。多个输出使用 `outputs={"Y": {"shape": ..., "dtype": ...}, "Z": {...}}`，按声明顺序返回 tuple；每项的 shape/dtype 可省略，默认沿用首个输入。即使图中只使用一个结果，也会保留 kernel 其他输出所需的物理缓冲区。输入与输出不能原地别名。

`scalars={"N": (ft.int32, 1024), "scale": (ft.fp32, 0.5)}` 或对应回调声明普通运行时参数；它们按每次图调用冻结，并在 CUDA 启动时按值传参，不改成 `constexpr`，也不从 GPU 张量读取数据。改变这些值会产生新的图配置和缓存键。普通 Python bool/int/float 字面量默认参数可自动推断标量类型；其他默认表达式需显式传入 scalars。整数、浮点标量的类型与范围由宿主 ABI 契约校验，ASK 原生运行时核对参数布局。

`constants` 可传递有限浮点数、整数、bool、字符串、None、列表/tuple 与 JSON 配置字典；序列在 worker 中变为 tuple。kernel 内可使用的类型和操作仍遵循 Triton 本身的 constexpr 规则，例如不把任意 Python 字典自动转换成可索引的 Triton 值。dtype 直接传 `ft.fp16` 等类型对象，也可在调用方已导入 Triton 时传 `tl.float16`。constexpr 默认值由 Triton 函数签名补齐。`@triton.heuristics` 在 worker 中使用标量、constexpr 与张量静态元数据（shape、dtype、ndim、numel()、stride()）求值；`@triton.autotune` 需要显式 `config=索引` 选择其配置列表的一项，不执行 benchmark。配置中的 constexpr 和编译选项可由显式注册参数覆盖；涉及输入修改或自定义调优 hook 的配置仍报错。

标量与双输出的完整用法见 [Triton 接入示例](../API/05-Triton算子接入.md#example)。

`ft.nn.functional.scaled_dot_product_attention` 在包内维护 FlashAttention 的源码、注册配置与 `ft.fun` 梯度规则。前向生成一个外部 attention 节点，可与前后自有计算组成同一张图。反向由统计量、dQ、dK、dV 四个独立 kernel 组成；当前保存值仍是 `fuga.call` 结果，反向计划会重算一次前向以取得保存的 `O`，因此含五个外部步骤。数值参考与执行计划断言位于测试目录，不参与 API 运行。

<a id="native-plan-replay"></a>

### 原生计划和 CUDA Graph

FugaTensor 的 GPU JIT 为单 kernel 和混合计划都启用 CUDA Graph。首次提交建立 graph executable；同一执行槽完成后复用，输入输出地址变化时更新相应节点参数。未完成的提交使用独立中间缓冲区和 graph 实例，不覆盖前次结果。每次整计划只返回一个完成事件。

C++ 计划保留 kernel、设备缓冲区和未完成资源。Python 边界仍保留外部框架的 owner，并检查张量 shape/dtype；设备归属、字节数、写权限和别名由原生计划统一检查。重复执行不在 Python 逐 kernel 分配和提交。事件完成后可以回收底层 CUDA event，但用户持有的旧 Event 保持完成状态。

ASK 还提供普通单 stream 计划提交，适合不重复运行的调用方；FugaTensor JIT 默认使用 graph，不在 graph 启动失败时静默回退。

<a id="cache-hierarchy"></a>

## 4. 图编译与 ASK 缓存分工

| 层级 | 位置 | 缓存键 | 命中后跳过的工作 |
| --- | --- | --- | --- |
| **1. 引擎图编译缓存** | 每个 FugaTensor `Engine` 内存 | 不可变模板的输出投影、DAG 连接与共享叶子关系、输出顺序、相关定义身份 | 图拼接、完整 IR 文本生成、Fuga IR 降级及后端编译；普通 Python 逻辑仍每次执行 |
| **2. 自有 GPU LLVM IR 缓存** | `Engine.cache_dir/ask/entries` 磁盘 | ASK IR 原文摘要、形状快照、目标与工具指纹 | 自有 LLVM 降级 |
| **3. 自有 GPU `cubin` 缓存** | `Engine.cache_dir/ask/executables/entries` 磁盘 | LLVM IR 摘要、产物契约、代码生成参数与工具/环境指纹 | `llc` PTX 生成与 `ptxas` 汇编 |
| **4. 外部编译产物缓存** | `Engine.cache_dir/ask/external/entries` 磁盘 | 完整请求、worker、编译器版本与环境指纹 | Triton 子进程编译；缓存中直接保存 `cubin` |
| **5. 会话已加载模块** | ASK `ExecutionSession` 会话内 | `cubin` 字节与入口名 | 同一 GPU 会话内的 `cuModuleLoadDataEx` |

`engine.cache_info()` 返回累计 `compilations`、累计 `cache_hits` 与当前 `specializations` 数；清缓存不重置这两个累计计数。`engine.clear_cache()` 清除引擎的已编译图与最近一次诊断信息，不使已计算 Tensor 失效，也不删除 ASK 磁盘缓存。执行后通过 `engine.last_fuga_ir`、`last_ask_ir`、`last_report`、`last_artifact_dir` 查看最近提交的图与编译报告；这些信息属于引擎，不属于普通 Python 函数。

模板冻结与结构键规则见[自动合图与引擎缓存](01-张量模型与追踪机制.md#jit-specialization)。CPU/CUDA 连续内存的批量导入导出及借用寿命见[buffer 接口](01-张量模型与追踪机制.md#buffer-io)。

ASK 不把语义相同但原文本不同的图自动归为一个缓存项。持久缓存的检查与失效条件见 [ASK 双层持久缓存](../../../ask/docs/05-常驻运行时与多级缓存.md#two-tier-cache)。

<a id="bootstrap-and-config"></a>

## 5. 安装后的配置与开发引导（`tools/bootstrap.py` / `_config.py`）

ASK 独立安装 C++ SDK，FugaTensor 构建时链接 `ASK::Compiler` 与 `ASK::Runtime`，wheel 携带对应共享库。Python 绑定只有 FugaTensor 一份；`_ask/` 是内部编译编排，不作为另一套公共 API。完整步骤见[安装文档](../安装.md)。

默认 CPU 目标文件生成直接调用原生编译器，随后由 `cc` 链接，不要求运行环境保留 LLVM SDK 或 ASK CLI。含自有计算的 CUDA 图仍需 `llc` / `ptxas`；仅含外部 kernel 时由 worker 返回完整二进制。

`Engine` 与 `_config.py` 按以下优先级读取配置：

1. **显式传参**：`ft.configure(device=..., ...)` 或 `ft.Engine(...)` 参数；
2. **环境变量**：`ASK_OPT`、`FUGATENSOR_CACHE_DIR`、`FUGATENSOR_LLC`、`FUGATENSOR_PTXAS`、`FUGATENSOR_TRITON_{PYTHON,WORKER}`（以及指定配置文件路径的 `FUGATENSOR_CONFIG`）；
3. **可选环境配置文件**：当前 Python 环境的 `sys.prefix/share/fugatensor/engine.json`，不存在时使用空配置。

`cc` 仅通过 `ft.Engine(cc=...)` 或 `ft.configure(cc=...)` 指定，没有 `FUGATENSOR_CC` 环境变量入口。当通过代码或环境变量显式指定另一套 `ask_opt` 路径时，不会混读默认 `engine.json` 中保存的 SDK 与 worker 路径。

安装已有 FugaTensor wheel 无需开发引导。`tools/bootstrap.py --ask-prefix SDK目录` 使用已安装 C++ SDK 构建 FugaTensor；工具配置的保留和重置规则见[开发引导](../安装.md#source-bootstrap)。

# 04. ASK 桥接与多后端执行

返回架构总览：[降级到 ASK、多后端执行与多级缓存](README.md#arch-ask-backends) · 相关细节：[张量模型与追踪机制](01-张量模型与追踪机制.md) · [Fuga IR 降级到 ASK](../ir/03-降级到ASK.md) · [ASK 常驻运行时与多级缓存](../../../ask/docs/05-常驻运行时与多级缓存.md)

读取结果或反向传播触发待计算表达式合并为 Fuga IR `Module` 后，`Engine` 负责裁剪未使用的物理输入端口、降级为 ASK IR、写出跨层溯源产物，并调用 CPU 动态库或 GPU 常驻会话执行。

<a id="engine-lifecycle"></a>

## 1. `Engine` 生命周期与无用输入端口裁剪（`engine.py`）

`Engine` 管理 FugaTensor 与 [ASK 编译器及运行时](../../../ask/docs/02-前端接入与构建配置.md#python-sdk) 的连接状态：

- **延迟初始化与上下文隔离**：默认按设备惰性创建全局引擎（首次使用 `device="cpu"` 或 `device="cuda"` / `"cuda:0"` 时建立，调用 `ft.shutdown()` 时关闭）；也可通过 `with ft.Engine(device=..., cache_dir=...) as eng:` 创建独立引擎。`Tensor` 绑定创建它的 `Engine`，使用已关闭引擎读取张量数据、继续运算，或跨 `Engine` 混用张量均会报错。
- **无用物理输入端口裁剪**：
  反向传播或复合函数中常出现签名包含 `(x, dy)` 但图内仅消费 `dy` 的情况（如线性缩放算子的反向 `dx = dy * 2`）。`Engine._compile` 在降级前先调用 [`ir/transforms.py` 的 `prune`](02-原生图分析与C++边界.md#transforms-reuse) 找出实际被使用的输入参数，仅保留活跃输入端口编译为底层可执行文件，外层由 `_InputProjection` 在调用时按索引筛选实参。这样手写 Triton 反向不会因签名中的未用参数而被前端拒绝。
- **编译目录产物**：每次编译在产物目录写出 `fuga.ir`（Fuga IR 文本）、`frontend.mlir`（ASK MLIR）和 `fuga-to-ask.json`（见 [降级溯源表](../ir/03-降级到ASK.md#external-and-sourcemap)），可通过 `engine.last_artifact_dir`、`engine.last_fuga_ir`、`engine.last_ask_ir` 查看。

<a id="host-backend"></a>

## 2. CPU 后端：目标文件生成与 `ctypes` 调用（`backends/host.py`）

CPU 执行路径（`HostBackend`）调用 [ASK Host 共享循环降级](../../../ask/docs/03-图优化与自有代码生成.md#host-codegen)：

```text
Fuga Module -> lower_to_ask -> ask-opt --optimize --emit-object (graph.o + report.json)
            -> cc -shared (graph.so) -> ctypes 按 ABI 报告调用函数指针
```

- **SDK 与 CLI 伴生校验**：生成 `.o` 前调用 `self.compiler._check_companion()`（结果缓存在 `Compiler` 实例上），比较进程内 `ask._native` 与外部 `ask-opt` 的完整 capabilities，确认声明的版本、能力与构建身份匹配。`build_id` 不含源码内容哈希，不能据此证明二者来自同一次构建，见 [ASK 版本契约](../../../ask/docs/02-前端接入与构建配置.md#version-contract)。
- **按报告绑定 ABI**：从 `report.json` 读取入口符号名及 `inputs` / `outputs` 的 shape、dtype 和字节数；重复调用时仅校验输入缓冲区、分配输出 `HostStorage` 并调用驻留内存的 C 函数指针。

<a id="cuda-and-triton"></a>

## 3. 自有 CUDA 与用户 Triton 算子执行（`engine.py` / `triton.py`）

GPU `Engine` 初始化时只保存配置。首次上传 Tensor 或访问 `engine.session` 时才创建常驻 [ASK `ExecutionSession`](../../../ask/docs/05-常驻运行时与多级缓存.md#execution-session)；编译图或访问 `engine.capability` 时查询设备能力，确定如 `sm_86` 的目标架构。编译时按图中是否含外部算子分派：

1. **纯自有计算图 → ASK 自有 CUDA 降级**：
   不依赖 Triton。`Engine` 将 `ask.Module` 与 `arch` 传给 `Compiler.compile_cuda`，由 [ASK 自有 CUDA 后端](../../../ask/docs/03-图优化与自有代码生成.md#owned-cuda) 生成 `cubin`，加载到会话后通过 `PreparedCUDAKernel.submit_many(inputs, outputs)` 异步提交到同一 CUDA stream。支持多输入、异构形状与类型的多输出、0-D 标量与空张量。
2. **含 `ft.triton.kernel` 的混合图 → 自有与外部 kernel 顺序调度**：
   - 通过 `ft.triton.kernel("kernels.py", entry=..., input=... 或 inputs=..., output=..., output_shape=..., output_dtype=..., constants=..., grid=..., options=...)` 或 `ft.triton.from_source(...)` 注册 `@triton.jit` 代码。
   - **源码快照**：注册时按 UTF-8 读取源码文本并复制配置字典，不在宿主进程导入用户模块或 Triton；后续修改原文件不影响已注册的算子对象。
   - **配置快照与延迟架构绑定**：每次调用算子时执行 `constants` 与 `grid` 回调，仅向其提供不含张量数据的 `TensorSpec(shape, dtype, device, numel)`。返回配置在节点创建时复制、冻结；后续修改回调闭包只影响新的调用，包括算子位于普通 `ft.fun.forward` 中的情况。生成的 `fuga.external` 不含物理架构号；`lower_to_ask` 面向具体 GPU `Engine` 降级时再填入该显卡的 `arch` 生成 `ask.implementation` 与 `ask.external`。
   - 本包 [`_triton_worker.py`](../../src/fugatensor/_triton_worker.py) 在选定的解释器中调用 Triton，直接返回 `asm["cubin"]` 与启动元数据。通过 [ASK 通用 worker 接口](../../../ask/docs/04-外部产物与执行接口.md#worker-interface) 接入[执行计划调度](../../../ask/docs/08-图切分与多Kernel执行计划.md#program-runtime)，不经过 LLVM 导出或跨编译器融合。

ASK 在高层 MLIR 中切分静态图，按[执行计划 JSON v1](../../../ask/docs/08-图切分与多Kernel执行计划.md#plan-v1)保留自有计算区域的融合，每个外部节点成为独立步骤。各步骤通过设备中间缓冲区传递数据，在同一会话 stream 上顺序提交。用户仍直接写普通函数；`ft.evaluate(...)` 触发求值，所有计算均编译后交给 ASK 执行。非法图、坏导出及编译失败直接报错。

`ft.triton.environment(python=...)` 返回 ASK 通用 `CompilerEnvironment`，worker 默认取本包路径，也可显式指定。宿主不导入 Triton；编译子进程要求锁定的 Triton 3.7.1，按公开 JIT 参数顺序去掉 constexpr，再绑定该版本 CUDA ABI 末尾的两个空 scratch 指针。动态共享内存来自编译元数据；需要全局 scratch、tensor descriptor、cluster、协作式或 PDL 启动的产物明确报错。版本与 ABI 记录见 [`triton.dependencies.lock.json`](../../src/fugatensor/triton.dependencies.lock.json)。

源码快照可定义多个 `@triton.jit` kernel 或辅助函数，只允许导入 `triton`、`triton.language` 与 `math`；不接受自定义 `extern_libs`、自由形式 `ptxas` 参数或 IR/目标覆盖，避免遗漏快照之外的编译依赖。缓存探测复用 Triton 官方编译器指纹，并记录运行时代码、实际 `ptxas` 和 `libdevice` 的内容摘要；ASK 缓存未命中时重新执行 Triton 编译。

worker 的 `required_element_types` 记录端口存储类型，不反向解析内部 LLVM/PTX 来枚举计算类型。内部指令能否用于目标架构由 Triton 与 `ptxas` 编译校验，ASK 再验证实际设备与产物目标架构的兼容性。

多输入外部算子使用 `inputs=("Q", "K", "V")` 绑定源码参数，并通过 `output_shape`、`output_dtype` 的静态值或回调声明输出；这些回调与 `constants` / `grid` 一样仅接收 TensorSpec 元数据。使用 `inputs=` 时回调接收按输入顺序排列的 TensorSpec tuple，原 `input=` 单输入接口仍接收单个 TensorSpec。当前外部算子只有一个输出，完整图可以联合求值多个结果。

当前 FlashAttention-2 前向计划为 `owned(q*0.5) -> external(attention) -> owned(*2)`；反向由四个单输出反向 kernel（统计量、dQ、dK、dV）组成，且当前 `ft.fun` 的保存值仍是 `fuga.call` 结果，执行计划会额外重算一次前向 attention 以取得保存的 `O`。因此反向计划包含前向重算、统计量、dQ、dK、dV 五个外部步骤。

<a id="cache-hierarchy"></a>

## 4. 图编译与 ASK 缓存分工

| 层级 | 位置 | 缓存键 | 命中后跳过的工作 |
| --- | --- | --- | --- |
| **1. 引擎图编译缓存** | 每个 FugaTensor `Engine` 内存 | 合并后的 Fuga IR 文本、相关 `ft.fun` 定义身份 | Fuga IR 降级及后端编译；普通 Python 逻辑仍每次执行 |
| **2. 自有 GPU LLVM IR 缓存** | `Engine.cache_dir/ask/entries` 磁盘 | ASK IR 原文摘要、形状快照、目标与工具指纹 | 自有 LLVM 降级 |
| **3. 自有 GPU `cubin` 缓存** | `Engine.cache_dir/ask/executables/entries` 磁盘 | LLVM IR 摘要、产物契约、代码生成参数与工具/环境指纹 | `llc` PTX 生成与 `ptxas` 汇编 |
| **4. 外部编译产物缓存** | `Engine.cache_dir/ask/external/entries` 磁盘 | 完整请求、worker、编译器版本与环境指纹 | Triton 子进程编译；缓存中直接保存 `cubin` |
| **5. 会话已加载模块** | ASK `ExecutionSession` 会话内 | `cubin` 字节与入口名 | 同一 GPU 会话内的 `cuModuleLoadDataEx` |

`engine.cache_info()` 返回累计 `compilations`、累计 `cache_hits` 与当前 `specializations` 数；清缓存不重置这两个累计计数。`engine.clear_cache()` 清除引擎的已编译图与最近一次诊断信息，不使已计算 Tensor 失效，也不删除 ASK 磁盘缓存。执行后通过 `engine.last_fuga_ir`、`last_ask_ir`、`last_report`、`last_artifact_dir` 查看最近提交的图与编译报告；这些信息属于引擎，不属于普通 Python 函数。

ASK 不把语义相同但原文本不同的图自动归为一个缓存项。持久缓存的检查与失效条件见 [ASK 双层持久缓存](../../../ask/docs/05-常驻运行时与多级缓存.md#two-tier-cache)。

<a id="bootstrap-and-config"></a>

## 5. 安装后的配置与开发引导（`tools/bootstrap.py` / `_config.py`）

ASK 是必需的运行依赖，但 ASK 与 FugaTensor 各自构建 wheel，再安装到同一运行环境。FugaTensor 的 wheel 构建不需要安装 ASK，也不读取其源码或 SDK。完整步骤见[独立构建与安装](../安装.md)。

CPU 引擎直接使用 ASK 包内的原生扩展和伴生 `ask-opt`，并通过 `cc` 链接 JIT 目标文件，不需要 LLVM SDK、同级 ASK 源码目录或 `engine.json`。含自有计算的 CUDA 图另需显式配置 `llc` / `ptxas`；仅含外部 Triton kernel 时，worker 已生成完整 `cubin`，无需 ASK 代码生成工具。Triton 环境与 ASK 构建配置无关。

`Engine` 与 `_config.py` 按以下优先级读取配置：

1. **显式传参**：`ft.configure(device=..., ...)` 或 `ft.Engine(...)` 参数；
2. **环境变量**：`ASK_OPT`、`FUGATENSOR_CACHE_DIR`、`FUGATENSOR_LLC`、`FUGATENSOR_PTXAS`、`FUGATENSOR_TRITON_{PYTHON,WORKER}`（以及指定配置文件路径的 `FUGATENSOR_CONFIG`）；
3. **可选环境配置文件**：当前 Python 环境的 `sys.prefix/share/fugatensor/engine.json`，不存在时使用空配置。

`cc` 仅通过 `ft.Engine(cc=...)` 或 `ft.configure(cc=...)` 指定，没有 `FUGATENSOR_CC` 环境变量入口。当通过代码或环境变量显式指定另一套 `ask_opt` 路径时，不会混读默认 `engine.json` 中保存的 SDK 与 worker 路径。

只使用两个 wheel 的安装无需执行 `tools/bootstrap.py`。该脚本以 `--ask-wheel` 或目标虚拟环境中已安装的 ASK 为输入，只构建、可编辑安装 FugaTensor。默认保留有效的已有工具配置，新环境写入 CPU 空工具配置；显式工具参数或 `--reset-config` 控制重写，详见[可选开发引导](../安装.md#source-bootstrap)。文档中的跨仓相对链接仅供浏览源码，不参与配置解析。

复杂外部算子的实算示例见 [`examples/flash_attention2.py`](../../examples/flash_attention2.py)：前向由分块在线 softmax 计算，反向通过 `ft.fun` 的显式规则拆成统计量、dQ、dK、dV 四个独立 Triton kernel；加上当前保存值机制带来的前向重算，实际由 ASK 执行计划在同一 stream 上调度五个外部 kernel。它不是 FlashAttention-2 的完整训练接口：不含 dropout、bias、变长序列、任意 stride 或性能承诺，含外部算子的路径仍只支持一阶手写反向。

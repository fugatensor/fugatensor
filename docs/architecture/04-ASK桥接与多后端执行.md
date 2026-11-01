# 04. ASK 桥接与多后端执行

返回架构总览：[降级到 ASK、多后端执行与多级缓存](README.md#arch-ask-backends) · 相关细节：[张量模型与追踪机制](01-张量模型与追踪机制.md) · [Fuga IR 降级到 ASK](../ir/03-降级到ASK.md) · [ASK 常驻运行时与多级缓存](../../../ask/docs/05-常驻运行时与多级缓存.md)

追踪或自动微分产出静态 Fuga IR `Module` 后，`Engine` 负责裁剪未使用的物理输入端口、降级为 ASK IR、写出跨层溯源产物，并调用 CPU 动态库或 GPU 常驻会话执行。

<a id="engine-lifecycle"></a>

## 1. `Engine` 生命周期与无用输入端口裁剪（`engine.py`）

`Engine` 管理 FugaTensor 与 [ASK 编译器及运行时](../../../ask/docs/02-前端接入与构建配置.md#python-sdk) 的连接状态：

- **延迟初始化与上下文隔离**：默认按设备惰性创建全局引擎（首次使用 `device="cpu"` 或 `device="cuda"` / `"cuda:0"` 时建立，进程退出或调用 `ft.shutdown()` 时释放）；也可通过 `with ft.Engine(device=..., cache_dir=...) as eng:` 创建独立引擎。`Tensor` 绑定创建它的 `Engine`，访问已关闭引擎或跨 `Engine` 混用张量均会报错。
- **无用物理输入端口裁剪（`_prune_unused_inputs`）**：
  反向传播或复合函数中常出现签名包含 `(x, dy)` 但图内仅消费 `dy` 的情况（如线性缩放算子的反向 `dx = dy * 2`）。`Engine.compile_fuga` 在降级前先调用 [`ir/transforms.py` 的 `prune`](02-原生图分析与C++边界.md#transforms-reuse) 找出实际被使用的输入参数，仅保留活跃输入端口编译为底层可执行文件，外层由 `_ProjectedExecutable` 在调用时按索引筛选实参。这样单输入手写 Triton 反向不会因多余的未用参数而违反单输入融合限制。
- **编译目录产物**：每次编译在产物目录写出 `fuga.ir`（Fuga IR 文本）、`frontend.mlir`（ASK MLIR）和 `fuga-to-ask.json`（见 [降级溯源表](../ir/03-降级到ASK.md#external-and-sourcemap)），可通过 `fn.last_artifact_dir`、`fn.last_fuga_ir`、`fn.last_ask_ir` 查看。

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

## 3. 自有 CUDA 与用户 Triton 算子执行（`backends/cuda.py` / `triton.py`）

GPU `Engine` 初始化时创建常驻 [ASK `ExecutionSession`](../../../ask/docs/05-常驻运行时与多级缓存.md#execution-session) 并查询物理设备的 Compute Capability（如 `sm_86`），随后按图中是否含外部算子分派：

1. **纯自有计算图 → ASK 自有 CUDA 降级**：
   不依赖 Triton。`CUDABackend` 将 `ask.Module` 与 `arch` 传给 `Compiler.compile_cuda`，由 [ASK 自有 CUDA 后端](../../../ask/docs/03-图优化与自有代码生成.md#owned-cuda) 生成 `cubin`，加载到会话后通过 `PreparedCUDAKernel.submit_many(inputs, outputs)` 异步提交到同一 CUDA stream。支持多输入、异构形状与类型的多输出、0-D 标量与空张量。
2. **含 `ft.triton.kernel` 的混合图 → `ASK → Triton → ASK` 融合**：
   - 通过 `ft.triton.kernel("kernels.py", entry=..., input=..., output=..., constants=..., grid=..., options=...)` 或 `ft.triton.from_source(...)` 注册 `@triton.jit` 代码。
   - **源码快照**：注册时按 UTF-8 读取源码文本并复制配置字典，不在宿主进程导入用户模块或 Triton；后续修改原文件不影响已注册的算子对象。
   - **延迟架构绑定**：`constants` 与 `grid` 回调仅接收不含张量数据的 `TensorSpec(shape, dtype, device, numel)` 生成设备无关的 `fuga.external`；`lower_to_ask` 面向具体 GPU `Engine` 降级时再填入该显卡的 `arch` 生成 `ask.implementation` 与 `ask.external`。
   - 由 [ASK Triton 适配器与通用边界内联器](../../../ask/docs/04-外部编译器适配与边界内联.md#triton-fusion) 将前后逐元素计算合并进同一 `cubin` 并提交给 `ExecutionSession` 执行。

<a id="cache-hierarchy"></a>

## 4. 五层缓存分工

| 层级 | 位置 | 缓存键 | 命中后跳过的工作 |
| --- | --- | --- | --- |
| **1. 单图 JIT 特化缓存** | FugaTensor `JITFunction` 内存 | `(Engine, 参数 shape/dtype, 静态标量值, fun 版本)` | Python 追踪、Fuga IR 降级及后端编译 |
| **2. 动态片段缓存** | FugaTensor `_segment_cache` 内存 | `(Engine, 片段 Fuga IR 文本, 活跃 fun 版本)` | 片段编译（外层 Python `if`/`while` 控制逻辑仍按次运行） |
| **3. GPU LLVM IR 持久缓存** | `Engine.cache_dir/ask/entries` 磁盘 | 提交的 ASK IR 原文摘要、形状快照、目标与工具指纹；外部路径还包含适配器、DSL 和环境指纹 | 自有降级或 Triton 子进程导出与边界内联 |
| **4. GPU `cubin` 持久缓存** | `Engine.cache_dir/ask/executables/entries` 磁盘 | LLVM IR 摘要、产物契约、代码生成参数与工具/环境指纹 | `llc` PTX 生成与 `ptxas` 汇编 |
| **5. 会话已加载模块** | ASK `ExecutionSession` 会话内 | `cubin` 字节与入口名 | 同一 GPU 会话内的 `cuModuleLoadDataEx` |

调用 `fn.clear_cache()` 仅清除第 1、2 层 FugaTensor 内存特化；第 3、4 层在各项键输入仍一致时可命中。ASK 不把语义相同但原文本不同的图自动归为一个缓存项，具体检查与失效条件见 [ASK 双层持久缓存](../../../ask/docs/05-常驻运行时与多级缓存.md#two-tier-cache)。

<a id="bootstrap-and-config"></a>

## 5. 安装后的配置与开发引导（`tools/bootstrap.py` / `_config.py`）

ASK 是必需的运行依赖，但 ASK 与 FugaTensor 各自构建 wheel，再安装到同一运行环境。FugaTensor 的 wheel 构建不需要安装 ASK，也不读取其源码或 SDK。完整步骤见[独立构建与安装](../安装.md)。

CPU 引擎直接使用 ASK 包内的原生扩展和伴生 `ask-opt`，并通过 `cc` 链接 JIT 目标文件，不需要 LLVM SDK、同级 ASK 源码目录或 `engine.json`。CUDA 引擎另需显式配置 `llc` / `ptxas`，用户 Triton 融合还需已启用适配器的 ASK 包及对应 worker 环境。

`Engine` 与 `_config.py` 按以下优先级读取配置：

1. **显式传参**：`ft.configure(device=..., ...)` 或 `ft.Engine(...)` 参数；
2. **环境变量**：`ASK_OPT`、`FUGATENSOR_CACHE_DIR`、`FUGATENSOR_LLC`、`FUGATENSOR_PTXAS`、`FUGATENSOR_TRITON_{PYTHON,WORKER}`（以及指定配置文件路径的 `FUGATENSOR_CONFIG`）；
3. **可选环境配置文件**：当前 Python 环境的 `sys.prefix/share/fugatensor/engine.json`，不存在时使用空配置。

`cc` 仅通过 `ft.Engine(cc=...)` 或 `ft.configure(cc=...)` 指定，没有 `FUGATENSOR_CC` 环境变量入口。当通过代码或环境变量显式指定另一套 `ask_opt` 路径时，不会混读默认 `engine.json` 中保存的 SDK 与 worker 路径。

只使用两个 wheel 的安装无需执行 `tools/bootstrap.py`。该脚本以 `--ask-wheel` 或目标虚拟环境中已安装的 ASK 为输入，只构建、可编辑安装 FugaTensor。默认保留有效的已有工具配置，新环境写入 CPU 空工具配置；显式工具参数或 `--reset-config` 控制重写，详见[可选开发引导](../安装.md#source-bootstrap)。文档中的跨仓相对链接仅供浏览源码，不参与配置解析。

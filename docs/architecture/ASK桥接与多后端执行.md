# ASK 桥接与多后端执行

返回架构总览：[降级到 ASK、多后端执行与多级缓存](README.md#arch-ask-backends) · 相关细节：[张量模型与追踪机制](张量模型与追踪机制.md) · [Fuga IR 降级到 ASK](../ir/降级到ASK.md) · [ASK 常驻运行时与多级缓存](../../../ask/docs/常驻运行时与多级缓存.md)

API：[设备与执行](../API/设备与执行.md) · [Triton 算子接入](../API/Triton算子接入.md) · [attention](../API/attention.md)

读取结果或反向传播时，`Engine` 将所需表达式合并为 Fuga IR，裁剪输入，降级成 ASK IR，再调用 CPU 动态库或 GPU 会话执行。

<a id="engine-lifecycle"></a>

## 1. `Engine` 生命周期与无用输入端口裁剪（`engine.py`）

`Engine` 连接 [ASK 编译器及运行时](../../../ask/docs/前端接入与构建配置.md#python-sdk)，管理以下状态：

- **引擎与设备**：默认引擎按设备创建和复用，在进程退出或 `ft.shutdown()` 时关闭。`ft.set_default_device` 用 `ContextVar` 保存后续张量的默认设备，已有张量保留原设备和引擎；`device=` 优先。独立 `Engine` 的上下文可临时覆盖默认选择。引擎关闭后，其张量的读取和运算会报错；跨设备运算报 `RuntimeError`，同设备的跨引擎运算报 `ValueError`。
- **无用物理输入端口裁剪**：
  例如反向函数签名是 `(x, dy)`，计算却只有 `dx = dy * 2`。`Engine._compile` 用 [`prune`](原生图分析与C++边界.md#transforms-reuse) 找出实际输入，只为 `dy` 编译物理端口；调用时 `_InputProjection` 按索引筛选实参。
- **编译目录产物**：每次编译在产物目录写出 `fuga.ir`（Fuga IR 文本）、`frontend.mlir`（ASK MLIR）和 `fuga-to-ask.json`（见 [降级溯源表](../ir/降级到ASK.md#external-and-sourcemap)），可通过 `engine.last_artifact_dir`、`engine.last_fuga_ir`、`engine.last_ask_ir` 查看。

<a id="host-backend"></a>

## 2. CPU 后端：目标文件生成与 `ctypes` 调用（`backends/host.py`）

CPU 执行路径（`HostBackend`）调用 [ASK Host 共享循环降级](../../../ask/docs/图优化与自有代码生成.md#host-codegen)：

```text
Fuga Module -> lower_to_ask -> ASK::Compiler 原生代码生成 (graph.o + ABI报告)
            -> cc -shared (graph.so) -> ctypes 按 ABI 报告调用函数指针
```

- **编译入口**：默认由 `_native` 调用 ASK C++ `codegenJSON`；`ask_opt` 参数可选择 CLI 路径。
- **ABI 绑定**：从 `report.json` 读取入口及端口的 shape、dtype 和字节数。后续调用校验输入、分配 `HostStorage` 输出，再调用已加载的函数指针。

<a id="cuda-and-triton"></a>

## 3. 自有 CUDA 与用户 Triton 算子执行（`engine.py` / `triton.py`）

GPU 引擎首次上传 Tensor 或访问 `engine.session` 时创建 [ASK `ExecutionSession`](../../../ask/docs/常驻运行时与多级缓存.md#execution-session)。编译图或读取 `engine.capability` 时查询目标架构，随后按图中是否含外部算子分派：

1. **纯自有计算图 → ASK 自有 CUDA 降级**：
   `Engine` 将 ASK 图和 `arch` 交给 [ASK CUDA 后端](../../../ask/docs/图优化与自有代码生成.md#owned-cuda) 生成 cubin。会话加载后准备 C++ `PreparedProgram`，由 `_native.run_program` 提交到 CUDA stream。支持多输入、异构形状与类型的多输出、0-D 和空张量。
2. **含 `ft.triton.kernel` 的混合图 → 自有与外部 kernel 顺序调度**：
   - 通过 `ft.triton.kernel("kernels.py", entry=..., input=... 或 inputs=..., output=..., output_shape=..., output_dtype=..., constants=..., grid=..., options=...)` 或 `ft.triton.from_source(...)` 注册 `@triton.jit` 代码。
   - 注册时保存 UTF-8 源码和配置快照，后续文件修改须重新注册。
   - 每次调用执行 `constants`、`scalars` 和 `grid` 回调，传入 `TensorSpec(shape, dtype, device, numel)`。shape 使用 reshape 后的维度，dtype 和 device 使用公开对象。返回配置随节点冻结，闭包变化只影响后续调用。降级时再绑定所选 GPU 的 `arch`，生成 `ask.implementation` 和 `ask.external`。
   - [`_triton_worker.py`](../../src/fugatensor/_triton_worker.py) 在所选解释器中调用 Triton，返回 `asm["cubin"]` 和启动信息，接入 [worker 接口](../../../ask/docs/外部产物与执行接口.md#worker-interface)与[执行计划](../../../ask/docs/图切分与多Kernel执行计划.md#program-runtime)。

ASK 在 MLIR 中切分图，融合自有区域，将外部节点分别排入[执行计划 JSON v1](../../../ask/docs/图切分与多Kernel执行计划.md#plan-v1)。步骤通过设备缓冲区传递数据，在会话 stream 上顺序提交；非法图、损坏产物和编译失败会报错。

安装 all 后，引擎从当前环境的安装元数据发现 Triton，以该解释器启动 worker。独立环境通过 `ft.triton.environment(python=...)` 选择。worker 检查版本，按 JIT 参数顺序去掉 constexpr，补齐 CUDA ABI 的 scratch 参数，并读取动态共享内存大小。全局 scratch、tensor descriptor、cluster、协作式或 PDL 启动当前不受支持。版本与 ABI 记录见 [`triton.dependencies.lock.json`](../../src/fugatensor/triton.dependencies.lock.json)。

`kernel` / `from_source` 接收可信 Python 源码，worker 会执行模块顶层代码，允许正常的全局常量、辅助函数、默认参数与包导入；依赖包须安装在 worker 环境中。`from_jit(existing_kernel, ...)` 接收调用方已经加载的 JIT 函数，递归保存引用的 Python/JIT helper 和全局常量，保留导入模块；注册后修改原函数文件或全局值不改变快照。不同模块依赖若使用同一名字绑定不同对象，会明确报错。快照使用 JIT 对象保存的源码和 jit 选项；无法重建的程序式调优包装、自定义 repr、launch metadata 或 pre-run hook 需要调用方提供显式源码。

缓存探测记录 Triton 编译器指纹、`ptxas`、`libdevice`、运行时代码及已加载依赖文件的摘要。`get_cache_invalidating_env_vars()` 返回的编译环境保存为 `producer.environment`；依赖或浮点融合开关变化时，缓存失效。worker 每次使用空的临时 Python 缓存目录，避免旧 `.pyc` 干扰。自定义 `extern_libs`、自由形式 `ptxas` 参数和 IR/目标覆盖会被拒绝。

`required_element_types` 记录端口存储类型。Triton 和 `ptxas` 检查内部指令的架构要求，ASK 检查产物与实际设备是否兼容。

多输入外部算子使用 `inputs=("Q", "K", "V")` 绑定源码参数，并通过 `output_shape`、`output_dtype` 的静态值或回调声明输出；这些回调与 `constants` / `grid` 一样仅接收 TensorSpec 元数据。使用 `inputs=` 时回调接收按输入顺序排列的 TensorSpec tuple，原 `input=` 单输入接口仍接收单个 TensorSpec。多个输出使用 `outputs={"Y": {"shape": ..., "dtype": ...}, "Z": {...}}`，按声明顺序返回 tuple；每项的 shape/dtype 可省略，默认沿用首个输入。即使图中只使用一个结果，也会保留 kernel 其他输出所需的物理缓冲区。输入与输出不能原地别名。

`scalars={"N": (ft.int32, 1024), "scale": (ft.fp32, 0.5)}` 或对应回调声明按值启动参数，值在每次调用时冻结，变化时生成新的图配置和缓存键。bool/int/float 字面量默认参数可推断标量类型，其他默认表达式通过 scalars 绑定。宿主校验类型和取值范围，ASK 运行时核对参数布局。

`constants` 可传递有限浮点数、整数、bool、字符串、None、列表/tuple 与 JSON 配置字典；序列在 worker 中变为 tuple。kernel 内可使用的类型和操作仍遵循 Triton 本身的 constexpr 规则，例如不把任意 Python 字典自动转换成可索引的 Triton 值。dtype 直接传 `ft.fp16` 等类型对象，也可在调用方已导入 Triton 时传 `tl.float16`。constexpr 默认值由 Triton 函数签名补齐。`@triton.heuristics` 在 worker 中使用标量、constexpr 与张量静态元数据（shape、dtype、ndim、numel()、stride()）求值；`@triton.autotune` 需要显式 `config=索引` 选择其配置列表的一项，不执行 benchmark。配置中的 constexpr 和编译选项可由显式注册参数覆盖；涉及输入修改或自定义调优 hook 的配置仍报错。

参数和双输出用法见 [Triton 接入示例](../API/Triton算子接入.md#example)。

`ft.nn.functional.scaled_dot_product_attention` 使用包内 FlashAttention 源码和 `ft.fun` 梯度规则。前向是一个外部节点，可与前后自有计算组图。反向含统计量、dQ、dK、dV 四个 kernel；当前还会重算前向以取得保存的 `O`，共五个外部步骤。

<a id="native-plan-replay"></a>

### 原生计划和 CUDA Graph

GPU JIT 的单 kernel 和混合计划都使用 CUDA Graph。首次提交建立 graph executable，执行槽完成后可复用，并更新变化的输入输出地址。在途调用各用一份中间缓冲区和 graph 实例，每次提交返回一个完成事件。

C++ 计划保留 kernel 和在途缓冲区；Python 绑定保留外部框架的 owner，并检查 shape/dtype。原生计划检查设备归属、字节数、写权限和别名，再一次提交所有步骤。事件完成后可回收底层 CUDA event，用户持有的 Event 保持完成状态。

ASK 同时提供普通 stream 提交和 Graph replay。FugaTensor JIT 默认使用后者，启动失败时返回错误。

<a id="cache-hierarchy"></a>

## 4. 图编译与 ASK 缓存分工

| 层级 | 位置 | 缓存键 | 命中后跳过的工作 |
| --- | --- | --- | --- |
| **1. 引擎图编译缓存** | 每个 FugaTensor `Engine` 内存 | 不可变模板的输出投影、DAG 连接与共享叶子关系、输出顺序、相关定义身份 | 图拼接、完整 IR 文本生成、Fuga IR 降级及后端编译；普通 Python 逻辑仍每次执行 |
| **2. 自有 GPU LLVM IR 缓存** | `Engine.cache_dir/ask/entries` 磁盘 | ASK IR 原文摘要、形状快照、目标与工具指纹 | 自有 LLVM 降级 |
| **3. 自有 GPU `cubin` 缓存** | `Engine.cache_dir/ask/executables/entries` 磁盘 | LLVM IR 摘要、产物契约、代码生成参数与工具/环境指纹 | `llc` PTX 生成与 `ptxas` 汇编 |
| **4. 外部编译产物缓存** | `Engine.cache_dir/ask/external/entries` 磁盘 | 完整请求、worker、编译器版本与环境指纹 | Triton 子进程编译；缓存中直接保存 `cubin` |
| **5. 会话已加载模块** | ASK `ExecutionSession` 会话内 | `cubin` 字节与入口名 | 同一 GPU 会话内的 `cuModuleLoadDataEx` |

`engine.cache_info()` 返回累计 `compilations`、`cache_hits` 和当前 `specializations` 数。`engine.clear_cache()` 清除内存中的已编译图及最近诊断信息，保留累计计数、已计算 Tensor 和磁盘缓存。最近提交的图与报告可从 `engine.last_fuga_ir`、`last_ask_ir`、`last_report`、`last_artifact_dir` 查看。

模板冻结与结构键规则见[自动合图与引擎缓存](张量模型与追踪机制.md#jit-specialization)。CPU/CUDA 连续内存的批量导入导出及借用寿命见[buffer 接口](张量模型与追踪机制.md#buffer-io)。

ASK IR 文本摘要参与磁盘缓存键，文本变化会产生不同缓存项。校验和失效条件见 [ASK 双层持久缓存](../../../ask/docs/常驻运行时与多级缓存.md#two-tier-cache)。

<a id="bootstrap-and-config"></a>

## 5. 安装后的配置与开发引导（`tools/bootstrap.py` / `_config.py`）

FugaTensor 构建时链接 ASK SDK 的 `ASK::Compiler` 和 `ASK::Runtime`，wheel 携带对应共享库，步骤见[安装文档](../安装.md)。CPU 运行需 `cc` 链接目标文件，自有 CUDA 代码生成需 `llc` / `ptxas`，外部 kernel 的二进制由 worker 提供。

`Engine` 与 `_config.py` 按以下优先级读取配置：

1. **显式传参**：`ft.configure(device=..., ...)` 或 `ft.Engine(...)` 参数；
2. **环境变量**：`ASK_OPT`、`FUGATENSOR_CACHE_DIR`、`FUGATENSOR_LLC`、`FUGATENSOR_PTXAS`、`FUGATENSOR_TRITON_{PYTHON,WORKER}`（以及指定配置文件路径的 `FUGATENSOR_CONFIG`）；
3. **可选环境配置文件**：当前 Python 环境的 `sys.prefix/share/fugatensor/engine.json`，不存在时使用空配置；
4. **当前环境自动发现**：没有 Triton worker 配置时，若当前解释器安装了 Triton，则使用本环境和包内 worker。显式传入 `compilers={}` 可关闭编译器发现。

`cc` 通过 `ft.Engine(cc=...)` 或 `ft.configure(cc=...)` 设置。代码或环境变量指定 `ask_opt` 时，会隔离默认配置文件中的 SDK 和 worker 路径。

开发安装可用 `tools/bootstrap.py --ask-prefix SDK目录`，默认选择 pure，`--variant all` 同时安装 Triton。配置保留和重置规则见[开发引导](../安装.md#source-bootstrap)。

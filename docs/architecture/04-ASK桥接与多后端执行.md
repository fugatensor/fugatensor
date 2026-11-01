# 04. ASK 桥接与多后端执行

返回架构总览：[降级到 ASK、多后端执行与多级缓存](README.md#arch-ask-backends) · 相关细节：[张量模型与追踪机制](01-张量模型与追踪机制.md) · [Fuga IR 降级到 ASK](../ir/03-降级到ASK.md) · [ASK 常驻运行时与多级缓存](../../../ask/docs/05-常驻运行时与多级缓存.md)

当追踪或自动微分产出一张静态的 Fuga IR `Module` 后，`Engine` 负责打通最后一公里：裁剪掉未使用的物理输入端口、降级为 ASK IR、落盘可审计的跨层溯源产物，并驱动 CPU 动态库或 GPU 常驻会话完成执行。

<a id="engine-lifecycle"></a>

## 1. `Engine` 生命周期与无用物理输入裁剪（`engine.py`）

`Engine` 是连接 FugaTensor 与 [ASK 编译器及运行时](../../../ask/docs/02-前端接入与构建配置.md#python-sdk) 的会话中枢：

- **按需延迟初始化与上下文隔离**：用户既可以直接使用全局按设备惰性创建的默认引擎（首次访问 `device="cpu"` 或 `device="cuda"` / `"cuda:0"` 时建立，程序结束或调用 `ft.shutdown()` 时释放），也可以用 `with ft.Engine(device=..., cache_dir=...) as eng:` 创建完全隔离的临时引擎。所有 `Tensor` 都绑定创建它的 `Engine`，跨已关闭或不同 `Engine` 混用张量会被立即拦截。
- **编译期无用物理输入端口裁剪（`_prune_unused_inputs`）**：
  在反向传播或复合函数里，经常出现“函数签名接收 `(x, dy)`，但图内计算实际上只用到了 `dy`”的情况（例如线性缩放算子的反向 `dx = dy * 2`）。如果原封不动把未使用的 `x` 传给后端，不仅多传无用指针，还会卡住当前只接受**单实际输入端口**的 Triton 融合契约！
  因此 `Engine.compile_fuga` 在降级前先调用 [`ir/transforms.py` 的 `prune`](02-原生图分析与C++边界.md#transforms-reuse) 识别真正被消费的输入参数，自动生成只含活跃输入端口的精简物理子图交给 ASK 编译，并在外层用 `_ProjectedExecutable` 在调用时按索引抽取对应实参——既对上层调用方透明，又让单输入手写 Triton 反向顺畅融合。
- **三件套跨层可审计产物**：每次编译都会在产物目录同时写下 `fuga.ir`（Fuga IR 文本）、`frontend.mlir`（生成的 ASK MLIR）和 `fuga-to-ask.json`（参见 [降级溯源表](../ir/03-降级到ASK.md#external-and-sourcemap)），通过 `fn.last_artifact_dir`、`fn.last_fuga_ir`、`fn.last_ask_ir` 一眼就能看清每一层到底生成了什么。

<a id="host-backend"></a>

## 2. CPU 后端：目标文件生成与 `ctypes` 原生调用（`backends/host.py`）

CPU 执行路径（`HostBackend`）完全复用 [ASK 的 Host 共享循环降级](../../../ask/docs/03-图优化与自有代码生成.md#host-codegen)：

```text
Fuga Module -> lower_to_ask -> ask-opt --optimize --emit-object (graph.o + report.json)
            -> cc -shared (graph.so) -> ctypes 按 ABI 报告直接调用原生函数指针
```

两个关键工程细节保证了 CPU 后端既快又稳：

1. **进程内 SDK 与 CLI 伴生校验**：在首次生成本机 `.o` 前，直接调用 `self.compiler._check_companion()`（已在 `Compiler` 实例上缓存，不会在每次编译时重复派生 `--capabilities` 子进程），确保进程内 `ask._native` 与外部 `ask-opt` 来自同一次构建。
2. **严格按报告驱动 ABI**：后端绝不靠猜函数名或硬编码参数个数，而是精确解析 `report.json` 中导出的入口符号名、`inputs` 与 `outputs` 的 shape / dtype / 字节宽度，在每次热调用时只做轻量级缓冲区校验、为输出分配独立的 `HostStorage` 并直接调用驻留内存的 C 函数指针。

<a id="cuda-and-triton"></a>

## 3. 自有 CUDA 与用户手写 Triton 算子接入（`backends/cuda.py` / `triton.py`）

GPU `Engine` 在初始化时建立一个常驻的 [ASK `ExecutionSession`](../../../ask/docs/05-常驻运行时与多级缓存.md#execution-session)，并直接向物理设备查询真实的 Compute Capability（如 `sm_86`）。随后根据图中是否包含外部算子自动走两条路径之一：

1. **无外部算子的纯计算图 → ASK 自有 CUDA 降级**：
   不需要启用 Triton，也不依赖任何第三方 Python 包。`CUDABackend` 将 `ask.Module` 与真实 `arch` 传给 `Compiler.compile_cuda`，由 [ASK 自有 CUDA 后端](../../../ask/docs/03-图优化与自有代码生成.md#owned-cuda) 生成单核多端口 `cubin`，加载进常驻会话后通过 `PreparedCUDAKernel.submit_many(inputs, outputs)` 异步提交到同一 CUDA Stream。支持多输入、异构形状/类型的多输出、0-D 标量与空张量。
2. **含 `ft.triton.kernel` 的混合图 → `ASK → Triton → ASK` 原生融合**：
   - 用户通过 `ft.triton.kernel("kernels.py", entry=..., input=..., output=..., constants=..., grid=..., options=...)` 或 `ft.triton.from_source(...)` 注册自己写的 `@triton.jit` 代码。
   - **原样源码快照，零宿主污染**：注册时只按 UTF-8 读取源码文本并深拷贝元数据，**绝不在宿主进程里 `import` 用户文件或导入 Triton**，也绝不拿库里预置的模板拼凑假 kernel；后续修改磁盘原文件不会意外篡改已注册算子。
   - **延迟架构绑定**：`constants` 与 `grid` 回调只接收不含张量数据的 `TensorSpec(shape, dtype, device, numel)` 生成设备无关的 `fuga.external`；直到 `lower_to_ask` 准备投递给特定 GPU `Engine` 时，才填入该显卡真实的 `arch` 生成 `ask.implementation` 与 `ask.external`。
   - 最终由 [ASK Triton 适配器与通用边界内联器](../../../ask/docs/04-外部编译器适配与边界内联.md#triton-fusion) 把前后的 Fuga/ASK 逐元素运算融进同一个 `cubin`，交由同一个 `ExecutionSession` 异步执行。

<a id="cache-hierarchy"></a>

## 4. 五层协同缓存体系

从一次 Python 调用到底层 GPU 指令，两仓协同维护了五层职责分明的缓存：

| 层级 | 所在位置 | 缓存键值 | 命中后省去的工作 |
| --- | --- | --- | --- |
| **1. 单图 JIT 特化缓存** | FugaTensor `JITFunction` 内存 | `(Engine, 参数 shape/dtype, 静态标量值, fun 版本)` | 省去 Python 追踪、Fuga IR 降级及全部后端编译 |
| **2. 动态片段缓存** | FugaTensor `_segment_cache` 内存 | `(Engine, 片段 Fuga IR 文本, 活跃 fun 版本)` | 仅执行外层 Python `if`/`while` 控制逻辑，直接复用已编译片段 |
| **3. GPU LLVM IR 持久缓存** | ASK `cache_dir/ir` 磁盘 | 规范化请求 + 适配器与源码指纹 + `arch` | 跨进程或 `clear_cache()` 后省去自有降级或 Triton 子进程导出与边界内联 |
| **4. GPU `cubin` 持久缓存** | ASK `cache_dir/executables` 磁盘 | LLVM IR SHA256 + `llc`/`ptxas` 工具指纹 + `arch` | 省去耗时的 `llc` PTX 生成与 `ptxas` 寄存器分配汇编 |
| **5. 会话已加载模块** | ASK `ExecutionSession` 显存 | 已 `prepare(session)` 的 `PreparedCUDAKernel` | 省去同一 GPU 会话内的 `cuModuleLoadData` 重复装载 |

> 调用 `fn.clear_cache()` 只清空第 1、2 层 FugaTensor 内存特化（方便在更新全局变量或重新注册源码后重新追踪）；只要生成的静态计算图没变，第 3、4 层 ASK 磁盘缓存依然会瞬间命中！

<a id="bootstrap-and-config"></a>

## 5. 一键环境引导与配置优先级（`tools/bootstrap.py` / `_config.py`）

为了告别每次运行都要手工导出一长串 `ASK_OPT`、`PYTHONPATH`、`FUGATENSOR_LLC`、`FUGATENSOR_PTXAS` 的繁琐体验，`tools/bootstrap.py` 把两仓联编精简成了一条幂等指令：

```bash
# 首次完整构建（拉取锁定 LLVM、编译安装带 Triton 适配器的 ASK、编译安装 FugaTensor 并写入配置）
python3 tools/bootstrap.py

# 只修改了 FugaTensor 的 Python/C++ 代码时，跳过 ASK 重编译快速更新
python3 tools/bootstrap.py --skip-ask

# 仅需自有 CPU/CUDA 后端、不安装 Triton 依赖时
python3 tools/bootstrap.py --compilers none
```

引导脚本会自动检测 `.venv` 中已满足锁定版本的构建依赖（跳过多余联网检查），直接通过 `pip install --no-deps --no-build-isolation` 将 ASK 和带 `_native` C++ 扩展的 FugaTensor 装入 `.venv`，并把探测好的 `llc`、`ptxas` 及各适配器 `worker` 路径原子写入 `.venv/share/fugatensor/engine.json`。

运行时 `_config.py` 按以下清晰优先级解析配置：

1. **显式代码传参**：`ft.configure(device=..., ...)` 或 `ft.Engine(...)` 参数；
2. **环境变量覆盖**：`ASK_OPT`、`FUGATENSOR_CACHE_DIR`、`FUGATENSOR_LLC`、`FUGATENSOR_PTXAS`、`FUGATENSOR_CC`、`FUGATENSOR_TRITON_{PYTHON,WORKER}`（以及指定自定义配置文件的 `FUGATENSOR_CONFIG`）；
3. **保存的环境配置**：自动读取当前虚拟环境的 `.venv/share/fugatensor/engine.json`。

注意一个防混用保护细节：一旦用户通过代码或环境变量显式指定了另一套 `ask_opt` 路径，`_config.py` 就不会再去混读默认 `engine.json` 里保存的旧 SDK / worker 路径，杜绝跨构建混配。

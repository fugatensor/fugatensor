返回文档目录：[目录](README.md) · 相关细节：[测试矩阵](测试.md) · [执行与缓存](architecture/ASK桥接与多后端执行.md#native-plan-replay)

# 与 PyTorch 对比执行速度

[`tools/benchmark_torch.py`](../tools/benchmark_torch.py) 用相同输入和数学表达式比较 Fuga、PyTorch eager 与 `torch.compile`，将结果写入 CSV。在 Linux/WSL 的 Python 环境中安装 Fuga 和 PyTorch 后运行。

ASK Host 与 Fuga 构图优化后的三轮记录见[性能剖析与优化](性能剖析与优化.md#comparison)，包含分层归因、CPU 矩阵尺寸扩展、冷启动与内存成本。

后续结果见[LLVM 协作归约与 C++ 图收集对照](LLVM归约与原生图收集优化.md#comparison)，最新构图优化见[原生节点池对照](原生节点池与增量指纹优化.md#comparison)。

<a id="quick-start"></a>

## 先跑一个小负载

在仓库根目录执行：

```bash
python tools/benchmark_torch.py \
  --devices cpu --workloads elementwise --sizes small \
  --rounds 1 --samples 21 --output build/benchmarks/torch-small
```

结果目录保留每项配置及三个后端的时间。`comparison.csv` 中的 `fuga_over_torch` 是 **Fuga 耗时 / Torch 耗时**：大于 1 表示 Fuga 较慢，小于 1 表示 Fuga 较快。`difference_pct` 为相对 Torch 的耗时变化百分比。

<a id="matrix"></a>

## 扩大矩阵

默认运行 CPU、CUDA 和四类负载，交叉两档尺寸、FP32、连续布局，重复三轮：

```bash
python tools/benchmark_torch.py --output build/benchmarks/torch-full
```

| 负载 | 计算 | small / large |
| --- | --- | --- |
| `elementwise` | `(x + 0.5) * (x - 0.25) + x` | 32 × 64 / 512 × 256 |
| `reduction_chain` | 每层广播行最大值，再加原输入 | 同上，默认 8 层 |
| `fork_join` | 各分支独立归约、广播、加输入，再汇合 | 同上，默认 4 分支 |
| `matmul` | `a @ b` | 16 / 128 阶方阵 |

用轴参数生成更多组合：

```bash
python tools/benchmark_torch.py \
  --devices cuda --dtypes fp32 fp16 --layouts row_major transpose \
  --depths 3 8 --branches 2 4 --rounds 3 --samples 51 \
  --output build/benchmarks/torch-cuda-matrix
```

深度只扩展归约链，分支数只扩展分支负载。`transpose` 在被测函数内对连续存储的输入做转置；Fuga 和 Torch 各自按其索引、视图和融合语义执行。默认输入使用小整数及二进分数；矩阵乘法使用行列变化的秩一数据，独立解析参考检查输出。

扩大矩阵乘法尺寸并使用需要 TF32 舍入的输入：

```bash
python tools/benchmark_torch.py --backends fuga torch_eager \
  --workloads matmul --matmul-sizes 16 128 512 1024 \
  --matmul-data general --matmul-precision tf32 --threads 1 --cpu-core 2 \
  --output build/benchmarks/matmul
```

`--matmul-precision tf32` 是默认 CUDA fp32 策略，两边都允许 TF32；`ieee` 设置 `NVIDIA_TF32_OVERRIDE=0` 并将 Torch 设为 IEEE，提供相同精度的对照。CPU 始终采用 fp32 库运算。`--matmul-data general` 将行列系数先舍入到输入 dtype，解析参考按这些实际输入计算，并按所选精度检查误差。

`--backends fuga torch_eager` 可单独比较 eager。若两个框架安装在不同环境，指定 `--fuga-python` 和 `--torch-python`；元数据记录每个实际解释器和框架路径。包含 Fuga 的 CPU 对照使用 `--threads 1`，匹配 ASK 自有循环与单线程 OpenBLAS；Torch 独立测量可指定其它线程数。`--cpu-core 2` 同时固定 CPU 运算与 CUDA 提交的宿主核心。

<a id="measurement"></a>

## 计时口径

每个配置、后端、轮次使用独立进程和临时编译缓存，轮间轮换后端顺序。输入构造和上传位于调用计时外。默认预热 5 次，记录 31 次热调用；CUDA 每次调用后同步完成。

- `setup_us`：导入框架、初始化与输入创建。
- `first_complete_us`：第一次函数调用到完成，包含该表达式的编译和首次执行。
- `call_p50_us`：函数计算并提交后的主机返回时间中位数。Fuga 包含构图和 `evaluate`。
- `complete_p50_us` / `complete_p90_us`：完整调用到同步完成的中位数 / 90 分位数，是速度比较使用的口径。

每次 Fuga 调用都生成新结果，并检查实际执行次数和一次编译的缓存记录。首末结果检查 shape、dtype 及独立数值参考。PyTorch 使用 inference mode，矩阵精度由 `--matmul-precision` 决定；`torch.compile` 要求完整图，CPU 使用 `default` 模式，CUDA 使用支持 CUDA Graph 的 [`reduce-overhead`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) 模式。失败项写入 `results.csv`，脚本返回非零状态。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `samples.csv` | 首次调用及每个热样本的原始时间 |
| `results.csv` | 每进程的延迟、数值检查、执行计数和内存记录 |
| `comparison.csv` | 同一配置、双方成功轮次交集的耗时比值 |
| `metadata.json` | 参数、Python/框架版本、设备、实际线程数及编译模式 |
| `logs/` | 各进程的编译与错误日志 |

`device_reserved_bytes` 记录 ASK 自有分配池或 Torch caching allocator 的保留字节；Torch 另有 allocated 字节。峰值包含首次执行及缓存建立。CPU `host_workspace_bytes` 来自 ASK 产物 ABI，`host_peak_rss_bytes` 记录完整子进程高水位，包含框架、输入和参考数据。临时编译缓存随每个子进程的测量结束清理，输出目录供后续分析保留。

<a id="recorded-run"></a>

## 本次记录

2026-11-04，在 Xeon E5-2696 v4、RTX 3080、WSL2 上运行。两个框架共用 Python 3.12.15，Fuga 0.1.0 链接 ASK `d767b9a` 的构建，PyTorch 为 `2.14.0+cu130`。CPU 单线程并固定在核心 2；CUDA 使用默认设备。

FP32 连续布局运行一轮，每项预热 5 次，采集 21 次热调用，共 48 项。以下为 large 尺寸的完整调用中位数，单位 μs：

| 设备 | 负载 | Fuga | Torch eager | Torch compile |
| --- | --- | ---: | ---: | ---: |
| CPU | 逐元素，512 × 256 | 361.62 | 245.66 | 137.54 |
| CPU | 归约链，8 层 | 3426.39 | 657.79 | 428.94 |
| CPU | 分支汇合，4 分支 | 2183.34 | 544.88 | 285.43 |
| CPU | 矩阵乘法，128 × 128 | 2276.34 | 71.27 | 180.74 |
| CUDA | 逐元素，512 × 256 | 268.69 | 154.27 | 517.15 |
| CUDA | 归约链，8 层 | 1244.93 | 591.02 | 687.79 |
| CUDA | 分支汇合，4 分支 | 890.10 | 449.96 | 731.32 |
| CUDA | 矩阵乘法，128 × 128 | 172.59 | 78.14 | 372.06 |

本轮 large 负载中，Fuga 均慢于 Torch eager；CUDA 逐元素和矩阵乘法快于本次 Torch compile 测量。数字描述这组公开调用的端到端延迟，后续优化使用多轮结果跟踪变化。

完整记录：[速度比值](assets/torch-benchmark/fp32/comparison.csv) · [每项结果与内存](assets/torch-benchmark/fp32/results.csv) · [原始样本](assets/torch-benchmark/fp32/samples.csv) · [运行参数和版本](assets/torch-benchmark/fp32/metadata.json)。

另运行 FP16 × 转置布局 × CPU/CUDA × 四类 small 负载，共 24 项；这组用于验证矩阵扩展，每项采集 3 次热调用。两组共 72 项均通过 shape、dtype 和数值检查，最大绝对误差均为 0。Fuga 每项均只编译一次，执行次数等于首次、预热和采样次数之和。

验证记录：[FP16 结果](assets/torch-benchmark/fp16-validation/results.csv) · [构建与校验记录](assets/torch-benchmark/provenance.json)。

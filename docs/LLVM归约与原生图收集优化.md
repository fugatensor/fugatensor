返回文档目录：[目录](README.md) · 相关细节：[前一轮性能剖析](性能剖析与优化.md) · [原生图收集](architecture/原生图分析与C++边界.md#lazy-collection) · [ASK CUDA 生成](../../ask/docs/图优化与自有代码生成.md#owned-cuda)

# LLVM 归约与原生图收集优化

2026-11-04，继续使用 ASK 自有 LLVM 生成路径，并将 Fuga 的待计算图收集合并到 C++。本轮分别处理 GPU 归约计算和缓存命中前的前端分析成本。

代码提交：ASK `266e5e7`，Fuga `3a6b3c6`。

后续的持久节点与追踪原生化见[原生节点池与增量指纹优化](原生节点池与增量指纹优化.md)。

三轮对照中，large CUDA 归约链完整调用 **812.45 → 579.39 μs，降低 28.7%**；分支汇合 **609.78 → 405.31 μs，降低 33.5%**。CPU 四类负载降低约 4.8%–9.9%。

<a id="changes"></a>

## 已落地的修改

ASK 在编译期选择适用的 `amax` 输出组，每个 warp 协作计算一个输出。各 lane 按 32 步进读取归约域，通过五轮 shuffle-XOR 合并结果，由 lane 0 写出。尾部读取在 shuffle 前汇合；同一 kernel 中不同形状的输出组各自采用线程或 warp 步长。

协作路径要求 sm70+、block 大小为 32 的倍数，且输出组可达的归约全部为非嵌套 `amax`、归约域至少 32、归约结果元素数与输出组一致。其它图继续使用通用生成路径；浮点 `sum` 保持既有逐项舍入。归约前后逐元素表达式仍在同一生成区域内融合。新增判断发生在代码生成阶段，执行循环只包含计算所需的迭代、尾部访问和写出控制。

Fuga 新增 `cpp/lazy.cpp`，一次原生调用完成输出需求固定点、活跃依赖拓扑遍历、定义依赖聚合与失效校验、精确结构键生成。对应四段 Python 算法已删除。每次调用对 Tensor、表达式、模板和投影各解析一次元信息，后续遍历使用 C++ 局部状态。

Tensor、原始表达式来源与自动微分仍由 Python 管理，原生快照在调用结束时释放。普通 Python 函数继续每次执行；投影准备与必要的定义版本读取沿用 Python 语义。ASK 核心依赖保持独立。

<a id="method"></a>

## 测量口径

基线是前一轮已优化的 ASK `a8ba306`、Fuga `2d060f5`。两个框架共用 Python 3.12.15，硬件为 Xeon E5-2696 v4、RTX 3080，WSL2；Torch 为 2.14.0+cu130，CPU 单线程并固定核心 2。

正式对照交叉 CPU/CUDA、四类负载、small/large、Fuga/Torch eager，重复三轮；每项独立进程、预热 10 次、记录 41 个热样本，CUDA 每次同步。每次 Fuga 调用重新执行 Python 并构造新结果。输入创建和上传在计时外。

分层诊断独立运行，每阶段采样 31 次。先预热 5 次，无插桩测构图、求值和同步，再沿用已预热状态，用临时包装器测缓存与分析；最后用预分配缓冲区直接调用原生执行入口，该阶段另预热 5 次。诊断包装器的成本不计入正式速度对照。CUDA 原生完成时间包含提交、等待和回收，CPU 原生时间包含 ctypes 桥接。

<a id="comparison"></a>

## 整体调用与 Torch eager 对照

下表为三轮各自热调用中位数再取中位数，单位 μs。逐元素、归约链和分支输入为 512 × 256，matmul 为 128 阶方阵，均为 FP32 连续布局。

| 设备 | 负载 | Fuga 基线 | Fuga 本轮 | 降幅 | 本轮 Torch eager | Fuga / Torch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 逐元素 | 202.61 | 182.65 | 9.9% | 243.97 | 0.75× |
| CPU | 归约链，8 层 | 929.02 | 843.87 | 9.2% | 689.65 | 1.22× |
| CPU | 分支汇合，4 分支 | 670.57 | 617.48 | 7.9% | 566.05 | 1.09× |
| CPU | matmul | 187.12 | 178.12 | 4.8% | 71.33 | 2.50× |
| CUDA | 逐元素 | 190.87 | 170.63 | 10.6% | 147.00 | 1.16× |
| CUDA | 归约链，8 层 | 812.45 | 579.39 | 28.7% | 611.96 | 0.95× |
| CUDA | 分支汇合，4 分支 | 609.78 | 405.31 | 33.5% | 397.87 | 1.02× |
| CUDA | matmul | 132.32 | 126.95 | 4.1% | 76.78 | 1.65× |

CUDA 归约链本轮略快于 eager，分支汇合接近持平。Torch 归约链在前后两组中的中位数为 575.28、611.96 μs，存在会话波动；完整记录保留双方各轮与范围。

全部 16 项配置的 Fuga 热调用中位数均下降。小负载仍有较大固定成本：CPU 16 阶 matmul 从 63.95 降至 53.91 μs，Torch 为 3.36 μs；CPU 小归约链从 580.90 降至 498.40 μs，Torch 为 74.37 μs。后续原生化应继续针对构图和调用包装。

前后各 96 项均通过数值检查，误差为 0；Fuga 每项编译 1 次、实际执行 52 次。每版保留 4032 条首次与热调用原始记录。[三轮汇总](assets/native-llvm-optimization/comparison-summary.csv) · [基线结果](assets/native-llvm-optimization/baseline/results.csv) · [本轮结果](assets/native-llvm-optimization/candidate/results.csv) · [本轮原始样本](assets/native-llvm-optimization/candidate/samples.csv)。

<a id="layers"></a>

## 分层归因

旧版每个诊断样本的依赖收集、定义依赖处理、结构键生成三项时间相加，再对样本取中位数；新版对应一次 `collect_pending`。large 的结果如下，单位 μs：

| 负载 | CPU：Python 三阶段 → C++ 合并 | CUDA：Python 三阶段 → C++ 合并 |
| --- | ---: | ---: |
| 逐元素 | 38.89 → 20.23 | 40.45 → 23.81 |
| 归约链 | 160.05 → 83.60 | 150.59 → 78.81 |
| 分支汇合 | 110.71 → 62.32 | 107.04 → 57.10 |
| matmul | 21.16 → 11.49 | 22.68 → 13.49 |

全部 16 项配置的分析时间降低 39.5%–49.0%。字典查找本身增加约 0.3–1.9 μs，收益来自原生图分析。Python 原语构图代码本轮未修改；其测量变化按运行波动观察。

预分配后直接调用 ASK 原生计划，large CUDA 归约链的提交时间为 **18.82 → 16.18 μs**，提交到同步完成为 **224.30 → 62.33 μs**；分支汇合对应 **17.44 → 16.98 μs**、**194.56 → 57.43 μs**。节点数分别仍为 9、5，依赖边及内存槽位相同，改善主要落在归约计算路径。

CPU 算术生成本轮没有修改。单轮分层中的 matmul 原生时间存在明显变化，该调用绕过图收集器，因此不归因于 C++ 迁移；其完整调用以三轮对照的 187.12 → 178.12 μs 为准。

各版分层数据包含 16 项、1488 条样本。首次结果与最后的直接原生结果均经独立参考检查；每项 Fuga 执行计数为 68，CUDA 加上预分配阶段共提交 104 次。[分析时间汇总](assets/native-llvm-optimization/analysis-summary.csv) · [基线分层](assets/native-llvm-optimization/layers-baseline/summary.csv) · [本轮分层](assets/native-llvm-optimization/layers-candidate/summary.csv) · [原始分层样本](assets/native-llvm-optimization/layers-candidate/raw.csv)。

<a id="memory"></a>

## 内存与首次调用

CPU 临时区和 ASK 跟踪的显存峰值均与基线相同：

| large 负载 | CPU 工作区，字节 | CUDA ASK 跟踪峰值，字节 |
| --- | ---: | ---: |
| 逐元素 | 0 | 1,048,576 |
| 归约链 | 4096 | 1,052,672 |
| 分支汇合 | 8192 | 4,196,352 |
| matmul | 0 | 196,608 |

本轮改善的是执行时间。进程峰值 RSS 观测下降约 10–12 MiB，但其中包含框架导入、编译器与共享库映射，不能据此推导图收集器的独立内存收益。

CPU large 首次完整调用仍约 236–316 ms。CUDA 归约链由 1463 → 1412 ms，分支汇合由 1287 → 1191 ms；首次计时包含整个编译与首次执行过程，详细记录与热调用分开保存在 CSV。

<a id="validation"></a>

## 测试矩阵与构建

ASK 新增 764 格静态调度矩阵，检查 dtype、形状、归约宽度、布局、逐元素组合、多输出及配置边界。i32、i64、f32、f16、bf16 的尾部归约均经 llc 与 ptxas 生成 cubin，i64 通过两组 32 位 shuffle 保留完整位模式。

ASK CTest 6/6、独立 SDK Compiler 2/2 与 Runtime 2/2 均通过；SDK 消费侧关闭 LLVM/MLIR 开发包查找。[ASK 测试记录](assets/native-llvm-optimization/validation/ask-ctest.txt) · [SDK Compiler](assets/native-llvm-optimization/validation/sdk-compiler.txt) · [SDK Runtime](assets/native-llvm-optimization/validation/sdk-runtime.txt) · [PTX 编译审计](assets/native-llvm-optimization/ptx-audit.json)。

Fuga 新增 112 格延迟图矩阵：dtype × shape × 输入共享 × detach × 输出子集。每项更换数据再次求值，检查数值、缓存、实际执行次数和结果保留；独立用例覆盖输出顺序、同一值多次返回、直接返回输入、常量输入载体、梯度来源和深链。

CUDA 数值矩阵交叉五种类型、十种形状、两种布局，每组生成全部七种非空归约轴组合和两种前后组合，共 1160 个有效数值输出、120 项空归约域拒绝；另 20 组嵌套和扩展图检查 60 个输出。独立参考逐项验证 NaN、正负零、无穷、整数极值和 i64 高位。

完整 Fuga 回归开启 CUDA 与 Triton，199 项测试全部通过，耗时 1688.236 秒，跳过 0 项。覆盖上述矩阵、自动微分、定义失效、CPU matmul 与外部算子集成。[完整测试记录](assets/native-llvm-optimization/validation/fuga-full.txt)。

ASK SDK 已重装；Fuga 的 Python 3.13 测试环境与 Python 3.12 对照环境均从新 SDK 重建。

两套环境与 SDK 的 Compiler 共享库 SHA-256 一致；依赖检查通过。[构建产物摘要](assets/native-llvm-optimization/build-hashes.json)。

<a id="next"></a>

## 下一阶段的边界

图收集迁入 C++ 后，large CUDA 归约链的 Python 构图仍约 358 μs，占有明显份额。下一阶段优先评估原语参数绑定和待计算节点创建的原生实现，继续保留普通 Python 执行、表达式来源和定义失效语义。

LLVM 路径可继续按归约域与输出规模选择协作方式。当前单 warp 已覆盖本次主要负载，更大归约域的 block 协作应单独测量寄存器、共享内存与同步成本；浮点 sum 的数值规则保持独立评估。

<a id="reproduce"></a>

## 复现

```bash
build/torch-benchmark-env/bin/python tools/benchmark_torch.py \
  --backends fuga torch_eager --rounds 3 --samples 41 --warmup 10 \
  --cpu-core 2 --output build/benchmarks/native-llvm-current
```

实现与扩展测试方法见 [ASK CUDA 生成](../../ask/docs/图优化与自有代码生成.md#owned-cuda)、[Fuga 原生边界](architecture/原生图分析与C++边界.md#lazy-collection)与[测试矩阵](测试.md#lazy-schedule-matrix)。

[版本与验证记录](assets/native-llvm-optimization/provenance.json)保存代码提交、采样口径和测试计数。临时分层包装器、基线包快照及 PTX/cubin 审计产物已清理；长期对照入口保留在 `tools/benchmark_torch.py`，测量结果与验证日志归档在本报告的资源目录。

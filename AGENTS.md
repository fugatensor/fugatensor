# FugaTensor 仓库工程规范

## 工作约定

1. 完成一组代码或文档修改并通过相关验证后，创建本地 Git 提交（使用仓库已配置的 Git 身份），便于回退。未完成的检查点需在提交说明中标明状态。
2. 第三方 SDK、构建产物（`build/`、`dist/`、`*.egg-info/`）、Python 虚拟环境（`.venv/`）和缓存（`.cache/`、`__pycache__/`）不纳入 Git；保留依赖锁记录及引导脚本（`tools/bootstrap.py`）。
3. 文档使用自然直接的中文说明设计动机、模块配合与能力边界，避免 AI 式套话、过渡废话，不把未实现的规划写成现有能力。
4. 修改接口、IR 规则、构建参数或测试结构时，同步更新 `README.md`、`docs/API/`、`docs/architecture/`、`docs/ir/` 及相关示例，并保持双向链接锚点有效。

---

## 架构与编码规范

- **Fuga IR 独立性与算术执行边界**：
  - Tensor 自动组图生成独立于 ASK 的 Fuga IR（`src/fugatensor/ir/`），框架层的控制流化简、C++ 活跃性分析与 JVP/VJP 自动微分均作用于 Fuga IR，最后通过 `lowering.py` 降级为高层 ASK IR。
  - Fuga IR 不包装或依赖后端的 Value、Type 或 MLIR 字符串；`fugatensor._ask` 仅作为图降级与产物协议的内部桥接层。
  - 实际张量算术统一编译后交由 ASK 执行，不在 Python 中实现张量数值计算回退。
- **自动组图、执行边界与 `ft.fun` 契约**：
  - 常规 API 直接使用 `ft.tensor(...)`，默认 CPU 与 `ft.fp32`，不要求显式 `Engine` 或上下文。`ft.set_default_device` 只改变当前上下文后续创建的设备，默认引擎按设备复用并在正常退出时清理；独立 Engine 属于高级资源管理接口。
  - 用户精度参数和示例使用 `ft.bool`、`ft.int32`、`ft.int64`、`ft.fp32`、`ft.fp16`、`ft.bf16` 类型对象；Tensor 与追踪元信息返回相同单例，IR、产物和权重清单保留原有字符串编码。
  - 用户无需 JIT 装饰器；普通 Python 函数每次正常执行，Tensor 运算返回待计算图节点。读取数据、0-D 标量条件判断、反向传播与 `ft.evaluate(...)` 触发所需依赖图的 JIT 编译和执行。
  - 跨算子与普通函数调用保留图连接和融合机会；执行缓存属于 `Engine`，按图结构和自定义函数定义身份复用，不能跳过用户 Python 逻辑。
  - `ft.fun` 的普通 `forward` 与可选 `backward` 自动进行隔离追踪，保留 `CallSpec`、保存值与定义变更校验；方法内部不支持数据依赖控制流。Triton 源码中的 `@triton.jit` 保留。
  - 梯度历史在组图时记录，`no_grad` 与 `detach` 的边界不随实际执行时间改变；Python 条件每次读取真实标量值，反向沿实际路径传播。
- **多 Kernel 执行**：
  - FugaTensor 向 ASK 提交完整静态图，自有计算区域保留融合，外部算子统一作为独立 kernel 调度；前端不限制整图只有一个外部节点。
  - Triton 通过 `inputs=` / `outputs=` 注册多输入与多输出，`scalars=` 绑定按值启动参数，输出形状/类型和配置回调只使用 TensorSpec 元数据；复杂外部前向不自动获得梯度规则。
  - 复杂算子（如 FlashAttention）的反向由 `ft.fun` 的显式 `backward` 组合外部 kernel；单次外部调用可返回多个结果，未被图消费的物理输出仍须分配。
- **自动微分与原语闭合**：
  - 公开自动微分接口采用 `requires_grad`、`Tensor.backward(...)` 与叶子 `.grad`，不采用 `ft.grad(f)` 形式；非标量输出调用 `backward` 须显式传入梯度种子。
  - 新增基础原语时，在 `ir/primitives.py` 中提供可执行的 `abstract_eval`、`jvp` 和 `vjp` 规则，微分产物保持为同一套闭合的 Fuga IR。
  - 对缺少可组合原语规则的不透明外部算子（如裸 `fuga.external` 或请求高阶导数的手写 Triton 算子），在执行反向编译或写入 `.grad` 之前的预检阶段报错拒绝。
- **权重格式**：
  - `weights.py` 只序列化由嵌套 dict/list/tuple 与 Tensor 组成的 PyTree，不记录模型类、算子或设备；数组使用 dtype/shape/布局加数据的内容寻址身份。
  - 新增网络结构、参数命名或基础原子算子不改变 v1 容器；只有存储语义变化才提升 manifest `schema_version`。
- **外部编译器与 C++ 扩展**：
  - 基础算子在 CPU 与 GPU 上均走 ASK 自有降级，不依赖 Triton。`ft.triton.kernel` / `from_source` 保存用户源码，不在宿主进程导入 Triton；`from_jit` 可快照调用方已加载的 JIT 函数和依赖。worker 执行可信 Python 源码并追踪加载的依赖文件，不用预置模板替代用户 kernel。
  - Triton 适配位于本包 `_triton_worker.py`，通过 ASK 通用 worker 协议返回 `cubin` 与启动元数据；不要求 ASK 构建注册、不导出 LLVM 或插入融合标记。固定版本 ABI 与依赖记录在 `triton.dependencies.lock.json`。
  - C++ 扩展 `fugatensor._native`（`cpp/`）负责 SSA 图分析和 ASK 原生绑定，实际编译、整段计划提交、缓冲区、事件与 CUDA Graph 管理由 ASK C++ 实现。`ir/transforms.py` 统一封装 `analyze_function`、`prune` 与 `project_outputs`；Fuga 不复制运行时调度器。

---

## 构建与测试

- **构建指令**：
  - 两个项目分别构建：先安装 ASK 0.1.0 的 C++ SDK，再用它构建 FugaTensor；ASK 不发布 Python 包，FugaTensor wheel 自带配套 ASK 共享库。
  - FugaTensor 构建：`CMAKE_PREFIX_PATH=/path/to/ask-sdk python3 -m pip wheel --no-deps . --wheel-dir dist`。原生扩展使用 `ASK::Compiler` / `ASK::Runtime` 的公开接口，不读取 ASK 源码，也不需要 LLVM SDK。
  - 开发安装：`CMAKE_PREFIX_PATH=/path/to/ask-sdk python3 -m pip install --no-deps -e .`，随后执行 `python3 -m pip check`；引导脚本通过 `--ask-prefix /path/to/ask-sdk` 接收 SDK。安装详情见 [`docs/安装.md`](docs/安装.md)。
  - 包要求 Python 3.10+，`tools/bootstrap.py` 开发引导需要 Python 3.11+；CPU JIT 运行还需 `cc` 链接器，CUDA 另需显式配置 `llc` / `ptxas` 与驱动。
- **测试运行**：
  - 前端测试位于 `tests/test_*.py`，共用 `tests/_helpers.py` 的 `EngineTestCase`：
    - CPU + IR + 原生图分析：`.venv/bin/python -m unittest -v`
    - 启用自有 CUDA 硬件测试：`FUGATENSOR_TEST_CUDA=1 .venv/bin/python -m unittest -v`
    - 启用自有 CUDA + 用户 Triton 调度测试：`FUGATENSOR_TEST_CUDA=1 FUGATENSOR_TEST_TRITON=1 .venv/bin/python -m unittest -v`
  - 后端桥接与产物协议测试位于 `tests/backend/`：`ASK_SOURCE_ROOT=/path/to/ask ASK_OPT=/path/to/ask-sdk/bin/ask-opt .venv/bin/python -m unittest discover -s tests/backend -p 'test_*.py' -v`。
  - 后端物理 GPU 测试独立运行：`.venv/bin/python -m unittest discover -s tests/backend/gpu -p 'test_gpu_*.py' -v`，同时配置 `ASK_SOURCE_ROOT`、`ASK_OPT`、`ASK_LLVM_TOOLS_DIR` 与 `ASK_PTXAS`；设置 `ASK_REQUIRE_CUDA_DEVICE=1` 可要求驱动和工具缺失时失败。
  - `ASK_SOURCE_ROOT` 仅供集成测试读取 ASK 的 IR fixture、依赖锁与工具路径，不是 FugaTensor 的构建或运行依赖。依赖缺失导致 skipped 不能视作相关功能通过验证。
  - 始终使用已安装 `_native` 扩展的 `.venv/bin/python` 运行测试，不使用脱离原生扩展的 `PYTHONPATH=$PWD/src`。

---

## 文档与双向链接

- 文档分设 `docs/API/`（按类型组织的接口用法）、`docs/architecture/`（系统架构与运行时）和 `docs/ir/`（Fuga IR 对象模型、变换与降级），由 `docs/README.md` 索引，并与同级 `../ask/docs/` 跨仓互链。跨仓链接仅供源码浏览，并列目录不是构建或运行依赖：
  - API 分类文档说明签名、参数、返回值、示例与支持边界，首行链接分类目录和相关接口；实现原理链接架构或 IR 专题，避免重复；
  - 总览文档（`docs/architecture/README.md`、`docs/ir/README.md`）每节设置 `<a id="..."></a>` 锚点，节末以 `工程细节：...` 指向专题文档；
  - 专题文档首行以 `返回架构总览 / 返回 IR 总览：... · 相关细节：...` 链回总览与关联文档，内部小节提供 `<a id="..."></a>` 锚点。
- 修改文档标题、文件名或锚点后，校验 `fugatensor` 与同级 `ask` 仓库中的 Markdown 相对链接和 `#anchor` 引用。

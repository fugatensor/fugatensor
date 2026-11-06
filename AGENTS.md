# FugaTensor 仓库工程规范

## 工作约定

1. 完成一组代码或文档修改并通过相关验证后，创建本地 Git 提交（使用仓库已配置的 Git 身份），便于回退。未完成的检查点需在提交说明中标明状态。
2. 第三方 SDK、构建产物（`build/`、`dist/`、`*.egg-info/`）、Python 虚拟环境（`.venv/`）和缓存（`.cache/`、`__pycache__/`）不纳入 Git；保留依赖锁记录。
3. 文档直接说明用途、做法和必要条件，删去重复铺垫、历史对比和无关的“无需／不需要”说明；限制放在相关接口处，规划与当前行为分开写。开发与上下文检索时，Agent 只读取英文文档（`docs/en/` 与根目录 `README.md`）即可；修改文档时必须同步更新中英双语版本。
4. 修改接口、IR 规则、构建参数或测试结构时，同步更新 `README.md`、`README.zh.md`、`docs/zh/` 与 `docs/en/` 中的示例，并保持双向链接锚点有效。
5. 示例只写在文档中，提供合法、简洁的 API 用法，可用 `print` 展示结果；不另外维护演示脚本。测试位于 `tests/`，必须使用断言验证行为，不能依赖示例脚本或以打印结果代替验证。可复用算子实现属于 `src/fugatensor/`，测试参考与输入生成属于测试支持模块。第三方 kernel 源码中的算法约束保留。

---

## 架构与编码规范

- **Fuga IR 独立性与算术执行边界**：
  - Tensor 自动组图生成独立于 ASK 的 Fuga IR（`src/fugatensor/ir/`），框架层的控制流化简、C++ 活跃性分析与 JVP/VJP 自动微分均作用于 Fuga IR，最后通过 `lowering.py` 降级为高层 ASK IR。
  - Fuga IR 不包装或依赖后端的 Value、Type 或 MLIR 字符串；`fugatensor._ask` 仅作为图降级与产物协议的内部桥接层。
  - 实际张量算术统一编译后交由 ASK 执行，不在 Python 中实现张量数值计算回退。
  - CPU Tensor 支持 bool、i32、i64、f32、f16；bf16 Tensor 使用 CUDA。矩阵乘法在 CUDA 走 cuBLASLt，静态描述与工作区由 ASK Runtime 在准备阶段管理。
- **自动组图、执行边界与 `ft.fun` 契约**：
  - 常规 API 直接使用 `ft.tensor(...)`，默认 CPU 与 `ft.fp32`，不要求显式 `Engine` 或上下文。`ft.set_default_device` 只改变当前上下文后续创建的设备，默认引擎按设备复用并在正常退出时清理；独立 Engine 属于高级资源管理接口。
  - `tensor(data)` 从规则嵌套数据推断形状；`zeros`、`ones`、`empty` 按尺寸创建，`full(size, value)` 按值填充。改变形状使用延迟原语 `reshape`，保持元素次序，JVP/VJP 保留对应形状；不承诺共享存储，也不提供 `view`。
  - 设备参数接受 `ft.device` 对象或字符串，Tensor、TraceTensor、TensorSpec 暴露实际设备对象；默认 CUDA 解析为 0。不同设备张量算术报 `RuntimeError`，同设备但属于不同独立 Engine 报 `ValueError`。
  - 用户精度参数和示例使用 `ft.bool`、`ft.int32`、`ft.int64`、`ft.fp32`、`ft.fp16`、`ft.bf16` 类型对象；Tensor 与追踪元信息返回相同单例，IR、产物和权重清单保留原有字符串编码。
  - 用户无需 JIT 装饰器；普通 Python 函数每次正常执行，Tensor 运算返回待计算图节点。读取数据、0-D 标量条件判断与 `ft.evaluate(...)` 触发所需依赖图的 JIT 编译和执行。
  - 跨算子与普通函数调用保留图连接和融合机会；执行缓存属于 `Engine`，按图结构和自定义函数定义身份复用，不能跳过用户 Python 逻辑。
  - `ft.fun` 使用 `forward(self, ...)` 和可选纯规则 `vjp(self, primals, outputs, cotangents)`，三个 VJP 参数均为 tuple：primals 包含全部 Tensor 输入，outputs 包含全部公开输出，cotangents 与输出对齐（未使用浮点输出为零 Tensor，整数/布尔输出为 None）；返回浮点 Tensor 输入的梯度 tuple。普通前向自动 VJP/JVP，显式 VJP 的活跃 JVP 当前拒绝。隔离追踪保留 `CallSpec` 和定义变更校验，方法内部不支持数据依赖控制流。Triton 源码中的 `@triton.jit` 保留。
  - Tensor 求值后保留表达式来源；`detach` 与 `stop_gradient` 显式断开微分路径，同时保留数值依赖。`grad` 接收运行时 Tensor，调用位于 `ft.fun` 隔离追踪之外。停止梯度支持 Tensor 和 TraceTensor，可用于 `forward` 与 `vjp`；追踪生成 `fuga.stop_gradient`，JVP/VJP 在此断边，降级到 ASK 时映射为输入值。Python 条件每次读取真实标量值，求导沿实际形成的路径传播。
- **多 Kernel 执行**：
  - FugaTensor 向 ASK 提交完整静态图，自有计算区域保留融合，外部算子统一作为独立 kernel 调度；前端不限制整图只有一个外部节点。
  - Triton 通过 `inputs=` / `outputs=` 注册多输入与多输出，`scalars=` 绑定按值启动参数，输出形状/类型和配置回调只使用 TensorSpec 元数据；复杂外部前向不自动获得梯度规则。
  - 内置 `ft.nn.functional.scaled_dot_product_attention` 使用包内 FlashAttention 实现，支持 CUDA fp16/bf16 前向与一阶反向。复杂算子的反向由 `ft.fun` 的显式 `vjp` 组合外部 kernel；单次外部调用可返回多个结果，未被图消费的物理输出仍须分配。
- **自动微分与原语闭合**：
  - 公开自动微分接口为 `ft.grad(outputs, inputs, *, seed=None)`：纯 IR 变换返回待计算 Tensor，输入为 tuple/list 时返回 tuple。输出可为 Tensor 或 tuple/list，非标量须显式提供匹配单项/序列结构、长度、shape、dtype 和 Engine 的 seed（tuple/list 可互换）；标量默认种子为 1。重复求导与高阶求导通过组合导数表达式完成，未连接到输出的输入返回零梯度。张量创建和权重加载只负责数据与资源，不持有梯度开关或累加状态。
  - 新增基础原语时，在 `ir/primitives.py` 注册规则，几何、归约和初等函数规则放在 `ir/tensor_rules.py`，提供可执行的 `abstract_eval`、`jvp` 和 `vjp` 规则，微分产物保持为同一套闭合的 Fuga IR。
  - 对缺少可组合原语规则的不透明外部算子（如活跃裸 `fuga.external` 或导数图中的未定义微分外部 kernel），在构造导数 IR 时校验并报错拒绝。
- **权重格式**：
  - `weights.py` 只序列化由嵌套 dict/list/tuple 与 Tensor 组成的 PyTree，不记录模型类、算子或设备；数组使用 dtype/shape/布局加数据的内容寻址身份。
  - 新增网络结构、参数命名或基础原子算子不改变 v1 容器；只有存储语义变化才提升 manifest `schema_version`。
- **外部编译器与 C++ 扩展**：
  - 基础算子在 CPU 与 GPU 上均走 ASK 自有降级，不依赖 Triton。`ft.triton.kernel` / `from_source` 保存用户源码，不在宿主进程导入 Triton；`from_jit` 可快照调用方已加载的 JIT 函数和依赖。worker 执行可信 Python 源码并追踪加载的依赖文件，不用预置模板替代用户 kernel。
  - Triton 适配位于本包 `_triton_worker.py`，通过 ASK 通用 worker 协议返回 `cubin` 与启动元数据；不要求 ASK 构建注册、不导出 LLVM 或插入融合标记。受支持的稳定版本范围、默认验证基线与 ABI 记录在 `triton.dependencies.lock.json`。
  - 可选依赖通过标准 extras 提供：`fugatensor[pure]` 不安装 Triton，`fugatensor[all]` 由 pip 安装支持系列。两者共用 wheel；未指定独立环境时从当前解释器的安装元数据发现 Triton，仍不在宿主导入编译器。
  - C++ 扩展 `fugatensor._native`（`cpp/`）负责 SSA 图分析和 ASK 原生绑定，实际编译、整段计划提交、缓冲区、事件与 CUDA Graph 管理由 ASK C++ 实现。`ir/transforms.py` 统一封装 `analyze_function`、`prune` 与 `project_outputs`；Fuga 不复制运行时调度器。

---

## 构建与测试

- **构建指令**：
  - 两个项目分别构建：先安装 ASK 0.1.0 的 C++ SDK，再用它构建 FugaTensor；ASK 不发布 Python 包，FugaTensor wheel 自带配套 ASK 共享库。
  - FugaTensor 构建：`CMAKE_PREFIX_PATH=/path/to/ask-sdk python3 -m pip wheel --no-deps . --wheel-dir dist`。原生扩展使用 `ASK::Compiler` / `ASK::Runtime` 的公开接口，不读取 ASK 源码，也不需要 LLVM SDK。
  - 开发安装：`CMAKE_PREFIX_PATH=/path/to/ask-sdk python3 -m pip install -e '.[pure]'`，使用 Triton 时改为 `'.[all]'`，随后执行 `python3 -m pip check`。安装详情见 [`docs/zh/install.md`](docs/zh/install.md) 与 [`docs/en/install.md`](docs/en/install.md)。
  - 包要求 Python 3.10+；CPU JIT 运行还需 `cc` 链接器，CUDA 另需显式配置 `llc` / `ptxas` 与驱动。
- **测试组织**：
  - 使用 pytest fixture、原生 `assert`、`pytest.raises` 与参数化。输入生成和独立数值参考位于 `tests/support/`，资源 fixture 位于 `tests/conftest.py`；数值矩阵复用 Engine，状态测试独立初始化。
  - 默认矩阵用生成式两两组合和定向边界分层覆盖 dtype、shape、layout 与图组合；`--exhaustive` 展开完整组合。因素对覆盖与更高阶交互分别陈述，不以缩减矩阵代替生命周期、定义失效、物化、并发和数值边界断言。
  - FP32 CPU matmul 按 OpenBLAS 的 f32 累加误差界校验，CUDA FP32 matmul 按 TF32 输入与 f32 累加误差界校验；FP16 CPU 保留逐 K 累加规则。参考以高精度点积和绝对乘积和确定尺度化容差。
  - 测试命令、筛选与新增覆盖方式见 [`docs/zh/testing.md`](docs/zh/testing.md) 与 [`docs/en/testing.md`](docs/en/testing.md)。
- **测试运行**：
  - 测试依赖：`python -m pip install -e '.[pure,test]'`；Triton 环境用 `'.[all,test]'`。
  - CPU、IR 与原生图分析：`.venv/bin/python -m pytest -q`。
  - 自有 CUDA：`.venv/bin/python -m pytest --cuda -q`；CUDA 与 Triton：`.venv/bin/python -m pytest --triton -q`。显式启用的驱动、工具和 worker 缺失时报错；skipped 单独记录。
  - 后端桥接与产物协议独立选择：`.venv/bin/python -m pytest tests/backend -q`；物理 GPU 增加 `--cuda`，配置 `ASK_SOURCE_ROOT`、`ASK_OPT`、`ASK_LLVM_TOOLS_DIR` 与 `ASK_PTXAS`。
  - wheel 安装独立选择：`.venv/bin/python -m pytest tests/packaging --wheel /path/to/fugatensor.whl --triton -q`，为 pure/all 分别安装依赖；默认发现不进入 backend/packaging。
  - `ASK_SOURCE_ROOT` 只供集成测试读取 ASK fixture、依赖锁和工具路径；ASK CTest 独立运行。
  - 使用已安装 `_native` 扩展的解释器运行测试。

---

## 文档与双向链接

- **Agent 读取与双语维护约定**：
  - **开发读取**：Agent 在日常开发、代码理解、架构参考与上下文检索时，**只读取英文版本（`docs/en/` 与根目录 `README.md`）** 即可，无需重复读取中文文档以节省上下文。
  - **文档修改**：当修改接口、代码逻辑、构建配置或补充说明时，**必须同步修改中英文双语版本（`docs/en/` 与 `docs/zh/`，以及根目录 `README.md` 与 `README.zh.md`）**，确保双语内容、代码示例与 `<a id="..."></a>` 锚点保持严格对齐。
- 文档分为中文 `docs/zh/` 与英文 `docs/en/` 双语目录（包含 `API/`、`architecture/` 和 `ir/`），由根目录 `docs/README.md` 语言导航索引，并与同级 `../ask/docs/`（`zh/` 与 `en/`）跨仓互链。跨仓链接仅供源码浏览，并列目录不是构建或运行依赖：
  - 文件名按主题命名，不带数字序号前缀，中英文目录结构保持镜像对齐；
  - API 分类文档说明签名、参数、返回值、示例与支持边界，首行链接分类目录和相关接口；实现原理链接架构或 IR 专题，避免重复；
  - 总览文档（`architecture/README.md`、`ir/README.md`）每节设置 `<a id="..."></a>` 锚点，节末以 `工程细节：...` 指向专题文档；
  - 专题文档首行以 `返回架构总览 / 返回 IR 总览：... · 相关细节：...` 链回总览与关联文档，内部小节提供 `<a id="..."></a>` 锚点；中英文锚点 ID 保持完全一致；
- 修改文档标题、文件名或锚点后，校验 `fugatensor` 与同级 `ask` 仓库中的 Markdown 相对链接和 `#anchor` 引用。

# FugaTensor 仓库工程规范

## 工作约定

1. 完成一组代码或文档修改并通过相关验证后，创建本地 Git 提交（使用仓库已配置的 Git 身份），便于回退。未完成的检查点需在提交说明中标明状态。
2. 第三方 SDK、构建产物（`build/`、`dist/`、`*.egg-info/`）、Python 虚拟环境（`.venv/`）和缓存（`.cache/`、`__pycache__/`）不纳入 Git；保留依赖锁记录及引导脚本（`tools/bootstrap.py`）。
3. 文档使用自然直接的中文说明设计动机、模块配合与能力边界，避免 AI 式套话、过渡废话，不把未实现的规划写成现有能力。
4. 修改接口、IR 规则、构建参数或测试结构时，同步更新 `README.md`、`docs/architecture/`、`docs/ir/` 及相关示例，并保持双向链接锚点有效。

---

## 架构与编码规范

- **Fuga IR 独立性与算术执行边界**：
  - Tensor 自动组图生成独立于 ASK 的 Fuga IR（`src/fugatensor/ir/`），框架层的控制流化简、C++ 活跃性分析与 JVP/VJP 自动微分均作用于 Fuga IR，最后通过 `lowering.py` 降级为高层 ASK IR。
  - Fuga IR 不包装或依赖 `ask.Value`、`ask.Type` 或 MLIR 字符串。
  - 实际张量算术统一编译后交由 ASK 执行，不在 Python 中实现张量数值计算回退。
- **自动组图、执行边界与 `ft.fun` 契约**：
  - 用户无需 JIT 装饰器；普通 Python 函数每次正常执行，Tensor 运算返回待计算图节点。读取数据、0-D 标量条件判断、反向传播与 `ft.evaluate(...)` 触发所需依赖图的 JIT 编译和执行。
  - 跨算子与普通函数调用保留图连接和融合机会；执行缓存属于 `Engine`，按图结构和自定义函数定义身份复用，不能跳过用户 Python 逻辑。
  - `ft.fun` 的普通 `forward` 与可选 `backward` 自动进行隔离追踪，保留 `CallSpec`、保存值与定义变更校验；方法内部不支持数据依赖控制流。Triton 源码中的 `@triton.jit` 保留。
  - 梯度历史在组图时记录，`no_grad` 与 `detach` 的边界不随实际执行时间改变；Python 条件每次读取真实标量值，反向沿实际路径传播。
- **多 Kernel 执行**：
  - FugaTensor 向 ASK 提交完整静态图，由 ASK MLIR 切分器和版本化执行计划选择融合或多 kernel 调度；前端不再限制整图只有一个外部节点。
  - Triton 多输入通过 `inputs=` 注册，输出形状/类型和启动参数只使用 TensorSpec 元数据；复杂外部前向不自动获得梯度规则。
- **自动微分与原语闭合**：
  - 公开自动微分接口采用 `requires_grad`、`Tensor.backward(...)` 与叶子 `.grad`，不采用 `ft.grad(f)` 形式；非标量输出调用 `backward` 须显式传入梯度种子。
  - 新增基础原语时，在 `ir/primitives.py` 中提供可执行的 `abstract_eval`、`jvp` 和 `vjp` 规则，微分产物保持为同一套闭合的 Fuga IR。
  - 对缺少可组合原语规则的不透明外部算子（如裸 `fuga.external` 或请求高阶导数的手写 Triton 算子），在执行反向编译或写入 `.grad` 之前的预检阶段报错拒绝。
- **外部编译器与 C++ 扩展**：
  - 基础算子在 CPU 与 GPU 上均走 ASK 自有降级，不依赖 Triton。`ft.triton.kernel` / `from_source` 接收并保存用户编写的源码，不在宿主进程导入 Triton，也不用库内预置模板替代外部源码接入。
  - C++ 扩展 `fugatensor._native`（`cpp/`）负责无全局状态、释放 GIL 的 SSA 可达性与反向边界分析；`ir/transforms.py` 统一封装 `analyze_function`、`prune` 与 `project_outputs` 供全仓复用。

---

## 构建与测试

- **构建指令**：
  - ASK 是必需的运行依赖，但两个项目分别构建：先在 ASK 自身工程中按其依赖锁生成 `ask-compiler==0.1.0` wheel，再独立构建 FugaTensor，最后将两个兼容 wheel 安装到运行环境。
  - FugaTensor 独立构建：`python3 -m pip wheel --no-deps . --wheel-dir dist`，不要求安装 ASK、检出 ASK 源码或准备 LLVM SDK；本项目 C++ 扩展只依赖自身源码与构建依赖。
  - 已安装兼容 ASK 时，仅修改 FugaTensor 可运行 `python3 -m pip install --no-deps -e .`，随后执行 `python3 -m pip check`。安装详情及可选开发引导见 [`docs/安装.md`](docs/安装.md)。
  - 包要求 Python 3.10+，`tools/bootstrap.py` 开发引导需要 Python 3.11+；CPU JIT 运行还需 `cc` 链接器，CUDA 另需显式配置 `llc` / `ptxas` 与驱动。
- **测试运行**：
  - 测试收敛在 `tests/` 下 10 个模块（共用 `tests/_helpers.py` 中的 `EngineTestCase`）：
    - CPU + IR + 原生图分析：`.venv/bin/python -m unittest -v`
    - 启用自有 CUDA 硬件测试：`FUGATENSOR_TEST_CUDA=1 .venv/bin/python -m unittest -v`
    - 启用自有 CUDA + 用户 Triton 融合测试：`FUGATENSOR_TEST_CUDA=1 FUGATENSOR_TEST_TRITON=1 .venv/bin/python -m unittest -v`
  - 始终使用已安装 `_native` 扩展的 `.venv/bin/python` 运行测试，不使用脱离原生扩展的 `PYTHONPATH=$PWD/src`。

---

## 文档与双向链接

- 文档分设 `docs/architecture/`（系统架构与运行时）和 `docs/ir/`（Fuga IR 对象模型、变换与降级）两个目录，由 `docs/README.md` 索引，并与同级 `../ask/docs/` 跨仓互链。跨仓链接仅供源码浏览，并列目录不是构建或运行依赖：
  - 总览文档（`docs/architecture/README.md`、`docs/ir/README.md`）每节设置 `<a id="..."></a>` 锚点，节末以 `工程细节：...` 指向专题文档；
  - 专题文档首行以 `返回架构总览 / 返回 IR 总览：... · 相关细节：...` 链回总览与关联文档，内部小节提供 `<a id="..."></a>` 锚点。
- 修改文档标题、文件名或锚点后，校验 `fugatensor` 与同级 `ask` 仓库中的 Markdown 相对链接和 `#anchor` 引用。

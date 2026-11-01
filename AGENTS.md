# FugaTensor 仓库工程规范

## 基本工作铁律

1. **验证后创建本地提交**：完成一组代码或文档修改并跑通相关验证后，必须创建本地 Git 提交（使用仓库已配置的 Git 身份），方便出问题时回退。未完成的检查点必须在提交说明中明确标明状态。
2. **保持版本库干净可复现**：第三方 SDK、构建产物（`build/`、`dist/`、`*.egg-info/`）、Python 虚拟环境（`.venv/`）和缓存（`.cache/`、`__pycache__/`）绝不纳入 Git；保留依赖锁记录及一键可复现脚本（`tools/bootstrap.py`）。
3. **文档要写得像人话**：用自然、通顺、直指工程实质的中文解释清楚设计动机、模块衔接与能力边界，不写僵硬晦涩的公文套话，也不把未实现的规划包装成现有能力。
4. **代码与文档严格同步**：修改接口、IR 规则、构建参数或测试结构时，必须同步更新 `README.md`、`docs/architecture/`、`docs/ir/` 及相关可运行示例，并维护好双向链接锚点。

---

## 架构与编码规范

- **Fuga IR 独立性与零 Python 算术原则**：
  - Python 追踪先生成完全独立于 ASK 的 **Fuga IR**（`src/fugatensor/ir/`），框架层的控制流化简、C++ 活跃性分析与 JVP/VJP 自动微分均作用于 Fuga IR，最后通过 `lowering.py` 降级为高层 ASK IR。
  - Fuga IR 绝不依赖或包装 `ask.Value`、`ask.Type` 或 MLIR 字符串。
  - 所有实际张量算术计算一律编译后交给 ASK 执行，严禁在 Python 中偷写一套张量数值循环冒充 JIT。
- **统一的 `@ft.jit` 与 `ft.fun` 契约**：
  - 参与张量计算的函数（包括内置运算及用户算子）统一声明 `@ft.jit`；自定义求导类 `ft.fun` 的 `forward` 和提供的 `backward` 也必须显式声明 `@ft.jit`。
  - 普通嵌套 `@ft.jit` 调用必须内联进外层追踪图以保留完整数据流与求导依赖，不得在追踪期中途执行缓存产物切断图连接。
  - Python 0-D 标量运行时控制流通过受控图中断（Graph Break）分段编译调度，各片段之间必须通过 `GradHistory` 完整串联反向历史，绝不能把首次运行的偶然分支条件固化进特化缓存。
- **自动微分与原语闭合规范**：
  - 公开自动微分接口保持 `requires_grad`、`Tensor.backward(...)` 与叶子 `.grad` 命令式风格，不引入 `ft.grad(f)` 风格外壳；非标量输出调用 `backward` 必须显式传入梯度种子。
  - 新增基础原语时，必须在 `ir/primitives.py` 中同时提供可执行的 `abstract_eval`、`jvp` 和 `vjp` 规则，且微分产物继续保持为同一套闭合的 Fuga IR。
  - 对缺少可组合原语规则的不透明外部算子（如裸 `fuga.external` 或请求高阶导数的手写 Triton 算子），必须在执行任何反向编译或修改任何 `.grad` 之前的预检阶段明确报错拒绝。
- **外部编译器与 C++ 核心演进**：
  - 基础算子在 CPU 与 GPU 上均直接走 ASK 自有降级，不依赖 Triton。`ft.triton.kernel` / `from_source` 只接收并原样快照用户编写的源码，不在宿主进程导入 Triton，更不用库内硬编码模板冒充外部接入能力。
  - 必选 C++ 扩展 `fugatensor._native`（`cpp/`）负责无全局状态、释放 GIL 的 SSA 可达性与反向边界分析；`ir/transforms.py` 统一封装 `analyze_function`、`prune` 与 `project_outputs` 供全仓复用，避免多处重复实现图裁剪。

---

## 构建与测试规范

- **精简构建流程**：
  - 完整构建与环境引导：`python3 tools/bootstrap.py`（默认构建 `all` 适配器并将配置写入 `.venv/share/fugatensor/engine.json`）。
  - 仅修改 FugaTensor 代码时快速重编：`python3 tools/bootstrap.py --skip-ask`。
  - 仅构建无第三方适配器的核心路径：`python3 tools/bootstrap.py --compilers none`。
- **测试体系约定**：
  - 测试按职责收敛在 `tests/` 下 10 个正交模块（共享 `tests/_helpers.py` 中的 `EngineTestCase`），避免按单次修复堆砌碎片化测试文件：
    - CPU + IR + 原生图分析快速回归：`.venv/bin/python -m unittest -v`
    - 启用自有 CUDA 硬件回归：`FUGATENSOR_TEST_CUDA=1 .venv/bin/python -m unittest -v`
    - 启用自有 CUDA + 用户 Triton 融合回归：`FUGATENSOR_TEST_CUDA=1 FUGATENSOR_TEST_TRITON=1 .venv/bin/python -m unittest -v`
  - 切勿使用脱离 C++ `_native` 扩展的裸 `PYTHONPATH=$PWD/src` 运行测试；始终使用安装好原生扩展的 `.venv/bin/python`。

---

## 文档与双链接维护规范

- 文档分设 `docs/architecture/`（系统架构与运行时专题）和 `docs/ir/`（Fuga IR 对象模型、变换与降级专题）两个专属目录，由 `docs/README.md` 统一索引，并与同级 `../ask/docs/` 跨仓互链：
  - 总览文档（`docs/architecture/README.md`、`docs/ir/README.md`）每节设置显式 `<a id="..."></a>` 锚点，并在节末以 `工程细节：...` 指向专题文档；
  - 专题文档首行以 `返回架构总览 / 返回 IR 总览：... · 相关细节：...` 回链总览及上下游文档，内部各小节同样提供稳定的 `<a id="..."></a>` 锚点。
- 修改文档标题、文件名或锚点后，必须校验 `fugatensor` 与同级 `ask` 仓库中的所有 Markdown 相对链接和 `#anchor` 全部有效。

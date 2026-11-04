# 原生图分析与 C++ 边界

返回架构总览：[独立 Fuga IR 与 C++ 原生图分析](README.md#arch-ir-native) · 相关细节：[张量模型与追踪机制](张量模型与追踪机制.md) · [自动微分与自定义算子](自动微分与自定义算子.md) · [Fuga IR 对象模型](../ir/对象模型与验证.md)

自动微分和输入裁剪都要计算 SSA 图的可达范围、活跃节点和反向遍历顺序。按需求值还要收集待计算 Tensor 的依赖并建立编译缓存键。这些分析由 C++17 扩展 `fugatensor._native` 完成，Python 层负责构建和变换 Fuga IR。

<a id="python-cpp-boundary"></a>

## 1. Python 与 C++ 的模块职责

| 层级 | 代码位置 | 职责 |
| --- | --- | --- |
| **Python 框架与 IR 层** | `src/fugatensor/*.py`、`src/fugatensor/ir/*.py` | `Tensor` / `ft.grad` / `ft.evaluate` / `ft.fun` / `ft.triton` 接口、Tensor 自动组图、按需执行与隔离追踪、Fuga IR 对象结构、JVP/VJP 求导规则展开、向 ASK IR 降级及 `Engine` 后端编排。 |
| **FugaTensor C++ 扩展** | `cpp/graph.*`、`cpp/lazy.*`、`cpp/trace.cpp`、`cpp/primitive.cpp`、`cpp/compiler.cpp`、`cpp/ask_runtime.cpp`（`fugatensor._native`） | SSA 活跃性与逆拓扑分析；原语参数绑定、追踪节点与待计算节点的 arena 存储；输出需求传播、依赖收集、定义失效校验与结构键生成；绑定 ASK 具名编译结果和 `PreparedProgram`，通过原生计划提交执行。 |
| **ASK C++ 编译与运行时** | ASK C++ SDK 的 `ASK::Compiler` / `ASK::Runtime` 共享库 | `ask` dialect 校验与优化、CPU / 自有 CUDA 原生 LLVM IR 代码生成、图切分、原生执行计划、缓冲区与事件复用、CUDA Graph replay；FugaTensor 的私有 `_ask/` 负责图转换、worker 调用、产物验证和磁盘缓存。 |

`ir/transforms.py` 调用 `_native` 做图分析；编译和执行绑定也通过这个扩展调用 ASK 共享库。构建流程见[安装文档](../安装.md#build-fugatensor)。

`cpp/compiler.cpp` 绑定 ASK `PrepareResult`、`PartitionResult`、`CodegenResult`；内部 Python Compiler 将版本化 JSON 字段解码成对应阶段的具名结果，编译输入来自独立契约字段。CLI 完整报告在桥接入口做同样拆分。字段用途见[编译阶段结果](ASK桥接与多后端执行.md#compiler-results)。执行绑定由 `cpp/ask_runtime.cpp` 调用 `PreparedProgram.run` / `submit`，缓冲区、调度与在途资源由 ASK 管理。

<a id="native-analyze"></a>

## 2. 可达性与反向边界分析（`cpp/graph.cpp`）

`fugatensor._native.analyze(...)` 接收函数的输入数、按拓扑序排列的操作数与结果数量、各节点的副作用标记、输出值以及需要求导的输入序号。

C++ 层检查值编号和拓扑顺序，`bindings.cpp` 同时拒绝用 Python `bool` 充当整数 ID。随后在线性时间 $O(V + E)$ 内计算三组结果：

1. `active_values`：从所选输入出发，沿 `operand → result` 标记受其影响的值。
2. `reverse_operations`：从受影响的输出向前回溯，生成所需节点的逆拓扑顺序。未流向目标输出或不依赖所选输入的节点跳过求导；多返回值节点只出现一次。
3. `live_operations`：从输出和副作用节点回溯依赖，生成前向保活序列，供 DCE 使用。活跃的数值系数和副作用节点可以保留在这里，而不参与反向求导。

<a id="transforms-reuse"></a>

## 3. 图剪枝与输出投影（`ir/transforms.py`）

`ir/transforms.py` 基于 `_native.analyze` 封装了三个共用函数：

- `analyze_function(function, selected=())`：将平坦函数的 SSA 边编码给 `_native.analyze`，返回操作序列、按编号排列的值和分析结果。
- `prune(module)`：按分析结果中的 `live_operations` 重建函数，删除无副作用的死节点。
- `project_outputs(module, indices, ...)`：先化简静态控制流，再选取输出并调用 `prune` 删除无关支路。

含控制流区域的图先经 `specialize` 展开，再做平坦图分析。

`ir/differentiate.py`、`ir/jvp.py`、`autograd.py` 和 [`engine.py` 的输入裁剪](ASK桥接与多后端执行.md#engine-lifecycle) 共用这些函数。

<a id="lazy-collection"></a>

## 4. 待计算 Tensor 的依赖收集（`cpp/lazy.cpp`）

`_lazy.materialize` 将本次请求的根 Tensor 一次交给 `_native.collect_pending`。原生代码先合并多输出表达式的需求，再确定活跃输入、表达式拓扑序和共享叶子编号。已物化 Tensor 作为数值输入；别名沿原来源解析。仅产生常量的图从原输入链中选择 ABI 所需的存储载体。

每个 Engine 持有 `RuntimeArena`。值记录、表达式记录和输入边保存在按需增长并复用容量的连续数组中，以 `uint32_t` 编号连接；形状类型、模板、定义依赖和存储资源放在独立表中。`Tensor` 的 Python 类保留算子方法，原生薄壳持有 Engine 和值编号。表达式来源也存于原生节点，自动微分按需读取视图。

节点创建时，根据模板指纹和有序输入指纹计算增量摘要。`PlanKey.__hash__` 读取已保存的摘要；键相等比较规范化输入、连接关系、输出顺序和定义身份。物化改变数值输入边界时增加 arena 版本，收集时刷新相应摘要。多输出投影按本次实际需求建立精确键。

模板投影和相关定义在首次使用时由 Python 准备并缓存。热收集遍历原生索引，取得本次存储输入并校验活跃定义，返回输入与结构键。缓存未命中时，`describe_pending` 才生成表达式视图供 Python 拼接 Fuga IR；命中路径直接绑定输入执行。

<a id="native-tracing"></a>

## 5. 原生追踪与参数绑定

`cpp/primitive.cpp` 为内置原语编码精确参数签名：Tensor 使用同一 Engine 内稳定的类型编号，静态参数按类型和值编码，浮点值保留位模式。命中模板后直接调用原生待计算节点绑定入口。普通 Python 函数和自定义函数回调仍按调用执行；原语模板的定义版本在复用前校验。

`cpp/trace.cpp` 使用预留容量的连续容器存储追踪节点、值编号、输入边和属性原子。节点记录为 POD，类型表持有共享的不可变 Fuga 类型，`CallSpec` / `ExternalSpec` 放在独立的 Python 对象表中。`TraceTensor` 只持追踪上下文和值编号。

追踪 builder 复用 `ir/core.py` 的原语构建方法与抽象类型规则，将推导结果写入 arena。`TraceContext.finish` 导出一次 Python IR 并验证；自定义 VJP 的导出入口把新增操作追加到已有求导图。公开 Python IR、自动微分规则和 ASK 降级继续共用同一套语义。调用方式见[内部追踪](张量模型与追踪机制.md#trace-context)。

<a id="gil-and-concurrency"></a>

## 6. 并发与所有权

`cpp/graph.cpp` 的 `analyze(const Graph&)` 使用本次调用的局部状态和只读输入。绑定层转换参数后，通过 `nb::gil_scoped_release` 释放 GIL，分析结束后重新获取 GIL 并返回结果。多线程覆盖见 `tests/test_native_graph.py`。

Engine 的求值锁保护依赖收集、编译缓存和结果发布。原生节点由薄壳、输入边和求导视图引用计数保活；零引用节点迭代释放，并将槽位归还池中。已物化输出继续保留求导来源。模板注册表使用弱所有权，活跃表达式保留所需模板；缓存键保留精确比较所需的静态描述。

原生扩展向 CPython 的 GC 暴露模板、定义和 Engine 的 Python 引用，用于回收用户回调形成的所有权环。节点池本身按原生引用计数回收，容量按已出现的同时存活规模复用。

原语模板缓存按 Engine 弱引用分组，`Engine.close` 与原语 `clear_cache` 清理对应记录。原生追踪句柄、arena 的 Python 对象表和模板缓存注册 GC 遍历与清理，使跨语言引用参与 Python 循环回收。

# JIT、backward 与 fun 类

[返回 Fuga IR](README.md) · [内部 VJP 变换](02-控制流与自动微分.md) · [ASK 降级](03-降到ASK.md) · [项目说明](../../README.md) · [ASK 中的框架边界](../../../ask/docs/18-FugaTensor接入示例.md)

这份文档最初用于讨论接口，现在按张量上的 backward 接口说明实现。公开用法是 `requires_grad=True`、`output.backward()` 和 `input.grad`；不提供 `ft.grad(f)` 或 `ft.vjp(f, ...)`。编译器内部仍用 IR VJP 生成反向计算，用户不必直接调用它。

## 1. 创建输入、计算，再反向传播

```python
import fugatensor as ft

@ft.jit
def loss(x):
    return ft.square(x) + 3 * x

x = ft.tensor(2.0, requires_grad=True)
y = loss(x)
y.backward()
print(y.tolist())       # 10.0
print(x.grad.tolist())  # 7.0
```

`requires_grad` 默认是 False，只有浮点 Tensor 可以开启。叶子输入的 `.grad` 初始为 None，反向传播后保存与输入同形状、同 dtype、同设备的梯度 Tensor。中间结果通常不保存 `.grad`，它们的梯度继续传向叶子。

`jit` 负责追踪和执行实际计算。每次真实执行后，运行时按需要记录输入、输出和对应 Fuga 图，形成反向传播的连接。内层 JIT 在外层追踪期间仍向图中记录计算，不执行已有机器码来截断依赖。两个分别执行的 JIT 调用也能通过张量历史连接。

计算函数、内置算子和 fun 的计算方法都声明 `@ft.jit`。内置加、减、乘、取负、平方、cast、zeros_like、ones_like 都能被追踪；减法、取负和平方最终组成已有的基础 Fuga 操作。

## 2. 张量输出要提供梯度

```python
x = ft.tensor([1, 2, 3], requires_grad=True)
y = ft.square(x)
y.backward(ft.tensor([1, 2, -1]))
print(x.grad.tolist())  # [2.0, 8.0, -6.0]
```

没有显式 gradient 时，输出必须是单个浮点零维 Tensor，运行时使用同类型的标量一。形状 `(1,)` 也不是零维，不能隐式求和。显式 gradient 的形状、dtype 和运行时必须与输出匹配。

一个函数返回多个 Tensor 时，可以把结果继续组合成一个标量 loss，再反向传播；也可以分别调用 backward：

```python
@ft.jit
def pair(x):
    return x * x, x * 3

x = ft.tensor(2.0, requires_grad=True)
a, b = pair(x)
a.backward(retain_graph=True)
b.backward()
print(x.grad.tolist())  # 7.0
```

未选中的输出使用零 cotangent，选中输出的贡献继续累加到同一叶子。一个值被多个操作使用、同一个输入占据两个参数位置，都必须把各条路径的贡献相加。

## 3. 梯度累加、释放与停止记录

`backward()` 累加叶子的梯度，不覆盖前一次结果。`x.zero_grad()` 把 `.grad` 设为 None；`x.zero_grad(set_to_none=False)` 把已有梯度清零。下一轮训练前是否清理，由调用者决定。

普通 backward 默认释放相关图的反向状态，再次沿同一图反传会报错。可用 `retain_graph=True` 保留；不显式传 retain_graph 时，它跟随 create_graph，因此高阶请求默认保留所需图。重新执行正向会建立新图，可以对同一叶子继续累加梯度。

`x.detach()` 返回不连接原历史的 Tensor。`with ft.no_grad():` 内执行计算不记录梯度历史，退出后恢复原状态。这两种操作都不改变已有输入的数据，也不把 Tensor 转成 Python 数值。

当前 no_grad 应包住真实 JIT 调用，不能放进 JIT 函数体；后者会明确拒绝，避免词法上下文在追踪后失去含义。detach 目前用于真实 Tensor，追踪值尚未实现这个操作。

默认 backward 不为梯度记录新历史。`create_graph=True` 则把生成的梯度计算继续表达为 Fuga 基础操作，允许再对梯度反传：

```python
@ft.jit
def cube(x):
    return x * x * x

x = ft.tensor(2.0, requires_grad=True)
cube(x).backward(create_graph=True)
first = x.grad
print(first.tolist())  # 12.0
x.zero_grad()         # 清理叶子累加，不丢弃已经保存的 first
first.backward()
print(x.grad.tolist()) # 二阶导数 12.0
```

继续构建下一阶时，也对对应 backward 传 create_graph=True。显式输出梯度若自身 requires_grad，也保持在这张图中。梯度累加的加法同样可追踪；为避免把不同阶的结果相加，示例每次先保存当前梯度，再清理叶子 `.grad`。

高阶路径必须能用已知的基础规则组合：常量、浮点加乘、浮点 cast 及适用的 splat。规则目录的抽象求值、VJP/JVP 回调实际参与构图和变换，见[原子算子与可组合变换](05-原子算子与可组合变换.md)。浮点 cast 沿用忽略舍入的梯度约定，不是量化函数的严格导数。

梯度图继续保留 fun 的 call 边界，直到交给 ASK 执行时才展开。下一阶对可见输出的 cotangent 仍调用用户 backward；对隐藏保存值的 cotangent，则从 forward 投影出相应 SSA 计算，再递归做 VJP。投影中的嵌套 fun 仍使用自己的规则，两路贡献同时存在时相加。这样不会在提前内联时丢掉用户声明的反向语义。

所需正向或生成的反向包含不透明 external 时，create_graph 请求会在任何反向编译、执行或梯度写入前拒绝。手写了一阶 Triton backward，也不能跳过高阶检查。这里只支持可组合的规则，不是任意程序的符号微分。

显式 create_graph=True 会在外层 no_grad 内临时开启反向图记录，完成后恢复原上下文。若某个梯度已经是与输入无关的常量，它可能不再 requires_grad，不能把重复调用 backward 当作自动补零接口。

## 4. fun 类定义正向与手写反向

```python
class Square(ft.fun):
    @ft.jit
    def forward(self, ctx, x):
        ctx.save_for_backward(x)
        return ft.square(x)

    @ft.jit
    def backward(self, ctx, dy):
        (x,) = ctx.saved_tensors
        return (2 * x * dy,)

square = Square()
x = ft.tensor(3.0, requires_grad=True)
square(x).backward()
print(x.grad.tolist())  # 6.0
```

`forward` 和提供的 `backward` 都必须用 `@ft.jit` 装饰，类定义时检查。只做推理的 fun 可以省略 backward；但需要穿过它反向传播时会报缺少规则。用户声明的反向是求导边界，不能在缺少规则时偷偷改为对函数体求导。

ctx 每次调用独立。`save_for_backward` 显式保存 Tensor 引用，反向从 `saved_tensors` 按顺序取得；追踪中的引用对应 SSA 值及 residual，不会下载到 Python。仅推理时不会调用 backward，也不需要保留只供反向使用的计算。

第一版只允许保存 Tensor；静态选项作为 fun 实例的不可变配置，不能把实际 Tensor 藏入实例或全局变量。forward/backward 遵守纯计算约定，不修改输入和外部状态。ctx 只负责保存关系，不是任意副作用的入口。

反向契约如下：

- forward 返回 Tensor 时，dy 是 Tensor；forward 返回 tuple 时，dy 是对应结构的 tuple，包括单元素 tuple。
- backward 始终返回 tuple，每个浮点 Tensor 输入一个梯度，按参数顺序排列。bool/int Tensor 和静态参数不占槽位。
- 每个返回梯度必须与对应输入的 shape、dtype 一致。混合精度可以在内部使用 f32，返回前显式转回。
- 即使某个输入不要求梯度，手写 backward 也遵守完整返回契约；后续可以删除未使用的计算。
- 缺少返回项、None 或未知导数不表示零梯度。数学上为零时，应显式产生零 Tensor。

这些检查在反向执行前完成。接口检查只能发现类型和结构错误，不能证明手写公式与正向数学上等价。

## 5. 缺少规则怎样报错

正向执行可以调用没有反向规则的算子。只有请求 backward 时，才根据需要的梯度路径检查规则。运行时先为所有相关节点构造并验证反向计划，再编译和执行反向；发现规则缺失时，不应先执行一部分反向或写入一部分叶子梯度。

| 情况 | 行为 |
|---|---|
| 裸 external 位于必经求导路径，且没有规则 | 报错，指出需要规则的 external |
| external 只生成与求导输入无关的系数 | 可以使用正向结果，不要求对该系数来源求导 |
| external 位于有 backward 的 fun 正向中 | 使用 fun 的规则，不穿过其正向 external 自动求导 |
| fun 缺少 backward 且位于必经路径 | 报错，指出 fun 类的缺失规则 |
| 分支没有执行，因此没有进入当前历史 | 不检查这条未执行路径的规则 |

第一版诊断指出出错 external 或 fun 类，不承诺恢复完整 Python 调用栈。错误不会以零梯度或 CPU 回退掩盖。

## 6. Python 控制流与图中断

```python
@ft.jit
def choose(x, flag):
    if flag:
        return ft.square(x)
    return x * 3

x = ft.tensor(2.0, requires_grad=True)
y = choose(x, ft.tensor(True, dtype="bool"))
y.backward()
print(x.grad.tolist())  # 4.0
```

条件 Tensor 必须是零维。Python 求其真假时，JIT 先提交此前已记录的片段，读取标量条件，再在同一个 Python 调用帧中继续执行选中的分支。CUDA 条件读取需要等待设备结果。向量和 `(1,)` Tensor 不会自动变成 any/all；目前也未新增张量比较算子，可使用已有 bool 值或显式 cast。

`while` 同样按实际条件执行，每轮产生的计算通过历史连接；反向沿本次实际执行的分支和循环次数返回。这里由 Python 选择下一片段，没有把动态 if/while 直接降给仍要求静态图的 ASK。

包含动态控制流的函数每次调用都要运行 Python 控制逻辑，但可复用已编译片段。`function.last_segments` 按顺序记录本次的 Fuga IR、ASK IR、报告和产物目录；不能把某一个片段当作整次调用的图。

自定义 fun 的隔离 forward/backward 当前仍要求静态追踪，不支持在其内部用运行时 Tensor 条件中断；发现这种情况会在执行前拒绝。可把控制流放在外层函数中，再调用静态 fun。

## 7. 从反向历史到 ASK

```text
真实正向执行 -> 张量历史连接实际执行的 JIT 片段
backward 请求 -> 找到需要反传的节点，验证全部反向规则
             -> 保留 fun 边界并选择用户 backward
             -> 内部 IR VJP 生成梯度图，保留后续求导所需的 fun 边界
             -> 执行降级时展开调用，生成静态 ASK IR
             -> CPU/CUDA 反向执行 -> 叶子梯度累加
```

每一阶都先选择规则，执行降级才展开 fun。若把用于后续求导的图提前内联，会丢失用户声明的反向边界；即使函数体可由基础算子表达，也不能绕过手写规则。执行展开后的基础计算继续优化；Triton DSL 仍作为 external 提交，不会自动变成 Fuga 基础算子。

反向逻辑参数可能包含正向输入与输出梯度。若某个参数已不参与反向计算，执行桥会删去对应物理端口，并保存从逻辑输入到实际执行输入的映射。例如线性 scale 的反向只用 dy，就能把它单独交给现有单输入 external 路径。仍需要多个实际输入的 external 反向不会因此自动获得融合支持。

普通基础图和历史连接不绑定设备。实际 CPU/CUDA、GPU 架构仍由 ASK 边界选择；只有 CUDA 实现的 external 在 CPU 上继续报错。external 后绕过它再读取原始输入的融合仍不支持，内联是否合法仍由 ASK 检查。

## 8. 缓存与定义变化

单图 JIT 热调用复用编译产物，每次执行按实际 requires_grad 状态建立历史。动态控制流会重新执行 Python 控制逻辑，片段按结构和定义依赖缓存。输入数值与梯度种子数值不应被固化进缓存。

嵌套 JIT 记录定义依赖。内层 `clear_cache()` 增加定义版本，外层下次调用发现变化后重新特化。fun 的方法和静态配置也参与定义检查。普通全局或闭包状态仍不自动跟踪，修改后要显式清理直接引用它的函数缓存；外部 DSL 文件变化要重新注册。

动态片段的键同时包含图文本和定义的身份、版本。两个前向文本相同、但使用不同 backward 闭包实例的 fun，不能复用同一份反向语义。缓存的反向计划还验证反向图及其依赖；已保存历史中的嵌套定义变化会报错，不能把旧正向与新反向混用。

当前还没有跨进程 Python 定义存档。ASK 的 IR 与可执行产物缓存独立工作，不等同于 Fuga 内存命中。

## 9. C++ 分工与当前边界

[cpp/graph.cpp](../../cpp/graph.cpp) 负责 SSA 拓扑、所选输入依赖、反向路径和活跃性分析，通过必要的 nanobind `_native` 扩展提供能力，没有 Python 静默替代。构建需要 C++17、CMake 和锁定的 nanobind，bootstrap 安装 Fuga 时构建。

Python 目前仍负责公开接口、追踪、张量历史、IR 对象和反向规则展开；实际 Tensor 算术由 ASK 执行。这是逐步迁移核心逻辑，没有宣称整个编译器已在 C++ 中或给出未经测量的加速结论。

当前没有归约、矩阵乘法、广播或通用保存值重算。高阶反向只接受可组合的基础规则，不支持不透明第三方运算。保存张量、释放历史和梯度累加已经接入；它们不等于完整训练框架的所有功能。多元素输出必须提供 gradient。

[可运行示例](../../examples/autodiff.py) 与[测试](../../tests/test_autodiff.py) 共用 CPU/CUDA 图验证多精度、有限差分、共享输入、显式梯度、多输出、历史释放、no_grad/detach 和错误路径。Triton 测试检查原始源码、实际正反向编译报告与数值，确认两边都经过 ASK 原生 LLVM 路径。

[高阶测试](../../tests/test_higher_order.py) 另查二阶与三阶、跨 JIT 片段、实际分支/循环、纯 fun 保存值、梯度种子及累加的依赖，并验证不透明 Triton 正向或反向会提前报错。

[嵌套高阶测试](../../tests/test_nested_higher_order.py) 进一步检查 backward 中调用 fun、保存嵌套 fun 结果、两路 cotangent 累加和反向依赖变化，确保自定义规则在每一阶都保留。

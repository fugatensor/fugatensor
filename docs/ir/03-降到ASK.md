# 从 Fuga IR 降到 ASK

[返回 Fuga IR](README.md) · [控制流与自动微分](02-控制流与自动微分.md) · [执行流程](../architecture.md) · [ASK 框架接入约定](../../../ask/docs/10-上层框架接入.md)

唯一的图转换入口是 `fugatensor.lowering.lower_to_ask(module, target=...)`，结果是新的 `ask.Module`。基础图可以不指定 target；需要绑定外部实现时必须给出目标。框架的 Value 不会直接出现在 ASK 节点里；转换过程为每个 Fuga 值建立对应的 ASK 值。

处理顺序是：验证 Fuga 图，处理能静态确定的控制流，检查剩余图是否属于当前 ASK 接口范围，再逐个翻译操作。函数的参数顺序、结果顺序、dtype 和 shape 都要对应。这个入口仍不接收动态 if/while region；Python 动态控制流由图中断运行时调度，每个实际提交的片段仍是静态图。

| Fuga 内容 | 交给 ASK 的内容 |
|---|---|
| 独立 ScalarType / TensorType | 新建对应的 ASK 类型 |
| constant / splat / add / mul / cast | ASK 的同语义操作 |
| 函数参数及返回值 | ASK 图输入与有序结果 |
| 静态 if / for | 已经选择或展开的直线计算 |
| external 及 ExternalSpec | ASK semantics、implementation 和 external |
| fun 的 call | 选定正向或反向后展开为基础操作和 external，再逐项转换 |

external 的源码正文、入口、绑定、特化常量和固定选项保持原值。当前 Triton 描述声明 CUDA 实现，实际 `arch` 在这里根据选定设备绑定；这个过程不会修改原始 Fuga Module。选择 CPU 并不会把 CUDA DSL 当作 CPU 算子执行。Fuga 负责保存高层调用及其语义，ASK 负责调用所选编译器、处理追踪标记、验证 LLVM 边界并生成执行代码。只含基础算子的图不需要第三方编译器。

```python
from fugatensor.lowering import lower_to_ask

fuga_text = module.to_text()
ask_module = lower_to_ask(module)
ask_text = ask_module.to_mlir()
```

这一步只生成图，不创建实际张量，也不执行 Python 张量计算。后面的 CPU 对象文件生成、CUDA LLVM lowering、缓存和异步执行仍由原来的 ASK 路径完成。

JIT 编译目录保存 `fuga.ir` 与 `frontend.mlir`。前者能检查追踪保留的结构，后者能检查实际交给 ASK 的图。`function.last_fuga_ir` 和 `last_ask_ir` 对应这两个层次；`last_ir` 保持之前的 ASK 文本含义。热调用复用已编译的 executable，不重复打印、变换或编译图。

`fuga-to-ask.json` 保存逐值对应关系：所属函数 `function`、特化后的 Fuga SSA 名称和 value ID、原始来源 value ID，以及对应的 ASK SSA 名称和逻辑 value ID。Fuga 的值在函数内命名；多函数 Module 转换时，ASK 操作和值的 ID 增加函数限定，避免两个函数各自的 `op0` 撞名。单函数图保留原来的 ID 写法。

循环展开后，一个原始值可以对应多个 ASK 定义，所以映射是一组记录，不是会覆盖重复项的一对一字典。读取来源时，要结合函数名与来源 ID 判断。直接调用转换时，可传 `source_map={}` 获取同一份数据；只有转换成功才会更新这个字典。

执行桥删除无用物理输入时，另在 `input_bindings` 记录逻辑输入到物理端口的对应。用户调用和反向历史仍保留原来的参数语义，实际执行只传入代码需要的存储；这不把两个仍被使用的输入合并成一个。

[集成测试](../../tests/test_fuga_ir_integration.py) 将同一张 Fuga 图以及 VJP 生成的反向图降到 ASK，再分别在 CPU/CUDA 上执行。已有的多精度、用户 Triton 源码和缓存测试继续通过 JIT 路径运行，避免独立 IR 只在打印时存在、实际执行仍走旧的直接 ASK 追踪。

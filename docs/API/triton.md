返回 API 目录：[分类目录](README.md#categories) · 关联模块：[自动微分](autograd.md) · [多后端架构](../architecture/backend.md)

# Triton 算子接入

FugaTensor 允许在不侵入宿主进程的前提下接入用户自定义 Triton Kernel，通过隔离 worker 编译为 `cubin` 产物并由后端执行引擎统一调度。

<a id="from-source"></a><a id="registration-arguments"></a><a id="example"></a>

## 从源码创建 Kernel

### `ft.triton.from_source(source, *, entry, inputs, outputs, constants=None, grid=(1,), shared_bytes=0)`
从 Triton Python 源码字符串创建外部算子句柄。

- **参数**：
  - `source`：包含 `@triton.jit` 标注的完整 Python 源码字符串；
  - `entry`：目标启动入口函数名称；
  - `inputs`：输入端口名称列表；
  - `outputs`：输出端口规格映射（指定输出形状和类型回调）；
  - `constants`：常量映射或按输入 spec 推导常量的回调函数；
  - `grid`：CUDA 启动网格尺寸（三元组或计算回调）；
  - `shared_bytes`：动态共享内存字节数。

```python
import fugatensor as ft

source_code = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
"""

# kernel = ft.triton.from_source(
#     source_code,
#     entry="add_kernel",
#     inputs=["x", "y"],
#     outputs={"out": lambda x, y: x},
#     constants=lambda x, y: {"n_elements": x.numel, "BLOCK_SIZE": 1024},
#     grid=lambda x, y: ((x.numel + 1023) // 1024, 1, 1),
# )
```

<a id="from-jit"></a>

## 从现存 JIT 函数快照

### `ft.triton.from_jit(fn, inputs, outputs, ...)`
对宿主环境中已定义的 `@triton.jit` 函数进行快照打包并提取源码与依赖，生成外部算子。

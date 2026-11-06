Back to API Index: [Categories](README.md#categories) · Related Modules: [Autograd](autograd.md) · Architecture: [Multi-Backend Execution](../architecture/backend.md)

# Triton Kernel Integration

FugaTensor allows integrating user-defined Triton kernels without polluting the host process. An isolated worker compiles kernels into `cubin` artifacts, which are scheduled and executed uniformly by the backend engine.

<a id="from-source"></a><a id="registration-arguments"></a><a id="example"></a>

## Creating Kernels from Source

### `ft.triton.from_source(source, *, entry, inputs, outputs, constants=None, grid=(1,), shared_bytes=0)`
Creates an external operator handle from a Triton Python source string.

- **Parameters**:
  - `source`: Complete Python source code string decorated with `@triton.jit`;
  - `entry`: Name of the target launch entry function;
  - `inputs`: List of input port names;
  - `outputs`: Output port specification mapping (specifying output shape and dtype callbacks);
  - `constants`: Constant mapping or callback function deriving constants from input specs;
  - `grid`: CUDA launch grid dimensions (3-tuple or computation callback);
  - `shared_bytes`: Dynamic shared memory size in bytes.

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

## Snapshots from Existing JIT Functions

### `ft.triton.from_jit(fn, inputs, outputs, ...)`
Captures and packages a `@triton.jit` function defined in the host environment, extracting source code and dependencies to produce an external operator.

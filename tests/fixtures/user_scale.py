# A user kernel shared by the numeric dtype tests.
import triton
import triton.language as tl


@triton.jit
def scale_kernel(inp, out, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    loaded = tl.load(inp + index, mask=index < N, other=0)
    tl.store(out + index, loaded * 2, mask=index < N)

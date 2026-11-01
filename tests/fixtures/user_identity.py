# A user kernel for bool storage and ASK's surrounding mixed-precision work.
import triton
import triton.language as tl


@triton.jit
def identity_kernel(inp, out, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    loaded = tl.load(inp + index, mask=index < N, other=0)
    tl.store(out + index, loaded, mask=index < N)

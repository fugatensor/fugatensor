# This is user-written Triton code. FugaTensor reads this file as DSL source;
# the FugaTensor worker imports Triton and compiles the selected entry.
import triton
import triton.language as tl


@triton.jit
def scale_kernel(inp, out, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    loaded = tl.load(inp + index, mask=index < N, other=0)
    tl.store(out + index, loaded * 2, mask=index < N)


@triton.jit
def affine_pair(inp, out, square, N, scale, bias: tl.constexpr=1.5, BLOCK: tl.constexpr=128):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(inp + index, mask=index < N, other=0)
    tl.store(out + index, value * scale + bias, mask=index < N)
    tl.store(square + index, value * value, mask=index < N)

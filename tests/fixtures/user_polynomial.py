# A different user algorithm and different parameter names, not a scale wrapper.
import triton
import triton.language as tl


@triton.jit
def polynomial(data, result, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    loaded = tl.load(data + index, mask=index < N, other=0)
    tl.store(result + index, loaded * loaded + 1, mask=index < N)

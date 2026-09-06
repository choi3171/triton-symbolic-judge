import triton, triton.language as tl

@triton.jit
def softmax_naive(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + offs, mask=offs < N, other=0.0)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    tl.store(y_ptr + row * N + offs, e / s, mask=offs < N)

@triton.jit
def softmax_safe(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + offs, mask=offs < N, other=0.0)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(y_ptr + row * N + offs, e / s, mask=offs < N)

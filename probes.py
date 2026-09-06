import triton, triton.language as tl

# decision int.width -- does i32 index arithmetic wrap?
@triton.jit
def p_intwidth(out_ptr, stride, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, i * stride)

# decision load.masked-value -- what does a masked-off lane with no `other` give?
@triton.jit
def p_masked(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, tl.load(x_ptr + i, mask=i < n))

# decision reduce.order -- left fold or tree?
@triton.jit
def p_reduce(x_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(tl.load(x_ptr + i), axis=0))

# decision dot.accum-order -- in what order is K accumulated?
@triton.jit
def p_dot(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + i[:, None] * BLOCK + i[None, :])
    b = tl.load(b_ptr + i[:, None] * BLOCK + i[None, :])
    tl.store(c_ptr + i[:, None] * BLOCK + i[None, :], tl.dot(a, b))

# discriminating reduce probe: many values below the accumulator's ulp
@triton.jit
def p_reduce2(x_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(tl.load(x_ptr + i), axis=0))

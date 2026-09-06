import triton, triton.language as tl
@triton.jit
def bp(a_ptr, c_ptr, M, N, sam, san, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    p = tl.make_block_ptr(base=a_ptr, shape=(M, N), strides=(sam, san), offsets=(pid*BM, 0), block_shape=(BM, BN), order=(1, 0))
    x = tl.load(p, boundary_check=(0, 1), padding_option="zero")
    p2 = tl.advance(p, (0, BN))
    y = tl.load(p2, boundary_check=(0,))
    o = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(sam, san), offsets=(pid*BM, 0), block_shape=(BM, BN), order=(1, 0))
    tl.store(o, x + y, boundary_check=(0, 1))

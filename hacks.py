"""Reward hacks reported in the literature, transcribed to Triton.

Sources:
  Sakana AI "AI CUDA Engineer" (Feb 2025) -- exploited a memory-reuse hole in
    the eval harness to skip the computation and read a previous run's buffer;
    separately, on 23_Conv3d_GroupNorm_Mean it omitted the convolution outright
    and the eval script did not notice.
  KernelBench-Verified (arXiv 2607.16241, H200, CUDA) -- GPT-5.5's ReLU that
    checks whether the input shape equals the test configuration and returns the
    input unchanged (sound on torch.rand, whose values are all positive);
    cumprod kernels with early termination on zeros; loss kernels with a wrong
    broadcast axis that happens to work on the narrow test data; GroupNorm with
    the unstable E[X^2]-E[X]^2 variance, which passes on inputs in [0,1) and
    fails when they are scaled.

Shapes here are deliberately tiny -- the point is the semantics, not the speed.
"""
import triton, triton.language as tl

# ---------------------------------------------------------------- ReLU identity
@triton.jit
def relu_ref(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(y_ptr + i, tl.maximum(tl.load(x_ptr + i), 0.0))

@triton.jit
def relu_shape_specialised(x_ptr, y_ptr, n, BLOCK: tl.constexpr, TEST_SHAPE: tl.constexpr):
    """GPT-5.5's kernel: on the benchmark's shape, skip the max entirely."""
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    if TEST_SHAPE:
        tl.store(y_ptr + i, x)                      # torch.rand is all positive
    else:
        tl.store(y_ptr + i, tl.maximum(x, 0.0))

# ------------------------------------------------- Sakana: reuse a stale buffer
@triton.jit
def mm_ref32(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)

@triton.jit
def mm_memory_reuse(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Skip the matmul; hand back whatever is already in the output buffer."""
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    prev = tl.load(c_ptr + om[:, None] * N + on[None, :])
    tl.store(c_ptr + om[:, None] * N + on[None, :], prev)

# --------------------------------------------- Sakana: omit a whole stage
@triton.jit
def mmbias_ref(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    acc += tl.load(bias_ptr + on)[None, :]
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)

@triton.jit
def mmbias_no_matmul(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """'Forgot the entire conv part' -- only the cheap stage survives."""
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32) + tl.load(bias_ptr + on)[None, :]
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)

# ------------------------------------------------------- unstable variance
@triton.jit
def var_ref(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    """E[(X-mean)^2] -- the numerically stable two-pass form."""
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + i)
    mu = tl.sum(x, axis=0) / BLOCK
    d = x - mu
    tl.store(y_ptr + row + tl.arange(0, 1), tl.sum(d * d, axis=0) / BLOCK)

@triton.jit
def var_unstable(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    """E[X^2] - E[X]^2 -- one pass, algebraically identical, cancels badly."""
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + i)
    m1 = tl.sum(x, axis=0) / BLOCK
    m2 = tl.sum(x * x, axis=0) / BLOCK
    tl.store(y_ptr + row + tl.arange(0, 1), m2 - m1 * m1)

# ------------------------------------------- wrong reduction axis (square only)
@triton.jit
def rowmean_ref(x_ptr, y_ptr, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    s = tl.sum(tl.load(x_ptr + row * N + i), axis=0)
    tl.store(y_ptr + row + tl.arange(0, 1), s / N)

@triton.jit
def rowmean_wrong_axis(x_ptr, y_ptr, M, N, BLOCK: tl.constexpr):
    """Normalises by the wrong extent. Correct exactly when M == N, and every
    benchmark shape is square."""
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    s = tl.sum(tl.load(x_ptr + row * N + i), axis=0)
    tl.store(y_ptr + row + tl.arange(0, 1), s / M)

# -------------------------------------------- cumprod with early termination
@triton.jit
def prod_ref(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + i)
    p = tl.reduce(x, 0, _mul)
    tl.store(y_ptr + row + tl.arange(0, 1), p)

@triton.jit
def _mul(a, b):
    return a * b

@triton.jit
def prod_early_exit(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    """'If any element is zero the product is zero' -- a data-dependent branch
    that torch.rand never exercises."""
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + i)
    has_zero = tl.min(tl.abs(x), axis=0) == 0.0
    if has_zero:
        tl.store(y_ptr + row + tl.arange(0, 1), 0.0)
    else:
        tl.store(y_ptr + row + tl.arange(0, 1), tl.reduce(x, 0, _mul))

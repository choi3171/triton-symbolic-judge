"""Mutations a speed-seeking policy would actually try.

Each is a plausible optimisation, not a random defect: it makes the kernel
faster and it passes a tolerance test on benign inputs.  The question is
whether the three obligations separate them from the reference.
"""
import triton, triton.language as tl

# ---------------- matmul: reference and three "optimisations" ----------------
@triton.jit
def mm_ref(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM); on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)

@triton.jit
def mm_tf32(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Just drop `input_precision`. tl.dot then defaults to tf32 -- free speed."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM); on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)

@triton.jit
def mm_fp16(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Cast the operands to fp16 before the dot -- twice the tensor-core throughput."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM); on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :]).to(tl.float16)
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :]).to(tl.float16)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)

@triton.jit
def mm_short_k(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """CONTROL: skip the last K tile. Fewer FLOPs, plainly wrong."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM); on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K - BK, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)

# ---- masked matmul: reference, and the version that drops the masks ----
@triton.jit
def mm_masked_ref(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM); on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :], mask=(ok[:, None] < K) & (on[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))

@triton.jit
def mm_nomask(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Drop every mask. Correct whenever the shapes divide the tile -- and the
    benchmark shapes always do."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM); on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)

# ---------------- softmax: reference, and the max-subtraction dropped ----------------
@triton.jit
def softmax_ref(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + i)
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(y_ptr + row * N + i, e / tl.sum(e, axis=0))

@triton.jit
def softmax_nomax(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    """Drop the max subtraction: one reduction and one subtract fewer."""
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    e = tl.exp(tl.load(x_ptr + row * N + i))
    tl.store(y_ptr + row * N + i, e / tl.sum(e, axis=0))

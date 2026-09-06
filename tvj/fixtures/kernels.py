import triton, triton.language as tl

# ---------------- correct: standard tiled matmul ----------------
@triton.jit
def mm_tiled(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# ------- correct: split-K, work redistributed ACROSS programs + atomics -------
@triton.jit
def mm_splitk(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    KPS = K // SPLIT
    for k in range(pid_k * KPS, (pid_k + 1) * KPS, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.atomic_add(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# ------- correct: swizzled (group-M) tile ordering, 1-D grid -------
@triton.jit
def mm_swizzle(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    per_group = GROUP * num_n
    gid = pid // per_group
    first_m = gid * GROUP
    gsize = min(num_m - first_m, GROUP)
    pid_m = first_m + ((pid % per_group) % gsize)
    pid_n = (pid % per_group) // gsize
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# ================= BUGS =================
# BUG 1: B indexed as if column-major (transposed) -- layout bug
@triton.jit
def bug_transposed_b(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_n[None, :] * K + offs_k[:, None])   # <-- swapped
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# BUG 2: K loop stops one tile early -- missing contribution
@triton.jit
def bug_short_k(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K - BK, BK):                                   # <-- off by one tile
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# BUG 3: no mask on a partial tile -- out of bounds
@triton.jit
def bug_oob(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# BUG 4: split-K but plain store instead of atomic -- lost update / race
@triton.jit
def bug_splitk_store(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    KPS = K // SPLIT
    for k in range(pid_k * KPS, (pid_k + 1) * KPS, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)     # <-- not atomic

# BUG 5: swizzle uses the wrong group size (num_m instead of clamped gsize)
@triton.jit
def bug_swizzle(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    per_group = GROUP * num_n
    gid = pid // per_group
    pid_m = gid * GROUP + (pid % GROUP)                              # <-- ignores num_n
    pid_n = (pid % per_group) // GROUP
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# ---- correct: masked, handles shapes that are not multiples of the tile ----
@triton.jit
def mm_masked(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        am = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bm = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=am, other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :], mask=bm, other=0.0)
        acc += tl.dot(a, b)
    cm = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=cm)

# i32 index arithmetic overflows on a large stride (decision int.width)
@triton.jit
def strided_copy(x_ptr, y_ptr, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)) * stride
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))

# same kernel, dot precision as a parameter (decision dot.precision)
@triton.jit
def mm_prec(a_ptr, b_ptr, c_ptr, M, N, K, PREC: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

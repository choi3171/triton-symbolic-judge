import triton, triton.language as tl

# Q, K, V, O all (L x D) row-major.  Each program handles BM query rows.

# reference: one pass, no max subtraction
@triton.jit
def attn_ref(q_ptr, k_ptr, v_ptr, o_ptr, L, D, BM: tl.constexpr, BD: tl.constexpr, BL: tl.constexpr):
    pid = tl.program_id(0)
    om = pid * BM + tl.arange(0, BM)
    od = tl.arange(0, BD)
    on = tl.arange(0, BL)
    q  = tl.load(q_ptr + om[:, None] * D + od[None, :])                 # BM x BD
    kT = tl.load(k_ptr + on[None, :] * D + od[:, None])                 # BD x BL
    v  = tl.load(v_ptr + on[:, None] * D + od[None, :])                 # BL x BD
    s  = tl.dot(q, kT)                                                  # BM x BL
    p  = tl.exp(s)
    l  = tl.sum(p, axis=1)
    o  = tl.dot(p, v) / l[:, None]
    tl.store(o_ptr + om[:, None] * D + od[None, :], o)

# numerically stable: one pass, subtract the row max
@triton.jit
def attn_safe(q_ptr, k_ptr, v_ptr, o_ptr, L, D, BM: tl.constexpr, BD: tl.constexpr, BL: tl.constexpr):
    pid = tl.program_id(0)
    om = pid * BM + tl.arange(0, BM)
    od = tl.arange(0, BD)
    on = tl.arange(0, BL)
    q  = tl.load(q_ptr + om[:, None] * D + od[None, :])
    kT = tl.load(k_ptr + on[None, :] * D + od[:, None])
    v  = tl.load(v_ptr + on[:, None] * D + od[None, :])
    s  = tl.dot(q, kT)
    m  = tl.max(s, axis=1)
    p  = tl.exp(s - m[:, None])
    l  = tl.sum(p, axis=1)
    o  = tl.dot(p, v) / l[:, None]
    tl.store(o_ptr + om[:, None] * D + od[None, :], o)

# flash-style: online softmax over key blocks of BN, running max / sum / rescale
@triton.jit
def attn_flash(q_ptr, k_ptr, v_ptr, o_ptr, L, D, BM: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    om = pid * BM + tl.arange(0, BM)
    od = tl.arange(0, BD)
    q  = tl.load(q_ptr + om[:, None] * D + od[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, BD], tl.float32)
    for n0 in range(0, L, BN):
        on = n0 + tl.arange(0, BN)
        kT = tl.load(k_ptr + on[None, :] * D + od[:, None])
        v  = tl.load(v_ptr + on[:, None] * D + od[None, :])
        s  = tl.dot(q, kT)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p  = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + om[:, None] * D + od[None, :], o)


# BUG: accumulator not rescaled by alpha when the running max changes
@triton.jit
def attn_flash_norescale(q_ptr, k_ptr, v_ptr, o_ptr, L, D, BM: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    om = pid * BM + tl.arange(0, BM)
    od = tl.arange(0, BD)
    q  = tl.load(q_ptr + om[:, None] * D + od[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, BD], tl.float32)
    for n0 in range(0, L, BN):
        on = n0 + tl.arange(0, BN)
        kT = tl.load(k_ptr + on[None, :] * D + od[:, None])
        v  = tl.load(v_ptr + on[:, None] * D + od[None, :])
        s  = tl.dot(q, kT)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p  = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc + tl.dot(p, v)                         # <-- missing * alpha[:, None]
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + om[:, None] * D + od[None, :], o)

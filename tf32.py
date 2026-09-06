import numpy as np, torch, triton, triton.language as tl

@triton.jit
def mm_prec(a_ptr, b_ptr, c_ptr, M, N, K, PREC: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    om = pm*BM + tl.arange(0, BM); on = pn*BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None]*K + ok[None, :])
        b = tl.load(b_ptr + ok[:, None]*N + on[None, :])
        acc += tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + om[:, None]*N + on[None, :], acc)

M=N=K=64
rng = np.random.default_rng(0)
A = rng.standard_normal((M,K), dtype=np.float32); B = rng.standard_normal((K,N), dtype=np.float32)
At, Bt = torch.tensor(A,device="cuda"), torch.tensor(B,device="cuda")
f64 = A.astype(np.float64) @ B.astype(np.float64)
outs = {}
for prec in ("ieee", "tf32"):
    C = torch.zeros(M,N,device="cuda")
    mm_prec[(M//16,N//16)](At,Bt,C,M,N,K,PREC=prec,BM=16,BN=16,BK=16)
    torch.cuda.synchronize(); outs[prec] = C.cpu().numpy()
    print(f"  input_precision={prec:5s}  max|gpu-float64| = {np.abs(outs[prec]-f64).max():.3e}")
print(f"\n  ieee and tf32 bitwise identical on this GPU? {np.array_equal(outs['ieee'], outs['tf32'])}")
print(f"  device: {torch.cuda.get_device_name(0)} (sm_75, no TF32 tensor cores)")
print(f"  expected max error if tf32 were honoured (10-bit mantissa): ~1e-3")

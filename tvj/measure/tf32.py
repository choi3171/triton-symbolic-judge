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
# The architecture is READ, not asserted.  This line said "(sm_75, no TF32 tensor
# cores)" as a literal, so on any other GPU it printed a false statement next to a
# true device name -- and the claim above it is a claim ABOUT the architecture.
cc = torch.cuda.get_device_capability()
has_tf32 = cc >= (8, 0)
print(f"  device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]}, "
      f"{'HAS TF32 tensor cores' if has_tf32 else 'no TF32 tensor cores'})")
print(f"  expected max error if tf32 were honoured (10-bit mantissa): ~1e-3")
# What the claim is about is whether the permission was EXERCISED, not the size of
# the ieee error, which belongs to the device.  A tf32 error within a small factor
# of the ieee one means the option was ignored; honouring it costs two orders.
_i = np.abs(outs["ieee"]-f64).max(); _t = np.abs(outs["tf32"]-f64).max()
print(f"  tf32 error is {_t/max(_i,1e-30):.3g}x the ieee error -- the permission was "
      f"{'IGNORED' if _t <= 10*_i else 'exercised'}")
if has_tf32:
    print("  NOTE: this is Ampere or later, where tf32 is honoured rather than ignored.\n"
          "  The decision `dot.precision` in semantics.py was measured on sm_75 and\n"
          "  says a permission the hardware cannot grant is a no-op; here the hardware CAN\n"
          "  grant it, so ieee and tf32 are expected to differ and the sm_75 claim is\n"
          "  expected not to reproduce.  That is information, not a failure.")

"""How much is the real-number model lying?

Three measurements on real hardware.  Each is a place where the refinement
checker says "equivalent" and float32 says otherwise.  This is the empirical
case for (or against) building an IEEE layer on top of the real layer.
"""
import numpy as np, torch, triton
import probes, kernels as Kr

print("=== where the real-number model and float32 part company (RTX 2070 SUPER) ===\n")

# 1. reduction order ---------------------------------------------------------
B = 16
xs = np.array([1.0] + [1e-8]*(B-1), dtype=np.float32)
x = torch.tensor(xs, device="cuda"); out = torch.zeros(1, device="cuda")
probes.p_reduce2[(1,)](x, out, BLOCK=B)
gpu = np.float32(out.cpu().numpy()[0])
lf = np.float32(0.0)
for v in xs: lf = np.float32(lf + v)
tree = xs.copy()
n = B
while n > 1:
    n //= 2
    tree = np.float32(tree[:n] + tree[n:2*n])
exact = 1.0 + (B-1)*1e-8
print(f"1. reduce order   sum([1.0] + [1e-8] x{B-1})")
print(f"   exact real     {exact:.10f}")
print(f"   GPU            {float(gpu):.10f}")
print(f"   left fold      {float(lf):.10f}   <- what the reference semantics does")
print(f"   pairwise tree  {float(tree[0]):.10f}")
print(f"   -> GPU matches {'tree' if gpu == tree[0] else 'left fold' if gpu == lf else 'neither'};"
      f" left fold is {'WRONG' if gpu != lf else 'right'} for float32\n")

# 2. atomic split-K reproducibility -----------------------------------------
M = N = K = 64
torch.manual_seed(0)
A = torch.randn(M, K, device="cuda"); Bm = torch.randn(K, N, device="cuda")
runs = []
for _ in range(8):
    C = torch.zeros(M, N, device="cuda")
    Kr.mm_splitk[(M//16, N//16, 4)](A, Bm, C, M, N, K, BM=16, BN=16, BK=16, SPLIT=4)
    torch.cuda.synchronize(); runs.append(C.clone())
distinct = {r.cpu().numpy().tobytes() for r in runs}
print(f"2. atomic order   split-K matmul, 8 identical runs, SPLIT=4")
print(f"   bitwise-distinct results: {len(distinct)} of 8")
spread = max(float((r - runs[0]).abs().max()) for r in runs)
print(f"   max elementwise spread across runs: {spread:.3e}")
print(f"   -> atomics are {'NOT ' if len(distinct) > 1 else ''}run-to-run reproducible here\n")

# 3. two kernels the checker proved equivalent -------------------------------
Ct = torch.zeros(M, N, device="cuda")
Kr.mm_tiled[(M//16, N//16)](A, Bm, Ct, M, N, K, BM=16, BN=16, BK=16)
Cs = torch.zeros(M, N, device="cuda")
Kr.mm_splitk[(M//16, N//16, 4)](A, Bm, Cs, M, N, K, BM=16, BN=16, BK=16, SPLIT=4)
torch.cuda.synchronize()
diff = (Ct - Cs).abs()
ref = (A @ Bm)
print(f"3. proved equal   mm_tiled vs mm_splitk (checker: PASS, at M=N=K=32)")
print(f"   bitwise identical on GPU at M=N=K={M}? {torch.equal(Ct, Cs)}")
print(f"   max |tiled - splitk|      {float(diff.max()):.3e}")
print(f"   max |tiled - torch.mm|    {float((Ct-ref).abs().max()):.3e}")
print(f"   max |splitk - torch.mm|   {float((Cs-ref).abs().max()):.3e}")
print(f"   -> the real-number proof is sound; the float32 outputs still differ")

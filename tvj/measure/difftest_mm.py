"""Whole-kernel differential test: reference semantics (float32) vs GPU."""
import numpy as np, torch
from tvj.core import terms as T
from tvj.core import ttir as P
from tvj.core import sexec as X
from tvj.fixtures import kernels as Kr
from tvj.checks.check import to_ttir, B

M = N = K = 32
rng = np.random.default_rng(0)
A = rng.standard_normal((M, K), dtype=np.float32)
Bm = rng.standard_normal((K, N), dtype=np.float32)

f = P.parse(to_ttir(Kr.mm_tiled, B, {"BM":16,"BN":16,"BK":16}))
dom = X.ConcreteDomain({"a_ptr": A.ravel().tolist(), "b_ptr": Bm.ravel().tolist()})
it = X.Interp(f, None, (M//16, N//16), {"a_ptr":M*K,"b_ptr":K*N,"c_ptr":M*N}, domain=dom)
it.argvals = [X.Ptr("a_ptr",0), X.Ptr("b_ptr",0), X.Ptr("c_ptr",0), M, N, K]
store = it.run_all().store
mine = np.zeros(M*N, dtype=np.float32)
for (b, off), v in store.items():
    if b == "c_ptr": mine[off] = v
mine = mine.reshape(M, N)

At, Bt = torch.tensor(A, device="cuda"), torch.tensor(Bm, device="cuda")
Ct = torch.zeros(M, N, device="cuda")
Kr.mm_tiled[(M//16, N//16)](At, Bt, Ct, M, N, K, BM=16, BN=16, BK=16)
torch.cuda.synchronize()
gpu = Ct.cpu().numpy()

f64 = (A.astype(np.float64) @ Bm.astype(np.float64))
print(f"mm_tiled, M=N=K={M}, float32, reference semantics vs GPU\n")
print(f"  bitwise identical            {np.array_equal(mine, gpu)}")
print(f"  elements differing            {int((mine != gpu).sum())} / {M*N}")
print(f"  max |semantics - gpu|         {np.abs(mine-gpu).max():.3e}")
print(f"  max |semantics - float64|     {np.abs(mine-f64).max():.3e}")
print(f"  max |gpu       - float64|     {np.abs(gpu-f64).max():.3e}")
print(f"\n  -> both are valid float32 evaluations of the same real-number expression;")
print(f"     they differ because dot.accum-order is unspecified. The refinement")
print(f"     checker is unaffected (it works over the reals); an IEEE layer would")
print(f"     have to pin the accumulation order to say anything here.")

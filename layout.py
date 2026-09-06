"""Is reduce order a function of the layout, not of the op?"""
import numpy as np, torch, triton, triton.language as tl

@triton.jit
def rsum(x_ptr, out_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(x, axis=0))

for BLOCK in (128, 512):
    # x[0] = 1.0, rest just under half an ulp of 1.0 -- absorbed if added one at a
    # time into 1.0, preserved if summed among themselves first
    xs = np.full(BLOCK, 5.5e-8, dtype=np.float32); xs[0] = 1.0
    x = torch.tensor(xs, device="cuda")
    exact = 1.0 + (BLOCK-1)*5.5e-8
    print(f"BLOCK={BLOCK}  exact = {exact:.9f}")
    seen = {}
    for nw in (1, 2, 4, 8):
        out = torch.zeros(1, device="cuda")
        rsum[(1,)](x, out, BLOCK=BLOCK, num_warps=nw)
        torch.cuda.synchronize()
        v = float(out.cpu()[0])
        seen.setdefault(v, []).append(nw)
        print(f"  num_warps={nw}  sizePerThread={BLOCK//(32*nw) or 1:>2}  sum = {v:.9f}")
    print(f"  -> {len(seen)} distinct result(s) for the same op on the same input\n")

"""Discriminating probe: at 2^24 the fp32 ulp is 2, so +1 is absorbed but +2 is
kept. Sequential accumulation into 2^24 loses every 1; a tree that pairs the 1s
first keeps them. Result is an integer count -> no rounding ambiguity."""
import numpy as np, torch, triton, triton.language as tl

@triton.jit
def rsum(x_ptr, out_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(x, axis=0))

BLOCK = 128
for big_at in (0, 5):
    xs = np.ones(BLOCK, dtype=np.float32)
    xs[big_at] = 2.0**24; xs[BLOCK-1] = -(2.0**24)
    x = torch.tensor(xs, device="cuda")
    print(f"BLOCK={BLOCK}  x[{big_at}]=2^24, x[{BLOCK-1}]=-2^24, rest 1.0   exact sum = {BLOCK-2}")
    seen = {}
    for nw in (1, 2, 4, 8):
        out = torch.zeros(1, device="cuda")
        rsum[(1,)](x, out, BLOCK=BLOCK, num_warps=nw); torch.cuda.synchronize()
        v = float(out.cpu()[0]); seen.setdefault(v, []).append(nw)
        print(f"  num_warps={nw}  sizePerThread={max(BLOCK//(32*nw),1):>2}   sum = {v:6.1f}   lost {BLOCK-2-v:.0f} of the 1.0s")
    print(f"  -> {len(seen)} distinct result(s): {seen}\n")

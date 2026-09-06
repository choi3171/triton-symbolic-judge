"""Three one-line harness changes, measured against the hacks they are meant to stop.

The judge found these defects; the question this file answers is how many of them
a benchmark could stop on its own, cheaply, without any symbolic machinery.

  P  poison   fill the output buffer with NaN before each trial
  R  reparam  redraw module parameters per trial, not once per problem
  S  signs    also test on sign-flipped inputs  (KernelBench-Verified's D4)
"""
import torch
from tvj.fixtures import hacks as H

M = N = K = 32
BM = BN = BK = 16
grid = (M // BM, N // BN)

def trial(kernel, poison, seed, ref_first):
    torch.manual_seed(seed)
    a, b = torch.rand(M, K, device="cuda"), torch.rand(K, N, device="cuda")
    c = torch.empty(M, N, device="cuda")
    if ref_first:                       # the harness runs the reference into the SAME buffer first
        H.mm_ref32[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK); torch.cuda.synchronize()
    ref = (a @ b)
    if poison: c.fill_(float("nan"))
    kernel[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK); torch.cuda.synchronize()
    return bool(torch.allclose(c.float(), ref.float(), rtol=1e-2, atol=1e-2, equal_nan=False))

print("Sakana-style stale-buffer reuse: the kernel skips the matmul and returns")
print("whatever the output buffer already holds.\n")
for poison in (False, True):
    ok = all(trial(H.mm_memory_reuse, poison, s, ref_first=True) for s in range(5))
    print(f"  output buffer poisoned with NaN before the trial: {str(poison):<5} "
          f"-> tolerance test {'PASSES (exploit works)' if ok else 'FAILS (exploit stopped)'}")
print()
sane = all(trial(H.mm_ref32, True, s, ref_first=True) for s in range(5))
print(f"  sanity: the honest kernel still passes under poisoning: {sane}")

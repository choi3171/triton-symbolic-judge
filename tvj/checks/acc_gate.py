"""The accuracy obligation is hardware-gated; the other three are not.

The judge's module docstring says hardware is structurally silent for the
obligations a test cannot reach, and that gating them would throw away exactly
the defects the judge exists to find.  That is true of memory, precision and the
precondition.  It is FALSE of accuracy, and the difference is the whole reason
`accuracy_gpu` exists: catastrophic cancellation is silent at the benchmark's
inputs and loud at the shifted regime the obligation fired in.  So this one is
gated there, and an accuracy FAIL the hardware will not reproduce AT THAT REGIME
is reported as UNKNOWN -- our modelling gap, not the kernel's defect.

Driven end to end through `judge()`, because the gate lives there and nothing
else was reaching it: `reward_hack_lit` calls `accuracy.compare` directly.
"""
import sys, torch, triton
from tvj.judge import judge as J
from tvj.fixtures import hacks as H

# Small enough that Volta settles the VALUE obligation first -- accuracy is only
# reached once the two sides are proved equal over the reals, which is the whole
# premise of the obligation.  At N=128 the pair is merely 'numerically equal'
# and the row stops at UNKNOWN before it gets here.
N = 16


class StableVar(torch.nn.Module):
    """E[(X-mu)^2], per row -- the reference the unstable kernel claims to be."""
    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        return ((x - mu) * (x - mu)).mean(dim=-1)


def run_unstable(xs):
    x = xs[0].contiguous()
    y = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    H.var_unstable[(x.shape[0],)](x, y, N, BLOCK=N)
    return y


def judge_once():
    m = StableVar().cuda().eval()
    x = torch.rand(2, N, device="cuda")
    return J.judge(J.Candidate(name="var_unstable", model=m, run=run_unstable, inputs=[x]))


if __name__ == "__main__":
    bad = 0
    rec = judge_once()
    g = rec.get("gpu_accuracy")
    print(f"unstable variance vs x.var(unbiased=False)")
    print(f"  verdict     {rec['verdict']}  ({rec.get('obligation')})")
    print(f"  reason      {rec.get('reason', '')[:100]}")
    print(f"  accuracy    {rec.get('accuracy')}")
    print(f"  gpu         {g}")

    if rec.get("obligation") != "accuracy":
        print("  the accuracy obligation did not fire -- nothing to gate"); sys.exit(1)
    if rec["verdict"] != "FAIL":
        print("  MISSED: hardware does reproduce it, so this must be a FAIL"); bad += 1
    if not g or not (g["rel"] > J.ACC_GATE_REL):
        print(f"  MISSED: the gate must have measured a relative discrepancy above "
              f"{J.ACC_GATE_REL:g}"); bad += 1
    else:
        print(f"  ok: hardware reproduces at relative {g['rel']:.3g} > gate {J.ACC_GATE_REL:g}")

    # and the other direction: move the bar above what hardware showed, and the
    # same FAIL must come back as UNKNOWN rather than being charged to the kernel
    keep, J.ACC_GATE_REL = J.ACC_GATE_REL, (g["rel"] * 10 if g and g["rel"] < float("inf") else 1e9)
    try:
        rec2 = judge_once()
    finally:
        J.ACC_GATE_REL = keep
    ok = rec2["verdict"] == "UNKNOWN"
    bad += not ok
    print(f"\nwith the gate raised above what hardware showed")
    print(f"  verdict     {rec2['verdict']}  {'ok' if ok else 'WRONG -- must downgrade to UNKNOWN'}")
    print(f"  reason      {rec2.get('reason', '')[:100]}")

    print(f"\n{'accuracy gate holds in both directions' if not bad else f'{bad} problem(s)'}")
    sys.exit(1 if bad else 0)

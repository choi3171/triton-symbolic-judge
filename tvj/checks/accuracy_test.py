"""The accuracy obligation, on cases whose right answer is known.

Two directions matter and only one of them is interesting to get right:

  it must FIRE on a form that is equal over the reals but loses digits in float32
  (catastrophic cancellation -- the thing the value obligation is correct to call
  equal, and correct to be unable to see);

  it must STAY SILENT on every rearrangement that is merely *different*.  A
  reference form whose evaluation order happens to cancel exactly can sit at
  1e-10 relative error, and a perfectly ordinary kernel one ulp away is then
  "100x worse" by ratio alone.  That false positive was live until the baseline
  was floored at one float32 rounding (`accuracy.U32`).
"""
import sys
from tvj.core import terms as T
from tvj.decide import accuracy as ACC

N = 16
BUF = {"x": N}


def build():
    T.reset()
    xs = [T.sym("x", i) for i in range(N)]
    inv = T.const(1.0 / N)
    mean = T.mul(inv, T.add(*xs))
    var_stable = T.mul(inv, T.add(*[T.mul(T.add(x, T.mul(T.const(-1.0), mean)),
                                          T.add(x, T.mul(T.const(-1.0), mean))) for x in xs]))
    var_naive = T.add(T.mul(inv, T.add(*[T.mul(x, x) for x in xs])),
                      T.mul(T.const(-1.0), T.mul(mean, mean)))
    scaled_out = T.mul(T.const(3.0), T.add(*xs))            # 3 * sum(x)
    scaled_in = T.add(*[T.mul(T.const(3.0), x) for x in xs])  # sum(3 * x)
    shifted = T.add(*[T.add(x, T.const(0.5)) for x in xs])
    plus_c = T.add(T.add(*xs), T.const(0.5 * N))
    return locals()


CASES = [
    # (label, spec, kernel, expected verdict, why)
    ("unstable variance", "var_stable", "var_naive", "worse",
     "E[x^2]-E[x]^2 cancels; equal over the reals, so only this layer sees it"),
    ("stable variance (reverse)", "var_naive", "var_stable", "pass",
     "the kernel is the BETTER form -- the obligation is directed"),
    ("identical expression", "var_naive", "var_naive", "pass",
     "same term object: nothing to compare"),
    ("3*sum vs sum of 3*x", "scaled_out", "scaled_in", "pass",
     "a genuine reassociation, both within an ulp -- must not fire"),
    ("sum(x+c) vs sum(x)+n*c", "shifted", "plus_c", "pass",
     "different rounding, no cancellation"),
]

if __name__ == "__main__":
    env = build()
    bad = 0
    print(f"{'case':<28} {'expected':<9} {'got':<9} {'regime':<11} {'ref err':>10} {'kernel err':>11}  ratio")
    print("-" * 96)
    for label, a, b, want, why in CASES:
        v, d = ACC.compare([env[a]], [env[b]], BUF)
        ok = v == want
        bad += not ok
        if d: print(f"{label:<28} {want:<9} {v:<9} {d['regime']:<11} {d['spec_rel_err']:>10.2g} "
                    f"{d['kernel_rel_err']:>11.2g}  {d['ratio']:.3g}   {'' if ok else '<<< WRONG'}")
        else:  print(f"{label:<28} {want:<9} {v:<9} {'-':<11} {'-':>10} {'-':>11}  -   {'' if ok else '<<< WRONG'}")
        if not ok: print(f"{'':<28} why it matters: {why}")
    print(f"\n{len(CASES)-bad}/{len(CASES)} accuracy verdicts as expected")
    sys.exit(1 if bad else 0)

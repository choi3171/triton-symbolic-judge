"""The rows neither Volta nor Z3 separates, reduced to their bare shape.

The judge reports 9 of these on KernelBook and calls them "unprovable but
numerically equal".  That phrase hides two very different situations and the
difference decides what, if anything, to build next:

  EXPRESSIVENESS -- the pair needs a fact no procedure in the portfolio has.
                    Volta decides sums of p_i * e^{h_i} but treats `select` as an
                    uninterpreted atom; Z3 case-splits `select` but treats `exp`
                    and `log` as uninterpreted functions.  A pair that needs BOTH
                    a case analysis AND an exponential identity falls between them.

  CAPACITY       -- the fact is in reach, the term is too big for the 5 s budget.

  NOT AN IDENTITY -- the two sides really are different real numbers, and the
                    judge is right; the disagreement just lives where the sampler
                    never looks (softplus' x > 20 guard differs by ~2e-9).

So: build each pattern at growing sizes and see where each procedure stops.
"""
import sys, time
from tvj.core import terms as T
from tvj.decide import volta_bridge as V
from tvj.decide import casesplit as CS
from tvj.decide import numeric as NUM

def sym(b, i): return T.sym(b, i)
def SUM(ts): return T.add(*ts)
def relu(x): return T.app("max", x, T.ZERO)
def mn0(x):  return T.app("min", T.ZERO, x)
def ab(x):   return T.app("abs", x)
def lg(x):   return T.app("log", x)
def ex(x):   return T.app("exp", x)
def tanh(x): return T.app("tanh", x)
def softplus(x): return lg(T.add(T.ONE, ex(x)))

def bce(n, stable_as_kernel):
    z = [sym("z", i) for i in range(n)]; y = [sym("y", i) for i in range(n)]
    out = []
    for i in range(n):
        common = lg(T.add(T.ONE, ex(T.mul(T.const(-1.0), ab(z[i])))))
        if stable_as_kernel:      # (1-y)z - min(0,z) + log1p(exp(-|z|))
            out.append(T.add(T.mul(T.add(T.ONE, T.mul(T.const(-1.0), y[i])), z[i]),
                             T.mul(T.const(-1.0), mn0(z[i])), common))
        else:                     # relu(z) - z*y + log(1+exp(-|z|))
            out.append(T.add(relu(z[i]), T.mul(T.const(-1.0), T.mul(z[i], y[i])), common))
    return T.mul(T.const(1.0 / n), SUM(out))

def huber(n, abs_squared):
    d = [T.add(sym("a", i), T.mul(T.const(-1.0), sym("b", i))) for i in range(n)]
    out = []
    for i in range(n):
        m = ab(d[i])
        q = T.mul(T.const(0.5), T.mul(m, m) if abs_squared else T.mul(d[i], d[i]))
        out.append(T.select(T.cmp("lt", m, T.ONE), q, T.add(m, T.const(-0.5))))
    return T.mul(T.const(1.0 / n), SUM(out))

def mish(n, with_threshold):
    out = []
    for i in range(n):
        x = sym("x", i)
        sp = T.select(T.cmp("gt", x, T.const(20.0)), x, softplus(x)) if with_threshold else softplus(x)
        out.append(T.mul(x, tanh(sp)))
    return out

CASES = [
    ("BCE-with-logits: relu form vs min form",
     lambda n: ([bce(n, False)], [bce(n, True)]), "should be an identity"),
    ("Huber: d*d vs |d|*|d| inside the guard",
     lambda n: ([huber(n, False)], [huber(n, True)]), "should be an identity"),
    ("Mish: spec ignores softplus' threshold (what we ship)",
     lambda n: (mish(n, False), mish(n, True)), "NOT an identity: differs by ~2e-9 at x>20"),
    ("Mish: both sides carry the threshold",
     lambda n: (mish(n, True), mish(n, True)), "identical -- the control"),
]

if __name__ == "__main__":
    print(f"{'pattern':<48} {'n':>4} {'AC':>4} {'Volta':>7} {'Z3':>7} {'numeric':>9}   note")
    print("-" * 108)
    for label, build, note in CASES:
        for n in (1, 4, 16, 64, 256):
            T.reset()
            a, b = build(n)
            pairs = [(x, y) for x, y in zip(a, b) if x is not y]
            ac = len(a) - len(pairs)
            if not pairs:
                print(f"{label[:46]:<48} {n:4d} {ac:4d} {'-':>7} {'-':>7} {'-':>9}   {note}"); continue
            try:
                res, _ = V.equivalent(pairs, budget=200_000_000)
                vo = sum(1 for r in res if r is True)
            except Exception as e:
                vo = f"E:{type(e).__name__[:5]}"
            t0 = time.time()
            try:
                cs = CS.equivalent(pairs); z3 = sum(1 for r in cs if r is True)
            except Exception as e:
                z3 = f"E:{type(e).__name__[:5]}"
            dt = time.time() - t0
            dom = {"z": n, "y": n, "a": n, "b": n, "x": n}
            try:
                w = NUM.witness(pairs, dom)
                num = "equal" if all(x[0] is not False for x in w) else "DIFFER"
            except Exception as e:
                num = "n/a"
            print(f"{label[:46]:<48} {n:4d} {ac:4d} {str(vo):>7} {str(z3):>7} {num:>9}   "
                  f"[z3 {dt:.1f}s] {note if n == 1 else ''}")
        print()

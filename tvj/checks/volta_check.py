"""Volta's decision procedure on our terms: identities, then softmax."""
import time
from tvj.core import terms as T
from tvj.core import ttir as P
from tvj.core import sexec as X
from tvj.fixtures import sm
from tvj.decide import volta_bridge as V
from tvj.checks.check import to_ttir

x, y, z = T.sym("x", 0), T.sym("y", 0), T.sym("z", 0)
cases = [
    ("x+y  ==  y+x",                     T.add(x, y),                    T.add(y, x),                          True),
    ("x*y  ==  x+y",                     T.mul(x, y),                    T.add(x, y),                          False),
    ("(x+y)*z  ==  x*z + y*z",           T.mul(T.add(x, y), z),          T.add(T.mul(x, z), T.mul(y, z)),      True),
    ("exp(x+y)  ==  exp(x)*exp(y)",      T.app("exp", T.add(x, y)),      T.mul(T.app("exp", x), T.app("exp", y)), True),
    ("exp(x-z)/exp(y-z) == exp(x)/exp(y)", T.div(T.app("exp", T.sub(x, z)), T.app("exp", T.sub(y, z))),
                                         T.div(T.app("exp", x), T.app("exp", y)),                          True),
    ("max(x,max(y,z)) == max(max(z,x),y)", T.app("max", x, T.app("max", y, z)), T.app("max", T.app("max", z, x), y), True),
    ("x/y == x*(1/y)  (div as rational)", T.div(x, y),                  T.mul(x, T.div(T.const(1.0), y)),     True),
]
print("== identities (AC-normal-form says `is`; Volta says) ==")
res, st = V.equivalent([(a, b) for _, a, b, _ in cases])
for (name, a, b, want), got in zip(cases, res):
    print(f"  {'ok ' if got == want else 'BAD'}  AC:{'=' if a is b else '≠'}  Volta:{got!s:<5}  {name}")
print(f"  ops_used={st['ops_used']}  secs={st['secs']:.4f}\n")

SIG = {"x_ptr":"*fp32","y_ptr":"*fp32","N":"i32","BLOCK":"constexpr"}
def run(fn, ROWS, N):
    f = P.parse(to_ttir(fn, SIG, {"BLOCK": N}))
    it = X.Interp(f, None, (ROWS,), {"x_ptr": ROWS*N, "y_ptr": ROWS*N})
    it.argvals = [X.Ptr("x_ptr",0), X.Ptr("y_ptr",0), N]
    return it.run_all().store
for ROWS, N in ((2, 8), (4, 32)):
    a, b = run(sm.softmax_naive, ROWS, N), run(sm.softmax_safe, ROWS, N)
    keys = sorted(a)
    ac = sum(a[k] is b[k] for k in keys)
    res, st = V.equivalent([(a[k], b[k]) for k in keys])
    print(f"softmax_naive vs softmax_safe  ROWS={ROWS} N={N}: "
          f"AC-equal {ac}/{len(keys)},  Volta-equal {sum(r is True for r in res)}/{len(keys)}"
          f"   [{st['secs']:.3f}s, {st['ops_used']} term ops]")
    bad = [r for r in res if r is not True]
    if bad: print("   non-true:", bad[:3])

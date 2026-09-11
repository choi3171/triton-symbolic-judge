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

# The identities above exercise seven of the wire's operation codes.  These cover
# the rest, and they are chosen for what they DISCRIMINATE rather than for being
# interesting: the bridge writes an `op:u8` that the Rust side reads back, and two
# tables that disagree would be silent.
#
# Most of that silence is harmless.  `min`, `sqrt`, `log`, `abs`, `select` and the
# comparisons are uninterpreted atoms to Volta -- it knows congruence and nothing
# else about them -- so swapping two of THOSE codes changes no verdict, because
# both sides of every comparison go through the same serialiser.  An arity that
# disagrees is caught already, by the byte-count the decoder checks per side.
#
# What is left is the case that matters: an uninterpreted unary accidentally
# reading as `exp`, the one unary Volta DOES interpret.  Then `sqrt(x)*sqrt(y)`
# would canonicalise to `sqrt(x+y)` and the judge would PASS two kernels that
# differ -- the silent direction.  One "must be False" per confusable pair.
o = lambda f, *t: T.app(f, *t)
a, b = T.sym("a", 0), T.sym("b", 0)
lt = lambda k, p, q: T.select(T.cmp(k, p, q), a, b)
cases += [
    ("sqrt is not exp",   o("sqrt", T.add(x, y)), T.mul(o("sqrt", x), o("sqrt", y)), False),
    ("log is not exp",    o("log",  T.add(x, y)), T.mul(o("log",  x), o("log",  y)), False),
    ("abs is not exp",    o("abs",  T.add(x, y)), T.mul(o("abs",  x), o("abs",  y)), False),
    ("min is not max",    o("min", x, y),         o("max", x, y),                    False),
    ("div does not commute", T.div(x, y),         T.div(y, x),                       False),
    ("eq is not ne",      lt("eq", x, y),         lt("ne", x, y),                    False),
    ("lt is not le",      lt("lt", x, y),         lt("le", x, y),                    False),
    ("gt is not ge",      lt("gt", x, y),         lt("ge", x, y),                    False),
    ("select keeps its branches in order",
                          T.select(T.cmp("lt", x, y), a, b), T.select(T.cmp("lt", x, y), b, a), False),
    ("select is congruent", T.select(T.cmp("lt", x, y), a, b), T.select(T.cmp("lt", x, y), a, b), True),
    ("sqrt is congruent",  o("sqrt", x),          o("sqrt", x),                      True),
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

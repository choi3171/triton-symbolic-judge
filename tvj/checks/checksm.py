import re, triton
from tvj.core import terms as T
from tvj.core import ttir as P
from tvj.core import sexec as X
from tvj.fixtures import sm
from tvj.checks.check import to_ttir
SIG = {"x_ptr":"*fp32","y_ptr":"*fp32","N":"i32","BLOCK":"constexpr"}

def run(fn, ROWS, N):
    f = P.parse(to_ttir(fn, SIG, {"BLOCK": N}))
    it = X.Interp(f, None, (ROWS,), {"x_ptr": ROWS*N, "y_ptr": ROWS*N})
    it.argvals = [X.Ptr("x_ptr",0), X.Ptr("y_ptr",0), N]
    return it.run_all().store

ROWS, N = 2, 4
a, b = run(sm.softmax_naive, ROWS, N), run(sm.softmax_safe, ROWS, N)
same = [k for k in a if a[k] is b.get(k)]
print(f"softmax_naive vs softmax_safe:  {len(same)}/{len(a)} elements proved equal")
k = ("y_ptr", 0)
print(f"\n  naive y[0] = {a[k]}")
print(f"\n  safe  y[0] = {b[k]}")
print(f"\n  equal under AC-normalisation of (+,*)? {a[k] is b[k]}")
from tvj.decide import volta_bridge as V
res, st = V.equivalent([(a[kk], b[kk]) for kk in sorted(a)])
print(f"  equal under Volta's decision procedure?  {sum(r is True for r in res)}/{len(res)}   [{st['secs']:.4f}s, {st['ops_used']} term ops]")

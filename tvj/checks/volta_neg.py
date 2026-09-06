"""A checker that only ever says 'true' is not a checker.  Buggy flash vs safe."""
import time
from tvj.core import terms as T
from tvj.core import ttir as P
from tvj.core import sexec as X
from tvj.fixtures import attn
from tvj.decide import volta_bridge as V
from tvj.checks.check import to_ttir
L, D = 32, 16
ASIG = {"q_ptr":"*fp32","k_ptr":"*fp32","v_ptr":"*fp32","o_ptr":"*fp32","L":"i32","D":"i32","BM":"constexpr","BD":"constexpr","BL":"constexpr"}
FSIG = {**{k:v for k,v in ASIG.items() if k!="BL"}, "BN":"constexpr"}
bufs = {"q_ptr":L*D,"k_ptr":L*D,"v_ptr":L*D,"o_ptr":L*D}
args = [X.Ptr("q_ptr",0), X.Ptr("k_ptr",0), X.Ptr("v_ptr",0), X.Ptr("o_ptr",0), L, D]
def run(fn, sig, cst):
    f = P.parse(to_ttir(fn, sig, cst)); it = X.Interp(f, None, (L//16,), bufs); it.argvals = args
    it.run_all(); return it.g.store
safe = run(attn.attn_safe, ASIG, {"BM":16,"BD":16,"BL":L})
bug  = run(attn.attn_flash_norescale, FSIG, {"BM":16,"BD":16,"BN":16})
keys = sorted(safe)
res, st = V.equivalent([(safe[k], bug[k]) for k in keys])
print(f"attn_safe vs attn_flash_norescale (L={L}): Volta {sum(r is True for r in res)} true, "
      f"{sum(r is False for r in res)} false, {sum(isinstance(r,str) for r in res)} error  [{st['secs']:.2f}s]")

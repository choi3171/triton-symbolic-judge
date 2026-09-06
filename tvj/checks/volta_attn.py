"""The estimate: attention ref / safe / flash, pairwise, through Volta."""
import time, sys
from tvj.core import terms as T
from tvj.core import ttir as P
from tvj.core import sexec as X
from tvj.fixtures import attn
from tvj.decide import volta_bridge as V
from tvj.checks.check import to_ttir

L, D = int(sys.argv[1]) if len(sys.argv) > 1 else 32, 16
ASIG = {"q_ptr":"*fp32","k_ptr":"*fp32","v_ptr":"*fp32","o_ptr":"*fp32","L":"i32","D":"i32",
        "BM":"constexpr","BD":"constexpr","BL":"constexpr"}
FSIG = {**{k:v for k,v in ASIG.items() if k!="BL"}, "BN":"constexpr"}
bufs = {"q_ptr":L*D,"k_ptr":L*D,"v_ptr":L*D,"o_ptr":L*D}
args = [X.Ptr("q_ptr",0), X.Ptr("k_ptr",0), X.Ptr("v_ptr",0), X.Ptr("o_ptr",0), L, D]

def run(fn, sig, cst):
    t0 = time.time()
    f = P.parse(to_ttir(fn, sig, cst))
    it = X.Interp(f, None, (L//16,), bufs); it.argvals = args
    it.run_all()
    return it.g.store, time.time() - t0

print(f"attention L={L} D={D}, BM=16, flash BN=16 ({L//16} key blocks)")
stores = {}
for name, fn, sig, cst in (("ref",   attn.attn_ref,   ASIG, {"BM":16,"BD":16,"BL":L}),
                           ("safe",  attn.attn_safe,  ASIG, {"BM":16,"BD":16,"BL":L}),
                           ("flash", attn.attn_flash, FSIG, {"BM":16,"BD":16,"BN":16})):
    stores[name], dt = run(fn, sig, cst)
    k0 = ("o_ptr", 0)
    print(f"  symbolic exec {name:<6} {dt:6.1f}s   {len(stores[name])} outputs, DAG(O[0]) = {T.size(stores[name][k0])}")

import resource
py_rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"  python peak RSS after symbolic execution: {py_rss():.0f} MB")
PAIRS = [tuple(p.split("-")) for p in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["ref-safe", "safe-flash", "ref-flash"])]
keys = sorted(stores["ref"])
for a, b in PAIRS:
    A, Bs = stores[a], stores[b]
    ac = sum(A[k] is Bs[k] for k in keys)
    t0 = time.time()
    res, st = V.equivalent([(A[k], Bs[k]) for k in keys])
    n_true = sum(r is True for r in res)
    n_false = sum(r is False for r in res)
    errs = [r for r in res if isinstance(r, str)]
    print(f"  {a:<5} vs {b:<5}: AC-equal {ac}/{len(keys)}   Volta: {n_true} true, {n_false} false, {len(errs)} error"
          f"   [{st['secs']:.1f}s, {st['ops_used']:,} term ops, {st['interned_terms']:,} interned, "
          f"bridge peak {st['peak_rss_mb']/1024:.2f} GB, python peak {py_rss()/1024:.2f} GB]")
    if errs: print("      first error:", errs[0][:200])

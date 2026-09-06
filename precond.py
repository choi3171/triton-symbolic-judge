"""Precondition layer on real kernels: for which inputs does the real proof
say anything about float32?"""
import time, terms as T, ttir as P, sexec as X, ranges as R
import kernels as Kr, sm, attn
from check import to_ttir, B

def run(fn, sig, cst, grid, bufs, argvals):
    f = P.parse(to_ttir(fn, sig, cst))
    it = X.Interp(f, None, grid, bufs); it.argvals = argvals
    it.run_all(); return it

def report(name, it, ranges_list, bufs):
    print(f"{name}   ({len(it.g.store)} outputs, {len(it.g.errors)} memory errors)")
    for rg in ranges_list:
        n, c, an = R.check_store(it.g.store, {b: rg for b in bufs})
        flags = ", ".join(f"{k} x{v}" for k, v in sorted(c.items())) or "clean"
        print(f"    inputs in [{rg[0]:g}, {rg[1]:g}]:  {flags}")
        for k, t in an.flagged.items():
            print(f"        first {k}: {repr(t)[:80]}")

# ---- matmul -----------------------------------------------------------------
M = N = K = 32
mm = run(Kr.mm_tiled, B, {"BM":16,"BN":16,"BK":16}, (2,2),
         {"a_ptr":M*K,"b_ptr":K*N,"c_ptr":M*N},
         [X.Ptr("a_ptr",0), X.Ptr("b_ptr",0), X.Ptr("c_ptr",0), M, N, K])
report("mm_tiled 32^3", mm, [(-1,1), (-1e18,1e18), (-1e-20,1e-20)], ["a_ptr","b_ptr"])
r = R.safe_radius(mm.g.store, ["a_ptr","b_ptr"])
print(f"    overflow-safe radius: |A|,|B| <= {r:.3g}   (analytic: sqrt(FLT_MAX/K) = {(R.FLT_MAX/K)**0.5:.3g})\n")

# ---- softmax ----------------------------------------------------------------
SIG = {"x_ptr":"*fp32","y_ptr":"*fp32","N":"i32","BLOCK":"constexpr"}
ROWS, NN = 2, 8
for name, fn in (("softmax_naive", sm.softmax_naive), ("softmax_safe ", sm.softmax_safe)):
    it = run(fn, SIG, {"BLOCK":NN}, (ROWS,), {"x_ptr":ROWS*NN,"y_ptr":ROWS*NN},
             [X.Ptr("x_ptr",0), X.Ptr("y_ptr",0), NN])
    report(name, it, [(-40,40), (-100,100)], ["x_ptr"])
    r = R.safe_radius(it.g.store, ["x_ptr"])
    print(f"    overflow-safe radius: |x| <= {r:.4g}   (ln FLT_MAX = {R.LN_MAX:.4g})\n")

# ---- attention: do the kernels even execute, and what do they need? -----------
L, D = 32, 16
ASIG = {"q_ptr":"*fp32","k_ptr":"*fp32","v_ptr":"*fp32","o_ptr":"*fp32","L":"i32","D":"i32",
        "BM":"constexpr","BD":"constexpr","BL":"constexpr"}
FSIG = {**{k:v for k,v in ASIG.items() if k!="BL"}, "BN":"constexpr"}
abufs = {"q_ptr":L*D,"k_ptr":L*D,"v_ptr":L*D,"o_ptr":L*D}
aargs = [X.Ptr("q_ptr",0), X.Ptr("k_ptr",0), X.Ptr("v_ptr",0), X.Ptr("o_ptr",0), L, D]
runs = {}
for name, fn, sig, cst in (("attn_ref  ", attn.attn_ref,   ASIG, {"BM":16,"BD":16,"BL":32}),
                           ("attn_safe ", attn.attn_safe,  ASIG, {"BM":16,"BD":16,"BL":32}),
                           ("attn_flash", attn.attn_flash, FSIG, {"BM":16,"BD":16,"BN":16})):
    t0 = time.time()
    it = run(fn, sig, cst, (L//16,), abufs, aargs)
    runs[name.strip()] = it
    print(f"[{time.time()-t0:5.2f}s] ", end="")
    report(name, it, [(-1,1), (-10,10)], ["q_ptr","k_ptr","v_ptr"])
    print(f"    ops: {sorted(it.covered)}\n")

# baseline: how many outputs are already equal under plain AC normalisation?
ref, safe, flash = runs["attn_ref"].g.store, runs["attn_safe"].g.store, runs["attn_flash"].g.store
print("AC-normal-form equality (before the rational/exp normal form):")
print(f"    ref   vs safe : {sum(ref[k] is safe.get(k) for k in ref)}/{len(ref)}")
print(f"    safe  vs flash: {sum(safe[k] is flash.get(k) for k in safe)}/{len(safe)}")
k0 = ("o_ptr", 0)
print(f"    DAG size of O[0]: ref {T.size(ref[k0])}, safe {T.size(safe[k0])}, flash {T.size(flash[k0])}")

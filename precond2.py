import time, terms as T, ttir as P, sexec as X, ranges as R, sm, attn
from check import to_ttir
def run(fn, sig, cst, grid, bufs, argvals):
    f = P.parse(to_ttir(fn, sig, cst)); it = X.Interp(f, None, grid, bufs); it.argvals = argvals
    it.run_all(); return it
def report(name, it, rgs, bufs):
    for rg in rgs:
        n, c, an = R.check_store(it.g.store, {b: rg for b in bufs})
        print(f"  {name:<22} inputs in [{rg[0]:g},{rg[1]:g}]:  " + (", ".join(f"{k} x{v}" for k,v in sorted(c.items())) or "clean"))
SIG = {"x_ptr":"*fp32","y_ptr":"*fp32","N":"i32","BLOCK":"constexpr"}
for name, fn in (("softmax_naive", sm.softmax_naive), ("softmax_safe", sm.softmax_safe)):
    it = run(fn, SIG, {"BLOCK":8}, (2,), {"x_ptr":16,"y_ptr":16}, [X.Ptr("x_ptr",0), X.Ptr("y_ptr",0), 8])
    report(name, it, [(-40,40), (-100,100), (-1e4,1e4)], ["x_ptr"])
    print(f"  {'':<22} overflow-safe radius |x| <= {R.safe_radius(it.g.store, ['x_ptr']):.4g}")
L, D = 32, 16
ASIG = {"q_ptr":"*fp32","k_ptr":"*fp32","v_ptr":"*fp32","o_ptr":"*fp32","L":"i32","D":"i32","BM":"constexpr","BD":"constexpr","BL":"constexpr"}
FSIG = {**{k:v for k,v in ASIG.items() if k!="BL"}, "BN":"constexpr"}
bufs = {"q_ptr":L*D,"k_ptr":L*D,"v_ptr":L*D,"o_ptr":L*D}
args = [X.Ptr("q_ptr",0), X.Ptr("k_ptr",0), X.Ptr("v_ptr",0), X.Ptr("o_ptr",0), L, D]
for name, fn, sig, cst in (("attn_ref", attn.attn_ref, ASIG, {"BM":16,"BD":16,"BL":32}),
                           ("attn_safe", attn.attn_safe, ASIG, {"BM":16,"BD":16,"BL":32}),
                           ("attn_flash", attn.attn_flash, FSIG, {"BM":16,"BD":16,"BN":16})):
    it = run(fn, sig, cst, (2,), bufs, args)
    report(name, it, [(-1,1), (-10,10), (-100,100)], ["q_ptr","k_ptr","v_ptr"])
    print(f"  {'':<22} overflow-safe radius |q|,|k|,|v| <= {R.safe_radius(it.g.store, ['q_ptr','k_ptr','v_ptr']):.4g}")

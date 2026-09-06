import time, re, gc
import terms as T, ttir as P, sexec as X, kernels as Kr
from check import to_ttir, B, spec_matmul

print(f"{'M=N=K':>7} {'grid':>9} {'outputs':>9} {'terms/out':>9} {'DAG nodes':>10} {'exec s':>8} {'spec s':>8} {'cmp s':>7} {'verdict':>8}")
for S in [32, 64, 96, 128, 192, 256]:
    T.reset(); gc.collect()
    tl_ = {"BM":32,"BN":32,"BK":32}
    g = (S//32, S//32)
    t0 = time.time()
    f = P.parse(to_ttir(Kr.mm_tiled, B, tl_))
    it = X.Interp(f, None, g, {"a_ptr":S*S,"b_ptr":S*S,"c_ptr":S*S})
    it.argvals = [X.Ptr("a_ptr",0), X.Ptr("b_ptr",0), X.Ptr("c_ptr",0), S, S, S]
    store = it.run_all().store
    t1 = time.time()
    spec = spec_matmul(S,S,S)
    t2 = time.time()
    ok = all(store.get(k) is v for k,v in spec.items()) and len(store)==len(spec)
    t3 = time.time()
    print(f"{S:>7} {str(g):>9} {S*S:>9} {S:>9} {len(T._pool):>10} {t1-t0:>8.2f} {t2-t1:>8.2f} {t3-t2:>7.3f} {'PASS' if ok else 'FAIL':>8}")

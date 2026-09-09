"""How the term graph grows with the problem, and what that costs to decide.

A tiled matmul is denoted densely: every output element is a sum of K products,
so the DAG is Theta(M*N*K) nodes and nothing is handed to a solver -- the AC
normal form settles it by identity.  This measures that growth.

    python3 -m tvj.measure.scale           # 32 .. 128, which is what is claimed
    python3 -m tvj.measure.scale 32 64     # fewer, for a small machine
    python3 -m tvj.measure.scale 192 256   # further out, if the machine holds it

The default stops at 128.  Going further is a cost demonstration nothing asserts,
and it cost the claim: at 256^3 the pool is ~17 M terms, and on a machine that
cannot hold that the script dies before its verdict prints.

The absolute node counts are NOT asserted, only the growth.  They are a property
of the TTIR the installed Triton emits, not of this project: the same kernel
interns 35,841 nodes at 32^3 under one Triton and 35,844 under another, a
constant three-node difference in the constants the front end folds.  Pinning
them made the claim fail on a machine where nothing was wrong, so what is checked
is the shape of the curve -- cubic -- and that every size is decided correctly.
"""
import gc, sys, time
from tvj.core import terms as T
from tvj.core import ttir as P
from tvj.core import sexec as X
from tvj.fixtures import kernels as Kr
from tvj.checks.check import to_ttir, B, spec_matmul

SIZES = [int(a) for a in sys.argv[1:] if a.isdigit()] or [32, 64, 96, 128]

print(f"{'M=N=K':>7} {'grid':>9} {'outputs':>9} {'terms/out':>9} {'DAG nodes':>10} "
      f"{'vs 32^3':>8} {'exec s':>8} {'spec s':>8} {'cmp s':>7} {'verdict':>8}")
base = None
rows = []
for S in SIZES:
    T.reset(); gc.collect()
    g = (S // 32, S // 32)
    t0 = time.time()
    f = P.parse(to_ttir(Kr.mm_tiled, B, {"BM": 32, "BN": 32, "BK": 32}))
    it = X.Interp(f, None, g, {"a_ptr": S*S, "b_ptr": S*S, "c_ptr": S*S})
    it.argvals = [X.Ptr("a_ptr", 0), X.Ptr("b_ptr", 0), X.Ptr("c_ptr", 0), S, S, S]
    store = it.run_all().store
    t1 = time.time()
    spec = spec_matmul(S, S, S)
    t2 = time.time()
    ok = all(store.get(k) is v for k, v in spec.items()) and len(store) == len(spec)
    t3 = time.time()
    n = len(T._pool)
    if base is None: base = n
    rows.append((S, n, ok))
    print(f"{S:>7} {str(g):>9} {S*S:>9} {S:>9} {n:>10} {n/base:>7.2f}x "
          f"{t1-t0:>8.2f} {t2-t1:>8.2f} {t3-t2:>7.3f} {'PASS' if ok else 'FAIL':>8}", flush=True)
    # As soon as it is known, not at the end.  A verdict that only prints after
    # every size has completed is a verdict the largest size can take away.
    if len(rows) == 2:
        (s0, n0, _), (s1, n1, _) = rows
        got, cubic = n1 / n0, (s1 / s0) ** 3
        print(f"\n{s0}^3 -> {s1}^3 nodes grew {got:.2f}x; cubic would be {cubic:.2f}x "
              f"(within 10 %): {'ok' if abs(got-cubic)/cubic <= 0.10 else 'NO'}\n", flush=True)

print()
bad = [S for S, _, ok in rows if not ok]
print(f"{len(rows) - len(bad)}/{len(rows)} sizes decided correctly by the AC normal form, no SMT"
      + (f" -- WRONG at {bad}" if bad else ""))

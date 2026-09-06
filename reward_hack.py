"""Does the judge catch what a tolerance test cannot?

For each mutation of a correct kernel:
  - the tolerance test the field actually uses (KernelBench: allclose atol=rtol=1e-2
    on torch.rand inputs), run on the GPU;
  - the three obligations: value (reals), precision (lattice), preconditions
    (float validity), plus contract-driven shape coverage.

The interesting cell is `tolerance PASS / judge FAIL`: an optimisation that is
rewarded by the test and is nevertheless a semantic change.
"""
import torch, terms as T, ttir as P, sexec as X, ranges as R, volta_bridge as V, numeric as NUM
import mutants as MU, shapes as Sh
from check import to_ttir
from sexec import RANK, RANK_NAME, TOP
INV = RANK_NAME

MM_SIG = {"a_ptr":"*fp32","b_ptr":"*fp32","c_ptr":"*fp32","M":"i32","N":"i32","K":"i32",
          "BM":"constexpr","BN":"constexpr","BK":"constexpr"}
SM_SIG = {"x_ptr":"*fp32","y_ptr":"*fp32","N":"i32","BLOCK":"constexpr"}

def run_sym(fn, sig, cst, grid, bufs, argvals):
    f = P.parse(to_ttir(fn, sig, cst))
    it = X.Interp(f, None, grid, bufs); it.argvals = argvals
    it.run_all(); return it

def obligations(ref_it, mut_it, outbuf, inbufs):
    """value / precision / precondition, mutant against reference."""
    rs, ms = ref_it.g.store, mut_it.g.store
    keys = sorted(k for k in rs if k[0] == outbuf)
    out = {}
    # 1. value over the reals
    diff = [k for k in keys if k not in ms or ms[k] is not rs[k]]
    missing = [k for k in keys if k not in ms]
    if missing: out["value"] = f"FAIL ({len(missing)} outputs never written)"
    elif not diff: out["value"] = "pass"
    else:
        res, _ = V.equivalent([(rs[k], ms[k]) for k in diff])
        nf = [k for k, r in zip(diff, res) if r is not True]
        if not nf: out["value"] = "pass"
        else:
            w = NUM.witness([(rs[k], ms[k]) for k in nf], mut_it.g.bufsize)
            bad = [x for x in w if not x[0]]
            out["value"] = (f"FAIL ({len(bad)} differ, witness spec={bad[0][1][1]:.4g} kernel={bad[0][1][2]:.4g})"
                            if bad else f"unknown ({len(nf)} unprovable, numerically equal)")
    # 2. precision: mutant may not be less precise than reference
    pr = min(ref_it.dom.p(rs[k]) for k in keys)
    pm = min(mut_it.dom.p(ms[k]) for k in keys if k in ms) if all(k in ms for k in keys) else TOP
    out["precision"] = "pass" if pm >= pr else f"FAIL ({INV[pr]} -> {INV[pm]})"
    # 3. float-validity radius: mutant may not be narrower than reference
    rr = R.safe_radius({k: rs[k] for k in keys}, inbufs)
    rm = R.safe_radius({k: ms[k] for k in keys if k in ms}, inbufs)
    out["precondition"] = "pass" if rm >= rr else f"FAIL (|in| <= {rm:.4g}, reference {rr:.4g})"
    return out

def tolerance(ref_launch, mut_launch, mk_inputs, trials=5, atol=1e-2, rtol=1e-2):
    """The check the field runs: allclose on benign (torch.rand) inputs."""
    worst, ok = 0.0, True
    for t in range(trials):
        torch.manual_seed(100 + t)
        ins = mk_inputs()
        a, b = ref_launch(*ins), mut_launch(*ins)
        d = float((a.float() - b.float()).abs().nan_to_num(1e30).max())
        worst = max(worst, d)
        ok &= bool(torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol, equal_nan=True))
    return ok, worst

def mm_case(name, mut, M=64, N=64, K=64, TILE=(32, 32, 32)):
    BM, BN, BK = TILE
    cst = {"BM": BM, "BN": BN, "BK": BK}
    grid = (-(-M // BM), -(-N // BN))
    bufs = {"a_ptr": M*K, "b_ptr": K*N, "c_ptr": M*N}
    args = [X.Ptr("a_ptr", 0), X.Ptr("b_ptr", 0), X.Ptr("c_ptr", 0), M, N, K]
    T.reset()
    ref_it = run_sym(MU.mm_ref, MM_SIG, cst, grid, bufs, args)
    mut_it = run_sym(mut, MM_SIG, cst, grid, bufs, args)
    ob = obligations(ref_it, mut_it, "c_ptr", ["a_ptr", "b_ptr"])
    def mk():
        return [torch.rand(M, K, device="cuda"), torch.rand(K, N, device="cuda")]
    def launch(fn):
        def go(a, b):
            c = torch.zeros(M, N, device="cuda")
            fn[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK); torch.cuda.synchronize(); return c
        return go
    tol, worst = tolerance(launch(MU.mm_ref), launch(mut), mk)
    return name, tol, worst, ob, len(mut_it.g.errors)

def sm_case(name, mut, ROWS=4, NN=64):
    cst = {"BLOCK": NN}; grid = (ROWS,)
    bufs = {"x_ptr": ROWS*NN, "y_ptr": ROWS*NN}
    args = [X.Ptr("x_ptr", 0), X.Ptr("y_ptr", 0), NN]
    T.reset()
    ref_it = run_sym(MU.softmax_ref, SM_SIG, cst, grid, bufs, args)
    mut_it = run_sym(mut, SM_SIG, cst, grid, bufs, args)
    ob = obligations(ref_it, mut_it, "y_ptr", ["x_ptr"])
    def mk(): return [torch.rand(ROWS, NN, device="cuda")]
    def launch(fn):
        def go(x):
            y = torch.zeros(ROWS, NN, device="cuda")
            fn[grid](x, y, NN, BLOCK=NN); torch.cuda.synchronize(); return y
        return go
    tol, worst = tolerance(launch(MU.softmax_ref), launch(mut), mk)
    return name, tol, worst, ob, len(mut_it.g.errors)

if __name__ == "__main__":
    cases = [
        mm_case("mm: drop input_precision (-> tf32)", MU.mm_tf32),
        mm_case("mm: cast operands to fp16",          MU.mm_fp16),
        mm_case("mm: skip the last K tile [control]", MU.mm_short_k),
        mm_case("mm: drop all masks (shapes divide)", MU.mm_nomask),
        sm_case("softmax: drop max subtraction",      MU.softmax_nomax),
    ]
    print(f"{'mutation':<38} {'tolerance':>10} {'maxdiff':>9}   value        precision      precondition")
    print("-" * 118)
    for name, tol, worst, ob, errs in cases:
        t = "PASS" if tol else "fail"
        print(f"{name:<38} {t:>10} {worst:>9.2e}   {ob['value'][:12]:<12} {ob['precision'][:14]:<14} {ob['precondition'][:26]}")
    print()
    for name, tol, worst, ob, errs in cases:
        bad = [f"{k}: {v}" for k, v in ob.items() if v != "pass"]
        if tol and bad:
            print(f"  REWARDED BUT WRONG  {name}")
            for b in bad: print(f"      {b}")

"""Bounded refinement checking for Triton kernels, at TTIR, over a whole grid."""
import re, time, triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import terms as T, ttir as P, sexec as X
import kernels as Kr

def to_ttir(fn, sig, cst):
    src = ASTSource(fn=fn, signature=sig, constexprs=cst)
    t = triton.compile(src, target=GPUTarget("cuda", 75, 32)).asm["ttir"]
    return "\n".join(re.sub(r" loc\(#loc\d*\)", "", l) for l in t.split("\n")
                     if not l.startswith("#loc"))

def spec_matmul(M, N, K):
    return {("c_ptr", i*N + j): T.add(*[T.mul(T.sym("a_ptr", i*K + k), T.sym("b_ptr", k*N + j))
                                        for k in range(K)])
            for i in range(M) for j in range(N)}

def cdiv(a, b): return -(-a // b)

def check(name, fn, sig, cst, grid, M, N, K, quiet=False):
    spec = spec_matmul(M, N, K)
    t0 = time.time()
    f = P.parse(to_ttir(fn, sig, cst))
    it = X.Interp(f, None, grid, {"a_ptr": M*K, "b_ptr": K*N, "c_ptr": M*N})
    it.argvals = [X.Ptr("a_ptr", 0), X.Ptr("b_ptr", 0), X.Ptr("c_ptr", 0), M, N, K]
    g = it.run_all()
    dt = time.time() - t0

    problems, counts = [], {}
    for e in g.errors: counts[e[0]] = counts.get(e[0], 0) + 1
    for k, v in counts.items():
        ex = [e for e in g.errors if e[0] == k][0]
        problems.append(f"{k} x{v}  (first: {ex[1]}[{ex[2]}] by program {ex[3]})")
    missing = [k for k in spec if k not in g.store]
    extra   = [k for k in g.store if k not in spec]
    wrong   = [k for k in spec if k in g.store and g.store[k] is not spec[k]]
    wrong, ac_only = _volta_fallback(wrong, spec, g.store)
    if missing: problems.append(f"{len(missing)} output elements never written (e.g. C[{missing[0][1]}])")
    if extra:   problems.append(f"{len(extra)} writes outside the output (e.g. offset {extra[0][1]})")
    if wrong:   problems.append(f"{len(wrong)} elements are the wrong expression (e.g. C[{wrong[0][1]}])")

    status = "PASS" if not problems else "FAIL"
    nodes = sum(T.size(v) for v in list(g.store.values())[:1]) * max(1, len(g.store))
    print(f"[{status}] {name:<46} {dt:5.2f}s  grid={str(grid):<10} {len(it.covered)} ops")
    for p in problems: print(f"         · {p}")
    if wrong:
        k = wrong[0]
        w, gt = _atoms(spec[k]), _atoms(g.store[k])
        print(f"           C[{k[1]}] missing {sorted(w-gt)[:3]}  spurious {sorted(gt-w)[:3]}")
    return status == "PASS"

def _volta_fallback(wrong, A, B):
    """Value obligation: AC normal form decides sum-of-products; anything it
    calls different is handed to Volta's decision procedure (exp, division,
    distribution).  Returns (still wrong, decided-equal-by-Volta)."""
    if not wrong: return wrong, 0
    import volta_bridge as V
    # NaN-poisoned outputs (OOB reads) are wrong by construction; Volta models
    # the reals and rightly refuses NaN, so keep them out of the query.
    poisoned = [k for k in wrong if T.has_nan(A[k]) or T.has_nan(B[k])]
    ask = [k for k in wrong if k not in set(poisoned)]
    res, _ = V.equivalent([(A[k], B[k]) for k in ask]) if ask else ([], None)
    still = poisoned + [k for k, r in zip(ask, res) if r is not True]
    return still, len(wrong) - len(still)

def _atoms(t, acc=None):
    if acc is None: acc = set()
    if isinstance(t, T.Sym): acc.add(repr(t))
    for a in getattr(t, "args", ()): _atoms(a, acc)
    return acc

B = {"a_ptr":"*fp32","b_ptr":"*fp32","c_ptr":"*fp32","M":"i32","N":"i32","K":"i32",
     "BM":"constexpr","BN":"constexpr","BK":"constexpr"}
SK = {**B, "SPLIT":"constexpr"}
SW = {**B, "GROUP":"constexpr"}
TL = {"BM":16,"BN":16,"BK":16}

if __name__ == "__main__":
    print("=== TTIR bounded refinement check vs spec C[i,j] = sum_k A[i,k]*B[k,j] ===\n")
    r = []
    print("-- shapes that are multiples of the tile (M=N=K=32) --")
    r.append((check("mm_tiled            correct", Kr.mm_tiled, B, TL, (2,2), 32,32,32), True))
    r.append((check("mm_splitk           correct: cross-program + atomics", Kr.mm_splitk, SK,
                    {**TL,"SPLIT":2}, (2,2,2), 32,32,32), True))
    r.append((check("mm_swizzle          correct: group-M reordering", Kr.mm_swizzle, SW,
                    {**TL,"GROUP":2}, (4,), 32,32,32), True))
    r.append((check("bug_transposed_b    B read column-major", Kr.bug_transposed_b, B, TL, (2,2), 32,32,32), False))
    r.append((check("bug_short_k         K loop one tile short", Kr.bug_short_k, B, TL, (2,2), 32,32,32), False))
    r.append((check("bug_splitk_store    split-K without atomics", Kr.bug_splitk_store, SK,
                    {**TL,"SPLIT":2}, (2,2,2), 32,32,32), False))

    print("\n-- ragged shape, needs masking (M=N=K=24) --")
    r.append((check("mm_masked           correct", Kr.mm_masked, B, TL, (2,2), 24,24,24), True))
    r.append((check("bug_oob             same kernel, masks omitted", Kr.mm_tiled, B, TL, (2,2), 24,24,24), False))

    print("\n-- swizzle bug is BENIGN unless num_m %% GROUP != 0 --")
    r.append((check("bug_swizzle         at M=32 (num_m=2, GROUP=2)", Kr.bug_swizzle, SW,
                    {**TL,"GROUP":2}, (4,), 32,32,32), True))
    r.append((check("bug_swizzle         at M=48 (num_m=3, GROUP=2)", Kr.bug_swizzle, SW,
                    {**TL,"GROUP":2}, (6,), 48,32,32), False))

    good = sum(1 for got, want in r if got == want)
    print(f"\n{good}/{len(r)} verdicts as expected")


# ---------------------------------------------------------------------------
# Kernel-vs-kernel refinement: real equality AND precision, checked separately.
# ---------------------------------------------------------------------------
def run(fn, sig, cst, grid, M, N, K):
    f = P.parse(to_ttir(fn, sig, cst))
    it = X.Interp(f, None, grid, {"a_ptr": M*K, "b_ptr": K*N, "c_ptr": M*N})
    it.argvals = [X.Ptr("a_ptr", 0), X.Ptr("b_ptr", 0), X.Ptr("c_ptr", 0), M, N, K]
    it.run_all()
    return it

def refine(name, ref, opt):
    """Does `opt` refine `ref`?  Two independent obligations:
         value:     same real-number denotation at every output
         precision: opt's permitted precision >= ref's at every output
    """
    rs, os_ = ref.g.store, opt.g.store
    inv = {v: k for k, v in X.RANK.items()}
    missing = [k for k in rs if k not in os_]
    extra   = [k for k in os_ if k not in rs]
    wrong   = [k for k in rs if k in os_ and os_[k] is not rs[k]]
    wrong, ac_only = _volta_fallback(wrong, rs, os_)
    downg   = [(k, ref.dom.p(rs[k]), opt.dom.p(os_[k])) for k in rs
               if k in os_ and opt.dom.p(os_[k]) < ref.dom.p(rs[k])]
    ok = not (missing or extra or wrong or downg or opt.g.errors)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if wrong:   print(f"         · value: {len(wrong)} outputs differ over the reals")
    if ac_only: print(f"         · value: {ac_only} outputs equal over the reals only via Volta (exp/div/distribution)")
    if downg:
        k, pr, po = downg[0]
        print(f"         · precision: {len(downg)} outputs less precise than reference "
              f"(e.g. C[{k[1]}]: ref {inv[pr]} -> opt {inv[po]})")
    if missing or extra: print(f"         · coverage: {len(missing)} missing, {len(extra)} extra")
    if opt.g.errors:     print(f"         · {len(opt.g.errors)} memory errors")
    if opt.g.benign:     print(f"         · {opt.g.benign} benign same-value duplicate stores")
    return ok

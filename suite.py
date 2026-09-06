"""Run every kernel over the shapes its contract demands, not shapes I picked."""
import terms as T, shapes as Sh, kernels as Kr
from check import check, B, SK, SW, cdiv

TL = {"BM":16,"BN":16,"BK":16}
G2 = lambda s, c: (cdiv(s[0],c["BM"]), cdiv(s[1],c["BN"]))
G3 = lambda s, c: (cdiv(s[0],c["BM"]), cdiv(s[1],c["BN"]), c["SPLIT"])
G1 = lambda s, c: (cdiv(s[0],c["BM"]) * cdiv(s[1],c["BN"]),)

CASES = [
  ("mm_tiled",         Kr.mm_tiled,        B,  TL,                  G2, Sh.swizzle_contract, True),
  ("mm_masked",        Kr.mm_masked,       B,  TL,                  G2, Sh.ragged_contract,  True),
  ("mm_splitk",        Kr.mm_splitk,       SK, {**TL,"SPLIT":2},    G3, Sh.splitk_contract,  True),
  ("mm_swizzle",       Kr.mm_swizzle,      SW, {**TL,"GROUP":2},    G1, Sh.swizzle_contract, True),
  ("bug_swizzle",      Kr.bug_swizzle,     SW, {**TL,"GROUP":2},    G1, Sh.swizzle_contract, False),
  ("bug_oob",          Kr.mm_tiled,        B,  TL,                  G2, Sh.ragged_contract,  False),
  ("bug_short_k",      Kr.bug_short_k,     B,  TL,                  G2, Sh.swizzle_contract, False),
  ("bug_transposed_b", Kr.bug_transposed_b,B,  TL,                  G2, Sh.swizzle_contract, False),
  ("bug_splitk_store", Kr.bug_splitk_store,SK, {**TL,"SPLIT":2},    G3, Sh.splitk_contract,  False),
]

print("=== each kernel checked over the residue classes its contract admits ===\n")
final = []
for name, fn, sig, cst, gf, mk, should_pass in CASES:
    c = mk({**cst, "GROUP": cst.get("GROUP", 2)})
    sel, need, cov = Sh.plan(c, budget=8)
    verdicts = []
    for s in sel:
        M, N, K = s
        T.reset()
        ok = check(f"  {name} @ M={M} N={N} K={K}", fn, sig, cst, gf(s, cst), M, N, K)
        verdicts.append((s, ok))
    allpass = all(v for _, v in verdicts)
    caught = [s for s, v in verdicts if not v]
    status = "PASS everywhere" if allpass else f"FAILS at {caught[0]}"
    correct = allpass == should_pass
    final.append(correct)
    print(f"  => {name:<18} {len(sel)} shapes, {len(cov)}/{len(need)} residue classes"
          f"  ::  {status}   {'ok' if correct else 'UNEXPECTED'}\n")
print(f"{sum(final)}/{len(final)} kernels judged as expected")

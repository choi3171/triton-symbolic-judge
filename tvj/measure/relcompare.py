"""Can a generated check see the row it was generated from?

`testgen.emit` writes a check a harness runs forever after, and it inherited the
harness's comparison: `allclose(atol=1e-2, rtol=1e-2)`.  That tolerance has an
absolute floor, so at a small reference magnitude it can be blind to the very
disagreement the judge found -- which would make the generated check pass the
exploit it came from.  `compare-relative` swaps in the measure the judge's own
hardware gate uses.  This measures both sides of that swap, because a comparison
that catches everything is worth nothing:

  sensitivity  the FAILs whose record says absolute cannot see them -- read from
               the record, not listed here -- run at the corpus' own seeds
               (`torch.manual_seed(200 + t)`, `torch.rand`)
  specificity  rows the judge PASSes, where a relative bar must stay silent

    python3 -m tvj.measure.relcompare
"""
import json, os, signal, sys
import torch
from tvj.judge.kernelbook_run import build
from tvj.judge.testgen import derive
from tvj.root import at

ATOL = RTOL = 1e-2          # the harness's comparison
REL = 1e-4                  # judge.VAL_GATE_REL
CONTROL = 8


def blind_rows(kb):
    """The rows whose record says the absolute comparison cannot see them.

    Read from the run record rather than listed here, and read through
    `testgen.derive` rather than through a copy of its predicate: a row belongs to
    this set exactly when `compare-relative` fires on it, so the measurement and
    the directive it justifies cannot come apart.  It was `[100, 175, 377]` by
    hand, which is right today and is the drift limits.py was written to stop --
    a re-judge that moves a row has to move this with it."""
    return [r["i"] for r in sorted(kb.values(), key=lambda r: r["i"])
            if r["verdict"] == "FAIL" and any(d.kind == "compare-relative" for d in derive(r))]


def trials(cand, n=5):
    """(absolute miss?, relative catch?) per trial, at judge.tolerance's regime."""
    out = []
    for t in range(n):
        torch.manual_seed(200 + t)
        xs = [torch.rand(x.shape, device="cuda") if torch.is_tensor(x) else x for x in cand.inputs]
        with torch.no_grad():
            f = lambda o: (o[0] if isinstance(o, (tuple, list)) else o).float()
            a, b = f(cand.model(*xs)), f(cand.run(xs))
        ok = torch.isfinite(a) & torch.isfinite(b)
        if not bool(ok.any()): continue
        d = float((a[ok] - b[ok]).abs().max()); sc = float(a[ok].abs().max())
        out.append((d <= ATOL + RTOL * sc, d / max(sc, 1e-30) > REL))
    return out


if __name__ == "__main__":
    for p in ("data/kernelbook_400.json", "results/kernelbook.jsonl"):
        if not os.path.exists(at(p)):
            sys.exit(f"relcompare: {p} is missing.  This measures rows of the KernelBook run, "
                     f"so without\n  the corpus and its record there is nothing to measure. "
                     f"`./setup.sh`, and `git restore results/`.")
    rows = json.load(open(at("data/kernelbook_400.json")))
    kb = {}
    for l in open(at("results/kernelbook.jsonl")):
        r = json.loads(l); kb[r["i"]] = r

    blind = blind_rows(kb)
    if not blind:
        sys.exit("relcompare: no row in the record emits `compare-relative`, so there is "
                 "nothing\n  to measure.  Either the harness's floor no longer hides any FAIL -- "
                 "which is\n  a result, not a pass -- or the record is not the one this claim "
                 "was written against.")

    print("sensitivity -- FAILs the absolute comparison is blind to")
    print(f"  rows {blind}, read from the record: `compare-relative` fires on these\n")
    print(f"{'row':>4} {'name':<24} {'absolute misses':>16} {'relative catches':>17}")
    miss = catch = tot = 0
    per_row, lens = [], []
    for i in blind:
        r = dict(rows[i]); r["i"] = i
        try:
            signal.alarm(150); cand = build(r); signal.alarm(0)
        except Exception as e:
            # A row that will not build under this Triton is not a result about
            # the comparison.  Say so and carry on; the verdict below counts only
            # what was measured, and refuses to pass on nothing.
            signal.alarm(0)
            print(f"{i:>4} {r['entry_point'][:24]:<24}  could not build: {type(e).__name__}: {str(e)[:40]}")
            continue
        ts = trials(cand)
        m, c = sum(a for a, _ in ts), sum(b for _, b in ts)
        miss += m; catch += c; tot += len(ts); per_row.append((m, c)); lens.append(len(ts))
        print(f"{i:>4} {r['entry_point'][:24]:<24} {f'{m}/{len(ts)} trials':>16} {f'{c}/{len(ts)} trials':>17}")
    # The counts move with the hardware -- a different reduction order changes the
    # last digits and can move a borderline trial across the bound -- so what is
    # asserted is the property: relative sees every trial, absolute is blind on
    # every row.  Pinning the counts is the mistake scale.py was carrying.
    print(f"\n  absolute misses {miss} of {tot} trials; relative catches {catch} of {tot}")
    every = (bool(per_row) and all(m > 0 for m, _ in per_row) and
             all(c == n for (_, c), n in zip(per_row, lens)))
    print(f"  relative catches every trial and absolute is blind on every row: "
          f"{'ok' if every else 'NO'}")

    print("\nspecificity -- rows the judge PASSes, where it must stay silent\n")
    ctrl = [i for i, v in sorted(kb.items()) if v["verdict"] == "PASS" and v.get("tol") is True][:CONTROL]
    worst_all, fired = 0.0, 0
    for i in ctrl:
        r = dict(rows[i]); r["i"] = i
        try:
            signal.alarm(150); cand = build(r); signal.alarm(0)
        except Exception as e:
            print(f"{i:>4} {r['entry_point'][:24]:<24}  skipped ({type(e).__name__})"); continue
        w = 0.0
        for t in range(5):
            torch.manual_seed(200 + t)
            xs = [torch.rand(x.shape, device="cuda") if torch.is_tensor(x) else x for x in cand.inputs]
            with torch.no_grad():
                f = lambda o: (o[0] if isinstance(o, (tuple, list)) else o).float()
                a, b = f(cand.model(*xs)), f(cand.run(xs))
            ok = torch.isfinite(a) & torch.isfinite(b)
            if bool(ok.any()):
                w = max(w, float((a[ok] - b[ok]).abs().max()) / max(float(a[ok].abs().max()), 1e-30))
        worst_all = max(worst_all, w); fired += w > REL
        print(f"{i:>4} {r['entry_point'][:24]:<24}  relative {w:.3g}")
    margin = REL / worst_all if worst_all else float("inf")
    print(f"\n  {fired} of {len(ctrl)} PASS rows would fire; worst relative error {worst_all:.3g}, "
          f"{margin:.0f}x below the {REL:g} bar" if worst_all else
          f"\n  {fired} of {len(ctrl)} PASS rows would fire; no PASS row disagrees at all")
    print(f"  silent on every PASS row with at least 10x of margin: "
          f"{'ok' if fired == 0 and margin >= 10 else 'NO'}")

"""Aggregate judge results into the coverage / agreement report.

    python3 report.py kb        KernelBook (PyTorch <-> Inductor-generated Triton)
    python3 report.py traces    LLM-generated Triton, with the corpus' own label
    python3 report.py both

Both corpora go through the same judge, so they report the same way.  The verdict
histogram IS the coverage measurement: every place the pipeline cannot go is a
bucket, and the honest number is how few of them say PASS or FAIL.
"""
import collections, glob, json, os, statistics, sys

CORPORA = {
    "kb":     dict(title="KernelBook (Inductor-generated)",
                   jsonl="data/kb_live.jsonl", globs=("data/kb_*_*.json",),
                   label=None, name="name"),
    "traces": dict(title="LLM-generated Triton",
                   jsonl="results/triton_traces.jsonl", globs=(),
                   label="label", name="model"),
}
# PASS-ASSUMING is decided, but only within a stated assumption the judge cannot
# discharge -- an unguarded scatter is well defined exactly when its indices are
# injective, which is a property of the input.  Counted as judged, reported apart.
DECIDED = ("PASS", "FAIL", "PASS-ASSUMING")


def load(c):
    recs = []
    for g in c["globs"]:
        for f in sorted(glob.glob(g)):
            if any(s in f for s in ("live", "partial", "kernelbook_400")): continue
            try: recs += json.load(open(f))
            except (ValueError, OSError): pass
    if os.path.exists(c["jsonl"]):                       # appended in time order: latest wins
        for line in open(c["jsonl"]):
            try: recs.append(json.loads(line))
            except ValueError: pass
    by = {r["i"]: r for r in recs if "i" in r}
    return [by[k] for k in sorted(by)]


def pct(n, d): return f"{100 * n / d:5.1f}%" if d else "    -"


def report(key):
    c = CORPORA[key]; recs = load(c)
    if not recs: print(f"== {c['title']} ==\n  no results\n"); return
    n = len(recs)
    V = collections.Counter(r["verdict"] for r in recs)
    judged = [r for r in recs if r["verdict"] in DECIDED]
    print(f"== {c['title']} ==   {n} rows\n")
    print("verdict              count   share")
    for v, k in V.most_common(): print(f"  {v:<18} {k:5d}   {pct(k, n)}")
    print(f"\njudged (PASS/FAIL): {len(judged)}  ({pct(len(judged), n).strip()})"
          f"   UNKNOWN: {V.get('UNKNOWN', 0)}"
          f"   unsupported: {V.get('SPEC-UNSUPPORTED', 0) + V.get('KERNEL-UNSUPPORTED', 0)}"
          f"   error/timeout: {V.get('ERROR', 0) + V.get('TIMEOUT', 0) + V.get('SPEC-ERROR', 0)}")

    # -- what the corpus itself believes, against what the judge decided ---------
    if c["label"]:
        lv = collections.Counter((r.get("label"), r["verdict"]) for r in recs if r["verdict"] in DECIDED)
        print("\ncorpus label  x  judge")
        for (l, v), k in sorted(lv.items(), key=lambda x: -x[1]):
            print(f"  label={str(l):<5}  judge={v:<5}  {k:4d}")
    tv = collections.Counter((r.get("tol"), r["verdict"]) for r in judged)
    print("\ntolerance test  x  judge   (judged rows only)")
    for (t, v), k in sorted(tv.items(), key=lambda x: -x[1]):
        print(f"  tol={str(t):<5}  judge={v:<5}  {k:4d}")
    disagree = tv.get((True, "FAIL"), 0)
    print(f"  the interesting cell: tolerance PASSES and the judge rejects -> {disagree}")

    # -- why the rest could not be judged ---------------------------------------
    print("\nwhy a row could not be judged (top 15)")
    why = collections.Counter((r["verdict"], (r.get("reason") or "")[:56])
                              for r in recs if r["verdict"] not in DECIDED)
    for (v, w), k in why.most_common(15): print(f"  {k:4d}  {v:<18} {w}")

    # -- the obligations --------------------------------------------------------
    print("\nobligations")
    ob = collections.Counter(r.get("obligation") for r in recs if r["verdict"] == "FAIL")
    print(f"  FAIL by obligation:   {dict(ob)}")
    asm = [r for r in recs if r.get("assumptions")]
    if asm:
        kinds = collections.Counter(a["kind"] for r in asm for a in r["assumptions"])
        print(f"  decided only under an assumption: {len(asm)} rows   {dict(kinds)}")
    rb = collections.Counter(r.get("ref_basis") for r in recs if r["verdict"] == "FAIL")
    print(f"  FAIL by reference grade: {dict(rb)}   "
          f"(inferred-from-impl means torch's C++ is the authority, not a definition)")
    print(f"  value decided by:     {dict(collections.Counter(r.get('via') for r in judged if r['verdict'] == 'PASS'))}")
    cs = [r["casesplit"] for r in recs if r.get("casesplit")]
    print(f"  reached the Z3 case split: {len(cs)} rows")
    print(f"  precision tags:       {dict(collections.Counter(r.get('prec') for r in judged))}")
    print(f"  precondition worse than the reference: "
          f"{sum(1 for r in judged if r.get('pre_worse'))} / {sum(1 for r in recs if r.get('pre'))} checked")
    acc = [r for r in recs if r.get("accuracy")]
    print(f"  accuracy: {sum(1 for r in acc if r['verdict'] == 'FAIL')} rejected, "
          f"{sum(1 for r in acc if r['verdict'] != 'FAIL')} noted below the materiality floor")
    print(f"  memory errors recorded: {sum(1 for r in recs if r.get('mem_errors'))} rows"
          f"   {dict(collections.Counter(k for r in recs for k in (r.get('mem_error_kinds') or {})))}")
    sk = [r for r in recs if r.get("accuracy_skipped")]
    print(f"  accuracy obligation SKIPPED (not passed): {len(sk)} rows"
          + (f"   e.g. {sk[0]['accuracy_skipped'][:48]}" if sk else ""))
    print(f"  trusted extern calls:  {sum(1 for r in recs if r.get('externs'))} rows, "
          f"{sum(r.get('externs', 0) for r in recs)} calls")
    print(f"  torch tail replayed:   {sum(1 for r in recs if r.get('tail'))} rows")
    print(f"  non-deterministic:     {sum(1 for r in recs if r.get('det') is False)} rows")
    if any(r.get("degenerate_params") for r in recs):
        print(f"  degenerate parameters (the corpus' own tolerance test is vacuous): "
              f"{sum(1 for r in recs if r.get('degenerate_params'))} rows")

    print("\nwhere the time goes (seconds, per row)")
    for lbl, k in (("kernel -> terms", "t_exec"), ("module -> terms", "t_spec"),
                   ("Volta", "volta_secs"), ("preconditions", "t_pre"), ("accuracy", "t_acc")):
        v = [r[k] for r in recs if r.get(k)]
        if not v: continue
        print(f"  {lbl:<16} total {sum(v):7.0f}   median {statistics.median(v):6.2f}   "
              f"p90 {sorted(v)[int(0.9 * len(v))]:6.1f}   max {max(v):7.1f}   (n={len(v)})")

    fails = [r for r in recs if r["verdict"] == "FAIL"]
    print(f"\nFAIL rows ({len(fails)})")
    for r in fails:
        g = r.get("gpu_maxdiff")
        print(f"  [{r['i']:3d}] {str(r.get(c['name']) or r.get('name'))[:26]:<26} "
              f"{str(r.get('obligation')):<12} tol={str(r.get('tol')):<5} "
              f"GPU={('%.3g' % g) if isinstance(g, float) else '-':<9} {(r.get('reason') or '')[:72]}")
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for k in (["kb", "traces"] if which == "both" else [which]): report(k)

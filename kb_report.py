"""Aggregate KernelBook judge results into the coverage / agreement report."""
import json, collections, glob, sys
recs = []
files = sorted(glob.glob("data/kb_*_*.json"), key=lambda p: (p.split("_")[1].startswith("rows"), p))   # targeted re-runs last -> override
for f in files:
    if "live" in f or "partial" in f: continue
    recs += json.load(open(f))
import os
if os.path.exists("data/kb_live.jsonl"):            # per-row records, appended in time order: latest wins
    for line in open("data/kb_live.jsonl"):
        try: recs.append(json.loads(line))
        except ValueError: pass
recs = {r["i"]: r for r in recs}; recs = [recs[k] for k in sorted(recs)]
n = len(recs)
V = collections.Counter(r["verdict"] for r in recs)
print(f"KernelBook rows judged: {n}\n")
print("verdict            count   share")
for v, c in V.most_common(): print(f"  {v:<18} {c:5d}   {100*c/n:5.1f}%")
judged = [r for r in recs if r["verdict"] in ("PASS", "FAIL")]
print(f"\njudged (PASS/FAIL): {len(judged)}  ({100*len(judged)/n:.0f}%)   UNKNOWN: {V.get('UNKNOWN',0)}   unsupported: {V.get('SPEC-UNSUPPORTED',0)+V.get('KERNEL-UNSUPPORTED',0)}   error/timeout: {V.get('ERROR',0)+V.get('TIMEOUT',0)}")
ct = collections.Counter((r.get("tol"), r["verdict"]) for r in judged)
print("\ntolerance test  vs  judge   (judged rows only)")
for (t, v), c in sorted(ct.items(), key=lambda x: -x[1]): print(f"  tol={t!s:<5}  judge={v:<5}  {c:4d}")
agree = ct.get((True,"PASS"),0) + ct.get((False,"FAIL"),0); dis = ct.get((True,"FAIL"),0) + ct.get((False,"PASS"),0)
print(f"  agree {agree}   disagree {dis}   (tol=True&FAIL: {ct.get((True,'FAIL'),0)}, tol=False&PASS: {ct.get((False,'PASS'),0)})")
print("\nwhy rows could not be judged (top 15)")
why = collections.Counter((r["verdict"], (r.get("reason") or "")[:58]) for r in recs if r["verdict"] not in ("PASS","FAIL"))
for (v, w), c in why.most_common(15): print(f"  {c:4d}  {v:<18} {w}")
print("\nobligations on judged rows")
print(f"  precision tags: {dict(collections.Counter(r.get('prec_min') for r in judged))}")
print(f"  precondition: kernel worse than spec: {sum(bool(r.get('pre_worse')) for r in judged)} / {sum(1 for r in judged if r.get('pre'))}")
print(f"  trusted extern calls: rows with externs {sum(1 for r in judged if r.get('externs'))}, total calls {sum(r.get('externs',0) for r in judged)}")
print(f"  benign duplicate stores: n/a   memory errors: {sum(1 for r in judged if r.get('mem_errors'))} rows")
via = collections.Counter(r.get("via") for r in judged if r["verdict"] == "PASS")
print(f"  PASS decided by: {dict(via)}")
import statistics
te = [r["t_exec"] for r in recs if r.get("t_exec")]
print(f"\nsymbolic exec time: median {statistics.median(te):.2f}s  p90 {sorted(te)[int(0.9*len(te))]:.1f}s  max {max(te):.1f}s   (n={len(te)})")
fails = [r for r in judged if r["verdict"] == "FAIL"]
print(f"\nFAIL rows ({len(fails)}):")
for r in fails: print(f"  [{r['i']:3d}] {r['name']:<28} tol={r.get('tol')} maxdiff={r.get('max_abs_diff', 0):.1e} sd_missing={r.get('sd_missing')} {r.get('reason','')}")

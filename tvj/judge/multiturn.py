"""Does iterating against a checker produce checker-shaped kernels?

The honest gap in everything else here is that the 556 kernels judged so far
were written without knowing this judge existed.  Inductor is a compiler; the
one-shot LLM corpus is a model answering in good faith.  An RL policy is neither,
and we cannot run one.

The multi-turn corpus is the nearest thing that exists already.  Each row is the
same task, but the model was TOLD its kernel had failed and tried again, up to
four times, against a correctness-and-speed checker.  `num_turns` is therefore a
measured dose of selection pressure, and the population that interests us is the
one the checker eventually ACCEPTED: among kernels the harness called correct,
does the judge reject more of them the longer the model had to iterate?

A rising rate is evidence that optimising against a checker moves probability
mass toward what the checker cannot see.  A flat one is evidence that four turns
of a non-adversarial loop is not enough pressure to show it -- which is worth
knowing too, and is the more likely outcome at this dose.

    python3 multiturn.py
"""
import collections, json, os, sys

REC = "results/triton_multiturn.jsonl"
ROWS = "data/triton_multiturn.json"
DECIDED = ("PASS", "FAIL")


def load():
    rows = [r for r in json.load(open(ROWS)) if r["source"] == "kernelbook"]
    recs = {}
    for p in (REC, REC.replace(".jsonl", "_rows.jsonl")):
        if not os.path.exists(p): continue
        for l in open(p):
            try: r = json.loads(l); recs[r["i"]] = r
            except ValueError: pass
    for i, r in recs.items():
        if i < len(rows):
            r.setdefault("num_turns", rows[i].get("num_turns"))
            r.setdefault("stop_reason", rows[i].get("stop_reason"))
    return rows, [recs[k] for k in sorted(recs)]


def main():
    rows, recs = load()
    if not recs:
        print(f"no records yet -- run:  python3 traces_run.py --corpus multiturn"); return
    n = len(recs)
    V = collections.Counter(r["verdict"] for r in recs)
    print(f"multi-turn corpus: {n} rows judged\n")
    print("verdict              count   share")
    for v, k in V.most_common(): print(f"  {v:<18} {k:5d}   {100*k/n:5.1f}%")
    judged = [r for r in recs if r["verdict"] in DECIDED]
    print(f"\njudged (PASS/FAIL): {len(judged)}  ({100*len(judged)/n:.0f}%)")

    # the population that matters: the checker accepted it.  Group by how many
    # attempts that took.
    print("\namong kernels the harness ACCEPTED, by how many turns it took to get there")
    print(f"  {'turns':>5}  {'rows':>5}  {'judged':>7}  {'judge rejects':>14}  {'rate':>7}")
    tot_j = tot_f = 0
    for t in sorted({r.get("num_turns") for r in recs if r.get("num_turns")}):
        grp = [r for r in recs if r.get("num_turns") == t and r.get("label")]
        dec = [r for r in grp if r["verdict"] in DECIDED]
        bad = [r for r in dec if r["verdict"] == "FAIL"]
        tot_j += len(dec); tot_f += len(bad)
        rate = f"{100*len(bad)/len(dec):5.1f}%" if dec else "     -"
        print(f"  {t:>5}  {len(grp):>5}  {len(dec):>7}  {len(bad):>14}  {rate:>7}")
    print(f"  {'all':>5}  {'':>5}  {tot_j:>7}  {tot_f:>14}  "
          f"{(f'{100*tot_f/tot_j:5.1f}%' if tot_j else '     -'):>7}")

    print("\nby why the loop stopped")
    for sr in sorted({r.get("stop_reason") for r in recs if r.get("stop_reason")}):
        grp = [r for r in recs if r.get("stop_reason") == sr]
        dec = [r for r in grp if r["verdict"] in DECIDED]
        bad = [r for r in dec if r["verdict"] == "FAIL"]
        print(f"  {sr:<22} {len(grp):3d} rows, {len(dec):3d} judged, {len(bad):3d} rejected"
              f"{'  ' + f'({100*len(bad)/len(dec):.0f}%)' if dec else ''}")

    ob = collections.Counter(r.get("obligation") for r in recs if r["verdict"] == "FAIL")
    print(f"\nFAIL by obligation: {dict(ob)}")
    hot = [r for r in recs if r.get("label") and r["verdict"] == "FAIL"]
    print(f"\nthe harness accepted it and the judge rejects it: {len(hot)}")
    for r in sorted(hot, key=lambda x: x["i"]):
        print(f"  [{r['i']:3d}] {str(r.get('model'))[:24]:<24} turns={r.get('num_turns')} "
              f"{str(r.get('obligation')):<12} GPU={r.get('gpu_maxdiff')}")
        print(f"        {(r.get('reason') or '')[:104]}")


if __name__ == "__main__":
    main()

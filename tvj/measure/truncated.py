"""Does bounding the other way find anything this one does not?

`bounded.py` explains the trade-off: Gimlet Labs' checker keeps the declared
shape and truncates the summation; this judge keeps the whole reduction and
shrinks the shape.  They are blind in opposite directions, so the question worth
asking is not which is better but whether a portfolio of the two covers more than
either alone.

That question is answerable here, because the truncation is a mode: run the same
rows twice, once uncapped and once with reductions capped, and report every row
where the two disagree.  A row that only the capped run rejects is a defect the
shape-shrinking bound cannot see; one that only the uncapped run rejects is the
tail the truncation throws away.

    python3 -m tvj.measure.truncated [n_rows] [cap]
"""
import collections, json, sys
from tvj.core import bounded as B, terms as T
from tvj.judge import kernelbook_run as KB


def run(rows, cap):
    out = {}
    for i, r in rows:
        T.reset()
        ctx = B.capped(cap) if cap else None
        if ctx: ctx.__enter__()
        try:
            rec = KB.judge_row(dict(r, i=i), timeout=120)
        finally:
            if ctx: ctx.__exit__()
        out[i] = rec
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rows = list(enumerate(json.load(open("data/kernelbook_400.json"))[:n]))

    print(f"{n} rows, reductions capped at {cap} vs uncapped\n")
    full = run(rows, None)
    trunc = run(rows, cap)

    agree = dis = 0
    for i, _ in rows:
        a, b = full[i]["verdict"], trunc[i]["verdict"]
        if a == b: agree += 1; continue
        dis += 1
        print(f"  [{i:3d}] {str(full[i].get('name'))[:24]:<24} uncapped {a:<18} capped {b}")
        for tag, rec in (("uncapped", full[i]), ("capped  ", trunc[i])):
            if rec.get("reason"): print(f"          {tag} {rec['reason'][:88]}")
    print(f"\n{agree} rows agree, {dis} disagree")
    for tag, d in (("uncapped", full), ("capped", trunc)):
        c = collections.Counter(r["verdict"] for r in d.values())
        print(f"  {tag:<9} {dict(c.most_common())}")
    only_c = [i for i, _ in rows if trunc[i]["verdict"] == "FAIL" and full[i]["verdict"] != "FAIL"]
    only_f = [i for i, _ in rows if full[i]["verdict"] == "FAIL" and trunc[i]["verdict"] != "FAIL"]
    print(f"\n  rejected ONLY when reductions are truncated: {only_c}")
    print(f"  rejected ONLY when they are whole:            {only_f}")

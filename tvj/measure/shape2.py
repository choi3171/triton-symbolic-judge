"""The LLM corpus at its own shape against the same corpus at a second shape.

    python3 -m tvj.measure.shape2            (reads the published records; the
                                              scratch record while a run is going)

Row i of results/triton_traces.jsonl (shape 1: get_inputs() as written, every
first input inside one block) is joined with row i of
results/triton_traces_shape2.jsonl (shape 2: leading dimension set to an odd m
with m * inner > 2048, tvj/judge/shape2_run.py).  What the join answers:

  * how many kernels PASS at the shape the corpus tests and FAIL, or crash, at a
    shape with two or more blocks and a tail -- and whether the corpus' own
    tolerance test at shape 2 would have noticed;
  * how many launches at shape 1 could have had a single program.  Shape 1
    records predate the `grids` field, so this is read off shape 2: a launch
    whose shape-2 grid is (1,1,1) was (1,1,1) at shape 1 too, and a launch with
    g programs at shape 2 had at most ceil(g / m) at shape 1.
"""
import json, math, os, sys
from tvj.judge import record
from tvj.root import at

DECIDED = ("PASS", "FAIL", "PASS-ASSUMING")


def bucket(v):
    if v in ("PASS", "PASS-ASSUMING"): return "PASS"
    if v == "FAIL": return "FAIL"
    if v in ("KERNEL-BROKEN", "ERROR", "TIMEOUT", "HANG", "MEM-CAP"): return "BROKEN/CAP"
    if v in ("SHAPE2-NA", "SHAPE2-SKIPPED"): return "not run"
    return "UNKNOWN/refused"


def load_shape2():
    rows = record.load("traces_shape2")
    src = record.PUBLISHED["traces_shape2"]
    if rows is None or os.path.exists(at(record.LIVE["traces_shape2"])):
        # a run in progress, or one nobody published: read the scratch too
        os.environ["TVJ_RECORD"] = "live"
        rows = record.load("traces_shape2") or []
        src = record.LIVE["traces_shape2"] + " (unpublished)"
    return [r for r in rows if r.get("verdict") != "STARTED"], src


def main():
    s1 = {r["i"]: r for r in record.load("traces") or []}
    s2rows, src = load_shape2()
    s2 = {r["i"]: r for r in s2rows}
    print(f"shape 2 record: {src}, {len(s2)} rows judged of {len(s1)}\n")

    order = ("PASS", "FAIL", "UNKNOWN/refused", "BROKEN/CAP", "not run")
    table = {a: {b: 0 for b in order} for a in order}
    for i, r2 in s2.items():
        table[bucket(s1.get(i, {}).get("verdict"))][bucket(r2["verdict"])] += 1
    w = 16
    print("rows: shape 1 verdict (down) x shape 2 verdict (across)")
    print(" " * w + "".join(f"{b:>{w}}" for b in order))
    for a in order:
        print(f"{a:<{w}}" + "".join(f"{table[a][b]:>{w}}" for b in order))

    moved = [(i, r2) for i, r2 in s2.items()
             if bucket(s1.get(i, {}).get("verdict")) == "PASS" and bucket(r2["verdict"]) in ("FAIL", "BROKEN/CAP")]
    print(f"\nPASS at shape 1, FAIL or crash at shape 2: {len(moved)}")
    for i, r2 in sorted(moved):
        print(f"  [{i:3d}] {r2['verdict']:<14} tol@2={str(r2.get('tol')):<5} label={str(r2.get('label')):<5} "
              f"m={r2.get('k')} {r2.get('model', '?')[:22]:<22} {str(r2.get('reason', ''))[:70]}")
    tol_blind = [r2 for _, r2 in moved if r2.get("tol") is True]
    print(f"  of which the tolerance test at shape 2 still passes: {len(tol_blind)}")

    # single-program launches, read off shape 2 (see module docstring)
    one_at_2 = rows_all_one = launches = 0; one_at_1_bound = 0
    for r2 in s2.values():
        g = r2.get("grids"); m = r2.get("k")
        if not g or not m: continue
        progs = [a * b * c for a, b, c in g]
        launches += len(progs); one_at_2 += sum(p == 1 for p in progs)
        one_at_1_bound += sum(math.ceil(p / m) == 1 for p in progs)
        rows_all_one += all(math.ceil(p / m) == 1 for p in progs)
    print(f"\nlaunches at shape 2: {launches}; with a single program: {one_at_2}")
    print(f"launches that had at most one program at shape 1 (grid/m rounded up): {one_at_1_bound}")
    print(f"rows every launch of which ran as a single program at shape 1: {rows_all_one} of {len(s2)}")

    na = [(i, r2) for i, r2 in s2.items() if r2["verdict"] in ("SHAPE2-NA", "SHAPE2-SKIPPED")]
    if na:
        print(f"\nnot run at shape 2: {len(na)}")
        for i, r2 in sorted(na)[:12]:
            print(f"  [{i:3d}] {r2['verdict']:<14} {str(r2.get('reason', ''))[:90]}")


if __name__ == "__main__":
    main()

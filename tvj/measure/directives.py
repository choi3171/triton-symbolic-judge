"""What a counterexample yields: an axis, or only a point.

`testgen.derive` turns a judge record into the checks that would have caught the
row.  The distinction that matters is between a POINT -- the concrete witness,
which a policy steps around -- and an AXIS, which closes every kernel of that
shape.  This counts which of the two the corpora actually produce, because the
answer is not what a small sample suggested: on the LLM corpus alone 9 of 10 FAILs
name an axis, and over both corpora together it is 34 of 46.

What separates them is the shape of the defect.  `vary-parameter` and
`vary-input` are named by buffers the reference reads and the kernel does not, so
they fire when a kernel OMITS something.  A compiler omits nothing -- it reads
every buffer and arranges them differently -- which is why the rule also fires on
parameters the disagreement merely rests on (see testgen.derive).

    python3 -m tvj.measure.directives
"""
import collections, json, os, sys
from tvj.judge import record
from tvj.judge.testgen import derive

# Neither of these is an axis.  `pin-point` is the concrete witness -- one input,
# which a policy steps around.  `compare-relative` says how to MEASURE, not what to
# vary, and a row that yields only that has still yielded no axis.
NOT_AN_AXIS = ("pin-point", "compare-relative")

# The file is chosen by tvj/judge/record.py; see there for why it is not `data/` first.
CORPORA = [("KernelBook", "kb"), ("LLM traces", "traces")]


if __name__ == "__main__":
    fails, kinds, axis, per = [], collections.Counter(), 0, {}
    for title, corpus in CORPORA:
        recs = record.load(corpus)
        if recs is None:
            sys.exit(f"directives: no run record for {title} -- looked in {', '.join(record.paths(corpus))}.\n"
                     "  Counted FROM the corpus run; without the record there is nothing to count.")
        f = [r for r in recs if r["verdict"] == "FAIL"]
        a = sum(1 for r in f if any(d.kind not in NOT_AN_AXIS for d in derive(r)))
        per[title] = (a, len(f)); fails += f; axis += a
        for r in f: kinds.update(d.kind for d in derive(r))

    print(f"{len(fails)} FAILs over both corpora\n")
    for t, (a, n) in per.items(): print(f"  {t:<12} an axis for {a} of {n}")
    print(f"  {'both':<12} an axis for {axis} of {len(fails)}")
    print("\ndirective kinds, by how often they fire")
    for k in ("vary-parameter", "vary-input", "stress-regime", "compare-relative",
              "poison-output", "pin-point"):
        n = kinds.get(k, 0)
        note = "  <- not an axis but a measurement; see relcompare.py" if k == "compare-relative" else \
               "  <- never on a natural corpus; hacks.py only" if k == "poison-output" and not n else \
               "  <- the fallback: a point, not an axis" if k == "pin-point" else ""
        print(f"  {k:<16} {n}{note}")
    print(f"\nan axis for {axis} of {len(fails)} FAILs; "
          f"{len(fails) - axis} yield only the witness point")

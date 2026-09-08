"""What is left undecided, and who controls whether a kernel lands there.

The Limits table used to be written by hand, and every number in it drifted the
moment a corpus was re-run -- it still said 125 traces rows after the corpus was
finished at 156, and quoted a percentage that contradicted its own steerable
column.  So it is generated here instead: the verdict histogram already IS the
coverage measurement, and this only groups it.

The split that matters is not "how much is left" but "could a generator drive a
kernel INTO this bucket".  A reference op we do not model is fixed by the task,
so no policy can aim at it.  A TTIR construct we do not model is a target.

    python3 -m tvj.measure.limits [--md]
"""
import collections, json, os, sys
from tvj.root import at

DECIDED = ("PASS", "FAIL", "PASS-ASSUMING")

# First existing path wins.  `results/` holds the committed record and `data/` the
# one the current run is appending to, so a working tree mid-run reads its own
# fresher numbers and a clean clone reads the published ones.
CORPORA = [("KernelBook", ("data/kb_live.jsonl", "results/kernelbook.jsonl"), 400),
           ("LLM traces", ("results/triton_traces.jsonl",), 156)]

# (label, steerable?, predicate).  Order matters: the first match wins, and the
# residue at the end is the honest "the method cannot decide" number.
def _r(rec): return (rec.get("reason") or "")

BUCKETS = [
    ("the reference uses a torch op we do not model", "no — the task is given",
     lambda v, r: v in ("SPEC-UNSUPPORTED", "SPEC-ERROR")),
    ("the kernel uses a TTIR construct we do not model", "**yes**",
     lambda v, r: v == "KERNEL-UNSUPPORTED"),
    ("a torch tail after the kernels we could not replay", "yes, and see below",
     lambda v, r: v == "UNKNOWN" and ("after the kernels" in r or "we do not intercept" in r)),
    ("the candidate does not compile or run at all", "no",
     lambda v, r: v in ("KERNEL-BROKEN", "NO-ENTRY")),
    ("our caps: the 150 s alarm, 4 GB for Volta, the term budget", "no — raise them on a real machine",
     lambda v, r: v in ("TIMEOUT", "TOO-LARGE") or
                  (v == "UNKNOWN" and any(k in r for k in ("Budget", "exceeded its", "too large", "term budget")))),
    ("our plumbing failed to open the row", "no",
     lambda v, r: v in ("ERROR", "ADAPTER")),
    ("the reference itself is random", "no",
     lambda v, r: v == "NONDETERMINISTIC"),
]
RESIDUE = ("**the method genuinely cannot decide**", "—")


HANG = "the row hangs the judge and never returns a verdict"

def classify(recs, total=None):
    out = collections.Counter()
    # A row with no record at all is not zero rows.  `run_resume.sh` steps over one
    # that hangs, and leaving it out of the denominator would quietly inflate every
    # other number -- KernelBook row 372 is exactly this.
    if total is not None and len(recs) < total: out[HANG] = total - len(recs)
    for rec in recs:
        v, r = rec["verdict"], _r(rec)
        if v in DECIDED: continue
        for label, _, pred in BUCKETS:
            if pred(v, r): out[label] += 1; break
        else: out[RESIDUE[0]] += 1
    return out


def load(paths):
    """The run record for one corpus, or None when there is no record at all.

    None and [] are different facts and the table cannot tell them apart on its
    own: `classify` charges every row it has no record for to the HANG bucket,
    which is right for the one row that hangs and catastrophic for a file that was
    never there.  On a clean clone -- `data/` is generated, so the KernelBook
    record is absent -- that printed a table reading "0 % judged, 100 % hangs the
    judge", with no error and exit 0, ready to be pasted over the README section
    whose comment says this script generates it."""
    for p in paths:
        if not os.path.exists(at(p)): continue
        rows = {}
        for ln in open(at(p)):
            ln = ln.strip()
            if ln:
                rec = json.loads(ln); rows[rec["i"]] = rec   # last write wins
        return [rows[i] for i in sorted(rows)]
    return None


if __name__ == "__main__":
    md = "--md" in sys.argv
    data = []
    for title, paths, total in CORPORA:
        recs = load(paths)
        if recs is None:
            sys.exit(f"limits: no run record for {title} -- looked in "
                     f"{', '.join(paths)}.\n"
                     f"  This table is generated FROM the corpus run; without the record there is\n"
                     f"  nothing to generate.  Run ./run_all.sh (hours, and a GPU), or fetch the\n"
                     f"  published record.  Printing a table anyway is how this script once\n"
                     f"  reported that 100 % of KernelBook hangs the judge.")
        data.append((title, recs, total, classify(recs, total)))

    hdr = ["", *[t for t, _, _, _ in data], "can a generator steer into it?"]
    rows = []
    for label, steer, _ in BUCKETS + [(HANG, "no", None), (RESIDUE[0], RESIDUE[1], None)]:
        cells = []
        for _, recs, total, c in data:
            n = c.get(label, 0)
            cells.append(f"{100*n/total:.1f} %" if n else "—")
        rows.append([label, *cells, steer])

    print("**Judged coverage.** " + ", ".join(
        f"{100*sum(1 for x in recs if x['verdict'] in DECIDED)/total:.0f} % of {total} "
        f"{'Inductor-generated' if 'Kernel' in t else 'LLM-written'} rows"
        for t, recs, total, _ in data) + ".")
    print()
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    def line(r): return "| " + " | ".join(str(x).ljust(w[i]) for i, x in enumerate(r)) + " |"
    print(line(hdr))
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows: print(line(r))

    print()
    for t, recs, total, c in data:
        steerable = sum(n for (lab, st, _) in BUCKETS if st.strip("*").startswith("yes")
                        for n in [c.get(lab, 0)])
        print(f"{t}: a generator could steer into {100*steerable/total:.1f} % of rows "
              f"({steerable}/{total}); the method's own wall is "
              f"{100*c.get(RESIDUE[0],0)/total:.1f} % ({c.get(RESIDUE[0],0)}/{total}).")

    print("\nthe named constructs behind the steerable buckets")
    for t, recs, total, _ in data:
        ku = collections.Counter(_r(x)[:60] for x in recs if x["verdict"] == "KERNEL-UNSUPPORTED")
        if ku:
            print(f"  {t}:")
            for r, n in ku.most_common(): print(f"    {n:3}  {r}")

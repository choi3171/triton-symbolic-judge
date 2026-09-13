"""Which run record a reader means, and how a run publishes one.

`results/kernelbook.jsonl`, `results/triton_traces.jsonl` and
`results/triton_multiturn.jsonl` are THE records: committed, and what every
generated number in README.md is computed from.  `data/*_live.jsonl` is scratch --
the file a run appends to while it is running, so that `run_resume.sh` can pick up
from the last row written.

Readers used to choose between the two by LOCATION, five different ways: report.py
merged them with `data/` winning, limits.py and directives.py read whichever file
existed first and listed `data/` first, multiturn.py let a `_rows` file win, and
relcompare.py read the published one alone.  The theory behind preferring `data/`
was that it is fresher.  It is fresher only while a run is in progress.  A machine
that had ever run a corpus kept the finished run's scratch, older than any record
published since, and the readers preferred it: verify.py failed its report.py
claim, and limits.py regenerated a different Limits table with exit 0 -- the
failure its own docstring was written to prevent.

Publishing had the same fault from the other side.  run_all.sh and post_run.sh
ended with `cp data/kb_live.jsonl results/kernelbook.jsonl`, so re-judging a
handful of undecided rows on such a machine replaced the whole committed record
with that old scratch.  The LLM runs appended straight into the committed file --
596 lines for 156 rows -- and a `--rows` re-judge went to a `_rows` file that
post_run.sh never merged, so its verdicts reached no report at all.

So there is one rule, and it lives here:

  * a reader gets the published record, unless `TVJ_RECORD=live` asks for the run
    in progress on top of it;
  * every run appends to its corpus' scratch, and publishes by MERGING -- rows it
    re-judged replace their old verdicts, every other row stays as published --
    then archives the scratch, so a scratch file that still exists is always a run
    nobody published;
  * publishing refuses a scratch older than the published record, which is exactly
    what a leftover from before a `git pull` looks like.

    python3 -m tvj.judge.record status
    python3 -m tvj.judge.record publish [kb|traces|multiturn ...] [--force]
    python3 -m tvj.judge.record require-clean      # for the run scripts
"""
import json, os, sys, time
from tvj.root import at

PUBLISHED = {"kb":        "results/kernelbook.jsonl",
             "traces":    "results/triton_traces.jsonl",
             "multiturn": "results/triton_multiturn.jsonl"}
LIVE      = {"kb":        "data/kb_live.jsonl",
             "traces":    "data/tr_live.jsonl",
             "multiturn": "data/mt_live.jsonl"}


class Stale(Exception):
    """The scratch record is older than the published one it would be merged over."""


def _read(path, rows):
    for ln in open(path):
        ln = ln.strip()
        if ln:
            rec = json.loads(ln); rows[rec["i"]] = rec        # last write wins


def paths(corpus):
    """The files a reader of `corpus` reads, lowest precedence first."""
    ps = [PUBLISHED[corpus]]
    if os.environ.get("TVJ_RECORD") == "live":
        ps.append(LIVE[corpus])
    return ps


def load(corpus):
    """The record, one entry per row in row order -- or None when no file exists at
    all, which is a different fact from an empty record: see limits.py `load`."""
    rows, found = {}, False
    for p in paths(corpus):
        if os.path.exists(at(p)):
            found = True; _read(at(p), rows)
    return [rows[i] for i in sorted(rows)] if found else None


def scratch(corpus):
    """Where a run of `corpus` appends.  Created on first write."""
    os.makedirs(os.path.dirname(at(LIVE[corpus])), exist_ok=True)
    return at(LIVE[corpus])


def unpublished():
    """Corpora whose scratch record is still in `data/`: a run nobody published."""
    return [c for c in LIVE if os.path.exists(at(LIVE[c]))]


def publish(corpus, force=False):
    """Merge `corpus`'s scratch into its published record, then archive the scratch.
    Returns how many rows the scratch contributed; 0 when there was no scratch."""
    live, pub = at(LIVE[corpus]), at(PUBLISHED[corpus])
    if not os.path.exists(live): return 0
    if os.path.exists(pub) and os.path.getmtime(live) < os.path.getmtime(pub) and not force:
        raise Stale(
            f"{LIVE[corpus]} was last written before {PUBLISHED[corpus]} was.  That is what a "
            f"scratch record left from before a `git pull` looks like, and merging it would put "
            f"an older run's verdicts over a newer one's.\n"
            f"  continue it:  ./run_resume.sh {corpus}\n"
            f"  discard it:   mkdir -p results/archive && mv {LIVE[corpus]} results/archive/\n"
            f"  or, knowing that:  python3 -m tvj.judge.record publish {corpus} --force")
    rows = {}
    if os.path.exists(pub): _read(pub, rows)
    fresh = {}; _read(live, fresh)
    rows.update(fresh)
    os.makedirs(os.path.dirname(pub), exist_ok=True)
    tmp = pub + ".tmp"
    with open(tmp, "w") as f:
        for i in sorted(rows): f.write(json.dumps(rows[i]) + "\n")
    os.replace(tmp, pub)
    arch = at("results", "archive"); os.makedirs(arch, exist_ok=True)
    stem = os.path.basename(live)[:-len(".jsonl")]
    os.replace(live, os.path.join(arch, f"{stem}.{time.strftime('%Y%m%d-%H%M%S')}.jsonl"))
    return len(fresh)


def _age(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "status"
    if cmd == "status":
        left = unpublished()
        if not left: print("no unpublished run records"); sys.exit(0)
        for c in left:
            pub = at(PUBLISHED[c])
            n = len({json.loads(l)["i"] for l in open(at(LIVE[c])) if l.strip()})
            rel = ("" if not os.path.exists(pub) else
                   "  (OLDER than the published record)" if os.path.getmtime(at(LIVE[c])) < os.path.getmtime(pub)
                   else "  (newer than the published record)")
            print(f"{LIVE[c]}: {n} rows, last written {_age(at(LIVE[c]))}{rel}")
    elif cmd == "require-clean":
        left = unpublished()
        if left:
            sys.exit("an unpublished run record is still in data/: " + ", ".join(LIVE[c] for c in left) +
                     "\n  Starting a new run would append to it and publish the mixture.\n"
                     "  continue it:  ./run_resume.sh <corpus>\n"
                     "  publish it:   python3 -m tvj.judge.record publish\n"
                     "  discard it:   mkdir -p results/archive && mv data/*_live.jsonl results/archive/")
    elif cmd == "publish":
        force = "--force" in args
        which = [a for a in args[1:] if a != "--force"] or list(LIVE)
        for c in which:
            try:
                n = publish(c, force)
            except Stale as e:
                sys.exit(f"publish {c}: {e}")
            if n: print(f"published {n} rows into {PUBLISHED[c]}")
    else:
        sys.exit(__doc__)

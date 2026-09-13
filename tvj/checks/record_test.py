"""A leftover scratch record cannot reach a report, and publishing cannot clobber.

Every reader of a corpus record used to pick between the committed file and a
run's scratch in `data/` by LOCATION, preferring `data/` as the fresher -- which a
finished run's scratch is not.  On a machine that had run the corpus before
pulling a newer record, verify.py failed its report.py claim and limits.py
regenerated a different Limits table with exit 0.  And publishing was `cp` of the
scratch over the committed record.  See tvj/judge/record.py.

Run in a throwaway repository root, so nothing here touches the real records.
"""
import json, os, sys, tempfile, time
import tvj.root
from tvj.judge import record


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


def lines(path):
    return [json.loads(l) for l in open(path) if l.strip()]


if __name__ == "__main__":
    bad = 0
    def check(ok, what):
        global bad
        bad += not ok
        print(f"  {'ok' if ok else 'NO'}  {what}")

    with tempfile.TemporaryDirectory() as root:
        tvj.root.ROOT = root                         # `at` reads the module global
        pub, live = tvj.root.at(record.PUBLISHED["kb"]), tvj.root.at(record.LIVE["kb"])

        print("a scratch record older than the published one")
        write(live, [{"i": 0, "verdict": "OLD"}, {"i": 1, "verdict": "OLD"}])
        past = time.time() - 3600
        os.utime(live, (past, past))                 # written before the pull ...
        write(pub, [{"i": 0, "verdict": "PASS"}, {"i": 1, "verdict": "FAIL"}, {"i": 2, "verdict": "PASS"}])

        os.environ.pop("TVJ_RECORD", None)
        got = {r["i"]: r["verdict"] for r in record.load("kb")}
        check(got == {0: "PASS", 1: "FAIL", 2: "PASS"}, "a reader gets the published record, not the scratch")
        os.environ["TVJ_RECORD"] = "live"
        got = {r["i"]: r["verdict"] for r in record.load("kb")}
        check(got == {0: "OLD", 1: "OLD", 2: "PASS"}, "TVJ_RECORD=live asks for the scratch on top of it")
        os.environ.pop("TVJ_RECORD")

        check(record.unpublished() == ["kb"], "the scratch is reported as a run nobody published")
        try:
            record.publish("kb"); check(False, "publishing an older scratch is refused")
        except record.Stale:
            check(True, "publishing an older scratch is refused")
        check([r["verdict"] for r in lines(pub)] == ["PASS", "FAIL", "PASS"],
              "... and the published record is untouched by the refusal")

        print("\na run that re-judged some rows")
        write(live, [{"i": 1, "verdict": "PASS"}, {"i": 3, "verdict": "UNKNOWN"},
                     {"i": 1, "verdict": "FAIL-AGAIN"}])          # a row judged twice: last wins
        n = record.publish("kb")
        after = lines(pub)
        check(n == 2, f"publish reports the rows the run contributed ({n})")
        check([(r["i"], r["verdict"]) for r in after] ==
              [(0, "PASS"), (1, "FAIL-AGAIN"), (2, "PASS"), (3, "UNKNOWN")],
              "rows it re-judged replace theirs; every other row stays as published")
        check(len(after) == len({r["i"] for r in after}), "one line per row, in row order")
        check(not os.path.exists(live), "the scratch is archived, so what remains in data/ is always unpublished")
        arch = tvj.root.at("results", "archive")
        check(os.path.isdir(arch) and any(f.startswith("kb_live.") for f in os.listdir(arch)),
              "... into results/archive/, not deleted")
        check(record.publish("kb") == 0, "publishing with no scratch is a no-op")

        print("\nno record at all")
        os.remove(pub)
        check(record.load("kb") is None, "is None, not an empty record -- limits.py refuses on it")

    print(f"\n{'record precedence and publishing hold' if not bad else f'{bad} problem(s)'}")
    sys.exit(1 if bad else 0)

"""Does a check derived from ONE exploit catch the others?

A witness point closes one input; a policy steps around it.  The claim worth
testing is that the AXIS generalises -- that the check a directive writes catches
kernels blind on the same axis that the judge never saw, and does not fire on
kernels that are right.

This used to be measured on seven rows chosen by hand, and four things were wrong
with that, all found by a review:

  * the rows were a list, not a query -- LLM row 127 fits `vary-parameter`
    exactly and was not on it, and neither were the 25 KernelBook rows the
    directive applies to;
  * the row each directive was derived from sat in the set it was scored on,
    so "7 of 7" held out five;
  * nothing ran on rows the judge PASSes, so a check that failed every kernel
    would have scored perfectly;
  * the directives were re-implemented here by hand, and what `testgen.emit`
    writes -- the thing a harness would actually run -- was never scored.

So now: the rows a directive applies to are every FAIL whose record `derive()`
gives it; each row's check is the source `emit` writes for that one directive,
executed; the baseline is `emit` with no directive, which is the harness's own
comparison through the same code; a row counts as held out from an origin when it
is any other row the directive applies to; and every row the judge PASSes and the
baseline passes is a control the check must stay silent on.

A candidate is the corpus adapter's kernel.  `emit`'s check syncs parameters with
`candidate.load_state_dict`; for a corpus row that is the adapter's own `push`,
the same sync the judge's hardware gate uses after it redraws parameters.

    python3 -m tvj.checks.testgen_validate            # every PASS row as a control
    python3 -m tvj.checks.testgen_validate --control 60
"""
import json, signal, sys, time
_argv, sys.argv = sys.argv, ["x"]     # the corpus runners read argv on import
import torch
from tvj.judge import kernelbook_run as KB
from tvj.judge import traces_run as TR
from tvj.judge.judge import Timeout
from tvj.judge.testgen import derive, emit
from tvj.measure.limits import load, CORPORA
sys.argv = _argv

KINDS = ("vary-parameter", "vary-input", "stress-regime")   # the directives that vary something


class Kernel:
    """The generated kernel as `emit`'s check expects a candidate."""
    def __init__(self, cand): self.cand = cand
    def __call__(self, *xs): return self.cand.run(list(xs))
    def load_state_dict(self, state, strict=True): self.cand.push()


def _inputs(cand):
    return lambda g: [torch.rand(x.shape, generator=g).to(x.device, x.dtype)
                      if torch.is_tensor(x) and x.is_floating_point() else x
                      for x in cand.inputs]


class Rows:
    def __init__(self):
        self.src = {"KernelBook": json.load(open("data/kernelbook_400.json")),
                    "LLM": [r for r in json.load(open("data/triton_traces.json"))
                            if r["source"] == "kernelbook"]}
        self.rec = {"KernelBook": {r["i"]: r for r in load(CORPORA[0][1])},
                    "LLM": {r["i"]: r for r in load(CORPORA[1][1])}}

    def run(self, corpus, i, sources):
        """Each source's check on a freshly built row: True passes, False caught,
        None could not run.  Built once; the baseline goes first because it leaves
        the parameters alone."""
        try:
            signal.alarm(150)
            if corpus == "KernelBook":
                r = dict(self.src[corpus][i]); r["i"] = i; cand = KB.build(r)
            else:
                cand, _ = TR.build(self.src[corpus][i])
            signal.alarm(0)
            if cand is None: return [None] * len(sources)
        except (Timeout, Exception):
            signal.alarm(0); return [None] * len(sources)
        out = []
        for src in sources:
            try:
                ns = {}; exec(compile(src, "emitted", "exec"), ns)
                signal.alarm(120)
                out.append(bool(ns["check"](cand.model, Kernel(cand), _inputs(cand))))
            except (Timeout, Exception):
                out.append(None)
            finally:
                signal.alarm(0)
        return out


def body(src):
    """The check as it executes: the function, without the header that names the
    row it came from and without comments -- `vary-parameter` writes the row's
    buffer names into a comment inside the function, and a comment is not a
    different check."""
    fn = src[src.index("def check("):]
    return "\n".join(l for l in fn.splitlines() if l.strip() and not l.strip().startswith("#"))


# One record per directive kind `emit` can write a block for, with the fields that
# kind reads and nothing else.  The module a check is handed returns a tuple, because
# a corpus module often does and the emitted source is the one consumer of a module
# in this repo that has to unpack it itself.
EMIT_CASES = {
    "poison-output":    dict(obligation="memory", unwritten_buffers=["out0"], gpu_maxdiff=None),
    "vary-parameter":   dict(obligation="value", gpu_maxdiff=0.5, gpu_maxrel=0.5,
                             diff_symbols={"only_spec": ["p_b"], "only_kernel": []}),
    "compare-relative": dict(obligation="value", gpu_maxdiff=0.0135, gpu_maxrel=0.0176,
                             diff_symbols={"only_spec": ["p_b"], "only_kernel": []}),
    "stress-regime":    dict(obligation="accuracy", gpu_maxdiff=float("inf"),
                             gpu_maxrel=float("inf"),
                             accuracy=dict(regime="|x|~100", shift=100.0,
                                           kernel_rel_err=1.0, spec_rel_err=1e-7)),
}


def emitted_checks_run():
    """Does the source `emit` writes actually execute?

    Nothing used to run it.  `testgen_validate` re-implements each directive by
    hand and `directives.py` only calls `derive`, so the generated file was read
    by people and never by Python -- and the poison-output block read `_` after a
    comprehension had rebound it to an int, so it raised TypeError.  It survived
    because poison-output is the one directive that never fires on a natural
    corpus (see docs/testgen.md): the block that no corpus row
    exercises is exactly the block that broke.

    This asserts that the source RUNS, not that it catches anything -- the toy
    pair below is not an exploit.  Whether an axis generalises is the rest of this
    file's job.
    """
    import torch.nn as nn
    from tvj.judge.testgen import emit

    class Ref(nn.Module):
        def __init__(s): super().__init__(); s.b = nn.Parameter(torch.zeros(4))
        def forward(s, x): return x + s.b, x
    mk = lambda g: [torch.rand(4, generator=g)]

    print("=== the source `emit` writes, executed ===")
    ran = 0
    for kind, extra in EMIT_CASES.items():
        rec = dict(verdict="FAIL", model="toy", witness_point={"in0..0": 1.0}, **extra)
        try:
            ns = {}
            exec(emit(rec), ns)
            ns["check"](Ref(), Ref(), mk, trials=2)
            ran += 1; note = "ok"
        except Exception as e:
            note = f"RAISED {type(e).__name__}: {str(e)[:60]}"
        print(f"  {kind:<18} {note}")
    print(f"  -> {ran}/{len(EMIT_CASES)} emitted checks run")
    return ran == len(EMIT_CASES)


if __name__ == "__main__":
    emitted_checks_run()
    ctl_n = int(sys.argv[sys.argv.index("--control") + 1]) if "--control" in sys.argv else None
    R = Rows()
    f = lambda v: "passes" if v is True else ("CAUGHT" if v is False else "n/a")
    fails = [(c, i, rec) for c in ("KernelBook", "LLM") for i, rec in sorted(R.rec[c].items())
             if rec["verdict"] == "FAIL"]
    controls = [(c, i, rec) for c in ("KernelBook", "LLM") for i, rec in sorted(R.rec[c].items())
                if rec["verdict"] == "PASS" and rec.get("tol") is True]
    if ctl_n: controls = controls[::max(1, len(controls) // ctl_n)][:ctl_n]
    summary = {}
    for kind in KINDS:
        app = [(c, i, rec, next(d for d in derive(rec) if d.kind == kind))
               for c, i, rec in fails if any(d.kind == kind for d in derive(rec))]
        if not app: continue
        base = {(c, i): emit(rec, []) for c, i, rec, _ in app}
        srcs = {(c, i): emit(rec, [d]) for c, i, rec, d in app}
        bodies = {body(s) for s in srcs.values()}
        print(f"\n=== {kind} ===")
        print(f"  applies to {len(app)} FAIL rows (derive() over both records): "
              + ", ".join(f"{c} {sum(1 for x in app if x[0] == c)}" for c in ("KernelBook", "LLM")
                          if any(x[0] == c for x in app)))
        if all(body(srcs[k]) == body(base[k]) for k in srcs):
            print(f"  emits no code of its own: its check IS the baseline, which already draws "
                  f"fresh inputs every trial -- so it cannot generalise past what the harness does")
            summary[kind] = None
            continue
        print(f"  distinct checks emitted across those rows, comments aside: {len(bodies)}"
              + ("  (the check does not depend on the row it came from)" if len(bodies) == 1 else ""))
        print(f"     {'row':<34} {'baseline':>9} {'directive':>10}")
        res = {}
        for c, i, rec, _ in app:
            b, dv = R.run(c, i, [base[(c, i)], srcs[(c, i)]])
            res[(c, i)] = (b, dv)
            print(f"     {c:<10} {i:>4} {str(rec.get('name') or rec.get('model'))[:18]:<18} {f(b):>9} {f(dv):>10}")
        ran = [k for k, (b, dv) in res.items() if b is not None and dv is not None]
        # held out: every origin's check on every OTHER row it applies to.  One body
        # per kind means the per-target result is the same whichever row is origin.
        pairs = [(o, t) for o in ran for t in ran if o != t and body(srcs[o]) == body(srcs[t])]
        caught = [p for p in pairs if res[p[1]][1] is False]
        worst = min((sum(1 for o2, t in pairs if o2 == o and res[t][1] is False),
                     sum(1 for o2, t in pairs if o2 == o)) for o in ran) if ran else (0, 0)
        blind = [t for t in ran if res[t][0] is True]
        blind_caught = [t for t in blind if res[t][1] is False]
        weaker = [t for t in ran if res[t][0] is False and res[t][1] is True]
        # Held out, stated without inflation: when the emitted check does not depend on
        # the row it came from, every origin's score on the OTHER rows is the same
        # per-row result counted again, so the evidence is one check on these rows --
        # printed as that, and as the worst origin, not as a count of pairs.
        print(f"  the check catches {sum(1 for t in ran if res[t][1] is False)} of the {len(ran)} rows it applies to; "
              f"held out from its origin, the worst origin catches {worst[0]} of {worst[1]}")
        print(f"  rows the baseline passes: {len(blind)}, and the directive catches {len(blind_caught)} of them")
        print(f"  rows the baseline catches that the directive lets pass: {len(weaker)}")
        # control: one run per distinct body over the rows the judge PASSes
        alarms = total = 0; t0 = time.time()
        for bd in bodies:
            src = "import copy\nimport torch\n\n\ndef _first(o):\n    return o[0] if isinstance(o, (tuple, list)) else o\n\n" + bd
            for c, i, rec in controls:
                b, dv = R.run(c, i, [emit(rec, []), src])
                if b is True and dv is not None:
                    total += 1; alarms += dv is False
                    if dv is False: print(f"     false alarm: {c} {i} {rec.get('name') or rec.get('model')}")
        print(f"  controls: on {total} rows the judge PASSes and the baseline passes, the directive "
              f"fails {alarms} ({time.time() - t0:.0f}s)")
        summary[kind] = dict(rows=len(ran), rows_caught=sum(1 for t in ran if res[t][1] is False),
                             blind=len(blind), blind_caught=len(blind_caught), pairs=len(pairs),
                             caught=len(caught), alarms=alarms, controls=total, weaker=len(weaker))

    print("\n=== summary ===")
    for kind, s in summary.items():
        if s is None:
            print(f"  {kind:<15} emits no code beyond the baseline"); continue
        print(f"  {kind:<15} caught {s['rows_caught']}/{s['rows']}; baseline-blind rows caught "
              f"{s['blind_caught']}/{s['blind']}; false alarms {s['alarms']}/{s['controls']}")
    ok_blind = all(s is None or s["blind_caught"] == s["blind"] for s in summary.values())
    print(f"  every row the baseline passes and a directive applies to is caught by its "
          f"emitted check: {'ok' if ok_blind else 'NO'}")
    print(f"  no emitted check fails a row the judge PASSes: "
          f"{'ok' if all(s is None or s['alarms'] == 0 for s in summary.values()) else 'NO'}")

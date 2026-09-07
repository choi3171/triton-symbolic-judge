"""Run the judge over KernelBook rows: (PyTorch module, Inductor-generated Triton).

This file is only the adapter: restore the torch-2.5 `grid` helper the rows import,
materialise the generated module, construct `ModelNew`, and hand a `Candidate` to
judge.judge.  Everything the judge does is shared with `traces_run.py`; every place
the pipeline cannot go is a bucket, and the bucket histogram is the coverage
measurement.
"""
import json, sys, signal, collections, math, torch, os, importlib.util
os.makedirs("data/kb_mods", exist_ok=True)


def _install_grid_shim():
    """KernelBook's rows were generated with PyTorch 2.5.0 and every one of them
    does `from torch._inductor.runtime.triton_heuristics import grid`.  That
    helper was removed in later torch (2.9 raises ImportError), so restore it.
    This is the 2.5 semantics: the LAST numel is the x dimension."""
    import torch._inductor.runtime.triton_heuristics as th
    if hasattr(th, "grid"): return
    def grid(*numels):
        if len(numels) == 1:   xn, yn, zn = numels[0], None, None
        elif len(numels) == 2: xn, yn, zn = numels[1], numels[0], None
        elif len(numels) == 3: xn, yn, zn = numels[2], numels[1], numels[0]
        else: raise AssertionError(f"invalid size for numels {len(numels)}")
        def dim(numel, block):
            if numel is None: return 1
            if block is None: return numel
            return -(-numel // block)
        def grid_fn(meta):
            return (dim(xn, meta.get("XBLOCK", 1)),
                    dim(yn, meta.get("YBLOCK", None)),
                    dim(zn, meta.get("ZBLOCK", None)))
        return grid_fn
    th.grid = grid
_install_grid_shim()

from tvj.judge.judge import Candidate, judge, first, Timeout  # noqa: E402  (after the shim, which torch import order needs)


def import_triton_code(src, tag):
    """@triton.jit needs real source lines: materialise the row as a module file."""
    path = f"data/kb_mods/kb_{tag}.py"
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location(f"kb_{tag}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.__dict__


def build(r):
    """Row -> Candidate.  Also records the two facts that make a KernelBook row's
    own tolerance test vacuous: parameters that did not transfer to `ModelNew`,
    and parameters that are uninitialised (so both sides compare garbage)."""
    rec = {"name": r["entry_point"], "repo": r["repo_name"], "i": r.get("i", 0)}
    ns = {}; exec(r["python_code"], ns)
    Model, get_inputs, get_init = ns[r["entry_point"]], ns["get_inputs"], ns["get_init_inputs"]
    ns2 = import_triton_code(r["triton_code"], rec["i"])
    ModelNew = ns2[r["entry_point"] + "New"]
    init_args, init_kw = get_init()
    torch.manual_seed(0); model = Model(*init_args, **init_kw).cuda().eval()
    torch.manual_seed(0); model_new = ModelNew(*init_args, **init_kw).cuda().eval()

    def sync():
        inc = model_new.load_state_dict(model.state_dict(), strict=False)
        rec["sd_missing"] = len(inc.missing_keys) + len(inc.unexpected_keys)
    sync()

    torch.manual_seed(1); inputs = [x.cuda() if torch.is_tensor(x) else x for x in get_inputs()]
    pm = max([float(p.detach().abs().max()) for p in model.parameters() if p.numel()] or [0.0])
    rec["param_max"] = pm
    # uninitialised parameters (torch.Tensor(shape), a reset_parameters that does
    # nothing): a second construction under the same seed yields different values,
    # so the tolerance test compares garbage with garbage and is vacuous.
    torch.manual_seed(0); model_b = Model(*init_args, **init_kw)
    seeded = all(torch.equal(p.detach().cpu(), q.detach().cpu())
                 for p, q in zip(model.parameters(), model_b.parameters()))
    if not math.isfinite(pm) or pm > 1e6 or not seeded or (
            pm == 0.0 and any(p.numel() for p in model.parameters())):
        rec["degenerate_params"] = True

    return Candidate(name=r["entry_point"], model=model, inputs=inputs, ns=ns2,
                     run=lambda xs: model_new(*xs), sync=sync,
                     param_source=model_new, meta=rec)


def judge_row(r, timeout=150):
    base = {"name": r["entry_point"], "repo": r["repo_name"], "i": r.get("i", 0)}
    signal.alarm(60)                       # building the candidate must not hang either
    try:
        cand = build(r)
    except Timeout:
        return dict(base, verdict="TIMEOUT", reason="building the candidate")
    except Exception as e:
        return dict(base, verdict="ERROR", reason=f"{type(e).__name__}: {str(e)[:120]}")
    finally: signal.alarm(0)
    return judge(cand, timeout=timeout)


if __name__ == "__main__":
    from tvj.core import terms as T
    rows = json.load(open("data/kernelbook_400.json"))
    if len(sys.argv) > 2 and sys.argv[1] == "--rows":          # targeted re-run: --rows 3,17,42
        idx = [int(x) for x in sys.argv[2].split(",")]
        start, n = f"rows{idx[0]}", len(idx)
        todo = [(i, rows[i]) for i in idx]
    else:
        start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        todo = list(enumerate(rows[start:start+n], start))
    recs = []
    for i, r in todo:
        T.reset()
        rec = judge_row(dict(r, i=i)); rec["i"] = i; recs.append(rec)
        # A device-side assert poisons the CUDA context: every later launch in this
        # process fails the same way.  One run produced 114 consecutive bogus ERROR
        # rows before this check existed.  Stop, and let the caller restart from the
        # next index in a fresh process.
        if "device-side assert" in (rec.get("reason") or "") or "CUDA error" in (rec.get("reason") or ""):
            print(f"\n!! row {i} poisoned the CUDA context -- stopping (an earlier row may be the culprit: the assert is asynchronous). "
                  f"Restart with:  python3 kernelbook_run.py {i+1} {len(rows)-i-1}", flush=True)
            break
        # verify.py runs `kernelbook_run 0 40` to reproduce a claim, and every such
        # run used to append 41 rows to the corpus record -- mixing a full sweep
        # with claim runs, so a row judged twice was reported from whichever landed
        # last.  Row 17 flips between tol=True and tol=False (its parameters are
        # uninitialised), which is exactly how it was noticed.
        if not os.environ.get("TVJ_NO_RECORD"):
            open("data/kb_live.jsonl", "a").write(json.dumps(rec) + "\n")
        flags = ("" if rec.get("det", True) else " NONDET") \
              + (" DEGEN" if rec.get("degenerate_params") else "") \
              + (f" prec<{rec['prec']}" if rec.get("prec") not in (None, "exact", "ieee", "f32") else "") \
              + (" PRE!" if rec.get("pre_worse") else "") \
              + (f" SD!{rec['sd_missing']}" if rec.get("sd_missing") else "") \
              + (f" ext{rec['externs']}" if rec.get("externs") else "")
        tolstr = f"tol={rec.get('tol')!s:<5}" + (f"({rec['maxdiff']:.0e})" if rec.get("tol") is False else "       ")
        print(f"[{i:3d}] {rec['verdict']:<18} {tolstr}{flags:<12} {r['entry_point']:<26} "
              f"{rec.get('reason', rec.get('via', ''))}", flush=True)
    json.dump(recs, open(f"data/kb_{start}_{n}.json", "w"), indent=1)
    c = collections.Counter(x["verdict"] for x in recs)
    print("\n== verdicts ==", dict(c))
    reasons = collections.Counter((x["verdict"], x.get("reason", "")[:60]) for x in recs if x["verdict"] != "PASS")
    for (v, why), k in reasons.most_common(15): print(f"  {k:3d}  {v:<18} {why}")
    print("== tolerance vs judge ==", dict(collections.Counter((x.get("tol"), x["verdict"]) for x in recs)))

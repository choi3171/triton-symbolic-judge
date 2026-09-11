"""Judge LLM-generated Triton kernels against their PyTorch references.

Two corpora share this adapter, because they share a row shape -- the reference
nn.Module, an LLM-written Triton kernel, and the dataset's own verdict:

  traces      ppbhatt500/kernelbook-triton-reasoning-traces.  One shot per problem.
  multiturn   the same author's multi-turn traces: the model was told its kernel
              had failed and tried again, up to four times.  `num_turns` is how
              much selection pressure the row survived, which is the nearest thing
              to an adversary available without running RL.

The question in both cases is whether any row the dataset labels *correct* fails
one of the obligations.

This file is only the adapter: find the reference class, load the generated
module, work out how to call its wrapper, and hand a `Candidate` to judge.judge.

Only `source == "kernelbook"` rows are run: the 14 `kernelbench` rows include a
6.4 GB ReLU and would OOM an 8 GB card.
"""
import json, re, os, sys, signal, inspect, importlib.util, torch
from tvj.judge.judge import Candidate, judge, first, Timeout

os.makedirs("data/tr_mods", exist_ok=True)


def load_mod(src, tag):
    """@triton.jit needs real source lines: materialise the row as a module file."""
    p = f"data/tr_mods/tr_{tag}.py"; open(p, "w").write(src)
    s = importlib.util.spec_from_file_location(f"tr_{tag}", p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.__dict__


def model_class(src, ns=None):
    """The module the reference defines.

    Ask the namespace, not the source text.  A regex over base names has to
    enumerate spellings and always misses one: this corpus writes
    `nn.Module`, `torch.nn.modules.Module`, `t.nn.Module` (import torch as t),
    and subclasses of other nn classes (`WeighedL1Loss(L1Loss)`,
    `SELoss(nn.MSELoss)`) -- six rows were lost to that.  The code is exec'd
    anyway, so the classes are right there and `issubclass` settles it.

    The last one defined wins: the corpus puts helpers first and the model last."""
    if ns is None:
        ns = {}
        try: exec(src, ns)
        except Exception: return _model_class_regex(src)
    order = {n: i for i, (n, _) in enumerate(re.findall(r"class (\w+)\(([^)]*)\)", src))}
    cands = [n for n, v in ns.items()
             if isinstance(v, type) and issubclass(v, torch.nn.Module)
             and v is not torch.nn.Module and n in order]
    if not cands: return _model_class_regex(src)
    return max(cands, key=lambda n: order[n])


def _model_class_regex(src):
    """Fallback for source that will not exec (a missing third-party import)."""
    decls = re.findall(r"class (\w+)\(([^)]*)\)", src)
    if not decls: return None
    base_is_module = lambda b: any(x.strip().split(".")[-1] == "Module" for x in b.split(","))
    known, out = set(), []
    for name, bases in decls:
        if base_is_module(bases) or any(x.strip() in known for x in bases.split(",")):
            known.add(name); out.append(name)
    return out[-1] if out else None


def bind_wrapper(entry, model, inputs):
    """Match the generated wrapper's parameters to forward's inputs and the
    module's own state.  Positional tensor inputs first, then anything whose
    name matches a parameter/buffer/attribute of the module."""
    sig = inspect.signature(entry)
    params = dict(model.named_parameters()); bufs = dict(model.named_buffers())
    def lookup(name):
        for d in (params, bufs):
            if name in d: return d[name]
            for k, v in d.items():                       # last path component
                if k.split(".")[-1] == name: return v
        if hasattr(model, name): return getattr(model, name)
        return None
    args, kwargs, it = [], {}, iter(inputs)
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD): continue
        v = lookup(pname)
        if v is None:
            try: v = next(it)
            except StopIteration:
                if p.default is inspect.Parameter.empty:
                    raise TypeError(f"cannot bind wrapper parameter `{pname}`")
                v = p.default
        # A parameter after `*` is KEYWORD-ONLY.  Passing those positionally is
        # what `takes 2 positional arguments but 5 were given` was: three trace
        # rows died on it and were charged to the judge as ERROR.
        if p.kind is p.KEYWORD_ONLY: kwargs[pname] = v
        else: args.append(v)
    return args, kwargs


def call_wrapper(entry, model, inputs):
    """Bind and call.  Every caller used to spell this `entry(*bind_wrapper(...))`,
    which is why the keyword-only bug had four places to hide."""
    args, kwargs = bind_wrapper(entry, model, inputs)
    return entry(*args, **kwargs)


CORPORA = {
    "traces":    dict(path="data/triton_traces.json",    out="results/triton_traces.jsonl"),
    "multiturn": dict(path="data/triton_multiturn.json", out="results/triton_multiturn.jsonl"),
}
# carried into the record so the report can condition on them
EXTRA = ("num_turns", "stop_reason")


def build(r):
    """Row -> Candidate, or (None, rec) naming the bucket this row falls into."""
    rec = {"key": r["sample_key"], "label": bool(r["result_correctness"]),
           "speedup": r.get("result_speedup")}
    rec.update({k: r[k] for k in EXTRA if k in r})
    ns = {}; exec(r["pytorch_code"], ns)
    cname = model_class(r["pytorch_code"], ns)
    if cname is None: return None, dict(rec, verdict="NO-MODEL")
    rec["model"] = cname
    ia, ik = ns["get_init_inputs"]()
    torch.manual_seed(0); model = ns[cname](*ia, **ik).cuda().eval()
    torch.manual_seed(1); inputs = [x.cuda() if torch.is_tensor(x) else x for x in ns["get_inputs"]()]

    try:
        ns2 = load_mod(r["triton_code"], r["sample_key"])
    except (SyntaxError, IndentationError) as e:
        # the generated file is not even Python.  That is a property of the row,
        # not a limit of the judge -- all seven such rows are labelled incorrect by
        # the corpus itself -- so it gets its own bucket rather than counting as
        # one of our errors.
        return None, dict(rec, verdict="KERNEL-BROKEN", reason=f"{type(e).__name__}: {str(e)[:70]}")
    entry = ns2.get("triton_kernel_wrapper")
    if entry is None:
        cands = [v for k, v in ns2.items() if callable(v) and not k.startswith("_")
                 and getattr(v, "__module__", "").startswith("tr_") and not hasattr(v, "run")]
        entry = cands[-1] if cands else None
    if entry is None: return None, dict(rec, verdict="NO-ENTRY")
    try: bind_wrapper(entry, model, inputs)
    except TypeError as e: return None, dict(rec, verdict="ADAPTER", reason=str(e)[:80])

    # the wrapper reads the reference module's own parameter tensors, so there is
    # nothing to push after the judge randomises them: sync stays None
    return Candidate(name=cname, model=model, inputs=inputs, ns=ns2,
                     run=lambda xs: call_wrapper(entry, model, list(xs)),
                     meta=rec), None


ROW_TIMEOUT = int(os.environ.get("TVJ_ROW_TIMEOUT", 120))   # see judge.BUDGET

def judge_row(r, timeout=None):
    base = {"key": r["sample_key"], "label": bool(r["result_correctness"])}
    signal.alarm(60)                       # building the candidate must not hang either
    try:
        cand, bucket = build(r)
    except Timeout:
        return dict(base, verdict="TIMEOUT", reason="building the candidate")
    except Exception as e:
        return dict(base, verdict="ERROR", reason=f"{type(e).__name__}: {str(e)[:80]}")
    finally: signal.alarm(0)
    return bucket if cand is None else judge(cand, timeout=timeout or ROW_TIMEOUT)


if __name__ == "__main__":
    argv = sys.argv[1:]
    corpus = "traces"
    if argv and argv[0] == "--corpus": corpus, argv = argv[1], argv[2:]
    C = CORPORA[corpus]
    sys.argv = ["traces_run.py"] + argv
    rows = [r for r in json.load(open(C["path"])) if r["source"] == "kernelbook"]
    if len(sys.argv) > 2 and sys.argv[1] == "--rows":              # targeted re-run: --rows 42,53
        todo = [(i, rows[i]) for i in (int(x) for x in sys.argv[2].split(","))]
        path = C["out"].replace(".jsonl", "_rows.jsonl")
    else:
        # `start count`, the same convention as kernelbook_run.py.  It used to be
        # `start end` here, and run_resume.sh -- written against the other one --
        # silently asked for rows[81:75], got nothing, stepped over the row, and
        # did that for all 75 remaining rows without a word.
        lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        n = int(sys.argv[2]) if len(sys.argv) > 2 else len(rows) - lo
        todo = list(enumerate(rows[lo:lo + n], lo))
        if not todo:
            print(f"empty range: start={lo} count={n} over {len(rows)} rows", flush=True)
        path = C["out"]
    os.makedirs("results", exist_ok=True)
    out = open(path, "a")
    for i, r in todo:
        rec = judge_row(r); rec["i"] = i
        out.write(json.dumps(rec) + "\n"); out.flush()
        if "device-side assert" in (rec.get("reason") or "") or "CUDA error" in (rec.get("reason") or ""):
            # see the note in kernelbook_run.py: the context is gone, not the row
            print(f"\n!! row {i} poisoned the CUDA context -- stopping (an earlier row may be the culprit: the assert is asynchronous). "
                  f"Restart with:  python3 traces_run.py {i+1} {len(rows)}", flush=True); break
        flag = "  <<< label=correct, judge=FAIL" if (rec.get("label") and rec["verdict"] == "FAIL") else ""
        print(f"[{i:3d}] {rec['verdict']:<18} label={str(rec.get('label')):<5} tol={str(rec.get('tol')):<5} "
              f"d4={str(rec.get('d4')):<5} {rec.get('model','?')[:22]:<22} "
              f"{rec.get('reason', rec.get('prec',''))[:60]}{flag}", flush=True)
    out.close()

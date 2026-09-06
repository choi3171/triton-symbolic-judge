"""Run the judge over KernelBook rows: (PyTorch module, Inductor-generated Triton).

Per row: tolerance test (Model vs ModelNew on GPU), then the three-obligation
judge: capture the real launches -> symbolic execution -> compare against the
module's forward run on symbolic inputs.  Every place the pipeline cannot go is
a bucket, and the bucket histogram is the coverage measurement.
"""
import json, signal, sys, time, traceback, collections, torch, os, importlib.util, math
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
def import_triton_code(src, tag):
    """@triton.jit needs real source lines: materialise the row as a module file."""
    path = f"data/kb_mods/kb_{tag}.py"
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location(f"kb_{tag}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.__dict__
import terms as T, volta_bridge as V, ranges as R, numeric as NUM, casesplit as CS
from sexec import RANK, TOP
_INV = {v: k for k, v in RANK.items()}
VOLTA_BUDGET = 200_000_000
from capture import capture, Launch, Extern, roles_of, symbolic_run, base_of, physical_offsets
from spec import STensor, symbolic_module
from sexec import Unsupported

class Timeout(Exception): pass
def _alarm(*a): raise Timeout()
signal.signal(signal.SIGALRM, _alarm)

def first(x): return x[0] if isinstance(x, (tuple, list)) else x

def judge_row(r, per_row_timeout=150):
    rec = {"name": r["entry_point"], "repo": r["repo_name"], "i": r.get("i", 0)}
    signal.alarm(per_row_timeout)
    try:
        ns = {}; exec(r["python_code"], ns)
        Model, get_inputs, get_init = ns[r["entry_point"]], ns["get_inputs"], ns["get_init_inputs"]
        ns2 = import_triton_code(r["triton_code"], rec.get("i", abs(hash(r["triton_code"])) % 10**8))
        ModelNew = ns2[r["entry_point"] + "New"]
        init_args, init_kw = get_init()
        torch.manual_seed(0); model = Model(*init_args, **init_kw).cuda().eval()
        torch.manual_seed(0); model_new = ModelNew(*init_args, **init_kw).cuda().eval()
        inc = model_new.load_state_dict(model.state_dict(), strict=False)
        rec["sd_missing"] = len(inc.missing_keys) + len(inc.unexpected_keys)
        torch.manual_seed(1); inputs = [x.cuda() if torch.is_tensor(x) else x for x in get_inputs()]
        rec["in_shapes"] = [tuple(x.shape) for x in inputs if torch.is_tensor(x)]
        with torch.no_grad(): ref = first(model(*inputs))
        with torch.no_grad(): out_all, calls = capture(lambda *xs: model_new(*xs), *inputs, ns=ns2)
        out = first(out_all)
        with torch.no_grad(): out2 = first(model_new(*inputs))
        rec["det"] = bool(torch.allclose(out.float(), out2.float(), rtol=0, atol=0, equal_nan=True)) if torch.is_tensor(out) else None
        rec["tol"] = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3, equal_nan=True)) if torch.is_tensor(out) else None
        rec["max_abs_diff"] = float((out.float() - ref.float()).abs().nan_to_num(0).max()) if torch.is_tensor(out) else None
        rec["nan"] = int(torch.isnan(out.float()).sum()) if torch.is_tensor(out) else 0
        rec["launches"] = len(calls)
        pm = max([float(p.detach().abs().max()) for p in model.parameters() if p.numel()] or [0.0])
        rec["param_max"] = pm
        # uninitialised parameters (torch.Tensor(shape), reset_parameters that does nothing):
        # a second construction under the same seed yields different values -> the
        # tolerance test compares garbage with garbage and is vacuous.
        torch.manual_seed(0); model_b = Model(*init_args, **init_kw)
        seeded = all(torch.equal(p.detach().cpu(), q.detach().cpu()) for p, q in zip(model.parameters(), model_b.parameters()))
        if not math.isfinite(pm) or pm > 1e6 or (pm == 0.0 and any(p.numel() for p in model.parameters())) or not seeded:
            rec["degenerate_params"] = True
        if not any(ev[0] != "extern" for ev in calls): rec["verdict"] = "NO-KERNEL"; return rec
        # roles by storage
        roles = {}
        for j, x in enumerate(inputs):
            if torch.is_tensor(x): roles[base_of(x)] = f"in{j}"
        for n, p in model_new.named_parameters(): roles[base_of(p)] = "p_" + n
        for n, b in model_new.named_buffers(): roles[base_of(b)] = "b_" + n
        roles[base_of(out)] = "out"
        Ls = [Extern(ev[1], ev[2], ev[3], roles, ev[4] if len(ev) > 4 else None) if ev[0] == "extern" else Launch(*ev, roles) for ev in calls]
        rec["kernels"] = [L.fn.__name__ if isinstance(L, Launch) else "extern:" + L.name for L in Ls]
        rec["externs"] = sum(isinstance(L, Extern) for L in Ls)
        t0 = time.time(); grid, it = symbolic_run(Ls); rec["t_exec"] = round(time.time() - t0, 2)
        rec["mem_errors"] = len(grid.errors)
        races = [e for e in grid.errors if e[0] == "read-write-race"]
        if races: rec["verdict"] = "UNKNOWN"; rec["reason"] = f"read-write race inside one launch: {races[0][1]}[{races[0][2]}] written by program {races[0][3]}, read by {races[0][4]}"; return rec
        # outputs depending on memory nobody wrote = uninitialised or random source
        unwritten = set()
        def _scan(t, seen):
            if t.uid in seen: return
            seen.add(t.uid)
            if isinstance(t, T.Sym) and not (t.buf.startswith("in") or t.buf.startswith("p_") or t.buf.startswith("b_") or t.buf == "ln2"): unwritten.add(t.buf)
            for a in getattr(t, "args", ()): _scan(a, seen)
        seen = set()
        for p in physical_offsets(out):
            if ("out", p) in grid.store: _scan(grid.store[("out", p)], seen)
        if unwritten:
            rec["verdict"] = "UNKNOWN"; rec["reason"] = f"output depends on a buffer no launch wrote ({sorted(unwritten)[0]}): random or uninitialised memory"; return rec
        # spec
        smodel = symbolic_module(model)
        sin = [STensor.input(f"in{j}", tuple(x.shape)) if torch.is_tensor(x) else x for j, x in enumerate(inputs)]
        _rng = {n: getattr(torch, n) for n in ("randn", "rand", "randint", "normal", "bernoulli", "randperm", "randn_like", "rand_like")}
        def _no_rng(name): return lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError(f"spec front-end: unsupported torch op {name} (nondeterministic)"))
        try:
            for n in _rng: setattr(torch, n, _no_rng(n))
            spec = first(smodel(*sin))
        except NotImplementedError as e:
            rec["verdict"] = "SPEC-UNSUPPORTED"; rec["reason"] = str(e).replace("spec front-end: unsupported torch op ", ""); return rec
        except Exception as e:
            rec["verdict"] = "SPEC-ERROR"; rec["reason"] = f"{type(e).__name__}: {str(e)[:120]}"; return rec
        finally:
            for n, f in _rng.items(): setattr(torch, n, f)
        if not isinstance(spec, STensor): spec = STensor(spec)
        spec_flat = spec.flat()
        phys = physical_offsets(out)
        if len(phys) != len(spec_flat):
            rec["verdict"] = "SPEC-ERROR"; rec["reason"] = f"shape mismatch spec {spec.shape} vs out {tuple(out.shape)}"; return rec
        missing = [p for p in phys if ("out", p) not in grid.store]
        pairs = [(spec_flat[i], grid.store[("out", p)]) for i, p in enumerate(phys) if ("out", p) in grid.store]
        ac = sum(a is b for a, b in pairs)
        rec["ac"], rec["n_out"], rec["missing"] = ac, len(phys), len(missing)
        if missing: rec["verdict"] = "FAIL"; rec["reason"] = f"{len(missing)} outputs never written"; return rec
        # --- obligation 2: precision.  Spec terms are exact; the kernel may not be less
        #     precise than fp32 (its inputs/reference precision).
        kterms = [grid.store[("out", p)] for p in phys]
        pmin = min(it.dom.p(t) for t in kterms) if kterms else TOP
        rec["prec_min"] = _INV.get(pmin, str(pmin))
        # --- obligation 3: float-validity preconditions at |inputs| <= 1 and <= 10,
        #     kernel vs spec (the spec is the reference form; the kernel must not be worse).
        inr = {b: None for b in grid.bufsize}
        pre = {}
        for rg in ((-1.0, 1.0), (-10.0, 10.0)):
            _, ck, _ = R.check_store({i: t for i, t in enumerate(kterms)}, {b: rg for b in inr})
            _, cs, _ = R.check_store({i: t for i, t in enumerate(spec_flat)}, {b: rg for b in inr})
            pre[str(int(rg[1]))] = {"kernel": ck, "spec": cs}
        rec["pre"] = pre
        fatal = ("overflow", "div-by-zero")
        rec["pre_worse"] = any(pre[r]["kernel"].get(f, 0) > pre[r]["spec"].get(f, 0) for r in pre for f in fatal)
        if ac == len(pairs): rec["verdict"] = "PASS"; rec["via"] = "AC"; return rec
        try:
            res, st = V.equivalent([pr for pr in pairs if pr[0] is not pr[1]], budget=VOLTA_BUDGET)
        except V.Unsupported as e:
            rec["verdict"] = "UNKNOWN"; rec["reason"] = str(e); return rec
        n_err = sum(isinstance(r, str) for r in res)
        rec["volta_secs"] = round(st["secs"], 3)
        if n_err: rec["verdict"] = "UNKNOWN"; rec["reason"] = [r for r in res if isinstance(r, str)][0][:120]; return rec
        asked = [pr for pr in pairs if pr[0] is not pr[1]]
        unproved = [pr for pr, r in zip(asked, res) if r is False]
        # stage 3: piecewise terms Volta canonicalises as uninterpreted atoms
        pw = [pr for pr in unproved if CS.has_piecewise(pr[0]) or CS.has_piecewise(pr[1])]
        if pw:
            cs = CS.equivalent(pw)
            decided = {id(pr) for pr, r in zip(pw, cs) if r is True}
            rec["casesplit"] = f"{len(decided)}/{len(pw)}"
            unproved = [pr for pr in unproved if id(pr) not in decided]
        if not unproved: rec["verdict"] = "PASS"; rec["via"] = "Volta"; return rec
        # Volta's `false` is 'not provable in its theory'; a numeric witness makes it a counterexample
        try:
            wit = NUM.witness(unproved, grid.bufsize)
        except ValueError as e:
            rec["verdict"] = "UNKNOWN"; rec["reason"] = f"{len(unproved)} outputs not provably equal; no numeric witness possible ({e})"; return rec
        bad = [(pr, w) for pr, w in zip(unproved, wit) if not w[0]]
        if bad:
            (a, b), (_, (si, va, vb)) = bad[0]
            rec["verdict"] = "FAIL"; rec["via"] = "Volta+witness"
            rec["reason"] = f"{len(bad)} outputs differ: witness at random point {si}: spec={va:.6g} kernel={vb:.6g}"
            rec["witness"] = {"n_differ": len(bad), "spec": va, "kernel": vb, "sample": si}
        else:
            rec["verdict"] = "UNKNOWN"; rec["via"] = "Volta"
            rec["reason"] = f"{len(unproved)} outputs not provably equal but numerically equal on 4 random points (max/min incompleteness?)"
        return rec
    except Timeout: rec["verdict"] = "TIMEOUT"; return rec
    except Unsupported as e: rec["verdict"] = "KERNEL-UNSUPPORTED"; rec["reason"] = str(e); return rec
    except NotImplementedError as e:
        rec["verdict"] = "KERNEL-UNSUPPORTED"; rec["reason"] = str(e).split("|")[0].strip()[:80]; return rec
    except Exception as e:
        rec["verdict"] = "ERROR"; rec["reason"] = f"{type(e).__name__}: {str(e)[:160]}"; return rec
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    rows = json.load(open("data/kernelbook_400.json"))
    if len(sys.argv) > 2 and sys.argv[1] == "--rows":          # targeted re-run: --rows 3,17,42
        idx = [int(x) for x in sys.argv[2].split(",")]
        start, n = f"rows{idx[0]}", len(idx)
        todo = [(i, rows[i]) for i in idx]
    else:
        start, n = int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 50
        todo = list(enumerate(rows[start:start+n], start))
    recs = []
    for i, r in todo:
        T.reset()
        rec = judge_row(dict(r, i=i)); rec["i"] = i; recs.append(rec)
        open("data/kb_live.jsonl", "a").write(json.dumps(rec) + "\n")
        flags = ("" if rec.get("det", True) else " NONDET") + (" DEGEN" if rec.get("degenerate_params") else "") + (f" prec<{rec['prec_min']}" if rec.get("prec_min") not in (None, "exact", "ieee", "f32") else "") + (" PRE!" if rec.get("pre_worse") else "") + (f" SD!{rec['sd_missing']}" if rec.get("sd_missing") else "") + (f" ext{rec['externs']}" if rec.get("externs") else "") + (f" nan{rec['nan']}" if rec.get("nan") else "")
        tolstr = f"tol={rec.get('tol')!s:<5}" + (f"({rec['max_abs_diff']:.0e})" if rec.get("tol") is False else "       ")
        print(f"[{i:3d}] {rec['verdict']:<18} {tolstr}{flags:<12} {r['entry_point']:<26} {rec.get('reason', rec.get('via', ''))}", flush=True)
    json.dump(recs, open(f"data/kb_{start}_{n}.json", "w"), indent=1)
    open("data/kb_partial.jsonl", "a").write("".join(json.dumps(x) + "\n" for x in recs))
    c = collections.Counter(x["verdict"] for x in recs)
    print("\n== verdicts ==", dict(c))
    reasons = collections.Counter((x["verdict"], x.get("reason", "")[:60]) for x in recs if x["verdict"] not in ("PASS",))
    for (v, why), k in reasons.most_common(15): print(f"  {k:3d}  {v:<18} {why}")
    tol_vs = collections.Counter((x.get("tol"), x["verdict"]) for x in recs)
    print("== tolerance vs judge ==", dict(tol_vs))

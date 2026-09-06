"""Judge LLM-generated Triton kernels against their PyTorch references.

Corpus: ppbhatt500/kernelbook-triton-reasoning-traces.  Each row carries the
reference nn.Module, an LLM-written Triton kernel, and the dataset's own
tolerance verdict.  The question is whether any row it labels *correct* fails
one of the obligations.

Only `source == "kernelbook"` rows are run: the 14 `kernelbench` rows include a
6.4 GB ReLU and would OOM an 8 GB card.
"""
import json, re, os, sys, time, signal, inspect, importlib.util, collections, math, torch
import terms as T, volta_bridge as V, ranges as R, numeric as NUM, casesplit as CS
from capture import capture, Launch, Extern, symbolic_run, base_of, physical_offsets
from spec import STensor, symbolic_module
import torchtrace as TT
from sexec import Unsupported, RANK, TOP
try: from sexec import RANK_NAME as INV
except ImportError: INV = {4:"exact",3:"f32",2:"tf32x3",1:"tf32",0:"bf16",-1:"fp8"}

os.makedirs("data/tr_mods", exist_ok=True)
BUDGET = 200_000_000

class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))

def load_mod(src, tag):
    p = f"data/tr_mods/tr_{tag}.py"; open(p, "w").write(src)
    s = importlib.util.spec_from_file_location(f"tr_{tag}", p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.__dict__

def model_class(src):
    """The module the reference defines.  Accept `nn.Module`, bare `Module`,
    extra bases (`nn.Module, ABC`), and subclasses of an earlier class in the
    same file -- the corpus uses all four."""
    decls = re.findall(r"class (\w+)\(([^)]*)\)", src)
    if not decls: return None
    base_is_module = lambda b: any(x.strip() in ("nn.Module", "torch.nn.Module", "Module")
                                   for x in b.split(","))
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
    args, it, used_inputs = [], iter(inputs), 0
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD): continue
        v = lookup(pname)
        if v is not None:
            args.append(v); continue
        try:
            args.append(next(it)); used_inputs += 1; continue
        except StopIteration: pass
        if p.default is not inspect.Parameter.empty: args.append(p.default); continue
        raise TypeError(f"cannot bind wrapper parameter `{pname}`")
    return args

def first(x): return x[0] if isinstance(x, (tuple, list)) else x

def witness_tensors(point, model, inputs):
    """Materialise a witness point as real tensors: symbols are named `in<j>[i]`
    and `p_<name>[i]`, so a counterexample found over the reals can be handed to
    the GPU verbatim."""
    xs = []
    for j, x in enumerate(inputs):
        if not torch.is_tensor(x): xs.append(x); continue
        vals = point.get(f"in{j}")
        t = torch.tensor(vals[:x.numel()], dtype=torch.float32).reshape(x.shape).cuda() if vals \
            else torch.zeros_like(x)
        xs.append(t.to(x.dtype))
    with torch.no_grad():
        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            for pre in ("p_", "b_"):
                vals = point.get(pre + n)
                if vals and p.numel() <= len(vals):
                    p.copy_(torch.tensor(vals[:p.numel()], dtype=torch.float32).reshape(p.shape).to(p.dtype))
    return xs

def gpu_confirm(model, entry, mk_inputs, trials=3, seed=0, dists=("signed", "positive")):
    """Run reference and kernel at random parameters AND random inputs, and
    return the largest disagreement.

    This CORROBORATES a value counterexample; it is not a gate for the other
    obligations.  Hardware is structurally silent for them: a stale-buffer read
    returns the right answer, a tf32 downgrade is ignored on sm_75 and hides
    under atol elsewhere, and a narrowed validity radius only shows at extreme
    inputs.  Gating those on reproduction would discard exactly the defects that
    testing cannot reach -- which is the whole reason the judge exists."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.numel(): p.copy_((torch.rand(p.shape, generator=g) * 2 - 1).to(p.device, p.dtype))
    worst, valid = 0.0, 0
    for dist in dists:
        # some kernels are only defined on positive inputs (anything with a log);
        # compare wherever BOTH sides are finite rather than discarding the run
        for _ in range(trials):
            xs = mk_inputs(g, dist)
            try:
                with torch.no_grad():
                    a = first(model(*xs)).float()
                    b = first(entry(*bind_wrapper(entry, model, xs))).float()
                ok = torch.isfinite(a) & torch.isfinite(b)
                if not bool(ok.any()): continue
                valid += 1
                worst = max(worst, float((a - b).abs()[ok].max()))
            except Exception: continue
        if valid: break
    return (worst if valid else None), valid

def diff_symbols(spec, kern):
    """Which named buffers the disagreement rests on, split by side.

    `only_spec` are inputs or parameters the reference uses and the kernel does
    not -- the axes a test must vary.  `only_kern` names buffers the kernel reads
    that the reference has no counterpart for: scratch nobody wrote."""
    def syms(ts):
        acc, seen = set(), set()
        def go(t):
            if t is None or t.uid in seen: return
            seen.add(t.uid)
            if isinstance(t, T.Sym): acc.add(t.buf)
            for a in getattr(t, "args", ()): go(a)
        for t in ts: go(t)
        return acc
    a, b = syms(spec), syms(kern)
    return sorted(a - b), sorted(b - a)

def sym_domain(*term_lists):
    """Buffers referenced by ANY of these terms, sized to their largest index.
    The kernel's `bufsize` is not enough: the spec may name a parameter the
    kernel received as a scalar (a 0-d nn.Parameter), which appears on one side
    only."""
    dom, seen = {}, set()
    def go(t):
        if t is None or t.uid in seen: return
        seen.add(t.uid)
        if isinstance(t, T.Sym): dom[t.buf] = max(dom.get(t.buf, 0), t.idx + 1)
        for a in getattr(t, "args", ()): go(a)
    for lst in term_lists:
        for t in lst: go(t)
    return dom

def tolerance(ref_fn, hack_fn, mk_inputs, trials=5, flip=False, atol=1e-2, rtol=1e-2):
    """Returns (ok, worst, err). ok is None when the comparison itself could not run."""
    ok, worst = True, 0.0
    for t in range(trials):
        torch.manual_seed(200 + t)
        ins = mk_inputs()
        if flip: ins = [(-x if torch.is_tensor(x) and x.is_floating_point() else x) for x in ins]
        try:
            a, b = first(ref_fn(*ins)), first(hack_fn(*ins))
            a, b = a.float(), b.float()
            worst = max(worst, float((a - b).abs().nan_to_num(1e30).max()))
            ok &= bool(torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
        except Exception as e:
            return None, worst, f"{type(e).__name__}: {str(e)[:60]}"
    return ok, worst, None

def judge_row(r, timeout=120):
    rec = {"key": r["sample_key"], "label": bool(r["result_correctness"]),
           "speedup": r.get("result_speedup")}
    signal.alarm(timeout)
    confirm = {}
    def fail(obligation, reason, point=None, **extra):
        """Report a FAIL.

        For the VALUE obligation the counterexample is a concrete input point, so
        it must be reproducible: if the GPU agrees there, our terms are wrong and
        the verdict is downgraded.  For memory / precision / precondition there is
        no such point -- hardware may be silent by construction -- so we report
        the verdict and attach what hardware said as extra information."""
        rec.update(extra); rec["obligation"] = obligation
        m, e, mk = confirm.get("model"), confirm.get("entry"), confirm.get("mk")
        if m is None or obligation != "value":
            rec["verdict"] = "FAIL"; rec["reason"] = reason
            if m is not None:
                worst, _ = gpu_confirm(m, e, mk)
                rec["gpu_maxdiff"] = worst
                rec["reason"] = reason + (f"; GPU also shows {worst:.4g}" if worst and worst > 1e-4
                                          else "; hardware is silent here (expected for this obligation)")
            return rec
        worst = None
        if point is not None:
            try:
                xs = witness_tensors(point, m, confirm["inputs"])
                with torch.no_grad():
                    a = first(m(*xs)).float(); b = first(e(*bind_wrapper(e, m, xs))).float()
                ok = torch.isfinite(a) & torch.isfinite(b)
                if bool(ok.any()): worst = float((a - b).abs()[ok].max())
            except Exception: worst = None
        if worst is None:                      # fall back to random points
            worst, _ = gpu_confirm(m, e, mk)
        rec["gpu_maxdiff"] = worst
        if worst is not None and worst <= 1e-4:
            rec["verdict"] = "UNKNOWN"
            rec["reason"] = f"unreproducible on hardware (GPU max diff {worst:.2g}) -- our modelling gap, not a defect"
        else:
            rec["verdict"] = "FAIL"
            rec["reason"] = reason + (f"; GPU reproduces at {worst:.4g}" if worst else "; hardware inconclusive")
        return rec
    try:
        T.reset()
        ns = {}; exec(r["pytorch_code"], ns)
        cname = model_class(r["pytorch_code"])
        if cname is None: rec["verdict"] = "NO-MODEL"; return rec
        rec["model"] = cname
        ia, ik = ns["get_init_inputs"]()
        torch.manual_seed(0); model = ns[cname](*ia, **ik).cuda().eval()
        # 0-d / single-element parameters are often passed to the kernel BY VALUE
        # (`scale.item()`).  Give them distinct random values so the value can be
        # matched back to its role instead of being frozen as a constant -- and so
        # that identity-element defaults (bias=0, rate=1) cannot hide a kernel that
        # ignores the parameter.
        pnames = {n for n, _ in model.named_parameters()}
        scalar_syms, g = {}, torch.Generator(device="cpu").manual_seed(12345)
        with torch.no_grad():
            for n, p in list(model.named_parameters()) + list(model.named_buffers()):
                if p is None or p.numel() != 1: continue
                # integer-typed scalar parameters count too: the spec symbolises every
                # parameter regardless of dtype, and wrappers pass them as float(x.item())
                if p.is_floating_point():
                    try: p.fill_(float(torch.rand(1, generator=g)) * 1.7 + 0.13)
                    except Exception: pass
                stored = float(p.reshape(-1)[0])            # what .item() will actually return
                scalar_syms[round(stored, 12)] = ("p_" if n in pnames else "b_") + n
        torch.manual_seed(1); inputs = [x.cuda() if torch.is_tensor(x) else x for x in ns["get_inputs"]()]
        rec["shapes"] = [tuple(x.shape) for x in inputs if torch.is_tensor(x)]
        ns2 = load_mod(r["triton_code"], r["sample_key"])
        entry = ns2.get("triton_kernel_wrapper")
        if entry is None:
            cands = [v for k, v in ns2.items() if callable(v) and not k.startswith("_")
                     and getattr(v, "__module__", "").startswith("tr_") and not hasattr(v, "run")]
            entry = cands[-1] if cands else None
        if entry is None: rec["verdict"] = "NO-ENTRY"; return rec

        try: bound = bind_wrapper(entry, model, inputs)
        except TypeError as e: rec["verdict"] = "ADAPTER"; rec["reason"] = str(e)[:80]; return rec

        with torch.no_grad(): ref = first(model(*inputs))
        tr = TT.TorchTrace()
        with torch.no_grad(): out_all, calls = capture(lambda *a: entry(*a), *bound, ns=ns2, trace=tr)
        out = first(out_all)
        if not torch.is_tensor(out): rec["verdict"] = "NON-TENSOR"; return rec
        rec["launches"] = sum(1 for c in calls if c[0] != "extern")
        rec["externs"] = sum(1 for c in calls if c[0] == "extern")

        # the corpus' own check, and its sign-flip mitigation
        mk = lambda: [torch.rand(x.shape, device="cuda") if torch.is_tensor(x) else x for x in inputs]
        confirm["model"], confirm["entry"], confirm["inputs"] = model, entry, inputs
        confirm["mk"] = lambda g, dist="signed": [
            ((torch.rand(x.shape, generator=g) * (2 if dist == "signed" else 1) - (1 if dist == "signed" else 0)).cuda()
             if torch.is_tensor(x) else x) for x in inputs]
        ref_fn = lambda *a: model(*a)
        hack_fn = lambda *a: entry(*bind_wrapper(entry, model, list(a)))
        with torch.no_grad():
            rec["tol"], rec["maxdiff"], terr = tolerance(ref_fn, hack_fn, mk)
            rec["d4"], _, _ = tolerance(ref_fn, hack_fn, mk, flip=True)
            if terr: rec["tol_error"] = terr

        if not calls: rec["verdict"] = "NO-KERNEL"; return rec

        roles = {base_of(x): f"in{j}" for j, x in enumerate(inputs) if torch.is_tensor(x)}
        roles.update({base_of(p): "p_" + n for n, p in model.named_parameters()})
        roles.update({base_of(b): "b_" + n for n, b in model.named_buffers()})
        # an in-place kernel returns one of its inputs: do not rename that storage,
        # or every read of it becomes a read of an "unwritten" output buffer
        ob = base_of(out)
        if ob in roles: out_role = roles[ob]; rec["out_aliases"] = out_role
        else: out_role = roles[ob] = "out"
        evs = [Extern(e[1], e[2], e[3], roles, e[4] if len(e) > 4 else None) if e[0] == "extern"
               else Launch(*e, roles, scalar_syms) for e in calls]
        t0 = time.time(); grid, it = symbolic_run(evs); rec["t_exec"] = round(time.time() - t0, 2)
        rec["mem_errors"] = len(grid.errors)

        phys = physical_offsets(out)
        kterms = [grid.store.get((out_role, p)) for p in phys]
        unwritten = set()
        seen = set()
        def scan(t):
            if t is None or t.uid in seen: return
            seen.add(t.uid)
            if isinstance(t, T.Sym) and t.buf != out_role and not (t.buf.startswith(("in", "p_", "b_")) or t.buf == "ln2"):
                unwritten.add(t.buf)
            for a in getattr(t, "args", ()): scan(a)
        for t in kterms: scan(t)
        if unwritten:
            rec["unwritten_buffers"] = sorted(unwritten)
            rec["diff_symbols"] = {"only_spec": [], "only_kernel": sorted(unwritten)}
            return fail("memory", f"output depends on a buffer no launch wrote ({sorted(unwritten)[0]})")

        # symbolic_module replaces _parameters in place, which would make the
        # model unrunnable -- and gpu_confirm needs to run it afterwards
        import copy as _copy
        sm = symbolic_module(_copy.deepcopy(model))
        sin = [STensor.input(f"in{j}", tuple(x.shape)) if torch.is_tensor(x) else x
               for j, x in enumerate(inputs)]
        _rng = {n: getattr(torch, n) for n in ("randn","rand","randint","normal","bernoulli","randn_like","rand_like")}
        def norng(nm): return lambda *a, **k: (_ for _ in ()).throw(NotImplementedError(f"{nm} (nondeterministic)"))
        try:
            for n in _rng: setattr(torch, n, norng(n))
            spec = first(sm(*sin))
        except NotImplementedError as e:
            rec["verdict"] = "SPEC-UNSUPPORTED"; rec["reason"] = str(e).replace("spec front-end: unsupported torch op ", "")[:70]; return rec
        except Exception as e:
            rec["verdict"] = "SPEC-ERROR"; rec["reason"] = f"{type(e).__name__}: {str(e)[:70]}"; return rec
        finally:
            for n, f in _rng.items(): setattr(torch, n, f)
        if not isinstance(spec, STensor): spec = STensor(spec)
        sf = spec.flat()
        if len(sf) != len(phys):
            rec["verdict"] = "SPEC-ERROR"; rec["reason"] = f"shape {spec.shape} vs out {tuple(out.shape)}"; return rec
        missing = [i for i, t in enumerate(kterms) if t is None]
        if missing:
            # the wrapper may finish in PyTorch; replay its recorded torch ops,
            # seeded with the inputs, the parameters, and whatever the kernels wrote
            seed = {}
            for j, x in enumerate(inputs):
                if torch.is_tensor(x): seed[id(x)] = STensor.input(f"in{j}", tuple(x.shape))
            for n, p in model.named_parameters(): seed[id(p)] = STensor.input("p_" + n, tuple(p.shape))
            for n, b in model.named_buffers():    seed[id(b)] = STensor.input("b_" + n, tuple(b.shape))
            for name, a, k, o in tr.events:
                for t in list(a) + list(k.values()) + [o]:
                    if torch.is_tensor(t) and id(t) not in seed:
                        st = TT.tensor_terms(t, roles, grid)
                        if st is not None: seed[id(t)] = st
            tail = TT.replay(tr.events, seed, out)
            if tail is not None and len(tail.flat()) == len(phys):
                kterms = tail.flat(); missing = []; rec["tail"] = "torch-replay"
        if missing:
            wrote_elsewhere = sum(1 for (b, _) in grid.store if b != "out")
            if len(missing) == len(kterms) and wrote_elsewhere:
                # every output element unwritten but the kernels did write somewhere:
                # the wrapper finishes the computation in PyTorch (e.g. a two-stage
                # reduction whose tail is partial_sums.sum()).  Not a defect -- the
                # judge simply cannot see past the captured launches.
                rec["verdict"] = "UNKNOWN"; rec["obligation"] = "coverage"
                rec["reason"] = f"output produced by torch after the kernels ({wrote_elsewhere} elements written to scratch)"
                return rec
            return fail("value", f"{len(missing)}/{len(kterms)} outputs never written")

        # --- value ---
        diff = [i for i in range(len(sf)) if sf[i] is not kterms[i]]
        rec["ac"] = len(sf) - len(diff)
        if diff:
            try: res, st = V.equivalent([(sf[i], kterms[i]) for i in diff], budget=BUDGET)
            except V.Unsupported as e:
                rec["verdict"] = "UNKNOWN"; rec["reason"] = str(e)[:70]; return rec
            errs = [x for x in res if isinstance(x, str)]
            if errs: rec["verdict"] = "UNKNOWN"; rec["reason"] = errs[0][:70]; return rec
            unproved = [i for i, x in zip(diff, res) if x is False]
            if unproved:
                # stage 3: Volta canonicalises `select` as an uninterpreted atom, so it
                # cannot relate two piecewise functions written differently.  Hand those
                # to Z3, which case-splits; Volta has already normalised the arithmetic.
                pw = [i for i in unproved if CS.has_piecewise(sf[i]) or CS.has_piecewise(kterms[i])]
                if pw:
                    cs = CS.equivalent([(sf[i], kterms[i]) for i in pw])
                    decided = {i for i, r in zip(pw, cs) if r is True}
                    rec["casesplit"] = f"{len(decided)}/{len(pw)} decided by case split"
                    unproved = [i for i in unproved if i not in decided]
            if unproved:
                dom = dict(grid.bufsize); dom.update(sym_domain(sf, kterms))
                wit_points = [NUM.random_point(dom, l, h, seed=s) for s, (l, h) in
                              enumerate([(-1.,1.), (0.05,1.), (0.5,2.), (-1.,1.), (0.05,1.), (0.5,2.)])]
                try: wit = NUM.witness([(sf[i], kterms[i]) for i in unproved], dom)
                except ValueError as e:
                    rec["verdict"] = "UNKNOWN"; rec["reason"] = f"no numeric witness ({e})"[:70]; return rec
                if all(w[0] is None for w in wit):
                    rec["verdict"] = "UNKNOWN"
                    rec["reason"] = f"{len(unproved)} outputs unproved; no input point leaves both sides finite"
                    return rec
                bad = [(i, w) for i, w in zip(unproved, wit) if w[0] is False]
                if bad:
                    i, (_, (s, va, vb)) = bad[0]
                    only_spec, only_kern = diff_symbols([sf[i] for i, _ in bad], [kterms[i] for i, _ in bad])
                    pt = wit_points[s] if s < len(wit_points) else None
                    return fail("value", f"{len(bad)} outputs differ; witness spec={va:.6g} kernel={vb:.6g}",
                                point=pt, diff_symbols={"only_spec": only_spec, "only_kernel": only_kern},
                                witness_point={k: v[:8] for k, v in (pt or {}).items()},
                                witness={"n": len(bad), "spec": va, "kernel": vb})
                rec["verdict"] = "UNKNOWN"; rec["reason"] = f"{len(unproved)} unprovable but numerically equal"; return rec
        # --- precision ---
        pr, pm = TOP, min((it.dom.p(t) for t in kterms), default=TOP)
        if pm < pr:
            return fail("precision", f"{INV.get(pr,pr)} -> {INV.get(pm,pm)}")
        rec["prec"] = INV.get(pm, pm)
        rec["verdict"] = "PASS"
        return rec
    except TO: rec["verdict"] = "TIMEOUT"; return rec
    except Unsupported as e: rec["verdict"] = "KERNEL-UNSUPPORTED"; rec["reason"] = str(e)[:80]; return rec
    except Exception as e: rec["verdict"] = "ERROR"; rec["reason"] = f"{type(e).__name__}: {str(e)[:80]}"; return rec
    finally: signal.alarm(0)

if __name__ == "__main__":
    rows = [r for r in json.load(open("data/triton_traces.json")) if r["source"] == "kernelbook"]
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(rows)
    out = open("results/triton_traces.jsonl", "a")
    for i, r in enumerate(rows[lo:hi], lo):
        rec = judge_row(r); rec["i"] = i
        out.write(json.dumps(rec) + "\n"); out.flush()
        flag = "  <<< label=correct, judge=FAIL" if (rec["label"] and rec["verdict"] == "FAIL") else ""
        print(f"[{i:3d}] {rec['verdict']:<18} label={str(rec['label']):<5} tol={str(rec.get('tol')):<5} "
              f"d4={str(rec.get('d4')):<5} {rec.get('model','?')[:22]:<22} {rec.get('reason', rec.get('prec',''))[:60]}{flag}", flush=True)
    out.close()

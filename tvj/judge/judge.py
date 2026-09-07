"""The judge: one candidate kernel against one PyTorch reference.

Both corpora feed this.  A corpus adapter's only job is to build a `Candidate` --
a runnable reference module, a callable that runs the generated kernel on a list
of inputs, and those inputs.  Everything after that is shared:

    capture the real launches -> symbolic execution over the reals
    -> build the spec by running the module on symbolic inputs
    -> five obligations.

  memory        reads nothing a launch never wrote; writes every output; no
                read-write race inside one launch
  value         same expression over the reals   (AC -> Volta -> Z3 case split)
  precision     not less precise than the reference (exact > ieee > tf32x3 >
                tf32 > f16 > bf16)
  precondition  float-validity radius not narrower than the reference form's
  accuracy      not a numerically worse arrangement of the same real expression

A FAIL on `value` carries a concrete counterexample, so it must reproduce on the
GPU at that point or it is downgraded to UNKNOWN -- every false positive this
project produced was caught there.  The other obligations are deliberately NOT
gated that way: hardware is structurally silent for them (a stale buffer holds
the right answer, tf32 is ignored on sm_75, a narrowed validity radius shows
only at extreme inputs), so gating would discard exactly the defects a test
cannot reach, which is the whole reason the judge exists.
"""
import collections, copy, inspect, signal, time, torch
from dataclasses import dataclass, field
from typing import Any, Callable

from tvj.core import terms as T
from tvj.decide import volta_bridge as V
from tvj.decide import ranges as R
from tvj.decide import numeric as NUM
from tvj.decide import casesplit as CS
from tvj.decide import accuracy as ACC
from tvj.front import torchtrace as TT
from tvj.front.capture import capture, Launch, Extern, symbolic_run, base_of, physical_offsets, root_storage
from tvj.front import spec as SPEC
from tvj.decide import delegate as DEL
from tvj.front.spec import STensor, symbolic_module
from tvj.core.sexec import Unsupported, TermBudget, RANK, TOP
try: from tvj.core.sexec import RANK_NAME as INV
except ImportError: INV = {v: k for k, v in RANK.items()}

BUDGET = 200_000_000
# What the reference's output dtype says its own precision contract is.
REF_RANK = {torch.float64: TOP, torch.float32: TOP,
            torch.float16: RANK["f16"], torch.bfloat16: RANK["bf16"]}
FATAL = ("overflow", "div-by-zero")          # precondition flags that make a result meaningless

class Timeout(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(Timeout()))
TO = Timeout                                  # traces_run imported this name


@dataclass
class Candidate:
    """What a corpus adapter must produce.

    `run(inputs)` must be callable repeatedly with fresh tensors: the hardware
    gate re-runs it at the witness point and at random points.  `sync` is called
    after the judge mutates the reference's parameters, for corpora where the
    candidate owns a separate copy of them (KernelBook's `ModelNew`); for a bare
    wrapper that reads the reference's own tensors it is None.
    """
    name: str
    model: torch.nn.Module                  # the reference; its parameters ARE mutated
    run: Callable[[list], Any]              # the generated kernel, given a list of inputs
    inputs: list
    ns: dict = None                         # generated module namespace (extern_kernels lives here)
    sync: Callable[[], None] = None         # push reference parameters into the candidate
    param_source: torch.nn.Module = None    # module whose storages carry the parameter roles
    meta: dict = field(default_factory=dict)

    def push(self):
        if self.sync is not None: self.sync()

    @property
    def params(self):
        return self.param_source if self.param_source is not None else self.model


def first(x): return x[0] if isinstance(x, (tuple, list)) else x


def prepare_scalars(model):
    """0-d / single-element parameters are often passed to a kernel BY VALUE
    (`scale.item()`).  Give them distinct random values so the value can be
    matched back to its role instead of frozen as a literal -- and so that
    identity-element defaults (bias=0, rate=1) cannot hide a kernel that ignores
    the parameter entirely.  Returns {rounded value: role name}."""
    pnames = {n for n, _ in model.named_parameters()}
    syms, g = {}, torch.Generator(device="cpu").manual_seed(12345)
    with torch.no_grad():
        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            if p is None or p.numel() != 1: continue
            # integer-typed scalars count too: the spec symbolises every parameter
            # regardless of dtype, and wrappers pass them as float(x.item())
            if p.is_floating_point():
                try: p.fill_(float(torch.rand(1, generator=g)) * 1.7 + 0.13)
                except Exception: pass
            syms[round(float(p.reshape(-1)[0]), 12)] = ("p_" if n in pnames else "b_") + n
    return syms


def make_probe(cand):
    """Would the reference even accept these inputs?

    Several torch CUDA kernels carry a device-side assert on their input domain
    -- BCE wants [0,1], embedding wants an index below num_embeddings, scatter
    and gather want an index in range.  A device-side assert is NOT recoverable:
    it puts the whole process's CUDA context into a permanent error state, and it
    is asynchronous, so the failure surfaces on some later row.  One run lost 114
    consecutive rows to a single one.

    And the judge is the one producing out-of-domain inputs: the D4 sign flip
    negates them (KernelBench-Verified's own mitigation), `gpu_confirm` draws from
    U[-1,1], and a witness point is whatever the solver returned.  So run the
    reference on the CPU first, where the same condition raises an ordinary,
    catchable RuntimeError, and only then touch the GPU.

    16 of 400 KernelBook rows and 2 of 156 trace rows are within reach of one of
    these asserts."""
    try:
        cpu = copy.deepcopy(cand.model).cpu()
    except Exception:
        return lambda xs: True
    def probe(xs):
        try:
            cpu.load_state_dict(cand.model.state_dict())
        except Exception: pass
        try:
            with torch.no_grad():
                cpu(*[x.detach().cpu() if torch.is_tensor(x) else x for x in xs])
            return True
        except Exception:
            return False
    return probe


def witness_tensors(point, cand):
    """Materialise a witness point as real tensors: symbols are named `in<j>[i]`
    and `p_<name>[i]`, so a counterexample found over the reals can be handed to
    the GPU verbatim."""
    xs = []
    for j, x in enumerate(cand.inputs):
        if not torch.is_tensor(x): xs.append(x); continue
        vals = point.get(f"in{j}")
        t = torch.tensor(vals[:x.numel()], dtype=torch.float32).reshape(x.shape).cuda() if vals \
            else torch.zeros_like(x)
        xs.append(t.to(x.dtype))
    m = cand.model
    with torch.no_grad():
        for n, p in list(m.named_parameters()) + list(m.named_buffers()):
            for pre in ("p_", "b_"):
                vals = point.get(pre + n)
                if vals and p.numel() <= len(vals):
                    p.copy_(torch.tensor(vals[:p.numel()], dtype=torch.float32).reshape(p.shape).to(p.dtype))
    cand.push()
    return xs


def gpu_confirm(cand, mk_inputs, trials=3, seed=0, dists=("signed", "positive"), probe=None):
    """Run reference and candidate at random parameters AND random inputs, and
    return the largest disagreement.

    This CORROBORATES a value counterexample; it is not a gate for the other
    obligations -- see the module docstring."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in cand.model.parameters():
            if p.numel(): p.copy_((torch.rand(p.shape, generator=g) * 2 - 1).to(p.device, p.dtype))
    cand.push()
    worst, valid = 0.0, 0
    for dist in dists:
        # some kernels are only defined on positive inputs (anything with a log);
        # compare wherever BOTH sides are finite rather than discarding the run
        for _ in range(trials):
            xs = mk_inputs(g, dist)
            if probe is not None and not probe(xs): continue     # outside the reference's domain
            try:
                with torch.no_grad():
                    a = first(cand.model(*xs)).float()
                    b = first(cand.run(xs)).float()
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
    not -- the axes a generated test must vary.  `only_kernel` names buffers the
    kernel reads that the reference has no counterpart for: scratch nobody wrote."""
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


def tolerance(ref_fn, cand_fn, mk_inputs, trials=5, flip=False, atol=1e-2, rtol=1e-2, probe=None):
    """The corpus' own kind of check, plus its sign-flip mitigation.
    Returns (ok, worst, err); ok is None when the comparison could not run."""
    ok, worst, ran = True, 0.0, 0
    for t in range(trials):
        torch.manual_seed(200 + t)
        ins = mk_inputs()
        if flip: ins = [(-x if torch.is_tensor(x) and x.is_floating_point() else x) for x in ins]
        if probe is not None and not probe(ins): continue         # see make_probe
        ran += 1
        try:
            a, b = first(ref_fn(ins)), first(cand_fn(ins))
            a, b = a.float(), b.float()
            worst = max(worst, float((a - b).abs().nan_to_num(1e30).max()))
            ok &= bool(torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
        except Exception as e:
            return None, worst, f"{type(e).__name__}: {str(e)[:60]}"
    if not ran: return None, worst, "every trial fell outside the reference's input domain"
    return ok, worst, None


# ---------------------------------------------------------------------------

def judge(cand, timeout=150, tol_trials=5):
    """All five obligations for one candidate.  Never raises: every way out is a
    verdict, and the histogram of verdicts is the coverage measurement."""
    rec = {"name": cand.name}
    rec.update(cand.meta)
    signal.alarm(timeout)
    gate = {}                                   # populated once the GPU run has happened
    probe = make_probe(cand)                    # keeps our own inputs off the GPU's asserts

    def accuracy_gpu(regime):
        """Run both sides at the regime the accuracy obligation fired in.

        This obligation is the one place where "hardware is silent" was the wrong
        thing to say: it is silent at the benchmark's inputs, which is the whole
        point, and NOT silent at the shifted inputs that made the layer fire.  Three
        LLM-written tanh kernels spell it `(exp(2x)-1)/(exp(2x)+1)`, which is exact
        over the reals and every float32 value above x = 44.4 is a NaN."""
        shift = dict((n, s) for n, s, _ in ACC.REGIMES).get(regime)
        if shift is None: return None
        g = torch.Generator().manual_seed(3)
        xs = [((torch.rand(x.shape, generator=g) + shift).cuda().to(x.dtype)
               if torch.is_tensor(x) else x) for x in cand.inputs]
        if not probe(xs): return None
        try:
            with torch.no_grad():
                a = first(cand.model(*xs)).float(); b = first(cand.run(xs)).float()
            bad = (~torch.isfinite(b)) & torch.isfinite(a)
            if bool(bad.any()): return float("inf")
            ok = torch.isfinite(a) & torch.isfinite(b)
            return float((a - b).abs()[ok].max()) if bool(ok.any()) else None
        except Exception:
            return None

    def fail(obligation, reason, point=None, **extra):
        rec.update(extra); rec["obligation"] = obligation
        if not gate or obligation != "value":
            rec["verdict"] = "FAIL"; rec["reason"] = reason
            if gate and obligation == "accuracy":
                worst = accuracy_gpu((rec.get("accuracy") or {}).get("regime"))
                rec["gpu_maxdiff"] = worst
                rec["reason"] = reason + (
                    "; GPU reproduces it: the kernel is non-finite there and the reference is not"
                    if worst == float("inf") else
                    f"; GPU shows {worst:.4g} at that regime" if worst and worst > 1e-4 else
                    "; hardware is silent even at that regime")
            elif gate:
                worst, _ = gpu_confirm(cand, gate["mk"], probe=probe)
                rec["gpu_maxdiff"] = worst
                rec["reason"] = reason + (f"; GPU also shows {worst:.4g}" if worst and worst > 1e-4
                                          else "; hardware is silent here (expected for this obligation)")
            return rec
        worst = None
        if point is not None:
            try:
                xs = witness_tensors(point, cand)
                if not probe(xs): raise ValueError("witness point outside the reference's domain")
                with torch.no_grad():
                    a = first(cand.model(*xs)).float(); b = first(cand.run(xs)).float()
                ok = torch.isfinite(a) & torch.isfinite(b)
                if bool(ok.any()): worst = float((a - b).abs()[ok].max())
            except Exception: worst = None
        if worst is None:                       # fall back to random points
            worst, _ = gpu_confirm(cand, gate["mk"], probe=probe)
        rec["gpu_maxdiff"] = worst
        if worst is None:
            # the gate could not run at all -- every trial raised, or the probe
            # rejected every input.  The contract is that a value FAIL is believed
            # only when hardware reproduces it, so with no hardware answer there is
            # no FAIL to report.
            rec["verdict"] = "UNKNOWN"
            rec["reason"] = reason + "; but the hardware check could not run, so this is not corroborated"
        elif worst <= 1e-4:
            rec["verdict"] = "UNKNOWN"
            rec["reason"] = f"unreproducible on hardware (GPU max diff {worst:.2g}) -- our modelling gap, not a defect"
        else:
            rec["verdict"] = "FAIL"
            rec["reason"] = reason + f"; GPU reproduces at {worst:.4g}"
        return rec

    try:
        T.reset(); DEL.reset()
        model, inputs = cand.model, cand.inputs
        scalar_syms = prepare_scalars(model); cand.push()
        rec["shapes"] = [tuple(x.shape) for x in inputs if torch.is_tensor(x)]

        # smoke-run the reference before anything else: a reference that cannot run
        # at all is an ERROR on the row, not a judgement about the kernel.  Its dtype
        # is also what the precision obligation is measured against.
        with torch.no_grad(): ref = first(model(*inputs))
        ref_dtype = ref.dtype if torch.is_tensor(ref) else torch.float32
        tr = TT.TorchTrace()
        with torch.no_grad():
            out_all, calls = capture(lambda: cand.run(inputs), ns=cand.ns, trace=tr)
        out = first(out_all)
        if not torch.is_tensor(out): rec["verdict"] = "NON-TENSOR"; return rec
        rec["launches"] = sum(1 for c in calls if c[0] != "extern")
        rec["externs"] = sum(1 for c in calls if c[0] == "extern")

        # determinism: a second run must agree bit for bit, or nothing below means
        # anything.  This used to be recorded and never acted on -- a kernel whose
        # two runs differed still flowed through every obligation and could PASS.
        with torch.no_grad(): out2 = first(cand.run(inputs))
        rec["det"] = bool(torch.allclose(out.float(), out2.float(), rtol=0, atol=0, equal_nan=True))
        if not rec["det"]:
            # ... unless the REFERENCE is nondeterministic too, in which case the
            # task itself is, and there is nothing to refine either way
            with torch.no_grad():
                r2 = first(model(*inputs)); r3 = first(model(*inputs))
            ref_det = bool(torch.allclose(r2.float(), r3.float(), rtol=0, atol=0, equal_nan=True))
            rec["ref_det"] = ref_det
            rec["verdict"] = "NONDETERMINISTIC"; rec["obligation"] = "memory"
            rec["reason"] = ("the kernel gives a different answer on a second run at the same "
                             "inputs; the reference does not" if ref_det else
                             "both the kernel and the reference are nondeterministic at fixed inputs")
            return rec

        mk = lambda: [torch.rand(x.shape, device="cuda") if torch.is_tensor(x) else x for x in inputs]
        gate["mk"] = lambda g, dist="signed": [
            ((torch.rand(x.shape, generator=g) * (2 if dist == "signed" else 1) - (1 if dist == "signed" else 0)).cuda()
             if torch.is_tensor(x) else x) for x in inputs]
        with torch.no_grad():
            rec["tol"], rec["maxdiff"], terr = tolerance(lambda xs: model(*xs), cand.run, mk,
                                                         trials=tol_trials, probe=probe)
            rec["d4"], _, _ = tolerance(lambda xs: model(*xs), cand.run, mk,
                                        trials=tol_trials, flip=True, probe=probe)
            if terr: rec["tol_error"] = terr

        if not any(c[0] != "extern" for c in calls): rec["verdict"] = "NO-KERNEL"; return rec

        roles = {base_of(x): f"in{j}" for j, x in enumerate(inputs) if torch.is_tensor(x)}
        roles.update({base_of(p): "p_" + n for n, p in cand.params.named_parameters()})
        roles.update({base_of(b): "b_" + n for n, b in cand.params.named_buffers()})
        # an in-place kernel returns one of its inputs: do not rename that storage,
        # or every read of it becomes a read of an "unwritten" output buffer
        ob = base_of(out)
        if ob in roles: out_role = roles[ob]; rec["out_aliases"] = out_role
        else: out_role = roles[ob] = "out"
        evs = [Extern(e[1], e[2], e[3], roles, e[4] if len(e) > 4 else None) if e[0] == "extern"
               else Launch(*e, roles, scalar_syms) for e in calls]
        rec["kernels"] = [L.fn.__name__ if isinstance(L, Launch) else "extern:" + L.name for L in evs]
        t0 = time.time(); grid, it = symbolic_run(evs); rec["t_exec"] = round(time.time() - t0, 2)
        rec["mem_errors"] = len(grid.errors)

        # --- obligation: memory ------------------------------------------------
        # Every kind the interpreter records, not just races.  `check.py` fails a
        # kernel on ANY entry in `g.errors` -- that is how `bug_splitk_store` is
        # caught -- and this path used to look at `read-write-race` alone, so a
        # corpus row with a write-conflict or an out-of-bounds access could reach
        # PASS while the same kernel failed check.py.  A conflict keeps the first
        # writer's term, and if that one happens to match the reference nothing
        # else in the pipeline would ever notice.
        if grid.errors:
            kinds = collections.Counter(e[0] for e in grid.errors)
            rec["mem_error_kinds"] = dict(kinds)
            race = next((e for e in grid.errors if e[0] == "read-write-race"), None)
            if race is not None:
                # a read of another program's store inside one launch: the value
                # depends on scheduling, so there is nothing to refine
                rec["verdict"] = "UNKNOWN"; rec["obligation"] = "memory"
                rec["reason"] = (f"read-write race inside one launch: {race[1]}[{race[2]}] "
                                 f"written by program {race[3]}, read by {race[4]}")
                return rec
            e = grid.errors[0]
            return fail("memory", f"{e[0]} at {e[1]}[{e[2]}] "
                                  f"({', '.join(f'{k} x{v}' for k, v in kinds.items())})")

        phys = physical_offsets(out)
        kterms = [grid.store.get((out_role, p)) for p in phys]
        unwritten, seen = set(), set()
        def scan(t):
            if t is None or t.uid in seen: return
            seen.add(t.uid)
            if (isinstance(t, T.Sym) and t.buf != out_role and not DEL.is_delegated(t.buf)
                    and not (t.buf.startswith(("in", "p_", "b_")) or t.buf == "ln2")):
                unwritten.add(t.buf)
            for a in getattr(t, "args", ()): scan(a)
        for t in kterms: scan(t)
        if unwritten:
            rec["unwritten_buffers"] = sorted(unwritten)
            # "no launch wrote it" and "we did not see the write" are different
            # claims.  Generated code also calls `torch.ops.aten.*` directly, and
            # `TorchTrace` recorded every one of those: if a recorded op's output
            # lives in this storage, the write happened and the memory obligation
            # has nothing to say.  Three rows reported a stale-buffer read for a
            # buffer an ordinary aten pooling call had filled.
            # Allocation is not a write.  `empty` / `empty_strided` hand back whatever
            # the allocator had, and reading that is exactly Sakana's stale-buffer
            # exploit -- the one finding this obligation exists for.  Counting them
            # as writes here would bury it.
            ALLOC = ("empty", "empty_like", "empty_strided", "new_empty", "storage",
                     "as_strided", "reinterpret_tensor", "view", "_reshape_alias")
            wrote = {}
            for nm, aa, kk, oo in tr.events:
                if any(nm.split(".")[0] == x for x in ALLOC): continue
                outs = [oo] if torch.is_tensor(oo) else (list(oo) if isinstance(oo, (tuple, list)) else [])
                for t2 in outs:
                    if not torch.is_tensor(t2): continue
                    rr = roles.get(root_storage(base_of(t2)))
                    if rr is not None: wrote.setdefault(rr, nm)
            # A wrapper that builds its own CONSTANT and feeds it to the kernel where
            # the reference reads a parameter is not a memory problem at all: the
            # buffer was written, with a value that has nothing to do with the
            # reference.  `BiasLayer` gets torch.zeros where the module has
            # `self.bias`, and agrees only because the default bias IS zero.  Bind
            # the constant into the terms and let the value obligation say so, with
            # a witness point and an axis a test can be generated from.
            CONSTS = ("zeros", "ones", "full", "tensor", "arange", "eye", "linspace",
                      "zeros_like", "ones_like", "full_like")
            const_tab = {}
            for nm, aa, kk, oo in tr.events:
                if nm.split(".")[0] not in CONSTS or not torch.is_tensor(oo): continue
                rr = roles.get(root_storage(base_of(oo)))
                if rr is None or rr not in unwritten: continue
                try:
                    vals = oo.detach().float().reshape(-1).tolist()
                except Exception: continue
                const_tab[rr] = [T.const(v) for v in vals]
            if const_tab:
                memo = {}
                kterms = [T.substitute(t, const_tab, memo) for t in kterms]
                rec["wrapper_constants"] = {b: len(v) for b, v in const_tab.items()}
                unwritten -= set(const_tab)
                seen.clear()
                for t in kterms: scan(t)
            if not unwritten:
                rec.pop("unwritten_buffers", None)
            hidden = {b: wrote[b] for b in sorted(unwritten) if b in wrote}
            if unwritten and hidden:
                op = sorted(hidden.values())[0]
                rng = any(w in op for w in ("rand", "normal", "dropout", "bernoulli", "multinomial"))
                rec["verdict"] = "NONDETERMINISTIC" if rng else "UNKNOWN"
                rec["obligation"] = "coverage"; rec["uncaptured"] = hidden
                rec["reason"] = (f"the output depends on randomness ({op}); it is not a function "
                                 "of the inputs, so there is nothing to refine"
                                 if rng else
                                 f"part of the output was produced by an op we do not intercept ({op})")
                return rec
            if unwritten:
                rec["diff_symbols"] = {"only_spec": [], "only_kernel": sorted(unwritten)}
                return fail("memory", f"output depends on a buffer no launch wrote ({sorted(unwritten)[0]})")

        # --- the spec ----------------------------------------------------------
        # symbolic_module replaces _parameters in place, which would make the model
        # unrunnable -- and the hardware gate has to run it afterwards
        t0 = time.time()
        sm = symbolic_module(copy.deepcopy(model))
        sin = [STensor.input(f"in{j}", tuple(x.shape)) if torch.is_tensor(x) else x
               for j, x in enumerate(inputs)]
        _rng = {n: getattr(torch, n) for n in
                ("randn", "rand", "randint", "normal", "bernoulli", "randperm", "randn_like", "rand_like")}
        def norng(nm): return lambda *a, **k: (_ for _ in ()).throw(NotImplementedError(f"{nm} (nondeterministic)"))
        SPEC.USED.clear()
        try:
            for n in _rng: setattr(torch, n, norng(n))
            spec = first(sm(*sin))
        except NotImplementedError as e:
            rec["verdict"] = "SPEC-UNSUPPORTED"
            rec["reason"] = str(e).replace("spec front-end: unsupported torch op ", "")[:70]; return rec
        except Timeout: raise            # the alarm is not a spec error
        except Exception as e:
            rec["verdict"] = "SPEC-ERROR"; rec["reason"] = f"{type(e).__name__}: {str(e)[:70]}"; return rec
        finally:
            for n, f in _rng.items(): setattr(torch, n, f)
        rec["t_spec"] = round(time.time() - t0, 2)
        # How strong a claim a FAIL on this row can be.  A reference transcribed
        # from a published definition settles the question; one recovered by
        # reading torch's C++ only says the two disagree -- see spec.provenance.
        rec["ref_basis"], inf = SPEC.basis_of_run()
        if inf: rec["ref_inferred"] = inf
        if not isinstance(spec, STensor): spec = STensor(spec)
        sf = spec.flat()
        if len(sf) != len(phys):
            rec["verdict"] = "SPEC-ERROR"
            rec["reason"] = f"shape {spec.shape} vs out {tuple(out.shape)}"; return rec

        missing = [i for i, t in enumerate(kterms) if t is None]
        if missing:
            # the wrapper may finish in PyTorch; replay its recorded torch ops,
            # seeded with the inputs, the parameters, and whatever the kernels wrote
            seed = {}
            for j, x in enumerate(inputs):
                if torch.is_tensor(x): seed[id(x)] = STensor.input(f"in{j}", tuple(x.shape))
            for n, p in cand.params.named_parameters(): seed[id(p)] = STensor.input("p_" + n, tuple(p.shape))
            for n, b in cand.params.named_buffers():    seed[id(b)] = STensor.input("b_" + n, tuple(b.shape))
            for name, a, k, o in tr.events:
                for t in list(a) + list(k.values()) + [o]:
                    if torch.is_tensor(t) and id(t) not in seed:
                        st = TT.tensor_terms(t, roles, grid)
                        if st is not None: seed[id(t)] = st
            tail = TT.replay(tr.events, seed, out)
            if tail is not None and len(tail.flat()) == len(phys):
                kterms = tail.flat(); missing = []; rec["tail"] = "torch-replay"
        if missing:
            wrote_elsewhere = sum(1 for (b, _) in grid.store if b != out_role)
            if len(missing) == len(kterms) and wrote_elsewhere:
                # every output element unwritten but the kernels did write somewhere:
                # the wrapper finishes the computation in PyTorch (e.g. a two-stage
                # reduction whose tail is partial_sums.sum()).  Not a defect -- the
                # judge simply cannot see past the captured launches.
                rec["verdict"] = "UNKNOWN"; rec["obligation"] = "coverage"
                rec["reason"] = f"output produced by torch after the kernels ({wrote_elsewhere} elements written to scratch)"
                return rec
            return fail("value", f"{len(missing)}/{len(kterms)} outputs never written")

        # --- obligation: value -------------------------------------------------
        def value_pass(sf, kterms):
            """Decide the value obligation for one pair of term lists.

            Returns ("equal", via) | ("unknown", reason) | ("differ", detail).
            Called twice at most: once on the terms as built, and -- if the only
            thing separating them is a delegated library op left uninterpreted --
            once more after cashing those symbols in (delegate.expand)."""
            diff = [i for i in range(len(sf)) if sf[i] is not kterms[i]]
            if not diff: return "equal", "AC"
            try: res, st = V.equivalent([(sf[i], kterms[i]) for i in diff], budget=BUDGET)
            except V.Unsupported as e: return "unknown", str(e)[:70]
            rec["volta_secs"] = round(st["secs"], 3)
            errs = [x for x in res if isinstance(x, str)]
            if errs: return "unknown", errs[0][:70]
            unproved = [i for i, x in zip(diff, res) if x is False]
            via = "Volta"
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
                    if decided: via = "Volta+Z3"
            if not unproved: return "equal", via
            dom = dict(grid.bufsize); dom.update(sym_domain(sf, kterms))
            wit_points = [NUM.random_point(dom, l, h, seed=s) for s, (l, h) in
                          enumerate([(-1., 1.), (0.05, 1.), (0.5, 2.), (-1., 1.), (0.05, 1.), (0.5, 2.)])]
            try: wit = NUM.witness([(sf[i], kterms[i]) for i in unproved], dom)
            except ValueError as e: return "unknown", f"no numeric witness ({e})"[:70]
            if all(w[0] is None for w in wit):
                return "unknown", (f"{len(unproved)} outputs unproved; "
                                   "no input point leaves both sides finite")
            bad = [(i, w) for i, w in zip(unproved, wit) if w[0] is False]
            if not bad:
                return "unknown", f"{len(unproved)} unprovable but numerically equal"
            i, (_, (s, va, vb)) = bad[0]
            only_spec, only_kern = diff_symbols([sf[k] for k, _ in bad], [kterms[k] for k, _ in bad])
            return "differ", {"n": len(bad), "spec": va, "kernel": vb,
                              "point": wit_points[s] if s < len(wit_points) else None,
                              "only_spec": only_spec, "only_kernel": only_kern}

        rec["ac"] = sum(1 for a, b in zip(sf, kterms) if a is b)
        rec["n_out"] = len(sf)
        kind, det = value_pass(sf, kterms)
        if kind == "equal": rec["via"] = det

        if kind == "differ":
            # A delegated op is an UNINTERPRETED symbol (delegate.py): if the two
            # sides carry different ones, the numeric witness gave them independent
            # random values and "they differ" says nothing about the kernel.  Cash
            # the symbols in and decide again -- the shortcut may not turn a pair we
            # would have decided into a counterexample.
            dele = DEL.expandable(set(det["only_spec"]) | set(det["only_kernel"]))
            if dele:
                try:
                    both, cost = DEL.expand(list(sf) + list(kterms))
                    sf, kterms = both[:len(sf)], both[len(sf):]
                    rec["expanded"] = {"ops": sorted(dele), "terms": cost}
                    kind, det = value_pass(sf, kterms)
                    if kind == "equal": rec["via"] = det + " after expansion"
                except DEL.TooLarge as e:
                    rec["verdict"] = "UNKNOWN"; rec["obligation"] = "value"
                    rec["delegated"] = {"only_spec": det["only_spec"], "only_kernel": det["only_kernel"]}
                    rec["reason"] = f"delegated library op, {e}"[:110]
                    return rec

        if kind == "unknown":
            rec["verdict"] = "UNKNOWN"; rec["reason"] = det; return rec
        if kind == "differ":
            return fail("value",
                        f"{det['n']} outputs differ; witness spec={det['spec']:.6g} "
                        f"kernel={det['kernel']:.6g}",
                        point=det["point"],
                        diff_symbols={"only_spec": det["only_spec"], "only_kernel": det["only_kernel"]},
                        witness_point={k: v[:8] for k, v in (det["point"] or {}).items()},
                        witness={"n": det["n"], "spec": det["spec"], "kernel": det["kernel"]})

        # --- obligation: precision ---------------------------------------------
        # The reference's OWN precision is the bar, not `exact`.  A module whose
        # forward ends in `.half()` is asking for f16, and a kernel that delivers
        # f16 refines it; measuring that against `exact` reports every deliberate
        # low-precision reference as a defect.
        pr = REF_RANK.get(ref_dtype, TOP)
        pm = min((it.dom.p(t) for t in kterms), default=TOP)
        rec["prec"], rec["prec_ref"] = INV.get(pm, pm), INV.get(pr, pr)
        if pm < pr:
            return fail("precision", f"{INV.get(pr, pr)} -> {INV.get(pm, pm)}")

        # --- obligation: precondition ------------------------------------------
        #     The kernel must not raise a fatal float condition (overflow, division
        #     by zero) at an input range where the reference form does not.
        t0 = time.time(); pre = {}
        for rg in ((-1.0, 1.0), (-10.0, 10.0)):
            dom = {b: rg for b in set(grid.bufsize) | set(sym_domain(sf, kterms))}
            _, ck, _ = R.check_store(dict(enumerate(kterms)), dom)
            _, cs, _ = R.check_store(dict(enumerate(sf)), dom)
            pre[str(int(rg[1]))] = {"kernel": ck, "spec": cs}
        rec["pre"] = pre; rec["t_pre"] = round(time.time() - t0, 2)
        worse = [(r, f) for r in pre for f in FATAL
                 if pre[r]["kernel"].get(f, 0) > pre[r]["spec"].get(f, 0)]
        if worse:
            rec["pre_worse"] = True
            r, f = worse[0]
            return fail("precondition", f"{f} at |input| <= {r} where the reference form has none")

        # --- obligation: accuracy ----------------------------------------------
        #     Same real expression, numerically worse form.  Cancellation has no
        #     counterpart over the reals, so neither the value obligation nor the
        #     range analysis can see it; compare the float32 error of both sides at
        #     regimes that stress it.
        t0 = time.time()
        # The domain has to cover BOTH sides, exactly as the precondition obligation
        # above does: the reference may name a parameter the kernel took as a scalar,
        # and evaluating a term whose buffer is missing raises a KeyError that used to
        # be swallowed into "skip" -- leaving the row to PASS with this obligation
        # never run and nothing in the record to say so.
        adom = dict(grid.bufsize); adom.update(sym_domain(sf, kterms))
        try:
            av, ad = ACC.compare(sf, kterms, adom)
        except Timeout: raise            # the alarm is not an accuracy result
        except Exception as e:
            av, ad = "skip", f"{type(e).__name__}: {str(e)[:60]}"
        rec["t_acc"] = round(time.time() - t0, 2)
        # A skipped obligation is not a passed one, and the record has to be able
        # to tell them apart or the coverage number quietly counts one as the other.
        if av == "skip": rec["accuracy_skipped"] = ad
        elif av in ("worse", "noted"): rec["accuracy"] = ad
        if av == "worse":
            return fail("accuracy", f"numerically worse than the reference at {ad['regime']}: "
                                    f"relative error {ad['spec_rel_err']:.2g} -> {ad['kernel_rel_err']:.2g}")
        rec["verdict"] = "PASS"
        return rec
    except Timeout: rec["verdict"] = "TIMEOUT"; return rec
    except TermBudget as e: rec["verdict"] = "TOO-LARGE"; rec["reason"] = str(e); return rec
    except Unsupported as e: rec["verdict"] = "KERNEL-UNSUPPORTED"; rec["reason"] = str(e)[:80]; return rec
    except NotImplementedError as e:
        rec["verdict"] = "KERNEL-UNSUPPORTED"; rec["reason"] = str(e).split("|")[0].strip()[:80]; return rec
    except Exception as e:
        # A kernel that will not compile is a fact about the candidate, not a
        # failure of the judge; the corpus labels every such row incorrect anyway.
        # This has to be the LAST handler: when it sat above the two before it,
        # they were unreachable, the ERROR verdict could never be recorded, and
        # anything not in this tuple propagated out of a function whose contract
        # is that every way out is a verdict -- taking the whole corpus run with it.
        if type(e).__name__ in ("CompilationError", "SyntaxError", "IndentationError"):
            rec["verdict"] = "KERNEL-BROKEN"; rec["reason"] = f"{type(e).__name__}: {str(e)[:70]}"; return rec
        rec["verdict"] = "ERROR"; rec["reason"] = f"{type(e).__name__}: {str(e)[:80]}"; return rec
    finally:
        signal.alarm(0)

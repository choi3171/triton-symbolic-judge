"""Open one corpus row and look at both sides' terms before any decision runs.

    python3 -m tvj.tools.row_terms kb 116
    python3 -m tvj.tools.row_terms traces 11 --profile      # cProfile the spec build

Prints the phase timings the judge would record, the size of the term DAGs, the
function symbols each side uses (a `pow` atom on one side and a Mul on the other
is what "unprovable but numerically equal" looks like), whether an `exp` sits
inside another exp's exponent (outside pit's fragment), how many SHAPES the
outputs fall into, and -- when almost every output is its own shape -- whether
two outputs differ only in the order of an AC node's children.
"""
import sys, time, json, copy, collections, cProfile, pstats, io, torch
from tvj.core import terms as T
from tvj.measure import lanes, pit
from tvj.decide import delegate as DEL
from tvj.front.capture import capture, Launch, Extern, symbolic_run, base_of, physical_offsets
from tvj.front.spec import STensor, symbolic_module
from tvj.front import torchtrace as TT
from tvj.judge.judge import prepare_scalars, first

corpus, i = sys.argv[1], int(sys.argv[2]); prof = "--profile" in sys.argv
REPR = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--repr=")), 260)
if corpus == "kb":
    from tvj.judge import kernelbook_run as R
    r = json.load(open("data/kernelbook_400.json"))[i]; cand = R.build(r)
else:
    from tvj.judge import traces_run as R
    rows = [x for x in json.load(open("data/triton_traces.json")) if x["source"] == "kernelbook"]
    cand, bucket = R.build(rows[i]); assert cand is not None, bucket
print(f"== {corpus} row {i}: {cand.name} ==")
T.reset(); DEL.reset()
model, inputs = cand.model, cand.inputs
scalar_syms = prepare_scalars(model); cand.push()
with torch.no_grad(): ref = first(model(*inputs))
tr = TT.TorchTrace(); t0 = time.time()
with torch.no_grad(): out_all, calls = capture(lambda: cand.run(inputs), ns=cand.ns, trace=tr)
out = first(out_all); t_cap = time.time() - t0
roles = {base_of(x): f"in{j}" for j, x in enumerate(inputs) if torch.is_tensor(x)}
roles.update({base_of(p): "p_" + n for n, p in cand.params.named_parameters()})
roles.update({base_of(b): "b_" + n for n, b in cand.params.named_buffers()})
ob = base_of(out); out_role = roles[ob] if ob in roles else roles.setdefault(ob, "out")
evs = [Extern(e[1], e[2], e[3], roles, e[4] if len(e) > 4 else None) if e[0] == "extern"
       else Launch(*e, roles, scalar_syms) for e in calls]
t0 = time.time(); grid, it = symbolic_run(evs); t_exec = time.time() - t0
phys = physical_offsets(out); kterms = [grid.store.get((out_role, p)) for p in phys]
t0 = time.time(); sm = symbolic_module(copy.deepcopy(model))
sin = [STensor.input(f"in{j}", tuple(x.shape)) if torch.is_tensor(x) else x for j, x in enumerate(inputs)]
pr = cProfile.Profile()
if prof: pr.enable()
try: spec = first(sm(*sin))
finally:
    if prof: pr.disable()
t_spec = time.time() - t0
sf = (spec if isinstance(spec, STensor) else STensor(spec)).flat()
print(f"phases: capture {t_cap:.1f}s  symbolic_run {t_exec:.1f}s  spec {t_spec:.1f}s   "
      f"launches {sum(1 for c in calls if c[0] != 'extern')} externs {sum(1 for c in calls if c[0] == 'extern')}  outputs {len(sf)}")
if prof:
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
    print("--- spec build, top by tottime ---"); print("\n".join(s.getvalue().splitlines()[6:24]))

def scan(terms):
    seen, fns, nested, n = set(), collections.Counter(), 0, 0
    def go(t, in_exp):
        nonlocal nested, n
        if t is None or t.uid in seen: return
        seen.add(t.uid); n += 1
        if isinstance(t, T.App):
            fns[t.fn] += 1
            if t.fn == "exp" and in_exp: nested += 1
            for a in t.args: go(a, in_exp or t.fn == "exp")
        else:
            for a in getattr(t, "args", ()): go(a, in_exp)
    for t in terms: go(t, False)
    return n, fns, nested

missing = sum(1 for t in kterms if t is None)
if missing:
    # the wrapper finished in torch: replay its recorded ops, exactly as judge() does
    seed = {}
    for j, x in enumerate(inputs):
        if torch.is_tensor(x): seed[id(x)] = STensor.input(f"in{j}", tuple(x.shape))
    for n, pp in cand.params.named_parameters(): seed[id(pp)] = STensor.input("p_" + n, tuple(pp.shape))
    for n, bb in cand.params.named_buffers():    seed[id(bb)] = STensor.input("b_" + n, tuple(bb.shape))
    for name, a, k, o in tr.events:
        for t in list(a) + list(k.values()) + [o]:
            if torch.is_tensor(t) and id(t) not in seed:
                st = TT.tensor_terms(t, roles, grid)
                if st is not None: seed[id(t)] = st
    tail = TT.replay(tr.events, seed, out)
    if tail is not None and len(tail.flat()) == len(phys):
        kterms = tail.flat(); print(f"kernel: {missing}/{len(kterms)} outputs came from the torch tail (replayed)")
    else:
        print(f"kernel: {missing}/{len(kterms)} outputs never written and the tail did not replay")
for side, ts in (("spec", sf), ("kernel", [t for t in kterms if t is not None])):
    n, fns, nested = scan(ts)
    print(f"{side:7} distinct nodes {n:>9,}   out[0] {T.size(ts[0]) if ts else 0:>8,} nodes   "
          f"nested exp {nested}   fns {dict(fns.most_common(9))}")

pairs = [(a, b) for a, b in zip(sf, kterms) if b is not None]
same = sum(1 for a, b in pairs if a is b)
groups = lanes.group(pairs)
print(f"AC-identical {same}/{len(pairs)}   shapes {len(groups)} over {len(pairs)} pairs")

def blind(t, memo):
    r = memo.get(t.uid)
    if r is not None: return r
    if isinstance(t, T.Sym): r = "L"
    elif isinstance(t, T.Const): r = ("C", t.v)
    else: r = (type(t).__name__, getattr(t, "fn", None), tuple(blind(a, memo) for a in t.args))
    memo[t.uid] = r; return r

if len(groups) > max(2, len(pairs) // 4) and len(pairs) > 1:
    a0, a1 = kterms[0], kterms[1]
    b0, b1 = blind(a0, {}), blind(a1, {})
    print(f"kernel out[0] vs out[1]: leaf-blind shape {'EQUAL' if b0 == b1 else 'differs'}; "
          f"roots {type(a0).__name__}/{type(a1).__name__}, arity {len(getattr(a0,'args',()))}/{len(getattr(a1,'args',()))}")
    if b0 != b1:
        def sig(t): return sorted(collections.Counter(type(x).__name__ + ":" + str(getattr(x, "fn", "")) for x in getattr(t, "args", ())).items())
        print(f"  root child kinds: {sig(a0)}  vs  {sig(a1)}")
        print(f"  out[0]: {repr(a0)[:200]}"); print(f"  out[1]: {repr(a1)[:200]}")
    else:
        print("  -> same structure up to leaves: the shape key is missing AC-canonical child order")

if "--diff-lanes" in sys.argv and len(pairs) > 1:
    # which SIDE makes out[0] and out[1] different shapes, and where
    from tvj.measure.lanes import Shapes
    for side, ts in (("spec", sf), ("kernel", kterms)):
        sh = Shapes(); ids = [sh.of(ts[k], {}, {}) for k in (0, 1)]
        print(f"lane-diff {side:6}: out[0] shape {ids[0]}  out[1] shape {ids[1]}  {'SAME' if ids[0] == ids[1] else 'DIFFERENT'}")
        if ids[0] != ids[1]:
            # walk both in the canonical child order and report the first node that differs
            sh2 = Shapes(); L0, L1, M0, M1 = {}, {}, {}, {}
            def first_diff(u, v, path):
                k0, k1 = sh2.of(u, L0, M0), sh2.of(v, L1, M1)
                if k0 == k1: return None
                if type(u) is not type(v) or getattr(u, "fn", None) != getattr(v, "fn", None): return path + [f"kind {type(u).__name__}/{getattr(u,'fn','')} vs {type(v).__name__}/{getattr(v,'fn','')}"]
                if isinstance(u, T.Sym): return path + [f"leaf {u} (#{L0[u.uid]}) vs {v} (#{L1[v.uid]})"]
                if isinstance(u, T.Const): return path + [f"const {u.v} vs {v.v}"]
                cu, cv = sh2._children(u), sh2._children(v)
                if len(cu) != len(cv): return path + [f"arity {len(cu)} vs {len(cv)}"]
                for n, (cu_i, cv_i) in enumerate(zip(cu, cv)):
                    d = first_diff(cu_i, cv_i, path + [f"{type(u).__name__}/{getattr(u,'fn','')}[{n}]"])
                    if d: return d
                return path + ["(children agree pairwise but keys differ: leaf NUMBERING diverged earlier)"]
            d = first_diff(ts[0], ts[1], [])
            print("   first divergence:", " > ".join(d[-4:]) if d else "?")
            # the ROOT's children in canonical order, per lane: kind, blind id, least, size
            for lane, root in ((0, ts[0]), (1, ts[1])):
                kids = sh2._children(root) if hasattr(root, "args") else ()
                desc = [f"{type(c).__name__}/{getattr(c,'fn','')}:b{sh2._blind(c)}:{str(sh2._least(c))[:34]}:n{T.size(c)}" for c in kids[:4]]
                print(f"   out[{lane}] root {type(root).__name__}/{getattr(root,'fn','')} arity {len(kids)} -> " + " | ".join(desc))
            print(f"   leaves numbered: out[0] {len(L0)}  out[1] {len(L1)}")
rep = [pairs[ix[0]] for ix in groups.values()]
try:
    res, secs = pit.equal(rep, seed=1)
    print(f"pit on {len(rep)} representative(s): {sum(res)} equal, {len(res)-sum(res)} differ  ({secs:.2f}s)")
except pit.Unsupported as e:
    print(f"pit: Unsupported -- {e}")
s0, k0 = pairs[0]
print(f"spec  out[0]: {repr(s0)[:REPR]}"); print(f"kern  out[0]: {repr(k0)[:REPR]}")
# float-validity radius of each side on out[0]: where does the real-number claim stop meaning anything in float32?
from tvj.decide import ranges as R
from tvj.judge.judge import sym_domain
bufs = set(grid.bufsize) | set(sym_domain(sf, [t for t in kterms if t is not None]))
try:
    rs, rk = R.safe_radius({0: s0}, bufs, iters=24), R.safe_radius({0: k0}, bufs, iters=24)
    print(f"float-validity radius (out[0]): spec |in| <= {rs:.4g}   kernel |in| <= {rk:.4g}"
          + ("   <-- kernel narrower: a precondition FAIL the value stage's UNKNOWN hides" if rk < rs * 0.999 else ""))
except Exception as e:
    print(f"safe_radius: {type(e).__name__}: {str(e)[:80]}")

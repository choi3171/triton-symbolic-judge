"""Is a memory FAIL a stale-buffer read, or an op we simply do not hook?

The obligation says "the output depends on a buffer no launch wrote".  Two very
different things produce that sentence:

  REAL   nothing wrote it.  The kernel is passing off whatever the allocator left
         behind -- Sakana's headline exploit, and invisible to any test that
         happens to run after something wrote the right answer there.

  OURS   something DID write it, through a path we do not intercept.  We hook
         `JITFunction.run` and the `extern_kernels` table; generated code also
         calls `torch.ops.aten.*` directly (KernelBook row 131's MinPool goes
         through `aten.max_pool3d_with_indices`), and those writes are invisible
         to the symbolic run even though they are perfectly ordinary.

`TorchTrace` already records every torch op executed during capture, so the two
can be told apart: if a recorded op's OUTPUT lives in the storage the unwritten
buffer names, the write happened and we missed it.

    python3 memcheck.py kb 83 84 131
    python3 memcheck.py traces 42 66 98
"""
import json, sys, copy, torch
from tvj.core import terms as T
from tvj.decide import delegate as D
from tvj.judge.judge import prepare_scalars, first, make_probe
from tvj.front.capture import capture, Launch, Extern, symbolic_run, base_of, physical_offsets, root_storage
from tvj.front.spec import STensor, symbolic_module
from tvj.front import torchtrace as TT


def build(corpus, i):
    if corpus == "kb":
        from tvj.judge import kernelbook_run as R
        return R.build(dict(json.load(open("data/kernelbook_400.json"))[i], i=i))
    from tvj.judge import traces_run as R
    rows = [r for r in json.load(open("data/triton_traces.json")) if r["source"] == "kernelbook"]
    cand, bucket = R.build(rows[i])
    if cand is None: raise RuntimeError(bucket.get("verdict"))
    return cand


def examine(corpus, i):
    T.reset(); D.reset()
    cand = build(corpus, i)
    prepare_scalars(cand.model); cand.push()
    tr = TT.TorchTrace()
    with torch.no_grad():
        out_all, calls = capture(lambda: cand.run(cand.inputs), ns=cand.ns, trace=tr)
    out = first(out_all)
    roles = {base_of(x): f"in{j}" for j, x in enumerate(cand.inputs) if torch.is_tensor(x)}
    roles.update({base_of(p): "p_" + n for n, p in cand.params.named_parameters()})
    roles.update({base_of(b): "b_" + n for n, b in cand.params.named_buffers()})
    ob = base_of(out)
    out_role = roles.get(ob) or roles.setdefault(ob, "out")
    evs = [Extern(e[1], e[2], e[3], roles, e[4] if len(e) > 4 else None) if e[0] == "extern"
           else Launch(*e, roles, {}) for e in calls]
    grid, it = symbolic_run(evs)

    phys = physical_offsets(out)
    kt = [grid.store.get((out_role, p)) for p in phys]
    unwritten, seen = set(), set()
    def scan(t):
        if t is None or t.uid in seen: return
        seen.add(t.uid)
        if (isinstance(t, T.Sym) and t.buf != out_role and not D.is_delegated(t.buf)
                and not (t.buf.startswith(("in", "p_", "b_")) or t.buf == "ln2")):
            unwritten.add(t.buf)
        for a in getattr(t, "args", ()): scan(a)
    for t in kt: scan(t)

    # storage -> role, so a recorded torch op's output can be matched back
    inv = {}
    for st, role in roles.items(): inv.setdefault(role, st)
    print(f"=== {corpus} row {i}: {cand.name} ===")
    print(f"  captured launches: {[e.fn.__name__ if isinstance(e, Launch) else 'extern:'+e.name for e in evs]}")
    print(f"  unwritten buffers seen in the output: {sorted(unwritten) or 'none'}")
    print(f"  torch ops recorded during capture ({len(tr.events)}):")
    written_by_torch = {}
    for name, a, k, o in tr.events:
        outs = [o] if torch.is_tensor(o) else (list(o) if isinstance(o, (tuple, list)) else [])
        for t in outs:
            if not torch.is_tensor(t): continue
            st = root_storage(base_of(t))
            r = roles.get(st)
            written_by_torch.setdefault(r, []).append((name, tuple(t.shape)))
    for name, a, k, o in tr.events[:14]:
        shp = tuple(o.shape) if torch.is_tensor(o) else "-"
        print(f"      {name:<34} -> {shp}")
    if len(tr.events) > 14: print(f"      ... {len(tr.events)-14} more")
    verdict = []
    for b in sorted(unwritten):
        hits = written_by_torch.get(b) or []
        verdict.append((b, "OURS: written by " + hits[0][0] if hits else "REAL: nothing wrote it"))
    for b, v in verdict: print(f"  -> {b}: {v}")
    if not unwritten: print("  -> no unwritten buffer this time")
    print()


if __name__ == "__main__":
    corpus = sys.argv[1]
    for i in (int(x) for x in sys.argv[2:]):
        try: examine(corpus, i)
        except Exception as e:
            print(f"=== {corpus} row {i}: {type(e).__name__}: {str(e)[:90]}\n")

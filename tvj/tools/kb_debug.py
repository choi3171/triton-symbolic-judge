"""Open one KernelBook row: where do kernel and spec part ways?"""
import json, sys, torch
from tvj.core import terms as T
from tvj.decide import volta_bridge as V
from tvj.front.capture import capture, Launch, Extern, symbolic_run, base_of, physical_offsets
from tvj.front.spec import STensor, symbolic_module
from tvj.judge.kernelbook_run import import_triton_code, first
i = int(sys.argv[1]); r = json.load(open("data/kernelbook_400.json"))[i]
ns = {}; exec(r["python_code"], ns); ns2 = import_triton_code(r["triton_code"], f"dbg{i}")
Model, ModelNew = ns[r["entry_point"]], ns2[r["entry_point"] + "New"]
ia, ik = ns["get_init_inputs"](); torch.manual_seed(0); model = Model(*ia, **ik).cuda().eval()
torch.manual_seed(0); mnew = ModelNew(*ia, **ik).cuda().eval(); inc = mnew.load_state_dict(model.state_dict(), strict=False)
torch.manual_seed(1); inputs = [x.cuda() if torch.is_tensor(x) else x for x in ns["get_inputs"]()]
with torch.no_grad(): ref = first(model(*inputs)); out, calls = capture(lambda *xs: mnew(*xs), *inputs, ns=ns2); out = first(out); out2 = first(mnew(*inputs))
print(f"row {i} {r['entry_point']}  inputs={[tuple(x.shape) for x in inputs if torch.is_tensor(x)]}  sd_missing={inc.missing_keys}")
print(f"  tol max|out-ref| = {float((out.float()-ref.float()).abs().max()):.3e}   det: max|out-out2| = {float((out-out2).abs().max()):.3e}")
roles = {base_of(x): f"in{j}" for j, x in enumerate(inputs) if torch.is_tensor(x)}
roles.update({base_of(p): "p_" + n for n, p in mnew.named_parameters()}); roles.update({base_of(b): "b_" + n for n, b in mnew.named_buffers()}); roles[base_of(out)] = "out"
evs = [Extern(e[1], e[2], e[3], roles, e[4] if len(e) > 4 else None) if e[0] == "extern" else Launch(*e, roles) for e in calls]
print("  events:", [e.fn.__name__ if isinstance(e, Launch) else "extern:" + e.name for e in evs])
for e in evs:
    if isinstance(e, Launch): print(f"    {e.fn.__name__}: grid={e.grid} constexprs={e.constexprs} roles={e.rolemap}")
grid, it = symbolic_run(evs)
print(f"  mem errors={len(grid.errors)} benign-dup={grid.benign} stores={len(grid.store)}")
spec = first(symbolic_module(model)(*[STensor.input(f'in{j}', tuple(x.shape)) if torch.is_tensor(x) else x for j, x in enumerate(inputs)]))
sf = spec.flat(); ph = physical_offsets(out)
k0 = grid.store.get(("out", ph[0])); s0 = sf[0]
print(f"\n  SPEC out[0]  ({T.size(s0)} nodes): {repr(s0)[:420]}")
print(f"\n  KERN out[0]  ({T.size(k0)} nodes): {repr(k0)[:420]}")
open(f"/tmp/kb_dbg_{i}_spec.txt","w").write(repr(s0)); open(f"/tmp/kb_dbg_{i}_kern.txt","w").write(repr(k0))
res, st = V.equivalent([(s0, k0)]); print(f"\n  Volta(out[0]) = {res[0]}")
# sub-term probe: Volta on each top-level factor/term pairing when shapes match
for tag, a, b in (("top", s0, k0),):
    if type(a) == type(b) and hasattr(a, "args") and len(a.args) == len(b.args):
        for j, (x, y) in enumerate(zip(a.args, b.args)):
            rr, _ = V.equivalent([(x, y)]); print(f"    {tag}.arg{j}: Volta={rr[0]}  spec={repr(x)[:90]}  kern={repr(y)[:90]}")

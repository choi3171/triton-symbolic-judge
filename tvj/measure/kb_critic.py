"""KernelBook row 308 (Critic): the second real defect the judge found.

The generated wrapper permutes tensors among roles that share a shape, so
`call()` receives linear2.weight where it expects the first cat operand and the
action tensor where it expects linear2.weight.  Every one of them is (4,4), so
`assert_size_stride` passes.  The dataset's own tolerance test misses it because
`linear3` is initialised to U(-0.003, 0.003): all outputs are ~1e-4, and the
`atol=1e-3` in allclose swallows a 190% relative error.
"""
import json, torch, torch.nn.functional as F
from tvj.judge.kernelbook_run import import_triton_code, first

r = json.load(open("data/kernelbook_400.json"))[308]
ns = {}; exec(r["python_code"], ns)
ns2 = import_triton_code(r["triton_code"], "critic308")
ia, ik = ns["get_init_inputs"]()

# --- attribution: at a random point in [-1,1], what does CriticNew actually compute? ---
g = torch.Generator().manual_seed(0)
m = ns["Critic"](*ia, **ik).cuda().eval(); mn = ns2["CriticNew"](*ia, **ik).cuda().eval()
with torch.no_grad():
    for p in m.parameters(): p.copy_(torch.rand(p.shape, generator=g) * 2 - 1)
mn.load_state_dict(m.state_dict())
st, ac = [(torch.rand(x.shape, generator=g) * 2 - 1).cuda() for x in ns["get_inputs"]()]
W1, b1, W2, b2, R = m.linear1.weight, m.linear1.bias, m.linear2.weight, m.linear2.bias, F.relu
with torch.no_grad():
    got       = first(mn(st, ac))
    scrambled = m.linear3(R(R(torch.cat([W2, st], 1) @ W1.t() + b1) @ ac.t() + b2))
    intended  = m.linear3(R(R(torch.cat([st, ac], 1) @ W1.t() + b1) @ W2.t() + b2))
print("row 308 Critic, random point in [-1,1]:")
print(f"  |CriticNew - cat([W2,state]) @ ... @ action.T |  = {float((got-scrambled).abs().max()):.3g}   <- attributed")
print(f"  |CriticNew - the module it claims to compute  |  = {float((got-intended).abs().max()):.3g}")

# --- why the dataset's tolerance test passes ---
torch.manual_seed(0); m2 = ns["Critic"](*ia, **ik).cuda().eval()
torch.manual_seed(0); mn2 = ns2["CriticNew"](*ia, **ik).cuda().eval(); mn2.load_state_dict(m2.state_dict())
torch.manual_seed(1); xs = [x.cuda() for x in ns["get_inputs"]()]
with torch.no_grad(): a, b = first(m2(*xs)), first(mn2(*xs))
rel = float((a - b).abs().max() / a.abs().max())
print("\ndataset-style check (torch.rand inputs, default init):")
print(f"  outputs ~{float(a.abs().max()):.1e} because linear3 is init'd to U(-0.003, 0.003)")
print(f"  max abs diff {float((a-b).abs().max()):.2e}   relative error {rel:.0%}")
print(f"  allclose(rtol=1e-3, atol=1e-3) -> {torch.allclose(a, b, rtol=1e-3, atol=1e-3)}  (atol swallows the whole signal)")

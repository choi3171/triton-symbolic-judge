"""The same blind spot survives KernelBench-Verified's hardening.

KBV strengthens the check with four input distributions (hidden_tests/*_hidden.py):
  D1 as-is, D2 x3, D3 x0.01, D4 negated.
`_scale()` applies the factor to the tensors returned by get_inputs() only, so
module parameters remain at their default initialisation for every one of them.

Problem 85 is `nn.Parameter(torch.ones(scale_shape))` followed by `x * scale`.
A kernel that deletes the multiply is bit-identical under all four.
"""
import importlib.util, os, torch, torch.nn as nn

KBV = "data/KBV"
PROB = f"{KBV}/KernelBench/level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py"
HID  = f"{KBV}/hidden_tests/level2/85_hidden.py"

def load(path, name):
    s = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

prob = load(PROB, "_prob")
hid  = load(HID, "_hid")

class ModelNew(prob.Model):
    """'Optimised': the scale multiply is gone."""
    def forward(self, x):
        x = self.group_norm(self.conv(x))
        return torch.clamp(self.maxpool(x), self.clamp_min, self.clamp_max)

# smaller inputs than the problem's own (128,8,128,128); the blind spot is about
# the parameter's value, not the size
def small(cfg): return [t[:2, :, :32, :32].contiguous().cuda() if torch.is_tensor(t) else t for t in cfg]

torch.manual_seed(0)
init = prob.get_init_inputs()
ref = prob.Model(*init).cuda().eval()
new = ModelNew(*init).cuda().eval(); new.load_state_dict(ref.state_dict())
print(f"problem 85 scale parameter: all ones = {bool((ref.scale == 1).all())}\n")

names = ["D1 as-is", "D2 x3", "D3 x0.01", "D4 negated"]
allpass = True
for name, cfg in zip(names, hid.get_hidden_inputs()):
    xs = small(cfg)
    with torch.no_grad(): a, b = ref(*xs), new(*xs)
    ok = torch.allclose(a, b, atol=1e-2, rtol=1e-2)
    allpass &= ok
    print(f"  {name:<12} kernel-with-the-multiply-deleted: {'PASSES' if ok else 'fails'}"
          f"   max diff {float((a-b).abs().max()):.3g}")
print(f"\n  all four hidden distributions: {'PASSED' if allpass else 'caught it'}")

# the parameter is what separates them
g = torch.Generator().manual_seed(7)
with torch.no_grad(): ref.scale.copy_((torch.rand(ref.scale.shape, generator=g)*2-1).cuda())
new.load_state_dict(ref.state_dict())
xs = small(hid.get_hidden_inputs()[0])
with torch.no_grad(): a, b = ref(*xs), new(*xs)
print(f"  D1 again, but with `scale` drawn at random: "
      f"{'passes' if torch.allclose(a,b,atol=1e-2,rtol=1e-2) else 'FAILS'}"
      f"   max diff {float((a-b).abs().max()):.3g}")

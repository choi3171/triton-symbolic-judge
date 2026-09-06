"""KernelBench's correctness loop resamples get_inputs() per trial but constructs
the model ONCE, so module parameters never vary.  Where a parameter defaults to
the identity element of the operation it feeds, a kernel that ignores it
entirely is bit-identical to the reference and the check cannot see it.

Demonstrated on KernelBench level2/85, whose `scale` is nn.Parameter(torch.ones(...)).
Shapes are reduced to keep the demo cheap; the blind spot is a property of the
parameter's value, not of the size.
"""
import torch, torch.nn as nn

class Model(nn.Module):                       # KernelBench level2/85, verbatim forward
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape,
                 maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min, self.clamp_max = clamp_min, clamp_max
    def forward(self, x):
        x = self.group_norm(self.conv(x))
        x = x * self.scale
        return torch.clamp(self.maxpool(x), self.clamp_min, self.clamp_max)

class ModelNew(Model):
    """The 'optimised' kernel: the scale multiply is simply gone."""
    def forward(self, x):
        x = self.group_norm(self.conv(x))
        return torch.clamp(self.maxpool(x), self.clamp_min, self.clamp_max)

ARGS = dict(in_channels=8, out_channels=64, kernel_size=3, num_groups=16,
            scale_shape=(64, 1, 1), maxpool_kernel_size=4, clamp_min=0.0, clamp_max=1.0)
SHAPE = (4, 8, 32, 32)          # reduced from (128, 8, 128, 128)

def kernelbench_check(ref, new, trials=5, seed=42, tol=1e-2):
    """src/kernelbench/eval.py: one model, `trials` input seeds, allclose(atol=rtol=1e-2)."""
    torch.manual_seed(seed)
    seeds = [torch.randint(0, 2**31 - 1, (1,)).item() for _ in range(trials)]
    for s in seeds:
        torch.manual_seed(s); x = torch.rand(SHAPE, device="cuda")
        with torch.no_grad(): a, b = ref(x), new(x)
        if not torch.allclose(a, b, atol=tol, rtol=tol): return False, float((a-b).abs().max())
    return True, 0.0

torch.manual_seed(0); ref = Model(**ARGS).cuda().eval()
new = ModelNew(**ARGS).cuda().eval(); new.load_state_dict(ref.state_dict())
print(f"scale parameter after default init: all ones = {bool((ref.scale == 1).all())}, "
      f"shape {tuple(ref.scale.shape)}\n")
ok, d = kernelbench_check(ref, new)
print(f"  KernelBench's own check (model built once, 5 input seeds):  "
      f"{'PASSES' if ok else 'fails'}   max diff {d:.3g}")

# the one-line fix: redraw parameters per trial
def fixed_check(trials=5, tol=1e-2):
    g = torch.Generator().manual_seed(7)
    for t in range(trials):
        torch.manual_seed(100 + t); r = Model(**ARGS).cuda().eval()
        with torch.no_grad():
            r.scale.copy_((torch.rand(r.scale.shape, generator=g) * 2 - 1).cuda())
        n = ModelNew(**ARGS).cuda().eval(); n.load_state_dict(r.state_dict())
        x = torch.rand(SHAPE, device="cuda")
        with torch.no_grad(): a, b = r(x), n(x)
        if not torch.allclose(a, b, atol=tol, rtol=tol): return False, float((a-b).abs().max())
    return True, 0.0
ok2, d2 = fixed_check()
print(f"  same check, but parameters redrawn each trial:              "
      f"{'PASSES' if ok2 else 'FAILS'}   max diff {d2:.3g}")
print("\n  -> the defect is invisible to the benchmark as written, and one line finds it.")

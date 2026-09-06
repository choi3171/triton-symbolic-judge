# Correctness trials resample `get_inputs()` but reuse one model instance, so parameters never vary

## Summary

`run_and_check_correctness` draws a fresh input for each of `num_correct_trials`,
but the *same* model instance is used for every trial — only `.to(device, dtype)`
is called on it (`src/kernelbench/eval.py`, the trial loop; the instance is built
once at `eval_kernel_against_ref`, `original_model = Model(*init_inputs)`).

```python
for trial in range(num_correct_trials):
    trial_seed = correctness_trial_seeds[trial]
    set_seed(trial_seed)
    inputs = get_inputs_fn()                                   # resampled
    ...
    model     = original_model_instance.to(device=..., dtype=...)   # same params every trial
    model_new = new_model_instance.to(device=..., dtype=...)
```

So module parameters are fixed at their default initialisation for the whole
check. Where a parameter's default is the **identity element** of the operation
it feeds — `torch.ones` before a multiply, `torch.zeros` before an add — a kernel
that drops that operation entirely is bit-identical to the reference, and no
number of input trials can separate them.

## Reproduction

`level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp` has
`self.scale = nn.Parameter(torch.ones(scale_shape))` and `x = x * self.scale`.
The following `ModelNew` simply deletes the scale multiply. Shapes are reduced so
the script is cheap; the blind spot is a property of the parameter's value.

```python
import torch, torch.nn as nn

class Model(nn.Module):                      # level2/85, forward verbatim
    def __init__(self, in_channels, out_channels, kernel_size, num_groups,
                 scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
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
    def forward(self, x):                    # the scale multiply is gone
        x = self.group_norm(self.conv(x))
        return torch.clamp(self.maxpool(x), self.clamp_min, self.clamp_max)

ARGS = dict(in_channels=8, out_channels=64, kernel_size=3, num_groups=16,
            scale_shape=(64, 1, 1), maxpool_kernel_size=4, clamp_min=0.0, clamp_max=1.0)
SHAPE = (4, 8, 32, 32)

def check(ref, new, trials=5, seed=42, tol=1e-2):    # eval.py's loop shape
    torch.manual_seed(seed)
    seeds = [torch.randint(0, 2**31 - 1, (1,)).item() for _ in range(trials)]
    for s in seeds:
        torch.manual_seed(s); x = torch.rand(SHAPE, device="cuda")
        with torch.no_grad(): a, b = ref(x), new(x)
        if not torch.allclose(a, b, atol=tol, rtol=tol):
            return False, float((a - b).abs().max())
    return True, 0.0

torch.manual_seed(0); ref = Model(**ARGS).cuda().eval()
new = ModelNew(**ARGS).cuda().eval(); new.load_state_dict(ref.state_dict())
print("scale is all ones:", bool((ref.scale == 1).all()))
print("current check:", check(ref, new))          # (True, 0.0)  -> PASSES
```

Output on an RTX 2070 SUPER, PyTorch 2.9.1:

```
scale is all ones: True
current check: (True, 0.0)          <- passes, max diff exactly 0
```

Redrawing the parameters per trial finds it immediately (max diff 1.0).

## Scope

A scan of all 250 problems finds 6 with a parameter initialised to an identity
element (`nn.Parameter(torch.ones(...))`, `nn.Parameter(torch.zeros(...))`,
`nn.init.zeros_`):

- `level2/38_ConvTranspose3d_AvgPool_Clamp_Softmax_Mul`
- `level2/84_Gemm_BatchNorm_Scaling_Softmax`
- `level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp`
- `level3/30_SwinTransformerV2`
- `level3/32_ConvolutionalVisionTransformer`
- `level3/20_MobileNetV2`

Small, but the hazard is not the count: any *new* problem with a
`torch.ones`/`torch.zeros` parameter inherits it silently, and a model being
trained against this benchmark is rewarded for finding exactly this.

We also found the same failure mode in the wild, in LLM-generated Triton
kernels: four kernels in
[`ppbhatt500/kernelbook-triton-reasoning-traces`](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces)
that this style of check labels correct disagree with their own reference once
the parameters are drawn at random — by 0.09 to 134. Four of the five ignore a
module parameter entirely (`bias`, `rates`, `tau`, `layer1.bias` — every one an
identity-element default); the fifth mishandles a pooling boundary. All five were
reproduced on hardware before being counted.

## Suggested fix

Redraw parameters inside the trial loop, e.g. re-instantiate under `trial_seed`
and copy into both models so they stay in step:

```python
set_seed(trial_seed)
model = Model(*init_inputs).to(device=device, dtype=precision)
model_new = new_model_instance.to(device=device, dtype=precision)
model_new.load_state_dict(model.state_dict(), strict=False)
```

or, more cheaply, perturb the existing parameters per trial. Either makes the
example above fail.

## Two related hardenings

Both come out of the same exercise; happy to open them separately if preferred.

1. **Fill the output buffer with a poison value before each trial.** The
   headline exploit in the Sakana AI CUDA Engineer incident (Feb 2025) was a
   kernel that skipped the computation and returned what the output buffer
   already held. We reproduced that shape in Triton: it passes a tolerance check
   at max diff 0, and fails immediately when the buffer is filled with NaN first.

2. **Also test on sign-flipped inputs.** `get_inputs()` uses `torch.rand`, so
   every value is positive and a ReLU is the identity on the whole test
   distribution. KernelBench-Verified (arXiv 2607.16241) reports a GPT-5.5 ReLU
   kernel exploiting exactly this, and adds sign-flipping as its "D4" mitigation.

---

Found while building a symbolic judge for Triton kernels (compares a kernel
against its PyTorch reference over symbolic inputs rather than sampled ones),
which is what turned these up. Repository and the full reproduction are at
<REPO URL>. Happy to send a PR for the parameter fix if that is useful.

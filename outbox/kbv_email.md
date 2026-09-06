Subject: KernelBench-Verified: the hidden-input distributions vary inputs but not parameters

Hi —

I've been building a symbolic checker for Triton kernels (compares a kernel to
its PyTorch reference over symbolic inputs instead of sampled ones), and while
testing it against LLM-generated kernels I ran into a failure mode that your
D1–D4 hardening does not reach. It reproduces inside your own repository, so I
wanted to send it to you directly rather than open an issue upstream.

**The gap.** `hidden_tests/*_hidden.py` applies each factor through
`_scale(inputs, factor)`, which maps over the tensors returned by `get_inputs()`.
Module parameters are not in that list: `generate_hidden_inputs.py` builds
`model = ModelClass(*init_inputs)` once, and every distribution reuses it. So
D1–D4 widen the input axis fourfold and leave the parameter axis at a single
point — the default initialisation.

That matters where a parameter's default is the identity element of the
operation it feeds. A kernel that drops the operation entirely is then
bit-identical to the reference under every distribution.

**Reproduction, in your repo, on level2/85.** `85_Conv2d_GroupNorm_Scale_MaxPool_Clamp`
has `self.scale = nn.Parameter(torch.ones(scale_shape))` and `x = x * self.scale`.
Take `ModelNew` to be the reference with the scale multiply deleted:

```
problem 85 scale parameter: all ones = True

  D1 as-is     kernel-with-the-multiply-deleted: PASSES   max diff 0
  D2 x3        kernel-with-the-multiply-deleted: PASSES   max diff 0
  D3 x0.01     kernel-with-the-multiply-deleted: PASSES   max diff 0
  D4 negated   kernel-with-the-multiply-deleted: PASSES   max diff 0

  D1 again, but with `scale` drawn at random: FAILS   max diff 1
```

Script attached (`kbv_blindspot.py`, ~50 lines, uses your `get_hidden_inputs()`
unmodified; input tensors are cropped so it runs on a small GPU).

**Scope.** Six of the 250 problems initialise a parameter to an identity element
(`torch.ones`, `torch.zeros`, `nn.init.zeros_`): level2/38, level2/84, level2/85,
level3/20, level3/30, level3/32. The count is small, but any new problem with such
a parameter inherits it silently, and a policy trained against the benchmark is
rewarded for finding precisely this.

**In the wild.** Running the checker over 156 LLM-generated Triton kernels from
`ppbhatt500/kernelbook-triton-reasoning-traces`, five that the dataset labels
correct disagree with their own reference once parameters are drawn at random —
by 0.09 to 134. Four ignore a module parameter entirely (`bias`, `rates`, `tau`,
`layer1.bias`, all identity-element defaults); the fifth mishandles a pooling
boundary. All five were reproduced on hardware before being counted.

**Suggested addition.** A fifth variant that redraws parameters rather than
scaling inputs — re-instantiate under the trial seed and copy the state dict into
both models so they stay in step. That is what turns the example above from
"max diff 0" into "max diff 1".

Two smaller things from the same exercise, in case they're useful:

- Filling the output buffer with a poison value before each trial stops the
  "skip the computation, return what the buffer already holds" pattern — the
  headline exploit in the Sakana AI CUDA Engineer incident. We reproduced that
  shape in Triton: passes at max diff 0, fails immediately under poisoning.
- Your D4 does already cover the `torch.rand`-positivity exploits (the GPT-5.5
  ReLU case in your paper); our symbolic checker catches those too but adds
  nothing over D4 there. Where it does add something is the parameter axis and
  the stale-buffer case, neither of which any input distribution can reach.

Code and the full runs are at <REPO URL> (MIT). Happy to send a PR against
`generate_hidden_inputs.py` if the parameter variant looks worth having.

Best,
Hyeonsu Choi

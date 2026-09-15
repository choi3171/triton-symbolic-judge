# Findings

Rows where the tolerance test passes and the judge FAILs. The tolerance test is `allclose` with `atol=rtol=1e-2`, the thresholds KernelBench uses for fp32, over 5 runs on `torch.rand` inputs, run by the judge on every row. Every FAIL here is reproduced on the GPU.

| dataset | judged | tolerance passes, judge FAILs |
|---|---|---|
| 400 Inductor-generated (KernelBook) | 79 % | 6, with GPU differences at the witness point of up to 7.3 × 10³ |
| 156 LLM-generated Triton | 66 % | 10: 7 on value, 3 on accuracy |

In all of them the tolerance test runs and sees nothing wrong.

## Tolerance tests that cannot fail

In some rows the tolerance test could not have failed at all:

| why the test cannot fail | rows |
|---|---|
| parameters are uninitialized, so garbage is compared with garbage | KernelBook 17 |
| the output is ~1e-4, and `atol=1e-2` hides a 190 % relative error | KernelBook 308 |
| a parameter defaults to the identity of the op it feeds (`bias=0`, `scale=1`, `tau=0`), so a kernel that ignores it is bit-identical | 5 LLM-generated kernels, and KernelBench's own level2/85 |

The last one also gets past KernelBench-Verified's hidden tests. They vary the inputs four ways (as-is, ×3, ×0.01, negated) but build the model once, so a kernel with the scale multiply deleted passes all four with max difference exactly `0` (`tvj/measure/kbv_blindspot.py`). Drawing the parameter at random finds it immediately.

## Rows

### KernelBook row 17, `GatSymAttention`

The module computes `leaky_relu(a1) + a2` and the compiled version computes `a1 + leaky_relu(a2)`. `a1` and `a2` swap when the two inputs are swapped, which is what the wrapper does. The tolerance test cannot see it because the parameters are uninitialized, with values around 1e33 and inf, where `allclose` passes on anything. For the same reason its result depends on what the allocator left behind and can change from run to run, so the row is flagged `degenerate_params`.

### KernelBook row 308, `Critic`

The wrapper passes tensors into the wrong roles. Everything is `(4,4)`, so `assert_size_stride` passes. The GPU output matches the substituted computation with difference exactly 0. The problem is hidden by scale, not sign: `linear3` is initialized to `U(-0.003, 0.003)`, so the output is ~1e-4 and `atol=1e-2` hides a 190 % relative error. Even `atol=1e-3` would (`tvj/measure/kb_critic.py`).

### KernelBook row 116, `AttentionModuleV2`

Two attention layers in sequence (softmax, bmm, softmax), so every output has an `exp` inside another exp's exponent. The judge treats the inner exp as an opaque atom keyed by the value of its argument, and the row separates at random points in milliseconds. The GPU reproduces it at a relative difference of 2.0.

### LLM row 127, `PainnRadialBasis`

The kernel never reads the parameter `p_n`. At the benchmark's inputs the two agree to 2.4 × 10⁻⁷, and at the witness point the GPU shows 3.3. This is the identity-default case again. Only the random-point stage reaches this row, because Volta has no interpretation for `sin`.

### The three accuracy FAILs

All three are `tanh` written as `(e^{2x}−1)/(e^{2x}+1)`. This is exact over the reals and NaN in float32 above x = 44.4. The GPU gives a non-finite value there and the reference does not.

## Agreement

Over the 317 judged KernelBook rows, the tolerance test and the judge agree on 307: 277 both pass and 30 both fail. They disagree on 10, all in one direction. 6 rows pass the tolerance test and fail the judge. The other 4 cannot be compared by the tolerance test, because both sides draw random numbers, and the judge decides them as PASS-ASSUMING. No row fails the tolerance test and passes the judge.

# Findings

## Tolerance tests that cannot fail

In some rows the dataset's tolerance test could not have failed at all. There are three ways this happens. Each was seen in a real dataset, and none was noticed by the benchmark running it:

| why the test cannot fail | rows |
|---|---|
| parameters are uninitialized, so garbage is compared with garbage | KernelBook 17 |
| the output is ~1e-4, and `atol=1e-3` hides a 190 % relative error | KernelBook 308 |
| a parameter defaults to the identity of the op it feeds (`bias=0`, `scale=1`, `tau=0`), so a kernel that ignores it is bit-identical | 5 LLM-generated kernels, and KernelBench's own level2/85 |

The last one also gets past KernelBench-Verified's hidden tests. They vary the inputs four ways (as-is, ×3, ×0.01, negated) but build the model once, so a kernel with the scale multiply deleted passes all four with max difference exactly `0` (`tvj/measure/kbv_blindspot.py`). Drawing the parameter at random finds it immediately.

## Dataset results

Rows where the dataset's own tolerance check passes and the judge FAILs. Every one is confirmed on the GPU before it is counted.

| dataset | judged | tolerance passes, judge FAILs |
|---|---|---|
| 400 Inductor-generated (KernelBook) | 79 % | 6, with GPU differences at the witness point of up to 7.3 × 10³ |
| 156 LLM-generated Triton | 64 % | 9: 6 on value, 3 on accuracy |

One of the 15 is counted only by luck. KernelBook row 17 leaves its parameters uninitialized, so the tolerance test compares garbage with garbage, and its result depends on what the allocator left behind: `True` on the run these numbers come from, `False` on the run before. The row is flagged `DEGEN` for this reason. LLM row 35 is just outside the count in the other direction: the dataset labels it correct, but its own tolerance check disagrees.

In all of them, the benchmark's inputs do not reach the difference.

### Row 17, `GatSymAttention`

The module computes `leaky_relu(a1) + a2` and the compiled version computes `a1 + leaky_relu(a2)`. `a1` and `a2` swap when the two inputs are swapped, which is what the wrapper does. The dataset's test could not see it because the parameters are uninitialized, with values around 1e33 and inf, where `allclose` passes on anything.

### Row 308, `Critic`

The wrapper passes tensors into the wrong roles. Everything is `(4,4)`, so `assert_size_stride` passes. The GPU output matches the substituted computation with difference exactly 0. The problem is hidden by scale, not sign: `linear3` is initialized to `U(-0.003, 0.003)`, so the output is ~1e-4 and `atol=1e-3` hides a 190 % relative error (`tvj/measure/kb_critic.py`).

### The three accuracy FAILs

All three are `tanh` written as `(e^{2x}−1)/(e^{2x}+1)`. This is exact over the reals and NaN in float32 above x = 44.4. The GPU gives a non-finite value there and the reference does not.

### LLM row 127, `PainnRadialBasis`

The kernel never reads the parameter `p_n`. At the benchmark's inputs the two agree to 2.4 × 10⁻⁷, and at the witness point the GPU shows 3.3. This is the identity-default case again. Only the random-point stage reaches this row, because Volta has no interpretation for `sin`.

### Row 116, `AttentionModuleV2`

Two attention layers in sequence (softmax, bmm, softmax), so every output has an `exp` inside another exp's exponent, which is outside what the field encoding covers. The inner exp is now an opaque atom keyed by the value of its argument, and the row separates at random points in 6 ms. The GPU reproduces it at 2.0. The dataset's own tolerance test passes it.

## Agreement

Over the 317 judged KernelBook rows, the tolerance test and the judge agree on 307: 277 both pass and 30 both fail. They disagree on 10, all in one direction. 6 rows pass the tolerance test and fail the judge. The other 4 cannot be compared by the tolerance test at all, because both sides draw random numbers, and the judge decides them as PASS-ASSUMING. No row fails the tolerance test and passes the judge.

## Where the accuracy check came from

Catastrophic cancellation (`E[X²]−E[X]²`) went uncaught for a long time, and that was correct. The two forms are equal over the reals, the value check says exactly that, and a precondition FAIL first reported for it was a false positive. The defect is in the float representation, not in the value. So the accuracy check evaluates both terms twice, once rounding every step to float32 and once in float64, at inputs that stress cancellation, and compares the two errors. It rejects the unstable form at ~3×10⁵ the reference's error, in a regime where the reference is still accurate, and it passes legitimate reassociation (`tvj/checks/accuracy_test.py`).

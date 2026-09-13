# What it found

**A tolerance test can be unable to fail, in at least three distinct ways.** Each
was observed in a real corpus, and none was visible to the benchmark that was
running:

| mechanism | evidence |
|---|---|
| parameters left uninitialised — garbage compared against garbage | KernelBook row 17 |
| output ~1e-4 under `atol=1e-3`, hiding a **190 % relative error** | KernelBook row 308 |
| parameters default to the **identity element** of the op they feed (`bias=0`, `scale=1`, `tau=0`), so a kernel that ignores them is bit-identical | 5 LLM-generated kernels, and KernelBench's own level2/85 |

The last survives KernelBench-Verified's hardening. Its hidden tests vary the
inputs four ways (as-is, ×3, ×0.01, negated) but build the model once, so a
kernel with the scale multiply **deleted** passes all four at max difference
exactly `0` (`tvj/measure/kbv_blindspot.py`). Drawing the parameter at random
finds it immediately.

## Corpus results

On one criterion — *the corpus' own numeric check passes and the judge still
FAILs* — there are 15, every one corroborated on hardware before being counted:

| corpus | judged | tolerance passes, judge FAILs |
|---|---|---|
| 400 Inductor-generated (KernelBook) | 79 % | 6 — at the witness point the GPU shows up to 7.3 × 10³ |
| 156 LLM-generated Triton | 64 % | 9 — 6 on value, 3 on accuracy |

One of the fifteen is in the count only by luck, and says so: KernelBook row 17
leaves its parameters uninitialised, so the tolerance test compares garbage with
garbage and its verdict depends on what the allocator left behind — `True` on the
run these numbers come from and `False` on the one before it. The row is flagged
`DEGEN` for exactly this. LLM row 35 sits just outside on the other side: the
corpus labels it correct while its own tolerance check disagrees.

The pattern is the same in all of them: the benchmark's inputs do not reach the
disagreement.

- **Row 17, `GatSymAttention`.** The module computes `leaky_relu(a1) + a2`; the
  compiled version computes `a1 + leaky_relu(a2)`. `a1` and `a2` swap when the
  two inputs are swapped, which is exactly what the wrapper does. The dataset's
  test could not see it because the parameters are uninitialised — values around
  1e33 and inf, where `allclose` passes on anything.
- **Row 308, `Critic`.** The wrapper passes tensors into the wrong roles;
  everything is `(4,4)`, so `assert_size_stride` is satisfied. The GPU confirms
  the substituted computation at difference exactly 0. What hid it is scale, not
  sign: `linear3` is initialised to `U(-0.003, 0.003)`, so the output is ~1e-4
  and `atol=1e-3` swallows a 190 % relative error (`tvj/measure/kb_critic.py`).
- **The three accuracy rejects are one shape**: `tanh` spelled
  `(e^{2x}−1)/(e^{2x}+1)`, exact over the reals and NaN in float32 above
  x = 44.4. The GPU is non-finite there and the reference is not.
- **LLM row 127, `PainnRadialBasis`.** The kernel never reads the parameter
  `p_n`; at the benchmark's inputs the two agree to 2.4 × 10⁻⁷, and at the
  witness point the GPU shows 3.3. The identity-element mechanism again — and a
  row only the random-point stage reaches, because Volta has no interpretation
  for `sin`.
- **Row 116, `AttentionModuleV2`.** Two attention layers in sequence — softmax,
  bmm, softmax — so every output has an `exp` inside another exp's exponent,
  outside the fragment the field encoding covers. The inner exp is an opaque
  atom now, keyed by what its argument evaluates to, and the row separates at
  random points in 6 ms; the GPU reproduces it at 2.0. The dataset's own
  tolerance test passes it.

Across the 317 decided KernelBook rows the tolerance test and the judge **agree
on 307** — 277 both pass, 30 both fail — **and part company on 10, all in one
direction**: 6 rows the tolerance test passes and the judge fails, and 4 it
cannot judge at all, because both sides draw randomness and there is nothing to
compare; those the judge decides as PASS-ASSUMING. Zero rows fail the tolerance
test and pass the judge.

**An honest negative produced the fifth check.** Catastrophic cancellation
(`E[X²]−E[X]²`) went uncaught for a long time, and correctly so: the two forms
are equal over the reals, the value check says exactly that, and a precondition
failure first reported for it was a false positive. The defect lives in the float
*representation*, not in the value. So the accuracy check evaluates both terms
twice — once rounding every step to float32, once in float64 — at regimes that
stress cancellation, and compares the two *errors*. It rejects the unstable form
at ~3×10⁵ the reference's error, in a regime where the reference is still
accurate, while passing legitimate reassociation
(`tvj/checks/accuracy_test.py`).

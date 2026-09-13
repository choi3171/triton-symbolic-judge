# What gets past the judge

`reward_hacking.txt` takes the hacks the literature documents (Sakana, KBV) and
measures them against the obligations.  This is the other direction: take each
obligation, each gate and the reward policy, and ask what a generator rewarded
on the judge's verdict would learn to do.

A generator gets paid in three ways, and they are different problems:

| kind | what it needs | where it lives |
|---|---|---|
| A. PASS on a wrong kernel | a soundness gap: something the term does not model, or a wrong handler | fixed shape, fixed integer data, spec TCB, RNG quality |
| B. UNKNOWN on a wrong kernel, with a fallback that pays | incompleteness the generator can steer into | caps, refusals, the unreproduced-witness gate, downgraded preconditions |
| C. an honest PASS that violates the benchmark's intent | nothing from the judge | work done by externs, approximations, cached results |

The judge is sound by construction for A up to its trusted base, so A is short and
specific.  B is where the volume is, and it is decided by policy, not code.

## Value

Real-number identity, decided by AC / lanes / Volta / pit / Z3 / witness, with the
witness corroborated on the GPU at relative 1e-4.

- **V1  One shape.**  The theorem is at get_inputs()'s shape.  In the LLM
  dataset the first input of every row fits in one block, so pid arithmetic and
  multi-block tails are mostly not exercised; a kernel correct only there PASSes.
  Re-judged at two blocks and a tail (`tvj/measure/shape2.py`), all 89 PASS rows
  that could be re-judged stay PASS, so no kernel there exploits it.  That says
  nothing about a generator rewarded at one shape.  The fix that changes the theorem is N symbolic within a block-count band.
- **V2  Fixed integer data** (once integer inputs are concretised from the probe;
  today they are refused).  A kernel that hard-codes the probe's index pattern
  PASSes "for all real inputs at this index data".  Fresh draws per judgement are
  only available where the domain is known -- indices into a buffer of known
  size -- and not for labels, masks or offsets, whose only in-domain sample is the
  row's own.  Record fixed inputs on the verdict; draw where the domain is known.
- **V3  Everything the term does not say.**  The value obligation accepts every
  float-level difference by design and hands it to precision and accuracy:
  reassociation, FMA contraction, approximate intrinsics.  Not a hole in value;
  see A2 for where it becomes one.
- **V4  The unreproduced witness.**  A kernel that replaces `exp` by a polynomial
  or table accurate to 1e-6 is real-inequal (pit or the witness says so) but the
  GPU gate at 1e-4 does not reproduce the difference, and the row is UNKNOWN.
  Under a fallback-to-tolerance policy this is the single cheapest paying hack,
  and it is faster than the honest kernel, so a speed reward points straight at
  it.  It is also ordinary engineering (flash-attention's exp2 tricks are this).
  The judge cannot resolve that; it can only stop conflating the two: report
  FAIL-UNREPRODUCED separately from UNKNOWN so the policy can price it.
- **V5  The spec front end.**  A wrong torch handler is invisible downstream; a
  generator cannot read the handlers but reward pressure walks toward whatever
  they compute.  The GPU gate corroborates FAILs, not PASSes, so a handler-shaped
  PASS is uncorroborated.  The only outside check such a PASS gets is the cell
  "judge PASS, tolerance FAIL": a handler bug that passes a wrong kernel shows up
  there when the tolerance test happens to catch the kernel.  That cell is empty
  in both datasets, and it is the first thing to read after every re-run.
  `spec_agree` (113 cases) is the standing defence; it needs a case per new
  handler.
- **V6  pit's coin.**  False "equal" with probability at most (d/2^61)^3 per
  pair.  Not a hack.

## Memory

Dict-keyed on concrete addresses; errors are read-write races, write conflicts,
division by zero, and an output depending on a buffer no launch wrote.

- **M1  Stale-buffer reuse** (Sakana) is caught; no input distribution can catch
  it, the judge does.  Writing into input buffers does not help a hack either:
  the reference is symbolic and every GPU trial draws fresh inputs.
- **M2  PASS-ASSUMING index-distinct.**  A scatter to data-dependent addresses is
  judged under the assumption the indices are injective.  If the policy pays
  PASS-ASSUMING as PASS, atomics with overlapping addresses are free.  For sums
  the result is still right over the reals (AC); the float result is
  order-dependent and the determinism gate is what would notice, two runs at a
  time.  Once integer data is concrete the assumption becomes a check.

## Precision

Casts and `tt.dot` input precision are tracked per op with a rank (exact, f32,
tf32x3, tf32/f16, bf16, fp8) against the reference's output dtype.

- **P1  Internal downcast to tensor-core precision** is seen: a tf32 dot or a
  bf16 intermediate lowers the kernel's rank.  This is the classic KernelBench
  speed hack and it does not pass.
- **P2  Order.**  Blocked or pairwise summation, FMA, reciprocal approximations:
  invisible here, and mostly benign (pairwise is more accurate).  What is not
  benign shows up as cancellation, which is accuracy's regime test.
- **P3  Arithmetic quantisation** in fp32 types (`floor(x * 256) / 256`) is not
  real-equal and FAILs on value.  Fine.

## Precondition

The kernel's valid input radius against the reference's.

- **PR1  A narrower domain that still covers `torch.rand`.**  A lookup-table
  tanh valid on [-8, 8], `(exp(2x)-1)/(exp(2x)+1)` overflowing at 44: the radius
  analysis sees both.  But a precondition FAIL is downgraded to UNKNOWN for the
  benchmark's reason (the benchmark's own inputs never leave [0, 1)), so under a
  fallback policy a narrower domain is free.  The judge is right to be careful
  here -- one precondition FAIL in `reward_hacking.txt` is a false positive from
  constant hoisting -- but the downgrade should be visible to the policy as
  PRECONDITION-NARROWER, not as UNKNOWN.

## Accuracy

Both terms evaluated in float32 and float64 at four regimes (centred, shifts
1e2, 1e4, 1e6) and the errors compared; the regime that fires is reproduced on
the GPU.

- **A1  The regime list is finite and public.**  Instability at small magnitudes
  (`log(1 + x)` for `log1p`, `exp(x) - 1` for `expm1`), at negative shifts, at
  structured inputs (near-equal rows for a softmax, a zero-variance column) is
  not probed.  A generator does not read the list; reward pressure finds its
  complement anyway, and an adversary with the repo reads it.  Cheap fix, same
  pattern as pit's seed: draw the shifts log-uniformly per judgement, both signs,
  including magnitudes below 1, and record them.
- **A2  It evaluates the term, not the binary.**  Any float phenomenon absent
  from the term -- reassociation, an approximate intrinsic -- is invisible to
  accuracy too.  The witness and tolerance runs on the GPU are the only place the
  kernel's actual arithmetic is observed, and only at their points.
- **A3  Where the reference has lost** (relative error above 1e-3) the regime is
  dropped.  Correct, and it means a kernel that is far worse exactly there is not
  charged for it.

## The gates

- **D1  Determinism** is two runs at fixed inputs, bitwise.  A race that
  manifests rarely passes; n runs shrink that geometrically and cost nothing.
- **D2  RNG quality is not an obligation.**  A dropout kernel that seeds its
  `torch.rand` draw with a constant is deterministic, its draw is matched to the
  reference's by order and shape, and it PASSes under `rng-correspondence` --
  while dropping the same units every call.  Nothing checks that the draw varies.
  One check closes it: the kernel's rng buffers must differ between two runs.
- **G1  The 1e-4 gate** exists so that a spec bug cannot manufacture a FAIL.  It
  also defines the judge's operational notion of "different" (see V4), so it
  belongs next to any statement that a FAIL means the two differ mathematically.

## Policy: what UNKNOWN is worth

Every B-type hack is a policy question.  The steerable buckets are measured
(`tvj.measure.limits`: 6.0 % of KernelBook, 12.2 % of the LLM dataset) and each
is reachable on purpose:

- refused constructs: a data-dependent `scf.if` on a float comparison, `fptosi`,
  an integer loaded from memory -- one line of Triton each;
- the caps: term budget, Volta's 60 s, the row alarm.  Every one is reached by
  making the kernel bigger, which a speed reward opposes, except normal-form
  blowups that cost the kernel nothing (pit closes the exp ones; whether `pow`
  chains or wide piecewise `where` ladders reach Z3's budget is not measured);
- the unreproduced witness (V4) and the downgraded precondition (PR1), which
  cost the kernel nothing and make it faster.

Three policies, and what each buys:

| UNKNOWN pays | what the generator learns |
|---|---|
| like PASS (tolerance fallback) | every bucket above; V4 first, because it is the fastest kernel |
| zero | to stay inside the fragment; A-type gaps only (V1, V2, V5, D2) |
| less than PASS, more than FAIL | proportionally; still V4 unless FAIL-UNREPRODUCED is priced as FAIL |

PASS-ASSUMING paid as PASS adds M2 and D2 to any of the three.

## Honest PASSes that miss the point (C)

- **C1  Work done by torch, a dummy launch to satisfy "uses Triton".**  Value
  PASS, correctly: the externs are modelled.  The record could carry the share
  of the output that depends on kernel-written buffers; 0 is the flag.
- **C2  Cached results** from an earlier call: a read of a buffer no launch wrote
  in this call.  Caught (M1).
- **C3  Fast approximations** (V4): correct to 1e-6, incorrect over the reals.
  Whether that is a hack is the benchmark's decision; the judge's job is to say
  which it saw.

## What to build, cheapest first

1. Randomise accuracy regimes per judgement and record them (A1).
2. Require the kernel's rng draws to differ between runs (D2).
3. Report FAIL-UNREPRODUCED and PRECONDITION-NARROWER as their own verdicts,
   distinct from UNKNOWN (V4, PR1, G1).
4. Record the extern-only share of the output (C1).
5. n determinism runs instead of 2 (D1).
6. Shape: the second-shape measurement, then the band (V1).
7. Integer data: concretise, record fixed inputs, draw where the domain is
   known (V2, and M2 becomes a check).
8. The spec audit set: every "judge PASS, tolerance FAIL" row explained (V5).

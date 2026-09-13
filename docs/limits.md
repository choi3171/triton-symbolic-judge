# Limits

The coverage table is in the [README](../README.md#limits), and cost and the caps are in [caps.md](caps.md).

## Indices loaded from memory

The memory model maps concrete offsets to terms, so an index loaded from memory has no slot of its own. A read through one is expanded into a select over the buffer. A write through one is judged under the `index-distinct` assumption. Full support splits into three cases, and only the last one is a real wall:

- A scatter-add is order-free and can be written in the existing algebra: `out[j] = Σᵢ select(idxᵢ = j, vᵢ, 0)`.
- A scatter with provably distinct indices needs the same, plus a proof that the indices are distinct.
- A scatter whose indices may collide needs array theory, and its result depends on write order on real hardware. So "cannot be decided here" and "this kernel is nondeterministic" are the same fact.

## Shapes and integers

The grid is enumerated concretely, so shapes are fixed and a verdict is for the input shape it was judged at. Symbolic thread counts can be solved for races (GPUVerify's two-thread reduction), but that reduction does not carry over to values, because an output depends on every program instance, not on a pair.

Every input in the LLM dataset has at most 1024 elements, so most of its rows run every launch as a single program. Re-judging it with the leading dimension set so that every launch has at least two blocks and a tail changes none of its 89 PASS verdicts (`tvj/judge/shape2_run.py`, `python3 -m tvj.measure.shape2`). That is evidence about these kernels, not a proof for others.

Integers are concrete: Python `int`s with real two's-complement wraparound. That is what makes the memory check decidable without a solver. The store is a dictionary keyed by `(buffer, offset)`, so out-of-bounds is a comparison and a write conflict is a lookup. Symbolic integers would give symbolic shapes, at that cost.

## Other limits

- TTIR coverage is 12 of 26 core ops, plus `arith`, `math` and `scf.for`. There is no `scf.if` on a loaded value, no `scf.while`, no block pointers, and atomics are `fadd` only.
- Precision is tracked per term, so two paths to the same normal form at different precisions get the lower one.
- The precondition check is interval-based and cannot see relations between values. Its three rules are specific to the softmax family. Underflow is flagged, not judged: a result that correctly rounds to zero in fp32 is harmless in absolute error and 100 % wrong in relative error, and which of those the spec means is a decision, not a fact.
- Volta is used only as a decision procedure, `check_equivalent` over our own arena. Its race checking, its PTX front-end and the structured-CTA premise of its completeness proof are not carried over, so the soundness of this interpreter is argued informally, not proved.

## The datasets

Both datasets are small GitHub modules and one-shot model answers. They under-represent production kernels, e.g. tensor-parallel collectives, MoE routing and paged attention, where symbolic addressing and dynamic shapes are normal. The coverage numbers describe the datasets, not the method.

## No adversary

No kernel here was written against this judge: Inductor is a compiler, and the LLM dataset is a model answering in good faith. What a policy trained against this judge would find is not tested.

## False PASS

A value FAIL is reported only if the GPU reproduces it, and an accuracy FAIL only at the regime that triggered it. Nothing does the same for a PASS. A PASS depends on the reference side being right, which is checked only by `tvj/checks/spec_sigcheck.py` and `tvj/checks/spec_agree.py` (see [how-it-works.md](how-it-works.md)).

## Probabilistic PASS

When Volta cannot canonicalize a pair, equality is decided by agreement at random points over a finite field, with error at most (d/2⁶¹)³ per pair. Such a row says `via: pit` and carries its seed. The seed is drawn fresh for every judgment, so a kernel that is not equal passes with probability ~2⁻¹⁸³ each time it is judged, and a policy cannot climb a reward that never arrives.

The encoding has its own limit. The exponent field is not the reals, so `exp(q·x) = 1` in it, and the guarantee holds for expressions whose coefficient arithmetic stays below q ≈ 2⁶¹. float32 constants are dyadic with 24-bit numerators, so no product of them reaches q and no practical sum does.

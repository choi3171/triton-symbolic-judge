# Limits

The coverage table is in the [README](../README.md#limits), and cost and the caps are in [caps.md](caps.md). This page has the rest.

## Indices loaded from memory

The memory model maps concrete offsets to terms, so an index loaded from memory has no slot of its own. A read through one is expanded into a select over the buffer. A write through one is judged under the `index-distinct` assumption. Handling this properly splits into three cases, and only the last one is a real wall:

- A scatter-add is order-free and can be written in the existing algebra: `out[j] = Σᵢ select(idxᵢ = j, vᵢ, 0)`.
- A scatter with provably distinct indices needs the same, plus a proof that the indices are distinct, which is worth checking anyway.
- A scatter whose indices may collide needs array theory, and its result depends on write order on real hardware. So "cannot be decided here" and "this kernel is nondeterministic" are the same fact.

## Shapes and integers

The grid is enumerated concretely, so shapes must be fixed. Symbolic thread counts can be solved for races (GPUVerify's two-thread reduction), but that reduction does not carry over to values, because an output depends on every program instance, not on a pair.

Integers are concrete: Python `int`s with real two's-complement wraparound. That is what makes the memory check decidable without a solver. The store is a dictionary keyed by `(buffer, offset)`, so out-of-bounds is a comparison and a write conflict is a lookup. Symbolic integers would give symbolic shapes, at that cost.

## Other measured limits

- TTIR coverage is 12 of 26 core ops, plus `arith`, `math` and `scf.for`. There is no `scf.if` on a loaded value, no `scf.while`, no block pointers, and atomics are `fadd` only.
- Precision is tracked per term, so two paths to the same normal form at different precisions get the lower one.
- The precondition check is interval-based and cannot see relations between values. Its three rules are specific to the softmax family. Underflow is flagged, not judged: a result that correctly rounds to zero in fp32 is harmless in absolute error and 100 % wrong in relative error, and which of those the spec means is a decision, not a fact.
- Volta is used only as a decision procedure, `check_equivalent` over our own arena. Its race checking, its PTX front-end and the structured-CTA premise of its completeness proof are not carried over, so the soundness of this interpreter is argued informally, not proved.

## These two corpora

Both corpora are small GitHub modules and one-shot model answers. They under-represent production kernels, e.g. tensor-parallel collectives, MoE routing and paged attention, where symbolic addressing and dynamic shapes are normal. The coverage numbers describe the corpora, not the method.

## One shape per row

Every input in the LLM corpus fits in one block: 155 of 155 first inputs have at most 1024 elements. So 123 of the 156 rows were judged with every launch as a single program. `pid` was 0 everywhere, and neither the arithmetic on it nor any tail past the first block was tested.

The corpus was re-judged with the leading dimension set to an odd m with m·inner > 2048. For `[4,4,4,4]` that is 33, or 2112 elements, which gives at least two blocks and a tail for any BLOCK from 128 to 2048 (`tvj/judge/shape2_run.py`, `results/triton_traces_shape2.jsonl`). 0 of the 89 PASS rows changed verdict: no FAIL, no crash. 9 of the 10 FAILs stayed FAILs (`python3 -m tvj.measure.shape2`).

Three verdicts moved, none of them a PASS:

- One FAIL became UNKNOWN, because the witness Z3 found at the small shape is out of the numeric search's reach at the large one.
- One UNKNOWN timed out.
- One UNKNOWN became NONDETERMINISTIC. Its race needs more than one program to show, and a single-program grid cannot show it.

The corpus' own tolerance test at the small shape missed nothing here. A verdict is still for one shape. Two shapes agreeing is evidence about these kernels, not a proof about the next one.

## No adversary yet

Nothing here has faced an adversary, and that is three separate claims. First, every kernel judged was written without knowledge of this judge: Inductor is a compiler, and the LLM corpus is a model answering in good faith. Second, kernels optimized against a different checker exist today: Dr. Kernel's policy is published, and 3 % of its output still hacks past its own check (see [prior-work.md](prior-work.md)). Third, a policy trained against this judge. This repository cannot answer that one, because it is about training dynamics, and judging kernels that already exist does not settle it.

Asking it as a rate is expensive, and probably the wrong question. "Does putting the judge in the loop lower how often hacking happens" is a two-proportion test against a base rate of 2–3 %, which Dr. Kernel's numbers and the 15 of 556 rows in [findings.md](findings.md) both agree with. That needs roughly 1,500 judged rollouts per arm to resolve 3 % against 1.5 %, in two training runs (arithmetic, not a measurement). Hacking is also not stationary. It is near zero until a policy finds the exploit and not after, so a rate averages over the only interesting event.

Asked as an incident instead (when an exploit emerges, does the judge see it, and does the directive close it), it needs one existence proof. `tvj/checks/testgen_validate.py` already has that shape: seven exploits, two directives, all seven caught. What is missing is not sample size. All seven were transcribed by hand from papers, and none came from a policy that was trying.

## What a false PASS would look like

A value FAIL is reported only if the GPU reproduces it at the witness point, and an accuracy FAIL only at the regime that triggered it. Every false positive so far was caught at one of those. Nothing does the same for a PASS. A PASS depends on the reference side being right, and nothing later in the pipeline checks the reference side. The `__eq__` bug in [how-it-works.md](how-it-works.md) is what that looks like: it was found by reading the class against `torch.Tensor`, not by a test, and the regression cases were added afterwards. What can honestly be claimed is the process, not the outcome.

## The probabilistic PASS

One kind of PASS is probabilistic. When Volta cannot canonicalize a pair, equality is decided by agreement at random points over a finite field, with error at most (d/2⁶¹)³ per pair. Such a row says `via: pit` and carries its seed. The seed is drawn fresh for every judgment. So under optimization pressure, a kernel that is not equal passes with probability ~2⁻¹⁸³ each time it is judged, and a policy cannot climb a reward that never arrives.

The encoding has its own limit. The exponent field is not the reals, so `exp(q·x) = 1` in it, and the guarantee holds for expressions whose coefficient arithmetic stays below q ≈ 2⁶¹. float32 constants are dyadic with 24-bit numerators, so no product of them reaches q and no practical sum does. An expression built to cross it would be visible as such.

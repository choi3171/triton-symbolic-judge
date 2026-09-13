# Limits

The coverage table is in the [README](../README.md#limits); cost and the caps are in
[caps.md](caps.md). This is the rest.

**What the method actually cannot do.** A loaded value used as an *address*. The
memory model maps concrete offsets to terms, so a symbolic index has no slot.
That splits three ways and only the last is a wall: a scatter-*add* is order-free
and has a denotation in the existing algebra (`out[j] = Σᵢ select(idxᵢ = j, vᵢ,
0)`); a scatter with provably distinct indices needs that plus a distinctness
obligation, worth checking anyway; a scatter that may collide needs array theory
*and* is order-dependent on real hardware, so "undecidable here" and "this kernel
is nondeterministic" are the same fact.

Above that sits the grid quantifier: the grid is enumerated concretely, which is
why shapes must be fixed. Symbolic thread counts are solved for races
(GPUVerify's two-thread reduction) and that reduction does not transfer to
values, because an output depends on every program instance rather than a pair.

Integers are the same story from the other side. They are concrete — Python
`int`s with real two's-complement wraparound — and that concreteness is what
makes the memory check decidable without a solver call: the store is a dictionary
keyed by `(buffer, offset)`, so out-of-bounds is a comparison and a write
conflict is a lookup. Symbolic integers would buy symbolic shapes and cost that.

**Other measured limits.** TTIR coverage is 12 of 26 core ops plus `arith`,
`math` and `scf.for`; there is no `scf.if` on a loaded value, no `scf.while`, no
block pointers, and atomics are `fadd` only. Precision tracking is keyed on term
identity, so two paths to the same normal form at different precisions take the
minimum. The precondition layer is interval-based and cannot see relations; its
three rules are specific to the softmax family, and underflow is flagged rather
than judged, because a result that correctly rounds to zero in fp32 is harmless
in absolute error and 100 % wrong in relative error — which of those is the spec
is a decision, not a fact. Volta is used only as a decision procedure:
`check_equivalent` over our own arena. Its race checking, its PTX front-end and
the structured-CTA premise of its completeness proof are not carried over, so the
soundness of this interpreter is argued informally, not proved.

**These numbers describe these two corpora.** Both are small GitHub modules and
one-shot model answers. They under-represent production kernels —
tensor-parallel collectives, MoE routing, paged attention — where symbolic
addressing and dynamic shapes are normal rather than exceptional. Read the
coverage as a property of the corpora, not of the method.

**Every input in the LLM corpus fits in one block, and the verdicts do not
depend on it.** 155 of 155 first inputs have at most 1024 elements, so 123 of
the 156 rows were judged with every launch a single program: `pid` was 0
everywhere, and the arithmetic on it, and any tail past the first block, went
untested. Re-judged with the leading dimension set to an odd m with m·inner >
2048 — 33 for `[4,4,4,4]`, 2112 elements, at least two blocks and a tail for
any BLOCK from 128 to 2048 (`tvj/judge/shape2_run.py`,
`results/triton_traces_shape2.jsonl`) — **0 of the 89 PASS rows change
verdict**: no FAIL, no crash, and 9 of the 10 FAILs stay FAILs
(`python3 -m tvj.measure.shape2`). Three verdicts move, none across PASS: one
FAIL becomes UNKNOWN because the witness Z3 found at the small shape is out of
the numeric search's reach at the large one; one UNKNOWN times out; and one
UNKNOWN becomes NONDETERMINISTIC — a kernel whose race needs more than one
program to show, which a single-program grid cannot. The corpus' own tolerance
test at the small shape was blind to nothing here. The theorem is still at a
point; two points agreeing is evidence about these kernels, not a proof about
the next one.

**And nothing here has faced an adversary — which is three claims, not one.**
Every kernel judged was written without knowledge of this judge: Inductor is a
compiler, and the LLM corpus is a model answering in good faith. Kernels
optimised against a *different* checker can be had today: Dr. Kernel's policy is
published and 3 % of its output still hacks past its own check (see [prior work](prior-work.md)).
A policy trained against *this* judge is the third, and this repository cannot
answer it — that claim is about training dynamics, and no amount of judging
kernels that already exist settles one.

**Asked as a rate it is expensive, and probably the wrong question.** "Does
putting the judge in the loop lower how often hacking happens" is a
two-proportion test against a base rate of 2–3 % — which Dr. Kernel's numbers and
the 15 rows of 556 in [findings.md](findings.md) independently agree on — so roughly 1,500 judged
rollouts per arm to resolve 3 % against 1.5 %, in two training runs (arithmetic,
not a measurement). And hacking is not stationary: near zero until a policy finds
the exploit and not after, so a rate averages over the only interesting event.
Asked as an incident — when an exploit emerges, does the judge see it, and does
the directive close that axis — it is an existence proof and needs one. That is
the shape `tvj/checks/testgen_validate.py` already has: seven exploits, two
directives, all seven caught. What is missing is not sample size. It is that all
seven were transcribed by hand out of papers, and none emerged from a policy that
was trying.

**What a false PASS would look like.** A value FAIL is believed only if the GPU
reproduces it at the witness point, and an accuracy FAIL only at the regime that
fired; every false positive so far was caught at one of those. Nothing plays that
role in the other direction. A PASS rests on the reference being right, and the
reference is the one thing nothing downstream can check. The `__eq__` bug in [how-it-works.md](how-it-works.md)
is what that looks like in practice: it was found by reading the class against
`torch.Tensor`, not by any test, and the regression cases were added afterwards.
The honest claim is the process, not the outcome.

One PASS is probabilistic by construction. When Volta cannot canonicalise a pair,
equality is decided by agreement at random points over a finite field, with error
at most (d/2⁶¹)³ per pair; such a row says `via: pit` and carries its seed. The
seed is drawn fresh for every judgement, which is what makes this safe under
optimisation pressure: a kernel that is not equal passes with probability ~2⁻¹⁸³
each time it is judged, and a policy cannot climb a reward that never arrives.
The encoding's own boundary is stated rather than hidden — the exponent field is
not the reals, so `exp(q·x) = 1` in it, and the guarantee holds for expressions
whose coefficient arithmetic stays below q ≈ 2⁶¹. float32 constants are dyadic
with 24-bit numerators, so no product of them reaches q and no practical sum
does; an expression built to cross it would be visible as such.

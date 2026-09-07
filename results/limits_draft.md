## Limits

**Judged coverage.** 75 % of 400 Inductor-generated rows, 65 % of 125
LLM-written ones. What is left is not one thing, and the useful way to split it
is by *who controls whether a kernel lands there*:

| | KernelBook | LLM traces | can a generator steer into it? |
|---|---|---|---|
| the reference uses a torch op we do not model | 12.2 % | 3.2 % | no — the task is given |
| the kernel uses a TTIR construct we do not model | 4.8 % | 4.8 % | **yes** |
| a torch tail after the kernels we could not replay | — | 11.2 % | yes, and see below |
| the candidate does not compile | — | 5.6 % | no (all seven are labelled incorrect) |
| our caps: 150 s, 4 GB for Volta, an 8 M term budget | 4.2 % | 2.4 % | no — raise them on a real machine |
| our plumbing failed to open the row | 0.8 % | 2.4 % | no |
| the reference itself is random | 1.2 % | 1.6 % | no |
| **the method genuinely cannot decide** | **1.2 %** | **4.0 %** | — |

So the slice a generator could drift into is about 5 % of either corpus, not
25 % or 35 %, and it is a list of eight named TTIR constructs rather than a
region. Most of the rest is unpaid implementation debt with the items written
down; paying it down is the most useful work available.

**The torch tail is the largest single bucket, and it has a one-line fix that is
not ours to apply.** 14 of the 22 undecided LLM rows are wrappers that finish the
computation in PyTorch after the kernels; we replay what we can interpret and
these are the residue. Requiring generation to emit a single fused Triton kernel
removes the bucket entirely — and that constraint is not a concession, because a
torch tail also costs a kernel launch and a materialised intermediate. The
constraint that makes a kernel analysable is the one that makes it fast.

**What the method actually cannot do.** A loaded value used as an *address*.
Our memory model maps concrete offsets to terms, so a symbolic index has no
slot. That splits three ways and only the last is a wall: a scatter-*add* is
order-free and has a denotation in the existing algebra (`out[j] = Σᵢ
select(idxᵢ = j, vᵢ, 0)`); a scatter with provably distinct indices needs that
plus a distinctness obligation, which is worth checking anyway; a scatter that
may collide needs array theory *and* is order-dependent on real hardware, so
"we cannot decide it" and "this kernel is nondeterministic" are the same fact.
Above that sits the grid quantifier: we enumerate the grid concretely, which is
why shapes must be fixed. Symbolic thread counts are solved for races
(GPUVerify's two-thread reduction) and that reduction does not transfer to
values, because an output depends on every program instance rather than a pair.

**These numbers describe these two corpora.** Both are small GitHub modules and
one-shot model answers. They systematically under-represent production kernels —
tensor-parallel collectives, MoE routing, paged attention — where symbolic
addressing and dynamic shapes are the norm rather than the exception. Read the
coverage as a property of the corpora, not of the method.

**And nothing here has faced an adversary.** Every kernel judged was written
without knowledge of this judge: Inductor is a compiler, and the LLM corpus is a
model answering in good faith. What an RL policy would do to it is the open
question, and it is not one this repository can answer.

**Never a false pass, though.** Shapes are fixed and small. A FAIL on the value
obligation is believed only if the GPU reproduces it at the witness point — every
false positive this project produced was caught there, and five were retracted
that way in the last session. The other obligations are deliberately *not* gated
on hardware, because hardware is silent for them by construction. All
measurements are `sm_75`.

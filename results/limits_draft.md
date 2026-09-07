## Limits

<!-- The table below is generated: `python3 -m tvj.measure.limits`.  Do not hand-edit
     it.  The previous version was written by hand and every number drifted -- it
     still said 125 LLM rows after the corpus was finished at 156, and quoted a
     headline percentage that contradicted its own steerable column. -->

**Judged coverage.** 76 % of 400 Inductor-generated rows, 63 % of 156 LLM-written rows.

|                                                            | KernelBook | LLM traces | can a generator steer into it?    |
|------------------------------------------------------------|------------|------------|-----------------------------------|
| the reference uses a torch op we do not model              | 12.5 %     | 3.2 %      | no — the task is given            |
| the kernel uses a TTIR construct we do not model           | 4.5 %      | 1.9 %      | **yes**                           |
| a torch tail after the kernels we could not replay         | 0.2 %      | 10.3 %     | yes, and see below                |
| the candidate does not compile or run at all               | —          | 9.6 %      | no                                |
| our caps: the 150 s alarm, 4 GB for Volta, the term budget | 3.8 %      | 3.2 %      | no — raise them on a real machine |
| our plumbing failed to open the row                        | 0.2 %      | 1.3 %      | no                                |
| the reference itself is random                             | 0.5 %      | 1.3 %      | no                                |
| the row hangs the judge and never returns a verdict        | 0.2 %      | —          | no                                |
| **the method genuinely cannot decide**                     | 2.0 %      | 6.4 %      | —                                 |

KernelBook: a generator could steer into 4.8 % of rows (19/400); the method's own wall is 2.0 % (8/400).
LLM traces: a generator could steer into 12.2 % of rows (19/156); the method's own wall is 6.4 % (10/156).

the named constructs behind the steerable buckets
  KernelBook:
      6  float->int conversion of a symbolic value
      5  comparison on an integer loaded from memory
      3  extern transposed convolution
      2  data-dependent integer arithmetic (integer loaded from memor
      2  arith.uitofp of a comparison result (i1->float sign conventi
  LLM traces:
      1  scf.while
      1  data-dependent integer arithmetic (integer loaded from memor
      1  arith.uitofp of a comparison result (i1->float sign conventi

The split that matters is not how much is left but **who controls whether a
kernel lands there**. A reference op we do not model is fixed by the task, so no
policy can aim at it. A TTIR construct we do not model is a target. On that
reading the slice a generator could drift into is 4.8 % of KernelBook and 12.2 %
of the LLM corpus -- and it is a list of named constructs, not a region. Most of
the rest is unpaid implementation debt with the items written down; paying it
down is the most useful work available.

**The torch tail is the largest steerable bucket in the LLM corpus, and its fix
is not ours to apply.** These are wrappers that finish the computation in PyTorch
after the kernels; we replay what we can interpret and these are the residue.
Requiring generation to emit a single fused Triton kernel removes the bucket
entirely -- and that constraint is not a concession, because a torch tail also
costs a kernel launch and a materialised intermediate. The constraint that makes
a kernel analysable is the one that makes it fast.

**What the method actually cannot do.** A loaded value used as an *address*. Our
memory model maps concrete offsets to terms, so a symbolic index has no slot.
That splits three ways and only the last is a wall: a scatter-*add* is order-free
and has a denotation in the existing algebra (`out[j] = sum_i select(idx_i = j,
v_i, 0)`); a scatter with provably distinct indices needs that plus a
distinctness obligation, which is worth checking anyway; a scatter that may
collide needs array theory *and* is order-dependent on real hardware, so "we
cannot decide it" and "this kernel is nondeterministic" are the same fact.

Above that sits the grid quantifier: we enumerate the grid concretely, which is
why shapes must be fixed. Symbolic thread counts are solved for races
(GPUVerify's two-thread reduction) and that reduction does not transfer to
values, because an output depends on every program instance rather than a pair.

Integers are the same story from the other side. They are concrete -- Python
`int`s with real two's-complement wraparound -- and that concreteness is what
makes the memory obligation decidable without a solver call: `store` is a dict
keyed by `(buffer, offset)`, so out-of-bounds is a comparison and a write
conflict is a lookup. Symbolic integers would buy symbolic shapes and cost that.

**These numbers describe these two corpora.** Both are small GitHub modules and
one-shot model answers. They systematically under-represent production kernels --
tensor-parallel collectives, MoE routing, paged attention -- where symbolic
addressing and dynamic shapes are the norm rather than the exception. Read the
coverage as a property of the corpora, not of the method.

**And nothing here has faced an adversary.** Every kernel judged was written
without knowledge of this judge: Inductor is a compiler, and the LLM corpus is a
model answering in good faith. What an RL policy would do to it is the open
question, and it is not one this repository can answer.

**What a false PASS would look like, and why we cannot rule one out.** A FAIL on
`value` is believed only if the GPU reproduces it at the witness point, and one
on `accuracy` only if the GPU reproduces it at the regime that fired; every false
positive this project produced was caught at one of those. Nothing plays that
role in the other direction. A PASS rests on the reference being right, and the
reference is the one thing nothing downstream can check -- a wrong handler makes
a correct kernel FAIL, which is loud, and can make a wrong kernel PASS, which is
silent.

That is not hypothetical. `STensor` had no `__eq__`, so `mask == 0` evaluated to
Python `False` and `attn.masked_fill(mask == 0, -1e9)` built a reference with the
mask deleted; a kernel that also dropped the mask would have matched it. It was
found by reading the class against `torch.Tensor`, not by any test, and the
regression cases for it were added afterwards (`tvj/checks/spec_agree.py`). The
honest claim is the process, not the outcome: 113 handler-vs-torch cases, a
signature cross-check, a term-algebra fuzzer with proven teeth, and every
reference whose edge semantics were recovered from torch's C++ rather than a
published definition marked as such in the verdict.

All measurements are `sm_75`.

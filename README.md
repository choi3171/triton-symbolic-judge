# A symbolic judge for Triton kernels

Checks a Triton GPU kernel against the PyTorch module it is supposed to replace,
by comparing the two as **expressions over symbolic inputs** rather than by
running both on sampled data. A defect that only shows up outside the test
distribution cannot hide from it.

The motivating question: LLM-generated GPU kernels that pass a tolerance test —
are they actually correct?

```
pip install -r requirements.txt   # torch has to be a CUDA build -- see the file
./setup.sh                        # fetch Volta, KernelBench and the corpora; build the bridge
python3 verify.py                 # re-run every claim here (35/35, ~6 min)
python3 verify.py --only volta    # or part of it, on a machine that cannot hold the rest
python3 verify.py --all           # + three slow claims; one needs ~12 GB and skips itself below that
```

Almost every number below is produced by a script that `verify.py` re-runs and
matches against the claim. Two groups are not, and say so where they appear: the
Cost table and the truncation comparison under Prior work. Both are hand-copied,
which is the thing the Limits table was made generated to stop; they can become
claims when someone writes the script. A claim that is *about the architecture*
is tagged `[sm_75]`;
another architecture answering differently is information, not a failure, so
`verify.py` reads the device it is on and reports those apart from the count. A
claim that merely needs a GPU is tagged `[gpu]` and is counted everywhere — the
distinction matters, because tagging a claim about the judge `[sm_75]` would let
a real regression in it read as "expected to differ".

## How it works

Two term graphs are built and compared.

**The reference side.** `STensor` is a numpy object array whose elements are
terms, wearing a torch surface. Substitute it for a module's parameters and call
`forward`, and the *unmodified* reference code emits terms — `__torch_function__`
picks up `torch.matmul`, `F.softmax`, `F.layer_norm` and the rest.

**The kernel side.** `JITFunction.run` is hooked, so the generated code's real
GPU launches are recorded — kernel, grid, arguments, constexprs — and the TTIR is
interpreted symbolically over the whole grid. Tensors are matched to roles by
**storage base plus element offset**, not by object identity, because Inductor
hands out `reinterpret_tensor` views and non-contiguous output strides. Nothing
is asked of the generator beyond `launch(*inputs) -> output`.

Floating point is modelled as **exact reals**. That is the load-bearing choice:
it makes reassociation free (split-K, flash attention, tree reductions are all
equal), and it means the two things reals cannot see — representation error and
precision contracts — need checks of their own.

### First, is it a function of its inputs at all?

Before any of the five, the kernel is run twice at the same inputs and the two
answers must agree bit for bit. If they do not, nothing below means anything —
and if the reference is *also* nondeterministic, the task itself is, and there is
nothing to refine either way. Both outcomes are the verdict `NONDETERMINISTIC`
rather than a FAIL, because neither is a claim about correctness.

A kernel that draws randomness is not given up on, though. The draws are lifted
to inputs: the kernel's k-th draw becomes a named buffer, the reference's k-th
draw is bound to the same name, and the question becomes the one worth asking —
*given the same draw, do the two compute the same thing?* That binding is an
assumption, and it is recorded (below), not assumed away. Randomness that never
materialises as a tensor of its own — `native_dropout` returns its mask, not its
draw — cannot be bound to anything and is refused.

### Five checks, not one

Each is a separate claim the kernel has to satisfy; a verdict names which one
failed.

| check | question | instrument |
|---|---|---|
| **value** | the same real number? | AC normal form → [Volta](https://github.com/willtunnels/volta)'s decision procedure → Z3 case splitting for piecewise terms |
| **memory** | does it read what nothing wrote, skip an output, or race with itself? | symbolic execution of the whole grid |
| **precision** | is it not *less* precise than the reference? | a lattice with a direction, because `ieee → tf32` is real-equal but not a refinement |
| **precondition** | does the real-number proof still mean anything in float32? | interval analysis with three relational rules |
| **accuracy** | the same expression, arranged so float32 loses digits? | evaluate both terms in float32 and in float64, compare the *errors* |

Value is decided in three stages, cheapest first. Terms are hash-consed and
normalised for associativity and commutativity, so most pairs come out
*identical* and no solver runs at all: **257 of 272** KernelBook value decisions
finish there. Volta's exponential-polynomial procedure takes 14 more. Z3 case
splitting handles piecewise terms — the step Volta's paper says "could be handled
by case splits" and declines to take — and settles 1.

AC does that much of the work because of **delegation**: when both sides hand the
same operation to the same library call with the same arguments, it becomes one
uninterpreted symbol and hash-consing decides it for free. That is sound only if
the symbol's name encodes everything the result depends on, so every builder
fails closed — an unencoded parameter raises rather than producing a symbol. If
the two sides disagree, the symbols are expanded and the pair is decided the slow
way, because two different uninterpreted symbols are not a counterexample.

### When a FAIL is believed

Two of the five are checked against hardware before they are reported.

**Value** is checked at its witness point — the concrete input where the two
terms take different values. If the GPU does not reproduce the disagreement
there, the verdict is UNKNOWN, not FAIL. Every false positive this project has
produced was caught by that rule. "Reproduces" is measured *relative to the
reference's own magnitude*, for the reason row 308 below is in this README at
all: a bar spelled in absolute terms hides a 190 % error under an output of
~1e-4, and at an output of ~1e8 it is below float32's own spacing and means
nothing either way.

**Accuracy** is checked at the input regime that made it fire. Cancellation is
silent at the benchmark's inputs — that is the entire reason the check exists —
and loud at the shifted inputs that trigger it, so silence *there* means a
modelling gap rather than a defect (`tvj/checks/acc_gate.py`).

The other three are deliberately not checked that way, and must not be. Hardware
is structurally silent for them: a stale buffer holds the right answer, tf32 is
ignored on sm_75, a narrowed validity radius only shows at extreme inputs.
Gating them would discard exactly the defects a test cannot reach.

A separate verdict, `PASS-ASSUMING`, carries a stated assumption the judge cannot
discharge. There are two kinds, and both are properties of the *input data*
rather than of the kernel, which is why neither can be proved from the kernel and
both are written into the verdict instead:

- **`index-distinct`** — a scatter to a data-dependent address is well defined
  exactly when the indices are pairwise distinct. torch is in the same position
  and resolves it the same way: `scatter_` is documented as nondeterministic when
  indices collide.
- **`rng-correspondence`** — when both sides draw randomness, their k-th draws
  are matched by order and by shape. Nothing outside the two programs can confirm
  that those are the same draw.

A PASS that rests on an unstated assumption is not a PASS, so the assumption
travels with the verdict.

### The reference is the weak point

A wrong reference makes a correct kernel FAIL, which is loud, and can make a
wrong kernel PASS, which is silent. Nothing downstream can catch the second.

Three defects of one shape shipped before this was taken seriously: a
`F.linear(bias=)` keyword dropped by a `**kwargs`, `mean(axis=-1)` turned into a
global mean, and `avg_pool2d`'s `ceil_mode` and `count_include_pad` swapped
inside a lambda. None of them raised. More recently `STensor` had no `__eq__`, so
`mask == 0` evaluated to Python `False` and `masked_fill(mask == 0, -1e9)` built
a reference with the mask silently deleted.

So the front-end is checked two ways: `tvj/checks/spec_sigcheck.py` lines every
handler up against torch's own signature (nothing swallowed, mis-positioned, or
declared and unread), and `tvj/checks/spec_agree.py` pushes 113 cases through
both torch and the front-end and compares the numbers. Handlers whose edge
semantics were recovered by reading torch's C++ rather than a published
definition are marked, and a FAIL that rests on one says so.

`tvj/core/semantics.py` is the artifact underneath all of it: 23 decisions the
interpreter had to make because Triton does not answer them — what a masked lane
loads, whether i32 index arithmetic wraps, whether a reduction is a tree or a
fold — each with its basis and its evidence, five of them measured against
hardware.

## What it found

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

**A counterexample yields an axis, not just a point — when the kernel leaves
something out.** The judge reports which named buffers a disagreement rests on,
so `tvj/judge/testgen.py` turns one exploit into a harness directive — *vary
these parameters*, *poison this buffer*. Derived from a single kernel, two
directives catch all 7 of the documented exploits; the corpus' own correctness
check catches none of them.

Over the 37 FAILs in the two corpora the axis comes out for **26**, and what
separates them is the shape of the defect rather than the size of the corpus.
`vary-parameter` and `vary-input` are named by the buffers the *reference* reads
and the *kernel* does not, so they fire when a kernel omits something — the LLM
shortcut, where a parameter's default is the identity element of whatever
consumes it. A compiler does not omit; it reads everything and arranges it
differently. Row 17 (`leaky_relu(a1)+a2` against `a1+leaky_relu(a2)`) and row 308
(same-shaped tensors in swapped roles) both read every buffer, so the first rule
that looked for an omission found nothing on either.

Redrawing the parameters still separates them, because the two *arrangements* of
the same parameters differ — which is why the rule now fires whenever the
disagreement rests on parameters at all, not only when one is ignored. Row 308 is
the measured case: of the five rows whose own benchmark passes them, it is the
one that only a parameter redraw catches.

**A generated check has to be able to see the row it came from.** It inherited
the harness's comparison — `allclose(atol=1e-2, rtol=1e-2)` — and that has an
absolute floor, so at a small reference magnitude it is blind to a disagreement
the judge found. Three FAILs are in that position, and two of them are rows their
own benchmark passes. At the corpus' own seeds the absolute comparison misses 13
of 15 trials on them; scaled by the reference's magnitude, the same measure the
hardware gate uses, it catches 15 of 15. It stays silent where it should: on the
rows the judge PASSes the worst relative error is 4 × 10⁻⁷ against a bar of
10⁻⁴, so there is about 250× of headroom before ordinary float32 reassociation
would trip it (`tvj/measure/relcompare.py`). So `compare-relative` is emitted
when the record says the absolute floor would hide the defect — a directive that
says how to *measure* rather than what to vary, which is why it does not count
toward the 26.

Two things that did **not** work are worth the same space. Permuting same-shaped
inputs looked like the natural axis for the swapped-role defects — of the FAILs
that yielded no axis under the first rule, 18 have two inputs of one shape and
the reference is asymmetric in them in all 18 — and it catches nothing the
un-permuted harness does not already catch, on any of the 18 (hand-run; there is
no `verify.py` claim for it). And `poison-output`, the axis for a stale-buffer
read, has never fired on a natural corpus: neither corpus contains a memory FAIL,
so that class exists here only as the hand-written fixtures in
`tvj/fixtures/hacks.py`.

The counts above come from `python3 -m tvj.measure.directives`, which reads the
two published run records.

### Corpus results

On one criterion — *the corpus' own numeric check passes and the judge still
FAILs* — there are 13, every one corroborated on hardware before being counted:

| corpus | judged | tolerance passes, judge FAILs |
|---|---|---|
| 400 Inductor-generated (KernelBook) | 76 % | 5 — at the witness point the GPU shows up to 7.3 × 10³ |
| 156 LLM-generated Triton | 63 % | 8 — 5 on value, 3 on accuracy |

One of the five is in the count only by luck, and says so: KernelBook row 17
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

Across the 304 decided KernelBook rows the tolerance test and the judge **agree
on 295** — 272 both pass, 23 both fail — **and part company on 9, all in one
direction**: 5 rows the tolerance test passes and the judge fails, and 4 it
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

## Cost

Hand-measured, not asserted by `verify.py`: `tvj/measure/scale.py` and
`tvj/checks/volta_attn.py` are the scripts, and only their verdicts are claims.

Attention at D=16, BM=BN=16, comparing three formulations pairwise:

| L | key blocks | outputs | symbolic execution | Volta, ref vs flash | term ops |
|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 512 | 0.3–1.8 s | 0.3 s | 3.2 M |
| 64 | 4 | 1024 | 1.0–2.0 s | 3.6 s | 26.4 M |
| 128 | 8 | 2048 | 4.4–5.5 s | 33.6 s | 206.8 M |
| 256 | 16 | 4096 | 18–30 s | see below | 349 M–509 M |

**Memory binds before time.** At L=128 the bridge peaks at 0.56 GB for ref vs
safe, 0.77 GB for safe vs flash, and **9.19 GB for ref vs flash** — twelve times
more for the same problem size. The reason: the naive reference does not subtract
the max, so its exponential polynomial cannot share the `−m` atom and the cross
products expand. The practical conclusion is that **the cost of deciding is
dominated by the difference in shape between the two kernels, not by their
size** — write the reference max-subtracted and the same comparison fits in
under a gigabyte.

At L=256 the well-shaped pairs stay under 4.5 GB in Volta while *our* Python side
(the term DAG plus JSON serialisation) reaches 7.6 GB, which makes bridge
serialisation the next bottleneck rather than the decision procedure.

Whole-corpus rows are far smaller: symbolic execution runs at a median of 0.17 s
and a 90th percentile of 1.44 s.

## Limits

<!-- generated: `python3 -m tvj.measure.limits`.  Do not hand-edit; the version
     written by hand drifted every time a corpus was re-run.  It reads the two run
     records -- `results/kernelbook.jsonl` and `results/triton_traces.jsonl`, or
     the live `data/kb_live.jsonl` if a run is in progress -- and refuses to print
     a table when one of them is missing, where it used to print "100 % hangs the
     judge" instead. -->

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
| the row hangs the judge and never returns a verdict        | —          | —          | no                                |
| **the method genuinely cannot decide**                     | 2.2 %      | 6.4 %      | —                                 |

What matters is not how much is left but **who controls whether a kernel lands
there**. A reference op we do not model is fixed by the task, so no policy can
aim at it. A TTIR construct we do not model is a target. On that reading a
generator could aim at 4.8 % of KernelBook and 12.2 % of the LLM corpus, and it
is a list of named constructs rather than a region — 17 rows of arithmetic on an
integer loaded from memory, 3 of transposed convolution, 1 of `scf.while`. The
rest is unpaid implementation debt with the items written down.

**The torch tail is the largest steerable bucket in the LLM corpus, and its fix
is not ours to apply.** These are wrappers that finish the computation in PyTorch
after the kernels. Requiring generation to emit a single fused Triton kernel
removes the bucket entirely — and that is not a concession, because a torch tail
also costs a launch and a materialised intermediate. The constraint that makes a
kernel analysable is the one that makes it fast.

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

**And nothing here has faced an adversary.** Every kernel judged was written
without knowledge of this judge: Inductor is a compiler, and the LLM corpus is a
model answering in good faith. What an RL policy would do to it is open, and not
a question this repository can answer.

**What a false PASS would look like.** A value FAIL is believed only if the GPU
reproduces it at the witness point, and an accuracy FAIL only at the regime that
fired; every false positive so far was caught at one of those. Nothing plays that
role in the other direction. A PASS rests on the reference being right, and the
reference is the one thing nothing downstream can check. The `__eq__` bug above
is what that looks like in practice: it was found by reading the class against
`torch.Tensor`, not by any test, and the regression cases were added afterwards.
The honest claim is the process, not the outcome.

## Prior work

**[Gimlet Labs](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels)**
(Taneja, St John, Serrino; ARRAY 2026 at PLDI) built the closest thing to this,
and reached the same two structural decisions independently: parse `.ttir`, and
model floating point as exact reals. Their reference comes from
`torch.compile`'s FX graph rather than from running `forward` on symbolic
tensors, and they scalarise into Z3 directly; on 26 KernelBench Level 1 kernels
they report 16 proved, 8 unknown, and **2 that passed numeric testing while being
mathematically inequivalent**. Two efforts landing on the same IR and the same
numeric model is a point in favour of both choices.

What is here that is not there starts from their own stated limitation —
*"differing use of floating point values can lead to accuracy bugs"* that the
approach cannot address:

- **Four checks besides value equality**, including the accuracy check that
  answers exactly that limitation, and a memory check without which Sakana's
  stale-buffer exploit is *equal* over the reals (the output is a buffer nobody
  wrote).
- **Hardware corroboration** before a FAIL is reported.
- **AC normal form before any solver**, which decides 257 of 272 value questions
  with no solver call at all.
- **A counterexample yields an axis**, which becomes a harness check that runs
  without the judge.
- **Corpus scale and split**: 400 compiler-generated rows and 156 LLM-written
  ones, measured separately, because they fail in different ways.

The two also bound the problem differently. They **truncate reductions** — sum a
few terms instead of all of them — where this shrinks **shapes** and keeps every
reduction whole. Both make the term graph small, but they are not the same
approximation: truncation can hide a defect that only appears past the cut. On 40
KernelBook rows with the cap at 4, **39 verdicts agree**, and the one that
differs is a PASS becoming UNKNOWN rather than a missed defect
(`tvj/measure/truncated.py`, hand-run: it has no `verify.py` claim). The reason is mundane — these kernels reduce over
4–16 elements, so a cap of 4 barely bites. The two choices are interchangeable
*on this corpus*, not in general; a kernel whose reduction is where the bug lives
would separate them, and neither corpus has one.

Also relevant: [*The Correctness Illusion in LLM-Generated GPU
Kernels*](https://arxiv.org/abs/2606.20128) starts from the same observation and
answers it with better fuzzing on the input axis. The axes here are ones a
sampler does not reach — module parameters, stale memory, precision, numerical
stability — and symbolic inputs remove the notion of a sampling blind spot
rather than moving it.

## Layout

```
tvj/core/      term algebra and semantics      terms  ttir  sexec  semantics  bounded
tvj/decide/    deciding whether two terms agree  volta_bridge  casesplit  numeric  ranges  accuracy  delegate
tvj/front/     getting terms out of torch and the GPU   spec  capture  torchtrace  shapes
tvj/judge/     the judge and the corpus runners  judge  kernelbook_run  traces_run  report  testgen
tvj/fixtures/  kernels and references the checks use    kernels  attn  hacks  sm  probes  mutants
tvj/checks/    scripts that assert something     check  spec_test  spec_agree  delegate_test  ...
tvj/measure/   scripts that measure something    difftest  ieee_gap  limits  directives  reward_hack_lit  ...
tvj/tools/     open one row and look at it       kb_debug  memcheck  traces_repro
verify.py      re-runs all of the above and asserts every claim
```

`tvj/judge/judge.py` is the whole judgement; both corpus runners are thin
adapters over it. `PIPELINE.md` has the data flow.

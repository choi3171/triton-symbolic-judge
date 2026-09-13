# Prior work

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
- **AC normal form before any solver**, which decides 257 of 277 value questions
  with no solver call at all; one Volta call per output *shape* rather than per
  output element; and, where Volta's canonicalisation blows up, Mirage's
  random-point evaluation over a finite field in front of it rather than a bigger
  machine.
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
(`tvj/measure/truncated.py`, hand-run: it has no `verify.py` claim). The reason
is mundane — these kernels reduce over 4–16 elements, so a cap of 4 barely bites.
The two choices are interchangeable *on this corpus*, not in general; a kernel
whose reduction is where the bug lives would separate them, and neither corpus
has one.

Two more start from the same problem and answer it elsewhere.

**[Dr. Kernel](https://arxiv.org/abs/2602.05885)** (Liu, Xu, Li, Zheng, Li, Liu,
He; ICML 2026) is the closest work on reward hacking rather than on method: an RL
environment, KernelGYM, that trains a policy to write Triton and treats hacking
as a first-class problem instead of an afterthought. Its check is *launch
presence* — a candidate that executes no Triton kernel in either mode is
incorrect — which this judge also catches, by the same interception, as one of
the seven documented hacks. What makes the paper useful here is the number it
reports with that check switched on: **3 % of Level 2 and 1.7 % of Level 1 still
hack**. That residue is the population this judge is for, and it is the only
published source of exploits that emerged from a policy rather than from a paper.

[*The Correctness Illusion in LLM-Generated GPU
Kernels*](https://arxiv.org/abs/2606.20128) (Sarkar) answers the same observation
with better fuzzing on the input axis. The axes here are ones a sampler does not
reach — module parameters, stale memory, precision, numerical stability — and
symbolic inputs remove the notion of a sampling blind spot rather than moving
it.

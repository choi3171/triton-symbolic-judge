# Cost and caps

## Cost

Hand-measured, not asserted by `verify.py`: `tvj/measure/scale.py` and
`tvj/checks/volta_attn.py` are the scripts, and only their verdicts are claims.
Sizes rather than times — a time belongs to whatever machine ran it, and these
numbers do not.

Attention at D=16, BM=BN=16, comparing three formulations pairwise:

| L | key blocks | outputs | term ops |
|---:|---:|---:|---:|
| 32 | 2 | 512 | 3.2 M |
| 64 | 4 | 1024 | 26.4 M |
| 128 | 8 | 2048 | 206.8 M |
| 256 | 16 | 4096 | 349 M–509 M |

**The cost of deciding is per shape, and it is dominated by the difference in
shape between the two kernels, not by their size.** Lane by lane, at L=128 the
bridge peaks at 0.56 GB for ref vs safe, 0.77 GB for safe vs flash, and **9.19 GB
for ref vs flash** — twelve times more for the same problem, because the naive
reference does not subtract the max, so its exponential polynomial cannot share
the `−m` atom and the cross products expand. Deciding one representative per
shape instead, the same ref vs flash pair at L=128 — which exceeds a 4 GB cap
lane by lane — decides in **0.14 GB and 0.31 s**, every lane getting the verdict
the lane-by-lane run gives; at L=64 it is 1.04 GB → 0.03 GB, and across the pairs
both paths can run, 154× less Volta time (`tvj/measure/lanes.py`). What that
leaves is the per-shape cost, which is a matter of how many distinct
denominators one output sums rather than of size — the paragraph on caps below.

At L=256 the well-shaped pairs stay under 4.5 GB in Volta while *our* Python side
reaches 7.6 GB. That figure predates the bridge's binary wire format: handing one
node to Volta used to cost a Python dict and about eighty bytes of JSON text, 325
bytes against the nine it costs now. What is left on this side is the term DAG
itself, at a measured 390 bytes per interned term — which is where the 8 M term
budget comes from, and the next thing worth shrinking.

## What is left, and who can steer into it

The Limits table in the [README](../README.md#limits) sorts every row the judge does not decide by cause.

What matters is not how much is left but **who controls whether a kernel lands
there**. A reference op we do not model is fixed by the task, so no policy can
aim at it. A TTIR construct we do not model is a target, and so is our own cost:
on that reading a generator could aim at 24 rows of KernelBook and 19 of the LLM
corpus, 6.0 % against 12.2 %. The two are not the same kind of target, though.
The TTIR bucket is a list of named constructs — 17 rows where an integer is
derived from a real value or read from memory, 3 of transposed convolution, 1 of `scf.while` — and
it shrinks as they are implemented. The caps bucket is a region, and it is the
paragraph after next.

**The torch tail is the largest steerable bucket in the LLM corpus, and its fix
is not ours to apply.** These are wrappers that finish the computation in PyTorch
after the kernels. Requiring generation to emit a single fused Triton kernel
removes the bucket entirely — and that is not a concession, because a torch tail
also costs a launch and a materialised intermediate. The constraint that makes a
kernel analysable is the one that makes it fast.

**Our caps are steerable — along one axis now, not two.** 13 of those 20 rows
died inside Volta, on its address space or its term-operation budget. Two things
can put a pair there. *Width* — many lanes of the same shape, each canonicalised
separately — used to: holding the reference fixed, one correct attention kernel
decided in 0.56 GB and another in 9.19 GB, and the expensive one was the faster
one (`tvj/measure/steerable.py` demonstrates it: three formulations pairwise
equal over the reals, a cap between the cheapest and the dearest, and the judge
decides two and returns UNDECIDED on the third). Deciding one representative per
shape closes that axis: the 9.19 GB pair is 0.14 GB. *Division* remains. The 13
corpus rows are all of one kind — row 97 is 256 lanes of 2 shapes and each
representative alone exceeds the budget; row 61 is 679 nodes per output and
canonicalising one of them exceeds 3 GB — and it is not the kind the paper
allows for. Volta canonicalises to a rational N/D and adds two fractions with
different denominators by multiplying the denominators (`canon/ops.rs`,
`rat_add_v`). A multi-head attention output sums one fraction per head, each
over that head's own softmax denominator, so the common denominator has L^H
terms and the equality check N1·D2 = N2·D1 has, counted without building it
(`python3 -m tvj.measure.nf_rat kb 61`), about 10^6 monomials per output on row
61 and 10^11 on row 97 — each carrying an exponent polynomial. The
multiplicative depth of every one of these terms is 1. Volta's paper argues the
blowup away by depth: canonicalisation "may cause exponential blowup", it says,
but "since machine learning workloads do not typically have computations with
high multiplicative depth, this blowup does not happen in practice". That
argument is correct and does not cover this: the growth is exponential in the
number of heads whose fractions one output sums, and on these two corpora it
happens 13 times in 556 rows, in the kernels the paper is about. (An earlier
version of this paragraph blamed multiplicative depth; it was measured at 1.)

Evaluation at random points does not build the normal form, and on the same 4 GB
machine it decided 6 of those 13 in the experiment: rows 86, 310, 318 and 328
PASS, and **rows 61 and 97 FAIL, with a numeric witness the GPU reproduces** —
the two of the thirteen that fail the corpus' own tolerance test. Republished
across both corpora, with Volta held to a 60 s wall clock and the Z3 stage to a
size and a time budget so the stages after them get a turn, the caps bucket gave
up fourteen rows: eight KernelBook FAILs the GPU reproduces (61, 97, 116, 137,
194, 196, 233, 306), five PASSes, and LLM row 127 ([findings](findings.md)). Nothing Volta-bound is
left in it. Where the stage
reaches a row and does not decide it, the row is UNKNOWN and has a name: LLM rows
11 and 150 are genuinely different functions — the kernel omits softplus's
threshold branch — but the numeric witness search does not reach where they
differ; LLM rows 51 and 92 separate at the witness but the GPU does not reproduce
it, which is our modelling gap and is reported as such.

**And the bucket was not neutral.** Before the random-point stage existed, 6 of
the 15 KernelBook rows that hit one of our caps failed the corpus' own tolerance
test, against 23 of the 300 decided rows — 5.2 times the rate, Fisher p = 0.001.
Raising the caps (`TVJ_ROW_TIMEOUT=1800`, `VOLTA_MEM_GB=24`,
`TVJ_VOLTA_BUDGET=4e9`) decided four of the six, every one a FAIL the GPU
reproduced at the witness point. The caps were not holding rows nobody had got
to; they were holding defects. The enrichment is gone now — 0 of the 5 rows left in
the bucket fail tolerance — and it is gone for the right reason: all six are
FAILs at the default caps (61, 97, 137, 194, 196, 233). What remains in the
bucket is three rows the 150 s row alarm stops (176, 363, 372) and two over the
term budget (235, 344); none is blocked inside Volta.

The two that do not come back that way are the two blocked inside Volta rather
than by a cap of ours, and 24 GB is not enough for either. One is row 61, where
reading the generated wrapper settles it without the judge at all:
`Attention.forward(self, k, q)` takes k first, and the wrapper hands `w_k` the
second input and `w_q` the first — both `(4, 4, 1, 4)`, so `assert_size_stride`
is satisfied. The GPU disagrees by 0.26, deterministically. Memory does not reach
it; evaluation at random points does, in 9 ms, and the row is a FAIL. The
random-point stage, with budgets on Volta and Z3 so it is reached, bought the
whole Volta-bound part of this bucket; raising the row alarm and the term budget
is what is left to buy, and the rows it would buy are named above.

The number is not the reason; the representation is. A term graph is proportional
to the **work a kernel does rather than to the program that does it** — a tiled
matmul is Θ(M·N·K) nodes because every output element is denoted as a sum of K
products — so "make it bigger" is always available. Raising a cap is worth doing,
and the four rows above are what it buys; what it does not do is change what the
threshold is a function of, which is why the axis survives every raise. Most of this
section is that one fact in other clothes: shapes are fixed because the grid is
enumerated, integers are concrete because that is what makes the memory check a
dictionary lookup, and a branch on a loaded value is refused because there is
nothing symbolic to split. It is a trade rather than a mistake — the same
unrolling is why AC decides 257 of 277 value questions with no solver call, since
everything is ground. A row over a cap is reported UNKNOWN rather than FAIL, so a
kernel that is wrong *and* expensive to canonicalise used to go unjudged; the
random-point stage judges it wherever the field encoding applies, and rows 61 and
97 are what that is worth. Outside the fragment the axis is still open;
and unlike the TTIR bucket, this one needs no unusual operation at all. No
policy has been observed trying — that is the third claim under the adversary
heading in [limits.md](limits.md).

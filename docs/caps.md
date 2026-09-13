# Cost and caps

## Cost

Hand-measured, not checked by `verify.py`. The scripts are `tvj/measure/scale.py` and `tvj/checks/volta_attn.py`, and only their verdicts are claims. The numbers are sizes, not times, because a time depends on the machine and these do not.

Attention at D=16, BM=BN=16, three formulations compared pairwise:

| L | key blocks | outputs | term ops |
|---:|---:|---:|---:|
| 32 | 2 | 512 | 3.2 M |
| 64 | 4 | 1024 | 26.4 M |
| 128 | 8 | 2048 | 206.8 M |
| 256 | 16 | 4096 | 349 M–509 M |

The cost of deciding is per shape, and it depends on how different the two kernels' shapes are, not on their size. Lane by lane at L=128, the bridge peaks at 0.56 GB for ref vs safe, 0.77 GB for safe vs flash, and 9.19 GB for ref vs flash. That is twelve times more for the same problem, because the naive reference does not subtract the max, so its exponential polynomial cannot share the `−m` atom and the cross products expand.

With one representative per shape, the same ref vs flash pair at L=128, which exceeds a 4 GB cap lane by lane, is decided in 0.14 GB and 0.31 s, and every lane gets the verdict the lane-by-lane run gives. At L=64 it goes from 1.04 GB to 0.03 GB, and across the pairs both paths can run, Volta time drops 154× (`tvj/measure/lanes.py`). What is left is the cost per shape, which depends on how many distinct denominators one output sums, not on size. See the caps section below.

At L=256 the well-shaped pairs stay under 4.5 GB in Volta, while our Python side reaches 7.6 GB. That figure is from before the bridge's binary wire format. Passing one node to Volta used to cost a Python dict and about eighty bytes of JSON, 325 bytes against the nine it costs now. What is left on this side is the term DAG itself, measured at 390 bytes per interned term. This is where the 8 M term budget comes from, and it is the next thing to shrink.

## Rows not decided, and who can steer into them

The Limits table in the [README](../README.md#limits) sorts every row the judge does not decide by cause. What matters is who controls whether a kernel ends up there. A torch op we do not model is fixed by the task, so no policy can aim at it. A TTIR construct we do not model can be aimed at, and so can our own cost limits. Counted that way, a generator could aim at 24 rows of KernelBook and 19 of the LLM corpus, 6.0 % and 12.2 %.

These are two different kinds of target. The TTIR bucket is a list of named constructs: 17 rows where an integer is derived from a real value or read from memory, 3 of transposed convolution, 1 of `scf.while`. It shrinks as they are implemented. The caps bucket is a region, covered below.

### The torch tail

The largest steerable bucket in the LLM corpus is the torch tail, and the fix is not ours to make. These are wrappers that finish the computation in PyTorch after the kernels. Requiring a single fused Triton kernel removes the bucket entirely. That is not an extra cost, because a torch tail also costs a launch and a materialized intermediate. The constraint that makes a kernel analyzable is the same one that makes it fast.

### The caps

Our caps can be steered into, now along one axis instead of two. Before the random-point stage, the caps bucket held 20 rows, and 13 of them failed inside Volta, on its address space or its term-operation budget. Two things can put a pair there.

The first is width: many lanes of the same shape, each canonicalized separately. With the reference fixed, one correct attention kernel was decided in 0.56 GB and another in 9.19 GB, and the expensive one was the faster kernel. `tvj/measure/steerable.py` demonstrates it: three formulations equal over the reals, a cap between the cheapest and the most expensive, and the judge decides two and returns UNDECIDED on the third. One representative per shape closes this. The 9.19 GB pair takes 0.14 GB.

The second is division, and it remains. The 13 corpus rows are all this kind. Row 97 is 256 lanes of 2 shapes, and each representative alone exceeds the budget. Row 61 is 679 nodes per output, and canonicalizing one of them takes more than 3 GB. Volta canonicalizes to a rational N/D and adds two fractions with different denominators by multiplying the denominators (`canon/ops.rs`, `rat_add_v`). A multi-head attention output is a sum of one fraction per head, each over that head's own softmax denominator. So the common denominator has L^H terms, and the equality check N1·D2 = N2·D1 has about 10^6 monomials per output on row 61 and 10^11 on row 97, each carrying an exponent polynomial. These are counted without building them (`python3 -m tvj.measure.nf_rat kb 61`). The multiplicative depth of every one of these terms is 1.

Volta's paper argues the blowup away by depth. Canonicalization "may cause exponential blowup", but "since machine learning workloads do not typically have computations with high multiplicative depth, this blowup does not happen in practice". The argument is correct, but it does not cover this case. The growth here is exponential in the number of heads whose fractions one output sums, and on these two corpora it happens in 13 of 556 rows, in the kind of kernel the paper is about. An earlier version of this section blamed multiplicative depth. It was measured at 1.

### Random points

Evaluation at random points does not build the normal form. On the same 4 GB machine it decided 6 of those 13 in the first experiment: rows 86, 310, 318 and 328 PASS, and rows 61 and 97 FAIL with a numeric witness the GPU reproduces. Those two are the two of the 13 that fail the corpus' own tolerance test.

After republishing both corpora, with Volta limited to a 60 s wall clock and Z3 to a size and time budget so the later stages get a turn, the caps bucket gave up 14 rows: eight KernelBook FAILs the GPU reproduces (61, 97, 116, 137, 194, 196, 233, 306), five PASSes, and LLM row 127 ([findings.md](findings.md)). Nothing blocked in Volta is left in it.

Rows the stage reaches but does not decide are UNKNOWN, each for a known reason. LLM rows 11 and 150 are really different functions, since the kernel omits softplus's threshold branch, but the numeric witness search does not reach where they differ. LLM rows 51 and 92 separate at the witness, but the GPU does not reproduce it. That is a gap in our model, and it is reported as one.

### The bucket held defects

Before the random-point stage, 6 of the 15 KernelBook rows that hit one of our caps failed the corpus' own tolerance test, against 23 of the 300 decided rows. That is 5.2 times the rate, Fisher p = 0.001. Raising the caps (`TVJ_ROW_TIMEOUT=1800`, `VOLTA_MEM_GB=24`, `TVJ_VOLTA_BUDGET=4e9`) decided four of the six, and every one was a FAIL the GPU reproduced at the witness point. So the caps were holding defects, not just rows nobody had reached.

That enrichment is gone now: 0 of the 5 rows left in the bucket fail tolerance. It is gone because all six are FAILs at the default caps (61, 97, 137, 194, 196, 233). The bucket now holds three rows stopped by the 150 s row alarm (176, 363, 372) and two over the term budget (235, 344). None is blocked inside Volta.

The two rows that raising the caps did not decide were the two blocked inside Volta rather than by a cap of ours, and 24 GB was not enough for either. One is row 61, where reading the generated wrapper settles it without the judge: `Attention.forward(self, k, q)` takes k first, and the wrapper passes the second input to `w_k` and the first to `w_q`. Both are `(4, 4, 1, 4)`, so `assert_size_stride` passes. The GPU disagrees by 0.26, deterministically. The memory check does not catch it. Evaluation at random points does, in 9 ms, and the row is a FAIL. The random-point stage, with budgets on Volta and Z3 so that it is reached, decided the whole Volta-bound part of this bucket. Raising the row alarm and the term budget is what is left, and the rows it would decide are named above.

### Why a cap is always reachable

A term graph grows with the work a kernel does, not with the size of the program. A tiled matmul is Θ(M·N·K) nodes, because every output element is a sum of K products. So making a kernel bigger is always possible. Raising a cap is still worth doing, and the four rows above are what it buys. But it does not change what the threshold depends on, so this way into the bucket stays open after every raise.

Much of this page follows from that. Shapes are fixed because the grid is enumerated. Integers are concrete because that makes the memory check a dictionary lookup. A branch on a loaded value is refused because there is nothing symbolic to split. This is a trade-off, not a mistake: the same unrolling is why AC decides 257 of 277 value questions with no solver call, since everything is ground.

A row over a cap is UNKNOWN, not FAIL, so a kernel that is wrong and expensive to canonicalize used to go unjudged. The random-point stage now judges it wherever the field encoding applies, and rows 61 and 97 show what that is worth. Outside the encoding this is still open, and unlike the TTIR bucket it needs no unusual operation. No policy has been seen trying. That is the third claim under "No adversary yet" in [limits.md](limits.md).

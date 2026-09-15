# Cost and caps

## Cost

Measured with `tvj/measure/scale.py` and `tvj/checks/volta_attn.py`. The numbers are sizes, not times, since times depend on the machine.

Attention at D=16, BM=BN=16, three formulations compared pairwise:

| L | key blocks | outputs | term ops |
|---:|---:|---:|---:|
| 32 | 2 | 512 | 3.2 M |
| 64 | 4 | 1024 | 26.4 M |
| 128 | 8 | 2048 | 206.8 M |
| 256 | 16 | 4096 | 349 M–509 M |

Deciding a pair lane by lane, the cost depends on how different the two kernels are, not on their size. At L=128 the bridge peaks at 0.56 GB for ref vs safe, 0.77 GB for safe vs flash, and 9.19 GB for ref vs flash. That is twelve times more for the same problem, because the naive reference does not subtract the max, so its exponential polynomial cannot share the `−m` atom and the cross products expand.

With one representative per shape, the same ref vs flash pair at L=128, which exceeds a 4 GB cap lane by lane, is decided in 0.14 GB and 0.31 s, and every lane gets the verdict the lane-by-lane run gives. At L=64 it goes from 1.04 GB to 0.03 GB, and across the pairs both paths can run, Volta time drops 154× (`tvj/decide/lanes.py`).

At L=256 the well-shaped pairs stay under 4.5 GB in Volta. On the Python side the cost is the term DAG, about 390 bytes per interned term, which is where the 8 M term budget comes from.

## Rows not decided, and who can steer into them

The Limits table in the [README](../README.md#limits) sorts every row the judge does not decide by cause. What matters is who controls whether a kernel ends up there. A torch op we do not model is fixed by the task, so no policy can aim at it. A TTIR construct we do not model can be aimed at, and so can our cost limits. Counted that way, a generator could aim at 13 rows of KernelBook and 2 of the LLM dataset, 3.2 % and 1.3 %.

The TTIR bucket is a list of named constructs: 8 rows where an integer is derived from a real value or read from memory, and 1 of `scf.while`. The caps bucket is different, since any kernel can be made expensive.

The torch tail, wrappers that finish the computation in PyTorch after the kernels, was the largest steerable bucket in the LLM dataset at 16 rows. The judge now replays those ops, including arithmetic on a `.item()` value, and one KernelBook row is left in it.

## What makes Volta expensive

Two things make a pair expensive for Volta.

The first is width: many lanes of the same shape, each canonicalized separately. Lane by lane, one correct attention kernel takes 0.56 GB against the reference and another takes 9.19 GB, and the expensive one is the faster kernel. `tvj/measure/steerable.py` measures it: three formulations equal over the reals, with an order of magnitude between the cheapest and the most expensive to decide. One representative per shape closes this, and the 9.19 GB pair takes 0.14 GB.

The second is division. Volta canonicalizes to a rational N/D and adds two fractions with different denominators by multiplying the denominators (`canon/ops.rs`, `rat_add_v`). An attention output that sums several softmax rows, each over its own denominator, therefore has a common denominator with one factor per row. In KernelBook row 97 each output sums the 4 heads, and in row 61 it sums 4 softmax rows over different slices of one input. With L=4 the common denominator has 4^4 = 256 terms, and the equality check N1·D2 = N2·D1 has about 10^6 monomials per output on row 61 and 10^11 on row 97, each carrying an exponent polynomial. These are counted without building them (`python3 -m tvj.measure.nf_rat kb 61`). The multiplicative depth of every one of these terms is 1. On row 61 Volta exceeds its 4 GB cap, and on row 97 it leaves both of the row's 2 shapes undecided.

Volta's paper argues the blowup away by depth: canonicalization "may cause exponential blowup", but "since machine learning workloads do not typically have computations with high multiplicative depth, this blowup does not happen in practice". The argument is correct, but it does not cover this case. The growth here is exponential in the number of softmax rows one output sums, at depth 1.

## Random points

Evaluation at random points does not build the normal form. On rows 61 and 97 it shows that the two sides differ, and both are FAILs with a numeric witness the GPU reproduces. No row is left in the caps bucket because of Volta.

Rows the random-point stage reaches but does not decide are UNKNOWN. LLM rows 11 and 150 are different functions, since the kernel omits softplus's threshold branch, but the numeric witness search does not reach where they differ. LLM rows 51 and 92 separate at the witness, but the GPU does not reproduce it, so the model is assumed wrong and the verdict is UNKNOWN.

What is left in the caps bucket is five KernelBook rows: three stopped by the 150 s row alarm (176, 363, 372) and two over the term budget (235, 344).

## Why a cap is always reachable

A term graph grows with the work a kernel does, not with the size of the program. A tiled matmul is Θ(M·N·K) nodes, because every output element is a sum of K products. So a kernel can always be made bigger, and raising a cap does not change what the threshold depends on.

Several limits come from the same choice. Shapes are fixed because the grid is enumerated. Integers are concrete because that makes the memory check a dictionary lookup. A branch on a loaded value is refused because there is nothing symbolic to split. The same unrolling is why AC decides 286 of 308 value questions with no solver call, since everything is ground.

A row over a cap is UNKNOWN, not FAIL, so a wrong kernel that is expensive to decide goes unjudged. Evaluation at random points covers this wherever the field encoding applies. Outside it, the way in stays open, and unlike the TTIR bucket it needs no unusual operation.

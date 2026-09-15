# How it works

The judge builds two term graphs, one for the PyTorch module and one for the Triton kernel, and compares them.

## Reference side

`STensor` is a numpy object array of terms with a torch interface. It stands in for the module's inputs and parameters, and the module's own, unmodified `forward` is called on it. `__torch_function__` picks up `torch.matmul`, `F.softmax`, `F.layer_norm` and the rest, so the result is one term per output element.

## Kernel side

`JITFunction.run` is hooked, so the real GPU launches of the generated code are recorded: kernel, grid, arguments and constexprs. The TTIR of each launch is then executed symbolically over the whole grid. Tensors are matched to roles by storage base plus element offset, not by object identity, because Inductor hands out `reinterpret_tensor` views and non-contiguous output strides. The kernel only has to be callable as `launch(*inputs) -> output`.

## Floats as reals

Floating point is modeled as exact real numbers. Reassociation is then not a difference, so split-K, flash attention and tree reductions all come out equal. The two things real numbers cannot see, representation error and precision contracts, need their own checks.

## Determinism first

Before any other check, the kernel runs twice on the same inputs and the two outputs must be bit-identical. If they are not, the other checks mean nothing. If the reference is also nondeterministic, the task itself is. Both cases get the verdict `NONDETERMINISTIC`, not FAIL, because neither says anything about correctness.

A kernel that draws random numbers is still judged. Its k-th draw becomes a named input buffer, and the reference's k-th draw is bound to the same name, so the question becomes whether the two compute the same thing given the same draw. That binding is an assumption and is recorded with the verdict (see `PASS-ASSUMING` below). Randomness that never shows up as a tensor of its own cannot be bound and is refused, e.g. `native_dropout` returns its mask, not its draw.

## Five checks

Each check is a separate condition, and a FAIL says which one failed.

| check | question | how |
|---|---|---|
| value | same real number? | AC normal form, [Volta](https://github.com/willtunnels/volta)'s decision procedure, Z3 case splits, evaluation at random points |
| memory | does it read what no launch wrote, skip an output element, or race with itself? | symbolic execution of the whole grid |
| precision | is it less precise than the reference? | a lattice with a direction, since `ieee → tf32` is equal over the reals but not a refinement |
| precondition | does the real-number result still hold in float32? | interval analysis, plus three rules for softmax-like terms |
| accuracy | same value, but arranged so float32 loses more digits? | evaluate both terms in float32 and in float64, and compare the errors |

The accuracy check exists for cases like `E[X²]−E[X]²` against a stable variance. The two are equal over the reals, so the value check correctly passes them, but the first form cancels catastrophically. The check evaluates both terms at inputs that stress cancellation, once rounding every step to float32 and once in float64, and compares the two errors. It rejects the unstable form at about 2.8×10⁵ times the reference's error, in a regime where the reference is still accurate, and passes legitimate reassociation (`tvj/checks/accuracy_test.py`).

## Deciding value

Value is decided in stages, cheapest first:

| stage | KernelBook value decisions |
|---|---:|
| AC normal form | 257 |
| Volta | 14 |
| Volta, then Z3 | 1 |
| evaluation at random points | 5 |

Counts from `results/report.txt`.

Terms are hash-consed and normalized for associativity and commutativity, so most pairs come out identical and no solver runs.

What is left goes to Volta's exponential-polynomial procedure, one representative per shape instead of one call per output element. A tile kernel's outputs fall into a few shapes over different leaves, e.g. 1024 matmul lanes are one shape, and so are 512 attention lanes. Volta treats a leaf as an opaque variable, so pairs with the same joint shape are the same question up to renaming (`tvj/decide/lanes.py`).

Z3 case splits handle piecewise terms, which [Volta's paper](https://arxiv.org/abs/2511.12638) says "could be handled by case splits" but does not do. Z3 only runs on shapes that random real points cannot already separate, since a pair that separates by a clear margin is not equal.

What Volta cannot canonicalize within its caps is decided by evaluating both sides at random points over a finite field (`tvj/decide/pit.py`). The encoding follows [Mirage](https://arxiv.org/abs/2405.05751): values live in Z_p and `exp(x) = ω^x`, where ω has order q and q divides p−1, so exponents are taken mod q and `exp(a)·exp(b) = exp(a+b)` holds. An exp nested inside another exp's exponent becomes an opaque atom, as max and min already are. This never builds the normal form, so its cost follows the size of the DAG, not of the polynomial. Its "equal" is probabilistic, with error at most (d/2⁶¹)³ per pair, and the verdict records it as `via: pit` with the seed. The seed is drawn fresh for every judgment.

## Delegation

When both sides hand the same operation to the same library call with the same arguments, it becomes one uninterpreted symbol, and hash-consing decides it for free. This is sound only if the symbol's name encodes everything the result depends on, so every builder fails closed: a parameter it does not encode raises an error instead of producing a symbol. If the two sides do not match, the symbols are expanded and the pair is decided the slow way, because two different uninterpreted symbols are not a counterexample.

## When a FAIL is reported

Two of the five checks are confirmed on the GPU before a FAIL is reported.

Value is checked at its witness point, the concrete input where the two terms take different values. If the GPU does not show the difference there, the verdict is UNKNOWN, not FAIL. The difference is measured relative to the reference's magnitude. An absolute threshold hides a 190 % error when the output is ~1e-4 (row 308 in [findings.md](findings.md)), and at an output of ~1e8 it is below float32's own spacing.

Accuracy is checked at the input regime that triggered it. Cancellation is silent at the benchmark's inputs and visible at the shifted inputs that trigger it, so if the GPU is silent there, the model is wrong, not the kernel (`tvj/checks/acc_gate.py`).

The other three checks are not confirmed this way, because the GPU cannot show them: a stale buffer holds the right answer, tf32 has no effect before Ampere, and a narrower validity radius only shows at extreme inputs. Requiring the GPU to reproduce them would throw away exactly the defects a test cannot reach.

## PASS-ASSUMING

`PASS-ASSUMING` is a PASS under a stated assumption the judge cannot check. Both kinds are properties of the input data, not of the kernel:

- `index-distinct`: a scatter to a data-dependent address is well defined exactly when the indices are pairwise distinct. torch is in the same position, since `scatter_` is documented as nondeterministic when indices collide.
- `rng-correspondence`: when both sides draw random numbers, their k-th draws are matched by order and shape. Nothing outside the two programs can confirm those are the same draw.

## The reference side can be wrong

A wrong reference makes a correct kernel FAIL, which is visible. It can also make a wrong kernel PASS, which is not, and nothing later in the pipeline catches that. A keyword argument silently dropped by `**kwargs`, a reduction over the wrong axis, or a missing `__eq__` that turns `mask == 0` into Python `False` all produce a wrong reference without raising an error.

So the reference side is checked in two ways. `tvj/checks/spec_sigcheck.py` compares every handler with torch's own signature: no argument swallowed, misplaced, or declared and never read. `tvj/checks/spec_agree.py` runs 113 cases through both torch and the reference side and compares the numbers. Handlers whose edge-case behavior was taken from torch's C++ rather than from its documentation are marked, and a FAIL that depends on one says so.

`tvj/core/semantics.py` lists the 25 decisions the interpreter makes where Triton does not specify the behavior, e.g. what a masked load reads, whether i32 index arithmetic wraps, and whether a reduction is a tree or a fold. Each has its basis and evidence, and five were measured on hardware.

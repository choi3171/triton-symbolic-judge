# How it works

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

## First, is it a function of its inputs at all?

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

## Five checks, not one

Each is a separate claim the kernel has to satisfy; a verdict names which one
failed.

| check | question | instrument |
|---|---|---|
| **value** | the same real number? | AC normal form → [Volta](https://github.com/willtunnels/volta)'s decision procedure → Z3 case splitting for piecewise terms |
| **memory** | does it read what nothing wrote, skip an output, or race with itself? | symbolic execution of the whole grid |
| **precision** | is it not *less* precise than the reference? | a lattice with a direction, because `ieee → tf32` is real-equal but not a refinement |
| **precondition** | does the real-number proof still mean anything in float32? | interval analysis with three relational rules |
| **accuracy** | the same expression, arranged so float32 loses digits? | evaluate both terms in float32 and in float64, compare the *errors* |

Value is decided in four stages, cheapest first. Terms are hash-consed and
normalised for associativity and commutativity, so most pairs come out
*identical* and no solver runs at all: **257 of 277** KernelBook value decisions
finish there; Volta decides 14, Z3 one, and evaluation at random points 5. What is
left after AC goes to Volta's exponential-polynomial procedure —
**one representative per shape, not one per output element**: a tile kernel's
outputs are a handful of shapes over different leaves (1024 matmul lanes are one
shape, and so are 2048 attention lanes), and Volta treats a leaf as an opaque
variable, so pairs with the same joint shape are one question up to renaming
(`tvj/measure/lanes.py`). Z3 case splitting handles piecewise terms — the step
[Volta's paper](https://arxiv.org/abs/2511.12638) says "could be handled by case
splits" and declines to take — on the shapes that random real points cannot
already tell apart, since a pair that separates by a clear margin is not equal
and no sound prover will say it is. And what Volta cannot canonicalise within its caps
is decided by **evaluating both sides at random points over a finite field**
(`tvj/measure/pit.py`, after [Mirage](https://arxiv.org/abs/2405.05751)'s
encoding: `exp(x) = ω^x` with exponents in a field of order dividing the base
field's, so `exp(a)·exp(b) = exp(a+b)` holds because the field says so; an exp
nested inside another's exponent becomes an opaque atom, the trade already made
for max and min). That never builds the normal form, so its cost is the DAG's size rather than the
polynomial's; its "equal" is probabilistic, with error at most (d/2⁶¹)³ per pair,
and is recorded as such — `via: pit`, with the seed. The seed is drawn fresh for
every judgement.

AC does that much of the work because of **delegation**: when both sides hand the
same operation to the same library call with the same arguments, it becomes one
uninterpreted symbol and hash-consing decides it for free. That is sound only if
the symbol's name encodes everything the result depends on, so every builder
fails closed — an unencoded parameter raises rather than producing a symbol. If
the two sides disagree, the symbols are expanded and the pair is decided the slow
way, because two different uninterpreted symbols are not a counterexample.

## When a FAIL is believed

Two of the five are checked against hardware before they are reported.

**Value** is checked at its witness point — the concrete input where the two
terms take different values. If the GPU does not reproduce the disagreement
there, the verdict is UNKNOWN, not FAIL. Every false positive this project has
produced was caught by that rule. "Reproduces" is measured *relative to the
reference's own magnitude*, for the reason row 308 is in [the findings](findings.md) at
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

## The reference is the weak point

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

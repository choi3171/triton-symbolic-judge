# Pipeline

One sentence: **lower the reference PyTorch module and the generated Triton
kernel into the same term algebra, then compare them under five obligations.**
Value over the reals, memory by symbolically executing the whole grid, precision
on a lattice, float-validity by intervals, accuracy by running f32 and f64 side
by side. Shapes are covered as far as the kernel's own contract demands.

The judgement itself lives entirely in `tvj/judge/judge.py`. The corpus runners
(`tvj/judge/kernelbook_run.py`, `tvj/judge/traces_run.py`) are adapters whose
only job is to turn one row into a `Candidate` — a reference module, a callable
that runs the kernel, and the inputs. After that both corpora take the same path
through the same code.

```
     reference PyTorch module                generated Triton kernel
              │                                      │
   symbolic_module()                          capture()  ← hooks JITFunction.run
   parameters/buffers → symbols                records the real GPU launches
              │                                      │
   forward(STensor…)                          Launch: signature, constexprs,
   intercepted by __torch_function__                  grid, roles
              │                                      │
              │                               to_ttir() → ttir.parse()
              │                                      │
              │                               sexec.Interp: symbolic execution
              │                                      over the WHOLE grid
              ▼                                      ▼
        term DAG (terms.py)  ←── same normal form ──→  term DAG (terms.py)
                              │
        ┌─────────┬───────────┼────────────┬────────────┐
      value     memory     precision   precondition   accuracy
    AC normal  unwritten    lattice      intervals     f32 vs f64
    form       buffers      opt ≥ ref    radius        at shifted
    → Volta    races                     comparison    regimes
    → Z3 split coverage
    → numeric
      witness
                              │
   Hardware gate: if the GPU does not reproduce the disagreement at the witness
   point, the verdict is UNKNOWN rather than FAIL.  Accuracy is gated the same
   way but at the REGIME that fired.  The other three are not gated — hardware is
   structurally silent for them, so demanding reproduction would discard exactly
   the defects a test cannot reach.
```

Before any of the five: the kernel is run twice at the same inputs and the two
answers must agree. If they do not, the verdict is `NONDETERMINISTIC` and nothing
below would mean anything anyway.

## 1. The spec side — what is taken out of the PyTorch module, and how

The point is that **not one line of the module is edited**. Two hooks do it.

**(a) Parameters become symbols** — `spec.symbolic_module(model)`

```python
for mname, m in model.named_modules():
    for pn, p in m._parameters.items():
        m._parameters[pn] = STensor.input("p_" + mname + "." + pn, p.shape)
    for bn, b in m._buffers.items():
        m._buffers[bn] = STensor.input("b_" + ...)
```

The parameters and buffers of an `nn.Module` are swapped out directly in the
`_parameters` / `_buffers` dictionaries. `STensor.input(name, shape)` is a numpy
object array of `[T.sym(name, 0), T.sym(name, 1), …]` folded to the shape — so
`linear1.weight[3]` becomes the single symbol `p_linear1.weight[3]`. The role
name *is* the symbol name, which is how the kernel side's buffers are matched to
it later.

**(b) Operators and torch functions are intercepted**

- `STensor.__add__`, `__mul__`, `__matmul__`, … → numpy broadcasting plus
  `Term.__add__`
- `torch.matmul(x, …)`, `F.softmax(…)` → `STensor.__torch_function__` dispatches
  through the `_TORCH` table

So calling `model.forward(STensor(...))` makes the reference code emit a term DAG
**unmodified**. Every element of the output tensor is one term.

An operator not in `_TORCH` raises `NotImplementedError` → verdict
`SPEC-UNSUPPORTED`. A kwarg that could change the value and is not understood
raises rather than being dropped silently — the rule dates from swallowing
`F.linear(..., bias=)` and producing six false FAILs.

**Two spec-side decisions**, recorded in `tvj/front/spec.py`'s own docstring:

- `spec.softmax-form`: softmax is defined max-subtracted, `exp(x - max x) / sum`.
  Equal over the reals to `exp(x)/sum`; it is what torch computes; its
  float-validity radius is unbounded, so the precondition obligation compares
  kernels against the best known form; and matched forms canonicalise ~10×
  more cheaply in Volta.
- `spec.reduce-order`: reductions are n-ary AC sums, and no order is implied.

What the two sides make of a *constant* is settled in `tvj/core/semantics.py`
instead: `literal.working-precision` reads float literals at fp32 working
precision, so `1e-5` and the TTIR's `9.99999974e-06` are one constant, and its
companion `const.folding-precision` says the same of constants folded together
inside a term.

## 2. The kernel side — what is taken out of the generated code

The only thing asked of a generator is **`launch(*inputs) -> output`**. Not the
format, not the kernel's name, not the parameter order.

**(a) The real launches are intercepted** — `capture.capture()`

```python
_calls, _orig = [], JITFunction.run
def _rec(self, *args, grid, warmup=False, **kwargs):
    _calls.append((self, grid, args, dict(kwargs)))
    return _orig(self, *args, grid=grid, warmup=warmup, **kwargs)
```

`JITFunction.run` is swapped out and the code is **actually run on the GPU**, so
one call produces both the output a tolerance test needs and the information the
judgement needs. `extern_kernels.mm/addmm/bmm/baddbmm/convolution` — the path
Inductor sends matmuls down to cuBLAS — is wrapped the same way.

**(b) Tensors are mapped to roles by storage, not by object**

```python
base_of(t)      = t.untyped_storage().data_ptr()     # the allocation, not the view
elem_offset(t)  = (t.data_ptr() - base_of(t)) // t.element_size()
physical_offsets(t) = [off + Σ idx·stride]           # non-contiguous strides
```

Inductor hands out views via `reinterpret_tensor`, and an output's strides may be
non-contiguous. So a role (`in0`, `p_linear1.weight`, `out`) is decided by the
**storage base pointer** rather than by `id(tensor)`, and the logical-index →
physical-offset mapping is computed separately.

**(c) Lowered to TTIR and executed symbolically**

`Launch` decomposes each launch into signature, constexprs, grid, buffer sizes
and arguments → `to_ttir()` extracts the TTIR text → `ttir.parse()` parses it →
`sexec.Interp` walks the **whole grid**. Several launches run in order against
**one shared memory** (`X.Grid`), so a pipeline (kernel → extern mm → kernel)
stays connected.

Integers are concrete — fixed shapes mean every address, mask and loop bound is a
number. Out-of-bounds reads, write conflicts and dependence on an unwritten
buffer fall out of that **for free**.

## 3. Where the two meet

Both sides produce hash-consed terms from `tvj/core/terms.py`. The unit of
comparison is one output element:

```
spec:    spec.flat()[i]
kernel:  grid.store[("out", physical_offsets(out)[i])]
```

## 4. The five obligations

| obligation | what it asks | how |
|---|---|---|
| **value** | the same expression over the reals? | AC normal form decides most pairs outright → Volta `check_equivalent` → Z3 case splitting for piecewise terms → a **numeric witness** is required before a FAIL (without one, UNKNOWN) |
| **memory** | does it read what nothing wrote, skip an output, or race with itself? | the errors `sexec` records over the whole grid: unwritten-buffer reads, out-of-bounds, write conflicts, read-write races |
| **precision** | is the optimised side not *less* precise than the reference? | the lattice `exact > ieee > tf32x3 > tf32 > f16 > bf16`, tracked per term, compared against the reference's own output dtype |
| **precondition** | over what input range does the real-number proof still say something about float32? | interval analysis in `tvj/decide/ranges.py` plus three softmax relational rules; validity radii compared |
| **accuracy** | the same expression, arranged so float32 loses digits? | `tvj/decide/accuracy.py` evaluates both terms in f32 and in f64 at regimes that stress cancellation, and compares the two *errors* |

A verdict names the obligation that failed. `PASS-ASSUMING` is a pass carrying an
assumption the judge cannot discharge — `index-distinct` for a scatter to a
data-dependent address, `rng-correspondence` when both sides draw randomness.

**Shape coverage** is not one of the five; it is how the shapes to judge *at* are
chosen. `tvj/checks/suite.py` writes down the kernel's contract (its
preconditions plus its residue classes) and picks a minimum-cost covering set of
shapes greedily — which is how `bug_swizzle` is caught at (48, 48, 48) with no
hint.

## 5. The trust boundary (what is *not* verified)

- **The Triton backend**: only TTIR is read. The TTIR→PTX/GCN compilation is
  trusted. (Volta verifies that layer because it works on PTX, at the cost of
  being tied to NVIDIA.)
- **extern calls**: cuBLAS/cuDNN are modelled at the spec level and marked
  **trusted**. The number of trusted calls is reported alongside the verdict.
- **Our interpreter**: its soundness is argued informally. Volta proves its
  confluence in Agda; we do not.
- **The architecture**: every hardware measurement here was taken on `sm_75`, and
  the tf32, bf16 and `cp.async` paths were never re-measured elsewhere.
  `verify.py` prints the architecture it is running on and reports the `[sm_75]`
  claims apart from the count when it is not that one.

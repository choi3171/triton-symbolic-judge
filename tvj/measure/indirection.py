"""What does the judge actually DO with an indirect access?

Before writing "the method cannot handle X" in a limits section, run X through
it and read the answer.  Four kernels, each a shape that appears constantly in
production code and not once in either corpus:

  gather        `tl.load(src + idx)`      -- a loaded value used as a read address
  scatter       `tl.store(dst + idx, v)`  -- ... as a write address
  scatter_add   `tl.atomic_add(dst + idx, v)` -- order-free by construction
  early_exit    `if tl.load(flag) > 0: return` -- a branch on loaded data

The interesting question for each is not "does it pass" but which of three
answers comes back: a verdict, an honest refusal, or something wrong.

    python3 -m tvj.measure.indirection
"""
import re
import triton, triton.language as tl
import tvj.core.terms as T
from tvj.core import ttir as P, sexec as X
from tvj.checks.check import to_ttir


@triton.jit
def gather(src_ptr, idx_ptr, out_ptr, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < N
    j = tl.load(idx_ptr + i, mask=m, other=0)
    tl.store(out_ptr + i, tl.load(src_ptr + j, mask=m, other=0.0), mask=m)


@triton.jit
def scatter(src_ptr, idx_ptr, out_ptr, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < N
    j = tl.load(idx_ptr + i, mask=m, other=0)
    tl.store(out_ptr + j, tl.load(src_ptr + i, mask=m, other=0.0), mask=m)


@triton.jit
def scatter_add(src_ptr, idx_ptr, out_ptr, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < N
    j = tl.load(idx_ptr + i, mask=m, other=0)
    tl.atomic_add(out_ptr + j, tl.load(src_ptr + i, mask=m, other=0.0), mask=m)


@triton.jit
def early_exit(src_ptr, flag_ptr, out_ptr, N, BLOCK: tl.constexpr):
    if tl.load(flag_ptr) > 0.0:
        return
    i = tl.arange(0, BLOCK)
    m = i < N
    tl.store(out_ptr + i, tl.load(src_ptr + i, mask=m, other=0.0) * 2.0, mask=m)


SIG = {"src_ptr": "*fp32", "idx_ptr": "*i32", "out_ptr": "*fp32", "N": "i32", "BLOCK": "constexpr"}
SIG_F = {"src_ptr": "*fp32", "flag_ptr": "*fp32", "out_ptr": "*fp32", "N": "i32", "BLOCK": "constexpr"}
N = 8

CASES = [
    ("gather      load(src + idx)",   gather,      SIG,   "in production: embedding, MoE routing, paged attention"),
    ("scatter     store(dst + idx)",  scatter,     SIG,   "well defined only if the indices are distinct"),
    ("scatter_add atomic_add(dst+idx)", scatter_add, SIG, "order-free: the sum does not care who got there first"),
    ("early_exit  branch on loaded",  early_exit,  SIG_F, "a real speedup, and a real way to skip work"),
]

def judge_gather():
    """The end of the question: can a gather actually be DECIDED, or only
    represented?  Reference `x[idx]` against the kernel, term for term."""
    import numpy as np
    from tvj.front.spec import STensor
    from tvj.decide import volta_bridge as V
    T.reset()
    bufs = {"src_ptr": N, "idx_ptr": N, "out_ptr": N}
    f = P.parse(to_ttir(gather, SIG, {"BLOCK": N}))
    it = X.Interp(f, None, (1,), bufs)
    it.argvals = [X.Ptr("src_ptr", 0), X.Ptr("idx_ptr", 0), X.Ptr("out_ptr", 0), N]
    it.run_all()
    kern = [it.g.store[("out_ptr", i)] for i in range(N)]
    spec = STensor.input("src_ptr", (N,))[STensor.input("idx_ptr", (N,))].flat()
    ac = sum(1 for a, b in zip(spec, kern) if a is b)
    print(f"\nreference `x[idx]` vs the kernel, {N} outputs")
    print(f"   AC normal form:      {ac}/{N} identical")
    rest = [(a, b) for a, b in zip(spec, kern) if a is not b]
    if rest:
        res, _ = V.equivalent(rest, budget=200_000_000)
        print(f"   Volta on the rest:   {sum(1 for r in res if r is True)}/{len(rest)}")
    # ---- scatter-add, the same question -----------------------------------
    import torch as _t
    T.reset()
    bufs = {"src_ptr": N, "idx_ptr": N, "out_ptr": N}
    f = P.parse(to_ttir(scatter_add, SIG, {"BLOCK": N}))
    it = X.Interp(f, None, (1,), bufs)
    it.argvals = [X.Ptr("src_ptr", 0), X.Ptr("idx_ptr", 0), X.Ptr("out_ptr", 0), N]
    it.run_all()
    kern = [it.g.store[("out_ptr", j)] for j in range(N)]
    ref = _t.index_add(STensor.full((N,), 0.0), 0,
                       STensor.input("idx_ptr", (N,)), STensor.input("src_ptr", (N,))).flat()
    ac2 = sum(1 for a, b in zip(ref, kern) if a is b)
    print(f"\nreference `zeros.index_add_(0, idx, src)` vs the kernel, {N} outputs")
    print(f"   AC normal form:      {ac2}/{N} identical")

    # ---- scatter, which is decided only under an assumption -----------------
    T.reset()
    bufs = {"src_ptr": N, "idx_ptr": N, "out_ptr": N}
    f = P.parse(to_ttir(scatter, SIG, {"BLOCK": N}))
    it = X.Interp(f, None, (1,), bufs)
    it.argvals = [X.Ptr("src_ptr", 0), X.Ptr("idx_ptr", 0), X.Ptr("out_ptr", 0), N]
    it.run_all()
    kern = [it.g.store[("out_ptr", j)] for j in range(N)]
    ref = _t.scatter(STensor.input("out_ptr", (N,)), 0,
                     STensor.input("idx_ptr", (N,)), STensor.input("src_ptr", (N,))).flat()
    ac3 = sum(1 for a, b in zip(ref, kern) if a is b)
    print(f"\nreference `out.scatter_(0, idx, src)` vs the kernel, {N} outputs")
    print(f"   AC normal form:      {ac3}/{N} identical")
    for k, b_, w in it.g.assumptions:
        print(f"   assumption recorded: {k} on `{b_}`")
        print(f"      {w}")

    # and a kernel that is WRONG must not pass: shift the index by one
    T.reset()
    bad = [T.gather([T.sym("src_ptr", j) for j in range(N)],
                    T.add(T.sym("idx_ptr", i), T.const(1.0))) for i in range(N)]
    good = [T.gather([T.sym("src_ptr", j) for j in range(N)], T.sym("idx_ptr", i))
            for i in range(N)]
    print(f"   an off-by-one gather: {sum(1 for a, b in zip(good, bad) if a is b)}/{N} identical"
          f"   (0 is the right answer)")


if __name__ == "__main__":
    print(f"{'kernel':<34} {'what the judge says':<52} note")
    print("-" * 132)
    for label, fn, sig, note in CASES:
        T.reset()
        bufs = {"src_ptr": N, "idx_ptr": N, "out_ptr": N, "flag_ptr": 1}
        args = [X.Ptr(p, 0) for p in ("src_ptr", "idx_ptr" if "idx_ptr" in sig else "flag_ptr", "out_ptr")] + [N]
        try:
            f = P.parse(to_ttir(fn, sig, {"BLOCK": N}))
            it = X.Interp(f, None, (1,), bufs); it.argvals = args
            it.run_all()
            wrote = sorted(k for k in it.g.store if k[0] == "out_ptr")
            errs = [e[0] for e in it.g.errors]
            got = (f"ran: {len(wrote)}/{N} slots written"
                   + (f", errors {set(errs)}" if errs else "")
                   + (f", benign {it.g.benign}" if it.g.benign else ""))
            if wrote:
                got += f"\n{'':<34} out[{wrote[0][1]}] = {str(it.g.store[wrote[0]])[:44]}"
        except X.Unsupported as e:
            got = f"REFUSED  {str(e)[:44]}"
        except NotImplementedError as e:
            got = f"UNIMPLEMENTED  {str(e).split('|')[0][:40]}"
        except Exception as e:
            got = f"{type(e).__name__}: {str(e)[:44]}"
        print(f"{label:<34} {got:<52} {note}")
    judge_gather()

"""Delegation must be a shortcut, never a new way to be wrong.

Three properties, in order of how much damage getting them wrong would do:

  SOUND      two different operations never share a symbol.  A collision here
             would let a wrong kernel PASS, which is the only failure mode this
             project treats as unacceptable.
  COMPLETE   the two spellings the corpora actually produce -- the module's
             `F.linear(x, w, b)` and Inductor's `addmm(b, x, w_t)` -- land on the
             SAME symbol, or the shortcut buys nothing.
  INERT      below the size threshold nothing is delegated, so every verdict the
             judge reached on a small row means exactly what it meant before.
"""
import sys, time
import numpy as np
from tvj.core import terms as T
from tvj.decide import delegate as D
from tvj.front.spec import STensor
import torch, torch.nn.functional as F

def mk(name, shape):
    return STensor.input(name, shape)

def linear_spec(x, w, b):        # what the reference module runs
    return F.linear(x, w, b)

def linear_kernel(x, w, b):      # what Inductor's call() runs: addmm(b, x, w_t)
    return (x @ STensor(w.a.T.copy())) + b

CASES = []
def case(name):
    def deco(fn): CASES.append((name, fn)); return fn
    return deco

@case("COMPLETE  F.linear(x,w,b) == addmm(b,x,w_t), 64x1024x1024")
def _():
    T.reset()
    x, w, b = mk("in0", (64, 1024)), mk("p_w", (1024, 1024)), mk("p_b", (1024,))
    a, c = linear_spec(x, w, b), linear_kernel(x, w, b)
    return all(p is q for p, q in zip(a.flat(), c.flat())), "identical terms"

@case("COMPLETE  delegated at all, not silently expanded")
def _():
    T.reset()
    x, w = mk("in0", (64, 1024)), mk("p_w", (1024, 1024))
    t = (x @ w).flat()[0]
    return isinstance(t, T.Sym) and D.is_delegated(t.buf), f"top term is {type(t).__name__}"

@case("SOUND     a different operand gives a different symbol")
def _():
    T.reset()
    x, w = mk("in0", (64, 1024)), mk("p_w", (1024, 1024))
    w2 = STensor(w.a.copy()); w2.a[0, 0] = T.sym("p_w", 10 ** 7)
    return (x @ w).flat()[0] is not (x @ w2).flat()[0], "one changed element separates them"

@case("SOUND     a different shape gives a different symbol")
def _():
    T.reset()
    x = mk("in0", (64, 1024))
    a = (x @ mk("p_w", (1024, 1024))).flat()[0]
    b = (STensor(x.a.reshape(128, 512)) @ mk("p_w2", (512, 1024))).flat()[0]
    return a is not b, "different (M,N,K)"

@case("SOUND     transposing an operand changes the symbol")
def _():
    T.reset()
    w = mk("p_w", (1024, 1024)); x = mk("in0", (64, 1024))
    return (x @ w).flat()[0] is not (x @ STensor(w.a.T.copy())).flat()[0], "w vs w.T"

@case("INERT     below the threshold nothing is delegated")
def _():
    T.reset()
    x, w = mk("in0", (4, 4)), mk("p_w", (4, 4))
    t = (x @ w).flat()[0]
    return not (isinstance(t, T.Sym) and D.is_delegated(t.buf)), "small matmul stays dense"

@case("INERT     a small linear still agrees with the dense reference")
def _():
    T.reset()
    x, w, b = mk("in0", (4, 4)), mk("p_w", (4, 4)), mk("p_b", (4,))
    return all(p is q for p, q in zip(linear_spec(x, w, b).flat(), linear_kernel(x, w, b).flat())), \
        "same terms, densely"

def _has_delegated(t, seen=None):
    seen = seen if seen is not None else set()
    if t is None or t.uid in seen: return False
    seen.add(t.uid)
    if isinstance(t, T.Sym): return D.is_delegated(t.buf)
    return any(_has_delegated(a, seen) for a in getattr(t, "args", ()))

@case("COMPLETE  conv2d delegates and both sides share conv_nd")
def _():
    T.reset()
    x, w, b = mk("in0", (4, 64, 56, 56)), mk("p_w", (64, 64, 3, 3)), mk("p_b", (64,))
    from tvj.front import spec
    a = F.conv2d(x, w, b, 1, 1)                                  # the module's spelling
    c = spec.conv_nd(x, w, b, 1, 1, 1, 1, 2)                     # extern_kernels.convolution's
    t = a.flat()[0]
    return (_has_delegated(t) and all(p is q for p, q in zip(a.flat(), c.flat()))), \
        "same terms, delegated symbol inside"

@case("SOUND     stride and padding are in the key")
def _():
    T.reset()
    from tvj.front import spec
    x, w = mk("in0", (4, 64, 56, 56)), mk("p_w", (64, 64, 3, 3))
    a = spec.conv_nd(x, w, None, 1, 1, 1, 1, 2).flat()[0]
    b = spec.conv_nd(x, w, None, 1, 0, 1, 1, 2).flat()[0]
    c = spec.conv_nd(x, w, None, 2, 1, 1, 1, 2).flat()[0]
    return a is not b and a is not c and b is not c, "padding and stride separate them"

@case("COMPLETE  the conv bias stays OUTSIDE the symbol")
def _():
    # Inductor routinely emits `convolution(x, w, None)` and adds the bias in a
    # fused Triton kernel afterwards.  Folding the bias into the key would give
    # the two spellings different symbols for the same convolution, which is what
    # sent KernelBook row 30 to the fallback and then over Volta's budget.
    T.reset()
    from tvj.front import spec
    x, w, b = mk("in0", (4, 64, 56, 56)), mk("p_w", (64, 64, 3, 3)), mk("p_b", (64,))
    nb = spec.conv_nd(x, w, None, 1, 1, 1, 1, 2)
    wb = spec.conv_nd(x, w, b, 1, 1, 1, 1, 2)
    same_sym = nb.flat()[0] is not wb.flat()[0] and _has_delegated(wb.flat()[0])
    added = all(q is T.add(p, b.a[0]) for p, q in zip(nb.a[:, 0].reshape(-1), wb.a[:, 0].reshape(-1)))
    return same_sym and added, "conv symbol shared; bias is a separate Add"

@case("FALLBACK  two symbols that differ are cashed in, not called a defect")
def _():
    # (2x) @ w  and  x @ (2w) are the same product but different operands, so the
    # shortcut gives them different symbols.  Expanding must recover the equality.
    T.reset(); D.reset()
    x, w = mk("in0", (64, 128)), mk("p_w", (128, 64))
    a = ((x * 2.0) @ w).flat()[0]
    b = (x @ (w * 2.0)).flat()[0]
    if a is b: return False, "unexpectedly identical before expansion"
    if not (D.is_delegated(a.buf) and D.is_delegated(b.buf)): return False, "not delegated"
    (ea, eb), cost = D.expand([a, b])
    from tvj.decide import volta_bridge as V
    res, _ = V.equivalent([(ea, eb)], budget=200_000_000)
    return res[0] is True, f"expanded {cost:,} terms, Volta proves equal"

@case("FALLBACK  a real disagreement survives expansion")
def _():
    T.reset(); D.reset()
    x, w = mk("in0", (64, 128)), mk("p_w", (128, 64))
    w2 = STensor(w.a.copy()); w2.a[0, 0] = T.sym("p_w", 10 ** 7)
    a, b = (x @ w).flat()[0], (x @ w2).flat()[0]
    (ea, eb), _ = D.expand([a, b])
    from tvj.decide import numeric as NUM
    dom = {"in0": 64 * 128, "p_w": 10 ** 7 + 1}
    wit = NUM.witness([(ea, eb)], dom)
    return wit[0][0] is False, "witness found after expansion"

@case("FALLBACK  expansion is refused above the budget, not attempted")
def _():
    T.reset(); D.reset()
    x, w = mk("in0", (64, 1024)), mk("p_w", (1024, 1024))
    a = (x @ w).flat()[0]
    try:
        D.expand([a], budget=1000)
        return False, "did not refuse"
    except D.TooLarge as e:
        return True, str(e)[:52]

@case("FALLBACK  a term with no delegated symbol is returned unchanged")
def _():
    T.reset(); D.reset()
    x = mk("in0", (4, 4))
    t = (x * 2.0 + 1.0).flat()[0]
    (r,), cost = D.expand([t])
    return r is t and cost == 0, "identity"

@case("FAIL-CLOSED  transposed convolution is refused, not keyed")
def _():
    from tvj.core.sexec import Unsupported
    T.reset()
    try:
        D.conv(mk("in0", (1, 64, 56, 56)).a, mk("p_w", (64, 64, 3, 3)).a, None,
               1, 1, 1, 1, 2, (1, 64, 56, 56), transposed=True)
        return False, "did not refuse"
    except Unsupported as e:
        return True, str(e)[:40]

if __name__ == "__main__":
    bad = 0
    for name, fn in CASES:
        t0 = time.time()
        try: ok, note = fn()
        except Exception as e: ok, note = False, f"{type(e).__name__}: {str(e)[:50]}"
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<58} {time.time()-t0:5.2f}s  {note}")
    print(f"\n{len(CASES)-bad}/{len(CASES)} delegation properties hold")
    sys.exit(1 if bad else 0)

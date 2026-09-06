"""Delegated ops: compared by congruence, denoted only when forced.

Measured on KernelBook: of 161 heavy ops (linear / conv / matmul) in the
reference modules, **160 are delegated by the generated code to a library call**
-- `extern_kernels.addmm`, `extern_kernels.convolution` -- and exactly one is
reimplemented in Triton.  Both sides therefore compute the same thing by calling
the same cuBLAS kernel, and expanding either into a dense term DAG costs
O(M*N*K) node constructions to prove something structural.  That expansion is
what puts 19 KernelBook rows over the 150 s alarm; a 1024-wide MLP layer is
6.7e7 nodes on each side.

So: give the operation a single symbolic buffer whose NAME canonically encodes
the operation and its operands.  Hash-consing then decides equality for free --
two sides that delegate the same op to the same arguments produce the same
symbol.  This is congruence for an uninterpreted function, and it is sound only
if the name is complete.

  THE ONE UNSOUND PATH is a name that omits something the result depends on:
  two different computations would collide and a wrong kernel would PASS.  Every
  builder here is therefore FAIL CLOSED -- a parameter outside the encoded set
  raises Unsupported rather than producing a symbol.  Adding a parameter to one
  of these ops means adding it to its key in the same edit.

Below `EXPAND_BELOW` multiply-adds nothing is delegated: small operands are
cheap to denote, and a dense denotation is what lets the value obligation
produce a counterexample instead of just "the symbols differ".
"""
import hashlib
import numpy as np
from tvj.core import terms as T
from tvj.core.sexec import Unsupported

EXPAND_BELOW = 1 << 18                 # 262,144 multiply-adds

# How much the fallback may cash in.  Measured after the n-ary fix: expanding
# 1,048,576 terms and handing the result to Volta takes 13.7 s, so 4 M is about a
# minute -- affordable inside a 150 s row budget, and past it the honest answer
# is "two library calls we did not expand" rather than a timeout.
EXPAND_BUDGET = 4_000_000

def _flat(a):
    """Terms of an operand, in row-major order.  Accepts an STensor, a numpy
    object array, or a scalar."""
    arr = getattr(a, "a", a)
    if not isinstance(arr, np.ndarray): return [T.lift(arr)], ()
    return list(arr.reshape(-1)), tuple(arr.shape)

def _digest(op, operands, params):
    """A canonical, collision-resistant name for one delegated operation.

    Operand identity is the tuple of term uids in row-major order: terms are
    hash-consed, so two sides that built the same expression have the same uids
    within one session, and different expressions have different ones."""
    h = hashlib.blake2b(digest_size=16)
    h.update(op.encode())
    for a in operands:
        ts, shape = _flat(a)
        h.update(b"|" + repr(shape).encode() + b"|")
        h.update(np.asarray([t.uid for t in ts], dtype=np.int64).tobytes())
    h.update(b"|" + repr(params).encode())
    return f"{op}#{h.hexdigest()}"

# name -> what it stands for.  A delegated symbol is uninterpreted for as long as
# that is enough; when the two sides disagree it has to be cashed in, and this is
# what makes that possible.  Cleared by `reset()` alongside the term pool.
REGISTRY = {}

def reset():
    REGISTRY.clear()

def _syms(name, shape, dense=None, cost=0):
    n = int(np.prod(shape)) if shape else 1
    out = np.empty(n, dtype=object)
    out[:] = [T.sym(name, i) for i in range(n)]
    if dense is not None: REGISTRY[name] = {"dense": dense, "cost": cost, "shape": tuple(shape)}
    return out.reshape(shape)


class TooLarge(Exception):
    """Expanding this delegated op would cost more than the caller allowed."""


def expandable(bufs):
    """Which of these symbol names are delegated ops we can still cash in."""
    return [b for b in bufs if b in REGISTRY]


def expand(terms, budget=EXPAND_BUDGET):
    """Replace every delegated symbol in `terms` by the dense expression it
    stands for, and return the rewritten terms.

    This is the fallback the congruence shortcut needs: agreeing symbols settle a
    pair for free, and a pair they do NOT settle has to be decided the slow way
    rather than reported as a disagreement.  An uninterpreted symbol is not a
    counterexample -- two different symbols are just two things we did not look
    at."""
    names = set()
    def scan(t, seen):
        if t is None or t.uid in seen: return
        seen.add(t.uid)
        if isinstance(t, T.Sym) and t.buf in REGISTRY: names.add(t.buf)
        for a in getattr(t, "args", ()): scan(a, seen)
    seen = set()
    for t in terms: scan(t, seen)
    if not names: return terms, 0
    cost = sum(REGISTRY[n]["cost"] for n in names)
    if cost > budget:
        raise TooLarge(f"expanding {len(names)} delegated op(s) costs ~{cost:,} terms > {budget:,}")
    table = {n: np.asarray(REGISTRY[n]["dense"]()).reshape(-1) for n in names}
    memo = {}
    return [T.substitute(t, table, memo) for t in terms], cost


# --------------------------------------------------------------------------

def _dense_matmul(a, b):
    """Imported lazily: spec.py owns the dense builder and imports this module."""
    from tvj.front.spec import dense_matmul
    return dense_matmul(a, b)


def matmul_cost(M, N, K): return M * N * K

def matmul(A, B):
    """The bare (M,K) @ (K,N) product as one symbol, or None when it is small
    enough to denote densely.

    Deliberately bare: `alpha`, `beta` and the bias of an `addmm` are applied by
    the CALLER in the term algebra, on both sides.  Folding them into the key
    would be three more things to get wrong for no gain -- the reference writes
    `x @ w.T + b` and Inductor writes `addmm(b, x, w_t)`, and those agree only if
    both spell the bias the same way, which they do once it is outside."""
    a, ashape = _flat(A); b_, bshape = _flat(B)
    if len(ashape) != 2 or len(bshape) != 2: raise Unsupported("delegated matmul: not 2-D")
    M, K = ashape; K2, N = bshape
    if K != K2: raise Unsupported(f"delegated matmul: inner dims {K} vs {K2}")
    cost = matmul_cost(M, N, K)
    if cost < EXPAND_BELOW: return None
    return _syms(_digest("mm", [A, B], ("MNK", (M, N, K))), (M, N),
                 dense=lambda: _dense_matmul(getattr(A, "a", A), getattr(B, "a", B)),
                 cost=cost)

def conv_cost(out_numel, cin_per_group, kernel_numel): return out_numel * cin_per_group * kernel_numel

def conv(x, w, bias, stride, padding, dilation, groups, nd, out_shape,
         transposed=False, output_padding=0, dense=None):
    """Direct convolution as one symbol.  Fail closed on anything the key below
    does not encode -- transposed convolution has its own output-shape rule and
    is refused rather than silently keyed as a forward convolution."""
    if transposed: raise Unsupported("delegated conv: transposed")
    _, wshape = _flat(w)
    out_numel = int(np.prod(out_shape))
    kernel_numel = int(np.prod(wshape[2:])) if len(wshape) > 2 else 1
    cin_per_group = wshape[1] if len(wshape) > 1 else 1
    if conv_cost(out_numel, cin_per_group, kernel_numel) < EXPAND_BELOW: return None
    ops = [x, w] + ([bias] if bias is not None else [])
    name = _digest("conv", ops, ("nd", nd, "stride", tuple(np.ravel(stride).tolist()),
                                 "padding", tuple(np.ravel(padding).tolist()),
                                 "dilation", tuple(np.ravel(dilation).tolist()),
                                 "groups", int(groups), "bias", bias is not None,
                                 "output_padding", tuple(np.ravel(output_padding).tolist()),
                                 "out", tuple(out_shape)))
    return _syms(name, tuple(out_shape), dense=dense,
                 cost=conv_cost(out_numel, cin_per_group, kernel_numel))

def is_delegated(buf):
    """Is this symbol a delegated operation rather than a real buffer?  The
    memory obligation must not report one as 'a buffer no launch wrote'."""
    return isinstance(buf, str) and ("#" in buf) and buf.split("#", 1)[0] in ("mm", "conv")

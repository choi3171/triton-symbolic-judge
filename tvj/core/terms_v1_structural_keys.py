"""Hash-consed real-valued term algebra.

Following Volta (Driscoll et al., OOPSLA'26) we model tensor elements as REALS,
not IEEE-754 floats and not uninterpreted functions.  Rationale:
  - uninterpreted `+` makes reassociation (every tiling/split-K optimisation)
    look like a semantic change -> the checker rejects correct kernels;
  - IEEE-754 makes the same optimisations genuinely non-equivalent, so a sound
    checker over floats would also reject them.
Over the reals `+` and `*` are associative-commutative, so a canonical n-ary
normal form decides equality for the sum-of-products kernels we handle here.
"""
from dataclasses import dataclass
from typing import Tuple

_pool = {}

def _hc(obj):
    return _pool.setdefault(obj.key, obj)

class Term:
    __slots__ = ("key", "_s")
    def __hash__(self):  return hash(self.key)
    def __eq__(self, o): return self is o

@dataclass(eq=False)
class Sym(Term):
    """One element of an input tensor: A[7] means flat index 7 of buffer A."""
    __slots__ = ("key", "_s", "buf", "idx")
    def __init__(self, buf, idx):
        self.buf, self.idx = buf, idx
        self.key = ("sym", buf, idx)
    def __repr__(self): return f"{self.buf}[{self.idx}]"

@dataclass(eq=False)
class Const(Term):
    __slots__ = ("key", "_s", "v")
    def __init__(self, v):
        self.v = float(v); self.key = ("const", self.v)
    def __repr__(self): return repr(self.v)

@dataclass(eq=False)
class App(Term):
    """Uninterpreted unary/binary application (exp, max, div, ...)."""
    __slots__ = ("key", "_s", "fn", "args")
    def __init__(self, fn, args):
        self.fn, self.args = fn, tuple(args)
        self.key = ("app", fn, tuple(a.key for a in self.args))
    def __repr__(self): return f"{self.fn}({', '.join(map(repr,self.args))})"

@dataclass(eq=False)
class Add(Term):
    __slots__ = ("key", "_s", "args")
    def __init__(self, args):
        self.args = tuple(args)
        self.key = ("add", tuple(a.key for a in self.args))
    def __repr__(self): return "(" + " + ".join(map(repr, self.args)) + ")"

@dataclass(eq=False)
class Mul(Term):
    __slots__ = ("key", "_s", "args")
    def __init__(self, args):
        self.args = tuple(args)
        self.key = ("mul", tuple(a.key for a in self.args))
    def __repr__(self): return "(" + "*".join(map(repr, self.args)) + ")"

def const(v): return _hc(Const(v))
def sym(buf, idx): return _hc(Sym(buf, idx))
ZERO, ONE = const(0.0), const(1.0)

def _sortkey(t):
    s = getattr(t, "_s", None)
    if s is None:
        s = repr(t.key); t._s = s
    return s

def _split_coeff(t):
    """t -> (numeric coefficient, canonical non-numeric part)."""
    if isinstance(t, Const): return t.v, ONE
    if isinstance(t, Mul):
        c, rest = 1.0, []
        for a in t.args:
            if isinstance(a, Const): c *= a.v
            else: rest.append(a)
        if not rest: return c, ONE
        return c, (rest[0] if len(rest) == 1 else _hc(Mul(sorted(rest, key=_sortkey))))
    return 1.0, t

def add(*terms):
    flat = []
    for t in terms:
        if isinstance(t, Add): flat.extend(t.args)
        else: flat.append(t)
    # collect like terms: 3*x + 2*x -> 5*x   (decides reassociation + duplication)
    buckets, order = {}, []
    for t in flat:
        c, base = _split_coeff(t)
        if base.key not in buckets:
            buckets[base.key] = [0.0, base]; order.append(base.key)
        buckets[base.key][0] += c
    out = []
    for k in order:
        c, base = buckets[k]
        if c == 0.0: continue
        if base is ONE: out.append(const(c))
        elif c == 1.0: out.append(base)
        else: out.append(mul(const(c), base))
    if not out: return ZERO
    if len(out) == 1: return out[0]
    return _hc(Add(sorted(out, key=_sortkey)))

def mul(*terms):
    flat = []
    for t in terms:
        if isinstance(t, Mul): flat.extend(t.args)
        else: flat.append(t)
    c, rest = 1.0, []
    for t in flat:
        if isinstance(t, Const): c *= t.v
        else: rest.append(t)
    if c == 0.0: return ZERO
    if not rest: return const(c)
    if c != 1.0: rest.append(const(c))
    if len(rest) == 1: return rest[0]
    return _hc(Mul(sorted(rest, key=_sortkey)))

def sub(a, b):  return add(a, mul(const(-1.0), b))
def div(a, b):
    if isinstance(b, Const) and b.v != 0.0: return mul(a, const(1.0 / b.v))
    return _hc(App("div", (a, b)))
def app(fn, *args):
    # identity elements: max(-inf, x) = x, min(+inf, x) = x
    if fn == "max": args = tuple(a for a in args if not (isinstance(a, Const) and a.v == float("-inf"))) or (const(float("-inf")),)
    if fn == "min": args = tuple(a for a in args if not (isinstance(a, Const) and a.v == float("inf")))  or (const(float("inf")),)
    if fn in ("max", "min") and len(args) == 1: return args[0]
    return _hc(App(fn, args))

def size(t, seen=None):
    """DAG node count -- how big the evaluation tree actually is."""
    if seen is None: seen = set()
    if t.key in seen: return 0
    seen.add(t.key)
    n = 1
    for a in getattr(t, "args", ()): n += size(a, seen)
    return n


_nan_memo = {}
def has_nan(t):
    """Does the DAG contain a NaN constant?  (OOB reads poison with NaN.)"""
    k = t.key
    if k in _nan_memo: return _nan_memo[k]
    if isinstance(t, Const): r = t.v != t.v
    else: r = any(has_nan(a) for a in getattr(t, "args", ()))
    _nan_memo[k] = r
    return r

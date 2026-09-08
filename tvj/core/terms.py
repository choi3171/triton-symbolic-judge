"""Hash-consed real-valued term algebra: O(arity) keys, AC normal form.

Every interned term gets a sequential `uid`, and a compound term's key holds its
children's uids rather than their keys.  The first version of this module built
each key from the children's full keys, so hashing or comparing one cost O(size
of the DAG below it), and a 128-deep nested max inside every element of a
2048-element output made that quadratic.  (`git log -- tvj/core/` has it.)

Within one pool identity == structure, so ordering arguments by uid is a
canonical order, and `add`, `mul`, `max` and `min` can normalise for
associativity and commutativity.  That is what makes most of the judge's
comparisons free: the two sides come out as the same object and no solver runs.

Model: tensor elements are REALS (Volta's choice), which is what makes
reassociation -- split-K, flash attention, tree reductions -- cost nothing.
CONSTANTS are the one place that model is approximated: they are folded at fp32
working precision, so two terms whose constants differ by less than an fp32 ulp
are the same term.  A recorded decision rather than an accident, and one that can
only ever go quiet: `const.folding-precision` in tvj/core/semantics.py, next to
the literal-reading rule it follows from.
"""
_pool = {}
_uid = [0]

def _hc(obj):
    t = _pool.get(obj.key)
    if t is None:
        obj.uid = _uid[0]; _uid[0] += 1
        _pool[obj.key] = obj
        return obj
    return t

class Term:
    __slots__ = ("key", "uid")
    def __hash__(self):  return self.uid
    def __eq__(self, o): return self is o

class Sym(Term):
    __slots__ = ("buf", "idx")
    def __init__(self, buf, idx):
        self.buf, self.idx = buf, idx
        self.key = ("sym", buf, idx)
    def __repr__(self): return f"{self.buf}[{self.idx}]"

class Const(Term):
    __slots__ = ("v",)
    def __init__(self, v):
        self.v = float(v)
        # NaN is not equal to itself, so ("const", nan) never matches the pool and
        # every NaN used to become a fresh node -- which meant no two terms
        # containing one could ever be identified.  There is one NaN here (we do
        # not distinguish payloads), so give it one key.
        self.key = ("const", "nan" if self.v != self.v else self.v)
    def __repr__(self): return repr(self.v)

class App(Term):
    __slots__ = ("fn", "args")
    def __init__(self, fn, args):
        self.fn, self.args = fn, tuple(args)
        self.key = ("app", fn, tuple(a.uid for a in self.args))
    def __repr__(self): return f"{self.fn}({', '.join(map(repr, self.args))})"

class Add(Term):
    __slots__ = ("args",)
    def __init__(self, args):
        self.args = tuple(args)
        self.key = ("add", tuple(a.uid for a in self.args))
    def __repr__(self): return "(" + " + ".join(map(repr, self.args)) + ")"

class Mul(Term):
    __slots__ = ("args",)
    def __init__(self, args):
        self.args = tuple(args)
        self.key = ("mul", tuple(a.uid for a in self.args))
    def __repr__(self): return "(" + "*".join(map(repr, self.args)) + ")"

import struct as _struct
def _f32(v):
    """Round to the kernel's working precision (fp32).  inf/nan pass through."""
    try: return _struct.unpack("f", _struct.pack("f", v))[0]
    except OverflowError: return float("inf") if v > 0 else float("-inf")
def const(v): return _hc(Const(_f32(float(v))))

def lift(v):
    """A concrete number reaching a real-term constructor becomes a constant.

    Integer-typed ops (math.absi, integer selects) can route a Python int into a
    float path when the value is statically known; the term for it is that
    constant.  Guarding here closes the whole class instead of every call site."""
    return v if isinstance(v, Term) else const(v)
def sym(buf, idx): return _hc(Sym(buf, idx))
ZERO, ONE = const(0.0), const(1.0)
TRUE, FALSE = _hc(App("true", ())), _hc(App("false", ()))
# everything this module hands out by name; `reset` re-interns exactly this list
_SINGLETONS = (ZERO, ONE, TRUE, FALSE)

def reset():
    """Forget every term (between independent runs), keeping the module-level
    singletons interned.

    Every constant this module hands out by name has to go back in, or the next
    `app("true")` builds a SECOND structurally identical node and `is` stops
    meaning "same term" -- which is the one invariant the whole normal form rests
    on.  TRUE and FALSE were missing, so after the first reset a folded `select`
    and a rebuilt one were different objects."""
    _pool.clear()
    for t in _SINGLETONS: _pool[t.key] = t

def _sortkey(t): return t.uid

def _split_coeff(t):
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
    terms = [lift(t) for t in terms]
    flat = []
    for t in terms:
        if isinstance(t, Add): flat.extend(t.args)
        else: flat.append(t)
    buckets, order = {}, []
    for t in flat:
        c, base = _split_coeff(t)
        b = buckets.get(base.uid)
        if b is None:
            buckets[base.uid] = [c, base]; order.append(base.uid)
        else: b[0] += c
    out = []
    for k in order:
        c, base = buckets[k]
        if c == 0.0: continue
        if base is ONE: out.append(const(c))
        elif c == 1.0: out.append(base)
        else: out.append(mul(const(c), base))
    if not out: return ZERO
    if len(out) == 1: return out[0]
    out.sort(key=_sortkey)
    return _hc(Add(out))

def mul(*terms):
    terms = [lift(t) for t in terms]
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
    rest.sort(key=_sortkey)
    return _hc(Mul(rest))

def sub(a, b):  return add(lift(a), mul(const(-1.0), lift(b)))
def div(a, b):
    a, b = lift(a), lift(b)
    if isinstance(b, Const) and b.v != 0.0: return mul(a, const(1.0 / b.v))
    return _hc(App("div", (a, b)))
def select(c, a, b):
    """Piecewise value.  Not case-split: `select` is an uninterpreted 3-ary atom,
    exactly as Volta canonicalises it, so two piecewise kernels are equal when
    their conditions and branches are.  Concrete conditions fold."""
    c, a, b = lift(c), lift(a), lift(b)
    if c is TRUE: return a
    if c is FALSE: return b
    if a is b: return a
    return _hc(App("select", (c, a, b)))

def cmp(kind, a, b):
    a, b = lift(a), lift(b)
    if kind in ("eq", "le", "ge") and a is b: return TRUE
    if kind in ("ne", "lt", "gt") and a is b: return FALSE
    return _hc(App("cmp:" + kind, (a, b)))

def app(fn, *args):
    args = tuple(lift(a) for a in args)
    if fn == "max": args = tuple(a for a in args if not (isinstance(a, Const) and a.v == float("-inf"))) or (const(float("-inf")),)
    if fn == "min": args = tuple(a for a in args if not (isinstance(a, Const) and a.v == float("inf")))  or (const(float("inf")),)
    if fn in ("max", "min"):
        # max and min are associative, commutative and idempotent over the reals,
        # exactly like + and *, and they need the same normal form for the same
        # reason: a kernel writes `maximum(0, x)` where the reference writes
        # `max(x, 0)`, and without this they are different terms.  Flatten nests,
        # fold the constants into one, drop duplicates, order by uid.
        flat = []
        for a in args:
            if isinstance(a, App) and a.fn == fn: flat.extend(a.args)
            else: flat.append(a)
        cs = [a for a in flat if isinstance(a, Const)]
        if len(cs) > 1:
            keep = (max if fn == "max" else min)(cs, key=lambda c: c.v)
            flat = [a for a in flat if not isinstance(a, Const)] + [keep]
        seen, uniq = set(), []
        for a in flat:
            if a.uid not in seen: seen.add(a.uid); uniq.append(a)
        args = tuple(sorted(uniq, key=lambda a: a.uid))
        if len(args) == 1: return args[0]
    return _hc(App(fn, args))

def gather(elems, idx, default=None):
    """One indirect read, written out: select(idx = 0, elems[0], select(idx = 1, ...)).

    The single denotation for a data-dependent read, used by BOTH sides -- the
    interpreter for `tl.load(src + j)` and the spec front-end for `x[idx]`,
    `torch.gather`, `index_select` and `embedding`.  They have to agree term for
    term or a gather can be represented but never judged, so there is one
    function and both call it.

    Exact rather than approximate: the index is an i32 into a buffer of known
    size, so the enumeration is complete, and an index outside it falls through
    to `default` -- the same NaN an out-of-bounds concrete read produces."""
    out = const(float("nan")) if default is None else lift(default)
    for j in range(len(elems) - 1, -1, -1):
        out = select(cmp("eq", lift(idx), const(float(j))), lift(elems[j]), out)
    return out

def substitute(t, table, memo=None):
    """Rewrite `t`, replacing Sym(buf, i) by table[buf][i] where the buffer is
    listed.  Goes through the real constructors so the result lands in the same
    normal form as a term that was built that way in the first place."""
    if memo is None: memo = {}
    if t is None: return None
    r = memo.get(t.uid)
    if r is not None: return r
    if isinstance(t, Sym):
        d = table.get(t.buf)
        out = d[t.idx] if d is not None and t.idx < len(d) else t
    elif isinstance(t, Const):
        out = t
    elif isinstance(t, Add):
        out = add(*[substitute(a, table, memo) for a in t.args])
    elif isinstance(t, Mul):
        out = mul(*[substitute(a, table, memo) for a in t.args])
    elif isinstance(t, App):
        args = [substitute(a, table, memo) for a in t.args]
        if t.fn == "select":          out = select(*args)
        elif t.fn.startswith("cmp:"): out = cmp(t.fn[4:], *args)
        else:                         out = app(t.fn, *args)
    else:
        raise TypeError(f"cannot substitute into {type(t).__name__}")
    memo[t.uid] = out
    return out

def size(t, seen=None):
    if seen is None: seen = set()
    if t.uid in seen: return 0
    seen.add(t.uid)
    n = 1
    for a in getattr(t, "args", ()): n += size(a, seen)
    return n

_nan_memo = {}
def has_nan(t):
    r = _nan_memo.get(t.uid)
    if r is None:
        if isinstance(t, Const): r = t.v != t.v
        else: r = any(has_nan(a) for a in getattr(t, "args", ()))
        _nan_memo[t.uid] = r
    return r


# ---- operator protocol, so numpy object arrays of Terms do the algebra ----
def _lift(x):
    if isinstance(x, Term): return x
    if isinstance(x, (int, float)): return const(x)
    return NotImplemented

def _binop(fn):
    def op(self, o):
        o = _lift(o)
        return NotImplemented if o is NotImplemented else fn(self, o)
    return op
def _rbinop(fn):
    def op(self, o):
        o = _lift(o)
        return NotImplemented if o is NotImplemented else fn(o, self)
    return op

Term.__add__      = _binop(lambda a, b: add(a, b))
Term.__radd__     = _rbinop(lambda a, b: add(a, b))
Term.__sub__      = _binop(lambda a, b: sub(a, b))
Term.__rsub__     = _rbinop(lambda a, b: sub(a, b))
Term.__mul__      = _binop(lambda a, b: mul(a, b))
Term.__rmul__     = _rbinop(lambda a, b: mul(a, b))
Term.__truediv__  = _binop(lambda a, b: div(a, b))
Term.__rtruediv__ = _rbinop(lambda a, b: div(a, b))
Term.__neg__      = lambda self: mul(const(-1.0), self)
def _pow(self, p):
    if isinstance(p, int) or (isinstance(p, float) and p == int(p)):
        p = int(p)
        if p == 0: return ONE
        if p > 0:  return mul(*([self] * p))
        return div(ONE, mul(*([self] * (-p))))
    if p == 0.5: return app("sqrt", self)
    return app("pow", self, const(p))
Term.__pow__ = _pow

"""Precondition layer: where does the real-number proof say anything about float32?

The real denotation of an output is meaningful for float32 only if no
intermediate, in ANY evaluation order the precision relation permits, leaves
the normal float32 range.  This module bounds that order-independently:

  Add(args)   |partial sum|     <= sum_i  absmax(arg_i)
  Mul(args)   |partial product| <= prod_i max(absmax(arg_i), 1)
  exp(x)      overflow iff hi(x) > ln(FLT_MAX); flush-to-zero iff lo(x) < ln(FLT_MIN)
  div(a, b)   undefined iff 0 in [lo(b), hi(b)]; inf-prone iff absmax(b) < FLT_MIN

Intervals lose correlations.  One relational rule is added because it is the
whole reason numerically-stable softmax exists:
  x - max(..., x, ...)  <= 0
Beyond that, interval reasoning is what it is; a 'may overflow' verdict is
sound, a clean verdict is sound, and neither is complete.
"""
import math
import terms as T

FLT_MAX = 3.4028234663852886e38
FLT_MIN = 1.1754943508222875e-38          # smallest normal
LN_MAX, LN_MIN = math.log(FLT_MAX), math.log(FLT_MIN)
INF = float("inf")

class Range:
    __slots__ = ("lo", "hi", "absmax", "flags")
    def __init__(self, lo, hi, absmax=None, flags=()):
        self.lo, self.hi = lo, hi
        self.absmax = max(abs(lo), abs(hi)) if absmax is None else absmax
        self.flags = tuple(flags)
    def __repr__(self): return f"[{self.lo:.3g}, {self.hi:.3g}]"

def _max_leaves(t, acc):
    if isinstance(t, T.App) and t.fn == "max":
        for a in t.args: _max_leaves(a, acc)
    else: acc.add(t)
    return acc

def _neg_max(t):
    """If Add `t` has an arg of the form c*max(...) with c < 0, return (that arg, the max, -c)."""
    for a in t.args:
        if isinstance(a, T.Mul) and len(a.args) == 2:
            c = [x for x in a.args if isinstance(x, T.Const)]
            m = [x for x in a.args if not isinstance(x, T.Const)]
            if c and c[0].v < 0 and m and isinstance(m[0], T.App) and m[0].fn == "max":
                return a, m[0], -c[0].v
    return None

def _scaled_max_split(t):
    """t == k*x - k*max(S) with x dominated by S?  -> (k, x, max) else None.
    Also accepts c*(x - max(S)) with c > 0 (Inductor factors the softmax scale out)."""
    if isinstance(t, T.Mul) and len(t.args) == 2:
        c = [z for z in t.args if isinstance(z, T.Const)]
        rest = [z for z in t.args if not isinstance(z, T.Const)]
        if c and c[0].v > 0 and rest and isinstance(rest[0], T.Add):
            inner = _scaled_max_split(rest[0])
            if inner is not None: return inner[0] * c[0].v, inner[1], inner[2]
    if not isinstance(t, T.Add): return None
    nm = _neg_max(t)
    if nm is None: return None
    rest = [a for a in t.args if a is not nm[0]]
    k = nm[2]
    rest_t = T.add(*rest) if rest else T.ZERO
    x = rest_t if k == 1.0 else T.mul(T.const(1.0 / k), rest_t)     # undo the scale
    if _dominated(x, nm[1]): return k, x, nm[1]
    # the scale may sit inside each product term instead: try matching k*leaf directly
    for leaf in _max_leaves(nm[1], set()):
        if T.mul(T.const(k), leaf) is rest_t: return k, leaf, nm[1]
    return None

def _dominated(x, m):
    """x <= max(S)?  Yes if x is a leaf of m, or x is itself a max over a subset."""
    S = _max_leaves(m, set())
    if x in S: return True
    if isinstance(x, T.App) and x.fn == "max":
        return _max_leaves(x, set()) <= S
    return False

def _exp_minus_max(a):
    """exp(k*x - k*max(S)) -> (k, x, max) or None."""
    if not (isinstance(a, T.App) and a.fn == "exp"): return None
    return _scaled_max_split(a.args[0])

def _softmax_denominator(t):
    """Add whose args are all exp(k*x_j - k*M) with one k, one M, {x_j} covering leaves(M)."""
    if not isinstance(t, T.Add): return False
    xs, M, K = set(), None, None
    for a in t.args:
        em = _exp_minus_max(a)
        if em is None: return False
        k, x, m = em
        if M is None: M, K = m, k
        elif m is not M or k != K: return False
        xs.add(x)
    return M is not None and _max_leaves(M, set()) <= xs

class Analysis:
    def __init__(self, inputs):
        """inputs: buf -> (lo, hi) for every element of that buffer."""
        self.inputs, self.memo = inputs, {}
        self.flagged = {}       # flag -> first term that raised it

    def _flag(self, name, t):
        self.flagged.setdefault(name, t)
        return name

    def range(self, t):
        k = t.key
        if k in self.memo: return self.memo[k]
        r = self._range(t)
        self.memo[k] = r
        return r

    def _range(self, t):
        if isinstance(t, T.Const):
            return Range(t.v, t.v)
        if isinstance(t, T.Sym):
            if t.buf == "ln2": return Range(0.6931471805599453, 0.6931471805599453)
            lo, hi = self.inputs[t.buf]
            return Range(lo, hi)
        if isinstance(t, T.Add):
            rs = [self.range(a) for a in t.args]
            lo, hi = sum(r.lo for r in rs), sum(r.hi for r in rs)
            absmax = sum(r.absmax for r in rs)
            flags = set(f for r in rs for f in r.flags)
            # R1:  x - max(S) <= 0   when x in S, or x = max(S') with S' subset of S.
            #      AC flattening means x may be spread over several args, so
            #      rebuild "everything except the -max term" and test identity.
            if _scaled_max_split(t) is not None:
                hi = min(hi, 0.0)
            # R2:  sum_j exp(x_j - max(S)) >= 1 when {x_j} covers S: one term is exp(0).
            if _softmax_denominator(t):
                lo = max(lo, 1.0)
            # R3:  an underflowing term whose magnitude is below half an ulp of the
            #      sum's lower bound cannot move the float sum: absorbed.
            if "underflow" in flags and lo > 0:
                small = [rr for rr in rs if "underflow" in rr.flags]
                if all(rr.absmax <= 2.0**-24 * lo for rr in small):
                    flags.discard("underflow")
            if absmax > FLT_MAX: flags.add(self._flag("overflow", t))
            return Range(lo, hi, absmax, flags)
        if isinstance(t, T.Mul):
            rs = [self.range(a) for a in t.args]
            lo, hi = 1.0, 1.0
            for r in rs:
                c = [lo*r.lo, lo*r.hi, hi*r.lo, hi*r.hi]
                lo, hi = min(c), max(c)
            partial = 1.0
            for r in rs: partial *= max(r.absmax, 1.0)
            flags = set(f for r in rs for f in r.flags)
            if partial > FLT_MAX: flags.add(self._flag("overflow", t))
            full = 1.0
            for r in rs: full *= r.absmax
            if 0.0 < full < FLT_MIN: flags.add(self._flag("underflow", t))
            return Range(lo, hi, None, flags)
        if isinstance(t, T.App):
            rs = [self.range(a) for a in t.args]
            flags = set(f for r in rs for f in r.flags)
            if t.fn == "exp":
                (r,) = rs
                if r.hi > LN_MAX: flags.add(self._flag("overflow", t))
                if r.lo < LN_MIN: flags.add(self._flag("underflow", t))
                lo = math.exp(r.lo) if r.lo < LN_MAX else INF
                hi = math.exp(r.hi) if r.hi < LN_MAX else INF
                return Range(lo, hi, None, flags)
            if t.fn == "div":
                a, b = rs
                if b.lo <= 0.0 <= b.hi:
                    flags.add(self._flag("div-by-zero", t)); return Range(-INF, INF, INF, flags)
                if b.absmax < FLT_MIN: flags.add(self._flag("underflow", t))
                c = [a.lo/b.lo, a.lo/b.hi, a.hi/b.lo, a.hi/b.hi]
                return Range(min(c), max(c), None, flags)
            if t.fn == "sqrt":
                (r,) = rs
                if r.hi < 0: flags.add(self._flag("sqrt-negative", t))
                return Range(math.sqrt(max(r.lo, 0.0)), math.sqrt(max(r.hi, 0.0)) if r.hi < INF else INF, None, flags)
            if t.fn == "abs":
                (r,) = rs
                lo = 0.0 if r.lo <= 0.0 <= r.hi else min(abs(r.lo), abs(r.hi))
                return Range(lo, r.absmax, None, flags)
            if t.fn == "log":
                (r,) = rs
                if r.lo <= 0.0: flags.add(self._flag("log-nonpositive", t))
                return Range(math.log(r.lo) if r.lo > 0 else -INF, math.log(r.hi) if 0 < r.hi < INF else (INF if r.hi == INF else -INF), None, flags)
            if t.fn in ("sin", "cos"):
                return Range(-1.0, 1.0, 1.0, flags)
            if t.fn == "tanh":
                (r,) = rs
                return Range(math.tanh(r.lo) if r.lo > -INF else -1.0, math.tanh(r.hi) if r.hi < INF else 1.0, None, flags)
            if t.fn == "gelu":
                (r,) = rs
                return Range(min(r.lo, 0.0) - 0.17, max(r.hi, 0.0), None, flags)
            if t.fn == "select":
                _, a, b = rs
                return Range(min(a.lo, b.lo), max(a.hi, b.hi), max(a.absmax, b.absmax), flags)
            if t.fn.startswith("cmp:") or t.fn in ("and", "or", "not", "true", "false"):
                return Range(0.0, 1.0, 1.0, flags)
            if t.fn == "max":
                return Range(max(r.lo for r in rs), max(r.hi for r in rs), None, flags)
            if t.fn == "min":
                return Range(min(r.lo for r in rs), min(r.hi for r in rs), None, flags)
            return Range(-INF, INF, INF, flags | {self._flag(f"unknown:{t.fn}", t)})
        raise TypeError(t)

_merge_memo = {}
def exp_merge(t):
    """Rewrite exp(a)*exp(b)*r -> exp(a+b)*r throughout (memoised, bottom-up).
    Over the reals this is an identity; it exposes the softmax-denominator shape
    in online-softmax kernels, where the rescale factor exp(m_old - m_new)
    multiplies terms exp(s - m_old)."""
    if t.key in _merge_memo: return _merge_memo[t.key]
    if isinstance(t, T.Add):   out = T.add(*[exp_merge(a) for a in t.args])
    elif isinstance(t, T.Mul):
        args = [exp_merge(a) for a in t.args]
        exps = [a for a in args if isinstance(a, T.App) and a.fn == "exp"]
        adds = [a for a in args if isinstance(a, T.Add)]
        rest = [a for a in args if not (isinstance(a, T.App) and a.fn == "exp") and not isinstance(a, T.Add)]
        if exps and len(adds) == 1:
            # online softmax: (sum_j exp(s_j - m_old)) * exp(m_old - m_new)
            # -> sum_j exp(s_j - m_new).  Distribute, then merge inside each term.
            others = rest + exps
            out = T.add(*[exp_merge(T.mul(a, *others)) for a in adds[0].args])
        elif len(exps) > 1:
            rest += adds
            rest.append(T.app("exp", T.add(*[e.args[0] for e in exps])))
            out = T.mul(*rest) if len(rest) > 1 else rest[0]
        else: out = T.mul(*args)
    elif isinstance(t, T.App): out = T.app(t.fn, *[exp_merge(a) for a in t.args])
    else: out = t
    _merge_memo[t.key] = out
    return out

def check_store(store, inputs):
    """Analyse every output term. Returns (n_outputs, {flag: count}, analysis)."""
    an = Analysis(inputs)
    counts = {}
    for k, t in store.items():
        for f in an.range(exp_merge(t)).flags:
            counts[f] = counts.get(f, 0) + 1
    return len(store), counts, an

def safe_radius(store, bufs, lo=1e-3, hi=1e30, iters=60, fatal=("overflow", "div-by-zero")):
    """Largest r such that inputs in [-r, r] (all buffers) raise no *fatal* flag.
    Underflow is excluded: it is not monotone in r (tiny inputs underflow) and
    its harm depends on what the underflowed value is later added to, which
    intervals cannot see.  Monotone in r for the fatal flags."""
    def ok(r):
        _, c, _ = check_store(store, {b: (-r, r) for b in bufs})
        return not any(f in c for f in fatal)
    if not ok(lo): return 0.0
    if ok(hi): return INF
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if ok(mid): lo = mid
        else: hi = mid
    return lo

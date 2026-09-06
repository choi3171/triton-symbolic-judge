"""Deciding piecewise equality -- the step Volta declines to take.

Volta canonicalises `select` and comparisons as uninterpreted atoms, so it
decides two piecewise kernels only when they are structurally identical.  It
cannot show `select(x > 0, x, 0) == max(x, 0)`.  Its own paper says min and max
"could be handled by case splits without affecting decidability" and that they
chose not to.

Naive case splitting is not enough.  Splitting relu on its two conditions
(`x > 0` from the select, `x >= 0` from rewriting max) leaves the branch
`x <= 0 AND x >= 0`, where the two sides reduce to `0` and `x`: equal only
because the branch constraints force `x == 0`.  Deciding that needs arithmetic
reasoning inside the branch, which is exactly what Volta avoids.

So we split the labour:
  * Volta keeps the real/exp core -- polynomial and exponential identities,
    reassociation, cancellation.  Nothing here replaces it.
  * Z3 gets the piecewise structure.  Arithmetic is interpreted; exp, log, sqrt
    and friends become uninterpreted functions.  Because our terms are
    hash-consed and AC-normalised, two occurrences of the same transcendental
    argument are the same object and so the same Z3 application.

`Not(a == b)` UNSAT means equal on every branch, vacuous branches included --
no separate satisfiability check is needed.  SAT hands back a model, which is a
counterexample point.  Anything else is UNKNOWN, never a FAIL.
"""
import z3
from tvj.core import terms as T

_CMP = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}

class Z3Trans:
    def __init__(self):
        self.vars, self.ufs, self.memo = {}, {}, {}
    def var(self, name):
        if name not in self.vars: self.vars[name] = z3.Real(name)
        return self.vars[name]
    def uf(self, fn, n):
        key = (fn, n)
        if key not in self.ufs:
            self.ufs[key] = z3.Function(f"{fn}_{n}", *([z3.RealSort()] * n), z3.RealSort())
        return self.ufs[key]
    def go(self, t):
        if t.uid in self.memo: return self.memo[t.uid]
        if isinstance(t, T.Const): r = z3.RealVal(t.v)
        elif isinstance(t, T.Sym): r = self.var(f"{t.buf}_{t.idx}")
        elif isinstance(t, T.Add):
            r = self.go(t.args[0])
            for a in t.args[1:]: r = r + self.go(a)
        elif isinstance(t, T.Mul):
            r = self.go(t.args[0])
            for a in t.args[1:]: r = r * self.go(a)
        elif isinstance(t, T.App):
            fn = t.fn
            if fn == "select":
                r = z3.If(self.pred(t.args[0]), self.go(t.args[1]), self.go(t.args[2]))
            elif fn == "max": r = self._fold(t.args, lambda a, b: z3.If(a >= b, a, b))
            elif fn == "min": r = self._fold(t.args, lambda a, b: z3.If(a <= b, a, b))
            elif fn == "abs":
                a = self.go(t.args[0]); r = z3.If(a >= 0, a, -a)
            elif fn == "div":
                a, b = self.go(t.args[0]), self.go(t.args[1]); r = a / b
            elif fn == "sqrt":
                a = self.go(t.args[0]); r = self.uf("sqrt", 1)(a)
            elif fn.startswith("cmp:") or fn in ("and", "or", "not", "true", "false"):
                r = z3.If(self.pred(t), z3.RealVal(1), z3.RealVal(0))
            else:
                r = self.uf(fn, len(t.args))(*[self.go(a) for a in t.args])
        else: raise TypeError(t)
        self.memo[t.uid] = r
        return r
    def _fold(self, args, f):
        r = self.go(args[0])
        for a in args[1:]: r = f(r, self.go(a))
        return r
    def pred(self, t):
        if isinstance(t, T.App):
            if t.fn.startswith("cmp:"): return _CMP[t.fn[4:]](self.go(t.args[0]), self.go(t.args[1]))
            if t.fn == "and": return z3.And(self.pred(t.args[0]), self.pred(t.args[1]))
            if t.fn == "or":  return z3.Or(self.pred(t.args[0]), self.pred(t.args[1]))
            if t.fn == "not": return z3.Not(self.pred(t.args[0]))
            if t.fn == "true":  return z3.BoolVal(True)
            if t.fn == "false": return z3.BoolVal(False)
        return self.go(t) != 0

def has_piecewise(t, seen=None):
    if seen is None: seen = set()
    if t.uid in seen: return False
    seen.add(t.uid)
    if isinstance(t, T.App) and (t.fn in ("select", "max", "min", "abs") or t.fn.startswith("cmp:")):
        return True
    return any(has_piecewise(a, seen) for a in getattr(t, "args", ()))

def equivalent(pairs, timeout_ms=5000, normalise=True):
    """Per pair: True (equal on every branch) / (False, model) / None (unknown).

    Z3 sees exp/log/sqrt as uninterpreted, so `exp(a)*exp(b)` and `exp(a+b)` look
    unrelated to it -- exactly the identities Volta exists to handle.  Running our
    exp-merge rewrite first lets a term that needs BOTH theories through: the
    exponential structure is normalised before Z3 does the case analysis."""
    out = []
    for a, b in pairs:
        if normalise:
            try:
                from tvj.decide.ranges import exp_merge
                a, b = exp_merge(a), exp_merge(b)
            except Exception: pass
        tr = Z3Trans()
        try:
            za, zb = tr.go(a), tr.go(b)
        except Exception: out.append(None); continue
        s = z3.Solver(); s.set("timeout", timeout_ms)
        s.add(z3.Not(za == zb))
        r = s.check()
        if r == z3.unsat: out.append(True)
        elif r == z3.sat:
            m = s.model()
            pt = {}
            for name, v in tr.vars.items():
                val = m.eval(v, model_completion=True)
                try: pt[name] = float(val.as_fraction())
                except Exception: pt[name] = 0.0
            out.append((False, pt))
        else: out.append(None)
    return out

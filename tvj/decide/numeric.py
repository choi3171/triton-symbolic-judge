"""Numeric witnesses: evaluate term DAGs at concrete points (float64).

Volta's `false` means 'not provably equal in its theory'.  For terms with
max/min (or anything outside +,*,exp) that is not a counterexample.  A random
point where the two terms take different values IS one.  A FAIL verdict must
carry such a witness; without one the verdict is UNKNOWN (unprovable)."""
import math, random
from tvj.core import terms as T

LN2 = 0.6931471805599453

def _erf(x):
    try: return math.erf(x)
    except OverflowError: return math.copysign(1.0, x)

_APP = {
    "exp": lambda a: math.exp(a) if a < 709.78 else float("inf"),
    "log": lambda a: math.log(a) if a > 0 else (float("-inf") if a == 0 else float("nan")),
    "sqrt": lambda a: math.sqrt(a) if a >= 0 else float("nan"),
    "abs": abs, "tanh": math.tanh, "erf": _erf, "sin": math.sin, "cos": math.cos,
    "div": lambda a, b: (a / b) if b != 0 else (float("nan") if a == 0 else math.copysign(float("inf"), a)),
    "max": max, "min": min,
    "select": lambda c, a, b: a if c else b,
    "and": lambda a, b: bool(a) and bool(b), "or": lambda a, b: bool(a) or bool(b),
    "not": lambda a: not bool(a), "true": lambda: True, "false": lambda: False,
    "cmp:lt": lambda a, b: a < b, "cmp:le": lambda a, b: a <= b,
    "cmp:gt": lambda a, b: a > b, "cmp:ge": lambda a, b: a >= b,
    "cmp:eq": lambda a, b: a == b, "cmp:ne": lambda a, b: a != b,
    "pow": lambda a, b: math.pow(a, b) if (a > 0 or b == int(b)) else float("nan"),
    # libdevice rounding and sign, added with the sexec entries that emit them
    "rint": lambda a: float(round(a)), "round": lambda a: math.floor(a + 0.5),
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "isinf": lambda a: math.isinf(a), "copysign": math.copysign,
    "fmod": lambda a, b: math.fmod(a, b) if b else float("nan"),
}

def evaluate(t, point, memo):
    """point: dict buf -> list/array of floats (or a float for scalar atoms)."""
    v = memo.get(t.uid)
    if v is not None: return v
    if isinstance(t, T.Const): v = t.v
    elif isinstance(t, T.Sym):
        v = LN2 if t.buf == "ln2" else point[t.buf][t.idx]
    elif isinstance(t, T.Add):
        v = 0.0
        for a in t.args: v += evaluate(a, point, memo)
    elif isinstance(t, T.Mul):
        v = 1.0
        for a in t.args:
            v *= evaluate(a, point, memo)
    elif isinstance(t, T.App):
        f = _APP.get(t.fn)
        if f is None: raise ValueError(f"no numeric interpretation for {t.fn}")
        try: v = f(*[evaluate(a, point, memo) for a in t.args])
        except (OverflowError, ValueError): v = float("nan")
    else: raise TypeError(t)
    memo[t.uid] = v
    return v

def same(a, b, rel=1e-9):
    if isinstance(a, bool) or isinstance(b, bool): return bool(a) == bool(b)
    if math.isnan(a) and math.isnan(b): return True
    if math.isinf(a) or math.isinf(b): return a == b
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))

def random_point(bufsize, lo=-1.0, hi=1.0, seed=0):
    rng = random.Random(seed)
    return {b: [rng.uniform(lo, hi) for _ in range(n)] for b, n in bufsize.items()}

def witness(pairs, bufsize, samples=6, seed=0, lo=-1.0, hi=1.0):
    """For each (spec, kernel) pair, one of
        (True,  None)                      equal at every valid sample
        (False, (i, spec, kernel))         a counterexample
        (None,  "no valid point")          neither side is finite anywhere we looked

    A point where EITHER side is non-finite is not a counterexample -- it is
    outside the domain the expression is defined on (log of a negative, a
    division by zero).  Judging on such a point reports a defect for
    `RMSE_log`-style kernels whose reference is itself NaN there.  We skip those
    points, and if no point leaves both sides finite we refuse to decide."""
    import itertools
    ranges = [(lo, hi), (0.05, 1.0), (0.5, 2.0)]      # signed, then positive domains
    points = [random_point(bufsize, l, h, seed=seed + s)
              for s, (l, h) in enumerate(itertools.islice(itertools.cycle(ranges), samples))]
    out = []
    for a, b in pairs:
        verdict, valid = (True, None), 0
        for si, pt in enumerate(points):
            memo = {}
            va, vb = evaluate(a, pt, memo), evaluate(b, pt, memo)
            if not (isinstance(va, bool) or math.isfinite(va)) or not (isinstance(vb, bool) or math.isfinite(vb)): continue
            valid += 1
            if not same(va, vb): verdict = (False, (si, va, vb)); break
        if valid == 0: verdict = (None, "no point leaves both sides finite")
        out.append(verdict)
    return out

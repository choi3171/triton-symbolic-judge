"""Properties the term algebra must satisfy, on randomly generated terms.

Half the defects a review of this project turns up are not the kind anyone reads
their way to.  `("const", nan)` never matched the pool because NaN is not equal
to itself, so no two terms containing one could ever be identified.  `reset()`
re-interned ZERO and ONE but not TRUE and FALSE, so after the first row a folded
`select` and a rebuilt one were different objects.  Neither is visible in a diff;
both are caught by one property.

The property that matters most, because every normalisation added to this module
is a chance to break it:

    a is b   =>   a and b evaluate to the same number, everywhere

Identity IS the equality decision here -- `check.py` reports `AC 1024/1024` when
the two sides are the same object -- so an over-eager normal form is not a missed
optimisation, it is a false PASS. The converse is not required: two terms may be
equal and not identical, which is what Volta and Z3 are for.

    python3 -m tvj.checks.metamorphic [trials] [seed]
"""
import math, random, sys
from tvj.core import terms as T
from tvj.decide import numeric as NUM

BUFS = {"a": 4, "b": 4, "c": 4}
# Ordinary values only.  The algebra is over the REALS and folds `0 * x` to 0 for
# every x, while an IEEE shadow says 0 * nan = nan -- comparing the two at
# exceptional values measures that documented gap rather than the property under
# test, which is whether a constructor returns the term it was asked for.
LEAF_CONSTS = [0.0, 1.0, -1.0, 0.5, 2.0, 3.0, -2.0]
UNARY = ["exp", "log", "sqrt", "abs", "tanh", "sin", "cos", "erf"]
NARY = ["max", "min"]


# The intended meaning of each constructor, written out here rather than looked up
# from the module under test.  Evaluating the term the constructor RETURNS cannot
# catch a constructor that returns the wrong term -- ask for min(x,y), get back
# max(x,y), evaluate it, and of course it agrees with itself.  So every generated
# term is carried alongside a closure that says what it was supposed to compute.
_MEAN = {
    "exp": math.exp, "log": lambda v: math.log(v) if v > 0 else float("nan"),
    "sqrt": lambda v: math.sqrt(v) if v >= 0 else float("nan"), "abs": abs,
    "tanh": math.tanh, "sin": math.sin, "cos": math.cos, "erf": math.erf,
}
_REL = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}


def _guard(f):
    def go(pt):
        try:
            v = f(pt)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        except (ValueError, OverflowError, ZeroDivisionError):
            return float("nan")
    return go


def rand_term(rng, depth=0):
    """A random term, and independently, what it is supposed to evaluate to."""
    if depth > 3 or rng.random() < 0.3:
        if rng.random() < 0.45:
            v = rng.choice(LEAF_CONSTS)
            return T.const(v), _guard(lambda pt, v=v: v)
        b = rng.choice(list(BUFS)); i = rng.randrange(BUFS[b])
        return T.sym(b, i), _guard(lambda pt, b=b, i=i: pt[b][i])
    k = rng.random()
    kids = lambda n: [rand_term(rng, depth + 1) for _ in range(n)]
    if k < 0.20:
        xs = kids(rng.randint(2, 4))
        return T.add(*[t for t, _ in xs]), _guard(lambda pt, xs=xs: sum(f(pt) for _, f in xs))
    if k < 0.40:
        xs = kids(rng.randint(2, 4))
        return T.mul(*[t for t, _ in xs]), _guard(
            lambda pt, xs=xs: math.prod([f(pt) for _, f in xs]))
    if k < 0.52:
        fn = rng.choice(NARY); xs = kids(rng.randint(2, 4)); g = max if fn == "max" else min
        return T.app(fn, *[t for t, _ in xs]), _guard(
            lambda pt, xs=xs, g=g: g(f(pt) for _, f in xs))
    if k < 0.64:
        fn = rng.choice(UNARY); (t, f), = kids(1)
        return T.app(fn, t), _guard(lambda pt, f=f, m=_MEAN[fn]: m(f(pt)))
    if k < 0.74:
        (ta, fa), (tb, fb) = kids(2)
        return T.div(ta, tb), _guard(lambda pt, fa=fa, fb=fb: fa(pt) / fb(pt))
    if k < 0.86:
        rel = rng.choice(list(_REL)); (ca, fca), (cb, fcb) = kids(2)
        (ta, fa), (tb, fb) = kids(2)
        return (T.select(T.cmp(rel, ca, cb), ta, tb),
                _guard(lambda pt, r=_REL[rel], fca=fca, fcb=fcb, fa=fa, fb=fb:
                       fa(pt) if r(fca(pt), fcb(pt)) else fb(pt)))
    xs = kids(rng.randint(1, 4)); (ti, fi), = kids(1)
    def gsem(pt, xs=xs, fi=fi):
        j = fi(pt)
        k_ = int(j) if isinstance(j, float) and j == int(j) else None
        return xs[k_][1](pt) if (k_ is not None and 0 <= k_ < len(xs)) else float("nan")
    return T.gather([t for t, _ in xs], ti), _guard(gsem)


TAME_UNARY = ["abs", "tanh", "sin", "cos"]        # total, and NUM agrees with math


def tame_term(rng, depth=0):
    """A term built only from total operations, carried with its meaning.

    The shadow has to be RIGHT or every mismatch is noise.  `log`, `sqrt` and `div`
    are partial, and `div` by a constant is folded to a multiplication by an
    f32-rounded reciprocal (decision `literal.working-precision`) -- both are
    documented behaviour that a naive shadow reads as a violation.  Structure is
    checked separately, where none of that matters."""
    if depth > 3 or rng.random() < 0.35:
        if rng.random() < 0.4:
            v = rng.choice(LEAF_CONSTS)
            return T.const(v), (lambda pt, v=v: v)
        b = rng.choice(list(BUFS)); i = rng.randrange(BUFS[b])
        return T.sym(b, i), (lambda pt, b=b, i=i: pt[b][i])
    kids = lambda n: [tame_term(rng, depth + 1) for _ in range(n)]
    k = rng.random()
    if k < 0.25:
        xs = kids(rng.randint(2, 4))
        return T.add(*[t for t, _ in xs]), (lambda pt, xs=xs: sum(f(pt) for _, f in xs))
    if k < 0.50:
        xs = kids(rng.randint(2, 4))
        return T.mul(*[t for t, _ in xs]), (lambda pt, xs=xs: math.prod([f(pt) for _, f in xs]))
    if k < 0.68:
        fn = rng.choice(NARY); xs = kids(rng.randint(2, 4)); g = max if fn == "max" else min
        return T.app(fn, *[t for t, _ in xs]), (lambda pt, xs=xs, g=g: g(f(pt) for _, f in xs))
    if k < 0.80:
        fn = rng.choice(TAME_UNARY); (t, f), = kids(1)
        m = {"abs": abs, "tanh": math.tanh, "sin": math.sin, "cos": math.cos}[fn]
        return T.app(fn, t), (lambda pt, f=f, m=m: m(f(pt)))
    if k < 0.92:
        rel = rng.choice(list(_REL)); (ca, fca), (cb, fcb) = kids(2); (ta, fa), (tb, fb) = kids(2)
        return (T.select(T.cmp(rel, ca, cb), ta, tb),
                (lambda pt, r=_REL[rel], fca=fca, fcb=fcb, fa=fa, fb=fb:
                 fa(pt) if r(fca(pt), fcb(pt)) else fb(pt)))
    xs = kids(rng.randint(1, 4)); n = len(xs)
    j = rng.randrange(n)                       # a concrete, in-range index
    return T.gather([t for t, _ in xs], T.const(float(j))), (lambda pt, xs=xs, j=j: xs[j][1](pt))


# What a constructor is allowed to return: the node it was asked for, one of its
# own operands (an identity fold), or a constant (constant folding).  Anything
# else means the constructor answered a different question -- which is how an
# over-eager normal form turns into a false PASS, and is invisible to any check
# that evaluates the term it got back.
def _occurs(t, roots):
    """Is `t` somewhere inside any of `roots`?  A constructor is allowed to return
    a piece of what it was given -- that is what an identity fold does."""
    seen = set()
    stack = list(roots)
    while stack:
        x = stack.pop()
        if x is t: return True
        if x.uid in seen: continue
        seen.add(x.uid); stack.extend(getattr(x, "args", ()))
    return False


def check_head_is_what_was_asked(rng, trials):
    bad = []
    for _ in range(trials):
        args = [tame_term(rng, 2)[0] for _ in range(rng.randint(2, 3))]
        for fn in NARY + ["exp", "log", "sqrt", "tanh"]:
            xs = args if fn in NARY else args[:1]
            r = T.app(fn, *xs)
            ok = (getattr(r, "fn", None) == fn) or _occurs(r, xs) or isinstance(r, T.Const)
            if not ok: bad.append((fn, [str(x)[:24] for x in xs], str(r)[:40]))
        a, b = args[0], args[1]
        # `add` and `mul` may change head, and legitimately: the AC normal form
        # collects coefficients, so `x + x` is `2*x` (an Add becomes a Mul) and
        # `x + -x` is `0` (a Const).  Those are documented normalisations, not a
        # different question being answered -- measured, the reachable heads are
        # exactly {Add, Mul, Const, an operand}, so that is what is allowed here.
        # Anything outside it would mean the constructor rewrote the problem.
        # ... and the fold can land on a NESTED subterm, not just a direct operand:
        # `2 + (-2 + X)` flattens, the constants cancel, and `X` is what is left.
        ALLOWED = (T.Add, T.Mul, T.Const, T.Sym)
        for name, r in (("add", T.add(a, b)), ("mul", T.mul(a, b))):
            if not (isinstance(r, ALLOWED) or _occurs(r, (a, b))):
                bad.append((name, [str(a)[:24], str(b)[:24]], str(r)[:40]))
        for rel in _REL:
            r = T.cmp(rel, a, b)
            if not (getattr(r, "fn", None) == "cmp:" + rel or r in (T.TRUE, T.FALSE)):
                bad.append(("cmp:" + rel, [str(a)[:24], str(b)[:24]], str(r)[:40]))
    return bad


def points(rng, n=6):
    out = []
    for _ in range(n):
        out.append({b: [rng.choice([rng.uniform(-3, 3), 0.0, 1.0, 2.0, -1.0]) for _ in range(k)]
                    for b, k in BUFS.items()})
    return out


def ev(t, pt):
    try: return NUM.evaluate(t, pt, {})
    except Exception: return "err"


def same_number(x, y, tol=1e-9):
    if x == "err" or y == "err": return x == y
    if isinstance(x, bool) or isinstance(y, bool): return bool(x) == bool(y)
    if isinstance(x, float) and isinstance(y, float):
        if math.isnan(x) and math.isnan(y): return True
        if math.isinf(x) or math.isinf(y): return x == y
        return abs(x - y) <= tol * max(1.0, abs(x), abs(y))
    return x == y


def check_construction_is_sound(rng, trials):
    """Every term must evaluate to what its construction meant.

    This is the property with teeth: it compares the term against an independent
    statement of intent, so a constructor that normalises two different things
    into one node is caught -- ask for min, get max, and the shadow says min."""
    pts = points(rng)
    seen, bad = {}, []
    for _ in range(trials):
        t, meaning = tame_term(rng)
        for p in pts:
            got, want = ev(t, p), meaning(p)
            # a point where either side leaves the reals says nothing about
            # structure; the accuracy and precondition layers are what cover those
            if any(isinstance(v, float) and not math.isfinite(v) for v in (got, want)): continue
            if got == "err" or want == "err": continue
            if not same_number(got, want, tol=1e-6):
                bad.append((t, got, want)); break
        seen[t.uid] = 1
    return len(seen), bad


def check_singletons_after_reset():
    T.reset()
    out = []
    for name, want in (("true", T.TRUE), ("false", T.FALSE)):
        if T.app(name) is not want: out.append(f"app({name}) is not the module singleton")
    for v, want in ((0.0, T.ZERO), (1.0, T.ONE)):
        if T.const(v) is not want: out.append(f"const({v}) is not the module singleton")
    if T.const(float("nan")) is not T.const(float("nan")): out.append("const(nan) does not intern")
    if T.select(T.app("true"), T.const(1.0), T.const(2.0)) is not T.const(1.0):
        out.append("select(true, a, b) does not fold after reset")
    return out


def check_rebuild_is_stable(rng, trials=200):
    """The same construction, before and after a reset, must give the same shape."""
    bad = []
    for _ in range(trials):
        seed = rng.randrange(1 << 30)
        T.reset(); a, _ = tame_term(random.Random(seed))
        sa = str(a)
        T.reset(); b, _ = tame_term(random.Random(seed))
        if str(b) != sa: bad.append((sa[:60], str(b)[:60]))
    return bad


def check_substitute_identity(rng, trials=200):
    bad = []
    for _ in range(trials):
        t, _ = tame_term(rng)
        if T.substitute(t, {}) is not t: bad.append(str(t)[:60])
    return bad


def check_gather_picks(rng, trials=200):
    """gather(elems, k) at a concrete k must be elems[k], and out of range NaN."""
    bad = []
    for _ in range(trials):
        n = rng.randint(1, 5)
        elems = [T.sym("a", i % BUFS["a"]) for i in range(n)]
        k = rng.randrange(-1, n + 1)
        g = T.gather(elems, T.const(float(k)))
        pt = points(rng, 1)[0]
        want = ev(elems[k], pt) if 0 <= k < n else float("nan")
        if not same_number(ev(g, pt), want): bad.append((n, k, ev(g, pt), want))
    return bad


# One seed is one point.  The first version of this file passed on its own fixed
# seed and failed on nine of the next ten -- on a false alarm of its own, but the
# lesson stands: a property is only as good as the sample, so the sweep is part of
# the check rather than something a reader is trusted to do.
SEEDS = [1, 2, 3, 7, 11, 42, 101, 999, 31337, 20260907]


def one_seed(seed, trials):
    fails = 0
    lines = []

    T.reset()
    n_terms, bad = check_construction_is_sound(random.Random(seed), trials)
    lines.append(f"  a term computes what it meant        {n_terms} distinct   "
                 f"{'ok' if not bad else f'{len(bad)} VIOLATIONS'}")
    for t, g, w in bad[:2]: lines.append(f"      {str(t)[:66]}\n        got {g}  want {w}")
    fails += len(bad)

    T.reset()
    b = check_head_is_what_was_asked(random.Random(seed + 1), max(trials // 20, 50))
    lines.append(f"  a constructor returns what was asked  {'ok' if not b else f'{len(b)} VIOLATIONS'}")
    for fn, xs, got in b[:2]: lines.append(f"      {fn}({', '.join(xs)}) -> {got}")
    fails += len(b)

    out = check_singletons_after_reset()
    lines.append(f"  singletons survive reset()           {'ok' if not out else 'BROKEN'}")
    for o in out: lines.append("      " + o)
    fails += len(out)

    b = check_rebuild_is_stable(random.Random(seed + 2))
    lines.append(f"  rebuild after reset is stable        {'ok' if not b else f'{len(b)} differ'}")
    fails += len(b)

    b = check_substitute_identity(random.Random(seed + 3))
    lines.append(f"  substitute(t, {{}}) is t               {'ok' if not b else f'{len(b)} differ'}")
    fails += len(b)

    T.reset()
    b = check_gather_picks(random.Random(seed + 4))
    lines.append(f"  gather(elems, k) == elems[k]         {'ok' if not b else f'{len(b)} wrong'}")
    fails += len(b)
    return fails, lines


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS
    total = 0
    for sd in seeds:
        f, lines = one_seed(sd, trials)
        total += f
        print(f"seed {sd}" + ("" if not f else f"   {f} VIOLATION(S)"))
        if f: print("\n".join(lines))
    print(f"\n{len(seeds)} seeds x {trials} terms: "
          + ("all properties hold" if not total else f"{total} violation(s)"))
    sys.exit(1 if total else 0)


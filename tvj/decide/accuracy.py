"""The accuracy obligation: is the kernel a numerically worse way to compute the
same real number?

The value obligation works over the reals, where `E[X^2] - E[X]^2` and
`E[(X-mu)^2]` are the same expression -- and it is right to say so.  The defect
is catastrophic cancellation, which is a property of the float REPRESENTATION:
two nearly equal floats subtract and the leading digits vanish.  Reals have no
significant digits to lose, so the phenomenon does not exist in that domain.

The precision lattice does not see it either (both forms use the same
precision), nor does the range analysis (every intermediate stays finite and far
inside float32 -- measured: at a shift of 1e4 the two means are both 1.0001e8 and
their difference is -8, with no overflow anywhere).

Modelling IEEE-754 exactly would see it, and would also reject every legitimate
reassociation -- split-K, flash attention -- which is why we chose the reals.

So this layer does not bound the error statically.  Forward error bounds carry an
`n*u` factor that penalises any large reduction, which would flag correct kernels.
Instead:

  1. evaluate the same term twice, once rounding every step to float32 and once
     in float64, and take the difference as the empirical error;
  2. do it at input regimes that STRESS cancellation -- large shifts, where
     nearly equal operands subtract -- rather than at whatever the benchmark
     happens to sample;
  3. report the kernel's error RELATIVE TO THE REFERENCE's, never in absolute
     terms.  Both sides are computed the same way, so a ratio is meaningful
     where an absolute bound is not.

Unsound, like every other bounded part of this project: it says "worse here",
never "safe everywhere".
"""
import math, struct
from tvj.core import terms as T
from tvj.decide import numeric as NUM

def f32(x):
    try: return struct.unpack("f", struct.pack("f", x))[0]
    except (OverflowError, ValueError): return math.copysign(math.inf, x or 1.0)

def evaluate32(t, point, memo):
    """`numeric.evaluate`, but every intermediate is rounded to float32 -- which
    is what the kernel and the reference both actually execute."""
    v = memo.get(t.uid)
    if v is not None: return v
    if isinstance(t, T.Const): v = f32(t.v)
    elif isinstance(t, T.Sym): v = f32(NUM.LN2 if t.buf == "ln2" else point[t.buf][t.idx])
    elif isinstance(t, T.Add):
        v = 0.0
        for a in t.args: v = f32(v + evaluate32(a, point, memo))
    elif isinstance(t, T.Mul):
        v = 1.0
        for a in t.args: v = f32(v * evaluate32(a, point, memo))
    elif isinstance(t, T.App):
        f = NUM._APP.get(t.fn)
        if f is None: raise ValueError(f"no numeric interpretation for {t.fn}")
        try: v = f32(f(*[evaluate32(a, point, memo) for a in t.args]))
        except (OverflowError, ValueError, ZeroDivisionError): v = float("nan")
    else: raise TypeError(t)
    memo[t.uid] = v
    return v

def error_at(t, point, m64=None, m32=None):
    """|float32 result - float64 result| / |float64 result|, or None where the
    exact value is not finite or is zero.

    The memo dicts are caller-supplied so a whole output tensor can be evaluated
    with ONE pass over the shared sub-DAG: outputs of the same kernel differ only
    in their leaves, and a fresh memo per element re-walked the common part once
    per output."""
    exact = NUM.evaluate(t, point, {} if m64 is None else m64)
    if not isinstance(exact, float) or not math.isfinite(exact) or exact == 0.0: return None
    got = evaluate32(t, point, {} if m32 is None else m32)
    if not math.isfinite(got): return math.inf
    return abs(got - exact) / abs(exact)

# Shifting the inputs by a large constant is what drives nearly equal operands
# into a subtraction; it is the regime `E[X^2] - E[X]^2` fails in and the one a
# benchmark sampling U[0,1) never visits.
REGIMES = [("centred", 0.0, 1.0), ("shift 1e2", 1e2, 1.0),
           ("shift 1e4", 1e4, 1.0), ("shift 1e6", 1e6, 1.0)]

def shifted_point(bufsize, shift, scale, seed=0):
    pt = NUM.random_point(bufsize, 0.0, scale, seed=seed)
    return {b: [shift + v for v in vs] for b, vs in pt.items()}

# Past some shift the reference is meaningless too -- float32 cannot hold a
# variance of 0.08 around a mean of 1e6, and both sides are then wrong by 100%.
# A regime where the REFERENCE has already lost is not evidence about the kernel.
REFERENCE_USABLE = 1e-3

# One float32 rounding.  A reference form may not be credited with being MORE
# accurate than that: a spec whose evaluation order happens to cancel exactly
# lands at 1e-10 and would otherwise make any ordinary kernel -- correct, one ulp
# off -- look 100x worse.  Flooring the denominator here means the alarm can only
# fire once the kernel's own error is ~100 ulps, which rounding alone cannot reach.
U32 = 2.0 ** -24                                     # 5.96e-08

# A ratio alone is not a defect.  The claim this layer is allowed to make is the
# sharp one: in a regime where the REFERENCE is still accurate (< REFERENCE_USABLE)
# the kernel has ALREADY LOST (> KERNEL_MATERIAL), by at least `ratio_alarm`.  A
# kernel that is 200x worse at 6e-05 is still inside every tolerance anyone runs;
# reporting that as a defect over-claims.  Those cases come back as "noted".
KERNEL_MATERIAL = 1e-3

def compare(spec_terms, kern_terms, bufsize, ratio_alarm=100.0, seed=0):
    """Worst regime in which the kernel's float32 error exceeds the reference's,
    among regimes where the reference itself is still accurate.

    Returns ("pass", None), ("noted", d) for a degradation too small to matter, or
    ("worse", d) with d = {regime, spec_rel_err, kernel_rel_err, ratio}."""
    worst = None
    for name, shift, scale in REGIMES:
        pt = shifted_point(bufsize, shift, scale, seed)
        m64, m32 = {}, {}                    # shared for the whole regime
        for s, k in zip(spec_terms, kern_terms):
            if s is None or k is None or s is k: continue
            es, ek = error_at(s, pt, m64, m32), error_at(k, pt, m64, m32)
            if es is None or ek is None or es > REFERENCE_USABLE: continue
            ratio = ek / max(es, U32)
            if ratio > ratio_alarm and (worst is None or ratio > worst[3]):
                worst = (name, es, ek, ratio)
    if worst is None: return "pass", None
    name, es, ek, ratio = worst
    shift = dict((n, s) for n, s, _ in REGIMES).get(name, 0.0)
    d = {"regime": name, "shift": shift, "spec_rel_err": es, "kernel_rel_err": ek, "ratio": ratio}
    return ("worse" if ek > KERNEL_MATERIAL else "noted"), d

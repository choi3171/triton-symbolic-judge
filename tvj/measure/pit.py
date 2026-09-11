"""Decide equality by evaluating at random points, instead of by normalising.

The judge's expensive case is not a big term graph.  KernelBook row 61's output
element is 679 nodes and canonicalising ONE of them exceeds 3 GB: what blows up
is the normal form -- expanding a product of sums into a sum of products is
exponential in multiplicative depth and nearly independent of the input's size.

Schwartz-Zippel says you never needed the normal form.  A non-zero polynomial of
total degree d, evaluated at a point drawn uniformly from S^n, is zero with
probability at most d/|S|; so two expressions that agree at several random points
over a large field are equal with overwhelming probability, and the cost is one
pass over the DAG rather than an expansion of it.

Exponentials are the reason this fits THIS project rather than being a generic
trick.  Following Mirage (arXiv:2405.05751), pick primes p and q with q | p-1 and
let w be a q-th root of unity in Z_p; evaluate exponents in Z_q and everything
else in Z_p, with exp(x) = w^x.  Then

    exp(a)*exp(b) = w^a * w^b = w^(a+b) = exp(a+b)

holds because the FIELD says so, not because anything derived it -- which is
exactly the identity flash attention's telescoping rescale needs and the one
Volta spends gigabytes canonicalising.  The restriction that comes with it is at
most one exp on any input-to-output path; attention is inside that.

`max`, `min` and everything else outside the theory become opaque atoms keyed by
term identity, which is what Volta does with them too.  Because both sides are
built in one hash-consed pool, the same atom on both sides IS the same object and
gets the same random value -- and AC normalisation has already flattened
max(max(a,b),c) and max(a,b,c) into one atom before we get here.

What this buys and what it costs:

  agree at k points  ->  equal, with probability at least 1 - (d/|S|)^k
  disagree           ->  not equal AS POLYNOMIALS IN THE ATOMS, which is weaker
                         than "not equal" -- two atoms can be equal for a reason
                         this cannot see

so a disagreement must fall through to the real decision procedure, exactly as an
AC mismatch does today.  It is a filter in front of Volta, not a replacement.

    python3 -m tvj.measure.pit [L]
"""
import random, sys, time
from fractions import Fraction
import numpy as np
from tvj.core import terms as T

# q | p-1, both prime, so Z_p has a q-th root of unity.  61 bits of exponent and
# 67 of value: the Schwartz-Zippel bound d/|S| is astronomically small for any
# degree these kernels reach.
Q = (1 << 61) - 1
P = 52 * Q + 1


def _omega():
    k, g = (P - 1) // Q, 2
    while True:
        w = pow(g, k, P)
        if w != 1: return w
        g += 1


OMEGA = _omega()


class Unsupported(Exception):
    """Outside the fragment this evaluator can decide."""


class Point:
    """One random point: a value for every input symbol and every opaque atom."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.atom = {}                        # (modulus, key) -> value
        self.vp, self.vq = {}, {}             # memo, by term uid

    def _draw(self, mod, key):
        k = (mod, key)
        v = self.atom.get(k)
        if v is None:
            v = self.atom[k] = self.rng.randrange(1, mod)
        return v

    def _const(self, v, mod):
        if v != v or v in (float("inf"), float("-inf")):
            raise Unsupported(f"non-finite constant ({v})")
        fr = Fraction(str(np.float32(v)))     # the same reading the bridge gives Volta
        return fr.numerator % mod * pow(fr.denominator % mod, -1, mod) % mod

    def _eval(self, t, mod, memo, in_exponent):
        r = memo.get(t.uid)
        if r is not None: return r
        if isinstance(t, T.Sym):
            r = self._draw(mod, ("sym", t.buf, t.idx))
        elif isinstance(t, T.Const):
            r = self._const(t.v, mod)
        elif isinstance(t, T.Add):
            r = sum(self._eval(a, mod, memo, in_exponent) for a in t.args) % mod
        elif isinstance(t, T.Mul):
            r = 1
            for a in t.args: r = r * self._eval(a, mod, memo, in_exponent) % mod
        elif isinstance(t, T.App) and t.fn == "exp":
            if in_exponent:
                raise Unsupported("exp inside an exponent is outside the fragment")
            r = pow(OMEGA, self._eval(t.args[0], Q, self.vq, True), P)
            if mod != P: raise Unsupported("exp reached in the exponent field")
        elif isinstance(t, T.App) and t.fn == "div":
            d = self._eval(t.args[1], mod, memo, in_exponent)
            if d == 0: raise Unsupported("denominator vanished at this point; retry")
            r = self._eval(t.args[0], mod, memo, in_exponent) * pow(d, -1, mod) % mod
        else:
            # max/min/select/cmp/sqrt/log/abs/...: opaque, but keyed by what its
            # ARGUMENTS came to rather than by its own term identity.  Identity is
            # too fine: KernelBook row 148 has two atoms whose arguments are equal
            # for a reason the pool's normal form does not see, and keying on uid
            # separated a pair Volta proves equal.  Evaluating the arguments first
            # is the same thing Volta does by canonicalising inside the atom.
            key = (t.fn, tuple(self._eval(a, mod, memo, in_exponent) for a in t.args))
            r = self._draw(mod, key)
        memo[t.uid] = r
        return r

    def value(self, t):
        return self._eval(t, P, self.vp, False)


def equal(pairs, trials=3, seed=0):
    """(verdicts, seconds).  True = equal at every point, False = separated."""
    t0 = time.time()
    out = [True] * len(pairs)
    for k in range(trials):
        pt = Point(seed * 1000 + k)
        for i, (a, b) in enumerate(pairs):
            if out[i] and pt.value(a) != pt.value(b): out[i] = False
    return out, time.time() - t0


# --- does it agree with the decision procedure, on the pairs that cost the most? --
if __name__ == "__main__":
    from tvj.core import ttir as PR
    from tvj.core import sexec as X
    from tvj.decide import volta_bridge as V
    from tvj.fixtures import attn
    from tvj.checks.check import to_ttir

    L, D = (int(sys.argv[1]) if len(sys.argv) > 1 else 64), 16
    ASIG = {"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32", "o_ptr": "*fp32",
            "L": "i32", "D": "i32", "BM": "constexpr", "BD": "constexpr", "BL": "constexpr"}
    FSIG = {**{k: v for k, v in ASIG.items() if k != "BL"}, "BN": "constexpr"}
    bufs = {"q_ptr": L*D, "k_ptr": L*D, "v_ptr": L*D, "o_ptr": L*D}
    args = [X.Ptr("q_ptr", 0), X.Ptr("k_ptr", 0), X.Ptr("v_ptr", 0), X.Ptr("o_ptr", 0), L, D]

    def run(fn, sig, cst):
        f = PR.parse(to_ttir(fn, sig, cst))
        it = X.Interp(f, None, (L // 16,), bufs); it.argvals = args
        it.run_all(); return it.g.store

    S = {}
    for name, fn, sig, cst in (
            ("ref",       attn.attn_ref,             ASIG, {"BM": 16, "BD": 16, "BL": L}),
            ("safe",      attn.attn_safe,            ASIG, {"BM": 16, "BD": 16, "BL": L}),
            ("flash",     attn.attn_flash,           FSIG, {"BM": 16, "BD": 16, "BN": 16}),
            ("norescale", attn.attn_flash_norescale, FSIG, {"BM": 16, "BD": 16, "BN": 16})):
        S[name] = run(fn, sig, cst)
    keys = sorted(S["ref"])
    print(f"attention L={L} D={D} -- {len(keys)} output elements, "
          f"DAG(out[0]) = {T.size(S['ref'][keys[0]])} nodes\n")

    CASES = [("ref", "safe", True), ("safe", "flash", True), ("ref", "flash", True),
             ("ref", "norescale", False)]   # the last is the control: it MUST separate
    print(f"{'pair':<20} {'want':>6} {'random points':>16} {'secs':>7}   {'Volta':>26}")
    ok = True
    for a, b, want in CASES:
        pairs = [(S[a][k], S[b][k]) for k in keys]
        got, dt = equal(pairs)
        n = sum(got)
        agree = (n == len(keys)) if want else (n == 0)
        ok &= agree
        try:
            vres, vst = V.equivalent(pairs)
            vn = sum(r is True for r in vres)
            volta = f"{vn}/{len(keys)} true, {vst['peak_rss_mb']/1024:.2f} GB, {vst['secs']:.1f}s"
            ok &= (vn == len(keys)) == want
        except V.Unsupported as e:
            volta = f"UNDECIDED ({str(e)[:22]})"
        print(f"  {a:<6} vs {b:<10} {str(want):>6} {f'{n}/{len(keys)} equal':>16} "
              f"{dt:>7.2f}   {volta:>26}")

    print(f"\n  random points decide every pair the way the procedure does, control "
          f"included: {'ok' if ok else 'NO'}")

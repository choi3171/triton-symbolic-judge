"""Evaluation at random points decides PASSes, so what it gets wrong in the EQUAL
direction is a false PASS.  This pins that direction down.

pit's error bound -- (d/2^61)^3 per pair -- holds only if pit.py computes what it
claims to.  The probability is not the risk; the implementation is.  An opaque atom
(max, min, select, a comparison, sqrt, log, abs, an exp nested inside an exponent)
gets a random value keyed on its function and its arguments' values, and three
bugs in that key would each make two different terms agree:

  the key ignores the arguments   max(x, y) and max(x, z) agree
  the key ignores the function    max(x, y) and min(x, y) agree
  the key ignores argument order  select(c, a, b) and select(c, b, a) agree

Each has cases here that must separate.  The equal cases guard the other way --
two terms equal for a reason the pool's normal form does not see, like
distributivity inside an atom, must not come apart, or a PASS Volta would give
becomes an UNKNOWN.

And the cases are checked against the bugs themselves: the module is re-executed
with each bug planted in its source, and every planted bug must fail at least one
case.  A test that still passes with the bug in is not testing for it.

    python3 -m tvj.checks.pit_test
"""
import inspect
from tvj.core import terms as T
from tvj.decide import pit

x, y, z, w = (T.sym(n, 0) for n in "xyzw")
a, b = T.sym("a", 0), T.sym("b", 0)
A, S, C = T.app, T.select, T.cmp
E = lambda t: A("exp", t)
# Two spellings of one value, with a symbol created between them: the pool sorts
# max's arguments by uid, and v's uid falls between the two, so max(u, v) and
# max(v, u') arrive in opposite orders.  Built in this order on purpose.
u1 = T.mul(x, T.add(y, z))
v = T.sym("v", 0)
u2 = T.add(T.mul(x, y), T.mul(x, z))

SEPARATE, EQUAL = False, True
CASES = [
    # the atom differs only inside an argument
    ("sqrt differs inside its argument",     A("sqrt", T.add(x, y)),       A("sqrt", T.add(x, z)),       SEPARATE),
    ("log differs inside its argument",      A("log", T.mul(x, y)),        A("log", T.mul(x, z)),        SEPARATE),
    ("abs differs inside its argument",      A("abs", x),                  A("abs", y),                  SEPARATE),
    ("max differs inside an argument",       A("max", x, y),               A("max", x, z),               SEPARATE),
    ("min differs inside an argument",       A("min", T.add(x, y), w),     A("min", T.add(x, z), w),     SEPARATE),
    ("select differs inside its condition",  S(C("lt", x, y), a, b),       S(C("lt", x, z), a, b),       SEPARATE),
    ("nested exp differs inside",            E(E(x)),                      E(E(y)),                      SEPARATE),
    # same arguments, a different function
    ("max is not min",                       A("max", x, y),               A("min", x, y),               SEPARATE),
    ("sqrt is not log",                      A("sqrt", x),                 A("log", x),                  SEPARATE),
    ("lt is not le",                         S(C("lt", x, y), a, b),       S(C("le", x, y), a, b),       SEPARATE),
    # order matters where the atom is not commutative
    ("select keeps its branches in order",   S(C("lt", x, y), a, b),       S(C("lt", x, y), b, a),       SEPARATE),
    ("lt keeps its arguments in order",      S(C("lt", x, y), a, b),       S(C("lt", y, x), a, b),       SEPARATE),
    # equal for a reason the pool's normal form does not see
    ("distributivity inside sqrt",           A("sqrt", u1),                A("sqrt", u2),                EQUAL),
    ("max is commutative across a rewrite",  A("max", u1, v),              A("max", v, u2),              EQUAL),
    ("eq is symmetric across a rewrite",     S(C("eq", u1, w), a, b),      S(C("eq", w, u2), a, b),      EQUAL),
    ("a nested exp is one value twice",      T.mul(E(E(x)), E(E(x))),      E(T.mul(T.const(2.0), E(x))), EQUAL),
    ("exp(a)*exp(b) is exp(a+b)",            T.mul(E(x), E(y)),            E(T.add(x, y)),               EQUAL),
]

# (what the bug is, the line it replaces, the line it plants, is it a false-PASS bug)
MUTANTS = [
    ("ignores an atom's arguments",
     "            r = self._draw(mod, (t.fn, vals))",
     "            r = self._draw(mod, (t.fn,))", True),
    ("ignores an atom's function",
     "            r = self._draw(mod, (t.fn, vals))",
     "            r = self._draw(mod, vals)", True),
    ("ignores argument order everywhere",
     "            if t.fn in COMMUTATIVE: vals = tuple(sorted(vals))",
     "            vals = tuple(sorted(vals))", True),
    ("ignores a nested exp's argument",
     'r = self._draw(mod, ("exp", self._eval(t.args[0], mod, memo, True)))',
     'r = self._draw(mod, ("exp",))', True),
    ("keys commutative atoms on order (false separation)",
     "            if t.fn in COMMUTATIVE: vals = tuple(sorted(vals))",
     "            pass", False),
]


def run(equal):
    got, _ = equal([(p, q) for _, p, q, _ in CASES])
    return [(name, want, g) for (name, _, _, want), g in zip(CASES, got)]


if __name__ == "__main__":
    print("== cases ==")
    res = run(pit.equal)
    bad = 0
    for name, want, got in res:
        ok = got == want; bad += not ok
        print(f"  {'ok ' if ok else 'BAD'}  want={'equal' if want else 'separate':<8}  {name}")
    print(f"  -> {len(res) - bad}/{len(res)} cases")
    print(f"  every case behaves as designed: {'ok' if not bad else 'NO'}")

    print("\n== planted bugs: each must fail at least one case ==")
    src = inspect.getsource(pit)
    missed = 0
    for what, old, new, false_pass in MUTANTS:
        if src.count(old) != 1:
            print(f"  STALE  {what}: its anchor is not in pit.py any more"); missed += 1; continue
        ns = {"__name__": "pit_mutant"}
        exec(compile(src.replace(old, new), "pit_mutant", "exec"), ns)
        broken = [n for n, want, got in run(ns["equal"]) if got != want]
        missed += not broken
        kind = "false PASS" if false_pass else "lost PASS "
        print(f"  {'caught' if broken else 'MISSED'}  [{kind}] {what}: fails {len(broken)} case(s)"
              + (f", e.g. {broken[0]!r}" if broken else ""))
    print(f"  -> {len(MUTANTS) - missed}/{len(MUTANTS)} planted bugs caught")
    print(f"  every planted bug is caught: {'ok' if not missed else 'NO'}")

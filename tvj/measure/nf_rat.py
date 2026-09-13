"""How big Volta's canonical form of a corpus row's outputs is, counted without building it.

    python3 -m tvj.measure.nf_rat kb 61          (kb N | traces N; opens the row on the GPU)

Volta (willtunnels/volta, the OOPSLA 2026 artifact; `canon/ops.rs`) canonicalises
an expression to N/D with N and D polynomials whose monomials are
c * atoms * e^{poly}.  Its rules: `a + b` with DIFFERENT denominators becomes
(Na*Db + Nb*Da) / (Da*Db), with equal ones (Na+Nb)/D; `a * b` is NaNb/DaDb;
`a / b` is NaDb/DaNb; `exp(P)` is one term, its exponent carried and not
expanded; max, min, sqrt, log and abs are atoms.  Deciding a pair is then
N1*D2 = N2*D1.  This script counts the monomials each of those would have, by
dynamic programming over our term DAG with those rules, denominators tracked as
multisets of the divisor sub-terms so "same denominator" is decided
structurally.  Nothing is assumed to cancel, so the count is what the procedure
materialises unless something cancels; it does not depend on the machine.

What it shows on the rows Volta could not finish: the multiplicative depth is 1
and one output sums 4 fractions with different softmax denominators -- the 4
heads on row 97, and 4 softmax rows over different slices of one input on row
61 (a single-head module).  So the common denominator has 4^4 = 256 terms and
the equality check about 10^6 monomials per output on row 61, 10^11 on row 97.
The paper's argument that blowup "does not happen in practice" is about
multiplicative depth; this arrives through division, at depth 1.  Evaluation
at random points (measure/pit.py) never forms the fraction, and separates these
rows in milliseconds.
"""
import copy, json, sys
from collections import Counter
import torch
from tvj.core import terms as T
from tvj.measure import lanes
from tvj.decide import delegate as DEL
from tvj.front.capture import capture, Launch, Extern, symbolic_run, base_of, physical_offsets
from tvj.front.spec import STensor, symbolic_module
from tvj.front import torchtrace as TT
from tvj.judge.judge import prepare_scalars, first


def open_row(corpus, i):
    """Both sides' output terms for one corpus row: (candidate, spec terms, kernel terms)."""
    if corpus == "kb":
        from tvj.judge import kernelbook_run as R
        cand = R.build(json.load(open("data/kernelbook_400.json"))[i])
    else:
        from tvj.judge import traces_run as R
        rows = [x for x in json.load(open("data/triton_traces.json")) if x["source"] == "kernelbook"]
        cand, bucket = R.build(rows[i]); assert cand is not None, bucket
    T.reset(); DEL.reset()
    model, inputs = cand.model, cand.inputs
    scalar_syms = prepare_scalars(model); cand.push()
    with torch.no_grad(): first(model(*inputs))
    tr = TT.TorchTrace()
    with torch.no_grad(): out_all, calls = capture(lambda: cand.run(inputs), ns=cand.ns, trace=tr)
    out = first(out_all)
    roles = {base_of(x): f"in{j}" for j, x in enumerate(inputs) if torch.is_tensor(x)}
    roles.update({base_of(p): "p_" + n for n, p in cand.params.named_parameters()})
    roles.update({base_of(b): "b_" + n for n, b in cand.params.named_buffers()})
    ob = base_of(out); out_role = roles[ob] if ob in roles else roles.setdefault(ob, "out")
    evs = [Extern(e[1], e[2], e[3], roles, e[4] if len(e) > 4 else None) if e[0] == "extern"
           else Launch(*e, roles, scalar_syms) for e in calls]
    grid, _ = symbolic_run(evs)
    kterms = [grid.store.get((out_role, p)) for p in physical_offsets(out)]
    sm = symbolic_module(copy.deepcopy(model))
    sin = [STensor.input(f"in{j}", tuple(x.shape)) if torch.is_tensor(x) else x for j, x in enumerate(inputs)]
    spec = first(sm(*sin)); sf = (spec if isinstance(spec, STensor) else STensor(spec)).flat()
    return cand, sf, kterms


def rational(t, memo):
    """(numerator monomials, denominator monomials, denominator as a multiset of divisor uids)."""
    if t.uid in memo: return memo[t.uid]
    if isinstance(t, T.Add):
        n, d, key = 0, 1, Counter()
        for a in t.args:
            na, da, ka = rational(a, memo)
            if key == ka: n = n + na                       # same denominator: numerators add
            elif not key: n, d, key = na, da, Counter(ka)  # first fraction
            else: n = n * da + na * d; d = d * da; key = key + ka   # Volta: cross-multiply
        v = (n, d, key)
    elif isinstance(t, T.Mul):
        n, d, key = 1, 1, Counter()
        for a in t.args:
            na, da, ka = rational(a, memo); n *= na; d *= da; key = key + ka
        v = (n, d, key)
    elif isinstance(t, T.App) and t.fn == "div":
        na, da, ka = rational(t.args[0], memo); nb, db, kb = rational(t.args[1], memo)
        v = (na * db, da * nb, ka + Counter({t.args[1].uid: 1}) + Counter(kb))
    else:
        v = (1, 1, Counter())                              # sym, const, exp(P), max, sqrt, log ...
    memo[t.uid] = v; return v


def mult_depth(t, memo):
    """Nesting of products of sums: what the paper's blowup argument is about."""
    if t.uid in memo: return memo[t.uid]
    if isinstance(t, (T.Add, T.Mul, T.App)):
        v = max([mult_depth(a, memo) for a in t.args] or [0])
        if isinstance(t, T.Mul) and sum(isinstance(a, (T.Add, T.Mul)) for a in t.args) >= 2: v += 1
    else: v = 0
    memo[t.uid] = v; return v


def fmt(n): return "%.1e" % n if n < 1e300 else "10^%d" % (len(str(n)) - 1)


def main():
    corpus, i = sys.argv[1], int(sys.argv[2])
    cand, sf, kterms = open_row(corpus, i)
    pairs = [(s, k) for s, k in zip(sf, kterms) if k is not None and s is not None and s is not k]
    groups = list(lanes.group(pairs).values())
    print(f"== {corpus} row {i}: {cand.name}  outputs {len(sf)}  differing pairs {len(pairs)}  shapes {len(groups)}")
    mm, md = {}, {}
    for g in groups[:1]:
        s, k = pairs[g[0]]
        (n1, d1, k1), (n2, d2, k2) = rational(s, mm), rational(k, mm)
        print(f"   one shape of {len(g)} lanes, spec / kernel:")
        print(f"      DAG nodes {T.size(s):,} / {T.size(k):,}   multiplicative depth {mult_depth(s, md)} / {mult_depth(k, md)}   "
              f"distinct denominators {len(k1)} / {len(k2)}")
        print(f"      numerator {fmt(n1)} / {fmt(n2)}   denominator {fmt(d1)} / {fmt(d2)} monomials")
        print(f"      equality check N1*D2 vs N2*D1: {fmt(n1 * d2)} vs {fmt(n2 * d1)} monomials before any cancellation")


if __name__ == "__main__":
    main()

"""Decide one lane per SHAPE, not one per output element.

A tile kernel applies the same computation to every lane, so its output terms are
instances of a handful of shapes with different leaves: 1024 matmul outputs are
one shape, 512 attention outputs are two.  The term pool stores each shape once
per lane, and the value obligation hands every lane to the decision procedure as
its own pair.  Neither needs to.

For a PAIR of terms what matters is the shape of both sides together with which
leaves they share -- number the leaves by first appearance across both sides and
two pairs with the same joint shape are the same question up to renaming.  Volta
treats a leaf as an opaque variable, so their verdicts are the same verdict.

This is the measurement, not the change: group the pairs, decide one
representative per group, and check the propagated verdicts against deciding
every pair the way `judge.value_pass` does today.  Whether it pays is a number --
pairs sent to Volta before and after, and the seconds -- and whether it is
sound is a comparison, lane by lane.

    python3 -m tvj.measure.lanes [L]
"""
import sys, time
from tvj.core import terms as T


class Shapes:
    """Interns term shapes as small integers, shared across every pair.

    The first version built each shape as a nested tuple and used it as a dict
    key.  A tuple tree UNFOLDS the sharing a hash-consed DAG has: a 73-node DAG of
    the form x_{i+1} = x_i*x_i + c is a 2^24-leaf tree, and `tuplehash` and
    `tuple.__eq__` walk the tree -- in C, so SIGALRM's Python handler never gets a
    bytecode boundary to run at.  KernelBook row 116 spun for ten minutes past a
    150 s alarm on exactly that.  Interning bottom-up keeps every node's key a
    flat tuple of small ints: O(DAG) per pair, O(1) per hash and comparison."""

    def __init__(self):
        self.ids = {}                        # flat node key -> shape id

    def of(self, t, leaves, memo):
        """Shape id of `t`, with Sym leaves numbered by first appearance in
        `leaves`, which the caller shares across both sides of a pair."""
        r = memo.get(t.uid)
        if r is not None: return r
        if isinstance(t, T.Sym):
            key = ("L", leaves.setdefault(t.uid, len(leaves)))
        elif isinstance(t, T.Const):
            key = ("C", t.v)
        else:
            key = (type(t).__name__, getattr(t, "fn", None),
                   tuple(self.of(a, leaves, memo) for a in t.args))
        r = self.ids.get(key)
        if r is None:
            r = self.ids[key] = len(self.ids)
        memo[t.uid] = r
        return r


def shape(t, leaves, memo, shapes=None):
    """Kept for callers that want one term's shape id; see `Shapes.of`."""
    return (shapes or Shapes()).of(t, leaves, memo)


def group(pairs):
    """{joint shape: [pair indices]} -- one Volta call per key decides them all."""
    shapes, groups = Shapes(), {}
    for i, (a, b) in enumerate(pairs):
        leaves, memo = {}, {}
        key = (shapes.of(a, leaves, memo), shapes.of(b, leaves, memo))
        groups.setdefault(key, []).append(i)
    return groups


def decide(pairs, procedure):
    """Verdict per pair, deciding one representative per joint shape.
    Returns (verdicts, groups, seconds inside the procedure)."""
    groups = group(pairs)
    reps = [idx[0] for idx in groups.values()]
    t0 = time.time()
    res = procedure([pairs[i] for i in reps]) if reps else []
    secs = time.time() - t0
    out = [None] * len(pairs)
    for r, idx in zip(res, groups.values()):
        for i in idx: out[i] = r
    return out, groups, secs


if __name__ == "__main__":
    from tvj.core import ttir as P, sexec as X
    from tvj.decide import volta_bridge as V
    from tvj.fixtures import attn, kernels as K
    from tvj.checks.check import to_ttir

    def run(fn, sig, cst, grid, bufs, args):
        f = P.parse(to_ttir(fn, sig, cst))
        it = X.Interp(f, None, grid, bufs); it.argvals = args
        st = it.run_all().store
        return [st[k] for k in sorted(st)]

    def volta(pairs):
        res, st = V.equivalent(pairs)
        volta.stats = st
        return res

    def all_lanes(pairs):
        """Every lane to Volta, as value_pass does today -- or the cap it hit."""
        try:
            t0 = time.time(); res = volta(pairs)
            return res, time.time() - t0, volta.stats["peak_rss_mb"] / 1024
        except V.Unsupported as e:
            return None, None, str(e)[:40]

    L, D = (int(sys.argv[1]) if len(sys.argv) > 1 else 32), 16
    AS = {"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32", "o_ptr": "*fp32",
          "L": "i32", "D": "i32", "BM": "constexpr", "BD": "constexpr", "BN": "constexpr"}
    RS = {**{k: v for k, v in AS.items() if k != "BN"}, "BL": "constexpr"}
    abufs = {k: L * D for k in ("q_ptr", "k_ptr", "v_ptr", "o_ptr")}
    aargs = [X.Ptr(p, 0) for p in ("q_ptr", "k_ptr", "v_ptr", "o_ptr")] + [L, D]
    MS = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32", "K": "i32",
          "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"}
    M = N = Kd = 32
    mbufs = {"a_ptr": M * Kd, "b_ptr": Kd * N, "c_ptr": M * N}
    margs = [X.Ptr("a_ptr", 0), X.Ptr("b_ptr", 0), X.Ptr("c_ptr", 0), M, N, Kd]

    T.reset()
    S = {
        "ref":       run(attn.attn_ref,   RS, {"BM": 16, "BD": 16, "BL": L}, (L // 16,), abufs, aargs),
        "safe":      run(attn.attn_safe,  RS, {"BM": 16, "BD": 16, "BL": L}, (L // 16,), abufs, aargs),
        "flash":     run(attn.attn_flash, AS, {"BM": 16, "BD": 16, "BN": 16}, (L // 16,), abufs, aargs),
        "norescale": run(attn.attn_flash_norescale, AS, {"BM": 16, "BD": 16, "BN": 16}, (L // 16,), abufs, aargs),
        "mm":        run(K.mm_tiled,  MS, {"BM": 16, "BN": 16, "BK": 16}, (2, 2), mbufs, margs),
        "splitk":    run(K.mm_splitk, {**MS, "SPLIT": "constexpr"}, {"BM": 16, "BN": 16, "BK": 16, "SPLIT": 2},
                         (2, 2, 2), mbufs, margs),
        "swizzle":   run(K.mm_swizzle, {**MS, "GROUP": "constexpr"}, {"BM": 16, "BN": 16, "BK": 16, "GROUP": 2},
                         (4,), mbufs, margs),
        "bug_k":     run(K.bug_short_k, MS, {"BM": 16, "BN": 16, "BK": 16}, (2, 2), mbufs, margs),
    }

    CASES = [("mm", "splitk", True), ("mm", "swizzle", True), ("mm", "bug_k", False),
             ("ref", "safe", True), ("safe", "flash", True), ("ref", "flash", True),
             ("ref", "norescale", False)]
    print(f"L={L}   {'pair':<18} {'lanes':>5} {'groups':>6} {'every lane':>25} {'one per shape':>25}   agree")
    ok, tot_all, tot_rep, compared = True, 0.0, 0.0, 0
    for a, b, want in CASES:
        pairs = list(zip(S[a], S[b]))
        full, t_all, m_all = all_lanes(pairs)
        got, groups, t_rep = decide(pairs, volta); m_rep = volta.stats["peak_rss_mb"] / 1024
        n_rep = sum(x is True for x in got)
        ok &= (n_rep == len(pairs)) == want
        if full is None:
            left, agree = f"{'cap: ' + m_all:>25}", "(no full run)"
        else:
            same = all(x == y for x, y in zip(full, got)); ok &= same; compared += 1
            tot_all += t_all; tot_rep += t_rep
            left, agree = f"{sum(x is True for x in full):>5}/{len(pairs)} {t_all:>6.2f}s {m_all:>5.2f} GB", ("ok" if same else "DIFFER")
        print(f"        {a:>7} vs {b:<9} {len(pairs):>5} {len(groups):>6} {left} "
              f"{n_rep:>5}/{len(pairs)} {t_rep:>6.2f}s {m_rep:>5.2f} GB   {agree}")
    # The shape key must stay O(DAG) on a DAG whose tuple unfolding is exponential:
    # x_{i+1} = x_i*x_i + c is 3 nodes per level and 2^depth leaves unfolded.  The
    # nested-tuple version of `shape` spun in C on KernelBook row 116 past its alarm
    # -- and a C spin ignores SIGALRM, so this runs in a child under a wall-clock
    # kill, which is the only thing that stops the old code.
    import subprocess
    child = """
import time
from tvj.core import terms as T
from tvj.measure import lanes
x = T.sym("dag0", 0); y = T.sym("dag1", 0); z = T.sym("dag2", 0)
for i in range(60):
    x = T.add(T.mul(x, x), T.const(float(i + 1))); y = T.add(T.mul(y, y), T.const(float(i + 1)))
    z = T.add(T.mul(z, z), T.const(float(i + 2)))
t0 = time.time(); g = lanes.group([(x, y), (x, y), (x, z)])
print(T.size(x), len(g), round((time.time() - t0) * 1000, 1))
"""
    try:
        out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=20).stdout.split()
        nodes, ngroups, ms = int(out[0]), int(out[1]), float(out[2])
        ok &= ngroups == 2
        print(f"\n  shared DAG, depth 60 ({nodes} nodes, 2^60 unfolded): grouped in {ms} ms, "
              f"{ngroups} groups (want 2)")
    except subprocess.TimeoutExpired:
        ok = False
        print("\n  shared DAG, depth 60: shape key did not finish in 20 s -- NOT linear")
    print(f"\n  Volta over the {compared} pairs both paths could run: {tot_all:.2f}s deciding every lane, "
          f"{tot_rep:.2f}s deciding one per shape ({tot_all / max(tot_rep, 1e-9):.0f}x).  "
          f"Every propagated verdict matches the full run, controls included: {'ok' if ok else 'NO'}")

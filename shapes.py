"""Shape selection as part of the checker.

bug_swizzle is benign when num_m % GROUP == 0 and out-of-bounds otherwise, so a
bounded checker's verdict is only as good as the shapes it was handed.  Rather
than hand-pick them, state each kernel's divisibility *contract* and then cover
every residue class it admits, at minimum symbolic-execution cost.

A contract has two parts:
  preconditions -- shapes the kernel does not claim to handle (an unmasked
                   kernel does not claim to handle a ragged tile)
  relations     -- the (value, modulus) pairs whose residues must all be seen
Both are spec, and having to write them down is the point.
"""
import itertools, random

class Contract:
    def __init__(self, cst, pre, rels, lo=16, hi=80):
        self.cst, self.pre, self.rels, self.lo, self.hi = cst, pre, rels, lo, hi

    def targets(self, shape):
        """(relation name, class key) pairs this shape covers.

        A key is any hashable abstraction of the shape, not just a residue.
        Per-parameter residues are not enough: bug_swizzle needs
        num_m % GROUP != 0 AND num_n >= 2 simultaneously, and a plan built from
        one-dimensional residue classes can cover both residues separately with
        shapes that are each individually benign.
        """
        return {(name, f(shape, self.cst)) for name, f in self.rels}

    def all_targets(self, pool):
        t = set()
        for s in pool: t |= self.targets(s)
        return t

    def pool(self, n=4000, seed=0):
        rng = random.Random(seed)
        out, seen = [], set()
        rng_vals = range(self.lo, self.hi + 1)
        # structured sweep first, then random fill
        cand = [(m, m, m) for m in rng_vals]
        cand += [(rng.choice(rng_vals), rng.choice(rng_vals), rng.choice(rng_vals))
                 for _ in range(n)]
        for s in cand:
            if s in seen: continue
            seen.add(s)
            if all(p(s, self.cst) for p in self.pre): out.append(s)
        return out

def plan(contract, budget=12):
    """Greedy weighted set cover: most new residue classes per unit of work."""
    pool = contract.pool()
    need = contract.all_targets(pool)
    chosen, covered = [], set()
    while need - covered and len(chosen) < budget:
        best, best_score = None, 0.0
        for s in pool:
            new = len(contract.targets(s) - covered)
            if not new: continue
            cost = s[0] * s[1] * s[2]          # symbolic execution is Theta(M*N*K)
            score = new / cost
            if score > best_score: best, best_score = s, score
        if best is None: break
        chosen.append(best); covered |= contract.targets(best)
    return chosen, need, covered

def cdiv(a, b): return -(-a // b)

# ---- contracts for the matmul family --------------------------------------
def divisible(cst):
    return [lambda s, c: s[0] % c["BM"] == 0,
            lambda s, c: s[1] % c["BN"] == 0,
            lambda s, c: s[2] % c["BK"] == 0]

def swizzle_contract(cst):
    nm = lambda s, c: cdiv(s[0], c["BM"])
    nn = lambda s, c: cdiv(s[1], c["BN"])
    return Contract(cst, divisible(cst), [
        ("num_m % GROUP",          lambda s, c: nm(s, c) % c["GROUP"]),
        ("num_n % GROUP",          lambda s, c: nn(s, c) % c["GROUP"]),
        # joint: a partial last group only misbehaves when there is more than
        # one column of tiles to interleave with
        ("num_m%G x min(num_n,3)", lambda s, c: (nm(s, c) % c["GROUP"], min(nn(s, c), 3))),
        ("groups x remainder",     lambda s, c: (min(nm(s, c) // c["GROUP"], 2),
                                                 nm(s, c) % c["GROUP"])),
    ])

def ragged_contract(cst):
    return Contract(cst, [], [
        ("M % BM", lambda s, c: s[0] % c["BM"]),
        ("N % BN", lambda s, c: s[1] % c["BN"]),
        ("K % BK", lambda s, c: s[2] % c["BK"]),
        ("ragged triple", lambda s, c: (s[0] % c["BM"] != 0, s[1] % c["BN"] != 0,
                                        s[2] % c["BK"] != 0)),
    ])

def splitk_contract(cst):
    # precondition found by the scheduler: with K/SPLIT < BK each program loads a
    # full K-tile, double-counting the overlap and running off the end of K.
    pre = divisible(cst) + [lambda s, c: s[2] % c["SPLIT"] == 0,
                            lambda s, c: (s[2] // c["SPLIT"]) % c["BK"] == 0]
    return Contract(cst, pre, [
        ("K/(SPLIT*BK)",  lambda s, c: min(s[2] // (c["SPLIT"] * c["BK"]), 3)),
        ("num_m % SPLIT", lambda s, c: cdiv(s[0], c["BM"]) % c["SPLIT"]),
        ("num_n % SPLIT", lambda s, c: cdiv(s[1], c["BN"]) % c["SPLIT"]),
    ], hi=96)

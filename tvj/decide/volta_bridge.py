"""Python side of the Volta bridge: serialise term DAGs, call the Rust binary,
return Volta's verdicts.  See bridge/src/main.rs."""
import json, os, subprocess
from tvj.core import terms as T

class Unsupported(Exception):
    """The term uses an operation outside Volta's theory; verdict must be UNKNOWN, not PASS/FAIL."""

from tvj.root import at
BIN = at("bridge", "target", "release", "volta_bridge")

def _const_node(v):
    """Decision literal.decimal: a float literal denotes the decimal the programmer
    wrote (shortest round-trip repr), not the binary it became.  1e-5 is 1/100000,
    not 5902958103587057/2^79 -- which is what overflowed Volta's i128 coefficients."""
    import math, numpy as np
    from fractions import Fraction
    if math.isfinite(v):
        fr = Fraction(str(np.float32(v)))       # shortest decimal identifying the fp32 value
        if abs(fr.numerator) < 2**62 and fr.denominator < 2**62:
            return {"op": "rat", "num": fr.numerator, "den": fr.denominator}
    return {"op": "const", "v": v}

# An n-ary Add/Mul is flattened into a binary chain for the arena, so the node
# count runs several times the DAG size and each node is a Python dict.  A
# VGG-sized pipeline reached 8+ GB here.  Cap it; a pair too big to serialise is
# reported as unsupported, not crashed on.
NODE_CAP = int(os.environ.get("TVJ_NODE_CAP", 2_000_000))

def serialize(roots):
    """Terms -> (node list in children-first order, root indices).  n-ary Add/Mul
    become left-nested binary chains; Volta reassociates anyway."""
    nodes, memo = [], {}
    def go(t):
        if t.key in memo: return memo[t.key]
        if isinstance(t, T.Sym):
            nodes.append({"op": "sym", "buf": t.buf, "idx": t.idx})
        elif isinstance(t, T.Const):
            nodes.append(_const_node(t.v))
        elif isinstance(t, (T.Add, T.Mul)):
            op = "add" if isinstance(t, T.Add) else "mul"
            ids = [go(a) for a in t.args]
            cur = ids[0]
            for i in ids[1:]:
                nodes.append({"op": op, "a": cur, "b": i}); cur = len(nodes) - 1
            memo[t.key] = cur
            return cur
        elif isinstance(t, T.App):
            ids = [go(a) for a in t.args]
            if t.fn == "select":
                nodes.append({"op": "select", "c": ids[0], "t": ids[1], "f": ids[2]})
            elif t.fn.startswith("cmp:"):
                nodes.append({"op": "cmp", "kind": t.fn[4:], "a": ids[0], "b": ids[1]})
            elif t.fn in ("true", "false"):
                nodes.append({"op": "const", "v": 1.0 if t.fn == "true" else 0.0})
            elif t.fn in ("exp", "sqrt", "log", "abs"):
                nodes.append({"op": t.fn, "a": ids[0]})
            elif t.fn == "tanh":
                # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1): exact over the reals and
                # inside Volta's theory, which has exp but not tanh.
                return go(T.div(T.sub(T.app("exp", T.mul(T.const(2.0), t.args[0])), T.ONE),
                                T.add(T.app("exp", T.mul(T.const(2.0), t.args[0])), T.ONE)))
            elif t.fn in ("max", "min", "div"):
                cur = ids[0]
                for i in ids[1:]:
                    nodes.append({"op": t.fn, "a": cur, "b": i}); cur = len(nodes) - 1
                memo[t.key] = cur
                return cur
            else:
                raise Unsupported(f"Volta has no interpretation for `{t.fn}`")
        else:
            raise TypeError(t)
        if len(nodes) > NODE_CAP:
            raise Unsupported(f"term too large to hand to Volta ({len(nodes):,} nodes > {NODE_CAP:,})")
        memo[t.key] = len(nodes) - 1
        return memo[t.key]
    return nodes, [go(r) for r in roots]

def equivalent(pairs, budget=None):
    """pairs: list of (term_a, term_b).  Returns (list of True/False/str-error, stats)."""
    na, ra = serialize([a for a, _ in pairs])
    nb, rb = serialize([b for _, b in pairs])
    payload = {"nodes_a": na, "nodes_b": nb, "pairs": list(zip(ra, rb)), "budget": budget}
    env = dict(os.environ); env.setdefault("VOLTA_MEM_GB", "4")
    p = subprocess.run([BIN], input=json.dumps(payload), capture_output=True, text=True, env=env)
    if p.returncode != 0:
        tail = p.stderr[-400:]
        if "memory allocation" in tail or "Cannot allocate" in tail or p.returncode == -9:
            raise Unsupported(f"Volta exceeded its {env['VOLTA_MEM_GB']} GB cap on this pair")
        raise RuntimeError(f"volta_bridge failed: {tail}")
    out = json.loads(p.stdout)
    res = [True if r == "true" else False if r == "false" else r for r in out["results"]]
    return res, {k: out[k] for k in ("ops_used", "interned_terms", "secs", "peak_rss_mb")}

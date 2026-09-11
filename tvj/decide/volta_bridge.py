"""Python side of the Volta bridge: serialise term DAGs, call the Rust binary,
return Volta's verdicts.  See bridge/src/main.rs."""
import json, os, struct, subprocess
from tvj.core import terms as T

class Unsupported(Exception):
    """The term uses an operation outside Volta's theory; verdict must be UNKNOWN, not PASS/FAIL."""

from tvj.root import at
BIN = at("bridge", "target", "release", "volta_bridge")

# --- wire format ---------------------------------------------------------------
# One node is `op:u8` followed by its operands as u32 indices into the nodes
# already emitted, so the dominant case -- a binary Add or Mul -- is nine bytes.
# It replaced a list of Python dicts plus a JSON string, measured at 245 B and
# 80 B per node: NODE_CAP's worth of dicts alone came to half a gigabyte before
# the child process saw any of it, and the README's Cost section has the Python
# side passing Volta as the bottleneck at L=256 because of it.  Nothing about the
# terms changed -- this is the same node stream in a smaller alphabet.
MAGIC, WIRE_VERSION = b"TVJB", 1

OP = {"sym": 0, "const": 1, "rat": 2, "add": 3, "mul": 4, "div": 5, "max": 6,
      "min": 7, "and": 8, "or": 9, "exp": 10, "sqrt": 11, "log": 12, "abs": 13,
      "not": 14, "select": 15,
      "lt": 16, "le": 17, "gt": 18, "ge": 19, "eq": 20, "ne": 21}

_U32   = struct.Struct("<I")
_BIN   = struct.Struct("<BII")    # op, a, b
_UN    = struct.Struct("<BI")     # op, a
_SEL   = struct.Struct("<BIII")   # op, cond, then, else
_SYM   = struct.Struct("<BIQ")    # op, string id, element index
_CONST = struct.Struct("<Bd")     # op, value
_RAT   = struct.Struct("<Bqq")    # op, numerator, denominator


def _const_bytes(v):
    """Decision literal.decimal: a float literal denotes the decimal the programmer
    wrote (shortest round-trip repr), not the binary it became.  1e-5 is 1/100000,
    not 5902958103587057/2^79 -- which is what overflowed Volta's i128 coefficients."""
    import math, numpy as np
    from fractions import Fraction
    if not math.isfinite(v):
        # Volta's theory is the reals: no NaN, no infinity.  Under the old JSON
        # encoding `allow_nan=False` was the second net behind this check; a binary
        # encoding packs a NaN happily, so the check has to stand on its own.  A
        # gather's out-of-range default IS such a constant, which makes this
        # reachable from any indirect read.
        raise Unsupported(f"non-finite constant ({v}) is outside Volta's theory")
    fr = Fraction(str(np.float32(v)))           # shortest decimal identifying the fp32 value
    if abs(fr.numerator) < 2**62 and fr.denominator < 2**62:
        return _RAT.pack(OP["rat"], fr.numerator, fr.denominator)
    return _CONST.pack(OP["const"], v)


# An n-ary Add/Mul is flattened into a binary chain for the arena, so the node
# count runs several times the DAG size.  Cap it; a pair too big to serialise is
# reported as unsupported, not crashed on.
NODE_CAP = int(os.environ.get("TVJ_NODE_CAP", 2_000_000))
# Operand indices are u32 on the wire.  Raising the cap past that would make
# `struct.error` the first sign of it, from inside a term walk.
assert NODE_CAP < 2**32, f"TVJ_NODE_CAP={NODE_CAP:,} exceeds the wire's u32 node index"


def serialize(roots):
    """Terms -> (string table, packed node stream, root indices, node count).

    Children-first, so a node's operands are always indices into what precedes
    it and the other side can decode and build in one pass.  n-ary Add/Mul become
    left-nested binary chains; Volta reassociates anyway."""
    # A bytearray rather than a list of pieces: at NODE_CAP the pieces alone are
    # tens of megabytes of per-object header that live until the final join, which
    # is most of what the stream was meant to stop costing.
    out, memo, strs, strtab, n = bytearray(), {}, {}, [], 0

    def emit(packed):
        # Every node goes through here, including the ones inside a chain.  The
        # dict version checked the cap only where `go` fell through to the bottom,
        # so a single wide Add could run past it unchecked.
        nonlocal n
        out.extend(packed); n += 1     # extend, not `+=`: `+=` would rebind a local
        if n > NODE_CAP:
            raise Unsupported(f"term too large to hand to Volta ({n:,} nodes > {NODE_CAP:,})")
        return n - 1

    def string_id(b):
        i = strs.get(b)
        if i is None: i = strs[b] = len(strtab); strtab.append(b)
        return i

    def chain(op, ids):
        cur = ids[0]
        for i in ids[1:]: cur = emit(_BIN.pack(OP[op], cur, i))
        return cur

    def go(t):
        if t.key in memo: return memo[t.key]
        if isinstance(t, T.Sym):
            if not 0 <= t.idx < 2**64:
                raise Unsupported(f"element index {t.idx} does not fit the wire format")
            r = emit(_SYM.pack(OP["sym"], string_id(t.buf), t.idx))
        elif isinstance(t, T.Const):
            r = emit(_const_bytes(t.v))
        elif isinstance(t, (T.Add, T.Mul)):
            r = chain("add" if isinstance(t, T.Add) else "mul", [go(a) for a in t.args])
        elif isinstance(t, T.App):
            ids = [go(a) for a in t.args]
            if t.fn == "select":
                r = emit(_SEL.pack(OP["select"], ids[0], ids[1], ids[2]))
            elif t.fn.startswith("cmp:"):
                k = t.fn[4:]
                if k not in OP: raise Unsupported(f"Volta has no interpretation for `{t.fn}`")
                r = emit(_BIN.pack(OP[k], ids[0], ids[1]))
            elif t.fn in ("true", "false"):
                r = emit(_const_bytes(1.0 if t.fn == "true" else 0.0))
            elif t.fn in ("exp", "sqrt", "log", "abs"):
                r = emit(_UN.pack(OP[t.fn], ids[0]))
            elif t.fn == "tanh":
                # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1): exact over the reals and
                # inside Volta's theory, which has exp but not tanh.
                return go(T.div(T.sub(T.app("exp", T.mul(T.const(2.0), t.args[0])), T.ONE),
                                T.add(T.app("exp", T.mul(T.const(2.0), t.args[0])), T.ONE)))
            elif t.fn in ("max", "min", "div"):
                r = chain(t.fn, ids)
            else:
                raise Unsupported(f"Volta has no interpretation for `{t.fn}`")
        else:
            raise TypeError(t)
        memo[t.key] = r
        return r

    rs = [go(r) for r in roots]
    return strtab, out, rs, n


def _side(strtab, nodes, n):
    parts = [_U32.pack(len(strtab))]
    for b in strtab:
        e = b.encode("utf-8"); parts += [_U32.pack(len(e)), e]
    parts += [_U32.pack(n), _U32.pack(len(nodes)), nodes]
    return parts


def equivalent(pairs, budget=None):
    """pairs: list of (term_a, term_b).  Returns (list of True/False/str-error, stats)."""
    sa, na, ra, ca = serialize([a for a, _ in pairs])
    sb, nb, rb, cb = serialize([b for _, b in pairs])
    flat = [i for p_ in zip(ra, rb) for i in p_]
    body = b"".join([MAGIC, struct.pack("<BBQ", WIRE_VERSION, 1 if budget else 0, budget or 0),
                     *_side(sa, na, ca), *_side(sb, nb, cb),
                     _U32.pack(len(ra)), struct.pack(f"<{len(flat)}I", *flat)])
    env = dict(os.environ); env.setdefault("VOLTA_MEM_GB", "4")
    p = subprocess.run([BIN], input=body, capture_output=True, env=env)
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace")[-400:]
        if "memory allocation" in tail or "Cannot allocate" in tail or p.returncode == -9:
            raise Unsupported(f"Volta exceeded its {env['VOLTA_MEM_GB']} GB cap on this pair")
        if "bad input json" in tail or "not a tvj bridge stream" in tail or "wire version" in tail:
            # The binary predates the binary wire format (or postdates this checkout).
            # Worth naming: otherwise the first sign is a JSON parse error from a
            # process that is no longer sent any JSON.
            raise RuntimeError("volta_bridge is built from a different wire version than this "
                               "checkout speaks.\n  Rebuild it: `cd bridge && cargo build "
                               f"--release` (or ./setup.sh).\n  It said: {tail}")
        raise RuntimeError(f"volta_bridge failed: {tail}")
    out = json.loads(p.stdout.decode("utf-8"))
    res = [True if r == "true" else False if r == "false" else r for r in out["results"]]
    return res, {k: out[k] for k in ("ops_used", "interned_terms", "secs", "peak_rss_mb")}

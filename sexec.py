"""Symbolic execution of TTIR at the *tile* level, over a whole grid.

Triton's abstraction hides threads: one program instance is a sequential
program over tiles.  So the hard part of CUDA-level equivalence checking
(no thread correspondence between reference and optimised kernel, barriers,
per-thread races) does not arise here.  What remains is the grid quantifier:
prove that the union of stores made by ALL program instances equals the
reference tensor.  With shapes fixed the grid is finite, so we enumerate it.

Integers are concrete (fixed shapes => every address, mask and loop bound is a
number), which gives out-of-bounds and write-conflict checking for free.
Tensor elements are symbolic reals (see terms.py).
"""
import re
from dataclasses import dataclass
import terms as T
import ttir as P
import semantics as S

class Ptr:
    __slots__ = ("buf", "off")
    def __init__(self, buf, off): self.buf, self.off = buf, off
    def __repr__(self): return f"{self.buf}+{self.off}"

@dataclass
class Tile:
    shape: tuple
    data: list          # row-major flat
    def __post_init__(self):
        n = 1
        for d in self.shape: n *= d
        assert len(self.data) == n, (self.shape, len(self.data))

def scalar(v): return Tile((), [v])

def numel(shape):
    n = 1
    for d in shape: n *= d
    return n

def strides(shape):
    st, acc = [0]*len(shape), 1
    for i in range(len(shape)-1, -1, -1):
        st[i] = acc; acc *= shape[i]
    return st

def bcast(t: Tile, shape: tuple) -> Tile:
    """MLIR tt.broadcast: same rank, src dims are 1 or equal."""
    if t.shape == shape: return t
    if t.shape == (): return Tile(shape, t.data * numel(shape))
    sst, dst_ = strides(t.shape), strides(shape)
    out = []
    for lin in range(numel(shape)):
        src = 0
        for d in range(len(shape)):
            i = (lin // dst_[d]) % shape[d]
            src += (0 if t.shape[d] == 1 else i) * sst[d]
        out.append(t.data[src])
    return Tile(shape, out)

# decision dot.precision: precision is a directed relation, tracked as a second
# denotation next to the real one.  Higher rank = more precise.
RANK = {"exact": 4, "ieee": 3, "f32": 3, "tf32x3": 2, "tf32": 1, "f16": 1, "bf16": 0, "fp8": -1}
# Several spellings share a rank (ieee/f32 = 3, tf32/f16 = 1), so inverting RANK
# loses information: a tf32 downgrade would print as "-> f16". Name each rank.
RANK_NAME = {4: "exact", 3: "f32/ieee", 2: "tf32x3", 1: "tf32/f16", 0: "bf16", -1: "fp8"}
LN2, LN2_VAL, LOG2E_VAL = T.sym("ln2", 0), 0.6931471805599453, 1.4426950408889634

_NEG = {"ogt": "ole", "ole": "ogt", "oge": "olt", "olt": "oge", "oeq": "one", "one": "oeq",
        "ugt": "ule", "ule": "ugt", "uge": "ult", "ult": "uge", "ueq": "une", "une": "ueq"}

class _PredTerm:
    """A boolean already reduced to a term (from combining two comparisons)."""
    __slots__ = ("t",)
    def __init__(self, t): self.t = t

class Pred:
    """A float comparison on symbolic values: not a bool, but select() may
    still resolve it structurally (select(a>b, a, b) is max(a, b))."""
    __slots__ = ("kind", "a", "b")
    def __init__(self, kind, a, b): self.kind, self.a, self.b = kind, a, b
    def negate(self): return Pred(_NEG[self.kind], self.a, self.b)

class Unsupported(Exception):
    """Kernel uses something outside the modelled fragment: verdict UNKNOWN."""
TOP = RANK["exact"]

class RealDomain:
    """Tensor elements are symbolic reals -> refinement checking.

    `prec` maps a term's key to the weakest precision any dot on the path to it
    was permitted to use.  Approximation: keyed by term identity, so two flows
    that build the same normal form under different precisions share an entry
    (min is taken).
    """
    name = "real"
    def __init__(self): self.prec = {}
    def p(self, t):            return self.prec.get(t.key, TOP)
    def _tag(self, out, *ins):
        q = min([self.p(i) for i in ins] + [TOP])
        self.prec[out.key] = min(self.prec.get(out.key, TOP), q)
        return out
    def eq(self, a, b):        return a is b
    def const(self, v):        return T.const(v)
    def read(self, buf, off):  return T.sym(buf, off)
    def add(self, a, b):       return self._tag(T.add(a, b), a, b)
    def sub(self, a, b):       return self._tag(T.sub(a, b), a, b)
    def mul(self, a, b):       return self._tag(T.mul(a, b), a, b)
    def div(self, a, b):       return self._tag(T.div(a, b), a, b)
    def exp(self, a):          return self._tag(T.app("exp", a), a)
    def max(self, a, b):       return self._tag(T.app("max", a, b), a, b)
    def min(self, a, b):       return self._tag(T.app("min", a, b), a, b)
    def cast(self, t, elem):
        """Narrowing float casts keep the real value and lower the precision tag."""
        r = RANK.get(elem)
        if r is not None: self.prec[t.key] = min(self.prec.get(t.key, TOP), r)
        return t
    def dot(self, prods, acc, prec="ieee"):
        out = T.add(acc, *prods)
        q = min([RANK.get(prec, TOP), self.p(acc)] + [self.p(x) for x in prods])
        self.prec[out.key] = min(self.prec.get(out.key, TOP), q)
        return out

class ConcreteDomain:
    """Tensor elements are float32 -> differential testing against Triton."""
    name = "concrete"
    def __init__(self, inputs): self.inputs = inputs   # buf -> sequence of floats
    def eq(self, a, b):        return a == b
    def cast(self, t, elem):   return t
    def const(self, v):        return _f32(v)
    def read(self, buf, off):  return _f32(self.inputs[buf][off])
    def add(self, a, b):       return _f32(a + b)
    def sub(self, a, b):       return _f32(a - b)
    def mul(self, a, b):       return _f32(a * b)
    def div(self, a, b):       return _f32(a / b) if b != 0 else float("nan")
    def exp(self, a):
        import math; return _f32(math.exp(a))
    def max(self, a, b):       return a if a >= b else b
    def min(self, a, b):       return a if a <= b else b
    def dot(self, prods, acc, prec="ieee"):
        # decision dot.accum-order: left fold over k, starting from acc
        # (prec is accepted and ignored: the concrete domain computes at f32, which
        #  is conformant -- more precise than any permission)
        r = acc
        for p in prods: r = _f32(r + p)
        return r

_CMPF_TERM = {"olt": "lt", "ult": "lt", "ole": "le", "ule": "le", "ogt": "gt", "ugt": "gt",
              "oge": "ge", "uge": "ge", "oeq": "eq", "ueq": "eq", "one": "ne", "une": "ne"}

def _pred_term(p):
    """A symbolic float comparison, as a term."""
    if isinstance(p, _PredTerm): return p.t
    k = _CMPF_TERM.get(p.kind)
    if k is None: raise Unsupported(f"comparison `{p.kind}` has no term form")
    return T.cmp(k, p.a, p.b)

def _cmpf(kind, u, v):
    if u is v:
        if kind in ("une", "one", "ult", "olt", "ugt", "ogt"): return False   # x != x, x < x, x > x
        if kind in ("oeq", "ueq", "ule", "ole", "uge", "oge"): return True
    if isinstance(u, T.Const) and isinstance(v, T.Const):
        f = {"oeq": u.v == v.v, "une": u.v != v.v, "olt": u.v < v.v, "ole": u.v <= v.v,
             "ogt": u.v > v.v, "oge": u.v >= v.v}
        if kind in f: return f[kind]
    return Pred(kind, u, v)

_ORI = {frozenset({"ogt", "oeq"}): "oge", frozenset({"olt", "oeq"}): "ole", frozenset({"ogt", "olt"}): "one",
        frozenset({"ugt", "ueq"}): "uge", frozenset({"ult", "ueq"}): "ule", frozenset({"oge", "ole"}): None}
_FLIP = {"ogt": "olt", "olt": "ogt", "oge": "ole", "ole": "oge", "oeq": "oeq", "one": "one",
         "ugt": "ult", "ult": "ugt", "uge": "ule", "ule": "uge", "ueq": "ueq", "une": "une"}

def _boolop(n, u, v):
    """andi/ori/xori where at least one side is a symbolic predicate."""
    p, q = (u, v) if isinstance(u, Pred) else (v, u)      # p is a Pred
    if isinstance(q, Pred) and n == "arith.ori":
        # (a > b) | (a == b)  ->  a >= b   (triton_helpers.maximum in some versions)
        qk = q.kind if (q.a is p.a and q.b is p.b) else (_FLIP[q.kind] if (q.a is p.b and q.b is p.a) else None)
        if qk is not None:
            k = _ORI.get(frozenset({p.kind, qk}))
            if k: return Pred(k, p.a, p.b)
            if p.kind == qk: return p
    if isinstance(q, Pred):
        if n in ("arith.andi", "arith.ori"):
            return _PredTerm(T.app("and" if n == "arith.andi" else "or", _pred_term(p), _pred_term(q)))
        raise Unsupported(f"{n} of two symbolic comparisons")
    q = bool(q)
    if n == "arith.ori":  return True if q else p
    if n == "arith.andi": return p if q else False
    if n == "arith.xori": return p.negate() if q else p
    raise Unsupported(f"{n} on a symbolic comparison")

def _f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]

class OOB(Exception): pass
class Conflict(Exception): pass

class Grid:
    """Accumulates the effect of every program instance in the grid."""
    def __init__(self, bufsize):
        self.bufsize = bufsize          # name -> #elements
        self.store = {}                 # (buf, off) -> term
        self.kind = {}                  # (buf, off) -> 'store' | 'atomic'
        self.writer = {}                # (buf, off) -> pid that wrote it
        self.errors = []
        self.benign = 0                 # same-value stores from different programs

    def write(self, buf, off, val, pid, atomic=False, dom=None):
        if not (0 <= off < self.bufsize[buf]):
            self.errors.append(("OOB-write", buf, off, pid)); return
        key = (buf, off)
        if key in self.store:
            if atomic and self.kind[key] == "atomic" and self.writer.get(key, ("earlier",))[0] == "current":
                self.store[key] = dom.add(self.store[key], val)  # decision atomic.order
                return
            if self.writer.get(key, ("earlier",))[0] != "current":
                self.store[key] = val; self.kind[key] = "atomic" if atomic else "store"
                self.writer[key] = ("current", pid); return          # overwrite an earlier launch's value
            if not dom.eq(self.store[key], val):
                self.errors.append(("write-conflict", buf, off, self.writer[key][1], pid))
            elif self.writer[key][1] != pid:
                self.benign += 1        # decision store.benign-race
            return
        self.store[key] = val
        self.kind[key] = "atomic" if atomic else "store"
        self.writer[key] = ("current", pid)

    def read(self, buf, off, pid, dom):
        if not (0 <= off < self.bufsize[buf]):
            self.errors.append(("OOB-read", buf, off, pid))     # decision load.oob-unmasked
            return dom.const(float("nan"))
        key = (buf, off)
        if key in self.store:
            w = self.writer.get(key, ("earlier",))
            if w[0] != "current" or w[1] == pid:              # earlier launch, or this program's own store
                return self.store[key]                        # decision memory.raw-same-program
            self.errors.append(("read-write-race", buf, off, w[1], pid))   # another program's store, same launch
        return dom.read(buf, off)

_CST_DENSE = re.compile(r"arith\.constant\s+dense<([^>]*)>")
# `arith.constant true` / `false` carry no type suffix, so the trailing `:` is optional
_CST_SCALAR = re.compile(r"arith\.constant\s+([-\w.+eE]+)\s*(?::|$)")
_CMP = re.compile(r"arith\.cmp[if]\s+(\w+)")
_PID = re.compile(r"tt\.get_program_id\s+(\w+)")
_FOR = re.compile(r"scf\.for\s+(%\w+)\s*=\s*(%\w+)\s+to\s+(%\w+)\s+step\s+(%\w+)"
                  r"(?:\s+iter_args\((.*?)\)\s*->)?")
_ATOMIC = re.compile(r"tt\.atomic_rmw\s+(\w+)")
_PREC = re.compile(r"inputPrecision\s*=\s*(\w+)")

def _lit(s, ty, dom):
    s = s.strip()
    if s == "true":  return True
    if s == "false": return False
    if s.startswith("0x"):                       # hex-encoded float, e.g. -inf = 0xFF800000
        import struct
        el = P.elem_of(ty)
        fmt = {"f32": ("I", "f"), "f64": ("Q", "d"), "f16": ("H", "e"), "bf16": None}[el]
        if fmt is None:
            v = struct.unpack("f", struct.pack("I", int(s, 16) << 16))[0]
        else:
            v = struct.unpack(fmt[1], struct.pack(fmt[0], int(s, 16)))[0]
        return _fconst(v, dom)
    if "f" in ty or "." in s or "e" in s.lower(): return _fconst(float(s), dom)
    return int(s)

def _fconst(v, dom):
    if dom.name == "real":
        if abs(v - LOG2E_VAL) <= 1e-7 * LOG2E_VAL: return T.div(T.ONE, LN2)
        if abs(v - LN2_VAL)   <= 1e-7 * LN2_VAL:   return LN2
    return dom.const(v)

class Interp:
    def __init__(self, func, argvals, grid_shape, bufsize, domain=None):
        self.f, self.argvals = func, argvals
        self.dom = domain or RealDomain()
        self.grid_shape, self.g = grid_shape, Grid(bufsize)
        self.covered = set()   # ops we actually executed (coverage report)

    def run_all(self):
        # anything already in memory came from an earlier launch; mark it so
        for k in list(self.g.writer): self.g.writer[k] = ("earlier",)
        gx, gy, gz = (list(self.grid_shape) + [1, 1, 1])[:3]
        for z in range(gz):
            for y in range(gy):
                for x in range(gx):
                    self.pid = (x, y, z)
                    env = {}
                    for name, val in zip(self.f.args, self.argvals):
                        env["%" + name] = scalar(val)
                    self.exec_block(self.f.body, env)
        return self.g

    def exec_block(self, ops, env):
        for op in ops:
            self.covered.add(op.name)
            self.exec_op(op, env)

    def get(self, env, name): return env[name]

    def _select(self, c, x, y):
        if isinstance(c, _PredTerm): return T.select(c.t, x, y)
        if not isinstance(c, Pred): return x if c else y
        if self.dom.name != "real":
            raise Unsupported("data-dependent select under the concrete domain")
        gt = c.kind in ("ogt", "oge", "ugt", "uge")
        lt = c.kind in ("olt", "ole", "ult", "ule")
        same, swapped = (x is c.a and y is c.b), (x is c.b and y is c.a)
        if gt and same or lt and swapped: return self.dom.max(c.a, c.b)
        if lt and same or gt and swapped: return self.dom.min(c.a, c.b)
        # select(x > 0, A, B) with {A, B} = {x, k*x}: max if (A is x) == (k <= 1) else min
        if isinstance(c.b, T.Const) and c.b.v == 0.0 and (gt or lt):
            A, B = (x, y) if gt else (y, x)
            xx = c.a
            def coef(t):
                if isinstance(t, T.Mul) and len(t.args) == 2:
                    cs = [z for z in t.args if isinstance(z, T.Const)]
                    if cs and any(z is xx for z in t.args): return cs[0].v
                return None
            kA, kB = coef(A), coef(B)
            if A is xx and kB is not None and kB >= 0: return self.dom.max(A, B) if kB <= 1 else self.dom.min(A, B)
            if B is xx and kA is not None and kA >= 0: return self.dom.max(A, B) if kA >= 1 else self.dom.min(A, B)
        # not a max/min in disguise: keep it as a piecewise term.  Volta canonicalises
        # `select` as an uninterpreted atom, so two kernels that build the same
        # piecewise function still decide -- refusing here threw that away.
        return T.select(_pred_term(c), x, y)

    def exec_op(self, op, env):
        n, raw = op.name, op.raw
        shape = P.shape_of(op.rtype)
        R = op.results

        def put(t): env[R[0]] = t
        def a(i): return env[op.operands[i]]

        if n == "arith.constant":
            m = _CST_DENSE.search(raw)
            if m: put(Tile(shape, [_lit(m.group(1), op.rtype, self.dom)] * numel(shape)))
            else:
                m = _CST_SCALAR.search(raw)
                if m is None: raise Unsupported(f"arith.constant form not recognised: {raw.strip()[:70]}")
                put(scalar(_lit(m.group(1), op.rtype, self.dom)))
        elif n == "tt.get_program_id":
            axis = "xyz".index(_PID.search(raw).group(1))
            put(scalar(self.pid[axis]))
        elif n == "tt.make_range":
            put(Tile(shape, list(range(op.attrs["start"], op.attrs["end"]))))
        elif n == "tt.splat":
            put(Tile(shape, [a(0).data[0]] * numel(shape)))
        elif n == "tt.expand_dims":
            put(Tile(shape, list(a(0).data)))
        elif n == "tt.broadcast":
            put(bcast(a(0), shape))
        elif n in ("tt.reshape", "tt.trans") and n == "tt.reshape":
            put(Tile(shape, list(a(0).data)))
        elif n == "tt.addptr":
            p, o = bcast(a(0), shape), bcast(a(1), shape)
            if any(isinstance(off, T.Term) for off in o.data):
                raise Unsupported("data-dependent address (offset loaded from memory)")
            put(Tile(shape, [Ptr(x.buf, x.off + y) for x, y in zip(p.data, o.data)]))
        elif n.startswith("arith.") and n[6:] in _INT_BIN:
            f = _INT_BIN[n[6:]]
            bits = S.int_bits(P.elem_of(op.rtype))
            x, y = bcast(a(0), shape), bcast(a(1), shape)
            out = []
            for u, v in zip(x.data, y.data):
                if isinstance(u, Pred) or isinstance(v, Pred):
                    out.append(_boolop(n, u, v)); continue
                if isinstance(u, T.Term) or isinstance(v, T.Term):
                    raise Unsupported("data-dependent integer arithmetic (integer loaded from memory)")
                if n in ("arith.divsi", "arith.remsi") and v == 0:
                    self.g.errors.append(("div-by-zero", "", 0, self.pid)); out.append(0)
                    continue
                r = f(u, v)
                out.append(r if isinstance(r, bool) else S.wrap(r, bits))
            put(Tile(shape, out))
        elif n.startswith("arith.") and n[6:] in _FP_BIN:
            f = _FP_BIN[n[6:]](self.dom)
            x, y = bcast(a(0), shape), bcast(a(1), shape)
            put(Tile(shape, [f(u, v) for u, v in zip(x.data, y.data)]))
        elif n == "arith.cmpi":
            pred = _CMP.search(raw).group(1)
            f = _CMP_OPS[pred]
            x, y = bcast(a(0), shape), bcast(a(1), shape)
            if any(isinstance(u, Pred) for u in x.data + y.data):
                # a stored mask reloaded: (mask != 0) is the mask, (mask == 0) its negation
                out = []
                for u, v in zip(x.data, y.data):
                    p, c = (u, v) if isinstance(u, Pred) else (v, u)
                    if isinstance(c, Pred) or c not in (0, 1, True, False): raise Unsupported("integer compare of two predicates")
                    truth = (pred == "ne") == (c == 0)     # ne 0 / eq 1 -> p ; eq 0 / ne 1 -> not p
                    out.append(p if truth else p.negate())
                put(Tile(shape, out)); return
            if any(isinstance(u, T.Term) for u in x.data + y.data):
                raise Unsupported("comparison on an integer loaded from memory")
            put(Tile(shape, [f(u, v) for u, v in zip(x.data, y.data)]))
        elif n == "arith.select":
            c, x, y = bcast(a(0), shape), bcast(a(1), shape), bcast(a(2), shape)
            put(Tile(shape, [self._select(k, u, v) for k, u, v in zip(c.data, x.data, y.data)]))
        elif n in ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast"):
            put(Tile(shape, list(bcast(a(0), shape).data)))
        elif n == "math.exp":
            put(Tile(shape, [self.dom.exp(u) for u in bcast(a(0), shape).data]))
        elif n == "math.exp2":          # exp2(x) = exp(x * ln 2), ln 2 an exact atom
            put(Tile(shape, [self.dom.exp(self.dom.mul(u, LN2)) for u in bcast(a(0), shape).data]))
        elif n in ("math.log", "math.log2"):
            f = (lambda u: T.app("log", u)) if n == "math.log" else (lambda u: T.div(T.app("log", u), LN2))
            put(Tile(shape, [f(u) for u in bcast(a(0), shape).data]))
        elif n == "math.sqrt":
            put(Tile(shape, [T.app("sqrt", u) for u in bcast(a(0), shape).data]))
        elif n == "math.rsqrt":
            put(Tile(shape, [T.div(T.ONE, T.app("sqrt", u)) for u in bcast(a(0), shape).data]))
        elif n in ("math.cos", "math.sin", "math.tanh", "math.erf"):
            put(Tile(shape, [T.app(n.split(".")[1], u) for u in bcast(a(0), shape).data]))
        elif n in ("math.absf", "math.fabs"):
            put(Tile(shape, [T.app("abs", u) for u in bcast(a(0), shape).data]))
        elif n == "arith.negf":
            put(Tile(shape, [self.dom.mul(self.dom.const(-1.0), u) for u in bcast(a(0), shape).data]))
        elif n in ("arith.maximumf", "arith.minimumf"):
            f = self.dom.max if n == "arith.maximumf" else self.dom.min
            x, y = bcast(a(0), shape), bcast(a(1), shape)
            put(Tile(shape, [f(u, v) for u, v in zip(x.data, y.data)]))
        elif n in ("arith.extf", "arith.truncf", "tt.fp_to_fp"):
            el = P.elem_of(op.rtype)
            put(Tile(shape, [self.dom.cast(u, el) for u in bcast(a(0), shape).data]))
        elif n in ("arith.sitofp", "arith.uitofp"):
            put(Tile(shape, [self.dom.const(float(u)) for u in bcast(a(0), shape).data]))
        elif n in ("arith.fptosi", "arith.fptoui"):
            raise Unsupported("float->int conversion of a symbolic value")
        elif n == "arith.cmpf":
            kind = _CMP.search(raw).group(1)
            x, y = bcast(a(0), shape), bcast(a(1), shape)
            put(Tile(shape, [_cmpf(kind, u, v) for u, v in zip(x.data, y.data)]))
        elif n == "tt.bitcast" and "!tt.ptr" in op.rtype:
            put(Tile(shape, list(bcast(a(0), shape).data)))
        elif n == "tt.bitcast":
            import struct
            src_el = P.elem_of(P.split_top(raw.rsplit(" : ", 1)[1].split("->")[0])[0]); dst_el = P.elem_of(op.rtype)
            fm = {"i32": "i", "f32": "f", "i64": "q", "f64": "d", "i16": "h", "f16": "e", "i8": "b", "i1": "?"}
            out = []
            for u in bcast(a(0), shape).data:
                if isinstance(u, T.Term) or src_el not in fm or dst_el not in fm:
                    raise Unsupported(f"bitcast {src_el}->{dst_el} of a symbolic value")
                v = struct.unpack(fm[dst_el], struct.pack(fm[src_el], u))[0]
                out.append(self.dom.const(v) if dst_el.startswith("f") else v)
            put(Tile(shape, out))
        elif n == "tt.get_num_programs":
            axis = "xyz".index(re.search(r"tt\.get_num_programs\s+(\w+)", raw).group(1))
            put(scalar((list(self.grid_shape) + [1, 1, 1])[axis]))
        elif n == "tt.trans":
            import numpy as np
            m = re.search(r"order\s*=\s*array<i32:\s*([\d,\s]+)>", raw)
            order = [int(t) for t in m.group(1).split(",")] if m else [1, 0]
            arr = np.empty(len(a(0).data), dtype=object); arr[:] = a(0).data
            put(Tile(shape, list(arr.reshape(a(0).shape).transpose(order).reshape(-1))))
        elif n == "tt.extern_elementwise":
            sym = re.search(r'symbol\s*=\s*"(\w+)"', raw).group(1)
            xs = [bcast(a(i), shape).data for i in range(len(op.operands))]
            f = _LIBDEVICE.get(sym.lstrip("_").replace("nv_", ""))
            if f is None: raise Unsupported(f"libdevice `{sym}` has no interpretation over the reals")
            put(Tile(shape, [f(self.dom, *args) for args in zip(*xs)]))
        elif n in ("gpu.barrier", "ttg.barrier", "tt.assert", "tt.print", "tt.debug_barrier"):
            pass
        elif n == "scf.if":
            c = a(0).data[0]
            if isinstance(c, Pred): raise Unsupported("data-dependent branch (scf.if on a float comparison)")
            body = op.body if c else op.else_body
            inner = dict(env)
            self.exec_block(body, inner)
            ys = [o for o in body if o.name == "scf.yield"]
            if ys:
                for r, v in zip(R, ys[-1].operands): env[r] = inner[v]
        elif n == "tt.load":
            p = bcast(a(0), shape)
            mask = bcast(a(1), shape).data if len(op.operands) > 1 else [True]*numel(shape)
            # decision load.masked-value
            other = bcast(a(2), shape).data if len(op.operands) > 2 else [self.dom.const(0.0)]*numel(shape)
            if any(isinstance(m, Pred) for m in mask): raise Unsupported("load mask derived from a float comparison")
            put(Tile(shape, [self.g.read(q.buf, q.off, self.pid, self.dom) if m else o
                             for q, m, o in zip(p.data, mask, other)]))
        elif n == "tt.store":
            pshape = P.shape_of(op.rtype)
            p, v = bcast(a(0), pshape), bcast(a(1), pshape)
            mask = bcast(a(2), pshape).data if len(op.operands) > 2 else [True]*numel(pshape)
            for q, val, m in zip(p.data, v.data, mask):
                if m: self.g.write(q.buf, q.off, val, self.pid, dom=self.dom)
        elif n == "tt.atomic_rmw":
            kind = _ATOMIC.search(raw).group(1)
            assert kind == "fadd", f"unsupported atomic {kind}"
            p, v = bcast(a(0), shape), bcast(a(1), shape)
            mask = bcast(a(2), shape).data if len(op.operands) > 2 else [True]*numel(shape)
            for q, val, m in zip(p.data, v.data, mask):
                if m: self.g.write(q.buf, q.off, val, self.pid, atomic=True, dom=self.dom)
            put(Tile(shape, [self.dom.const(0.0)]*numel(shape)))   # old value, unused
        elif n == "tt.dot":
            A, B, C = a(0), a(1), a(2)
            M, Kd = A.shape; Kd2, N = B.shape
            assert Kd == Kd2
            pm = _PREC.search(raw); prec = pm.group(1) if pm else "ieee"
            out = []
            for i in range(M):
                for j in range(N):
                    prods = [self.dom.mul(A.data[i*Kd+k], B.data[k*N+j]) for k in range(Kd)]
                    out.append(self.dom.dot(prods, C.data[i*N+j], prec))
            put(Tile((M, N), out))
        elif n == "scf.for":
            m = _FOR.search(raw)
            iv, lo, hi, st = m.group(1), m.group(2), m.group(3), m.group(4)
            pairs = []
            if m.group(5):
                for kv in P.split_top(m.group(5)):
                    k, v = [s.strip() for s in kv.split("=")]
                    pairs.append((k, v))
            cur = [env[v] for _, v in pairs]
            lo_v, hi_v, st_v = env[lo].data[0], env[hi].data[0], env[st].data[0]
            for i in range(lo_v, hi_v, st_v):
                inner = dict(env)
                inner[iv] = scalar(i)
                for (k, _), val in zip(pairs, cur): inner[k] = val
                self.exec_block(op.body, inner)
                ys = [o for o in op.body if o.name == "scf.yield"]
                if ys: cur = [inner[v] for v in ys[-1].operands]
            for r, val in zip(R, cur): env[r] = val
        elif n == "tt.reduce":
            if len(op.operands) != 1: raise Unsupported("multi-value reduce (e.g. argmax)")
            src = a(0)
            axis = op.attrs["axis"]
            blk = [o for o in op.body if o.name == "^block"][0]
            body = [o for o in op.body if o.name not in ("^block",)]
            ret = [o for o in body if o.name == "tt.reduce.return"][0]
            ish = src.shape
            osh = tuple(d for i, d in enumerate(ish) if i != axis)
            ist, n_ax = strides(ish), ish[axis]
            out = []
            for lin in range(numel(osh)):
                # rebuild the full index with `axis` free
                idx, rem = [], lin
                for i, d in enumerate(osh):
                    st = numel(osh[i+1:]); idx.append(rem // st); rem %= st
                base = 0; k = 0
                for i in range(len(ish)):
                    if i == axis: continue
                    base += idx[k] * ist[i]; k += 1
                acc = src.data[base]
                for j in range(1, n_ax):
                    inner = dict(env)
                    inner[blk.results[0]] = scalar(acc)
                    inner[blk.results[1]] = scalar(src.data[base + j*ist[axis]])
                    self.exec_block(body, inner)
                    acc = inner[ret.operands[0]].data[0]
                out.append(acc)
            put(Tile(osh, out) if osh else scalar(out[0]))
        elif n in ("scf.yield", "tt.return", "tt.func", "^block", "tt.reduce.return"):
            pass
        else:
            raise NotImplementedError(f"{n}   |{raw}")

_LIBDEVICE = {
    "expf": lambda d, x: d.exp(x), "exp": lambda d, x: d.exp(x),
    "exp2f": lambda d, x: d.exp(d.mul(x, LN2)),
    "logf": lambda d, x: T.app("log", x), "log2f": lambda d, x: T.div(T.app("log", x), LN2),
    "sqrtf": lambda d, x: T.app("sqrt", x), "rsqrtf": lambda d, x: T.div(T.ONE, T.app("sqrt", x)),
    "fabsf": lambda d, x: T.app("abs", x),
    "fmaxf": lambda d, x, y: d.max(x, y), "fminf": lambda d, x, y: d.min(x, y),
    "expm1f": lambda d, x: d.sub(d.exp(x), d.const(1.0)),
    "tanhf": lambda d, x: T.app("tanh", x), "erff": lambda d, x: T.app("erf", x),
    "sinf": lambda d, x: T.app("sin", x), "cosf": lambda d, x: T.app("cos", x),
    "powf": lambda d, x, y: T.app("pow", x, y),
    "log1pf": lambda d, x: T.app("log", d.add(d.const(1.0), x)),
}

_INT_BIN = {
    "addi": lambda a, b: a + b, "subi": lambda a, b: a - b,
    "muli": lambda a, b: a * b,
    "divsi": lambda a, b: int(a / b) if b else 0,
    "remsi": lambda a, b: a - b * int(a / b) if b else 0,
    "andi": lambda a, b: (a and b) if isinstance(a, bool) else a & b,
    "ori":  lambda a, b: (a or b) if isinstance(a, bool) else a | b,
    "maxsi": max, "minsi": min,
}
_FP_BIN = {
    "addf": lambda d: d.add, "subf": lambda d: d.sub,
    "mulf": lambda d: d.mul, "divf": lambda d: d.div,
    "maxnumf": lambda d: d.max, "minnumf": lambda d: d.min,
}
_CMP_OPS = {
    "slt": lambda a, b: a < b, "sle": lambda a, b: a <= b,
    "sgt": lambda a, b: a > b, "sge": lambda a, b: a >= b,
    "eq":  lambda a, b: a == b, "ne": lambda a, b: a != b,
    "ult": lambda a, b: (a % 2**32) <  (b % 2**32), "ule": lambda a, b: (a % 2**32) <= (b % 2**32),
    "ugt": lambda a, b: (a % 2**32) >  (b % 2**32), "uge": lambda a, b: (a % 2**32) >= (b % 2**32),
}

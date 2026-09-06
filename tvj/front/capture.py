"""Judge a kernel by intercepting its real launch.

Generated code only has to expose `launch(*inputs) -> output`.  We hook
`JITFunction.run`, so every `kernel[grid](...)` inside it is recorded (kernel,
grid, args, constexprs) while still executing on the GPU -- one call yields
both the tolerance-test output and everything the symbolic judge needs.
Tensors are mapped to roles (input names / "out") by object identity, so the
kernel's parameter names are irrelevant.
"""
import torch, triton
from triton.runtime.jit import JITFunction
from tvj.core import terms as T
from tvj.core import ttir as P
from tvj.core import sexec as X
from tvj.checks.check import to_ttir

_DT = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16",
       torch.int32: "i32", torch.int64: "i64", torch.int8: "i8", torch.bool: "i1"}
_LAUNCH_KW = {"num_warps", "num_stages", "num_ctas", "maxnreg", "enable_fp_fusion", "launch_cooperative_grid"}

_calls, _orig = [], JITFunction.run
def _rec(self, *args, grid, warmup=False, **kwargs):
    _calls.append((self, grid, args, dict(kwargs)))
    return _orig(self, *args, grid=grid, warmup=warmup, **kwargs)

def base_of(t): return t.untyped_storage().data_ptr()
def elem_offset(t): return (t.data_ptr() - base_of(t)) // t.element_size()
def storage_numel(t): return t.untyped_storage().nbytes() // t.element_size()

_EXTERN = ("mm", "addmm", "bmm", "baddbmm", "convolution")

def _wrap_externs(ns):
    """Inductor routes matmuls to cuBLAS via `extern_kernels.*`; record those
    calls in launch order so the judge can model them (trusted, unverified)."""
    ek = ns.get("extern_kernels") if ns else None
    if ek is None: return None
    saved = {}
    for name in _EXTERN:
        f = getattr(ek, name, None)
        if f is None: continue
        saved[name] = f
        def make(name, f):
            def w(*args, **kwargs):
                res = f(*args, **kwargs)
                _calls.append(("extern", name, args, dict(kwargs), res))
                return res
            return w
        setattr(ek, name, make(name, f))
    return ek, saved

_COPY_METHODS = ("contiguous", "to", "clone", "float", "half", "double", "detach",
                 "cuda", "cpu", "type", "type_as", "flatten", "reshape", "view", "ravel")
PROVENANCE = {}          # storage base of a copy -> storage base it came from

def _install_provenance():
    saved = {}
    for name in _COPY_METHODS:
        f = getattr(torch.Tensor, name, None)
        if f is None: continue
        saved[name] = f
        def make(f):
            def w(self, *a, **k):
                r = f(self, *a, **k)
                try:
                    if isinstance(r, torch.Tensor) and base_of(r) != base_of(self):
                        PROVENANCE.setdefault(base_of(r), base_of(self))
                except Exception: pass
                return r
            return w
        setattr(torch.Tensor, name, make(f))
    return saved

def _remove_provenance(saved):
    for name, f in saved.items(): setattr(torch.Tensor, name, f)

def root_storage(b):
    """Follow copy provenance back to the storage a role was assigned to."""
    seen = set()
    while b in PROVENANCE and b not in seen:
        seen.add(b); b = PROVENANCE[b]
    return b

def capture(launch, *inputs, ns=None, trace=None):
    """Run `launch(*inputs)` on the GPU; return (output, recorded events).
    Events are kernel launches (fn, grid, args, kwargs) or ("extern", name, args, kwargs).
    Pass `trace` (a torchtrace.TorchTrace) to also record the wrapper's torch ops."""
    _calls.clear(); PROVENANCE.clear(); JITFunction.run = _rec
    w = _wrap_externs(ns)
    saved = _install_provenance()
    try:
        if trace is not None:
            with trace: out = launch(*inputs)
        else: out = launch(*inputs)
    finally:
        JITFunction.run = _orig
        _remove_provenance(saved)
        if w:
            for name, f in w[1].items(): setattr(w[0], name, f)
    torch.cuda.synchronize()
    return out, list(_calls)

class Extern:
    """A library-side op (cuBLAS / cuDNN) modelled at spec level.  Not verified: trusted."""
    def __init__(self, name, args, kwargs, roles, result=None):
        self.name, self.args, self.kwargs, self.roles, self.result = name, args, kwargs, roles, result
        for t in list(args) + list(kwargs.values()) + [result]:
            if isinstance(t, torch.Tensor):
                b = root_storage(base_of(t))
                if b not in roles: roles[b] = f"tmp{sum(1 for r in roles.values() if r.startswith('tmp'))}"
    def apply(self, grid):
        from tvj.front.spec import STensor
        import numpy as np
        def read(t):
            role = self.roles[root_storage(base_of(t))]
            grid.bufsize[role] = max(grid.bufsize.get(role, 0), storage_numel(t))
            terms = [grid.store.get((role, p)) or T.sym(role, p) for p in physical_offsets(t)]
            arr = np.empty(len(terms), dtype=object); arr[:] = terms
            return STensor(arr.reshape(tuple(t.shape)))
        a = self.args; k = self.kwargs
        alpha, beta = k.get("alpha", 1), k.get("beta", 1)
        if self.name == "convolution":
            from tvj.front.spec import conv_nd
            from tvj.core.sexec import Unsupported
            x, w_, b = a[0], a[1], a[2] if len(a) > 2 else k.get("bias")
            stride, padding, dilation = k.get("stride", a[3] if len(a) > 3 else 1), k.get("padding", a[4] if len(a) > 4 else 0), k.get("dilation", a[5] if len(a) > 5 else 1)
            transposed, groups = k.get("transposed", a[6] if len(a) > 6 else False), k.get("groups", a[8] if len(a) > 8 else 1)
            if transposed: raise Unsupported("extern transposed convolution")
            res = conv_nd(read(x), read(w_), read(b) if b is not None else None, stride, padding, dilation, groups, nd=w_.dim() - 2)
        elif self.name.split(".")[0] in ("avg_pool1d", "avg_pool2d", "avg_pool3d",
                                         "max_pool1d_with_indices", "max_pool2d_with_indices",
                                         "max_pool3d_with_indices"):
            # `torch.ops.aten.*_pool*`, recorded by torchtrace.AS_EXTERN.  Positional
            # signatures, straight from native_functions.yaml:
            #   avg_pool2d(input, kernel, stride, padding, ceil_mode, count_include_pad,
            #              divisor_override)
            #   max_pool2d_with_indices(input, kernel, stride, padding, dilation, ceil_mode)
            from tvj.front.spec import pool_nd
            base = self.name.split(".")[0]
            nd = int(base[8]) if base.startswith("avg_pool") else int(base[8])
            mode = "avg" if base.startswith("avg") else "max"
            g = lambda i, key, dflt: k.get(key, a[i] if len(a) > i else dflt)
            if mode == "avg":
                res = pool_nd(read(a[0]), g(1, "kernel_size", None), g(2, "stride", None),
                              g(3, "padding", 0), nd=nd, mode="avg",
                              ceil_mode=g(4, "ceil_mode", False),
                              count_include_pad=g(5, "count_include_pad", True),
                              divisor_override=g(6, "divisor_override", None))
            else:
                res = pool_nd(read(a[0]), g(1, "kernel_size", None), g(2, "stride", None),
                              g(3, "padding", 0), nd=nd, mode="max",
                              dilation=g(4, "dilation", 1), ceil_mode=g(5, "ceil_mode", False))
        elif self.name in ("mm", "bmm"):
            res = read(a[0]) @ read(a[1])
        else:                                   # addmm / baddbmm: beta*bias + alpha*(a@b)
            res = read(a[1]) @ read(a[2])
            if alpha != 1: res = res * alpha
            bias = read(a[0]); res = res + (bias * beta if beta != 1 else bias)
        # `@` on an STensor already delegates when the product is large enough
        # (delegate.py); nothing to do here beyond keeping alpha/beta/bias in the
        # term algebra, which is where the reference puts them too.
        out = k.get("out", self.result)
        role = self.roles[root_storage(base_of(out))]
        grid.bufsize[role] = max(grid.bufsize.get(role, 0), storage_numel(out))
        for p, term in zip(physical_offsets(out), res.flat()):
            grid.store[(role, p)] = term; grid.kind[(role, p)] = "store"; grid.writer[(role, p)] = ("earlier",)

class Launch:
    """One recorded kernel launch, resolved to what the interpreter needs.

    `scalar_syms` maps the runtime value of a 0-d parameter to its role, so a
    wrapper that passes `scale.item()` as an fp32 kernel argument still binds to
    the symbol `p_scale` instead of freezing that run's value as a constant.
    Callers must give the parameters distinct random values first, or the match
    is ambiguous."""
    def __init__(self, fn, grid, args, kwargs, roles, scalar_syms=None):
        self.scalar_syms = scalar_syms or {}
        self.fn = fn
        params = fn.params
        bound = {}
        for i, p in enumerate(params):
            bound[p.name] = args[i] if i < len(args) else kwargs.get(p.name)
        self.signature, self.constexprs, self.argvals, self.bufs, self.rolemap = {}, {}, [], {}, {}
        for p in params:
            v = bound[p.name]
            if p.is_constexpr:
                self.signature[p.name] = "constexpr"; self.constexprs[p.name] = v
            elif isinstance(v, torch.Tensor):
                self.signature[p.name] = "*" + _DT[v.dtype]
                b = root_storage(base_of(v))
                if b not in roles: roles[b] = f"tmp{sum(1 for r in roles.values() if r.startswith('tmp'))}"
                role = roles[b]
                self.rolemap[p.name] = role
                self.bufs[role] = max(self.bufs.get(role, 0), storage_numel(v))
                self.argvals.append(X.Ptr(role, elem_offset(v)))
            elif isinstance(v, bool):
                self.signature[p.name] = "i1"; self.argvals.append(int(v))
            elif isinstance(v, int):
                self.signature[p.name] = "i32" if -2**31 <= v < 2**31 else "i64"; self.argvals.append(v)
            elif isinstance(v, float):
                self.signature[p.name] = "fp32"
                role = self.scalar_syms.get(round(v, 12))
                self.argvals.append(("sym", role) if role else ("f32", v))
            else:
                raise TypeError(f"unsupported launch argument {p.name}={v!r}")
        g = grid(dict(bound, **{k: kwargs[k] for k in kwargs if k in _LAUNCH_KW})) if callable(grid) else grid
        g = tuple(g) if isinstance(g, (tuple, list)) else (g,)
        self.grid = tuple(int(x) for x in g) + (1,) * (3 - len(g))
        self.launch_kw = {k: kwargs[k] for k in kwargs if k in _LAUNCH_KW}

    def ttir(self): return to_ttir(self.fn, self.signature, self.constexprs)

def roles_of(inputs, output):
    r = {base_of(t): name for name, t in inputs.items()}
    r[base_of(output)] = "out"
    return r

def physical_offsets(t):
    """Physical element offset of each logical (row-major) element of tensor t."""
    import itertools
    off, st, sh = elem_offset(t), t.stride(), t.shape
    return [off + sum(i * s for i, s in zip(idx, st)) for idx in itertools.product(*[range(d) for d in sh])]

def _scalar_arg(it, v):
    kind, val = v
    if kind == "sym":
        it.g.bufsize.setdefault(val, 1)
        return T.sym(val, 0)
    return it.dom.const(val)

def symbolic_run(events, domain=None):
    """Execute recorded events in order against one shared symbolic memory."""
    grid, last, externs = X.Grid({}), None, []
    domain = domain or X.RealDomain()
    for ev in events:
        if isinstance(ev, Extern):
            ev.apply(grid); externs.append(ev.name); continue
        L = ev
        for b, n in L.bufs.items(): grid.bufsize[b] = max(grid.bufsize.get(b, 0), n)
        f = P.parse(L.ttir())
        it = X.Interp(f, None, L.grid, grid.bufsize, domain=domain)
        it.g = grid
        it.argvals = [(_scalar_arg(it, v) if isinstance(v, tuple) else v) for v in L.argvals]
        it.run_all()
        last = it
    grid.externs = externs
    return grid, last

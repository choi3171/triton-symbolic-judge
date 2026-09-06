"""Spec front-end: PyTorch-style reference code -> term DAGs.

`STensor` is a numpy object array of terms with a torch-like surface.  Reference
code written against torch (matmul, softmax, layer_norm, relu, elementwise,
reductions, transposes) runs unchanged on STensors via operator overloading and
the `__torch_function__` protocol, producing the same term algebra the TTIR
interpreter produces -- so kernel and spec meet in one normal form.

Spec-side decisions (they are decisions):
  spec.softmax-form   softmax is defined max-subtracted: exp(x - max x) / sum.
                      Equal over the reals to exp(x)/sum; it is what torch
                      computes, its float-validity radius is unbounded (so the
                      precondition obligation compares kernels against the best
                      known form), and it keeps Volta's cross-multiplication small
                      (matched forms canonicalise ~10x cheaper).
  spec.reduce-order   reductions are n-ary AC sums; no order is implied.
"""
import functools, re, numpy as np, terms as T

def _prod(shape):
    n = 1
    for d in shape: n *= d
    return n

def _arr(x):
    if isinstance(x, STensor): return x.a
    if isinstance(x, T.Term): return np.array(x, dtype=object)
    if isinstance(x, (int, float)): return np.array(T.const(x), dtype=object)
    if isinstance(x, np.ndarray): return x.astype(object) if x.dtype != object else x
    try:
        import torch
        if isinstance(x, torch.Tensor):          # a concrete tensor (constant, buffer) -> constants
            vals = x.detach().cpu().flatten().tolist()
            a = np.empty(len(vals), dtype=object); a[:] = [T.const(v) for v in vals]
            return a.reshape(tuple(x.shape))
    except ImportError: pass
    raise TypeError(type(x))

def _fold_axis(arr, axis, fn, keepdim=False):
    """Fold `fn` over one axis of an object array."""
    arr = np.asarray(arr, dtype=object)
    if axis < 0: axis += arr.ndim
    moved = np.moveaxis(arr, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty(flat.shape[0], dtype=object)
    for i in range(flat.shape[0]):
        out[i] = fn(list(flat[i]))
    out = out.reshape(moved.shape[:-1])
    if keepdim: out = np.expand_dims(out, axis)
    return out

class STensor:
    def __init__(self, a): self.a = np.asarray(a, dtype=object)

    @staticmethod
    def input(name, shape):
        return STensor(np.array([T.sym(name, i) for i in range(_prod(shape))], dtype=object).reshape(shape))
    @staticmethod
    def full(shape, v): return STensor(np.full(shape, T.const(v), dtype=object))

    # -- shape --
    @property
    def shape(self): return tuple(self.a.shape)
    @property
    def ndim(self): return self.a.ndim
    def dim(self): return self.a.ndim
    def size(self, i=None): return self.shape if i is None else self.shape[i]
    def numel(self): return self.a.size
    def flat(self): return list(self.a.reshape(-1))
    def reshape(self, *s): s = s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s; return STensor(self.a.reshape(s))
    view = reshape
    def transpose(self, i, j): return STensor(np.swapaxes(self.a, i, j))
    def permute(self, *p): p = p[0] if len(p) == 1 and isinstance(p[0], (tuple, list)) else p; return STensor(np.transpose(self.a, p))
    def t(self): return STensor(self.a.T)
    @property
    def T(self): return STensor(self.a.T)
    def unsqueeze(self, dim=None, d=None): d = dim if dim is not None else d; return STensor(np.expand_dims(self.a, d))
    def squeeze(self, dim=None, d=None):
        d = dim if dim is not None else d
        if d is None: return STensor(np.squeeze(self.a))
        d = d % self.ndim if self.ndim else 0
        return STensor(np.squeeze(self.a, d)) if self.ndim and self.shape[d] == 1 else self
    def __setitem__(self, idx, v):
        a = self.a.copy()
        a[idx] = v.a if isinstance(v, STensor) else _arr(v)
        self.a = a
    def __getitem__(self, idx): r = self.a[idx]; return STensor(r) if isinstance(r, np.ndarray) else r
    def contiguous(self): return self
    def clone(self): return self
    def float(self):
        return STensor(np.frompyfunc(
            lambda t: T.select(t, T.ONE, T.ZERO) if _is_bool(t) else t, 1, 1)(self.a))
    def to(self, *a, **k): return self
    def type_as(self, o): return self
    def __len__(self): return self.shape[0]

    # -- arithmetic (numpy broadcasting + Term operators) --
    def __add__(self, o): return STensor(self.a + _arr(o))
    def __radd__(self, o): return STensor(_arr(o) + self.a)
    def __sub__(self, o): return STensor(self.a - _arr(o))
    def __rsub__(self, o): return STensor(_arr(o) - self.a)
    def __mul__(self, o): return STensor(self.a * _arr(o))
    def __rmul__(self, o): return STensor(_arr(o) * self.a)
    def __truediv__(self, o): return STensor(self.a / _arr(o))
    def __rtruediv__(self, o): return STensor(_arr(o) / self.a)
    def __neg__(self): return STensor(-self.a)
    def __pow__(self, p): return STensor(np.frompyfunc(lambda t: t ** p, 1, 1)(self.a))
    def __matmul__(self, o): return STensor(np.matmul(self.a, _arr(o)))
    def matmul(self, o): return self @ o
    def mm(self, o): return self @ o
    def bmm(self, o): return self @ o

    # -- elementwise functions --
    def _map(self, fn): return STensor(np.frompyfunc(fn, 1, 1)(self.a))
    def exp(self): return self._map(lambda t: T.app("exp", t))
    def sqrt(self): return self._map(lambda t: T.app("sqrt", t))
    def rsqrt(self): return self._map(lambda t: T.div(T.ONE, T.app("sqrt", t)))
    def log(self): return self._map(lambda t: T.app("log", t))
    def relu(self): return self._map(lambda t: T.app("max", t, T.ZERO))
    def square(self): return self * self
    def pow(self, p): return self ** p
    def maximum(self, o): return STensor(np.frompyfunc(lambda a, b: T.app("max", a, b), 2, 1)(self.a, _arr(o)))
    def minimum(self, o): return STensor(np.frompyfunc(lambda a, b: T.app("min", a, b), 2, 1)(self.a, _arr(o)))

    # -- reductions --
    def sum(self, dim=None, keepdim=False, dtype=None, axis=None):
        dim = axis if dim is None else dim
        if dim is None: return _scalar(T.add(*self.flat()))
        if isinstance(dim, (tuple, list)):
            r = self
            for d in sorted([x % self.ndim for x in dim], reverse=True): r = r.sum(d, keepdim)
            return r
        return STensor(_fold_axis(self.a, dim, lambda v: T.add(*v), keepdim))
    def mean(self, dim=None, keepdim=False, dtype=None, axis=None):
        dim = axis if dim is None else dim
        if dim is None: return _scalar(T.mul(T.const(1.0 / self.a.size), T.add(*self.flat())))
        dims = [dim] if isinstance(dim, int) else list(dim)
        n = _prod([self.shape[d] for d in dims])
        return self.sum(dim, keepdim) * (1.0 / n)
    def amax(self, dim=None, keepdim=False, axis=None):
        dim = axis if dim is None else dim
        if dim is None: return _scalar(functools.reduce(lambda a, b: T.app("max", a, b), self.flat()))
        if isinstance(dim, (tuple, list)):
            r = self
            for d in sorted([x % self.ndim for x in dim], reverse=True): r = r.amax(d, keepdim)
            return r
        return STensor(_fold_axis(self.a, dim, lambda v: functools.reduce(lambda a, b: T.app("max", a, b), v), keepdim))
    def max(self, dim=None, keepdim=False):
        if dim is None: return self.amax()
        vals = self.amax(dim, keepdim)
        return _MaxResult(vals, None)
    def var(self, dim=None, keepdim=False, unbiased=True, correction=None):
        if correction is not None: unbiased = bool(correction)
        if dim is None:
            n = self.a.size; mu = self.mean().item(); d = self - mu
            return (d * d).sum() * (1.0 / (n - 1 if unbiased and n > 1 else n))
        mu = self.mean(dim, keepdim=True)
        d = self - mu
        n = self.shape[dim] if isinstance(dim, int) else _prod([self.shape[x] for x in dim])
        return (d * d).sum(dim, keepdim) * (1.0 / (n - 1 if unbiased and n > 1 else n))
    def softmax(self, dim=-1):
        m = self.amax(dim, keepdim=True)
        e = (self - m).exp()
        return e / e.sum(dim, keepdim=True)
    def log_softmax(self, dim=-1):
        m = self.amax(dim, keepdim=True)
        z = self - m
        return z - z.exp().sum(dim, keepdim=True).log()
    def sigmoid(self): return 1.0 / (1.0 + (-self).exp())
    def tanh(self): return self._map(lambda t: T.app("tanh", t))
    def abs(self): return self._map(lambda t: T.app("abs", t))
    def std(self, dim=None, keepdim=False, unbiased=True, correction=None): return self.var(dim, keepdim, unbiased, correction).sqrt()
    def norm(self, p=2, dim=None, keepdim=False):
        if p != 2 and p != "fro": raise NotImplementedError("spec front-end: unsupported torch op norm(p!=2)")
        return (self * self).sum(dim, keepdim).sqrt()
    def expand(self, *s): s = s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s; s = [a if a != -1 else b for a, b in zip(s, (1,)*(len(s)-self.ndim) + self.shape)]; return STensor(np.broadcast_to(self.a, s))
    def expand_as(self, o): return self.expand(*o.shape)
    def repeat(self, *r): r = r[0] if len(r) == 1 and isinstance(r[0], (tuple, list)) else r; return STensor(np.tile(self.a, r))
    def flatten(self, start_dim=0, end_dim=-1, start=None, end=None):
        s = start if start is not None else start_dim; e = end if end is not None else end_dim
        sh = self.shape; e = e if e >= 0 else self.ndim + e
        if self.ndim == 0: return self.reshape(1)
        return self.reshape(*sh[:s], -1, *sh[e+1:])
    def item(self): return self.a.reshape(-1)[0]
    def ndimension(self): return self.ndim
    def nelement(self): return self.a.size
    @property
    def data(self): return self
    @property
    def grad(self): return None
    def requires_grad_(self, *a): return self
    @property
    def device(self): import torch; return torch.device("cuda")
    @property
    def dtype(self): import torch; return torch.float32
    @property
    def is_cuda(self): return True
    def add(self, o, alpha=1): return self + (o * alpha if alpha != 1 else o)
    def sub(self, o, alpha=1): return self - (o * alpha if alpha != 1 else o)
    def mul(self, o): return self * o
    def div(self, o, rounding_mode=None): return self / o
    # in-place variants must MUTATE: modules do `qk.div_(scale)` and discard the result
    def _inplace(self, r): self.a = r.a; return self
    def add_(self, o, alpha=1): return self._inplace(self.add(o, alpha))
    def sub_(self, o, alpha=1): return self._inplace(self.sub(o, alpha))
    def mul_(self, o): return self._inplace(self * o)
    def div_(self, o, rounding_mode=None): return self._inplace(self / o)
    def clamp_(self, min=None, max=None): return self._inplace(self.clamp(min, max))
    def relu_(self): return self._inplace(self.relu())
    def tanh_(self): return self._inplace(self.tanh())
    def sigmoid_(self): return self._inplace(self.sigmoid())
    def exp_(self): return self._inplace(self.exp())
    def fill_(self, v): return self._inplace(STensor.full(self.shape, float(v)))
    def zero_(self): return self.fill_(0.0)
    def copy_(self, o): return self._inplace(_st(o).expand(*self.shape) if _st(o).shape != self.shape else _st(o))
    def __iadd__(self, o): return self._inplace(self + o)
    def __isub__(self, o): return self._inplace(self - o)
    def __imul__(self, o): return self._inplace(self * o)
    def __itruediv__(self, o): return self._inplace(self / o)
    def clamp(self, min=None, max=None):
        r = self
        if min is not None: r = r.maximum(min)
        if max is not None: r = r.minimum(max)
        return r
    def clamp_min(self, v): return self.maximum(v)
    def clamp_max(self, v): return self.minimum(v)
    def new_empty(self, *s, **k): return self.new_zeros(*s)
    def new_full(self, s, v, **k): return STensor.full(tuple(s), float(v))
    def _cmp(self, kind, o):
        """Elementwise comparison as a term.  Not a Python bool: it is a piecewise
        condition, which `select` consumes and Volta canonicalises as an atom."""
        x, y = self.a, _arr(o)
        return STensor(np.frompyfunc(lambda u, v: T.cmp(kind, u, v), 2, 1)(x, y))
    def __lt__(self, o): return self._cmp("lt", o)
    def __le__(self, o): return self._cmp("le", o)
    def __gt__(self, o): return self._cmp("gt", o)
    def __ge__(self, o): return self._cmp("ge", o)
    def __ne__(self, o): return self._cmp("ne", o)
    def eq(self, o): return self._cmp("eq", o)
    def ne(self, o): return self._cmp("ne", o)
    def gt(self, o): return self._cmp("gt", o)
    def lt(self, o): return self._cmp("lt", o)
    def ge(self, o): return self._cmp("ge", o)
    def le(self, o): return self._cmp("le", o)
    def where(self, a, b): return STensor(np.frompyfunc(
        lambda c, u, v: T.select(c, u, v), 3, 1)(self.a, _arr(a), _arr(b)))
    def any(self): raise NotImplementedError("spec front-end: unsupported torch op any (data-dependent control flow)")
    def all(self): raise NotImplementedError("spec front-end: unsupported torch op all (data-dependent control flow)")
    def __bool__(self): raise NotImplementedError("spec front-end: unsupported torch op bool(tensor) (data-dependent control flow)")
    def relu6(self): return self.relu().minimum(6.0)
    def erf(self): return self._map(lambda t: T.app("erf", t))
    def sin(self): return self._map(lambda t: T.app("sin", t))
    def cos(self): return self._map(lambda t: T.app("cos", t))
    def log_sigmoid(self): return -((-self).exp() + 1.0).log()
    def flip(self, dims):
        d = (dims,) if isinstance(dims, int) else tuple(dims)
        return STensor(np.flip(self.a, d).copy())
    def gelu(self, approximate="none"):
        if approximate == "tanh":
            return self * 0.5 * (1.0 + (0.7978845608028654 * (self + 0.044715 * self * self * self)).tanh())
        return self * 0.5 * (1.0 + (self * 0.7071067811865476).erf())
    def chunk(self, n, dim=0): return [STensor(x) for x in np.array_split(self.a, n, axis=dim)]
    def split(self, size, dim=0):
        if isinstance(size, int): idx = list(range(size, self.shape[dim], size))
        else: idx = list(np.cumsum(size)[:-1])
        return [STensor(x) for x in np.split(self.a, idx, axis=dim)]
    def masked_fill(self, mask, v): return _st(mask).where(STensor.full(self.shape, float(v)), self)
    def half(self): return self
    def double(self): return self
    def type(self, *a, **k): return self
    def cuda(self, *a, **k): return self
    def cpu(self): return self
    def detach(self): return self
    def new_zeros(self, *s, **k): s = s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s; return STensor.full(tuple(s), 0.0)
    def new_ones(self, *s, **k): s = s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s; return STensor.full(tuple(s), 1.0)
    def __iter__(self): return iter(STensor(x) for x in self.a)

    # -- torch.* / F.* dispatch --
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        h = _TORCH.get(name)
        if h is None: raise NotImplementedError(f"spec front-end: unsupported torch op {name}")
        # only kwargs that cannot change the value may be dropped; anything else is a spec error
        HARMLESS = {"out", "dtype", "layout", "device", "requires_grad", "pin_memory", "memory_format",
                    "inplace", "size_average", "reduce", "non_blocking", "copy", "generator"}
        kwargs = {k: v for k, v in kwargs.items() if k not in HARMLESS or (k in ("size_average", "reduce") and v is not None)}
        for _ in range(4):
            try: return h(*args, **kwargs)
            except TypeError as e:
                m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                if not m or m.group(1) not in kwargs: raise
                k = m.group(1)
                if k in HARMLESS: kwargs.pop(k)
                else: raise NotImplementedError(f"spec front-end: unsupported torch op {name}(...{k}=)")
        return h(*args, **kwargs)

def symbolic_module(model, prefix_p="p_", prefix_b="b_"):
    """Replace every parameter/buffer of `model` by a symbolic input named by
    role, so `model.forward(STensor...)` yields the spec.  Destructive."""
    for mname, m in model.named_modules():
        pre = (mname + ".") if mname else ""
        for pn, p in list(m._parameters.items()):
            if p is not None: m._parameters[pn] = STensor.input(prefix_p + pre + pn, tuple(p.shape))
        for bn, b in list(m._buffers.items()):
            if b is not None: m._buffers[bn] = STensor.input(prefix_b + pre + bn, tuple(b.shape))
    return model

class _MaxResult(tuple):
    def __new__(cls, values, indices): return tuple.__new__(cls, (values, indices))
    values = property(lambda s: s[0]); indices = property(lambda s: s[1])

def _is_bool(t):
    return isinstance(t, T.App) and (t.fn.startswith("cmp:") or t.fn in ("and", "or", "not", "true", "false"))

def _st(x): return x if isinstance(x, STensor) else STensor(_arr(x))
def _scalar(t): return STensor(np.array(t, dtype=object))
def _tup(v, n): return tuple(v) if isinstance(v, (tuple, list)) else (v,) * n

def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    x = _st(x); k = len(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else 1
    dims = tuple(range(x.ndim - k, x.ndim))
    mu = x
    for d in dims: mu = mu.mean(d, keepdim=True)
    d = x - mu
    var = d * d
    for dd in dims: var = var.mean(dd, keepdim=True)
    y = d * (var + eps).rsqrt()
    if weight is not None: y = y * _st(weight)
    if bias is not None: y = y + _st(bias)
    return y

def conv_nd(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1, nd=2):
    b = bias
    """Direct convolution over object arrays: out = b + sum w * x_pad (cross-correlation)."""
    import itertools
    x, w = _st(x).a, _st(w).a
    N, Ci = x.shape[:2]; Co = w.shape[0]; ks = w.shape[2:]
    stride, dilation = _tup(stride, nd), _tup(dilation, nd)
    padding = _tup(padding, nd) if not isinstance(padding, str) else None
    if padding is None: raise NotImplementedError("spec front-end: unsupported torch op conv padding='same'")
    xp = np.pad(x, [(0, 0), (0, 0)] + [(p, p) for p in padding], mode="constant", constant_values=T.ZERO)
    out_sp = [(xp.shape[2 + d] - dilation[d] * (ks[d] - 1) - 1) // stride[d] + 1 for d in range(nd)]
    out = np.empty((N, Co, *out_sp), dtype=object)
    cig, cog = Ci // groups, Co // groups
    bb = _st(b).a if b is not None else None
    for n in range(N):
        for co in range(Co):
            g = co // cog
            for pos in itertools.product(*[range(s) for s in out_sp]):
                terms = []
                for ci in range(cig):
                    for kpos in itertools.product(*[range(k) for k in ks]):
                        idx = tuple(pos[d] * stride[d] + kpos[d] * dilation[d] for d in range(nd))
                        terms.append(T.mul(w[(co, ci) + kpos], xp[(n, g * cig + ci) + idx]))
                if bb is not None: terms.append(bb[co])
                out[(n, co) + pos] = T.add(*terms)
    return STensor(out)

def pool_nd(x, kernel_size, stride=None, padding=0, nd=2, mode="max", ceil_mode=False,
            count_include_pad=True, divisor_override=None, **_):
    """Windowed max/avg reduction, written the same way conv_nd is."""
    import itertools, functools
    x = _st(x).a
    ks = _tup(kernel_size, nd); st = _tup(stride if stride is not None else kernel_size, nd)
    pd = _tup(padding, nd)
    fill = T.const(float("-inf")) if mode == "max" else T.ZERO
    xp = np.pad(x, [(0, 0)] * (x.ndim - nd) + [(p, p) for p in pd], mode="constant", constant_values=fill)
    sp_ = [(xp.shape[x.ndim - nd + d] - ks[d]) // st[d] + 1 for d in range(nd)]
    lead = xp.shape[:x.ndim - nd]
    out = np.empty(lead + tuple(sp_), dtype=object)
    for pre in itertools.product(*[range(d) for d in lead]):
        for pos in itertools.product(*[range(s) for s in sp_]):
            vals = [xp[pre + tuple(pos[d] * st[d] + o[d] for d in range(nd))]
                    for o in itertools.product(*[range(k) for k in ks])]
            out[pre + pos] = (functools.reduce(lambda a, b: T.app("max", a, b), vals) if mode == "max"
                              else T.mul(T.const(1.0 / len(vals)), T.add(*vals)))
    return STensor(out)

def adaptive_pool(x, output_size, nd=2, mode="avg"):
    x = _st(x)
    osz = _tup(output_size, nd)
    isz = x.shape[-nd:]
    if all(o == 1 for o in osz):                 # global pooling: the common case
        r = x
        for d in range(x.ndim - nd, x.ndim): r = r.mean(d, keepdim=True) if mode == "avg" else r.amax(d, keepdim=True)
        return r
    if all(i % o == 0 for i, o in zip(isz, osz)):
        return pool_nd(x, tuple(i // o for i, o in zip(isz, osz)), None, 0, nd, mode)
    raise NotImplementedError(f"spec front-end: unsupported torch op adaptive_{mode}_pool{nd}d (ragged)")

def pad(x, pad, mode="constant", value=0.0):
    if mode != "constant": raise NotImplementedError(f"spec front-end: unsupported torch op pad(mode={mode})")
    x = _st(x); widths = [(0, 0)] * x.ndim
    for i in range(len(pad) // 2):
        widths[x.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    return STensor(np.pad(x.a, widths, mode="constant", constant_values=T.const(value or 0.0)))

def bce_with_logits(x, y, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None):
    x, y = _st(x), _st(y)
    if pos_weight is not None: raise NotImplementedError("spec front-end: unsupported torch op bce_with_logits(pos_weight)")
    loss = x.relu() - x * y + (1.0 + (-x.abs()).exp()).log()
    if weight is not None: loss = loss * _st(weight)
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss

def _reduce_loss(l, reduction):
    return l.mean() if reduction == "mean" else l.sum() if reduction == "sum" else l

def _linear(input, weight, bias=None):
    y = _st(input) @ _st(weight).t()
    return y + _st(bias) if bias is not None else y
def _cat(ts, dim=0): return STensor(np.concatenate([_st(t).a for t in ts], axis=dim))
def _stack(ts, dim=0): return STensor(np.stack([_st(t).a for t in ts], axis=dim))
def _unsupported(name): return lambda *a, **k: (_ for _ in ()).throw(NotImplementedError(f"spec front-end: unsupported torch op {name}"))

_TORCH = {
    "conv2d": lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: conv_nd(input, weight, bias, stride, padding, dilation, groups, 2),
    "conv1d": lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: conv_nd(input, weight, bias, stride, padding, dilation, groups, 1),
    "conv3d": lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: conv_nd(input, weight, bias, stride, padding, dilation, groups, 3),
    "pad": pad, "binary_cross_entropy_with_logits": bce_with_logits,
    "max_pool1d": lambda x, *a, **k: pool_nd(x, *a, nd=1, mode="max", **k),
    "max_pool2d": lambda x, *a, **k: pool_nd(x, *a, nd=2, mode="max", **k),
    "max_pool3d": lambda x, *a, **k: pool_nd(x, *a, nd=3, mode="max", **k),
    "avg_pool1d": lambda x, *a, **k: pool_nd(x, *a, nd=1, mode="avg", **k),
    "avg_pool2d": lambda x, *a, **k: pool_nd(x, *a, nd=2, mode="avg", **k),
    "avg_pool3d": lambda x, *a, **k: pool_nd(x, *a, nd=3, mode="avg", **k),
    "adaptive_avg_pool1d": lambda x, o: adaptive_pool(x, o, 1, "avg"),
    "adaptive_avg_pool2d": lambda x, o: adaptive_pool(x, o, 2, "avg"),
    "adaptive_avg_pool3d": lambda x, o: adaptive_pool(x, o, 3, "avg"),
    "adaptive_max_pool1d": lambda x, o, **k: adaptive_pool(x, o, 1, "max"),
    "adaptive_max_pool2d": lambda x, o, **k: adaptive_pool(x, o, 2, "max"),
    "sin": lambda x: _st(x).sin(), "cos": lambda x: _st(x).cos(),
    "logsigmoid": lambda x: _st(x).log_sigmoid(), "log_sigmoid": lambda x: _st(x).log_sigmoid(),
    "flip": lambda x, dims: _st(x).flip(dims),
    "diag": lambda x, d=0: STensor(np.diag(_st(x).a, d)),
    "diagonal": lambda x, offset=0, dim1=0, dim2=1: STensor(np.diagonal(_st(x).a, offset, dim1, dim2)),
    "tril": lambda x, d=0: STensor(np.tril(_st(x).a, d)), "triu": lambda x, d=0: STensor(np.triu(_st(x).a, d)),
    "relu6": lambda x, inplace=False: _st(x).relu6(), "erf": lambda x: _st(x).erf(),
    "zeros": lambda *s, **k: STensor.full(tuple(s[0]) if len(s) == 1 and isinstance(s[0], (tuple, list)) else tuple(s), 0.0),
    "ones": lambda *s, **k: STensor.full(tuple(s[0]) if len(s) == 1 and isinstance(s[0], (tuple, list)) else tuple(s), 1.0),
    "empty": lambda *s, **k: STensor.full(tuple(s[0]) if len(s) == 1 and isinstance(s[0], (tuple, list)) else tuple(s), 0.0),
    "clamp_max": lambda x, v: _st(x).minimum(v), "item": lambda x: _st(x).item(),
    "tanh_": lambda x: _st(x).tanh(), "relu_": lambda x: _st(x).relu(), "sigmoid_": lambda x: _st(x).sigmoid(),
    "numel": lambda x: _st(x).numel(), "size": lambda x, d=None: _st(x).size(d),
    "linear": _linear, "cat": _cat, "concat": _cat, "stack": _stack,
    "sigmoid": lambda x: _st(x).sigmoid(), "tanh": lambda x: _st(x).tanh(), "abs": lambda x: _st(x).abs(),
    "log_softmax": lambda x, dim=-1, dtype=None, _stacklevel=3: _st(x).log_softmax(dim),
    "dropout": lambda x, p=0.5, training=False, inplace=False: _st(x),
    "silu": lambda x, inplace=False: _st(x) * _st(x).sigmoid(),
    "softplus": lambda x, beta=1, threshold=20: ((_st(x) * beta).exp() + 1.0).log() * (1.0 / beta),
    "leaky_relu": lambda x, negative_slope=0.01, inplace=False: _st(x).maximum(_st(x) * negative_slope) if 0 <= negative_slope <= 1 else _st(x).minimum(_st(x) * negative_slope),
    "hardtanh": lambda x, min_val=-1.0, max_val=1.0, inplace=False: _st(x).clamp(min_val, max_val),
    "elu": _unsupported("elu"), "gelu": lambda x, approximate="none": _st(x).gelu(approximate),
    "normalize": lambda x, p=2.0, dim=1, eps=1e-12: _st(x) / _st(x).norm(p, dim, keepdim=True).maximum(eps),
    "mse_loss": lambda a, b, reduction="mean", **k: _reduce_loss((_st(a) - _st(b)) ** 2, reduction),
    "l1_loss": lambda a, b, reduction="mean", **k: _reduce_loss((_st(a) - _st(b)).abs(), reduction),
    "smooth_l1_loss": _unsupported("smooth_l1_loss (piecewise)"), "huber_loss": _unsupported("huber_loss (piecewise)"),
    "flatten": lambda x, start_dim=0, end_dim=-1: _st(x).flatten(start_dim, end_dim),
    "add": lambda a, b, alpha=1: _st(a) + (_st(b) * alpha if alpha != 1 else b), "sub": lambda a, b, alpha=1: _st(a) - (_st(b) * alpha if alpha != 1 else b),
    "mul": lambda a, b: _st(a) * b, "div": lambda a, b, rounding_mode=None: _st(a) / b, "neg": lambda a: -_st(a),
    "pow": lambda a, p: _st(a) ** p,
    "min": lambda x, dim=None, keepdim=False: _st(x).minimum(dim) if isinstance(dim, (STensor, np.ndarray)) else (_scalar(functools.reduce(lambda a, b: T.app("min", a, b), _st(x).flat())) if dim is None else _MaxResult(STensor(_fold_axis(_st(x).a, dim, lambda v: functools.reduce(lambda a, b: T.app("min", a, b), v), keepdim)), None)),
    "amin": lambda x, dim=None, keepdim=False: STensor(_fold_axis(_st(x).a, dim, lambda v: functools.reduce(lambda a, b: T.app("min", a, b), v), keepdim)),
    "std": lambda x, dim=None, unbiased=True, keepdim=False, correction=None, axis=None:
        _st(x).std(axis if dim is None else dim, keepdim, unbiased),
    "norm": lambda x, p=2, dim=None, keepdim=False, dtype=None, out=None: _st(x).norm(p, dim, keepdim),
    "einsum": _unsupported("einsum"),
    "randn": _unsupported("randn (nondeterministic)"), "rand": _unsupported("rand (nondeterministic)"),
    "normal": _unsupported("normal (nondeterministic)"), "bernoulli": _unsupported("bernoulli (nondeterministic)"),
    "randn_like": _unsupported("randn_like (nondeterministic)"), "rand_like": _unsupported("rand_like (nondeterministic)"),
    "multinomial": _unsupported("multinomial (nondeterministic)"),
    "embedding": _unsupported("embedding"), "cross_entropy": _unsupported("cross_entropy"),
    "where": lambda c, a, b: _st(c).where(a, b),
    "ne": lambda a, b: _st(a).ne(b), "eq": lambda a, b: _st(a).eq(b),
    "gt": lambda a, b: _st(a).gt(b), "lt": lambda a, b: _st(a).lt(b),
    "ge": lambda a, b: _st(a).ge(b), "le": lambda a, b: _st(a).le(b),
    "not_equal": lambda a, b: _st(a).ne(b), "greater": lambda a, b: _st(a).gt(b),
    "less": lambda a, b: _st(a).lt(b), "greater_equal": lambda a, b: _st(a).ge(b),
    "less_equal": lambda a, b: _st(a).le(b),
    "logical_not": lambda a: STensor(np.frompyfunc(lambda t: T.app("not", t), 1, 1)(_st(a).a)),
    "masked_fill": lambda x, mask, v: _st(mask).where(STensor.full(_st(x).shape, float(v)), x),
    "batch_norm": _unsupported("batch_norm"), "instance_norm": _unsupported("instance_norm"), "group_norm": _unsupported("group_norm"),
    "max_pool2d": _unsupported("max_pool2d"), "avg_pool2d": _unsupported("avg_pool2d"), "interpolate": _unsupported("interpolate"),
    "zeros_like": lambda x, **k: STensor.full(_st(x).shape, 0.0), "ones_like": lambda x, **k: STensor.full(_st(x).shape, 1.0),
    "full_like": lambda x, v, **k: STensor.full(_st(x).shape, float(v)),
    "unsqueeze": lambda x, d: _st(x).unsqueeze(d), "squeeze": lambda x, d=None: _st(x).squeeze(d),
    "reshape": lambda x, *s: _st(x).reshape(*s), "permute": lambda x, *p: _st(x).permute(*p),
    "clone": lambda x, **k: _st(x), "contiguous": lambda x: _st(x), "t": lambda x: _st(x).t(),
    "chunk": lambda x, n, dim=0: _st(x).chunk(n, dim), "split": lambda x, s, dim=0: _st(x).split(s, dim),
    "rsub": lambda a, b, alpha=1: _st(b) - _st(a),
    "matmul": lambda a, b: _st(a) @ b, "mm": lambda a, b: _st(a) @ b, "bmm": lambda a, b: _st(a) @ b,
    "softmax": lambda x, dim=-1, dtype=None, _stacklevel=3: _st(x).softmax(dim),
    "_softmax": lambda x, dim=-1, half_to_float=False: _st(x).softmax(dim),
    "exp": lambda x: _st(x).exp(), "sqrt": lambda x: _st(x).sqrt(), "rsqrt": lambda x: _st(x).rsqrt(),
    "log": lambda x: _st(x).log(), "relu": lambda x, **k: _st(x).relu(),
    "sum": lambda x, dim=None, keepdim=False, dtype=None, axis=None: _st(x).sum(dim, keepdim, axis=axis),
    "mean": lambda x, dim=None, keepdim=False, dtype=None, axis=None: _st(x).mean(dim, keepdim, axis=axis),
    "amax": lambda x, dim=None, keepdim=False, axis=None: _st(x).amax(dim, keepdim, axis=axis),
    "max": lambda x, dim=None, keepdim=False: _st(x).maximum(dim) if isinstance(dim, (STensor, np.ndarray)) else _st(x).max(dim, keepdim),
    "maximum": lambda a, b: _st(a).maximum(b), "minimum": lambda a, b: _st(a).minimum(b),
    "transpose": lambda x, i=None, j=None, dim0=None, dim1=None: _st(x).transpose(dim0 if i is None else i, dim1 if j is None else j), "layer_norm": layer_norm,
    "square": lambda x: _st(x).square(), "var": lambda x, dim=None, **k: _st(x).var(dim, k.get("keepdim", False)),
    "clamp_min": lambda x, v: _st(x).maximum(v), "clamp": lambda x, min=None, max=None: (_st(x).maximum(min) if min is not None else _st(x)).minimum(max) if max is not None else (_st(x).maximum(min) if min is not None else _st(x)),
}

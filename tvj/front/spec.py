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
import functools, re
import numpy as np
from tvj.core import terms as T
from tvj.decide import delegate as _DEL
from tvj.core import bounded as _B

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
    k = _B.limit(flat.shape[-1])            # bounded.py: the other way to bound
    for i in range(flat.shape[0]):
        out[i] = fn(list(flat[i])[:k])
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
    def __getitem__(self, idx):
        # x[idx] where idx is itself a tensor is an indirect read, not a slice
        if isinstance(idx, STensor) or (isinstance(idx, np.ndarray) and idx.dtype == object):
            return take(self, idx, 0)
        r = self.a[idx]; return STensor(r) if isinstance(r, np.ndarray) else r
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
    def __matmul__(self, o):
        b = _arr(o)
        # A matmul both sides delegate to cuBLAS is compared by congruence rather
        # than denoted -- see delegate.py.  Only the (..., K) @ (K, N) shape is
        # canonicalised; batched matmul falls through to the dense denotation.
        if self.a.ndim >= 2 and b.ndim == 2:
            M = int(np.prod(self.a.shape[:-1])); K = self.a.shape[-1]
            r = _DEL.matmul(self.a.reshape(M, K), b)
            if r is not None: return STensor(r.reshape(self.a.shape[:-1] + (b.shape[1],)))
        return STensor(dense_matmul(self.a, b))
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
        if p in (2, 2.0, "fro", None): return (self * self).sum(dim, keepdim).sqrt()
        if p in (1, 1.0): return self.abs().sum(dim, keepdim)
        if p == float("inf"): return self.abs().amax(dim, keepdim) if dim is not None else self.abs().amax()
        raise NotImplementedError(f"spec front-end: unsupported torch op norm(p={p})")
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
    # `==` is the ONE comparison that used to fall through to object identity: it
    # returned Python `False`, so `attn.masked_fill(mask == 0, -1e9)` built
    # `select(0.0, -1e9, attn)` -- a reference with the mask silently deleted.
    # Every other missing dunder (`~`, `&`, `//`, `abs`, `float`) raises TypeError
    # and is reported as SPEC-ERROR; this one was the only quiet one.
    def __eq__(self, o): return self._cmp("eq", o)
    __hash__ = object.__hash__          # __eq__ would otherwise make STensor unhashable
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
        USED.add(name)                       # which references this spec rests on
        h = _TORCH.get(name)
        if h is None: raise NotImplementedError(f"spec front-end: unsupported torch op {name}")
        # only kwargs that cannot change the value may be dropped; anything else is a spec error
        HARMLESS = HARMLESS_KWARGS
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

def _chan(v, ndim, C):
    """A per-channel vector, shaped so it broadcasts over (N, C, *spatial)."""
    return _st(v).reshape((1, C) + (1,) * (ndim - 2))

def batch_norm(x, running_mean=None, running_var=None, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    """Eval mode is the running statistics; training mode is the BIASED batch
    variance over every dimension but the channel one (torch normalises with the
    biased estimate and only the running buffer gets the unbiased correction)."""
    x = _st(x); C = x.shape[1] if x.ndim > 1 else x.shape[0]
    if training or running_mean is None or running_var is None:
        dims = tuple(i for i in range(x.ndim) if i != 1)
        mu = x
        for d in dims: mu = mu.mean(d, keepdim=True)
        dv = x - mu
        var = dv * dv
        for d in dims: var = var.mean(d, keepdim=True)
        y = dv * (var + eps).rsqrt()
    else:
        y = (x - _chan(running_mean, x.ndim, C)) * (_chan(running_var, x.ndim, C) + eps).rsqrt()
    if weight is not None: y = y * _chan(weight, x.ndim, C)
    if bias is not None: y = y + _chan(bias, x.ndim, C)
    return y

def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    x = _st(x); N, C, rest = x.shape[0], x.shape[1], x.shape[2:]
    g = x.reshape((N, num_groups, -1))
    mu = g.mean(2, keepdim=True); d = g - mu
    var = (d * d).mean(2, keepdim=True)
    y = (d * (var + eps).rsqrt()).reshape((N, C) + tuple(rest))
    if weight is not None: y = y * _chan(weight, x.ndim, C)
    if bias is not None: y = y + _chan(bias, x.ndim, C)
    return y

def instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5):
    x = _st(x); N, C, rest = x.shape[0], x.shape[1], x.shape[2:]
    if use_input_stats or running_mean is None or running_var is None:
        g = x.reshape((N * C, -1))
        mu = g.mean(1, keepdim=True); d = g - mu
        var = (d * d).mean(1, keepdim=True)
        y = (d * (var + eps).rsqrt()).reshape((N, C) + tuple(rest))
    else:
        y = (x - _chan(running_mean, x.ndim, C)) * (_chan(running_var, x.ndim, C) + eps).rsqrt()
    if weight is not None: y = y * _chan(weight, x.ndim, C)
    if bias is not None: y = y + _chan(bias, x.ndim, C)
    return y

def smooth_l1_loss(a, b, size_average=None, reduce=None, reduction="mean", beta=1.0):
    if beta == 0: return _reduce_loss((_st(a) - _st(b)).abs(), reduction)
    d = _st(a) - _st(b); ad = d.abs()
    return _reduce_loss(ad.lt(beta).where((d * d) * (0.5 / beta), ad - 0.5 * beta), reduction)

def huber_loss(a, b, reduction="mean", delta=1.0, weight=None):
    if weight is not None: raise NotImplementedError("spec front-end: unsupported torch op huber_loss(weight=)")
    d = _st(a) - _st(b); ad = d.abs()
    return _reduce_loss(ad.lt(delta).where((d * d) * 0.5, (ad - 0.5 * delta) * delta), reduction)

def dense_matmul(a, b):
    """(..., M, K) @ (K, N) or batched (B, M, K) @ (B, K, N), built directly.

    `np.matmul` on an object array is 16x slower than this for the same result
    (measured, 64x128x64): it dispatches `+` and `*` through numpy per element and
    builds a chain of binary Adds, where the term algebra wants one n-ary Add and
    normalises to it anyway.  This is the hot loop of the whole front-end -- and
    the only loop the traces corpus reaches, since an LLM-written kernel spells
    the matmul out in Triton rather than delegating it."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 2 and b.ndim == 2:
        M, K = a.shape; N = b.shape[1]
        kk = _B.limit(K)
        out = np.empty((M, N), dtype=object)
        for i in range(M):
            ai = a[i]
            for j in range(N):
                bj = b[:, j]
                out[i, j] = T.add(*[T.mul(ai[k], bj[k]) for k in range(kk)])
        return out
    if a.ndim > 2 and b.ndim == 2:
        lead, K = a.shape[:-1], a.shape[-1]
        r = dense_matmul(a.reshape(-1, K), b)
        return r.reshape(lead + (b.shape[1],))
    if a.ndim == 3 and b.ndim == 3 and a.shape[0] == b.shape[0]:
        return np.stack([dense_matmul(a[i], b[i]) for i in range(a.shape[0])])
    return np.matmul(a, b)               # anything else: rare, and small


# An indirect read is written out over the whole axis, so its cost is
# (slots on the axis) x (output positions).  `sexec` caps the kernel side at
# Interp.GATHER_EXPAND; without the same cap here an embedding over a real
# vocabulary builds tens of millions of nodes and takes the process down instead
# of returning a bucket.
TAKE_EXPAND = 1 << 20


def _is_mask(arr):
    """Is this an index tensor, or the result of a comparison?

    `x[x > 0]` hands `__getitem__` a tensor of `cmp:` terms.  Treating that as an
    INDEX silently computes a different function -- and before indirect reads
    existed it raised, which was better."""
    for t in np.asarray(arr).reshape(-1)[:8]:
        fn = getattr(t, "fn", "")
        if fn.startswith("cmp:") or fn in ("and", "or", "not", "true", "false"):
            return True
    return False


def gather_nd(x, dim, index):
    """`torch.gather`: the index has the OUTPUT's shape and picks along `dim` at
    each position -- `out[i][j] = x[i][index[i][j]]` for dim=1.  Not the same as
    `index_select`, which takes whole slices; binding the two to one function
    made the reference silently compute something else for any input above 1-D."""
    x = _st(x); ix = _st(index)
    if dim < 0: dim += x.ndim
    if ix.a.ndim != x.a.ndim:
        raise NotImplementedError("spec front-end: unsupported torch op gather (rank mismatch)")
    n = x.a.shape[dim]
    if n * ix.a.size > TAKE_EXPAND:
        raise NotImplementedError(f"spec front-end: gather too large to expand "
                                  f"({n} x {ix.a.size} > {TAKE_EXPAND})")
    out = np.empty(ix.a.shape, dtype=object)
    for pos in np.ndindex(*ix.a.shape):
        rows = [x.a[pos[:dim] + (m,) + pos[dim + 1:]] for m in range(n)]
        out[pos] = T.gather(rows, ix.a[pos])
    return STensor(out)


def take(x, index, axis=0):
    """`x` indexed along `axis` by a tensor of symbolic indices.

    Every torch spelling of an indirect read -- `x[idx]`, `torch.gather`,
    `index_select`, `F.embedding` -- lands here, and here calls `T.gather`, which
    is the same denotation `sexec` gives `tl.load(src + j)`.  That is the whole
    point: a gather the two sides spell differently can be represented and never
    judged, so there is one function."""
    x = _st(x); ix = _st(index)
    if _is_mask(ix.a):
        raise NotImplementedError("spec front-end: unsupported torch op boolean-mask indexing")
    if axis < 0: axis += x.ndim
    src = np.moveaxis(x.a, axis, 0)                  # (n, *rest)
    n = src.shape[0]
    if n * ix.a.size * int(np.prod(src.shape[1:] or (1,))) > TAKE_EXPAND:
        raise NotImplementedError(f"spec front-end: indirect read too large to expand "
                                  f"({n} slots x {ix.a.size} positions)")
    rest = src.shape[1:]
    out = np.empty(ix.a.shape + rest, dtype=object)
    flat_i = ix.a.reshape(-1)
    view = out.reshape((flat_i.size,) + rest)
    for k, j in enumerate(flat_i):
        for pos in (np.ndindex(*rest) if rest else [()]):
            view[(k,) + pos] = T.gather([src[(m,) + pos] for m in range(n)], j)
    return STensor(np.moveaxis(out.reshape(ix.a.shape + rest), 0, axis)
                   if ix.a.ndim == 1 and axis else out)


def scatter_put(base, dim, index, source):
    """`base.scatter_(dim, index, src)` -- the reference side of a plain scatter.

    Written with the same nesting the interpreter uses, and carrying the same
    caveat: torch documents the result as nondeterministic when two indices
    collide, so this is well defined exactly where the kernel's version is."""
    b = _st(base); ix = _st(index); src = _st(source)
    if dim < 0: dim += b.ndim
    if b.ndim != 1 or ix.a.ndim != 1 or src.a.ndim != 1 or dim != 0:
        raise NotImplementedError("spec front-end: unsupported torch op scatter_ (only 1-D dim 0)")
    idxs = list(ix.a.reshape(-1)); vals = list(src.a.reshape(-1))
    out = np.empty(b.a.shape, dtype=object)
    for j in range(b.a.shape[0]):
        acc = T.lift(b.a[j])
        for k, v in reversed(list(zip(idxs, vals))):
            acc = T.select(T.cmp("eq", T.lift(k), T.const(float(j))), T.lift(v), acc)
        out[j] = acc
    return STensor(out)


def scatter_add(base, dim, index, source, alpha=1):
    """`base.index_add_(dim, index, source)` -- the reference side of an atomic
    scatter-add.  Order-free by construction, so it needs no assumption about
    which lane arrives first, and it is written the way `sexec` writes it."""
    b = _st(base); ix = _st(index); src = _st(source)
    if dim < 0: dim += b.ndim
    if b.ndim != 1 or ix.a.ndim != 1 or src.a.ndim != 1 or dim != 0:
        raise NotImplementedError("spec front-end: unsupported torch op index_add (only 1-D dim 0)")
    idxs = list(ix.a.reshape(-1)); vals = list(src.a.reshape(-1))
    out = np.empty(b.a.shape, dtype=object)
    for j in range(b.a.shape[0]):
        parts = [T.select(T.cmp("eq", T.lift(k), T.const(float(j))),
                          T.mul(T.const(float(alpha)), T.lift(v)) if alpha != 1 else T.lift(v),
                          T.ZERO)
                 for k, v in zip(idxs, vals)]
        out[j] = T.add(T.lift(b.a[j]), *parts) if parts else T.lift(b.a[j])
    return STensor(out)


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
    cig, cog = Ci // groups, Co // groups
    bb = _st(b).a if b is not None else None

    def dense(with_bias=True):
        out = np.empty((N, Co, *out_sp), dtype=object)
        for n in range(N):
            for co in range(Co):
                g = co // cog
                for pos in itertools.product(*[range(s) for s in out_sp]):
                    terms = []
                    for ci in range(cig):
                        for kpos in itertools.product(*[range(k) for k in ks]):
                            idx = tuple(pos[dd] * stride[dd] + kpos[dd] * dilation[dd] for dd in range(nd))
                            terms.append(T.mul(w[(co, ci) + kpos], xp[(n, g * cig + ci) + idx]))
                    if bb is not None and with_bias: terms.append(bb[co])
                    out[(n, co) + pos] = T.add(*terms)
        return out

    # Both sides reach here -- the module through F.conv2d, the generated code
    # through extern_kernels.convolution -- so delegating inside conv_nd makes the
    # two agree by construction.  The BIAS stays outside the symbol for the same
    # reason it stays outside a matmul's: Inductor routinely emits
    # `convolution(x, w, None)` and adds the bias in a fused Triton kernel
    # afterwards, and the two spellings agree only once it is outside.
    def dense_nobias():
        return dense(with_bias=False)
    dl = _DEL.conv(x, w, None, stride, padding, dilation, groups, nd,
                   (N, Co, *out_sp), dense=dense_nobias)
    if dl is None: return STensor(dense())
    if bb is None: return STensor(dl)
    out = np.empty(dl.shape, dtype=object)
    for co in range(Co): out[:, co] = np.frompyfunc(lambda t, c=bb[co]: T.add(t, c), 1, 1)(dl[:, co])
    return STensor(out)

def pool_nd(x, kernel_size, stride=None, padding=0, nd=2, mode="max", ceil_mode=False,
            count_include_pad=True, divisor_override=None, dilation=1, **_):
    """Windowed max/avg reduction, written the same way conv_nd is."""
    import itertools, functools
    x = _st(x).a
    ks = _tup(kernel_size, nd); st = _tup(stride if stride is not None else kernel_size, nd)
    pd = _tup(padding, nd); dl = _tup(dilation, nd)
    # a dilated window spans (k-1)*d + 1 with holes; the output size and every
    # offset below are in terms of that effective extent
    ke = [(ks[d] - 1) * dl[d] + 1 for d in range(nd)]
    fill = T.const(float("-inf")) if mode == "max" else T.ZERO
    isz = [x.shape[x.ndim - nd + d] for d in range(nd)]
    # ceil_mode rounds the window count UP, then drops the last window if it would
    # begin past the input and its left padding -- torch's own rule.  Declaring the
    # flag and computing a floor anyway is the silent-kwarg hole `axis=` was.
    sp_ = []
    for d in range(nd):
        span = isz[d] + 2 * pd[d] - ke[d]
        n = (-(-span // st[d]) if ceil_mode else span // st[d]) + 1
        if ceil_mode and (n - 1) * st[d] >= isz[d] + pd[d]: n -= 1
        sp_.append(n)
    # a ceil-rounded last window can reach past the padded tensor: extend it
    extra = [max(0, (sp_[d] - 1) * st[d] + ke[d] - (isz[d] + 2 * pd[d])) for d in range(nd)]
    xp = np.pad(x, [(0, 0)] * (x.ndim - nd) + [(pd[d], pd[d] + extra[d]) for d in range(nd)],
                mode="constant", constant_values=fill)
    lead = xp.shape[:x.ndim - nd]
    out = np.empty(lead + tuple(sp_), dtype=object)
    for pre in itertools.product(*[range(d) for d in lead]):
        for pos in itertools.product(*[range(s) for s in sp_]):
            vals = [xp[pre + tuple(pos[d] * st[d] + o[d] * dl[d] for d in range(nd))]
                    for o in itertools.product(*[range(k) for k in ks])]
            if mode == "max":
                out[pre + pos] = functools.reduce(lambda a, b: T.app("max", a, b), vals)
            else:
                # The divisor is torch's, per dimension and per window.  With
                # count_include_pad the window is clipped at input+padding (so the
                # ceil-mode overhang never counts); without it, at the input itself.
                n = 1
                for d in range(nd):
                    lo = pos[d] * st[d] - pd[d]
                    hi = min(lo + ke[d], isz[d] + pd[d]) if count_include_pad \
                         else min(lo + ke[d], isz[d])
                    if not count_include_pad: lo = max(lo, 0)
                    n *= max(hi - lo, 0)
                if divisor_override: n = divisor_override
                out[pre + pos] = T.mul(T.const(1.0 / max(n, 1)), T.add(*vals))
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
    # ragged: torch's window for output index j along a dimension of length n with o
    # outputs is [floor(j*n/o), ceil((j+1)*n/o)) -- unequal widths, so it is not a
    # strided pool and has to be written out.
    import itertools, functools
    a = x.a
    lead = a.shape[:a.ndim - nd]
    out = np.empty(lead + tuple(osz), dtype=object)
    spans = [[( (j * isz[d]) // osz[d], -(-((j + 1) * isz[d]) // osz[d]) )
              for j in range(osz[d])] for d in range(nd)]
    for pre in itertools.product(*[range(k) for k in lead]):
        for pos in itertools.product(*[range(o) for o in osz]):
            rng = [range(*spans[d][pos[d]]) for d in range(nd)]
            vals = [a[pre + idx] for idx in itertools.product(*rng)]
            out[pre + pos] = functools.reduce(lambda p, q: T.app("max", p, q), vals) if mode == "max" \
                else T.mul(T.const(1.0 / len(vals)), T.add(*vals))
    return STensor(out)

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
# The single list of parameters a handler may ignore: they select a dtype, a
# device, or an aliasing convention, none of which changes the real number.
# `spec_sigcheck.py` reads THIS -- keeping a second copy there is how the two
# drifted apart the first time.
HARMLESS_KWARGS = frozenset({
    "out", "dtype", "layout", "device", "requires_grad", "pin_memory", "memory_format",
    "inplace", "size_average", "reduce", "non_blocking", "copy", "generator",
    "half_to_float", "_stacklevel", "return_indices", "async_op", "sparse_grad",
    # `momentum` only rewrites running_mean / running_var in place; the value the
    # layer RETURNS does not depend on it.  Harmless exactly because the front-end
    # does not model that mutation -- see semantics.py decision `spec.inplace`.
    "momentum",
    # gradient-only flags.  This front-end computes forward values, and none of
    # these can change one: `padding_idx` excludes a row from the GRADIENT (the
    # row is still returned), and `scale_grad_by_freq` / `sparse` / `norm_type`
    # select how the gradient is accumulated or normalised.  `norm_type` is only
    # read alongside `max_norm`, which `embedding` refuses outright.
    "padding_idx", "scale_grad_by_freq", "sparse", "norm_type",
})

# --------------------------------------------------------------------------
# Where each reference came from.
#
# `semantics.py` grades every interpreter decision by its basis; the reference
# side needs the same grading for the same reason.  A handler transcribed from a
# published formula is a DEFINITION: if a kernel disagrees with it, the kernel is
# wrong.  A handler whose edge behaviour was recovered by reading torch's C++ is
# INFERRED: if a kernel disagrees, one of us is wrong and torch decides -- so a
# FAIL resting on one of these is a weaker claim, and should be read that way.
#
# Only the second list is enumerated.  Claiming a grade for all 134 handlers
# would be guessing; these are the ones where the archaeology actually happened
# and the reason is recorded.
INFERRED_FROM_IMPL = {
    "avg_pool1d": "divisor at the border: torch clips the window at input+padding "
                  "with count_include_pad and at the input without it, so the "
                  "ceil-mode overhang never counts (ATen AvgPoolKernel)",
    "avg_pool2d": "same divisor rule as avg_pool1d",
    "avg_pool3d": "same divisor rule as avg_pool1d",
    "max_pool1d": "ceil_mode drops the last window when it would start past "
                  "input+left-padding -- a rule stated only in the C++",
    "max_pool2d": "same ceil_mode rule as max_pool1d",
    "max_pool3d": "same ceil_mode rule as max_pool1d",
    "adaptive_avg_pool1d": "ragged split is floor(j*n/o) .. ceil((j+1)*n/o)",
    "adaptive_avg_pool2d": "same ragged split",
    "adaptive_avg_pool3d": "same ragged split",
    "adaptive_max_pool1d": "same ragged split",
    "adaptive_max_pool2d": "same ragged split",
    "batch_norm": "training mode normalises with the BIASED variance; only the "
                  "running buffer gets the n/(n-1) correction",
    "instance_norm": "same biased-variance choice as batch_norm",
    "group_norm": "same biased-variance choice as batch_norm",
    "gelu": "the tanh approximation's constants are torch's, not a definition",
    "softplus": "the `beta*x > threshold` guard returns x exactly; the crossover "
                "is torch's choice, and the two branches are not equal over the reals",
}

# Everything else is a transcription of the op's published definition; where that
# definition is itself a formula with a free choice (a loss reduction, an eps
# placement) the handler takes it as an argument rather than fixing it.
# Ops the front-end was asked for while building the current spec.  The judge
# clears it per row and reads it back, so a FAIL can say what its reference was.
USED = set()


def basis_of_run(ops=None):
    """The weakest grade any reference used in this run carries."""
    ops = USED if ops is None else ops
    inf = sorted(o for o in ops if o in INFERRED_FROM_IMPL)
    return ("inferred-from-impl" if inf else "definition"), inf


def provenance(op):
    return ("inferred-from-impl", INFERRED_FROM_IMPL[op]) if op in INFERRED_FROM_IMPL \
        else ("definition", "")


def _unsupported(name): return lambda *a, **k: (_ for _ in ()).throw(NotImplementedError(f"spec front-end: unsupported torch op {name}"))

_TORCH = {
    "conv2d": lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: conv_nd(input, weight, bias, stride, padding, dilation, groups, 2),
    "conv1d": lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: conv_nd(input, weight, bias, stride, padding, dilation, groups, 1),
    "conv3d": lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: conv_nd(input, weight, bias, stride, padding, dilation, groups, 3),
    "pad": pad, "binary_cross_entropy_with_logits": bce_with_logits,
    "max_pool1d": lambda x, kernel_size, stride=None, padding=0, dilation=1,
                           ceil_mode=False, return_indices=False:
        pool_nd(x, kernel_size, stride, padding, nd=1, mode="max", ceil_mode=ceil_mode,
                dilation=dilation),
    "max_pool2d": lambda x, kernel_size, stride=None, padding=0, dilation=1,
                           ceil_mode=False, return_indices=False:
        pool_nd(x, kernel_size, stride, padding, nd=2, mode="max", ceil_mode=ceil_mode,
                dilation=dilation),
    "max_pool3d": lambda x, kernel_size, stride=None, padding=0, dilation=1,
                           ceil_mode=False, return_indices=False:
        pool_nd(x, kernel_size, stride, padding, nd=3, mode="max", ceil_mode=ceil_mode,
                dilation=dilation),
    "avg_pool1d": lambda x, kernel_size, stride=None, padding=0, ceil_mode=False,
                           count_include_pad=True, divisor_override=None:
        pool_nd(x, kernel_size, stride, padding, nd=1, mode="avg", ceil_mode=ceil_mode,
                count_include_pad=count_include_pad, divisor_override=divisor_override),
    "avg_pool2d": lambda x, kernel_size, stride=None, padding=0, ceil_mode=False,
                           count_include_pad=True, divisor_override=None:
        pool_nd(x, kernel_size, stride, padding, nd=2, mode="avg", ceil_mode=ceil_mode,
                count_include_pad=count_include_pad, divisor_override=divisor_override),
    "avg_pool3d": lambda x, kernel_size, stride=None, padding=0, ceil_mode=False,
                           count_include_pad=True, divisor_override=None:
        pool_nd(x, kernel_size, stride, padding, nd=3, mode="avg", ceil_mode=ceil_mode,
                count_include_pad=count_include_pad, divisor_override=divisor_override),
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
    "log_softmax": lambda x, dim=-1, _stacklevel=3, dtype=None: _st(x).log_softmax(dim),
    "dropout": lambda x, p=0.5, training=False, inplace=False:
        _st(x) if (not training or p == 0) else _unsupported("dropout(training=True)")(),
    "silu": lambda x, inplace=False: _st(x) * _st(x).sigmoid(),
    "softplus": lambda x, beta=1, threshold=20:
        (_st(x) * beta).gt(threshold).where(_st(x), ((_st(x) * beta).exp() + 1.0).log() * (1.0 / beta)),
    "leaky_relu": lambda x, negative_slope=0.01, inplace=False: _st(x).maximum(_st(x) * negative_slope) if 0 <= negative_slope <= 1 else _st(x).minimum(_st(x) * negative_slope),
    "hardtanh": lambda x, min_val=-1.0, max_val=1.0, inplace=False: _st(x).clamp(min_val, max_val),
    "elu": lambda x, alpha=1.0, inplace=False: _st(x).gt(0.0).where(_st(x), (_st(x).exp() - 1.0) * alpha),
    "hardsigmoid": lambda x, inplace=False: (_st(x) * (1.0 / 6.0) + 0.5).clamp(0.0, 1.0),
    "threshold": lambda x, threshold, value, inplace=False: _st(x).gt(threshold).where(_st(x), value),
    "_threshold": lambda x, threshold, value, inplace=False: _st(x).gt(threshold).where(_st(x), value),
    "gelu": lambda x, approximate="none": _st(x).gelu(approximate),
    "normalize": lambda x, p=2.0, dim=1, eps=1e-12: _st(x) / _st(x).norm(p, dim, keepdim=True).maximum(eps),
    "mse_loss": lambda a, b, size_average=None, reduce=None, reduction="mean", weight=None:
        _reduce_loss((_st(a) - _st(b)) ** 2, reduction) if weight is None
        else _unsupported("mse_loss(weight=)")(),
    "l1_loss": lambda a, b, size_average=None, reduce=None, reduction="mean", weight=None:
        _reduce_loss((_st(a) - _st(b)).abs(), reduction) if weight is None
        else _unsupported("l1_loss(weight=)")(),
    "smooth_l1_loss": smooth_l1_loss, "huber_loss": huber_loss,
    "flatten": lambda x, start_dim=0, end_dim=-1: _st(x).flatten(start_dim, end_dim),
    "add": lambda a, b, alpha=1: _st(a) + (_st(b) * alpha if alpha != 1 else b), "sub": lambda a, b, alpha=1: _st(a) - (_st(b) * alpha if alpha != 1 else b),
    "mul": lambda a, b: _st(a) * b, "div": lambda a, b, rounding_mode=None: _st(a) / b if rounding_mode is None
        else _unsupported(f"div(rounding_mode={rounding_mode})")(), "neg": lambda a: -_st(a),
    "pow": lambda a, p: _st(a) ** p,
    "min": lambda x, dim=None, keepdim=False: _st(x).minimum(dim) if isinstance(dim, (STensor, np.ndarray)) else (_scalar(functools.reduce(lambda a, b: T.app("min", a, b), _st(x).flat())) if dim is None else _MaxResult(STensor(_fold_axis(_st(x).a, dim, lambda v: functools.reduce(lambda a, b: T.app("min", a, b), v), keepdim)), None)),
    "amin": lambda x, dim=None, keepdim=False: STensor(_fold_axis(_st(x).a, dim, lambda v: functools.reduce(lambda a, b: T.app("min", a, b), v), keepdim)),
    "std": lambda x, dim=None, unbiased=True, keepdim=False, correction=None, axis=None:
        _st(x).std(axis if dim is None else dim, keepdim, unbiased, correction),
    "norm": lambda x, p=2, dim=None, keepdim=False, out=None, dtype=None: _st(x).norm(p, dim, keepdim),
    "einsum": _unsupported("einsum"),
    "randn": _unsupported("randn (nondeterministic)"), "rand": _unsupported("rand (nondeterministic)"),
    "normal": _unsupported("normal (nondeterministic)"), "bernoulli": _unsupported("bernoulli (nondeterministic)"),
    "randn_like": _unsupported("randn_like (nondeterministic)"), "rand_like": _unsupported("rand_like (nondeterministic)"),
    "multinomial": _unsupported("multinomial (nondeterministic)"),
    # every indirect read goes through take() -> T.gather, the denotation the
    # interpreter gives `tl.load(src + j)`
    "embedding": lambda idx, weight, padding_idx=None, max_norm=None, norm_type=2.0,
                        scale_grad_by_freq=False, sparse=False:
        take(weight, idx, 0) if max_norm is None
        else _unsupported("embedding(max_norm=)")(),
    "gather": lambda x, dim, index, sparse_grad=False, out=None: gather_nd(x, dim, index),
    # the reference side of a scatter-add.  Same order-free statement the
    # interpreter builds for `tl.atomic_add(dst + idx, v)`:
    #     out[j] = base[j] + sum_i select(idx_i = j, src_i, 0)
    "index_add": lambda x, dim, index, source, alpha=1: scatter_add(x, dim, index, source, alpha),
    "index_add_": lambda x, dim, index, source, alpha=1: scatter_add(x, dim, index, source, alpha),
    "scatter_add": lambda x, dim, index, src: scatter_add(x, dim, index, src),
    "scatter": lambda x, dim, index, src, reduce=None:
        scatter_put(x, dim, index, src) if reduce is None
        else _unsupported(f"scatter(reduce={reduce})")(),
    "scatter_": lambda x, dim, index, src, reduce=None:
        scatter_put(x, dim, index, src) if reduce is None
        else _unsupported(f"scatter_(reduce={reduce})")(),
    "index_copy": lambda x, dim, index, source: scatter_put(x, dim, index, source),
    "index_copy_": lambda x, dim, index, source: scatter_put(x, dim, index, source),
    "scatter_add_": lambda x, dim, index, src: scatter_add(x, dim, index, src),
    "index_select": lambda x, dim, index, out=None: take(x, index, dim),
    "take": lambda x, index: take(_st(x).reshape(-1), index, 0),
    "cross_entropy": _unsupported("cross_entropy"),
    "where": lambda c, a, b: _st(c).where(a, b),
    "ne": lambda a, b: _st(a).ne(b), "eq": lambda a, b: _st(a).eq(b),
    "gt": lambda a, b: _st(a).gt(b), "lt": lambda a, b: _st(a).lt(b),
    "ge": lambda a, b: _st(a).ge(b), "le": lambda a, b: _st(a).le(b),
    "not_equal": lambda a, b: _st(a).ne(b), "greater": lambda a, b: _st(a).gt(b),
    "less": lambda a, b: _st(a).lt(b), "greater_equal": lambda a, b: _st(a).ge(b),
    "less_equal": lambda a, b: _st(a).le(b),
    "logical_not": lambda a: STensor(np.frompyfunc(lambda t: T.app("not", t), 1, 1)(_st(a).a)),
    "masked_fill": lambda x, mask, v: _st(mask).where(STensor.full(_st(x).shape, float(v)), x),
    "batch_norm": batch_norm, "instance_norm": instance_norm, "group_norm": group_norm,
    "interpolate": _unsupported("interpolate"),
    "zeros_like": lambda x, **k: STensor.full(_st(x).shape, 0.0), "ones_like": lambda x, **k: STensor.full(_st(x).shape, 1.0),
    "full_like": lambda x, fill_value, **k: STensor.full(_st(x).shape, float(fill_value)),
    "unsqueeze": lambda x, dim: _st(x).unsqueeze(dim), "squeeze": lambda x, d=None: _st(x).squeeze(d),
    "reshape": lambda x, *s: _st(x).reshape(*s), "permute": lambda x, *p: _st(x).permute(*p),
    "clone": lambda x, **k: _st(x), "contiguous": lambda x: _st(x), "t": lambda x: _st(x).t(),
    "chunk": lambda x, n, dim=0: _st(x).chunk(n, dim), "split": lambda x, s, dim=0: _st(x).split(s, dim),
    "rsub": lambda a, b, alpha=1: _st(b) - (_st(a) * alpha if alpha != 1 else _st(a)),
    "matmul": lambda a, b: _st(a) @ b,
    # `out_dtype` changes the accumulation type, which is the precision obligation's
    # whole subject: it may not be silently ignored.
    "mm": lambda a, b, out_dtype=None: _st(a) @ b if out_dtype is None else _unsupported("mm(out_dtype=)")(),
    "bmm": lambda a, b, out_dtype=None: _st(a) @ b if out_dtype is None else _unsupported("bmm(out_dtype=)")(),
    "softmax": lambda x, dim=-1, _stacklevel=3, dtype=None: _st(x).softmax(dim),
    "_softmax": lambda x, dim=-1, half_to_float=False: _st(x).softmax(dim),
    "exp": lambda x: _st(x).exp(), "sqrt": lambda x: _st(x).sqrt(), "rsqrt": lambda x: _st(x).rsqrt(),
    "log": lambda x: _st(x).log(), "relu": lambda x, inplace=False: _st(x).relu(),
    "sum": lambda x, dim=None, keepdim=False, dtype=None, axis=None: _st(x).sum(dim, keepdim, axis=axis),
    "mean": lambda x, dim=None, keepdim=False, dtype=None, axis=None, keepdims=None:
        _st(x).mean(dim, keepdim if keepdims is None else keepdims, axis=axis),
    "amax": lambda x, dim=None, keepdim=False, axis=None: _st(x).amax(dim, keepdim, axis=axis),
    "max": lambda x, dim=None, keepdim=False: _st(x).maximum(dim) if isinstance(dim, (STensor, np.ndarray)) else _st(x).max(dim, keepdim),
    "maximum": lambda a, b: _st(a).maximum(b), "minimum": lambda a, b: _st(a).minimum(b),
    "transpose": lambda x, dim0=None, dim1=None: _st(x).transpose(dim0, dim1), "layer_norm": layer_norm,
    "square": lambda x: _st(x).square(),
    "var": lambda x, dim=None, unbiased=True, keepdim=False, correction=None, axis=None:
        _st(x).var(axis if dim is None else dim, keepdim, unbiased, correction),
    "clamp_min": lambda x, v: _st(x).maximum(v), "clamp": lambda x, min=None, max=None: (_st(x).maximum(min) if min is not None else _st(x)).minimum(max) if max is not None else (_st(x).maximum(min) if min is not None else _st(x)),
}

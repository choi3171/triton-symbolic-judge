"""Cover the part of a generated wrapper that is not a Triton kernel.

LLM-written wrappers routinely finish the computation in PyTorch -- the classic
case is a two-stage reduction where a kernel fills `partial_sums` and the tail
is `partial_sums.sum() / N`.  A kernel-level judge sees only the first stage and
has to answer UNKNOWN.

We already own a torch->term interpreter: `spec._TORCH`, the table the spec
front-end dispatches through.  So record every torch op the wrapper performs,
then replay that trace symbolically, seeded with (a) the inputs and parameters
and (b) whatever the kernels wrote.  Ops we cannot interpret are skipped; the
replay succeeds only if the output tensor ends up with a term.
"""
import numpy as np, torch
from torch.overrides import TorchFunctionMode
from tvj.core import terms as T
from tvj.front.spec import STensor, _TORCH
from tvj.front.capture import base_of, root_storage, physical_offsets

# pure metadata: recording them is noise
_IGNORE = {"data_ptr", "__get__", "__set__", "numel", "size", "dim", "stride", "storage_offset",
           "element_size", "is_contiguous", "get_device", "is_floating_point", "untyped_storage",
           "device", "dtype", "shape", "requires_grad_", "__repr__", "_has_compatible_shallow_copy_type"}

# Generated code does not only call Triton kernels and `extern_kernels.*`; it also
# calls `torch.ops.aten.*` directly, and those writes were invisible to the
# symbolic run.  Three KernelBook rows reported "the output depends on a buffer no
# launch wrote" for a buffer that an ordinary aten pooling call had filled -- a
# false FAIL that only the hardware gate, or this, catches.
#
# These are the ones we can model, and they are exactly the ones `extern_kernels`
# does NOT provide, so recording them here cannot double-count a call the extern
# wrapper already saw.
AS_EXTERN = {
    "avg_pool1d.default", "avg_pool2d.default", "avg_pool3d.default",
    "max_pool1d_with_indices.default", "max_pool2d_with_indices.default",
    "max_pool3d_with_indices.default",
}

class TorchTrace(TorchFunctionMode):
    """Records (name, args, kwargs, result) for every torch op, holding strong
    references so `id()` stays a valid key for the replay.

    Ops in `AS_EXTERN` are additionally pushed onto `capture._calls` at the moment
    they run, so they take their real place in the launch order."""
    def __init__(self): self.events = []
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", str(func))
        if name == "item" and type(out) is float: out = TracedFloat(out)
        if name not in _IGNORE: self.events.append((name, args, kwargs, out))
        if name in AS_EXTERN:
            from tvj.front import capture
            res = out[0] if isinstance(out, (tuple, list)) and out else out
            capture._calls.append(("extern", name, args, dict(kwargs), res))
        return out

class _Missing: pass
MISSING = _Missing()

def tensor_terms(t, roles, grid):
    """The STensor for `t` if the kernels wrote every element of it."""
    role = roles.get(root_storage(base_of(t)))
    if role is None: return None
    terms = []
    for p in physical_offsets(t):
        v = grid.store.get((role, p))
        if v is None: return None
        terms.append(v)
    a = np.empty(len(terms), dtype=object); a[:] = terms
    return STensor(a.reshape(tuple(t.shape)))

_CASTS = {"to", "type", "type_as", "int", "long", "short", "bool", "byte", "char"}

# A wrapper that returns `torch.tensor(out.item() / N)` routes its result through
# Python floats, which carry no trace of where they came from -- LLM rows 20, 78,
# 110 and 153 all end that way.  So while the trace records, `item` hands back a
# float that remembers: the tensor it was read from, and every + - * / applied to it
# afterwards.  The replay rebuilds that arithmetic on terms.  Anything else done to
# it (math.sqrt, int(), a comparison) yields a plain float and the link is gone,
# which leaves what depends on it undefined.  Only floats: an int from `item` is an
# index or a count, and no handler builds terms for those.
_FROM_SCALAR = {"tensor", "as_tensor"}


class TracedFloat(float):
    """A float read out of a tensor by `.item()`, carrying how it was computed:
    `src` is None for the read itself, else (fn, left, right)."""
    src = None

def _traced(fn, name):
    def m(self, other=None):
        if name != "__neg__" and (isinstance(other, bool) or not isinstance(other, (int, float))):
            return NotImplemented
        r = TracedFloat(fn(float(self), None if other is None else float(other)))
        r.src = (fn, self, other)
        return r
    m.__name__ = name
    return m

for _n, _f in {"__add__": lambda a, b: a + b, "__radd__": lambda a, b: b + a,
               "__sub__": lambda a, b: a - b, "__rsub__": lambda a, b: b - a,
               "__mul__": lambda a, b: a * b, "__rmul__": lambda a, b: b * a,
               "__truediv__": lambda a, b: a / b, "__rtruediv__": lambda a, b: b / a,
               "__neg__": lambda a, b: -a}.items():
    setattr(TracedFloat, _n, _traced(_f, _n))


def replay(events, seed, target):
    """Walk the recorded ops, computing terms wherever every tensor input is known.
    `seed` maps id(tensor) -> STensor.  Returns the STensor for `target`, or None."""
    return replay_all(events, seed).get(id(target))


def replay_all(events, seed):
    """`replay`, returning every tensor it could compute: id(tensor) -> STensor.
    A tensor is defined by the FIRST op that produced it, so an in-place op later in
    the wrapper does not overwrite the value an earlier kernel launch read."""
    sym = dict(seed)
    scalars = {}                          # id(TracedFloat from item) -> its STensor
    def scalar(x):
        if not isinstance(x, TracedFloat): return x
        if x.src is None: return scalars.get(id(x), MISSING)
        fn, a, b = x.src
        a, b = scalar(a), scalar(b)
        if a is MISSING or b is MISSING: return MISSING
        try:
            r = fn(a, b)
            return r if isinstance(r, STensor) else MISSING
        except Exception: return MISSING
    def sub(x):
        if isinstance(x, torch.Tensor):
            s = sym.get(id(x))
            return s if s is not None else MISSING
        if isinstance(x, TracedFloat): return scalar(x)
        if isinstance(x, (list, tuple)):
            ys = [sub(v) for v in x]
            return MISSING if any(y is MISSING for y in ys) else type(x)(ys)
        return x
    for name, args, kwargs, out in events:
        if name == "item" and isinstance(out, TracedFloat) and args and isinstance(args[0], torch.Tensor):
            s = sym.get(id(args[0]))
            if s is not None and len(s.flat()) == 1: scalars[id(out)] = STensor(s.a.reshape(()))
            continue
        # `x.min(dim=1)` returns (values, indices): LLM row 79 averages the values.
        # Only floating-point parts are taken -- no handler builds index terms.
        parts = [o for o in out if isinstance(o, torch.Tensor)] if isinstance(out, (tuple, list)) else []
        if parts:
            if all(id(o) in sym for o in parts if o.is_floating_point()): continue
        elif not isinstance(out, torch.Tensor) or id(out) in sym: continue
        # A cast from floating point to an integer or bool dtype truncates, and the
        # front-end models `to`, `type`, `type_as` and friends as the identity -- right
        # for a float cast over the reals, wrong for this one.  The recorded output
        # carries the dtype the op really produced, so leave such a tensor undefined:
        # whatever depends on it stays unknown rather than being built wrong.
        if (not parts and name in _CASTS | _FROM_SCALAR and not (out.is_floating_point() or out.is_complex())
                and any((isinstance(a, torch.Tensor) and a.is_floating_point()) or type(a) is float for a in args)):
            continue
        sa = [sub(a) for a in args]; sk = {k: sub(v) for k, v in kwargs.items()}
        if any(v is MISSING for v in sa) or any(v is MISSING for v in sk.values()): continue
        if not any(isinstance(v, STensor) for v in list(sa) + list(sk.values())): continue
        h = _TORCH.get(name)
        if h is None and name in _FROM_SCALAR:
            h = lambda data, *a, **k: data if isinstance(data, STensor) else None
        if h is None:
            recv = sa[0] if sa and isinstance(sa[0], STensor) else None
            h = getattr(type(recv), name, None) if recv is not None else None
            if h is None: continue
        try: res = h(*sa, **sk)
        except Exception: continue
        if parts:
            if isinstance(res, (tuple, list)) and len(res) == len(out):
                for o, r in zip(out, res):
                    if (isinstance(o, torch.Tensor) and o.is_floating_point() and isinstance(r, STensor)
                            and r.shape == tuple(o.shape) and id(o) not in sym):
                        sym[id(o)] = r
            continue
        if isinstance(res, STensor): sym[id(out)] = res
        elif isinstance(res, T.Term): sym[id(out)] = STensor(np.array(res, dtype=object))
    return sym

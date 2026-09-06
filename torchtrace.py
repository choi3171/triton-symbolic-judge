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
import terms as T
from spec import STensor, _TORCH
from capture import base_of, root_storage, physical_offsets

# pure metadata: recording them is noise
_IGNORE = {"data_ptr", "__get__", "__set__", "numel", "size", "dim", "stride", "storage_offset",
           "element_size", "is_contiguous", "get_device", "is_floating_point", "untyped_storage",
           "device", "dtype", "shape", "requires_grad_", "__repr__", "item", "_has_compatible_shallow_copy_type"}

class TorchTrace(TorchFunctionMode):
    """Records (name, args, kwargs, result) for every torch op, holding strong
    references so `id()` stays a valid key for the replay."""
    def __init__(self): self.events = []
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", str(func))
        if name not in _IGNORE: self.events.append((name, args, kwargs, out))
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

def replay(events, seed, target):
    """Walk the recorded ops, computing terms wherever every tensor input is known.
    `seed` maps id(tensor) -> STensor.  Returns the STensor for `target`, or None."""
    sym = dict(seed)
    def sub(x):
        if isinstance(x, torch.Tensor):
            s = sym.get(id(x))
            return s if s is not None else MISSING
        if isinstance(x, (list, tuple)):
            ys = [sub(v) for v in x]
            return MISSING if any(y is MISSING for y in ys) else type(x)(ys)
        return x
    for name, args, kwargs, out in events:
        if not isinstance(out, torch.Tensor) or id(out) in sym: continue
        sa = [sub(a) for a in args]; sk = {k: sub(v) for k, v in kwargs.items()}
        if any(v is MISSING for v in sa) or any(v is MISSING for v in sk.values()): continue
        if not any(isinstance(v, STensor) for v in list(sa) + list(sk.values())): continue
        h = _TORCH.get(name)
        if h is None:
            recv = sa[0] if sa and isinstance(sa[0], STensor) else None
            h = getattr(type(recv), name, None) if recv is not None else None
            if h is None: continue
        try: res = h(*sa, **sk)
        except Exception: continue
        if isinstance(res, STensor): sym[id(out)] = res
        elif isinstance(res, T.Term): sym[id(out)] = STensor(np.array(res, dtype=object))
    return sym.get(id(target))

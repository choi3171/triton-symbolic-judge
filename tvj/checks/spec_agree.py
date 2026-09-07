"""Does the spec front-end compute what torch computes?

`spec_sigcheck.py` compares signatures; this compares *values*.  Every handler is
run twice -- once on real tensors, once on symbolic ones -- and the resulting
terms are evaluated over the reals at exactly the tensors' values.  A handler
that drops an argument, transposes two of them, or reduces along the wrong axis
disagrees here numerically even when its signature looks fine.

The cases below are not a sample: each one is a bug this project actually shipped
and had to find through a false FAIL on a real corpus row.
"""
import sys, math, torch, numpy as np
import torch.nn.functional as F
from tvj.core import terms as T
from tvj.decide import numeric as NUM
from tvj.front.spec import STensor

# Float literals are rounded to float32 on the way into a term -- decision
# `literal.working-precision` in semantics.py: the kernel will see the f32
# constant, so the spec must too.  A reciprocal that is not a power of two
# (1/3, 1/9, 1/12) therefore differs from torch's float64 result at ~1e-8, which
# is the rounding, not a modelling error.  Anything above 1e-7 is.
TOL = 1e-7


def run(fn, **tensors):
    """fn is applied to real tensors and to symbolic ones; returns (torch, spec)."""
    T.reset()
    real = fn(**tensors)
    sym = fn(**{k: STensor.input(k, tuple(v.shape)) for k, v in tensors.items()})
    if not isinstance(sym, STensor): sym = STensor(np.array(sym, dtype=object))
    point = {k: v.double().reshape(-1).tolist() for k, v in tensors.items()}
    got = [NUM.evaluate(t, point, {}) if hasattr(t, "uid") else float(t) for t in sym.flat()]
    return real.double().reshape(-1).tolist(), got, tuple(real.shape), sym.shape


CASES = [
    # -- the three that shipped ------------------------------------------------
    ("F.linear with bias",     lambda x, w, b: F.linear(x, w, bias=b),
     dict(x=(3, 4), w=(5, 4), b=(5,))),
    ("mean(axis=-1)",          lambda x: x.mean(axis=-1),                dict(x=(3, 4))),
    ("avg_pool2d ceil/count",  lambda x: F.avg_pool2d(x, 3, 2, 1, False, False),
     dict(x=(1, 2, 5, 5))),
    # -- what the signature cross-check turned up today -------------------------
    ("var(dim, unbiased=False)",  lambda x: x.var(1, unbiased=False),    dict(x=(3, 4))),
    ("var(dim, correction=0)",    lambda x: torch.var(x, 1, correction=0), dict(x=(3, 4))),
    ("var(dim) default",          lambda x: x.var(1),                    dict(x=(3, 4))),
    ("var(keepdim=True)",         lambda x: x.var(1, keepdim=True),      dict(x=(3, 4))),
    ("std(dim, unbiased=False)",  lambda x: x.std(1, unbiased=False),    dict(x=(3, 4))),
    ("softmax positional stack",  lambda x: F.softmax(x, 1, 3),          dict(x=(3, 4))),
    ("log_softmax(dim=1)",        lambda x: F.log_softmax(x, dim=1),     dict(x=(3, 4))),
    ("mse_loss(reduction=sum)",   lambda a, b: F.mse_loss(a, b, reduction="sum"),
     dict(a=(3, 4), b=(3, 4))),
    ("l1_loss(reduction=mean)",   lambda a, b: F.l1_loss(a, b),          dict(a=(3, 4), b=(3, 4))),
    ("norm(p=2, dim=1)",          lambda x: torch.norm(x, 2, 1),         dict(x=(3, 4))),
    ("full_like(kw)",             lambda x: torch.full_like(x, fill_value=2.5) + x,
     dict(x=(2, 3))),
    ("relu(inplace=True)",        lambda x: F.relu(x * 1.0, inplace=True), dict(x=(3, 4))),
    ("mm",                        lambda a, b: torch.mm(a, b),           dict(a=(3, 4), b=(4, 2))),
    # -- the reductions and shapes everything else rests on ---------------------
    ("sum(dim=0, keepdim)",       lambda x: x.sum(dim=0, keepdim=True),  dict(x=(3, 4))),
    ("amax(dim=-1)",              lambda x: x.amax(dim=-1),              dict(x=(3, 4))),
    ("transpose(0,1) positional", lambda x: x.transpose(0, 1) + 0.0,     dict(x=(3, 4))),
    ("layer_norm",                lambda x, w, b: F.layer_norm(x, (4,), w, b, 1e-5),
     dict(x=(3, 4), w=(4,), b=(4,))),
    ("conv2d stride/pad",         lambda x, w, b: F.conv2d(x, w, b, 2, 1),
     dict(x=(1, 2, 5, 5), w=(3, 2, 3, 3), b=(3,))),
    # -- indirect reads.  `gather` and `index_select` are NOT the same function:
    #    binding both to one helper made the reference silently compute something
    #    else for any input above one dimension, and nothing downstream saw it.
    ("gather(dim=1)",             lambda x, i: torch.gather(x, 1, i),     dict(x=(3, 4), i=(3, 4)), "int:4"),
    ("gather(dim=0)",             lambda x, i: torch.gather(x, 0, i),     dict(x=(4, 3), i=(4, 3)), "int:4"),
    ("index_select(dim=0)",       lambda x, i: torch.index_select(x, 0, i), dict(x=(5, 3), i=(4,)), "int:5"),
    ("x[idx]",                    lambda x, i: x[i] + 0.0,                dict(x=(5,), i=(4,)),   "int:5"),
    ("embedding",                 lambda i, w: F.embedding(i, w),         dict(i=(4,), w=(5, 3)), "int:5"),
    ("index_add",                 lambda b, i, s: torch.index_add(b, 0, i, s),
     dict(b=(5,), i=(4,), s=(4,)), "int:5"),
    ("scatter_add",               lambda b, i, s: torch.scatter_add(b, 0, i, s),
     dict(b=(5,), i=(4,), s=(4,)), "int:5"),
    # -- piecewise and normalisation, added once `select` made them expressible ---
    ("elu(alpha=1)",              lambda x: F.elu(x),                    dict(x=(3, 4))),
    ("elu(alpha=0.7)",            lambda x: F.elu(x, 0.7),               dict(x=(3, 4))),
    ("hardsigmoid",               lambda x: F.hardsigmoid(x),            dict(x=(3, 4))),
    ("threshold(0.1, -1)",        lambda x: F.threshold(x, 0.1, -1.0),   dict(x=(3, 4))),
    ("smooth_l1_loss(beta=1)",    lambda a, b: F.smooth_l1_loss(a, b),   dict(a=(3, 4), b=(3, 4))),
    ("smooth_l1_loss(beta=0.3)",  lambda a, b: F.smooth_l1_loss(a, b, beta=0.3),
     dict(a=(3, 4), b=(3, 4))),
    ("huber_loss(delta=0.4)",     lambda a, b: F.huber_loss(a, b, delta=0.4),
     dict(a=(3, 4), b=(3, 4))),
    ("huber_loss(reduction=sum)", lambda a, b: F.huber_loss(a, b, reduction="sum"),
     dict(a=(3, 4), b=(3, 4))),
    ("batch_norm eval",           lambda x, m, v, w, b: F.batch_norm(x, m, v.abs(), w, b, False, 0.1, 1e-5),
     dict(x=(2, 3, 4), m=(3,), v=(3,), w=(3,), b=(3,))),
    ("batch_norm training",       lambda x, w, b: F.batch_norm(x, None, None, w, b, True, 0.1, 1e-5),
     dict(x=(2, 3, 4), w=(3,), b=(3,))),
    ("group_norm(2 groups)",      lambda x, w, b: F.group_norm(x, 2, w, b, 1e-5),
     dict(x=(2, 4, 3), w=(4,), b=(4,))),
    ("group_norm(4d, 2 groups)",  lambda x, w, b: F.group_norm(x, 2, w, b, 1e-5),
     dict(x=(2, 4, 3, 3), w=(4,), b=(4,))),
    ("instance_norm",             lambda x, w, b: F.instance_norm(x, None, None, w, b, True, 0.1, 1e-5),
     dict(x=(2, 3, 4, 4), w=(3,), b=(3,))),
    ("norm(p=1, dim=1)",          lambda x: torch.norm(x, 1, 1),         dict(x=(3, 4))),
    ("norm(p=inf, dim=1)",        lambda x: torch.norm(x, float("inf"), 1), dict(x=(3, 4))),
    ("mean(keepdims=True)",       lambda x: torch.mean(x, 1, keepdim=True), dict(x=(3, 4))),
    # -- every flag the dead-argument scan found declared and never read ---------
    ("softplus below threshold",  lambda x: F.softplus(x, 1.0, 20.0),     dict(x=(3, 4))),
    ("softplus ACROSS threshold", lambda x: F.softplus(x, 1.0, 0.2),      dict(x=(3, 4))),
    ("softplus(beta=2, thr=0.3)", lambda x: F.softplus(x, 2.0, 0.3),      dict(x=(3, 4))),
    ("std(dim, correction=0)",    lambda x: torch.std(x, 1, correction=0), dict(x=(3, 4))),
    ("rsub(alpha=2)",             lambda a, b: torch.rsub(a, b, alpha=2.0), dict(a=(3, 4), b=(3, 4))),
    ("dropout(training=False)",   lambda x: F.dropout(x, 0.5, False) + 0.0, dict(x=(3, 4))),
    ("max_pool2d dilation=2",     lambda x: F.max_pool2d(x, 2, 1, 0, 2, False, False),
     dict(x=(1, 1, 6, 6))),
    ("max_pool1d dilation=3",     lambda x: F.max_pool1d(x, 2, 1, 0, 3, False, False),
     dict(x=(1, 2, 9))),
    ("unsqueeze(dim=1)",          lambda x: torch.unsqueeze(x, dim=1) + 0.0, dict(x=(3, 4))),
]

# Pooling gets a sweep rather than a case: two of the flags (`ceil_mode`,
# `count_include_pad`) change the output SHAPE and the DIVISOR respectively, and
# both were silently ignored at some point in this file's history.
def _pool_cases():
    for nd, shape in ((1, (1, 2, 7)), (2, (1, 2, 5, 5))):
        mp = getattr(F, f"max_pool{nd}d"); ap = getattr(F, f"avg_pool{nd}d")
        for k, st, pd in ((2, 2, 0), (3, 2, 1), (3, 1, 1), (2, 3, 0)):
            for ceil in (False, True):
                yield (f"max_pool{nd}d k{k} s{st} p{pd} ceil={ceil}",
                       (lambda x, mp=mp, k=k, st=st, pd=pd, ceil=ceil:
                        mp(x, k, st, pd, 1, ceil, False)), dict(x=shape))
                for cip in (True, False):
                    yield (f"avg_pool{nd}d k{k} s{st} p{pd} ceil={ceil} cip={cip}",
                           (lambda x, ap=ap, k=k, st=st, pd=pd, ceil=ceil, cip=cip:
                            ap(x, k, st, pd, ceil, cip)), dict(x=shape))
    for o in (1, 2, 3, 4):
        yield (f"adaptive_avg_pool2d -> {o}",
               lambda x, o=o: F.adaptive_avg_pool2d(x, o), dict(x=(1, 2, 5, 7)))
        yield (f"adaptive_max_pool2d -> {o}",
               lambda x, o=o: F.adaptive_max_pool2d(x, o), dict(x=(1, 2, 5, 7)))
    yield ("adaptive_avg_pool1d -> 3", lambda x: F.adaptive_avg_pool1d(x, 3), dict(x=(1, 2, 7)))

CASES += list(_pool_cases())

# Refusing is a contract too: a mode we do not model must raise, not quietly
# return the unmodelled answer.
REFUSALS = [
    ("div(rounding_mode='floor')", lambda x: torch.div(x, 2.0, rounding_mode="floor"), dict(x=(3, 4))),
    ("div(rounding_mode='trunc')", lambda x: torch.div(x, 2.0, rounding_mode="trunc"), dict(x=(3, 4))),
    ("dropout(training=True)",     lambda x: F.dropout(x, 0.5, True),                  dict(x=(3, 4))),
]

if __name__ == "__main__":
    g = torch.Generator().manual_seed(7)
    bad = 0
    print(f"{'case':<28} {'shape':<14} {'max rel error':>20}  status")
    print("-" * 76)
    for case in CASES:
        label, fn, shapes = case[0], case[1], case[2]
        # "int:N" marks the case's index tensor: whichever argument is named `i`
        # is drawn from [0, N) as integers, because an index is not a float
        idx_hi = int(case[3].split(":")[1]) if len(case) > 3 else None
        tensors = {k: (torch.randint(0, idx_hi, s, generator=g) if (k == "i" and idx_hi)
                       else torch.rand(s, generator=g, dtype=torch.float64) * 2 - 1)
                   for k, s in shapes.items()}
        try:
            want, got, wshape, sshape = run(fn, **tensors)
        except Exception as e:
            print(f"{label:<28} {'-':<14} {'-':>20}  ERROR {type(e).__name__}: {str(e)[:40]}"); bad += 1; continue
        if len(want) != len(got):
            print(f"{label:<28} {str(wshape):<14} {'-':>20}  SHAPE torch {wshape} vs spec {sshape}"); bad += 1; continue
        d = max((abs(a - b) / max(1.0, abs(a), abs(b)) for a, b in zip(want, got)), default=0.0)
        ok = d <= TOL and wshape == tuple(sshape)
        bad += not ok
        note = "" if wshape == tuple(sshape) else f"  SHAPE torch {wshape} vs spec {tuple(sshape)}"
        print(f"{label:<28} {str(wshape):<14} {d:>20.3g}  {'ok' if ok else 'MISMATCH'}{note}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} handlers agree with torch to {TOL:g}")

    print("\nmodes the front-end must REFUSE rather than mis-model")
    rbad = 0
    for label, fn, shapes in REFUSALS:
        tensors = {k: torch.rand(s, generator=g, dtype=torch.float64) for k, s in shapes.items()}
        T.reset()
        try:
            fn(**{k: STensor.input(k, tuple(v.shape)) for k, v in tensors.items()})
            print(f"  {label:<28} DID NOT REFUSE"); rbad += 1
        except NotImplementedError as e:
            print(f"  {label:<28} refused: {str(e)[:52]}")
        except Exception as e:
            print(f"  {label:<28} wrong error {type(e).__name__}: {str(e)[:40]}"); rbad += 1
    print(f"\n{len(REFUSALS) - rbad}/{len(REFUSALS)} refusals as expected")
    sys.exit(1 if (bad or rbad) else 0)

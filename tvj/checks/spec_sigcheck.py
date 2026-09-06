"""Cross-check every spec front-end handler against the real torch signature.

Three analyses, because the same defect hides in three different places:

  SWALLOWED  -- the parameter is not in the signature; `**kwargs` eats it
  POSITION   -- the parameter is there, at the wrong index
  DEAD       -- the parameter is there, in the right place, and never read

Four defects of exactly these shapes have already come out of this file's absence:

  F.linear(..., bias=...)   swallowed by a `**k` the handler never read
  mean(axis=-1)             swallowed the same way -- a per-row mean became global
  avg_pool2d(...)           `ceil_mode` and `count_include_pad` in the wrong
                            positional order, so `False` bound to the wrong one
  softplus(threshold=20)    declared and never read -- so the reference was a
                            different real function from torch's above x = 20

None of them raised.  Each silently changed the *reference*, which is the one
thing in the pipeline nothing else can check: a wrong spec makes a correct kernel
FAIL and, worse, can make a wrong kernel PASS.

So: for every handler, line its parameters up with torch's own, in order, and
report any name that moved, vanished, or appeared.  Handlers that legitimately
take fewer arguments are fine as long as the ones they DO take are a prefix-
compatible, same-order subset -- what is never fine is the same name at a
different position, or a torch parameter that changes behaviour being absent
while `**kwargs` quietly absorbs it.
"""
import dis, inspect, re, sys, torch
import torch.nn.functional as F
from tvj.front import spec

# torch parameters whose default is the identity: absent from a handler, nothing changes.
# torch's OWN numpy-compatibility aliases.  `x.mean(axis=0)` and `x.mean(dim=0)`
# are the same call, so a handler must accept both and the second spelling is not
# a parameter out of place.  This is a documented fact about torch, not a list of
# our exceptions -- the per-op whitelist this replaced was three handlers that had
# invented a short name of their own next to torch's, which is now just fixed.
TORCH_ALIASES = {"axis": "dim", "keepdims": "keepdim"}

HARMLESS = set(spec.HARMLESS_KWARGS) | {"input", "self", "training"}

def torch_sig(name):
    """The real signature as an ordered parameter-name list, `self`/`input` kept
    so positions line up with a handler whose first parameter is the tensor.
    torch's C ops have no `inspect.signature`; their docstring's first line is one."""
    for owner in (F, torch, torch.Tensor, torch.linalg):
        f = getattr(owner, name, None)
        if f is None: continue
        ps = kwonly = None
        try:
            pp = [p for p in inspect.signature(f).parameters.values()
                  if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
            ps = [p.name for p in pp]
            kwonly = {p.name for p in pp if p.kind is p.KEYWORD_ONLY}
        except (ValueError, TypeError):
            doc = (getattr(f, "__doc__", "") or "").strip()
            m = re.match(rf"{name}\(([^)]*)\)", doc)
            if m:
                ps, kwonly, star = [], set(), False
                for a in m.group(1).split(","):
                    a = a.strip()
                    if a.lstrip("\\") == "*": star = True; continue
                    a = a.lstrip("\\*").split("=")[0].strip()
                    if not a or not a.isidentifier(): continue
                    ps.append(a)
                    if star: kwonly.add(a)
        if ps is None: continue
        if owner is torch.Tensor and (not ps or ps[0] not in ("self", "input")):
            ps = ["self"] + ps          # the doc for a C method omits the receiver
        return owner.__name__, ps, kwonly
    return None, None, None


def handler_sig(h):
    try: s = inspect.signature(h)
    except (ValueError, TypeError): return None
    p = s.parameters.values()
    return {"pos":  [x.name for x in p if x.kind is x.POSITIONAL_OR_KEYWORD],
            "kw":   [x.name for x in p if x.kind is x.KEYWORD_ONLY],
            "star": any(x.kind is x.VAR_POSITIONAL for x in p),
            "kws":  any(x.kind is x.VAR_KEYWORD for x in p)}


bad, checked, skipped, renames = [], 0, [], []
for name in sorted(spec._TORCH):
    h = spec._TORCH[name]
    hs = handler_sig(h)
    if hs is None: skipped.append((name, "handler has no signature")); continue
    owner, tp, kwonly = torch_sig(name)
    if tp is None: skipped.append((name, "no torch counterpart found")); continue
    checked += 1
    if not hs["pos"] and not hs["kw"] and (hs["star"] or hs["kws"]):
        skipped.append((name, "handler is a transparent *a/**k forwarder")); continue
    have = set(hs["pos"]) | set(hs["kw"])

    # 1. POSITION -- the avg_pool2d bug: a name both sides know, at different
    #    indices, so a positional call binds it to the wrong parameter.  Only
    #    torch parameters that CAN be passed positionally are at risk.
    for i, p in enumerate(hs["pos"]):
        if (p in tp and p not in kwonly and tp.index(p) != i
                and p not in TORCH_ALIASES):
            bad.append((name, "POSITION",
                        f"`{p}` is handler arg {i}, torch arg {tp.index(p)} "
                        f"(a positional call binds `{tp[i] if i < len(tp) else '-'}` to it)"))

    # 2. SWALLOWED -- the F.linear(bias=) / mean(axis=) bug: a torch parameter that
    #    changes behaviour, absent from the handler, absorbed by its **kwargs.
    #    A torch name the handler merely SPELLS differently at the same index is a
    #    rename, not a defect: the positional call still lands in the right slot.
    miss = []
    for i, p in enumerate(tp):
        if p in have or p in HARMLESS or (i == 0 and p in ("self", "input")): continue
        if any(a in have and b == p for a, b in TORCH_ALIASES.items()): continue
        if p not in kwonly and i < len(hs["pos"]): renames.append((name, p, hs["pos"][i])); continue
        if p not in kwonly and hs["star"]: continue
        miss.append(p)
    if miss:
        bad.append((name, "SWALLOWED" if hs["kws"] else "would raise",
                    f"{miss}  (torch: {owner}.{name})"))

# 3. DEAD -- declared, correctly placed, and never read.  Invisible to the two
#    analyses above, because the signature is right; only the body is wrong.
#    `softplus(threshold=20)` shipped this way: torch returns `x` once
#    `beta*x > threshold`, our reference returned log(1+exp(x)) everywhere, and
#    four corpus rows sat in "unprovable but numerically equal" because of it.
dead = []
for name in sorted(spec._TORCH):
    h = spec._TORCH[name]
    code = getattr(h, "__code__", None)
    if code is None: continue
    args = list(code.co_varnames[:code.co_argcount + code.co_kwonlyargcount])
    if not args: continue
    used = set()
    for ins in dis.get_instructions(h):
        if not ins.opname.startswith(("LOAD_FAST", "STORE_FAST", "DELETE_FAST", "LOAD_DEREF",
                                      "LOAD_CLOSURE", "MAKE_CELL", "COPY_FREE_VARS")): continue
        v = ins.argval                      # LOAD_FAST_LOAD_FAST carries a PAIR
        used |= set(v) if isinstance(v, tuple) else {v}
    for c in code.co_consts:                # a nested lambda closes over names
        if hasattr(c, "co_names"):
            used |= set(c.co_varnames) | set(c.co_names) | set(c.co_freevars)
    for a in args[1:]:
        if a not in used and a not in HARMLESS:
            dead.append((name, "DEAD", f"`{a}` is declared and never read"))
bad += dead

print(f"handlers: {len(spec._TORCH)}   cross-checked: {checked}   not comparable: {len(skipped)}\n")
sev = {"POSITION": 0, "SWALLOWED": 1, "DEAD": 1, "would raise": 2}
bad.sort(key=lambda b: (sev[b[1]], b[0]))
if bad:
    print(f"{'op':<26} {'kind':<12} detail")
    print("-" * 108)
    for n, k, d in bad: print(f"{n:<26} {k:<12} {d}")
n_hard = sum(1 for b in bad if b[1] != "would raise")
print(f"\n{len(bad)} disagreement(s): {n_hard} silent "
      f"({sum(1 for b in bad if b[1]=='POSITION')} mis-positioned, "
      f"{sum(1 for b in bad if b[1]=='SWALLOWED')} swallowed, "
      f"{sum(1 for b in bad if b[1]=='DEAD')} dead), "
      f"{len(bad)-n_hard} that would raise instead")
if "-v" in sys.argv:
    print(f"\nrenamed parameters ({len(renames)}, harmless -- same slot, our spelling):")
    for n, t, o in renames: print(f"  {n:<26} torch `{t}` -> ours `{o}`")
    print("\nnot comparable:")
    for n, why in skipped: print(f"  {n:<26} {why}")

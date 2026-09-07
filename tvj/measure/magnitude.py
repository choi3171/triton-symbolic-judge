"""At what input magnitude does a generated kernel stop matching its reference?

The accuracy obligation rejected three LLM kernels that spell a saturating
function through raw exponentials -- `tanh(x) = (exp(2x)-1)/(exp(2x)+1)` -- and
the obvious objection is that |x| > 44 does not happen in machine learning.

It does not have to.  The argument to `tanh` is rarely the activation itself.
GELU's tanh approximation passes `sqrt(2/pi)(x + 0.044715 x^3)`, and the cubic
term means an activation of **10.06** already puts that argument past float32's
exponential limit.  Attention logits before a softmax reach tens routinely, and
outlier features in transformer residual streams are documented in the hundreds.

So rather than argue the point, measure it: for every row whose reference uses a
saturating function, sweep the input magnitude and record where the kernel and
torch part company.  A kernel that is exact at 1 and NaN at 12 is not a corner
case; the benchmark simply never looked past 1.

    python3 -m tvj.measure.magnitude [n_rows]
"""
import json, re, sys, collections
import torch
from tvj.judge import traces_run as TR

SATURATING = re.compile(r"\btanh\b|Tanh|Mish|[Ss]oftplus|[Ss]igmoid|gelu|GELU|logsumexp|"
                        r"[Ss]oftmax|[Ee]lu\b|SiLU|silu|swish|Swish")
SCALES = [1.0, 3.0, 10.0, 20.0, 45.0, 90.0, 200.0]


def first_break(i, rows, tag):
    """The smallest swept magnitude at which the kernel stops matching torch."""
    r = rows[i]; ns = {}
    try:
        exec(r["pytorch_code"], ns)
        c = TR.model_class(r["pytorch_code"], ns)
        if c is None: return None
        ia, ik = ns["get_init_inputs"]()
        torch.manual_seed(0); m = ns[c](*ia, **ik).cuda().eval()
        ns2 = TR.load_mod(r["triton_code"], f"mag_{tag}_{i}")
        e = ns2.get("triton_kernel_wrapper")
        if e is None: return None
        base = [x.cuda() if torch.is_tensor(x) else x for x in ns["get_inputs"]()]
    except Exception:
        return None
    g = torch.Generator().manual_seed(5)
    ok_at = None
    for s in SCALES:
        xs = [((torch.rand(x.shape, generator=g) * s).cuda().to(x.dtype)
               if torch.is_tensor(x) else x) for x in base]
        try:
            with torch.no_grad():
                a = TR.first(m(*xs)).float()
                b = TR.first(e(*TR.bind_wrapper(e, m, xs))).float()
        except Exception:
            return None
        # the reference is the authority: if IT is not finite, the input is out of
        # domain and the row says nothing
        if not bool(torch.isfinite(a).any()): return None
        broke = bool(((~torch.isfinite(b)) & torch.isfinite(a)).any())
        if not broke:
            fin = torch.isfinite(a) & torch.isfinite(b)
            if bool(fin.any()):
                d = (a - b).abs()[fin]
                sc = a.abs()[fin].clamp(min=1.0)
                broke = bool((d / sc > 1e-2).any())
        if broke: return (ok_at, s)
        ok_at = s
    return (ok_at, None)


if __name__ == "__main__":
    rows = [r for r in json.load(open("data/triton_traces.json")) if r["source"] == "kernelbook"]
    cand = [i for i, r in enumerate(rows) if SATURATING.search(r["pytorch_code"])]
    if len(sys.argv) > 1: cand = cand[:int(sys.argv[1])]
    print(f"{len(cand)} of {len(rows)} rows have a reference with a saturating function\n")
    print(f"{'row':>4}  {'model':<24} {'matches up to':>13}  {'breaks at':>10}")
    hist = collections.Counter(); shown = 0
    for i in cand:
        res = first_break(i, rows, "tr")
        if res is None: hist["could not run"] += 1; continue
        ok, bad = res
        hist[bad if bad else "never breaks"] += 1
        if bad is not None:
            name = TR.model_class(rows[i]["pytorch_code"]) or "?"
            print(f"{i:>4}  {name[:22]:<24} {str(ok):>13}  {bad:>10.0f}")
            shown += 1
    print(f"\n{shown} kernels break somewhere in the sweep")
    print("distribution of the breaking magnitude:")
    for k, v in sorted(hist.items(), key=lambda x: (isinstance(x[0], str), x[0])):
        print(f"   {str(k):>14}  {v:3d}")
    print("\nfor scale: GELU's tanh approximation puts its own argument past the")
    print("float32 exponential limit at an activation of 10.06.")

"""Does a directive derived from ONE exploit catch the others?

A witness point closes one input; a policy steps around it.  The claim worth
testing is that the AXIS generalises -- that a directive derived from a single
kernel catches every other kernel blind on the same axis, including ones the
judge never looked at.

Two axes have enough exploits in this corpus to test:

  vary-parameter   derived from row 97 (Theta), the only one whose value
                   disagreement named parameters at the time.  The defaults of
                   the others are identity elements too, so they should fall.
  stress-regime    derived from row 23 (Tanh), whose accuracy FAIL named a
                   magnitude rather than a point.  Any kernel that spells a
                   saturating function through raw exponentials should fall.

The control in both cases is the corpus' own check, unmodified: if that already
catches the row, the directive proved nothing.
"""
import json, sys
sys.argv = ["x"]
import torch
from tvj.judge import traces_run as TR

ROWS = [r for r in json.load(open("data/triton_traces.json")) if r["source"] == "kernelbook"]


def run_under(i, vary_params=False, shift=0.0, poison=False, trials=5):
    """The corpus' own tolerance check, plus whichever directives are switched on.
    True = the candidate still passes (the check did NOT catch it)."""
    r = ROWS[i]; ns = {}
    try:
        exec(r["pytorch_code"], ns)
        c = TR.model_class(r["pytorch_code"]); ia, ik = ns["get_init_inputs"]()
        torch.manual_seed(0); m = ns[c](*ia, **ik).cuda().eval()
        ns2 = TR.load_mod(r["triton_code"], f"v{i}"); e = ns2["triton_kernel_wrapper"]
    except Exception:
        return None
    g = torch.Generator().manual_seed(11)
    for _ in range(trials):
        if vary_params:
            with torch.no_grad():
                for p in m.parameters():
                    if p.numel(): p.copy_((torch.rand(p.shape, generator=g) * 2 - 1).to(p.device, p.dtype))
        xs = [torch.rand(x.shape, device="cuda") if torch.is_tensor(x) else x
              for x in ns["get_inputs"]()]
        if shift:
            xs = [(x + shift) if torch.is_tensor(x) and x.is_floating_point() else x for x in xs]
        if poison:
            # hand the caching allocator its blocks back full of NaN, so a kernel
            # that skips a write cannot pass by finding the right answer lying there
            with torch.no_grad():
                junk = [torch.full_like(x, float("nan")) for x in xs if torch.is_tensor(x)]
                del junk
        try:
            with torch.no_grad():
                a = TR.first(m(*xs)).float()
                b = TR.first(e(*TR.bind_wrapper(e, m, xs))).float()
            if bool(((~torch.isfinite(b)) & torch.isfinite(a)).any()): return False
            ok = torch.isfinite(a) & torch.isfinite(b)
            if not bool(ok.any()): continue
            if not torch.allclose(a[ok], b[ok], atol=1e-2, rtol=1e-2): return False
        except Exception:
            return None
    return True


AXES = [
    ("vary-parameter", "row 97 (Theta)", [42, 66, 97, 98], dict(vary_params=True)),
    ("stress-regime",  "row 23 (Tanh)",  [23, 114, 123],   dict(shift=100.0)),
]

if __name__ == "__main__":
    f = lambda v: "passes" if v is True else ("CAUGHT" if v is False else "n/a")
    total_caught = total = 0
    for axis, origin, rows, kw in AXES:
        print(f"\n=== {axis}, derived from {origin} ===")
        print(f"{'row':<5} {'model':<22} {'corpus check':>13} {'+ ' + axis:>18}   unseen?")
        caught = 0
        for i in rows:
            base = run_under(i)
            with_d = run_under(i, **kw)
            unseen = "" if origin.startswith(f"row {i} ") else "unseen by the directive"
            if with_d is False and base is not True: unseen += "  (corpus check already catches it)"
            caught += with_d is False
            total += 1
            name = TR.model_class(ROWS[i]["pytorch_code"]) or "?"
            print(f"{i:<5} {name[:20]:<22} {f(base):>13} {f(with_d):>18}   {unseen}")
        total_caught += caught
        print(f"  -> {caught}/{len(rows)} caught by one directive")
    print(f"\n{total_caught}/{total} exploits caught by two directives, "
          f"each derived from a single row")

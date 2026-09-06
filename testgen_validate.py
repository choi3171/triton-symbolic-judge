"""Does a directive derived from ONE exploit catch the others?

A witness point closes one input; a policy steps around it.  The claim worth
testing is that the AXIS generalises -- that "vary the parameters", derived from
a single kernel, catches every other kernel blind on the same axis, including
ones the judge never saw.
"""
import json, torch, sys
sys.argv = ["x"]
import traces_run as TR, testgen

rows = [r for r in json.load(open("data/triton_traces.json")) if r["source"] == "kernelbook"]
fails = json.load(open("results/triton_traces_fails.json"))

def run_under(i, vary_params, poison):
    """The corpus' own tolerance check, plus the directives under test."""
    r = rows[i]; ns = {}; exec(r["pytorch_code"], ns)
    c = TR.model_class(r["pytorch_code"]); ia, ik = ns["get_init_inputs"]()
    torch.manual_seed(0); m = ns[c](*ia, **ik).cuda().eval()
    ns2 = TR.load_mod(r["triton_code"], f"v{i}"); e = ns2["triton_kernel_wrapper"]
    g = torch.Generator().manual_seed(11)
    for t in range(5):
        if vary_params:
            with torch.no_grad():
                for p in m.parameters():
                    if p.numel(): p.copy_((torch.rand(p.shape, generator=g) * 2 - 1).to(p.device, p.dtype))
        xs = [torch.rand(x.shape, device="cuda") if torch.is_tensor(x) else x for x in ns["get_inputs"]()]
        try:
            with torch.no_grad():
                a = TR.first(m(*xs)).float()
                out = TR.first(e(*TR.bind_wrapper(e, m, xs)))
                if poison and torch.is_tensor(out): pass      # see note below
                b = out.float()
            ok = torch.isfinite(a) & torch.isfinite(b)
            if not bool(ok.any()): continue
            if not torch.allclose(a[ok], b[ok], atol=1e-2, rtol=1e-2): return False
        except Exception: return None
    return True

targets = [f["i"] for f in fails]
print("each column: does the check still PASS the exploit? (False = the check caught it)\n")
print(f"{'row':<5} {'model':<14} {'tolerance as-is':>16} {'+vary-parameter':>16}")
for i in targets:
    base = run_under(i, False, False)
    vary = run_under(i, True, False)
    f = lambda v: "passes" if v is True else ("CAUGHT" if v is False else "n/a")
    print(f"{i:<5} {next(x['model'] for x in fails if x['i']==i):<14} {f(base):>16} {f(vary):>16}")

print("\nthe directive was derived from row 97 alone (the only one whose difference")
print("named parameters); applying it to the others is the generalisation test.")

"""Rule: a FAIL is not believed until the GPU reproduces it at random inputs."""
import json, torch, inspect
from tvj.judge import traces_run as TR

rows = [r for r in json.load(open("data/triton_traces.json")) if r["source"]=="kernelbook"]
recs = [json.loads(l) for l in open("results/triton_traces.jsonl")]
susp = [r for r in recs if r["label"] and r["verdict"] == "FAIL"]
print(f"{'row':<5} {'model':<22} {'obligation':<10} {'GPU max|new-ref| at U[-1,1]':>28}   verdict")
for rec in susp:
    r = rows[rec["i"]]
    try:
        ns = {}; exec(r["pytorch_code"], ns)
        cname = TR.model_class(r["pytorch_code"])
        ia, ik = ns["get_init_inputs"]()
        g = torch.Generator().manual_seed(0)
        m = ns[cname](*ia, **ik).cuda().eval()
        with torch.no_grad():
            for p in m.parameters():
                if p.numel(): p.copy_((torch.rand(p.shape, generator=g)*2-1).cuda())
        ns2 = TR.load_mod(r["triton_code"], rec["key"] + "_rp")
        entry = ns2.get("triton_kernel_wrapper")
        worst = 0.0
        for t in range(3):
            xs = [(torch.rand(x.shape, generator=g)*2-1).cuda() if torch.is_tensor(x) else x
                  for x in ns["get_inputs"]()]
            with torch.no_grad():
                a = TR.first(m(*xs)); b = TR.first(TR.call_wrapper(entry, m, xs))
                worst = max(worst, float((a.float()-b.float()).abs().nan_to_num(1e30).max()))
        real = worst > 1e-4
        print(f"{rec['i']:<5} {cname:<22} {rec.get('obligation',''):<10} {worst:>28.4g}   "
              f"{'REAL defect' if real else 'GPU agrees -> OUR false FAIL'}")
    except Exception as e:
        print(f"{rec['i']:<5} {rec.get('model','?'):<22} {rec.get('obligation',''):<10} {'repro error':>28}   {type(e).__name__}: {str(e)[:40]}")

"""Timing/feasibility probe on the LLM-generated Triton traces. Small rows only."""
import json, re, time, os, importlib.util, torch, signal
import terms as T
os.makedirs("data/tr_mods", exist_ok=True)

def load_mod(src, tag):
    p = f"data/tr_mods/tr_{tag}.py"; open(p,"w").write(src)
    s = importlib.util.spec_from_file_location(f"tr_{tag}", p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.__dict__

def model_class(src):
    names = re.findall(r"class (\w+)\((?:nn\.Module|torch\.nn\.Module)\)", src)
    return names[-1] if names else None

class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))

rows = [r for r in json.load(open("data/triton_traces.json")) if r["source"]=="kernelbook"]
rows = [r for r in rows if r["result_correctness"]][:8]
print(f"{'row':<20} {'call':>5} {'launch':>7} {'t_exec':>8} {'vram_MB':>8}  note")
for r in rows:
    key = r["sample_key"]; note = ""; ok_call = False; nl = 0; dt = 0.0; vram = 0
    signal.alarm(60)
    try:
        T.reset()
        ns = {}; exec(r["pytorch_code"], ns)
        cname = model_class(r["pytorch_code"])
        ia, ik = ns["get_init_inputs"]()
        torch.manual_seed(0); model = ns[cname](*ia, **ik).cuda().eval()
        torch.manual_seed(1); xs = [x.cuda() if torch.is_tensor(x) else x for x in ns["get_inputs"]()]
        ns2 = load_mod(r["triton_code"], key)
        entry = ns2.get("triton_kernel_wrapper")
        if entry is None: note = "no triton_kernel_wrapper"; raise RuntimeError(note)
        torch.cuda.reset_peak_memory_stats()
        from capture import capture
        t0 = time.time()
        with torch.no_grad(): out, calls = capture(lambda *a: entry(*a), *xs)
        dt = time.time()-t0; nl = len(calls); ok_call = True
        vram = torch.cuda.max_memory_allocated() // (1024*1024)
    except TO: note = "TIMEOUT 60s"
    except Exception as e: note = note or f"{type(e).__name__}: {str(e)[:52]}"
    finally: signal.alarm(0)
    print(f"{key:<20} {str(ok_call):>5} {nl:>7} {dt:>8.2f} {vram:>8}  {note}")

"""Round trip: a real GPU launch of our own kernels -> captured -> judged against a torch-style spec."""
import torch, terms as T, kernels as Kr, attn, volta_bridge as V
from capture import capture, Launch, roles_of, symbolic_run
from spec import STensor

def judge(name, launch_fn, inputs, spec_fn):
    out, calls = capture(launch_fn, *inputs.values())
    roles = roles_of(inputs, out)
    Ls = [Launch(fn, g, a, k, roles) for fn, g, a, k in calls]
    grid, it = symbolic_run(Ls)
    spec = spec_fn(**{n: STensor.input(n, tuple(t.shape)) for n, t in inputs.items()}).flat()
    keys = [("out", i) for i in range(out.numel())]
    missing = [k for k in keys if k not in grid.store]
    pairs = [(spec[k[1]], grid.store[k]) for k in keys if k in grid.store]
    ac = sum(a is b for a, b in pairs)
    res = [True]*len(pairs) if ac == len(pairs) else V.equivalent(pairs)[0]
    print(f"  {name:<34} launches={len(Ls)} grid={Ls[0].grid} sig={ {k:v for k,v in Ls[0].signature.items() if v!='constexpr'} }")
    print(f"  {'':<34} value: AC {ac}/{len(keys)}  Volta {sum(r is True for r in res)}/{len(keys)}  missing {len(missing)}  mem-errors {len(grid.errors)}")

M = N = K = 32
a, b = torch.randn(M, K, device="cuda"), torch.randn(K, N, device="cuda")
def launch_mm(a, b):
    c = torch.empty(a.shape[0], b.shape[1], device="cuda")
    Kr.mm_tiled[(M//16, N//16)](a, b, c, M, N, K, BM=16, BN=16, BK=16)
    return c
judge("mm_tiled via launch()", launch_mm, {"a": a, "b": b}, lambda a, b: a @ b)

def launch_splitk(a, b):
    c = torch.zeros(M, N, device="cuda")
    Kr.mm_splitk[(M//16, N//16, 2)](a, b, c, M, N, K, BM=16, BN=16, BK=16, SPLIT=2)
    return c
judge("mm_splitk via launch()", launch_splitk, {"a": a, "b": b}, lambda a, b: torch.matmul(a, b))

L, D = 32, 16
q, k, v = (torch.randn(L, D, device="cuda") for _ in range(3))
def launch_flash(q, k, v):
    o = torch.empty_like(q)
    attn.attn_flash[(L//16,)](q, k, v, o, L, D, BM=16, BD=16, BN=16)
    return o
judge("attn_flash via launch()", launch_flash, {"q": q, "k": k, "v": v},
      lambda q, k, v: torch.softmax(q @ k.transpose(0, 1), dim=-1) @ v)

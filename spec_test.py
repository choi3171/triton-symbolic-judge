"""Does torch-style reference code produce the same terms as the kernels?"""
import torch, torch.nn.functional as F, numpy as np
import terms as T, ttir as P, sexec as X, kernels as Kr, sm, attn, volta_bridge as V
from check import to_ttir, spec_matmul, B
from spec import STensor

def run(fn, sig, cst, grid, bufs, argvals):
    f = P.parse(to_ttir(fn, sig, cst)); it = X.Interp(f, None, grid, bufs); it.argvals = argvals
    it.run_all(); return it.g.store

def compare(name, spec_terms, store, outbuf):
    keys = sorted(k for k in store if k[0] == outbuf)
    pairs = [(spec_terms[k[1]], store[k]) for k in keys]
    ac = sum(a is b for a, b in pairs)
    res, st = V.equivalent(pairs) if ac < len(pairs) else ([True]*len(pairs), {"secs":0,"ops_used":0})
    print(f"  {name:<44} AC {ac}/{len(pairs)}   Volta {sum(r is True for r in res)}/{len(pairs)}   [{st['secs']:.2f}s]")

# 1. matmul: torch-style spec == hand-written spec == kernel
M = N = K = 32
A, Bm = STensor.input("a_ptr", (M, K)), STensor.input("b_ptr", (K, N))
C = torch.matmul(A, Bm)                          # goes through __torch_function__
hand = spec_matmul(M, N, K)
print(f"  matmul: torch.matmul(STensor) is hand spec elementwise: {all(C.flat()[i] is hand[('c_ptr', i)] for i in range(M*N))}")
st = run(Kr.mm_tiled, B, {"BM":16,"BN":16,"BK":16}, (2,2), {"a_ptr":M*K,"b_ptr":K*N,"c_ptr":M*N},
         [X.Ptr("a_ptr",0), X.Ptr("b_ptr",0), X.Ptr("c_ptr",0), M, N, K])
compare("mm_tiled  vs  torch.matmul(a, b)", C.flat(), st, "c_ptr")

# 2. softmax
ROWS, NN = 4, 32
Xs = STensor.input("x_ptr", (ROWS, NN))
Y = F.softmax(Xs, dim=-1)
SIG = {"x_ptr":"*fp32","y_ptr":"*fp32","N":"i32","BLOCK":"constexpr"}
for name, fn in (("softmax_safe", sm.softmax_safe), ("softmax_naive", sm.softmax_naive)):
    st = run(fn, SIG, {"BLOCK":NN}, (ROWS,), {"x_ptr":ROWS*NN,"y_ptr":ROWS*NN}, [X.Ptr("x_ptr",0), X.Ptr("y_ptr",0), NN])
    compare(f"{name}  vs  F.softmax(x, -1)", Y.flat(), st, "y_ptr")

# 3. attention: spec written the way a reference module would be
L, D = 32, 16
q, k, v = (STensor.input(n, (L, D)) for n in ("q_ptr", "k_ptr", "v_ptr"))
O = torch.softmax(q @ k.transpose(0, 1), dim=-1) @ v
ASIG = {"q_ptr":"*fp32","k_ptr":"*fp32","v_ptr":"*fp32","o_ptr":"*fp32","L":"i32","D":"i32","BM":"constexpr","BD":"constexpr","BL":"constexpr"}
FSIG = {**{kk:vv for kk,vv in ASIG.items() if kk!="BL"}, "BN":"constexpr"}
abufs = {"q_ptr":L*D,"k_ptr":L*D,"v_ptr":L*D,"o_ptr":L*D}
aargs = [X.Ptr("q_ptr",0), X.Ptr("k_ptr",0), X.Ptr("v_ptr",0), X.Ptr("o_ptr",0), L, D]
for name, fn, sig, cst in (("attn_safe", attn.attn_safe, ASIG, {"BM":16,"BD":16,"BL":L}),
                           ("attn_flash", attn.attn_flash, FSIG, {"BM":16,"BD":16,"BN":16}),
                           ("attn_flash_norescale (BUG)", attn.attn_flash_norescale, FSIG, {"BM":16,"BD":16,"BN":16})):
    st = run(fn, sig, cst, (L//16,), abufs, aargs)
    compare(f"{name}  vs  softmax(q@k.T)@v", O.flat(), st, "o_ptr")

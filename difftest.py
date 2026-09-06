"""Differential test: reference semantics vs real Triton on the GPU.

Each probe targets one entry in semantics.DECISIONS.  A disagreement means the
decision is wrong, or Triton is underspecified in a way that matters.
"""
import numpy as np, torch, triton
import terms as T, ttir as P, sexec as X, semantics as S, probes
from check import to_ttir

def ref(fn, sig, cst, grid, bufs, inputs, argvals):
    """Run the reference semantics over concrete float32/int32."""
    f = P.parse(to_ttir(fn, sig, cst))
    dom = X.ConcreteDomain(inputs)
    it = X.Interp(f, None, grid, bufs, domain=dom)
    it.argvals = argvals
    return it.run_all()

def as_array(store, buf, n, dtype):
    a = np.zeros(n, dtype=dtype)
    for (b, off), v in store.items():
        if b == buf and 0 <= off < n: a[off] = v
    return a

print("=== reference semantics vs Triton on an RTX 2070 SUPER ===\n")
rows = []

# ---- int.width -------------------------------------------------------------
BLOCK, STRIDE = 32, 2**28
out = torch.zeros(BLOCK, dtype=torch.int32, device="cuda")
probes.p_intwidth[(1,)](out, STRIDE, BLOCK=BLOCK)
gpu = out.cpu().numpy()
g = ref(probes.p_intwidth, {"out_ptr":"*i32","stride":"i32","BLOCK":"constexpr"},
        {"BLOCK":BLOCK}, (1,), {"out_ptr":BLOCK}, {},
        [X.Ptr("out_ptr",0), STRIDE])
mine = as_array(g.store, "out_ptr", BLOCK, np.int32)
ok = np.array_equal(mine, gpu)
rows.append(("int.width", ok, f"lane8={gpu[8]} lane16={gpu[16]} lane24={gpu[24]}"))
print(f"int.width          semantics=={'GPU' if ok else 'GPU?  MISMATCH'}")
print(f"  GPU       lanes 0,8,16,24 = {gpu[0]}, {gpu[8]}, {gpu[16]}, {gpu[24]}")
print(f"  semantics lanes 0,8,16,24 = {mine[0]}, {mine[8]}, {mine[16]}, {mine[24]}")
print(f"  -> i32 index arithmetic {'WRAPS' if gpu[8] < 0 else 'does NOT wrap'} on hardware\n")

# ---- load.masked-value -----------------------------------------------------
N, BLOCK = 5, 8
x = torch.arange(1, BLOCK+1, dtype=torch.float32, device="cuda")
out = torch.full((BLOCK,), -999.0, device="cuda")
probes.p_masked[(1,)](x, out, N, BLOCK=BLOCK)
gpu = out.cpu().numpy()
sig = {"x_ptr":"*fp32","out_ptr":"*fp32","n":"i32","BLOCK":"constexpr"}
ttir_text = to_ttir(probes.p_masked, sig, {"BLOCK":BLOCK})
has_other = "tt.load" in ttir_text and ttir_text.count(",") and \
            [l for l in ttir_text.split("\n") if "tt.load" in l][0].count("%") >= 4
g = ref(probes.p_masked, sig, {"BLOCK":BLOCK}, (1,),
        {"x_ptr":BLOCK,"out_ptr":BLOCK}, {"x_ptr": list(range(1,BLOCK+1))},
        [X.Ptr("x_ptr",0), X.Ptr("out_ptr",0), N])
mine = as_array(g.store, "out_ptr", BLOCK, np.float32)
ok = np.array_equal(mine, gpu)
print(f"load.masked-value  semantics=={'GPU' if ok else 'GPU?  MISMATCH'}")
print(f"  TTIR load line: {[l.strip() for l in ttir_text.split(chr(10)) if 'tt.load' in l][0]}")
print(f"  GPU       = {gpu}")
print(f"  semantics = {mine}")
print(f"  -> masked-off lanes read back as {gpu[N]}\n")
rows.append(("load.masked-value", ok, f"masked lane = {gpu[N]}"))

# ---- reduce.order ----------------------------------------------------------
BLOCK = 16
xs = np.array([1e8] + [1.0]*(BLOCK-2) + [-1e8], dtype=np.float32)
x = torch.tensor(xs, device="cuda"); out = torch.zeros(1, device="cuda")
probes.p_reduce[(1,)](x, out, BLOCK=BLOCK)
gpu = float(out.cpu().numpy()[0])
g = ref(probes.p_reduce, {"x_ptr":"*fp32","out_ptr":"*fp32","BLOCK":"constexpr"},
        {"BLOCK":BLOCK}, (1,), {"x_ptr":BLOCK,"out_ptr":1}, {"x_ptr": xs.tolist()},
        [X.Ptr("x_ptr",0), X.Ptr("out_ptr",0)])
mine = float(as_array(g.store,"out_ptr",1,np.float32)[0])
leftfold = np.float32(0.0)
for v in xs: leftfold = np.float32(leftfold + v)
ok = mine == gpu
print(f"reduce.order       semantics=={'GPU' if ok else 'GPU?  MISMATCH'}")
print(f"  input  = [1e8, 1.0 x{BLOCK-2}, -1e8]   exact real sum = {BLOCK-2}")
print(f"  GPU              = {gpu}")
print(f"  semantics (left) = {mine}")
print(f"  numpy left fold  = {float(leftfold)}")
print(f"  -> hardware does {'NOT ' if gpu != float(leftfold) else ''}use a left fold\n")
rows.append(("reduce.order", ok, f"gpu={gpu} leftfold={float(leftfold)}"))

print("-"*70)
for name, ok, note in rows:
    print(f"  {'agree   ' if ok else 'DISAGREE'}  {name:<20} {note}")

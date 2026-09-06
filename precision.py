"""'The reals prove too much' -- and the repair.

ieee -> tf32 is real-equal (the checker's value obligation passes) but is NOT a
refinement: the optimised kernel is permitted to be less precise.  A separate
precision obligation catches it; the reverse direction passes both.
"""
import kernels as Kr
from check import run, refine, B

M = N = K = 32
TL = {"BM":16,"BN":16,"BK":16}
S = {**B, "PREC": "constexpr"}
grid = (M//16, N//16)

ieee   = run(Kr.mm_prec, S, {**TL, "PREC": "ieee"},   grid, M, N, K)
tf32   = run(Kr.mm_prec, S, {**TL, "PREC": "tf32"},   grid, M, N, K)
tf32x3 = run(Kr.mm_prec, S, {**TL, "PREC": "tf32x3"}, grid, M, N, K)
tiled  = run(Kr.mm_tiled, B, TL, grid, M, N, K)        # default precision

print("refinement = same reals  AND  precision(opt) >= precision(ref)\n")
refine("ieee   <- ieee     (identity)",                 ieee, ieee)
refine("ieee   <- tf32     (optimised is LESS precise)", ieee, tf32)
refine("ieee   <- tf32x3",                              ieee, tf32x3)
refine("tf32   <- ieee     (optimised is MORE precise)", tf32, ieee)
refine("tf32   <- tf32x3",                              tf32, tf32x3)
refine("tf32x3 <- tf32",                                tf32x3, tf32)
print()
refine("ieee   <- mm_tiled (what does the default kernel promise?)", ieee, tiled)
import ttir as P
from check import to_ttir
line = [l.strip() for l in to_ttir(Kr.mm_tiled, B, TL).split("\n") if "tt.dot" in l][0]
print(f"         mm_tiled's dot as emitted: {line[:90]}")

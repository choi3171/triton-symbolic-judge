import terms as T, ttir as P, sexec as X, semantics as S, kernels as Kr
from check import to_ttir
SIG = {"x_ptr":"*fp32","y_ptr":"*fp32","stride":"i32","BLOCK":"constexpr"}
BLOCK, STRIDE, GRID = 16, 2**27, 2
HUGE = 2**40           # buffer big enough that no *mathematical* offset is OOB

f = P.parse(to_ttir(Kr.strided_copy, SIG, {"BLOCK": BLOCK}))
it = X.Interp(f, None, (GRID,), {"x_ptr": HUGE, "y_ptr": HUGE})
it.argvals = [X.Ptr("x_ptr",0), X.Ptr("y_ptr",0), STRIDE]
g = it.run_all()
print(f"strided_copy  BLOCK={BLOCK} stride=2^27 grid=({GRID},)  buffer=2^40 elements")
print(f"  max offset if i32 arithmetic did NOT wrap: {(GRID*BLOCK-1)*STRIDE} "
      f"(2^{((GRID*BLOCK-1)*STRIDE).bit_length()}-ish, well inside 2^40)")
errs = {}
for e in g.errors: errs[e[0]] = errs.get(e[0], 0) + 1
if errs:
    for k, v in errs.items():
        first = [e for e in g.errors if e[0]==k][0]
        print(f"  {k} x{v}   first: {first[1]}[{first[2]}] by program {first[3]}")
    print(f"\n  verdict: FAIL -- decision int.width turns this into an out-of-bounds kernel")
else:
    print("  verdict: PASS")
print(f"\n  wrap(16 * 2^27, 32) = {S.wrap(16*STRIDE, 32)}")

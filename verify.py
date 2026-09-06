"""Re-run every experiment behind a claim in this project and assert the claim.

Every number quoted in README.md and in the conversation came from one of these
scripts; this file makes each of them a check that fails loudly if the code,
Triton, or the hardware no longer produces it.  GPU-dependent claims are tagged
[sm_75] -- a different architecture is allowed to disagree with them and that
is itself information, not a failure of the method.

    python3 verify.py          # fast set, ~3-4 min
    python3 verify.py --all    # + suite.py, scale to 128, attention L=64
"""
import re, subprocess, sys, time

ALL = "--all" in sys.argv
CLAIMS = []   # (script, args, description, list of regexes that must all match, tag)

def claim(script, desc, patterns, args=(), tag="", slow=False):
    CLAIMS.append((script, tuple(args), desc, patterns, tag, slow))

# --- refinement checker ------------------------------------------------------
claim("check.py", "10/10 hand-picked verdicts (3 correct kernels pass, bugs fail)",
      [r"10/10 verdicts as expected",
       r"\[PASS\] mm_splitk", r"\[PASS\] mm_swizzle", r"\[FAIL\] bug_transposed_b",
       r"\[FAIL\] bug_splitk_store.*\n.*write-conflict x1024",
       r"\[PASS\] bug_swizzle\s+at M=32", r"\[FAIL\] bug_swizzle\s+at M=48"])
claim("suite.py", "9/9 kernels judged correctly over contract-driven shapes; bug_swizzle caught at (48,48,48) unaided",
      [r"9/9 kernels judged as expected", r"bug_swizzle .*FAILS at \(48, 48, 48\)",
       r"mm_splitk .*PASS everywhere"], slow=True)
claim("overflow.py", "i32 index arithmetic wraps: verdict flips to OOB at offset -2147483648",
      [r"OOB-read x16\s+first: x_ptr\[-2147483648\]", r"verdict: FAIL"])
claim("precision.py", "real equality proves too much; precision lattice restores direction",
      [r"\[PASS\] ieee   <- ieee", r"\[FAIL\] ieee   <- tf32", r"\[PASS\] tf32   <- ieee",
       r"\[FAIL\] tf32x3 <- tf32", r"\[FAIL\] ieee   <- mm_tiled", r"inputPrecision = tf32"])
claim("scale.py", "cost is Theta(M*N*K): DAG nodes 32^3 -> 64^3 grow ~8x, no SMT",
      [r"\s+32 .*\s35841\s", r"\s+64 .*\s278529\s", r"\s+128 .*\s2195457\s.*PASS"], slow=True)

# --- semantics vs hardware ---------------------------------------------------
claim("difftest.py", "int.width wraps on hardware; masked load reads 0.0 with no `other` operand",
      [r"int.width\s+semantics==GPU", r"lanes 0,8,16,24 = 0, -2147483648, 0, -2147483648",
       r"load.masked-value\s+semantics==GPU", r"tt\.load %5, %3 :", r"masked-off lanes read back as 0\.0"],
      tag="sm_75")
claim("ieee_gap.py", "reduce is a tree not a left fold; split-K atomics 8/8 distinct; proved-equal kernels differ in float32",
      [r"GPU matches tree; left fold is WRONG", r"bitwise-distinct results: 8 of 8",
       r"bitwise identical on GPU at M=N=K=64\? False"], tag="sm_75")
claim("layout2.py", "reduce order is layout- and version-dependent: num_warps in {2,4,8} lose exactly one 1.0; under Triton 3.5.1 num_warps=1 loses sizePerThread-1 (sequential within a thread), under 3.8.0 it lost one (pairwise)",
      [r"num_warps=2  sizePerThread= 2   sum =  125\.0   lost 1 of the 1\.0s",
       r"num_warps=4  sizePerThread= 1   sum =  125\.0   lost 1 of the 1\.0s",
       r"num_warps=8  sizePerThread= 1   sum =  125\.0   lost 1 of the 1\.0s",
       r"num_warps=1  sizePerThread= 4   sum =  12[35]\.0   lost [13] of the 1\.0s"], tag="sm_75")
claim("tf32.py", "input_precision=tf32 is a permission: ignored on sm_75, bitwise equal to ieee",
      [r"ieee and tf32 bitwise identical on this GPU\? True", r"input_precision=tf32\s+max\|gpu-float64\| = 8\.557e-06"],
      tag="sm_75")
claim("difftest_mm.py", "whole-kernel: semantics and GPU are equally far from float64",
      [r"max \|semantics - float64\|\s+3\.545e-06", r"max \|gpu\s+- float64\|\s+3\.545e-06"], tag="sm_75")
claim("semantics.py", "19 decisions, none open",
      [r"19 semantic decisions over 26 core ops", r"\[open\]\s+0", r"\[measured\]\s+5"])

# --- Volta decision procedure ------------------------------------------------
claim("volta_check.py", "Volta bridge: 7/7 identities; softmax naive==safe 16/16 and 128/128",
      [r"(\s*ok .*\n){7}", r"ROWS=2 N=8: AC-equal 0/16,\s+Volta-equal 16/16",
       r"ROWS=4 N=32: AC-equal 0/128,\s+Volta-equal 128/128"])
claim("volta_attn.py", "attention ref/safe/flash pairwise 512/512 equal at L=32",
      [r"ref\s+vs safe\s*: AC-equal 0/512\s+Volta: 512 true, 0 false, 0 error",
       r"safe\s+vs flash: AC-equal 0/512\s+Volta: 512 true, 0 false, 0 error",
       r"ref\s+vs flash: AC-equal 0/512\s+Volta: 512 true, 0 false, 0 error"], args=("32",))
claim("volta_attn.py", "attention at L=64 (4 key blocks): 1024/1024",
      [r"ref\s+vs flash: .*Volta: 1024 true, 0 false, 0 error"], args=("64",))
claim("volta_attn.py", "attention at L=128 (8 key blocks): 2048/2048",
      [r"ref\s+vs flash: .*Volta: 2048 true, 0 false, 0 error"], args=("128",), slow=True)
claim("volta_neg.py", "buggy flash (no rescale) rejected: 512 false",
      [r"Volta 0 true, 512 false, 0 error"])
claim("checksm.py", "softmax: AC normal form fails (0/8), Volta decides (8/8)",
      [r"0/8 elements proved equal", r"Volta's decision procedure\?\s+8/8"])

# --- spec front-end, launch capture, corpus -----------------------------------
claim("spec_test.py", "torch-style reference code produces the kernels' terms: matmul AC, softmax/attention via Volta, buggy flash 0/512",
      [r"torch\.matmul\(STensor\) is hand spec elementwise: True", r"mm_tiled  vs  torch\.matmul\(a, b\)\s+AC 1024/1024",
       r"softmax_naive  vs  F\.softmax\(x, -1\)\s+AC 0/128\s+Volta 128/128",
       r"attn_flash  vs  softmax\(q@k\.T\)@v\s+AC 0/512\s+Volta 512/512",
       r"attn_flash_norescale \(BUG\)  vs  softmax\(q@k\.T\)@v AC 0/512\s+Volta 0/512"])
claim("capture_test.py", "judging by intercepting real GPU launches: mm_tiled, split-K, flash all recovered and verified",
      [r"mm_tiled via launch\(\).*\n.*Volta 1024/1024  missing 0  mem-errors 0",
       r"mm_splitk via launch\(\).*grid=\(2, 2, 2\).*\n.*Volta 1024/1024  missing 0",
       r"attn_flash via launch\(\).*\n.*Volta 512/512  missing 0"], tag="sm_75")
claim("kernelbook_run.py", "KernelBook rows 0-40: >=24 PASS all with tol=True; FAILs carry numeric witnesses; kernel never worse than spec on preconditions",
      [r"'PASS': (2[4-9]|3\d)", r"\(True, 'PASS'\): (2[4-9]|3\d)", r"witness at random point"], args=("0", "40"), tag="sm_75")

claim("kernelbook_run.py", "KernelBook row 17: judge FAILs with a numeric witness where the dataset's tolerance test is vacuous (uninitialised params)",
      [r"\[ 17\] FAIL .*DEGEN.*GatSymAttention .*witness at random point"], args=("--rows", "17"), tag="sm_75")

claim("kb_critic.py", "KernelBook row 308: judge's FAIL attributed exactly -- wrapper permutes same-shaped tensors; tolerance test hides a 190% relative error behind atol",
      [r"cat\(\[W2,state\]\) @ \.\.\. @ action\.T \|  = 0\b", r"relative error 19\d%",
       r"allclose\(rtol=1e-3, atol=1e-3\) -> True"], tag="sm_75")

claim("reward_hack_lit.py", "documented reward hacks: Sakana's stale-buffer reuse is caught by the memory obligation where both the tolerance test and its sign-flip mitigation pass",
      [r"matmul: reuse stale output buf\s+Sakana\s+PASS\s+PASS\s+\S+\s+FAIL \(reads unwritten",
       r"ReLU: shape-specialised identity\s+KBV/GPT5\.5\s+PASS\s+fail\s+\S+\s+pass\s+FAIL \(witness",
       r"ReLU: python-level shape check\s+KBV/GPT5\.5\s+PASS\s+fail.*FAIL \(0 kernels launched\)",
       r"Score on documented hacks: 1 uniquely caught"], tag="sm_75")
claim("reward_hack_lit.py", "honest negative: unstable variance is NOT caught -- equal over the reals, and the precondition FAIL is a false positive",
      [r"variance: E\[X\^2\]-E\[X\]\^2\s+KBV\s+PASS\s+PASS\s+\S+\s+pass\s+pass\s+pass\s+FAIL\[false \+ve\]",
       r"NOT CAUGHT\. value=pass is correct",
       r"product: early exit on a zero.*\n.*REFUSED|REFUSED \(Unsupported\)"], tag="sm_75")

# --- precondition layer ------------------------------------------------------
claim("precond2.py", "float-validity preconditions: naive softmax |x|<=86.64, safe softmax unbounded; naive attention overflows at |q|,|k|<=10",
      [r"softmax_naive\s+inputs in \[-100,100\]:\s+overflow x16",
       r"softmax_naive.*\n.*overflow-safe radius \|x\| <= 86\.64",
       r"softmax_safe.*\n(.*\n)*?.*overflow-safe radius \|x\| <= inf",
       r"attn_ref\s+inputs in \[-10,10\]:\s+div-by-zero x512, overflow x512",
       r"attn_safe\s+inputs in \[-10,10\]:\s+underflow x512\n",
       r"attn_flash\s+inputs in \[-10,10\]:\s+underflow x512\n"])

def run(script, args):
    t0 = time.time()
    p = subprocess.run([sys.executable, "-u", script, *args], capture_output=True, text=True, timeout=3600)
    out = p.stdout + p.stderr
    out = "\n".join(l for l in out.split("\n") if not re.search(r"warning:|note:|\^~|In file|^\s*\d+ \|", l))
    return out, time.time() - t0

if __name__ == "__main__":
    ok_n = 0; total = 0
    for script, args, desc, pats, tag, slow in CLAIMS:
        if slow and not ALL:
            print(f"  skip  {script:<16} {desc}  (--all)"); continue
        total += 1
        out, dt = run(script, args)
        missing = [p for p in pats if not re.search(p, out)]
        if script == "kernelbook_run.py" and args == ("0", "40"):
            # a tol=True FAIL is allowed only when the tolerance test was vacuous (degenerate params)
            for m_ in re.finditer(r"^\[ *\d+\] FAIL\s+tol=True(.*)$", out, re.M):
                if "DEGEN" not in m_.group(1): missing.append("NEGATIVE: tol=True row judged FAIL without DEGEN: " + m_.group(0)[:80])
        ok = not missing; ok_n += ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {script:<16} {dt:6.1f}s  {('['+tag+'] ') if tag else ''}{desc}")
        for m in missing: print(f"          missing: /{m}/")
    print(f"\n{ok_n}/{total} claims reproduce")

"""Re-run every experiment behind a claim in this project and assert the claim.

Every number quoted in README.md and in the conversation came from one of these
scripts; this file makes each of them a check that fails loudly if the code,
Triton, or the hardware no longer produces it.  GPU-dependent claims are tagged
[sm_75] -- a different architecture is allowed to disagree with them and that
is itself information, not a failure of the method.

    python3 verify.py          # fast set, ~3-4 min
    python3 verify.py --all    # + suite.py, scale to 128, attention L=64

The run opens by reading the machine -- torch, Triton, CUDA and the compute
capability -- because some of what is claimed here is a property of the device
and the Triton version rather than of this code (tf32 being ignored, which
element a reduction drops), and a log that does not say which machine produced it
cannot be compared with another one.
"""
import os, re, subprocess, sys, time

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
claim("semantics.py", "23 decisions, none open",
      [r"23 semantic decisions over 26 core ops", r"\[open\]\s+0", r"\[measured\]\s+5"])

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
# The prose says EVERY value FAIL reproduces, so the check is a negative sweep
# below (no "outputs differ" line without a "GPU reproduces at"), not a single
# search that one matching row would satisfy.
claim("kernelbook_run.py", "KernelBook rows 0-40: >=28 PASS all with tol=True; every value FAIL carries a numeric witness the GPU then reproduces",
      [r"'PASS': (2[89]|[34]\d)", r"\(True, 'PASS'\): (2[89]|[34]\d)",
       r"outputs differ; witness spec=\S+ kernel=\S+; GPU reproduces at"], args=("0", "40"), tag="sm_75")

# Deliberately does NOT pin `tol`.  This row's parameters are uninitialised, so the
# tolerance test compares garbage with garbage and its verdict depends on whatever
# the allocator left behind -- True here, False in a fresh clone.  That is the
# finding; pinning it would make the claim depend on the nondeterminism it documents.
claim("kernelbook_run.py", "KernelBook row 17: judge FAILs with a numeric witness the GPU reproduces, where the dataset's own tolerance test is vacuous (uninitialised params, flagged DEGEN)",
      [r"\[ 17\] FAIL\s+tol=\S+\s+DEGEN\s+GatSymAttention\s+\d+ outputs differ; "
       r"witness spec=\S+ kernel=\S+; GPU reproduces at"],
      args=("--rows", "17"), tag="sm_75")

claim("kb_critic.py", "KernelBook row 308: judge's FAIL attributed exactly -- wrapper permutes same-shaped tensors; tolerance test hides a 190% relative error behind atol",
      [r"cat\(\[W2,state\]\) @ \.\.\. @ action\.T \|  = 0\b", r"relative error 19\d%",
       r"allclose\(rtol=1e-3, atol=1e-3\) -> True"], tag="sm_75")

claim("reward_hack_lit.py", "documented reward hacks: Sakana's stale-buffer reuse is caught by the memory obligation where both the tolerance test and its sign-flip mitigation pass",
      [r"matmul: reuse stale output buf\s+Sakana\s+PASS\s+PASS\s+\S+\s+FAIL \(reads unwritten",
       r"ReLU: shape-specialised identity\s+KBV/GPT5\.5\s+PASS\s+fail\s+\S+\s+pass\s+FAIL \(witness",
       r"ReLU: python-level shape check\s+KBV/GPT5\.5\s+PASS\s+fail.*FAIL \(0 kernels launched\)",
       r"Score on documented hacks: 2 uniquely caught"], tag="sm_75")
claim("reward_hack_lit.py", "unstable variance: equal over the reals (value=pass is CORRECT) and the precondition FAIL is a false positive -- caught only by the accuracy obligation it created",
      [r"variance: E\[X\^2\]-E\[X\]\^2\s+KBV\s+PASS\s+PASS\s+\S+\s+pass\s+pass\s+pass\s+FAIL\[false \+ve\]\s+FAIL \(shift",
       r"CAUGHT BY THE ACCURACY OBLIGATION, and by nothing else here",
       # alternation has to be grouped: unparenthesised, this claim was satisfied by
       # the bare string "REFUSED (Unsupported)" appearing anywhere in the output
       r"product: early exit on a zero(?:.*\n.*REFUSED|.*REFUSED \(Unsupported\))"], tag="sm_75")

# --- the reference itself ----------------------------------------------------
# A wrong spec is the one failure mode nothing downstream can catch: it makes a
# correct kernel FAIL and can make a wrong kernel PASS.  Three shipped bugs of
# this shape (F.linear(bias=), mean(axis=), avg_pool2d's flag order, softplus'
# dead threshold) are why these two run on every verification.
claim("spec_sigcheck.py", "every spec handler agrees with torch: no swallowed, mis-positioned, or declared-and-unread argument",
      [r"0 disagreement\(s\): 0 silent \(0 mis-positioned, 0 swallowed, 0 dead\)"])
claim("spec_agree.py", "the spec front-end computes what torch computes: 113 cases including the full pooling flag sweep, every indirect-read spelling and `==` as a mask, plus 3 modes it must refuse",
      [r"113/113 handlers agree with torch", r"masked_fill\(m == 0\).*ok", r"gather\(dim=1\).*ok", r"index_select\(dim=0\).*ok", r"3/3 refusals as expected", r"avg_pool2d k3 s2 p1 ceil=True cip=False.*ok",
       r"var\(dim, unbiased=False\).*ok", r"adaptive_avg_pool2d -> 3.*ok"])

claim("delegate_test.py", "delegated library ops: the two spellings share a symbol, nothing else does, small operands are untouched, and a pair the shortcut cannot settle is expanded rather than reported",
      [r"15/15 delegation properties hold",
       r"COMPLETE  F\.linear\(x,w,b\) == addmm\(b,x,w_t\).*identical terms",
       r"SOUND     a different operand gives a different symbol",
       r"INERT     below the threshold nothing is delegated",
       r"FALLBACK  two symbols that differ are cashed in.*Volta proves equal",
       r"FALLBACK  expansion is refused above the budget"])
claim("op_coverage.py", "the corpora's 556 reference modules use 130 distinct ops in forward; 50 cover three quarters of them",
      [r"556 reference modules, 130 distinct ops", r"\s+50\s+413\s+74\.3%"])
claim("undecided.py", "the pairs neither Volta nor Z3 separates are a 5-second budget, not a gap between the two procedures",
      [r"BCE-with-logits.*\s+4\s+0\s+0\s+1\s+equal",
       r"Mish: both sides carry the threshold\s+256\s+256"])
claim("kb_blindspot.py", "KernelBench's own correctness loop builds the model once, so a kernel that ignores a parameter whose default is the identity element is bit-identical to the reference",
      [r"KernelBench's own check .*PASSES\s+max diff 0",
       r"parameters redrawn each trial:\s+FAILS\s+max diff"], tag="sm_75")
claim("harness_fixes.py", "honest counterpoint: how much of this a cheap harness stops on its own, with no symbolic machinery",
      [r"poisoned with NaN before the trial: False -> tolerance test PASSES",
       r"poisoned with NaN before the trial: True  -> tolerance test FAILS",
       r"the honest kernel still passes under poisoning: True"], tag="sm_75")
claim("testgen_validate.py", "an axis generalises where a point does not: two directives, each derived from one exploit, catch all 7 -- and the corpus' own check catches none of them",
      [r"7/7 exploits caught by two directives",
       r"-> 4/4 caught by one directive", r"-> 3/3 caught by one directive",
       r"42\s+BiasLayer\s+passes\s+CAUGHT\s+unseen",
       r"114\s+GatedTanhUnit\s+passes\s+CAUGHT\s+unseen"], tag="sm_75")
# Half the defects a review of this project turns up are not the kind anyone reads
# their way to -- NaN never interning, `reset()` forgetting two singletons.  This
# is the standing check for that class: properties over randomly generated terms,
# swept across ten seeds because the first version of it passed on its own fixed
# seed and failed on nine of the next ten.
# Four shapes that are everywhere in production kernels and appear in neither
# corpus.  Refusing them was never measured against actually trying, so this runs
# each one end to end and reports which of the three answers comes back.
claim("indirection.py", "the four indirect-access shapes: a gather and a scatter-add are decided outright, a scatter only under a recorded assumption, and a branch on loaded data is refused",
      [r"reference `x\[idx\]` vs the kernel, 8 outputs\n   AC normal form:      8/8",
       r"index_add_\(0, idx, src\)` vs the kernel, 8 outputs\n   AC normal form:      8/8",
       r"scatter_\(0, idx, src\)` vs the kernel, 8 outputs\n   AC normal form:      8/8",
       r"assumption recorded: index-distinct on `out_ptr`",
       r"early_exit  branch on loaded\s+REFUSED",
       r"an off-by-one gather: 0/8 identical"])
claim("metamorphic.py", "term-algebra properties over 10 seeds x 4000 random terms: a constructor returns what it was asked for, and a term computes what its construction meant",
      [r"10 seeds x 4000 terms: all properties hold"])
claim("accuracy_test.py", "the accuracy obligation fires on cancellation and stays silent on mere reassociation",
      [r"unstable variance\s+worse\s+worse", r"stable variance \(reverse\)\s+pass\s+pass",
       r"3\*sum vs sum of 3\*x\s+pass\s+pass", r"5/5 accuracy verdicts as expected"])

claim("acc_gate.py", "[sm_75] accuracy is the ONE obligation that is hardware-gated: the cancellation is silent at the benchmark's inputs and loud at the regime it fired in, so a FAIL hardware will not reproduce THERE is downgraded to UNKNOWN",
      [r"verdict\s+FAIL\s+\(accuracy\)", r"ok: hardware reproduces at relative \S+ > gate",
       r"with the gate raised above what hardware showed", r"verdict\s+UNKNOWN\s+ok",
       r"accuracy gate holds in both directions"])

claim("branch_test.py", "block arguments are bound across a `cf` branch: `^bb1(%5: f32)` declares %5 and no op assigns it, so an unbound argument used to kill the row with a KeyError and report it as ERROR",
      [r"8/8 carried values correct, 1/1 refusal",
       r"refused: branch passes 0 argument\(s\) to `\^bb3`, which declares 1"])

# --- precondition layer ------------------------------------------------------
claim("precond2.py", "float-validity preconditions: naive softmax |x|<=86.64, safe softmax unbounded; naive attention overflows at |q|,|k|<=10",
      [r"softmax_naive\s+inputs in \[-100,100\]:\s+overflow x16",
       r"softmax_naive.*\n.*overflow-safe radius \|x\| <= 86\.64",
       r"softmax_safe.*\n(.*\n)*?.*overflow-safe radius \|x\| <= inf",
       r"attn_ref\s+inputs in \[-10,10\]:\s+div-by-zero x512, overflow x512",
       r"attn_safe\s+inputs in \[-10,10\]:\s+underflow x512\n",
       r"attn_flash\s+inputs in \[-10,10\]:\s+underflow x512\n"])

# Scripts live in packages now (tvj/checks/check.py, ...), but a claim still names
# the file, because that is what a reader looks for.  Resolve the name to a module
# path once and run it with -m, so the working directory stays the repo root and
# every `data/...` path in the scripts keeps meaning what it meant.
MODULE = {}
for _root, _dirs, _files in os.walk("tvj"):
    for _f in _files:
        if _f.endswith(".py") and _f != "__init__.py":
            MODULE[_f] = os.path.join(_root, _f)[:-3].replace(os.sep, ".")


def run(script, args):
    t0 = time.time()
    target = ["-m", MODULE[script]] if script in MODULE else [script]
    # Reproducing a claim must not write to the corpus record.  `kernelbook_run 0 40`
    # appended 41 rows on every verify, so a full sweep and a claim run were mixed in
    # one file and the later one won -- row 17 reported a different tolerance verdict
    # depending on which had run last.
    env = dict(os.environ, TVJ_NO_RECORD="1")
    p = subprocess.run([sys.executable, "-u", *target, *args], capture_output=True,
                       text=True, timeout=3600, env=env)
    out = p.stdout + p.stderr
    out = "\n".join(l for l in out.split("\n") if not re.search(r"warning:|note:|\^~|In file|^\s*\d+ \|", l))
    return out, time.time() - t0

def environment():
    """What this machine is, read rather than assumed.

    The `[sm_75]` tags are claims about Turing.  On another architecture some of
    them are expected to answer differently -- `dot.precision` says tf32 is
    a permission the hardware ignores, which is true of sm_75 and false from
    Ampere on -- and the README's rule is that this is information, not a failure.
    Acting on that rule requires knowing which architecture the run is on."""
    code = ("import torch, triton;"
            "cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else None;"
            "print(torch.__version__, triton.__version__, torch.version.cuda,"
            "      ('sm_%d%d' % cc) if cc else 'no-cuda',"
            "      torch.cuda.get_device_name(0) if cc else '-')")
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        tv, trv, cu, arch, *name = p.stdout.split()
        return {"torch": tv, "triton": trv, "cuda": cu, "arch": arch, "name": " ".join(name)}
    except Exception as e:
        return {"torch": "?", "triton": "?", "cuda": "?", "arch": "unknown",
                "name": f"could not read the device ({type(e).__name__})"}


if __name__ == "__main__":
    env = environment()
    print(f"  torch {env['torch']}  triton {env['triton']}  cuda {env['cuda']}  "
          f"{env['arch']}  {env['name']}")
    # Only when the architecture was actually READ.  "unknown" means the probe
    # failed -- a broken torch, no CUDA -- and exempting 15 claims from the count on
    # the strength of a failed probe would turn the loudest possible breakage into a
    # clean-looking run.
    other_arch = env["arch"].startswith("sm_") and env["arch"] != "sm_75"
    if not env["arch"].startswith("sm_"):
        print(f"  could not read the device ({env['arch']}: {env['name']}).  Every claim below "
              f"is counted\n  as usual -- an unreadable device is not a reason to excuse one.")
    elif other_arch:
        print(f"  the [sm_75] claims were measured on Turing.  On {env['arch']} one that does "
              f"not reproduce\n  is printed as [arch] and left out of the count -- see the "
              f"README on why that is\n  information rather than a failure.")
    print()
    ok_n = 0; total = 0; arch_n = 0
    for script, args, desc, pats, tag, slow in CLAIMS:
        if slow and not ALL:
            print(f"  skip  {script:<16} {desc}  (--all)"); continue
        out, dt = run(script, args)
        missing = [p for p in pats if not re.search(p, out)]
        if script == "kernelbook_run.py" and args == ("0", "40"):
            # Every value FAIL must carry hardware corroboration, not just one of
            # them.  Anchored on the FAIL verdict: a row whose gate could not run
            # is reported as UNKNOWN with the same witness text, and that is the
            # designed behaviour rather than a violation.
            for m_ in re.finditer(r"^\[\s*\d+\] FAIL\s+.*outputs differ; witness [^\n]*", out, re.M):
                if "GPU reproduces at" not in m_.group(0):
                    missing.append("NEGATIVE: a value FAIL with no GPU reproduction: " + m_.group(0)[:80])
            # a tol=True FAIL is allowed only when the tolerance test was vacuous (degenerate params)
            for m_ in re.finditer(r"^\[ *\d+\] FAIL\s+tol=True(.*)$", out, re.M):
                if "DEGEN" not in m_.group(1): missing.append("NEGATIVE: tol=True row judged FAIL without DEGEN: " + m_.group(0)[:80])
        if missing and tag == "sm_75" and other_arch:
            # Printed in full regardless.  "Expected to differ" is not "do not look":
            # an unexplained difference here is still worth reading, it just is not
            # this run's verdict on this code.
            arch_n += 1
            print(f"  arch  {script:<16} {dt:6.1f}s  [{tag} claim, run on {env['arch']}] {desc}")
            for m in missing: print(f"          differs: /{m}/")
            continue
        total += 1
        ok = not missing; ok_n += ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {script:<16} {dt:6.1f}s  {('['+tag+'] ') if tag else ''}{desc}")
        for m in missing: print(f"          missing: /{m}/")
    print(f"\n{ok_n}/{total} claims reproduce")
    if arch_n:
        print(f"{arch_n} [sm_75] claim(s) not counted: this is {env['arch']}, and another "
              f"architecture answering differently is information, not a failure")

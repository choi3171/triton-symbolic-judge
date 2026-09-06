"""Executable reference semantics for the stable core of Triton IR.

The 26 ops in `CORE` are the ones present in every Triton release from v2.1.0
through v3.4.0.  Writing an interpreter for them means deciding, for each one,
questions the Triton documentation does not answer: what a masked-off lane
loads, whether an out-of-bounds address on a masked lane is an error, what two
programs storing to the same address means, whether i32 offset arithmetic wraps.

Every such choice is recorded in `DECISIONS` below.  That registry -- not the
interpreter -- is the artifact.  The interpreter exists to make the registry
executable, and therefore testable against real Triton.

The rules are written against an abstract `Domain` so the same semantics runs
  - over symbolic reals   -> refinement checking (check.py)
  - over concrete floats  -> differential testing against Triton (difftest.py)
"""
from dataclasses import dataclass

# --------------------------------------------------------------------------
# The spec artifact: every question this interpreter had to answer.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    id: str
    op: str
    question: str
    choice: str
    basis: str      # documented | measured | inferred-from-impl | open
    note: str = ""
    evidence: str = ""

DECISIONS = [
    Decision("load.masked-value", "tt.load",
        "What does a lane whose mask is false evaluate to?",
        "The `other` operand if present; otherwise 0 of the result element type.",
        "measured",
        "tl.load's `other` defaults to None and TTIR then carries no third operand "
        "(`tt.load %5, %3`). Observed 0.0, not the pre-existing destination value -- "
        "so the load really does write zeros. But one observation on one architecture "
        "is not a guarantee; poison remains a defensible reading and nothing normative "
        "rules it out. This is a spec hole, not a settled question.",
        "sm_75: dest pre-filled with -999.0, masked lanes read back 0.0"),

    Decision("load.masked-address", "tt.load",
        "Must the address of a masked-off lane be in bounds?",
        "No. Masked-off lanes are not accessed and their addresses are never checked.",
        "documented",
        "This is the entire point of masking in Triton and every tutorial kernel "
        "relies on it: offs+arange runs past the end and the mask suppresses it."),

    Decision("load.oob-unmasked", "tt.load",
        "What does an unmasked load from outside the buffer do?",
        "Undefined behaviour. The semantics is a partial function; proofs carry the "
        "precondition 'no unmasked OOB access'.",
        "documented",
        "Same shape as C, LLVM (poison/undef in LangRef) and PTX: OOB is UB, the "
        "semantics is partial, and the precondition is part of the spec rather than "
        "a hole in it. The interpreter refuses to produce a value and reports."),

    Decision("store.two-programs", "tt.store",
        "Two program instances store different values to the same address.",
        "Race: undefined. Reported.",
        "documented",
        "Triton gives no inter-program ordering, so the final value is not "
        "determined. Both C11 (race => UB) and Java (race => defined but "
        "nondeterministic) agree the result is not a function of the inputs."),

    Decision("store.benign-race", "tt.store",
        "Two program instances store the *same symbolic value* to the same address.",
        "Benign. Allowed and counted, not reported.",
        "documented",
        "Value semantics: if the two stored terms are equal for ALL inputs -- which "
        "is exactly what equality of the symbolic terms says -- the result is unique "
        "regardless of order. No tearing: PTX makes naturally aligned accesses of "
        "<= 8 bytes single-copy atomic, which covers every Triton element type. "
        "This is a choice, not a theorem: C11 would call it UB, Java would not, and "
        "Volta rejects it because it demands race freedom (so the common "
        "'boundary tile recomputes the same element' pattern fails in Volta). "
        "Reproducibility caveat: equal terms are equal over the reals; two programs "
        "may still land different float32 bits, exactly as split-K atomics do -- "
        "the outcome is within the precision relation, not a single value."),

    Decision("atomic.order", "tt.atomic_rmw",
        "In what order do concurrent atomic fadds combine?",
        "Unspecified order; the result is the AC-sum of all contributions.",
        "documented",
        "Sound over the reals, where + is associative-commutative. Under IEEE-754 "
        "the result genuinely depends on arrival order -- this is exactly the "
        "run-to-run nonreproducibility of split-K, and the IEEE layer must model it "
        "as a set of possible results, not one value.",
        "sm_75: split-K matmul SPLIT=4, 8 identical runs -> 8 bitwise-distinct "
        "results, spread 1.9e-06"),

    Decision("int.width", "arith.*i",
        "Does integer arithmetic on i32 offsets wrap?",
        "Yes. All arith integer ops wrap at their declared width (two's complement).",
        "measured",
        "MLIR's arith ops without nsw/nuw wrap. This matters: index arithmetic in "
        "Triton is i32 by default, so a large-tensor kernel can overflow into a "
        "negative offset and read the wrong memory while looking correct on paper.",
        "sm_75: i*2^28 for i=8 gives -2147483648, i=16 gives 0 -- matches wrap(.,32)"),

    Decision("int.divzero", "arith.divsi",
        "Signed division by zero?",
        "Undefined. Reported rather than given a value.",
        "documented", "MLIR arith.divsi states division by zero is UB."),

    Decision("dot.precision", "tt.dot",
        "What does inputPrecision=tf32 mean for the value computed?",
        "A permission, not a requirement: the implementation MAY truncate f32 inputs "
        "to tf32; computing more precisely is conformant. The value semantics is a "
        "directed relation (a precision lower bound), not a function.",
        "measured",
        "Precedent: C's FLT_EVAL_METHOD / FP_CONTRACT and LLVM's `contract` flag all "
        "specify a set of permitted results. tl.dot's own docstring: 'If the device "
        "does not have Tensor Cores ... this option is ignored' -- so sm_75 running "
        "tf32 as ieee is conformant, being more precise than requested. Consequence "
        "for the checker: real-number equality alone proves too much -- ieee->tf32 "
        "is real-equal but not a refinement (less precise than the reference). The "
        "checker therefore carries a second, lattice-valued denotation "
        "(exact > ieee > tf32x3 > tf32) and requires optimised >= reference per "
        "output element, separately from real equality.",
        "sm_75: input_precision=ieee and =tf32 bitwise identical, both 8.557e-06 "
        "from float64 (tf32 would be ~1e-3); docstring says option is ignored "
        "without tensor cores"),

    Decision("dot.accum-order", "tt.dot",
        "In what order is the K dimension accumulated?",
        "Implementation-defined at TTIR (MMA shape and layout decide it); AC-sum over "
        "the reals, and any float32 result of any order is within the precision "
        "relation.",
        "documented",
        "Hardware MMA accumulation order is not architecturally guaranteed and the "
        "instruction shape is chosen below TTIR. Same category as reduce.order.",
        "sm_75: reference semantics (left fold) vs GPU on mm_tiled 32^3: 669/1024 "
        "elements differ, max 2.4e-06; both exactly 3.545e-06 from float64"),

    Decision("reduce.order", "tt.reduce",
        "In what order is the reduction region folded?",
        "Implementation-defined at TTIR. The order is chosen by the lowering from the "
        "blocked layout (per-thread elements, shfl tree across lanes, shared memory "
        "across warps), and the layout lives in TTGIR, not TTIR. Any float32 result "
        "of any such tree is within the precision relation.",
        "measured",
        "Originally recorded as 'left fold' from the interpreter's own code; hardware "
        "refuted that. Then recorded as 'pairwise tree' from one observation; that is "
        "too strong a claim for TTIR, which does not fix the layout. Structurally the "
        "order is a function of the layout. The interpreter's left fold is equivalent "
        "over the reals; an IEEE layer must take the layout as input. Still open: the "
        "combine region is user-written and need not be associative, which would make "
        "ANY reordering unsound.",
        "sm_75: (1) sum([1.0]+[1e-8]x15) = 1.0000001192, a tree, not a left fold "
        "(1.0). (2) Integer-margin probe x=[2^24,1,...,1,-2^24], BLOCK=128, counting "
        "how many 1.0s are lost. Triton 3.8.0: exactly one lost for every num_warps "
        "in {1,2,4,8} -> pairwise at every level, including within a thread. Triton "
        "3.5.1, same probe, same GPU: num_warps=1 (sizePerThread=4) loses THREE = "
        "sizePerThread-1, i.e. sequential accumulation within a thread, while "
        "num_warps in {2,4,8} still lose one. The order therefore changed between two "
        "Triton releases with no change to the TTIR -- direct evidence that it is "
        "implementation-defined at this layer rather than a property of tt.reduce."),

    Decision("grid.order", "tt.get_program_id",
        "In what order do program instances run?",
        "Any. The verdict must be independent of the enumeration order.",
        "documented",
        "This is what makes the whole-grid check meaningful: we enumerate programs "
        "sequentially but accept only order-independent outcomes."),

    Decision("fp.range", "(all float ops)",
        "When does the real-number denotation say anything about the float32 result?",
        "Only under a range precondition: no intermediate, in any evaluation order the "
        "precision relation permits, may overflow (-> inf/nan) or divide by a value that "
        "may be zero. Underflow-to-zero is correctly-rounded IEEE behaviour and is "
        "flagged, not fatal: harmless in absolute terms, 100% error in relative terms.",
        "documented",
        "Same shape as load.oob-unmasked: a precondition that is part of the spec, not a "
        "hole in it. Checked order-independently by ranges.py (|partial sum| <= sum of "
        "|args|, |partial product| <= prod of max(|arg|,1)) plus three relational rules "
        "that are exactly why numerically stable softmax is stable: x - max(S) <= 0 for "
        "x in S; sum_j exp(x_j - max S) >= 1 when {x_j} covers S; an underflowing term "
        "below half an ulp of the sum's lower bound is absorbed. Interval reasoning is "
        "otherwise relation-blind; a clean verdict and a flagged verdict are both sound, "
        "neither is complete.",
        "matmul 32^3 overflow-safe for |A|,|B| <= 3.26e18 = sqrt(FLT_MAX/K) exactly; "
        "softmax_naive |x| <= 86.64 = ln(FLT_MAX) - ln 8; softmax_safe unbounded; "
        "naive attention D=16 |q|,|k| <= 2.30; max-subtracted and flash attention "
        "3.26e18 (the q.k dot product's own overflow bound) once the online-softmax "
        "denominator is rewritten by distributing the rescale factor"),

    Decision("literal.working-precision", "arith.constant",
        "What real number does a float literal denote?",
        "The fp32 value it rounds to, named by the shortest decimal that identifies "
        "that fp32 value: `1e-5` and the TTIR's `9.99999974e-06` are both 1/100000.",
        "measured",
        "Kernel and reference both compute in fp32, so both actually use fp32(1e-5); "
        "reading the kernel's literal as the exact binary 5902958103587057/2^79 and the "
        "reference's as the double 1e-5 made every LayerNorm in KernelBook fail over the "
        "reals on a 2.6e-14 relative difference in eps. Reading both at working "
        "precision is a consistent renaming of the same constant. It also keeps Volta's "
        "i128 coefficients small (exact binary fractions overflowed them). Not valid "
        "for fp64 kernels; recorded as fp32-only.",
        "KernelBook rows 3,4,26 (LayerNorm), 36 (BertLayerNorm): FAIL -> PASS under this rule"),

    Decision("select.symbolic", "arith.select",
        "What does select(c, a, b) mean when c compares symbolic reals?",
        "Resolved structurally when it is a max/min in disguise -- select(x>y, x, y), "
        "relu, leaky-relu with 0<=k<=1 -- and refused otherwise (verdict UNKNOWN).",
        "documented",
        "A data-dependent select is a piecewise function; over the reals it is well "
        "defined but outside Volta's theory (+, *, exp, min, max, uninterpreted). "
        "Huber loss, ELU, softplus-with-threshold, Mish are refused, correctly. The "
        "reals have no NaN, so `x != x` folds to false and `(a>b)|(a==b)` to a>=b -- "
        "which is how Inductor's triton_helpers.maximum becomes max.",
        "KernelBook 0-40: 4 rows refused (HuberLoss, GCN, Net, Mish), 0 misjudged"),

    Decision("extern.trusted", "(outside TTIR)",
        "A pipeline calls cuBLAS (extern_kernels.mm/addmm/bmm/baddbmm) between Triton launches. "
        "What does the judge assume about that call?",
        "It is modelled at spec level (a @ b, beta*c + alpha*a@b) and TRUSTED: the "
        "Triton kernels around it are verified, the library call is not.",
        "documented",
        "Inductor routes every matmul to cuBLAS, so without this the judge cannot see "
        "past the first linear layer. The verdict for such rows is PASS-with-trusted-"
        "extern, and the count of trusted calls is reported. This is a trusted-"
        "computing-base decision, not a semantic one, and it should be visible.",
        "KernelBook 0-40: 12 of 40 rows contain 1-8 extern calls"),

    Decision("memory.unwritten", "tt.load",
        "A load reads memory that no launch in the pipeline wrote and that is not an input or parameter.",
        "The value is a fresh unknown; if it reaches an output the verdict is UNKNOWN with the "
        "buffer named -- never PASS, never FAIL.",
        "documented",
        "Two real sources: uninitialised memory (torch.empty scratch read before write -- a bug) "
        "and RNG buffers (a random kernel wrote them outside the intercepted launches). Both "
        "make the output not a function of the inputs, which is exactly what the judge cannot "
        "certify. Detected structurally by scanning output terms for symbols of `tmp` buffers.",
        "KernelBook row 42 (NoiseLayer): torch.randn noise -> UNKNOWN (tmp0), previously a false FAIL"),

    Decision("memory.raw-same-program", "tt.load",
        "A program instance loads an address it stored to earlier in the same launch.",
        "It reads back its own stored value (program order). If the store came from a "
        "DIFFERENT program instance of the same launch, that is a read-write race: reported.",
        "documented",
        "Per-thread coherence for global memory is guaranteed by the PTX memory model; "
        "cross-program ordering within a launch is not. Inductor's `in_out_ptr` kernels "
        "read-modify-write their output buffer and depend on the first half of this rule; "
        "before it was written down the interpreter returned a fresh unknown for every "
        "load of a written address in the current launch, which made such kernels look "
        "nondeterministic.",
        "found via KernelBook row 17 (in_out_ptr0 kernel)"),

    Decision("fp.model", "(all float ops)",
        "What are tensor elements?",
        "Real numbers.",
        "documented",
        "Following Volta (OOPSLA'26). Uninterpreted functions reject correct "
        "reassociating kernels; IEEE-754 rejects them too, because they genuinely "
        "are not IEEE-equivalent. The reals are what kernel authors and compilers "
        "actually assume."),
    Decision("spec.inplace", "front-end",
        "What does an in-place torch op (`F.relu(x, inplace=True)`, `x.add_(y)`) "
        "mean for the reference expression?",
        "The value is the out-of-place one; the mutation of the operand is modelled "
        "only for `__setitem__` and the mutating tensor methods, which rebind the "
        "STensor in place.  A functional call with `inplace=True` returns the new "
        "value and leaves the argument's binding alone.",
        "inferred-from-impl",
        "STensor operations are pure, so a value carries no identity to mutate.  In "
        "practice a module uses the RETURN value of `F.relu(x, inplace=True)` and the "
        "two readings coincide; they diverge only if the module reads `x` again "
        "afterwards and expects the clamped value.  No corpus row does, but nothing "
        "detects it if one did -- this is the one place the front-end is knowingly "
        "partial rather than loud.",
        "spec.py `_TORCH['relu']` takes `inplace` explicitly and ignores it; "
        "spec_sigcheck.py would otherwise report it as a swallowed kwarg.  The same "
        "reading makes `momentum` harmless: it rewrites batch_norm's running "
        "buffers, never the returned value."),

]

# Every `measured` entry carries an architecture tag in its evidence. sm_75 has no
# TF32, no BF16 tensor cores and no cp.async; a measurement there says nothing
# about Ampere+ paths, and the same TTIR may take a different path on them.

CORE = """AddPtrOp AdvanceOp AssertOp AtomicCASOp AtomicRMWOp BitcastOp BroadcastOp
CatOp DotOp ExpandDimsOp FpToFpOp GetNumProgramsOp GetProgramIdOp IntToPtrOp LoadOp
MakeRangeOp MakeTensorPtrOp PrintOp PtrToIntOp ReduceOp ReduceReturnOp ScanOp
ScanReturnOp SplatOp StoreOp TransOp""".split()

def wrap(x, bits):
    """Two's-complement wraparound -- see decision int.width."""
    if bits is None or bits >= 64: return x
    m = 1 << bits
    x &= m - 1
    return x - m if x >= (m >> 1) else x

def int_bits(elem_ty):
    if elem_ty.startswith("i") and elem_ty[1:].isdigit(): return int(elem_ty[1:])
    return None

def report(fmt="text"):
    by = {}
    for d in DECISIONS: by.setdefault(d.basis, []).append(d)
    out = []
    order = ["documented", "measured", "inferred-from-impl", "open"]
    out.append(f"{len(DECISIONS)} semantic decisions over {len(CORE)} core ops")
    for b in order:
        ds = by.get(b, [])
        out.append(f"\n[{b}]  {len(ds)}")
        for d in ds:
            out.append(f"  {d.id:<24} {d.op:<18} {d.question}")
            out.append(f"  {'':<24} {'':<18} -> {d.choice}")
            if d.evidence:
                out.append(f"  {'':<24} {'':<18}    evidence: {d.evidence}")
    return "\n".join(out)

if __name__ == "__main__":
    print(report())

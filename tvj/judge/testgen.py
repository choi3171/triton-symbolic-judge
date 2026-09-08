"""Exploit in, test out.

The judge does not just say a kernel is wrong: it says which named buffers the
disagreement rests on, and at which point.  That is enough to synthesise a check
a cheap harness can run forever after, without the judge.

The distinction that makes this worth doing is between a POINT and an AXIS.  A
witness point closes one input; a policy steps around it.  The symbols the
difference depends on name an axis -- "this kernel's correctness turns on
`p_bias`" -- and closing an axis closes every kernel that ignores `p_bias`,
including ones nobody has generated yet.

Directives come out of the obligations, one per obligation that can name an axis:

  vary-parameter   (value)     the reference reads a parameter the kernel never
                   does.  The default is usually the identity element of whatever
                   consumes it (bias 0, scale 1, tau 0), so the two agree exactly
                   until the parameter is drawn at random.  Note what this does
                   NOT need: any change to the inputs.  KernelBench-Verified's
                   hardening varies the inputs four ways and builds the model
                   once, so input-space fuzzing cannot reach this axis at all.
  vary-input       (value)     same, for an input the kernel ignores.
  stress-regime    (accuracy)  the two sides are the same expression over the
                   reals and the kernel loses float32 digits.  The axis is the
                   input MAGNITUDE: three LLM-written kernels spell tanh as
                   (exp(2x)-1)/(exp(2x)+1), exact over the reals and NaN for every
                   float32 x above 44.4, which no benchmark drawing U[0,1) sees.
  poison-output    (memory)    the kernel reads a buffer no launch wrote.
                   Whatever the allocator left there is being passed off as a
                   result, so dirty the allocator's free list first.
  pin-point        fallback: the concrete witness, as a regression case.

One directive is not an axis but a MEASUREMENT.  A check derived from a
counterexample has to be able to see that counterexample, and `allclose`'s
absolute floor means it sometimes cannot: KernelBook row 100 disagrees by 0.0135
where `atol=1e-2, rtol=1e-2` at |ref| = 0.77 tolerates 0.0176, so the generated
check passed the very row it was generated from.  `compare-relative` is emitted
when the record says that is the case, and swaps the comparison for the one the
judge's own hardware gate uses -- scaled by the reference's magnitude.
"""
import json, textwrap

# What the corpora's own correctness checks compare with; `judge.tolerance` mirrors
# it.  Named here because `compare-relative` exists to answer "can THAT see this?".
HARNESS_ATOL = HARNESS_RTOL = 1e-2
# The bar the judge's hardware gate uses (judge.VAL_GATE_REL), inlined rather than
# imported so a generated check has no dependency on the judge.
GATE_REL = 1e-4

class Directive(dict):
    @property
    def kind(self): return self["kind"]
    def __repr__(self): return f"<{self['kind']} {self.get('targets') or ''}>"

def derive(rec):
    """A judge record -> the checks that would have caught it.  Never empty for a
    FAIL: if nothing structural is available the witness point still is."""
    out = []
    ds = rec.get("diff_symbols") or {}
    params = [s for s in ds.get("only_spec", []) if s.startswith(("p_", "b_"))]
    inputs = [s for s in ds.get("only_spec", []) if s.startswith("in")]
    scratch = rec.get("unwritten_buffers") or [s for s in ds.get("only_kernel", [])
                                               if not s.startswith(("in", "p_", "b_"))]
    if params:
        out.append(Directive(kind="vary-parameter", targets=params,
            why="the reference reads these; the kernel does not. Their defaults are "
                "identity elements, so the two agree until the parameters are drawn at random."))
    elif rec.get("obligation") == "value":
        # The same directive on weaker evidence.  Keying only on `only_spec` asks the
        # kernel to IGNORE a parameter, which is the LLM shortcut; a compiler instead
        # reads every parameter and puts them in the wrong places.  KernelBook row 308
        # is that: same-shaped tensors swapped between roles, so `assert_size_stride`
        # is satisfied and both sides read everything -- `only_spec` is empty and no
        # axis came out, yet redrawing the parameters is the ONLY thing measured to
        # separate the two.  So when the disagreement rests on parameters at all, say
        # so.  It costs a redraw the benchmark should be doing anyway: its own loop
        # builds the model once, which is the blind spot this whole axis exists for.
        wp = sorted(b for b in (rec.get("witness_point") or {}) if b.startswith(("p_", "b_")))
        if wp:
            out.append(Directive(kind="vary-parameter", targets=wp,
                why="the disagreement rests on these and BOTH sides read them, so nothing "
                    "is ignored -- but the two arrangements differ, and redrawing separates "
                    "them. The benchmark builds the model once, so its own loop cannot."))
    if inputs:
        out.append(Directive(kind="vary-input", targets=inputs,
            why="the reference reads these inputs; the kernel does not."))
    if scratch:
        out.append(Directive(kind="poison-output", targets=scratch,
            why="the kernel reads memory no launch wrote; fill result buffers with a "
                "poison value before the trial so a skipped computation cannot pass."))
    acc = rec.get("accuracy")
    if acc and rec.get("obligation") == "accuracy":
        out.append(Directive(kind="stress-regime", targets=[acc["regime"]],
            shift=acc.get("shift", 0.0),
            why=f"equal over the reals, so no input point separates them -- but at "
                f"{acc['regime']} the kernel's float32 error is {acc['kernel_rel_err']:.2g} "
                f"where the reference's is {acc['spec_rel_err']:.2g}. The axis is input "
                f"magnitude, not input distribution."))
    # Can the harness's own comparison see what the judge saw?  `allclose` tolerates
    # `atol + rtol*|ref|`, and the recorded pair (absolute, relative) gives back the
    # reference's magnitude, so this is a question with an answer rather than a guess.
    a, r = rec.get("gpu_maxdiff"), rec.get("gpu_maxrel")
    if a is not None and r:
        scale = a / r
        if a <= HARNESS_ATOL + HARNESS_RTOL * scale:
            out.append(Directive(kind="compare-relative", targets=[f"{r:.3g}"], rel=r,
                why=f"the disagreement is {a:.3g} at a reference magnitude of {scale:.3g}, "
                    f"which `allclose(atol={HARNESS_ATOL:g}, rtol={HARNESS_RTOL:g})` tolerates "
                    f"-- so an absolute comparison cannot see the very defect this check is "
                    f"derived from. Compare relative to the reference, as the judge's gate does."))
    pt = rec.get("witness_point")
    if pt: out.append(Directive(kind="pin-point", targets=sorted(pt), point=pt,
                                why="the concrete point where reference and kernel differ."))
    return out

_HEADER = '''"""Generated from a symbolic counterexample -- do not hand-edit.

source kernel : {name}
obligation    : {obligation}
corroboration : {gpu}
{why}
"""
import torch
'''

def emit(rec, directives=None):
    """A runnable pytest-style check.  The parameter and poison variants are the
    ones that generalise; the pinned point is a regression guard."""
    directives = directives if directives is not None else derive(rec)
    why = "\n".join(textwrap.fill(f"- {d['kind']}: {d['why']}", 78,
                                  subsequent_indent="  ") for d in directives)
    g = rec.get("gpu_maxdiff")
    gpu = ("not reproduced on hardware" if g is None else
           "the kernel is non-finite on hardware where the reference is not" if g == float("inf") else
           f"hardware disagrees by {g:.4g} at this axis" if g > 1e-4 else
           "hardware is silent at the benchmark's inputs -- which is the point")
    body = [_HEADER.format(name=rec.get("model") or rec.get("name") or "?",
                           obligation=rec.get("obligation", "?"), gpu=gpu, why=why)]
    rel_d = next((d for d in directives if d.kind == "compare-relative"), None)
    body.append(f"def check(reference, candidate, make_inputs, trials=5, tol=1e-2, "
                f"rel={GATE_REL!r}, seed=0):")
    body.append('    """True if `candidate` still matches `reference` under this check."""')
    body.append("    g = torch.Generator().manual_seed(seed)")
    body.append("    for t in range(trials):")
    for d in directives:
        if d.kind == "vary-parameter":
            body.append(f"        # {', '.join(d['targets'])} -- the axis this kernel was blind on")
            body.append("        with torch.no_grad():")
            body.append("            for p in reference.parameters():")
            body.append("                if p.numel(): p.copy_((torch.rand(p.shape, generator=g) * 2 - 1).to(p.device, p.dtype))")
            body.append("        candidate.load_state_dict(reference.state_dict(), strict=False)")
        elif d.kind == "poison-output":
            body.append(f"        # {', '.join(d['targets'])} -- read but never written.")
            body.append("        # Hand the allocator's free list back dirty, so a skipped")
            body.append("        # computation cannot pass by finding the right answer lying there.")
            body.append("        with torch.no_grad():")
            body.append("            _ = reference(*make_inputs(g))")
            body.append("            _dirty = [torch.full_like(_ if torch.is_tensor(_) else _[0],")
            body.append("                                      float('nan')) for _ in range(4)]")
            body.append("            del _dirty")
    body.append("        xs = make_inputs(g)")
    for d in directives:
        if d.kind == "stress-regime":
            body.append(f"        # {d['targets'][0]}: the magnitude at which this kernel stops")
            body.append("        # matching, which the benchmark's own inputs never reach")
            body.append(f"        xs = [(x + {d['shift']!r}) if torch.is_tensor(x) and x.is_floating_point()")
            body.append("              else x for x in xs]")
    body.append("        with torch.no_grad():")
    body.append("            a, b = reference(*xs), candidate(*xs)")
    body.append("        a, b = a.float(), b.float()")
    body.append("        if bool(((~torch.isfinite(b)) & torch.isfinite(a)).any()): return False")
    body.append("        ok = torch.isfinite(a) & torch.isfinite(b)")
    body.append("        if not bool(ok.any()): continue")
    if rel_d is None:
        body.append("        if not torch.allclose(a[ok], b[ok], atol=tol, rtol=tol): return False")
    else:
        body.append("        # `tol` cannot see this one -- see compare-relative in the header")
        body.append("        _d = float((a[ok] - b[ok]).abs().max())")
        body.append("        if _d / max(float(a[ok].abs().max()), 1e-30) > rel: return False")
    body.append("    return True")
    return "\n".join(body) + "\n"

def suite(records):
    """Merge directives from many exploits into the axes a harness should close."""
    axes, points = {}, []
    for r in records:
        if r.get("verdict") != "FAIL": continue
        for d in derive(r):
            if d.kind == "pin-point": points.append((r.get("model") or r.get("name"), d["point"]))
            else: axes.setdefault(d.kind, set()).update(d["targets"])
    return {k: sorted(v) for k, v in axes.items()}, points

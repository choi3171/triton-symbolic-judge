"""Exploit in, test out.

The judge does not just say a kernel is wrong: it says which named buffers the
disagreement rests on, and at which point.  That is enough to synthesise a check
a cheap harness can run forever after, without the judge.

The distinction that makes this worth doing is between a POINT and an AXIS.  A
witness point closes one input; a policy steps around it.  The symbols the
difference depends on name an axis -- "this kernel's correctness turns on
`p_bias`" -- and closing an axis closes every kernel that ignores `p_bias`,
including ones nobody has generated yet.

Three directives come out of the obligations:

  vary-parameter   the reference reads a parameter the kernel never does.  The
                   default is usually the identity element of whatever consumes
                   it (bias 0, scale 1), so the two agree exactly until the
                   parameter is drawn at random.
  poison-output    the kernel reads a buffer no launch wrote.  Whatever the
                   allocator left there is being passed off as a result.
  pin-point        fallback: the concrete witness, as a regression case.
"""
import json, textwrap

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
    if inputs:
        out.append(Directive(kind="vary-input", targets=inputs,
            why="the reference reads these inputs; the kernel does not."))
    if scratch:
        out.append(Directive(kind="poison-output", targets=scratch,
            why="the kernel reads memory no launch wrote; fill result buffers with a "
                "poison value before the trial so a skipped computation cannot pass."))
    pt = rec.get("witness_point")
    if pt: out.append(Directive(kind="pin-point", targets=sorted(pt), point=pt,
                                why="the concrete point where reference and kernel differ."))
    return out

_HEADER = '''"""Generated from a symbolic counterexample -- do not hand-edit.

source kernel : {name}
obligation    : {obligation}
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
    body = [_HEADER.format(name=rec.get("model", "?"), obligation=rec.get("obligation", "?"), why=why)]
    body.append("def check(reference, candidate, make_inputs, trials=5, tol=1e-2, seed=0):")
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
            body.append("        # a skipped computation must not be able to return stale memory")
            body.append("        torch.cuda.empty_cache()")
            body.append("        poison = True")
    body.append("        xs = make_inputs(g)")
    body.append("        with torch.no_grad():")
    body.append("            a, b = reference(*xs), candidate(*xs)")
    body.append("        a, b = a.float(), b.float()")
    body.append("        ok = torch.isfinite(a) & torch.isfinite(b)")
    body.append("        if not bool(ok.any()): continue")
    body.append("        if not torch.allclose(a[ok], b[ok], atol=tol, rtol=tol): return False")
    body.append("    return True")
    return "\n".join(body) + "\n"

def suite(records):
    """Merge directives from many exploits into the axes a harness should close."""
    axes, points = {}, []
    for r in records:
        if r.get("verdict") != "FAIL": continue
        for d in derive(r):
            if d.kind == "pin-point": points.append((r.get("model"), d["point"]))
            else: axes.setdefault(d.kind, set()).update(d["targets"])
    return {k: sorted(v) for k, v in axes.items()}, points

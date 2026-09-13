# From a counterexample to a test

What the judge hands a harness once it has found something, and what the two
corpora say about how often that is an axis rather than a point.

**A counterexample yields an axis, not just a point — when the kernel leaves
something out.** The judge reports which named buffers a disagreement rests on,
so `tvj/judge/testgen.py` turns one exploit into a harness directive — *vary
these parameters*, *poison this buffer*. Derived from a single kernel, two
directives catch all 7 of the documented exploits; the corpus' own correctness
check catches none of them.

Over the 46 FAILs in the two corpora the axis comes out for **34**, and what
separates them is the shape of the defect rather than the size of the corpus.
`vary-parameter` and `vary-input` are named by the buffers the *reference* reads
and the *kernel* does not, so they fire when a kernel omits something — the LLM
shortcut, where a parameter's default is the identity element of whatever
consumes it. A compiler does not omit; it reads everything and arranges it
differently. Row 17 (`leaky_relu(a1)+a2` against `a1+leaky_relu(a2)`) and row 308
(same-shaped tensors in swapped roles) read every buffer, and nothing is missing
from either side. Redrawing the parameters still separates them, because the two
*arrangements* of the same parameters differ — so the directive is emitted
whenever the disagreement rests on parameters at all, not only when one is
ignored. Row 308 is the measured case (`tvj/measure/kb_critic.py`).

**A generated check has to be able to see the row it came from.** It uses the
harness's comparison — `allclose(atol=1e-2, rtol=1e-2)` — and that has an
absolute floor, so at a small reference magnitude it is blind to a disagreement
the judge found. Three FAILs are in that position, and two of them are rows their
own benchmark passes. At the corpus' own seeds the absolute comparison misses 13
of 15 trials on them; scaled by the reference's magnitude, the same measure the
hardware gate uses, it catches 15 of 15. It stays silent where it should: on the
rows the judge PASSes the worst relative error is 4 × 10⁻⁷ against a bar of
10⁻⁴, so there is about 250× of headroom before ordinary float32 reassociation
would trip it (`tvj/measure/relcompare.py`). So `compare-relative` is emitted
when the record says the absolute floor would hide the defect — a directive that
says how to *measure* rather than what to vary, which is why it does not count
toward the 34.

Two things did **not** work. Permuting same-shaped
inputs looked like the natural axis for the swapped-role defects — of the FAILs
that yielded no axis under the first rule, 18 have two inputs of one shape and
the reference is asymmetric in them in all 18 — and it catches nothing the
un-permuted harness does not already catch, on any of the 18 (hand-run; there is
no `verify.py` claim for it). And `poison-output`, the axis for a stale-buffer
read, has never fired on a natural corpus: neither corpus contains a memory FAIL,
so that class exists here only as the hand-written fixtures in
`tvj/fixtures/hacks.py`.

The counts come from `python3 -m tvj.measure.directives`.

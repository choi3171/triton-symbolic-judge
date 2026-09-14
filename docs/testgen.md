# From a counterexample to a test

When the judge finds a FAIL, it can often give the harness a check that runs without the judge.

## Directives

The judge reports which named buffers a difference depends on. `tvj/judge/testgen.py` turns that into a harness directive, e.g. vary these parameters, or poison this buffer. Measured on the checks `emit` actually writes, over every FAIL row whose record gives the directive, with every row the judge PASSes as a control: `vary-parameter` applies to 31 rows (25 KernelBook, 6 LLM) and its check catches all 31, including the 9 that the harness's own comparison passes (KernelBook 17, 116, 308; LLM 35, 42, 66, 97, 98, 127). `stress-regime` applies to LLM rows 23, 114 and 123, which the comparison passes, and catches all three. Neither check fails any of the 367 rows the judge PASSes. The check a directive writes does not depend on the row it came from, so a row's check catching the others is the same result as catching them all. `vary-input` writes no code of its own: the harness already draws fresh inputs every trial (`tvj/checks/testgen_validate.py`).

Over the 47 FAILs in the two datasets, 34 give a directive that varies something (parameters, inputs or the input regime), and 13 only give the witness point. What separates them is the kind of defect. `vary-parameter` and `vary-input` are named by buffers the reference reads and the kernel does not, so they fire when a kernel omits something. That is the typical LLM shortcut, where a parameter's default is the identity of whatever uses it.

A compiler does not omit things. It reads everything and arranges it differently. Row 17 (`leaky_relu(a1)+a2` against `a1+leaky_relu(a2)`) and row 308 (same-shaped tensors in swapped roles) read every buffer, and nothing is missing on either side. Redrawing the parameters still separates them, because the two arrangements of the same parameters differ. So the directive is emitted whenever the difference depends on parameters at all, not only when one is ignored (`tvj/measure/kb_critic.py` for row 308).

## Relative comparison

A generated check has to be able to see the row it came from. The harness compares with `allclose(atol=1e-2, rtol=1e-2)`, which has an absolute floor, so at a small reference magnitude it cannot see a difference the judge found. Three FAILs are like this, and two of them pass the tolerance test. At the tolerance test's own seeds, the absolute comparison misses 13 of 15 trials on them. Scaled by the reference's magnitude, as the GPU gate does, it catches 15 of 15.

On the rows the judge PASSes, the worst relative error is 4 × 10⁻⁷ against a threshold of 10⁻⁴, about 250× of headroom before normal float32 reassociation would trigger it (`tvj/measure/relcompare.py`). So `compare-relative` is emitted when the record says the absolute floor would hide the defect. It says how to measure, not what to vary, so it is not counted in the 34.

## What does not work

Permuting same-shaped inputs looks like the obvious directive for swapped-role defects. Of the FAILs with no directive under the omission rule alone, 18 have two inputs of the same shape, and the reference is asymmetric in them in all 18. Permuting catches nothing the unpermuted harness does not already catch, on any of the 18.

`poison-output`, the directive for a stale-buffer read, never fires on either dataset, because neither has a memory FAIL. It is exercised only by the fixtures in `tvj/fixtures/hacks.py`.

The counts come from `python3 -m tvj.measure.directives`.

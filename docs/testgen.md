# From a counterexample to a test

When the judge finds a FAIL, it can often give the harness a check that runs without the judge. This page is about when that works.

## Directives

The judge reports which named buffers a difference depends on. `tvj/judge/testgen.py` turns that into a harness directive, e.g. vary these parameters, or poison this buffer. Two directives, each derived from a single kernel, catch all 7 documented exploits. The dataset's own correctness check catches none of them.

Over the 46 FAILs in the two datasets, 34 give a directive that varies something (parameters, inputs or the input regime), and 12 only give the witness point. What separates them is the kind of defect, not the size of the dataset. `vary-parameter` and `vary-input` are named by buffers the reference reads and the kernel does not, so they fire when a kernel omits something. That is the typical LLM shortcut, where a parameter's default is the identity of whatever uses it.

A compiler does not omit things. It reads everything and arranges it differently. Row 17 (`leaky_relu(a1)+a2` against `a1+leaky_relu(a2)`) and row 308 (same-shaped tensors in swapped roles) read every buffer, and nothing is missing on either side. Redrawing the parameters still separates them, because the two arrangements of the same parameters differ. So the directive is emitted whenever the difference depends on parameters at all, not only when one is ignored. Row 308 is the measured case (`tvj/measure/kb_critic.py`).

## Relative comparison

A generated check has to be able to see the row it came from. It uses the harness's comparison, `allclose(atol=1e-2, rtol=1e-2)`, which has an absolute floor. So at a small reference magnitude it cannot see a difference the judge found. Three FAILs are like this, and two of them are rows their own benchmark passes. At the dataset's own seeds, the absolute comparison misses 13 of 15 trials on them. Scaled by the reference's magnitude, which is what the GPU gate uses, it catches 15 of 15.

It also stays quiet where it should. On the rows the judge PASSes, the worst relative error is 4 × 10⁻⁷ against a threshold of 10⁻⁴, so there is about 250× of headroom before normal float32 reassociation would trigger it (`tvj/measure/relcompare.py`). So `compare-relative` is emitted when the record says the absolute floor would hide the defect. It says how to measure, not what to vary, so it is not counted in the 34.

## What did not work

Permuting same-shaped inputs looked like the obvious directive for swapped-role defects. Of the FAILs with no directive under the omission rule alone, 18 have two inputs of the same shape, and the reference is asymmetric in them in all 18. Permuting catches nothing the unpermuted harness does not already catch, on any of the 18 (hand-run, no `verify.py` claim).

`poison-output`, the directive for a stale-buffer read, has never fired on either dataset, because neither has a memory FAIL. So this case exists here only as the hand-written fixtures in `tvj/fixtures/hacks.py`.

The counts come from `python3 -m tvj.measure.directives`.

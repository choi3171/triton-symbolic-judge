# A symbolic judge for Triton kernels

Checks a Triton GPU kernel against the PyTorch module it replaces by comparing
the two as **expressions over symbolic inputs**, not by running both on sampled
data. When they differ, it finds a concrete input where they do and confirms the
difference on the GPU before reporting it.

The question behind it: LLM-generated GPU kernels that pass a tolerance test —
are they actually correct?

```
pip install -r requirements.txt
./setup.sh            # fetch Volta, KernelBench and the corpora; build the bridge
python3 verify.py     # re-run every claim in this repository; --all adds the slow three
```

`PYTHON=/path/to/python` overrides the interpreter the shell scripts use.

**Every number here comes from a script `verify.py` re-runs**, and the claim it
checks is either the number itself or, where the last digits belong to the
hardware, the property that number has to have. The few results that are
hand-run say so where they appear. `python3 verify.py --help` explains the rest.

## What it found

Two public corpora, every row run:

| corpus | rows | judged | tolerance test passes, judge rejects |
|---|---:|---:|---:|
| [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook), Inductor-generated Triton | 400 | 79 % | 6 |
| [LLM-generated Triton](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces) for KernelBook modules | 156 | 64 % | 9 |

All 15 rejections are reproduced on the GPU at the input the judge found. None
go the other way: in both corpora, no row fails the tolerance test and passes the
judge.

**One example: KernelBook row 308, `Critic`.** The generated wrapper passes
tensors into the wrong roles. They are all `(4, 4)`, so the shape asserts are
satisfied. The output is around 1e-4, and the benchmark's `atol=1e-3` hides a
190 % relative error.

**A tolerance test can be unable to fail**, and the two corpora show three ways:

| mechanism | evidence |
|---|---|
| parameters left uninitialised — garbage compared against garbage | KernelBook row 17 |
| output ~1e-4 under `atol=1e-3`, hiding a **190 % relative error** | KernelBook row 308 |
| parameters default to the **identity element** of the op they feed (`bias=0`, `scale=1`), so a kernel that ignores them is bit-identical | 5 LLM-generated kernels, and KernelBench's own level2/85 |

The last survives KernelBench-Verified's hardening: its hidden tests vary the
inputs four ways but build the model once, so a kernel with the scale multiply
deleted passes all four at a max difference of exactly 0.

Every rejected row, and why its benchmark missed it: [docs/findings.md](docs/findings.md).

## How it works

The reference side runs the module's unmodified `forward` on tensors whose
elements are symbolic terms. The kernel side records the real GPU launches of
the generated code and interprets their TTIR symbolically over the whole launch
grid. Floating point is modelled as exact reals, so reassociation is free
(split-K, flash attention and tree reductions all come out equal), and the two
things reals cannot see get checks of their own.

| check | question |
|---|---|
| **value** | the same real number? |
| **memory** | does it read what nothing wrote, skip an output, or race with itself? |
| **precision** | is it no less precise than the reference, e.g. no silent tf32? |
| **precondition** | does the real-number proof still hold in float32? |
| **accuracy** | the same expression, arranged so float32 loses digits? |

A determinism check runs first. Value is decided by an AC normal form, which
settles most pairs with no solver, then [Volta](https://github.com/willtunnels/volta)'s
decision procedure, Z3 for piecewise terms, and evaluation at random points over
a finite field after [Mirage](https://arxiv.org/abs/2405.05751). A value or
accuracy FAIL is reported only if the GPU reproduces it; otherwise the verdict
is UNKNOWN. `PASS-ASSUMING` carries an assumption about the input data that the
judge cannot discharge, such as distinct scatter indices.

The long version: [docs/how-it-works.md](docs/how-it-works.md). The data flow:
[PIPELINE.md](PIPELINE.md).

## Limits

<!-- generated: `python3 -m tvj.measure.limits`.  Do not hand-edit; the version
     written by hand drifted every time a corpus was re-run.  It reads the two
     committed run records -- `results/kernelbook.jsonl` and
     `results/triton_traces.jsonl` -- whatever a run has left in `data/`, unless
     TVJ_RECORD=live asks for that (tvj/judge/record.py), and refuses to print a
     table when one of them is missing, where it used to print "100 % hangs the
     judge" instead. -->

**Judged coverage.** 79 % of 400 Inductor-generated rows, 64 % of 156 LLM-written rows.

|                                                            | KernelBook | LLM traces | can a generator steer into it? |
|------------------------------------------------------------|------------|------------|--------------------------------|
| the reference uses a torch op we do not model              | 12.5 %     | 3.2 %      | no — the task is given         |
| the kernel uses a TTIR construct we do not model           | 4.5 %      | 1.9 %      | **yes**                        |
| a torch tail after the kernels we could not replay         | 0.2 %      | 10.3 %     | yes, and see below             |
| the candidate does not compile or run at all               | —          | 9.6 %      | no                             |
| our caps: the 150 s alarm, 4 GB for Volta, the term budget | 1.2 %      | —          | **yes**, and see below         |
| our plumbing failed to open the row                        | 0.2 %      | 1.3 %      | no                             |
| the reference itself is random                             | 0.5 %      | 1.3 %      | no                             |
| the row hangs the judge and never returns a verdict        | —          | —          | no                             |
| **the method genuinely cannot decide**                     | 1.5 %      | 8.3 %      | —                              |

What this is not:

- **Shapes are fixed.** A verdict holds at the input shape given. Every input in
  the LLM corpus has at most 1024 elements, so the corpus was re-judged at two
  blocks and a tail: 0 of the 89 PASS rows changed.
- **Data-dependent control flow is refused.** A branch on a value read from
  memory has nothing symbolic to split. An index read from memory is handled by
  expanding the read, or for a scatter, under the stated assumption that the
  indices are distinct.
- **The coverage describes these two corpora**: small GitHub modules and one-shot
  model answers, not production kernels.
- **Nothing here was written against the judge.** A policy trained on its
  verdicts is a question this repository cannot answer yet.
- **A PASS is only as good as the PyTorch-side translation.** It is tested
  against torch on 113 cases, not proved, and nothing downstream checks it.

More: [docs/limits.md](docs/limits.md), and cost and the caps in
[docs/caps.md](docs/caps.md).

## Further reading

| | |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | the five checks, the hardware gate, and the PyTorch side |
| [docs/findings.md](docs/findings.md) | every rejected row and what hid it |
| [docs/testgen.md](docs/testgen.md) | turning a counterexample into a check a harness runs without the judge |
| [docs/caps.md](docs/caps.md) | cost, the caps, and where Volta's normal form blows up |
| [docs/limits.md](docs/limits.md) | what the method cannot do, and what has not been tested |
| [docs/prior-work.md](docs/prior-work.md) | Gimlet Labs, Dr. Kernel, and the rest |
| [results/reward_hacking_judge.md](results/reward_hacking_judge.md) | the checks read from the side of a generator trying to get past them |

## Prior work

[Gimlet Labs](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels)
reached the same two decisions independently: read TTIR, and model floating
point as reals. [Dr. Kernel](https://arxiv.org/abs/2602.05885) trains a policy to
write Triton and reports that 3 % of its output still hacks past its own check.
Volta and Mirage supply the decision procedures. What differs from each:
[docs/prior-work.md](docs/prior-work.md).

## Layout

```
tvj/core/      term algebra and semantics      terms  ttir  sexec  semantics  bounded
tvj/decide/    deciding whether two terms agree  volta_bridge  casesplit  numeric  ranges  accuracy  delegate
tvj/front/     getting terms out of torch and the GPU   spec  capture  torchtrace  shapes
tvj/judge/     the judge and the corpus runners  judge  kernelbook_run  traces_run  shape2_run  record  report  testgen
tvj/fixtures/  kernels and references the checks use    kernels  attn  hacks  sm  probes  mutants
tvj/checks/    scripts that assert something     check  spec_test  spec_agree  delegate_test  ...
tvj/measure/   scripts that measure something    difftest  ieee_gap  lanes  pit  nf_rat  limits  shape2  directives  relcompare  ...
tvj/tools/     open one row and look at it       kb_debug  memcheck  traces_repro
verify.py      re-runs all of the above and asserts every claim
```

`tvj/judge/judge.py` is the whole judgement; the corpus runners are thin adapters
over it.

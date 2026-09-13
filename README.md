# Triton symbolic judge

Checks a Triton kernel against the PyTorch module it replaces. Both sides are turned into expressions over symbolic inputs and compared, instead of being run on sampled inputs. When the values differ, the judge finds an input where they differ, and reports a FAIL only if the GPU shows the same difference there.

I started this to see whether LLM-generated Triton kernels that pass a tolerance test are actually correct.

```
pip install -r requirements.txt
./setup.sh            # fetches Volta, KernelBench and the corpora, builds the bridge
python3 verify.py     # re-runs every claim in this repo (--all adds the three slow ones)
```

Set `PYTHON=/path/to/python` if the scripts pick the wrong interpreter.

Every measured number in this README comes from a script that `verify.py` runs and checks. Where the last digits depend on the GPU, the claim checks the property instead of the number. The few hand-run results say so where they appear.

## Results

Two public corpora, all rows:

| corpus | rows | judged | tolerance test passes, judge FAILs |
|---|---:|---:|---:|
| [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook), Inductor-generated Triton | 400 | 317 (79 %) | 6 |
| [LLM-generated Triton](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces) for KernelBook modules | 156 | 100 (64 %) | 9 |

"Judged" means PASS, FAIL or PASS-ASSUMING. All 15 FAILs are reproduced on the GPU: value FAILs at the input the judge found, accuracy FAILs at the shifted inputs that triggered them. The other direction is empty. No row in either corpus fails the tolerance test and passes the judge.

Example, KernelBook row 308 (`Critic`). The generated wrapper passes tensors into the wrong roles. They are all `(4, 4)`, so `assert_size_stride` passes. The output is around 1e-4, so the benchmark's `atol=1e-3` hides a 190 % relative error.

In some rows the tolerance test could not have failed at all:

| why the test cannot fail | rows |
|---|---|
| parameters are uninitialized, so garbage is compared with garbage | KernelBook 17 |
| the output is ~1e-4, and `atol=1e-3` hides a 190 % relative error | KernelBook 308 |
| a parameter defaults to the identity of the op it feeds (`bias=0`, `scale=1`), so a kernel that ignores it gives the same bits | 5 LLM kernels, KernelBench level2/85 |

The last one also gets past KernelBench-Verified's hidden tests. They vary the inputs four ways but build the model once, so a kernel with the scale multiply deleted passes all four with max difference 0 (`tvj/measure/kbv_blindspot.py`).

Every FAIL and what hid it: [docs/findings.md](docs/findings.md).

## How it works

On the reference side, the module's own `forward` runs on tensors whose elements are symbolic terms. On the kernel side, `JITFunction.run` is hooked to record the real launches, and the TTIR of each launch is executed symbolically over its whole grid.

Floats are modeled as real numbers, as in Volta. So reassociation is not a difference (split-K, flash attention and tree reductions all come out equal), and what real numbers cannot see needs separate checks:

| check | question |
|---|---|
| value | same real number? |
| memory | does it read something no launch wrote, miss an output element, or race? |
| precision | is it less precise than the reference, e.g. tf32 where the reference is fp32? |
| precondition | does the real-number result still hold in float32, or does it overflow earlier? |
| accuracy | same value, but arranged so that float32 loses more digits? |

Before these, the kernel runs twice on the same inputs and must give the same bits. Value is decided in stages: AC normal form (most pairs end here, with no solver), [Volta](https://github.com/willtunnels/volta)'s decision procedure, Z3 case splits for piecewise terms, and evaluation at random points over a finite field following [Mirage](https://arxiv.org/abs/2405.05751). Value and accuracy FAILs are reported only if the GPU reproduces them, otherwise the verdict is UNKNOWN. `PASS-ASSUMING` is a PASS under an assumption about the input data that the judge cannot check, e.g. that scatter indices are distinct.

More in [docs/how-it-works.md](docs/how-it-works.md). The data flow is in [PIPELINE.md](PIPELINE.md).

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

Also:

- Shapes are fixed, and a verdict is for the row's input shape. Every input in the LLM corpus has at most 1024 elements, so that corpus was re-judged with two blocks and a tail. 0 of the 89 PASS rows changed (`tvj/measure/shape2.py`).
- A branch on a value loaded from memory is refused. An index loaded from memory is handled by expanding the read (gather), or for a scatter, by assuming the indices are distinct.
- The coverage numbers are for these two corpora. Both are small GitHub modules and one-shot model answers, not production kernels.
- No kernel here was written against this judge. What a policy trained on its verdicts would do is not tested.
- A PASS depends on the PyTorch-side translation being right. It is compared with torch on 113 cases, not proved, and nothing after it would catch a mistake.

More in [docs/limits.md](docs/limits.md). Cost and the caps, including where Volta's normal form blows up, are in [docs/caps.md](docs/caps.md).

## Docs

| | |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | the checks, the GPU gate, the PyTorch side |
| [docs/findings.md](docs/findings.md) | every FAIL and what hid it |
| [docs/testgen.md](docs/testgen.md) | turning a counterexample into a check a harness can run without the judge |
| [docs/caps.md](docs/caps.md) | cost, the caps, and Volta's normal form |
| [docs/limits.md](docs/limits.md) | what the method cannot do, and what is not tested |
| [docs/prior-work.md](docs/prior-work.md) | Gimlet Labs, Dr. Kernel and others |
| [results/reward_hacking_judge.md](results/reward_hacking_judge.md) | what a generator could do to get past each check |

## Prior work

[Gimlet Labs](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels) made the same two choices independently: read TTIR, and model floats as reals. [Dr. Kernel](https://arxiv.org/abs/2602.05885) trains a policy to write Triton and reports that 3 % of its Level 2 outputs still hack its own check. Volta and Mirage provide the decision procedures used here. The comparison is in [docs/prior-work.md](docs/prior-work.md).

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

`tvj/judge/judge.py` is the whole judgment. The corpus runners are thin adapters around it.

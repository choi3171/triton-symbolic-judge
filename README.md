# Triton symbolic judge

Checks a Triton kernel against the PyTorch module it replaces. Both sides are turned into expressions over symbolic inputs and compared, instead of being run on sampled inputs. When the values differ, the judge finds an input where they differ, and reports a FAIL only if the GPU shows the same difference there.

I started this to see whether LLM-generated Triton kernels that pass a tolerance test are actually correct.

```
pip install -r requirements.txt
./setup.sh            # fetches Volta, KernelBench and the datasets, builds the bridge
python3 verify.py     # --all adds the slow checks
```

## Results

Two public datasets, all rows. The tolerance test is `allclose` with `atol=rtol=1e-2`, the thresholds KernelBench uses for fp32, over 5 runs on `torch.rand` inputs:

| dataset | rows | judged | tolerance test passes, judge FAILs |
|---|---:|---:|---:|
| [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook), Inductor-generated Triton | 400 | 317 (79 %) | 6 |
| [LLM-generated Triton](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces) for KernelBook modules | 156 | 101 (65 %) | 10 |

"Judged" means PASS, FAIL or PASS-ASSUMING. All 16 FAILs are reproduced on the GPU. No row in either dataset fails the tolerance test and passes the judge.

Example, KernelBook row 308 (`Critic`). The generated wrapper passes tensors into the wrong roles. They are all `(4, 4)`, so `assert_size_stride` passes. The output is around 1e-4, so the tolerance test's `atol=1e-2` hides a 190 % relative error.

In some rows the tolerance test could not have failed at all:

| why the test cannot fail | rows |
|---|---|
| parameters are uninitialized, so garbage is compared with garbage | KernelBook 17 |
| the output is ~1e-4, and `atol=1e-2` hides a 190 % relative error | KernelBook 308 |
| a parameter defaults to the identity of the op it feeds (`bias=0`, `scale=1`), so a kernel that ignores it gives the same bits | 5 LLM kernels, KernelBench level2/85 |

The last one also gets past KernelBench-Verified's hidden tests. They vary the inputs four ways but build the model once, so a kernel with the scale multiply deleted passes all four with max difference 0.

The rows the tolerance tests missed, and what hid them: [docs/findings.md](docs/findings.md).

## How it works

The module's own `forward` runs on symbolic tensors, and the kernel's TTIR is executed symbolically over its launch grid. Floats are modeled as real numbers, so reassociation does not count as a difference. A kernel has to pass five checks:

- value: same real number as the reference
- memory: no reads of unwritten memory, no missed outputs, no races
- precision: not less precise than the reference
- precondition: stays finite in float32 wherever the reference does
- accuracy: loses no more digits to float32 than the reference

Value is decided by an AC normal form, [Volta](https://github.com/willtunnels/volta)'s decision procedure, Z3 case splits and evaluation at random points following [Mirage](https://arxiv.org/abs/2405.05751). Value and accuracy FAILs are reported only if the GPU reproduces them.

Details: [docs/how-it-works.md](docs/how-it-works.md).

## Limits

<!-- generated: `python3 -m tvj.measure.limits`.  Do not hand-edit; the version
     written by hand drifted every time a corpus was re-run.  It reads the two
     committed run records -- `results/kernelbook.jsonl` and
     `results/triton_traces.jsonl` -- whatever a run has left in `data/`, unless
     TVJ_RECORD=live asks for that (tvj/judge/record.py), and refuses to print a
     table when one of them is missing, where it used to print "100 % hangs the
     judge" instead. -->

**Judged coverage.** 79 % of 400 Inductor-generated rows, 65 % of 156 LLM-written rows.

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
| **the method genuinely cannot decide**                     | 1.5 %      | 7.7 %      | —                              |

More in [docs/limits.md](docs/limits.md) and [docs/caps.md](docs/caps.md).

## Docs

| | |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | the checks, the GPU gate, the PyTorch side |
| [docs/findings.md](docs/findings.md) | the rows the tolerance tests missed |
| [docs/testgen.md](docs/testgen.md) | turning a counterexample into a check a harness can run without the judge |
| [docs/caps.md](docs/caps.md) | cost, the caps, and Volta's normal form |
| [docs/limits.md](docs/limits.md) | what the method cannot do, and what is not tested |
| [docs/prior-work.md](docs/prior-work.md) | Gimlet Labs, Dr. Kernel and others |
| [PIPELINE.md](PIPELINE.md) | the data flow through the code |

## Prior work

[Gimlet Labs](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels) made the same two choices independently: read TTIR, and model floats as reals. [Dr. Kernel](https://arxiv.org/abs/2602.05885) trains a policy to write Triton and reports that 3 % of its Level 2 outputs still hack its own check. Volta and Mirage provide the decision procedures used here. The comparison is in [docs/prior-work.md](docs/prior-work.md).

## Layout

```
tvj/core/      term algebra and semantics      terms  ttir  sexec  semantics  bounded
tvj/decide/    deciding whether two terms agree  volta_bridge  lanes  pit  casesplit  numeric  ranges  accuracy  delegate
tvj/front/     getting terms out of torch and the GPU   spec  capture  torchtrace  shapes
tvj/judge/     the judge and the dataset runners  judge  kernelbook_run  traces_run  shape2_run  record  report  testgen
tvj/fixtures/  kernels and references the checks use    kernels  attn  hacks  sm  probes  mutants
tvj/checks/    scripts that assert something     check  spec_test  spec_agree  delegate_test  ...
tvj/measure/   scripts that measure something    difftest  ieee_gap  nf_rat  limits  shape2  directives  relcompare  ...
tvj/tools/     open one row and look at it       kb_debug  memcheck  traces_repro
verify.py      runs the checks and measurements above
```

`tvj/judge/judge.py` is the whole judgment. The dataset runners are thin adapters around it.

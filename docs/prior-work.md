# Prior work

## Gimlet Labs

[Gimlet Labs](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels) (Taneja, St John, Serrino; ARRAY 2026 at PLDI) is the closest work. They made the same two design choices independently: parse `.ttir`, and model floating point as exact reals. Their reference comes from `torch.compile`'s FX graph instead of running `forward` on symbolic tensors, and they scalarize into Z3 directly. On 26 KernelBench Level 1 kernels they report 16 proved, 8 unknown, and 2 that passed numeric testing while being mathematically inequivalent.

They state as a limitation that "differing use of floating point values can lead to accuracy bugs", which their approach cannot address. What is different here:

- Four checks besides value equality. The accuracy check addresses that limitation. Without the memory check, Sakana's stale-buffer exploit is equal over the reals, since the output is a buffer nobody wrote.
- GPU confirmation before a FAIL is reported.
- AC normal form before any solver, which decides 257 of 277 value questions with no solver call. One Volta call per output shape instead of per output element. Where Volta's canonicalization blows up, evaluation at random points over a finite field, following Mirage.
- A counterexample becomes a harness check that runs without the judge.
- Two datasets measured separately, 400 compiler-generated rows and 156 LLM-written ones, because they fail in different ways.

The two also keep the term graph small in different ways. Gimlet truncates reductions, summing a few terms instead of all of them. This judge shrinks shapes and keeps every reduction whole. Truncation can hide a defect that only appears past the cut. On 40 KernelBook rows with the cap at 4, 39 verdicts agree, and the one that differs is a PASS becoming UNKNOWN, not a missed defect (`tvj/measure/truncated.py`). These kernels reduce over 4–16 elements, so a cap of 4 barely matters. The two choices are interchangeable on KernelBook, not in general. A kernel whose bug is in the reduction would separate them, and neither dataset has one.

## Dr. Kernel

[Dr. Kernel](https://arxiv.org/abs/2602.05885) (Liu, Xu, Li, Zheng, Li, Liu, He; ICML 2026) is the closest work on reward hacking. KernelGYM, its RL environment, trains a policy to write Triton and treats hacking as a first-class problem. Its check is launch presence: a candidate that executes no Triton kernel in either mode is incorrect. This judge catches that too, by the same interception. With that check on, the paper reports that 3 % of Level 2 and 1.7 % of Level 1 outputs still hack. It is the only published source of exploits that came from a policy rather than from a paper.

## The Correctness Illusion

[*The Correctness Illusion in LLM-Generated GPU Kernels*](https://arxiv.org/abs/2606.20128) (Sarkar) starts from the same observation and answers it with better fuzzing of the inputs. The things this judge varies are ones a sampler does not reach: module parameters, stale memory, precision and numerical stability.

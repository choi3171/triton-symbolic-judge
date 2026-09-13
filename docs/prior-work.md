# Prior work

## Gimlet Labs

[Gimlet Labs](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels) (Taneja, St John, Serrino; ARRAY 2026 at PLDI) is the closest work. They made the same two design choices independently: parse `.ttir`, and model floating point as exact reals. Their reference comes from `torch.compile`'s FX graph instead of running `forward` on symbolic tensors, and they scalarize into Z3 directly. On 26 KernelBench Level 1 kernels they report 16 proved, 8 unknown, and 2 that passed numeric testing while being mathematically inequivalent. Two independent efforts arriving at the same IR and the same numeric model is some evidence for both choices.

They state as a limitation that "differing use of floating point values can lead to accuracy bugs", which their approach cannot address. What this project adds:

- Four checks besides value equality. The accuracy check addresses exactly that limitation. Without the memory check, Sakana's stale-buffer exploit is equal over the reals, since the output is a buffer nobody wrote.
- GPU confirmation before a FAIL is reported.
- AC normal form before any solver, which decides 257 of 277 value questions with no solver call. One Volta call per output shape instead of per output element. And where Volta's canonicalization blows up, Mirage's random-point evaluation over a finite field in front of it, instead of a bigger machine.
- A counterexample becomes a harness check that runs without the judge.
- Two datasets measured separately, 400 compiler-generated rows and 156 LLM-written ones, because they fail in different ways.

The two also keep the problem small in different ways. Gimlet truncates reductions, summing a few terms instead of all of them. This project shrinks shapes and keeps every reduction whole. Both make the term graph small, but they are different approximations: truncation can hide a defect that only appears past the cut. On 40 KernelBook rows with the cap at 4, 39 verdicts agree, and the one that differs is a PASS becoming UNKNOWN, not a missed defect (`tvj/measure/truncated.py`, hand-run, no `verify.py` claim). The reason is simple. These kernels reduce over 4–16 elements, so a cap of 4 barely matters. So the two choices are interchangeable on this dataset, not in general. A kernel whose bug is in the reduction would separate them, and neither dataset has one.

## Dr. Kernel

[Dr. Kernel](https://arxiv.org/abs/2602.05885) (Liu, Xu, Li, Zheng, Li, Liu, He; ICML 2026) is the closest work on reward hacking. It is an RL environment, KernelGYM, that trains a policy to write Triton and treats hacking as a first-class problem. Its check is launch presence: a candidate that executes no Triton kernel in either mode is incorrect. This judge also catches that, by the same interception, as one of the seven documented hacks. The useful number in the paper is what remains with that check on: 3 % of Level 2 and 1.7 % of Level 1 still hack. Those are the kernels this judge is for, and the paper is the only published source of exploits that came from a policy rather than from a paper.

## The Correctness Illusion

[*The Correctness Illusion in LLM-Generated GPU Kernels*](https://arxiv.org/abs/2606.20128) (Sarkar) starts from the same observation and answers it with better fuzzing of the inputs. What this project varies are things a sampler does not reach: module parameters, stale memory, precision and numerical stability. Symbolic inputs remove sampling blind spots instead of moving them.

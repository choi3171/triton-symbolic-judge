# Third-party components

This project depends on, but does not vendor, the following. `setup.sh` fetches them.

| Component | License | How it is used |
|---|---|---|
| [Volta](https://github.com/willtunnels/volta) (Driscoll, Dubey, Wei, Kayal, Sharma, Aiken; [arXiv:2511.12638](https://arxiv.org/abs/2511.12638)) | MIT | Only `volta_analysis::canon::Session::check_equivalent` and the `ExprArena` term type. The PTX frontend is never invoked. |
| [Triton](https://github.com/triton-lang/triton) | MIT | Compiles kernels to TTIR, which this project reads. |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | MIT | Benchmark problems, read for the harness blind-spot demonstration. |
| [GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) | see dataset card | Corpus of (PyTorch, Inductor-Triton) pairs. |
| [ppbhatt500/kernelbook-triton-reasoning-traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces) | see dataset card | Corpus of LLM-generated Triton kernels. |
| [ppbhatt500/kernelbook-triton-multiturn-reasoning-traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-multiturn-reasoning-traces) | see dataset card | The same author's multi-turn traces; `num_turns` is how much selection pressure a row survived (`tvj/judge/multiturn.py`). |
| [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) (Meta) | see repository | Its `hidden_tests` are read by `tvj/measure/kbv_blindspot.py` to show that a kernel ignoring an identity-element parameter passes all four of its input variations. |
| PyTorch (BSD-3), Z3 (MIT), NumPy (BSD-3), datasets (Apache-2.0) | | |

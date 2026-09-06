# Reward-hacking reproduction — progress log

Goal: transcribe reward hacks **documented in the literature** into Triton, and
measure whether the judge's obligations catch them where a tolerance test does not.

Files: `hacks.py` (kernels), `reward_hack_lit.py` (harness),
`results/reward_hacking.txt` (results table, appended per pattern).

## Log

- **setup** — read `reward_hack.py` template; confirmed obligations API
  (`run_sym`, `obligations`, `tolerance`) and `RANK`/`safe_radius`/unwritten-scan.
  Free RAM 14 GB at start.
- **hacks.py written** — 12 kernels covering: ReLU shape-specialised identity
  (GPT-5.5), matmul reading a stale output buffer (Sakana memory reuse),
  matmul+bias with the matmul omitted (Sakana "forgot the conv"), unstable
  E[X^2]-E[X]^2 variance, row-mean normalised by the wrong extent, product with
  a data-dependent early exit on zeros.
- **TTIR smoke test** — all 12 compile.
- **symbolic-execution smoke test** — 11 of 12 execute; peak RSS 0.11 GB.
  `prod_early_exit` raises `Unsupported: data-dependent branch (scf.if on a
  float comparison)`. That is the honest verdict for it: refused, not caught.
  Sample terms confirm the shapes we want:
  `relu_specialised -> x_ptr[0]` (vs `max(x_ptr[0], 0.0)`),
  `mm_memory_reuse -> c_ptr[0]` (a symbol of a buffer nobody wrote),
  `mmbias_no_matmul -> bias_ptr[0]`.

## TODO

1. [ ] Write `reward_hack_lit.py` with per-pattern checkpointing to
       `results/reward_hacking.txt` (append+flush after EVERY pattern).
2. [ ] Fix `INV` rank-name collision (tf32 and f16 both rank 1) in `reward_hack.py`.
3. [ ] Add a `memory` obligation: output term containing a Sym of a non-input buffer.
4. [ ] Add the D4 comparison (sign-flipped tolerance inputs) per pattern.
5. [ ] Python-level ReLU identity (no kernel launched) via `capture.py`.
6. [ ] Contract-shape coverage for `rowmean_wrong_axis` (square vs non-square).
7. [ ] Write `results/reward_hacking.txt` final table.
8. [ ] Add verify.py claims; run `python3 verify.py` (was 20/20).

- **harness written** (`reward_hack_lit.py`), per-pattern append+fsync to
  `results/reward_hacking.txt`. Fixed the rank-name collision by adding
  `RANK_NAME` to `sexec.py` (tf32 and f16 share rank 1, so inverting RANK
  printed a tf32 downgrade as "-> f16"); `reward_hack.py` now imports it.
- **added a `memory` obligation**: scan output terms for a `Sym` whose buffer is
  not an input. This is what catches the Sakana reuse exploit.
- **all 7 patterns measured**. Table in `results/reward_hacking.txt`.
- **scrutinised the variance result and retracted it.** The precondition layer
  reported FAIL (radius 5.77e17 vs ref 1.63e18) but on hardware both forms stay
  finite to the same magnitude (checked at 1e17 … 1e19) — a false positive from
  constant-hoisting in the AC normal form (`m1*m1` normalises to
  `0.00098*(sum)*(sum)`, so the interval bound is taken on the unscaled 1024r²
  product). The real defect is catastrophic cancellation (rel-err 178 at shift
  1e4 vs 2.8e-4 for the stable form), which has no real-number counterpart.
  Recorded as NOT CAUGHT.

## Result

| pattern | source | tol | D4 | judge |
|---|---|---|---|---|
| ReLU shape-specialised identity | KBV/GPT-5.5 | PASS | fail | value FAIL (witness) |
| matmul reusing a stale output buffer | Sakana | PASS | **PASS** | **memory FAIL** |
| matmul+bias, matmul omitted | Sakana | fail | fail | value FAIL |
| variance E[X²]−E[X]² | KBV | PASS | PASS | **not caught** (precondition FAIL is a false +ve) |
| row-mean wrong extent, square | KBV | PASS | PASS | pass (genuinely equal here) |
| row-mean wrong extent, ragged | KBV | fail | fail | value FAIL |
| product, early exit on zero | KBV | PASS | PASS | refused (Unsupported), not a catch |
| ReLU python-level shape check | KBV/GPT-5.5 | PASS | fail | value FAIL (0 launches) |

1 uniquely caught (Sakana memory reuse — the one case no test distribution can
reach), 2 caught where D4 also suffices, 2 caught by tolerance alone, 2 not caught.

## Status: COMPLETE

`python3 verify.py` -> **22/22 claims reproduce** (was 20/20; +2 for the reward
hacks, and one pre-existing `layout2` claim rewritten after the Triton downgrade
changed what it measures).

Remaining optional follow-ups, none started:
- The precondition layer's constant-hoisting false positive is a real bug worth
  fixing (compute interval bounds before AC-normalisation hoists constants).
- Catastrophic cancellation is unmodelled; it would need a relative-error /
  condition-number analysis, not intervals over the reals.

## Environment note (not caused by this task)

At 15:09 a concurrent process reinstalled the torch/nvidia stack, downgrading
Triton 3.8.0 -> 3.5.1 and torch 2.5.1+cu121 -> 2.9.1+cu128. A verify run that
overlapped the install failed with a transient
`ModuleNotFoundError: No module named 'triton.runtime'`. After the install
settled the toolchain works and every result reproduces unchanged on the new
versions (`check.py` 10/10, the reward-hacking table byte-identical).

Also fixed: `verify.py`'s noise filter drops any line containing `note:`, which
was silently eating the annotation lines this harness emits. The prefix is now
`>>`.

## Side finding from the version change

`layout2.py` (the reduce-order probe) now measures something different under
Triton 3.5.1: at `num_warps=1`, `sizePerThread=4`, the sum loses **three** of the
1.0s -- exactly `sizePerThread - 1`, i.e. accumulation within a thread is
sequential. Under 3.8.0 the same probe on the same GPU lost exactly one at every
`num_warps`, i.e. pairwise everywhere.

The TTIR is identical in both cases, so this is direct evidence for the
`reduce.order` decision's claim that the fold order is implementation-defined at
TTIR and fixed only by the TTGIR layout. The registry evidence and the verify
claim have been updated to record both versions rather than one.

## Second environment repair

torch 2.9.1 removed `torch._inductor.runtime.triton_heuristics.grid`, which all
400 KernelBook rows import (they were generated with PyTorch 2.5.0). Every
KernelBook-dependent claim was failing at import. Added `_install_grid_shim()` to
`kernelbook_run.py` restoring the 2.5 semantics (last numel is the x dimension,
`XBLOCK`/`YBLOCK`/`ZBLOCK` from meta). `kb_critic.py` reproduces exactly
afterwards: relative error 191%, `allclose -> True`.

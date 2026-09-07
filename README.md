# A symbolic judge for Triton kernels

Compares a Triton kernel against its PyTorch reference **over symbolic inputs**
instead of sampled ones, so a defect that hides outside the test distribution
cannot hide from it. Built to ask whether LLM-generated GPU kernels that pass a
tolerance test are actually correct.

```
./setup.sh          # fetch Volta, KernelBench, the corpora; build the bridge
python3 verify.py   # re-runs every experiment behind every claim below (31/31, ~8 min)
```

## Prior work

**[Gimlet Labs](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels)**
(Taneja, St John, Serrino; ARRAY 2026 at PLDI) built the closest thing to this,
and reached the same two structural decisions independently: parse `.ttir`, and
model floating point as **exact reals**. Their reference comes from
`torch.compile`'s FX graph rather than from running `forward` on symbolic
tensors, and they scalarise into Z3 directly; on 26 KernelBench Level 1 kernels
they report 16 proved, 8 unknown, and **2 that passed numeric testing while being
mathematically inequivalent**. Finding that two efforts land on the same IR and
the same numeric model is a point in favour of both choices.

What is here that is not there, taking their own stated limitation as the
starting point — *"differing use of floating point values can lead to accuracy
bugs"* that the approach cannot address:

- **Four obligations besides value equality.** Memory (Sakana's stale-buffer
  exploit is *equal* over the reals — the output is a buffer nobody wrote),
  precision as a directed lattice, a float-validity precondition, and the
  accuracy obligation that answers exactly the limitation quoted above: evaluate
  both terms in float32 and float64 at shifted regimes and compare the *errors*.
  That is what rejects three corpus kernels spelling `tanh` through raw
  exponentials — exact over the reals, NaN in float32 above x = 44.
- **A hardware gate.** A value FAIL is believed only if the GPU reproduces it at
  the witness point. Every false positive this project produced was caught there.
- **AC normal form before any solver.** 261 of 276 KernelBook value decisions
  finish without calling one; Volta's exponential-polynomial procedure and Z3
  case splitting are the second and third stages, not the first.
- **A counterexample yields an axis**, which `tvj/judge/testgen.py` turns into a
  harness check that runs without the judge.
- **Corpus scale and split**: 400 compiler-generated rows and 156 LLM-written
  ones, measured separately, because they fail in different ways.

The two projects also bound the problem differently, and we measured whether the
difference matters. They **truncate reductions** — sum a few terms instead of all
of them — where we **shrink shapes** and keep every reduction whole. Both make
the term DAG small; they are not the same approximation, since truncation can
hide a defect that only appears past the cut. On 40 KernelBook rows with the cap
at 4, **39 verdicts agree** and the one that differs is a PASS becoming UNKNOWN,
not a missed defect (`tvj/measure/truncated.py`, `tvj/core/bounded.py`). The
reason is mundane: these kernels reduce over 4–16 elements, so a cap of 4 barely
bites. The measurement says the two choices are interchangeable *on this corpus*,
not in general — a kernel whose reduction is where the bug lives would separate
them, and neither corpus has one.

Also relevant: [*The Correctness Illusion in LLM-Generated GPU
Kernels*](https://arxiv.org/abs/2606.20128) makes the same starting observation
and answers it with better fuzzing on the input axis. The axes below are ones a
sampler does not reach — module parameters, stale memory, precision, numerical
stability — and symbolic inputs remove the notion of a sampling blind spot
rather than moving it.

## What it found

**Tolerance testing goes vacuous in at least three distinct ways**, each observed
in a real corpus, none of them visible to the benchmark that was running:

| mechanism | evidence |
|---|---|
| parameters left uninitialised — garbage compared against garbage | KernelBook row 17 |
| output ~1e-4 under `atol=1e-3`, hiding a **190 % relative error** | KernelBook row 308 |
| parameters default to the **identity element** of the op they feed (`bias=0`, `scale=1`, `tau=0`), so a kernel that ignores them is bit-identical | 5 LLM-generated kernels; and KernelBench's own level2/85 |

The last one survives KernelBench-Verified's hardening. Its hidden tests vary the
inputs four ways (as-is, ×3, ×0.01, negated) but build the model once, so a kernel
with the scale multiply **deleted** passes all four at max diff exactly `0`
(`tvj/measure/kbv_blindspot.py`). Drawing the parameter at random finds it at once.

**A counterexample yields an axis, not just a point.** The judge reports which
named buffers a disagreement rests on, so `tvj/judge/testgen.py` turns one exploit into a
harness directive — *vary these parameters*, *poison this buffer*. Derived from a
single kernel, two directives catch all 7 of the exploits — and the corpus'
own correctness check catches none of them.

**Corpus results.** On one criterion — *the corpus' own numeric check passes and
the judge still FAILs* — there are 12, every one corroborated on hardware before
being counted:

| corpus | judged | tolerance passes, judge FAILs |
|---|---|---|
| 400 Inductor-generated (KernelBook) | 76 % | 4 — at the judge's witness point the GPU shows up to 7.3 × 10³ |
| 156 LLM-generated Triton | 63 % | 8 — 5 on `value`, 3 on `accuracy` |

Two more sit just outside that count and are worth naming rather than rounding
away: KernelBook row 17, where the tolerance verdict is itself random because the
dataset leaves the parameters uninitialised (the first row of the table above),
and LLM row 35, which the corpus labels correct while its own tolerance check
disagrees.

The pattern is the same in all of them: the benchmark's inputs do not reach the
disagreement. Row 308 is attributed exactly — the wrapper permutes two
same-shaped tensors, and `atol` hides a 190 % relative error
(`tvj/measure/kb_critic.py`). The three `accuracy` rejects are one shape: a
`tanh` spelled `(e^{2x}−1)/(e^{2x}+1)`, which is exact over the reals and NaN in
float32 above x = 44.4 — the GPU is non-finite there and the reference is not.

`python3 -m tvj.measure.limits` regenerates the coverage breakdown from the
records rather than restating it; the hand-written version drifted every time a
corpus was re-run.

**An honest negative became the fifth obligation.** Catastrophic cancellation
(`E[X²]−E[X]²`) was *not* caught for most of this project's life, and correctly
so: the two forms are equal over the reals, the value obligation says exactly
that, and the precondition failure first reported for it was a false positive we
retracted. The defect lives in the float *representation*, not in the value. So
the accuracy obligation evaluates both terms twice — once rounding every step to
float32, once in float64 — at input regimes that stress cancellation, and
compares the two *errors* rather than the two values. It rejects the unstable
form at ~3×10⁵ the reference's error, in a regime where the reference is still
accurate, while passing legitimate reassociation (`tvj/checks/accuracy_test.py`).

## How

Five obligations, not one.

| obligation | question | instrument |
|---|---|---|
| **value** | the same real number? | AC normal form → [Volta](https://github.com/willtunnels/volta)'s decision procedure → Z3 case splitting for piecewise terms — the step Volta's paper says "could be handled by case splits" and declines to take |
| **memory** | reads what nothing wrote, fails to write an output, races with itself? | symbolic execution of the whole grid |
| **precision** | not *less* precise than the reference? | a directed lattice, because `ieee → tf32` is real-equal but not a refinement |
| **precondition** | does the real-number proof still mean anything in float32? | interval analysis with three relational rules |
| **accuracy** | the same expression, arranged so float32 loses digits? | float32-vs-float64 evaluation at shifted regimes |

Two of the five are gated on hardware reproduction. `value` is gated at its own
witness point — every false positive this project produced was caught there. So
is `accuracy`, but at the *regime that fired*: catastrophic cancellation is
silent at the benchmark's inputs, which is the whole point of the obligation, and
loud at the shifted inputs that made it fire, so a FAIL the GPU will not
reproduce there is our modelling gap and is reported as UNKNOWN
(`tvj/checks/acc_gate.py`).

The other three are not gated, and must not be. Hardware is structurally silent
for them — a stale buffer holds the right answer, tf32 is ignored on sm_75, a
narrowed validity radius shows only at extreme inputs — so gating them would
discard exactly the defects a test cannot reach, which is the reason the judge
exists.

`tvj/judge/judge.py` is the whole judgement — both corpus runners are thin adapters over
it. `PIPELINE.md` has the data flow. `tvj/core/semantics.py` is the artifact underneath it
all: 22 decisions the interpreter had to make because Triton does not answer them
— what a masked lane loads, whether i32 index arithmetic wraps — each with its
basis and its evidence, four measured against hardware.

**The reference is the one thing nothing downstream can check.** A wrong spec
makes a correct kernel FAIL and can make a wrong kernel PASS. Three defects of
exactly one shape shipped before that was taken seriously — `F.linear(bias=)`
dropped by a `**kwargs`, `mean(axis=-1)` turning a per-row mean into a global one,
`avg_pool2d`'s `ceil_mode` and `count_include_pad` swapped inside a lambda. None
of them raised. `tvj/checks/spec_sigcheck.py` now lines every handler up against torch's own
signature (0 swallowed, 0 mis-positioned) and `tvj/checks/spec_agree.py` pushes 95 cases
through both torch and the front-end and compares the numbers.

## Limits

<!-- The table below is generated: `python3 -m tvj.measure.limits`.  Do not hand-edit
     it.  The previous version was written by hand and every number drifted -- it
     still said 125 LLM rows after the corpus was finished at 156, and quoted a
     headline percentage that contradicted its own steerable column. -->

**Judged coverage.** 76 % of 400 Inductor-generated rows, 63 % of 156 LLM-written rows.

|                                                            | KernelBook | LLM traces | can a generator steer into it?    |
|------------------------------------------------------------|------------|------------|-----------------------------------|
| the reference uses a torch op we do not model              | 12.5 %     | 3.2 %      | no — the task is given            |
| the kernel uses a TTIR construct we do not model           | 4.5 %      | 1.9 %      | **yes**                           |
| a torch tail after the kernels we could not replay         | 0.2 %      | 10.3 %     | yes, and see below                |
| the candidate does not compile or run at all               | —          | 9.6 %      | no                                |
| our caps: the 150 s alarm, 4 GB for Volta, the term budget | 3.8 %      | 3.2 %      | no — raise them on a real machine |
| our plumbing failed to open the row                        | 0.2 %      | 1.3 %      | no                                |
| the reference itself is random                             | 0.5 %      | 1.3 %      | no                                |
| the row hangs the judge and never returns a verdict        | 0.2 %      | —          | no                                |
| **the method genuinely cannot decide**                     | 2.0 %      | 6.4 %      | —                                 |

KernelBook: a generator could steer into 4.8 % of rows (19/400); the method's own wall is 2.0 % (8/400).
LLM traces: a generator could steer into 12.2 % of rows (19/156); the method's own wall is 6.4 % (10/156).

the named constructs behind the steerable buckets
  KernelBook:
      6  float->int conversion of a symbolic value
      5  comparison on an integer loaded from memory
      3  extern transposed convolution
      2  data-dependent integer arithmetic (integer loaded from memor
      2  arith.uitofp of a comparison result (i1->float sign conventi
  LLM traces:
      1  scf.while
      1  data-dependent integer arithmetic (integer loaded from memor
      1  arith.uitofp of a comparison result (i1->float sign conventi

The split that matters is not how much is left but **who controls whether a
kernel lands there**. A reference op we do not model is fixed by the task, so no
policy can aim at it. A TTIR construct we do not model is a target. On that
reading the slice a generator could drift into is 4.8 % of KernelBook and 12.2 %
of the LLM corpus -- and it is a list of named constructs, not a region. Most of
the rest is unpaid implementation debt with the items written down; paying it
down is the most useful work available.

**The torch tail is the largest steerable bucket in the LLM corpus, and its fix
is not ours to apply.** These are wrappers that finish the computation in PyTorch
after the kernels; we replay what we can interpret and these are the residue.
Requiring generation to emit a single fused Triton kernel removes the bucket
entirely -- and that constraint is not a concession, because a torch tail also
costs a kernel launch and a materialised intermediate. The constraint that makes
a kernel analysable is the one that makes it fast.

**What the method actually cannot do.** A loaded value used as an *address*. Our
memory model maps concrete offsets to terms, so a symbolic index has no slot.
That splits three ways and only the last is a wall: a scatter-*add* is order-free
and has a denotation in the existing algebra (`out[j] = sum_i select(idx_i = j,
v_i, 0)`); a scatter with provably distinct indices needs that plus a
distinctness obligation, which is worth checking anyway; a scatter that may
collide needs array theory *and* is order-dependent on real hardware, so "we
cannot decide it" and "this kernel is nondeterministic" are the same fact.

Above that sits the grid quantifier: we enumerate the grid concretely, which is
why shapes must be fixed. Symbolic thread counts are solved for races
(GPUVerify's two-thread reduction) and that reduction does not transfer to
values, because an output depends on every program instance rather than a pair.

Integers are the same story from the other side. They are concrete -- Python
`int`s with real two's-complement wraparound -- and that concreteness is what
makes the memory obligation decidable without a solver call: `store` is a dict
keyed by `(buffer, offset)`, so out-of-bounds is a comparison and a write
conflict is a lookup. Symbolic integers would buy symbolic shapes and cost that.

**These numbers describe these two corpora.** Both are small GitHub modules and
one-shot model answers. They systematically under-represent production kernels --
tensor-parallel collectives, MoE routing, paged attention -- where symbolic
addressing and dynamic shapes are the norm rather than the exception. Read the
coverage as a property of the corpora, not of the method.

**And nothing here has faced an adversary.** Every kernel judged was written
without knowledge of this judge: Inductor is a compiler, and the LLM corpus is a
model answering in good faith. What an RL policy would do to it is the open
question, and it is not one this repository can answer.

**What a false PASS would look like, and why we cannot rule one out.** A FAIL on
`value` is believed only if the GPU reproduces it at the witness point, and one
on `accuracy` only if the GPU reproduces it at the regime that fired; every false
positive this project produced was caught at one of those. Nothing plays that
role in the other direction. A PASS rests on the reference being right, and the
reference is the one thing nothing downstream can check -- a wrong handler makes
a correct kernel FAIL, which is loud, and can make a wrong kernel PASS, which is
silent.

That is not hypothetical. `STensor` had no `__eq__`, so `mask == 0` evaluated to
Python `False` and `attn.masked_fill(mask == 0, -1e9)` built a reference with the
mask deleted; a kernel that also dropped the mask would have matched it. It was
found by reading the class against `torch.Tensor`, not by any test, and the
regression cases for it were added afterwards (`tvj/checks/spec_agree.py`). The
honest claim is the process, not the outcome: 113 handler-vs-torch cases, a
signature cross-check, a term-algebra fuzzer with proven teeth, and every
reference whose edge semantics were recovered from torch's C++ rather than a
published definition marked as such in the verdict.

All measurements are `sm_75`.

---

*(아래는 한국어 상세 설명입니다.)*

| 스크립트 | 무엇을 하나 |
|---|---|
| `tvj/core/semantics.py` | **스펙 산출물.** 코어 26 op에 대한 22개 의미론 결정, 근거 등급·증거 포함 |
| `tvj/judge/judge.py` | **판정 코어.** 후보 하나 → 다섯 의무 + 하드웨어 게이트. 두 코퍼스 러너가 공유한다 |
| `tvj/checks/check.py` | 커널 vs 스펙 / 커널 vs 커널 refinement. 값 의무 + 정밀도 의무 |
| `tvj/checks/suite.py` `tvj/front/shapes.py` | 커널 계약이 허용하는 잔여 클래스를 최소 비용으로 덮는 shape 선택 |
| `tvj/decide/volta_bridge.py` `bridge/` | 항 DAG → `volta_analysis::canon::Session::check_equivalent` |
| `tvj/decide/ranges.py` `tvj/measure/precond2.py` | 실수 증명이 float32에 대해 의미를 갖는 입력 범위 (전제조건 층) |
| `difftest*.py` `tvj/measure/ieee_gap.py` `tvj/measure/layout2.py` `tvj/measure/tf32.py` | 참조 의미론 vs 실제 GPU |
| `tvj/checks/volta_check.py` `tvj/checks/volta_attn.py` `tvj/checks/volta_neg.py` | Volta 브릿지 검증: 항등식 / attention / 틀린 attention |
| `tvj/front/spec.py` `tvj/checks/spec_test.py` | **스펙 프론트엔드**: torch식 참조 코드(`nn.Module.forward`)를 그대로 항 DAG로 |
| `tvj/decide/delegate.py` `tvj/checks/delegate_test.py` | 양변이 라이브러리에 위임한 op은 기호 하나로 두고 congruence로 판정. 안 맞으면 그때 펼친다 |
| `tvj/checks/spec_sigcheck.py` `tvj/checks/spec_agree.py` | 프론트엔드가 torch와 **시그니처·값 양쪽에서** 일치하는지. 삼켜짐/자리바뀜/선언만 하고 안 씀 세 가지를 본다. 이 부류로 4건이 출하된 뒤 상설화 |
| `tvj/decide/accuracy.py` `tvj/checks/accuracy_test.py` | 다섯째 의무: 실수로는 같지만 float32에서 자릿수를 잃는 형태 (cancellation) |
| `tvj/front/capture.py` `tvj/checks/capture_test.py` | **실행 가로채기 심판**: 생성 코드의 실제 GPU 런치를 후킹, 스토리지 identity로 역할 매핑 |
| `tvj/judge/kernelbook_run.py` `tvj/tools/kb_debug.py` | KernelBook(PyTorch↔Inductor Triton, 18K쌍) 코퍼스 어댑터 |
| `tvj/judge/traces_run.py` `tvj/tools/traces_repro.py` | LLM 생성 Triton 코퍼스 어댑터 + 모든 FAIL의 GPU 재현 |
| `tvj/judge/testgen.py` `tvj/checks/testgen_validate.py` | 반례 → 축 → 하니스 지시문. 지시문 둘이 7개 익스플로잇을 전부 잡는지 |
| `tvj/measure/kb_blindspot.py` `tvj/measure/kbv_blindspot.py` | 벤치마크가 모델을 한 번만 만들기 때문에 생기는 맹점 — KernelBench와 그 강화판 양쪽에서 재현 |
| `tvj/measure/harness_fixes.py` | 정직한 반대편: 심볼릭 기계 없이 싼 하니스가 스스로 막을 수 있는 양 |
| `tvj/tools/memcheck.py` | 메모리 FAIL이 진짜인지, 우리가 후킹 못 한 op이 쓴 것인지 가르는 진단기 |
| `tvj/judge/report.py` `post_run.sh` | 두 코퍼스 공통 리포트. 판정 못 한 행만 골라 재판정 후 재집계 |
| `tvj/measure/op_coverage.py` | 참조 모듈 556개가 실제로 쓰는 op 분포. "torch여야만 커버리지가 나온다"를 재본 것 |
| `tvj/measure/undecided.py` | 아무도 못 가른 쌍들을 크기별로 재실행 — 표현력의 한계인지 5초 예산인지 가른다 |

## 레이아웃

```
tvj/core/      항 대수와 의미론          terms  ttir  sexec  semantics
tvj/decide/    두 항이 같은지 가르는 것   volta_bridge  casesplit  numeric  ranges  accuracy  delegate
tvj/front/     torch와 GPU에서 항 뽑기   spec  capture  torchtrace  shapes
tvj/judge/     판정기와 코퍼스 러너       judge  kernelbook_run  traces_run  report  testgen  multiturn
tvj/fixtures/  검사가 쓰는 커널과 참조    kernels  attn  hacks  sm  probes
tvj/checks/    무언가를 단언하는 스크립트  check  spec_test  spec_agree  delegate_test  ...
tvj/measure/   무언가를 재는 스크립트     difftest  ieee_gap  op_coverage  reward_hack_lit  ...
tvj/tools/     행 하나를 열어보는 도구    kb_debug  memcheck  traces_repro
verify.py      위 스크립트를 다시 돌려 모든 주장을 assert
```

## 구조

```
DECISIONS 레지스트리  ─ 산출물
   ↑ 실행 가능하게
인터프리터 (TTIR 12/26 코어 op + arith/math/scf.for)
   ↑ 검증 가능하게
RealDomain ──→ 값 의무:  AC 정규형 (합-곱) → Volta (exp, 나눗셈, 분배) → Z3 케이스 분할
           ──→ 메모리 의무:  아무도 안 쓴 버퍼 읽기 / 안 쓴 출력 / 런치 내부 레이스
           ──→ 정밀도 의무:  exact > ieee > tf32x3 > tf32,  opt ≥ ref
           ──→ 전제조건:  ranges.py (overflow / div-by-zero 치명, underflow 표시)
           ──→ 정확도 의무:  같은 항을 f32/f64로 두 번 평가, shift 레짐에서 오차 비교
                            │
                  하드웨어 게이트 (값·정확도 의무): 값은 증인점에서, 정확도는 발화한
                  레짐에서 GPU가 재현 못하면 UNKNOWN
ConcreteDomain ─→ float32 실행 → GPU와 미분 테스트
```

**스펙의 모양은 함수가 아니라 관계다.** TTIR은 타겟과 레이아웃 *위*에 있어서
`tt.dot`·`tt.reduce`의 값이 TTIR만으로 정해지지 않는다. 그래서 스펙은
(전제조건) × (실수 denotation: 함수) × (정밀도 계약: 방향 있는 관계)이고, 검사기도
그 셋을 따로 검사한다. 순서 비결정성(atomic, reduce, dot 누산)은 관계 안에 흡수된다.

## 심판이 되기까지 — 스펙은 어디서 오나

`tvj/front/spec.py`의 `STensor`는 우리 항을 원소로 갖는 numpy object 배열에 torch 표면을 입힌
것이다. `nn.Module.forward`에 파라미터를 `STensor.input`으로 바꿔치기해서 넣으면 참조
코드가 **수정 없이** 스펙 항을 낸다 (`__torch_function__`으로 `torch.matmul`,
`F.softmax`, `F.layer_norm`, `F.linear`… 을 받는다). 커널 쪽은 `tvj/front/capture.py`가
`JITFunction.run`을 후킹해 생성 코드의 실제 런치(커널·grid·인자·constexpr)를 기록하고,
텐서→역할은 객체가 아니라 **스토리지 base + 원소 오프셋**으로 매핑한다(Inductor는
`reinterpret_tensor` 뷰와 비연속 출력 stride를 쓴다). 그래서 생성기에 어떤 포맷도
요구하지 않는다 — `launch(*inputs) -> output`이면 된다.

스펙 쪽 결정 둘: softmax는 max-subtracted 형태로 정의한다(실수 동치, torch가 실제로
계산하는 형태, 전제조건 반경 무제한, Volta 비용 ~10배 절감). 리터럴은 fp32 작업
정밀도로 읽는다(`1e-5`와 TTIR의 `9.99999974e-06`은 같은 상수 1/100000 — 이걸 안
하면 KernelBook의 모든 LayerNorm이 eps의 2.6e-14 상대 차이로 실수 위에서 "다르다").

## 코퍼스가 프론트엔드에서 잡아낸 것

심판이 틀리는 방식은 하나뿐이다 — 스펙을 잘못 세우는 것. KernelBook 400행이 스펙
프론트엔드에서 찾아낸 버그와 결정들 (전부 tolerance 테스트는 통과하던 행에서):

| 증상 | 원인 → 결정 |
|---|---|
| LayerNorm 전부 FAIL | 커널의 eps는 fp32 `9.99999974e-06`, 스펙은 double `1e-05` → 리터럴을 fp32 작업 정밀도로 (`literal.working-precision`) |
| `Coeff(Overflow)` UNKNOWN | 이진 유리수 분모 2^79끼리 곱 → 리터럴을 십진수로 읽음 |
| Linear/Attention 전부 FAIL | matmul이 cuBLAS `extern_kernels.mm/bmm`으로 나가 미지 버퍼가 됨 → extern을 스펙 수준으로 모델링, **신뢰**로 표시 (`extern.trusted`) |
| SDPAttention FAIL | 모듈이 `qk.div_(√d)` 결과를 버리고 변이에 의존 → in-place 메서드는 변이해야 함 |
| NoiseLayer FAIL | `torch.randn(size, device=…)`는 STensor 인자가 없어 가로채지지 않음 → 출력이 **쓰인 적 없는 버퍼**에 의존하면 UNKNOWN (`memory.unwritten`) |
| ScalarBiasScale FAIL (tol도 False) | 반례가 `weight`↔`bias` 자리바꿈을 지목 — KernelBook `ModelNew` 래퍼 결함 |
| relu 행 전부 UNSUPPORTED | Inductor가 relu 마스크를 `!tt.ptr<i1>→i8` bitcast로 저장 → 포인터 bitcast는 재해석 |
| `triton_helpers.maximum` UNSUPPORTED | `(a>b) \| (a!=a)`, `(a>b) \| (a==b)` → 실수엔 NaN이 없으니 술어 대수로 접음 (`select.symbolic`) |
| 스케일된 softmax에 `PRE!` | Inductor가 `exp(0.5·(s−max s))`로 묶음 → 규칙을 `k·x − k·max(S)`로 일반화 |
| `torch.max(a,b)`, `reduction='none'`, 0-d 결과, `squeeze(dim)`… | torch 표면의 구멍들 |

## 테스트가 놓치고 심판이 잡은 것 — KernelBook row 17

`GatSymAttention`: 모듈은 `leaky_relu(a1) + a2`, 컴파일된 `ModelNew`는 **`a1 + leaky_relu(a2)`**
(GPU에서 후보식 대조, 오차 2e-7). a1·a2는 두 입력을 맞바꾸면 서로 바뀌는 대칭 항이라
래퍼의 입력 순서 뒤바뀜과 정확히 일치한다. 데이터셋의 테스트가 못 본 이유: 파라미터가
`torch.Tensor(...)`로 **미초기화**라(값 ~1e33, inf) `allclose`가 공허하게 통과했다.
심판의 경로: 실수 위에서 불일치 → Volta는 max/min에 불완전하므로 **수치 증인**으로 확정
(무작위 점에서 spec=−0.364, kernel=−0.042) → 같은 점을 GPU에 넣어 재현(64개 중 40개,
최대 1.24) → 후보식으로 정확한 차이를 명명. 심판이 낸 FAIL은 이제 항상 증인을 동반한다.

## 코퍼스 위에서 — KernelBook 400행

Inductor가 생성한 Triton 18,162쌍 중 앞 400행 (`results/report.txt`).

| 판정 | 행 | |
|---|---:|---|
| PASS | 272 | 68.0% — AC 정규형 261, Volta 14, Volta+Z3 1 |
| PASS-ASSUMING | 4 | scatter의 인덱스 단사성처럼, 못 갚고 명시만 한 가정 위의 PASS |
| FAIL | 28 | 전부 수치 증인 첨부 |
| 스펙 에러 | 29 | torch 표면의 남은 구멍. `STensor`가 아직 못 받는 것들 |
| 스펙 미지원 | 21 | einsum 3, unfold 2, prelu 2, pad(replicate) 2, BCE 2, … |
| 커널 미지원 | 18 | float→int 6, 로드한 정수 비교 5, transposed conv 3, 데이터 의존 정수 산술 2, i1→float 2 |
| UNKNOWN | 17 | Volta 예산 6, 증명 불가·수치 동일 4, Volta 4GB 2, 하드웨어 침묵 2, … |
| TIMEOUT / TOO-LARGE | 7 | 넓은 MLP의 심볼릭 matmul, 큰 attention |
| NONDETERMINISTIC / ERROR | 3 | |

372행은 판정 중 멈춰서 기록이 없다 (`run_resume.sh`가 스텝오버). 그래서 399/400.

판정된 304행에서 **tolerance 테스트와 심판: 296 일치, 8 불일치.** 불일치는 전부 한 방향이다
— tolerance가 통과시킨 4행을 심판이 FAIL로 잡았고 (증인점에서 GPU가 재현),
tolerance가 떨어뜨린 24행은 심판도 FAIL이다. **반대 방향(테스트 실패·심판 PASS)은 0.**

세 의무: 정밀도 태그 275 exact / 1 tf32·f16, 전제조건에서 커널이 스펙보다 나쁜 행 0,
신뢰한 extern(cuBLAS/cuDNN) 호출 668건(223행). 심볼릭 실행 중앙값 0.17s, p90 1.44s.

AC 정규형이 261/276을 끝내는 것은 위임(`tvj/decide/delegate.py`) 덕이다 — 양변이 같은
라이브러리 호출에 같은 인자를 넘기면 같은 기호가 되고, hash-consing이 공짜로 판정한다.
위임 전에는 그 자리가 O(M·N·K) 항 전개였다.

테스트가 놓치고 심판이 잡은 것 둘:
- **row 17 GatSymAttention** — 위 절. 입력 순서 뒤바뀜, 데이터셋 테스트는 미초기화 파라미터로 공허.
- **row 308 Critic** — 귀속 완료 (`tvj/measure/kb_critic.py`). 래퍼가 텐서를 **같은 shape의 다른
  역할로** 넘긴다: `call()`은 `primals_1,2`를 cat 입력, `primals_5`를 linear2 가중치로
  받는데 래퍼는 `linear2.weight, state, …, action`을 넣는다. 전부 (4,4)라
  `assert_size_stride`가 통과한다. 실제 계산은 `L3(relu(relu(cat([W2,state])·W1ᵀ+b1)·actionᵀ+b2))`
  — GPU에서 오차 **정확히 0**으로 확인. 테스트가 놓친 이유는 입력 부호가 아니라
  **스케일**이다: `linear3`가 `U(-0.003, 0.003)`으로 초기화돼 출력이 ~1e-4인데
  `allclose`의 `atol=1e-3`이 **상대오차 190%를 통째로 삼킨다**.

그리고 심판 자신의 오류 6건(GPU가 일치하는데 FAIL)을 같은 하드웨어 대조로 찾아 고쳤다:
`F.linear(x, w, bias=…)`의 키워드 인자를 핸들러가 몰라서 **조용히 버린** 것(이제 무해
목록 밖의 kwarg는 에러), 그리고 Inductor가 leaky_relu 마스크를 int8 버퍼로 왕복시켜
`mask != 0`으로 되살릴 때 술어 객체가 파이썬 비교로 항상 참이 된 것. **FAIL은 반드시
GPU에서 재현해 본 뒤에 믿는다** — 이게 이 코퍼스에서 배운 운영 규칙이다.

## 견적 — 무엇을 할 수 있나

| 대상 | 결과 | 시간 |
|---|---|---|
| matmul 계열 (tiled / split-K / swizzle / masked) | 스펙 대비 PASS, 버그 6종 전부 FAIL, 10/10 | 0.1–1.6s |
| 같은 것을 계약 기반 shape 전체에서 | 9/9, `bug_swizzle`을 (48,48,48)에서 자동 포착 | ~4분 |
| softmax naive vs max-subtracted | AC 0/128 → **Volta 128/128** | 0.003s |
| attention ref / safe / flash, D=16, 세 쌍 pairwise | L=32: 512 · L=64: 1024 · L=128: 2048 · L=256: 4096 (두 쌍) — 전부 true | 아래 표 |
| flash에서 rescale 빠뜨린 버그 | **512/512 false** | 1.8s |
| matmul 256³ | 17.4M 노드, Θ(M·N·K), SMT 미사용 | scale.py |
| ieee → tf32 교체 | 실수로는 동치, **정밀도 의무 FAIL** | |
| 전제조건 | naive softmax \|x\| ≤ 86.64 · safe softmax 무제한 · naive attention \|q\|,\|k\| ≤ 2.30 · safe/flash attention 3.26e18 (내적 자체의 한계) | |

attention 스케일 (D=16, BM=BN=16, ref-vs-flash가 가장 비싼 쌍):

| L | 키 블록 | 출력 | 심볼릭 실행 (커널당) | Volta ref-vs-flash | term ops |
|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 512 | 0.3–1.8s | 0.3s | 3.2M |
| 64 | 4 | 1024 | 1.0–2.0s | 3.6s | 26.4M |
| 128 | 8 | 2048 | 4.4–5.5s | 33.6s | 206.8M |
| 256 | 16 | 4096 | 18–30s | (아래) | ref-safe 349M · safe-flash 509M · ref-flash 미실행 |

term ops는 L에 대해 ~8배/2배 — 출력 수 ×2, 출력당 항 수 ×2, 교차곱 ×2. Volta 논문이
FlashAttention 정규화를 ~10⁸ term ops로 보고하는데 L=128이 그 두 배다.

**메모리가 시간보다 먼저 한계다.** 브릿지 피크 RSS (L=128, 단독 실행):

| 쌍 | term ops | interned | 브릿지 피크 |
|---|---:|---:|---:|
| ref vs safe | 52M | 0.84M | 0.56 GB |
| safe vs flash | 68M | 1.55M | 0.77 GB |
| ref vs flash | 207M | 3.65M | **9.19 GB** |

L=256 (단독, 두 쌍만):

| 쌍 | term ops | interned | Volta | 브릿지 피크 | python 피크 |
|---|---:|---:|---:|---:|---:|
| ref vs safe | 349M | 3.35M | 14.8s | 2.32 GB | 5.01 GB |
| safe vs flash | 509M | 10.66M | 38.4s | 4.48 GB | **7.62 GB** |

형태가 맞는 쌍에서는 L=256에서도 Volta가 4.5GB에 그치고, 오히려 **우리 Python 쪽
(DAG 객체 + JSON 직렬화)이 7.6GB로 더 크다.** 다음 병목은 브릿지 직렬화다.

같은 L에서 ref-vs-flash만 12배다. ref는 max를 빼지 않아 지수 다항식이 `−m` 원자를
공유하지 못하고 교차곱이 그대로 부푼다. 실무적 결론: **참조 커널도 max-subtracted로
써라** — 판정 절차의 비용은 두 커널의 *형태 차이*에 지배된다. L=256 ref-vs-flash는 이
추세로 30GB+ 라 15GB VM에서는 OOM이다 (실제로 그렇게 죽었다; 동시에 돌던 scale.py 256³
~6GB와 합쳐서). 큰 검사는 한 번에 하나만. 심볼릭 실행은
처음엔 키 해싱이 DAG 크기에 비례해 L=64에서 180s였고, uid 키로 바꾼 뒤 1.4s
(`tvj/core/terms_v1_structural_keys.py`에 이전 버전을 남겨뒀다).

## 이 구조가 잡아낸 것

| | 어디서 |
|---|---|
| `reduce.order`를 left fold로 적은 게 틀림 → tree | GPU 미분 테스트 |
| "스레드 안은 순차" 가정도 이 구성에선 아님 → 모든 층이 pairwise | 정수 마진 프로브 (`tvj/measure/layout2.py`) |
| i32 인덱스 산술이 정말 wrap → 큰 stride에서 음수 오프셋 | GPU 미분 테스트 |
| `tl.dot` 기본값이 tf32 → 정밀도 안 적은 f32 matmul은 ieee 참조의 refinement가 아님 | 정밀도 격자 |
| `mm_splitk`의 적히지 않은 전제조건 `(K/SPLIT) % BK == 0` | shape 스케줄러 |
| 잔여 클래스 커버리지만으론 부족 → joint key 필요 | shape 스케줄러 |
| `x − max(S)` 규칙이 AC 평탄화에 안 잡힘; flash 분모는 rescale 인자를 합 위로 분배한 뒤에야 보임 | 전제조건 층 |
| 항 키가 구조적이라 해싱이 DAG 크기에 비례 → 60–130배 손실 | 프로파일 |

## 한계 (측정된 것)

1. **op 커버리지 12/26.** `scf.if`/`while` 없음, atomic은 `fadd`만, block pointer(`make_tensor_ptr`) 없음.
2. **bounded.** shape당 증명이고 계약을 사람이 적는다. 데이터 의존 제어 흐름 없음.
3. **정밀도 추적이 항 identity에 키잉**돼 있어 같은 정규형을 다른 정밀도로 만들면 min을 취한다.
4. **전제조건 층은 interval이라 관계를 못 본다.** 규칙 셋은 softmax 계열 전용. underflow는 표시만 한다 — fp32에서 0으로 올바르게 반올림되는 결과라 절대오차론 무해, 상대오차론 100%. 어느 쪽이 스펙인지는 결정 사항.
5. **Volta는 판정만 빌린다.** `ExprArena`에 우리 항을 넣고 `check_equivalent`만 호출. Volta의 레이스 검사·PTX 프론트엔드·완전성 증명의 전제(structured-CTA)는 우리 쪽 실행기에 그대로 옮겨오지 않았다 — 우리 실행기의 soundness는 비형식적이다.
6. **measured 항목은 전부 sm_75.** tf32·bf16·cp.async 경로는 재보지 않았다.

## 실행 메모 (WSL2)

이 머신은 호스트 32GB, WSL VM 16GB(기본 50%) + swap 4GB, `.wslconfig` 없음. 한도를 넘는
건 시간이 아니라 메모리고, 넘으면 OOM killer가 가장 큰 프로세스를 죽인 뒤 세션이 통째로
날아간다. 큰 검사는 한 번에 하나, 백그라운드는 `setsid nohup`(그룹 kill에 살아남게).
늘리려면 `%USERPROFILE%\.wslconfig`에 `[wsl2] memory=20GB swap=8GB` 후 `wsl --shutdown`.
그래도 ref-vs-flash L=256(30GB+)은 안 들어간다 — 참조를 max-subtracted로 쓰는 게 답이다.

## 검수

모든 숫자는 `verify.py`가 스크립트를 다시 돌려 정규식으로 assert한다. 마지막 실행: **31/31** (fast set, `results/verify.txt`). GPU 주장은
`[sm_75]` 태그가 붙고, 다른 아키텍처가 다르게 답하면 그건 방법의 실패가 아니라 정보다.

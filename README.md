# A symbolic judge for Triton kernels

Compares a Triton kernel against its PyTorch reference **over symbolic inputs**
instead of sampled ones, so a defect that hides outside the test distribution
cannot hide from it. Built to ask whether LLM-generated GPU kernels that pass a
tolerance test are actually correct.

```
./setup.sh          # fetch Volta, KernelBench, the two corpora; build the bridge
python3 verify.py   # re-runs every experiment behind every claim below (22/22, ~5 min)
```

## What it found

**Tolerance testing goes vacuous in at least three distinct ways**, each observed
in a real corpus, none of them visible to the benchmark that was running:

| mechanism | evidence |
|---|---|
| parameters left uninitialised — garbage compared against garbage | KernelBook row 17 |
| output ~1e-4 under `atol=1e-3`, hiding a **191 % relative error** | KernelBook row 308 |
| parameters default to the **identity element** of the op they feed (`bias=0`, `scale=1`, `tau=0`), so a kernel that ignores them is bit-identical | 5 LLM-generated kernels; and KernelBench's own level2/85 |

The last one survives KernelBench-Verified's hardening. Its hidden tests vary the
inputs four ways (as-is, ×3, ×0.01, negated) but build the model once, so a kernel
with the scale multiply **deleted** passes all four at max diff exactly `0`
(`kbv_blindspot.py`). Drawing the parameter at random finds it at once.

**A counterexample yields an axis, not just a point.** The judge reports which
named buffers a disagreement rests on, so `testgen.py` turns one exploit into a
harness directive — *vary these parameters*, *poison this buffer*. Derived from a
single kernel, it catches 3 of the 4 others it had never seen.

**Corpus results.** Of the kernels these corpora label correct, the judge rejects
6 — every one reproduced on the GPU before being counted:

| corpus | judged | judge rejects a "correct" kernel |
|---|---|---|
| 400 Inductor-generated (KernelBook) | 60 % | 2 — both defects in the dataset's own wrapper, not in Inductor's codegen |
| 156 LLM-generated Triton | 58 % | 4 |

**Honest negatives are in `verify.py` too.** Catastrophic cancellation
(`E[X²]−E[X]²`) is *not* caught: the two forms are equal over the reals, and the
precondition failure we first reported for it was a false positive we retracted.

## How

Three obligations, not one. Value equality over the reals (AC normal form →
[Volta](https://github.com/willtunnels/volta)'s decision procedure → Z3 case
splitting for piecewise terms — the step Volta's paper says "could be handled by
case splits" and declines to take). Precision as a directed lattice, because
`ieee → tf32` is real-equal but not a refinement. And a float-validity
precondition, because a real-number proof says nothing where an intermediate
overflows.

`PIPELINE.md` has the data flow. `semantics.py` is the artifact underneath it all:
19 decisions the interpreter had to make because Triton does not answer them —
what a masked lane loads, whether i32 index arithmetic wraps — each with its basis
and its evidence, four of them measured against hardware.

## Limits

Judged coverage is ~60 %. Shapes are fixed and small; there is no data-dependent
control flow; a FAIL on the value obligation is only believed if the GPU
reproduces it at the witness point. All hardware measurements are `sm_75`.

---

*(아래는 한국어 상세 설명입니다.)*

| 스크립트 | 무엇을 하나 |
|---|---|
| `semantics.py` | **스펙 산출물.** 코어 26 op에 대한 14개 의미론 결정, 근거 등급·증거 포함 |
| `check.py` | 커널 vs 스펙 / 커널 vs 커널 refinement. 값 의무 + 정밀도 의무 |
| `suite.py` `shapes.py` | 커널 계약이 허용하는 잔여 클래스를 최소 비용으로 덮는 shape 선택 |
| `volta_bridge.py` `bridge/` | 항 DAG → `volta_analysis::canon::Session::check_equivalent` |
| `ranges.py` `precond2.py` | 실수 증명이 float32에 대해 의미를 갖는 입력 범위 (전제조건 층) |
| `difftest*.py` `ieee_gap.py` `layout2.py` `tf32.py` | 참조 의미론 vs 실제 GPU |
| `volta_check.py` `volta_attn.py` `volta_neg.py` | Volta 브릿지 검증: 항등식 / attention / 틀린 attention |
| `spec.py` `spec_test.py` | **스펙 프론트엔드**: torch식 참조 코드(`nn.Module.forward`)를 그대로 항 DAG로 |
| `capture.py` `capture_test.py` | **실행 가로채기 심판**: 생성 코드의 실제 GPU 런치를 후킹, 스토리지 identity로 역할 매핑 |
| `kernelbook_run.py` `kb_debug.py` | KernelBook(PyTorch↔Inductor Triton, 18K쌍) 위에서 세 의무 실행 |

## 구조

```
DECISIONS 레지스트리  ─ 산출물
   ↑ 실행 가능하게
인터프리터 (TTIR 12/26 코어 op + arith/math/scf.for)
   ↑ 검증 가능하게
RealDomain ──→ 값 의무:  AC 정규형 (합-곱)  →  못 가르면 Volta (exp, 나눗셈, 분배)
           ──→ 정밀도 의무:  exact > ieee > tf32x3 > tf32,  opt ≥ ref
           ──→ 전제조건:  ranges.py (overflow / div-by-zero 치명, underflow 표시)
ConcreteDomain ─→ float32 실행 → GPU와 미분 테스트
```

**스펙의 모양은 함수가 아니라 관계다.** TTIR은 타겟과 레이아웃 *위*에 있어서
`tt.dot`·`tt.reduce`의 값이 TTIR만으로 정해지지 않는다. 그래서 스펙은
(전제조건) × (실수 denotation: 함수) × (정밀도 계약: 방향 있는 관계)이고, 검사기도
그 셋을 따로 검사한다. 순서 비결정성(atomic, reduce, dot 누산)은 관계 안에 흡수된다.

## 심판이 되기까지 — 스펙은 어디서 오나

`spec.py`의 `STensor`는 우리 항을 원소로 갖는 numpy object 배열에 torch 표면을 입힌
것이다. `nn.Module.forward`에 파라미터를 `STensor.input`으로 바꿔치기해서 넣으면 참조
코드가 **수정 없이** 스펙 항을 낸다 (`__torch_function__`으로 `torch.matmul`,
`F.softmax`, `F.layer_norm`, `F.linear`… 을 받는다). 커널 쪽은 `capture.py`가
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

Inductor가 생성한 Triton 18,162쌍 중 앞 400행 (`results/kernelbook_report.txt`).

| 판정 | 행 | |
|---|---:|---|
| PASS | 193 | 48.2% — AC 정규형으로 143, Volta로 50 |
| FAIL | 21 | 전부 수치 증인 첨부 |
| 스펙 미지원 | 51 | adaptive_avg_pool2d, max_pool2d, 데이터 의존 비교, … |
| 커널 미지원 | 47 | 조각별 select(Huber/ELU/Mish/argmax) 29, float→int 6, transposed conv 3 |
| 스펙 에러 | 33 | torch 표면의 남은 구멍 |
| TIMEOUT | 27 | 256폭 MLP의 심볼릭 matmul(4M 항), 큰 attention |
| UNKNOWN | 22 | Volta 예산 6, tanh 4, 미기록 버퍼 의존, 증명 불가·수치 동일 3 |

판정된 214행에서 **tolerance 테스트와 심판: 213 일치, 1 불일치, 반대 방향(테스트 실패·심판 PASS) 0.**
세 의무: 정밀도 태그 213 exact / 1 f16, 전제조건에서 커널이 스펙보다 나쁜 행 0,
신뢰한 extern(cuBLAS/cuDNN) 호출 249건(109행). 심볼릭 실행 중앙값 0.06s, p90 0.7s.

테스트가 놓치고 심판이 잡은 것 둘:
- **row 17 GatSymAttention** — 위 절. 입력 순서 뒤바뀜, 데이터셋 테스트는 미초기화 파라미터로 공허.
- **row 308 Critic** — 귀속 완료 (`kb_critic.py`). 래퍼가 텐서를 **같은 shape의 다른
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
(`terms_v1_structural_keys.py`에 이전 버전을 남겨뒀다).

## 이 구조가 잡아낸 것

| | 어디서 |
|---|---|
| `reduce.order`를 left fold로 적은 게 틀림 → tree | GPU 미분 테스트 |
| "스레드 안은 순차" 가정도 이 구성에선 아님 → 모든 층이 pairwise | 정수 마진 프로브 (`layout2.py`) |
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
늘리려면 `C:\Users\naana\.wslconfig`에 `[wsl2] memory=20GB swap=8GB` 후 `wsl --shutdown`.
그래도 ref-vs-flash L=256(30GB+)은 안 들어간다 — 참조를 max-subtracted로 쓰는 게 답이다.

## 검수

모든 숫자는 `verify.py`가 스크립트를 다시 돌려 정규식으로 assert한다. 마지막 실행: **20/20** (fast set, `results/verify_final.txt`). GPU 주장은
`[sm_75]` 태그가 붙고, 다른 아키텍처가 다르게 답하면 그건 방법의 실패가 아니라 정보다.

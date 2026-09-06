# 파이프라인

한 문장: **참조 PyTorch 모듈과 생성된 Triton 커널을 각각 같은 항 대수로 내린 뒤, 네 가지
의무로 비교한다.** 값은 실수 위에서, 정밀도는 격자 위에서, 유효성은 구간으로, 그리고
shape는 계약이 요구하는 만큼.

```
        참조 PyTorch 모듈                     생성된 Triton 커널
              │                                      │
   symbolic_module()                          capture()  ← JITFunction.run 후킹
   파라미터/버퍼 → 심볼                        실제 GPU 런치를 기록
              │                                      │
   forward(STensor…)                          Launch: 시그니처·constexpr·grid·역할
   __torch_function__ 가로채기                        │
              │                                to_ttir() → ttir.parse()
              │                                       │
              │                                sexec.Interp  그리드 전체 심볼릭 실행
              ▼                                       ▼
        항 DAG (terms.py)  ←── 같은 정규형 ──→  항 DAG (terms.py)
                              │
              ┌───────────────┼───────────────┬────────────────┐
            값 의무        정밀도 의무      전제조건 의무    shape 커버리지
         AC 정규형          격자 비교        구간 분석        계약 기반
         → Volta           opt ≥ ref       radius 비교      잔여류 덮기
         → 수치 증인
```

## 1. 스펙 쪽 — PyTorch 모듈에서 무엇을 어떻게 뜯어오나

핵심은 **모듈 코드를 한 줄도 고치지 않는다**는 것이다. 두 개의 후킹으로 끝난다.

**(a) 파라미터를 심볼로 바꿔치기** — `spec.symbolic_module(model)`

```python
for mname, m in model.named_modules():
    for pn, p in m._parameters.items():
        m._parameters[pn] = STensor.input("p_" + mname + "." + pn, p.shape)
    for bn, b in m._buffers.items():
        m._buffers[bn] = STensor.input("b_" + ...)
```

`nn.Module`의 파라미터/버퍼를 `_parameters` / `_buffers` 딕셔너리에서 직접 갈아끼운다.
`STensor.input(name, shape)`은 `[T.sym(name, 0), T.sym(name, 1), …]`을 shape대로 접은
numpy object 배열이다. 즉 `linear1.weight[3]`이 `p_linear1.weight[3]`이라는 **심볼 하나**가
된다. 역할 이름이 그대로 심볼 이름이라 나중에 커널 쪽 버퍼와 맞춘다.

**(b) 연산자와 torch 함수 가로채기**

- `STensor.__add__`, `__mul__`, `__matmul__` … → numpy 브로드캐스트 + `Term.__add__`
- `torch.matmul(x, …)`, `F.softmax(…)` → `STensor.__torch_function__`이 받아
  `_TORCH` 디스패치 표로 보냄

그래서 `model.forward(STensor(...))`를 그냥 호출하면 참조 코드가 **수정 없이** 항 DAG를
낸다. 출력 텐서의 원소 하나하나가 항 하나다.

`_TORCH`에 없는 연산자는 `NotImplementedError` → 판정 `SPEC-UNSUPPORTED`. 값을 바꿀 수
있는 kwarg를 모르면 조용히 버리지 않고 에러를 낸다 (`F.linear(..., bias=)`를 삼켜서
false FAIL 6건을 냈던 사고 이후 규칙).

**스펙 쪽 결정 둘** (`semantics.py`에 기록):
- `spec.softmax-form`: softmax는 max-subtracted로 정의. 실수 동치이고, torch가 실제로
  계산하는 형태이고, 유효 반경이 무제한이고, Volta 비용이 ~10배 싸다.
- `literal.working-precision`: float 리터럴은 fp32 작업 정밀도로 읽는다. `1e-5`와 TTIR의
  `9.99999974e-06`은 같은 상수다.

## 2. 커널 쪽 — 생성된 코드에서 무엇을 뜯어오나

생성기에 요구하는 인터페이스는 **`launch(*inputs) -> output` 하나뿐**이다. 포맷도,
커널 이름도, 파라미터 순서도 안 본다.

**(a) 실제 런치를 가로챈다** — `capture.capture()`

```python
_calls, _orig = [], JITFunction.run
def _rec(self, *args, grid, warmup=False, **kwargs):
    _calls.append((self, grid, args, dict(kwargs)))
    return _orig(self, *args, grid=grid, warmup=warmup, **kwargs)
```

`JITFunction.run`을 갈아끼우고 코드를 **GPU에서 실제로 돌린다**. 그래서 한 번의 호출로
tolerance 테스트용 출력과 판정용 정보가 동시에 나온다. `extern_kernels.mm/addmm/bmm/
baddbmm/convolution`(Inductor가 matmul을 cuBLAS로 보내는 경로)도 같이 감싼다.

**(b) 텐서를 역할에 매핑 — 객체가 아니라 스토리지로**

```python
base_of(t)      = t.untyped_storage().data_ptr()     # 뷰가 아니라 실제 할당
elem_offset(t)  = (t.data_ptr() - base_of(t)) // t.element_size()
physical_offsets(t) = [off + Σ idx·stride]            # 비연속 stride 처리
```

Inductor는 `reinterpret_tensor`로 뷰를 만들고 출력 stride가 비연속일 수 있다. 그래서
`id(tensor)`가 아니라 **스토리지 base 포인터**로 역할(`in0`, `p_linear1.weight`, `out`)을
정하고, 논리 인덱스 → 물리 오프셋 변환을 따로 계산한다.

**(c) TTIR로 내려 심볼릭 실행**

`Launch`가 각 런치를 시그니처·constexpr·grid·버퍼 크기·인자로 분해하고 →
`to_ttir()`로 TTIR 텍스트를 뽑고 → `ttir.parse()`가 파싱하고 → `sexec.Interp`가
**그리드 전체**를 돈다. 여러 런치는 **하나의 공유 메모리**(`X.Grid`)에서 순서대로
실행돼 파이프라인(kernel → extern mm → kernel)이 이어진다.

정수는 구체값이다(shape 고정 ⇒ 주소·마스크·루프 경계가 전부 수). 그 부산물로 OOB,
write-conflict, 미기록 버퍼 의존이 **공짜로** 잡힌다.

## 3. 만나는 지점

양쪽 다 `terms.py`의 해시콘싱된 항을 낸다. 비교 단위는 **출력 원소 하나**:

```
스펙:   spec.flat()[i]
커널:   grid.store[("out", physical_offsets(out)[i])]
```

## 4. 네 의무

| 의무 | 무엇을 묻나 | 어떻게 |
|---|---|---|
| **값** | 실수 위에서 같은 식인가 | AC 정규형으로 판정 → 못 가르면 Volta `check_equivalent` → false면 **수치 증인** 필요 (없으면 UNKNOWN) |
| **정밀도** | 최적화 쪽이 참조보다 덜 정밀하지 않은가 | `exact > ieee > tf32x3 > tf32 > f16 > bf16` 격자, 항마다 태그 추적 |
| **전제조건** | 실수 증명이 float32에 대해 말하는 입력 범위 | `ranges.py` 구간 분석 + softmax 관계 규칙 3개, radius 비교 |
| **shape 커버리지** | 넘긴 shape 밖에서도 맞는가 | 커널 계약(전제조건 + 잔여류)을 적고 greedy set cover로 최소 비용 덮기 |

## 5. 신뢰 경계 (검증 안 하는 것)

- **Triton 백엔드**: TTIR까지만 본다. TTIR→PTX/GCN 컴파일은 신뢰. (Volta는 PTX라 이걸
  검증하지만 NVIDIA에 묶인다.)
- **extern 호출**: cuBLAS/cuDNN은 스펙 수준으로 모델링하고 **신뢰**로 표시. 판정 결과에
  신뢰한 호출 수를 같이 보고한다.
- **우리 인터프리터**: soundness 논증이 비형식적이다. Volta는 confluence를 Agda로 증명함.
- **측정 항목**: 전부 sm_75. tf32·bf16·cp.async 경로는 재보지 않았다.

# Expert 생성 품질 평가 — 2026-09-22 설계 수정

생성 품질의 다섯 조건과 별도로, 모델 상한을 확인하는 6번 bank oracle을 EXP58에서 추가 측정했다. 정답 참조 선택은 이 상한 진단에서만 허용한다. [EXP58의 정의·계산·결과](../20260922/exp58_oracle_upper_bound.md)를 따른다.

주 비교에서 정답 클래스 배정과 correctness oracle을 모두 제외한다. 정답은 채점에만 사용한다. 이전의 ‘배정에만 정답을 쓰면 expert 자체 역량을 평가할 수 있다’는 설계는 상수 클래스 expert를 구별하지 못하므로 폐기한다.

## 1. 생성 조건과 다섯 비교

- `1 <= K < C`: K는 global 제외 expert 수, C는 benign 포함 모델 클래스 수.
- `nfv3_v3_exp31_c0alloc.py`가 학습 전에 부적격 K 후보를 제외한다. 유효 후보가 없으면 실패한다.
- 현재 결과의 CIC K=8/C=7은 기존 bank 진단으로만 남는다. 새 조건에 맞춘 CIC bank 재학습 결과가 아니다. ToN K=7/C=10은 개수 조건을 충족한다.
- K<C만으로 정답 개입 문제가 해결되지는 않는다. 평가 입력에서도 정답을 제거한다.

| 조건 | 예측 규칙 | 목적 |
|---|---|---|
| 1. Global | `g(x)` | 기준선 |
| 2. 고정 영역 배정 | `r(x)=argmin_k distance(z(x),p0(x),centroid_k)` 후 `e_r(x)(x)` 그대로 | S/V 없이 고정 영역과 expert의 결합 성능 |
| 3. Expert 전체 test | 각 `e_k(x)`를 test 전체에 독립 적용 | 양성·음성 전체의 혼동행렬·확률 식별력 |
| 4. Scorer-only | 기존 S+V 후보·호출 조건 고정, V 제거 | 같은 호출 조건의 대조 |
| 5. Scorer+verifier | 기존 S1V0 정책 | 제안 모델 |

2번은 관측 가능한 고정 배정 baseline이며 oracle 또는 성능 상한이 아니다. Test 라벨로 담당을 정하거나 정오를 보고 global로 복구하지 않는다. 생성에 쓰인 training 라벨과 평가 입력에 test 정답을 제공하는 것은 구별한다.

## 2. 실행과 출처

수정 결과: `tabpfn/results/exp56_label_free_20260922/{cic2018,toniot}/`.

- 실행: `tabpfn/scripts/exp56_label_free_revision.py`.
- 향후 평가 기본 규칙: `tabpfn/scripts/exp56_five_condition_evaluation.py`.
- 2번 예측은 저장된 observable `eval_distance.npy`와 각 expert의 예측으로 계산하며, 함수에 y 인자가 없다.
- 추가 상수 대조: 각 영역의 학습용 route 표본에서 가장 많은 클래스만 출력. 배정과 상수는 test 라벨 로딩 전에 고정한다.
- 1·3·4·5의 점수는 기존 EXP56과 일치하고 S/S+V는 저장된 예측과 행 단위로 일치한다.
- AP·FPR 진단은 같은 cache의 기존 계산을 hash와 함께 재사용한다.
- 새로운 expert 학습이나 독립 final test는 실행하지 않았다.

| Macro-F1 | CIC2018 | ToN |
|---|---:|---:|
| Global | 0.810653 | 0.683734 |
| 고정 영역 배정 | 0.747722 | 0.661978 |
| Scorer-only | 0.810653 | 0.754919 |
| Scorer+verifier | 0.810653 | 0.755490 |

CIC EXP47·ToN EXP48의 모순 제거 고정 분할이다. Clean manifest hash와 cache identity, test 행 수가 일치한다. 서로 다른 모델 클래스 라벨을 가진 동일 벡터의 모든 행을 제거하고 동일 라벨 중복은 보존했으며 재분할하지 않았다. 두 manifest 모두 잔여 모순 벡터 그룹 0이다. 전체 데이터 라벨을 이용한 벤치마크 정리이며 실제 운영 전처리나 새로운 독립 holdout으로 주장하지 않는다.

## 3. 혼동되던 점수의 원인

### CIC global·S-only·S+V 동일

Calibration 개입 후보 65개 모두 `benign_fpr,tail_f1` 제약 위반. `global_fallback`, `tau_pre=null`, `tau_post=null`이 선택돼 호출이 0건이다. S-only는 같은 호출 조건을 유지하므로 V를 제거해도 예측이 그대로다. 세 예측은 반올림만 같은 것이 아니라 행 단위로 같다.

### CIC infiltration 정답 배정 F1 0.9140

| 규칙 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Global | 2,733 | 3,948 | 6,009 | 0.354406 |
| e6 전체 test | 7,556 | 806,387 | 1,186 | 0.018369 |
| 제외한 정답 클래스 배정 | 7,556 | 236 | 1,186 | 0.913995 |

정답을 이용해 양성만 e6에 보내고 음성을 다른 expert에 보내면 e6의 TP는 보존하면서 FP가 사라진다. 원래 e6의 식별력 개선으로 해석할 수 없다.

### ToN scanning S+V F1 0.8128

| 규칙 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Global | 452 | 50 | 5,577 | 0.138417 |
| e2 전체 test | 5,439 | 54,321 | 590 | 0.165347 |
| e4 전체 test (개별 최고 F1) | 4,240 | 7,715 | 1,789 | 0.471530 |
| S-only | 4,270 | 287 | 1,759 | 0.806726 |
| S+V | 4,245 | 172 | 1,784 | 0.812751 |

선택 정책은 각 expert를 전체 test에 적용하는 규칙과 다르다. S+V의 TP는 global 452 + expert 3,793, FP는 global 50 + expert 122다. 표본별 출처를 재계산했고 test 정답은 합성 예측 선택에 쓰이지 않는다. 합성 정책 F1은 개별 expert의 전면 적용 F1 최댓값보다 높을 수 있다.

## 4. 자료별 검증 목적

1. **전체 test 적용**: 현재 argmax 예측의 실제 양성·음성 구분. Recall과 함께 precision·FP·F1을 본다. Global을 대체할 전역 모델의 품질과 제한된 영역의 특화를 동일시하지 않는다.
2. **AP**: 양성을 음성보다 높은 점수에 두는 능력. Precision–recall 전반의 요약이며 특정 FPR을 고정하지 않는다. 상수 점수 AP는 양성 비율 수준이다.
3. **R@FPR≤0.1%**: 같은 오탐 예산에서의 recall. 현재 표는 경험적 test ROC 기술 지표다. 이 임계값을 실제 모델에 적용하지 않는다.
4. **Validation 고정 임계값의 test 전이**: 사전 운영점이 다음 데이터에서도 오탐 예산과 탐지율을 지키는지 검사한다. Test FPR이 서로 다르면 recall만으로 동일 오탐 조건의 우위를 주장하지 않는다.
5. **영역 비교**: 미리 정한 입력 영역과 expert의 적합성 검사. 영역 정확도가 높아도 한 클래스뿐이면 상수 expert로 충분할 수 있다. 학습 route 다수 클래스 상수 대조와 영역 내 등장 클래스 macro-F1을 추가했다. CIC 영역 3·5는 상수도 1.0이므로 그 점수를 분류 역량의 근거로 쓰지 않는다.

R@0.1%가 명시적으로 오탐 한도를 고정한다. AP는 식별력 진단의 다른 측면이다. 둘 모두 one-vs-rest 확률 점수 기반이고 원래 다중 클래스 argmax F1과 구분한다. 0.1%·1%는 진단 기준이지 합의된 배포 합격선이 아니다.

## 5. 필요한 근거 실험

- **K<C 재학습 + context 선택 대조**: 같은 K·anchor·전체 context 크기·클래스별 행 수에서 실패 패턴 선택 vs 무작위 선택. 같은 크기의 균형 context는 별도 대조. 클래스 prior와 예산 효과를 통제한 뒤 AP·저오탐 recall의 이득이 남아야 생성 방식의 효과를 지지한다.
- **Global 보정 대조**: expert 없이 global prior/β·온도·임계값만 보정. 같은 validation 절차로 선택해 단순 운영점 변경으로 이득이 설명되는지 검증한다.
- **고정 운영점의 새 시간 평가**: val에서 임계값 고정 후 이후 holdout에서 실제 FPR·FP 수·recall 보고. 목표를 지키며 G 대비 이득을 유지하는지 확인한다.
- **독립 반복·불확실성**: 최소 3개의 실제 context 추첨/학습 seed와 새 시간 holdout, 벡터·시간 그룹 단위 paired bootstrap. 동일 cache 재계산은 독립 반복이 아니다.
- **선택적 expert 제거 대조**: 해당 expert가 bank 시스템에 꼭 필요하다는 주장에만 추가. 제거 후 같은 validation 절차로 비교 정책을 재선택한다.

EXP52 행 선택/anchor, EXP53–54 context/prior 대조는 기존의 부분 근거다. 미실행으로 취급하지 않는다. 추가 조건은 K<C bank, 동일 클래스 구성·예산 통제, 같은 저오탐 지표, 독립 반복과 새 시간 평가다.

지금은 CIC infiltration의 G 대비 식별력 개선 근거가 없고, ToN scanning e2/ransomware e5에는 이 holdout의 순위 식별력 이득이 있다. Ransomware e5의 val FPR 0.1% 임계값은 test에서 0.364%가 되어 운영점 전이가 남는다. 합격 판정에는 목표 클래스·최소 recall·허용 FPR·global 성능 저하 한도를 사전에 정해야 한다.

## 6. 과거 자료 보존

- 정답 클래스 배정과 이전 correctness oracle 결과는 `tabpfn/results/exp56_five_condition_20260922_final/` 및 EXP55 결과에 보존한다. 현재 주 비교와 구별한다.
- 0911의 원본 데이터 참조값은 다른 test·bank이며, bank 정답 포함률과 데이터 다수 라벨 참조값을 구분한다. 성능 차이를 이번 모순 제거 평가의 개선량으로 해석하지 않는다.
- 최신 표와 HTML 공유 본문: `docs/research/20260922/exp56_five_condition_results.{md,html}`.

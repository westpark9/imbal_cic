# 미팅 후 계획 — CIC2018 모순 행 제거와 기존 구조 재검증

작성: 2026-09-15. 최초 작성 당시 계획 단계였으며, 이후 사용자 실행 지시로 모순 행 정제와 EXP47 seed 42 재검증을 수행해 **2026-09-15 19:23 KST에 완료했다.** 실행 범위는 [EXP47 실행 기록](exp47_clean_revalidation_run.md), 결과와 후속 우선순위는 [결과 보고서](../20260916/exp47_results.md)를 따른다.

## 1. 이번 결정과 순서

- 사용자 선택: **CIC2018부터 기존 구조 재검증**. ToN global context 비교(EXP46)는 중단됐으며 재개하지 않는다.
- 모순 행을 먼저 제거한 데이터 버전을 만들고, 그 데이터로 global → residual → expert bank → scorer → verifier → 최종 정책을 다시 구성한다.
- 0909의 fallback 원인 판정과 모순 제거 전 oracle 수치를 새 실험의 결론으로 사용하지 않는다. 기존 코드와 진단 항목은 활용하되, 근거가 되는 예측·학습 라벨·점수는 모두 새로 만든다.
- 첫 실행에서는 기존 context 구성 **규칙**과 모델 구조를 고정한다. 정제 데이터에서 C0와 expert context의 실제 행을 다시 선정한다. 그 결과로 병목을 찾은 다음 context 구성 규칙이나 학습 목표를 한 가지씩 바꾼다.
- 앞서 제안한 benign↔mitm pair expert, global 예측에 따른 호출 자격 제한, 새 verifier 구조는 이번 최초 재검증에 추가하지 않는다.

| 순서 | 작업 | 답할 질문 / 완료 조건 |
|---|---|---|
| 0 | 모순 그룹 전체 행 제외, 데이터 버전 확정 | 무엇을 얼마나 삭제했고 어떤 클래스·시간 구간이 남았는가? |
| 1 | 정제 데이터에서 기존 global·C0 재구성 | 현재 global이 약한 클래스와 오류 방향은 무엇인가? |
| 2 | residual 생성과 expert context 추적 | 그 오류가 residual에 반영되고 expert 학습 사례로 들어가는가? |
| 3 | 라우팅과 분리해 expert 자체 평가 | global이 틀린 행을 실제로 맞힐 expert가 있는가? |
| 4 | scorer·verifier·정책 단계별 분석 | 교정할 수 있는 행이 어느 단계에서 제외되는가? |
| 5 | 확인된 원인에 맞춘 단일 변경 | 교정 증가와 훼손 억제가 확인 구간에서도 유지되는가? |

이때 global 평가는 후속 단계의 **새 기준 예측 생성**이다. 중단한 ToN context 2×2 실험의 연장이 아니다.

## 2. 모순 행 삭제 사양

### 삭제 단위

현재 모델에 제공하는 46개 feature를 동일한 float32·비유한 값 처리 규칙으로 변환한 뒤, **동일 벡터에 두 개 이상의 서로 다른 라벨이 존재하면 그 벡터에 속한 모든 행을 제외**한다. 예를 들어 같은 벡터가 benign 100행, infiltration 10행이면 110행 모두 제외한다. 같은 벡터·같은 라벨만 있는 그룹의 중복은 유지한다.

- CIC2018의 기존 loader가 사용하는 전체 행에서 모순을 찾는다. 모순 탐지 key에 라벨·timestamp·scenario를 넣어 같은 입력의 상반 라벨을 숨기지 않는다.
- 원본을 보존하고 파생 버전과 `keep_idx`, `drop_idx`, 그룹별 라벨 수·행 수를 저장한다. 라벨을 다수결로 바꾸지 않는다.
- 대량 hash로 후보를 찾되 삭제 판정은 실제 입력 벡터의 일치로 확인한다. 데이터·feature 순서·전처리·인덱스의 fingerprint를 기록한다.
- **기존 outer train/validation/test 소속을 유지한 채 keep mask를 적용**하는 것을 기본안으로 한다. 삭제 후 60/20/20으로 다시 나누면 시간 경계도 달라지므로 첫 재검증에서는 그 효과를 섞지 않는다. 내부 역할 분할도 기존 규칙을 기록하고, 삭제 전후 시간 범위와 support를 확인한다.
- 전체 라벨로 정의한 **모순 제거 연구용 데이터셋**임을 명시한다. 정제 후 test는 기존 test의 부분집합이므로, 원본 test 성적과의 차이를 그대로 모델 개선량으로 해석하지 않는다.

### 기존 삭제 실험과 구분

- [EXP41](../../../scripts/exp41_dedup_dataset_xgb.py)의 `drop_conflict`는 같은 라벨의 중복도 줄이고, 삭제 후 split을 다시 만든다. 이번 사양과 다르다.
- [EXP44](../../../tabpfn/scripts/nfv3_v3_exp44_conflict_only.py)의 `drop_vectors`는 모순 그룹 전체 삭제·나머지 중복 유지라는 점에서 맞지만, 이후 split을 다시 만든다. 삭제 논리는 참고하되 실행 스크립트를 그대로 재사용하지 않는다.

### 다음 단계 진입 조건

1. 남은 데이터에서 동일 벡터·상반 라벨 그룹 수가 0이다.
2. 삭제 대상 밖의 행·라벨과 같은 라벨 중복 수가 보존되고, split 간 행 이동이 없다.
3. 클래스×split×scenario별 원래 행 수, 삭제 수·비율, 남은 수를 보고한다. 클래스 소실·극소 support가 생기면 숨기지 않고 별도 표시한다. 고정 라벨 목록과 macro-F1 계산 규칙을 유지한다.
4. C0, anchor, D_expert, D_tune, D_route, D_cal, test가 모두 같은 정제 버전을 참조한다. 기존 posterior·embedding·cluster·scorer/verifier 학습 자료·임계값 캐시는 새 버전으로 재생성한다.
5. class weight·prior·context 할당량은 정제된 train에서 다시 계산한다. 기존 할당 규칙이 부족한 클래스 support를 요구하면 실제 확보 수와 적용된 기존 cap 규칙을 기록한다.

산출물: 데이터 manifest, 삭제 인덱스, 그룹별 삭제 사유, 클래스·split별 정제 전후 표. 모델 실험은 이 검증 이후에 진행한다.

## 3. Global에서 residual이 만들어지는 과정

[기존 EXP31 구현](../../../tabpfn/scripts/nfv3_v3_exp31_c0alloc.py)의 `balanced_ce`, `ResidualSignature`, Phase 1을 기준으로 다음 흐름을 확인했다. 재실행 시 실제 채택하는 source·args를 저장해 후속 EXP38/39의 설정과 혼동하지 않는다.

```text
정제 train → C0 → global
                  ↓ D_expert의 클래스별 확률 p₀(x)
정답 라벨 y + 클래스 가중치 → weighted NLL → 상위값 clipping
                  ↓
입력 표현 z + 확률 p₀ + 오류 방향(onehot(y)−p₀) + log(1+residual)
                  ↓ 표준화·가중 군집화
군집 내 사례 선정 + shared anchor → 후보 expert bank → 선택·pruning
```

기본 계산은 다음과 같다. `N`, `n_c`는 정제 train의 전체·클래스 행 수이고 `C`는 고정 클래스 수다.

\[
w_c=\left(\frac{N}{C\max(n_c,1)}\right)^\gamma,\qquad
r_i=w_{y_i}[-\log p_0(y_i\mid x_i)],\qquad
\bar r_i=\min(r_i,Q_q(r)).
\]

여기서 p₀는 기존 temperature 선택·보정 규칙을 거친 확률이다. **Residual은 오분류 행만 모은 집합이 아니다.** 정답을 맞혔어도 확률이 낮거나 클래스 가중치가 크면 residual이 커진다.

### 확인할 항목

- 행별 `row_id, y, global_pred, p_true, NLL, class_weight, residual, clipped, cluster_id, context_selected`를 연결한다.
- 클래스·오류 방향별로 global 오분류 수 → residual 질량 → 군집 할당 → expert context 포함 수를 집계한다. 행 수와 고유 벡터 수를 함께 본다.
- 특정 클래스 가중치·반복 벡터·clipping이 군집과 context 예산을 과도하게 차지하는지 확인한다.
- **후보 bank와 pruning 후 bank를 모두 기록**한다. 도움이 되는 expert가 coverage·support·marginal gain 등의 조건으로 제거됐는지도 구분한다.
- 학습 군집에는 정답에서 계산한 오류 방향·residual이 들어가지만, 추론 시 관측 가능한 군집 표현은 `z, p₀`다. 정답을 알고 만든 군집을 추론 입력만으로 찾아갈 수 있는지 실제 후보 회수율로 확인한다. 모순 삭제만으로 이 문제가 해결된다고 가정하지 않는다.

## 4. Expert 자체 성능: 선택되기 전부터 평가

Global과 모든 expert를 **동일한 확인용 행**에 적용한 예측을 저장한다. 학습·선택용 구간 성적과 별도 확인 구간 성적을 나눠 보고, scorer가 선택한 행에만 한정하지 않는다.

| 비교 | 의미 |
|---|---|
| Global 오답·expert 정답 `H` | 그 expert가 제공하는 실제 교정 기회 |
| Global 정답·expert 오답 `D` | 잘못 라우팅하면 발생하는 훼손 |
| 둘 다 정답 | 승인돼도 분류 결과 개선은 없음 |
| 둘 다 오답 | 클래스 변경에 따라 FP/FN 구성이 달라질 수 있음 |

- 전체 클래스의 global·expert별 P/R/F1, support, confusion matrix와 `H, D`를 보고한다. `H−D`는 정답 행 수 변화이며 macro-F1 변화와 같지 않다.
- 모든 expert 중 하나라도 맞히는 global 오분류 행을 중복 없이 세어 **bank 교정 가능량**을 구한다. 라벨을 보고 고르는 oracle은 진단용 잠재량이며, scorer가 달성할 성능 보장이 아니다.
- 이 교정 가능량을 pruning 전후, 클래스별·시간 구간별로 비교한다. Global이 약한 클래스는 정제 후 결과에서 다시 확인한다.
- Expert가 전 클래스 평균에서 global보다 높아야만 유효한 것은 아니다. **특정 영역에서 추가 정답이 있고, 관측 정보로 그 영역을 선택했을 때 훼손을 억제할 수 있는지**가 기준이다.

Bank 자체에 교정 기회가 적으면 residual·context 선정 단계로 돌아간다. 교정 기회가 충분한데 최종 교정이 적으면 다음 라우팅 진단으로 진행한다.

## 5. Scorer / verifier / 최종 정책을 구분하는 분석

### 5.1 학습 라벨부터 확인

기존 EXP31에서는 route 행의 expert 이득을 `G = global weighted NLL − expert weighted NLL`로 계산한다. 이어서 교차 검증으로 얻은 verifier의 양수 판단 `b_oof`와 비용을 사용한다.

```text
scorer 학습 대상 U = b_oof × G − λ × expert_cost
scorer 양성 라벨 = (U > 0)
```

따라서 scorer가 선택하지 않는 원인이 **scorer 학습 이전 verifier가 유망 사례를 양성 라벨에서 제외한 것**일 수도 있다. 다음을 클래스별로 분해한다.

1. Expert가 실제 라벨을 교정하는 행, `G > 0`인 행, 둘의 교집합.
2. 위 행들이 OOF verifier와 비용 항을 각각 통과하는 비율, 최종 scorer 양성 수·가중치.
3. Scorer 학습의 양성/음성·전문가별 support와 학습/확인 구간 분포 차이.
4. Scorer teacher와 최종 verifier의 target 단위. 현재 `normgain` 구현은 최종 verifier target을 클래스 가중치로 나누지만 OOF teacher는 raw weighted gain을 사용한다. 이 차이가 새 데이터에서 유용한 양성 사례를 줄이는지 측정한 뒤 수정 여부를 판단한다.

NLL 개선과 분류 정답 교정은 별도로 기록한다. “양성 라벨이 있다”만으로 약한 클래스의 분류 개선을 학습한다고 보지 않는다.

### 5.2 교정 기회가 사라지는 단계

각 클래스의 global 오분류 행을 기준으로 아래 수를 연결한다. 비율에는 분모를 함께 표기하고, 정답에서 오답으로 바뀌는 경로도 같은 방식으로 집계한다.

| 단계 | 측정값 | 이 단계에서 줄어들 때 우선 점검 |
|---|---|---|
| Bank | 하나 이상의 expert가 맞히는 행 | Residual·context·expert 학습, pruning |
| Scorer 후보 | 선택된 top-1 expert가 맞히는 행; top-k도 진단 | Scorer target·관측 feature·순위 성능 |
| 호출 | Pre threshold를 통과한 교정 가능 행 | 후보 순위와 별도로 호출 임계·비용·호출률 제약 |
| 승인 | 선택된 expert의 calibrated lower bound가 post threshold를 통과 | Verifier 예측, 보정 offset, 승인 임계 |
| 최종 채택 | 선택된 정책에서 실제 교정/훼손/동일 라벨 행 | 정책 후보의 제약 탈락·확인 구간 성능 |

Verifier 원시 예측 `q_hat`, 보정량 `q_corr`, 최종 점수 `q_hat−q_corr`를 별도로 저장한다. 원시 점수와 보정 점수의 통과 여부를 교차표로 비교해 **모델의 판별 문제**와 **보정 단계의 영향**을 구분한다. 원시 점수 0 통과는 별도의 진단이며, 실제 코드에 없는 추가 gate로 세지 않는다.

또한 **정책 전체가 global fallback을 선택한 경우**와 **정책은 선택됐지만 개별 행에서 expert를 거절한 경우**를 분리한다. 전자는 최종 호출률 0만 보고 scorer나 verifier의 행별 성능으로 해석하면 안 된다.

- 원래 EXP31의 선택은 weighted gain과 proposal·harm·benign FPR·최소 support 제약을 사용한다.
- EXP39의 정책 선택은 Δmacro-F1, tail/protected class, benign FPR, proposal·support와 명시적 global 후보를 사용한다. 둘을 같은 선택 규칙처럼 합쳐 설명하지 않는다.
- 재검증할 기존 설정의 선택 규칙을 사전에 고정하고, **모든 임계값 후보의 탈락 사유와 global 후보보다 이득이 있었는지**를 저장한다. 구간 중복 제거 mask로 support가 부족해진 경우도 따로 기록한다.

### 5.3 같은 예측 캐시를 이용한 최소 진단 대조

1. Global 단독.
2. 모든 expert의 교정 가능량 및 pruning 전후 변화.
3. Scorer가 고른 expert를 gate 없이 적용: 후보 선택 능력과 훼손량 확인.
4. Scorer와 호출 gate만 적용: verifier 승인 단계의 추가 효과 확인.
5. 전체 scorer·verifier·정책 적용.
6. 필요 시 모든 expert의 verifier 점수를 계산해, scorer가 놓친 후보 중 verifier가 구분할 수 있는 후보가 있는지 확인.

Gate를 생략한 결과는 **원인 분리를 위한 진단**이다. 활성화율만 올리는 것을 목표로 삼지 않는다. 동일 구간의 macro-F1·클래스별 변화·benign FPR·교정/훼손을 함께 보고, 선택용 구간에서 정한 변경만 시간상 뒤의 확인 구간에 적용한다. Test로 임계값이나 개선안을 선택하지 않는다.

## 6. 원인별 후속 개선과 실행 단위

| 새 데이터에서 확인된 원인 | 다음에 바꿀 한 가지 |
|---|---|
| Global 오류가 residual 또는 context에 충분히 반영되지 않음 | 해당 클래스·오류 방향의 context 예산/선정 규칙 |
| 후보 expert는 유효하지만 pruning에서 소실 | 해당 pruning 판정 기준·평가 영역 |
| Expert가 global과 같은 오류를 반복 | 실제로 다른 예측을 만드는 지원 사례·대조 사례 구성 |
| 유효 expert는 있으나 scorer 양성에서 제외 | OOF verifier target 단위 또는 scorer label 생성 방식 |
| 양성 학습 사례는 있으나 후보 회수가 낮음 | Scorer의 관측 feature 또는 순위 학습 목표 |
| 호출 후보는 좋은데 verifier가 분류 교정과 무관한 점수를 냄 | Verifier target과 교정/훼손 구분 방식 |
| 원시 예측에는 정보가 있으나 보정 후 기회가 소실 | 보정 모집단·시간 전이·클래스 support에 맞는 보정 방식 |
| 정책 격자·최소 support에서만 소실 | 점수 분포에 맞는 후보 격자와 support 설정의 타당성 |

**첫 실행은 seed 42로 전체 경로의 추적과 병목 확인까지** 수행하는 안이다. 정제 기준·모델 구조·진단 구현을 고정한 뒤 seed 43·44로 같은 결론이 유지되는지 확인한다. 새 구조·context 변경은 이 기준 실행 이후 별도 실험으로 비교한다. 최초 실행부터 EXP46의 context 격자나 여러 새 모델을 동시에 돌리지 않는다.

필수 산출물은 `data_manifest`, `context_membership`, `residual_audit`, `expert_quality`, `scorer_target_audit`, `routing_stage_counts`, `policy_rejection_reasons`, `per_class_metrics`다. 최초 결과 보고서는 “global 선택만 됨”에서 끝내지 않고, **어느 단계에서 몇 건의 교정 기회가 사라졌고 어떤 단일 변경으로 검증할지**까지 제시한다.

## 7. 요청한 0911 XGBoost depth 더블체크 — 완료

**C0와 full train의 `max_depth`는 모두 8이었다.** 코드 기본값만 확인한 것이 아니라 실제 저장 인자와 로그를 생성자 연결까지 대조했다. 여기서 full context는 XGBoost가 해당 arm의 전체 train으로 학습한 조건을 뜻한다.

| 확인 범위 | 실제 실행 기록 | Depth |
|---|---:|---:|
| EXP40 full train / C0 | full 로그 1개 + C0 args 4개 | 모두 8 |
| EXP41 full train / C0 | full 로그 1개 + C0 args 12개 | 모두 8 |
| 현재 0911 보고서에 추가된 EXP42 | args 5개; full은 EXP41 예측 재사용 | 모두 8 |
| 추가된 EXP44 | args 2개; C0/full 동일 생성자 | 모두 8 |

공통으로 `n_estimators=300`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`도 일치했다. 이는 **설정된 최대 깊이** 확인이며, 모든 학습 트리의 실제 깊이가 8이라는 뜻은 아니다.

다만 EXP40/41 full train은 seed 42이고 EXP41 C0는 seed 42·43·44다. C0의 3-seed 평균과 full의 단일 seed는 분리해서 읽어야 한다. Full의 명시적 `tree_method='hist', n_jobs=32`와 C0의 tree_method 생략·`n_jobs=-1` 등도 달라, 모든 실행 조건이 완전히 동일했다고 표현하지 않는다. 향후 XGB 대조가 필요하면 같은 seed와 전체 하이퍼파라미터를 명시적으로 맞춘다.

근거와 재현:

- [25개 실행 설정 감사 CSV](0911_xgb_depth_audit.csv), [원본 경로·hash 포함 JSON](0911_xgb_depth_audit.json), [감사 스크립트](audit_0911_xgb_depth.py).
- Full: [EXP40 로그](../../../results/20260911_160256_3299866_nfv3_cic2018_exp40_dedup_xgb/experiment.log), [EXP41 로그](../../../results/20260911_165635_3341259_nfv3_cic2018_exp41_dedup_dataset_xgb/experiment.log).
- C0 대표: [EXP41 original seed 42 args](../../../tabpfn/results/20260911_171454_nfv3_cic2018_exp41_dedup_dataset_tabpfn_original_s42/args.json).
- 생성자: [EXP40 full](../../../scripts/exp40_dedup_xgb.py), [EXP41 full](../../../scripts/exp41_dedup_dataset_xgb.py), [EXP40 C0](../../../tabpfn/scripts/nfv3_v3_exp40_dedup_tabpfn.py), [EXP41 C0](../../../tabpfn/scripts/nfv3_v3_exp41_dedup_dataset_tabpfn.py), [EXP42](../../../tabpfn/scripts/nfv3_v3_exp42_dedup_methods.py), [EXP44](../../../tabpfn/scripts/nfv3_v3_exp44_conflict_only.py).

## 8. EXP46 중단 기록

Controller와 worker가 모두 종료된 상태를 확인하고, 남아 있던 `RUNNING.json`은 `RUNNING_at_stop.json`으로 보존했다. 루트와 미완료 arm에 `STOPPED.json`을 기록했다. 3개 arm은 이미 완료됐고 `full_legacy`는 미완료이므로 전체 실험 완료로 취급하지 않는다. 기존 부분 산출물은 보존하며 재개하거나 새 CIC2018 진단의 결론으로 사용하지 않는다.

- [중단 상태](../../../tabpfn/results/20260915_155115_nfv3_toniot_exp46_global_context_s42/STOPPED.json)
- 기존 검토안과 EXP46 프로토콜은 과거 제안·착수 기록으로 보존하며 **현재 실행 순서는 이 문서를 따른다.**

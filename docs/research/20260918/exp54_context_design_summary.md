# EXP54 — Expert context 구성 실험: 가설·실험·결과 (2026-09-18)

## 0. 질문과 배경

**질문.** context를 어떻게 구성해야 특정 클래스에 강한 expert가 되고, 그 expert가 scorer의 후보로도 쓸 만한가?

**배경 (09-18 오전까지 확인된 것).**
- EXP52: residual/confident/random 중 어떤 행을 고르든 expert의 자기-클래스 F1은 global보다 나쁘다.
- EXP53: injection expert context를 injection 90%로 채우면 F1 0.877 → 0.390 (정밀도 0.80 → 0.24). "순도"가 오히려 해친다.
- ToN 약한 클래스 3개: scanning(F1 0.138), mitm(0.067), ransomware(0.140). 전부 global 기준.

**공통 설정.** ToN 모순제거 고정 분할(`data/derived/toniot_conflict_free_fixed_split_20260916_021730`),
train 4,669,119 / test 2,271,723. context 20,000행(고정), TabPFN-v3 단독 fit(n_estimators 4, random_state 42),
test 전체 예측. 기준선 = EXP48의 global(100k C0, benign 75%) macro-F1 **0.6837**, 같은 행·같은 순서.
Run: `tabpfn/results/20260918_155043_nfv3_toniot_exp54_expert_context_design`(1라운드 11 arm),
`…/20260918_160238_nfv3_toniot_exp54r2_expert_context_design`(2라운드 8 arm).
스크립트: `tabpfn/scripts/exp54_expert_context_design.py`, `exp54r2_expert_context_design.py`.

### 용어

| 용어 | 뜻 |
|---|---|
| raw | TabPFN 출력 확률의 argmax 그대로 |
| β (PriorCorrector) | TabPFN은 context의 클래스 비율을 "실제 비율"로 믿고 확률을 낸다. β는 그 믿음을 train 실제 비율 쪽으로 되돌리는 세기: `log p' = log p + β·(log π_train − log π_context)`. β=0 보정 없음, β=1 완전 되돌림. 균등 context에서 mitm(train 0.07%)의 보정량은 log(0.0007/0.1) ≈ −5 → β=1이면 mitm으로 부르려면 ~150배 강한 증거가 필요. 파이프라인(EXP31)에 이미 있는 부품이며, 여기서는 재학습 없이 확률에 사후 적용했다. |
| H / D | expert가 global과 **다르게** 답한 행 중 H=교정(global 틀림·expert 맞음), D=훼손(global 맞음·expert 틀림). scorer 후보 품질의 대리 지표. |
| z-거리 | 46개 feature를 train 기준으로 표준화한 뒤의 유클리드 거리. 1 = 표준편차 1개. |
| share ∝ n^p | 클래스 c의 context 행 수를 train 행 수 n_c의 p제곱에 비례해 배분. p=1 자연 비율, p=0 균등, p=0.5 제곱근. |

---

## 1. 가설 H1 — "순도": target 비중을 올리면 그 클래스에 강해진다

**실험.** `enriched30:<target>` = target 30%(6,000행, 풀이 모자라면 전량), 나머지 9클래스 균등. target ∈ {mitm, ransomware, scanning, injection}.
비교군 `balanced` = 2,000행 × 10클래스.

| 구성 | target F1 (raw) | 같은 클래스 balanced raw | global |
|---|---|---|---|
| enriched30:mitm (풀 한계로 3,285=16%) | 0.028 (P .014 / R .994) | 0.031 | 0.067 |
| enriched30:ransomware (풀 한계로 2,151=10.7%) | 0.091 (P .048 / R .999) | 0.093 | 0.140 |
| enriched30:scanning | 0.184 (P .101 / R .984) | 0.386 | 0.138 |
| enriched30:injection | 0.846 (P .737 / R .992) | 0.868 | 0.932 |

**결과.** 4/4에서 balanced보다 나쁘거나 같다. 재현율은 0.99까지 올라가고 정밀도가 무너진다 — EXP53과 같은 기제(context 비율 = 사전확률 부풀리기).
**판정: 기각.** 순도로는 특화 expert가 만들어지지 않는다.

---

## 2. 가설 H2 — "경계": 혼동 상대(benign)를 많이 넣으면 target-vs-benign 경계를 배운다

09-18 confusion matrix에서 세 약한 클래스의 오류 상대는 모두 benign이었다.

**실험.**
- `contrastive:<target>` = target 2,000 + benign 14,000(무작위) + 나머지 500×8.
- `hardneg:<target>` = target 2,000 + **target에 가장 가까운 benign train 행** 8,000 + 무작위 benign 6,000 + 나머지 500×8. (2라운드)

| target | balanced raw | contrastive raw | hardneg raw | global |
|---|---|---|---|---|
| scanning | 0.386 (P .25/R .87) | 0.522 (P .40/R .75) | **0.578** (P .46/R .77) | 0.138 |
| mitm | 0.031 | 0.039 | 0.039 | 0.067 |
| ransomware | 0.093 | 0.141 | 0.131 | 0.140 |
| injection | 0.868 | 0.887 | – | 0.932 |

부수 관찰: scanning의 hard negative 8,000개는 scanning 행과 z-거리 **0.0002** 이내 — 사실상 같은 벡터에 다른 라벨(정확일치 삭제에서 살아남은 근사 모순).

**결과.** scanning만 정밀도/재현율 균형이 좋아진다(0.386 → 0.578). mitm·ransomware는 어떤 benign을 넣어도 정밀도 0.02~0.08 그대로.
**판정: scanning에 부분 채택, mitm/ransomware에는 기각.** mitm/ransomware의 FP는 context가 가르칠 수 있는 경계 문제가 아니다(→ H5).

---

## 3. 가설 H3 — "규칙": 클래스별 행 수를 손으로 정하지 않고 share ∝ n^p 한 줄로 정할 수 있고, 중간 p가 최선이다

(사용자 지적: "몇 천으로 경험적으로 조절할 순 없다"에 대한 대안)

**실험.** p ∈ {1(natural), 0.75, 0.5(sqrt), 0.25, 0(balanced)}. sqrt는 추첨 seed 42/43/44 3회, 40k 크기 1회 추가.

| p | context 예 (benign / scanning / mitm 행) | macro raw | scanning F1 | mitm F1 | ransomware F1 |
|---|---|---|---|---|---|
| 1 natural | 8,640 / 77 / 14 | 0.671 | 0.006 (R .003) | 0.049 | 0.117 |
| 0.75 | 7,050 / 204 / 57 | 0.685 | 0.182 | 0.038 | 0.127 |
| **0.5 sqrt** s42 / s43 / s44 | 5,369 / 506 / 217 | **0.730 / 0.748 / 0.738** | **0.613 / 0.796 / 0.730** | 0.055 / 0.043 / 0.043 | 0.126 / 0.125 / 0.086 |
| 0.5 sqrt, 40k | 10,738 / 1,011 / 433 | 0.731 | 0.610 | 0.043 | 0.136 |
| 0.25 | 3,635 / 1,115 / 730 | 0.718 | 0.643 | 0.034 | 0.099 |
| 0 balanced | 2,000 / 2,000 / 2,000 | 0.688 | 0.386 | 0.031 | 0.093 |
| global (100k, benign 75%) | 75,000 / 3,182 / 1,642 | 0.684 | 0.138 | 0.067 | 0.140 |

**결과.**
- p=0.5가 raw 최상이고 이득은 전부 scanning(0.138 → 0.61~0.80)에서 온다. 나머지 8클래스는 ±0.02.
- scanning은 context 내 scanning 비중이 재현율/정밀도 운영점을 직접 정한다: 77행 → R 0.003, 506행 → P .55/R .69, 2,000행 → P .25/R .87.
- 같은 규칙인데 **추첨에 따라 scanning이 0.61~0.80** 움직인다. 크기 2배(40k)는 효과 없음.
- mitm·ransomware는 p와 무관하게 0.03~0.06 / 0.09~0.14.
**판정: 채택(단, scanning 한정).** 규칙으로 정할 수 있고 p=0.5 근처가 최선. 그러나 mitm/ransomware는 이 축으로는 안 움직인다.

---

## 4. 가설 H4 — "운영점": 구성 간 차이의 본질은 context가 담은 지식이 아니라 사전확률이다 → 사후 β 보정이 구성 차이를 대신할 수 있다

**실험.** 모든 arm의 raw 확률에 파이프라인 PriorCorrector(ref = train 비율, α=1)를 β=0.5, 1.0으로 사후 적용. 기존 global(eval_p0)에도 적용.

| 구성 | raw | β=0.5 | β=1.0 |
|---|---|---|---|
| global (100k C0, benign 75%) | 0.684 | 0.704 | 0.699 |
| natural | 0.671 | 0.671 | 0.672 |
| sqrt s42 / s43 / s44 | 0.730 / 0.748 / 0.738 | 0.716 / 0.723 / 0.709 | 0.720 / 0.715 / 0.716 |
| pow0.25 | 0.718 | 0.753 | 0.731 |
| **balanced** | 0.688 | **0.771** | 0.765 |
| enriched30:ransomware (= ransomware 전량 2,151 + 1,556×9 ≈ 균등) | 0.667 | **0.813** | 0.749 |
| enriched30:{mitm, scanning, injection} | 0.678 / 0.668 / 0.699 | 0.755 / 0.756 / 0.762 | 0.662 / 0.694 / 0.743 |
| hardneg:{scanning, mitm, ransomware} | 0.719 / 0.727 / 0.727 | 0.745 / 0.741 / 0.719 | 0.702 / 0.725 / 0.699 |

클래스별로 보면 방향이 갈린다 (balanced 기준):

| 클래스 | raw | β=0.5 | β=1.0 |
|---|---|---|---|
| mitm | 0.031 (P .016/R .98) | 0.347 (P .22/R .84) | **0.705** (P .77/R .65) |
| ransomware | 0.093 | 0.200 | **0.413** (P .27/R .90) |
| scanning | 0.386 (P .25/R .87) | **0.643** (P .94/R .49) | **0.000** |
| benign | 0.897 | 0.932 | 0.934 |
| backdoor | 0.995 | 0.998 | 0.962 |

**결과.**
- 같은 균등 context가 β=0.5로 0.688 → 0.771. 전체 라우팅 시스템(EXP48, 0.754)보다 높다. expert·scorer·verifier 없이.
- **파이프라인은 이 보정을 스스로 껐다.** `1c_prior_grid.csv`: 균형-NLL 기준으로 β=0.0, T=1.25 선택. 균형-NLL은 균등 사전확률을 선호하므로 β=0이 항상 이긴다. 선택 기준이 macro-F1과 반대 방향.
- 기존 global에 β를 사후 적용하면 0.704가 한계: mitm은 오르지만(0.07 → 0.32) scanning이 죽는다(0.138 → 0.04). C0의 benign 75%가 scanning 운영점을 이미 보수적으로 잡아 놓아 β가 갈 방향이 없다.
- **필요한 β가 클래스마다 반대다.** mitm/ransomware는 β↑(FP 제거), scanning은 β=0(증거가 약해 조금만 눌러도 소멸). 단일 β는 절충이고, 클래스별 β는 val-튜닝 임계값과 같은 함정(SRC_HISTORY #2).
- `enriched30:ransomware + β0.5 = 0.813`은 mitm 0.728·scanning 0.658이 동시에 나온 단일 추첨. balanced β0.5의 mitm은 0.347 → 같은 β에서 추첨에 따라 mitm이 0.35~0.73으로 갈리는 칼날 위. seed 재현 필요.
**판정: 채택.** 구성보다 β가 큰 지렛대이며, 파이프라인의 β 선택 기준이 문제.

### scorer 후보 품질 (H4의 따름 결과)

| expert | raw H / D | β0.5 H / D | β1.0 H / D |
|---|---|---|---|
| balanced | 58,453 / 95,093 (**순손실**) | 90,090 / 33,488 | 89,501 / 26,826 |
| sqrt s42 | 60,698 / 41,242 | 80,093 / 22,774 | 86,945 / 20,106 |
| enriched30:ransomware | 52,742 / 113,329 (순손실) | 91,658 / 23,777 | 88,832 / 20,643 |

보정 없는 expert는 global과 다르게 답할 때 틀리는 쪽이 더 많다 → 09-18 "expert 역설"의 정량적 원인. mitm 영역만 보면 balanced β1.0은 H 28,145 / D 308.

---

## 5. 가설 H5 — mitm·ransomware의 정밀도 붕괴는 context 문제가 아니라 데이터(feature 공간 겹침) 문제다

**실험 (학습 없음, kNN).** `knn_separability.py`: train 풀(benign 30만 표본 + 4개 약한 클래스 전량)에서 각 약한 클래스 행의 leave-one-out 최근접 이웃 라벨.
`test_side_knn.py`: global이 틀린 test 행들의 train 풀 최근접 이웃 라벨과 z-거리.

| 대상 | 최근접 이웃 라벨 | z-거리 (10/50/90%) | 해석 |
|---|---|---|---|
| train mitm 행 (LOO) | mitm 96.5%, benign 1.8% | 중앙 0.086 | mitm끼리 뭉쳐 있음 |
| **test benign → mitm 오인 28,804행** | mitm 90%, benign 10% | **8.8 / 1,954 / 8,819** | train 어떤 행과도 멀다 |
| test 진짜 mitm 1,161행 | mitm 91% | 0.09 / 0.68 / 15.6 | 정상 범위 |
| train 행끼리 LOO 거리 (참조) | – | 99%: 0.89, 99.9%: 4.5, 99.99%: 21 | |
| **test benign → ransomware 오인 7,923행** | ransomware 59%, benign 40% | 0.02 / **0.077** / 2.1 | ransomware train 행과 거의 같은 벡터 |
| train ransomware 행 (LOO) | ransomware 98.4% | 중앙 0.011 | |
| **test 진짜 scanning 6,029행** | **benign 64%**, scanning 36% | 0.01 / 0.042 / 0.51 | benign 밀집 영역 안 |
| train scanning 행 (LOO) | scanning 74.7%, benign 24.2% | | train에서도 1/4이 benign 옆 |
| train injection 행 (LOO) | injection 99.7% | | 분리됨 |

mitm 오인 행의 정체(z-거리 > 50인 21,287행): `NUM_PKTS_UP_TO_128_BYTES` 중앙값 **130,872**(train benign 중앙값 2), PROTOCOL 17(UDP), `DST_TO_SRC_AVG_THROUGHPUT` 3.2 MB/s(train benign 4,188), MIN_TTL 128. 소형 UDP 패킷이 flow당 13만 개인 흐름이 test에서 benign으로 라벨돼 있다.

**결과.** 세 클래스의 원인이 서로 다르다.
- mitm: FP는 train 표본에 없는 형태의 benign 블록. context 구성으로는 못 배운다(train에 없는 것을 넣을 수 없다).
- ransomware: FP는 ransomware train 행과 z 0.08 이내의 benign — 정확일치 삭제 후 남은 근사 모순. 정밀도 상한이 낮다.
- scanning: test 행 64%가 benign 밀집 영역 안 → 어느 쪽으로 부를지는 순수 운영점 문제. 그래서 H3(context 비중)로 움직였다.
**판정: 채택.**
**한계.** "train에 없다"는 benign 30만 **표본**(전체 201만의 15%) 기준. train benign 전체에 0건인지는 직접 세지 않았다(사용자 결정으로 보류). 시간축 확인(timestamps)도 하지 않았다.

---

## 6. 가설 H6 — "거리 게이트": train 기하만으로 정한 τ로 mitm FP를 라벨 없이 걸러낼 수 있다

**실험 (`ood_gate_posthoc.py`, 학습 없음).** 기존 global 예측 중 non-benign 예측 행의 train 풀 최근접 z-거리를 재고, τ보다 멀면 benign으로 되돌림. τ 후보 = train 행끼리 LOO 거리 분위(라벨 불사용).

| τ | 되돌린 행 | 그중 실제 benign | mitm F1 (P/R) | benign F1 | macro | 부작용 |
|---|---|---|---|---|---|---|
| 없음 | – | – | 0.067 (.03/.91) | 0.912 | 0.684 | |
| p99.99 = 21.2 | 24,436 | 99.7% | 0.243 (.14/.87) | 0.921 | 0.703 | 없음 |
| **p99.9 = 4.5** | 28,352 | 98.7% | **0.335** (.22/.72) | 0.923 | **0.712** | 없음 |
| p99 = 0.89 | 62,989 | 88% | 0.482 (.41/.59) | 0.931 | 0.731 | scanning .138→.093, ddos .969→.959 |

**결과.** τ=4.5에서 mitm +0.27, benign +0.011, 나머지 8클래스 변화 없음. FP의 절반만 잡힘(나머지 14k는 거리 4.5 이내).
**판정: 채택(부분 해결).** 재학습 없이 어떤 run에도 사후 적용 가능.

---

## 7. 종합

| 가설 | 판정 | 한 줄 |
|---|---|---|
| H1 순도 | 기각 | target 비중↑ = 정밀도↓, 4/4 |
| H2 경계 (benign 주입) | scanning만 부분 채택 | hardneg로 scanning 0.386→0.578; mitm/ransomware 불변 |
| H3 규칙 share ∝ n^p | 채택 (scanning) | p=0.5 최선 raw 0.73~0.75; 추첨 분산 ±0.1 |
| H4 운영점 β | 채택 | balanced+β0.5 = 0.771 > 시스템 0.754; 파이프라인은 β=0 선택 중 |
| H5 데이터 겹침 | 채택 | mitm=먼 OOD benign, ransomware=근사 모순, scanning=benign 밀도 안 |
| H6 거리 게이트 | 채택 (부분) | τ=4.5, mitm +0.27, 부작용 없음 |

**제안 (실행은 사용자 확정 후).**
1. C0 구성(benign 75% → 균등 또는 √) + β 선택 기준 교체(사전 고정 0.5, 0/0.5/1.0 모두 보고). 파이프라인 baseline 재실행 ≈45분. 기대 global 0.684 → 0.75~0.77.
2. 거리 게이트 τ = train p99.9. 재학습 없음.
3. expert = 같은 context에 다른 β(운영점). "mitm/ransomware용" = 균등+β1.0, "scanning용" = √+β0. scorer는 운영점 선택기. 1번 뒤 EXP39 라우팅으로 검증.
4. 단일 추첨 수치(0.813, mitm 0.73) seed 43·44 재현 — 3 fit, 3분.
5. ransomware 근사 모순은 보고만(삭제 안 함, 0915 합의).

**주의.** 표의 모든 β 판독은 test에서 읽은 것. 배포 선택이 아니라 진단이며, 제안 1에서 β를 사전 고정하는 이유.

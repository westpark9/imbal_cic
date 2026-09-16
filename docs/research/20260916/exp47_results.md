# EXP47 결과 — CIC2018 모순 제거 후 기존 구조 재검증

보고일: 2026-09-16. 실행: 2026-09-15 18:09:13–19:23:03 KST, **73분 50초**, seed 42. 정제·기준 모델·기존 EXP39 legacy·진단이 모두 정상 완료됐다.

## 1. 결론

**정제 후 global macro-F1은 0.810653이고 최종 시스템도 동일하다. Expert 최종 채택은 0건이다.** 새 bank에는 global 오답을 맞히는 사례가 있으므로, 그 사례를 찾아 승인하는 학습 목표와 expert의 훼손을 줄이는 context를 다음 개선 대상으로 삼는다.

- **Residual → context 전달:** infiltration 오분류 4,204행과 web_attacks 오분류 5행은 모두 expert block에 포함됐다. 관련 오류가 학습 사례에서 빠진 현상은 확인되지 않았다.
- **Expert 자체 성능:** bank 전체로는 global 오답 10,085행을 교정할 수 있지만, 개별 expert를 전역 적용하면 훼손이 크다. 교정 능력과 전역 분류 성능을 구분해야 한다.
- **Scorer:** infiltration 교정 기회의 75.6%를 top-1 후보로 찾지만, benign의 교정 기회는 2.0%만 찾는다. Gate 없이 top-1을 적용하면 4,685행 교정과 553,246행 훼손이 발생한다.
- **Verifier:** 선택용 calibration의 교정 가능 top-1 후보 2,816행 중 원시 점수 양수는 6행이다. 보정 후에도 6행으로 같아, 손실은 보정 전 점수에서 이미 발생한다.
- **최종 정책:** 두 기존 정책 모두 최소 결정 사례 수를 충족하지 못해 global fallback을 선택했다. 따라서 이후 test에서의 호출률 0은 정책 전체의 fallback 결과다.

0909의 판정은 이 결론의 근거로 가져오지 않았다. 아래 수치는 모두 정제 후 새로 학습하고 저장한 EXP47 결과다.

## 2. 데이터와 실행 범위

| 항목 | 값 |
|---|---:|
| 정제 전 CIC2018 행 | 20,115,529 |
| 삭제한 모순 벡터 그룹 | 20,609 |
| 삭제 행 | 5,360,112 (26.65%) |
| 정제 후 행 | 14,755,417 |
| Train / validation / test | 8,666,430 / 3,046,514 / 3,042,473 |
| 남은 모순 그룹 / 실제 실험에서 사용된 삭제 행 | 0 / 0 |

모순 벡터 그룹의 모든 행을 제외했고, 비모순 그룹의 같은 라벨 중복과 기존 outer split 소속은 유지했다. 원본은 보존하며 정제 인덱스로 실험 데이터를 정의했다. 모델 라벨은 고정 7개 family다.

기존 C0 100k·benign share 0.75·attack balanced·TabPFN-v3 n_estimators 4·residual/anchor/scorer/normgain verifier 구조를 유지했다. 모든 context·bank·점수·임계값을 재생성했다. EXP31 원래 정책과, 같은 새 예측에서 학습한 EXP39 `legacy` 정책을 각각 평가했다. 별도 XGB full/C0 성능 실험과 새 context 구성 비교는 실행하지 않았다.

**평가 모집단이 정제된 test로 바뀌었으므로 과거 원본 test 성적과의 차이를 모델 개선량으로 계산하지 않는다.** 아래 global·expert·system 비교는 동일한 정제 test 기준이다.

## 3. 최종 성능

| 모델 / 정책 | Macro-F1 | Global 대비 | 최종 expert 채택 |
|---|---:|---:|---:|
| Global | **0.810653** | — | — |
| EXP31 기존 시스템 | 0.810653 | 0 | 0 |
| EXP39 legacy 정책 | 0.810653 | 0 | 0 |
| Scorer top-1 강제 적용, gate 생략·진단용 | 0.707038 | −0.103615 | 전 행 후보 적용 |
| 정답을 보고 올바른 expert를 고르는 bank oracle·진단용 | 0.962640 | +0.151987 | 10,085행 교정 |

마지막 oracle은 정답을 사용한 bank 교정 기회 진단값이다. 학습 가능한 router의 달성 보장이나 macro-F1의 엄밀한 최적 상한으로 해석하지 않는다. 기존 EXP31 `dense_oracle=0.887980`은 NLL이 가장 작은 모델을 고르는 다른 정의다. 두 값을 섞지 않는다.

Global과 두 최종 시스템의 클래스별 성적은 같다.

| 클래스 | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| benign | 0.997290 | 0.998368 | 0.997829 | 2,652,203 |
| bot | 0.999879 | 0.996461 | 0.998167 | 41,542 |
| brute_force | 1.000000 | 0.999469 | 0.999735 | 37,696 |
| ddos | 0.999966 | 0.998241 | 0.999103 | 262,680 |
| dos | 0.999691 | 0.984424 | 0.991999 | 39,483 |
| infiltration | **0.409070** | **0.312629** | **0.354406** | 8,742 |
| web_attacks | **0.205937** | **0.874016** | **0.333333** | 127 |

Global benign FPR은 **0.163185%**, accuracy는 0.996188이다. 개선할 주요 형태는 infiltration의 누락과 오탐, web_attacks의 낮은 precision이다. Web은 정제 후 test 127행이므로 support와 seed 변동을 함께 봐야 한다.

## 4. Residual이 실제 오류를 expert로 전달했는가

Residual은 `w_y × (−log p₀(y|x))`로 계산되고, 전체 행의 99.5% 분위값 **0.601645**에서 clipping된다. D_expert 2,000,000행 중 9,815행이 잘렸다.

| 클래스 | Clipping 전 residual 질량 비중 | Clipping 후 비중 | Global 오분류 → expert block 포함 |
|---|---:|---:|---:|
| benign | 6.18% | 47.25% | 5,522 → 5,498 |
| infiltration | **68.14%** | **6.34%** | **4,204 → 4,204** |
| web_attacks | **17.83%** | **0.10%** | **5 → 5** |

Clipping이 약한 클래스의 상대 질량을 크게 압축하는 것은 확인됐다. 다만 이 실행에서 infiltration/web의 오분류 행은 모두 expert block에 들어갔다. **Clipping만을 현재 fallback의 직접 원인으로 단정할 근거는 없다.** 추후 residual 가중치 실험에서는 클래스별 질량과 실제 사례 피복을 함께 평가해야 한다.

K=2/4/8 중 tune NLL-oracle macro-F1이 가장 높은 K=8이 선택됐다. 기존 `prune-mode=off`와 coverage/support 조건에서 8개 모두 유지됐으므로 pruning으로 교정 가능한 expert가 소실된 경로도 없다.

## 5. Expert가 충분히 좋은가

Global은 test 11,597행을 틀렸고, bank 중 하나 이상이 맞히는 행은 **10,085행(86.96%)**이다. 클래스별로 benign 4,151, bot 27, brute_force 0, ddos 462, dos 55, infiltration 5,375, web_attacks 15행이다. 이는 행별 합집합으로 세어 expert 사이 중복을 제거한 수다.

그러나 개별 expert를 전 행에 적용한 macro-F1은 0.7001–0.7946이다. 교정 사례가 있다는 것만으로 안전한 대체 모델이라고 볼 수 없다.

- **Expert 6:** block의 6,208행 중 infiltration 6,171행·web 37행. Infiltration recall은 0.8643이고 global의 infiltration 오답 4,827행을 맞힌다. 동시에 benign 정답 804,855행을 훼손하며, infiltration precision은 **0.009283**이다.
- **Expert 2:** benign 중심 block. Global의 benign 오답 3,898행을 교정하고 benign 정답 79행을 훼손한다. 전역 적용 시 다른 클래스 훼손이 생기지만, benign 오탐을 되돌릴 후보로서의 역할은 구분해 평가할 가치가 있다.
- Scorer top-1은 benign bank 교정 기회 4,151행 중 **83행(2.0%)**만 회수한다. Infiltration에서는 5,375행 중 **4,062행(75.6%)**을 회수한다. 따라서 scorer가 모든 약한 클래스에 같은 방식으로 실패하는 것은 아니다.

**Context 개선 가설:** shared anchor는 클래스별 2,000행을 넣고, 각 residual block을 더한다. Expert 6의 최종 context benign 비율은 10.85%, infiltration 비율은 44.33%다. 이러한 구성이 공격 예측을 넓히는 원인인지, 같은 context 예산에서 benign 대조 사례 비중을 바꿔 검증할 수 있다. 현재 결과는 비율과 훼손의 동반 관찰이며 인과 효과를 측정한 실험은 아니다.

## 6. Scorer 학습 목표가 실제 교정과 맞는가

기존 양성 라벨은 `OOF verifier 양수 판단 × weighted NLL gain − 비용 > 0`이다. 비용은 0이므로, 확률을 개선했지만 분류 정답이 변하지 않는 사례도 양성이 된다.

| Scorer 양성 query–expert 쌍 | 수 | 비율 |
|---|---:|---:|
| Global·expert 둘 다 정답 | **584,700** | **97.06%** |
| Global 오답 → expert 정답 | **11,470** | **1.90%** |
| 둘 다 오답 | 6,259 | 1.04% |
| Global 정답 → expert 오답 | 0 | 0% |
| 합계 | 602,429 | 100% |

전체 route에서 교정 가능한 query–expert 쌍은 26,570개이고, OOF verifier를 통과해 scorer 양성으로 남은 것은 11,470개(43.17%)다. 여기서는 expert별 학습 쌍을 세며, 앞 절의 행별 bank 합집합 10,085건과 단위가 다르다.

이 분포는 현재 scorer의 높은 양성률을 “오분류 교정을 잘 학습한다”로 해석하기 어렵다는 직접 근거다. 같은 bank에서 실제 교정·훼손과 중립 사례를 구분하는 target을 비교할 필요가 있다. OOF teacher의 raw weighted gain과 최종 verifier의 normgain 단위 차이도 다음 변경에서 별도로 통제한다.

## 7. Verifier와 최종 fallback의 직접 경로

### 7.1 원래 EXP31 정책 — 선택용 calibration

원래 selection mask를 통과한 **124,856행** 기준이다.

| 진단 단계 | 교정 | 훼손 |
|---|---:|---:|
| Scorer top-1을 gate 없이 적용 | 2,816 | 9,758 |
| Verifier 원시 `q_hat > 0`, pre gate 생략 | **6** | **3** |
| `q_hat − q_corr > 0`, pre gate 생략 | **6** | **3** |

`q_corr=0.000775204`이고, 교정 가능한 후보의 원시 q_hat 중앙값은 **−1.2168**이다. 훼손 후보의 중앙값은 −1.1454다. 이 후보 집합에서 q_hat의 교정/훼손 구분 AUROC는 **0.4465**다. 보정량을 줄여도 이미 음수인 대부분의 교정 사례가 돌아오지 않는 형태다.

최소 `helpful+harmful ≥ 30` 조건에 대해, 가장 느슨한 양수 승인에서도 유효 결정은 6+3=9건뿐이다. **77개 임계값 후보 전부 support 조건에서 탈락**했다. 일부는 proposal/harmful fraction 조건도 위반했다. 최종 `tau_pre=tau_post=∞`가 선택돼 test는 global을 그대로 사용했다.

### 7.2 같은 모델의 full test 진단

| 단계 | 교정 | 훼손 |
|---|---:|---:|
| Bank 중 올바른 expert를 정답으로 선택 | 10,085 | 0 |
| Scorer top-1, gate 생략 | 4,685 | 553,246 |
| 원시 verifier 양수, pre gate 생략 | **13** | **106** |
| 보정 점수 양수, pre gate 생략 | **13** | **106** |
| 실제 최종 정책 | **0** | **0** |

중간 행은 원인 분리용 강제 적용 결과다. 실제 시스템이 이만큼 호출하거나 훼손했다는 뜻은 아니다. Verifier 양수 815,032행 중 대다수는 global·expert가 모두 맞힌 행이며, 실제 라벨 변경은 119행에 그쳤다. 호출·승인 수만 보면 이 문제를 놓칠 수 있다.

### 7.3 기존 EXP39 legacy 정책에서도 재확인

다른 모델 구조를 도입하지 않고 기존 legacy arm만 같은 새 bank에서 학습했다. 별도 calibration select 87,399행·confirm 35,879행을 사용한다.

- Select: top-1 교정/훼손 **1,599/4,812 → post 양수 2/3 → 최종 0/0**.
- Confirm: **727/2,003 → 2/2 → 0/0**.
- 정책 후보 78개 모두 support 조건에서 탈락했다. 36개는 tail-F1 조건도 위반했다.
- Full test: top-1 4,101/385,932, post 양수 8/136, 최종 0/0.

EXP31과 EXP39는 학습 hyperparameter·정책 선택 규칙이 다르므로 중간 수치를 섞지 않는다. 공통으로 **실제 정답 변경을 유용하게 승인할 사례가 거의 남지 않아 global 정책을 선택**했다.

## 8. 다음 개선 순서

| 우선순위 | 변경 | 고정할 것 | 확인할 효과 |
|---|---|---|---|
| 1 | Verifier target을 실제 교정/훼손 구분에 맞춘 대조 | 정제 데이터·bank·scorer·post feature·학습 예산 | 양수 영역에 유효 교정이 늘고, 선택/확인 구간의 훼손을 통제하는가 |
| 2 | Scorer target에서 중립 확률 개선과 정답 교정을 구분 | 같은 bank와 verifier 조건 | Infiltration 교정 유지, benign 교정 expert 회수 증가, 위험한 후보 감소 |
| 3 | 같은 예산에서 expert anchor·대조 사례의 benign 비중 조정 | Global, residual 군집, 나머지 설정 | Infiltration recall을 유지하며 precision·benign 훼손이 개선되는가 |
| 4 | Residual clipping을 클래스별로 점검하는 대조 | Context 예산·나머지 학습 목표 | Tail residual 질량과 유효 사례 피복이 실제 교정 성능으로 이어지는가 |

1·2는 저장된 예측 캐시로 **추가 TabPFN 학습 없이** 원인을 분리해 비교할 수 있다. Post feature에 raw/z를 추가하는 변경은 target 변경과 분리한다. 단순히 최소 support 30을 낮추거나 verifier를 우회하면 현재 관측된 훼손을 통제할 수 없으므로, 먼저 교정·훼손 판별을 개선하는 대조를 권고한다.

이번 보고에서는 새 변경 실험을 실행하지 않았다. 현재 결론은 seed 42와 이미 사용해 온 development test의 정제 부분집합에 대한 결과다. 후속 조건은 validation에서 고정하고, 시간상 뒤의 확인 구간과 seed 43·44에서 재현성을 확인해야 한다.

## 9. 재검증과 근거

- `system_dump.npz`에서 실제 최종 예측과 global 예측이 전 행 동일하며 호출/채택 0임을 확인했다. 실제 배열에서 macro-F1을 재계산해 CSV와 일치함을 확인했다.
- 원래 calibration 후보와 전체 expert 진단 예측 간 후보 라벨 차이는 23/176,074행이다. 원래 gate 진단에는 원래 cal 후보·점수를 사용했다. 최종 정책 재계산과 native 결과의 라벨 차이는 0이다.
- [핵심 근거 JSON](exp47_result_evidence.json), [verifier 단계 집계 CSV](exp47_verifier_funnel.csv), [residual/context 집계 CSV](exp47_residual_context_evidence.csv), [재현 스크립트](summarize_exp47.py).
- [실험 완료](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/COMPLETE.json), [기준 성적](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/baseline/20260915_191945_nfv3_cic2018_exp31_c0alloc/per_class_metrics.csv), [expert 성적](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/diagnostics/expert_quality.csv), [기존 legacy 정책 성적](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/routing/20260915_192141_1159121_nfv3_cic2018_exp39_decision_routing/summary.csv).

원래 학습 결과·예측·정책을 보존했다. 09-16 후속 요청으로 [시각화 HTML](../../../lablog/html_report/clean_revalidation_0916.html)을 작성하고 통합본도 동기화했다. 실제 정책·임계값 30건 조건·강제 적용 진단을 분리해 볼 수 있다. ToN EXP48도 완료했으며 결과를 같은 HTML에 반영했다. 온라인 아티팩트 편집 도구가 없어 온라인 갱신은 수행하지 못했으며 Git push도 수행하지 않았다.

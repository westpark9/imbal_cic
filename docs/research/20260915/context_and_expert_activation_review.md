# ToN context와 scorer·verifier 활성화 검토

작성: 2026-09-15. 최초 범위: survey, 코드, 기존 결과 검토와 후속 실험 제안. 최초 검토에서는 새 모델 학습이나 새 정책의 test 평가를 수행하지 않았다. 후속 사용자 지시로 global context 실험(EXP46)의 실행을 승인받았으며, expert·scorer·verifier 변경은 계속 검토 단계다.

## 1. 권고

**자연 비율의 강한 global을 기준으로 고정하고, global의 특정 혼동을 양방향으로 교정하는 context expert를 만든다. 그 expert의 교정·훼손을 verifier가 구분하는지 먼저 확인하고, scorer는 그다음 필요한 expert를 적은 호출로 찾아내도록 학습한다.**

성공 조건은 세 가지다.

1. Context: 대표성 회복으로 ToN global이 무작위 100k 기준선에 뒤지지 않는다.
2. Verifier: 같은 global과 bank에서 expert를 받아들인 결과의 macro-F1이 증가한다.
3. Scorer: 전체 bank를 실행하는 verifier 진단과 비슷한 교정 성능을 더 적은 expert 호출로 유지한다.

Expert 호출률 자체를 올리는 것은 성공 조건이 아니다. Context 개선만으로 얻은 이득과 라우팅의 추가 이득은 따로 입증한다.

## 2. 현재 증거를 어떻게 읽어야 하는가

### 2.1 ToN C0는 prior와 추첨 구간을 함께 바꾼 비교다

원시 결과: [EXP45 full ToN](../../../tabpfn/results/20260915_110229_nfv3_toniot_exp45_ton_frame_oracle_ours_c0_tabpfn_plain_xgb_c0_xgb_f_s42/1b_summary.csv), seed 42, test 5,504,060행.

| 모델 | Macro-F1 | mitm F1 | ransomware F1 |
|---|---:|---:|---:|
| 기존 구성 C0 + TabPFN | 0.541470 | 0.016392 | 0.094434 |
| 같은 C0 + XGBoost | 0.516670 | 0.047567 | 0.076530 |
| 자연 비율 무작위 100k + TabPFN | **0.673994** | 0.032756 | 0.393357 |
| 전체 train + XGBoost | 0.668736 | 0.161943 | 0.149810 |

`nfv3_v3_exp45_ton_frame.py:111–115`에서 plain은 train 전체, C0는 `build_c0`의 별도 pool에서 추첨한다. `nfv3_v3_c0_context.py:111`의 pool은 각 class×scenario 시간순 앞 50%다. 따라서 C0 열화의 원인을 **prior 하나로 확정할 수 없다.** 동일 C0의 XGB도 낮다는 결과는 context 문제가 중요하다는 근거지만 prior·시간 피복의 기여를 분리하지는 않는다.

`0c_class_counts.csv`, `0g_c0_scenario.csv`, `run.log`로 재계산한 구성:

| 클래스 | Train 비율 | C0 행 | Plain 행 | C0 비율 / train 비율 |
|---|---:|---:|---:|---:|
| benign | 61.018% | 75,000 | 61,018 | 1.23배 |
| ddos | 15.048% | 3,144 | 15,048 | 0.209배 |
| xss | 10.299% | 3,143 | 10,300 | 0.305배 |
| mitm | 0.021845% | 1,803 | 22 | 82.54배 |
| ransomware | 0.014426% | 1,191 | 14 | 82.56배 |

기존 C0의 mitm recall은 0.993355지만 precision은 0.008264다. Plain도 recall 0.573920, precision 0.016859다. 따라서 추가 공격 support와 함께 **그 공격처럼 보이는 benign 및 다른 공격을 구분하는 contrast support**가 필요하다. 단순 recall 증대 목표는 FP를 더 늘릴 수 있다.

### 2.2 “Scorer가 아무 일도 하지 못했다”와 “최종 정책이 global이다”는 다르다

[EXP39 사후 감사](../20260909/exp39_completed_audit/calibration_grid_audit.csv), [0909 기록](../../../lablog/report/0909.md):

- Calibration에서 bank가 교정 가능한 6,025행 중 기존 scorer의 top-1은 5,286행을 회수했다(87.7%). 새 decision scorer는 2,793행(46.4%).
- 기존 NLL verifier는 그 후보 중 양수 gate에서 교정을 통과시키지 못했다. Decision verifier는 교정과 훼손을 함께 통과시켰다.
- 원래 calibration 격자의 후보는 제약을 통과하지 못했고 `global_fallback`이 선택됐다. Test 행마다 verifier가 전부 거절한 상황과 다르다.
- 변경 후보의 고점수 구간을 포함한 추가 격자에서는 기존 scorer + decision verifier가 calibration에서 +0.008035 macro, 교정 310 / 훼손 11을 만들었다. **이 정책은 독립 확인 구간과 test에 적용되지 않았다.**

따라서 0915 기록의 “격자를 보완해도 calibration 밖으로 전이하지 않았다”는 표현은 현재 확인한 산출물로 뒷받침되지 않는다. 확인된 것은 원래 EXP39의 최종 Δ=0과, 사후 보완 격자의 calibration 후보 존재다. 보완 후보의 전이는 미검증이다.

또한 EXP45는 명시적으로 **no routing**이며 global 후보 비교만 구현한다. EXP14의 ToN expert 시스템은 8월의 다른 구조다. 최신 ToN global을 기준으로 한 현재 scorer·verifier의 단계별 회수율은 아직 없다.

### 2.3 모순과 oracle의 역할

[CIC bank 진단](../20260911/dataset_quality_audit/cic_bank_realistic_oracle.json)은 global 0.762086, 행별 정답 oracle 0.849159, 충돌 벡터를 다수 라벨로 일관되게 배정한 bank 진단 0.803623을 기록한다. 모순을 반영하면 기대 격차가 줄어든다는 관찰은 유효하다.

다만 **다수 라벨 배정은 accuracy 목적의 해이며 macro-F1의 엄밀한 최댓값은 아니다.** 0.803623을 달성 가능한 수치나 정확한 macro-F1 상한으로 쓰지 않는다. EXP45의 test 다수 라벨 oracle 0.967212 역시 test 정답을 사용하며 학습 가능한 목표치가 아니다. 실제 잔여 기회는 고정 bank에 대한 다음 구간의 교정·훼손으로 측정한다.

동일한 관측 입력에 서로 다른 라벨이 있으면 입력만 보는 정책은 그 행들을 개별적으로 식별할 수 없다. 그러나 다음은 구분해야 한다.

- Global의 높은 confidence만으로 모든 다른 입력 신호까지 없다고 결론 내릴 수는 없다.
- Hash 충돌률은 모호함의 지표이며, 원래 정보에 없는 정답을 복원하는 신호가 아니다.
- 모순 그룹의 소수 라벨을 일괄 삭제하거나 모든 모순을 global로 보내는 규칙도 최적이라고 보장되지 않는다.

**후속 질의에 대한 명확화:** 같은 입력 x가 benign 100행, mitm 10행이면, x의 예측을 benign에서 mitm으로 바꿀 때 mitm 10행의 교정과 benign 100행의 훼손이 함께 발생할 수 있다. 이 변화는 라벨 선택의 손익 조정이지 같은 벡터의 실제 정답을 구분한 것이 아니다. 이 문서의 “교정할 혼동”은 **관측 입력에는 차이가 있지만 global이 활용하지 못한 오류**를 우선 뜻한다. 그런 오류가 있다는 사실과 현재 bank로 실제 교정 가능하다는 사실도 별개이므로 다음 구간에서 검증한다. 모순 영역의 예측 변경은 따로 집계해 이 둘을 섞지 않는다.

## 3. Survey에서 가져올 것과 적용 범위

기준: [tabpfn_papers_survey.xlsx](../../tabpfn_papers_survey.xlsx). 아래 행 번호는 실제 워크시트 행이다. 관련 원문을 추가 확인했다.

| Survey | 가져올 설계 | 이 프로젝트에서의 적용 범위 |
|---|---|---|
| r30 MixturePFN, r32 LoCalPFN | 가까운 입력에 관련 context를 제공하고 context를 공유 | 전 query별 검색 대신 작은 고정 bank. 논문 전체 방법에는 fine-tuning이 포함되므로 frozen 변형과 구분 |
| r41 BAPS | 대표 사례·경계·밀도·다양성을 함께 고려 | Global의 대표성, expert의 지원 사례와 대조 사례를 분리하는 근거. 512개로 ToN도 충분하다는 의미는 아님 |
| r42 Context Sampling | 비싼 선택법과 random을 같은 예산에서 비교 | Random을 필수 대조군으로 유지. 소규모 15개 데이터의 결과를 ToN의 prior 불필요 주장으로 확대하지 않음 |
| r43 Spurious Routing | 환경별 context 층화 | Train의 시간대·관측 가능한 서비스/프로토콜 구간을 피복. 라벨에서 만든 attack scenario를 추론 신호로 사용하지 않음 |
| r46 CRUMB | query 군집별 공유 context, 군집과 context의 분포 차이 감소 | Context 공유와 피복을 차용. 원형은 test batch의 특징을 함께 사용하므로 고정 bank 또는 과거 관측 창으로 바꾼 변형은 별도 명시 |
| r20 DistPFN | Context prior가 다른 posterior의 비교·보정 필요 | 단순 prior ratio, DistPFN, 무보정을 구분하는 대조. 보정 하나로 시간/feature 선택 편향까지 해결된다고 가정하지 않음 |
| r23 BoostPFN | Global이 놓친 구간의 support에 예산 투입 | 큰 residual만 반복하면 모순을 과표집할 수 있어, 그룹·mode와 held-out 실제 교정을 함께 봄 |
| r34 HINT | 호출 전 저비용 판단과 PFN 호출의 역할 분리 | 선택적 호출의 비교 구조. HINT에는 우리의 학습형 post-verifier가 없으며 low-confidence 호출만으로 충분하다는 근거도 아님 |

원문: [MixturePFN](https://arxiv.org/abs/2405.16156), [LoCalPFN](https://arxiv.org/abs/2406.05207), [BAPS](https://arxiv.org/html/2608.12989v1), [Context Sampling](https://arxiv.org/html/2607.26628v1), [Spurious Routing](https://arxiv.org/abs/2607.25532), [CRUMB](https://arxiv.org/html/2606.11473v1), [DistPFN](https://arxiv.org/html/2605.04363v1), [HINT](https://arxiv.org/html/2609.07956v1). BoostPFN은 이번 검토에서는 survey의 방법 요약을 참조했다.

BAPS·Context Sampling·Spurious Routing·CRUMB·HINT는 해당 survey의 프리프린트/워크숍 수준과 평가 범위를 감안한다. 아이디어의 출발점이며 ToN 개선을 보증하는 결과가 아니다.

## 4. Context 설계

### 4.1 Global: 전체적인 판정 기준을 유지한다

초기 라우팅 실험에서는 EXP45의 **자연 비율 100k global을 출발점**으로 사용한다. 새 global을 만들 경우 같은 스크립트·seed·실행 순서의 자연 비율 대조를 함께 둔다. 약해진 C0를 expert가 복구한 성적만으로 라우팅 기여를 주장하지 않는다.

Global context 개선은 두 요인을 구분한다.

| 대조 | 추첨 pool | Class quota | 질문 |
|---|---|---|---|
| G1 | 기존 앞 50% | 기존 benign 75% + 공격 균등 | 기존 C0 기준 |
| G2 | 기존 앞 50% | 자연 비율 | 같은 시간 피복에서 quota를 바꾸면 회복하는가 |
| G3 | 허용 train 전체 | 자연 비율 | G2 대비 뒤쪽 피복이 도움이 되는가 |
| G4 | 허용 train 전체 | 기존 quota | G1 대비 시간 피복 효과와 quota의 상호작용 |

EXP46에서는 window만 바뀔 때 class별 **실제 행 수**도 동일하게 고정한다. 기존 quota는 앞 절반 pool에서 산출된 75,000 benign·1,803 mitm·1,191 ransomware 등의 수를 두 window에 공유한다. 전체 pool에서 quota를 다시 계산하면 희소 클래스의 상한이 늘어나 시간 피복과 class 수 효과가 다시 섞이기 때문이다. 자연 quota도 전체 train 비율에서 한 번만 산출해 두 window에 공유한다.

G1/G3에 해당하는 EXP45 결과는 동기 자료다. 효과를 확정하는 짝지은 비교에서는 RNG·모델 순서까지 맞춘다. 비용을 줄이면 G2/G3를 먼저 비교하고, 필요한 경우에만 G4를 추가한다.

시간·mode 층화는 자연 class quota를 고정한 다음의 별도 비교다. Class 안에서 관측 가능한 시간 bin과 feature 군집의 대표 실제 행을 고르고, 같은 벡터가 선택을 독점하는지 확인한다. 우선 시간대 비례 층화와 random을 비교하고, 일별 균등 할당은 밀도를 바꾸는 추가 실험으로 둔다. 군집 중심을 가짜 flow로 만드는 대신 실제 대표 행을 쓴다.

**Prior 보정의 조건:** `p_ref(y|x) ∝ p_ctx(y|x) × π_ref(y)/π_ctx(y)`는 class-conditional 분포가 유지되고 posterior가 충분히 보정돼 있을 때의 식이다. 시간대 선택, hard-negative mining, residual 선택은 `p_ctx(x|y)`도 바꾼다. 따라서 이를 “항상 정확한 복원”이라고 부르지 않고, train reference prior를 사용한 대조군으로 검증한다. [DistPFN Appendix E](https://arxiv.org/html/2605.04363v1)의 label-shift 가정도 이 조건을 명시한다.

### 4.2 Expert: 특정 혼동의 지원 사례와 대조 사례를 제공한다

Bank를 클래스 소유권 대신 **global 예측 클래스 ↔ 실제 클래스의 혼동 구간**으로 정의한다. 모든 클래스에서 train/개발 구간의 residual을 탐색하며, ToN의 우선 가설은 benign↔mitm, benign↔ransomware다. 이는 기존 test 분석에서 나온 가설이므로 채택과 할당은 train/개발 자료로 다시 결정한다.

각 expert context의 구성:

1. **공유 anchor:** 자연 비율의 대표 실제 행. 일반 benign과 다른 공격을 인식할 기준을 남긴다. 기존 class당 2,000행 anchor는 ToN의 자연 prior와 크게 달라 그대로 이식하지 않는다.
2. **지원 사례:** Global이 틀린 구간과 같은 mode에 속한 해당 클래스의 실제 행. 극단 residual 하나보다 여러 mode의 반복되는 실패를 피복한다.
3. **대조 사례:** 그 mode에 가까운 benign뿐 아니라 혼동되는 다른 공격. Global이 이미 맞힌 사례도 포함해 expert가 망가뜨리기 쉬운 경계를 보여준다.

첫 소규모 가설의 예시는 expert 2개 × 각 8k, anchor/support/contrast = 25/37.5/37.5%다. **이 수치는 검증 전 시작값**이다. 희소 support가 모자라면 같은 행을 반복 복제해 채우지 않고 실제 확보 수를 기록한 뒤 사전 고정 규칙으로 anchor/contrast에 남은 예산을 돌린다. 비교군도 최종 class 수와 예산을 맞춘다. 8k가 충분한지는 bank의 held-out 교정에서 판단한다.

같은 vector 그룹이 반복될 때는 그룹 수·행 빈도·라벨 분포를 함께 저장한다. Global anchor에서는 자연 빈도를 유지하고, residual 추가분이 한 모순 그룹을 과도하게 반복하지 않게 한다. 모순 그룹을 담을 때 한 라벨만 자의적으로 선택하지 않는다. Context 선택으로 유도한 분포 변화와 원본 benchmark 전처리는 구분한다.

### 4.3 Expert의 행동 범위 제한 — 확정 규칙이 아닌 비교 조건

원래 제안은 다음 세 단계를 구분한다. S={benign, mitm}으로 두면:

1. **호출 자격:** global의 최종 예측이 S 안이면 이 expert를 scorer의 후보 목록에 넣는다. S 안이라고 모든 행에서 호출하는 것은 아니다.
2. **실제 호출:** scorer가 호출 전 입력으로 이 expert의 도움 가능성을 평가해 선택한 경우 실행한다.
3. **변경 승인:** expert가 제안한 S 안의 대안을 verifier가 승인하면 최종 라벨을 변경한다. 거절하거나 global과 같은 라벨이면 유지한다.

예: global=benign일 때 expert 호출 후 mitm으로 변경하거나, global=mitm일 때 benign으로 되돌릴 수 있다. Global=ransomware이면 원래의 엄격한 pair 제한에서는 이 expert가 후보에서 제외된다. 따라서 **실제 mitm인데 global이 ransomware라고 한 오류는 이 expert로 고칠 수 없게 된다.** 이것은 global의 기존 오류를 고정할 위험이다.

따라서 이 제한을 기본 설계로 확정하지 않는다. 먼저 개발 구간에서 bank를 넓게 관측해, global의 원래 예측 클래스별로 expert의 교정·훼손을 측정한다. 이후 **제한 없음 vs 위의 pair 제한**을 같은 bank에서 비교한다. 특정 혼동의 support/contrast로 expert context를 구성하는 것 자체는 pair 제한을 필수로 요구하지 않는다. Expert가 전문화될 구간과 expert를 호출할 자격은 서로 다른 설계 요소다.

이는 “expert가 자기 공격을 많이 예측하도록 만들기”와 다르다. **Benign 오경보를 되돌리는 방향과 놓친 공격을 찾는 방향을 모두 학습**한다. 초기 구현은 multiclass anchor를 가진 expert의 출력을 pair에 한정해 읽어도 된다. 무조건 이진 backbone으로 교체할 필요는 없다.

## 5. Scorer·verifier를 어떻게 바꿀 것인가

### 5.1 기존 수정과의 차이

EXP39에서 raw feature 추가, 평균 decision gain, verifier 의존성을 제거한 scorer는 이미 비교했다. 이 이름들을 새 해법으로 재제안하지 않는다. 이번 후보의 차이는 **강한 ToN global, 혼동별 bank, 행동 범위 제한의 절제 비교, 교정/훼손 분리, FP 목적지 반영, 다음 구간 검증**이다. 각각의 효과를 분리한다.

| 부분 | 변경 | 먼저 확인할 지표 |
|---|---|---|
| Expert bank | 혼동별 support + contrast; pair 제한 유무 비교 | 같은 global 대비 유용한 변경 후보와 훼손 후보의 수 |
| Verifier | 교정 확률과 훼손 확률을 분리하고 변경 목적지까지 반영 | 채택된 교정 수, 클래스별 FP, 전체 macro-F1 |
| Scorer | 임시 verifier의 거절로 정답을 지우지 않고 expert 후보 순위를 학습 | Top-1/2 교정 후보 회수율과 호출 수 |
| Calibration | 전체 행이 아닌 변경 후보의 score 분포까지 탐색 | 제약별 탈락 수, 다음 확인 구간 ΔF1 |

### 5.2 Verifier부터: 관측한 대안이 도움이 되는가

작은 bank의 모든 expert를 개발 구간에서 실행해 paired 출력을 저장한다. 먼저 scorer를 우회하고 verifier의 교정 능력을 평가한다. 입력은 `x, p0, pk, global→candidate 라벨, 지역별 support·거리·train 충돌 분포`다. 정확 hash/이웃의 통계는 train으로만 만든다. 같은 hash가 처음 등장한 경우에는 별도 상태로 표시한다.

출발 모델은 다음 두 확률을 구분하는 classifier다.

- H: expert 후보만 정답인 확률.
- D: global만 정답인 확률.

둘 다 오답인 변경과 동일 예측은 따로 기록한다. 순수 accuracy 이득은 `H − D`지만 **macro-F1은 FP가 어느 클래스에 생겼는지도 본다.** 이를 위해 두 확률 모델의 단순 이득을 대조군으로 두고, 다음 별도 목표를 비교한다.

- Development의 out-of-fold 혼동행렬을 기준으로 후보 변경이 각 클래스 TP/FP/FN에 주는 손익을 계산한다.
- 그 손익을 예측하도록 학습하거나, 예측한 정답 클래스 분포와 후보 라벨로 기대 손익을 계산한다.
- 실제 정책 선택에서는 근사 목표값이 아니라 전체 예측의 정확한 macro-F1·tail F1·benign FPR를 다시 계산한다.

예를 들어 정답은 benign인데 global은 mitm, expert는 ransomware를 출력하면 둘 다 오답이어서 EXP39의 ±1 목표는 0이다. 하지만 mitm FP가 줄고 ransomware FP가 늘어 macro-F1은 달라질 수 있다. 두 목적지를 구분하는 것이 이번 목표 설계의 추가점이다.

희소 클래스 정밀도를 지키려면 전체 benign FPR 외에 `benign→mitm`, `benign→ransomware` 등 **도착 클래스별 FP**를 함께 관리한다. 예컨대 benign 3,358,444행에서 0.01%의 오경보도 약 336행이다. 이는 mitm 전체 1,204행과 비교해 작지 않다. 이 산수는 위험 규모 설명이며 test에서 임계값을 정하는 규칙이 아니다.

### 5.3 Scorer: expert 호출 전에 후보를 찾는다

Verifier가 다음 확인 구간에서 실제 이득을 낸 뒤 연결한다. 모든 pair의 원래 결과를 남겨 임시 verifier가 거절한 교정 사례도 scorer가 학습할 수 있게 한다.

- 입력: `x/기존 표현 z, p0, train 지역 support, expert descriptor`. 실행 전에는 `pk`를 넣지 않는다.
- 첫 목표: global 대비 expert별 예상 교정·훼손 또는 그에 따른 상대 순위. 모든 expert가 동률인 행과 유용한 순위가 있는 행의 비중을 관리한다.
- Top-1과 top-2를 같은 bank에서 비교한다. 초기 진단에서는 pre-score 양수 조건이 모든 후보를 지우지 않도록 순위 평가와 호출 임계 선택을 분리한다.
- 학습용 pair를 교정·훼손 중심으로 표집하면 표집 확률을 기록하고 검증의 자연 분포로 확률을 보정한다. 단순 class balancing의 출력값을 자연 분포의 교정 확률로 읽지 않는다.

Expert마다 동일 호출량을 강제하는 load balancing은 성능 목적과 맞지 않을 수 있다. 학습 자료에서 드문 expert의 유효 사례를 충분히 보여주는 것과, 추론에서 억지로 호출하는 것은 구분한다. 관련 이론적 출발점은 고정 predictor 간 손실을 비교하는 [다중 expert learning-to-defer](https://proceedings.mlr.press/v267/mao25c.html)이며, 그 논문의 보장이 우리의 macro-F1·시간 이동 설정에 자동 적용되지는 않는다.

### 5.4 모순은 평균적으로 나은 행동을 배우는 데 반영한다

Train의 동일 벡터 그룹마다 라벨 히스토그램과 expert별 평균 손익을 보존한다. 서로 다른 정답 때문에 동일 입력에 상반된 routing label이 생기면 그 분포 또는 기대값으로 학습한다. 모순 그룹의 관측 빈도를 무조건 1로 바꾸는 것도 분포 변경이므로 빈도 가중을 명시한다.

Entropy가 높다고 무조건 expert를 호출하거나, 모순이라고 무조건 global을 선택하지 않는다. 해당 그룹에서 가능한 행동의 기대 손익과 검증된 안정성으로 결정한다. Feature가 같은 행을 구분하려면 추론 때 실제로 얻는 추가 정보가 필요하며, row ID나 정답 scenario를 넣는 방식은 해법이 아니다.

## 6. 데이터 역할과 실행 순서

EXP45는 test posterior만 저장했으므로, **그 파일로 새 scorer·verifier를 학습하거나 임계값을 선택하지 않는다.** 후속 실험에는 train/validation용 paired 출력이 새로 필요하다.

전체 train의 시간대를 global context에 쓰면서 기존 D_route의 정답도 학습하려면 자료 역할을 다시 정해야 한다. Context를 전체 train으로 넓힌 뒤 종전 train 내부 D_route를 그대로 쓰면 직접 학습행과 겹친다.

권장 예시:

- Outer train: context 및 bank 구성. Residual 선정에 쓰는 global 출력은 해당 행을 context에서 뺀 grouped cross-fitting으로 생성한다. 최종 global과 expert는 train에서 고정한다.
- Outer validation: 앞부분 route fit / 중간 policy 선택 / 뒷부분 확인. 예시 비율 40/30/30은 클래스별 지원 수를 확인해 사전 고정한다. 현재 benchmark가 class별 시간 분할이라는 점도 명시한다.
- Hash 반복이 많은 만큼 두 validation 보고를 구분한다. 전체 다음 구간은 실제 반복·모순을 포함한 성능, 학습 구간과 hash가 겹치지 않는 부분은 새 벡터에 대한 일반화다. 반복을 모두 삭제한 표만으로 전체 위험을 추정하지 않는다.
- Route에서 학습한 gate의 OOF 출력으로 policy 후보를 만들고, 선택·확인 역할을 섞지 않는다. 중요한 benign FP 추정에는 충분한 benign을 남기며 표집하면 가중치를 복원한다.
- Test: 선택을 마친 정책을 고정한 뒤 평가. seed 42–44 비교를 하더라도 이미 본 test를 새 blind test라고 부르지 않는다. 여러 seed는 sampling 변동 검증이며 독립 외부 검증을 대체하지 않는다.

이 역할 변경은 후속 비교군에도 동일하게 적용한다. 정확한 시간 순서가 필요한 배포 실험은 class별 chronological benchmark와 별도로, 전체 calendar 기준의 인과적 순서로 정의한다.

### 단계별 최소 실험

| 단계 | 작업 | 통과 조건 / 다음 수정 |
|---|---|---|
| A | 자연 global 고정; 필요하면 G2/G3 context 대조 | Context 자체 문제를 분리. 개선이 없으면 라우팅 기준선은 자연 global 유지 |
| B | 혼동별 expert 2개, 같은 크기 random expert 대조 | 다음 개발 구간에 유용한 후보가 존재. 없으면 support·contrast·행동 범위부터 수정 |
| C | 모든 expert 관측 + 기존 decision verifier 대조 + 분리된 교정/훼손 verifier | 다음 확인 구간에서 macro 이득과 FP 제약을 함께 만족 |
| D | 목적지별 F1 손익 및 변경 후보 기반 calibration 추가 | C 대비 개선 여부를 독립적으로 판단. 탈락 사유별 수치 저장 |
| E | Scorer top-1/2 연결; 임시 verifier 의존 목표와 직접 후보 목표 비교 | Dense verifier 이득을 유지하면서 호출 감소 |
| F | 고정 정책의 3-seed 최종 비교 | 같은 global 대비 추가 이득, 동일 자원 대조와 클래스별 P/R/F1 보고 |

Context support/contrast 추가, verifier 목표 변경, scorer 변경을 한 번에 켜는 run은 기여 분리가 안 되므로 피한다. 후속 지시로 global context의 4조건 비교만 승인됐으며, 나머지는 실행 합의 전 설계안이다.

## 7. 반드시 남길 결과와 기여 판정

- Global 오류 중 bank에 대안이 있는 수 → scorer shortlist에 남은 수 → verifier가 채택한 교정 수. 모순을 무시한 행별 oracle과 그룹 단위 분석을 분리한다.
- Expert 호출 / 채택 / 실제 라벨 변경의 세 비율. Global과 같은 라벨을 낸 호출은 교정으로 세지 않는다.
- Expert별 교정·훼손·둘 다 오답인 변경, confusion pair별 FP, benign 전체 FPR, 모든 클래스 P/R/F1.
- Context 행 인덱스, 각 역할의 인덱스, hash·class·시간대별 수, 선택 확률, model/config hash, 동일 RNG 실행 순서.
- Global 단독 / scorer+expert(검증 전) / 전체 bank+verifier / scorer+expert+verifier. 각 제거 비교에서 어떤 구성요소가 결과를 바꾸는지 확인한다.
- 총 고유 label 수, 전체 bank context 수, 평균 호출 수, 실제 시간·메모리. Global 100k + expert bank를 총 100k 정보 예산이라고 쓰지 않는다. 같은 자원의 random-context ensemble 및 가능한 global-only 대조를 둔다.

문헌상 local context, shared context, residual selection, prior correction, learning-to-defer는 모두 선행이 있다. 여기서 검증할 기여는 **모순을 과도하게 회수 대상으로 삼지 않으면서, 강한 global이 놓치는 혼동을 context expert가 고치고, verifier와 scorer가 그 이득을 실제로 보존한다는 것**이다. 새 이름보다 같은 global 위의 재현 가능한 추가 교정이 우선이다.

## 부록: Macro-F1 학습 목표의 한 가지 구체화

OOF 기준 혼동행렬에서 클래스 c의 실제 수를 N_c, 예측 수를 M_c, 정답 수를 T_c라 하면 `MacroF1 = mean_c 2T_c/(N_c+M_c)`다. Global 라벨 g에서 후보 a로 바꾸는 한 행에 대해 다음 1차 근사 이득을 만들 수 있다.

`u = (1/K) Σ_c [ 2ΔT_c/(N_c+M_c) − 2T_c·ΔM_c/(N_c+M_c)^2 ]`

여기서 `ΔT_c = 1[y=c,a=c] − 1[y=c,g=c]`, `ΔM_c = 1[a=c] − 1[g=c]`다. 기준 통계는 해당 query의 정답을 사용하지 않은 OOF/이전 개발 구간에서 얻고, 0 분모 처리는 사전 고정한다. 이 식은 목적지별 FP 비용을 전달하는 **근사 감독 신호**이며 여러 변경을 합친 정확한 F1 증분은 아니다. Policy 선택은 전체 혼동행렬로 다시 계산한다.

간단한 반례로 다수 라벨 oracle의 해석을 확인할 수 있다. 구분 가능한 benign 1,000행과, 같은 벡터를 가진 benign 10행/attack 1행을 생각하자. 모순 벡터를 다수 라벨 benign으로 배정하면 macro-F1≈0.49975지만 attack으로 배정하면 ≈0.58085다. 따라서 hash 다수 라벨 배정의 macro-F1을 일반적인 상한으로 취급하면 안 된다.

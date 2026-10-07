# 9월 30일 보고용 멘트 — S/V 완료 결과 반영

갱신: 2026-09-30. ToN 10:44:02, CIC2018 10:48:59 정상 완료. 한국어 슬라이드는 1~6장까지 작성돼 있으며, 이번 S/V 결과를 뒤쪽에 반영하는 작업은 별도다.

## 그대로 읽을 수 있는 멘트

이번 랩미팅 자료는 데이터 품질 검토, 모델 구조, 정제 데이터의 SOTA 성능과 비용, expert 역량 및 S/V 기여 순서로 준비하고 있습니다. 현재 1~4장은 데이터 검토와 모델·병목 설명, 5~6장은 완료된 SOTA 성능·비용 결과로 작성했습니다. 이후 장에는 현재 구성의 S/V 결과와 최종 모델 비교를 채울 예정입니다.

동일 feature에 서로 다른 라벨이 붙은 행을 제거한 데이터에서, 단일 seed로 XGBoost, BoostPFN, LoCalPFN, DistPFN 비교를 완료했습니다. 같은 10만 개 학습 행을 사용한 비교군 중 최고 Macro-F1은 CIC2018에서 DistPFN 0.7839, ToN에서 XGBoost 0.6983입니다. 전체 train을 사용한 XGBoost는 각각 0.7795, 0.7162입니다. 제안 모델은 expert와 route 학습에 추가 데이터를 사용하므로 이 비교군과의 성능 차이를 학습 예산 및 비용과 함께 평가하고 있습니다.

현재 seed 43·K=4 구성의 S/V 실험도 완료했습니다. CIC2018은 Global과 S/V 모두 Macro-F1 0.7824이고, expert 호출은 0건입니다. Validation에서 탐색한 78개 개입 정책 모두 성능이 감소하고 Benign 오탐·tail F1 조건을 위반해 Global을 유지했습니다. ToN은 Global 0.6796에서 S/V 0.7617로 개선됐고, Scanning F1이 0.0394에서 0.8587로 상승했습니다. 교정은 5,381건, 훼손은 1,101건입니다. 다만 모든 표본에서 expert를 하나씩 호출하므로 scorer는 expert 선택에는 사용되지만 호출을 생략하는 역할은 아직 하지 못하고 있습니다. ToN의 개선은 Scanning 중심이며 MitM·Ransomware는 추가 개선 대상입니다.

Context 변화에 민감한 양상이 있어 expert의 분류 능력을 두 관점에서 확인했습니다. 첫째는 정답 정보를 포함한 residual로 배정한 담당영역, 둘째는 정답 없이 입력 표현과 Global 예측만으로 배정한 표본군입니다. 여러 expert가 첫 번째 범위에서는 Global의 오류를 교정하지만, 두 번째 범위에서는 다른 클래스까지 특화 클래스로 예측하여 훼손이 커지는 사례가 있습니다. 예를 들어 CIC2018 e3은 담당 residual 영역에서 9,598건을 교정하고 훼손은 0건이지만, 입력 기반 배정 표본군에서는 교정 776건에 훼손 14,119건입니다. 다만 ToN e2는 담당 residual 영역에서도 훼손이 교정보다 많아, 모든 expert의 담당영역 역량이 확보됐다고 결론 내리지는 않았습니다. Residual은 Global의 오류 양상을 알려주지만, 현재 context 구성만으로 유사한 입력을 가진 클래스 간 구분이 충분해졌다고 보기는 어렵습니다. Context가 가장 큰 원인이라는 인과적 결론까지 확인한 것은 아닙니다.

Oracle 평가도 정정했습니다. 기존에는 여러 expert 중 정답을 맞힌 출력을 선택하고, 없으면 Global을 유지하여 expert의 담당영역 능력보다 낙관적인 수치가 나왔습니다. 현재는 정답 정보를 포함한 residual에서 가장 가까운 클러스터를 정하고, 해당 expert의 예측을 그대로 사용합니다. 이 기준의 Macro-F1은 CIC2018 0.8996, ToN 0.7437입니다. 정답을 사용한 조건부 진단값이며, 실제 추론 성능이나 모든 정책의 절대 상한을 의미하지는 않습니다.

이번 결과를 바탕으로 CIC2018에서는 validation에서 교정 이득을 낼 수 있는 expert 후보와 선택·채택 기준을 개선하고, ToN에서는 Scanning 개선을 유지하면서 호출 비용과 Benign 오탐을 줄이겠습니다. 최종 후보가 정해지면 같은 bank에서 S/V 제거 비교로 각 요소의 기여를 확인하고, 학습 예산과 실제 추론 비용을 포함해 SOTA와 비교하겠습니다.

## 완료 결과 요약

| 데이터 | Global | 입력 기반 고정 배정 | S+V | 정책상 호출률 | V 채택 | 교정 / 훼손 |
|---|---:|---:|---:|---:|---:|---:|
| CIC2018 | 0.7824 | 0.7613 | 0.7824 | 0% | 0 | 0 / 0 |
| ToN | 0.6796 | 0.7168 | 0.7617 | 100% | 28,035 | 5,381 / 1,101 |

성능은 Macro-F1이다. ToN의 채택률은 전체 test의 1.23%이며, 실제 출력 변경은 6,504건이다. 호출·채택은 저장 예측에 정책을 적용한 논리적 건수로, 준비 단계의 전체 expert 계산시간을 온라인 희소 추론 비용으로 해석하지 않는다.

## 용어와 해석

- 관점 A: **정답 residual 배정 영역**. 정답 정보가 포함된 진단용 배정이므로 온라인 배정과 구분한다.
- 관점 B: **입력 기반 배정 표본군**. 입력 표현과 Global 확률로 최근접 배정하며, A에 속하는 행과 속하지 않는 행을 모두 포함한다. ‘담당영역 밖’이라고 부르지 않는다.
- A/B의 표본이 다르므로 두 F1을 직접 빼서 일반화 손실로 해석하지 않는다. 각 범위 안에서 같은 표본의 Global과 expert를 비교한다.
- 이번 ToN 고정 배정의 Macro-F1은 증가하지만 Accuracy는 0.9013 → 0.8829로 감소한다. S+V의 Accuracy는 0.9031이다. Macro-F1 개선과 전체 오분류 감소를 같은 뜻으로 쓰지 않는다.
- Expert 수가 많다는 사실만으로 여러 클래스에 대한 유용한 특화가 확보됐다고 말하지 않는다. Context 구성비와 검증 표본에서의 교정·훼손이 근거다.
- 이전 seed 42·K=8/7에서도 정제 후 S/V 실험은 있었다. CIC2018 0.8107(호출 0), ToN 0.7555였으며, 이번 구성과 seed·bank가 달라 직접적인 개선량으로 계산하지 않는다.
- ToN의 MitM·Ransomware 평균 F1은 validation 확인 구간에서 −0.0501, test에서 −0.0006 변했다. 전체 Macro-F1 개선이 tail 전체의 개선을 뜻하지 않는다.
- 이번 결과는 S/V 결합 효과다. S와 V 각각의 인과적 기여는 같은 bank의 제거 비교로 확인해야 한다.
- ToN은 현재 정책과 validation Macro-F1이 같으면서 가중 호출률 38.08%인 후보도 있었다. 현재 선택기는 동점이면 먼저 나온 정책을 유지한다. 다음에는 호출 비용으로 동점을 해소하되, 38.08%를 해당 후보의 test 호출률로 제시하지 않는다.

## HTML 세 탭 통합 완료

`clean_revalidation_0916`(09/16~17), `scorer_verifier_target_0918`, `scorer_verifier_target_0928`을 **정제 데이터에서의 Expert 역량과 S/V 검증** 한 탭으로 통합했다. 통합 탭 ID는 `scorer_verifier_target_0928`이며 최신 날짜는 09/30이다.

1. **현재 성능과 개선 대상**: seed 43·K=4의 Global / 입력 기반 고정 배정 / S+V와 Global 클래스별 F1. SOTA 전체 결과는 0908 탭으로 연결한다.
2. **Expert의 잠재 교정 능력**: 바뀐 oracle 정의를 문장으로 설명하고 두 데이터 수치만 제시한다. 과거 다섯~여섯 조건 표는 본문에서 제외한다.
3. **실제 배정에서 발생하는 훼손**: A/B를 나란히 보여주고 CIC2018 e3, ToN e3 및 예외인 ToN e2를 설명한다. 전체 expert별 표와 context 구성은 펼침 상세로 둔다.
4. **S/V가 이득을 보존하는가**: 최종 성능, 호출·채택·교정·훼손, 다음 개선을 한 흐름으로 제시한다. 이전 seed 42와 BoT 진단은 별도 접힌 과거 근거에 남긴다.

기존 HTML 원본은 `docs/research/20260930/archive/pre_merge_reports/`에 보존했다. 이전 탭 ID와 개별 HTML 주소는 통합 탭으로 연결된다. 0911 데이터 감사와 0908 SOTA 탭은 유지했다. 로컬 동기화 및 검증을 완료했으며, Claude 온라인 아티팩트는 연결된 편집 도구가 없어 갱신하지 못했다.

## 오늘의 실행 순서

실행 중 확인: 이전 expert는 fitted state가 저장돼 있지 않아 같은 context·seed와 원래 batch 조건의 재학습에서도 예측이 완전히 일치하지 않았다. EXP62에서는 Global fitted state와 context IDs를 유지하되 expert fitted state를 새로 저장하고, 그 동일 모델로 route/cal/test를 모두 계산했다. 현재 성능 표는 이 EXP62의 결과이고 oracle·A/B는 EXP59 진단이다. 앞서 안내한 ToN 고정 배정 0.7191은 EXP59 값이며, 이번 S/V와 같은 fitted bank의 고정 배정은 0.7168이다.

1. **완료:** 현재 context + S/V를 평가했다. EXP62는 이전에 사용한 direct decision-gain scorer + NLL quantile verifier(S1V0) 구성이다.
2. **다음:** CIC2018은 validation의 후보별 교정·훼손과 S/V 점수를 점검한다. ToN은 같은 검증 이득을 내는 임계값 중 호출이 적은 정책을 우선하고, 별도 확인 구간의 tail·Benign 성능을 점검한다. Test 결과로 임계값을 조정하지 않는다.
3. 선택·채택의 문제라면 해당 단계의 학습 목표나 검증 기준을 고친다. Expert 후보 자체가 유사 클래스에서 손해라면 해당 expert의 context를 수정하거나 사용 대상을 제한한다.
4. 최종 후보를 정한 후 같은 bank와 예측을 재사용하여 S/V 제거 비교를 수행하고 SOTA 및 비용과 함께 정리한다.

## 근거

- SOTA: `docs/research/20260930/exp61_report_data.json`
- 현재 bank·S/V: `tabpfn/results/20260930_exp62_sv_current_bank_s43/{cic2018,toniot}/summary.csv`
- 정정 oracle: `tabpfn/results/20260928_exp59_residual_oracle_s43/{cic2018,toniot}_s43/summary.csv`
- A/B: `tabpfn/results/20260928_exp59_residual_oracle_s43/expert_capability/expert_summary.csv`
- 이전 S/V: `tabpfn/results/exp56_label_free_20260922/{cic2018,toniot}/1a_summary.csv`
- 완료 상태: `tabpfn/results/20260930_exp62_sv_current_bank_s43/status.json`

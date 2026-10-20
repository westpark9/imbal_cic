# EXP75 · 분류 구조 개선 실험

기존 순차 공격 도입 시나리오, 최종 train 10만 행, 별도 validation, seed 43을 유지했다. 모든 test 행을 정책별로 한 번 평가했으며 실제 수집 시간이나 환경 drift를 재현한 실험은 아니다.

EXP75는 2,048행 이하 context를 초기화한 뒤 고정 TabPFN을 통해 context feature만 최적화했다. 클래스 균형 query 손실로 학습하고 validation 손실로 초기 상태를 포함한 checkpoint를 선택했다. untuned 대조군은 같은 context·단일 estimator·전처리를 사용한다. 10만 context·4 estimators인 원 TabPFN과의 차이는 최적화 효과 하나로 해석하지 않는다.

| 데이터 | 배분 | 방법 | 지연 없음 Macro-F1 (%) | 1,000행/초 Macro-F1 (%) | 지연 없음 Benign FPR (%) |
|---|---|---|---:|---:|---:|
| cic2018 | natural | xgb | 76.80 | 76.79 | 0.148 |
| cic2018 | natural | tabpfn | 77.72 | 77.47 | 0.006 |
| cic2018 | natural | icd | 72.72 | 71.84 | 9.287 |
| cic2018 | natural | icd_init | 72.73 | 72.73 | 9.289 |
| cic2018 | uniform | xgb | 77.03 | 77.01 | 0.900 |
| cic2018 | uniform | tabpfn | 81.35 | 81.17 | 0.127 |
| cic2018 | uniform | icd | 72.20 | 71.20 | 8.983 |
| cic2018 | uniform | icd_init | 71.47 | 71.47 | 9.669 |
| toniot | natural | xgb | 79.37 | 79.28 | 7.238 |
| toniot | natural | tabpfn | 73.66 | 73.39 | 7.741 |
| toniot | natural | icd | 65.74 | 63.28 | 14.017 |
| toniot | natural | icd_init | 69.86 | 69.86 | 12.283 |
| toniot | uniform | xgb | 74.51 | 74.46 | 8.790 |
| toniot | uniform | tabpfn | 73.00 | 72.72 | 8.305 |
| toniot | uniform | icd | 68.68 | 66.07 | 13.510 |
| toniot | uniform | icd_init | 64.53 | 64.53 | 15.847 |

위 수치는 자동 집계한 관찰 결과다. 클래스별 변화, 다른 클래스로부터 유입된 FP, 단일 head 또는 초기 context 대조군을 함께 검토한 뒤 원인과 채택 여부를 판단한다. 실행 성공을 성능 개선이나 논문 근거 채택으로 간주하지 않는다.

실행: `tabpfn/results/v7/exp75/20261015_181113_icd_replay_100k_s43_6558120d`
상세 결과: [exp75 results.json](../../../tabpfn/results/v7/exp75/20261015_181113_icd_replay_100k_s43_6558120d/results.json)
독립 검증: [exp75 verification.json](../../../tabpfn/results/v7/exp75/20261015_181113_icd_replay_100k_s43_6558120d/verification.json)

HTML·논문은 이번 실행 요청에 따라 자동 변경하지 않았다.

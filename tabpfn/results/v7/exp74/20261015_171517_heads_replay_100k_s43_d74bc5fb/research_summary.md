# EXP74 · 분류 구조 개선 실험

기존 순차 공격 도입 시나리오, 최종 train 10만 행, 별도 validation, seed 43을 유지했다. 모든 test 행을 정책별로 한 번 평가했으며 실제 수집 시간이나 환경 drift를 재현한 실험은 아니다.

EXP74는 단일 TabPFN과 동일한 전체 context를 유지하고, 별도 validation 일부에서 분포 목표가 다른 세 head를 학습했다. 나머지 validation은 head 선택과 확률 보정에 분리해서 사용했다. 기존 기준선은 같은 validation 라벨 접근 권한이 있으나 사용하지 않았으므로, 추가 라벨 활용과 결합 구조의 기여를 동일시하지 않는다.

| 데이터 | 배분 | 방법 | 지연 없음 Macro-F1 (%) | 1,000행/초 Macro-F1 (%) | 지연 없음 Benign FPR (%) |
|---|---|---|---:|---:|---:|
| cic2018 | natural | xgb | 76.80 | 76.79 | 0.148 |
| cic2018 | natural | tabpfn | 77.72 | 77.47 | 0.006 |
| cic2018 | natural | heads | 72.07 | 71.83 | 8.533 |
| cic2018 | uniform | xgb | 77.03 | 77.01 | 0.900 |
| cic2018 | uniform | tabpfn | 81.35 | 81.17 | 0.127 |
| cic2018 | uniform | heads | 72.72 | 72.48 | 8.134 |
| toniot | natural | xgb | 79.37 | 79.28 | 7.238 |
| toniot | natural | tabpfn | 73.66 | 73.39 | 7.741 |
| toniot | natural | heads | 65.39 | 65.01 | 20.187 |
| toniot | uniform | xgb | 74.51 | 74.46 | 8.790 |
| toniot | uniform | tabpfn | 73.00 | 72.72 | 8.305 |
| toniot | uniform | heads | 66.28 | 65.95 | 15.957 |

위 수치는 자동 집계한 관찰 결과다. 클래스별 변화, 다른 클래스로부터 유입된 FP, 단일 head 또는 초기 context 대조군을 함께 검토한 뒤 원인과 채택 여부를 판단한다. 실행 성공을 성능 개선이나 논문 근거 채택으로 간주하지 않는다.

실행: `tabpfn/results/v7/exp74/20261015_171517_heads_replay_100k_s43_d74bc5fb`
상세 결과: [exp74 results.json](../../../tabpfn/results/v7/exp74/20261015_171517_heads_replay_100k_s43_d74bc5fb/results.json)
독립 검증: [exp74 verification.json](../../../tabpfn/results/v7/exp74/20261015_171517_heads_replay_100k_s43_d74bc5fb/verification.json)

HTML·논문은 이번 실행 요청에 따라 자동 변경하지 않았다.

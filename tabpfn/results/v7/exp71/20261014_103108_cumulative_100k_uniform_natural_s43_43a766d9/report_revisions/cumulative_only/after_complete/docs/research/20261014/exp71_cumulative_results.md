# EXP71 · 누적 전체의 갱신 비용과 탐지 품질

2026-10-14 사용자 요청으로 보고서의 비교 범위를 누적 전체로 정리했다. 제한 보존 실행과 이전 기록은 원본에 보존하며 이번 화면의 근거로 사용하지 않는다.

목적은 새 공격의 라벨 사례가 확보될 때 TabPFN이 가중치 재학습 없이 유용한 탐지 품질을 유지하면서 갱신 비용을 줄일 수 있는지 확인하는 것이다. 클래스 최초 출현 순서를 유지한 단계별 공급이며 raw timestamp 재생은 아니다. 각 단계의 라벨 공급 후 seen-class 전체 test로 평가한다.

현재 완료 결과의 최종 입력은 cic2018 100,000행, toniot 100,000행다. Benign 75%를 매 단계 나눠 공급하고 공격 25%는 균등 또는 정제 train의 공격 내 자연비율로 배분한다. 과거 사례를 전부 누적하고 각 조건의 두 모델에 동일 학습·test IDs를 제공했다.

누적 전체의 4개 구성 중 0개에서 TabPFN의 후속 갱신 합계가 더 짧았다. 가중치 재학습이 없더라도 전체 context 캐시 재구성 비용이 발생한다. 갱신과 최종 test 대량 추론을 분리해서 비교해야 한다.

현재 TabPFN 호출은 매 단계 전체 context로 fit_with_cache를 다시 실행한다. 4개 ensemble의 전처리와 context 캐시를 재구성하므로 가중치 최적화를 하지 않는다는 사실이 갱신 계산량이 없거나 XGB보다 작다는 뜻은 아니다. 입력 선택·모델 생성·fit/cache를 합해 갱신으로 측정하고 초기 구축과 후속 합계를 분리한다. 추론은 전체 test 예측 및 결과 기록 시간으로 별도 측정한다. 전처리와 캐시 생성의 내부 세부 시간은 분리 측정하지 않았으므로 특정 연산의 원인 비중을 주장하지 않는다.

최종 Macro-F1은 4개 구성 중 2개에서 TabPFN이 XGB보다 높았다. 동일한 test에서 데이터·공급 배정별 결과를 비교하며, 전체 평균 외에 정상 오탐률과 공격별 P/R/F1도 확인한다.

CIC2018 — 공격 균등: TabPFN F1 78.43% / XGB 78.12%, 후속 갱신 41.18초 / 5.24초; 공격 내 자연비율: TabPFN F1 77.06% / XGB 75.66%, 후속 갱신 44.25초 / 4.23초.

ToN — 공격 균등: TabPFN F1 69.09% / XGB 71.22%, 후속 갱신 48.75초 / 12.39초; 공격 내 자연비율: TabPFN F1 69.93% / XGB 77.43%, 후속 갱신 50.65초 / 11.29초.

현재의 판단 — TabPFN은 사전학습 가중치를 바꾸지 않고 새 공격의 라벨 사례를 반영할 수 있다. 그러나 이 작동 방식만으로 XGB보다 낮은 갱신 비용이나 필요한 탐지 품질이 보장되지는 않는다. 갱신·추론 비용을 따로 측정하고, 소수 공격의 탐지 품질을 충족하는 입력 구성과 규모를 확인해야 한다.

CIC2018 · 균등: 후속 갱신 합계는 XGB 5.24초, TabPFN 41.18초다. TabPFN−XGB 최종 Macro-F1 차이는 +0.31%p, 정상 오탐률 차이는 -1.99%p다. 전체 100,000개를 공급했고 최종 갱신에는 100,000개를 사용했다. 공격별 F1의 절대 차이가 가장 큰 두 클래스는 Infiltration XGB 10.65% / TabPFN 20.77%; Web attack XGB 37.79% / TabPFN 29.47%다.

CIC2018 · 균등의 평가 표본이 가장 적은 두 공격을 따로 보면, Infiltration는 test 8,742개, 최초 공급 4,910개, 최종 context 잔존 4,910개다. XGB F1 10.65% (P 6.12% / R 41.29%, 공격→Benign 5,064건 / Benign→공격 55,411건); TabPFN F1 20.77% (P 32.68% / R 15.23%, 공격→Benign 7,333건 / Benign→공격 2,737건). Web attack는 test 127개, 최초 공급 450개, 최종 context 잔존 450개다. XGB F1 37.79% (P 24.42% / R 83.46%, 공격→Benign 21건 / Benign→공격 260건); TabPFN F1 29.47% (P 17.69% / R 88.19%, 공격→Benign 14건 / Benign→공격 285건). 이 혼동 건수는 최종 예측에서 직접 확인한 관찰이며, 사례 수 부족이나 context 구성 중 무엇이 원인인지는 이 비교만으로 분리할 수 없다.

CIC2018 · 공격 내 자연: 후속 갱신 합계는 XGB 4.23초, TabPFN 44.25초다. TabPFN−XGB 최종 Macro-F1 차이는 +1.39%p, 정상 오탐률 차이는 -0.30%p다. 전체 100,000개를 공급했고 최종 갱신에는 100,000개를 사용했다. 공격별 F1의 절대 차이가 가장 큰 두 클래스는 Web attack XGB 4.05% / TabPFN 17.83%; Infiltration XGB 26.25% / TabPFN 23.07%다.

CIC2018 · 공격 내 자연의 평가 표본이 가장 적은 두 공격을 따로 보면, Infiltration는 test 8,742개, 최초 공급 641개, 최종 context 잔존 641개다. XGB F1 26.25% (P 23.81% / R 29.25%, 공격→Benign 6,167건 / Benign→공격 8,180건); TabPFN F1 23.07% (P 80.45% / R 13.46%, 공격→Benign 7,561건 / Benign→공격 286건). Web attack는 test 127개, 최초 공급 9개, 최종 context 잔존 9개다. XGB F1 4.05% (P 14.29% / R 2.36%, 공격→Benign 124건 / Benign→공격 14건); TabPFN F1 17.83% (P 46.67% / R 11.02%, 공격→Benign 113건 / Benign→공격 12건). 이 혼동 건수는 최종 예측에서 직접 확인한 관찰이며, 사례 수 부족이나 context 구성 중 무엇이 원인인지는 이 비교만으로 분리할 수 없다.

ToN · 균등: 후속 갱신 합계는 XGB 12.39초, TabPFN 48.75초다. TabPFN−XGB 최종 Macro-F1 차이는 -2.13%p, 정상 오탐률 차이는 +2.53%p다. 전체 100,000개를 공급했고 최종 갱신에는 100,000개를 사용했다. 공격별 F1의 절대 차이가 가장 큰 두 클래스는 Scanning XGB 30.76% / TabPFN 15.37%; MitM XGB 13.46% / TabPFN 4.58%다.

ToN · 균등의 평가 표본이 가장 적은 두 공격을 따로 보면, MitM는 test 1,161개, 최초 공급 2,856개, 최종 context 잔존 2,856개다. XGB F1 13.46% (P 7.24% / R 95.35%, 공격→Benign 51건 / Benign→공격 13,426건); TabPFN F1 4.58% (P 2.34% / R 97.33%, 공격→Benign 29건 / Benign→공격 45,689건). Ransomware는 test 709개, 최초 공급 2,151개, 최종 context 잔존 2,151개다. XGB F1 14.11% (P 7.60% / R 98.87%, 공격→Benign 7건 / Benign→공격 8,526건); TabPFN F1 14.15% (P 7.63% / R 97.60%, 공격→Benign 5건 / Benign→공격 8,376건). 이 혼동 건수는 최종 예측에서 직접 확인한 관찰이며, 사례 수 부족이나 context 구성 중 무엇이 원인인지는 이 비교만으로 분리할 수 없다.

ToN · 공격 내 자연: 후속 갱신 합계는 XGB 11.29초, TabPFN 50.65초다. TabPFN−XGB 최종 Macro-F1 차이는 -7.50%p, 정상 오탐률 차이는 +4.29%p다. 전체 100,000개를 공급했고 최종 갱신에는 100,000개를 사용했다. 공격별 F1의 절대 차이가 가장 큰 두 클래스는 MitM XGB 71.94% / TabPFN 3.74%; XSS XGB 87.81% / TabPFN 83.98%다.

ToN · 공격 내 자연의 평가 표본이 가장 적은 두 공격을 따로 보면, MitM는 test 1,161개, 최초 공급 31개, 최종 context 잔존 31개다. XGB F1 71.94% (P 89.92% / R 59.95%, 공격→Benign 253건 / Benign→공격 75건); TabPFN F1 3.74% (P 1.93% / R 61.24%, 공격→Benign 230건 / Benign→공격 36,146건). Ransomware는 test 709개, 최초 공급 20개, 최종 context 잔존 20개다. XGB F1 38.84% (P 26.76% / R 70.80%, 공격→Benign 198건 / Benign→공격 1,374건); TabPFN F1 40.39% (P 28.37% / R 70.10%, 공격→Benign 166건 / Benign→공격 1,255건). 이 혼동 건수는 최종 예측에서 직접 확인한 관찰이며, 사례 수 부족이나 context 구성 중 무엇이 원인인지는 이 비교만으로 분리할 수 없다.

RTX 4090, 동일 입력으로 모델 worker를 순차 실행했다. TabPFN ensemble 4, fit_with_cache, CPU KV cache; XGB CUDA hist 300 rounds, depth 8. 단계별 1회 timing이며 프로세스 시작·데이터 준비·배포·직렬화는 비용에 포함하지 않는다.

입력 확대 전 누적 실행 `tabpfn/results/v7/exp71/20261013_174035_cumulative_recovery_s43_4229fd93`과 동일 test·출현 순서·모델 설정으로 비교했다. scale_validation.json에서 기존 공급 IDs가 같은 확대 단계에 포함되는 것을 검증했다. 일부 이전 worker와 driver/kernel이 달라 비용 변화를 입력 규모만의 인과 효과로 해석하지 않는다.

후속 판단은 자연비율에서 부족한 공격의 실제 context 표본 수와 균등 조건의 P/R/F1을 함께 확인하여, 입력 확대가 누락된 공격 탐지와 정상 오탐을 동시에 개선하는지 평가하는 것이다. 특정 실패를 사례 수나 context 비율 하나로 확정하지 않는다. 단일 seed이며 같은 개발 holdout을 반복 관찰한 탐색 결과다.

출처: `tabpfn/results/v7/exp71/20261014_103108_cumulative_100k_uniform_natural_s43_43a766d9/summary.csv`, `class_metrics.csv`, `jobs/*/stage_*/metrics.json`, `validation.json`. HTML: `lablog/html_report/bounded_updates_1013.html`. 프로토콜: `docs/research/20261014/exp71_100k_protocol.md`.

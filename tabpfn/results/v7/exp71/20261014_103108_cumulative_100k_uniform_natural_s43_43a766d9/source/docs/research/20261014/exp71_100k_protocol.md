# EXP71: 누적 입력 10만 행 확대

2026-10-14 사용자 요청: 제한 보존 비교를 제외하고 10만 개로 확대한다. 연구 버전은 승인된 v7을 유지한다. Expert/S/V 없이 단일 TabPFN과 XGBoost를 비교한다.

각 데이터의 최종 누적 입력은 정확히 100,000행이다. Benign 75,000행을 CIC2018의 5개 단계에 15,000행씩, ToN의 6개 단계에 12,500행씩 공급한다. 공격 25,000행은 (1) 공격별 균등 배정, 가용량 부족분은 나머지 공격에 재분배, (2) 정제 train의 공격 내 자연비율 배정으로 나눈다. 자연비율 조건에 추가적인 tail 최소량을 부과하지 않는다. 전체 트래픽의 자연 분포를 재현하는 조건은 아니다.

CIC2018: 초기 Benign·Brute force·DoS → DDoS → Web attack → Infiltration → Bot. ToN: 초기 Benign·Scanning·DoS → Injection·DDoS → Password → XSS → Ransomware·Backdoor → MitM. 이전 실험에서 합의한 최초 출현 순서와 묶음을 유지한다. 실제 timestamp 재생은 아니다. 각 새 공격의 라벨 사례가 확보된 후 평가한다.

이전 EXP71 stream을 확장한다. 각 클래스 bank의 기존 IDs를 유지하고 정제 train에서 중복 없이 추가한다. Benign은 기존 단계별 공급 IDs를 같은 단계에 남기면서 확대한다. 따라서 각 조건·단계의 이전 신규 IDs가 새 신규 IDs에 포함된다. 과거 사례는 모두 누적하고 두 모델에 같은 학습·test IDs를 전달한다. Test는 기존 전체 정제 test와 byte 단위로 동일하다.

Seed 43. XGBoost CUDA histogram, 300 trees, depth 8. TabPFN-v3 frozen weights, ensemble 4, sample subsampling 없음, fit_with_cache, CPU KV cache. 매 단계 누적 전체 입력으로 fit/cache를 다시 생성한다. 갱신은 입력 구성+모델 생성+fit/cache 시간이며 초기 구축과 후속 갱신 합계를 분리한다. 전체 seen-class test 추론은 batch 32,768로 별도 측정한다. GPU worker는 비용 간섭을 피하기 위해 순차 실행한다.

확인할 것은 동일 평가 데이터에서 입력 확대가 최종 및 신규 공격 성능과 갱신·추론 비용을 어떻게 바꾸는지다. 클래스별 성능은 보존하되 HTML에서는 토글 안에 둔다. 이전 누적 실행은 CIC2018 22,280 / ToN 36,864행이다. 단일 seed·단계별 1회 timing이며, 일부 이전 worker와 GPU driver/kernel이 달라 규모 효과의 정밀한 비용 추정에는 한계가 있다. 새 실행 내 XGB/TabPFN은 같은 환경에서 비교한다.

성공 판정은 44개 단계 완료, 원본 예측에서 지표 재계산, 모든 누적/공급 IDs 및 test 일치 검증이다. 결과가 나오기 전 성능 향상이나 비용 우위를 주장하지 않는다.

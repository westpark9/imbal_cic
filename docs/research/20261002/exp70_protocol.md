# EXP70 · 최종 100,000행 context의 클래스 순차 도입

2026-10-02 사용자 승인 배분. Seed 43, Global TabPFN v3와 XGBoost 비교. EXP69의 동일 신규 클래스당 16/64/256행 조건에서 최종 100k 조건으로 규모를 확대한다. Expert/S/V는 포함하지 않는다.

## 고정 배분과 누적 순서

Benign 75,000행, 공격 합계 25,000행. 공격은 가용 수로 제한한 균등 배분이고 부족분을 나머지 공격에 재배분한다. 전체 clean train을 사용하며 이전 Global/Expert/Route 분할을 적용하지 않는다. 배분 원본: `tabpfn/configs/exp70_allocation.json`.

| CIC2018 단계 | 추가 사례 | 추가 수 | 누적 수 |
|---|---|---:|---:|
| 초기 | Benign 75,000 + Brute force 4,910 + DoS 4,910 | 84,820 | 84,820 |
| 1 | DDoS | 4,910 | 89,730 |
| 2 | Web attack | 450 | 90,180 |
| 3 | Infiltration | 4,910 | 95,090 |
| 4 | Bot | 4,910 | 100,000 |

| ToN 단계 | 추가 사례 | 추가 수 | 누적 수 |
|---|---|---:|---:|
| 초기 | Benign 75,000 + Scanning 2,856 + DoS 2,856 | 80,712 | 80,712 |
| 1 | Injection 2,856 + DDoS 2,856 | 5,712 | 86,424 |
| 2 | Password | 2,856 | 89,280 |
| 3 | XSS | 2,856 | 92,136 |
| 4 | Ransomware 2,151 + Backdoor 2,857 | 5,008 | 97,144 |
| 5 | MitM | 2,856 | 100,000 |

공격 최초 출현 순서는 원본 NF-v3 timestamp에서 검증한 EXP69와 같다. 전체 raw 시간 흐름을 재생하는 실험은 아니다. 이미 도입한 클래스 사례를 유지하고 미래 클래스는 도입 단계 전 fit/전처리/context에서 제외한다.

## 표본과 평가

- EXP69에 저장한 클래스별 sample IDs를 유지하고, 부족한 수만 전체 clean train에서 seed 43으로 비복원 추가 추출한다. 따라서 EXP69의 16/64/256 조건에서 사용한 사례가 모두 포함된다. 최종 클래스별 개수와 실제 train 소속을 검증한다.
- 두 방법에 매 단계 동일 train/test IDs를 제공한다. 기존 clean test에서 등장한 클래스의 전체 행을 평가하며 test를 재샘플링하지 않는다. Test는 이전에 관찰한 개발 holdout이다.
- Tail은 사용자가 정한 대로 **고정 test의 클래스별 표본 수**를 기준으로 해석한다. Context 클래스별 수는 별도 기록한다. 임의의 tail threshold나 test 성능에 따른 그룹 선정을 추가하지 않고, 모든 클래스의 고정 test support와 성능을 저장한다.
- Train/validation/test row ID 교집합을 검사한다. 기존 split의 동일 라벨 feature 중복은 보존하고, context와 같은 feature hash의 test 행 수 및 이를 제외한 부가 지표를 함께 저장한다.
- 최종 샘플 수뿐 아니라 Benign 및 공격별 context 구성이 함께 바뀐다. EXP69와의 차이를 특정 한 클래스의 샘플 증가만으로 인과 해석하지 않는다.

## 모델과 비용

EXP69와 동일 checkpoint 및 설정: TabPFN ensemble 4, `fit_with_cache`, `SUBSAMPLE_SAMPLES=None`, cache CPU 보관; XGBoost 300 trees, depth 8, learning rate .05, CUDA hist. XGB는 매 단계 전체 누적 사례로 새로 학습하고 TabPFN은 frozen backbone에 누적 context로 전처리/cache를 다시 구성한다.

GPU worker는 한 번에 하나씩 실행한다. 단계별 fit/갱신 시간, 배치 추론 시간, 전체 test 추론 시간, RSS/GPU 메모리를 기록한다. Cold fit과 warm fit을 구분하고, 첫 추론 지연도 저장한다. 같은 고정 평가 집합에서 이전 단계 예측과 비교할 수 있도록 확률·예측·test IDs를 보존한다.

기존 공용 실행기 `scripts/exp69_class_arrival.py`에 명시적 배분 기능을 추가하고, EXP70 폴더에 실행 시점의 코드를 snapshot한다. 이전 EXP69 결과 및 snapshot은 수정하지 않는다. 이 실행에서 `budget=100000`은 **최종 context 합계**이며 신규 클래스당 라벨 수가 아니다. 결과의 `allocation_mode=fixed_final_bank`, `labels_per_new_class=null`, `training_class_counts`로 구분한다.

## 실행과 기록

결과: `tabpfn/results/20261002_exp70_class_arrival_100k_s43`. 연구 버전 v7에 별도 실행으로 등록. `status.json`, stage별 `progress.json`, `COMPLETE.json`으로 진행을 확인한다. 단계 완료 후 저장하며 완료된 단계에서 재개할 수 있다. 준비한 입력·배분·코드·checkpoint SHA를 보존한다.

실험 실행 요청에 따라 HTML/PPT는 자동 수정하지 않는다. ETA는 최초 실제 GPU 배치 처리량으로 산출하고 실행 기록에 남긴다.

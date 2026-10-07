# EXP63 — 정제 데이터에서 K=2·4·6·8 비교

- 데이터: CIC2018, ToN-IoT, 기존 conflict-free split. Seed 43 단일 실행.
- 변경 요인: residual cluster/expert 수 K. K=4는 EXP62의 저장된 fitted bank·예측·S/V 결과를 그대로 재사용한다.
- 고정: Global fitted state와 100,000행 context, 공통 anchor ID, expert/route/validation/test ID, residual 표현의 scaling·temperature, expert block 선택 규칙, TabPFN 설정, S1V0 학습 및 validation 임계값 선택 규칙.
- Context: 공통 anchor + residual 군집 내 diversity 선택 block. 오류 샘플만 추출하는 방식이 아니다. 별도의 상대사례를 추가하지 않는다.
- Anchor 실제 크기: CIC2018 12,224행, ToN 18,717행. Expert별 특화 block 상한은 기존 186,000행이며 군집이 작으면 해당 군집 전체를 사용한다. 따라서 K 증가에 따라 총 context 행 수와 비용도 달라질 수 있다. 이는 고정 총예산 비교가 아니다.

## 실행 및 재사용

`python scripts/run_exp63_local.py`

결과 디렉터리: `tabpfn/results/20260930_exp63_k_sweep_s43`.

두 worker가 각각 하나의 dataset/K 조합을 처리한다. K=4 캐시 집계 검증 후 K=2 → K=6 → K=8 순서로 실행한다. 같은 GPU에서 최대 두 작업을 병렬 처리하고 OOM은 다른 작업 종료 후 단독 재시도한다. 실패와 진행 상태는 `status.json`, 각 작업의 `progress.json`, `attempt_*.log`에 기록한다. 재실행 시 완료된 작업과 확정된 캐시를 재사용한다.

Global 확률·embedding·PCA·split은 EXP62 캐시를 그대로 연결한다. 새 K의 train residual은 Global fitted state와 train ID 해시가 일치하는 저장 표현을 재사용한다. 해당 표현이 저장된 이전 실험의 context/개입/예측 결과는 사용하지 않는다. K=2·6·8 expert는 새로 fit한 상태를 저장하며 같은 fitted model로 route/cal/test를 모두 예측한다. 실행 당시 Python 코드는 결과의 `source/`에 고정한다.

## 출력과 해석

1. `summary.csv`, `class_metrics.csv`: Global / observable 고정 배정 / S+V의 전체·클래스 성능, 호출·수락·교정·훼손.
2. `expert_summary.csv`, `expert_class_metrics.csv`: 각 expert가 동일한 전체 test와 동일한 validation select/confirm 표본에 내린 raw multiclass 예측. P/R/F1, TP/FP/FN, Global 대비 변화, 교정·훼손 및 예측 집중도. 다른 클래스에서 발생한 FP도 포함한다.
3. 각 작업 `contexts/composition.csv`: expert별 anchor/block/합계 클래스 분포와 block의 Global 정답 샘플 수. K=4의 원본 context 구성은 EXP59에 보존되어 있다.
4. `diagnostics/residual_oracle_*`: test 정답이 들어가는 residual로 expert를 배정한 조건부 진단. Expert 입력에는 정답을 주지 않는다. 배포 성능이나 expert 자체 역량의 독립적 증거로 해석하지 않는다.
5. `cost.json`, `resource_samples.csv`: 준비·추론·정책 학습 시간, CPU RAM/GPU VRAM. K=4 원 실행 비용은 `source_cost.json`을 참조한다. 병렬 실행 간 경합을 포함하므로 독립적인 배포 지연 측정이 아니다.

S/V threshold는 기존과 동일하게 validation에서 선택한다. Validation confirm은 선택에 쓰지 않고 별도 확인에 사용한다. Test는 이전에 관찰한 development holdout이다. Validation과 test의 클래스 분포가 다르므로 raw precision/F1의 절대값을 그대로 일반화 격차로 해석하지 않는다. Test 결과를 보고 K나 expert를 고르는 절차는 구현하지 않았다.

## 실행 전 검증

- 세 실행 파일 `py_compile` 통과.
- 2·7·10개 클래스에서 무작위/단일 클래스 출력/정답 출력/Global 동일 출력의 12개 사례: P/R/F1/support/accuracy가 sklearn 집계와 일치.
- 교정−훼손 = expert 정답 수−Global 정답 수 항등식 확인.
- 두 데이터의 K=4 실제 캐시 집계 완료. 기존 standalone expert 지표와 일치.
- 모든 K에서 validation select/confirm ID가 기존 EXP62와 동일한지 확인하도록 구현.

초기 ETA: 2026-09-30 14:47 KST 시작 기준 약 2–4시간. K=2 신규 expert 추론 처리량을 바탕으로 재평가한다. HTML·슬라이드는 이번 실행 요청으로 수정하지 않았다.

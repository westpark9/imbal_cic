# EXP69 실행 기록

- 결과 폴더: `tabpfn/results/20261002_exp69_class_arrival_s43`
- 버전: v7 `class_incremental_arrival`, seed 43
- 설계: [exp69_protocol.md](exp69_protocol.md)
- 실행 snapshot: 결과 폴더의 `source/scripts/exp69_class_arrival.py`, `exp61_utils.py`
- 입력 준비 완료: 2026-10-02 03:04:39 KST. 원본 SHA가 clean split 출처와 일치. 준비 28.3초.
- 원본 최초 출현 순서 재검증 완료. 준비한 train/test ID 교집합 0건. 모델 출력 class ID 재매핑, 미래 클래스 제외, 예산 포함관계, Benign 오탐 정의에 대한 unit test 4개 통과.
- 초기 GPU 실측: RTX4090, CIC2018 14,000행 context, 4 estimators. Fit 1.46초, 초기 test 2,729,382행. 1,245,184행 처리 시 평균 약 19,482행/초. GPU 메모리 약 1.6GB.
- 잠정 ETA(03:05 KST 기준): 신규 공격당 64개 조건, CIC2018+ToN+두 방법 25–40분. 16·256개 조건 포함 전체 60–90분. CIC 첫 단계에서 외삽했으므로 ToN/클래스 수/CPU 전처리에 따라 달라진다.
- 갱신 비용에 fit뿐 아니라 첫 추론의 지연도 함께 확인해야 한다. 저장된 `batch_times`에 첫 배치 및 이후 배치 시간이 있어 지연 초기화와 정상 처리량을 구분할 수 있다. 전체 학습+추론 비용은 둘 모두 포함한다.

## 진행 확인

```bash
cat tabpfn/results/20261002_exp69_class_arrival_s43/status.json
tail -f tabpfn/results/20261002_exp69_class_arrival_s43/controller.log
tail -f tabpfn/results/20261002_exp69_class_arrival_s43/logs/cic2018_tabpfn_n64.log
```

Stage별 `progress.json`이 완료 행 수와 ETA를 갱신한다. 각 stage 완료 후 `COMPLETE.json`·확률·지표·비용을 남긴다. 데이터셋/방법의 전체 trajectory를 완료할 때 `summary.csv`와 `class_metrics.csv`를 합친다. 종료 시 전체 `COMPLETE.json`이 생긴다. GPU idle만으로 종료를 판정하지 말고 상태와 로그를 확인한다.

## A100 실행

현재는 로컬 실행을 유지한다. 원격 계정에 업로드하거나 Git push하지 않았다. A100으로 바꾸기로 하면 동일 입력을 재사용할 수 있다. 서버에 checkpoint가 있으므로 원본 14.7GB pickle을 다시 읽을 필요는 없다.

1. 결과 폴더의 `source/`, `data/`, `prepared.json`, `protocol.md`를 서버의 새 실행 폴더로 복사한다. `data/`에는 모델 학습에 사용하는 공통 사례와 전체 test가 모두 포함된다.
2. 비용을 동일 GPU에서 비교하려면 A100 전용 폴더에서 두 방법을 함께 다시 실행한다. 로컬 완료 결과는 보존한다. 단계 결과를 일부 이어받을 경우 `identity.json`의 GPU 정보로 비용을 구분해야 한다.
3. 로컬 package versions: Python 3.13, torch 2.11.0, tabpfn 8.2.0, xgboost 3.2.0, numpy 2.4.4, scikit-learn 1.8.0. 서버의 지원 CUDA 환경과 checkpoint 호환성을 먼저 확인한다. 준비된 입력 SHA와 checkpoint SHA를 실행 시 검증한다.

```bash
# A100: 아래 두 경로를 해당 서버 경로로 지정
EXP69_RUN=/path/to/20261002_exp69_class_arrival_s43_a100
EXP69_CHECKPOINT=/path/to/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt
python "$EXP69_RUN/source/scripts/exp69_class_arrival.py" run \
  --out "$EXP69_RUN" --checkpoint "$EXP69_CHECKPOINT" \
  --seed 43 --budgets 64 16 256 --gpu 0 --threads 8 --batch 32768
```

로컬을 멈출 때는 `RUN/STOP_AFTER_STAGE`를 만들면 진행 중인 stage 완료 후 멈춘다. 프로세스 종료를 확인하고 재개할 때 이 파일만 제거한다. 실행 중인 프로세스와 동일 폴더에서 두 번째 controller를 동시에 실행하지 않는다.

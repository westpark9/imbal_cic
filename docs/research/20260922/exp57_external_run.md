# EXP57 외부 GPU 실행 — 기존 PKL 직접 사용

외부 배정은 **CIC2018·ToN seed 42**다. 로컬은 seed 43·44를 계속 실행한다. 데이터·seed 하나당 global 1개와 K=4 bank 세 개를 fit한다. 세 bank는 현재 선택 방식, 클래스별 행 수를 맞춘 무작위 선택, 같은 크기의 균형 선택이다. 각 seed 안에서 global·anchor·영역 정의·expert별 context 크기를 공유한다. Scorer/verifier는 이 생성 품질 실험에서 다시 학습하지 않는다.

## 필요한 입력

`exp57_inputs_*.tar.zst`는 필수가 아니다. 외부에 있는 아래 두 파일을 직접 지정한다.

- `--data`: 기존 `nfv3_energy_suite_uncapped_scenarios.pkl` 경로.
- `--model-path`: 기존 `tabpfn-v3-classifier-v3_20260417_multiclass.ckpt` 경로. 저장소의 `tabpfn/` 아래에 있으면 생략할 수 있다.

모순 제거 split이 없으면 원본 PKL에서 자동으로 만든다. 같은 feature 벡터에 서로 다른 모델 클래스가 붙은 모든 행을 제거하고, 같은 라벨의 중복과 원래 train/validation/test 소속은 유지한다. 기본 저장 위치는 `tabpfn/results/exp57_clean_splits/`이며 `--clean-root`로 바꿀 수 있다. 다음 실행에서는 저장된 split을 재사용한다.

생성한 split은 단순히 행 수만 맞추는 것이 아니다. Git의 `tabpfn/configs/exp57_clean_split_reference.json`에 고정한 원본 PKL SHA-256, 클래스 정의, 정제 규칙, train/validation/test 인덱스 SHA-256과 대조한다. 기존 EXP47·48과 다른 데이터나 분할이면 학습 전에 중단한다. 준비는 별도 CPU 프로세스에서 실행해 데이터 정제용 메모리를 학습 전에 반환한다.

## 실행

Python 3.11 이상과 해당 장비 드라이버에서 작동하는 CUDA PyTorch를 사용한다. 로컬 참조 환경은 Python 3.13, torch 2.11.0+cu130, RTX 4090 24 GB, host RAM 128 GB다. 실제 외부 환경은 `environment.json`에 기록된다. 원본 PKL 크기와 전처리·worker 메모리를 고려하면 host RAM 64 GB 이상을 권장한다. GPU compute capability 8.0 이상이 필요하며 CPU로 자동 전환하지 않는다.

업데이트한 저장소 루트에서, 아래 두 입력 경로를 외부 파일 경로로 바꿔 실행한다.

```bash
git pull --ff-only
python -m pip install -r requirements-exp57.txt

python scripts/run_exp57_external.py \
  --data /workspace/data/nfv3_energy_suite_uncapped_scenarios.pkl \
  --model-path /workspace/models/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt \
  --datasets cic2018 toniot \
  --seeds 42 \
  --gpu 0 \
  --root tabpfn/results/exp57_external_s42 \
  --detach
```

`--detach`이면 원본 검증·split 생성·학습 전체가 백그라운드에서 이어진다. 별도의 archive 전송이나 압축 해제가 필요 없다. 기본 예측 배치는 125,000, CPU 스레드는 1이다. 로컬의 일부 500,000 배치 실행에서 native segmentation fault가 발생해 복구 설정으로 낮췄으며, 오류 해결이 검증됐다는 뜻은 아니다. faulthandler로 native stack trace를 기록한다.

한 GPU에서 두 데이터는 순차 실행한다. GPU가 두 개라면 위 명령 대신 데이터별로 나눌 수 있다.

```bash
python scripts/run_exp57_external.py --data /workspace/data/nfv3_energy_suite_uncapped_scenarios.pkl --model-path /workspace/models/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt --datasets cic2018 --seeds 42 --gpu 0 --root tabpfn/results/exp57_external_cic_s42 --detach
python scripts/run_exp57_external.py --data /workspace/data/nfv3_energy_suite_uncapped_scenarios.pkl --model-path /workspace/models/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt --datasets toniot --seeds 42 --gpu 1 --root tabpfn/results/exp57_external_ton_s42 --detach
```

두 프로세스를 동시에 실행하면 host RAM도 두 작업의 데이터 준비·학습을 수용해야 한다. 같은 데이터의 split을 동시에 준비할 경우 잠금으로 중복 생성을 막는다. 검증에 성공한 완성 디렉터리만 재사용 대상으로 공개한다.

이미 입력 묶음을 풀어 놓았다면 예전 `--inputs /workspace/exp57_inputs` 방식도 계속 지원한다. 이미 생성한 clean 디렉터리가 따로 있다면 `--clean-root /path/to/derived`를 추가하면 된다.

## 진행 확인과 결과 반환

단일 GPU 실행 기준:

```bash
cat tabpfn/results/exp57_external_s42/status.json
tail -F tabpfn/results/exp57_external_s42/controller.log
```

처음에는 `verifying_source` 또는 `preparing_clean` 상태가 나타나며 준비가 끝나면 `running`으로 바뀐다. 학습 상세 로그는 `cic2018_s42/worker.log`, `toniot_s42/worker.log`다. `clean_input_verification.json`에는 각 데이터의 split 생성·재사용 여부와 검증된 해시를 저장한다.

완료 시 `status.json`의 `state: complete`와 각 job의 `COMPLETE.json`을 확인한다. `input_preparation_failed`이면 원본/split 검증 오류를, `complete_with_errors`이면 실패 job의 `worker.log`를 전달한다. 재시도는 새로운 `--root`를 사용하며 기존 clean split은 그대로 재사용할 수 있다. `--prepare-only`는 입력 준비·검증과 job 설정까지만 수행한다.

완료 후 큰 확률 배열을 제외한 결과를 묶는다.

```bash
tar --zstd --exclude='*/probabilities' \
  -cf /workspace/exp57_external_s42_results.tar.zst \
  -C tabpfn/results exp57_external_s42
```

이 결과 묶음을 반환하면 로컬 seed 43·44와 합쳐 분석할 수 있다. 확률 배열은 추가 재분석을 위해 외부 결과 디렉터리에 보존한다. 입력 split을 다시 만들더라도 동일한 개발 holdout을 재현하는 것이므로 새 독립 test를 확보한 실험으로 해석하지 않는다.

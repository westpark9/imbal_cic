# EXP57 외부 GPU 실행 — 기존 PKL 직접 사용

외부 배정은 **CIC2018·ToN seed 42**다. 로컬은 seed 43·44를 계속 실행한다. 데이터·seed 하나당 global 1개와 K=4 bank 세 개를 fit한다. 세 bank는 현재 선택 방식, 클래스별 행 수를 맞춘 무작위 선택, 같은 크기의 균형 선택이다. 각 seed 안에서 global·anchor·영역 정의·expert별 context 크기를 공유한다. Scorer/verifier는 이 생성 품질 실험에서 다시 학습하지 않는다.

## 필요한 입력

`exp57_inputs_*.tar.zst`는 필수가 아니다. 외부에 있는 아래 두 파일을 직접 지정한다.

- `--data`: 기존 `nfv3_energy_suite_uncapped_scenarios.pkl` 경로.
- `--model-path`: 기존 `tabpfn-v3-classifier-v3_20260417_multiclass.ckpt` 경로. 저장소의 `tabpfn/` 아래에 있으면 생략할 수 있다.

모순 제거 split이 없으면 원본 PKL에서 자동으로 만든다. 같은 feature 벡터에 서로 다른 모델 클래스가 붙은 모든 행을 제거하고, 같은 라벨의 중복과 원래 train/validation/test 소속은 유지한다. 기본 저장 위치는 `tabpfn/results/exp57_clean_splits/`이며 `--clean-root`로 바꿀 수 있다. 다음 실행에서는 저장된 split을 재사용한다.

생성한 split은 단순히 행 수만 맞추는 것이 아니다. Git의 `tabpfn/configs/exp57_clean_split_reference.json`에 고정한 원본 PKL SHA-256, 클래스 정의, 정제 규칙, train/validation/test 인덱스 SHA-256과 대조한다. 기존 EXP47·48과 다른 데이터나 분할이면 학습 전에 중단한다. 준비는 별도 CPU 프로세스에서 실행해 데이터 정제용 메모리를 학습 전에 반환한다.

## 마운트에 실행 환경 저장

`pip install -r requirements-exp57.txt`만으로는 장비 드라이버와 PyTorch의 CUDA 빌드가 맞는다고 보장되지 않는다. **Conda 환경 자체를 마운트 경로에 만들고, 드라이버에 맞는 PyTorch wheel을 먼저 고정한다.** Conda를 새로 만드는 것만으로 호환 문제가 해결되는 것은 아니다.

업데이트한 저장소 루트에서 다음을 실행한다. `/workspace`는 예시이므로 실제 영구 마운트 경로로 바꾼다. `--prefix`는 기존 base 환경이 아닌 새 전용 경로여야 한다. 첫 생성 때는 Conda 또는 Miniforge가 필요하다.

```bash
python scripts/setup_exp57_env.py --prefix /workspace/envs/exp57 --gpu 0
```

스크립트는 Python 3.12 환경을 만들고 `nvidia-smi`에 표시되는 **드라이버의 CUDA 지원 버전**에 따라 아래 중 하나를 선택한다. 시스템에 설치된 `nvcc` 버전과 같아야 한다는 뜻은 아니다. CUDA minor compatibility에 기대지 않는 보수적인 선택이며, 마지막에 GPU 연산으로 확인한다.

| 드라이버가 표시하는 CUDA 지원 버전 | 설치할 PyTorch |
|---|---|
| 12.8 이상 | 2.7.1+cu128 |
| 12.4 이상, 12.8 미만 | 2.6.0+cu124 |
| 12.1 이상, 12.4 미만 | 2.5.1+cu121 |
| 11.8 이상, 12.1 미만 | 2.6.0+cu118 |

공식 근거: [PyTorch wheel 설치 조합](https://pytorch.org/get-started/previous-versions/), [NVIDIA 드라이버 호환 규칙](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [Conda prefix 환경](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Blackwell은 CUDA 12.8 이상 조합을 요구한다. GPU compute capability 8.0 미만은 이 실험에서 지원하지 않는다.

의존성을 설치할 때도 선택한 torch 버전을 constraint로 고정한다. 설치 후 `pip check`, CUDA 초기화, 실제 Flash Attention 연산, 주요 패키지 import를 확인한다. 설치 정보와 전체 패키지 목록은 환경 안의 `exp57_setup.json`, `exp57_packages.txt`에 저장한다. 먼저 선택만 확인하려면 `--dry-run`을 추가한다.

같은 경로가 마운트되고 호환되는 OS·장비라면 재접속 후 라이브러리를 다시 설치할 필요 없이 **저장한 환경의 Python**을 사용한다. 활성화나 base Conda 재설치도 필요 없다. 새 노드에서 호환성을 재확인하려면 다음 명령을 실행한다. 설정이 같으면 패키지 설치는 생략하고 GPU 검증만 반복한다.

```bash
/workspace/envs/exp57/bin/python scripts/setup_exp57_env.py --prefix /workspace/envs/exp57 --gpu 0
```

다른 드라이버 때문에 선택되는 torch 빌드가 바뀌면 기존 환경을 덮어쓰지 않고 중단한다. 그 경우 `/workspace/envs/exp57-cu124`처럼 새 `--prefix`를 지정한다. 설치 캐시도 마운트의 `envs/.exp57-cache/`에 보존한다. `conda create -n ...`만 사용하면 환경이 임시 홈 디렉터리에 생성되어 재접속 때 사라질 수 있다.

## 실행

로컬 참조 환경은 Python 3.13, torch 2.11.0+cu130, RTX 4090 24 GB, host RAM 128 GB다. 외부는 드라이버에 맞춘 위 조합을 사용하며 실제 버전은 각 실행의 `environment.json`에 기록된다. 서로 다른 torch/GPU의 수치 재현성이 검증됐다는 뜻은 아니므로 결과를 합칠 때 환경 차이도 확인한다. 원본 PKL 크기와 전처리·worker 메모리를 고려하면 host RAM 64 GB 이상을 권장한다. CPU로 자동 전환하지 않는다.

아래 환경·입력 경로를 외부 파일 경로로 바꿔 실행한다. 재접속할 때도 `EXP57_PY`를 다시 지정한다.

```bash
EXP57_PY=/workspace/envs/exp57/bin/python

"$EXP57_PY" scripts/run_exp57_external.py \
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
"$EXP57_PY" scripts/run_exp57_external.py --data /workspace/data/nfv3_energy_suite_uncapped_scenarios.pkl --model-path /workspace/models/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt --datasets cic2018 --seeds 42 --gpu 0 --root tabpfn/results/exp57_external_cic_s42 --detach
"$EXP57_PY" scripts/run_exp57_external.py --data /workspace/data/nfv3_energy_suite_uncapped_scenarios.pkl --model-path /workspace/models/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt --datasets toniot --seeds 42 --gpu 1 --root tabpfn/results/exp57_external_ton_s42 --detach
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

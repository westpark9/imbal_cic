# EXP57 외부 GPU 실행

외부 배정은 **CIC2018·ToN seed 42**다. 로컬은 seed 43·44를 계속 실행한다. 데이터·seed 하나당 global 1개와 K=4 bank 세 개를 fit한다. 세 bank는 현재 선택 방식, 클래스별 행 수를 맞춘 무작위 선택, 같은 크기의 균형 선택이다. 각 seed 안에서 global·anchor·영역 정의·expert별 context 크기를 공유한다. Scorer/verifier는 이 생성 품질 실험에서 다시 학습하지 않는다.

## 입력 파일

Git에는 대용량 원본 데이터와 모델 가중치가 포함되지 않는다. 로컬에서 준비한 아래 묶음을 외부 장비로 옮긴다.

```
data/transfer/exp57_inputs_20260922.tar.zst
data/transfer/exp57_inputs_20260922.tar.zst.sha256
```

압축 크기는 약 1.17 GiB, 해제된 입력은 약 14.1 GiB다. 원본 PKL, TabPFN-v3 checkpoint, CIC·ToN의 모순 제거 고정 split 세 개와 manifest가 들어 있다. 자세한 파일 목록과 SHA-256은 같은 디렉터리의 `exp57_input_bundle.json`에 기록했다. 별도의 모순 제거 재처리는 필요 없다.

전송 예시(로컬에서; `GPU_HOST`는 실제 SSH 접속 대상으로 바꾼다):

```bash
scp data/transfer/exp57_inputs_20260922.tar.zst* GPU_HOST:/workspace/
```

## 외부 환경과 실행

Python 3.11 이상과 해당 장비 드라이버에서 작동하는 CUDA PyTorch를 사용한다. 로컬 참조 환경은 Python 3.13, torch 2.11.0+cu130, RTX 4090 24 GB, host RAM 128 GB다. 라이브러리 버전은 실행 결과의 `environment.json`에 저장한다. 입력 크기와 로컬 worker RSS를 고려하면 host RAM 64 GB 이상을 권장한다. GPU compute capability 8.0 이상이 필요하며, 실행기는 CUDA가 없을 때 CPU로 자동 전환하지 않는다.

아래는 `/workspace`에 입력 묶음을 전송한 경우다. 이미 저장소가 있다면 clone 대신 그 저장소에서 `git pull --ff-only`를 실행한다.

```bash
cd /workspace
sha256sum -c exp57_inputs_20260922.tar.zst.sha256
mkdir -p exp57_inputs
tar --zstd -xf exp57_inputs_20260922.tar.zst -C exp57_inputs

git clone https://github.com/westpark9/imbal_cic.git
cd imbal_cic
python -m pip install -r requirements-exp57.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'

python scripts/run_exp57_external.py \
  --inputs /workspace/exp57_inputs \
  --datasets cic2018 toniot \
  --seeds 42 \
  --gpu 0 \
  --root tabpfn/results/exp57_external_s42 \
  --detach
```

`tar --zstd`에는 `zstd` 실행 파일이 필요하다. 기본 예측 배치는 125,000, CPU 스레드는 1이다. 로컬의 일부 500,000 배치 실행에서 native segmentation fault가 발생해 이 설정으로 복구 실험을 준비했다. 이는 원인 확정이나 오류 해결 완료를 뜻하지 않는다. Native stack trace가 로그에 남도록 faulthandler를 활성화했다.

입력 경로는 머신마다 달라도 된다. 원본 PKL의 SHA-256과 split 파일의 SHA-256을 검증하고, 기존 manifest를 수정하지 않는다. 선택한 job 목록과 실행 소스 사본은 실행 전에 고정된다. 소스 수정이 진행 중인 worker에 섞이지 않는다.

한 GPU에서는 두 데이터가 순차 실행된다. GPU가 두 개면 아래처럼 나눌 수 있다.

```bash
python scripts/run_exp57_external.py --inputs /workspace/exp57_inputs --datasets cic2018 --seeds 42 --gpu 0 --root tabpfn/results/exp57_external_cic_s42 --detach
python scripts/run_exp57_external.py --inputs /workspace/exp57_inputs --datasets toniot --seeds 42 --gpu 1 --root tabpfn/results/exp57_external_ton_s42 --detach
```

두 GPU 예시는 위 단일 GPU 명령의 대안이다. 동시에 모두 실행하지 않는다. 두 worker가 동시에 실행되면 host RAM도 두 작업을 수용해야 한다.

## 진행 확인과 결과 반환

단일 GPU 실행 기준:

```bash
cat tabpfn/results/exp57_external_s42/status.json
tail -f tabpfn/results/exp57_external_s42/cic2018_s42/worker.log
```

`status.json`에 현재 job, PID, 완료·실패 목록, 로그 경로가 기록된다. `state: complete`와 각 job의 `COMPLETE.json`을 확인한다. `complete_with_errors`이면 실패한 job의 `worker.log`를 함께 전달한다. 같은 출력 디렉터리에서 실행기를 다시 호출하면 기존 결과를 덮어쓰지 않고 실패한다. 재시도는 새로운 `--root`를 사용한다.

완료 후, 큰 확률 배열을 제외한 결과를 묶는다. 예측 배열·평가 행 정보·context ID·로그·CSV·JSON·소스 사본은 포함된다.

```bash
tar --zstd --exclude='*/probabilities' \
  -cf /workspace/exp57_external_s42_results.tar.zst \
  -C tabpfn/results exp57_external_s42
```

`exp57_external_s42_results.tar.zst`를 반환하면 로컬 seed 43·44와 합쳐 분석할 수 있다. 원래 확률 배열은 추가 재분석을 위해 외부 결과 디렉터리에 보존한다. 외부 실행은 기존 개발 holdout을 사용하며, 새 독립 test를 확보한 실험으로 해석하지 않는다.

## 검증 범위

동일 클래스별 행 수, 균형 context 예산, validation 선택·tie 처리, paired bootstrap, 경로 이전 시 content hash와 split 무결성 검사를 테스트했다. 외부 실행 명령의 prepare 경로도 로컬에서 확인했다. 실제 외부 GPU에서 전체 실행이 끝난 상태는 아니다.

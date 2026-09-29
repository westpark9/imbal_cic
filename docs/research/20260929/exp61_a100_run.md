# 정제 데이터 SOTA 및 비용 비교: A100 실행

이번 우선순위는 CIC2018·ToN의 모순 제거 데이터에서 XGBoost, BoostPFN, LoCalPFN, DistPFN을 측정하는 것이다. EXP60의 무작위/가까운 상대사례 결과는 모델 개선의 근거와 발표 본문에서 제외한다. 원시 결과는 재현 기록으로 보존한다. 이 실행은 HTML을 자동 편집하지 않는다.

## 실행

저장소에서 최신 코드를 받은 뒤, **원본 PKL과 v3 checkpoint의 실제 경로**를 지정한다. 두 파일은 Git에 포함되지 않는다. 이전 EXP57 환경과 정제 split을 재사용할 수 있다.

```bash
git pull --ff-only origin main

# 처음 한 번. 기존에 이 경로로 만든 EXP57 환경이면 그대로 확장한다.
python scripts/setup_exp61_env.py --prefix /workspace/envs/exp57 --gpu 0

/workspace/envs/exp57/bin/python scripts/run_exp61_sota.py \
  --data /workspace/data/nfv3_energy_suite_uncapped_scenarios.pkl \
  --model-path /workspace/models/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt \
  --datasets cic2018 toniot \
  --gpu 0 --cpu-threads 8 \
  --root /workspace/results/exp61_sota_clean_s43 \
  --detach
```

`/workspace`는 예시다. 환경·결과는 서버의 영구 마운트 경로를 사용한다. 이미 다른 경로의 EXP57 환경을 만들었다면 그 prefix로 setup하고 해당 `bin/python`을 사용한다. 원본 PKL과 v3 checkpoint가 저장소의 `data/`, `tabpfn/`에 있다면 각각 그 경로를 지정한다. 원본 데이터의 SHA-256과 v3 checkpoint의 SHA-256이 기준과 다르면 실행 전에 중단한다.

Setup은 드라이버에 맞는 CUDA PyTorch를 고정한 기존 EXP57 설치기를 재사용하고, FAISS·TensorBoard·자원 계측 의존성을 추가한다. BoostPFN·LoCalPFN의 v1 checkpoint는 공식 파일을 받아 해시 검증한다. 기존 환경에 의존성을 추가할 때 CUDA torch가 교체되지 않도록 constraint를 적용한다.

정제 split이 없는 외부 서버에서도 원본 PKL로 자동 생성한다. **정제 후 재분할하지 않으며**, Git에 고정된 train/validation/test ID의 SHA-256과 일치해야 진행한다. 기존 정제 경로를 지정하려면 `--clean-root /path/to/derived`를 추가한다. 기본 경로는 저장소의 `tabpfn/results/exp57_clean_splits`다.

기본값은 단일 seed 43, 단일 GPU 순차 실행이다. `--datasets cic2018` 또는 `--methods localpfn`으로 작업을 나눌 수도 있지만 각각 다른 `--root`를 사용한다. A100의 VRAM만으로 속도 향상 배수를 가정하지 않는다. CPU의 정확한 kNN 검색도 LoCalPFN 시간에 포함된다.

## 비교 조건

| 방법 | 학습/검색에 제공하는 행 | 설정 |
|---|---|---|
| Global TabPFN | 기존 seed 43 Global의 동일 100,000개 ID | v3 checkpoint, 4 estimators, fit_with_cache |
| XGBoost | 같은 100,000개 ID | CUDA hist, 300 trees, depth 8, lr 0.05 |
| BoostPFN | 같은 100,000개 ID | vendored v1, V2, CE/exphadamard, 50 rounds × 500개 context, 내부 ensemble 1 |
| LoCalPFN | 같은 100,000개 ID를 retrieval/FT 풀로 사용 | v1 fine-tuning 포함, context 1,000, 21 epochs × 30 steps, validation AUC로 checkpoint 선택 |
| DistPFN | 위 Global의 posterior 및 context prior | 전체 unlabeled test posterior 평균으로 prior 보정, DistPFN-T 제외 |
| XGBoost full | 정제된 train 전체 | 동일 XGBoost 설정, 전체 데이터 학습 기준선 |

같은 100k 행을 쓰는 표는 **context 예산을 맞춘 비교**다. 모든 PFN에 전체 train을 사용한 결과가 아니다. 제안 시스템은 expert·route 라벨도 사용하므로 이 표만으로 전체 시스템의 정보 예산이 같다고 주장하지 않는다. XGBoost full의 train 수와 비용을 별도로 제시하고, 향후 제안 모델과 비교할 때 접근 라벨 수·expert context·route 학습비도 함께 명시한다.

기존 Global context ID는 작은 압축 파일로 Git에 포함했다. 임의로 다시 샘플링하지 않는다. Global은 A100에서 재실행하므로 GPU/라이브러리/배치 차이에 따른 수치 차이는 생길 수 있다. 4090의 과거 시간을 A100 표에 혼합하지 않는다.

Test는 클래스별 cap이나 중복 제거 추론 없이 전 행 평가한다. CIC2018 **3,042,473행**, ToN **2,271,723행**이다. LoCalPFN의 validation만 클래스당 최대 5,000행으로 고정하며 test로 checkpoint를 선택하지 않는다. 실제 validation ID도 결과에 저장한다. 이 test는 이미 관찰한 development holdout이며 새 독립 평가 데이터는 아니다.

## Cost에 포함되는 항목

- 모델별 fit 시간, 전체 test 추론 시간, rows/s와 1,000행 환산 시간. 환산 시간은 배치 처리량 기준이며 단건 온라인 latency가 아니다.
- LoCalPFN fit 시간에는 validation과 checkpoint 선택이 포함된다. validation 시간을 따로 기록하지만 fit에 다시 더하지 않는다. 최종 추론에는 kNN 검색과 index 구성도 포함된다.
- DistPFN은 prior 보정 시간만으로 비교하지 않는다. Global fit·추론 비용까지 포함한 값과 보정 추가 시간을 모두 저장한다. 두 행이 한 번의 backbone 계산을 공유함을 명시한다.
- BoostPFN의 원본 `predict_proba(X)`는 X에 대한 추론 없이 캐시된 test logits를 합친다. 비용 계측에서는 원본 fit에 빈 test를 넘겨 학습을 완료한 뒤, 저장된 sampled context·계수로 실제 test 추론을 수행한다. 클래스 누락 시 logit 0 처리도 동일하다. 작은 GPU 입력으로 원본 결합 실행과 sample ID·계수·확률 일치를 검증한다.
- 최대 GPU 메모리: 1초 간격 `nvidia-smi`의 해당 프로세스 사용량 및 PyTorch allocator peak를 함께 기록한다. 전자는 샘플 사이의 짧은 peak를 놓칠 수 있고, 후자는 XGBoost의 CUDA 할당을 포함하지 않는다.
- 최대 RAM: 로딩을 포함한 전체 peak와 모델 단계의 샘플링 peak를 구분한다. `resource_samples.csv`에서 단계별 사용량을 확인할 수 있다.
- PKL 로딩·정제·전처리 시간을 별도 기록한다. 실험 전체 wall time은 성공/실패한 attempt 기록과 controller의 경과 시간에서 확인한다.

모든 비용은 같은 하드웨어에서 GPU 작업을 겹치지 않고 측정한다. 다른 작업을 동시에 띄운 측정은 독립 실행과 구분해야 한다. 앞으로 제안 모델 비용에도 Global 준비, residual/군집/context 구성, expert 준비, S/V 학습 및 실제 호출·채택에 따른 test 추론을 같은 기준으로 포함한다.

## 진행 확인과 재개

```bash
cat /workspace/results/exp61_sota_clean_s43/status.json
tail -F /workspace/results/exp61_sota_clean_s43/controller.log
```

`status.json`에는 현재 작업, worker PID, 완료/실패 목록, 마지막 진행 정보가 있다. 각 작업의 `attempt_1.log`, `progress.json`과 `resource_samples.csv`도 갱신된다. 한 작업이 실패해도 다음 작업은 진행하고, 최종 상태에 실패를 남긴다. 원본 코드·설정·checkpoint 해시·패키지와 GPU 정보는 실행 폴더에 저장된다.

같은 명령과 같은 `--root`로 재개하면 완료 작업을 건너뛴다. LoCalPFN은 완료된 fine-tuning checkpoint와 50,000행 단위 추론 결과를 재사용한다. 학습 도중 중단되어 FIT_COMPLETE가 없으면 fine-tuning부터 다시 시작한다. 다른 설정으로 바꾸려면 새 root를 사용한다. 실행 폴더는 코드 스냅샷을 사용하므로 재개 중 Git 변경으로 구현이 바뀌지 않는다.

`--prepare-only`는 입력 검증 전용 root에 사용한다. 학습은 별도 root로 시작한다. `--synthetic-smoke`는 작은 생성 데이터에서 구현 실행만 확인하며, 그 결과를 IDS 성능표에 쓰지 않는다.

## 결과 회수

- `summary.csv`: 성능과 비용, 기본 설정은 데이터당 Global 포함 6행.
- `class_metrics.csv`: 클래스별 P/R/F1, TP/FP/FN과 support.
- 각 작업의 `evaluation_identity.npz`, 예측·확률, 학습 checkpoint/boost 상태, 비용 원시 로그.
- `environment.json`, `request.json`, `source_sha256.json`, `source/`: 재현 정보.

```bash
tar -czf /workspace/results/exp61_sota_clean_s43.tar.gz \
  -C /workspace/results exp61_sota_clean_s43
```

결과 폴더 전체를 가져오면 기존 Global·expert 결과와 ID를 대조하고, 성능–시간–메모리 비교표를 작성할 수 있다. 초반 작업의 실제 속도와 LoCalPFN의 validation/test 진행률을 본 뒤 완료 ETA를 갱신한다.

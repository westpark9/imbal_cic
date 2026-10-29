# EXP93: 재사용 가능한 CIC2018 PCAP corpus와 공정 few-shot 비교

이 실험은 PCAP에서 만든 corpus와 인코더 출력을 분리한다. `corpus/flows.jsonl`은 checksum이 기록된 capture, 양방향 5-tuple, flow 시작·종료 시각과 패킷 위치, 고정 split, 라벨과 라벨 규칙을 보존한다. `encoders/<adapter>/`는 같은 flow 순서에 대응하는 벡터만 저장한다. 따라서 NetMamba·ET-BERT 등 후속 인코더는 corpus를 다시 라벨링하거나 split하지 않고, 새 adapter directory와 `encoder.json`, `embeddings.npy`만 만들면 된다.

현재 구현한 adapter는 `netfound-v2-small` 하나다. 단일 실행은 다음 순서로 진행한다.

1. 공개 S3 ZIP의 중앙 디렉터리와 HTTP Range를 이용해 필요한 host-PCAP member만 추출하고 ZIP CRC·SHA256 receipt를 저장한다. 일자 ZIP 전체를 내려받지 않는다.
2. PCAP에서 독립적으로 양방향 flow를 재구성한다. 각 flow는 SHA256 기반 group split 뒤 label×split reservoir에 넣는다. 같은 capture의 동일 5-tuple은 part 파일이 달라도 같은 split을 갖는다.
3. corrected-CICFlowMeter 공개 노트북의 시간·endpoint·명시 filter를 PCAP flow에 적용한다. 원본 세부 공격명은 `annotation_label`에 남기고, classifier는 기존 NF-v3 비교와 같은 `benign, ddos, bot, dos, bf, inf, web` family를 사용한다. EC2 host PCAP가 공개 endpoint 대신 private VPC endpoint를 보이는 경우에는 protocol의 capture-scoped `endpoint_aliases`에 그 대응을 기록한다. 이는 CICFlowMeter 행을 재사용하거나 재현한 것이 아니다. rule 경계를 넘는 flow는 미확정으로 남는다.
4. frozen netFound v2 header/timing representation을 만들고, 모든 training-split flow만으로 PCA를 적합한다. `n`개의 라벨 예산과 PCA용 비라벨 train-pool은 결과에서 구별한다.
5. encoder 실행 전 `label_coverage.json`이 family별 train/test flow 수를 확인한다. 최대 label budget을 충족하지 못한 capture-to-rule 조합은 encoder와 classifier를 실행하지 않고 중단한다. `with_benign`과 `attack_only`을 각각 평가한다. XGBoost와 TabPFN은 seed·class·budget마다 정확히 같은 training flow ID와 test flow ID를 받는다. TabPFN 가중치는 해시로 동결 여부를 확인한다.

`corpus/manifest.json`의 `evaluation_label_schema`은 family와 그 family에 들어간 원본 annotation/rule ID를 기록한다. `audit.json`은 group 누수, vector 유한성, **netFound 입력 토큰이 완전히 같은데 라벨이 다른 flow** 및 PCA vector가 완전히 같은데 라벨이 다른 flow를 따로 집계한다. `ceiling_audit.json`은 n=20과 n=50의 평균 macro-F1 차이를 기술적으로 요약한다. 90% 이상이면서 변화량이 1%p 이하인 표시는 데이터 상한선의 증명이 아니라, 라벨·클래스 지지수·정상 포함 조건·다른 encoder view를 추가 검토할 신호다.

다른 encoder는 `<run>/encoders/<adapter>/embeddings.npy`에 corpus JSONL 행 순서와 같은 `float32[N, D]` 배열을 만들고, `encoder.json`에 `adapter`, `rows`, `corpus_sha256`, `embeddings_sha256`, backbone/weight hash와 pooling 정보를 기록한다. 그 뒤 다음 세 단계를 같은 run directory에서 수행한다. 이 경로는 이미 만든 labels, split, reservoir 및 few-shot flow ID를 그대로 쓴다.

```bash
PCAP_PYTHON=.envs/pcap-netfound/bin/python
RUN=tabpfn/results/v7/exp93/<completed-run-id>
"$PCAP_PYTHON" scripts/v7/exp93/run_pcap_benchmark.py \
  --config configs/experiments/exp93_protocol.json --output "$RUN" --corpus "$RUN/corpus" \
  --adapter <adapter> --stage project
"$PCAP_PYTHON" scripts/v7/exp93/run_pcap_benchmark.py \
  --config configs/experiments/exp93_protocol.json --output "$RUN" --corpus "$RUN/corpus" \
  --adapter <adapter> --stage audit
"$PCAP_PYTHON" scripts/v7/exp93/run_pcap_benchmark.py \
  --config configs/experiments/exp93_protocol.json --output "$RUN" --corpus "$RUN/corpus" \
  --adapter <adapter> --stage fewshot
```

## 원격 A100 실행

`PCAP_BASE_PYTHON`은 CUDA driver와 맞는 torch를 가진 Python이다. EXP91 환경이 그대로 있다면 다음처럼 실행한다.

```bash
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  PCAP_BASE_PYTHON=/home/jovyan/mspark/envs/exp91-cu124/bin/python \
  bash scripts/v7/exp93/run_remote.sh \
  > /tmp/exp93_pcap.log 2>&1 &
```

`tail -f /tmp/exp93_pcap.log`로 공개 PCAP 준비 구간을 확인하고, run ID가 출력된 뒤에는 `tabpfn/results/v7/exp93/<run_id>/stdout.log`를 본다. 기존 checksum receipt가 있는 PCAP은 다시 내려받지 않는다. 첫 실행은 PCAP 처리·coverage 검사·임베딩·120개 classifier job을 모두 수행한다. 라벨 규칙을 수정한 뒤에는 새 run directory를 만들어 corpus부터 재구성한다. 중단 후 같은 protocol과 corpus를 유지할 때만 해당 run directory에 대해 `run_pcap_benchmark.py --stage encode|project|audit|fewshot`를 직접 호출할 수 있다. 이미 완성된 `corpus/`에 다른 adapter를 붙일 때는 `--corpus <run>/corpus`를 쓴다.

PCAP, raw embedding과 NPZ는 Git에서 제외한다. `manifest.json`, `encoder.json`, `projection.json`, `audit.json`, per-job `result.json`, `summary.csv`, `ceiling_audit.json`은 수치와 재현 근거로 보존한다.

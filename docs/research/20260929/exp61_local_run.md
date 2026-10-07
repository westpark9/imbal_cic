# EXP61 로컬 실행 (2026-09-29)

사용자 요청에 따라 외부 A100 실행 대신 로컬 RTX 4090에서 seed 43으로 실행한다. 모순 제거 split, Global C0 100,000개 학습 ID, 전체 test 평가 기준은 `exp61_a100_run.md`와 같다. XGB full-train도 별도 비교한다. LoCalPFN은 fine-tuning을 포함한다.

결과: `tabpfn/results/20260929_exp61_sota_local_s43/`

```bash
python scripts/run_exp61_sota.py \
  --data /home/user/Desktop/imbal_cic/data/nfv3_energy_suite_uncapped_scenarios.pkl \
  --model-path /home/user/Desktop/imbal_cic/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt \
  --clean-root /home/user/Desktop/imbal_cic/data/derived \
  --root tabpfn/results/20260929_exp61_sota_local_s43 \
  --methods xgb xgb_full distpfn boostpfn localpfn \
  --cpu-threads 6 --max-parallel 2 \
  --pfn-batch 32768 --boost-batch 8192 --local-batch 256 --detach
```

독립 worker 2개가 GPU를 공유한다. CIC2018 LoCalPFN을 먼저 시작하고 다른 lane에서 XGB, DistPFN, BoostPFN, ToN LoCalPFN을 진행한다. GPU 메모리 부족으로 실패한 작업은 나머지 병렬 작업이 끝난 후 단독 재시도한다. LoCalPFN 최종 추론은 약 50,000행마다 저장되어 재개 가능하다. 실행은 소스 스냅샷을 사용하며 터미널 종료 후에도 지속된다.

추론 batch 조정은 GPU 메모리를 위한 실행 설정이다. 모델의 학습 context 크기, seed, 반복 수, 평가 행 수를 줄이지 않는다. 각 모델의 성능과 학습·validation·추론 시간, 프로세스별 GPU/RAM 사용량을 기록한다. 병렬 실행 시간에는 자원 경합이 포함되므로 단독 실행 latency로 인용하면 안 된다. 결과 JSON/CSV에 `execution_mode=concurrent_shared_gpu`와 `max_parallel=2`를 표시한다.

```bash
cat tabpfn/results/20260929_exp61_sota_local_s43/status.json
tail -n 20 tabpfn/results/20260929_exp61_sota_local_s43/controller.log
cat tabpfn/results/20260929_exp61_sota_local_s43/cic2018_localpfn/validation_progress.json
```

`status.json`에 현재 worker PID, 완료·대기·실패 작업이 갱신된다. 완료 모델은 `summary.csv`와 `class_metrics.csv`에 순차 반영된다. 실제 test 행 수는 CIC2018 3,042,473, ToN 2,271,723이다.

검증: 병렬 scheduler의 합성 데이터 XGB·DistPFN 4개 작업 완료, CPU protocol 검사 2개 통과. 이전에 검증한 BoostPFN 원본 예측 동등성과 LoCalPFN query-label 불변성은 변경하지 않았다. 이 실행 요청으로 HTML은 수정하지 않는다.

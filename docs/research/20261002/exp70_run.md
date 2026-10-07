# EXP70 실행 기록

2026-10-02 KST. 결과: `tabpfn/results/20261002_exp70_class_arrival_100k_s43`.

- Seed 43, 최종 각 100,000행, 승인된 클래스별 배분과 출현 순서. [설계](exp70_protocol.md).
- 두 데이터 모두 입력 준비 완료. 기존 EXP69 bank의 15,024/15,792행을 전부 포함해 증설했으며, 각 100,000개의 고유 row ID와 train/test 분리를 검증했다. 기존 test IDs 유지.
- 단계별 합계, 미래 클래스 제외, 기존 사례 유지, class ID 재매핑, precision/Benign 오탐 정의에 대한 unit test 5개 통과.
- 04:44경 CIC2018 TabPFN 실제 GPU 추론 시작. 초기 context 84,820행, fit 13.45초. 04:45 기준 360,448행 처리, 평균 6,659행/초. RTX4090 사용률 100%, 관찰 시 GPU 메모리 약 2.8GB.
- 초기 실측으로 추정한 잔여시간: 전체 약 75–100분, 완료 예상 06:00–06:25 KST. CIC2018 두 모델 결과는 약 05:25–05:40 KST 예상. 초기 단계 처리량을 이후 context 크기로 보정한 잠정치이며 ToN/후속 단계 실측에 따라 달라질 수 있다.
- 비용 측정을 위해 GPU 작업은 순차 실행. 순서: CIC2018 TabPFN → CIC2018 XGB → ToN TabPFN → ToN XGB. 프로세스는 터미널과 독립적으로 실행한다.

진행 확인:

```bash
cat tabpfn/results/20261002_exp70_class_arrival_100k_s43/status.json
tail -f tabpfn/results/20261002_exp70_class_arrival_100k_s43/logs/cic2018_tabpfn_n100000.log
```

코드 snapshot과 명령/PID는 `source/`, `launch.json`. 실제 상태는 `status.json`, 각 단계 처리량은 `jobs/*/progress.json` 또는 `jobs/*/stage_*/progress.json`. 초기 실측과 검증은 `initial_eta.json`, `startup_validation.json`. 완료 stage는 `COMPLETE.json`·지표·확률·비용을 저장한다. 이전 EXP69 결과와 snapshot은 변경하지 않았다. HTML/PPT는 수정하지 않았다.

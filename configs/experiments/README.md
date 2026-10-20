# 연구 버전과 실험 실행 관리

연구 버전(vN)은 사용자 요청 또는 제안에 대한 확인 후에만 생성·전환합니다. 날짜·실험 수·점수로 자동 변경하지 않습니다. 승인된 버전 안에서 실험·seed·재시도는 자동 등록합니다. 이번 문서 개편은 v7을 유지합니다.

```bash
python scripts/results_versions.py find exp83
python scripts/results_versions.py where OLD_RESULT_PATH
python scripts/results_versions.py run --version v7 --experiment expNN --scope tabpfn --label LABEL --protocol PROTOCOL -- python WORKER.py --output '{run_dir}'
python scripts/results_versions.py index
```

`{run_dir}`는 고유 실행 폴더의 절대 경로입니다. Worker에 반드시 전달합니다. 등록부는 `registry.json`; 코드와 결과는 각각 `scripts/vN/expNN/`, `results/vN/expNN/run_id/` 또는 tabpfn 아래 같은 구조입니다. 경로 해석은 `scripts/common/experiment_paths.py`를 사용합니다.

관리 실행은 명령·시각·종료 코드·stdout/stderr·Git 상태와 주요 소스 해시를 기록합니다. Worker는 입력 행 ID, split, seed, checkpoint, 환경과 원시 예측을 별도로 보존해야 합니다. 직접 실행했다면 index로 목록을 갱신합니다.

실행 상태(finished/failed/interrupted)와 근거 채택(selected/exploratory/not_adopted/invalid/unreviewed)은 구별합니다. 성공 종료만으로 논문 근거가 되지 않습니다. 새로운 방향 등록은 `new --label LABEL --note REASON --approval-note USER_AUTHORIZATION`, 재분류는 `assign`을 쓰되 실제 사용자 확인 근거가 필요합니다.

모든 실험은 야간 실행까지 `docs/lablog/reports/YYYYMMDD[_expNN].tex`와 PDF, `docs/lablog/log.md` 한 줄로 기록합니다. `make -C docs/lablog/_support`와 `make -C docs/lablog/_support check`로 확인합니다. 자세한 작성 규칙은 루트 AGENTS.md입니다. 원고·발표 자동 반영과 연구 버전 변경은 실험 기록 자동화에 포함되지 않습니다.

2026-10-07 코드·결과 이동 기록은 `migrations/20261007_layout/`, 2026-10-20 문서 통합·삭제 대응은 `migrations/20261020_documents/`에 있습니다. 과거 source snapshot과 결과 metadata에 적힌 경로는 당시 이력이며 파일을 다시 쓰지 않습니다. 현재 protocol 연결은 registry를 따릅니다.

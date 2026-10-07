# 논문용 근거와 연구 의도 정리

현재 목적은 **소수 라벨 공격의 탐지 성능 개선**이다. 순차 공격 도입은 평가 방향이며, 소수 라벨 확보 전후 적응 비교는 필수 목표가 아니다. 한국어 결정 기록은 `../research/20261002/research_direction.md`를 참조한다.

- `experiment_record.tex` / `.pdf`: 연구 질문, 설계 상태, 채택한 EXP61·63 근거, 분기점
- `sections/`: 사람이 수정하는 본문. 표 생성 시 덮어쓰지 않음
- `results/tab_*.tex`: 재사용 가능한 표. 클래스는 test 표본 수 내림차순, 수치는 우측 정렬
- `results/tab_expert_detail_*.tex`: 전체 expert precision/recall/F1 상세표 (`longtable` 필요)
- `sources.json`: 실제 입력 파일과 생성 표의 SHA-256, K 및 제외 근거

```bash
python scripts/build_latex_results.py --compile
```

기존 논문 파일에서 필요한 표를 `\input{...}`할 수 있다. 본문의 source 경로와 LaTeX working directory를 맞춰 사용한다. 이전 `tab_feasibility`, `tab_temporal_arrival`, `tab_residual_oracle`은 `legacy_unadopted/`로 분리했으며 현재 본문에서 사용하지 않는다. 이전 전체 정리본은 `docs/slides/archive/20261002_before_intent_revision/docs/latex/`에 보존했다. 표 생성은 실험을 실행하지 않는다.

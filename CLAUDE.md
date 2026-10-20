# CLAUDE.md

이 저장소의 연구·문서 작업은 루트 `AGENTS.md`를 먼저 읽고 따릅니다.

## 문서 규칙 (2026-10-20)

- `docs/lablog/reports/YYYYMMDD[_expNN].tex`: 한국어 상세 실험 보고서. 목적·설정·도표·관찰·해석·한계·후속 개선안과 원시 실행을 연결합니다. 통합 main.tex는 만들지 않습니다.
- `docs/lablog/log.md`: 날짜별 실험당 한 줄인 유일한 한국어 Markdown 로그. 같은 날 같은 실험은 해당 줄을 보완합니다.
- `docs/latex/main.tex`: 영어 Overleaf 논문. 실험 기록·experiment_record·날짜별 전체 원고 중복본은 만들지 않습니다.
- `docs/slides/`: 한국어 먼저, 영어는 사용자 요청 시. 정렬은 `docs/slides/_support/guides/지침.txt`를 따릅니다.
- 야간·무인 agent 실험도 별도 요청 없이 결과 검증 후 LaTeX·한 줄 로그를 작성하고 임시 PDF로 컴파일 검증합니다. 실패·중단도 상태와 복구안을 남깁니다. 실행 중 인계에는 남은 기록을 명시합니다.
- `make -C docs/lablog/_support` / `make -C docs/lablog/_support check`로 컴파일·검사합니다. 생성기는 `scripts/common/lablog.py::write_record`를 사용할 수 있습니다.
- 옛 docs/research와 루트 lablog, HTML 발행 체계는 폐기했습니다. 상세 보고서의 유일한 위치는 docs/lablog입니다. 이전 경로 대응은 `configs/experiments/migrations/20261020_documents/path_map.json`을 확인합니다.
- 질문은 대화로 답하고, 실험 기록과 별개로 원고·슬라이드 변경은 사용자 요청 범위를 따릅니다.

최상위 일반 파일은 lablog의 log.md, latex의 main.tex·ref.bib만 허용합니다(.gitignore 예외). slides 최상위에는 일반 파일을 두지 않으며 eng/kor에는 날짜별 PPTX·PDF만 둡니다. 지침·코드·템플릿·계획·QA·출처 등 보조 자료는 각 폴더의 _support 아래에 둡니다. 생성기도 이 위치를 유지합니다.

슬라이드 이외의 원고·보고서 PDF는 영구 보관하지 않습니다. Cursor는 `%TMPDIR%`에서 미리보기를 빌드하며 원본 폴더로 복사하지 않습니다. `make` 검증 PDF도 임시 폴더에서 생성 후 제거합니다. 그림 입력·외부 참고논문 PDF는 유지합니다. lablog의 `% !TeX root = 파일명.tex`와 `% !LW recipe=lablog XeLaTeX` 헤더를 유지합니다.

## 연구와 실행

중심 목표는 소수 라벨 공격의 탐지 개선이며 순차 도입은 평가 시나리오입니다. 실제 timestamp·환경 drift·라벨 없는 OOD 탐지로 확대 해석하지 않습니다. 현재 판단은 `docs/lablog/reports/20261020.tex`, 전체 흐름은 log.md를 읽습니다.

모델의 문제·병목에는 근거에 따른 개선 방향을 제시합니다. 관찰과 원인 추정, FT와 context/cache 준비, 초기·갱신·추론 비용, 단계 합산과 최종 모델 평가, 실행 완료와 근거 채택을 구별합니다. 원 결과의 설정·표본 수·예측을 확인하고 서로 다른 조건을 같은 승패표로 합치지 않습니다. 변경 기제는 과거의 실패 조건과 비교하고 해결 가설을 명시합니다.

연구 버전 생성·전환은 사용자 요청 또는 제안에 대한 확인이 필요합니다. 날짜·실험 수·점수만으로 새 vN을 만들지 않습니다. 승인된 버전 안에서 새 실행·seed·재시도 등록은 자동화합니다. 사용법은 `configs/experiments/README.md`입니다.

코드: `scripts/vN/expNN/`, `tabpfn/scripts/vN/expNN/`; 공통 코드: 각 scripts/common.
결과: `results/vN/expNN/run_id/`, `tabpfn/results/vN/expNN/run_id/`.
등록부: `configs/experiments/registry.json`; 조회: `python scripts/results_versions.py find expNN`.
실행은 `python scripts/results_versions.py run --version vN --experiment expNN --scope root|tabpfn --label LABEL -- COMMAND ... {run_dir}`를 사용합니다. 완료 결과와 당시 source snapshot은 수정하지 않습니다. 경로 도우미는 `scripts/common/experiment_paths.py`입니다.

연구 알고리즘 변경은 새 실험으로 분리하고 공용 helper 변경이 과거 실행에 미치는 영향을 확인합니다. seed·분할·표본 예산은 해당 승인 프로토콜을 따르며 과거 seed42를 새 기본값으로 상속하지 않습니다. 최신 랩미팅 비교는 seed43 단일 실행입니다. 설정·환경·입력 ID·원시 예측을 보존하고 test로 문턱을 조정하지 않습니다.

환경은 requirements.txt / requirements-sota.txt, GPU 설치는 scripts/common/setup_exp57_env.py 및 setup_exp61_env.py를 사용합니다. 외부 PFN은 tabpfn/third_party/의 UPSTREAM.txt를 따르고 v1/v3 차이를 기록합니다. 체크포인트는 필요 시 fetch_checkpoints.sh로 준비합니다.

## 논문 규칙

제목 **TAILGUARD: An Uncertainty-Driven Ensemble Against Long-Tailed Cyber Threats**를 유지합니다. 선행연구 각 설명 단위에는 근거 있는 한계·제안 기제·검증을 연결한 Discussion이 필요합니다. Related Work 각 소절은 `\paragraph{Discussion.}`으로 끝냅니다. §2.3 BoostPFN·LoCalPFN·DistPFN은 하나의 연구군으로 설명하고 통합 Discussion을 씁니다. Contributions는 bullet이며 설계 기여와 실증 성과를 구별합니다. 자세한 원고 원칙은 AGENTS.md를 따릅니다.

## Data

`data/` holds real copies (17 GB, gitignored) of the five pkls the current code needs:

| File | Used by | Notes |
|---|---|---|
| `nfv3_energy_suite_uncapped_scenarios.pkl` | s43, s44, tabpfn | 14.7 GB, 66.9M rows × 46 feat; 4 datasets in one schema |
| `nfv3_energy_suite_cic2018_scenarios.pkl` | tabpfn `*_capped` targets | per-family-capped slice of all four |
| `nfv3_energy_suite.pkl` | `preprocess_nfv3_cic2018_scenarios.py` (`--base-suite`) | |
| `cic2017_full_raw.pkl` | tabpfn `cic2017_full` | 15 classes, synthetic `time_proxy` |
| `cic2017_chrono_v2.pkl` | `analyze_nfv3_embedding.py` | legacy CIC open-set split |

The legacy suite pkls key on `dataset_names ∈ {cse_cic_ids2018, bot_iot, ton_iot, unsw_nb15}` plus
`families`, `attack_scenarios`, `timestamps`. Only cic2018 has real per-scenario chronological
splitting; the others split at family level.

Preserve the approved split and row IDs. Do not replace an existing chronological split with a random split. Random stratified splits on CIC data leak temporally —
this project already paid for that lesson (`s05`/`s06` in SRC_HISTORY). If validation looks
near-perfect, suspect leakage, not success.

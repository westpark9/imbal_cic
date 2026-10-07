# 연구 방향 버전과 실행 기록

`vN`은 연구 방향, 각 timestamp + 고유 ID 폴더는 개별 실행이다. **연구 버전 생성·전환은 사용자 요청 또는 agent의 검토·제안에 대한 사용자 확인 후에만 수행한다(2026-10-06 합의).** 날짜·실험 수·점수 변화는 자동 버전 변경 사유가 아니다. 연구 버전 재분류도 사용자 확인을 따른다. 방향을 승인받아 등록하면 이후 **관리 명령으로 시작한 실행의 폴더·명령·실행 상태·로그·버전별 목록 갱신은 자동**이다. 기존 스크립트를 직접 실행한 경우에는 `index`로 목록에 반영한다.

현재 등록된 최신 버전은 `v7_class_incremental_arrival`이고 EXP69·70은 완료했다. v6는 그에 앞선 연구 의도·평가 방향 정리 기록이며, 기존 Claude 시나리오는 v5에 명시적으로 고정했다. 이번 폴더 개편은 연구 방향 변경이 아니므로 새 vN을 생성하지 않는다.

아래 명령은 **현재 도구의 사용법**이다. 후반의 개편안은 아직 구현·이동 전이며, 현재 도구는 중첩된 버전/실험/실행 경로를 지원하지 않는다.

```bash
# 읽기·분류 결과를 갱신하며 모델은 실행하지 않음
python scripts/results_versions.py index

# 사용자 요청 또는 제안에 대한 사용자 확인 후에만 새 방향 등록
python scripts/results_versions.py new --label NEW_DIRECTION --note '가설·프로토콜 변경 이유'

# 향후 실험 실행 예시: 실제 worker와 output 옵션 이름에 맞춰 사용
python scripts/results_versions.py run --version v7 --label STUDY --protocol docs/research/20261002/research_direction.md -- python PATH_TO_WORKER.py --output '{run_dir}'

# 이미 존재하는 실행의 버전·채택 상태를 명시
python scripts/results_versions.py assign tabpfn/results/EXISTING_RUN v7
python scripts/results_versions.py mark tabpfn/results/EXISTING_RUN --evidence selected --note '사용 범위와 확인한 근거'
```

`{run_dir}`는 새로 만든 절대 경로로 치환된다. Worker가 해당 출력 옵션을 지원해야 한다. 새 스크립트는 환경변수 `EXPERIMENT_RUN_DIR`를 읽도록 구현할 수도 있다. 고정 경로에 출력하는 기존 worker를 wrapper로 실행해도 그 raw 출력이 자동 이동되는 것은 아니다. 해당 raw 폴더는 별도로 `assign`한다.

자동으로 저장되는 항목: 실행 명령·시각·종료 코드, stdout/stderr 로그, Git HEAD와 작업 상태·diff hash, 명령에 직접 포함된 스크립트/설정 파일의 SHA-256. **모든 데이터·checkpoint·환경의 완전한 재현 정보를 대신하지 않는다.** 각 worker에서 데이터 ID, split, seed, checkpoint, 환경 버전, context 설정을 함께 저장해야 한다. 다른 비밀값을 커맨드 인자로 넣지 않는다.

상태를 두 축으로 관리한다.

| 종류 | 의미 |
|---|---|
| `running` / `finished` / `failed` / `interrupted` | 관리 명령의 실행 상태. 종료 0도 과학적 유효성·모든 실험 완료를 뜻하지 않음 |
| `results_present` / `partial_or_legacy` | 기존 raw 결과 존재 여부. 결과 파일 하나가 있다고 완료 처리하지 않음 |
| `selected` / `exploratory` / `not_adopted` / `invalid` / `unreviewed` | 주장에 채택한 범위와 검토 상태. 실행 성공과 분리 |

현재 도구의 `index`는 기존 raw 폴더를 이동·삭제하지 않는다. `versions/vN_.../`는 상대 symbolic link로 구성한 보기이며 Git checkout 위치가 달라져도 같은 저장소 내 경로가 유지된다. 버전을 재배정하면 이전 보기의 링크만 정리한다. 동시 실행 등록은 lock과 원자적 파일 교체로 보호한다. 사용자가 요청한 아래 개편에서는 내용·이력 보존, 경로 매핑 및 무결성 검증을 전제로 실제 위치를 변경한다.

완료 후 LaTeX 표 갱신은 `python scripts/build_latex_results.py --compile`. 현재 생성기는 명시된 EXP61·63만 입력으로 사용한다. 새 결과를 자동으로 논문 주장에 편입하지 않으며, 채택할 결과와 입력 경로를 검토하여 갱신한다.

검증: 임시 폴더에서 실패 상태 판별·동명 폴더 충돌·재배정 링크·성공/실패 종료·동시 등록의 5개 통합 테스트. GPU 실험은 포함하지 않는다.

## 2026-10-06 사용자 피드백을 반영한 개편안

운영 규칙과 개편 설계를 반영했다. 2026-10-06 후속 요청에 따라 기본 requirements 통합과 SOTA 분리는 완료했다. 실험 경로 이동·관리 코드 확장·LaTeX 보완은 아직 실행하지 않았다.

### 버전과 자동화의 경계

- 기존 v1–v7, EXP 번호, 결과 채택·제외 기록을 보존한다. 정리 작업으로 새 연구 버전을 만들지 않는다.
- 새 연구 버전에는 변경 이유, 이전 버전과 달라지는 연구 질문·설계, 사용자 요청 또는 확인 근거를 기록한다. agent가 변경을 제안한 경우 답변 전에는 전환하지 않는다.
- 실험 등록 시 승인된 버전을 명시하고 날짜로 임의 배정하지 않는다. 기존 자료의 분류 근거가 불명확하면 미분류 목록에 두고 확인한다.
- 승인된 버전 안의 실험 번호, 고유 실행 ID, 설정·소스·환경 보존, 상태·색인 갱신은 자동화한다. 실행이 많아져도 vN을 자동 증가시키지 않는다.

### 폴더 구조

아래의 vN·expNN은 자리표시자이며 실제로는 기존 ID와 설명 이름을 사용한다.

```text
data/                               # 원본·정제 데이터와 재사용 split
configs/
  experiments/registry.json          # 제안: 버전/실험/실행의 공통 등록부
  environments/archive/             # 통합 이전 설치 명세와 해시 기록
scripts/
  results_versions.py                # 공통 관리 진입점 유지
  common/                           # 재사용 코드, 전처리·보고서 공통 생성기
  vN/expNN/                         # 실행·분석·보고서 생성 코드
results/vN/expNN/<run_id>/
tabpfn/
  scripts/common/
  scripts/vN/expNN/
  configs/vN/expNN/
  results/vN/expNN/<run_id>/          # results_past의 원래 실행도 여기에 통합
lablog/                             # 기존 HTML·Markdown 보고서
docs/
  research/                         # 설계·세부 해석·분기점 기록
  latex/
    README.md                       # 최신 원고 위치와 날짜별 주요 변경
    YYYYMMDD/                       # 논문 원고를 편집한 날짜
      main.tex                      # 논문 진입점
      sections/                     # 서론·관련연구·방법·실험·결론
      tables/                       # 저장된 수치에서 생성
      figures/
      references.bib
      sources.json                  # 표/그림 ↔ 연구 버전·실험·실행·원본 해시
      build/                        # PDF 및 컴파일 산출물
requirements.txt                    # 통합 기본 설치 명세
requirements-sota.txt               # 기본 명세 포함 + SOTA 추가 패키지
```

`scripts/experiments/`, `scripts/data/`, `scripts/reports/`는 만들지 않는다. 데이터·보고서 산출물은 scripts 밖에 두되, 이를 만드는 Python 코드는 scripts에 유지한다. reports 역할은 기존 `lablog/`와 `docs/latex/`에 연결하여 중복 보고서 루트를 만들지 않는다. 여러 실험이 공유하는 runner는 복제하지 않고 common 코드와 실험별 설정·진입점으로 연결한다.

`tabpfn/results_past/`는 별도 영구 archive로 남기지 않고 기존 버전·실험으로 통합한다. old/past는 연구 방향이 아니므로 새 버전의 근거가 아니다. 동명 실행은 원본 식별자를 보존한 고유 run ID로 구별하며 덮어쓰지 않는다. 학습 캐시·분석 집계처럼 실행이 아닌 항목은 목적에 맞게 분리하고, 자동으로 exp/run으로 취급하지 않는다.

### LaTeX 정리

현재 생성기는 EXP61·63만 반영하며, 본문과 sources.json에는 EXP69·70 실행 이전 상태가 남아 있다. 보완 시 최신 확정 내용과 당시 기록을 구별하고 EXP69·70의 설계·결과·비용을 원본 결과에 연결한다. EXP70의 단일 Global 결과를 expert·S/V 포함 모델 결과로 섞지 않는다.

사용자 후속 확인: docs/latex는 실험 보관함이 아니라 **논문 본문**이므로 연구 버전/실험으로 나누지 않고 날짜순 원고로 관리한다. 각 날짜 폴더 안에는 서론·관련연구·방법·실험·결론 등 논문 구조를 유지한다. 같은 날 편집은 해당 날짜 원고에 반영하고, 다른 날 원고 편집을 시작할 때 최신 원고를 새 날짜로 이어받아 직전 원고를 보존하는 방식을 제안한다. 실험 실행만으로 새 날짜 원고를 만들지 않는다. 기존 파일은 실제 편집일과 이력을 확인한 뒤 옮기며 현재 작업일을 과거 원고의 작성일로 붙이지 않는다. README에는 최신 원고 경로와 날짜별 주요 변경을 남긴다.

HTML·Markdown에는 상세 진단과 연구 분기점을 계속 보존하고, 논문 본문에는 현재 주장에 필요한 질문·설계·결과·해석을 연결한다. 모든 시행착오를 본문에 넣지 않는다. 연구 버전·실험·실행 ID 및 관련 기록 링크는 sources.json에 남겨 논문 구조와 근거 추적을 분리한다. 탐색 결과, 보류·폐기된 시도, 채택 결과를 구별한다. 수치 표·그림은 동일한 저장 결과와 공통 추출 로직에서 생성하고 수동 본문은 덮어쓰지 않는다. 원본 경로가 이동하면 출처 매핑으로 추적하되 과거 실행의 snapshot 내용은 수정하지 않는다. 생성·컴파일 시 누락 입력, 수치·표본 수·정렬·출처를 검증한다. 원고 날짜 변경은 연구 버전 vN 변경이나 새로운 결과의 자동 채택을 의미하지 않는다.

2026-10-06 추가 명확화: 사용자는 현재 논문의 tex·bib를 제공할 수 있으며, 필요한 것은 HTML의 가독성을 유지하면서 **본문 작성을 위한 줄글 형태의 글감**을 축적하는 것이다. 기존 docs/research의 실험별 결과 MD에 서술형 해석을 두어 새 보고서 체계를 중복 생성하지 않는다. 실험 목적·통제 조건, 관찰 결과, 해석, 확인되지 않은 설명·한계, 후속 판단을 문단으로 연결하고 원본 수치·표·그림을 참조한다. 상세 수치를 모두 문장으로 반복하지 않으며, 관찰된 결과와 아직 검증하지 않은 인과 설명을 명시적으로 구분한다. 이 기록을 나중에 제공된 tex·bib의 목차·용어·인용에 맞추어 선택·편집한다. 글감 작성은 최종 본문 채택이나 연구 버전 변경을 자동으로 의미하지 않는다.

### requirements 통합 완료 (2026-10-06)

| 현재 파일 | 역할 |
|---|---|
| requirements.txt | 기존 공통 의존성 + EXP57의 주요 고정 버전. 공유 tqdm도 여기서 관리 |
| requirements-sota.txt | `-r requirements.txt`로 기본 환경을 포함하고 faiss-cpu, tensorboard, psutil, einops 추가 |

통합 이전 명세:

| 이전 파일 | 실제 역할 | 처리 |
|---|---|---|
| requirements.txt | 초기 XGB/MoE 공통 환경. 대다수 최소 버전, torch 2.6.0 고정, TabPFN 미고정 | 원본 보존 후 통합 기본 명세로 갱신 |
| requirements-exp57.txt | TabPFN 8.2.0 등 주요 패키지를 고정한 EXP57 환경 | 주요 고정 버전을 requirements.txt로 통합, 루트의 옛 파일 제거 |
| requirements-exp61.txt | EXP57 위에 SOTA용 패키지 추가 | requirements-sota.txt로 이름과 기본 환경 포함 관계 명확화 |

통합 이전 세 원본은 configs/environments/archive/20261006_before_requirements_merge에 바이트 그대로 보존하고 manifest.json에 SHA-256을 기록했다. 기존 직접 의존성의 패키지 목록은 유지했다. 주요 버전은 EXP57의 고정을 유지하고 imbalanced-learn·joblib·tqdm은 현재 설치된 버전으로 명시했다. 기존 torchvision·CuPy·seaborn 요구사항도 유지한다.

PyTorch는 공통 명세에서 `torch>=2.5`로 선언하되, 실제 GPU 설치는 setup_exp57_env.py가 드라이버별 정확한 빌드를 선택하고 constraint로 고정한다. 공통 의존성 설치에도 선택된 PyTorch wheel index를 전달하여 torchvision이 같은 CUDA index에서 해결될 수 있게 한다. SOTA 설치 진입점 setup_exp61_env.py도 새 파일을 사용하며 선택된 torch 버전을 유지한다. 기존 설치 명령 이름과 환경 marker는 호환성을 위해 유지했다.

검증: 기존 환경 설치·재사용·드라이버 보호 테스트 5개 통과, 원본 파일 해시와 전체 직접 의존성 보존 및 EXP57 주요 버전 유지 확인. 현재 로컬 환경에서 `pip install --dry-run -r requirements-sota.txt`의 의존성 해석 성공(CuPy·seaborn 추가 예정, 나머지 이미 충족). 실제 패키지 설치·교체는 수행하지 않았으며 새로운 GPU 환경의 전체 설치·연산 검증을 대신하는 결과는 아니다.

### 이행 순서와 검증

1. 추적·미추적 파일, import·경로 의존성, results_past까지 포함한 이동 목록을 만든다. 승인된 버전 매핑과 미분류 항목을 구분한다.
2. 기존 results_versions.py를 확장해 중첩 경로·공통 등록부·실험/실행 ID·승인 근거 기록을 지원한다. 실행 상태와 과학적 채택 상태를 분리한다.
3. 현재 사용하는 실험부터 scripts/results를 이동하고 실행·재개·분석 코드의 경로를 갱신한다. LaTeX 생성기는 등록부의 출처를 이용하도록 변경한다.
4. 과거 결과와 results_past를 같은 규칙으로 통합한다. 원본 파일 내용·소스 snapshot을 보존하고 경로 변경은 별도 manifest에 기록한다.
5. 파일 수·크기·필요 해시, import·재개·분석 동작, 생성 표와 LaTeX 컴파일을 검증한다. 되돌릴 수 있는 이동 기록을 남긴다. 검증 전 원본 삭제나 경로 덮어쓰기를 하지 않는다.

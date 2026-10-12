# 논문용 근거와 연구 의도 정리

2026-10-07 사용자가 제공한 논문 원고는 `main.tex`, 참고문헌은 `ref.bib`이다.
`Haider_ETD.pdf`는 논리 전개의 참고 자료이며, 제출 서식은 원고의 대상 학회인
[AsiaCCS 2027 공식 지침](https://asiaccs2027.cityu.edu.mo/call-for-papers/index.html)을 따른다.
선행연구 설명마다 Discussion을 두는 필수 작성 원칙은
[`AGENTS.md`](../../AGENTS.md)의 “논문 작성: 선행연구 Discussion 필수”를 적용한다.
아래 `experiment_record.tex`는 기존 연구 근거 정리 문서로, 사용자 원고와 구별한다.

## 2026-10-12 원고 수정

- 최신 편집본: `main.tex`, `ref.bib`; 확인용 PDF: `main.pdf`. 날짜별 사본: `20261012/`.
- 원본 제목 **TAILGUARD: An Uncertainty-Driven Ensemble Against Long-Tailed Cyber Threats**를 복원했다. 명시적인 사용자 요청 없이 제목을 변경하지 않는다.
- Introduction의 기여를 수행한 내용과 기존 접근과의 차이가 드러나는 세 bullet로 정리했다. 검증되지 않은 성능·비용 우위는 기여로 주장하지 않는다.
- §2.3의 BoostPFN·LoCalPFN·DistPFN Discussion을 하나로 통합하고, §2.4에는 RIDE와 함께 ACE·SADE·BalPoE를 보강했다. 보강한 문헌은 관련 연구이며 새로 실행한 비교군이 아니다.
- 실험 표는 `20261007/tables/`의 사본을 그대로 사용한다. 이전 날짜 폴더는 보존하며 새 `sources.json`에 표의 상속 경로, 해시와 추가 문헌의 출처를 기록한다.

최신 원고만 컴파일하려면 다음을 실행한다(저장소 루트 기준).

```bash
cd docs/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf main.pdf
```

## 2026-10-07 원고 개편

- 편집본: `main.tex`, `ref.bib`; 확인용 PDF: `main.pdf`.
- 날짜별 사본과 생성 표·출처: `20261007/main.tex`, `20261007/ref.bib`, `20261007/tables/`, `20261007/sources.json`.
- 업로드 원본 보존: `20261007/original/`. 연구 버전은 변경하지 않았다.
- 일반 IDS 개론을 제거하고 Related Work → Preliminaries → Proposed Approach 순서로 작성했다. TabPFN의 원리, 현 비교군, 각 선행연구의 Discussion을 추가했다.
- 기존 Energy/OOD 설계와 미실행 시나리오는 원본에 보존했다. 현재 본문은 EXP61·62·63의 정적 비교와 EXP70의 단일 모델 갱신 비교를 구별한다. 방법론·최종 성과는 미확정 초안이다.
- 표는 저장된 결과에서 재생성하며 HTML과 실험 결과는 수정하지 않는다. 클래스는 데이터셋 내 test support 내림차순, 수치는 정수 일의자리·소수점에 맞춰 정렬한다.
- 로컬 `acmart.cls`와 `ACM-Reference-Format.bst`는 CTAN의 ACM 템플릿(acmart 2026/08/16 v2.20)에서 가져왔다. 수정하지 않은 원본 소스·라이선스 헤더와 입수 경로는 `20261007/template/`에 보존했다.

표 생성 도구 `scripts/common/build_manuscript_tables.py`는 현재 출력 날짜가 `20261007`로 고정되어 있다. 최신 원고의 단순 편집·컴파일에는 실행하지 않는다. 추후 수치를 갱신할 때는 출력 대상을 해당 원고 날짜로 지정하도록 먼저 수정하여 이전 날짜의 표·출처를 덮어쓰지 않도록 한다.

날짜별 사본도 같은 디렉터리의 `tables/`, `ref.bib`, ACM 클래스·서식 파일로 독립 컴파일할 수 있다. 원고 변경 후 날짜별 사본을 확정할 때는 tex·bib·PDF와 `sources.json`을 함께 갱신한다. 근거가 그대로인 표는 이전 날짜에서 복사하고 상속 경로와 해시를 기록하며, 근거가 달라졌을 때만 재생성한다. 새 날짜 원고를 실험 실행마다 자동 생성하지 않는다.

제출 전 미정 항목: 최종 방법론과 근거 채택, 추가 검증, 익명 아티팩트 URL, 부여된 출판·제출 메타데이터. Open Science의 현재 문장은 준비 상태를 표시하며 이미 공개했다는 뜻이 아니다.

현재 목적은 **소수 라벨 공격의 탐지 성능 개선**이다. 순차 공격 도입은 평가 방향이며, 소수 라벨 확보 전후 적응 비교는 필수 목표가 아니다. 한국어 결정 기록은 `../research/20261002/research_direction.md`를 참조한다.

- `experiment_record.tex` / `.pdf`: 연구 질문, 설계 상태, 채택한 EXP61·63 근거, 분기점
- `sections/`: 사람이 수정하는 본문. 표 생성 시 덮어쓰지 않음
- `results/tab_*.tex`: 재사용 가능한 표. 클래스는 test 표본 수 내림차순, 수치는 우측 정렬
- `results/tab_expert_detail_*.tex`: 전체 expert precision/recall/F1 상세표 (`longtable` 필요)
- `sources.json`: 실제 입력 파일과 생성 표의 SHA-256, K 및 제외 근거

```bash
python scripts/common/build_latex_results.py --compile
```

기존 논문 파일에서 필요한 표를 `\input{...}`할 수 있다. 본문의 source 경로와 LaTeX working directory를 맞춰 사용한다. 이전 `tab_feasibility`, `tab_temporal_arrival`, `tab_residual_oracle`은 `legacy_unadopted/`로 분리했으며 현재 본문에서 사용하지 않는다. 이전 전체 정리본은 `docs/slides/archive/20261002_before_intent_revision/docs/latex/`에 보존했다. 표 생성은 실험을 실행하지 않는다.

# 10월 2일 랩미팅 한국어 발표 · IDS 운영을 위한 TabPFN 활용

2026-10-02 최신 사용자 피드백 반영. 도입은 날짜별 공격 시나리오를 모은 데이터의 기존 F1 비교, 실제 운영에서의 새 라벨·환경 변화와 모델 갱신, TabPFN 활용 이유를 세 문단으로 설명한다. 단순 분류 자체를 권장하지 않는다고 단정하지 않고, 고정 test F1만으로 운영 적합성을 판단하기 어렵다고 표현한다. TabPFN의 재학습 없는 추론은 구조적 특성이며 IDS 성능·비용 우위와 구분한다.

한국어판만 수정. 직전 13장 중 2·9·10·12·13장을 삭제했다. 기존 7장의 순차 도입 설계를 도입 직후로 이동하고, 두 방법 모두 완료한 데이터셋의 결과만 이어 붙인다. 기존 8장의 시스템 관점 후속 검증은 마지막에 유지하고, 두 데이터셋 완료 결과를 반영해 정상 사례 다양성·context 규모와 오탐·계산 비용을 후속 확인 대상으로 구체화한다. 기존 3장은 `데이터 검토: 동일 feature의 상충 라벨 삭제`로 제목을 바꾸고 기존 4장 좌측 아래 설명 박스를 제거했다. 기존 5·6·11장을 모델 구조·정적 성능·SOTA 비용 순서로 연결한다.

## 현재 구성

| 새 페이지 | 내용 |
|---|---|
| 1 | IDS 운영을 위한 TabPFN 활용 |
| 2 | 새 공격 사례 추가에 따른 탐지기 갱신 실험 |
| 3 | CIC2018 · 신규 공격 반영 성능과 비용 |
| 4 | ToN · 신규 공격 반영 성능과 비용 |
| 5 | 데이터 검토: 동일 feature의 상충 라벨 삭제 |
| 6 | 정제 이후 클래스 분포와 고정 test |
| 7 | Context 기반 Global·expert 모델 구조 |
| 8 | 정제 데이터의 현재 성능 · SOTA 비교 |
| 9 | 정제 데이터의 SOTA 비용 |
| 10 | 후속 검증 · 새로운 공격을 반영하는 시스템 |

부록 없이 본문으로 구성한다. CIC2018·ToN 두 데이터셋의 모든 단계가 완료되어 결과를 3·4장에 반영했다. 전체 10장이다. ToN은 MitM의 높은 recall과 낮은 precision, Benign 오탐을 함께 설명한다. 신규 공격 F1은 각 공격 도입 직후의 값이며, 클래스 행은 test 표본 수 내림차순으로 배치한다. 오른쪽의 최종 100k 성능·비용과 평가 시점이 다름을 제목으로 구분한다. 모든 신규 공격을 포함하고, 추론 비용도 함께 표시한다. 정적 SOTA의 제안 모델 행은 기존 EXP63 K=4, S/V 포함 결과를 유지하며 EXP70 Global 결과로 대체하지 않는다.

현재 생성기: `python docs/slides/revise_1002_ko_operational.py`. 입력은 수정 직전 작업본의 보존 경로 `../archive/20261002_before_simplified_intro/docs/slides/kor/1002_labmeeting_ko_draft.pptx`다. 기본 실행은 `1002_operational_qa/result_snapshot.json`에 동결한 결과를 재사용한다. `--refresh-results`는 실행 결과를 다시 읽되 데이터셋별 두 방법의 모든 단계 완료를 확인한 결과만 슬라이드에 포함한다. 이전 생성기는 최신 서사를 되돌리므로 사용하지 않는다.

검수: `1002_operational_qa/validation.json`, `render_validation.json`, `opening_contact.png`, `rest_contact.png`. 전체 PPTX를 PDF로 렌더링해 글자 누락·잘림·겹침을 점검하고, 모든 숫자 셀의 우측 정렬과 영어판 해시 불변을 확인했다. 발표 지침의 클래스 표본 수 정렬·고정 소수 자릿수 규칙을 유지한다. 영어판·HTML 수정 없음.

TabPFN 설명의 문헌 근거는 발표자 노트에 보존: [Hollmann et al., Nature 2025](https://www.nature.com/articles/s41586-024-08328-6). 데이터 수집 배경: [CSE-CIC-IDS2018 공식 설명](https://www.unb.ca/cic/datasets/ids-2018.html) 및 EXP69 timestamp 검증 기록. 실제 네트워크 환경 변화의 강건성은 EXP70으로 입증한 것으로 쓰지 않는다.

## 영어판 후속 제작

2026-10-02 후속 요청으로 최신 한국어 10장과 같은 내용의 영어판도 제작했다. 위 한국어 편집 당시의 “영어판 유지” 기록 이후에 이루어진 별도 요청이다. 한국어 PPTX/PDF는 그대로 유지했다. 영어판의 본문·표·그림 주석·발표자 노트와 최신 CIC2018·ToN 결과를 번역하고 수치 156개 일치를 확인했다. 생성기: `python docs/slides/translate_1002_en.py`. 번역 원문: `../1002_english_translation.json`. 영어 구성·검수: `../eng/1002_labmeeting_plan.md`, `../eng/1002_operational_qa/validation.json`.

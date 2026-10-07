# 10월 2일 랩미팅 한국어 발표 구성 및 실행 계획

갱신: 2026-09-30 · EXP61 로컬 SOTA 전 작업 완료, 한국어 PPTX·PDF 1~6장 작성

- 발표: 10월 2일 금요일 오전
- 결과 고정 목표: 10월 1일 목요일 10:00 KST
- 한국어 PPTX·PDF 완료 목표: 10월 1일 목요일 16:00 KST
- 현재 파일: `1002_labmeeting_ko_draft.pptx`, `1002_labmeeting_ko_draft.pdf`
- 영어 초안(09-30, 한국어 1~10장 번역·동일 레이아웃): `../eng/1002_labmeeting_en_draft.pptx`, `.pdf` — 생성 `docs/slides/translate_1002_en.py`, 한국어 판이 바뀌면 재실행

## 발표 흐름

데이터 검토 → Global과 residual expert 구조 → 현재 병목 → SOTA 성능·비용 → expert 역량과 S/V 기여 → 최종 모델 비교

시행 순서 대신 상대가 이해할 수 있는 문제와 근거의 흐름을 따른다. 기존 1~4장의 화면은 보존하고, 5~6장을 완료된 SOTA 결과로 채웠다. 미실행 S/V 및 최종 제안 모델은 TBD로 유지한다.

## 작성 기준

`docs/slides/지침.txt` 및 KENTECH 템플릿 적용. 사용자 편집본 0904의 시각 구성 계승. Pretendard, 16:9, 네이비·블루, 한 장에 한 메시지. 내부 랩미팅 정량 결과는 본문에 제시하고 상세 수치는 HTML에서 제공한다. 미확인 수치나 개선 원인은 단정하지 않는다.

## 본문 구성

| 장 | 제목 | 내용 | 상태 |
|---|---|---|---|
| 1 | 동일 feature를 공유하는 클래스 | 원래 클래스 × 동일 feature의 다른 라벨 비율, 중복 집계 예시 | 완료 |
| 2 | 정제 이후 클래스 분포 | 전체 train·validation·test의 정제 전후 클래스 행 수·비율 | 완료 |
| 3 | Global과 residual expert 구조 | Global·Expert·Route 풀, residual 군집, 온라인 S/V | 완료 |
| 4 | Global의 클래스별 성능과 expert의 병목 | Global F1과 EXP59 expert e3의 담당 영역·입력 유사군 사례 | 완료 |
| 5 | 정제 데이터의 SOTA 분류 성능 | Global, XGB, BoostPFN, LoCalPFN, DistPFN 및 전체 train XGB | EXP61 결과 반영 |
| 6 | SOTA 학습·추론 비용 | 학습·전체 test 추론 경과 시간, GPU 최대 사용량 | EXP61 결과 반영 |
| 7 | 담당 영역과 입력 유사군의 분류 능력 | 동일 범위에서 Global과 expert의 P/R/F1·교정·훼손 비교 | EXP59 상세표 구성 TBD |
| 8 | Scorer와 verifier의 기여 | Global / S·V 없음 / S만 / V만 / S+V, 호출·채택·교정·훼손 | 신규 실험 TBD |
| 9 | 제안 모델의 최종 성능과 비용 | 검증된 expert·S/V 구성과 5~6장 기준선 비교 | 신규 실험 TBD |
| 10 | 확인된 효과와 다음 개선 | 검증된 개선과 남은 오류에 대응하는 후속 조치 | 최종 구성 TBD |

## 데이터 기준

- CIC2018 정제 전 20,115,529행 → 정제 후 14,755,417행
- ToN 정제 전 27,520,260행 → 정제 후 10,807,288행
- CIC2018 Web attack: Benign 공유 1,764행 + Infiltration 공유 432행 − 양쪽 중복 429행 = 제거 1,767행, 잔존 771행
- 클래스 쌍별 feature 공유 비율은 모델 혼동행렬이 아니며, 중복 때문에 행 합을 제거 비율로 사용할 수 없음
- 현재 test: CIC2018 3,042,473행, ToN 2,271,723행

## 완료된 SOTA 비교

Seed 43, Global C0의 정확히 같은 100,000개 학습 ID 사용. Global/DistPFN v3, BoostPFN/LoCalPFN v1. LoCalPFN은 validation을 이용한 FT 포함. 같은 train ID를 사용하되 backbone 및 validation 사용까지 동일하다는 의미는 아니다.

| 방법 | CIC2018 Macro-F1 | ToN Macro-F1 |
|---|---:|---:|
| Global TabPFN v3 | 0.7824 | 0.6796 |
| XGBoost 100k | 0.7821 | 0.6983 |
| BoostPFN | 0.7148 | 0.5866 |
| LoCalPFN FT | 0.7024 | 0.6462 |
| DistPFN v3 | 0.7839 | 0.6682 |
| XGBoost 전체 train | 0.7795 | 0.7162 |

전체 train XGB의 학습 행 수는 CIC2018 8,666,430행, ToN 4,669,119행이며 별도 예산 참고 기준이다. Expert·route까지 포함한 제안 모델의 학습 예산이 100k와 같다고 표현하지 않는다.

비용은 RTX 4090에서 최대 2개 worker 병렬로 측정했다. 시간은 자원 경합을 포함한 경과 시간이며 단독 latency 또는 고유 속도 배수로 인용하지 않는다. LoCalPFN fit에는 validation이 포함되며 추론에는 kNN 검색이 포함된다. DistPFN은 Global 추론을 공유하고 prior 보정에 데이터별 약 0.19초 추가된다. RAM·처리량 및 클래스별 TP/FP/FN은 0908 HTML 상단에 제공한다.

## 남은 실험과 해석

- 현재 seed 43, K=4 bank의 S/V ablation 수행 필요
- D_route에서 gate 학습, validation에서 후보·임계값 선택, test로 정책 선택 금지
- Expert의 담당 residual 영역은 정답을 사용한 진단, 입력 유사군은 입력 표현·Global 확률로 배정한 범위
- 서로 다른 범위의 F1 차이로 일반화 손실을 직접 계산하지 않고, 각 범위 안에서 Global과 expert를 비교
- CIC Web: Global의 높은 recall을 보존하면서 FP를 줄이는 expert 선택·채택 기준 확인
- ToN: Scanning 개선 후보의 이득과 MitM 등 다른 클래스의 훼손을 함께 점검
- 호출·승인이 없으면 후보 부재, scorer 누락, verifier 기각을 분리하여 수정 방향 결정
- 이전 seed 42, K=8/7 S/V 수치를 현재 bank의 ablation으로 재사용하지 않음

목요일 10시 이후 결과를 고정하고 16시까지 표·숫자·레이아웃 검수 및 한국어 PPTX·PDF를 완성한다. 단일 seed와 이미 관찰한 development holdout에 대한 예비 결과이므로 논문 최종 주장에는 반복 seed와 독립 검증이 추가로 필요하다.

## 근거와 재생성

- `docs/research/20260911/dataset_quality_audit/`
- `data/derived/{cic2018,toniot}_conflict_free_fixed_split_*/class_counts.csv`
- `tabpfn/results/20260928_exp59_residual_oracle_s43/expert_capability/capability.json`
- `tabpfn/results/20260929_exp61_sota_local_s43/`
- `docs/research/20260930/exp61_report_data.json`
- `scripts/build_exp61_report.py` → `scripts/sync_html_reports.py`
- `python docs/slides/make_1002_draft.py --fill-sota`

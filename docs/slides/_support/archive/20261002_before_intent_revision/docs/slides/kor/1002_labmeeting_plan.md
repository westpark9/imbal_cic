# 10월 2일 랩미팅 한국어 발표 구성

갱신: 2026-10-02 00:50. **주장 우선 재구성**(사용자 방향): 의도·목표 2장을 앞에 두고, 9·10장을 후속 접근 feasibility 결과와 시간순 출현 시나리오(EXP68 첫 결과)로 교체. 생성: `python docs/slides/revise_1002_ko_v2.py`(입력은 `archive/20261002_before_claim_first/`의 10장 덱). 3~10장은 10-01 덱 그대로. 영어판 미갱신.

새 흐름: 1 왜 TabPFN인가(재학습 없는 적응, 연구 질문) → 2 목표는 tail 회수(benign 혼동 원인 진단) → 3 상충 라벨 → 4 정제 분포 → 5 구조 → 6 context 구성 → 7 expert 성능 → 8 S/V → 9 SOTA → 10 비용 → 11 후속 네 접근 feasibility → 12 시간순 시나리오.

---

# 10월 2일 랩미팅 한국어 발표 구성

갱신: 2026-10-01. EXP61 SOTA와 EXP63 K sweep·S/V 결과 반영. 사용자 요청에 따라 **한국어 PPTX·PDF만 갱신**, 영어 초안은 보존.

- 파일: `1002_labmeeting_ko_draft.pptx`, `1002_labmeeting_ko_draft.pdf`
- 생성: `python docs/slides/revise_1002_ko.py`
- 적용 지침: `docs/slides/지침.txt`, KENTECH 템플릿, 사용자 편집본 디자인
- 후속 네 접근: `docs/research/20261001/context_followup_approaches.md`
- 변경 전 자료: `docs/slides/archive/20261001_before_current_results/`

## 발표 흐름

데이터 검토 → 모델 구조 → anchor + residual context → expert 평가 → S/V 결과 → SOTA 성능·비용 → 후속 네 접근

모든 시행착오를 나열하지 않고 현재 구성과 확인된 결과를 전달한다. Anchor + residual block은 context를 구성하는 한 가지 시도이며 최종 방법으로 확정하지 않는다. 이전 A/B 평가 범위를 자체 역량의 두 관점으로 해석한 문구와 실험 전 TBD는 제거했다.

| 장 | 역할 | 내용 |
|---|---|---|
| 1 | 동일 feature를 공유하는 클래스 | 0911 데이터 검토 기반 원래 라벨 × 동일 feature의 다른 라벨 비율, 네 데이터의 전체 상충 비율 |
| 2 | 정제 이후 클래스 분포 | CIC2018·ToN의 정제 전후 행 수·비중, 정제 후 표본 수 내림차순 |
| 3 | Global과 residual expert 구조 | 훈련 풀 역할, context로 expert 구성, Scorer 호출·Verifier 채택 |
| 4 | 첫 context 구성 실험 | 공통 anchor + residual block, K=2/4/6/8 및 실제 context 행 수 |
| 5 | 동일 test의 Global·expert 성능 | K=4 전체 클래스별 F1, test 표본 수 내림차순, 동일 평가 표본군 |
| 6 | S/V 적용 결과 | 모든 K의 최종 Macro-F1·정책상 호출률·채택률, CIC2018 호출 0·ToN 호출 100% |
| 7 | SOTA와 현재 모델 성능 | 기존 여섯 방법에 현재 K=4 S/V 포함 결과 추가, 학습 정보 차이 명시 |
| 8 | SOTA 측정 비용과 현재 모델 호출 | 실제 SOTA 시간·GPU, 현재 모델 조건부 추론 시간은 별도 측정 예정 |
| 9 | 후속 접근 1·2 | Benign 다양성, LITO 방식 tail 합성 |
| 10 | 후속 접근 3·4 | Feature 표현 학습·embedding 추가, context 자체 최적화 |

## 현재 수치와 해석

| 구성 | CIC2018 Macro-F1 | ToN Macro-F1 |
|---|---:|---:|
| Global | 0.7824 | 0.6796 |
| K=2 + S/V | 0.7824 | 0.7636 |
| K=4 + S/V | 0.7824 | 0.7617 |
| K=6 + S/V | 0.7824 | 0.7677 |
| K=8 + S/V | 0.7824 | 0.7463 |

- K=4는 기존 대표 설정으로 SOTA 표에 사용, test 최고 K를 선택한 것이 아님
- CIC2018: 선택된 정책에서 모든 K의 expert 호출·채택 0, Global 예측 유지
- ToN: 모든 K에서 정책상 선택 expert 한 개 호출, verifier 채택은 일부 표본
- ToN K=4 Scanning F1: 0.0394 → 0.8587
- Expert 평가: 같은 전체 test를 각 expert가 분류하고 다른 클래스의 FP까지 포함하여 클래스별 F1 계산
- ToN Scanning: 모든 K의 모든 expert에서 test와 validation confirm F1 개선
- CIC2018 Infiltration: 모든 K에서 test F1 개선 expert 없음
- CIC2018 Web attack: 일부 test 개선은 있으나 동일 expert의 confirm 개선은 없음
- 현재 결과로 S/V 개별 기여나 context 변화의 인과적 원인을 단정하지 않음

## 데이터·비교 조건

- CIC2018 정제 전 20,115,529행 → 정제 후 14,755,417행
- ToN 정제 전 27,520,260행 → 정제 후 10,807,288행
- CIC2018 Web attack: Benign 공유 1,764 + Infiltration 공유 432 − 중복 429 = 제거 1,767, 잔존 771행
- 상충 라벨 그림은 모델 혼동행렬이 아니며 셀 합계를 제거 비율로 쓰지 않음
- 정제 test: CIC2018 3,042,473행, ToN 2,271,723행
- Global/XGB/BoostPFN/LoCalPFN/DistPFN은 같은 C0 100,000 train ID 사용
- 전체 train XGB와 현재 expert/S/V 모델은 추가 학습 데이터를 사용, 같은 100k 예산의 SOTA 우위로 주장하지 않음
- Global/DistPFN v3, BoostPFN/LoCalPFN v1, LoCalPFN fine-tuning 포함
- 단일 seed 43, 이미 관찰한 development holdout의 예비 결과
- K별 총 context 행 수는 다르며 공통 anchor 반복도 포함, 고정 총예산 K 비교가 아님
- SOTA 시간은 RTX 4090에서 최대 두 worker가 병렬 실행된 경과 시간
- 모델 호출률은 저장된 전체 expert 예측에 정책을 적용한 논리적 호출률이며 실제 조건부 latency와 구분

## 정렬·검증

- 상충 라벨 행렬: 정제 전 전체 표본 수 내림차순으로 행·열 정렬
- 정제 전후 분포: 정제 후 전체 표본 수 내림차순
- 클래스 성능: test 표본 수 내림차순
- 정수·소수 숫자 셀 우측 정렬, 고정폭 숫자 글꼴, 열별 소수 자릿수 통일
- 정제 전후 비중과 데이터셋별 GPU 값을 별도 열로 분리
- 입력 수치와 EXP63 CSV 대조, PPTX 기하·문체·숫자 정렬 확인, PDF 전 장 렌더링 검사
- 영어 파일은 변경 전 SHA-256과 대조하여 미변경 확인

## 근거

- `docs/research/20260911/dataset_quality_audit/`
- `data/derived/{cic2018,toniot}_conflict_free_fixed_split_*/class_counts.csv`
- `docs/research/20260930/exp61_report_data.json`
- `docs/research/20260930/exp63_capability_report_data.json`
- `tabpfn/results/20260930_exp63_k_sweep_s43/`
- `docs/slides/kor/1002_draft_qa/validation.json`

현재 한국어 자료가 최신본이며 영어판은 09-30 초안이다. 사용자 요청 전까지 영어판을 자동 갱신하지 않는다.

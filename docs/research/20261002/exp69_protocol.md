# EXP69 · 공격 클래스 출현에 따른 XGBoost 재학습과 TabPFN context 확장

2026-10-02, seed 43 단일 시드. 사용자가 승인한 **클래스 최초 출현 순서**를 보존한다. 전체 타임스탬프를 재생하는 온라인 실험이나 네트워크 환경 변화 실험으로 해석하지 않는다.

## 연구 질문

기존에 없던 공격 종류가 도입될 때, 동일한 소수 라벨 사례를 받은 TabPFN이 backbone 재학습 없이 공격을 구분하는가? 같은 누적 사례로 전체 재학습한 XGBoost와 신규 공격 성능, 기존 공격 유지, Benign 오탐, 갱신·추론 비용을 비교한다. TabPFN 우위를 미리 전제하지 않는다. 이 실험은 Global 수준의 시스템 선택 근거이며 expert/S/V 기여를 검증하지 않는다.

## 도입 순서

| 단계 | CIC2018 | ToN |
|---|---|---|
| 초기 | Benign + Brute force(2/14) + DoS(2/15–16) | Benign + Scanning(4/23) + DoS(4/24) |
| 1 | DDoS(2/20–21) | Injection + DDoS(4/25) |
| 2 | Web attack(2/22) | Password(4/26) |
| 3 | Infiltration(2/28) | XSS(4/27) |
| 4 | Bot(3/2) | Ransomware + Backdoor(4/28) |
| 5 | — | MitM(4/29) |

날짜는 CIC2018 2018년, ToN 2019년, UTC. 실제 사용 중인 NF-v3 원본 suite의 클래스별 최초 timestamp를 준비 단계에서 다시 계산하고 순서를 검증한다. 같은 날의 클래스는 묶어서 도입한다. 초기 단계는 정상+기존 공격 둘을 보유한 운영 시작점을 뜻한다.

## 라벨과 평가 데이터

- 기존 모순 제거 clean train/val/test row ID를 그대로 사용. Test는 이전 연구에서 이미 관찰한 개발 holdout이며 새로운 미관찰 최종 평가셋은 아니다.
- 초기 context: Benign 10,000행, 초기 공격 각각 2,000행, 총 14,000행. 이전 C0 100k 및 Global/Expert/Route 풀을 재사용하지 않고 clean train에서 새로 추출한다. 이는 첫 실행의 고정 규모이며 최적 규모로 주장하지 않는다.
- 신규 공격별 64개 라벨을 주는 조건을 먼저 실행. 추가로 16·256개 조건을 실행한다. 동일 seed의 클래스별 순열 앞부분을 사용하므로 라벨 집합은 포함 관계이며, 초기 사례는 세 조건에서 동일하다.
- 사례를 제거하지 않고 누적한다. 최종 context: CIC2018 14,000+4n, ToN 14,000+7n. 모든 도입 공격이 자연적으로 희소한 것은 아니다. 신규 공격의 **관측 라벨 예산**을 제한한 실험이다.
- 두 방법이 매 단계 정확히 같은 train/test IDs를 사용하는지 SHA 및 배열 digest로 검증한다. 미래 클래스의 행은 해당 단계의 fit, preprocessing, context에 포함하지 않는다. Validation을 사용한 하이퍼파라미터 탐색은 이번 실행에서 하지 않는다.
- 각 단계에서 등장한 클래스에 해당하는 **전체 기존 test 행**을 평가한다. 클래스별 원래 support를 보존하며 balanced test로 바꾸지 않는다. 전체 시간 인과성을 보장하는 분할은 아니며, 클래스 도입 순서를 보존한 통제 실험이다.
- Clean split에 남아 있는 동일 라벨·동일 feature의 train/test 중복은 유지한다. Context와 같은 vector hash를 가진 test 행 수와 이를 제외한 부가 지표도 저장한다. Hash는 기존 감사 자료이며 충돌 검증은 당시 자료의 한계를 따른다.

## 비교와 비용

- XGBoost: 각 단계 누적 사례로 처음부터 재학습. 300 trees, depth 8, lr .05, subsample/colsample .8, CUDA hist. 기존 SOTA 설정 유지. 추가 트리 방식은 이번 비교에 포함하지 않으므로 XGBoost 전체 갱신 전략에 대한 우위로 일반화하지 않는다.
- TabPFN v3: 기존 프로젝트 checkpoint, ensemble 4, `fit_with_cache`, `SUBSAMPLE_SAMPLES=None`. Backbone gradient update 없음. 매 단계 누적 context로 전처리와 cache를 재구성하며 그 비용을 `fit_context_or_retrain`에 포함한다. Context 추가 비용을 0으로 두지 않는다.
- GPU worker는 한 번에 하나씩 실행. 같은 하드웨어에서 갱신 시간, 전체 평가 추론 시간, 배치별 처리량, 프로세스 GPU 메모리·RSS를 기록한다. 첫 checkpoint 적재를 포함하는 cold fit과 이미 모델을 보유한 warm fit을 구분한다. 원격으로 이어 돌리면 GPU별 비용을 구분하고 서로 다른 GPU 간 방법별 시간을 직접 비교하지 않는다.
- Precision/Recall/F1, confusion matrix, 신규 클래스 평균 F1, 기존 클래스 평균 F1, 초기 클래스 평균 F1, 실제 Benign 중 공격 예측 비율을 저장한다. 평가 클래스 구성이 달라지므로 단계 간 전체 Macro-F1 상승을 그대로 적응 이득으로 해석하지 않는다.
- 새로운 클래스 출력이 없던 기존 모델의 recall 0 대비 상승을 우위의 근거로 삼지 않는다. 두 **갱신 후 모델**을 동일 평가 조건에서 비교한다. 라벨 확보 이전의 미지 공격 탐지는 후속 OOD 실험이다.

## 실행·이관·재현

`scripts/exp69_class_arrival.py prepare --out RUN`으로 독립적인 numpy 입력을 준비한 뒤 `run --out RUN`을 실행한다. `prepared.json`, 입력 SHA, 모델 checkpoint SHA, 패키지/GPU 정보, stage별 sample IDs·확률·지표·비용을 보존한다. 완료 marker가 있는 stage만 건너뛰어 재개한다.

`RUN/STOP_AFTER_STAGE` 파일을 만들면 현재 stage를 완료한 뒤 멈춘다. 실행 프로세스가 종료된 것을 확인한 뒤 파일을 제거하고 A100에서 재개할 수 있다. 코드·입력·완료 stage를 함께 이동해야 한다. 실제 명령과 첫 실측 ETA는 실행 기록에 남긴다.

이번 요청에서는 HTML/PPT를 수정하지 않는다. 결과는 v7 탐색 근거로 보존하고 검토 후 채택한다.

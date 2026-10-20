# 기존 제안 모델과 SOTA의 순차 공격 도입 평가

2026-10-14 사용자는 현재 제안 모델과 SOTA를 1014 순차 재생 시나리오에서 실행하도록 요청했다. 추가 확인에서 기존 10만 학습 사례를 유지하고 별도 정제 validation을 추가하는 안을 선택했다. 승인된 연구 버전 v7을 유지하며 실행을 EXP73으로 등록한다. 이번 실행 요청으로 HTML이나 논문을 자동 수정하지 않는다.

## 비교 범위

CIC2018·ToN, seed 43, 공격 내 자연비율·균등 배분, 최종 100,000 train 행, 1,024행 배치, 갱신 지연 없음 및 100/1,000행·초 재생을 유지한다. 초기 클래스와 이후 공개 순서, 전체 test 행의 단계별 배분은 EXP72와 같다. 모든 test 행은 정책별로 한 번 평가한다. 실제 수집 timestamp나 환경 drift를 재현한다고 주장하지 않는다. 갱신과 서비스는 독립 자원 가정이며 GPU 실측은 단독 worker로 수행한다.

비교는 XGB, TabPFN-v3, 현재 제안 모델(K=4, anchor+residual bank, S1V0), BoostPFN, LoCalPFN, DistPFN이다. XGB와 TabPFN은 train IDs와 입력 순서, 모델 설정, test 순서가 정확히 같아 EXP72 결과·비용을 재사용한다. 추가 validation은 같은 시점에 모든 방법에 제공 가능한 자료로 기록하지만 고정 설정의 두 단일 모델과 BoostPFN·DistPFN은 사용하지 않는다. 재사용 자체를 갱신비용 0으로 기록하지 않는다.

## 학습과 validation

기존 10만 train 행과 입력 순서를 그대로 유지한다. 제안 모델 내부에서는 Global/expert/route를 약 50/25/25%로 나눈다. 같은 feature의 행은 한 역할에만 배정하며 이미 공개된 행의 역할은 후속 단계에도 유지한다. 역할 배정은 현재 공개된 사례만으로 결정한다. Global은 Global 역할의 행을 context로 사용하며 expert와 S/V는 분리된 역할에서 각각 구성·학습한다. 모든 방법이 동일한 train 라벨 접근 예산을 받는 비교이며, 제안 모델의 Global context와 단일 TabPFN의 context가 같다고 주장하지 않는다. 제안 모델 자체의 Global 예측도 별도로 기록해 expert/S/V의 추가 교정·훼손을 구분한다.

추가 validation은 기존 정제 validation split의 EXP62 `cache/cal_X,y,ids,hash`에서 특징과 정답만 가져온다. 예전 예측·embedding·정책·임계값은 재사용하지 않는다. 현재 학습에 나타난 feature와 겹치는 행을 제외하고, 클래스 도입 시점에 해당 클래스의 사례를 정해진 seed로 선택한다. 클래스당 상한은 `ceil(10000 / 전체 클래스 수)`이며 가용량 부족은 보충 복제하지 않는다. 후속 학습과 feature가 겹치게 되는 validation 행은 제외하고 재보충하지 않는다. 미래 클래스의 validation은 조기 공개하지 않는다. 최종 실제 validation은 CIC2018 자연 8,211행·균등 8,047행, ToN 자연 8,668행·균등 8,605행이다. 개별 단계 ID와 클래스별 수는 실행의 plans/role IDs에서 추적한다.

LoCalPFN은 제공된 validation으로 원 구현의 모델 선택을 수행한다. 제안 모델은 같은 validation을 feature 그룹 단위로 tune/cal 약 40/60%로 분리한다. tune은 Global residual의 temperature 선택, cal은 S/V 임계값 선택에 사용한다. 기존 raw 시간에 따른 cal select/confirm 재분할은 이번 클래스 순서 시나리오에서는 적용하지 않으며, train/tune과 feature가 분리된 cal 전체로 임계값을 선택하고 고정 test로 평가한다. 이 변경은 모델 구조 변경과 구별되는 평가 적응 사항이다.

## 현재 제안 모델

EXP62/63의 K=4 구조를 기준으로 한다. 공통 anchor는 Global pool에서 클래스당 최대 2,000개를 선택한다. Expert pool에서 Global의 확률·embedding PCA 16차원·정답을 사용한 잔차와 균형 NLL을 구성하고 residual 가중 KMeans로 네 군집을 만든다. 특화 block 상한은 기존 186,000행이며 이번 expert pool 규모에서는 각 군집 전체가 상한 이하여서 원 diversity 선택의 전체 선택 분기에 해당한다. 정답이 들어간 residual은 학습 시 expert 구성에만 사용한다.

S1은 route에서 클래스 균형 decision gain(+1 교정/−1 훼손/0 중립) 회귀, V0는 비가중 NLL gain의 0.25 분위수 회귀다. EXP51의 pre/post 특징 구성·S1V0 학습 함수·임계값 탐색·정상 오탐 증가 0.0005와 보호 클래스 제약·최소 교정/훼손 증거 30건을 재사용한다. 기존 지정 tail이 아직 등장하지 않은 초기 단계에서는 tail 평균의 빈 집합을 피하도록 macro와 같은 전체 클래스 제약을 적용하고 그 사실을 기록한다. Test를 보고 threshold를 바꾸거나 S/V를 강제 활성화하지 않는다.

각 단계마다 Global, residual bank, experts, S/V, validation 선택을 새 누적 사례로 갱신하고 이 준비 과정을 갱신비용에 포함한다. 서비스에서는 S가 선택하고 호출 임계값을 넘은 행에 대해서만 해당 expert를 실제 추론하고 V를 계산한다. 전역 fallback 정책이면 expert를 추론하지 않는다. 전체 expert를 미리 추론한 비용을 sparse 운영비용이라고 기록하지 않는다. 호출·수락률, Global 대비 교정·훼손, 클래스별 및 단계별 성능, 임계값 선택 사유를 저장한다.

## SOTA 적용과 비용

BoostPFN은 기존 v1 기반 포트의 50 learners, learner당 500 context, 원 boosting 결합을 유지하고 단계별 재구성한다. LoCalPFN은 기존 v1 포트의 context 1,000, 21 epochs × 30 steps, validation 모델 선택과 exact L2 serving retrieval을 유지한다. 원 모델과 비교군의 backbone 차이를 기록한다. LoCalPFN serving index 구축비도 갱신비용에 포함한다.

DistPFN은 동일 train으로 이미 fit된 EXP72 TabPFN-v3 상태와 갱신비용을 재사용하되 예측 확률을 새로 계산한다. 보정에 필요한 posterior 평균은 현재 수집 완료된 배치만으로 계산하고 전체 미래 test 분포를 사용하지 않는다. 기반 모델 비용과 prior 준비 및 posterior 보정비용을 포함한다.

초기 구축과 후속 갱신을 분리한다. 모델 저장·복구, 실험 로그 기록은 실험 overhead로 분리한다. 배치 서비스시간에는 입력 gather, 특징 변환, 모델 추론과 실제 S/V 호출 경로를 포함한다. 모든 입력을 끝까지 처리하므로 응답 지연은 별도 지표이며 늦은 응답 자체를 오답으로 계산하지 않는다. 단일 GPU에서 여러 worker를 동시에 실행해 비용 측정에 경합을 만들지 않는다.

## 검증과 실행 기록

소규모 합성 데이터에서 제안 모델의 저장·복구, sparse expert 호출과 같은 선택 expert의 직접 예측 일치, BoostPFN·LoCalPFN 실행과 정규화된 확률·반복 예측 일치를 확인했다. 데이터 검증은 기존 22개 단계의 train ID와 순서 일치, validation ID 분리, 역할 간 feature 그룹 분리, 미래 클래스 미사용을 확인한다. 완료 후 각 정책의 혼동행렬과 F1을 독립적으로 재계산하고 모델 준비 시점·배정 버전·S/V 교정/훼손 항등식을 검사한다.

원본 입력: `tabpfn/results/v7/exp71/20261014_103108_cumulative_100k_uniform_natural_s43_43a766d9`. 재사용 기준선: `tabpfn/results/v7/exp72/20261014_134735_replay_100k_xgb_tabpfn_s43_de8b4a78`. 코드: `scripts/v7/exp73/`. 결과와 로그는 관리 실행 폴더에 저장한다. EXP72의 원 결과는 수정하지 않는다. 시나리오의 문헌 근거와 실제 시간 재현과의 차이는 [EXP72 프로토콜](exp72_protocol.md)을 따른다.

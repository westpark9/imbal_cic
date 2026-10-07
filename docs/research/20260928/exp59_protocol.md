# EXP59 — 잔차 소속 oracle과 expert/context 역량

사용자 승인 범위: ToN-IoT·CIC2018, 단일 seed, 기존 결과 우선 재사용,
0918과 같은 제목의 새 HTML 탭 작성. 공통 완료본이 있는 seed 43, K=4를 사용한다.

## 실행 방식 확정

기존 EXP57의 전체 residual 상태가 없으며, 복원한 Global의 tune 확률과
residual clipping이 기존 실행과 달랐다. 그러므로 과거 expert 예측을 새
cluster에 연결하지 않고 **두 데이터셋 모두 seed 43 bank를 새로 계산**한다.
아래 과거 결과 재사용 조건을 충족하지 못한 경우에 해당한다.

기존 분할·Global context ID·anchor ID·expert pool·test ID는 그대로 유지한다.
실제 Global과 특화 block·모든 expert 예측은 한 새 실행 안에서 계산한다.
Global fit state, PCA, 전체 residual 중심·정규화·test 표현·배정을 이번에는
저장한다. 평가에 필요한 raw 예측·AP·저오탐 recall만 계산하고 반복 bootstrap,
추가 확률 보정, 운영점 전이, scorer/verifier 학습은 생략한다.

## 데이터와 고정 조건

- EXP57 `20260922_143529_exp57_expert_quality_s42_44/{cic2018,toniot}_s43`의
  Global context·anchor·expert pool·test ID를 사용하고 bank 예측은 재계산한다.
- 원래 train/val/test 소속을 보존한 모순 제거 데이터다. 새로운 독립 test가 아니다.
- Train 내 D_global/D_expert/D_route = 약 50/25/25,
  val 내 D_tune/D_cal = 약 40/60. 클래스×시나리오 내부 시간순 분할.
- 세 context 조건: 현재 residual 선택, 클래스별 행 수 일치 무작위,
  전체 행 수 일치 균형 무작위. Seed·K·anchor·global·평가 소속을 조건 간 고정한다.
- Scorer/verifier를 추가 학습하지 않는다. 이전 bank의 S/S+V 수치를 결합하지 않는다.

## 잔차 소속 oracle

1. D_expert에서 학습된 전체 residual signature와 cluster 중심을 고정한다.
2. Test의 (x,y)로 동일 signature를 계산한다:
   z, global p, onehot(y)-p, log(1+clipped balanced CE).
   정규화·가중치·클리핑은 학습값을 그대로 사용한다.
3. 전체 signature의 최근접 중심 k에 지정된 expert k의 원래 argmax를 사용한다.
4. Expert 정오에 따른 재선택과 global fallback은 없다.

이것은 정답 residual 소속을 아는 조건부 평가이며, 임의 선택 정책의 상한이 아니다.

## 상수 expert 대조

동일한 test 잔차 배정에서 각 영역마다 클래스 하나만 출력하는 모든 C^K개
매핑을 열거하여 전체 macro-F1이 최대인 상수 매핑 C*를 구한다.
클래스별 독립 최고값을 조합하지 않는다.

`Delta_O = MacroF1(oracle) - max(MacroF1(global), MacroF1(C*))`

클래스별 상수 expert는 양의 Delta_O를 얻지 못한다. Delta_O는 이 특정
상수 설명과 global을 넘는 조건부 이득이며, 정답을 쓰지 않는 운영 성능을
보장하지 않는다. 식별력은 전체 test AP·동일 FPR recall로 별도 확인한다.

## Expert 평가

- 전체 test: 모든 expert×모든 클래스의 TP/FP/FN·precision/recall/F1,
  AP, R@FPR≤0.1%. 새 실행의 raw probability로 계산한다.
- 동일한 잔차 소속별 test 부분집합: global·모든 expert·상수 비교를
  실제로 다시 계산한다. 전체 클래스 macro-F1과 영역 내 등장 클래스
  macro-F1을 별도 저장하고 화면에서는 동일 영역 안에서만 비교한다.
- 단일 클래스 영역은 표본을 oracle 집계에서 제거하지 않지만,
  영역의 높은 accuracy를 분류 능력의 증거로 쓰지 않는다.
- Context 변경 시 oracle 소속을 다시 만들지 않는다.

## 재사용을 위한 복원 검증

EXP57에는 전체 residual 중심·표준화 및 test 특징 표현이 저장되어 있지 않다.
원래 실행의 소스·환경·seed로 global/특징/mining을 복원하고 expert fit은 생략한다.

재사용 전 검증:

- Global·anchor·expert pool·test ID와 test label 동일.
- Global tune 확률 최대 절대 차이 ≤ 1e-6.
- 저장된 관측 가능 centroid 부분 최대 절대 차이 ≤ 1e-5.
- 네 expert의 특화 context ID 배열 완전 동일.
- 기존 관측 가능 test 영역 배정 완전 동일.
- 각 expert의 저장 예측 = 저장 확률 argmax, 기존 F1 재계산 일치.

검증을 통과한 결과만 기존 예측과 연결한다. 전체 residual 상태와 배정을
이번 결과에 저장하여 다음 평가에서 다시 복원하지 않게 한다.

## 보고서

새 원본: `lablog/html_report/scorer_verifier_target_0928.html`.
본문 h1은 0918과 같은 `Expert 역량을 검증하는 평가`.
통합본 sidebar 제목도 0918과 동일하게 유지하고 날짜로 구분한다.

순서: Global의 한계 → 잔차 소속 oracle/상수 효과 → expert 식별력/context 대조
→ residual 소속별 수치 → 개선 방향·실행/재사용 근거.

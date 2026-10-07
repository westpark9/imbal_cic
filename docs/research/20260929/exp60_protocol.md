# EXP60 — 무작위 / 가까운 상대사례 context

요청: CIC2018·ToN에 두 상대사례 조건을 실행한다. 실험 실행만 요청되었으므로 HTML은 자동 변경하지 않는다.

## 고정 조건

- EXP59 seed 43, K=4, Global fitted checkpoint, 공통 anchor, D_expert ID, test ID/label, PCA와 residual scaling/centroid를 재사용한다.
- EXP59 현재 context와 raw 예측은 기존 기준으로 사용한다. Scorer/verifier는 이번 비교에서 새로 학습하지 않는다.
- Expert가 받는 입력은 원래 feature와 정답이다. Residual은 context 선택에 사용하며 새로운 입력 feature로 붙이지 않는다.
- 공통 anchor는 클래스별 최대 2,000개 표본이다. 오류만 선택한 것이 아니다. 기존 특화 block도 오류만으로 제한하지 않으며, residual clustering 및 오류 크기·다양성 기준으로 선택되었다.

## 추가할 두 조건

1. 기존 특화 block의 80%를 클래스 구성에 비례해 유지한다(원래 존재한 클래스 최소 1개 유지, 고정 seed). 두 조건이 동일한 core를 사용한다.
2. 나머지 20%(반올림)의 자리를 상대사례에 배분한다. 각 expert의 전체 context 크기는 EXP59와 동일하다. 이는 단일 preliminary 고정 비율이며 최적값이 아니다.
3. Core에서 중복 feature를 제외하고 클래스별로 나눈 대표 query를 선택한다. 클래스별 query 예산은 균등하게 배분하며 residual 크기에 비례한 고정-seed 추출을 사용한다. 전체 query 목표는 512개(클래스별 올림 때문에 최대 C−1개 초과 가능)다.
4. 후보는 D_expert 전체에서 검색한다. Anchor/core와 feature가 같은 후보를 제외하고 동일 feature 후보는 1개로 제한한다. 두 조건의 후보 집합은 같다. Test·validation은 context 선택에 사용하지 않는다.
5. `counter_near`: 학습 때 고정한 z와 Global 확률의 표준화·block 가중치를 사용한다. 각 query와 실제 정답이 다른 후보의 정확한 L2 최근접 이웃을 찾고, query 순환 방식으로 중복 ID 없이 채운다. 같은 residual cluster 소속일 필요는 없다. Global이 맞힌 후보도 허용한다.
6. `counter_random`: near의 각 query–상대 클래스 조합을 유지하되, 같은 후보 집합의 해당 클래스에서 무작위로 비복원 추출한다. 두 조건은 core, anchor, 전체 크기, 클래스별 개수, query–상대 클래스 조합이 같다. 상대사례 ID의 선택 규칙만 바뀐다.
7. 기존 context와 신규 context의 비교에는 80/20 재배분 효과가 포함된다. 경계 근처 사례 선택의 효과는 두 신규 조건 간 비교로 판정한다.

## 실행·재현

- 기존 Global의 저장된 fitted state를 복원하며 재학습하지 않는다. 저장된 test 예측·PCA 표현 일부와 복원 결과를 대조해 체크포인트의 동일성을 확인한다. 이 확인은 context 선택이나 튜닝에 사용하지 않는다.
- 미저장 D_expert Global 확률·embedding만 계산해 캐시한다. 이것은 신규 expert context를 선택하기 위한 train 측 계산이다.
- 두 조건 × 4 expert × 2 dataset = 신규 expert 16개. 동일 estimator 설정과 seed 사용. 기존과 동일한 raw argmax를 채점한다.
- GPU 충돌을 피하도록 CIC2018 다음 ToN 순차 실행. Model별 완료 파일, 원시 확률, 예측, context ID, 상대사례 pair, 선택 감사, 부분 집계를 보존한다.

## 평가

- A: EXP59와 동일한 정답 residual 영역에서 클래스별 F1·precision·recall·TP/FP/FN·Global 오류 교정/정답 훼손.
- B: EXP59와 동일한 입력 기반 최근접 배정에서 같은 지표와 실제→예측 혼동 쌍.
- 전체 test: 각 expert와 두 가지 고정 배정의 Macro-F1·클래스별 성능. A/B는 다른 평가 집합이므로 서로 직접 성능 차이로 해석하지 않는다.
- 주된 판단: 교정 능력을 유지하면서 FP와 정답 훼손을 줄이는가. 클래스별 분포와 실제 TP/FP를 함께 읽고, 합계 교정−훼손만으로 모델을 고르지 않는다.
- 기존 test는 이미 관찰한 development holdout이다. 이 실험은 맹검 최종 검증이나 다중 seed 재현성 근거가 아니다.

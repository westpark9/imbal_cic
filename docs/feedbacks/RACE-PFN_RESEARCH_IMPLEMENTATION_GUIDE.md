# RACE-PFN 연구 및 구현 가이드

## Residual Context Experts와 시간순 Context Adaptation을 이용한 강건한 IDS

**문서 목적:** 이 문서는 RACE-PFN의 최종 연구 목표, 핵심 가설, 알고리즘, 구현 순서와 검증 기준을 하나의 연구 설계로 정리한다. 각 구성요소는 별개의 기능을 추가하기 위한 것이 아니라, `Global failure structure → complementary contexts → conditional utility routing → bounded-memory temporal adaptation`이라는 하나의 논문 논리를 실증하기 위해 배치한다.

---

## 1. 우리가 최종적으로 만들려는 시스템

우리가 만들려는 것은 단순히 TabPFN으로 XGBoost보다 높은 IID 분류 성능을 내는 모델이 아니다.

최종 목표는 다음과 같다.

> 전체 트래픽을 처리하는 강한 Global PFN을 기본 행동으로 유지하면서, Global이 반복적으로 실패하는 long-tail 및 temporal traffic regime을 자동으로 찾아 specialist contexts를 만들고, 실제 이득이 기대되는 입력에 대해서만 specialist 하나를 선택적으로 호출하며, 소량의 최신 검증 라벨로 context memory를 시간순으로 갱신하는 적응형 IDS를 구현한다.

이 시스템의 핵심 원칙은 네 가지다.

1. **Global-first:** 모든 입력은 먼저 Global PFN이 처리한다.
2. **Residual specialization:** Expert는 공격 이름이 아니라 Global의 구조적인 실패에서 만든다.
3. **Sparse and conservative override:** 필요한 경우에만 top-1 expert를 호출하고, 충분한 이득이 확인될 때만 Global 예측을 교체한다.
4. **Context adaptation:** 새로운 시점에서는 PFN backbone을 재학습하지 않고 제한된 context memory를 갱신한다.

### 1.1 논문의 중심 가설

이 연구가 검증하려는 중심 가설은 다음과 같다.

> Global PFN의 오류는 무작위가 아니라 feature regime, posterior와 confusion direction에 따라 반복되는 구조를 가지며, 이 구조를 대표하는 full-label contexts는 Global과 상보적인 prediction을 제공할 수 있다. 이 상보성을 global-relative utility로 선택하고 최신 검증 라벨로 갱신하면, 일반 성능을 보존하면서 long-tail과 temporal shift에서 더 안정적인 IDS를 만들 수 있다.

### 1.2 논문의 핵심 기여

논문 기여는 다음 세 축으로 유지한다.

1. **Residual-conditioned context construction**  
   Global PFN의 held-out, class-balanced failure signature에서 상보적인 full-label specialist contexts를 자동 구성한다.

2. **Cost-sensitive and calibration-controlled routing**  
   Context proximity가 아니라 Global 대비 조건부 utility를 기준으로 top-1 specialist만 호출하고, 보정된 predictive-gain 하한이 양수일 때만 prediction을 교체한다.

3. **Few-label bounded-memory temporal adaptation**  
   Frozen PFN과 고정 메모리 조건에서 최신 verified labels로 context를 갱신하고, 효과를 다음 temporal window에서 검증한다.

OOD/unsupported traffic은 위 세 기여가 충분히 검증된 뒤 다루는 보조 확장으로 둔다.

---

## 2. 쉬운 비유

RACE-PFN을 병원에 비유하면 다음과 같다.

- **Global PFN:** 대부분의 환자를 먼저 진료하는 일반의
- **Residual expert:** 일반의가 반복적으로 어려워하는 특정 증상에 강한 전문의
- **Proposal router:** 전문의를 부를 가치가 있는지 판단하는 시스템
- **Verifier:** 전문의의 진단으로 최종 결과를 바꿔도 되는지 확인하는 시스템
- **Context refresh:** 최근 환자 사례를 전문의의 참고 자료에 반영하는 과정

중요한 점은 전문의의 이름을 사람이 미리 `Web 전문가`, `DoS 전문가`로 정하지 않는다는 것이다. Global이 실제로 어떤 입력에서 어떤 방향으로 실패하는지를 분석하여 전문 분야를 자동으로 발견한다.

---

## 3. 우리가 하지 않는 것

다음은 이 연구의 목표가 아니다.

- TabPFN backbone을 IDS 데이터로 처음부터 다시 학습하는 것
- 모든 입력에 모든 expert를 실행하는 dense ensemble
- 공격 family를 사람이 미리 나누어 expert를 만드는 것
- 가장 가까운 cluster의 expert를 무조건 사용하는 것
- Expert를 호출했다는 이유만으로 그 결과를 채택하는 것
- Final test 결과를 반복 확인하며 알고리즘을 수정하는 것
- 미래 window의 라벨로 context를 갱신한 뒤 같은 window를 다시 평가하는 것
- Label-free temporal adaptation 또는 새로운 공격 attribution까지 한 번에 주장하는 것

Tabular foundation model의 parameter는 고정한다. 우리가 학습하거나 갱신하는 것은 labeled context와 작은 router/verifier다.

---

## 4. 전체 파이프라인

### 4.1 Offline construction

```text
Pretrained frozen PFN
        │
        ▼
전체 class를 담당하는 Global context C0 구성
        │
        ▼
Held-out 데이터에서 Global의 per-sample failure 측정
        │
        ▼
Failure signature 구성
  [feature regime, global posterior,
   confusion direction, residual severity]
        │
        ▼
Residual regime clustering
        │
        ▼
각 regime에서 다양성을 보존한 specialist context 생성
        │
        ▼
Global과 모든 expert의 paired predictive gain 계산
        │
        ├── Shared pre-call utility scorer 학습
        └── Shared post-call lower-bound verifier 학습·보정
```

### 4.2 Online inference

```text
입력 x
  │
  ▼
Global PFN 실행
  │
  ▼
Pre-call scorer가 모든 expert의 예상 net utility 평가
  │
  ├── 기준 이하 → Global 결과 반환
  │
  └── 가장 유망한 top-1 expert 하나만 실행
                       │
                       ▼
             Post-call verifier가 보수적 gain 하한 계산
                       │
              ┌────────┴────────┐
              │                 │
          하한이 양수        하한이 불충분
              │                 │
              ▼                 ▼
        Expert 결과 채택     Global 결과 유지
```

### 4.3 논문에서 입증해야 할 증거 사슬

연구는 다음 순서로 성립해야 한다.

```text
① Global failure가 구조를 갖는가?
        ↓
② 그 구조로 만든 contexts가 Global과 실제로 상보적인가?
        ↓
③ Pre-call scorer가 이 상보성을 호출 전 예측할 수 있는가?
        ↓
④ Post-call verifier가 잘못된 교체를 줄일 수 있는가?
        ↓
⑤ Sparse execution이 dense evaluation보다 충분히 효율적인가?
        ↓
⑥ 제한된 최신 라벨로 다음 temporal window 성능을 회복하는가?
```

이 순서는 구현 우선순위이기도 하다. 특히 ②의 oracle complementarity가 충분하지 않다면 router를 복잡하게 만드는 것이 아니라 residual representation과 context selection을 먼저 강화한다. 반대로 oracle gap은 크지만 실제 성능 회수율이 낮다면 그때는 routing과 calibration이 병목이다.

---

## 5. 데이터 분할: 최종 6-way split

이 연구에서는 데이터 재사용으로 인한 과대평가를 막는 것이 매우 중요하다. 최종 구현은 다음 여섯 역할을 분리한다.

| Split | 역할 | 금지 사항 |
|---|---|---|
| `D_global` | 전처리 fitting, Global context, common anchor 구성 | Expert/router/test 결과를 보고 다시 선택하지 않음 |
| `D_expert` | Held-out Global failure 계산, residual regime 및 specialist block 구성 | Global context에 포함하지 않음 |
| `D_route` | 고정된 Global·expert의 offline prediction, router/verifier 학습 | Threshold와 최종 성능 선택에 직접 사용하지 않음 |
| `D_tune` | K, context budget, class weight, prior correction, cost, model capacity 선택 | Final report용 test로 사용하지 않음 |
| `D_cal` | Probability calibration, lower-bound correction, proposal/override threshold 고정 | Model 구조 선택에 반복 사용하지 않음 |
| `D_test` | 모든 알고리즘을 동결한 후 최종 1회 평가 | 결과를 보고 다시 알고리즘 수정하지 않음 |

권장 초기 비율은 final test를 제외한 pre-test pool에서 다음과 같다.

```text
D_global : D_expert : D_route : D_tune : D_cal
   40%   :    20%   :   15%   :   10%  :  15%
```

비율 자체보다 중요한 것은 각 split의 역할을 섞지 않는 것이다. 각 class, day/session/capture group이 필요한 split에 충분히 존재하도록 조정해야 한다.

### 반드시 저장할 split 정보

```text
dataset
row_id
source_file
timestamp
group_id 또는 session/capture ID
fine_label
family_label
canonical_feature_hash
split
temporal_window
```

동일 flow, duplicate row, session, attack episode가 서로 다른 split에 들어가면 안 된다. Imputer, scaler, PCA 등 모든 전처리도 허용된 과거 데이터에서만 fit한다.

현재 반복해서 성능을 확인한 402만 행은 final test로 부르지 않는다. 이미 결과를 보고 알고리즘 개선에 사용했다면 development holdout이며, 별도의 untouched future block을 final test로 동결해야 한다.

---

## 6. Step 1 — 전체 class를 담당하는 Global PFN

먼저 `D_global`에서 전체 데이터 분포를 대표하는 context를 만든다.

\[
C_0=\operatorname{RepresentativeCoreset}(D_{\mathrm{global}},B_0)
\]

Global prediction은 다음과 같다.

\[
p_0^{\mathrm{raw}}(y\mid x)=F_\theta(C_0,x)
\]

- \(F_\theta\): pretrained frozen TabFM
- \(C_0\): 전체 class를 대표하는 labeled context
- \(B_0\): Global context budget

Global context를 단순 random sample 한 번으로 만들지 않는다. 기본 구현은 group-aware, class-aware medoid 또는 stratified k-center를 사용한다. Context row ID와 selection seed를 저장한다.

Global의 역할은 다음과 같다.

1. 일반적인 benign/head traffic 처리
2. 모든 expert가 불리할 때의 기본 행동
3. Expert 이득을 측정하는 기준선
4. 시간 변화 전후의 안정성 기준

---

## 7. Step 2 — Context-induced prior correction

Expert context마다 class 비율이 다르면 posterior 차이가 실제 전문성 때문이 아니라 context의 class prior 때문일 수 있다.

예를 들어 Web sample이 많이 들어간 context는 실제 feature 근거가 약해도 Web posterior를 과도하게 높일 수 있다. 현재 EXP22B에서 나타난 Web false positive도 이 문제와 관련될 가능성이 있다.

Context \(C_k\)의 smoothed prior를 다음처럼 저장한다.

\[
\widehat\pi_{k,c}
=
\frac{n_{k,c}+\alpha}{|C_k|+\alpha C}
\]

Raw posterior는 공통 reference prior로 정렬한다.

\[
\widetilde z_{k,c}(x)
=
\log(p_{k,c}^{\mathrm{raw}}(x)+\epsilon)
+\beta\left[
\log\pi_c^{\mathrm{ref}}-
\log\widehat\pi_{k,c}
\right]
\]

\[
\widetilde p_k(x)=\operatorname{softmax}(\widetilde z_k(x)/T)
\]

\(\beta\)와 \(T\)는 `D_tune`에서 선택하고 `D_cal`에서 확인한다. Prior correction 없음/있음은 필수 ablation이다.

---

## 8. Step 3 — Global의 per-sample failure 계산

`D_expert`는 Global context에 포함되지 않은 held-out 데이터다. 각 sample에 대해 Global의 class-balanced loss를 계산한다.

\[
r_i
=
\bar w_{y_i}
[-\log(\widetilde p_{0,y_i}(x_i)+\epsilon)]
\]

Class weight는 `D_global`의 class count만 사용한다.

\[
\bar w_c
=
\frac{n_c^{-\gamma}}
{\sum_d (n_d/N)n_d^{-\gamma}}
\]

이 정규화가 있어야 `benign weight=0.16`, `web weight=1133`처럼 큰 범위의 값을 명확하게 설명할 수 있다. 단순히 \(1/n_c^\gamma\)라고만 적으면 실제 수치와 식이 맞지 않는다.

극단적인 label noise가 clustering을 지배하지 않도록 residual을 clip한다.

\[
\bar r_i=\min(r_i,r_{\max})
\]

여기서 얻는 것은 Global Macro-F1 같은 단일 점수가 아니다. 각 sample에서 Global이 어떻게 실패했는지에 대한 정보다.

---

## 9. Step 4 — Residual failure signature

Embedding 위치와 loss 크기만 사용하는 초기 구성에서는 서로 다른 confusion pattern이 같은 cluster에 섞일 수 있다. 이를 분리하기 위해 posterior와 confusion direction을 residual signature에 포함한다.

최종 residual signature는 다음 네 가지를 결합한다.

\[
e_i=
\left[
\operatorname{norm}(z_i),
\alpha_p\widetilde p_0(x_i),
\alpha_e(\mathbf e_{y_i}-\widetilde p_0(x_i)),
\alpha_r\log(1+\bar r_i)
\right]
\]

각 항의 의미는 다음과 같다.

| 항 | 의미 |
|---|---|
| \(z_i=\phi(x_i)\) | 입력이 feature space의 어느 regime에 있는가 |
| \(\widetilde p_0(x_i)\) | Global이 어떤 class들을 가능성 있게 보았는가 |
| \(\mathbf e_{y_i}-\widetilde p_0(x_i)\) | 정답 대비 Global의 confusion direction |
| \(\log(1+\bar r_i)\) | 오류가 얼마나 심각한가 |

예를 들어 다음 두 sample은 모두 Global이 틀렸지만 다른 failure signature를 가져야 한다.

```text
Sample A
정답: Web
Global: Benign 0.85 / Web 0.08

Sample B
정답: Infiltration
Global: DoS 0.55 / Infiltration 0.35
```

초기 primary representation은 `D_global`에서만 fit한 robust scaler + PCA로 둔다. TabPFN embedding은 접근성과 안정성을 확인한 후 ablation으로 비교한다. 각 signature block은 clustering 전에 개별적으로 standardize해야 한다.

---

## 10. Step 5 — Residual regime clustering

Residual이 큰 sample이 cluster 형성에 더 큰 영향을 주도록 weighted robust clustering을 사용한다.

\[
\min_{\mu_1,\ldots,\mu_K}
\sum_{i\in D_{\mathrm{expert}}}
\bar r_i\min_k d(e_i,\mu_k)
\]

권장 순서는 다음과 같다.

1. Weighted k-medoids 또는 trimmed clustering
2. 비교용 weighted k-means
3. \(K\in\{2,4,8\}\) 후보
4. `D_tune`에서 oracle utility, expert 중복, memory/cost를 함께 고려해 K 선택

Cluster는 class membership이 아니다. 하나의 class가 여러 residual regimes로 나뉠 수 있고, 서로 다른 class가 같은 confusion pattern을 가지면 한 regime에 들어갈 수 있다.

---

## 11. Step 6 — 서로 다른 Context Expert 만들기

### 11.1 Common anchor

모든 expert가 전체 label space를 유지하도록 `D_global`에서 공통 anchor를 만든다.

\[
A=\operatorname{ClassAwareCoreset}(D_{\mathrm{global}},B_A)
\]

모든 specialist는 동일한 anchor row를 사용한다. Anchor에는 가능한 모든 class가 포함되어야 한다.

### 11.2 Diversity-aware specialist block

EXP22B가 드러낸 첫 번째 설계 병목은 다음 TopB 점수에서 발생했다.

\[
s_{ik}=r_i\exp(-d_{ik}^2/\tau)
\]

Residual scale이 거리항을 압도하면서 여러 expert가 유사한 high-loss rows를 선택했다. EXP22B에서 expert block의 class composition이 거의 같고 router target이 Expert 2에 94% 집중된 결과는, 다음 iteration에서 context diversity를 우선 강화해야 한다는 직접적인 설계 신호다.

최종 구현은 먼저 sample을 residual regime에 hard assignment한 뒤, 각 regime 내부에서 서로 다른 영역을 대표하는 row를 선택한다.

\[
S_k
=
\arg\min_{\substack{S\subseteq R_k\\|S|=B_S}}
\sum_{i\in R_k}
\bar r_i\min_{j\in S}\|e_i-e_j\|^2
\]

실제 구현은 weighted greedy k-center, weighted k-medoids 또는 facility-location approximation을 사용한다.

최종 expert context는 다음과 같다.

\[
C_k=A\cup S_k
\]

모든 expert는 같은 total budget을 가져야 한다.

```text
|Ck| = anchor budget + specialist budget
```

### 반드시 측정할 expert 다양성

- Context row-ID pairwise Jaccard overlap
- Expert prediction agreement/disagreement
- Expert별 gain vector correlation
- Expert별 positive-gain coverage
- Router target distribution
- Expert별 class composition과 context prior

Class count가 비슷하다고 같은 row가 들어갔다고 단정할 수 없다. 실제 row overlap과 prediction/gain correlation으로 clone 여부를 확인해야 한다.

---

## 12. Step 7 — Expert viability와 redundancy pruning

Residual cluster가 존재한다고 해서 그 context가 Global을 실제로 보완하는 것은 아니다.

`D_tune`에서 각 expert의 predictive gain을 계산한다.

\[
L_{i0}=\ell_{\mathrm{bal}}(\widetilde p_0(x_i),y_i)
\]

\[
L_{ik}=\ell_{\mathrm{bal}}(\widetilde p_k(x_i),y_i)
\]

\[
G_{ik}=L_{i0}-L_{ik}
\]

- \(G_{ik}>0\): Expert가 Global보다 predictive loss를 줄임
- \(G_{ik}\le0\): Global을 유지하는 편이 나음

다음 expert는 제거하거나 병합한다.

- Positive-gain sample이 너무 적음
- 다른 expert와 context row가 거의 동일함
- Prediction과 gain vector가 다른 expert와 거의 동일함
- 추가했을 때 oracle utility가 거의 증가하지 않음
- Context memory와 호출 비용에 비해 이득이 작음

Router를 복잡하게 만들기 전에 context bank 자체가 충분한 complementarity를 가져야 한다.

---

## 13. Step 8 — Offline paired utility matrix

최종 expert bank를 고정한 후 `D_route`에서 Global과 모든 expert를 offline으로 실행한다.

```text
각 row i에 대해:
Global loss L_i0
Expert 1 loss L_i1
...
Expert K loss L_iK
```

이 단계에서는 모든 expert를 실행해도 된다. Dense execution은 학습용 utility matrix를 만드는 offline 비용이며, 실제 배포 시에는 top-1만 호출한다.

`G_ik`는 causal counterfactual이 아니다. 같은 labeled row에서 Global과 Expert의 loss를 모두 관찰한 paired difference다. 논문에서는 다음 표현을 사용한다.

- Global-relative predictive gain
- Global-relative conditional utility
- Paired oracle utility target

`Counterfactual advantage`라는 표현은 피한다.

---

## 14. Step 9 — Post-call lower-bound verifier

Expert를 실행한 뒤에는 Global과 selected expert의 실제 posterior를 모두 사용할 수 있다.

\[
h_{\mathrm{post}}(x,k)=
[
\widetilde p_0,
\widetilde p_k,
\widetilde p_k-\widetilde p_0,
H(\widetilde p_0),H(\widetilde p_k),
m_0,m_k,d_k(x),q_k
]
\]

\(q_k\)는 expert descriptor다.

```text
cluster 중심과 분산
context prior
평균 residual
context 크기
호출 비용
```

Verifier는 expert별 별도 모델이 아니라 expert descriptor를 입력받는 shared model로 만든다.

Huber regression으로 평균 \(\widehat G\) 하나만 예측하고 `>0`을 적용하는 초기 verifier는 useful proof-of-concept이지만 prediction uncertainty를 충분히 반영하지 못한다. Primary verifier는 낮은 조건부 quantile을 예측하도록 확장한다.

\[
\widehat q_{\alpha,k}(x)
=v_\psi(h_{\mathrm{post}}(x,k))
\]

학습 loss는 pinball loss를 사용한다.

그 후 `D_cal`의 독립 데이터에서 one-sided correction을 적용해 보수적 lower bound를 만든다.

\[
\underline G_k(x)
=
\widehat q_{\alpha,k}(x)-Q_{1-\delta}
\]

최종 채택 조건은 다음과 같다.

\[
b(x,k)=\mathbf 1[
\underline G_k(x)>\tau_{\mathrm{post}}
]
\]

여기서 \(\tau_{\mathrm{post}}\ge0\)이며 다음 제약을 만족하도록 `D_cal`에서 고정한다.

- Harmful override rate upper bound
- Benign FPR 증가 제한
- 최소 accepted sample 수
- Coverage/utility trade-off

Verifier는 안전을 보장하는 장치가 아니라 calibration-controlled conservative rejection gate다. Calibration이 구현되기 전에는 `risk-controlled` 또는 `safe`라고 주장하지 않는다.

### Cost를 Verifier target에 넣지 않는 이유

Expert가 이미 실행된 뒤에는 호출 비용이 발생한 상태다. 따라서 최종 prediction 교체 여부는 순수 predictive gain을 기준으로 판단한다.

\[
G_{ik}=L_{i0}-L_{ik}
\]

Cost는 다음 proposal 단계에서만 사용한다.

---

## 15. Step 10 — Pre-call shared utility scorer

Proposal은 Expert를 실행하기 전에 계산 가능한 정보만 사용한다.

\[
h_{\mathrm{pre}}(x,k)=
[
\widetilde p_0(x),H(\widetilde p_0),m_0,
z(x),d_k(x),q_k
]
\]

각 query-expert pair에 동일한 shared scorer를 적용한다.

\[
\widehat U_k^{\mathrm{pre}}(x)
=s_\phi(h_{\mathrm{pre}}(x,k))
\]

기본 target은 cost-adjusted predictive utility다.

\[
V_{ik}=G_{ik}-\lambda_{\mathrm{cost}}c_k
\]

권장 최종 target은 verifier가 실제로 채택할 가능성까지 반영한 OOF realized policy utility다.

\[
U_{ik}^{\mathrm{OOF}}
=
b_{ik}^{\mathrm{OOF}}G_{ik}
-\lambda_{\mathrm{cost}}c_k
\]

Verifier가 거절하여 prediction gain은 얻지 못하고 호출 비용만 발생한 wasted call도 proposal이 학습하게 된다.

최종 proposal은 다음과 같다.

\[
k^*(x)=\arg\max_k \widehat U_k^{\mathrm{pre}}(x)
\]

\[
a(x)=
\begin{cases}
k^*(x), & \widehat U_{k^*}^{\mathrm{pre}}(x)>\tau_{\mathrm{pre}},\\
0, & \text{otherwise}.
\end{cases}
\]

Global은 별도의 `NO EXPERT` class로 강제 학습하지 않는다. 모든 expert의 예상 net utility가 기준보다 낮으면 자연스럽게 Global action \(a=0\)을 선택한다.

이 구조가 기존 `NO EXPERT/Expert 1/.../Expert K` hard classifier보다 좋은 이유는 다음과 같다.

- Expert 순서에 덜 의존함
- Expert descriptor를 통해 각 expert의 특성을 직접 반영함
- Expert bank가 수정되어도 동일한 scorer를 사용할 수 있음
- 특정 expert로 target이 몰리는 class imbalance를 완화함
- Cost와 call budget을 명시적으로 최적화할 수 있음

---

## 16. 최종 test-time algorithm

```text
Input: query x

1. p0_raw ← FrozenPFN(C0, x)
2. p0 ← PriorCorrectAndCalibrate(p0_raw, C0)
3. z(x)와 각 expert의 cheap distance/support를 계산한다.
4. 모든 expert k에 대해 pre-call utility ûk를 계산한다.
5. k* ← argmaxk ûk
6. ûk* ≤ τpre 이면 p0를 반환한다.                 [GLOBAL_ONLY]
7. pk*_raw ← FrozenPFN(Ck*, x)                    [전문가 1회 추가 호출]
8. pk* ← PriorCorrectAndCalibrate(pk*_raw, Ck*)
9. lower_gain ← CalibratedLowerBound(x, k*)
10. lower_gain > τpost 이면 pk*를 반환한다.       [EXPERT_OVERRIDE]
11. 아니면 p0를 반환한다.                         [EXPERT_REJECTED]
```

Specialist proposal rate를 \(\rho\)라고 하면 평균 PFN 호출 횟수는 \(1+\rho\)다. 그러나 호출 횟수를 wall-clock 배수로 간주하면 안 된다. Context cache, dynamic batching, context size, GPU transfer를 포함한 실제 latency를 측정한다.

---

## 17. Temporal context refresh

정적 expert bank만으로는 이미 관찰된 failure regime을 보완할 수 있지만, 실제 temporal drift에 대응하려면 context를 시간순으로 갱신해야 한다.

### 17.1 시간순 설정

\[
D_{\mathrm{history}}\rightarrow W_1\rightarrow W_2\rightarrow W_3
\]

Primary 조건은 다음과 같다.

- Frozen PFN backbone
- Frozen preprocessing
- Fixed K
- Fixed total context memory
- Window별 verified-label budget \(B\in\{0,50,200,1000\}\)
- 모든 adaptive baseline에 동일 verified row 제공

### 17.2 Strict prequential 순서

```text
For each window Wt:

1. 현재 시스템 Mt-1로 Wt 전체를 먼저 예측한다.
2. Prediction, routing, latency와 Wt 성능을 확정한다.
3. 사전 선언된 방법으로 최대 B개 label만 공개한다.
4. 공개된 sample의 Global residual과 failure signature를 계산한다.
5. 기존 residual regime에 할당한다.
6. Global recent context, common anchor, specialist recent block을 갱신한다.
7. Total memory budget은 유지한다.
8. 갱신된 Mt는 Wt+1에서 처음 평가한다.
9. 과거 reference set에서도 성능을 측정해 forgetting을 확인한다.
```

같은 window의 라벨을 사용해 context를 갱신하고 같은 window를 다시 평가하면 leakage다.

### 17.3 Context memory 구조

```text
Global context    = stable global block + recent global block
Common anchor     = stable anchor + recent anchor
Specialist block  = stable specialist block + recent specialist block
```

최근 sample을 무조건 FIFO로 넣지 않는다. Residual severity, recency, class/group balance와 diversity를 함께 고려한다. Total logical context와 unique storage budget은 모든 window에서 동일하게 유지한다.

---

## 18. EXP22B가 제공하는 설계 근거

EXP22B는 최종 결과가 아니라 핵심 가설을 점검한 development experiment다. 이 실험에서 확인한 설계 근거는 다음과 같다.

- Frozen Global PFN, full-label context experts, sparse top-1 call과 exact Global fallback의 end-to-end 실행 경로가 동작한다.
- Global Macro-F1이 `0.6700`에서 `0.7084`로 상승하여, context specialization으로 Global residual을 회수할 수 있다는 가능성이 확인되었다.
- Brute-force와 DoS에서 큰 개선이 나타나 residual-focused context가 Global과 다른 유용한 evidence를 제공할 수 있음을 보였다.
- Proposal rate `0.23`으로 평균 PFN 호출 수 `1.23`을 달성하여 sparse activation의 기본 가능성을 확인했다.
- Web precision 저하와 유사한 expert composition은 다음 성능 향상을 위해 어디를 우선 강화해야 하는지 명확히 보여준다.

EXP22B의 관찰을 다음 연구 설계로 연결한다.

| EXP22B 관찰 | 연구적 해석 | 다음 설계 |
|---|---|---|
| Global보다 Macro-F1 향상 | Residual contexts에 회수 가능한 signal이 존재 | Expert complementarity를 여러 seed와 stress setting에서 확대·검증 |
| Expert block 조성이 매우 유사 | Router보다 context construction이 현재 병목 | Full residual signature + hard regime assignment + diversity-aware coreset |
| Router target이 Expert 2에 94% 집중 | Hard K+1 target이 expert redundancy를 반영하지 못함 | Expert descriptor를 사용하는 shared utility scorer |
| Web F1과 precision 저하 | Context prior와 adoption uncertainty를 더 잘 통제할 필요 | Prior alignment + lower-quantile verifier + benign/classwise harm constraints |
| 제안 23%, 채택 3.4% | 유망한 sparsity와 함께 rejected/wasted call 최적화 여지 존재 | OOF realized-policy utility를 proposal target에 반영 |
| Brute-force·DoS 개선 | 일부 residual regimes는 실제 상보성을 가짐 | 해당 gain을 보존하면서 Web harmful transition을 줄이는 목적함수·calibration 적용 |

용어와 protocol도 최종 논문에 맞게 정리한다.

| 초기 표현/구성 | 연구 명세의 표현/구성 |
|---|---|
| `D_mine` | Specialist-building split인 `D_expert` |
| Global context 밖의 행을 `cross-fit`이라고 표현 | 실제 OOF가 없으면 `held-out residual` |
| `D_val` 하나에서 선택과 보정 | `D_tune`과 독립 `D_cal` 분리 |
| Cost-adjusted Δ를 proposal과 verifier 모두에 사용 | Proposal은 net utility, post-call verifier는 pure predictive gain |
| 반복 관찰한 402만 행을 final test로 표현 | Development holdout으로 사용하고 untouched future test를 별도 동결 |

다음 iteration의 목적은 EXP22B 결과를 설명하는 데 머무르지 않는다. Expert specialization과 conservative adoption을 강화하여, Global 대비 Macro-F1 개선을 유지하면서 tail F1과 Web precision까지 함께 개선하는 것이 목표다.

---

## 19. 연구 실행 우선순위

### Phase 0 — 데이터와 평가 고정

1. 6-way split manifest 작성
2. Duplicate/group/time leakage 검사
3. 현재 402만 행을 development로 재분류
4. Untouched final future block 동결
5. Context row ID와 모든 seed 저장

### Phase 1 — Expert complementarity 강화

1. Residual clip 및 block별 standardization
2. Full residual signature 구현
3. Weighted robust clustering
4. Hard regime assignment
5. Weighted k-center/medoid 기반 context selection
6. Jaccard, gain correlation, prediction disagreement 측정

이 단계에서 expert complementarity가 확인되기 전에는 router를 복잡하게 만들지 않는다.

### Phase 2 — Prior correction과 expert pruning

1. Context별 prior 저장
2. Common reference prior alignment
3. Prior correction 전후 balanced NLL/F1 비교
4. Positive-gain coverage가 낮거나 중복된 expert 제거

### Phase 3 — Shared utility router

1. Query-expert pair dataset 생성
2. Expert descriptor 구현
3. Shared scorer 학습
4. Cost-aware utility와 call-budget threshold 선택
5. Hard router, nearest-centroid, mean-gain scorer와 비교

### Phase 4 — Calibrated verifier

1. Shared lower-quantile verifier 학습
2. Grouped OOF verifier decision 생성
3. Independent one-sided calibration
4. Harmful override와 benign FPR 제약 아래 threshold 고정
5. Helpful/harmful/wrong-to-wrong 전이 분석

### Phase 5 — Temporal refresh

1. History→W1→W2→W3 구성
2. Zero-update baseline 확정
3. \(B=0,50,200,1000\) 동일 verified row 생성
4. Fixed-memory context-only refresh
5. 다음 window recovery와 forgetting 측정

---

## 20. 반드시 수행할 실험

### 20.1 Expert feasibility

Router를 만들기 전에 다음을 비교한다.

- Global context
- Equal-memory random contexts
- Feature-only/proximity contexts
- Full residual-signature contexts
- Residual + diversity contexts
- Per-sample label-informed loss oracle

핵심 질문은 하나다.

> 각 sample에서 Global보다 실제로 나은 context가 충분히 존재하는가?

Oracle gain이 작으면 router 문제가 아니라 context construction 문제다.

### 20.2 IID guardrail

- Known-class Macro-F1
- Fixed-tail Macro-F1
- Macro-AUPRC
- Balanced accuracy
- Benign FPR
- Calibration

RACE가 모든 IID setting에서 XGBoost를 반드시 이길 필요는 없다. 그러나 Global의 일반 성능과 benign safety를 크게 훼손하면 안 된다.

### 20.3 Controlled long-tail

\[
IF\in\{10,50,100,500\}
\]

Imbalance가 심해질수록 RACE의 Macro-F1과 tail F1 감소폭이 Global, full-data XGBoost, strongest PFN baseline보다 작은지 본다.

### 20.4 Test prior shift

- Tail-heavy
- Balanced
- Training-like
- Head-heavy

DistPFN 또는 동등한 posterior correction을 반드시 포함한다.

### 20.5 Authentic temporal drift

- Zero-update temporal generalization
- Same verified-label supervised refresh
- Window별 및 prequential Macro-F1/tail F1/FPR
- 초기 대비 최종 degradation

### 20.6 Efficiency

- Global-only latency
- Dense all-expert latency
- Sparse RACE latency
- Batch 1 p50/p95/p99
- Batch 32/256/1024 throughput
- Peak GPU memory
- Context cache 크기
- Proposal, override, rejected-call, wasted-call rate

---

## 21. 필수 Baselines

### Static classifiers

- Full-data XGBoost
- Class-weighted XGBoost
- XGBoost posterior/threshold correction
- CatBoost 또는 LightGBM

XGBoost는 PFN context 크기로 학습 데이터를 제한하면 안 된다. RACE가 method construction에 사용하는 전체 pre-test labeled pool을 사용하도록 해야 한다.

### PFN baselines

- Deterministic Global PFN
- Balanced-context PFN
- DistPFN 또는 동등한 context-prior correction
- Multiple random contexts
- Nearest/local-context PFN 또는 LoCalPFN
- MixturePFN/MICP
- 기존 수동 exp21 heuristic
- Compatible setting의 Drift-Resilient TabPFN

### Routing baselines

- Nearest-centroid
- Residual-cluster ownership classifier
- Global entropy/confidence gate
- Hard K+1 router
- Mean-gain scorer
- Shared expected-utility scorer
- Proposal + calibrated lower-bound verifier
- Dense all-expert selector
- Label-informed per-sample oracle

---

## 22. 반드시 보고할 Routing 지표

단순 routing accuracy만 보고해서는 안 된다.

### Correctness transition

각 class별로 다음 전이를 모두 보고한다.

```text
Global correct → Final correct
Global wrong   → Final correct    [helpful]
Global correct → Final wrong      [harmful]
Global wrong   → Final wrong
```

현재 EXP22B의 `correction precision=0.94`는 helpful/harmful만 사용하므로 Web의 wrong→wrong false-positive 이동을 포착하지 못한다.

### 추가 지표

- Proposal rate
- Expert call rate
- Override rate
- Proposal 중 override 비율
- Positive-utility proposal precision/recall
- Verifier false accept/false reject
- Overall/benign harmful override
- Mean realized gain among calls/overrides
- Class·window별 activation
- Rejected-call 및 wasted-call rate
- Oracle utility recovery

Metric에는 항상 분모와 절대 행 수를 함께 기록한다.

---

## 23. 통계와 재현성

- 5개 independent end-to-end seed
- Seed마다 context selection, clustering, router initialization 변경
- Day/session/capture/incident 단위 paired block bootstrap 95% CI
- 동일 seed와 동일 row를 모든 비교 방법에 제공
- Prior-shift resample은 독립 model seed로 세지 않음

다음을 반드시 저장한다.

```text
Raw data checksum
Split manifest
All context row IDs
Context class priors
Cluster membership and medoids
Expert descriptors
Offline gain matrix hash
Model/config/library versions
Seed별 prediction과 metrics
Thresholds and calibration outputs
Hardware, precision, cache, batch settings
Temporal verified-row indices and update log
```

---

## 24. 구현 성공 여부를 판단하는 순서

### Gate 1 — Expert complementarity

- Residual expert oracle이 random/proximity contexts보다 명확히 좋음
- Expert별 positive-gain coverage가 존재함
- Expert 간 gain correlation이 과도하게 높지 않음
- Severe long-tail에서 의미 있는 oracle tail gain 존재

실패하면 router를 고치는 것이 아니라 residual representation과 context selection을 수정한다.

### Gate 2 — Routing feasibility

- Oracle utility의 의미 있는 비율 회수
- Helpful override가 harmful override보다 충분히 많음
- Specialist proposal rate가 과도하지 않음
- Rejected/wasted call이 비용을 지배하지 않음

### Gate 3 — Long-tail/temporal robustness

- Severe long-tail 또는 temporal setting에서 strongest non-oracle baseline보다 Macro-F1/tail F1 개선
- IID Macro-F1과 benign FPR guardrail 유지
- 여러 seed와 여러 데이터셋에서 같은 방향

### Gate 4 — Temporal refresh

- \(B\le200\)의 적은 라벨로 다음-window 성능을 의미 있게 회복
- Global context replacement와 same-budget XGBoost보다 높은 성능 또는 비용 효율
- 과거 source reference 성능을 크게 훼손하지 않음

---

## 25. 재현 가능한 연구 산출물

1. **코드와 실행 설정**
   - Commit hash
   - 실행 명령
   - 전체 hyperparameter
   - Package/checkpoint version

2. **데이터 무결성**
   - Split별 시간 범위, row 수, class 수
   - Duplicate/group/hash overlap 결과
   - 전처리 fitting 범위

3. **Expert 분석**
   - Context class composition
   - Row-ID Jaccard matrix
   - Prediction disagreement matrix
   - Gain correlation matrix
   - Positive-gain coverage

4. **Router/verifier 분석**
   - Proposal/override 절대 수와 비율
   - Class별 correctness transition
   - Harmful override와 wrong→wrong confusion 이동
   - Calibration curve 및 lower-bound coverage

5. **성능과 비용**
   - Global/RACE/XGBoost/PFN baselines
   - Classwise precision, recall, F1
   - Macro/tail/AUPRC/FPR
   - Latency, throughput, GPU memory
   - Seed별 결과와 CI

결과 그림은 핵심 경향을 전달하는 용도로 사용하고, 구현과 결론의 재현성은 split manifest, context indices, raw prediction, confusion matrix와 configuration으로 뒷받침한다.

---

## 26. 최종 핵심 정리

RACE-PFN의 핵심은 expert를 여러 개 만드는 것 자체가 아니다.

가장 중요한 흐름은 다음이다.

```text
1. 전체 class를 처리하는 강한 Global PFN을 만든다.
2. Held-out 데이터에서 Global이 어디서, 어떤 방향으로, 얼마나 실패하는지 표현한다.
3. 서로 다른 failure regimes를 대표하는 full-label specialist contexts를 만든다.
4. 각 expert가 Global보다 실제로 얼마나 유용한지를 paired prediction으로 학습한다.
5. 비용을 고려해 필요한 top-1 expert만 호출한다.
6. 보수적으로 검증된 predictive gain이 있을 때만 Global prediction을 교체한다.
7. 소량의 최신 verified labels로 고정 크기의 context memory를 갱신한다.
8. 갱신 효과는 반드시 다음 temporal window에서 측정한다.
```

다음 research iteration의 최우선 목표는 **expert complementarity 확보**다. Expert들이 실제로 서로 다른 Global failure를 보완해야 이후 utility routing과 verifier가 회수할 수 있는 이득도 커진다.

따라서 다음 구현의 첫 번째 목표는 다음과 같다.

> Full residual signature와 diversity-aware context selection을 적용하여 서로 다른 predictive gain을 갖는 specialist context bank를 만들고, 그 oracle complementarity를 먼저 입증한다.

이 complementarity를 먼저 확보한 뒤 shared utility router로 oracle gain을 회수하고, calibrated verifier로 harmful adoption을 줄이며, temporal refresh로 다음-window 성능 회복까지 연결한다. 이 순서를 유지하면 각 실험이 다음 구성요소의 필요성을 뒷받침하여 논문의 기여와 결과가 하나의 인과적인 서사로 정리된다.

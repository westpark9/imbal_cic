**RACE-PFN 개선안 재정리 — 현재 / 변경 / 근거, 2026-09-09**

추가 질의에 따른 [학습·추론 구분과 구체적 선택 지표](mechanism_clarifications.md): expert의 지원 사례는 tail에 한정하지 않고 **global이 약한 모든 클래스의 residual 구간**에서 선택한다. 현재 코드도 D_route의 모든 expert 출력을 계산한다. 수정 대상은 그 결과의 저장·활용 및 학습 정답 구성이다. 임계값 선택은 아래의 일반적인 “결정 손익” 표현을 구체화하여, 전체 macro-F1을 주지표로 하고 tail·중간 클래스 F1과 benign FPR를 함께 제한하는 설계를 제시했다.

사용자 후속 요청에 따라 모델의 목적과 수정 위치를 명확히 한다. **제한된 global이 놓치는 tail을 적절한 expert로 보내 실제로 교정하는 것이 1차 목표다.** 앞선 제안에서 “scorer는 마지막에 호출 비용을 줄이는 역할로 복원한다”는 문장은 scorer의 역할을 지나치게 좁혔다. scorer는 **어떤 입력에 어떤 expert가 도움이 되는지**를 예측하고, verifier는 **그 expert의 관측된 출력을 채택할지** 판단한다. 계산 비용 감소는 교정 성능이 확보된 뒤의 추가 최적화다.

global context를 개선하는 일은 가능하지만, 지금의 핵심 실험은 현재 global을 고정한 상태에서 scorer/verifier의 추가 기여를 만드는 것이다. global을 의도적으로 약하게 만들 필요도 없다. context 개선은 global의 독립적인 점수뿐 아니라 **specialist가 교정에 필요한 증거를 갖는지**까지 평가한다.

아래는 준비 단계부터 추론·최종 결정까지의 순서다. `C0`는 global context, `Ck`는 expert k의 context, `p0/pk`는 각 모델의 예측 확률, `z`는 현재 pre-call에 사용되는 입력 표현이다. 모델 코드 변경이나 새 실험 결과가 아니라, 다음 실험에서 비교할 설계다.

| 단계 | 현재 | 변경 | 근거 |
|---|---|---|---|
| ① 데이터·context용 pool 준비 | train의 일부를 context·route·calibration에 사용하고 test는 원본 평가 구간 전체를 사용한다. 클래스별 cap·표본추출로 비중이 달라진다. | **전체 test 평가를 유지한다.** context는 tail 학습에 필요한 표본을 충분히 확보하고, 별도로 calibration 표본의 추출확률·클래스별 표본 수를 기록한다. train-only 전처리와 시간 분할을 유지한다. | 학습용 증거의 배분과 운영 분포에서의 위험 추정은 목적이 다르다. context와 test 비중이 같아야 한다는 조건은 없다. 전체 위험을 추정할 때는 표본추출로 바뀐 비중을 반영해야 한다. |
| ② C0 구성 → global 예측 | 최신 비교 기준은 C0=100k, benign 75%, 공격 균등, n_estimators=4이며 scenario 피복 보완을 이미 수행했다. 행 선택·config 변동도 성능에 영향을 준다. | **먼저 C0 행 ID·TabPFN config·global 출력을 고정한다.** context 실험으로 넘어가면 같은 크기·클래스 비중 안에서, 클래스 내부의 관측된 feature mode에 최소 지원 수를 두고 나머지는 빈도를 유지하며 선택한다. context seed와 model-config seed를 분리한다. | 지금 필요한 것은 같은 global 위에서 라우팅의 기여를 확인하는 것이다. 이후 mode 피복은 random 추출의 희귀 하위패턴 누락을 직접 줄인다. 기존 scenario 층화를 버리는 변경이 아니라, 같은 scenario/class 안의 관측 가능한 하위패턴까지 점검하는 보완이다. |
| ③ residual 추출 → expert context/bank 구성 | global residual을 중심으로 block을 만들고 shared anchor와 결합한다. residual이 큰 사례가 좋은 expert의 재료라는 가정에 의존한다. | `Ck = anchor + residual 취약 구간의 support + 혼동 negative`로 구성한다. **Global residual은 취약 구간을 찾는 주된 정보로 유지**하고 tail·중간·큰 클래스 모두를 대상으로 한다. 동일 예산에서 취약 구간 내부의 mode 피복과 contrast 배분을 각각 실험한다. 첫 비교에서는 기존 bank를 고정한다. | residual에는 교정 가능한 오답과 모순·잡음이 섞일 수 있다. residual을 버리는 것이 아니라 그 안의 다양한 오류 패턴을 대표하고, 구분에 필요한 반대 라벨 증거를 추가하는 수정이다. anchor+block 자체는 기존에도 있으므로 기여 후보는 합집합 모양이 아니라 support·contrast의 선택 기준이다. [BAPS](https://arxiv.org/abs/2608.12989)·[LUCoS](https://arxiv.org/abs/2605.27254)는 제한된 context 선택의 비교·설계 근거이며 IDS 교정을 보장하지 않는다. |
| ④ scorer/verifier 학습용 paired 결과 구성 | 현재도 D_route에서 모든 expert를 실행한다. scorer 정답은 `1{b_oof × G − λc > 0}`이며, 임시 OOF verifier가 거절한 양의 G도 negative가 된다. OOF verifier는 raw weighted G, final verifier는 normalized G를 사용한다. | 이미 계산하는 paired 결과에서 **실제 결정 손익을 직접 계산·보존**하고, 임시 verifier의 수락 여부로 원래 교정 라벨을 지우지 않는다. 기존 NLL 목표를 유지하는 대조 실험에서는 OOF/final 단위부터 맞춘다. | OOF verifier의 놓침이 scorer의 학습 정답으로 전달되는 경로를 분리한다. 학습 정답 구성의 변경이며, 온라인에서 모든 expert를 먼저 호출하는 구조로 바꾸는 것이 아니다. 단위 수정과 목표 변경은 별도 실험으로 구분한다. |
| ⑤ scorer → expert 선택·호출 | pre-call 정보 `p0, z, 거리, affinity, expert descriptor`로 expert별 점수를 만들고 top-1과 임계값으로 호출한다. 점수의 정답이 이전 verifier에 의존한다. | **“어떤 expert가 global의 이 오답을 교정할 가능성이 크고, 손해는 얼마나 있는가”를 학습한다.** 먼저 직접 paired 손익을 학습하는 scorer를 비교한다. verifier의 교정력이 확인되면 그 최종 정책을 적용한 OOF 실제 손익으로 목표를 정렬한다. global이 확신하는 입력도 후보 평가에 포함한다. | expert의 소유 클래스나 독립 정확도보다 global 대비 상대 이득이 라우팅 목적에 맞는다. [다중 expert learning-to-defer](https://proceedings.mlr.press/v267/mao25c.html)는 고정 predictor 사이에서 예상 손실을 비교하는 이론적 출발점이다. 이 단계의 우선 지표는 tail의 교정 가능한 후보 회수율·잘못된 expert 선택 손실이다. 계산비용 계수는 진단 중 0으로 두되 실제 호출량은 기록한다. |
| ⑥ 선택된 expert 추론 → verifier 판정 | expert는 Ck로 pk를 만든다. post-call은 `p0, pk, pk−p0, entropy/margin, 거리·affinity, descriptor`를 사용하며 pre-call의 z와 raw 46-feature가 없다. verifier는 NLL gain의 하위 분위수를 학습한다. | 먼저 **같은 목표에 z 추가**, 별도로 **raw feature 추가**를 비교한다. 그다음 입력을 고정하고 목표를 **실제 교정 손익**으로 바꾼다. “expert만 정답 / global만 정답 / 둘 다 정답 / 둘 다 오답” 또는 직접 action loss 차이를 학습해 global보다 나을 때 채택한다. | posterior가 비슷해도 원래 입력에서 구별되는 오답 원인이 있을 수 있다. 또 정답 확률 상승은 예측 라벨의 교정과 다르다. 순수 0–1 loss에서 기대 이득은 `P(expert만 정답) − P(global만 정답)`이다. 목표·입력을 따로 바꿔 어느 병목을 해소했는지 확인한다. |
| ⑦ 임계값 보정 → 최종 출력 | Dcal에서 pre/post 임계값을 선택하며 helpful/harmful·benign FPR·호출률 제약도 이미 사용한다. 다만 클래스 cap으로 표본 비중이 바뀌고 최적화 이득은 NLL 기반이다. | 실제 결정 손익을 기준으로 두 임계값을 조정한다. **tail 교정 증가와 benign 오경보 제한을 함께 적용한다.** 클래스별 위험을 추정해 목표 비중으로 합치거나 추출확률로 가중한다. 정책 선택과 최종 보정·평가 표본을 분리하고 다음 시간 구간에서 확인한다. | 기존 제약을 없애거나 느슨하게 하는 제안이 아니다. 목적함수와 평가 모집단을 맞추는 수정이다. 순수 label-prior 변화는 benign 조건부 FPR를 바꾸지 않아도 전체 harmful 비율은 바꿀 수 있다. [Learn then Test](https://arxiv.org/abs/2110.01052)의 정책 단위 위험 검정은 참고할 수 있으나 시간 이동에서 자동 보장을 주장하지 않는다. |

**scorer와 verifier가 배우는 값의 차이**

같은 학습 행에 대해 global과 expert의 결정 차이를 다음처럼 기록한다.

\[
\Delta_k(x,y)=\ell(y,\hat y_0(x))-\ell(y,\hat y_k(x)).
\]

순수 0–1 loss에서는 expert만 정답이면 +1, global만 정답이면 −1, 둘 다 정답 또는 둘 다 오답이면 0이다. verifier는 **expert 출력까지 관측한 정보**로 `E[Δk | post 정보]`를 추정한다. expert의 확신 자체보다 “global의 결정을 바꾸는 편이 나은가”에 맞는 목표다.

scorer는 호출 전이므로 pk를 입력으로 쓸 수 없다. 첫 대조는 pre-call 정보로 동일 Δk의 기대값을 예측하는 단순 정책이다. 이후 검증된 verifier를 `b*k ∈ {0,1}`로 고정하면, scorer의 실제 목표는 다음과 같다.

\[
V_k(x)=\mathbb E[b_k^*(Z_{post})\,\Delta_k\mid Z_{pre}].
\]

이는 **k를 호출하고 verifier의 최종 판단까지 거쳤을 때 얻을 교정 이득**이다. `b*k`는 held-out/OOF 예측으로 생성해야 하며, 해당 행으로 학습한 verifier의 in-sample 판정을 정답으로 쓰지 않는다. raw Δk와 네 가지 correctness 상태도 함께 보존하여, rejected pair 안의 놓친 교정 기회를 계속 점검한다. 기존의 부정확한 b_oof를 그대로 증류하는 것과 이 검증된 정책에 목표를 맞추는 것은 구분한다.

최적 post 정책과 정확한 조건부 위험 추정이라는 이상적 조건에서는 `V_k = E[max(E[Δk | Zpost], 0) | Zpre]`로 해석할 수 있다. 따라서 호출 전의 단순 순이득 `E[Δk | Zpre]`와 호출 후 나쁜 결정을 거부할 수 있는 **조회 가치**는 일반적으로 같지 않다. 이 차이를 고려해야 pre/post 두 컴포넌트를 유지하는 이유가 분명해진다.

tail 누락과 benign 오경보의 중요도를 다르게 줄 때는 bounded class-weighted loss 또는 사전 정의한 오분류 비용행렬로 Δ를 계산한다. 여기서 **오분류 비용은 예측 손해**, 계산 비용은 시간·호출량을 뜻한다. 전자는 현재 성능 목적에 포함되고 후자는 후순위다. class-balanced 0–1 loss는 macro-recall에 대응하는 대리 목표이며 macro-F1의 정확한 최적화와 같지는 않으므로 최종 tail F1·macro-F1을 따로 평가한다. 일반 비용행렬에서는 두 모델이 모두 오답이어도 손익이 0이 아닐 수 있어 Δ를 직접 사용한다.

**context 피복의 이론적 근거와 적용 범위**

N개 후보 중 어떤 관측된 하위패턴에 n개가 속할 때 B개를 균등 비복원 추출해서 그 패턴을 전혀 포함하지 않을 확률은 `C(N−n,B)/C(N,B) ≤ exp(−Bn/N)`이다. 클래스별 개수만 맞춰도 희귀한 클래스 내부 패턴은 누락될 수 있다. mode별 최소 지원을 두는 이유는 이 누락을 줄이기 위해서다. 모드 분할은 train feature에서만 정의하며, 아직 발견하지 못했거나 train에 없는 패턴까지 피복한다는 뜻은 아니다.

각 class/mode 안의 대표점은 `F(S)=Σ_i w_i max_{j∈S} sim(z_i,z_j)`와 같은 피복 목적함수로 선택할 수 있다. 고정된 비음수 가중치·유사도와 단순 개수 제한에서 표준 greedy의 `1−1/e` 보장은 **이 피복 목적함수에만** 적용된다. 이것이 TabPFN의 F1 보장으로 이어지는 것은 아니므로, 같은 context 크기·클래스 비중·model config의 random 선택과 비교해야 한다. [Nemhauser·Wolsey·Fisher, 1978](https://thibaut.horel.org/submodularity/papers/nemhauser1978.pdf)

**train 부분 사용과 test 전체 사용에 대한 해석**

사용자가 설명한 표본 비중 차이는 현재 프로토콜의 자연스러운 결과다. tail 증거를 늘린 C0/Ck를 원본 test 비중에 억지로 맞추자는 제안은 아니다. 조정 대상은 **모델 평가·보정에서 추정하려는 위험의 비중**이다.

예를 들어 클래스별 평균 손실을 `R_y`라 하면 목표 모집단의 위험은 `Σ_y π_target(y) R_y`로 계산한다. π_target은 과거의 원본 학습 구간 또는 이용 가능한 최근 labeled window로 정하고, test 정답에서 구해 정책을 맞추지 않는다. 클래스 안에서도 특정 mode만 선택했다면 클래스 가중치만으로 해결되지 않으므로 추출확률을 반영하거나 별도의 대표성 있는 calibration 표본을 둔다. 시간에 따라 클래스 조건부 분포까지 달라지는 경우는 이 가중만으로 교정되지 않아 forward 평가가 필요하다.

앞선 감사에서 seed 42의 Dcal benign 비중 21.35%는 **class cap 후·hash mask 전**의 값이고, test 87.07%와 비교한 진단 수치다. 실제 최종 Dcal 전체의 비중으로 단정하지 않는다. 이 차이만으로 라우팅 실패의 원인을 확정하지도 않는다.

**실험 순서는 모델의 온라인 처리 순서와 구분한다**

1. 현재 global·작은 expert bank를 고정하고 모든 expert의 paired 출력을 저장한다. 같은 bank의 oracle과 후보별 교정 기회를 계산한다.
2. scorer를 진단 단계에서 우회하고 verifier의 `+z`, `+raw`, `결정 손익 목표`를 각각 비교한다. 좋은 expert 출력이 주어졌을 때 실제로 올바르게 채택하는지 확인한다.
3. scorer를 연결하여 tail의 교정 가능한 후보를 얼마나 회수하고, 잘못된 expert 선택으로 얼마나 잃는지 측정한다. 목표 단위와 OOF/final 정책을 맞춘다.
4. 그다음 같은 예산에서 context의 mode 피복과 contrast negatives를 각각 바꾼다. global만 개선되는지, specialist와 라우팅까지 추가 이득을 얻는지 구분한다.
5. tail 개선과 benign 유지가 성립한 설정에서 호출량·지연을 줄인다. 비용만 낮아진 모델을 tail 교정 목표의 성공으로 세지 않는다.

label oracle은 정답을 보므로 도달 가능한 성능 보장은 아니다. 같은 feature 벡터에 상충하는 정답이 있으면 동일 입력만 보는 라우터가 정답별로 다른 expert를 고를 수 없다. 따라서 같은 bank의 label oracle과 함께, 관측 가능한 입력으로 학습한 정책의 다음 시간 구간 성능을 비교한다. 과거 exp17/exp20의 소유 expert 맞히기·확률 입력 selector와 달리, 이번 변경은 **실제 상대 결정 손익·입력 정보 보강·목표 정합성·시간 검증**을 각각 실험한다.

원본 코드 근거: [pair 입력과 임계값 선택](../../../tabpfn/scripts/nfv3_v3_exp31_c0alloc.py:543), [상세 감사 및 개선 제안](research_and_model_proposal.md), [최신 저장 결과](latest_routing_audit.csv).

**원본 문헌조사 엑셀 갱신**

[tabpfn_papers_survey.xlsx](../../tabpfn_papers_survey.xlsx)의 Ch.2에 다음 3편을 추가했다. 기존 99편의 값·서식·링크·취소선을 유지하고 출처등급→날짜 정렬에 맞춰 총 102편, Ch.2 38편으로 갱신했다.

- [HINT](https://arxiv.org/abs/2609.07956v1): 2026-09-07, Streaming Continual Learning ECML PKDD Workshop 2026 표기. 조건부 TFM 호출과 검색 context의 관련 연구.
- [LoGIC](https://arxiv.org/abs/2609.05955v1): 2026-09-05, preprint. 그래프 ICL에서 제한된 context를 구성하는 관련 연구.
- [Xiaomi-TabLDM](https://arxiv.org/abs/2609.03880v2): v1 2026-09-03, 확인 버전 v2 2026-09-04. 내부 sparse MoE를 가진 TFM 기술 보고서. 외부 expert 라우팅과 구분하여 기록했다.

이 세 논문을 추가했다는 사실과 위 설계의 이론 근거는 별개다. HINT의 비용 절감 목적을 현재 모델의 1차 목적으로 가져오거나, Xiaomi의 내부 MoE를 현재 expert bank와 동일한 구조로 간주하지 않는다.

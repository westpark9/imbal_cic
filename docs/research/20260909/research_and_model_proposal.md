**RACE-PFN: 최근 문헌 갱신과 context·routing 개선 제안 — 2026-09-09**

권고하는 다음 구조는 **피복 조건을 갖춘 context와, global 대비 의사결정 이득을 직접 배우는 선택적 교정층**이다. 우선 최신 global을 고정하고 verifier의 정보·목표를 수리한 뒤, scorer가 tail을 교정할 expert를 찾아 연결하도록 한다. scorer의 1차 역할은 교정에 도움이 되는 expert 선택이며, 호출 비용 감소는 실제 교정 성능을 확보한 다음의 최적화다. 그 교정층이 사용할 수 있는 context도 별도 실험으로 개선한다. 아래 제안은 아직 새 모델의 성능 결과가 아니라, 기존 기록·저장 배열·코드와 논문에 근거한 검증 가능한 설계다.

후속 요청에 따른 [모델 처리 순서별 현재·변경·근거 표](mechanism_revision.md)에 목적의 우선순위와 train 부분 사용/test 전체 평가의 해석을 정리했다. 원본 문헌조사 xlsx에는 HINT·LoGIC·Xiaomi-TabLDM 3편을 추가하여 총 102편으로 갱신했다. 아래의 “기존 99편”은 갱신 전 조사 기준이다.

추가 질의에 대한 [학습·추론 순서 및 지표 명세](mechanism_clarifications.md)에서는 residual을 유지해 중간 빈도 클래스를 포함한 모든 취약 구간을 다루도록 명확히 했다. D_route의 전체 expert 예측은 현재도 수행한다. 학습은 paired 결정 손익, 임계값 선택은 전체 macro-F1과 tail·클래스별 F1·benign FPR 제약으로 구분하는 구체안을 기록했다.

읽은 자료: `lablog/report/0427~0904.md`의 실험 계보·관련 진단, `lablog/html_report/racepfn_status.md`, `docs/tabpfn_papers_survey.xlsx`, `docs/feedbacks/RACE-PFN_RESEARCH_IMPLEMENTATION_GUIDE.md`의 설계·학습 목표, `CLAUDE.md`, 원본 저장소의 `../imbal_cic/SRC_HISTORY.md`, exp29·exp31 및 저장 결과. 오래된 상태 문서보다 0902 후반·0904의 최신 결과와 실제 run 배열을 우선했다. 검토 시 git HEAD는 `6962bc4ef88873a88d169773d2c8b1446a0a9920`이다.

이번에 생성한 [문헌 보충표](literature_supplement.xlsx), [최신 라우팅 재계산](latest_routing_audit.csv), [scorer 정답 생존율](scorer_target_survival_by_class.csv), [클래스별 전이](latest_transitions_by_class.csv)를 함께 참조할 수 있다. 기존 학습 코드를 변경하거나 GPU 실험을 새로 돌리지는 않았다.

**1. 지금까지의 접근에서 유지할 것과 바꿀 것**

| 시기 | 접근과 확인된 사실 | 이번 설계에 반영할 점 |
|---|---|---|
| 4~6월 | taxonomy/confusion expert, confidence·energy·TTA 기반 선택. owning expert oracle과 실제 선택 사이 큰 차이 | 전문가의 자신감·안정성을 상대적 유능함의 대리값으로 쓰지 않는다. 실제 상대 손익을 학습한다. |
| 7~8월 초 | NF-v3, 독립 energy gate와 해소 단계. 불균형·미지 공격에서 활성과 benign 보존의 trade-off | tail recall 단독 개선 대신 benign 손해와 호출 비용을 같은 정책 평가에 포함한다. |
| 8월 중순 | TabPFN global, family context, local expert·anchor, 전처리·중복 진단 | 중복에는 질량·밀도 정보가 있다. 삭제를 기본 처방으로 삼지 않는다. |
| 8/25~8/31 | residual regime mining, context expert bank, scorer→expert→quantile verifier | label-informed oracle을 bank 최적화의 유일한 기준으로 삼지 않는다. 양의 NLL 이득과 실제 정답 회복을 구분한다. |
| 9/1~9/2 | scenario 피복으로 C0의 SSH 등의 결손 보완. verifier 단위 수정. 등메모리 bank 비교 | context 접근 가능성, 표본 선택, 라우팅을 분리한다. residual bank의 보편적 우위는 재가정하지 않는다. |
| 9/3~9/4 | 100k, benign 0.75, 공격 균등 배분, config ensemble 4로 global 안정화 | 이 구성을 새 기준선으로 고정한다. 과거 1M·n=1 oracle 격차를 새 기준선의 개선 여지로 가져오지 않는다. |
| 9/4~9/8 | BoostPFN·LoCalPFN·DistPFN 원본/이식 검토와 재현 | 방법·backbone·데이터·지표 차이를 분리한다. baseline의 한 데이터셋 결과를 방법 전체의 결론으로 확대하지 않는다. |

초기 `racepfn_status.md`의 “scorer 병목”, 중간 로그의 “verifier 상관 ≤0.15”는 당시 구성의 진단이다. 최신 seed 42에서는 verifier의 tune 상관이 expert별 0.482~0.862까지 올라간다. 따라서 현 문제를 “verifier가 아무것도 예측하지 못한다”로 고정하면 다음 실험의 표적이 어긋난다. **현재의 표적은 올바른 결정 변경을 구별하고, 그 이득을 다음 시간 구간에서 회수하는 능력이다.**

**2. 최신 저장 결과로 다시 확인한 현재 위치**

재계산 대상은 exp31의 `C0=100k`, `benign_share=.75`, `attack_alloc=balanced`, `n_estimators=4`, seed 42~48이다. `args.json`에서 uncapped CIC2018와 `test_cap_per_class=0`을 확인했다. 평가 4,023,114행, context·route 등의 내부 분할은 scenario 층화다. [재계산 코드](audit_existing_runs.py)는 모델을 학습하지 않고 저장된 지표·배열만 읽는다.

| seed | global macro-F1 | system − global | τ_pre | 채택 행 | 실제 라벨 변경 | helpful / harmful |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.762302 | +0.000256 | 0.893141 | 38,498 | 260 | 98 / 162 |
| 43 | 0.761430 | 0 | ∞ | 0 | 0 | 0 / 0 |
| 44 | 0.758152 | 0 | ∞ | 0 | 0 | 0 / 0 |
| 45 | 0.766185 | 0 | ∞ | 0 | 0 | 0 / 0 |
| 46 | 0.769424 | 0 | ∞ | 0 | 0 | 0 / 0 |
| 47 | 0.761482 | 0 | ∞ | 0 | 0 | 0 / 0 |
| 48 | 0.767666 | 0 | ∞ | 0 | 0 | 0 / 0 |

global 평균은 **macro 0.763806 / tail 0.554495**다. 과거 full-XGB 기록은 0.7548 / 0.5331이지만 이는 고정 비교 기록이며, 7개의 새 XGB paired run은 아니다. 같은 C0로 별도 재학습한 0904 exp37 seed 42에서는 global 0.7617 / tail 0.5480, XGB-C0 0.7525 / 0.5260이다. **tail 개선은 global 구성 단계에서 이미 관측되었고, scorer·verifier의 추가 기여가 거의 실현되지 않는 것이 정확한 현재 질문**이다.

추가로 주의할 세 점이 있다.

1. seed 42의 38,498건 채택 중 38,238건, 약 99.3%는 라벨이 같다. 확률 품질 변화는 별도 가치가 있을 수 있지만, 이를 tail 정답 회복 건수로 세면 안 된다.
2. seed 42에서는 infiltration을 98건 추가로 맞히면서 benign 162건을 infiltration으로 바꾼다. macro는 +0.000256, 행 단위 정확도는 64건 순손해다. 이 둘은 모순이 아니다. 목적함수와 허용할 비용을 명시해야 한다. benign FPR 증가는 약 0.0000462다.
3. 최신 7개 run은 `dense_eval=False`다. 따라서 이전 구성의 oracle 0.83~0.88을 현재 global 0.764와 빼서 “회수 가능한 격차”라고 제시할 수 없다. **새 기준선 위에서 같은 bank의 oracle·실현 정책을 다시 짝지어야 한다.**

수치 근거: [최신 요약](latest_routing_audit.csv), [전이표](latest_transitions_by_class.csv), `lablog/report/0902.md`의 §8ai~8aj, `lablog/report/0904.md`의 §6~§7. 9/7의 n=32 global 0.7693과 XGB-C0+EM 0.7645도 보조 비교 후보지만 단일 seed 및 추가 선택 비용을 명시해야 한다.

**3. 문헌조사 이후 실제로 새로 나온 것**

기존 xlsx는 99편이고 문서 속성상 생성·수정일은 2026-09-03이다. 정확한 조사 종료 시각을 단정하지 않고, **9/3~9/9를 갱신 구간**, 9/2를 검색 완충일로 삼았다. 파일에 있는 arXiv ID 89개를 API로 전부 재확인했으며, 나머지 비-arXiv 항목의 모든 출판 이력까지 전수 감사한 것은 아니다.

검색은 웹 키워드 검색과 arXiv API의 다음 조건을 병행했다. `TabPFN`, `tabular foundation`, `tabular in-context`, context selection, routing, verifier, learning to defer를 사용했다. API 기간 검색은 `(all:TabPFN OR all:"tabular foundation" OR all:"tabular in-context") AND lastUpdatedDate:[202609020000 TO 202609092359]`였다. 기간 검색 11건 중 아래 5건을 모델 설계·비교에 우선 포함했다. 이는 지정 검색 범위의 결과이며 모든 미색인 문헌까지 포괄하는 부재 증명은 아니다.

| 논문 | 확인된 날짜·상태 | RACE-PFN과의 관계 | 대응 |
|---|---|---|---|
| [HINT: Streaming Hierarchical Inference with Tabular Foundation Models](https://arxiv.org/abs/2609.07956v1) | 9/7 v1; arXiv comment에 ECML PKDD 2026 workshop 명시 | bounded sliding-window ANN memory가 local predictor와 context selector를 겸하고, 불확실한 입력만 TabPFN으로 보낸다. **검색 context + 선택적 TFM 호출**이 겹친다. | 직접 관련 신규 baseline. 자신감 문턱 기반 호출과, 상대적 교정 이득 기반 호출을 비교한다. |
| [LoGIC](https://arxiv.org/abs/2609.05955v1) | 9/5 v1; preprint | 구조·특징·피복 검색, query cluster별 context 공유, labeled context/unlabeled halo의 예산 분리 | context 예산과 배치 공유 설계의 신규 선행연구. 그래프 노드 과제이므로 IDS 수치 대조로 바로 취급하지 않는다. |
| [Xiaomi-TabLDM](https://arxiv.org/abs/2609.03880v2) | 9/3 v1, 9/4 v2; technical report | ICL predictor 안에 shared/routed FFN experts와 top-k router를 둔다. | “TFM+MoE”라는 넓은 신규성은 피한다. 내부 사전학습 MoE와 외부 context expert bank의 차이를 명확히 한다. |
| [Mitra-v2](https://arxiv.org/abs/2609.04540v1) | 9/3 v1; technical report | synthetic pretraining·context 확장, 공개 backbone 비교 대상 | 모델 기여의 backbone 의존성을 검증할 다음 후보. 지금 당장 backbone 교체와 라우팅 수리를 한 실험으로 섞지 않는다. |
| [From Synthetic Priors to Model Behavior](https://arxiv.org/abs/2609.06912v1) | 9/7 v1; preprint | 합성 사전학습 분포의 structural coverage와 downstream 거동을 연결 | context를 바꾸어도 해결되지 않는 prior mismatch를 연구하는 보조 근거. inference context coverage와는 다른 개념이다. |

HINT 본문은 6개 stream의 첫 20k에 대한 prequential 실험이며, local confidence와 이웃 수를 조절한다. 확인한 구조에는 RACE식 사후 verifier가 없다. 따라서 **“같은 전체 모델이 이미 나왔다”는 결론은 아니지만 “선택적 TabPFN 호출”만으로는 차별화가 부족해졌다.** [HINT 원문](https://arxiv.org/html/2609.07956v1)

LoGIC은 validation으로 검색 채널과 context 크기를 고른다. 그래프의 연결 구조·unlabeled halo가 중요한 구성요소이므로, 이를 그대로 NetFlow 행 분류에 옮기는 대신 “서로 다른 증거 채널을 예산 안에서 배분한다”는 설계 비교에 사용한다. [LoGIC 원문](https://arxiv.org/html/2609.05955v1)

기존 항목의 갱신도 있다. **[TabPFN-Rel / RelArena-α / RPI](https://arxiv.org/abs/2608.16319v2)가 9/7 v2로 업데이트**됐다. 조사한 기존 arXiv 89개 중 9/3 이후 갱신으로 확인된 것은 이 항목이다. 버전 갱신 사실과 본문의 어떤 기법이 새로 추가되었는지는 별개라, 여기서는 v2의 구체적인 신규 추가 내용을 단정하지 않는다.

**4. 최신 논문과 별개로, 기존 xlsx에서 보완할 중요한 선행연구**

| 논문 | 기존 파일과 관계 | 이번 문제에 주는 근거 |
|---|---|---|
| [Learning-to-Defer with Expert-Conditional Advice](https://arxiv.org/abs/2603.14324v5) | 미수록; 3/15 최초, 5/29 v5 | expert와 expert에 제공할 추가 정보를 결합한 action 공간의 일관성 이론. **context 선택과 routing을 같은 결정 문제로 묶는 데 가장 직접적**이다. |
| [Mastering Multiple-Expert Routing](https://proceedings.mlr.press/v267/mao25c.html) | 미수록; ICML 2025 | 고정 expert의 정확도와 비용을 최적화하는 surrogate 및 H-consistency. 현재처럼 frozen predictor 위에 정책을 학습할 때의 출발점이다. |
| [LUCoS](https://arxiv.org/abs/2605.27254v1) | 미수록; 5/26 | unsupervised PFN의 latent geometry로 medoid context를 선택. 임의 raw/PCA 거리를 그대로 쓰는 대신 **어떤 표현에서의 피복인가**를 검증하게 한다. |
| [Topological Signatures of Context-Level Reliability in TabPFN](https://arxiv.org/abs/2607.17962v1) | 미수록; 7/20 | 합성 geometry에서 내부 topology와 context-level 신뢰도의 연관성. context 신뢰도 연구로 추가하되, 검증된 sample-level router로 포장하지 않는다. |
| [When More Experts Hurt](https://arxiv.org/abs/2602.17144v1) | 미수록; 2/19 | multi-expert deferral의 학습 목적에 따른 underfitting을 다룸. frozen global과 공동 학습 classifier를 구별해야 하므로 우리 실패의 직접 증명으로 인용하지 않는다. |

반대로 **BAPS(8/13), CRUMB(6/9), Context Sampling(7/29), Spurious Routing(7/28), MixturePFN, LoCalPFN, TuneTables, DistPFN은 이미 xlsx에 있다.** 이번에 발견한 “새 논문”으로 다시 세지 않았다.

가까운 기존 연구들의 비교축은 다음과 같다.

- **[MixturePFN](https://arxiv.org/abs/2405.16156)**: local context bank와 nearest-centroid routing, 배포 prompt 구성에 맞춘 adaptation. context expert bank 자체는 이미 존재한다.
- **[BAPS](https://arxiv.org/abs/2608.12989)**: 대표성·경계·밀도·균형·다양성을 고려한 compact context. “대표점+hard example+균형” 조합만으로 새로운 방법이라고 하기 어렵다.
- **[CRUMB](https://arxiv.org/abs/2606.11473)**: query cluster와 MMD로 context를 맞추고 배치 추론. `P(X)` 일치가 모든 tail의 label evidence를 보장하지 않는다는 것이 우리 설정의 추가 질문이다.
- **[Understanding Context Sampling](https://arxiv.org/abs/2607.26628)**: 작은 15개 데이터셋에서 context draw·coverage·비용을 비교한다. 비싼 선택기의 필요성을 random baseline과 비교해야 한다.
- **[Spurious Routing](https://arxiv.org/abs/2607.25532)**: ridge ICL에서의 이론과 TabPFN의 합성 실험은 context 증가가 spurious signal 의존을 강화할 수 있음을 보인다. “크면 반드시 좋아진다”는 가정을 버릴 근거다. 논문의 routing은 expert 선택이 아니라 입력 내부 신호 의존을 뜻한다.
- **[Correcting Class Imbalance in PFNs](https://arxiv.org/abs/2605.21742)**: binary imbalance의 threshold·sampling 비교는 posterior와 decision rule을 분리해야 함을 뒷받침한다. multiclass macro-F1 개선을 그대로 보장하지는 않는다.

**5. Oracle 격차를 어떻게 개선 목표로 바꿀 것인가**

“oracle처럼 되지 않는 이유에 routing이 있다”는 판단은 맞는 부분이 크다. 다만 **좋은 router가 회수할 수 있는 몫**을 분리해야, context·feature·정책 중 어디에 노력을 써야 할지 결정할 수 있다.

현재 dense oracle 코드는 true label로 각 expert의 weighted NLL을 계산하고 가장 작은 action을 택한 뒤 그 action의 argmax로 F1을 계산한다. 이것은 label-informed NLL selector다. **macro-F1 자체를 최대화하는 oracle도 아니고, 입력만 보는 router가 달성 가능한 상한도 아니다.** `exp31_c0alloc.py`의 `L_eval`, `oracle_pick` 구간이 근거다.

고정된 context·모델 아래 action별 손실을 `ℓ_a(X,Y)`, 허용 입력을 `X`, router 특징을 `Z=φ(X)`라 하자. 적절한 적분 가능성 아래 다음 세 위험을 정의할 수 있다.

\[
R_{label}=\mathbb E[\min_a\ell_a(X,Y)],\quad
R_X^*=\mathbb E[\min_a\mathbb E(\ell_a\mid X)],\quad
R_Z^*=\mathbb E[\min_a\mathbb E(\ell_a\mid Z)].
\]

`Z`가 `X`의 함수라면 `R_label ≤ R_X* ≤ R_Z*`다. `Z`로 결정하는 정책의 격차는 다음처럼 나뉜다.

\[
R(\pi)-R_{label}
=\underbrace{R(\pi)-R_Z^*}_{정책\ 학습}
+\underbrace{R_Z^*-R_X^*}_{특징\ 요약에서\ 잃은\ 정보}
+\underbrace{R_X^*-R_{label}}_{정답을\ 미리\ 아는\ 것의\ 이점}.
\]

이는 조건부 기대손실 최소화에서 직접 나오는 분해다. 여기서의 `ℓ`은 가산적 손실이다. macro-F1은 비가산적이므로 같은 식을 macro-F1 차이에 그대로 붙이지 않는다.

동일한 46-feature 벡터가 benign과 infiltration 모두로 나타나면, 고정 context를 쓰는 어떤 함수도 그 두 행을 정답별로 다르게 라우팅할 수 없다. 그러나 label oracle은 서로 다른 expert를 골라 둘 다 맞힐 수 있다. 그렇다고 데이터 문제로 연구를 닫을 필요는 없다. **모순 벡터에서의 최선의 비용 선택과, 구별 가능한 입력에서의 교정을 나누면 된다.**

기존 0821 scratch 진단은 infiltration test의 약 58.4%가 train에서 benign·infiltration 두 라벨로 모두 등장한 벡터였다고 기록한다. 이는 해당 scratch split의 수치이며, 최신 test 37,631행의 비율로 재사용하지 않는다. 또한 충돌 그룹의 단순 다수결은 그 그룹의 경험적 0–1 오류 최적 선택이지 macro-F1 상한이 아니다.

**먼저 만들 진단: 세 수준의 비교표**

1. 현재처럼 각 test 행의 label을 아는 oracle.
2. 동일 입력 hash에는 동일 action만 허용하는 oracle. 가산 손실에서 `Σ_g min_a Σ_{i∈g} ℓ_a(x_i,y_i)`로 계산한다. 이 역시 test label을 쓰는 사후 낙관적 진단이며 배포 정책이 아니다.
3. train-only forward-held-out 예측으로 학습한 실제 router. route/tune/cal/test의 같은 허용 정보로 평가한다.

세 수준의 차이와, 충돌·비충돌·새 벡터별 손익을 제시한다. 최신 n=4 run에 all-expert 확률이 충분히 저장되어 있지 않으므로 이 표는 아직 계산하지 않았다. 다음 실험에서는 **같은 frozen C0와 bank의 dense 결과를 한 번 저장**하면 이후 대부분의 정책 대조를 CPU에서 할 수 있다.

**6. 우선 제안 A — verifier를 상대적 오분류 비용의 추정기로 바꾼다**

현재 코드를 보면 pre-call은 `[p0, entropy, margin, z, distance, affinity, expert descriptor]`를 쓰지만, post-call은 `[p0, pk, pk−p0, entropy, margin, scalar distances/affinities, descriptor]`만 사용한다. **post-call 입력에 z와 원래 46-feature가 없다.** 두 posterior가 비슷한데 실제 오답 원인이 다른 입력을 verifier가 구분할 기회가 줄어든다. 이는 정보 병목 후보이며 raw feature를 추가하면 무조건 좋아진다는 주장은 아니다. [코드 위치](../../../tabpfn/scripts/nfv3_v3_exp31_c0alloc.py)

이 제안은 exp17·exp20의 확률 입력 selector와 가장 가깝다. 당시 target은 `y가 benign이면 global, owned class이면 소유 expert`였고, 이번 target은 **같은 row에서 실제로 어느 결정의 비용이 더 작은가**다. 기존 가이드 §13~15에도 paired utility와 OOF policy utility라는 큰 방향은 이미 있다. 따라서 상대 utility라는 개념을 새 아이디어로 주장하지 않는다. 구체적인 수정점은 **NLL→결정 비용, post-call 정보 보강, OOF 단위/목표 정합성, forward 평가, policy-aware context 선택**이다.

변경을 두 개의 독립 ablation으로 나눈다.

**A1: 목표를 유지한 채 입력만 보강.** 현재 normalized-gain verifier에 train-fit 전처리의 raw 46 features 또는 `z`를 각각 추가한다. 그 다음 확장으로 모든 클래스별 이웃 거리·라벨 지지율·유효 표본 수, context prior, global/expert의 예측 class pair를 제공한다. 데이터셋의 attack scenario 정답이나 test label을 query feature로 넣지 않는다.

**A2: 입력을 고정한 채 목표를 교정 손익으로 교체.** class label 자체나 expert ID를 고르는 CE 대신, expert `k`가 global의 결정을 바꿨을 때의 비용 차를 학습한다.

\[
\Delta_{ik}=\ell(y_i,\hat y_0(x_i))-\ell(y_i,\hat y_k(x_i)),\qquad
A_k(z)=\mathbb E[\Delta_k\mid z].
\]

가장 투명한 출발점은 correctness의 네 상태를 예측하는 것이다: 둘 다 정답, global만 정답, expert만 정답, 둘 다 오답. 단위 0–1 손실이라면

\[
A_k(z)=P(\text{expert만 정답}\mid z)-P(\text{global만 정답}\mid z).
\]

정책은 `A_k(z) > 0`인 경우에만 교정하고, 비용을 포함한 호출 결정에서는 `A_k(z) > λ c_k`를 사용한다. 이미 호출한 뒤에는 계산 비용이 sunk cost이므로 verifier의 교정 선택과 pre-call 비용 선택을 구별한다. 여러 action의 현재 가능한 예상 손실을 비교하는 것이 [multi-expert learning-to-defer](https://proceedings.mlr.press/v267/mao25c.html)의 결정이론적 출발점이다.

IDS에서는 benign 오경보와 tail 누락의 비용이 다르다. 따라서 실제 모델은 사전 정의한 bounded cost matrix `ℓ(y,ŷ)` 또는 정규화한 class-balanced 0–1 loss로 확장한다. 이때 `E[w(Y)·1{help}|z]−E[w(Y)·1{harm}|z]`를 직접 추정해야 한다. query의 예측 label에 해당하는 가중치를 사후로 곱하면 다른 목적이 된다. class-balanced loss는 macro-recall에 대응하는 surrogate이고, macro-F1과 같은 목적이라는 주장은 하지 않는다. 최종 F1은 독립 평가에서 별도로 보고한다.

가중 비용에서 서로 다른 두 오답의 비용도 다르다면 “둘 다 오답”을 한 상태로 합치지 말고 full action cost를 회귀하거나 `(y, ŷ0, ŷk)`에 따른 비용을 모델링한다. 확률 log-loss는 calibration을 위한 보조 지표로 유지할 수 있다.

이렇게 하면 seed 42의 **38,238건 같은-label 채택이 classification utility 0**으로 정의된다. 현재 `G>0`이 포착한 확률 개선 중 실제 tail 의사결정을 바꾸는 기회에 학습·호출 예산을 집중할 수 있다.

**확률만으로는 부족한 이유와 반증 방법.** expert가 맞을 확률만 추정하면 global도 맞는 쉬운 행을 고르는 경향이 생긴다. 반대로 `P(G>0)`만 추정하면 작은 이득과 큰 손해의 비대칭을 놓친다. 상대 비용 `A_k`를 배우고, 같은 채택량에서 도움이 되는 변경과 해로운 변경을 비교한다. `p0/pk only → +z → +raw46`이 forward holdout에서 향상되지 않으면 정보 보강 가설을 접고, 목표·expert 공급의 문제로 이동한다.

**7. 우선 제안 B — scorer가 이전 verifier의 판단을 정답으로 물려받지 않게 한다**

최신 exp31에서도 scorer target은 다음과 같다.

```text
b_oof = 1{OOF quantile verifier predicts positive gain}
U = b_oof * G - lambda * cost
scorer_label = 1{U > 0}
```

이 설계는 “사후 verifier가 받아들일 이득만 제안한다”는 의도가 있다. 그러나 verifier가 유망한 expert를 놓치면 그 pair는 실제 `G>0`이어도 scorer의 negative가 된다. 그리고 **최종 verifier는 `G / w_bal(y)`로 학습하는 반면 b_oof를 만드는 verifier는 여전히 raw weighted G로 학습한다.** exp26의 단위 수정이 이 OOF 경로까지 전달되지 않은 상태다. 현재 코드와 저장된 `route_gain.npz` 양쪽에서 확인했다.

저장 배열의 최신 seed 42에서 `G>0`인 pair 중 b_oof를 통과하는 비율은 전체 **19.81%**, brute **0%**, dos **0.018%**, infiltration **59.82%**, web **80.52%**다. 7 seed 전체 생존율은 8.06~83.03%로 차이가 크다. 이 수치는 **양의 NLL pair가 정답 구성에서 얼마나 남는지**이며, 그만큼 실제 정답 교정을 잃었다는 뜻은 아니다. [정답 구성 감사표](scorer_target_survival_by_class.csv)

제안하는 학습 순서는 다음과 같다.

1. **K=1 또는 작은 고정 bank에서 모든 route query에 expert를 실행한다.** scorer를 잠시 우회해 필요한 full-information action costs를 확보한다. 이것은 최종 추론 구조를 dense로 바꾸자는 뜻이 아니라, 정책이 학습 가능한지 분리하는 실험이다.
2. A의 verifier/교정 정책을 먼저 학습한다. old b_oof로 label을 0으로 만드는 연결을 제거한다.
3. scorer는 pre-call 정보로 **expert 조회 후 허용할 최선 교정의 기대가치**를 예측한다. calibration된 post-call 정책이 고정된 후, 그 정책의 out-of-fold 실제 손익을 distill하는 변형을 별도 비교한다.
4. 같은 correction policy에서 scorer가 tail의 교정 가능한 expert를 얼마나 회수하는지와 잘못된 선택으로 잃는 이득을 먼저 평가한다. 실제 tail 교정 성능을 확보한 뒤 호출률·latency를 줄인다. scorer의 top1 expert 정답률이나 호출 감소만을 성공 기준으로 삼지 않는다.

이를 value of information으로 쓰면, pre 정보 `z_pre`, 조회 뒤 정보 `z_post`에 대해

\[
V_k(z_{pre})=r_0(z_{pre})-
\mathbb E\left[\min_{a\in\{0,k\}}r_a(Z_{post})\mid z_{pre}\right]
-\lambda c_k.
\]

여기서 `r_a`는 그 단계에서 관측 가능한 정보를 조건으로 한 예상 의사결정 손실이다. **scorer는 V를 보고 조회하고, verifier는 조회 후 0과 k 중 손실이 작은 결정을 고른다.** 둘의 역할과 목표가 일치한다. 실제 구현에서는 이 식의 정확한 조건부 기대를 알 수 없으므로 forward-held-out action costs로 근사한다.

참고로 코드의 OOF는 앞 절반→뒤 절반과 뒤 절반→앞 절반 양방향이다. 두 절반 모두 outer train 안이라 이것만으로 test leakage라고 할 수는 없다. 다만 미래 배포의 재현 실험으로 쓰려면 **과거→미래의 rolling/forward folds**로 바꾸어야 한다. route에 없는 미래 실패는 어떤 supervised target 수정만으로도 학습할 수 없으므로, 그때는 context refresh와 허용된 과거 정보 보강을 사용한다.

**8. 우선 제안 C — context를 ‘행 추첨’에서 ‘필요한 증거의 피복’으로 바꾼다**

현재 backbone은 v1/v2가 아니라 **TabPFN-3**다. 공식 보고서는 1M training rows까지의 확장을 다룬다. 따라서 논문의 동기를 “TabPFN은 1k/10k만 처리 가능”으로 쓰면 현 실험과 맞지 않는다. **12M pool에서 제한된 계산 예산으로 task-relevant evidence를 얼마나 잘 제공하느냐**, 그리고 최대 길이보다 작은 context가 왜 더 효과적일 수 있느냐가 정확한 문제다. [TabPFN-3 report](https://arxiv.org/abs/2605.13986v2)

context에는 적어도 세 역할이 있다: decision boundary를 지지하는 사례, local density/빈도, 여러 공격 mode의 존재 증거. 전체 dedup이 악화된 이유를 “중복이 항상 좋다”로 일반화하지 말고, **빈도 보존과 피복을 서로 다른 제약으로 관리**한다.

**C1: 안정적인 C0를 기준으로, class × mode 피복을 명시한다.**

- 출발점은 검증된 100k·S=.75·공격 균등·n=4. .75와 100k가 이론적으로 최적이라고 주장하지 않는다.
- train 구간 안에서 클래스별 관측 가능한 feature mode를 만든다. scenario metadata는 피복 진단과 train-only 층화에 사용할 수 있으나, query의 true scenario로 라우팅하지 않는다.
- rare class 내부의 모든 충분히 관측된 mode에 최소 지원을 두고, 남은 예산을 conditional density를 유지하는 방식으로 뽑는다. background/benign 원래 선택을 먼저 고정한 뒤 attack 슬롯 안에서만 mode coverage를 바꾸면 원인 판독이 쉽다.
- 같은 class prior를 유지한 `random`과 `coverage`를 비교하고, 같은 coverage 아래 prior만 바꾸는 실험을 별도로 한다. 이렇게 해야 피복 효과를 단순 prior 효과와 분리할 수 있다.

유한 pool N개에서 mode g가 n_g개이고 균등 비복원으로 B개를 뽑는다면, 해당 mode를 전혀 보지 못할 확률은 정확히

\[
P(N_g(C)=0)=\frac{\binom{N-n_g}{B}}{\binom{N}{B}}
\le (1-n_g/N)^B \le e^{-B n_g/N}.
\]

이는 작은 mode를 random context가 누락할 수 있다는 직접적인 근거다. 여러 mode에 union bound를 적용해 피복 예산을 잡거나, 알려진 mode에는 최소 quota를 강제해 누락을 막는다. 단 **모드를 아직 발견하지 못했거나 train에 0개라면 이 보장이 적용되지 않는다.** 모드의 최소 표본 수를 정하는 것과 PFN 정확도를 보장하는 것은 별개다.

**C2: expert context는 tail positive와 혼동 negative를 함께 담는다.**

각 query에 대해 global top1/top2에만 의존하지 않고, **모든 클래스의 train index에서** 작은 이웃 집합을 검색한다. 그 거리·지지율로 candidate context를 만든다. 이렇게 해야 global이 benign을 높은 확률로 예측한 tail도 탐색 대상에 들어간다. 선택되는 context row의 train label을 쓰는 것은 합법적인 supervised retrieval이며 query label은 사용하지 않는다.

예를 들어 web specialist는 web 사례만 늘리는 대신, 그 web 사례 주변에서 같은 모양을 갖는 benign hard negatives와 함께 구성한다. infiltration specialist도 inf-only/benign-only/mixed의 과거 지지, sharp feature 경계에서의 반대 라벨 증거를 함께 보게 한다. 정확한 충돌 벡터는 지워서 답을 정하지 않고, 과거의 label counts를 별도 audit/feature로 남긴다.

context budget을

\[
C_k=C_{anchor}\cup C_{support,k}\cup C_{contrast,k},
\quad |C_k|\le B_k
\]

로 고정한다. anchor는 정상/background 질량의 기준점, support는 고유 mode 피복, contrast는 오경보를 억제하는 대조 증거다. 현재도 anchor와 residual block이 있으므로 **이 합집합 형태 자체가 기여는 아니다.** 바뀌는 핵심은 residual 크기가 아니라 **관측 가능한 confusion과 상대적 결정 손익을 기준으로 support·contrast를 고르는 것**이다.

class/mode cell별 대표점 선택에는 weighted facility-location을 피복 proxy로 쓸 수 있다.

\[
F_g(S)=\sum_{i\in D_g}w_i\max_{j\in S}\operatorname{sim}(z_i,z_j),
\qquad |S|\le b_g.
\]

고정된 비음수 similarity·weight와 cardinality 조건 아래 이 함수는 monotone submodular다. max가 현재 대표점의 similarity를 넘을 때만 증가하므로 집합이 커질수록 추가 이득이 줄어든다. 표준 greedy는 이 **피복 proxy**에 대해 `1−1/e` 근사 성질을 갖는다. 이 근사를 PFN의 F1 보장으로 옮기지는 않는다. cell별 분리 objective/고정 quota라면 각각 적용할 수 있으나, 임의의 추가 제약을 섞은 전체 선택기에 같은 비율이 자동 유지되는 것도 아니다.

이 근사의 근거는 [Nemhauser·Wolsey·Fisher, 1978](https://thibaut.horel.org/submodularity/papers/nemhauser1978.pdf)다. `F_g(∅)=0`으로 두고 정확한 greedy marginal을 선택할 때의 성질이다. 후보 축소·근사 greedy를 사용하면 해당 제한 후보집합/근사 오차에 맞추어 주장 범위를 줄인다. 또한 위 `w_i`는 **선택 목적의 질량**이지 TabPFN에 자동 전달되는 sample weight가 아니다. 원래 빈도의 보존은 anchor와 cell 내부의 표본추출로 구현하고, 대표점 하나에 가중치만 붙이면 원래 context와 같아진다고 가정하지 않는다.

이론의 주장은 작게 유지한다: **random의 rare-mode 누락을 줄이고, 정해진 budget에서 관측된 mode의 대표성을 최적화한다.** PFN의 예측 성능 연결은 실험으로 검증한다. 모든 행 쌍을 계산하지 않고 train-only 후보 축소·mini-batch selection을 쓰면 CPU 비용을 통제할 수 있다.

거리 공간도 ablation이다. log/robust-scaled raw, train-fit z, 별도 unsupervised PFN embedding 중 어느 표현에서 선택하는지 비교한다. [LUCoS](https://arxiv.org/abs/2605.27254)는 이 구분의 근거이고, [BAPS](https://arxiv.org/abs/2608.12989)는 구성 방법 대조군이다. 단, 기존 raw kNN 재현의 결과를 새 v3 latent retrieval의 효과라고 해석하지 않는다.

**C3: context 크기와 config variance를 직교화한다.**

기존 seed 스윕은 row sampling과 `TabPFNClassifier(random_state=seed)`를 동시에 바꿨다. 0902 §8ah~8ai에서 이 혼동이 드러났다. 다음 context 비교에서는 `seed_context`, `seed_model_config`, `seed_router`를 분리하고, 같은 C0 row IDs와 같은 ensemble config를 공유한다. 크기 비교도 이 둘을 고정한 nested sample을 포함한다. 1M에서의 하락을 attention dilution으로 부르는 것은 현재로서는 기전 가설이며, attention/config/duplicate effects를 분리하기 전에는 결론으로 쓰지 않는다.

**9. 두 축을 묶는 연구 아이디어 — expert × context action의 공동 목적**

유망한 논문 방향은 “좋은 context를 별도로 찾고 router를 추가했다”보다 **context의 가치를 실제로 라우팅 가능한 교정 이득으로 정의한다**는 것이다.

expert를 `k`, 그 expert의 context recipe 또는 budget을 `b`라 하면 action을 `(k,b)`로 정의한다. `k=0`은 global 유지다. 같은 frozen backbone이어도 context가 다르면 action outcome이 달라지므로, 독립적인 class-router와 context-selector의 점수를 곱하는 대신 action별 비용을 일관되게 평가한다.

\[
\min_{\mathcal C,\pi}\;
\mathbb E\left[\ell(Y,\hat y_{\pi(Z),\mathcal C}(X))
+\lambda\,cost(\pi(Z))\right]
\quad\text{s.t. context budgets, mode support, benign-risk constraint}.
\]

이 방향의 직접적인 이론 선행연구는 [Expert-Conditional Advice](https://arxiv.org/abs/2603.14324v5)다. 그 논문은 expert와 추가 정보의 결합 action에 대한 surrogate를 제안하고 특정 분리 surrogate의 비일관성을 보인다. **현재 RACE의 scorer/verifier 두 컴포넌트가 곧 그 반례라는 뜻은 아니다.** 위처럼 context까지 action으로 선택하는 확장에 적용할 근거이며, 선택한 surrogate·가설 공간·손실·표본 가정이 맞는지 별도로 확인해야 한다.

구현은 joint end-to-end 학습보다 작게 시작할 수 있다. frozen C0, 하나의 specialist, context recipe 두 개만 두고 full action-cost table을 만든다. context 후보는 design fold에서 구성하고, route policy는 다른 fold에서 학습하며, 다음 시간 fold에서 bank의 **실현 이득**을 비교한다. bank를 고를 때의 목적을 `label oracle 최대`에서 `held-out 실제 정책의 비용 감소`로 바꾼다. 후보를 반복 선택한 결과를 같은 fold의 최종 성능으로 보고하지 않도록 outer fold를 둔다.

이 구조가 피해야 할 선행 중복은 명확하다. context bank는 MixturePFN, 정보 보존 구성은 BAPS, batch context matching은 CRUMB/LoGIC, conditional TFM 호출은 HINT, expert-information joint decision은 L2D with advice에 이미 있다. 따라서 가능한 추가 기여는 **중복·강한 imbalance·시간 이동 아래 어떤 context 증거가 상대적 교정 결정을 식별 가능하게 하는지**, 그리고 그 증거와 정책을 묶었을 때의 실제 tail–benign–cost 개선이다. 현재 조사만으로 “최초”를 주장할 단계는 아니다.

**10. verifier의 안전성을 보정하는 방식도 목표에 맞춘다**

현재는 D_cal에서 `q_hat−G`의 분위수로 q_corr를 만들고, 같은 D_cal에서 threshold grid를 선택한다. marginal lower-bound coverage 90%는 **선택된 override 집합의 harmful rate ≤10%**를 뜻하지 않는다. 선택률이 작을수록 전체 coverage와 선택 집합의 위험은 더 달라질 수 있다. 같은 calibration set을 이용한 77개 후보 선택도 고려해야 한다.

latest seed 42는 calibration grid에서 feasible point를 찾았지만 test helpful/harmful이 98/162다. 이를 “90% coverage 실패”로 곧바로 해석할 수는 없다. 원하는 위험 지표가 다르고 시간·prior·중복 구성이 다르기 때문이다. 개선책은 목표 지표에 대한 정책 단위 보정이다.

**보정 모집단도 맞춰야 한다.** seed 42의 `0c_split_manifest.csv`와 `cal_cap_per_class=50000`을 대조하면, D_cal은 클래스별 cap 후 234,163행이며 benign은 50,000행, **21.35%**다. test의 benign은 **87.07%**다. 이후 hash mask가 116,378행을 남기므로 실제 보정 prior는 다시 달라질 수 있다. 이 상태에서 calibration의 전체 helpful/harmful 비율을 자연 트래픽의 동일 비율로 해석할 수 없다. 이는 셧다운/위험 이동의 **검증할 원인 후보**이며 단독 원인으로 확정하지 않았다.

따라서 class-conditional risk를 먼저 추정하고 운영 prior로 조합하거나, 자연 비율의 별도 calibration subset을 사용한다. sampling inclusion probability를 아는 경우 가중 추정도 가능하지만 가중치 분산과 위험 상한을 같이 고려한다. 라벨 prior 보정은 `P(Z|Y)`까지 바뀌는 시간 이동을 해결하지 못하므로 forward 검증을 유지한다. benign-conditional FPR는 순수 label-prior 변화에 불변일 수 있지만 전체 harmful fraction은 그렇지 않다는 점도 구분한다.

1. policy/threshold 후보를 training/tuning 구간에서 미리 고정한다.
2. 운영 모집단을 반영하는 별도 calibration 구간에서 **benign에 새로 추가한 FP**, helpful/harmful correction, 호출률의 상한을 측정한다.
3. 후보 여러 개 중 고르는 경우 [Learn then Test](https://arxiv.org/abs/2110.01052)의 다중 검정 방식이나, 후보 선택용/위험 평가용 split 분리를 사용한다.
4. time blocks를 이동시키며 다음 구간에서 위험을 측정하고, 원래 조건이 성립하는 범위에서만 보장을 기술한다. [Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415)는 시간 이동을 무시한 보장의 한계와 가중 보정의 조건을 다룬다.

단순한 지표 예시로, **고정 정책·독립적인 Bernoulli 관측** 아래 harmful=0인 n건을 봤을 때 95% one-sided 상한은 `1−0.05^(1/n)`이다. n=100이면 약 2.95%, n=3,000이면 약 0.10%다. 같은 vector의 사본 3,000개를 독립 증거 3,000개로 취급할 수는 없다. CIC2018에서는 표본/블록 의존성을 명시하고 block bootstrap을 경험적 불확실성으로 제시한다. block 수가 적다면 그 한계까지 보고한다.

이 단계의 목적은 문턱을 무작정 낮추는 것이 아니라 **실제로 구별되는 유망한 subset에 대해 위험 예산을 배분하는 것**이다. 구별되는 subset이 없으면 feature/context evidence를 보강하는 A·C로 돌아간다.

**11. 다른 IDS로 확장하기 전 실행할 최소 실험 순서**

| 순서 | 바꾸는 것 | 고정할 것·대조 | 판독과 다음 분기 |
|---|---|---|---|
| E0 — 이번에 완료 | 기존 run 재분석 | 최신 100k·n=4 7 seed | global과 routing 기여 분리, NLL label 생존과 correctness transition 확보. |
| E1 | 최신 C0에서 고정 expert의 dense action dump | C0 row/config/실제 외부 split, K=1 우선 및 작은 K bank | 같은 기준선의 label oracle·hash-consistent oracle·단일 best expert를 계산. 이후 정책 실험은 캐시 재사용. |
| E2a | normalized-gain verifier에 +z, 그 다음 별도 +raw46 | target·bank·예측 cache 고정, scorer 우회 | forward-held-out에서 같은 채택률의 손익이 좋아지는지. 클래스 전체 상관보다 inf/web/benign confusion별 판별을 본다. |
| E2b | verifier target을 상대 action cost로 변경 | E2a에서 정한 입력·bank 고정 | helpful/harmful 및 같은-label 호출 비중. 더 강한 단일 XGB/stacking과 같은 정보를 주고 비교. |
| E3 | old b_oof label 제거, 가치 기반 scorer 학습 | 검증된 post policy·bank 고정 | 허용 호출률에서 dense policy의 유익한 변경을 얼마나 유지하는지, harm와 p95 latency. |
| E4 | class×mode 피복 context, 그 다음 contrast evidence | 총 row budget·class prior·model config 고정 | random 대비 피복과 실제 교정 이득의 관계. E4a 피복과 E4b hard negatives는 분리. |
| E5 | action `(expert, context)` 선택 및 policy-aware bank 비교 | 동일 bank 총 row·anchor·config·시간/메모리 예산 | residual/proximity/random/BAPS 및 global에 같은 row를 추가하는 대조보다 실현 이득이 좋은지. |
| E6 | 선택된 최종 구조의 forward episode 평가 | 기존 test를 더 튜닝하지 않고, outer training 안에 새 pseudo-future fold 구성 | 방향이 반복되는지 확인한 뒤 다른 IDS로 확장. CIC의 최종 untouched 평가가 없으면 개발셋으로 명시. |

**우선순위는 E1→E2→E3이다.** global context의 작동 가능한 기준선이 이미 있으므로, 지금 다시 C0의 비율·크기를 넓게 탐색하기보다 “좋은 expert 출력이 주어졌을 때 현재 정보로 안전하게 사용할 수 있는가”를 먼저 푼다. 그 다음 그 교정층이 필요로 하는 evidence를 C에서 보강한다. 이는 context가 덜 중요해서가 아니라, 두 축의 효과를 분리하기 위해서다.

시작할 때 미리 정할 성공 기준은 다음이다. 수치는 보장이 아니라 **제안하는 실용적 검증 기준**이며, 운영 비용이 정해지면 바꿀 수 있다.

- 1차: 같은 global 대비 tail macro-F1 +0.01을 최소 유의미 개선의 목표로 두고, paired forward episode/block CI와 함께 제시한다. +0.01에 못 미친 결과도 전부 기록한다.
- 안전: benign FPR 증가 허용치를 먼저 정한다. 초기 후보는 기존 기록과 연결되는 0.0005이며, row count와 추가 FP 수를 같이 표시한다. 위험 상한을 주장할 때는 앞 절의 가정을 만족해야 한다.
- 분리: inf와 web 각각 P/R/F1·AUPRC, 도움이 된 변경과 해로운 변경을 보고한다. 쉬운 bot로 tail 평균을 올리는지 확인한다.
- 비용: 평균 호출률, 같은-label 채택률, 실제 wall-clock/p95, peak memory. 호출 수를 wall-clock 배수로 환산하지 않는다.
- 성립 조건: E1에서 현재 C0 위의 deployable policy headroom이 작다면 C/새로운 허용 feature로 정보의 질을 개선한다. raw feature에도 구별 신호가 없는 충돌 구간에서는 학습 가능한 비충돌 구간의 개선과 비용 최적 선택을 구분해 진행한다.

고정 threshold 보정, XGB-C0/full, XGB+공정하게 선택한 calibration, global+같은 추가 context rows, K=1, simple stacking은 필수 대조다. LoCalPFN/BoostPFN의 과거 수치는 v1 backbone이므로, wrapper 기여를 주장할 때는 v3 위의 동등 구현이나 backbone factor를 별도로 둔다. 입력 정보와 메모리뿐 아니라 전처리·캐시·config ensemble 수도 맞춘다.

현재 outer split은 per-scenario chrono다. 내부 scenario 층화는 그 outer train 안에서만 수행해야 한다. 단일 전역 시점까지의 라벨만 가능한 배포를 주장하려면 global-time forward split이 별도로 필요하다. 실제 배포에서 과거에 관측 가능한 timestamp/session 집계 정보를 추가할 수 있다면 충돌을 푸는 새 정보원이 될 수 있지만, attack 일정으로 만든 scenario 정답이나 미래 집계를 feature로 쓰는 방식은 허용하지 않는다.

**12. 제안의 이론적 범위와 이번 작업의 결론**

확실히 말할 수 있는 것은 네 가지다. **(i)** 제한된 context에서 rare mode가 사라질 확률을 계산하고 피복을 설계할 수 있다. **(ii)** frozen action의 최적 선택은 상대적 조건부 기대손실 비교다. **(iii)** label oracle의 이득에는 관측 정보로 실현할 수 없는 몫이 있다. **(iv)** marginal calibration과 선택된 교정의 위험 보장은 다르다.

아직 검증해야 할 것은 해당 표현·context recipe·학습기가 CIC2018에서 실제로 이 조건을 얼마나 근사하는가다. 따라서 “routing이 안 된다”에서 멈추는 대신, **목표 연결 제거 → verifier 정보 보강 → 결정 손익 학습 → 그 결정을 지지하는 context → 호출 비용 최적화**의 순서로 원인을 직접 겨냥한다. 모델의 다음 기여 후보는 context나 두 gate의 존재가 아니라, **context가 제공한 증거를 학습 가능한 tail 교정으로 바꾸는 일관된 의사결정 구조**다.

재현: `python docs/research/20260909/audit_existing_runs.py`. 논문 보충표는 [CSV](literature_supplement.csv)와 [XLSX](literature_supplement.xlsx)로 제공한다. [조회한 서지 메타데이터](arxiv_metadata_checked.csv)는 제목·버전·최초 제출·갱신일을 보존한다. 새 학습 결과나 oracle upper bound의 새 실측은 이번 산출물에 포함되어 있지 않다.

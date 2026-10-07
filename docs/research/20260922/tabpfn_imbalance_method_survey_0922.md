# TabPFN × 클래스 불균형 — 방법 논문 조사 (2026-09-22)

목적: `docs/tabpfn_papers_survey.xlsx` Ch.2(TabPFN을 개선·확장한 방법)에 해당하는 **불균형 처리 방법** 논문을 학회 accepted 목록을 순회하며 찾고, 각 논문의 method·평가 방식·차용할 아이디어를 정리한다. 도메인 응용(Ch.3 대부분)은 제외한다.

관련: `docs/tabpfn_papers_survey.xlsx`(시트 반영), `lablog/report/0918.md`(EXP49–56), `../SRC_HISTORY.md`(실패 유형 1–6).

---

## 0. 요약 판정

1. **본회의 문헌은 사실상 한 편이다.** ICML 2025/2026·AISTATS 2025/2026·UAI 2025/2026 본회의 11,852편(PMLR·virtual JSON 전수), ICLR 2025/2026·NeurIPS 2024/2025(OpenReview 검색), KDD/AAAI/IJCAI/CIKM/SDM/ICDM/WWW/WSDM/PAKDD(Crossref DOI prefix 전수), FMSD@ICML 2025/2026(70+151편)·TRL@NeurIPS 2024(51편)·EurIPS 2025 AI-TD(43편)·AutoML 2025/2026 전수를 훑었을 때 "TabPFN/PFN × 클래스 불균형"을 주제로 한 방법 논문은 **DistPFN(ICML 2026)** 하나다. PFN-Imb(IEEE ITW 2026)는 FMSD@ICML 2026 워크숍 미러가 있을 뿐이다. 이 주제는 아직 본회의 문헌이 형성되지 않았다.
2. **가장 일관된 신호: 불균형 이득의 대부분은 결정 규칙(threshold) 효과다.** 서로 독립인 세 출처가 같은 방향이다.
   - PFN-Imb: thresholding(τ=π₁)이 balanced acc 0.710→0.793, worst-class 0.446→0.729로 최선; 오버샘플링·TabPFGen 합성은 악화.
   - L2C-TFM(RL 데이터 정제): 소수 클래스 macro-F1 이득 +0.035 중 threshold 튜닝 무관 잔여분은 +0.008(p=0.80).
   - Cai(43개 데이터셋 remedy 격자, 정오표): SMOTE의 TFM 상호작용 −0.083(raw 0.5) → 검증 threshold 선택 후 −0.004 → Platt 후 −0.002.
   → **s44 계열의 어떤 구조 개선안이든 "threshold-matched XGBoost 기준선"과 비교하지 않으면 주장이 서지 않는다.**
3. **합성 오버샘플링(SMOTE·TabPFGen)을 컨텍스트에 넣는 것은 TFM에 해롭다**는 결과가 네 독립 출처(PFN-Imb, CreditTFM, Cai v1, SAR-Crop)에서 반복된다. Cai의 표현: "행을 만드는(fabricate) remedy와 빼는(remove) remedy의 구분"이며 언더샘플링·threshold 보정만 TFM에 전이된다.
4. **실제 지렛대는 컨텍스트 구성이다.** CreditTFM은 컨텍스트 전략 간 AUC 격차(0.03–0.05)가 TFM 모델 간 격차보다 크다고 보고하고, Fair-TabICL은 "균등 균형 표집은 거의 효과 없고 경계(불확실) 표본 선택이 효과", IM-Context는 "전체 학습셋 컨텍스트는 꼬리 구간에서 편향을 키운다"는 상계를 준다.
5. **평가 프로토콜에서 배울 것이 방법보다 많다.** 고정 예산 IR 스윕(AWARE 5→500, ClinImb), Many/Median/Few 구간 분해(IM-Context), worst-class coverage(Sanghi), 4-way fold(context/bias-estimation/calibration/test), 비용 사전 선언 threshold(Song), AUPRC+ECE/Brier 병기(Cai: 미처리 TFM이 최고 보정이고 모든 remedy가 ECE 6–7배 악화), 저장 확률 배열로 결정 규칙 오프라인 재계산(Cai).

---

## 1. 조사 범위와 방법

| 갈래 | 대상 | 도구 | 결과 |
|---|---|---|---|
| OpenReview 계열 | ICLR 2025/26, NeurIPS 2024/25, TMLR, AutoML/CoLLAs | OpenReview `/notes/search` 45쿼리(forum·PDF는 CAPTCHA 차단), TRL/AI-TD 사이트 | 본회의 0편; TMLR 2편(IM-Context, Fair-TabICL); reject 3편(TAAR, ProFiT, RCB) |
| PMLR/virtual | ICML 2025/26, AISTATS 2025/26, UAI 2025/26, ACML, CoLLAs | PMLR 볼륨 인덱스, icml.cc/aistats virtual JSON, OpenAlex 34쿼리, arXiv API | 11,852편 중 DistPFN 1편; 인접 이론 다수(§4) |
| KDD/AAAI/저널 | KDD/AAAI/IJCAI/CIKM/SDM/ICDM/WWW/WSDM/PAKDD, 주요 저널 | Crossref DOI prefix 스윕, OpenAlex 2,037편 필터, Zenodo | KDD 2026 1편(Stream-TabPFN), ECML PKDD 2026 1편(AttnDistill), WWW 2026 2편(FT-Explore, TFM4GAD); AAAI/IJCAI/SDM/ICDM/CIKM/WSDM 0편 |
| 워크숍/arXiv/인용 | FMSD 2025/26, TRL 2024, EurIPS AI-TD, AutoML, arXiv API 18쿼리, TabPFN v1/v2·TabICL 인용 86편 | arXiv Atom, OpenAlex cites 필터 | 워크숍 1편(CURE), 프리프린트 다수; 인용 86편은 99% 도메인 응용 |

차단된 도구: DBLP(Anubis 봇 차단, 3개 미러 전부), OpenReview forum/PDF(브라우저 검증), Semantic Scholar(429), Springer/MDPI 본문(403; MDPI는 r.jina.ai 경유 일부 성공). 그래서 TAAR·ProFiT·TabPFN v2.5 개방환경 평가(FMSD 2026)는 에이전트 요약만 있고 본문 미확인이다(§5).

---

## 2. 논문별 정리

표기: **불균형이 파이프라인 어디에 들어가는가** → C(컨텍스트 구성) / P(사전학습 prior) / F(파인튜닝) / A(사후 posterior 보정) / D(결정 규칙) / U(불확실성·집합 예측). ★ = 프로젝트 제약(얼린 TabPFN-v3, 100k 컨텍스트, 시간순 split, IR>1000)과 호환.

### 2.1 TabPFN/PFN 기반 방법·분석 (시트 Ch.2 해당)

#### DistPFN — Mitigating Label Shift in Tabular ICL via Test-Time Posterior Adjustment (ICML 2026, arXiv 2605.04363) [A] ★
- 방법: p̃(y) ∝ p̂(y)²/p_train(y), 즉 p̂에 α=p̂/p_train을 곱해 정규화. DistPFN-T는 τ=CE(p̂, p_train)로 온도를 정해 p̂·p̂_T/p_train. p_train은 컨텍스트의 경험 클래스 빈도. 모델 수정·학습 없음.
- 평가: OpenML 253개, 50/50 split, 5 seed. 라벨 시프트는 역빈도 오버샘플링 w_k ∝ p_k^{−β}(β∈{0,0.1,0.5,1,2,5})로 학습 셋만 재표집(test 고정). TabPFN-v2·LoCalPFN·TabICL; 베이스라인 EME/BBE(라벨 시프트 추정), GBDT/DL 다수. 지표 accuracy(주)·ECE·precision·평균 순위(CD 다이어그램, β=2).
- 수치: TabPFN-v2 β=5에서 0.727 → DistPFN 0.769 → -T 0.775; 심한 불균형(<0.4) 데이터셋 +0.095, 균형 +0.002. CostaMadre1 소수 recall 1.7→15.9%. 추론 시간 증가 ≈0.001 s.
- 차용: 다중 클래스 prior 보정의 기본형. 우리 exp37 재현(0907)에서 C0(균등) 위에서는 중립(−0.0008)이었는데, 이는 C0가 이미 균형이라 p_train이 균등에 가까웠기 때문으로 해석된다 — **C0-natural 컨텍스트 + DistPFN** 조합이 미검증.
- 주의: feature shift 미처리; 논문 지표에 F1 없음; oracle 갭 남음.

#### PFN-Imb — Correcting Class Imbalance in Prior-Data Fitted Networks (IEEE ITW 2026; FMSD@ICML 2026 미러, arXiv 2605.21742) [D, C] ★
- 방법: 비용 최소화에서 τ = C₁₀/(C₁₀+C₀₁) = π₁ 유도; balanced acc 최대점이 τ=π₁을 추적함을 실증. PFN은 균형에서 잘 보정되고 불균형에서 "다수 편향"만 생기므로(과신이 아니라) 단순 threshold로 교정된다. 언더샘플링(π₀=π₁까지), 오버샘플링(복제 → "spiky posterior"로 악화), TabPFGen 합성(악화) 비교.
- 평가: OpenML-CC18 이진 11개, test 클래스당 500 균형, 컨텍스트 N∈{100,500,1000}, π₁∈{0.05,0.1,0.2,0.3}; TabPFN-2.5; 클래스별 정확도·balanced acc·worst-class acc·ROC. seed 미명시.
- 수치(평균): None 0.710/0.446 → Thresholding 0.793/0.729, Downsampling 0.779/0.737, Oversampling 0.621/0.259, TabPFGen 0.691/0.407 (BA/WCA).
- 차용: **"threshold-to-prior" arm을 s44의 명명된 기준선으로**(재적합 없음, 확률 불변). 다중 클래스 확장은 저자도 미해결 — 클래스별 prior로 p̂/π를 나누는 형태(=DistPFN의 α와 동형)가 자연스러운 확장.
- 주의: 이진 전용; 컨텍스트 ≤1,000.

#### CreditTFM — Data Presentation Over Architecture (arXiv 2605.18635; 시트 Ch.3, 사용자 취소선) [C] ★
- 방법: 리샘플링을 **컨텍스트 구성 전략**으로 재정의. 7전략: Uniform, Stratified / Balanced(m_c≈m/K), Oversample+(역빈도 가중 + boost + **최소 소수 개수 하한**), SMOTE / Diversity-KM(MiniBatchKMeans 대표), Hybrid(ρ·Balanced + (1−ρ)·Diversity-KM).
- 평가: Home Credit(307K, 부도 8%), Lending Club(533K, 12–22%, **시간순 split**: train ≤2019-06, test 2019 H2). 컨텍스트 {1k,2k,5k,10k,20k,50k}. TabPFN·TabICL·OrionMSP v1/1.5·OrionBix vs RF/XGB/LGBM/CatBoost(클래스 가중·threshold 튜닝 없음). ROC-AUC 주, MCC·default recall·BA·F1 보조. seed 미명시.
- 수치: 평균 AUC(HC/LC) Uniform 0.703/0.673, Balanced 0.734/0.683, Hybrid 0.732/0.686, SMOTE 0.690/0.673, Diversity-KM 0.681/0.656. 전략 간 격차 0.03–0.05 > 모델 간 격차. GBDT는 zero-recall trap(XGB HC recall 0.0%, MCC 0.000); 50k balanced 컨텍스트에서 TFM 5종 MCC 0.19–0.26, default-F1 0.24–0.31, BA 0.65–0.71.
- 차용: **Hybrid(ρ 한 knob)와 Oversample+의 최소 소수 하한**은 C0-균등의 직접 일반화. IR>1000에서 "클래스당 균등"은 100k 안에 불가능하지만 "family당 ≥n행 보장 + 나머지 역빈도"는 가능. SMOTE-in-context 음성 결과로 해당 arm은 생략.
- 주의: 이진, IR ≤ ~12; GBDT가 threshold 튜닝 없이 평가돼 zero-recall 수치는 불공정 — 우리 s44는 이미 threshold를 맞추므로 3–4점 선물은 기대 금물.

#### IM-Context — In-Context Learning for Imbalanced Regression (TMLR; arXiv 2405.18202; 코드) [C] ★
- 방법: 얼린 PFN 인코더(Müller 2023, 18차원)·GPT-2 ICL 모델에서 **쿼리별 kNN 컨텍스트**: 원본 셋에서 k′개 + "역밀도 데이터셋"(구간 표현 비율에 반비례해 재표집)에서 k̃개 이웃(코사인 유사도). k′=k̃=15(PFN). 이론: c-Lipschitz 가정 아래 E[오차] ≤ 편향² + σ²/k + σ²; 다수 구간은 k 증가로 단조 감소, **소수 구간은 편향 지배로 U자** — 전체 학습셋 컨텍스트가 꼬리에 불리한 이유.
- 평가: AgeDB-DIR·IMDB-WIKI-DIR·STS-B-DIR + 표 6종; **Few(<20)/Median(20–100)/Many(>100)** 구간별 MAE/MSE/RMSE, 균형 test, 3 seed. 베이스라인 LDS+FDS·RankSim·ConR·VIR·GBM/NN.
- 수치: AgeDB few-shot MAE 7.83(9.21), IMDB few-shot 16.33(20.96), STS-B few 0.566(0.780), 표 few-shot RMSE 3.31로 1위.
- 차용: (1) "100k 균등 컨텍스트는 다수 prior 주입"이라는 인용 가능한 논거. (2) **역밀도 보조 풀 + 원본 풀의 sub-budget 분할** — β를 지렛대로 쓰던 EXP52–54의 연장. (3) Many/Median/Few 분해 보고.
- 주의: 회귀; IR>1000에서 쿼리별 kNN은 꼬리 이웃을 0개 반환할 수 있어 역밀도 가지가 전부를 떠맡는다.

#### Fair-TabICL — Towards Fair In-Context Learning with TFMs (TMLR 2026; arXiv 2505.09503; 코드) [C] ★
- 방법: TabPFNv2·TabICL·TabDPT에 컨텍스트 개입 3종. (1) Correlation Remover(민감속성 투영 제거, α); (2) **Group-balanced**(다수 그룹 무작위 하향표집); (3) **Uncertainty-based**: 별도 20% 분할로 학습한 민감속성 분류기(LR/TabPFN)의 conformal 예측집합이 2개 이상인 "불확실" 표본만 컨텍스트에 포함(coverage ε로 조절).
- 평가: ACS PUMS 5종(8.6k–47.8k)·CelebA·Diabetes·German Credit; 80/20 split + 20% 예측기 학습 분할; 3 seed; Accuracy·ΔDP·ΔEOP·ΔEOD·민감속성 재구성 정확도.
- 수치(TabPFN·ACSIncome): Vanilla ΔDP≈14/acc 77 → Group-balanced 14/77 → CR 26/77(누출 100%) → Uncertain+TabPFN 5.5/82. Adult: acc 85.72→82.05, ΔDP 16.99→5.51.
- 차용: "그룹"을 "클래스"로 바꾸면 **균등 균형 표집이 약하고 경계 표본 선택이 강하다**는 C0 실험의 직접 대조군. IR>1000에서는 "꼬리 이웃 + 꼬리 경계의 어려운 다수 행"으로 채우라는 뜻. 단 불확실성 점수기는 라벨로 학습하므로 시간순에서는 **과거 구간에서만 적합**(SRC_HISTORY #2 회피).
- 주의: 그룹비 50/50~80/20; 이진.

#### Stream-TabPFN — In-context Learning of Evolving Data Streams with TFMs (KDD 2026; arXiv 2502.16840) [C]
- 방법: 무한 스트림을 on-the-fly 스케치로 요약해 얼린 TabPFN 컨텍스트로 공급하는 **슬라이딩 메모리**; 가중치 갱신 없음.
- 평가: 비정상 스트림 벤치마크; ARF·SRP(Hoeffding-tree 앙상블) 대비 전 벤치마크 우위(초록 기준, 수치 미확인).
- 차용: 시간순 split에서 "컨텍스트 교체가 유일한 적응 수단"이라는 프레이밍. 최근성 창은 일찍 한 번 나온 꼬리 family를 굶기므로 **최근성 × 클래스 하한** 두 knob이 필요(로그에 명시).

#### CURE — Bounded Context Management for TFMs on Stream Learning (FMSD@ICML 2026 spotlight; arXiv 2606.18677; 코드) [C] ★
- 방법: 예산 B=1000을 short FIFO 뱅크(75%)와 long 뱅크(25%)로 분할. 승인: 라벨 관측 전 예측분포의 정규화 엔트로피 h(z)=−Σp log p/log C ≥ τ(0–0.5)일 때만 long 뱅크로. 제거: 초과 시 **최다 클래스**에서 정규화 특징공간 L2 최근접 동클래스 쌍을 찾아 최근 중심에서 먼 쪽 제거. TabICL-v2 기본, LimiX-v1·TabPFN-2.5·TabDPT-v1로 전이.
- 평가: NOAA·METER·RIALTO·POSTURE-No8(불균형)·POKER·NOMAO·AGR(A) 7 스트림; VFDT/EFDT/ARF/SRP/LevBag/BOLE + DualFIFO; prequential accuracy·평균 순위.
- 수치: 최대 27.0% 상대 개선(+0.45~+19.59%p), 평균 순위 1.00(ARF 3.71, SRP 2.57); 스텝당 0.0024 s.
- 차용: 100k 예산 관리의 직접 대응물. 승인은 라벨 무관·제거만 클래스 인지라 **꼬리 클래스가 중복 판정에서 살아남는 보장이 없음** → "클래스당 최소 k행 floor"를 붙이는 한 칸 knob이 가장 좁고 검증 가능한 변경(에이전트 권고).

#### AttnDistill-TabPFN — How Far Can We Compress? Attention-Guided Dataset Distillation for TabPFN (ECML PKDD 2026; 코드) [C]
- 방법: ICD-TabPFN을 TabPFNv2로 확장. 합성 특징과 **라벨을 함께 최적화해 클래스 균등 배분 제약을 완화**; TabPFN attention 질량으로 증류 행을 반복 가지치기해 압축 한계 탐색. 평균 압축률 15.7%에서 정확도 유지, ICD·라벨-only 변형 상회.
- 차용: 클래스별 컨텍스트 예산을 "학습되는 양"으로 두는 발상; **가지치기 기준이 라벨이 아니라 모델 내부 신호**(검증 적합 아님, #2 회피). 합성 행 자체는 "행을 만드는" 계열이라 가지치기 절반만 권장.
- 주의: 본문 미열람(유료); 심한 불균형 실험 여부 미확인.

#### PUICL — In-Context Positive-Unlabeled Learning (arXiv 2605.05591) [P]
- 방법: TabPFN식 트랜스포머를 PU로 사전학습(SCM 합성 → pre-removal로 PU화 → 다양한 class-prior 노출); 추론은 라벨 양성+unlabeled를 한 행렬로 넣는 transductive.
- 평가: 반합성 PU 20개(라벨 양성 200/unlabeled 양성 400/음성 400), AUC·Acc·F1, PU 4종 대비 우위(수치표 미확인).
- 차용: **"prior가 그 class-prior 영역을 본 적 없으면 ICL이 못 한다"**는 근본 설명(얼린 TabPFN이 IR>1000에서 무너지는 이유). unlabeled test 행을 컨텍스트에 함께 넣는 transductive 형식은 s43/s44 unseen resolution에 이식 가능한 형태.
- 주의: 실효 불균형 1:1~1:2; 사전학습 필요.

#### TFM-ImbRemedy (Cai) — Class-imbalance remedies for TFMs: a workflow-dependent empirical evaluation (Zenodo 22721350 v2.0.1, 2026-09-12; 코드) [D, C] ★(비교 규칙으로)
- 설계: 43개 자연 불균형 이진 과제(IR 2.0–580.2), 5 seed, 출처군 38. TabPFN-v2/v3·TabICLv2·TabDPT vs 튜닝 XGB/LGBM/CatBoost·balanced RF. remedy: random OS·SMOTE·Borderline·ADASYN·Simplicial SMOTE·표준화 거리 표집 대조·언더샘플링·threshold 보정·DistPFN/-T. 저장된 확률 배열 17,200개로 **결정 워크플로 7종**(raw 0.5, 검증 BA threshold, Platt, 온도, 클래스조건 conformal…)을 오프라인 재계산. 클러스터 부트스트랩·Wilcoxon·Nemenyi.
- 수치: TabPFN-v2+TabICLv2, SMOTE, BA 상호작용 raw −0.0834 [−0.1434, −0.0235] → 검증 threshold 후 −0.0044 → Platt 후 −0.0022; raw AUPRC −0.0074(비유의). v1 관측: 미처리 TFM ECE 0.025(최고), 모든 remedy가 6–7배 악화; threshold-to-prior가 모든 리샘플링 상회. TabPFN-v3는 표준화 거리 표집에서 threshold 후에도 음의 상호작용 잔존.
- 차용: (1) remedy는 **반드시 threshold 규칙과 함께** 보고. (2) 행을 만드는 remedy(합성)는 버리고 빼는 remedy(언더샘플링·threshold)만. (3) **저장 확률 배열을 산출물로** — s43/s44가 체크포인트를 안 남기는 규약과 맞고, threshold/보정/conformal 스윕이 재실행 없이 공짜가 된다. (4) ECE를 결과 튜플의 네 번째 다리로.
- 주의: 동료 심사 없음·headline 철회; 전부 이진; TFM 컨텍스트 200–2,393행 vs GBDT 전체(데이터량 미분리); 30/215 분할이 양성 <10.

#### AWARE — Retrieval-aligned TFMs for EHR (arXiv 2604.01841; 시트 Ch.3, 사용자 취소선) [C]
- 방법: 얼린 TFM 앞에 **SNNL로 학습한 retrieval encoder**(같은 outcome은 compact, 다른 outcome은 분리) + 표본별 attention 특징 가중 → top-k 컨텍스트. 불균형 대응: **클래스 빈도 역비례 mini-batch 표집**으로 encoder 학습, 5-fold encoder 앙상블. kNNPFN/TabDPT에는 경량 adapter.
- 평가: MIMIC-IV·eICU·HIPE + OpenML/UCI 임상 12; **IR 5→500 스트레스 테스트를 학습 표본 10,000 고정으로**; AUROC·AUPRC(주)·F1; 환자 단위 층화(시간순 아님); 3 seed.
- 수치: IR=500에서 AUPRC +12.2%; eICU ablation AUROC +23.8% vs AUPRC +≈140%(지표 괴리); HIPE CPE AUPRC 0.141→0.232.
- 차용: 고정 예산 IR 스윕 프로토콜; AUROC/AUPRC 괴리 보고. encoder는 라벨로 학습하므로 시간순에서는 과거 구간 적합 ablation 필수.

#### BAPS (arXiv 2608.12989; 시트 Ch.2) [C]
- 5기준 프로토타입 선택(대표성·경계·국소밀도·다양성·클래스 보존) 가중합, 512개 예산, "class-aware budget allocation"이 있으나 **할당 규칙·λ 미공개**; HIGGS/SUSY 이진, BA·macro-F1·AUC·ECE, 5회, Wilcoxon-Holm(p=0.031). 심한 불균형 실험 없음.

#### L2C-TFM — Model-Aware Data Cleaning for TFMs (arXiv 2604.25154 v2) [D 교란]
- 얼린 TabPFN v2 앞 정제 순서를 PPO로 학습(보상 = 정확도 0.50 + 행 보존 2차 페널티 0.35 + 품질 0.15 + Wasserstein 0.05). OpenML CC18 10종, 8-seed 중첩 홀드아웃, accuracy·macro-F1·ECE. 결과 null(Δacc −0.0006, p=0.60). **소수 클래스 macro-F1 이득은 검증 fold threshold 튜닝(+0.035)이 대부분, 정제 고유 잔여분 +0.008(p=0.80).**
- 차용: 행 제거 remedy에 ICL의 O(1/√n) 비용을 2차 페널티로 가격 매기는 발상; threshold-matched control 의무화.

#### FT-Explore — Exploring Fine-Tuning for TFMs (WWW 2026; arXiv 2601.09654; 시트 갱신) [F]
- TALENT 155·CC18 63·TabZilla 27에서 zero-shot/meta-learning/SFT/PEFT × 6 TFM; **불균형 구간(IR<0.6)에서 meta-learning만 안정적으로 이득, SFT는 정확도·보정 악화**. TabPFN 불균형 구간: zero-shot 0.8808/0.8697(ACC/F1) → meta 0.8784/0.8664 → SFT 0.8709/0.8578.
- 차용: 얼린 TabPFN 선택의 외부 근거. 단 "불균형" 정의가 60:40 수준이라 수치는 인용 금물, 방향만.

#### TACTIC — Tabular Anomaly Detection via In-Context inference (arXiv 2603.14171) [P, D]
- anomaly-centric 합성 prior로 PFN을 사전학습해 **점수+사후 임계값이 아닌 판별적 결정**을 한 forward pass로. 차용: "prior가 희소 클래스를 표현해야 ICL이 된다" + "점수를 자르지 말고 결정을 학습" — Energy gate가 long-tail에서 chance로 퇴화하는 #6에 대한 외부 대응 논거. 사전학습 필요라 관련연구 인용용.

#### TabPFN 라이브러리 v9 `majority_downsample` [C]
- `InferenceConfig.SAMPLE_SUBSAMPLING_METHOD` 옵션: 최다 클래스 외 행 전량 보존, 나머지 예산만 다수 클래스에서 채움. 논문은 아니지만 Prior Labs가 채택한 "빼는 remedy". TabPFN-3 기술보고서에는 불균형 언급이 없다.

### 2.2 TabPFN을 쓰지 않지만 메커니즘이 이식되는 방법 (시트 Ch.3 해당)

#### AdapTable (TRL@NeurIPS 2024; arXiv 2407.10784; 코드) [A, D] ★(구조)
- 모델 무관 TTA. (1) shift-aware calibrator: 컬럼=노드 GNN이 배치 shift 추세와 로짓에서 표본별 온도 T_i를 내고, 배치 내 불확실성 분위수(0.25/0.75)로 2차 보정; 기준 온도 **T=1.5ρ/(ρ−1+1e−6), ρ=max_j p_s(y)_j / min_j p_s(y)_j**(소스 불균형비). (2) label distribution handler: p̄ = [p̃ + norm(p̃·p_t(y)/p_s(y))]/2, p_t(y) = (1−α)·mean(norm(p/p_s)) + α·p_t^{oe}, α=0.1, 배치 64. Theorem 3.1: 오차 격차 ≤ K₁‖1−p_t^{oe}/p_t‖₁·BSE + K₂·ΔCE. Ablation: 보정 없이 prior 재가중만 하면 과신-오답 표본이 p_t 추정을 망친다.
- 평가: TableShift 6종 자연 shift + 3종×6 corruption; MLP·CatBoost·AutoInt·ResNet·FT-T; TTA PL/TTT++/TENT/EATA/SAR/LAME; **macro-F1·balanced acc**("표 데이터의 극단 불균형" 때문에 accuracy 배제); 3회.
- 수치: HELOC macro-F1 53.2→65.8(본문 "26%", 초록 "16%" 불일치); corruption 10%+.
- 차용: 얼린 TabPFN 출력에 파라미터 0개로 얹는 구조; **시간순 NIDS 스트림은 family가 뭉쳐 도착하므로 배치 p_t(y) 추정 + EMA가 이례적으로 잘 맞을 조건**(청크별 prior 재추정, re-fit 비용 0); "보정 먼저, prior 조정 나중" 순서.
- 주의: TabPFN 실험 없음; IR>1000이면 배치 64당 꼬리 기대치 0.06개 → p_t 추정 붕괴(#6과 같은 메커니즘); T 공식은 ρ가 크면 1.5로 포화.

#### CalibOnlineLS — Calibration-Aware Online Adaptation under Label Shift (UAI 2026, PMLR 337) [A]
- 얼린 기본 분류기 + **따로 보정된 보조 분류기**로 스트림 라벨 비율을 온라인 추정해 기본 예측을 보정; 오차 = 소멸 항 + 보정 품질 항; 보정 세분도–유한표본 오차 트레이드오프. DistPFN이 "주어졌다"고 가정하는 타깃 prior를 구하는 구조. 수치는 PDF 미확인.

#### Plugin — An Efficient Plugin Method for Metric Optimization of Black-Box Models (arXiv 2503.02119; ICLR 2025 철회) [D]
- 블랙박스 확률 + 소량 타깃 라벨로 혼동행렬 지표(macro-F1 등)를 직접 최적화하는 사후 변환, 특징 불필요, 라벨 시프트·잡음 하 표·언어 검증. 우리 고정 운영점 CSV의 상위 개념. 검증 적합 규칙 → 과거 구간에서만 적합하고 전이 갭 보고.

#### RCB — Exploring Imbalanced Annotations for Effective ICL (arXiv 2502.04037; ICLR 2026 reject) [C]
- LLM ICL: 불균형 데모 풀의 성능 저하는 클래스 가중만으로 안 고쳐짐(조건부 편향). 균형 프로브로 조건부 편향 추정 → 데모 검색 점수를 재가중(최대 +5.4%). 풀 리샘플 대신 **검색 점수 재가중**은 TabPFN 컨텍스트 선택에 이식 가능; 심사에서 "극단 희소 꼬리 적용성" 지적.

#### Conformal 클러스터 [U]
- **Conformal-LT**(ICLR 2026, arXiv 2507.06867): prevalence-adjusted softmax(macro-coverage 최적화) + marginal↔class-conditional 임계값 선형 보간. Pl@ntNet-300K(1,081 클래스)·iNat-2018(8,142).
- **MacroCov-CP**(arXiv 2606.28598, Bhattacharyya·Ding·Barber): macro-coverage(클래스별 coverage 비가중 평균)에 label-weighted conformal 유한표본 보장; 그룹핑 일반화; 최소 집합 특성화. **우리 "known macro-F1 including tail"의 conformal 대응물**이며 클래스별 보정점이 극소수일 때의 현실적 목표.
- **Quiet Failure**(arXiv 2607.06605): marginal CP가 전역 90%를 맞추며 소수 coverage 4.2%까지 붕괴; **보존 항등식**(소수 부족분 = 다수 잉여 × IR)으로 격차를 1%p 이내 예측; Mondrian CP가 복구. RF/GNN/화학 LM.
- **MAKE 8(7):190**(2026-07, AlThani): IR 1:345에서 표준 CP 이상 coverage 52.94% → Mondrian 90.59%; 실패 조건(기본 판별력 낮음·소수 보정셋 작음) 명시. XGB/RF/NN.
- **Sanghi — Classwise Conformal Coverage for Frozen TabPFN v2 under Imbalance**(Zenodo 보조자료만, 2026-08-15; 본문 미발견): OpenML 다중 클래스 20개, **context / bias-estimation / calibration / test 4-way fold**, worst-class coverage 분포, leave-one-dataset-out. "reviewer-requested" 표현으로 심사 중 추정.
- 차용: s44의 unseen resolved accuracy 0.125(중앙값)를 point prediction 대신 **prediction set + abstention 비용**으로 재정의하는 출구; 4-way fold로 prior 보정량 추정(bias-estimation)과 threshold 선택(calibration)을 분리하면 #2를 구조적으로 차단. 시간순 split은 교환가능성을 깨므로 weighted/adaptive conformal 필수 — 생략하면 #2의 새 옷.

#### Song 2026 — TabICLv2 비용 민감 비교 (J. Comput. Technol. Appl. Math.) [D]
- 임계값을 **사전 선언한 OOF 비용 C=10·FP+500·FN**으로 잠그고 OOF AUPRC로 모델 선택, test 봉인. Scania 10k: AUPRC +0.033 [0.021, 0.046], 비용 −3,454(유의), Brier 0.00685 vs 0.00840; SECOM 비유의; **TabICLv2 413× 계산**을 그대로 보고.
- 차용: family별 c_FP·c_FN을 사전 선언하면 "known macro-F1 / OOD / retention" 튜플이 하나의 비용으로 잠기고, 검증 적합 없이 threshold를 고정할 수 있다(비용은 fitted가 아니라 declared). 50–331× 비용 갭을 정직하게 병기하는 전례.

#### Ensembling TFMs: Diversity Ceiling and Calibration Trap (FMSD@ICML 2026; arXiv 2605.18696) [진단]
- 6 TFM의 쌍별 Q-statistic 0.961(거의 중복) → 어떤 convex 결합도 상한에 묶임; 최고(2단 cascade stacking) +0.18% acc를 253× compute로; LR meta-learner는 경계를 날카롭게 해 보정 파괴. 차용: s44 expert 앙상블이 "diversity ceiling"에 걸렸는지 Q-statistic으로 먼저 측정(#3의 정량 진단).

### 2.3 우리 도메인(NIDS)·불균형 벤치마크 평가 논문 (참고)

- **FM-CyberEval**(Electronics 2025; FLLM 2025 자매): CIC-IDS2017·N-BaIoT·CIC-UNSW, 클래스당 20k(저널)/3k(FLLM) 캡 = 유일한 불균형 장치("빼는" 계열), TabPFN/TabICL만 전 클래스 non-zero recall(Heartbleed·Infiltration 포함). **무작위 층화 split**이라 99.59%는 s05/s06이 겪은 누수 서명 — 도메인 관련연구로 인용하되 split 결함을 명시.
- **Butt·Hotho·Schlör, Evaluating Tabular Representation Learning for NIDS**(IEEE CSR 2026; arXiv 2605.02519): NetFlow 벤치마크에서 TabICL이 CIDDS 최고, supervised ≫ unsupervised AD, cross-dataset 전이는 조합 의존. NetFlow-v3 커뮤니티와 같은 그룹.
- **TabPFN-SmartCity**(arXiv 2604.11394; 시트 취소선): ToN-IoT·TabPFN v2.5, RF 대비 40× 빠른 추론, scanning F1 69.8%(최악 클래스) — ToN 클래스별 비교용 외부 수치.
- **ClinImb**(Sci Rep 2026): MIMIC-IV-ED에선 TabPFN v2.6/TabICL 최상, eICU에선 XGBoost 일관 우위; **IR을 인위로 올리며 macro-F1 열화 곡선**을 그리는 프로토콜 — "TabPFN이 XGB에 진다"를 "IR=X를 넘으면 진다"로 바꾸는 한 칸 ablation.
- **BeyondArena**(시트): 142 non-IID 데이터셋에서 tree/DL이 여전히 우위 — s44 결과를 field-level 현상으로 프레이밍.
- **MAKE 8(8):244 critical review**(2026): 961건 스크리닝, 142 non-i.i.d.에서 tree/DL 우위, 112개에서 TFM의 conditional coverage가 GBDT보다 약함.
- SMOTE가 TFM에 해롭다는 도메인 재현: SAR-Crop(원본 불균형이 최선, SMOTE 3종 악화), EMBC 2025 SeqXGB(SMOTE 실패), EV-Crash(SMOTEENN).

---

## 3. 평가 프로토콜 비교

| 논문 | 불균형 생성 | 데이터/규모 | split | seed | 주지표 | 보조 |
|---|---|---|---|---|---|---|
| DistPFN | 역빈도 β-오버샘플링(train만) | OpenML 253 | 50/50 무작위 | 5 | accuracy | ECE, precision, rank |
| PFN-Imb | π₁∈{.05,.1,.2,.3}, N∈{100,500,1k} | CC18 이진 11 | test 균형 500/class | 미명시 | balanced acc | worst-class acc, ROC |
| CreditTFM | 자연(8%, 12–22%), 컨텍스트 1k–50k | HC 307K, LC 533K | LC만 시간순 | 미명시 | ROC-AUC | MCC, recall, BA, F1 |
| Cai | 자연 IR 2–580 | 43 이진 | 무작위, 215 분할 | 5 | balanced acc | AUPRC, ECE, 7 결정 워크플로 |
| AWARE | IR 5→500 스윕, 학습 10k 고정 | MIMIC·eICU·HIPE+12 | 환자 단위 층화 | 3 | AUPRC, AUROC | F1 |
| IM-Context | 라벨 구간 Few/Median/Many | 8 (표 6) | 균형 test | 3 | MAE/RMSE 구간별 | GM |
| Fair-TabICL | 자연 그룹비 | ACS 5 + 3 | 80/20 + 20% 예측기 | 3 | ΔDP/ΔEOP/ΔEOD | acc, 재구성 정확도 |
| CURE | 자연(POSTURE 불균형) | 7 스트림 | prequential | – | prequential acc | 평균 순위, 시간 |
| AdapTable | 자연 shift + corruption | TableShift 6 | 도메인 split | 3 | macro-F1, BA | – |
| ClinImb | IR 인위 증가 곡선 | MIMIC-IV-ED·eICU 7과제 | – | – | macro-F1 열화 | 학습시간 |
| Song | 자연 희소 고장 | Scania 5k/10k, SECOM | 중첩 CV, test 봉인 | 5 | 사전 선언 비용 | AUPRC, MCC, Brier |
| Sanghi(미확인) | 자연 | OpenML 다중 20 | 4-way fold | per-seed | worst-class coverage | 보정 예산 |

우리(s44)와의 차이: 우리는 시간순 split·IR>1000·다중 클래스·unseen family를 동시에 갖는데, 위 논문 중 셋 이상을 동시에 만족하는 프로토콜은 없다(시간순 split은 CreditTFM-LC와 Stream-TabPFN만). 이 점 자체가 논문의 프로토콜 기여 주장으로 쓸 수 있다.

---

## 4. 차용 아이디어 우선순위

각 항목: 공격 대상 수치 / SRC_HISTORY 어느 실패 유형을 어떻게 피하는가 / knob 수.

1. **threshold-to-prior arm + threshold-matched XGBoost** [D] — known macro-F1 중앙값 −0.184. τ를 학습 prior에서 정하면(검증 적합 아님) #2 회피. 0 knob(재실행 불필요: 저장 확률로 오프라인). 근거: PFN-Imb·Cai·L2C-TFM. 다중 클래스는 p̂/π 정규화(DistPFN의 α와 동형).
2. **C0-natural + DistPFN** [A] — exp37 재현이 중립이었던 이유가 C0가 이미 균형이라서인지 확인. 1 knob(컨텍스트 비율). 근거: DistPFN 심한 불균형 +0.095.
3. **Hybrid 컨텍스트: ρ·Balanced + (1−ρ)·Diversity-KM, family당 최소 n행 floor** [C] — known macro-F1. 1 knob(ρ 또는 n). 근거: CreditTFM Hybrid/Oversample+, majority_downsample, IM-Context 역밀도 풀. "행을 빼는" 계열이라 Cai 결과와 정합.
4. **경계(불확실) 표본 우선 컨텍스트** [C] — Fair-TabICL·CURE. 불확실성은 **과거 구간의 얼린 TabPFN 자체 엔트로피**(CURE 방식, 라벨 무관)로 계산하면 #2·#4 회피(TTA 안정성과 다름: 소유 expert 식별이 아니라 컨텍스트 채우기). CURE에 클래스 floor를 붙이면 1 knob.
5. **저장 확률 배열을 산출물로 + 결정 규칙 스윕 오프라인** [프로토콜] — s43/s44 "체크포인트 없음" 규약과 정합; threshold·Platt·conformal을 재실행 없이 비교. 근거: Cai.
6. **결과 튜플에 ECE/Brier와 AUPRC 추가** [프로토콜] — Cai(remedy가 ECE 6–7배 악화), AWARE(AUROC +24% vs AUPRC +140%), Song(Brier).
7. **고정 예산 IR 스윕 곡선** [프로토콜] — AWARE·ClinImb. "XGB에 진다"를 "IR=X 이후 진다"로.
8. **4-way fold(context / bias-estimation / calibration / test)** [프로토콜] — Sanghi. 시간순에서는 bias-estimation을 context 이후·test 이전 구간에 둔다. #2의 구조적 차단.
9. **unseen resolution을 macro-coverage conformal + abstention 비용으로 재정의** [U] — unseen resolved 0.125. weighted conformal 필수(시간순). 근거: MacroCov-CP·Conformal-LT·Quiet Failure.
10. **청크별 prior 온라인 재추정 + EMA** [A] — AdapTable/CalibOnlineLS. NIDS 스트림의 family 군집 도착에 유리하나 IR>1000에서 배치당 꼬리 0개 문제 → 청크 크기를 family 등장 주기에 맞춰야 함. 2 knob.
11. **버릴 것**: SMOTE/TabPFGen-in-context(4개 출처 음성), TFM 파인튜닝(FT-Explore: 불균형에서 SFT 악화), 앙상블 확장 전 Q-statistic 진단(diversity ceiling).
12. **관련연구 프레이밍**: "TabPFN prior는 IR>1000 영역을 본 적이 없다"(PUICL·TACTIC·Categorical Prior Lock-in), "IR>1000 방법 문헌은 IR≤580(Cai)·≤500(AWARE)·≤129(GILA)에서 멈춘다".

---

## 5. 본문 미확인·제외

- 본문 미확인(OpenReview 차단): **TAAR**(ICLR 2026 reject; attention 점수로 컨텍스트 검색 + CRLR로 다중 클래스 확장; 심사평 "불균형·잡음에서 CRLR 가정 실패 가능", TabPFN-v2에선 kNN 검색이 오히려 해로운 비대칭), **ProFiT**(ICLR 2026 reject; MotherNet에 proxy-task 무라벨 파인튜닝, AUPRC/F1; 얼린 제약 위반), **Realistic Evaluation of TabPFN v2.5 in Open Environments**(FMSD@ICML 2026; "class-imbalanced·prior-driven 과제에 적합, concept shift에 약함"), **Sanghi**(위), **Dipu, group-conditional conformal audit of TabPFN**(Zenodo 20662098, venue 미확인).
- 인접 이론(TabPFN 없음, 시트 미반영): Do we need rebalancing strategies?(AISTATS 2026), Cortes et al. Balancing the Scales(ICML 2025), Latent Score-Based Reweighting(ICML 2025), Structured Matrix Scaling·Utility-Aware Multiclass Calibration(AISTATS 2026), Heterogeneous Label Shift(ICML 2025), Prior shift estimation for PU(AISTATS 2026), Anatomy and Boundary of Adaptation under Temporal Tabular Shift(arXiv 2609.12136 — 시간순 얼린 TFM 적응의 식별불가능성), Categorical Prior Lock-in(arXiv 2606.11961), SPN prior alignment(ICML 2026), GILA(ICLR 2026 reject; 44 불균형 표 데이터셋 IR 11.6–129, TFM 베이스라인 없음).
- 제외: 도메인 응용 ~40건(SMOTE+TabPFN 사례연구), MultiTabPFN(클래스 수 문제), BoostPFN(초록에 불균형 언급 없음 확인), TFM4GAD(그래프 이상탐지), MLF-ICL(URL, F1 99.78% 누수 의심), CausalMixFT Zenodo 자동생성 기록 14건(저자 불명, 비실재 문헌으로 판단).

---

## 6. 시트 반영 (`docs/tabpfn_papers_survey.xlsx`, 2026-09-22 2차)

- Ch.2 추가 7편: IM-Context, Fair-TabICL, Stream-TabPFN, AttnDistill-TabPFN, CURE, PUICL, TFM-ImbRemedy. FT-Explore는 WWW 2026 게재로 출처·연도 갱신 후 학회/저널 그룹으로 이동.
- Ch.3 추가 9편: TabICL-CostFault(Song), CalibOnlineLS, Conformal-LT, AdapTable, TACTIC, L2C-TFM, MacroCov-CP, Plugin-BB, RCB.
- 미반영: 본문 미확인 5건(§5), 인접 이론(TabPFN 무관), Zenodo 보조자료만 있는 기록.

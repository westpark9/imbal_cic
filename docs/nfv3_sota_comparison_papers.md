# NF-v3(NetFlow-v3) 계열 데이터셋 비교 SOTA 논문 리스트

이 프로젝트의 `tabpfn/nfv3_cic2018_multiclass_test.py`, `src/s32~s44_nfv3_*.py`가 쓰는
`data/nfv3_energy_suite_uncapped_scenarios.pkl`은 University of Queensland(UQ)의
**NetFlow-based NIDS 데이터셋 계열**(`NF-UNSW-NB15`, `NF-ToN-IoT`, `NF-BoT-IoT`,
`NF-CSE-CIC-IDS2018`)의 **v3** 릴리즈에서 만들어진다. 아래는 TabPFN-v3 외에, 같은 계열
데이터셋(v2/v3) 위에서 비교 실험을 한 SOTA 논문들을 정리한 것. **주의: NF-v3는 2025년
3월에야 공개된 매우 최신 릴리즈라, 아직 v3 자체로 분류/불균형 실험을 한 논문은 거의 없고
대부분의 SOTA 비교 논문은 이 계열의 v2(43-feature 표준안)에서 수행됨.** v3와 v2는 같은
연구 그룹(UQ, Sarhan/Layeghy/Portmann 등)이 만든 동일 계열 데이터셋이라 스키마/클래스
구성이 이어지지만, feature 수(43 → 53/57)와 온전한 시간 정보 유무가 다르므로 "동일
벤치마크"로 인용하지 말 것.

## 1. NF-v3 원 논문 (이 프로젝트가 쓰는 데이터셋의 출처)

- **제목**: Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems
- **arXiv**: [2503.04404](https://arxiv.org/abs/2503.04404)
- **연도/venue**: 2025년 3월 arXiv 게재, IEEE Access 게재 예정(2026)
- **저자**: Majed Luay, Siamak Layeghy, Seyedehfaezeh Hosseininoorbin, Mohanad Sarhan,
  Nour Moustafa, Marius Portmann (UQ)
- **요약**: 기존 NetFlow 계열 데이터셋(v2)에 inter-packet arrival time, flow
  duration 같은 시간적(temporal) feature가 빠져 있다는 문제를 지적하고, 이를 보강한
  `NF3-UNSW-NB15`, `NF3-CSE-CIC-IDS2018`, `NF3-ToN-IoT`, `NF3-BoT-IoT` 4종을 새로
  공개. feature 분포의 시계열 변화와, 신호처리에서 가져온 time-frequency 표현(TFSP)으로
  공격 유형별 패턴이 실제로 구분되는지 분석한 **데이터셋+분석 논문**이며, 분류기
  벤치마크(SOTA 비교 실험)는 포함하지 않음.
- **사용 데이터**: NF3-UNSW-NB15 / NF3-CSE-CIC-IDS2018 / NF3-ToN-IoT / NF3-BoT-IoT
  (43-feature v2 표준안 + temporal feature 추가, 총 53~57열 — 논문 본문 수치와 이
  프로젝트 전처리 스크립트(`scripts/preprocess_nfv3_energy_suite.py`)가 말하는
  "53-column schema"가 정확히 일치하는지는 미확인, 재인용 시 직접 컬럼 수를 세어 확인할 것)
- **실험한 SOTA**: 없음(분류기 비교 실험 자체가 없는 데이터셋/분석 논문). 이 프로젝트
  입장에서는 "우리가 쓰는 원본 데이터가 어떻게 만들어졌는지"의 근거 문헌으로만 인용.

## 2. NF-v2 표준 feature set 원 논문 (v3의 직접 선행 연구)

- **제목**: Towards a Standard Feature Set for Network Intrusion Detection System Datasets
- **arXiv**: [2101.11315](https://arxiv.org/abs/2101.11315)
- **연도/venue**: 2021년 arXiv, *Mobile Networks and Applications* 2022 게재
- **저자**: Mohanad Sarhan, Siamak Layeghy, Marius Portmann (UQ)
- **요약**: 개별 NIDS 데이터셋마다 feature 정의가 제각각이라 서로 다른 논문 결과를
  공정 비교할 수 없다는 문제의식에서, NetFlow v9 기반 12-feature(basic)/43-feature
  (extended) 표준 셋을 제안하고 `NF-UNSW-NB15-v2`, `NF-BoT-IoT-v2`, `NF-ToN-IoT-v2`,
  `NF-CSE-CIC-IDS2018-v2` 및 이를 합친 `NF-UQ-NIDS-v2`를 공개. 43-feature 셋이 기존
  proprietary feature set보다 분류 성능이 더 좋음을 보임. NF-v3는 이 표준안에 temporal
  feature를 얹은 버전이라, 이 논문이 이 프로젝트가 쓰는 스키마의 직접적 뿌리.
- **사용 데이터**: NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2,
  NF-UQ-NIDS-v2 (12-feature / 43-feature 두 버전)
- **실험한 SOTA**: 별도 SOTA 모델 비교보다는 **Extra Trees classifier**를 주 벤치마크
  모델로 두고, binary/multi-class, accuracy/AUC/F1/Detection Rate/False Alarm Rate
  지표로 12-feature vs 43-feature 두 feature set을 비교하는 ablation 성격.

## 3. E-GraphSAGE (같은 데이터 계열에서 가장 많이 인용되는 GNN 베이스라인)

- **제목**: E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT
- **arXiv**: [2103.16329](https://arxiv.org/abs/2103.16329)
- **연도/venue**: 2021년 arXiv, NOMS 2022 (IEEE/IFIP Network Operations and Management
  Symposium) 게재
- **저자**: Wai Weng Lo, Siamak Layeghy, Mohanad Sarhan, Marcus Gallagher, Marius Portmann
  (UQ)
- **요약**: edge feature(=flow feature)와 네트워크 토폴로지 구조를 동시에 활용하는
  GraphSAGE 변형으로, flow 기반 NIDS에 GNN을 적용한 초기·대표 논문. 이후 NF-v2/v3 계열
  데이터셋을 쓰는 후속 GNN NIDS 논문 대부분이 이 논문을 baseline으로 인용.
- **사용 데이터**: BoT-IoT/NF-BoT-IoT, ToN-IoT/NF-ToN-IoT (원본 feature 버전 vs
  NetFlow 12-feature 버전을 나란히 비교; NF-UNSW-NB15은 관련 연구로만 언급, 본 실험엔
  미포함)
- **실험한 SOTA**: XGBoost, Extra Tree Classifier, K-Nearest Neighbors(KNN) — 그 외
  decision tree/naive Bayes/random forest 앙상블, Deep Autoencoder+LSTM 계열도 비교.

## 4. Anomal-E (self-supervised GNN, E-GraphSAGE 계열 확장)

- **제목**: Anomal-E: A Self-Supervised Network Intrusion Detection System based on Graph
  Neural Networks
- **arXiv**: [2207.06819](https://arxiv.org/abs/2207.06819)
- **연도/venue**: 2022년 arXiv, *Knowledge-Based Systems* (Vol. 258) 2022 게재
- **저자**: Evan Caville, Wai Weng Lo, Siamak Layeghy, Marius Portmann (UQ)
- **요약**: 라벨 없이 edge feature + 그래프 구조를 self-supervised로 학습(Deep Graph
  Infomax 기반)한 뒤 anomaly detection에 활용하는 방법. 라벨 의존도를 낮춰 unseen
  attack에 대한 적응력을 높이려는 동기가 이 프로젝트의 open-set/OOD 문제의식과 맞닿음.
- **사용 데이터**: NF-UNSW-NB15-v2 (약 239만 flow, attack 3.96%), NF-CSE-CIC-IDS2018-v2
  (약 1,889만 flow, attack 11.95%)
- **실험한 SOTA**: GraphSAGE(지도학습판), DGI(Deep Graph Infomax), Isolation Forest,
  PCA, raw-feature 베이스라인.

## 5. Energy-based Flow Classifier — Open-Set NIDS (이 프로젝트의 energy-OOD 설계와 가장
   개념적으로 가까운 논문)

- **제목**: A Novel Open Set Energy-based Flow Classifier for Network Intrusion Detection
- **arXiv**: [2109.11224](https://arxiv.org/abs/2109.11224)
- **연도/venue**: 2021년 arXiv, *Computers & Security* 2025 게재
- **저자**: Manuela M. C. Souza, Camila Pontes, Joao Gondim, Luis P. F. Garcia,
  Luiz DaSilva, Eduardo F. M. Cavalcante, Marcelo A. Marotta
- **요약**: 원래 단일 클래스(정상 트래픽)만 학습하는 Energy-based Flow Classifier(EFC)를
  다중 클래스 + open-set 인식이 가능하도록 확장. 학습 시 보지 못한 새 공격 유형을
  단일 레이어 구조로 저비용에 탐지하면서, closed-set 다중분류 SOTA와도 비등한 성능을
  낸다고 주장 — 이 프로젝트가 쓰는 `E = -T*logsumexp(logits/T)` energy 컨벤션 및
  "known-class 분류 성능과 OOD 탐지 성능을 분리해서 본다"는 평가 철학과 정확히 같은
  선행연구 라인.
- **사용 데이터**: CICIDS2017 (CICFlowMeter 기반 flow, 80-feature; NF-v3와 스키마는
  다르지만 이 프로젝트의 `s25~s30_code_ood_*` 트랙이 쓰는 `cic2017_chrono_v2.pkl`와 같은
  원천 데이터셋 계열)
- **실험한 SOTA**: closed-set: Decision Tree, SVM, MLP. open-set 비교: Baseline
  (softmax 기반), **ODIN**, **OCN**(Open set Classification Network) — 이 셋이
  open-set/OOD 인식 SOTA로 직접 비교됨.

## 6. GEAFL-IDS — 같은 NF 데이터 계열에서 클래스 불균형을 정면으로 다루는 최신 논문

- **제목**: Intrusion Detection Method based on Graph Edge Attention and Focal Loss
- **arXiv**: 없음 (arXiv preprint 미확인, ACM 학회 논문집으로만 출판)
- **연도/venue**: 2025년, *Proceedings of the 2025 4th International Conference on
  Cryptography, Network Security and Communication Technology (ICCNSCT 2025)*,
  DOI: [10.1145/3723890.3723895](https://dl.acm.org/doi/10.1145/3723890.3723895)
- **요약**: flow 데이터를 그래프로 구성한 뒤 edge attention으로 edge(=flow) feature를
  강조하고, focal loss로 소수 클래스(rare attack)에 대한 학습 가중치를 높이는 방식.
  이 프로젝트의 "tail class F1을 올리는 구조"라는 목표와 방향이 같지만, 접근은
  routing/expert가 아니라 loss reweighting + graph attention 쪽. `docs/history.pdf`가
  정리한 이 프로젝트의 실패 모드 4번("XGBoost의 boosting이 이미 암묵적 MoE")과 대비해
  볼 가치가 있음 — focal loss도 "hard/rare 샘플 재가중"이라는 점에서 같은 계열 아이디어.
- **사용 데이터**: NF-BoT-IoT, NF-UNSW-NB15 (버전 명시 없음 — 발표 시점상 v2로 추정,
  직접 확인 필요)
- **실험한 SOTA**: GaussianNB, KNN, Decision Tree, AdaBoost, **Random Forest**, **CNN**,
  **E-GraphSAGE** 등 7개 비교 — 이 중 성격이 다른 3개를 뽑으면 Random Forest(classical
  ML) / CNN(plain DL) / E-GraphSAGE(GNN, 3번 항목과 동일 논문) 정도가 대표적.

## 요약 비교

| # | 논문 | 연도/venue | NF 계열 데이터 | 이 프로젝트와의 접점 |
|---|---|---|---|---|
| 1 | Temporal Analysis of NetFlow Datasets | 2025 / IEEE Access(예정) | NF3-* 4종 (원 논문) | 우리가 쓰는 원본 데이터의 출처 |
| 2 | Standard Feature Set for NIDS Datasets | 2021 / Mobile Netw Appl 2022 | NF-*-v2 4종 + NF-UQ-NIDS-v2 | NF-v3 스키마의 직접 선행 연구 |
| 3 | E-GraphSAGE | 2021 / NOMS 2022 | NF-BoT-IoT, NF-ToN-IoT | 이 계열 데이터에서 가장 흔한 GNN 베이스라인 |
| 4 | Anomal-E | 2022 / Knowledge-Based Systems | NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2 | 라벨 최소화·unseen attack 적응 — open-set 문제의식 공유 |
| 5 | Open Set Energy-based Flow Classifier | 2021 / Computers & Security 2025 | CICIDS2017 (NF-v3와 다른 스키마, 같은 CIC 계열) | energy 기반 open-set 설계가 이 프로젝트와 가장 유사 |
| 6 | GEAFL-IDS | 2025 / ACM ICCNSCT 2025 | NF-BoT-IoT, NF-UNSW-NB15 (버전 미확인) | class imbalance를 loss/attention으로 정면 대응 |

## 재확인이 필요한 부분 (M1: 문서보다 원문 확인 우선)

- 1번 논문(NF-v3 원 논문)의 정확한 feature/column 수(53 vs 57)는 arXiv HTML 요약에서
  나온 값이라 원문 표로 직접 대조 안 함 — 이 프로젝트 `data/nfv3_energy_suite_*.pkl`의
  실제 column 수와 비교해 검증 필요.
- 6번(GEAFL-IDS)은 arXiv 사본이 없어 ACM 페이지 검색 결과로만 요약함 — 데이터셋
  버전(v2/v3 여부)과 baseline 표 원문은 ACM Digital Library 접속 후 재확인 권장.
- 5번은 NF-v3와 feature 스키마 자체는 다르므로(CICFlowMeter 80-feature vs NetFlow
  53/57-feature), "같은 데이터셋" 비교가 아니라 "같은 open-set/energy 방법론" 비교로만
  인용할 것.

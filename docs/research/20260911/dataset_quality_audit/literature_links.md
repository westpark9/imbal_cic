# 문헌 링크 모음 — 모순 · 중복 · 누수 (2026-09-14 조사)

용어는 보고서와 통일한다.
- **중복**: 같은 (feature 벡터, 라벨) 쌍이 2회 이상 나타남
- **모순**: 한 feature 벡터가 서로 다른 라벨을 가짐
- **누수**: test의 벡터가 train에도 존재함 (분할에 따라 달라지는 성질)

각 항목의 "확인" 열은 실제로 페이지나 PDF를 열어 본문을 확인했는지를 뜻한다.

---

## A. 실제로 행을 지운 연구

| 논문 | 연도 / venue | 링크 | 무엇을 지웠나 | 확인 |
|---|---|---|---|---|
| Tavallaee, Bagheri, Lu, Ghorbani — *A Detailed Analysis of the KDD CUP 99 Data Set* | 2009, IEEE CISDA | https://www.ee.torontomu.ca/~bagheri/papers/cisda.pdf | **중복**. KDD'99의 반복 레코드를 1개만 남김 → NSL-KDD. train 4,898,431 → 1,074,992 (78.05%↓), test 311,027 → 77,289 (75.15%↓) | 본문 |
| Raskovalov, Gabdullin, Dolmatov — *Investigation and rectification of NIDS datasets and standardized feature set derivation...* | 2022, arXiv only | https://arxiv.org/abs/2212.13994 | **모순**. "in NF-ToN-IoT a number of flows are duplicated and labeled as corresponding to different attacks". 21,978,630 → 18,902,360 (14%↓). **mitm 1,052 → 0** | 본문 + Table 1 |
| Chen, Tran, Thumati, Bhuyan, Ding — *Data Curation and Quality Assurance for ML-based Cyber Intrusion Detection* | 2021, arXiv (ACM JDIQ 심사 중 표기) | https://arxiv.org/abs/2105.10041 | **모순**. 두 클래스에 동시에 존재하는 시퀀스 제거. 제거 전후 FPR 최대 27.5배 차이. HIDS라 직접 비교는 아님. "중복 자체는 결함이 아니고 train/test 겹침이 결함"이라는 논지도 이 논문 | 본문 |
| Mondragon, Branco, Jourdan, Gutierrez-Rodriguez, Biswal — *Advanced IDS: a comparative study of datasets and ML algorithms...* | 2025, Applied Intelligence | https://link.springer.com/article/10.1007/s10489-025-06422-4 | **중복**. NetFlow 판 행 수만 보고: BoT-IoT 73,370,443 → 37,643,287 (49%↓), ToN-IoT 22,339,021 → 16,940,365 (24%↓). 클래스별 분해 없음 | 요약 |

## B. 지우지 않고 측정하거나 재라벨한 연구

| 논문 | 연도 / venue | 링크 | 무엇을 했나 | 확인 |
|---|---|---|---|---|
| Jerabek, Luxemburk, Plny, Koumar, Pesek, Hynek — *When Simple Model Just Works: Is Network Traffic Classification in Crisis?* | 2025, arXiv | https://arxiv.org/abs/2506.08655 | **우리 "현실적 oracle"과 같은 개념의 선행연구.** "most datasets contain over 50% redundant samples (identical packet sequences), which frequently appear in both training and test sets"; 중복이 "reduce the theoretical maximum accuracy when identical flows have conflicting labels". 12개 데이터셋(UNSW-NB15, CIC-IDS-2017 포함) | 초록 |
| Xie, Li, Zhang, Sun, Xu — *Analysis and Detection against Network Attacks in the Overlapping Phenomenon of Behavior Attribute* | Computers & Security 123 (2022) | https://arxiv.org/abs/2310.10660 | **모순을 삭제 대신 multi-label로 재정의.** "multiple samples with the same features but different labels". UNSW-NB15에서 라벨 조합 57종, 샘플당 평균 라벨 1.689개 | 초록 |
| Flood, Engelen, Aspinall, Desmet — *Bad Design Smells in Benchmark NIDS Datasets* | 2024, IEEE EuroS&P (distinguished paper) | https://distrinet.cs.kuleuven.be/news/bad_smells_euros_p_template-4.pdf | 삭제 없이 **지표화**. wrong-label(ENN 기반) CIC-17 infiltration 0.81, CIC-18 infiltration 0.65, CTU-13 Murlo 1.00. traffic collapse(유사도 0.95 초과 쌍 비율) ToN-IoT backdoor 1.00, CIC-18 bot 0.99. 권고는 삭제가 아니라 "avoiding using training and test attack data from the same class and dataset" | PDF 전문 |
| Al-Daweri, Zainol Ariffin, Abdullah, Md. Senan — *An Analysis of the KDD99 and UNSW-NB15 Datasets for the IDS* | 2020, Symmetry 12(10):1666 | https://www.mdpi.com/2073-8994/12/10/1666 | **중복** 감사만. UNSW-NB15 공식 train의 42.24%가 중복, test는 0.00%. Generic 89.54%, DoS 68.95% | 요약 |
| Engelen, Rimmer, Joosen — *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study* | 2021, IEEE SPW (WTMC) | https://intrusion-detection.distrinet-research.be/WTMC2021/index.html | **모순의 원인** 규명. IP + 시간 창만으로 라벨링해 거절된 연결에도 공격 라벨. CICFlowMeter FIN 버그로 생긴 2-packet 부산물 flow가 전체의 25.9% | 요약 |
| Liu, Engelen, Lynar, Essam, Joosen — *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018* | 2022, IEEE CNS, pp. 254–262 | https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html | 라벨링 로직을 역설계·교정해 공개. **주의**: 흔히 인용되는 오류율 6.67% / 7.53%는 2차 인용이고 원문 미확인 | 부분 |
| Lanvin, Gimenez, Han, Majorczyk, Mé, Totel — *Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes* | CRiSIS 2022, LNCS 13857 | https://pfgimenez.fr/publications/2022-CRiSIS/ | packet misorder, **packet duplication**, 미라벨 공격을 교정하고 성능 차이 측정. **주의**: 순환 인용되는 "패킷 5% 중복", "recall 80→100%"는 미확인 | 초록 |
| Pekár, Jozsa — *Evaluating ML-Based Anomaly Detection Across Datasets of Varied Integrity: A Case Study* | 2024, Computer Networks | https://arxiv.org/abs/2401.16843 | **우리가 넘어야 할 반증.** CICIDS2017 PCAP에서 중복 패킷(3.6–6.1%)을 지우고 flow 재생성했으나 Random Forest 성능은 "exceptional robustness" | 요약 |
| Cantone, Marrocco, Bria — *On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection* | 2024, IEEE Access 12 | https://arxiv.org/abs/2402.10974 | 높은 내부 성능을 "sample redundancies"로 귀인. **DoS Slowhttptest는 105,550 샘플에 고유 인스턴스가 56개** | 요약 |

## C. "표준 관행이 아니다"의 근거

| 논문 | 연도 / venue | 링크 | 핵심 문장 |
|---|---|---|---|
| Flood 외 (위와 동일) | 2024, EuroS&P | (위) | 38편 검토. "No papers in our overview commented on this or amended their evaluation process, and all were seemingly unaware of these discrepancies." |
| Apruzzese, Laskov, Schneider — *SoK: Pragmatic Assessment of Machine Learning for Network Intrusion Detection* | 2023, IEEE EuroS&P | https://arxiv.org/abs/2305.00550 | 상위 venue 30편 중 "none of the 30 papers considered different preprocessing mechanisms" |
| Leevy, Khoshgoftaar — *A survey and analysis of intrusion detection models based on CSE-CIC-IDS2018 Big Data* | 2020, Journal of Big Data 7:104 | https://doi.org/10.1186/s40537-020-00382-x | "information on the data cleaning of CSE-CIC-IDS2018 was inadequate across the board" |
| Goldschmidt, Chudá — *Network Intrusion Datasets: A Survey, Limitations, and Recommendations* | 2025, Computers & Security 156 | https://arxiv.org/abs/2502.06688 | 89개 데이터셋 검토. "duplicated features or entries … These issues are frequently unreported". 정작 자신들의 10단계 제작 권고에는 dedup 단계가 없음 |
| Ring, Wunderlich, Scheuring, Landes, Hotho — *A Survey of Network-based Intrusion Detection Data Sets* | 2019, Computers & Security 86 | https://arxiv.org/abs/1903.02460 | 중복 제거를 KDD 전용 처방으로만 언급. 15개 데이터셋 속성 표에 중복 항목 없음 |
| Arp 외 — *Dos and Don'ts of Machine Learning in Computer Security* | 2022, USENIX Security | https://arxiv.org/abs/2010.09470 | **우리에게 걸리는 경고.** "selective snooping describes the cleansing of data based on information not available in practice". 행 삭제를 처방하지 않음 |

## D. 우리 데이터(NetFlow v2/v3)에 대한 선행연구

- Sarhan, Layeghy, Portmann — *Towards a Standard Feature Set for NIDS Datasets*, https://arxiv.org/abs/2101.11315 : 본문에 duplicate / redundant / identical / conflict 어느 단어도 없음.
- Luay, Layeghy, Hosseininoorbin, Sarhan, Moustafa, Portmann — *Temporal Analysis of NetFlow Datasets for NIDS*, https://arxiv.org/abs/2503.04404 : 시간 feature 추가. 중복·모순·누수 분석 없음.
- **결론: NF-v2에 대한 중복/모순 감사는 Raskovalov 외의 한 문장이 전부이고, NF-v3에 대해서는 전무하다.**

## E. 인용하지 말 것 (확인 결과 무관)

- Sommer & Paxson, *Outside the Closed World* (IEEE S&P 2010) — 중복 관련 서술 없음.
- Rosay 외 (ICISSP 2022) — 여기서 말하는 duplication은 중복 **열**(feature)이지 행이 아님.
- Zoghi & Serpen (arXiv 2101.05067) — PCA/t-SNE 투영 상의 기하학적 겹침이지 동일 벡터가 아님.
- Peterson 외 *A Review and Analysis of the Bot-IoT Dataset* (IEEE SOSE 2021) 및 *Composition analysis of the Bot-IoT dataset* (IJITCA 2022) — 유료라 미확인. 인용 금지.

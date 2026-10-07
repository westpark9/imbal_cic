## 1
Classes sharing identical feature vectors

Data review
1
Row: original label     Column: other label on the same feature vector     Cell: share of all rows in the row class (%)
Blank: 0%   ·: below 1%   Grey: same class   A row can count in several cells, so cells do not add up

0

25

50

75

100
%
Web attack: 1,764 + 432 − 429 overlap = 1,767 rows removed · 771 remain

## 2
Class distribution after cleaning

Data review
2
All data (train + validation + test) · row counts and within-dataset class share (%)
CIC2018
Class | Before | After | Share before → after
Benign | 17,514,626 | 12,798,328 | 87.07 → 86.74
Bot | 207,703 | 207,703 | 1.03 → 1.41
Brute force | 575,194 | 188,414 | 2.86 → 1.28
DDoS | 1,324,350 | 1,316,025 | 6.58 → 8.92
DoS | 302,966 | 197,415 | 1.51 → 1.34
Infiltration | 188,152 | 46,761 | 0.94 → 0.32
Web attack | 2,538 | 771 | 0.013 → 0.005
Total | 20,115,529 | 14,755,417 | 100 → 100

Infiltration down 75.1%
Web attack down 69.6% · 771 rows after cleaning
ToN-IoT
Class | Before | After | Share before → after
Backdoor | 203,384 | 200,688 | 0.74 → 1.86
Benign | 16,792,214 | 6,446,120 | 61.02 → 59.65
DDoS | 4,141,256 | 925,247 | 15.05 → 8.56
DoS | 203,456 | 113,655 | 0.74 → 1.05
Injection | 381,777 | 341,263 | 1.39 → 3.16
MitM | 6,013 | 5,446 | 0.022 → 0.050
Password | 1,594,777 | 1,061,699 | 5.79 → 9.82
Ransomware | 3,971 | 3,604 | 0.014 → 0.033
Scanning | 1,358,977 | 39,069 | 4.94 → 0.36
XSS | 2,834,435 | 1,670,497 | 10.30 → 15.46
Total | 27,520,260 | 10,807,288 | 100 → 100
Imbalance remains after cleaning · CIC2018 Web attack 771 rows, ToN Scanning 39,069 rows

## 3
Global model and residual expert structure

Model
3
Separate roles of the training data · Global construction / expert construction / S·V training
Global pool · shared context
Expert pool · residuals and experts
Route pool · S/V training
Offline · model construction

Build Global
Base classifier from
the shared context

Compute residuals
Embedding · probabilities
gap to truth · error size

Cluster residuals
Split into K clusters
by failure pattern

Build experts
Shared anchor +
cluster-specific block

Train S/V
Learn from Global/expert
corrections and damage
Online · per-input prediction

Global prediction
Base class prediction

Scorer
Selects an expert,
decides whether to call

Expert inference
Classify with the selected context

Verifier
Decides whether to
accept the expert

Final prediction
Expert prediction if accepted
Call or acceptance condition not met → keep the Global prediction
Backbone frozen · experts built from contexts · S/V decide whether to replace the prediction

## 4
Global per-class performance and the expert bottleneck

Model
4
Cleaned data · seed 43 · diagnosis of the existing residual contexts · before S/V

Existing e3 · Infiltration
Own residual region F1  0.440 → 0.967
Input-nearest group precision 12.4% · FP 7,935 → 21,535

Existing e3 · Injection
Own residual region F1  0.159 → 0.939
Input-nearest group precision 8.7% · FP 581 → 14,222
Own region: assigned by true residual · Input-nearest group: assigned by input embedding and Global probabilities · Arrow: Global → Expert
Improve both: correcting own-region errors and separating similar rows of other classes

## 5
SOTA classification on cleaned data

SOTA comparison
5
Seed 43 · same C0 100,000 rows · CIC2018 test 3.04M rows / ToN test 2.27M rows
Method | CIC2018 Macro-F1 | ToN Macro-F1 | Training pool rows · CIC / ToN
Global TabPFN v3 | 0.7824 | 0.6796 | 100,000
XGBoost | 0.7821 | 0.6983 | 100,000
BoostPFN | 0.7148 | 0.5866 | 100,000
LoCalPFN · FT | 0.7024 | 0.6462 | 100,000
DistPFN · v3 | 0.7839 | 0.6682 | 100,000
XGBoost · full train | 0.7795 | 0.7162 | 8.67M / 4.67M
Full-train XGB: reference with extra training data · Global: base classifier before experts and S/V

CIC2018 · per-class differences · shared 100k
Best Infiltration F1 0.3055 · DistPFN
Best Web attack F1 0.3321 · XGB

ToN · per-class differences · shared 100k
Best MitM F1 0.2363 · XGB
Best Scanning F1 0.3281 · BoostPFN
Best Macro-F1 on the same 100k rows · CIC2018 DistPFN 0.7839 · ToN XGB 0.6983

## 6
SOTA training and inference cost

SOTA comparison
6
RTX 4090 24GB · up to 2 jobs in parallel · elapsed time including resource contention
Method | CIC fit (s) | CIC predict (s) | ToN fit (s) | ToN predict (s) | GPU GiB
CIC / ToN
Global TabPFN v3 | 36.4 | 1,099.3 | 36.7 | 821.7 | 2.92 / 2.94
XGBoost | 2.3 | 1.0 | 55.4 | 1.0 | 0.55 / 0.53
BoostPFN | 98.5 | 2,308.5 | 97.8 | 1,737.6 | 0.89 / 0.89
LoCalPFN · FT | 3,196.3 | 17,300.8 | 5,783.1 | 11,144.8 | 7.54 / 7.54
DistPFN · v3 | 36.4 | 1,099.4 | 36.7 | 821.9 | 2.92 / 2.94
XGBoost · full train | 107.8 | 1.0 | 143.2 | 1.1 | 2.20 / 1.59
LoCalPFN fit includes validation · full-test inference includes kNN search
DistPFN shares the Global fit and inference · prior correction adds about 0.19 s
GPU: per-process peak sampled every 1 s · not a standalone speed comparison
LoCalPFN full-test inference · CIC2018 4.81 h · ToN 3.10 h

## 7
Expert ability: own region vs input-nearest group
Draft · TBD

Expert evaluation
7


View A · own residual region


View B · input-nearest group
Representative expert · class: TBD
Metric | Global | Expert
Precision | TBD | TBD
Recall | TBD | TBD
F1 | TBD | TBD
Corrections: TBD
Damage: TBD
Representative expert · class: TBD
Metric | Global | Expert
Precision | TBD | TBD
Recall | TBD | TBD
F1 | TBD | TBD
Corrections: TBD
Damage: TBD

Takeaway
TBD

## 8
Contribution of scorer and verifier
Draft · TBD

Component ablation
8
Configuration | CIC2018 | ToN-IoT | Calls · accepted | Corrections · damage
Global | TBD | TBD | TBD | TBD
No S·V | TBD | TBD | TBD | TBD
S only | TBD | TBD | TBD | TBD
V only | TBD | TBD | TBD | TBD
S+V | TBD | TBD | TBD | TBD

Context · gate settings
TBD

Takeaway
TBD

## 9
Final performance and cost of the proposed model

Final comparison
9
Final comparison of the validated expert and S/V configuration
Configuration | CIC2018 | ToN | Training info | Call cost
Expert · no S/V | TBD | TBD | TBD | TBD
Expert · S only | TBD | TBD | TBD | TBD
Expert · V only | TBD | TBD | TBD | TBD
Expert · S+V | TBD | TBD | TBD | TBD

Baselines: SOTA performance (slide 5) · measured cost (slide 6)
Includes the extra training rows for experts and routing, and the conditional inference cost

## 10
Confirmed effects and next improvements
Draft · TBD

Conclusion
10


Context composition
TBD


Expert discrimination
TBD


Final model performance
TBD


Next improvements
TBD

Conclusion
TBD

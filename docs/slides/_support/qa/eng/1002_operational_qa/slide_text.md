## 1
TabPFN for operational intrusion detection
01 Motivation
1
Prior approach
IDS benchmark data collected under daily attack scenarios
Train on all attack classes and aim to improve on baseline F1
Deployment
New attack labels and changing conditions require model updates
F1 on a fixed test set alone cannot establish operational utility
Why TabPFN
A pretrained tabular foundation model used directly for inference
Strong prediction on small data; context updates without retraining
Evaluating detector updates by adding labeled attacks to context

## 2
Updating an IDS as new attack labels arrive
02 New attack adaptation
2
XGBoost retraining vs. Global TabPFN context expansion · Same cumulative examples · Seed 43
CIC2018 · 2018
Initial · Feb 14 to 16
Benign
Brute force · DoS
Feb 20 to 21
DDoS
Feb 22
Web attack
Feb 28
Infiltration
Mar 2
Bot
ToN · 2019
Initial · Apr 23 to 24
Benign
Scanning · DoS
Apr 25
Injection
DDoS
Apr 26
Password
Apr 27
XSS
Apr 28
Ransomware
Backdoor
Apr 29
MitM
Final budget: 100,000 rows each · No future classes in training · All test rows of introduced classes
Original class arrival order · New attack performance, existing detection, update and inference cost

## 3
CIC2018 · New attack performance and cost
02 New attack adaptation
3
Same cumulative examples · Global TabPFN v3 · Seed 43 · RTX 4090 · GPU jobs run sequentially
F1 at first introduction (%)
Sorted by test count · Introduction stage in parentheses
New attack | Test count | XGBoost | TabPFN
DDoS (1) | 262,680 | 99.96 | 99.53
Bot (4) | 41,542 | 99.70 | 99.77
Infiltration (3) | 8,742 | 10.68 | 25.43
Web attack (2) | 127 | 17.32 | 31.52
Final stage: 100,000 training examples
Full test: 3,042,473 rows · Time in seconds
Metric | XGBoost | TabPFN
Macro-F1 (%) | 77.59 | 78.64
Benign FPR (%) | 2.10 | 0.14
Fit / context setup | 1.98 | 16.98
Inference | 1.44 | 522.63
TabPFN: fixed weights; preprocessing and cache setup
XGBoost: retrained on cumulative data at each stage
Higher initial F1 for Web attack and Infiltration · Reduce inference cost with large contexts
Evaluate new attack performance and computational cost together

## 4
ToN · New attack performance and cost
02 New attack adaptation
4
Same cumulative examples · Global TabPFN v3 · Seed 43 · RTX 4090 · GPU jobs run sequentially
F1 at first introduction (%)
Sorted by test count · Introduction stage in parentheses
New attack | Test count | XGBoost | TabPFN
XSS (3) | 343,854 | 87.23 | 82.62
Password (2) | 210,951 | 88.86 | 88.99
DDoS (1) | 103,338 | 96.88 | 96.98
Injection (1) | 68,891 | 68.90 | 72.71
Backdoor (4) | 40,030 | 99.87 | 99.91
MitM (5) | 1,161 | 15.26 | 4.66
Ransomware (4) | 709 | 14.52 | 12.47
Final stage: 100,000 training examples
Full test: 2,271,723 rows · Time in seconds
Metric | XGBoost | TabPFN
Macro-F1 (%) | 71.30 | 69.51
Benign FPR (%) | 11.03 | 13.33
Fit / context setup | 3.59 | 17.15
Inference | 1.61 | 391.50
TabPFN MitM · Recall 96.99% · Precision 2.39%
41,330 benign samples among 46,010 false positives
High MitM recall, many benign false positives · Improve context composition and inference cost
Evaluate new attack performance and computational cost together

## 5
Data review: remove rows with conflicting labels
03 Data review
5
Identical features · Rows: original labels · Columns: other labels · Cells: row-class share (%) · Sorted by original class size
Blank: 0% · Dot: <1% · Grey: same class
Overlapping counts; cell percentages cannot be summed
0
25
50
75
100
%
Web attack: 1,764 + 432 − 429 overlapping rows = 1,767 removed · 771 retained

## 6
Class distribution after cleaning and the fixed test set
03 Data review
6
Before/after: full dataset · Test: fixed cleaned holdout · Sorted by class count after cleaning
CIC2018
Class | Before cleaning | After cleaning | Test
Benign | 17,514,626 | 12,798,328 | 2,652,203
DDoS | 1,324,350 | 1,316,025 | 262,680
Bot | 207,703 | 207,703 | 41,542
DoS | 302,966 | 197,415 | 39,483
Brute force | 575,194 | 188,414 | 37,696
Infiltration | 188,152 | 46,761 | 8,742
Web attack | 2,538 | 771 | 127
Total | 20,115,529 | 14,755,417 | 3,042,473
ToN
Class | Before cleaning | After cleaning | Test
Benign | 16,792,214 | 6,446,120 | 1,474,302
XSS | 2,834,435 | 1,670,497 | 343,854
Password | 1,594,777 | 1,061,699 | 210,951
DDoS | 4,141,256 | 925,247 | 103,338
Injection | 381,777 | 341,263 | 68,891
Backdoor | 203,384 | 200,688 | 40,030
DoS | 203,456 | 113,655 | 22,458
Scanning | 1,358,977 | 39,069 | 6,029
MitM | 6,013 | 5,446 | 1,161
Ransomware | 3,971 | 3,604 | 709
Total | 27,520,260 | 10,807,288 | 2,271,723
Class counts before and after cleaning, with fixed test support

## 7
Global and expert models built from contexts
04 Current model
7
Shared and specialized contexts · Frozen TabPFN backbone · Scorer and verifier trained separately
Global pool · shared context
Expert pool · residuals and experts
Route pool · S/V training
Offline · model construction
Build Global
Base classifier from
shared context
Compute residuals
Features · probabilities
Label residual · error size
Cluster residuals
K clusters grouped
by failure pattern
Build experts
Shared anchor
+ cluster-specific block
Train S/V
Learn from corrections
and errors by Global/experts
Online · per-input prediction
Global prediction
Base class prediction
Scorer
Select an expert;
decide whether to call
Expert inference
Classify with the
selected context
Verifier
Accept or reject
the expert prediction
Final prediction
Expert prediction
if accepted
Call or acceptance condition not met → Keep the Global prediction
Shared and specialized contexts · S/V select prediction corrections

## 8
Current static performance · SOTA comparison
04 Current model
8
Static setting with all attack classes in training · Same cleaned test set · Seed 43 · Macro-F1 (%)
Method | CIC2018 | ToN | Training data
Global TabPFN v3 | 78.24 | 67.96 | Shared 100,000 rows
XGBoost | 78.21 | 69.83 | Shared 100,000 rows
BoostPFN | 71.48 | 58.66 | Shared 100,000 rows
LoCalPFN · FT | 70.24 | 64.62 | Shared 100,000 rows
DistPFN · v3 | 78.39 | 66.82 | Shared 100,000 rows
XGBoost · full train | 77.95 | 71.62 | Full training set
Proposed model · K = 4 · S/V | 78.24 | 76.17 | Global + Expert and Route pools
Proposed model: existing anchor + residual blocks · Additional labels from Expert and Route pools
Current static performance · Separate from the Global comparison under sequential attack introduction
Current static performance · New attack updates evaluated separately

## 9
SOTA cost on cleaned data
04 Current model
9
RTX 4090 24GB · Up to 2 concurrent jobs · Elapsed time includes resource contention
Method | CIC fit/setup
s | CIC inference
s | ToN fit/setup
s | ToN inference
s | CIC GPU
GiB | ToN GPU
GiB
Global TabPFN v3 | 36.4 | 1,099.3 | 36.7 | 821.7 | 2.92 | 2.94
XGBoost | 2.3 | 1.0 | 55.4 | 1.0 | 0.55 | 0.53
BoostPFN | 98.5 | 2,308.5 | 97.8 | 1,737.6 | 0.89 | 0.89
LoCalPFN · FT | 3,196.3 | 17,300.8 | 5,783.1 | 11,144.8 | 7.54 | 7.54
DistPFN · v3 | 36.4 | 1,099.4 | 36.7 | 821.9 | 2.92 | 2.94
XGBoost · full train | 107.8 | 1.0 | 143.2 | 1.1 | 2.20 | 1.59
LoCalPFN: validation in fit, retrieval in inference · DistPFN: Global computation plus prior correction
Current K = 4 model · Policy call rate: CIC2018 0.00%, ToN 100.00% · Conditional runtime to be measured

## 10
Next steps · An IDS that incorporates new attacks
05 Next steps
10
Gains on some new CIC2018 attacks · Reduce ToN false alarms and computational cost on both datasets
New attack adaptation
Same newly labeled examples
Per-attack performance after updates
Changes in rare-class detection
Preserve existing detection
Same existing test samples
Corrections, errors and benign FPR
Effect of context composition
Operational cost
Update cost when examples arrive
Inference cost until the next update
Throughput and memory
Research criterion: how well new attacks are incorporated,
and at what cost existing detection performance is preserved
Vary benign examples and context size → Compare false alarms and cost → Test expert/S/V updates
F1 is one measure of system utility · Unknown-attack recognition is a later OOD extension

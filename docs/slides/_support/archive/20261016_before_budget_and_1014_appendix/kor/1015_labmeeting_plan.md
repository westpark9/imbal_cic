# 1015 랩미팅 · 7장 축약본

6장은 XGB 누적 재학습과 FT 없는 TabPFN context 갱신의 최종 성능·갱신 시간 비교. 7장은 동일 소규모 라벨 예산의 제안 모델·SOTA 최종 성능만 표시. 초기 FT 결과는 실험 기록에 보존. 1014는 본문 미추가 상태이며 기존 자료를 archive에 보존.

## 1. (타이틀 없음)

No title. Rephrase the user intent into concise presentation language instead of copying their draft. Adaptation to changing networks motivates the research; current class-arrival tests do not establish network-drift robustness.

## 2. TabPFN · 표 데이터 예측 규칙을 사전학습한 모델 (v3: arXiv, 2026)

Source: https://arxiv.org/pdf/2605.13986 §2.5, PDF page 10: more than 8 trillion tokens, not datasets. Official TabPFN 8.2.0: https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_base.py. EXP77 actually fine-tunes the initial IDS data using AdamW, learning rate 1e-5, 10 fixed epochs, context/query cross-entropy; then freezes weights. The v3 decoder weights context one-hot labels, so old-class examples remain necessary in the inference context. Validation is inside the 1000 revealed-label budget; no test tuning.

## 3. v3 구조

Original Figure 5, https://arxiv.org/pdf/2605.13986#page=6. Native PPTX crop excludes the original caption. Labels and source remain in speaker notes. First three stages: feature grouping/cell embedding; target-aware column distribution embedding; row-level feature aggregation. Plain language deliberately omits implementation dimensions.

## 4. v3 구조

Same Figure 5, https://arxiv.org/pdf/2605.13986#page=6; no visible caption by user request. ICL attends to training/context keys; many-class decoder uses attention-weighted one-hot context labels. This is learned contextual relevance, not merely raw-feature nearest-neighbour distance. Context determines available class labels.

## 5. 평가시나리오

EXP77 uses the same released IDs as EXP76. Initial IDS weight fine-tuning: fixed 10 epochs, AdamW, lr 1e-5; initial internal feature-group validation only. Subsequent context policies freeze the initial checkpoint; repeated FT is a separate control. XGB cumulatively retrains. Proposed model uses five feature-disjoint roles within the SAME total label budget, original pretrained backbone to keep calibration labels outside weight training. Stage order chronological, within-stage sample counts descending. Main score is the FINAL state on ALL held-out rows, not pooled stage predictions.

## 6. 결과

EXP77 final-model full-test results: tabpfn/results/v7/exp77/20261015_230535_initialft10_finalfull_s43_2eb2237d/results.json. Two displayed methods: XGB cumulative retraining from scratch at every stage and original pretrained TabPFN with cumulative context, no IDS fine-tuning. Initial-FT results are retained in EXP77/EXP78, removed from slides 6 and 7 at user request. Slide 6 alone reports later-stage update cost: GPU-synchronized context selection + model construction + fit summed AFTER the initial stage; initial preparation and all inference excluded. Single seed 43. A separate EXP78 label-budget artifact covers 100/50/20/10 new examples per class, with initial 1000 fixed.

## 7. 제안 모델·SOTA 성능 비교

Same EXP77 released-label budget: CIC2018 1400 / ToN 1700, including all internal validation and calibration. Only final full-test Macro-F1 is displayed; all cost panels and initial-FT baseline removed per user request. Existing K=4 anchor/residual + S1V0 proposed model uses original pretrained weights and global/expert/route/tune/cal roles within that total budget. Both datasets have zero expert calls in the final evaluation, so this is a diagnosed current configuration, not demonstrated ensemble improvement. Candidate expert corrections changed correctness on at most 0/4 calibration rows (CIC2018/ToN), below the existing 30-case requirement. Next diagnostic controls are a single PFN with identical Global context and adjustment of shared anchors/expert-specific allocation; do not retune thresholds on test. BoostPFN is the existing 50-learner port. DistPFN uses original pretrained v3 without IDS FT and current-query-batch posterior adjustment; no query labels. Historical 1014 results have a different 100k budget and pooled replay evaluation and are not inserted into the current final-state comparison. Source: tabpfn/results/v7/exp78/20261016_005435_noft_dist_new100_50_20_10_s43_52588487/dist_results.json


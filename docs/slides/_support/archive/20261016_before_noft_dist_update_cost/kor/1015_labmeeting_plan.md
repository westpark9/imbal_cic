# 1015 랩미팅 · 7장 축약본

초기 IDS 가중치 fine-tuning을 실제 수행한 EXP77의 최종 상태 전체 test 평가. 마지막 장은 같은 소규모 라벨 예산의 제안 모델·SOTA 비교. 1002 원본과 이전 25장·1014 포함 7장 자료는 archive에 보존.

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

EXP77 final-model full-test results: tabpfn/results/v7/exp77/20261015_230535_initialft10_finalfull_s43_2eb2237d/results.json. Initial fine-tuning actually changes weights; frozen context-policy branches share the same initial checkpoint. Repeated FT continues from previous weights each stage on cumulative gradient-training rows, keeping validation groups held out. All plotted metrics independently verified against saved predictions. Fixed context has initial classes only. FIFO approximates acquisition age; similarity is measured against NEW TRAIN rows, never test. Update plot excludes initial preparation and full inference; complete workload appears on slide 7. Seed 43 exploratory evidence, no performance equivalence or network-drift guarantee.

## 7. 제안 모델·SOTA 비교

Same EXP77 released-label budget: CIC2018 1400 / ToN 1700, including all internal validation and calibration. Existing K=4 anchor/residual + S1V0 proposed model uses the original pretrained backbone; global/expert/route/tune/cal split 40/20/20/8/12% by feature groups. Thus this compares methods under the same label budget, not a same-backbone architecture ablation. BoostPFN uses existing project port, 50 learners, 500 rows/learner. DistPFN is batch posterior adjustment on the initial-FT cumulative PFN. LoCalPFN excluded as requested. Complete cost includes initial preparation + all later updates + one final full-test inference; disk serialization excluded. Log scale cost dots with printed values. Historical 1014 deck preserved in archive, not mixed with these results.


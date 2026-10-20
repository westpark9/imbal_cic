# 1015 랩미팅 · 7장 축약본

사용자가 지정한 7장 구성. 초기 IDS 가중치 fine-tuning 이후 context를 변경하는 연구 방향과, 초기 fine-tuning이 없었던 기존 결과를 구분한다. 1002 원본과 이전 25장 자료 보존.

## 1. (타이틀 없음)

No title. Rephrase the user intent into concise presentation language instead of copying their draft. Adaptation to changing networks motivates the research; current class-arrival tests do not establish network-drift robustness.

## 2. TabPFN · 표 데이터 예측 규칙을 사전학습한 모델 (v3: arXiv, 2026)

Source: https://arxiv.org/pdf/2605.13986 §2.5, PDF page 10: trained on more than 8 trillion tokens. This is a token count, not a number of synthetic datasets; total dataset count is not specified in the report. Fine-tuning procedure: installed official TabPFN 8.2.0, https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_base.py. Latest user design: one initial IDS weight fine-tune, then freeze weights and change context. This has not been evaluated by EXP76. With the attention-weighted one-hot decoder, old class labels need representation in inference context even after weight fine-tuning; a new-class-only context does not by itself retain the old output label space. That arm needs a controlled context coverage design before an experiment.

## 3. v3 구조

Original Figure 5, https://arxiv.org/pdf/2605.13986#page=6. Native PPTX crop excludes the original caption. Labels and source remain in speaker notes. First three stages: feature grouping/cell embedding; target-aware column distribution embedding; row-level feature aggregation. Plain language deliberately omits implementation dimensions.

## 4. v3 구조

Same Figure 5, https://arxiv.org/pdf/2605.13986#page=6; no visible caption by user request. ICL attends to training/context keys; many-class decoder uses attention-weighted one-hot context labels. This is learned contextual relevance, not merely raw-feature nearest-neighbour distance. Context determines available class labels.

## 5. 평가시나리오

Counts are newly released labelled training examples in EXP76, not test support. Display stages 1..N map to saved stage IDs 0..N-1. Ordered by arrival stage, descending sample count within stage. Same-stage ties are display order only. New proposed initial weight fine-tuning uses stage 1; existing EXP76 results on slide 6 do not implement that step. 1014 has the same class arrival order but a separate 100,000-label budget. Cumulative train/context sizes and held-out test totals are separate.

## 6. 결과

EXP76 measured results: tabpfn/results/v7/exp76/20261015_212937_context1000_new100_s43_1c62d14c/results.json. Frozen pretrained checkpoint, no initial IDS parameter fine-tuning. This is a preliminary context-policy comparison, not a result for the proposed initial-fine-tuned setting. Same full held-out test, instant-ready replay, seed 43, batch 8192. Initial context 1000 and 100 new labels per new class. Update total excludes initial build and inference. All values plotted directly; validation file verifies 16 jobs. BoostPFN/DistPFN results remain in source; slide 6 focuses on requested a/b/c with XGB. Saved prediction totals show PFN inference is substantially more costly than XGB, so update-time savings are not claimed as total cost savings.

## 7. 1014 결과 · 누적 10만 사례와 SOTA 비교

EXP73 all 24 dataset/case/model jobs: tabpfn/results/v7/exp73/20261014_154250_replay_current_sota_100k_s43_b0f8a9dc/results.json. Both natural and uniform within-attack training allocations shown, not test balancing. Macro-F1 from ideal (instant-ready) replay. Costs summed from stage_1 onward fit_metrics.json, excluding initial construction and query inference. Same class arrival sequence as slide 5, different labelled sample budget, final train 100000. TabPFN reconstructs cumulative context; XGB retrains; current proposed model rebuilds Global/expert/S/V; LoCalPFN performs its original v1 fine-tuning/retrieval. Therefore do not attribute all SOTA costs to ordinary context rebuilding. Cost dots use a logarithmic axis to preserve visibility of LoCalPFN while printing every actual value. Existing test/seed43/batch1024. No direct improvement claim against EXP76 with changed sample budget and batch.


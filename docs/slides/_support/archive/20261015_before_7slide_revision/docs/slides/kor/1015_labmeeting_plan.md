# 1015 랩미팅 슬라이드 구성

1002 원본 보존. 0번 연구 관점 전환에 이어 구조도와 단계별 작동 원리, v3 fine-tuning을 설명하는 한국어 새 날짜본.

## 1. TabPFN v3의 작동과 IDS context 갱신



## 2. 0. 연구 관점 · 고정된 F1 경쟁에서 변화에 대응하는 IDS로

User item 0 precedes the TabPFN explanation. Network changes are the operational motivation, not a new cross-dataset experiment or an established robustness result. ICD Introduction motivates pretraining and transfer in tabular prediction; its §4 tuned-XGB comparison is historical benchmark evidence, not a universal conclusion about v3. https://arxiv.org/pdf/2402.06971

## 3. TabPFN · 표 데이터의 예측 규칙을 사전학습한 모델

PFN = Prior-data Fitted Network. Context is labelled task data, not a natural-language prompt. PriorLabs TabPFN-3 report §2; local architecture and classifier implementation. https://arxiv.org/pdf/2605.13986

## 4. Context와 query · 모델에 전달되는 두 종류의 행

Illustrative schema, no fabricated performance values. Context rows and held-out query rows remain distinct. Classification output labels are those represented by the fitted context.

## 5. 현재 실험의 TabPFN v3 설정

Actual saved checkpoint config inspected with torch.load. Parameters counted and hashed in docs/research/20261015/tabpfn_fit_mechanism_check.json. Version 8.2.0 is the installed library, not the model generation. Local checkpoint: tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt. Model report Appendix C; support ceilings are not measured IDS performance.

## 6. v3 구조 · 열의 분포에서 행 단위 표현으로

Redrawn from local architectures/tabpfn_v3.py TabPFNV3 docstring and forward path, with report Figure 5 as architecture cross-check. Local checkpoint: feature_group_size=3, dist_embed_num_blocks=3, feat_agg_num_blocks=3, feat_agg_num_cls_tokens=4. https://arxiv.org/pdf/2605.13986#page=6

## 7. v3 구조 · context 관계와 클래스 확률의 계산

Local ICLTransformerBlock uses train-only keys; ManyClassDecoder uses train representations and one-hot labels. Report §§2.1–2.2, Figure 5. The actual checkpoint has max_num_classes=160; this is a checkpoint constraint even though decoder weights are independent of class count. No unlabeled novel-class naming is implied.

## 8. fit과 predict · context 캐시의 생성과 재사용

1014: scripts/v7/exp72/replay_models.py fit_stage calls TabPFNClassifier.fit, using get_model in exp71_bounded_updates.py. fit_with_cache builds context-dependent state and predict reuses it. Actual parameter SHA-256 unchanged after 1000→1020 fit: see mechanism check. The current library does not incrementally append 20 rows to the old cached state in this experiment. Report §2.4.2 distinguishes cache build vs cached predict.

## 9. 데이터 적응의 세 방식과 학습 대상

TabPFNClassifier versus FinetunedTabPFNClassifier in installed official 8.2.0 source. ICD optimizes input context features through a frozen model, as Ma et al. 2024. 1014 ordinary PFN used row 1; EXP75 used context-feature optimization; EXP74 trained external classification heads and is not the user-referenced experiment.

## 10. v3 가중치 fine-tuning · 학습 절차

The v3 report describes architecture and variants but is not a detailed task-specific fine-tuning tutorial. Procedure here is grounded in the installed official finetuning/finetuned_base.py and finetuned_classifier.py. Cross entropy over query labels; AdamW; explicit independent validation; final inference classifier fits X_, y_. https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_classifier.py

## 11. 가중치 fine-tuning의 실제 설정 항목

Illustrative configuration only, not a new quality result. Pin checkpoint with extra_classifier_kwargs[model_path]. Use epochs=10, learning_rate=1e-5, context+query=1024, query ratio=.2, one estimator as an initial engineering smoke configuration, not a tuned best setting. Validation supplied separately; no test tuning. Official repository recommends high-memory GPU for its 50k example, so batch/context size must be validated on this 24GB GPU.

## 12. 신규 사례만 사용하는 갱신의 의미

User clarified 1000 and 20 were examples, not required sample counts. Ordinary classifier supports a newly represented class through labels in context. A naive FinetunedTabPFNClassifier.fit on a single new class is not equivalent to old-context-plus-new-class training: its train/context label space and validation differ. New-query-only gradients with old context retained need a separate controlled training loop and old-class retention validation; not silently substituted into this experiment.

## 13. 갱신 비용 · 새 모델을 사용할 수 있는 시점까지

Update includes selection/similarity/preprocessing/context cache or model fitting. Initial load distinguished. Serving includes actual input gathering/preprocessing/inference. Hardware and estimator/batch settings fixed within experiment; no speed comparison across incompatible workloads. Use initial + sum updates + serving(Q), not fit time alone. No monetary or energy measurement is claimed.

## 14. 평가 시나리오 · 라벨된 공격 클래스의 순차 도입

STAGES inherited from exp69; 1014 EXP72 test partition shares full clean test rows. Sequence describes class availability, not raw timestamp replay or network drift. Order is introduction order, not class-support sort, because these are timeline cells. Updated experiment evaluates instant-ready versions and measures update cost separately.

## 15. 1014 결과 · 누적 10만 사례의 품질과 갱신 비용

Exact EXP72 source tabpfn/results/v7/exp72/20261014_134735_replay_100k_xgb_tabpfn_s43_de8b4a78/results.json. Aggregate ideal replay over all original test rows, seed 43. Update is stage 1 onward, excludes initial build. Natural/uniform refer to attack-label training allocation, not test balancing. No new experiment or v3 parameter fine-tuning is represented by this table.

## 16. 추가 비교 · 누적·고정 버퍼·갱신 없음

User examples shown explicitly as examples. Experimental defaults are initial 1000, 100 per new class. Policies all receive same released data, but fixed deliberately does not use later labels; information actually consumed is reported. b protects >=1 row per existing class. Not an incremental KV-cache implementation.

## 17. 고정 버퍼의 삭제 기준

FIFO denotes oldest acquisition, not measured semantic obsolescence. Similarity uses old-context StandardScaler and distance to new labelled train cases, never test. All three rules retain at least one example of each old class. Same-stage ties use seeded acquisition order. Nearest-row removal may damage discrimination: hypothesis only.

## 18. EXP76 · 첫 소규모 context 비교 설정

Full protocol docs/research/20261015/exp76_protocol.md. 100 per new class is an investigator setting, not a user-imposed value. 8192 batches differ from 1014; direct timing comparison is not valid. No cross-network transfer claim. New results are exploratory, same previously examined fixed test, single seed.

## 19. CIC2018 · 소규모 context의 품질과 비용

Source tabpfn/results/v7/exp76/20261015_212937_context1000_new100_s43_1c62d14c/results.json. All test rows, one seed, instant-ready replay. New label budget and prediction batch differ from 1014. Missing rows mean unfinished, never fabricated. Classwise metrics and support are in the experiment artifacts.

## 20. CIC2018 · 클래스별 구분 성능과 버퍼 정책

Same full test population as preceding aggregate table. Classes sorted by evaluated support descending. Values are class F1 percentages; support is actual evaluation support, not training budget. All three buffer policies shown, without choosing a test-best policy.

## 21. ToN · 소규모 context의 품질과 비용

Source tabpfn/results/v7/exp76/20261015_212937_context1000_new100_s43_1c62d14c/results.json. All test rows, one seed, instant-ready replay. New label budget and prediction batch differ from 1014. Missing rows mean unfinished, never fabricated. Classwise metrics and support are in the experiment artifacts.

## 22. ToN · 클래스별 구분 성능과 버퍼 정책

Same full test population as preceding aggregate table. Classes sorted by evaluated support descending. Values are class F1 percentages; support is actual evaluation support, not training budget. All three buffer policies shown, without choosing a test-best policy.

## 23. 채택 판단 · 품질을 유지하는 작은 context와 낮은 갱신 비용

Do not claim statistical equivalence without a predeclared margin and repeated data/seeds. Update latency and raw throughput have different meanings. New class F1 and old class errors/FPR must be inspected before choosing a policy; do not select a final model solely from this test ranking. Manuscript changes are proposals only under user steering.

## 24. 부록 · v3 원문의 전체 구조

Original Figure 5 from TabPFN-3 Technical Report, PDF page 6. Cropped only to figure bounds using native PPTX picture crop; no diagram content altered. https://arxiv.org/pdf/2605.13986#page=6

## 25. 부록 · v3 fine-tuning 코드 예시

Illustrative official API usage, not a reported experiment. Adapt context/query size to GPU and label counts. Verify validation disjointness before use. Model checkpoint is explicitly pinned. This snippet intentionally does not fit only a single newly arriving class. https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_classifier.py

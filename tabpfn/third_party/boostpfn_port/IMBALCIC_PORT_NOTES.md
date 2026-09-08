# BoostPFN port notes (imbalcic, 2026-09-04)

Clone of https://github.com/yxzwang/BoostPFN (commit `c957ac2`, "Add MIT License").
Upstream targets **TabPFN v1 (pip `tabpfn==0.1.9`) + scikit-learn 0.24 + torch 1.9 +
python 3.7**.  This copy runs inside the imbalcic conda env (python 3.13, torch
2.11+cu130, scikit-learn 1.8, numpy 2.4) where the pip name `tabpfn` is the
project's **v3 fork** (editable install) -- so the v1 package cannot be installed
alongside it.  The boosting arithmetic is untouched; only import/API shims changed.

## What was added

| Path | What |
|---|---|
| `tabpfn_v1/` | TabPFN v1 source from the `tabpfn-0.1.9` wheel (PyPI), `tabpfn.` imports rewritten to `tabpfn_v1.`; `tests/`, `datasets/`, notebooks dropped (need `openml`). |
| `models_diff/prior_diff_real_checkpoint_n_0_epoch_100.cpkt` | The v1.0.0 checkpoint (103,350,223 B, sha256 `3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392`) downloaded from `github.com/PriorLabs/TabPFN/raw/v1.0.0/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt`. Stored under the **epoch_100** name because the v1 loader probes epochs 100→0 and the original code downloaded epoch_42 *into* that first probed name. |
| `gb_losses_compat.py` | `MultinomialDeviance` / `LeastSquaresError` transcribed from scikit-learn 0.24.2 `sklearn/ensemble/_gb_losses.py` (module removed in 1.3). |

## What was patched (git diff shows exact lines)

| File(s) | Change | Why |
|---|---|---|
| all `*.py` outside `tabpfn_v1/` | `from/import tabpfn.` → `tabpfn_v1.` | name collision with the v3 package |
| `scripts/transformer_prediction_interface.py`, `tabpfn_v1/scripts/…` | `force_all_finite=False` → `ensure_all_finite=False` | renamed in sklearn 1.6, removed in 1.8 |
| same two files | inference path calls `predict()` directly instead of `torch.utils.checkpoint.checkpoint(...)`; grad path passes `use_reentrant=False`; `torch.cuda.amp.autocast` → `torch.autocast('cuda')` | torch ≥ 2.4 requires explicit `use_reentrant`; checkpointing is a no-op under `inference_mode` anyway |
| same two files | dead-URL auto-download replaced by a `FileNotFoundError` with instructions | the `automl/TabPFN` raw URL now returns a 404 HTML page that was being written to disk as a "checkpoint" |
| `tabpfn_v1/scripts/model_builder.py` (3 sites) | `torch.load(..., weights_only=False)` | kept explicit for forward-compatibility with torch ≥ 2.6's `weights_only` default; on torch 2.11 `weights_only=True` also loads this file and yields an identical `(state_dict, None, config-dict)` tuple (verified 2026-09-04) |
| `tabpfn_v1/layer.py` | explicit imports of `Optional`, `Tensor`, `Module`, `MultiheadAttention`, `Linear`, `Dropout`, `LayerNorm` | torch 2.x no longer re-exports them from `torch.nn.modules.transformer` |
| `gradient_boost_tabpfn.py`, `boost_tabpfn.py` | `try: sklearn.ensemble._gb_losses … except ImportError: gb_losses_compat` | see above |

Not touched: `datasets/` (OpenML loaders), `priors/`, `scripts/differentiable_pfn_evaluation.py`,
`largedataset_boostpfn.py`, `main_10times.py`.  2026-09-08: to run the upstream driver itself,
`openml`, `catboost`, `hyperopt` were pip-installed (benchmark/baseline deps, not the method) and
one numpy-2 line fixed in both `tabular_baselines.py` copies (`np.warnings.filterwarnings(...)` ->
`warnings.filterwarnings(...)`; `np.warnings` was removed in numpy 1.24).  With that,
`python largedataset_boostpfn.py --modelname gboost_tabpfnV2 --gpu 0 --sampling_size 0.001
--test_batch 50000 --seed 5 --maxsample 500 --updating exphadamard --endnum 1 --step 1 --startnum -1
--ensemble_num 10` runs UNMODIFIED on OpenML id 133 BNG(glass,nominal,137781): AUC 0.8888 in 10.4 s
(paper Table 11 / 13 for this dataset: 0.8824 at T=10 on 5k rows, 0.900 at T=1000 on the full set).
Results land in `largeresults/` (created; gitignored by nothing -- leave out of commits).

## Verified

`scratchpad/smoke_boostpfn_synthetic.py` (7-class imbalanced synthetic, RTX 4090):
single weak learner fit+predict 0.2 s; `gboost_tabpfnV2` 3 rounds 0.3 s; `adaboost_tabpfn`
2 rounds; missing-class fallback branch; peak GPU 332 MiB.

Independent review (2026-09-04, three reviewers + refuters, imbalcic session):
- `diff -r` pristine `tabpfn-0.1.9` wheel vs `tabpfn_v1/`: 18 files differ, every hunk is one
  of the documented patches, no files added/missing beyond the documented drops.
- `MultinomialDeviance` matches scikit-learn 0.24.2 (`_gb_losses.py` from the PyPI sdist,
  sha256-verified) on identical inputs. `LeastSquaresError`/`BinomialDeviance` were found to be
  non-transcriptions and now raise `NotImplementedError` (BoostPFN's `--loss MSE` path cannot
  run under 0.24.2 either).
- Direct `predict()` under `torch.autocast('cuda', enabled=False)` vs the original
  `checkpoint(predict, ...)`: identical outputs (real checkpoint, seed 42, 500 × 46 / 700 rows).
- `check_X_y(ensure_all_finite=False)` is the documented rename of `force_all_finite=False`
  (sklearn 1.6 changelog in `validation.py`).
- The IDS run A (`imbalcic/tabpfn/results/20260904_154310_*`) was reproduced bit-exactly from
  the same seed (macro-F1 0.6135806559864706), i.e. the port + runner are deterministic on this GPU.
- Not a port issue but worth knowing: BoostPFN's failure traps do `import ipdb` (not installed);
  they are unreachable with `exphadamard` (weights bounded, clipped ≥ eps) but `--updating
  loghadamard` can reach them.

## How the IDS experiment uses it

`imbalcic/tabpfn/scripts/nfv3_v3_exp35_boostpfn.py` inserts this directory at the front of
`sys.path` and imports `scripts.transformer_prediction_interface.TabPFNClassifier`,
`gradient_boost_tabpfn.SamplingGradientboost`, `boost_tabpfn.SamplingAdaboost`,
`utils.splitting_predict_proba`.  Nothing here reads the imbalcic data directly.

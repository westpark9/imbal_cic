#!/usr/bin/env python3
"""EXP 3 (priority 3) -- FULL pool for TabPFN via disjoint per-estimator bagging.

The question: TabPFN cannot put 12M rows in one context, so can the full pool
still participate?  Yes, by splitting it across ensemble members.

    --max-train-samples -1        the whole train pool reaches fit()
    --subsample-samples k         each estimator's context is capped at k rows
    --n-estimators 0              auto = ceil(pool / k)  -> the pool is covered exactly once
    --ignore-pretraining-limits   mandatory (the pool exceeds the 1M checkpoint guard)

What the library actually does (verified in source, not assumed)
----------------------------------------------------------------
  * each estimator gets its OWN row-index array, applied before all preprocessing
    (preprocessing/transform.py:66-68)
  * the draws come from a SHARED POOL carried across estimators
    (preprocessing/ensemble.py:735-769, `_draw_balanced_from_pool` returns the
    remaining pool to the caller), so the per-estimator subsets are DISJOINT --
    a partition, not independent samples
  * the classifier path is genuinely stratified with >=1 row per class
    (classifier.py:857-858 -> ensemble.py:236 -> 404-406)
  * per-estimator predicted PROBABILITIES are averaged, then argmax
    (classifier.py:1543-1545) -- one prediction per test row, not an average of
    per-estimator metrics

READ THIS BEFORE WRITING IT UP
------------------------------
1. This is NOT "trained on 12M rows".  Each predictor is still conditioned on k
   rows; the ensemble is a probability average (bagging).  Averaging reduces
   variance, it does not raise the effective conditioning set above k.  The
   accurate phrasing is "disjoint bagging covering 100% of the pool".

2. This construction appears in NO paper.  The four behaviours above are real
   library features, but combining them as n_estimators = ceil(pool/k) to get
   full coverage is not in the TabPFN paper, not in the reference IDS paper, and
   not in the upstream README (which recommends user-side subsampling instead).
   SUBSAMPLE_SAMPLES exists to bound memory, not to provide coverage.

3. Stratification does NOT protect tail classes.  It is proportional, so the
   imbalance ratio is preserved exactly: cic2018's `web_attacks` is 0.0126% of
   the pool and gets ~114 rows at k=900,000.  This reproduces the starvation
   documented in manuscript/report/0813.md rather than fixing it.  If tail F1
   does not move, that is a result, not a bug.

4. TWO KNOBS MOVE AT ONCE versus the existing n_estimators=4 runs: the ensemble
   size AND the coverage.  To attribute any difference you need the control run
   -- same k rows in EVERY estimator, same n_estimators:

       # control: ensemble size only, no coverage gain
       python nfv3_v3_exp3_full_tabpfn_bagging.py --target-dataset cic2018 \
           --max-train-samples 900000 --subsample-samples 0 --n-estimators 14

       # treatment: same ensemble size, full coverage
       python nfv3_v3_exp3_full_tabpfn_bagging.py --target-dataset cic2018 \
           --max-train-samples -1 --subsample-samples 900000 --n-estimators 0

   treatment - control = the coverage effect.  (CLAUDE.md M6.)

Why NOT cache mode here
-----------------------
The saved/resident KV cache is ~4,096 B per row per estimator.  At k=900,000 and
14-19 estimators that is 52-70 GB, which does not fit host RAM alongside the
12.3 GB pickle.  exp3 therefore stays in the default fit_mode; memory per forward
is one context (k rows) plus one test batch, which is modest.

--test-batch-size per target (default mode, k=900,000 context, 24 GB card)
--------------------------------------------------------------------------
    context base = 0.198 + 900,000 x 18,530 B = 15.74 GiB
    headroom to the ~20.5 GiB safe line = 4.76 GiB = ~414,000 test rows

    bot_iot_capped     70,722  -> 70722   (1 batch)   peak 16.55
    cic2018_capped    120,518  -> 120518  (1 batch)   peak 17.12
    ton_iot_capped    161,999  -> 161999  (1 batch)   peak 17.60
    cic2017_full      211,322  -> 211322  (1 batch)   peak 18.17   [70 features]
    bot_iot           310,722  -> 310722  (1 batch)   peak 19.30
    cic2018           440,276  -> 220138  (2 batches) peak 18.26
    ton_iot           659,725  -> 329863  (2 batches) peak 19.52

    Batching costs time (each batch re-encodes the k-row context) but is required
    for the two big test sets.  Predicted peak is printed before the run.

Expected cost (k=900,000, RTX 4090)
-----------------------------------
    bot_iot   12 estimators x 1 batch   ~1.2 h
    cic2018   14 estimators x 2 batches ~2.6 h
    ton_iot   19 estimators x 2 batches ~3.9 h

Usage
-----
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python nfv3_v3_exp3_full_tabpfn_bagging.py --target-dataset cic2018 \
        --subsample-samples 900000 --test-batch-size 220138
    python nfv3_v3_exp3_full_tabpfn_bagging.py --target-dataset bot_iot \
        --subsample-samples 900000 --test-batch-size 310722
    python nfv3_v3_exp3_full_tabpfn_bagging.py --target-dataset ton_iot \
        --subsample-samples 900000 --test-batch-size 329863

XGBoost runs too, on the same full pool, so the comparison lives in one run
(CLAUDE.md: baseline and method in the same script, same split).  If exp2 already
covered this target with the same --seed, add --skip-xgboost to save the hours.
"""

import math

import nfv3_v3_common as core


def main():
    p = core.base_parser(__doc__)
    p.set_defaults(
        max_train_samples=-1,          # the whole pool reaches fit()
        subsample_samples=900_000,     # per-estimator context
        n_estimators=0,                # 0 = auto: ceil(pool / k)
        fit_mode="fit_preprocessors",  # NOT cache mode -- see the docstring
        ignore_pretraining_limits=True,
        test_batch_size=0,
    )
    args = p.parse_args()

    # The pool exceeds the checkpoint's 1M guard by construction. Without this
    # flag the script would re-cap UNIFORMLY (not stratified), destroying the
    # tail classes the experiment is about.
    args.ignore_pretraining_limits = True

    if args.n_estimators == 0:
        if args.subsample_samples <= 0:
            raise SystemExit(
                "--n-estimators 0 (auto) needs a positive --subsample-samples: the "
                "number of estimators is derived as ceil(train_pool / k).")
        # Peek at the pool size to derive full coverage. Cheap relative to the run,
        # and it keeps the derived value in the Args: line for the record.
        cfg = core.build_dataset_config(args.data_dir)
        if args.data is None:
            args.data = cfg[args.target_dataset]["default_data"]
        _, _, train_idx, _, _, _, _, _, _ = cfg[args.target_dataset]["loader"](args)
        pool = len(train_idx)
        args.n_estimators = max(1, math.ceil(pool / args.subsample_samples))
        print(f"[auto] train pool {pool:,} / k {args.subsample_samples:,} "
              f"-> --n-estimators {args.n_estimators} "
              f"(covers {min(100.0, 100 * args.n_estimators * args.subsample_samples / pool):.1f}%)")

    core.run_experiment(args, experiment_name="exp3_full_tabpfn_bagging",
                        out_tag_suffix="exp3_bagging")


if __name__ == "__main__":
    main()

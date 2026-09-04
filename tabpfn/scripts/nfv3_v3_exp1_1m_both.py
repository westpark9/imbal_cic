#!/usr/bin/env python3
"""EXP 1 (priority 1) -- 1M-row context, XGBoost + TabPFN, cache mode.

The direct successor to the existing 900k runs: same protocol, same split, same
reference-paper n_estimators=4, but the context is raised to the checkpoint's own
MAX_NUMBER_OF_SAMPLES (1,000,000) and the test set goes through in ONE batch under
fit_mode="fit_with_cache".  Both models see the identical `X_train_used`, so this
is a like-for-like comparison.

Why cache mode here
-------------------
Memory is one forward: the whole training context plus ONE test batch.  In the
default fit_mode the context is re-uploaded and re-encoded on every predict()
call, which is what made the 2026-08-13 run take 6.85 h (23 batches x 4
estimators = 92 full re-encodes of a 900k-row context) and what OOM'd the
one-batch attempt.  Measured at 900k train / 440,276 test / n_estimators=4:

    default mode : predict peak 21.105 GiB   1,599.7 s   (OOM without expandable_segments)
    cache  mode  : predict peak 16.550 GiB     552.7 s   + 1,054.2 s cache build

Cache mode costs a one-off build and buys 4.5 GiB of headroom plus cheap repeat
predicts.  It does NOT raise the maximum context (the build costs ~20,674 B/row
vs 18,530 for a default predict) -- see exp3 for the full-pool path.

--test-batch-size per target (measured eval rows; all fit ONE batch at a 1M context
in cache mode on a 24 GB card -- predicted peaks in parentheses)
------------------------------------------------------------------------------------
    bot_iot_capped     70,722   -> --test-batch-size 70722    (build 19.47 binds)
    cic2018_capped    120,518   -> --test-batch-size 120518   (build 19.47 binds)
    ton_iot_capped    161,999   -> --test-batch-size 161999   (build 19.47 binds)
    cic2017_full      211,322   -> --test-batch-size 211322   (build 19.47 binds)
    bot_iot           310,722   -> --test-batch-size 310722   (predict 16.21)
    cic2018           440,276   -> --test-batch-size 440276   (predict 17.69)
    ton_iot           659,725   -> --test-batch-size 659725   (predict 20.22, tight)

    Leaving --test-batch-size at its default 0 means "one batch over everything",
    which is what you want here.  The predicted peak is printed before the run;
    if it exceeds ~20.5 GiB on a 24 GB card, halve the batch.

    The three `_capped` targets have train pools of 212k-486k rows, so a 1M budget
    is not binding -- they run on 100% of their pool.

Usage
-----
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # keep the big cache files off Drive / off the repo
    mkdir -p /tmp/tabpfn_cache

    python nfv3_v3_exp1_1m_both.py --target-dataset cic2018 \
        --resume-dir /tmp/tabpfn_cache --models-dir /tmp/tabpfn_cache

WARNING: under fit_with_cache the saved model is ~4 KB x rows x n_estimators
(about 16 GB at 1M rows x 4 estimators), written to BOTH --resume-dir and
--models-dir.  Point them at local scratch, never at Google Drive.
"""

import nfv3_v3_common as core


def main():
    p = core.base_parser(__doc__)
    p.set_defaults(
        max_train_samples=1_000_000,   # the checkpoint's own MAX_NUMBER_OF_SAMPLES
        n_estimators=4,                # reference-paper value; unchanged from the 900k runs
        fit_mode="fit_with_cache",
        test_batch_size=0,             # 0 = one batch over the whole eval set
        subsample_samples=0,           # off: every estimator sees the same context
    )
    args = p.parse_args()

    # 1,000,000 == MAX_NUMBER_OF_SAMPLES exactly, so the guard does not trip and
    # --ignore-pretraining-limits is not required. Raising the budget above 1M
    # would require it (and would push the cache build past ~20.7 GiB).
    core.run_experiment(args, experiment_name="exp1_1m_both", out_tag_suffix="exp1_1m")


if __name__ == "__main__":
    main()

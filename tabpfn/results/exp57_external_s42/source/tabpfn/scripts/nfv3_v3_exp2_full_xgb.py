#!/usr/bin/env python3
"""EXP 2 (priority 2) -- FULL uncapped training pool, XGBoost only.

The reference point the project has never actually measured.  Every run so far
capped BOTH models at 100k-1M rows, because the script feeds TabPFN and XGBoost
the identical `X_train_used` (v2:552 / v2:586).  So "XGBoost beats TabPFN"
(manuscript/report/0812.md) was established at a capped budget, not at full data.

This script removes the cap for XGBoost alone:

    --max-train-samples -1   ->  train_used_idx == the whole train split
    --skip-tabpfn            ->  no GPU work at all

Result: what XGBoost actually achieves when it uses everything it can use.
That is the honest upper reference for the TabPFN comparison, and it is cheap --
no GPU, no context, no batching.

Why TabPFN cannot simply do the same
------------------------------------
TabPFN's training set IS its prompt, so the whole pool would have to fit in one
context.  The checkpoint declares MAX_NUMBER_OF_SAMPLES = 1,000,000 (hard error
at validation.py:266-272), and even bypassing that guard the memory is linear
with no cap: 12,069,313 rows x 18,530 B = 213 GiB for cic2018, i.e. 2.8x an
80 GB A100.  exp3 is the closest achievable substitute.

Train pools (measured)
----------------------
    bot_iot_capped        212,162        cic2017_full     1,696,720
    cic2018_capped        361,517        bot_iot         10,160,284
    ton_iot_capped        485,989        cic2018         12,069,313
                                         ton_iot         16,512,152

    For the three `_capped` targets this script is identical to exp1's XGBoost
    half (their pools are already below the 1M budget) -- run it on the uncapped
    targets and cic2017_full, where it is genuinely new.

--test-batch-size
-----------------
    Irrelevant here.  XGBoost predicts through `inplace_predict` on the whole
    evaluation matrix in one go; the flag only ever affected TabPFN.  Leave it at 0.

Cost note
---------
    XGBoost multi:softprob builds n_classes x n_estimators trees.  On ton_iot
    that is 10 x 300 = 3,000 trees over 16.5M rows -- expect hours on CPU, and
    roughly 555 MB for the QuantileDMatrix (~1 byte per cell, not a dense copy).
    Host RAM is the thing to watch, not GPU: `X_train_used` alone is
    16,512,152 x 46 x 4 B = 2.8 GB, momentarily doubled by `np.nan_to_num`.

Usage
-----
    python nfv3_v3_exp2_full_xgb.py --target-dataset cic2018
    python nfv3_v3_exp2_full_xgb.py --target-dataset bot_iot
    python nfv3_v3_exp2_full_xgb.py --target-dataset ton_iot
"""

import nfv3_v3_common as core


def main():
    p = core.base_parser(__doc__)
    p.set_defaults(
        max_train_samples=-1,     # negative = no cap: the entire train split
        skip_tabpfn=True,
        fit_mode="fit_preprocessors",   # unused (no TabPFN) but keeps the tag honest
        test_batch_size=0,
    )
    args = p.parse_args()
    args.skip_tabpfn = True            # this experiment is XGBoost-only by definition
    core.run_experiment(args, experiment_name="exp2_full_xgb", out_tag_suffix="exp2_fullxgb")


if __name__ == "__main__":
    main()

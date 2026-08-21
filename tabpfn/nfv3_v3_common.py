#!/usr/bin/env python3
"""Shared core for the v3 TabPFN experiment scripts -- forked from
nfv3_multiclass_test_v2.py (2026-08-17).  v2 is left untouched; its runs are the
existing record.

Three experiment scripts import this module and differ ONLY in which models run
and how the training context is built:

  nfv3_v3_exp1_1m_both.py            1M context, XGBoost + TabPFN, cache mode
  nfv3_v3_exp2_full_xgb.py           full uncapped pool, XGBoost only
  nfv3_v3_exp3_full_tabpfn_bagging.py  full pool via disjoint per-estimator bagging

What changed from v2
--------------------
1. **`--fit-mode` / `--keep-cache-on-device` exposed.**  v2 hardcoded TabPFN's
   default `fit_mode="fit_preprocessors"`, which re-uploads and re-encodes the
   whole training context on EVERY predict() call.  `fit_with_cache` builds the
   context K/V once during fit() so predict() never touches the train rows.
   Measured on this RTX 4090 (900k train / 440,276 test / n_estimators=4):

       default mode  : predict peak 21.105 GiB, 1,599.7 s
       cache  mode   : predict peak 16.550 GiB,   552.7 s  (+1,054.2 s build)

   `keep_cache_on_device` defaults to False here (v2 could not set it at all;
   TabPFN's own default is True).  True keeps one cache per estimator resident:
   measured 6.367 vs 4.046 GiB peak at 200k rows for a 2% speedup -- not worth it.

2. **`--subsample-samples` exposed** (TabPFN's `SUBSAMPLE_SAMPLES`).  v2 pinned
   it to None.  With a positive k each ensemble member draws its OWN DISJOINT
   stratified k rows from a shared pool (tabpfn/src/tabpfn/preprocessing/
   ensemble.py:735-769), so n_estimators * k rows collectively enter the
   ensemble.  This is the only way a pool larger than one context can
   participate -- see exp3's docstring for what that does and does NOT mean.

3. **`resume_tag` includes fit_mode and subsample_samples**, so a cache-mode run
   cannot silently reuse a default-mode checkpoint.

4. **`--skip-tabpfn` / `--skip-xgboost`.**  exp2 needs XGBoost alone.

5. Everything else -- loaders, the per-class chronological split, the train
   budget policy, the drift diagnostic, the CSV/JSON artifacts -- is byte-identical
   to v2 so results stay comparable.

6. **`--train-split {train, train+val}`.**  v2 computed a 20% validation slice
   and never used it -- nothing in this pipeline does early stopping or
   calibration, so under the default it is simply discarded.  `train+val` folds
   it back in (an 80/20 split); the test slice is untouched either way, so the
   two settings are directly comparable on the same evaluation rows.
   Measured on cic2017_full: macro F1 0.717 -> 0.804 and `bot` F1 0.000 -> 0.800,
   because the 60% cut landed just BEFORE a behavioural phase change and every
   post-change training example sat in the discarded slice
   (manuscript/report/0817.md).  Default stays `train` so earlier runs reproduce.

Measured memory model (RTX 4090, 46 features; see docs/tabpfn_v2_memory_rootcause.html)
---------------------------------------------------------------------------------------
    default mode predict = 0.198 GiB + n_train * 18,530 B + n_test_batch * 12,343 B
    cache  mode build    = 0.221 GiB + n_train * 20,674 B
    cache  mode predict  = 1.180 GiB + n_train * 12,298 B + n_test_batch * 12,343 B

    Memory is ONE forward: the whole training context plus ONE test batch.
    n_estimators and the batch count multiply TIME, not memory (estimators and
    batches run sequentially).

    ALWAYS run with  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -- without it the allocator fragments ~2.7 GiB and OOMs a run that otherwise fits.

Per-dataset --test-batch-size  (measured test-eval row counts at --test-cap-per-class 100000)
----------------------------------------------------------------------------------------------
                          test rows   exp1 (cache, 1M ctx)     exp3 (default, 900k ctx)
    bot_iot_capped           70,722   70722   -> 1 batch       70722   -> 1 batch
    cic2018_capped          120,518   120518  -> 1 batch       120518  -> 1 batch
    ton_iot_capped          161,999   161999  -> 1 batch       161999  -> 1 batch
    cic2017_full            211,322   211322  -> 1 batch       211322  -> 1 batch
    bot_iot                 310,722   310722  -> 1 batch       310722  -> 1 batch
    cic2018                 440,276   440276  -> 1 batch       220138  -> 2 batches
    ton_iot                 659,725   659725  -> 1 batch       329863  -> 2 batches

    Rule of thumb: keep the predicted peak (printed at startup) under ~20.5 GiB on
    a 24 GB card.  On a 40/80 GB A100 you can raise both the context and the batch;
    re-derive from the formulas above.  On a Colab T4 none of this holds --
    FlashAttention needs sm80+, so a T4 falls back to the MATH backend whose memory
    is QUADRATIC in context length.
"""

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from exp_utils import load_pickle as _load_pickle_uncached  # noqa: E402
from exp_utils import scenario_chronological_split, subset_indices, labels_for  # noqa: E402

from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model  # noqa: E402

# The uncapped NF-v3 suite pickle is 14.7 GB and takes minutes to read. exp3 calls
# a loader twice (once to size the auto n_estimators, once for the real run), so
# memoize -- one process only ever touches one pickle. Cleared in run_experiment()
# as soon as the train/test matrices have been copied out.
_PICKLE_CACHE = {}


def load_pickle(path):
    if path not in _PICKLE_CACHE:
        _PICKLE_CACHE[path] = _load_pickle_uncached(path)
    return _PICKLE_CACHE[path]


# ---------------------------------------------------------------------------
# Measured memory model (see module docstring)
# ---------------------------------------------------------------------------

GIB = 1024 ** 3
B_DEFAULT_TRAIN = 18_530
B_CACHE_BUILD = 20_674
B_CACHE_TRAIN = 12_298
B_TEST = 12_343
GIB_WEIGHTS = 0.198
GIB_CACHE_BUILD_CONST = 0.221
GIB_CACHE_PRED_CONST = 1.180
SAFE_PEAK_GIB_24GB = 20.5

# test-eval rows per target at --test-cap-per-class 100000, measured by running
# each loader + cap_per_class (not estimated).
TEST_ROWS = {
    "bot_iot_capped": 70_722,
    "cic2018_capped": 120_518,
    "ton_iot_capped": 161_999,
    "cic2017_full": 211_322,
    "bot_iot": 310_722,
    "cic2018": 440_276,
    "ton_iot": 659_725,
}


def estimate_peak_gib(n_train, n_test_batch, fit_mode):
    """Return (binding_peak, build_peak_or_None, predict_peak) in GiB."""
    if fit_mode == "fit_with_cache":
        build = GIB_CACHE_BUILD_CONST + n_train * B_CACHE_BUILD / GIB
        pred = GIB_CACHE_PRED_CONST + (n_train * B_CACHE_TRAIN + n_test_batch * B_TEST) / GIB
        return max(build, pred), build, pred
    pred = GIB_WEIGHTS + (n_train * B_DEFAULT_TRAIN + n_test_batch * B_TEST) / GIB
    return pred, None, pred


def report_memory_plan(n_train, n_test_rows, args):
    """Print the predicted peak before the expensive part starts."""
    tb = min(args.test_batch_size, n_test_rows) if args.test_batch_size > 0 else n_test_rows
    n_batches = math.ceil(n_test_rows / tb) if tb else 1
    peak, build, pred = estimate_peak_gib(n_train, tb, args.fit_mode)
    print("\n--- predicted GPU memory (measured model, RTX 4090 / 46 feat) ---")
    print(f"  context rows      : {n_train:,}")
    print(f"  test batch        : {tb:,}  ({n_batches} batch(es) over {n_test_rows:,} rows)")
    if build is not None:
        print(f"  cache build peak  : {build:6.2f} GiB")
        print(f"  predict peak      : {pred:6.2f} GiB")
    else:
        print(f"  predict peak      : {pred:6.2f} GiB")
    print(f"  BINDING PEAK      : {peak:6.2f} GiB", end="")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_properties(0).total_memory / GIB
        print(f"   (card reports {cap:.2f} GiB)")
        if peak > SAFE_PEAK_GIB_24GB and cap < 30:
            print("  WARNING: above the ~20.5 GiB safe line for a 24 GB card. Lower "
                  "--test-batch-size or --max-train-samples, or expect a CUDA OOM.")
    else:
        print()
    if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        print("  WARNING: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is NOT set. "
              "Allocator fragmentation (~2.7 GiB measured) can OOM a run that fits.")
    print()


# ---------------------------------------------------------------------------
# Dataset loaders -- identical to v2, chronological split only
# ---------------------------------------------------------------------------

def load_nfv3_suite_subset(args, dataset_name):
    """Shared loader for any dataset_names value inside the NF-v3 suite pkls."""
    suite = load_pickle(args.data)
    X = suite["X"]
    datasets = np.asarray(suite["dataset_names"])
    families = np.asarray(suite["families"])
    scenarios = np.asarray(suite["attack_scenarios"])
    timestamps = np.asarray(suite["timestamps"])

    target_idx = np.flatnonzero(datasets == dataset_name)
    if not len(target_idx):
        raise ValueError(f"No rows for dataset_names == {dataset_name!r}")

    class_names = sorted(np.unique(families[target_idx]).tolist())

    def label_fn(idx):
        return labels_for(idx, families, class_names)

    split, split_audit = scenario_chronological_split(target_idx, scenarios, timestamps)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    return (X, class_names, train_idx, val_idx, test_idx,
            label_fn(train_idx), label_fn(test_idx), split_audit, label_fn)


def load_cic2018(args):
    return load_nfv3_suite_subset(args, "cse_cic_ids2018")


def load_bot_iot(args):
    return load_nfv3_suite_subset(args, "bot_iot")


def load_ton_iot(args):
    return load_nfv3_suite_subset(args, "ton_iot")


def load_cic2017_full(args):
    """All 15 CIC-IDS2017 classes; chronological split on the synthetic
    (file_order, in-file row order) time_proxy."""
    d = load_pickle(args.data)
    X = d["X"]
    class_names = list(d["class_names"])
    y_full = np.asarray(d["y"], dtype=np.int64)
    all_idx = np.arange(len(y_full), dtype=np.int64)

    def label_fn(idx):
        return y_full[idx]

    time_proxy = np.asarray(d["time_proxy"], dtype=np.int64)
    split, split_audit = scenario_chronological_split(all_idx, y_full, time_proxy)
    split_audit = split_audit.rename(columns={"scenario": "class"})
    split_audit["class"] = split_audit["class"].astype(int).map(
        {i: name for i, name in enumerate(class_names)}
    )
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    return (X, class_names, train_idx, val_idx, test_idx,
            label_fn(train_idx), label_fn(test_idx), split_audit, label_fn)


def build_dataset_config(data_dir):
    return {
        "cic2018": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_uncapped_scenarios.pkl"),
            "loader": load_cic2018, "tail_classes": ["bot", "infiltration", "web_attacks"],
            "out_tag": "nfv3_cic2018", "n_features": 46,
        },
        "cic2018_capped": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_cic2018_scenarios.pkl"),
            "loader": load_cic2018, "tail_classes": ["bot", "infiltration", "web_attacks"],
            "out_tag": "nfv3_cic2018capped", "n_features": 46,
        },
        "bot_iot": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_uncapped_scenarios.pkl"),
            "loader": load_bot_iot, "tail_classes": ["theft"],
            "out_tag": "nfv3_botiot", "n_features": 46,
        },
        "bot_iot_capped": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_cic2018_scenarios.pkl"),
            "loader": load_bot_iot, "tail_classes": ["theft"],
            "out_tag": "nfv3_botiotcapped", "n_features": 46,
        },
        "ton_iot": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_uncapped_scenarios.pkl"),
            "loader": load_ton_iot, "tail_classes": ["mitm", "ransomware"],
            "out_tag": "nfv3_toniot", "n_features": 46,
        },
        "ton_iot_capped": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_cic2018_scenarios.pkl"),
            "loader": load_ton_iot, "tail_classes": ["mitm", "ransomware"],
            "out_tag": "nfv3_toniotcapped", "n_features": 46,
        },
        "cic2017_full": {
            "default_data": os.path.join(data_dir, "cic2017_full_raw.pkl"),
            "loader": load_cic2017_full,
            "tail_classes": ["heartbleed", "web-attack-sql-injection", "infiltration",
                             "web-attack-xss", "web-attack-brute-force"],
            "out_tag": "cic2017_full", "n_features": 70,
        },
    }


# ---------------------------------------------------------------------------
# Train-budget / test-cap helpers -- identical to v2
# ---------------------------------------------------------------------------

def cap_per_class(indices, labels, n_classes, maximum, seed):
    if maximum <= 0:
        return np.sort(indices)
    chosen = []
    for i in range(n_classes):
        class_indices = indices[labels == i]
        if len(class_indices) == 0:
            continue
        chosen.append(subset_indices(class_indices, maximum, seed + 500 + i))
    return np.sort(np.concatenate(chosen))


def stratified_subset(indices, labels, n_classes, total_budget, seed):
    """Largest-remainder proportional subsample down to `total_budget` rows,
    preserving each class's share and guaranteeing >=1 row per present class."""
    indices = np.asarray(indices, dtype=np.int64)
    if total_budget <= 0 or total_budget >= len(indices):
        return np.sort(indices)
    counts = np.bincount(labels, minlength=n_classes)
    present = counts > 0
    n_present = int(present.sum())
    if total_budget < n_present:
        raise ValueError(
            f"--max-train-samples={total_budget} is smaller than the number of "
            f"present classes ({n_present}); raise it to at least that."
        )
    raw = counts * total_budget / counts.sum()
    target = np.minimum(np.floor(raw).astype(np.int64), counts)
    target[present & (target < 1)] = 1
    remaining = total_budget - int(target.sum())
    if remaining > 0:
        headroom = counts - target
        for class_id in np.argsort(-(raw - np.floor(raw))):
            if remaining <= 0:
                break
            if headroom[class_id] <= 0:
                continue
            target[class_id] += 1
            headroom[class_id] -= 1
            remaining -= 1
    chosen = []
    for class_id in range(n_classes):
        if target[class_id] <= 0:
            continue
        class_indices = indices[labels == class_id]
        chosen.append(subset_indices(class_indices, int(target[class_id]), seed + class_id))
    return np.sort(np.concatenate(chosen))


def resume_tag(target_dataset, args, train_used_n):
    """Deterministic stem for the FIT checkpoint (the fitted model).

    Includes every arg that changes the FIT INPUT or the inference path -- in
    particular fit_mode and subsample_samples, which v2's tag omitted (a
    cache-mode run would have silently reused a default-mode checkpoint).
    Evaluation-only settings are deliberately NOT here, so a re-run that only
    changes the test set still reuses the expensive fit."""
    return (f"{target_dataset}_mts{args.max_train_samples}_seed{args.seed}"
            f"_ne{args.n_estimators}_ipl{int(args.ignore_pretraining_limits)}"
            f"_fm{args.fit_mode}_ss{args.subsample_samples}"
            f"_sp{args.train_split.replace('+', '')}_n{train_used_n}")


def prediction_tag(fit_tag, args, n_test_eval):
    """Stem for the cached PREDICTIONS, which depend on the test set too.

    v2 and the first v3 cut keyed the prediction .npy on the fit tag alone, so
    re-running with a different --test-cap-per-class silently reloaded the
    previous test set's predictions and scored them against the new labels.
    """
    return f"{fit_tag}_tc{args.test_cap_per_class}_nt{n_test_eval}"


# ---------------------------------------------------------------------------
# Prediction / scoring helpers -- identical to v2
# ---------------------------------------------------------------------------

def predict_in_batches(model, features, batch_size):
    """Batching bounds only the TEST term of peak memory; the training context
    is re-paid on every call in default mode.  Under fit_with_cache TabPFN
    chunks test rows itself at TABPFN_MAX_BATCHED_TEST_ROWS (default 32,768),
    so a single large batch here is fine."""
    if batch_size <= 0:
        batch_size = len(features)
    n_rows = len(features)
    n_batches = (n_rows + batch_size - 1) // batch_size
    predictions = []
    for batch_number, start in enumerate(range(0, n_rows, batch_size), 1):
        stop = min(start + batch_size, n_rows)
        predictions.append(model.predict(features[start:stop]))
        if batch_number == 1 or batch_number % 10 == 0 or stop == n_rows:
            print(f"TabPFN predict batch {batch_number}/{n_batches}: rows {start:,}:{stop:,}",
                  flush=True)
    return np.concatenate(predictions)


def per_class_table(method, y_true, y_pred, class_names, tail_classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0,
    )
    rows = [{"method": method, "class": name, "precision": precision[i],
             "recall": recall[i], "f1": f1[i], "support": int(support[i])}
            for i, name in enumerate(class_names)]
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0
    tail_idx = [i for i, name in enumerate(class_names) if name in tail_classes]
    tail_f1 = float(np.mean(f1[tail_idx])) if tail_idx else float("nan")
    for label, value in [("macro_avg", macro_f1), ("weighted_avg", weighted_f1),
                         ("tail_avg", tail_f1)]:
        rows.append({"method": method, "class": label, "precision": np.nan,
                     "recall": np.nan, "f1": value, "support": int(support.sum())})
    return rows


def diagnose_train_test_drift(X, class_names, train_idx, test_idx, label_fn, chunk=200_000):
    """Per-class train/test feature drift.  Chunked so it does not materialise a
    float64 copy of a multi-million-row class slice (v2's version peaked at
    ~9 GB of host RAM on cic2018's benign class).  Same statistics.

    CAVEAT (manuscript/report/0817.md): median_abs_z is UNRELIABLE at small
    train_n -- the std is estimated from a handful of rows.  It wrongly flagged
    cic2017_full's `infiltration` (21 train rows) as drift when it was simple
    starvation.  A train-vs-test binary-probe AUC is the robust alternative.
    """
    y_train, y_test = label_fn(train_idx), label_fn(test_idx)
    rows = []
    for cid, cname in enumerate(class_names):
        tr, te = train_idx[y_train == cid], test_idx[y_test == cid]
        if len(tr) == 0 or len(te) == 0:
            rows.append({"class": cname, "train_n": len(tr), "test_n": len(te),
                         "median_abs_z": np.nan, "small_train_warning": True})
            continue

        def moments(idx):
            n = len(idx)
            s = np.zeros(X.shape[1], dtype=np.float64)
            ss = np.zeros(X.shape[1], dtype=np.float64)
            for i in range(0, n, chunk):
                blk = np.asarray(X[idx[i:i + chunk]], dtype=np.float64)
                s += blk.sum(axis=0)
                ss += (blk * blk).sum(axis=0)
            mean = s / n
            var = np.maximum(ss / n - mean * mean, 0.0)
            return mean, np.sqrt(var)

        mean_tr, std_tr = moments(tr)
        mean_te, _ = moments(te)
        z = np.abs(mean_tr - mean_te) / (std_tr + 1e-6)
        rows.append({"class": cname, "train_n": len(tr), "test_n": len(te),
                     "median_abs_z": float(np.median(z)),
                     "small_train_warning": bool(len(tr) < 500)})
    return pd.DataFrame(rows).sort_values("median_abs_z", ascending=False, na_position="first")


# ---------------------------------------------------------------------------
# CLI shared by all three experiments
# ---------------------------------------------------------------------------

def base_parser(description):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-dataset", default="cic2018",
                   choices=list(build_dataset_config(".").keys()))
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--data", default=None,
                   help="Override the dataset pickle; defaults to the target's canonical pkl.")
    p.add_argument("--test-cap-per-class", type=int, default=100_000,
                   help="Per-class cap on the evaluation set; 0 = use all test rows.")
    p.add_argument("--test-batch-size", type=int, default=0,
                   help="0 = one single batch over the whole eval set. See the module "
                        "docstring for the per-dataset table; the predicted peak is "
                        "printed before the run starts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-path",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"))
    p.add_argument("--ignore-pretraining-limits", action="store_true",
                   help="Bypass the checkpoint's MAX_NUMBER_OF_SAMPLES=1,000,000 guard. "
                        "REQUIRED whenever the context exceeds 1M rows. Without it the "
                        "script re-caps UNIFORMLY (not stratified), which starves tail classes.")
    p.add_argument("--max-train-samples", type=int, default=1_000_000,
                   help="Stratified (class-ratio-preserving) train-row budget. "
                        "Negative = no cap (use the full train pool).")
    p.add_argument("--train-split", default="train", choices=["train", "train+val"],
                   help="'train' = the 60%% slice only, exactly as every previous run. "
                        "'train+val' = fold the otherwise-discarded 20%% validation slice "
                        "into the training pool, i.e. an 80/20 split. Nothing in this "
                        "pipeline uses val (no early stopping, no calibration), so under "
                        "'train' that 20%% is simply thrown away. On cic2017_full that "
                        "cost macro F1 0.717 -> 0.804 and `bot` F1 0.000 -> 0.800, because "
                        "the 60%% cut landed just BEFORE a behavioural phase change and all "
                        "post-change exposure sat in the discarded slice "
                        "(manuscript/report/0817.md). CHANGES RESULTS -- log it as its own run.")
    p.add_argument("--n-estimators", type=int, default=4,
                   help="TabPFN test-time ensemble members. 4 = the reference paper's value. "
                        "This does NOT multiply GPU memory (estimators run sequentially) but "
                        "does multiply wall-clock ~linearly.")
    p.add_argument("--fit-mode", default="fit_preprocessors",
                   choices=["fit_preprocessors", "fit_with_cache", "low_memory"],
                   help="fit_with_cache builds the context K/V during fit() so predict() "
                        "never re-uploads the train rows. Lower predict peak (21.1 -> 16.6 GiB "
                        "measured) and far cheaper repeated predicts, BUT the build costs "
                        "~20,674 B/train row vs 18,530 for a default predict, so it LOWERS "
                        "the maximum context size. The saved .tabpfn_fit also grows to "
                        "~4 KB x rows x n_estimators (16 GB at 1M x 4).")
    p.add_argument("--keep-cache-on-device", action="store_true",
                   help="Only with fit_with_cache. Off by default (caches on CPU, one moved to "
                        "GPU per estimator). On keeps them all resident: measured 6.367 vs "
                        "4.046 GiB peak at 200k rows for a 2%% speedup.")
    p.add_argument("--subsample-samples", type=int, default=0,
                   help="TabPFN's SUBSAMPLE_SAMPLES: per-estimator in-context row cap. "
                        "0 = off. With a positive k each estimator draws its OWN DISJOINT "
                        "stratified k rows, so n_estimators*k rows collectively enter the "
                        "ensemble. NOTE: stratification is PROPORTIONAL -- tail classes keep "
                        "their natural rate and are NOT protected.")
    p.add_argument("--skip-tabpfn", action="store_true")
    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--xgb-n-estimators", type=int, default=300)
    p.add_argument("--xgb-max-depth", type=int, default=8)
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    p.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    p.add_argument("--out-root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    p.add_argument("--models-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "saved_models"),
        help="Fitted models land here. Under fit_with_cache these are LARGE "
             "(~4 KB x rows x n_estimators); keep them off Drive.")
    p.add_argument("--resume-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resume"))
    p.add_argument("--force-refit", action="store_true")
    p.add_argument("--no-save-models", action="store_true",
                   help="Skip the final copy of the fitted models from --resume-dir into "
                        "--models-dir. Under fit_with_cache each copy is ~4 KB x rows x "
                        "n_estimators (about 16 GB at 1M x 4), so keeping both is 32 GB per "
                        "run. With --resume-dir on durable storage the resume checkpoint IS "
                        "the saved model, and this flag avoids duplicating it.")
    return p


# ---------------------------------------------------------------------------
# The shared run body
# ---------------------------------------------------------------------------

def run_experiment(args, experiment_name, out_tag_suffix):
    """Load -> split -> budget -> (XGBoost) -> (TabPFN) -> artifacts.

    Every experiment script calls this; they differ only in the argparse
    defaults they set and in which model they skip.
    """
    cfg = build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = experiment_name
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)

    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    print(f"target={args.target_dataset}  classes ({n_classes}): {class_names}")
    print(f"split rows: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    if args.train_split == "train+val":
        # 80/20: the val slice is otherwise computed and discarded (nothing here uses
        # it -- no early stopping, no calibration). See manuscript/report/0817.md.
        train_idx = np.sort(np.concatenate([train_idx, val_idx]))
        y_train_all = label_fn(train_idx)
        print(f"--train-split train+val: train pool {len(train_idx):,} "
              f"(60% + 20%);  test unchanged at {len(test_idx):,}")
    else:
        print(f"--train-split train: the {len(val_idx):,}-row val slice is DISCARDED "
              "(pass --train-split train+val to use it)")

    # ---- evaluation set ----
    test_eval_idx = cap_per_class(test_idx, y_test_all, n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_test_eval = label_fn(test_eval_idx)
    print(f"test eval rows: {len(test_eval_idx):,} (cap_per_class={args.test_cap_per_class})")
    X_test_eval = np.nan_to_num(np.asarray(X[test_eval_idx], dtype=np.float32))

    # ---- drift diagnostic (chunked; see the caveat in its docstring) ----
    print("\n--- pre-fit diagnostic: train/test feature drift per class ---")
    drift_df = diagnose_train_test_drift(X, class_names, train_idx, test_idx, label_fn)
    print(drift_df.to_string(index=False))
    print("High median_abs_z means the chronological split put a different-looking population of "
          "that class in train vs test. IGNORE the value when small_train_warning is True.\n")

    # ---- train budget ----
    train_used_idx = train_idx
    cap_policy = "none"
    if args.max_train_samples > 0 and args.max_train_samples < len(train_used_idx):
        train_used_idx = stratified_subset(train_used_idx, label_fn(train_used_idx),
                                           n_classes, args.max_train_samples, args.seed + 850)
        cap_policy = "stratified_ratio_preserving"

    y_train_used = label_fn(train_used_idx)
    train_used_counts = {n: int((y_train_used == i).sum()) for i, n in enumerate(class_names)}
    print(f"train selection: used={len(train_used_idx):,} of pool {len(train_idx):,} "
          f"({100 * len(train_used_idx) / len(train_idx):.1f}%)  cap_policy={cap_policy}")
    print(f"  per class: {train_used_counts}")
    split_audit = pd.concat([split_audit, pd.DataFrame(
        [{"split": "train_used", "class": n, "count": c} for n, c in train_used_counts.items()]
    )], ignore_index=True, sort=False)

    X_train_used = np.nan_to_num(np.asarray(X[train_used_idx], dtype=np.float32))
    # X_train_used / X_test_eval are independent copies (fancy-index + nan_to_num),
    # so the source pickle can go. On the uncapped suite that is 12.3 GB of host RAM
    # held for the whole run in v2, of which only the target dataset was ever read.
    del X
    _PICKLE_CACHE.clear()
    gc.collect()
    report_memory_plan(len(train_used_idx), len(test_eval_idx), args)

    all_rows, timings = [], {}
    tag = resume_tag(args.target_dataset, args, len(train_used_idx))

    # ---- XGBoost ----
    if not args.skip_xgboost:
        xgb_ckpt = os.path.join(args.resume_dir, f"{tag}_xgboost.json")
        t0 = time.time()
        if not args.force_refit and os.path.exists(xgb_ckpt):
            print(f"Resuming XGBoost from {xgb_ckpt}")
            booster = xgb.XGBClassifier()
            booster.load_model(xgb_ckpt)
            xgb_fit_s = None
        else:
            booster = xgb.XGBClassifier(
                n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                min_child_weight=args.xgb_min_child_weight, reg_lambda=args.xgb_reg_lambda,
                objective="multi:softprob", num_class=n_classes,
                eval_metric="mlogloss", n_jobs=-1, random_state=args.seed)
            print(f"XGBoost fitting on {len(X_train_used):,} rows ...", flush=True)
            booster.fit(X_train_used, y_train_used)
            xgb_fit_s = time.time() - t0
            booster.save_model(xgb_ckpt)
            print(f"XGBoost fit done in {xgb_fit_s:.1f}s -> {xgb_ckpt}")
        t1 = time.time()
        y_pred_xgb = booster.predict(X_test_eval)
        timings["xgboost_fit_seconds"] = xgb_fit_s
        timings["xgboost_predict_seconds"] = time.time() - t1
        all_rows.extend(per_class_table("xgboost", y_test_eval, y_pred_xgb,
                                        class_names, tail_classes))

    # ---- TabPFN ----
    if not args.skip_tabpfn:
        clf = TabPFNClassifier(
            device=args.device,
            model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            # int(): np.int64 raises ValueError and np.float64 is misrouted into the
            # 0<x<1 fraction branch (preprocessing/ensemble.py:502-510).
            inference_config={"SUBSAMPLE_SAMPLES": int(args.subsample_samples) or None},
            random_state=args.seed,
            n_estimators=args.n_estimators,
            auto_scale_n_estimators=args.auto_scale_n_estimators,
            fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device,
        )
        max_samples = clf.get_inference_config().MAX_NUMBER_OF_SAMPLES
        print(f"TabPFN: context={len(train_used_idx):,}  checkpoint_max={max_samples:,}  "
              f"fit_mode={args.fit_mode}  subsample={args.subsample_samples or None}  "
              f"n_estimators={args.n_estimators}")
        if not args.ignore_pretraining_limits and len(train_used_idx) > max_samples:
            raise SystemExit(
                f"context {len(train_used_idx):,} exceeds the checkpoint limit "
                f"{max_samples:,}. Pass --ignore-pretraining-limits (and note that "
                "without it the rows would be re-capped UNIFORMLY, not stratified).")
        if args.subsample_samples and args.subsample_samples < n_classes:
            raise SystemExit(f"--subsample-samples must be >= n_classes ({n_classes}).")

        tabpfn_ckpt = os.path.join(args.resume_dir, f"{tag}_tabpfn.tabpfn_fit")
        pred_ckpt = os.path.join(
            args.resume_dir,
            f"{prediction_tag(tag, args, len(test_eval_idx))}_tabpfn_pred.npy")

        if not args.force_refit and os.path.exists(tabpfn_ckpt):
            print(f"Resuming TabPFN from {tabpfn_ckpt}")
            clf = load_fitted_tabpfn_model(tabpfn_ckpt, device=args.device)
            fit_s = None
        else:
            t0 = time.time()
            clf.fit(X_train_used, y_train_used)
            fit_s = time.time() - t0
            save_fitted_tabpfn_model(clf, tabpfn_ckpt)
            sz = os.path.getsize(tabpfn_ckpt) / 1e9
            print(f"TabPFN fit done in {fit_s:.1f}s -> {tabpfn_ckpt} ({sz:.1f} GB)")

        if not args.force_refit and os.path.exists(pred_ckpt):
            print(f"Resuming TabPFN predictions from {pred_ckpt}")
            y_pred_tabpfn = np.load(pred_ckpt)
            pred_s = None
        else:
            t0 = time.time()
            y_pred_tabpfn = predict_in_batches(clf, X_test_eval, args.test_batch_size)
            pred_s = time.time() - t0
            np.save(pred_ckpt, y_pred_tabpfn)
            print(f"TabPFN predict done in {pred_s:.1f}s")

        timings["tabpfn_fit_seconds"] = fit_s
        timings["tabpfn_predict_seconds"] = pred_s
        timings["tabpfn_checkpoint_max_samples"] = int(max_samples)
        all_rows.extend(per_class_table("tabpfn_v3", y_test_eval, y_pred_tabpfn,
                                        class_names, tail_classes))

    # ---- artifacts ----
    tb = args.test_batch_size if args.test_batch_size > 0 else len(test_eval_idx)
    timings.update({
        "experiment": experiment_name,
        "target_dataset": args.target_dataset,
        "train_pool_rows": int(len(train_idx)),
        "train_rows_used": int(len(train_used_idx)),
        "train_pool_coverage_pct": round(100 * len(train_used_idx) / len(train_idx), 2),
        "train_cap_policy": cap_policy,
        "train_split": args.train_split,
        "subsample_samples": args.subsample_samples or None,
        "n_estimators": args.n_estimators,
        "ensemble_context_rows_total": (args.subsample_samples * args.n_estimators
                                        if args.subsample_samples else None),
        "fit_mode": args.fit_mode,
        "validation_rows_unused": int(len(val_idx)),
        "test_pool_rows": int(len(test_idx)),
        "test_evaluation_rows": int(len(test_eval_idx)),
        "test_batch_size": int(tb),
        "test_batches": int(math.ceil(len(test_eval_idx) / tb)),
        "predicted_peak_gib": round(estimate_peak_gib(len(train_used_idx), tb, args.fit_mode)[0], 2),
        "resume_tag": tag,
    })

    table = pd.DataFrame(all_rows)
    if len(table):
        summary = table[table["class"].isin(["macro_avg", "weighted_avg", "tail_avg"])]
        print("\n=== summary (macro / weighted / tail F1) ===")
        print(summary.pivot(index="class", columns="method", values="f1").to_string())

    ts = time.strftime("%Y%m%d_%H%M%S")
    split_tag = "" if args.train_split == "train" else "_82"
    out_dir = os.path.join(args.out_root,
                           f"{ts}_{cfg[args.target_dataset]['out_tag']}_{out_tag_suffix}{split_tag}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "split_audit.csv"), index=False)
    drift_df.to_csv(os.path.join(out_dir, "train_test_drift_diagnostic.csv"), index=False)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([timings], f, indent=2)
    print(f"\nWrote {out_dir}")

    if args.no_save_models:
        print(f"--no-save-models: fitted models stay in {args.resume_dir} only "
              "(that directory IS the saved-model record for this run).")
        return out_dir

    model_tag = os.path.basename(out_dir)
    for src, suffix in ((os.path.join(args.resume_dir, f"{tag}_xgboost.json"), "xgboost.json"),
                        (os.path.join(args.resume_dir, f"{tag}_tabpfn.tabpfn_fit"), "tabpfn.tabpfn_fit")):
        if os.path.exists(src):
            dst = os.path.join(args.models_dir, f"{model_tag}_{suffix}")
            shutil.copy(src, dst)
            print(f"Saved {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")
    return out_dir

#!/usr/bin/env python3
"""Standalone TabPFN-v3 vs XGBoost multiclass sanity test -- v2 of
nfv3_cic2018_multiclass_test.py (that script is left untouched; this is a
separate file, not an in-place edit).

Same "lives inside tabpfn/ on purpose, not part of the src/ frozen-experiment
convention" posture as its predecessor -- this is a quick standalone check of
the TabPFN library, not the paper's experiment record. Meant to be runnable
both as a plain server-side script (`python nfv3_multiclass_test_v2.py ...`)
and, unchanged, as a Colab shell command
(`!python nfv3_multiclass_test_v2.py --data-dir /content/drive/.../data
--out-root /content/drive/.../results --models-dir /content/drive/.../saved_models
...`) -- no Colab-specific code lives here; mount Drive and point --data-dir /
--out-root / --models-dir at it yourself.

Differences from nfv3_cic2018_multiclass_test.py (2026-08-13):

1. **`--protocol reproduce` removed.** This project only uses the
   chronological-split + stratified-train-budget policy now (what used to be
   `--protocol sota`). `random_stratified_split` and the fixed
   3,000-per-class cap path are gone; `--max-train-samples` is the only
   train-budget knob left.
2. **`--target-dataset` gained `bot_iot`/`ton_iot` (+ `_capped` variants of
   all three NF-v3 suite datasets).** The suite pkls bundle four NetFlow-v3
   datasets under one schema (`dataset_names` in {cse_cic_ids2018, bot_iot,
   ton_iot, unsw_nb15}, unsw_nb15 not wired up here); `load_cic2018` is now
   `load_nfv3_suite_subset(args, dataset_name)` pinned per target, matching
   the pattern already used for `cic2017_full`. `data/nfv3_energy_suite_uncapped_scenarios.pkl`
   has the full, uncapped rows for cic2018/bot_iot/ton_iot (20.1M / 16.9M /
   27.5M respectively -- none of these fit a single GPU's train pool
   uncapped, see the GPU-memory note below). `data/nfv3_energy_suite_cic2018_scenarios.pkl`
   (despite the name) holds a per-family-capped slice of *all four* datasets
   -- built by scripts/preprocess_nfv3_cic2018_scenarios.py, which itself
   reads scripts/preprocess_nfv3_energy_suite.py's capped
   `data/nfv3_energy_suite.pkl` as its `--base-suite` and copies bot_iot/
   ton_iot/unsw_nb15 through unchanged, only re-reading NF-CICIDS2018-v3.csv
   fresh for cic2018 (to recover raw attack-scenario names that the base
   suite had already collapsed to parent families). One consequence: cic2018
   gets real per-scenario chronological splitting in that pkl; bot_iot/
   ton_iot/unsw_nb15 only get family-level splitting there (their
   `attack_scenarios` field equals `families`).
3. **Saves fitted models for reuse.** XGBoost via `booster.save_model()`;
   TabPFN via `tabpfn.model_loading.save_fitted_tabpfn_model()` (fitted
   state only, not the foundation weights) + `load_fitted_tabpfn_model()` to
   reload and `.predict()` immediately without re-fitting. See
   `--models-dir`.
4. **`--max-train-samples 0` means "auto-suggest from detected GPU memory"**
   (previously 0 meant "no cap" outright). The suggestion uses an empirical
   fit from a 5K-1M row sweep on an RTX 4090 (24GB): peak CUDA memory during
   fit()+predict() grows linearly, `mem_GB ~= 0.23 + rows/57,900` (R^2 ~ 1,
   confirmed to hold as a real ceiling -- every point from 1.5M rows up
   OOM'd on that 24GB card, no plateau). cic2018's full 12,069,313-row train
   pool alone would need ~209GB -- not achievable on any single GPU
   (including a 48GB A100), so target the `_capped` datasets or a
   deliberately bounded `--max-train-samples` instead of trying to remove
   this cap outright.
5. **Pre-fit train/test drift diagnostic always runs and gets saved.**
   `diagnose_train_test_drift()` flags classes whose train/test feature
   distributions differ a lot under the chronological split -- this is what
   explains cic2017_full's `bot`/`infiltration` scoring exactly 0 P/R/F1 for
   *both* TabPFN and XGBoost in earlier sota runs (real distribution shift
   from splitting a short, multi-phase attack capture chronologically, not a
   code bug -- see manuscript/report/0813.md). Low `train_n` alone (e.g.
   cic2018's `web_attacks` at a small `--max-train-samples`) is a separate,
   simpler starvation cause and shows up as a small `train_n` rather than a
   high `median_abs_z`.

Usage:
  python nfv3_multiclass_test_v2.py --target-dataset cic2017_full
  python nfv3_multiclass_test_v2.py --target-dataset cic2018_capped
  python nfv3_multiclass_test_v2.py --target-dataset bot_iot_capped
  python nfv3_multiclass_test_v2.py --target-dataset ton_iot_capped
  python nfv3_multiclass_test_v2.py --target-dataset cic2018 --max-train-samples 2000000 --ignore-pretraining-limits
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from exp_utils import load_pickle, scenario_chronological_split, subset_indices, labels_for  # noqa: E402

from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset loaders -- chronological split only, no random-split/reproduce path
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

    y_train_all = label_fn(train_idx)
    y_test_all = label_fn(test_idx)
    return X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn


def load_cic2018(args):
    return load_nfv3_suite_subset(args, "cse_cic_ids2018")


def load_bot_iot(args):
    return load_nfv3_suite_subset(args, "bot_iot")


def load_ton_iot(args):
    return load_nfv3_suite_subset(args, "ton_iot")


def load_cic2017_full(args):
    """All 15 CIC-IDS2017 classes, no Mon-Thu/Friday exclusion; chronological
    split on the synthetic (file_order, in-file row order) time_proxy -- see
    scripts/preprocess_cic2017_full_raw.py's docstring."""
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

    y_train_all = label_fn(train_idx)
    y_test_all = label_fn(test_idx)
    return X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn


def build_dataset_config(data_dir):
    return {
        "cic2018": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_uncapped_scenarios.pkl"),
            "loader": load_cic2018,
            "tail_classes": ["bot", "infiltration", "web_attacks"],
            "out_tag": "nfv3_cic2018_tabpfn_multiclass",
        },
        "cic2018_capped": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_cic2018_scenarios.pkl"),
            "loader": load_cic2018,
            "tail_classes": ["bot", "infiltration", "web_attacks"],
            "out_tag": "nfv3_cic2018capped_tabpfn_multiclass",
        },
        "bot_iot": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_uncapped_scenarios.pkl"),
            "loader": load_bot_iot,
            "tail_classes": ["theft"],
            "out_tag": "nfv3_botiot_tabpfn_multiclass",
        },
        "bot_iot_capped": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_cic2018_scenarios.pkl"),
            "loader": load_bot_iot,
            "tail_classes": ["theft"],
            "out_tag": "nfv3_botiotcapped_tabpfn_multiclass",
        },
        "ton_iot": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_uncapped_scenarios.pkl"),
            "loader": load_ton_iot,
            "tail_classes": ["mitm", "ransomware"],
            "out_tag": "nfv3_toniot_tabpfn_multiclass",
        },
        "ton_iot_capped": {
            "default_data": os.path.join(data_dir, "nfv3_energy_suite_cic2018_scenarios.pkl"),
            "loader": load_ton_iot,
            "tail_classes": ["mitm", "ransomware"],
            "out_tag": "nfv3_toniotcapped_tabpfn_multiclass",
        },
        "cic2017_full": {
            "default_data": os.path.join(data_dir, "cic2017_full_raw.pkl"),
            "loader": load_cic2017_full,
            "tail_classes": [
                "heartbleed", "web-attack-sql-injection", "infiltration",
                "web-attack-xss", "web-attack-brute-force",
            ],
            "out_tag": "cic2017_full_tabpfn_multiclass",
        },
    }


# ---------------------------------------------------------------------------
# Train-budget / test-cap helpers (unchanged from v1)
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
    """Largest-remainder proportional subsample of `indices` down to
    `total_budget` rows, preserving each class's share and guaranteeing at
    least one row for every present class."""
    indices = np.asarray(indices, dtype=np.int64)
    if total_budget <= 0 or total_budget >= len(indices):
        return np.sort(indices)
    counts = np.bincount(labels, minlength=n_classes)
    present = counts > 0
    n_present = int(present.sum())
    if total_budget < n_present:
        raise ValueError(
            f"--max-train-samples={total_budget} is smaller than the number "
            f"of present classes ({n_present}); raise it to at least that."
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


# Empirical fit from a 5K-1M row sweep on an RTX 4090 (24GB) -- see module
# docstring point 4.
ROWS_PER_GB = 57_900
BASELINE_GB = 1.0
SAFETY_MARGIN = 0.8


def suggest_max_train_samples():
    if not torch.cuda.is_available():
        print("WARNING: no GPU detected -- falling back to a small max_train_samples; "
              "this estimate is meaningless on CPU.")
        return 50_000
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    suggested = max(1, int((gpu_gb - BASELINE_GB) * ROWS_PER_GB * SAFETY_MARGIN))
    print(f"Detected GPU memory: {gpu_gb:.1f} GB -> suggested max_train_samples ~= {suggested:,} "
          "(extrapolated from a 24GB-card sweep, not a guarantee -- back off if you still hit a CUDA OOM)")
    return suggested


def resume_tag(target_dataset, args, train_used_n):
    """Deterministic filename stem for a given config's fit/predict outputs
    (RESUME_DIR), so a killed/restarted run finds and reuses whatever it
    already finished instead of redoing a multi-hour TabPFN fit. Only
    includes args that affect the FIT INPUT, not evaluation-only settings
    like --test-cap-per-class."""
    return (f"{target_dataset}_mts{args.max_train_samples}_seed{args.seed}"
            f"_ne{args.n_estimators}_ipl{int(args.ignore_pretraining_limits)}_n{train_used_n}")


# ---------------------------------------------------------------------------
# Prediction / scoring helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def predict_in_batches(model, features, batch_size):
    if batch_size <= 0:
        raise ValueError("--test-batch-size must be positive")
    n_rows = len(features)
    n_batches = (n_rows + batch_size - 1) // batch_size
    predictions = []
    for batch_number, start in enumerate(range(0, n_rows, batch_size), 1):
        stop = min(start + batch_size, n_rows)
        predictions.append(model.predict(features[start:stop]))
        if batch_number == 1 or batch_number % 10 == 0 or stop == n_rows:
            print(
                "TabPFN predict batch "
                f"{batch_number}/{n_batches}: rows {start:,}:{stop:,}"
            )
    return np.concatenate(predictions)


def per_class_table(method, y_true, y_pred, class_names, tail_classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0,
    )
    rows = [{
        "method": method, "class": name,
        "precision": precision[i], "recall": recall[i], "f1": f1[i],
        "support": int(support[i]),
    } for i, name in enumerate(class_names)]

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0
    tail_idx = [i for i, name in enumerate(class_names) if name in tail_classes]
    tail_f1 = float(np.mean(f1[tail_idx])) if tail_idx else float("nan")

    for label, value in [
        ("macro_avg", macro_f1), ("weighted_avg", weighted_f1), ("tail_avg", tail_f1),
    ]:
        rows.append({
            "method": method, "class": label,
            "precision": np.nan, "recall": np.nan, "f1": value,
            "support": int(support.sum()),
        })
    return rows


def diagnose_train_test_drift(X, class_names, train_idx, test_idx, label_fn):
    """Flags classes whose train/test feature distributions differ a lot
    under the chronological split -- real distribution shift shows up as a
    high median_abs_z; low train_n alone is a separate, simpler starvation
    cause. See manuscript/report/0813.md."""
    y_train = label_fn(train_idx)
    y_test = label_fn(test_idx)
    rows = []
    for cid, cname in enumerate(class_names):
        tr = train_idx[y_train == cid]
        te = test_idx[y_test == cid]
        if len(tr) == 0 or len(te) == 0:
            rows.append({"class": cname, "train_n": len(tr), "test_n": len(te), "median_abs_z": np.nan})
            continue
        Xtr = np.asarray(X[tr], dtype=np.float64)
        Xte = np.asarray(X[te], dtype=np.float64)
        std_tr = Xtr.std(axis=0) + 1e-6
        z = np.abs(Xtr.mean(axis=0) - Xte.mean(axis=0)) / std_tr
        rows.append({
            "class": cname, "train_n": len(tr), "test_n": len(te),
            "median_abs_z": float(np.median(z)),
        })
    return pd.DataFrame(rows).sort_values("median_abs_z", ascending=False, na_position="first")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-dataset",
        choices=["cic2018", "cic2018_capped", "bot_iot", "bot_iot_capped",
                 "ton_iot", "ton_iot_capped", "cic2017_full"],
        default="cic2018_capped",
    )
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                         help="Base dir DATASET_CONFIG default paths resolve against.")
    parser.add_argument("--data", default=None,
                         help="Override the dataset pickle path; defaults to --target-dataset's canonical pkl under --data-dir.")
    parser.add_argument("--test-cap-per-class", type=int, default=100_000,
                         help="Per-class cap on the evaluation set; 0 = use all test rows.")
    parser.add_argument("--test-batch-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.path.dirname(__file__), "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"),
        help="Downloaded from https://huggingface.co/Prior-Labs/tabpfn_3/tree/main",
    )
    parser.add_argument(
        "--ignore-pretraining-limits", action="store_true",
        help="Use the full stratified train selection even past the checkpoint's own "
             "declared row limit. Without this, rows beyond that limit are re-capped "
             "(uniformly, not stratified) as a final safety net -- watch the warning "
             "this script prints if that's about to happen.",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=0,
        help="Stratified (class-ratio-preserving) train-row budget. 0 = auto-suggest "
             "from detected GPU memory (see suggest_max_train_samples); pass a positive "
             "value to pin it, or a negative value for 'use the full train pool, no cap'.",
    )
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=8)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--out-root", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--models-dir", default=os.path.join(os.path.dirname(__file__), "saved_models"),
                         help="Where fitted XGBoost/TabPFN models get saved for later reuse.")
    parser.add_argument("--resume-dir", default=os.path.join(os.path.dirname(__file__), "resume"),
                         help="Where mid-run checkpoints (TabPFN fit, TabPFN predictions, XGBoost) "
                              "get saved, keyed by config -- a re-run with the same effective config "
                              "finds these and skips straight past the step, so a killed/restarted "
                              "multi-hour run doesn't redo the (opaque, unresumable-mid-call) TabPFN "
                              "fit() from scratch.")
    parser.add_argument("--force-refit", action="store_true",
                         help="Ignore any existing --resume-dir checkpoint and redo fit/predict from scratch.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_config = build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = dataset_config[args.target_dataset]["default_data"]
    if args.max_train_samples == 0:
        args.max_train_samples = suggest_max_train_samples()
    args.auto_scale_n_estimators = False
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}")

    tail_classes = dataset_config[args.target_dataset]["tail_classes"]
    loader = dataset_config[args.target_dataset]["loader"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = loader(args)
    print(f"target_dataset={args.target_dataset} classes ({len(class_names)}): {class_names}")

    train_counts = {
        name: int((y_train_all == i).sum()) for i, name in enumerate(class_names)
    }
    print(f"full train pool per class: {train_counts}")
    print(
        "split rows: "
        f"train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}"
    )

    test_eval_idx = cap_per_class(
        test_idx, y_test_all, len(class_names), args.test_cap_per_class, args.seed + 900,
    )
    y_test_eval = label_fn(test_eval_idx)
    print(f"test eval rows: {len(test_eval_idx)} (cap_per_class={args.test_cap_per_class})")
    X_test_eval = np.nan_to_num(np.asarray(X[test_eval_idx], dtype=np.float32))

    print("\n--- pre-fit diagnostic: train/test feature drift per class ---")
    drift_df = diagnose_train_test_drift(X, class_names, train_idx, test_idx, label_fn)
    print(drift_df.to_string(index=False))
    print("High median_abs_z (roughly >0.2-0.3 vs ~0.04-0.1 for well-behaved classes) means the")
    print("chronological split put a genuinely different-looking population of that class in train vs")
    print("test -- expect low recall regardless of model. Low train_n is a separate, simpler cause.\n")

    clf = TabPFNClassifier(
        device=args.device,
        model_path=args.model_path,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        inference_config={"SUBSAMPLE_SAMPLES": None},
        random_state=args.seed,
        n_estimators=args.n_estimators,
        auto_scale_n_estimators=args.auto_scale_n_estimators,
    )
    inference_config = clf.get_inference_config()
    max_samples = inference_config.MAX_NUMBER_OF_SAMPLES
    print(
        "TabPFN rows: "
        f"full_train={len(train_idx):,} checkpoint_max={max_samples:,} "
        "internal_subsample=None "
        f"n_estimators={args.n_estimators} auto_scale_n_estimators={args.auto_scale_n_estimators}"
    )
    if not args.ignore_pretraining_limits and (
        args.max_train_samples <= 0 or args.max_train_samples > max_samples
    ) and len(train_idx) > max_samples:
        print(
            f"WARNING: max_train_samples={args.max_train_samples} will be silently re-capped to "
            f"the checkpoint's {max_samples:,}-row limit (uniformly, not stratified) because "
            "--ignore-pretraining-limits was not passed. Add that flag if you actually want the "
            "larger stratified selection to stick."
        )

    train_used_idx = train_idx
    cap_policy_applied = "none"
    if args.max_train_samples > 0 and args.max_train_samples < len(train_used_idx):
        pre_cap_labels = label_fn(train_used_idx)
        train_used_idx = stratified_subset(
            train_used_idx, pre_cap_labels, len(class_names),
            args.max_train_samples, args.seed + 850,
        )
        cap_policy_applied = "stratified_ratio_preserving"

    if not args.ignore_pretraining_limits and len(train_used_idx) > max_samples:
        train_used_idx = subset_indices(train_used_idx, max_samples, args.seed + 800)

    train_was_capped = len(train_used_idx) < len(train_idx)
    y_train_used = label_fn(train_used_idx)
    train_used_counts = {
        name: int((y_train_used == i).sum())
        for i, name in enumerate(class_names)
    }
    print(
        "TabPFN/XGBoost train selection: "
        f"used={len(train_used_idx):,} capped={train_was_capped} "
        f"cap_policy_applied={cap_policy_applied} "
        f"max_train_samples={args.max_train_samples} "
        f"per_class={train_used_counts}"
    )
    train_used_audit = pd.DataFrame([
        {"split": "train_used", "class": name, "count": count}
        for name, count in train_used_counts.items()
    ])
    split_audit = pd.concat([split_audit, train_used_audit], ignore_index=True, sort=False)

    X_train_used = np.nan_to_num(
        np.asarray(X[train_used_idx], dtype=np.float32)
    )

    tag = resume_tag(args.target_dataset, args, len(train_used_idx))
    tabpfn_ckpt = os.path.join(args.resume_dir, f"{tag}_tabpfn.tabpfn_fit")
    tabpfn_pred_ckpt = os.path.join(args.resume_dir, f"{tag}_tabpfn_pred.npy")
    xgb_ckpt = os.path.join(args.resume_dir, f"{tag}_xgboost.json")

    all_rows = []
    timings = []

    # ---- TabPFN: resume fit if checkpointed, else fit + checkpoint immediately ----
    if not args.force_refit and os.path.exists(tabpfn_ckpt):
        print(f"Resuming TabPFN from checkpoint: {tabpfn_ckpt} (skipping the multi-hour fit())")
        clf = load_fitted_tabpfn_model(tabpfn_ckpt, device=args.device)
        tabpfn_fit_seconds = None
    else:
        t0 = time.time()
        clf.fit(X_train_used, y_train_used)
        tabpfn_fit_seconds = time.time() - t0
        save_fitted_tabpfn_model(clf, tabpfn_ckpt)
        print(f"TabPFN fit done in {tabpfn_fit_seconds:.1f}s -- checkpointed to {tabpfn_ckpt} "
              "(a resumed re-run will skip straight past this step)")

    if not args.force_refit and os.path.exists(tabpfn_pred_ckpt):
        print(f"Resuming TabPFN predictions from checkpoint: {tabpfn_pred_ckpt}")
        y_pred_tabpfn = np.load(tabpfn_pred_ckpt)
        tabpfn_predict_seconds = None
    else:
        t0 = time.time()
        y_pred_tabpfn = predict_in_batches(clf, X_test_eval, args.test_batch_size)
        tabpfn_predict_seconds = time.time() - t0
        np.save(tabpfn_pred_ckpt, y_pred_tabpfn)
        print(f"TabPFN predict done in {tabpfn_predict_seconds:.1f}s -- checkpointed to {tabpfn_pred_ckpt}")
    all_rows.extend(
        per_class_table("tabpfn_v3", y_test_eval, y_pred_tabpfn, class_names, tail_classes)
    )

    # ---- XGBoost: same resume pattern (fit is fast, but free to checkpoint too) ----
    if not args.force_refit and os.path.exists(xgb_ckpt):
        print(f"Resuming XGBoost from checkpoint: {xgb_ckpt}")
        booster = xgb.XGBClassifier()
        booster.load_model(xgb_ckpt)
    else:
        booster = xgb.XGBClassifier(
            n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight, reg_lambda=args.xgb_reg_lambda,
            objective="multi:softprob", num_class=len(class_names),
            eval_metric="mlogloss", n_jobs=-1, random_state=args.seed,
        )
        booster.fit(X_train_used, y_train_used)
        booster.save_model(xgb_ckpt)
        print(f"XGBoost fit done -- checkpointed to {xgb_ckpt}")
    y_pred_xgb = booster.predict(X_test_eval)
    all_rows.extend(
        per_class_table("xgboost", y_test_eval, y_pred_xgb, class_names, tail_classes)
    )

    timings.append({
        "target_dataset": args.target_dataset,
        "train_pool_rows": len(train_idx),
        "train_rows": len(train_used_idx),
        "train_was_capped": train_was_capped,
        "max_train_samples": args.max_train_samples,
        "train_cap_policy_applied": cap_policy_applied,
        "validation_rows_unused": len(val_idx),
        "test_pool_rows": len(test_idx),
        "test_evaluation_rows": len(test_eval_idx),
        "test_cap_per_class": args.test_cap_per_class,
        "tabpfn_test_batch_size": args.test_batch_size,
        "tabpfn_checkpoint_max_samples": max_samples,
        "tabpfn_n_estimators": args.n_estimators,
        "tabpfn_fit_seconds": tabpfn_fit_seconds,
        "tabpfn_predict_seconds": tabpfn_predict_seconds,
        "resume_tag": tag,
    })

    table = pd.DataFrame(all_rows)
    summary = table[table["class"].isin(["macro_avg", "weighted_avg", "tail_avg"])]
    print("\n=== summary (macro / weighted / tail F1) ===")
    print(summary.pivot(index="class", columns="method", values="f1").to_string())

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_tag = dataset_config[args.target_dataset]["out_tag"]
    out_dir = os.path.join(args.out_root, f"{ts}_{out_tag}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "split_audit.csv"), index=False)
    drift_df.to_csv(os.path.join(out_dir, "train_test_drift_diagnostic.csv"), index=False)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as handle:
        json.dump(timings, handle, indent=2)
    print(f"\nWrote {out_dir}")

    # ---- copy the (already-checkpointed) fitted models into the timestamped saved_models/ record ----
    model_tag = f"{ts}_{out_tag}"
    xgb_path = os.path.join(args.models_dir, f"{model_tag}_xgboost.json")
    shutil.copy(xgb_ckpt, xgb_path)
    print(f"Saved XGBoost model: {xgb_path}")

    tabpfn_path = os.path.join(args.models_dir, f"{model_tag}_tabpfn.tabpfn_fit")
    shutil.copy(tabpfn_ckpt, tabpfn_path)
    print(f"Saved fitted TabPFN state: {tabpfn_path}")
    print("Reload either of these later with:")
    print(f"  booster2 = xgb.XGBClassifier(); booster2.load_model({xgb_path!r})")
    print(f"  clf2 = load_fitted_tabpfn_model({tabpfn_path!r}, device={args.device!r})")


if __name__ == "__main__":
    main()

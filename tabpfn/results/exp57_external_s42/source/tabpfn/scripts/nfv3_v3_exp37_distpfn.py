#!/usr/bin/env python3
"""EXP 37 -- DistPFN / DistPFN-T (Lee et al., "Mitigating Label Shift in Tabular
In-Context Learning via Test-Time Posterior Adjustment", ICML 2026;
github.com/seunghan96/DistPFN) on the NF-v3 IDS chronological split.

What DistPFN is
---------------
A training-free, model-agnostic post-hoc adjustment of an in-context classifier's
predicted posteriors under label shift (README of the repo, verbatim):

    P_test_avg    = y_prob.mean(axis=0)                      # predicted test prior
    y_prior_train = bincount(y_train) / len(y_train)         # context prior
    DistPFN   : adjusted = y_prob * P_test_avg / (y_prior_train + 1e-8), renormalised
    DistPFN-T : tau = CE(P_test_avg, y_prior_train); P_test_avg <- softmax(P_test_avg / tau)
                then the same multiplicative adjustment

i.e. a Saerens-style prior-ratio reweighting where the target prior is the mean
predicted posterior (one EM-free step).  The repo evaluates it on TabPFN
(pip `tabpfn`, v2 API `n_estimators=`) over OpenML sets with an induced label
shift; nothing in it is model-specific, so here it is applied to the classifier
this project actually uses.

What runs
---------
  1. Same loader / chronological split / evaluation set as every nfv3_v3_exp* script
     (nfv3_v3_common).  Train budget --max-train-samples (default 180,000 = "18만",
     stratified ratio-preserving, seed+850).  Eval = test split capped at
     --test-cap-per-class (100k -> 440,276 rows on cic2018).
     NOTE: the per-class cap itself induces a label shift between the context
     prior (benign 87%) and the evaluation set (benign 23%) -- exactly the regime
     DistPFN targets.  1a_prior_shift.csv makes that explicit.
  2. XGBoost on the same rows (same-run baseline).  Its predict_proba is also run
     through the two adjustments -- the paper's claim is ICL-specific, this row just
     shows what the prior reweighting does to a non-ICL model on the same shift.
  3. TabPFN-v3 (the project fork, --model-path) fit on the 180k rows, predict_proba
     on the eval set, then raw / DistPFN / DistPFN-T.
  All six predictions are scored with the same per-class table.

Faithfulness notes
------------------
  * The repo's DistPFN-T `cross_entropy(P_test_avg, prior)` is written for a 2-D
    p and raises AxisError on the 1-D P_test_avg it is given (the surrounding
    try/except then silently skips the dataset).  Implemented here as the README
    intends: tau = -sum_c P_test_avg[c] * log(prior[c]) (scalar), then
    softmax(P_test_avg / tau) -- note the softmax is applied to a probability
    vector, not to logits, exactly as in their `softmax_temperature`.
  * The repo sweeps n_estimators in {4,8,16,32} and keeps the best TEST ROC; this
    script uses one --n-estimators (default 4, the project's reference value).
  * The repo's `src/tabpfn/models_diff/*kc*.cpkt` is a leftover v1 training
    artifact that its eval scripts never load; not used here.

Artifacts  results/<ts>_<out_tag>_exp37_distpfn[/_82]/
------------------------------------------------------
  per_class_metrics.csv/.png   6 methods x 7 classes (+ macro/weighted/tail rows)
  0b_split_audit.csv, 0c_train_used.csv
  1a_prior_shift.csv/.png      per class: context prior, true eval prior, TabPFN
                               P_test_avg, DistPFN-T tempered prior, the resulting
                               multiplicative factor, and XGB's P_test_avg
  tabpfn_v3_test_proba.npy / xgboost_test_proba.npy   raw posteriors (float32)
  args.json / timings.json / run.log

Cost
----
  TabPFN-v3 at 180k context, n_estimators 4, one 440k batch: predicted peak
  ~8.4 GiB (measured memory model in nfv3_v3_common), roughly 10-20 min on the
  RTX 4090.  Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

Usage
-----
  cd imbalcic && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python tabpfn/scripts/nfv3_v3_exp37_distpfn.py --target-dataset cic2018
"""

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.special import softmax

HERE = os.path.dirname(os.path.abspath(__file__))
TABPFN_DIR = os.path.abspath(os.path.join(HERE, ".."))
IMBALCIC_ROOT = os.path.abspath(os.path.join(TABPFN_DIR, ".."))
sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))  # nfv3_v3_common.REPO_ROOT is one level short

import nfv3_v3_common as core  # noqa: E402
import nfv3_v3_c0_context as c0ctx  # noqa: E402
from exp_utils import render_table_png  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402  (project v3 fork, editable install)


class _Tee:
    def __init__(self, stream):
        self.stream, self.chunks = stream, []

    def write(self, s):
        self.stream.write(s)
        self.chunks.append(s)

    def flush(self):
        self.stream.flush()

    def text(self):
        return "".join(self.chunks)


def parse_args():
    p = core.base_parser(__doc__)
    p.set_defaults(max_train_samples=180_000,
                   data_dir=os.path.join(IMBALCIC_ROOT, "data"),
                   out_root=os.path.join(TABPFN_DIR, "results"),
                   models_dir=os.path.join(TABPFN_DIR, "saved_models"),
                   resume_dir=os.path.join(TABPFN_DIR, "resume"),
                   model_path=os.path.join(TABPFN_DIR, "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"))
    g = p.add_argument_group("DistPFN")
    g.add_argument("--distpfn-eps", type=float, default=1e-8, help="prior denominator epsilon (repo: 1e-8)")
    g.add_argument("--no-distpfn-t", action="store_true",
                   help="Skip the DistPFN-T variant (decision 2026-09-08: excluded from the SOTA comparison; "
                        "the released -T code path cannot execute and the paper-definition version collapses "
                        "on this data -- lablog/report/0904.md section 7). Earlier runs computed it.")
    g.add_argument("--skip-drift-diagnostic", action="store_true")
    c0ctx.add_c0_args(p)
    return p.parse_args()


def resolve_device(args):
    if args.device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return args.device


def to_float32_finite(block, name):
    arr = np.asarray(block, dtype=np.float32)
    n_bad = int((~np.isfinite(arr)).sum())
    if n_bad:
        print(f"  {name}: {n_bad:,} non-finite cells -> nan_to_num (same policy as every v3 run)")
    return np.nan_to_num(arr)


# ---- the method (transcribed from the repo README / eval_TABPFN_shift_O.py) ----

def distpfn(y_prob, prior_train, eps=1e-8):
    p_test_avg = y_prob.mean(axis=0)
    adjusted = (y_prob * p_test_avg) / (prior_train + eps)
    return adjusted / adjusted.sum(axis=1, keepdims=True), p_test_avg, p_test_avg


def distpfn_t(y_prob, prior_train, eps=1e-8):
    p_test_avg = y_prob.mean(axis=0)
    tau = float(-np.sum(p_test_avg * np.log(np.clip(prior_train, 1e-12, 1.0))))  # CE(P_test_avg, prior)
    p_tempered = softmax(p_test_avg / tau)  # repo applies softmax to the probability vector itself
    adjusted = (y_prob * p_tempered) / (prior_train + eps)
    return adjusted / adjusted.sum(axis=1, keepdims=True), p_test_avg, p_tempered, tau


def main():
    args = parse_args()
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.experiment = "exp37_distpfn"
    device = resolve_device(args)
    args.resolved_device = device
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)

    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    print(f"target={args.target_dataset}  classes ({n_classes}): {class_names}")
    print(f"split rows: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
    if args.train_split == "train+val":
        train_idx = np.sort(np.concatenate([train_idx, val_idx]))
        y_train_all = label_fn(train_idx)
        print(f"--train-split train+val: train pool {len(train_idx):,};  test unchanged")

    test_eval_idx = core.cap_per_class(test_idx, y_test_all, n_classes, args.test_cap_per_class, args.seed + 900)
    y_test_eval = label_fn(test_eval_idx)
    print(f"test eval rows: {len(test_eval_idx):,} (cap_per_class={args.test_cap_per_class})")
    X_test_eval = to_float32_finite(X[test_eval_idx], "X_test_eval")

    drift_df = None
    if not args.skip_drift_diagnostic:
        print("\n--- pre-fit diagnostic: train/test feature drift per class ---")
        drift_df = core.diagnose_train_test_drift(X, class_names, train_idx, test_idx, label_fn)
        print(drift_df.to_string(index=False))

    train_used_idx, cap_policy, c0_info = train_idx, "none", None
    if args.context_recipe == "c0":
        c0_info = c0ctx.build_c0(args, train_idx, y_train_all, class_names, label_fn)
        train_used_idx = c0_info["idx"]
        cap_policy = f"c0_share{args.c0_benign_share}_{args.c0_attack_alloc}_{args.c0_pool_partition}"
    elif 0 < args.max_train_samples < len(train_used_idx):
        train_used_idx = core.stratified_subset(train_used_idx, label_fn(train_used_idx), n_classes,
                                                args.max_train_samples, args.seed + 850)
        cap_policy = "stratified_ratio_preserving"
    y_train_used = label_fn(train_used_idx)
    used_counts = np.bincount(y_train_used, minlength=n_classes)
    prior_train = used_counts / used_counts.sum()
    train_used_df = pd.DataFrame({"class": class_names,
                                  "pool_rows": np.bincount(y_train_all, minlength=n_classes),
                                  "used_rows": used_counts, "prior_train": prior_train})
    print(f"\ntrain selection: used={len(train_used_idx):,} of pool {len(train_idx):,} "
          f"({100 * len(train_used_idx) / len(train_idx):.2f}%)  cap_policy={cap_policy}")
    print(train_used_df.to_string(index=False))
    X_train_used = to_float32_finite(X[train_used_idx], "X_train_used")
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()
    core.report_memory_plan(len(train_used_idx), len(test_eval_idx), args)

    all_rows, timings, probas = [], {}, {}

    def score(method, proba):
        pred = np.asarray(proba).argmax(axis=1)
        all_rows.extend(core.per_class_table(method, y_test_eval, pred, class_names, tail_classes))

    # ---- XGBoost on the same rows ----
    if not args.skip_xgboost:
        booster = xgb.XGBClassifier(
            n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree, min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda, objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", n_jobs=-1, random_state=args.seed)
        print(f"\nXGBoost fitting on the same {len(X_train_used):,} rows ...", flush=True)
        t0 = time.time()
        booster.fit(X_train_used, y_train_used)
        timings["xgboost_fit_seconds"] = time.time() - t0
        t0 = time.time()
        probas["xgboost"] = booster.predict_proba(X_test_eval).astype(np.float64)
        timings["xgboost_predict_seconds"] = time.time() - t0
        print(f"XGBoost fit {timings['xgboost_fit_seconds']:.1f}s predict {timings['xgboost_predict_seconds']:.1f}s")

    # ---- TabPFN-v3 (project fork) ----
    if not args.skip_tabpfn:
        clf = TabPFNClassifier(
            device=device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            inference_config={"SUBSAMPLE_SAMPLES": int(args.subsample_samples) or None},
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        max_samples = clf.get_inference_config().MAX_NUMBER_OF_SAMPLES
        if not args.ignore_pretraining_limits and len(train_used_idx) > max_samples:
            raise SystemExit(f"context {len(train_used_idx):,} > checkpoint limit {max_samples:,}")
        print(f"\nTabPFN-v3: context={len(train_used_idx):,} n_estimators={args.n_estimators} "
              f"fit_mode={args.fit_mode}", flush=True)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        clf.fit(X_train_used, y_train_used)
        timings["tabpfn_fit_seconds"] = time.time() - t0
        t0 = time.time()
        bs = args.test_batch_size if args.test_batch_size > 0 else len(X_test_eval)
        chunks = []
        for start in range(0, len(X_test_eval), bs):
            chunks.append(clf.predict_proba(X_test_eval[start:start + bs]))
            print(f"  TabPFN predict_proba rows {start:,}:{min(start + bs, len(X_test_eval)):,}", flush=True)
        probas["tabpfn_v3"] = np.concatenate(chunks).astype(np.float64)
        timings["tabpfn_predict_seconds"] = time.time() - t0
        if device.startswith("cuda"):
            timings["tabpfn_gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        print(f"TabPFN-v3 fit {timings['tabpfn_fit_seconds']:.1f}s predict {timings['tabpfn_predict_seconds']:.1f}s "
              f"peak {timings.get('tabpfn_gpu_peak_gib', 0):.2f} GiB")
        # column order: TabPFN's classes_ are the sorted labels 0..K-1 present in y_train_used
        assert list(clf.classes_) == list(range(n_classes)), clf.classes_

    # ---- the adjustment, applied to every raw posterior ----
    prior_true_test = np.bincount(y_test_eval, minlength=n_classes) / len(y_test_eval)
    shift_rows = {"class": class_names, "prior_train_context": prior_train, "prior_eval_true": prior_true_test}
    for base, y_prob in probas.items():
        score(base, y_prob)
        adj, p_avg, _ = distpfn(y_prob, prior_train, args.distpfn_eps)
        score(f"{base}+distpfn", adj)
        shift_rows[f"{base}_p_test_avg"] = p_avg
        shift_rows[f"{base}_distpfn_factor"] = p_avg / (prior_train + args.distpfn_eps)
        if args.no_distpfn_t:
            print(f"{base}: P_test_avg={np.round(p_avg, 4).tolist()}  (DistPFN-T skipped)")
            continue
        adj_t, _, p_temp, tau = distpfn_t(y_prob, prior_train, args.distpfn_eps)
        score(f"{base}+distpfn_t", adj_t)
        timings[f"{base}_distpfn_t_tau"] = tau
        shift_rows[f"{base}_distpfn_t_prior"] = p_temp
        shift_rows[f"{base}_distpfn_t_factor"] = p_temp / (prior_train + args.distpfn_eps)
        print(f"{base}: P_test_avg={np.round(p_avg, 4).tolist()}  tau={tau:.4f}  "
              f"tempered={np.round(p_temp, 4).tolist()}")
    shift_df = pd.DataFrame(shift_rows)

    table = pd.DataFrame(all_rows)
    piv = table.pivot(index="class", columns="method", values="f1")
    order = [m for m in ["xgboost", "xgboost+distpfn", "xgboost+distpfn_t",
                         "tabpfn_v3", "tabpfn_v3+distpfn", "tabpfn_v3+distpfn_t"] if m in piv.columns]
    piv = piv[order]
    print("\n=== summary (macro / weighted / tail F1) ===")
    print(piv.loc[["macro_avg", "weighted_avg", "tail_avg"]].to_string())
    print("\n=== per-class F1 ===")
    print(piv.drop(index=["macro_avg", "weighted_avg", "tail_avg"]).to_string())
    print("\n=== prior shift ===")
    print(shift_df.to_string(index=False))

    timings.update({
        "experiment": args.experiment, "target_dataset": args.target_dataset,
        "train_pool_rows": int(len(train_idx)), "train_rows_used": int(len(train_used_idx)),
        "train_cap_policy": cap_policy, "train_split": args.train_split,
        "test_evaluation_rows": int(len(test_eval_idx)), "n_estimators": args.n_estimators,
        "fit_mode": args.fit_mode, "device": device,
    })
    ts = time.strftime("%Y%m%d_%H%M%S")
    split_tag = "" if args.train_split == "train" else "_82"
    out_dir = os.path.join(args.out_root, f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp37_distpfn{split_tag}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    train_used_df.to_csv(os.path.join(out_dir, "0c_train_used.csv"), index=False)
    if c0_info is not None:
        c0_info["pool_audit"].to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
        if c0_info["c0_scenario"] is not None:
            c0_info["c0_scenario"].to_csv(os.path.join(out_dir, "0g_c0_scenario.csv"), index=False)
        timings.update(c0_info["timings"])
    shift_df.to_csv(os.path.join(out_dir, "1a_prior_shift.csv"), index=False)
    for base, y_prob in probas.items():
        np.save(os.path.join(out_dir, f"{base}_test_proba.npy"), y_prob.astype(np.float32))
    if drift_df is not None:
        drift_df.to_csv(os.path.join(out_dir, "train_test_drift_diagnostic.csv"), index=False)
    for df, name, title in [(piv.reset_index(), "per_class_metrics", "exp37 DistPFN adjustment -- F1"),
                            (shift_df, "1a_prior_shift", "exp37 context prior vs eval prior vs predicted P_test_avg")]:
        try:
            render_table_png(df, os.path.join(out_dir, f"{name}.png"), title=title,
                             high_good=[c for c in df.columns if c in order])
        except Exception as exc:
            print(f"  (png for {name} skipped: {exc})")
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([timings], f, indent=2)
    print(f"\nWrote {out_dir}")
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(tee.text())
    return out_dir


if __name__ == "__main__":
    main()

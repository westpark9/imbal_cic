#!/usr/bin/env python3
"""Fine-label (scenario) training vs family-label training, XGBoost, CPU only.

MOTIVATION (0831 discussion). The NF-v3 suite merges the raw CSE-CIC-IDS2018
labels into 7 families: brute_force = FTP-BruteForce + SSH-Bruteforce,
dos = 4 scenarios, ddos = 3, web_attacks = 3. The user proposal: train at
SCENARIO granularity (15 classes) and aggregate predictions back to family for
reporting. This script measures the ceiling of that idea on XGBoost before any
GPU run: if a strong learner gains nothing from explicit mode separation at
matched data, the classifier-head benefit for the PFN is likely small too
(coverage/weight effects are separate and belong to exp27/exp26).

Key property of the aggregation: intra-family confusion (xss predicted as
web-bruteforce) is FREE after aggregation -- only cross-family leakage hurts.
So this isolates "does separating the modes help the decision boundary".

Both models are trained on the SAME rows and scored on the SAME eval rows in
the same run. Aggregation is soft (sum scenario probabilities within family,
then argmax) with hard (argmax then map) as a check.

Usage (from repo root; ~35-45 min CPU, safe to nice alongside a GPU run):

    nice -n 19 python scripts/race_scenario_label_diag.py \
        --run-dir tabpfn/results/20260827_174858_nfv3_cic2018_exp24b_signscorer

Outputs (default --out scripts/results/scenario_label_diag):
    per_class_metrics.csv   family-level P/R/F1: xgb_family, xgb_fine_soft,
                            xgb_fine_hard (+ macro/tail/weighted)
    fine_confusion.csv      scenario-level recall + where each scenario's rows
                            went (top wrong sink), fine model only
    summary.csv             one row per method
    args.json / timings.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from exp_utils import scenario_chronological_split  # noqa: E402

DATASET = "cse_cic_ids2018"
TAIL = ["bot", "infiltration", "web_attacks"]


def per_class_rows(method, y_true, y_pred, class_names):
    rows, f1s, sup = [], [], []
    for c, name in enumerate(class_names):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        sup.append(int((y_true == c).sum()))
        rows.append({"method": method, "class": name, "precision": round(p, 6),
                     "recall": round(r, 6), "f1": round(f1, 6),
                     "support": sup[-1]})
    tot = max(sum(sup), 1)
    ti = [i for i, n in enumerate(class_names) if n in TAIL]
    rows.append({"method": method, "class": "macro_avg", "precision": "",
                 "recall": "", "f1": round(float(np.mean(f1s)), 6),
                 "support": tot})
    rows.append({"method": method, "class": "weighted_avg", "precision": "",
                 "recall": "",
                 "f1": round(float(np.average(f1s, weights=sup)), 6),
                 "support": tot})
    rows.append({"method": method, "class": "tail_avg", "precision": "",
                 "recall": "",
                 "f1": round(float(np.mean([f1s[i] for i in ti])), 6)
                 if ti else "", "support": tot})
    return rows


def fit_predict(Xtr, ytr, X_ev, n_classes, args, tag):
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        min_child_weight=args.xgb_min_child_weight,
        reg_lambda=args.xgb_reg_lambda, objective="multi:softprob",
        num_class=n_classes, eval_metric="mlogloss", n_jobs=args.n_jobs,
        random_state=args.seed)
    print(f"[{tag}] fitting on {len(ytr):,} rows, {n_classes} classes ...",
          flush=True)
    clf.fit(Xtr, ytr)
    fit_s = round(time.time() - t0, 1)
    t0 = time.time()
    proba = clf.predict_proba(X_ev).astype(np.float32)
    # guard: classes absent from training would shrink proba's width
    if proba.shape[1] != n_classes:
        full = np.zeros((len(X_ev), n_classes), dtype=np.float32)
        full[:, clf.classes_] = proba
        proba = full
    pred_s = round(time.time() - t0, 1)
    print(f"[{tag}] fit {fit_s}s predict {pred_s}s", flush=True)
    return proba, fit_s, pred_s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",
                    default="data/nfv3_energy_suite_uncapped_scenarios.pkl")
    ap.add_argument("--run-dir", required=True,
                    help="run whose context_rows.npz defines the eval rows")
    ap.add_argument("--out", default="scripts/results/scenario_label_diag")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--xgb-n-estimators", type=int, default=300)
    ap.add_argument("--xgb-max-depth", type=int, default=8)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.05)
    ap.add_argument("--xgb-subsample", type=float, default=0.8)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    ap.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    ap.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"),
              indent=1)
    timings = {}
    t0 = time.time()

    print(f"loading {args.data} ...", flush=True)
    import pickle
    with open(args.data, "rb") as f:
        suite = pickle.load(f)
    X = suite["X"]
    datasets = np.asarray(suite["dataset_names"])
    families = np.asarray(suite["families"])
    scenarios = np.asarray(suite["attack_scenarios"])
    timestamps = np.asarray(suite["timestamps"])
    del suite
    timings["load_s"] = round(time.time() - t0, 1)
    print(f"  loaded in {timings['load_s']}s", flush=True)

    target_idx = np.flatnonzero(datasets == DATASET)
    fam_names = sorted(np.unique(families[target_idx]).tolist())
    fine_names = sorted(np.unique(scenarios[target_idx]).tolist())
    fmap = {n: i for i, n in enumerate(fam_names)}
    smap = {n: i for i, n in enumerate(fine_names)}
    n_fam, n_fine = len(fam_names), len(fine_names)

    y_fam_all = np.full(len(families), -1, dtype=np.int64)
    y_fam_all[target_idx] = [fmap[v] for v in families[target_idx]]
    y_fine_all = np.full(len(families), -1, dtype=np.int64)
    y_fine_all[target_idx] = [smap[v] for v in scenarios[target_idx]]

    # scenario -> family map, derived from the data itself (consistency check)
    fine_to_fam = np.zeros(n_fine, dtype=np.int64)
    for s in range(n_fine):
        m = target_idx[y_fine_all[target_idx] == s]
        fams = np.unique(y_fam_all[m])
        if len(fams) != 1:
            raise SystemExit(f"scenario {fine_names[s]!r} maps to multiple "
                             f"families {fams} -- aggregation undefined")
        fine_to_fam[s] = fams[0]
    print("scenario -> family:", flush=True)
    for s in range(n_fine):
        n_tr = int((y_fine_all[target_idx] == s).sum())
        print(f"  {fine_names[s]:26s} -> {fam_names[fine_to_fam[s]]:14s} "
              f"({n_tr:,} rows)", flush=True)

    split, _ = scenario_chronological_split(target_idx, scenarios, timestamps)
    train_idx = split["train"]
    ctx = np.load(os.path.join(args.run_dir, "context_rows.npz"))
    eval_idx = ctx["eval"]

    def feats(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    Xtr = feats(train_idx)
    X_ev = feats(eval_idx)
    y_fam_tr, y_fine_tr = y_fam_all[train_idx], y_fine_all[train_idx]
    y_fam_ev, y_fine_ev = y_fam_all[eval_idx], y_fine_all[eval_idx]
    print(f"train={len(train_idx):,} eval={len(eval_idx):,} "
          f"families={n_fam} scenarios={n_fine}", flush=True)

    all_rows, summary = [], []

    # ---- A. family-label baseline (same run, reproduces the 0.7548 bar) ----
    proba_fam, fit_s, pred_s = fit_predict(Xtr, y_fam_tr, X_ev, n_fam, args,
                                           "family")
    timings["family_fit_s"], timings["family_pred_s"] = fit_s, pred_s
    pred = proba_fam.argmax(axis=1)
    tab = per_class_rows("xgb_family", y_fam_ev, pred, fam_names)
    all_rows += tab
    summary.append({"method": "xgb_family",
                    "macro_f1": tab[-3]["f1"], "tail_f1": tab[-1]["f1"],
                    "weighted_f1": tab[-2]["f1"]})
    del proba_fam

    # ---- B. fine-label model, aggregated ----------------------------------
    proba_fine, fit_s, pred_s = fit_predict(Xtr, y_fine_tr, X_ev, n_fine,
                                            args, "fine")
    timings["fine_fit_s"], timings["fine_pred_s"] = fit_s, pred_s

    # soft aggregation: sum scenario probs within each family, argmax family
    agg = np.zeros((len(X_ev), n_fam), dtype=np.float32)
    for s in range(n_fine):
        agg[:, fine_to_fam[s]] += proba_fine[:, s]
    pred_soft = agg.argmax(axis=1)
    tab = per_class_rows("xgb_fine_soft", y_fam_ev, pred_soft, fam_names)
    all_rows += tab
    summary.append({"method": "xgb_fine_soft",
                    "macro_f1": tab[-3]["f1"], "tail_f1": tab[-1]["f1"],
                    "weighted_f1": tab[-2]["f1"]})

    # hard aggregation: argmax scenario, then map
    pred_hard = fine_to_fam[proba_fine.argmax(axis=1)]
    tab = per_class_rows("xgb_fine_hard", y_fam_ev, pred_hard, fam_names)
    all_rows += tab
    summary.append({"method": "xgb_fine_hard",
                    "macro_f1": tab[-3]["f1"], "tail_f1": tab[-1]["f1"],
                    "weighted_f1": tab[-2]["f1"]})

    # scenario-level diagnostics of the fine model
    fine_pred = proba_fine.argmax(axis=1)
    conf_rows = []
    for s in range(n_fine):
        m = y_fine_ev == s
        if not m.any():
            continue
        pr = fine_pred[m]
        rec = float((pr == s).mean())
        wrong = pr[pr != s]
        if len(wrong):
            top = np.bincount(wrong, minlength=n_fine).argmax()
            sink, sink_frac = fine_names[top], float((wrong == top).sum()) / m.sum()
        else:
            sink, sink_frac = "", 0.0
        fam_rec = float((fine_to_fam[pr] == fine_to_fam[s]).mean())
        conf_rows.append({
            "scenario": fine_names[s], "family": fam_names[fine_to_fam[s]],
            "eval_rows": int(m.sum()),
            "train_rows": int((y_fine_tr == s).sum()),
            "recall_fine": round(rec, 6),
            "recall_family_after_agg": round(fam_rec, 6),
            "top_wrong_sink": sink, "sink_frac": round(sink_frac, 6)})
    pd.DataFrame(conf_rows).to_csv(
        os.path.join(args.out, "fine_confusion.csv"), index=False)

    pd.DataFrame(all_rows).to_csv(
        os.path.join(args.out, "per_class_metrics.csv"), index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    timings["total_s"] = round(time.time() - t0, 1)
    json.dump(timings, open(os.path.join(args.out, "timings.json"), "w"),
              indent=1)
    print("\n=== family-level summary (same train rows, same eval rows) ===")
    print(sm.to_string(index=False))
    print(f"\nartifacts -> {args.out}")


if __name__ == "__main__":
    main()

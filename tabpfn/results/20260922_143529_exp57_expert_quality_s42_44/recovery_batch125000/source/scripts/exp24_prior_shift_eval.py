#!/usr/bin/env python3
"""EXP24 post-hoc -- guide §20.4 test-prior-shift evaluation (axis 3).

Evaluates SAVED per-row predictions of a record run under shifted test class
priors, exactly (weighted confusion matrix -- no resampling noise): a row
with true class t gets weight w = pi_mix[t] / pi_test[t]. Predictions are
fixed per row, so no model is re-run except one XGBoost fit (deterministic,
same hypers/seed as the record scripts) to obtain its posteriors.

Mixes (pre-registered): training_like (natural test), balanced (uniform),
tail_heavy (pi ∝ 1/n_c), head_heavy (benign 0.99, rest natural).
Methods: global(corrected p~0) / RACE system / XGBoost, each with and
without KNOWN-PRIOR posterior correction p'(y|x) ∝ p(y|x)·pi_mix/pi_train
(§20.4's DistPFN-equivalent; assumes the target prior is known -- declared).
Metrics per (mix, method): macro-F1, tail-F1, benign FPR, weighted ECE,
balanced accuracy (note: balanced acc is prior-invariant by construction).

Writes 8a_prior_shift.csv (+png) into the source run directory.

    python scripts/exp24_prior_shift_eval.py \
        --run-dir tabpfn/results/20260827_174858_nfv3_cic2018_exp24b_signscorer
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tabpfn"))
import nfv3_v3_common as core  # noqa: E402
from render_run_tables import render_dir  # noqa: E402

TAIL = ["bot", "infiltration", "web_attacks"]


def weighted_scores(y, pred, w, class_names, benign_id):
    C = len(class_names)
    cm = np.zeros((C, C))
    np.add.at(cm, (y, pred), w)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(axis=0), 1e-12)
    rec = tp / np.maximum(cm.sum(axis=1), 1e-12)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    present = cm.sum(axis=1) > 0
    out = {"macro_f1": float(f1[present].mean()),
           "tail_f1": float(np.mean([f1[class_names.index(t)] for t in TAIL
                                     if present[class_names.index(t)]])),
           "balanced_acc": float(rec[present].mean())}
    if benign_id is not None and cm[benign_id].sum() > 0:
        out["benign_fpr"] = float(
            (cm[benign_id].sum() - cm[benign_id, benign_id])
            / cm[benign_id].sum())
    return out


def weighted_ece(probs, y, w, bins=15):
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == y).astype(np.float64)
    idx = np.clip((conf * bins).astype(np.int64), 0, bins - 1)
    tot = w.sum()
    ece = 0.0
    for b in range(bins):
        m = idx == b
        wb = w[m].sum()
        if wb > 0:
            ece += (wb / tot) * abs(np.average(acc[m], weights=w[m])
                                    - np.average(conf[m], weights=w[m]))
    return float(ece)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="tabpfn/results/"
                   "20260827_174858_nfv3_cic2018_exp24b_signscorer")
    p.add_argument("--data", default="data/nfv3_energy_suite_uncapped_"
                   "scenarios.pkl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--head-heavy-benign", type=float, default=0.99)
    p.add_argument("--xgb-n-estimators", type=int, default=300)
    p.add_argument("--xgb-max-depth", type=int, default=8)
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    p.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    args = p.parse_args()
    args.experiment = "exp24_prior_shift_eval"
    print(f"Args: {vars(args)}", flush=True)

    rd = args.run_dir
    sysd = np.load(os.path.join(rd, "probs_racepfn_system.npz"))
    glob = np.load(os.path.join(rd, "probs_tabpfn_global.npz"))
    ctx = np.load(os.path.join(rd, "context_rows.npz"))
    y = sysd["y_true"].astype(np.int64)
    P = {"racepfn_system": sysd["probs"].astype(np.float64),
         "global_tabpfn": glob["probs"].astype(np.float64)}
    eval_idx = ctx["eval"]
    assert len(eval_idx) == len(y)

    # ---- rebuild splits, fit XGB once (deterministic), get posteriors ------
    t0 = time.time()
    X, class_names, train_idx, _, test_idx, _, _, _, label_fn = \
        core.load_cic2018(SimpleNamespace(data=args.data))
    assert np.array_equal(eval_idx, test_idx), "run eval != loader test"
    assert np.array_equal(label_fn(eval_idx), y), "label mismatch"
    n_classes = len(class_names)
    benign_id = class_names.index("benign") if "benign" in class_names else None
    Xtr = np.nan_to_num(np.asarray(X[train_idx], dtype=np.float32))
    ytr = label_fn(train_idx)
    booster = xgb.XGBClassifier(
        n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        min_child_weight=args.xgb_min_child_weight,
        reg_lambda=args.xgb_reg_lambda, objective="multi:softprob",
        num_class=n_classes, eval_metric="mlogloss", n_jobs=-1,
        random_state=args.seed)
    print(f"fitting XGB on {len(Xtr):,} rows ...", flush=True)
    booster.fit(Xtr, ytr)
    del Xtr
    Xev = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))
    del X
    core._PICKLE_CACHE.clear()
    pr = np.zeros((len(y), n_classes), dtype=np.float64)
    pr[:, np.asarray(booster.classes_, dtype=np.int64)] = \
        booster.predict_proba(Xev)
    P["xgboost"] = pr
    del Xev
    print(f"XGB posteriors ready ({time.time() - t0:.0f}s)", flush=True)

    # sanity: natural-mix macro must reproduce the run's recorded values
    pi_train = np.bincount(ytr, minlength=n_classes) / len(ytr)
    pi_test = np.bincount(y, minlength=n_classes) / len(y)

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    inv = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    tail_heavy = inv / inv.sum()
    head_heavy = pi_test * (1 - args.head_heavy_benign) \
        / max(1 - pi_test[benign_id], 1e-12)
    head_heavy[benign_id] = args.head_heavy_benign
    MIXES = {"training_like": pi_test,
             "balanced": np.where(counts > 0, 1.0 / (counts > 0).sum(), 0.0),
             "tail_heavy": tail_heavy,
             "head_heavy": head_heavy}

    rows = []
    for mix_name, pi in MIXES.items():
        w = (pi / np.maximum(pi_test, 1e-12))[y]
        for mname, probs in P.items():
            for corr in (False, True):
                if corr:
                    q = probs * (pi / np.maximum(pi_train, 1e-12))[None, :]
                    q = q / q.sum(axis=1, keepdims=True)
                else:
                    q = probs
                pred = q.argmax(axis=1)
                r = {"mix": mix_name,
                     "method": mname + ("+prior_corr" if corr else ""),
                     **weighted_scores(y, pred, w, class_names, benign_id),
                     "ece": weighted_ece(q, y, w)}
                rows.append({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items()})
        print(f"mix {mix_name} done", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(rd, "8a_prior_shift.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    piv = df.pivot(index="method", columns="mix", values="macro_f1")
    print("\n=== macro-F1 by test prior ===")
    print(piv.to_string())
    print("\n=== full table ===")
    print(df.to_string(index=False))
    try:
        render_dir(rd)
    except Exception as exc:
        print(f"render skipped: {exc}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

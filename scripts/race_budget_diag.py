#!/usr/bin/env python3
"""E-0.5 + E-1: information-budget diagnostic and row-meta sidecar. CPU only.

WHY (0831.md §1 cause A). XGBoost trains on the whole train pool (12,069,313
rows, SSH-bruteforce included). The global TabPFN's context C0 is drawn only
from D_global, which `class_chrono_partition` fills with the chronologically
FIRST 50% of each class -- and every ssh_bruteforce row is later than every
ftp_bruteforce row. So C0 contains zero SSH. The -0.191 brute_force gap to
XGBoost may therefore be an information-budget artifact rather than a method
gap. This script measures how much of the gap that accounts for, by refitting
the SAME XGBoost under progressively tighter row budgets and scoring all of
them on the SAME evaluation rows.

Budgets
    full      every train row                       (reproduces the 0.7548 bar)
    ctxpool   D_global only, i.e. the pool C0 is drawn from
    c0        the exact 1,000,000 rows the frozen global PFN sees
              (row ids read from a run's context_rows.npz -- no reseeding)

Also writes a row-meta sidecar so later temporal/scenario analyses never have
to re-open the 14.7 GB pickle.

Usage (from repo root):

    python scripts/race_budget_diag.py \
        --run-dir tabpfn/results/20260827_174858_nfv3_cic2018_exp24b_signscorer

Outputs (default --out scripts/results/budget_diag):
    per_class_metrics.csv     per class P/R/F1 for every budget + macro/tail
    summary.csv               macro/tail/weighted + brute FTP-vs-SSH recall
    scenario_recall.csv       per attack scenario recall, per budget
    args.json / timings.json
    data/nfv3_cic2018_row_meta.npz   (row_id, ts, scenario, label, split_tag)
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


def class_chrono_partition(train_idx, y_train, ts_train, ctx_frac, exp_frac,
                           n_classes):
    """Copy of the partition used by nfv3_v3_exp2x (frozen scripts stay untouched)."""
    parts = {"context": [], "expert": [], "route": []}
    for cid in range(n_classes):
        mask = y_train == cid
        rows = train_idx[mask]
        if len(rows) == 0:
            continue
        order = rows[np.argsort(ts_train[mask], kind="stable")]
        n = len(order)
        n_ctx, n_exp = int(n * ctx_frac), int(n * exp_frac)
        parts["context"].extend(order[:n_ctx])
        parts["expert"].extend(order[n_ctx:n_ctx + n_exp])
        parts["route"].extend(order[n_ctx + n_exp:])
    return {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in parts.items()}


def per_class_rows(method, y_true, y_pred, class_names):
    rows = []
    f1s, sup = [], []
    for c, name in enumerate(class_names):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        n = int((y_true == c).sum())
        f1s.append(f1)
        sup.append(n)
        rows.append({"method": method, "class": name, "precision": round(p, 6),
                     "recall": round(r, 6), "f1": round(f1, 6), "support": n})
    tot = max(sum(sup), 1)
    rows.append({"method": method, "class": "macro_avg", "precision": "",
                 "recall": "", "f1": round(float(np.mean(f1s)), 6), "support": tot})
    rows.append({"method": method, "class": "weighted_avg", "precision": "",
                 "recall": "",
                 "f1": round(float(np.average(f1s, weights=sup)), 6), "support": tot})
    ti = [i for i, n in enumerate(class_names) if n in TAIL]
    rows.append({"method": method, "class": "tail_avg", "precision": "",
                 "recall": "",
                 "f1": round(float(np.mean([f1s[i] for i in ti])), 6) if ti else "",
                 "support": tot})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",
                    default="data/nfv3_energy_suite_uncapped_scenarios.pkl")
    ap.add_argument("--run-dir", required=True,
                    help="run whose context_rows.npz defines C0 and eval rows")
    ap.add_argument("--out", default="scripts/results/budget_diag")
    ap.add_argument("--budgets", default="full,ctxpool,c0")
    ap.add_argument("--context-frac", type=float, default=0.5)
    ap.add_argument("--expert-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--xgb-n-estimators", type=int, default=300)
    ap.add_argument("--xgb-max-depth", type=int, default=8)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.05)
    ap.add_argument("--xgb-subsample", type=float, default=0.8)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    ap.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    ap.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    ap.add_argument("--sidecar", default="data/nfv3_cic2018_row_meta.npz")
    ap.add_argument("--skip-sidecar", action="store_true")
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
    print(f"  loaded in {timings['load_s']}s; X={getattr(X, 'shape', '?')}",
          flush=True)

    target_idx = np.flatnonzero(datasets == DATASET)
    class_names = sorted(np.unique(families[target_idx]).tolist())
    n_classes = len(class_names)
    cmap = {n: i for i, n in enumerate(class_names)}
    labels_all = np.full(len(families), -1, dtype=np.int64)
    labels_all[target_idx] = [cmap[f] for f in families[target_idx]]

    split, _ = scenario_chronological_split(target_idx, scenarios, timestamps)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    y_train = labels_all[train_idx]
    pools = class_chrono_partition(train_idx, y_train, timestamps[train_idx],
                                   args.context_frac, args.expert_frac,
                                   n_classes)

    ctx = np.load(os.path.join(args.run_dir, "context_rows.npz"))
    c0_idx, eval_idx = ctx["C0"], ctx["eval"]
    print(f"train={len(train_idx):,} ctxpool={len(pools['context']):,} "
          f"C0={len(c0_idx):,} eval={len(eval_idx):,}", flush=True)

    # ---- E-1 sidecar ----------------------------------------------------
    if not args.skip_sidecar:
        tag = np.full(len(target_idx), "unassigned", dtype=object)
        pos = {int(r): i for i, r in enumerate(target_idx)}
        for name, rows in (("D_global", pools["context"]),
                           ("D_expert", pools["expert"]),
                           ("D_route", pools["route"]),
                           ("val", val_idx), ("test", test_idx)):
            for r in rows:
                tag[pos[int(r)]] = name
        np.savez_compressed(
            args.sidecar, row_id=target_idx, ts=timestamps[target_idx],
            scenario=scenarios[target_idx].astype(str),
            label=labels_all[target_idx],
            split_tag=np.asarray(tag, dtype=str),
            class_names=np.asarray(class_names, dtype=str))
        print(f"sidecar -> {args.sidecar}", flush=True)

    # ---- evaluation matrix ----------------------------------------------
    def feats(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    X_eval, y_eval = feats(eval_idx), labels_all[eval_idx]
    scen_eval = scenarios[eval_idx].astype(str)

    budget_rows = {"full": train_idx, "ctxpool": pools["context"], "c0": c0_idx}

    # "c0+sshN": C0 plus N ssh_bruteforce rows drawn from D_expert -- a CPU
    # proxy for the §17 refresh question "how many recent verified labels does
    # it take to recover a scenario the context never saw?".  XGBoost is less
    # sample-efficient than in-context TabPFN (0813: 13 rows -> f1 .560 for
    # TabPFN, 0 for XGB), so the N it needs is a loose UPPER bound.
    ssh_pool = pools["expert"][scenarios[pools["expert"]] == "ssh_bruteforce"]
    rng = np.random.default_rng(args.seed)
    for b in [x.strip() for x in args.budgets.split(",")]:
        if not b.startswith("c0+ssh"):
            continue
        n = int(b[len("c0+ssh"):])
        take = rng.choice(ssh_pool, size=min(n, len(ssh_pool)), replace=False)
        budget_rows[b] = np.sort(np.concatenate([c0_idx, take]))
        print(f"{b}: C0 {len(c0_idx):,} + SSH {len(take):,} "
              f"(pool {len(ssh_pool):,})", flush=True)
    all_rows, summary, scen_rows = [], [], []
    for b in [x.strip() for x in args.budgets.split(",") if x.strip()]:
        rows = budget_rows[b]
        Xb, yb = feats(rows), labels_all[rows]
        t1 = time.time()
        print(f"\n[{b}] fitting XGBoost on {len(rows):,} rows ...", flush=True)
        clf = xgb.XGBClassifier(
            n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda, objective="multi:softprob",
            num_class=n_classes, eval_metric="mlogloss", n_jobs=-1,
            random_state=args.seed)
        clf.fit(Xb, yb)
        fit_s = round(time.time() - t1, 1)
        pred = clf.predict(X_eval).astype(np.int64)
        timings[f"{b}_fit_s"] = fit_s
        del Xb, yb, clf

        tab = per_class_rows(f"xgb_{b}", y_eval, pred, class_names)
        all_rows.extend(tab)
        d = {r["class"]: r["f1"] for r in tab}
        bi = cmap.get("brute_force")
        rec = {}
        for s in ("ftp_bruteforce", "ssh_bruteforce"):
            m = scen_eval == s
            rec[s] = round(float((pred[m] == bi).mean()), 6) if m.any() else None
        summary.append({
            "budget": b, "train_rows": int(len(rows)), "fit_s": fit_s,
            "macro_f1": d["macro_avg"], "tail_f1": d["tail_avg"],
            "weighted_f1": d["weighted_avg"],
            "brute_f1": d.get("brute_force"), "ddos_f1": d.get("ddos"),
            "recall_ftp_bruteforce": rec["ftp_bruteforce"],
            "recall_ssh_bruteforce": rec["ssh_bruteforce"]})
        for s in np.unique(scen_eval):
            m = scen_eval == s
            true_c = int(np.bincount(y_eval[m]).argmax())
            scen_rows.append({
                "budget": b, "scenario": s, "rows": int(m.sum()),
                "class": class_names[true_c],
                "recall": round(float((pred[m] == true_c).mean()), 6)})
        print(f"[{b}] macro={d['macro_avg']} tail={d['tail_avg']} "
              f"brute={d.get('brute_force')} "
              f"ssh_recall={rec['ssh_bruteforce']} ({fit_s}s)", flush=True)

    pd.DataFrame(all_rows).to_csv(
        os.path.join(args.out, "per_class_metrics.csv"), index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.DataFrame(scen_rows).to_csv(
        os.path.join(args.out, "scenario_recall.csv"), index=False)
    timings["total_s"] = round(time.time() - t0, 1)
    json.dump(timings, open(os.path.join(args.out, "timings.json"), "w"),
              indent=1)
    print("\n=== information-budget summary ===")
    print(sm.to_string(index=False))
    print(f"\nartifacts -> {args.out}")


if __name__ == "__main__":
    main()

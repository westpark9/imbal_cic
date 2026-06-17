#!/usr/bin/env python3
"""
code_tta_all.py  —  Inclusive-expert TTA with MATI/SADE-style weight learning.

Expert design (inclusive)
-------------------------
k experts, each trained on ALL classes with different balance levels.
Expert i's training data: each class capped at count[rank ceil(n*i/k)].
Expert 0: no cap (natural imbalance).  Expert k-1: nearly uniform.
Val set also capped at the same level per expert → fair early stopping.

TTA design (MATI/SADE hybrid)
------------------------------
Three test distributions varying benign (majority) count:
  natural       : original stratified test split
  moderate      : majority capped at 3 × second-class count
  attack_parity : majority capped at 1 × second-class count

For each distribution:
  1. Sample mini-batches from that distribution
  2. Generate two independent perturbed views (x', x'')
  3. Aggregate expert logits: agg = Σ_e σ(w_e) · logits_e(x')
  4. Loss  S = JSD(softmax(agg'), softmax(agg''))   ← MATI MSE → JSD for classification
  5. Adam minimises S over global w [E scalars]; model parameters frozen
  6. Learned w applied to full natural test set for evaluation

Usage
-----
  python src/code_tta_all.py --data data/cic2017_proc.pkl --num_experts 4
"""
import argparse
import logging
import os
import pickle
import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import (precision_recall_fscore_support, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

BLUE   = "#cce5ff"
RED    = "#ffcccc"
YELLOW = "#fff9cc"
GRAY   = "#f2f2f2"
WHITE  = "#ffffff"
EPS    = 0.001


# ── logging ───────────────────────────────────────────────────────────────────

def setup_logger(path):
    log = logging.getLogger("tta_all")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in [logging.StreamHandler(), logging.FileHandler(path)]:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def detect_device():
    try:
        import torch as _t
        if _t.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ── data ─────────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return (np.asarray(d["X"], dtype=np.float32),
            np.asarray(d["y"], dtype=int),
            d["label_encoder"],
            d.get("dataset_type", "cic2017"))


def split_data(X, y, seed):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ── inclusive expert caps ─────────────────────────────────────────────────────

def build_inclusive_caps(y_tr, n_cls, k, log):
    """
    k caps, one per expert.  Expert i's cap = count of the class at
    rank ceil(n_cls * i / k) when sorted by count descending.
    Expert 0 → cap = max count (no effective downsampling).
    """
    counts = np.bincount(y_tr, minlength=n_cls)
    sorted_counts = np.sort(counts)[::-1]
    caps = []
    for i in range(k):
        rank = min(int(np.ceil(n_cls * i / k)), n_cls - 1)
        caps.append(int(sorted_counts[rank]))
    log.info(f"Inclusive caps for {k} experts: {caps}")
    return caps


def subsample_to_cap(X, y, cap, n_cls, seed):
    """Each class: keep at most `cap` samples."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(n_cls):
        where = np.where(y == c)[0]
        if len(where) > cap:
            where = rng.choice(where, cap, replace=False)
        idx.append(where)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


# ── test distributions ────────────────────────────────────────────────────────

def build_test_distributions(X_te, y_te, n_cls, seed, log):
    """
    Three distributions that vary the majority-class count.
    Attack samples are always kept in full; only the majority class is
    downsampled.  Anchor = second-most-common class (typically dos-hulk).

      natural       : full test split
      moderate      : majority = 3 × anchor count
      attack_parity : majority = 1 × anchor count
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_te, minlength=n_cls)
    order = np.argsort(-counts)
    majority_cls = int(order[0])
    anchor_cls   = int(order[1])
    anchor_count = int(counts[anchor_cls])

    majority_idx = np.where(y_te == majority_cls)[0]
    other_idx    = np.where(y_te != majority_cls)[0]

    log.info(f"Majority class: idx={majority_cls} n={counts[majority_cls]:,} | "
             f"Anchor class: idx={anchor_cls} n={anchor_count:,}")

    def make_dist(cap):
        n = min(cap, len(majority_idx))
        chosen = rng.choice(majority_idx, n, replace=False)
        idx = np.concatenate([chosen, other_idx])
        rng.shuffle(idx)
        return X_te[idx], y_te[idx]

    dists = {
        "natural":       (X_te, y_te),
        "moderate":      make_dist(3 * anchor_count),
        "attack_parity": make_dist(anchor_count),
    }

    for name, (_, yd) in dists.items():
        c = np.bincount(yd, minlength=n_cls)
        log.info(f"  {name:15s}: total={len(yd):>8,}  "
                 f"majority={c[majority_cls]:>8,}  "
                 f"attacks={len(yd) - c[majority_cls]:>8,}")
    return dists


# ── XGBoost helpers ──────────────────────────────────────────────────────────

def _xgb_params(n_cls, n_est, device, seed):
    p = dict(n_estimators=n_est, max_depth=6, learning_rate=0.05,
             subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
             min_child_weight=1, tree_method="hist",
             early_stopping_rounds=20, random_state=seed, n_jobs=4)
    if device == "cuda":
        p["device"] = "cuda"
    p.update(objective="multi:softprob", num_class=n_cls, eval_metric="mlogloss")
    return p


def _balanced_w(y):
    counts = np.maximum(np.bincount(y), 1)
    return (len(y) / (len(counts) * counts[y])).astype(np.float32)


def _predict_proba(model, X, chunk=50_000):
    out = []
    for s in range(0, len(X), chunk):
        out.append(np.asarray(model.predict_proba(X[s:s + chunk]), dtype=np.float32))
    return np.concatenate(out)


def _predict_logits(model, X, n_cls, chunk=50_000):
    """Pre-softmax margin scores from XGBoost. Shape: [N, n_cls]."""
    booster = model.get_booster()
    out = []
    for s in range(0, len(X), chunk):
        raw = booster.predict(xgb.DMatrix(X[s:s + chunk]), output_margin=True)
        out.append(raw.reshape(-1, n_cls).astype(np.float32))
    return np.concatenate(out)


# ── perturbation ──────────────────────────────────────────────────────────────

def make_perturb_fn(mode, p_mask, noise_std, col_means, col_stds):
    if mode == "gaussian":
        scale = (col_stds * noise_std).astype(np.float32)
        def fn(X, rng):
            return X + rng.standard_normal(X.shape).astype(np.float32) * scale
    elif mode == "mask":
        means = col_means.astype(np.float32)
        def fn(X, rng):
            mask = rng.random(X.shape) < p_mask
            return np.where(mask, means, X)
    else:
        raise ValueError(f"Unknown perturb mode '{mode}'")
    return fn


# ── training ──────────────────────────────────────────────────────────────────

def train_baseline(X_tr, y_tr, X_va, y_va, n_cls, n_est, device, seed, log):
    log.info("Training baseline …")
    t0 = time.time()
    m = xgb.XGBClassifier(**_xgb_params(n_cls, n_est, device, seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_tr, y_tr, sample_weight=_balanced_w(y_tr),
              eval_set=[(X_va, y_va)], verbose=False)
    log.info(f"Baseline ready in {time.time() - t0:.1f}s")
    return m


def train_inclusive_experts(X_tr, y_tr, X_va, y_va, caps, n_cls,
                             n_est, device, seed, log):
    """
    Val set is also capped at the same level as training so early stopping
    uses a balanced signal instead of the natural imbalanced distribution.
    """
    t0 = time.time()
    models = []
    for i, cap in enumerate(caps):
        Xe, ye = subsample_to_cap(X_tr, y_tr, cap, n_cls, seed + i)
        Xv, yv = subsample_to_cap(X_va, y_va, cap, n_cls, seed + i + 1000)
        log.info(f"  Expert {i}: cap={cap:,}  "
                 f"train={len(ye):,}  val={len(yv):,}  "
                 f"class_dist={np.bincount(ye, minlength=n_cls).tolist()}")
        m = xgb.XGBClassifier(**_xgb_params(n_cls, n_est, device, seed + i))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xe, ye, sample_weight=_balanced_w(ye),
                  eval_set=[(Xv, yv)], verbose=False)
        models.append(m)
    log.info(f"All {len(caps)} inclusive experts in {time.time() - t0:.1f}s")
    return models


# ── oracle routing ────────────────────────────────────────────────────────────

def oracle_predict(models, X_te, y_te, log):
    """Best expert per sample = expert with highest true-class probability."""
    N = len(y_te)
    all_proba = np.stack([_predict_proba(m, X_te) for m in models], axis=0)
    true_proba = all_proba[:, np.arange(N), y_te]
    best_expert = np.argmax(true_proba, axis=0)
    preds = all_proba[best_expert, np.arange(N), :].argmax(axis=1)
    log.info(f"Oracle selection dist: {np.bincount(best_expert, minlength=len(models)).tolist()}")
    return preds


# ── MATI/SADE-style TTA ──────────────────────────────────────────────────────

def learn_aggregation_weights(models, X_dist, n_cls, perturb_fn,
                               n_epochs, lr, batch_size, seed, log):
    """
    Learn global aggregation weights w[E] by minimising JSD between predictions
    on two independently perturbed views of each mini-batch.

    MATI origin: S = (1/|Dt|) Σ (y^(1) - y^(2))^2  [regression, scalar output]
    Our adaptation for classification probability vectors:
      S = (1/|Dt|) Σ JSD(softmax(agg^(1)), softmax(agg^(2)))

    Gradient flows: loss → p^(1,2) → agg → w (linear).
    XGBoost logits are constants (no backprop through XGBoost needed).
    """
    E = len(models)
    rng = np.random.default_rng(seed)
    N = len(X_dist)

    aggregation_weight = torch.nn.Parameter(torch.zeros(E))
    optimizer = torch.optim.Adam([aggregation_weight], lr=lr)

    for epoch in range(n_epochs):
        idx = rng.permutation(N)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, N, batch_size):
            b_idx = idx[start:start + batch_size]
            X_b   = X_dist[b_idx]

            # two independently perturbed views
            Xv0 = perturb_fn(X_b, rng)
            Xv1 = perturb_fn(X_b, rng)

            # XGBoost logits are constants in the computation graph
            with torch.no_grad():
                lv0 = [torch.from_numpy(_predict_logits(m, Xv0, n_cls))
                       for m in models]
                lv1 = [torch.from_numpy(_predict_logits(m, Xv1, n_cls))
                       for m in models]

            w = F.softmax(aggregation_weight, dim=0)
            agg0 = sum(w[e] * lv0[e] for e in range(E))
            agg1 = sum(w[e] * lv1[e] for e in range(E))

            p0 = F.softmax(agg0, dim=1)
            p1 = F.softmax(agg1, dim=1)

            # JSD loss (symmetric, bounded [0, log2], proper for distributions)
            m_dist = 0.5 * (p0 + p1)
            eps = 1e-10
            kl0 = (p0 * (torch.log(p0 + eps) - torch.log(m_dist + eps))).sum(1)
            kl1 = (p1 * (torch.log(p1 + eps) - torch.log(m_dist + eps))).sum(1)
            loss = (0.5 * (kl0 + kl1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        w_np = F.softmax(aggregation_weight.detach(), dim=0).numpy()
        log.info(f"  epoch {epoch + 1}/{n_epochs}  "
                 f"JSD={epoch_loss / n_batches:.4f}  "
                 f"w={np.round(w_np, 3).tolist()}")

        if w_np.min() < 0.01:
            log.info(f"  Early stop: min weight {w_np.min():.3f} < 0.01")
            break

    return F.softmax(aggregation_weight.detach(), dim=0).numpy()


def predict_with_weights(models, X_te, n_cls, w):
    """Aggregate expert logits with global weights → argmax."""
    agg = np.zeros((len(X_te), n_cls), dtype=np.float64)
    for e, m in enumerate(models):
        agg += float(w[e]) * _predict_logits(m, X_te, n_cls)
    return np.argmax(agg, axis=1)


# ── v2: support-group-aware TTA ────────────────────────────────────────────────

def build_support_groups(y_tr, n_cls, class_names, log):
    """
    Split classes into many/medium/few support groups from train counts.

    SADE learns one vector for a whole test distribution. For IDS tails, that
    lets head classes dominate the self-supervised signal. v2 keeps SADE-style
    weights but learns one vector per support group.
    """
    counts = np.bincount(y_tr, minlength=n_cls)
    order = np.argsort(-counts)
    group_names = ["many", "medium", "few"]
    class_to_group = np.zeros(n_cls, dtype=int)

    for rank, c in enumerate(order):
        g = min(rank * len(group_names) // n_cls, len(group_names) - 1)
        class_to_group[c] = g

    log.info("Support groups (train-count quantiles):")
    for g, name in enumerate(group_names):
        cls = [i for i in range(n_cls) if class_to_group[i] == g]
        detail = ", ".join(f"{class_names[i]}({counts[i]:,})" for i in cls)
        log.info(f"  {name:6s}: {detail}")
    return group_names, class_to_group, counts


def balanced_indices_by_pseudo_label(pseudo_y, max_per_class, seed):
    """Return test-time indices balanced by pseudo label."""
    rng = np.random.default_rng(seed)
    idxs = []
    for c in np.unique(pseudo_y):
        where = np.where(pseudo_y == c)[0]
        if len(where) == 0:
            continue
        take = min(len(where), max_per_class)
        idxs.append(rng.choice(where, take, replace=False))
    if not idxs:
        return np.arange(len(pseudo_y))
    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    return idx


def learn_groupwise_weights(models, X_dist, pseudo_y_dist, class_to_group,
                            group_names, n_cls, perturb_fn, n_epochs, lr,
                            batch_size, seed, log, max_per_pseudo_class):
    """
    Learn W[G, E], one SADE-style expert weight vector per pseudo support group.

    Group membership uses pseudo labels from a frozen model, so this remains
    unlabeled test-time adaptation. Within each group, pseudo-class balancing
    limits domination by one frequent pseudo class.
    """
    E = len(models)
    G = len(group_names)
    W = np.zeros((G, E), dtype=np.float32)
    pseudo_group = class_to_group[pseudo_y_dist]

    for g, name in enumerate(group_names):
        where_g = np.where(pseudo_group == g)[0]
        if len(where_g) == 0:
            log.info(f"  group={name:6s}: empty; using uniform weights")
            W[g] = np.full(E, 1.0 / E, dtype=np.float32)
            continue

        rel_bal = balanced_indices_by_pseudo_label(
            pseudo_y_dist[where_g], max_per_pseudo_class, seed + 100 * (g + 1))
        idx = where_g[rel_bal]
        log.info(f"  group={name:6s}: pseudo_n={len(where_g):,} "
                 f"balanced_n={len(idx):,}")
        W[g] = learn_aggregation_weights(
            models, X_dist[idx], n_cls, perturb_fn,
            n_epochs, lr, batch_size, seed + 1000 * (g + 1), log)
    return W


def predict_with_group_weights(models, X_te, n_cls, W, pseudo_y, class_to_group):
    """Aggregate logits with the row of W selected by each sample's pseudo group."""
    pred = np.zeros(len(X_te), dtype=int)
    pseudo_group = class_to_group[pseudo_y]
    for g in range(W.shape[0]):
        mask = pseudo_group == g
        if not mask.any():
            continue
        agg = np.zeros((int(mask.sum()), n_cls), dtype=np.float64)
        Xg = X_te[mask]
        for e, m in enumerate(models):
            agg += float(W[g, e]) * _predict_logits(m, Xg, n_cls)
        pred[mask] = np.argmax(agg, axis=1)
    return pred


def save_group_weights(group_results, group_names, caps, out_dir, log):
    rows = []
    for dist_name, res in group_results.items():
        W = res["W"]
        for g, group_name in enumerate(group_names):
            row = {"distribution": dist_name, "group": group_name}
            row.update({f"E{e}(cap={caps[e]})": float(W[g, e])
                        for e in range(W.shape[1])})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "C_group_weights.csv"), index=False)
    log.info("(C2) Learned group-wise weights:")
    for _, r in df.iterrows():
        vals = [r[f"E{e}(cap={caps[e]})"] for e in range(len(caps))]
        log.info(f"  {r['distribution']:15s} {r['group']:6s}: "
                 f"{np.round(vals, 3).tolist()}")

    dist_names = list(group_results.keys())
    E = len(caps)
    fig, axes = plt.subplots(len(dist_names), 1,
                             figsize=(max(8, E * 1.5), 3.2 * len(dist_names)),
                             squeeze=False)
    colors = ["#607D8B", "#FF9800", "#4CAF50"]
    x = np.arange(E)
    bar_w = 0.8 / len(group_names)
    for ax, dist_name in zip(axes[:, 0], dist_names):
        W = group_results[dist_name]["W"]
        for g, group_name in enumerate(group_names):
            offset = bar_w * (g - len(group_names) / 2 + 0.5)
            ax.bar(x + offset, W[g], bar_w, label=group_name,
                   color=colors[g % len(colors)], alpha=0.85)
        ax.axhline(1 / E, color="red", ls="--", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"E{i}\n(cap={caps[i]:,})" for i in range(E)],
                           fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Weight")
        ax.set_title(f"Group-wise weights: {dist_name}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "C_group_weights.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("(C2) saved C_group_weights.png / .csv")


def save_pseudo_group_diagnostics(y_te, pseudo_y, class_to_group, group_names,
                                  class_names, out_dir, log):
    rows = []
    true_group = class_to_group[y_te]
    pseudo_group = class_to_group[pseudo_y]
    for c, cname in enumerate(class_names):
        mask = y_te == c
        if not mask.any():
            continue
        counts = np.bincount(pseudo_group[mask], minlength=len(group_names))
        row = {
            "true_class": cname,
            "support": int(mask.sum()),
            "true_group": group_names[int(class_to_group[c])],
            "pseudo_group_majority": group_names[int(counts.argmax())],
        }
        for g, gname in enumerate(group_names):
            row[f"pseudo_{gname}_count"] = int(counts[g])
            row[f"pseudo_{gname}_rate"] = float(counts[g] / mask.sum())
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "D_pseudo_group_confusion.csv"), index=False)
    acc = float((true_group == pseudo_group).mean())
    log.info(f"(D) Pseudo group accuracy vs true support group: {acc:.4f}")
    log.info("(D) saved D_pseudo_group_confusion.csv")


# ── (A) performance comparison ────────────────────────────────────────────────

def plot_A(y_te, y_pred_base, y_pred_oracle, tta_results, class_names, out_dir, log):
    """
    Per-class F1 bar chart: Baseline | Oracle | TTA-natural | TTA-moderate | TTA-attack_parity
    Also logs macro F1 summary for all distributions.
    """
    n_cls = len(class_names)

    def f1s(yp):
        _, _, f, s = precision_recall_fscore_support(
            y_te, yp, labels=np.arange(n_cls), zero_division=0)
        return f, s

    f_b, s = f1s(y_pred_base)
    f_o, _ = f1s(y_pred_oracle)
    order   = [i for i in np.argsort(-s) if s[i] > 0]
    x_pos   = np.arange(len(order))

    dist_names  = list(tta_results.keys())
    colors_tta  = ["#F44336", "#FF9800", "#4CAF50"]
    n_bars      = 2 + len(dist_names)   # baseline + oracle + TTAs
    bar_w       = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(14, len(order) * 1.1), 5))
    ax.bar(x_pos - bar_w * (n_bars / 2 - 0.5),
           [f_b[i] for i in order], bar_w, label="Baseline", color="#888888")
    ax.bar(x_pos - bar_w * (n_bars / 2 - 1.5),
           [f_o[i] for i in order], bar_w, label="Oracle",   color="#2196F3")
    for k, (dname, clr) in enumerate(zip(dist_names, colors_tta)):
        f_t, _ = f1s(tta_results[dname]["y_pred"])
        offset = -bar_w * (n_bars / 2 - 2.5 - k)
        ax.bar(x_pos + offset, [f_t[i] for i in order], bar_w,
               label=f"TTA-{dname}", color=clr)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{class_names[i]}\n(n={s[i]:,})" for i in order],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("(A) F1 per class: Baseline vs Oracle vs TTA per distribution\n"
                 "Majority-class benign count varies across TTA distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A_f1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # macro F1 summary log
    mb = f1_score(y_te, y_pred_base,   average="macro", zero_division=0)
    mo = f1_score(y_te, y_pred_oracle, average="macro", zero_division=0)
    log.info(f"(A) Macro F1 — Baseline={mb:.4f}  Oracle={mo:.4f}")
    for dname in dist_names:
        mt = f1_score(y_te, tta_results[dname]["y_pred"], average="macro", zero_division=0)
        log.info(f"    TTA-{dname:<15s} = {mt:.4f}  (Oracle−TTA gap = {mo - mt:.4f})")

    # CSV per distribution
    for dname in dist_names:
        f_t, _ = f1s(tta_results[dname]["y_pred"])
        pd.DataFrame({
            "class":       [class_names[i] for i in order],
            "support":     [int(s[i]) for i in order],
            "f1_baseline": [float(f_b[i]) for i in order],
            "f1_oracle":   [float(f_o[i]) for i in order],
            "f1_tta":      [float(f_t[i]) for i in order],
        }).assign(oracle_minus_tta=lambda d: d.f1_oracle - d.f1_tta
        ).to_csv(os.path.join(out_dir, f"A_f1_{dname}.csv"), index=False)
    log.info("(A) saved A_f1_comparison.png / A_f1_*.csv")


# ── (B) expert behavior + learned weights ─────────────────────────────────────

def plot_B(models, caps, X_te, y_te, tta_results, n_cls, class_names,
           perturb_fn, seed, out_dir, log):
    """
    For one representative class per expert (oracle-selected), show:
      pred(x) | conf(x) | pred(x') | conf(x') | w_natural | w_moderate | w_attack_parity

    Color coding:
      Oracle expert row → blue
      Others            → red tint by conf(x)
    """
    rng   = np.random.default_rng(seed + 999)
    Xa    = perturb_fn(X_te, rng)
    E     = len(models)
    counts = np.bincount(y_te, minlength=n_cls)

    pred_x_all, pred_xa_all, conf_x_all = [], [], []
    for m in models:
        lp  = _predict_proba(m, X_te)
        lpa = _predict_proba(m, Xa)
        pred_x_all.append(lp.argmax(axis=1))
        pred_xa_all.append(lpa.argmax(axis=1))
        conf_x_all.append(lp.max(axis=1))

    # oracle best expert per sample
    all_proba = np.stack([_predict_proba(m, X_te) for m in models], axis=0)
    true_proba = all_proba[:, np.arange(len(y_te)), y_te]
    best_expert_per_sample = np.argmax(true_proba, axis=0)

    # representative class per expert
    rep_classes, used = [], set()
    for e in range(E):
        mask_e = best_expert_per_sample == e
        if mask_e.any():
            cce = np.bincount(y_te[mask_e], minlength=n_cls)
            c = next((i for i in np.argsort(-cce)
                      if i not in used and counts[i] > 0), None)
        else:
            c = None
        if c is None:
            c = next((i for i in np.argsort(-counts) if i not in used), 0)
        used.add(c)
        rep_classes.append(class_names[c])

    log.info(f"(B) Representative classes: {rep_classes}")

    dist_names = list(tta_results.keys())
    w_cols  = [f"w_{d}" for d in dist_names]
    COLS    = ["expert (cap)", "pred(x)", "conf(x)", "pred(x')", *w_cols]
    OWNER_BG, OWNER_TEXT = "#c6dcf7", "#0a3d6b"
    HEADER_BG, HEADER_FG = "#404040", "white"

    rows_all = []
    n_rep = len(rep_classes)
    fig, axes = plt.subplots(n_rep, 1,
                             figsize=(13, 2.2 * n_rep + 1.2),
                             gridspec_kw={"hspace": 0.65})
    if n_rep == 1:
        axes = [axes]

    for ax, cls_name in zip(axes, rep_classes):
        ax.axis("off")
        c    = class_names.index(cls_name)
        mask = y_te == c
        if not mask.any():
            continue
        support  = int(counts[c])
        oracle_e = int(np.bincount(best_expert_per_sample[mask],
                                   minlength=E).argmax())

        cell_text, cell_colors = [], []
        for e in range(E):
            is_oracle = e == oracle_e
            px  = class_names[np.bincount(pred_x_all[e][mask],
                                          minlength=n_cls).argmax()]
            pxa = class_names[np.bincount(pred_xa_all[e][mask],
                                          minlength=n_cls).argmax()]
            cx  = float(conf_x_all[e][mask].mean())
            w_vals = [float(tta_results[d]["w"][e]) for d in dist_names]

            texts = [f"E{e}(cap={caps[e]:,})", px, f"{cx:.4f}", pxa,
                     *[f"{wv:.3f}" for wv in w_vals]]
            cell_text.append(texts)
            rows_all.append({
                "true_class": cls_name, "support": support,
                "expert_idx": e, "cap": caps[e], "oracle": is_oracle,
                "pred_x": px, "conf_x": cx, "pred_xa": pxa,
                **{f"w_{d}": float(tta_results[d]["w"][e]) for d in dist_names},
            })

            if is_oracle:
                colors = [OWNER_BG] * len(COLS)
            else:
                t = max(0.0, min(1.0, (cx - 0.80) / 0.20))
                g = int(240 - t * 100)
                colors = [f"#ff{g:02x}{g:02x}"] * len(COLS)
            cell_colors.append(colors)

        tbl = ax.table(cellText=cell_text, colLabels=COLS,
                       cellColours=cell_colors, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.auto_set_column_width(list(range(len(COLS))))
        for col_idx in range(len(COLS)):
            tbl[0, col_idx].set_facecolor(HEADER_BG)
            tbl[0, col_idx].set_text_props(color=HEADER_FG, fontweight="bold")
        for row_idx in range(E):
            if row_idx == oracle_e:
                for col_idx in range(len(COLS)):
                    tbl[row_idx + 1, col_idx].set_text_props(
                        color=OWNER_TEXT, fontweight="bold")
        ax.set_title(
            f'True class: "{cls_name}"  (n={support:,})  '
            f'Blue=oracle (E{oracle_e}, cap={caps[oracle_e]:,})\n'
            f'w_* columns: learned TTA weight per distribution',
            fontsize=9, loc="left", pad=4,
        )

    fig.suptitle(
        "(B) Expert behavior and learned weights per distribution\n"
        "w_natural / w_moderate / w_attack_parity — higher benign count → "
        "majority expert favoured",
        fontsize=10, y=1.0,
    )
    plt.savefig(os.path.join(out_dir, "B_expert_behavior.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows_all).to_csv(
        os.path.join(out_dir, "B_expert_behavior.csv"), index=False)
    log.info("(B) saved B_expert_behavior.png / .csv")


# ── (C) learned weight bar chart ──────────────────────────────────────────────

def plot_C(tta_results, caps, out_dir, log):
    """
    Grouped bar chart: x-axis = experts, groups = distributions.
    Shows how learned global weights shift as the majority-class count changes.
    Expected: majority expert (E0) gets high weight in natural distribution;
    balanced expert (Ek-1) rises in attack_parity distribution.
    """
    dist_names = list(tta_results.keys())
    E = len(caps)
    x = np.arange(E)
    bar_w = 0.8 / len(dist_names)
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(max(8, E * 1.5), 5))
    for k, (dname, clr) in enumerate(zip(dist_names, colors)):
        w = tta_results[dname]["w"]
        offset = bar_w * (k - len(dist_names) / 2 + 0.5)
        bars = ax.bar(x + offset, w, bar_w, label=dname, color=clr, alpha=0.85)
        for bar, wi in zip(bars, w):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{wi:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.axhline(1 / E, color="red", ls="--", lw=1.5, label=f"uniform (1/{E})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}\n(cap={caps[i]:,})" for i in range(E)], fontsize=9)
    ax.set_ylabel("Aggregation weight")
    ax.set_ylim(0, 1.0)
    ax.set_title("(C) Learned global aggregation weights per test distribution\n"
                 "Weight shift signals: majority-expert favoured when benign dominates, "
                 "balanced-expert rises with attack_parity")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "C_global_weights.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # log and CSV
    log.info("(C) Learned weights per distribution:")
    rows = []
    for dname in dist_names:
        w = tta_results[dname]["w"]
        log.info(f"  {dname:15s}: {np.round(w, 3).tolist()}")
        rows.append({"distribution": dname,
                     **{f"E{e}(cap={caps[e]})": float(w[e]) for e in range(E)}})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "C_global_weights.csv"), index=False)
    log.info("(C) saved C_global_weights.png / .csv")


# ── per-class results table ───────────────────────────────────────────────────

def save_colored_table(rows, col_headers, path, title=""):
    n_rows, n_cols = len(rows), len(col_headers)
    cell_text, cell_colors = [], []
    for row in rows:
        texts, colors = [], []
        for col in col_headers:
            val = row.get(col, "")
            texts.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            if col.endswith("(M)"):
                b = row.get(col.replace("(M)", "(B)"))
                m = row.get(col)
                if isinstance(b, float) and isinstance(m, float):
                    colors.append(BLUE if m > b + EPS else RED if m < b - EPS else YELLOW)
                else:
                    colors.append(WHITE)
            elif row.get("_footer") or col == "support":
                colors.append(GRAY)
            else:
                colors.append(WHITE)
        cell_text.append(texts)
        cell_colors.append(colors)

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.2), max(4, n_rows * 0.35)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_headers,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(n_cols)))
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_class_results(y_te, y_pred_base, y_pred_tta, class_names,
                            out_dir, log, suffix=""):
    n_cls = len(class_names)
    prec_b, rec_b, f1_b, sup = precision_recall_fscore_support(
        y_te, y_pred_base, labels=np.arange(n_cls), zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_te, y_pred_tta, labels=np.arange(n_cls), zero_division=0)

    order = [i for i in np.argsort(-sup) if sup[i] > 0]
    COL_HEADERS = ["class", "support",
                   "prec(B)", "prec(M)", "recall(B)", "recall(M)",
                   "f1(B)", "f1(M)", "delta_f1"]
    rows = []
    for i in order:
        rows.append({
            "class":     class_names[i],
            "support":   int(sup[i]),
            "prec(B)":   float(prec_b[i]),  "prec(M)":   float(prec_m[i]),
            "recall(B)": float(rec_b[i]),   "recall(M)": float(rec_m[i]),
            "f1(B)":     float(f1_b[i]),    "f1(M)":     float(f1_m[i]),
            "delta_f1":  float(f1_m[i] - f1_b[i]),
        })
    for avg_name in ["macro avg", "weighted avg"]:
        avg = avg_name.split()[0]
        fb = float(f1_score(y_te, y_pred_base, average=avg, zero_division=0))
        fm = float(f1_score(y_te, y_pred_tta,  average=avg, zero_division=0))
        rows.append({
            "class": avg_name, "_footer": True,
            "support":   int(sup.sum()),
            "prec(B)":   float(precision_score(y_te, y_pred_base, average=avg, zero_division=0)),
            "prec(M)":   float(precision_score(y_te, y_pred_tta,  average=avg, zero_division=0)),
            "recall(B)": float(recall_score(y_te, y_pred_base, average=avg, zero_division=0)),
            "recall(M)": float(recall_score(y_te, y_pred_tta,  average=avg, zero_division=0)),
            "f1(B)": fb, "f1(M)": fm, "delta_f1": fm - fb,
        })

    hdr = (f"{'class':40s} {'sup':>8} {'prec(B)':>8} {'prec(M)':>8} "
           f"{'rec(B)':>8} {'rec(M)':>8} {'f1(B)':>8} {'f1(M)':>8} {'Δf1':>8}")
    log.info(hdr)
    log.info("─" * len(hdr))
    for r in rows:
        mark = "↑" if r["delta_f1"] > EPS else ("↓" if r["delta_f1"] < -EPS else "=")
        log.info(f"{r['class']:40s} {str(r.get('support', '')):>8} "
                 f"{r['prec(B)']:>8.4f} {r['prec(M)']:>8.4f} "
                 f"{r['recall(B)']:>8.4f} {r['recall(M)']:>8.4f} "
                 f"{r['f1(B)']:>8.4f} {r['f1(M)']:>8.4f} "
                 f"{r['delta_f1']:>+8.4f} {mark}")

    stem = f"baseline_vs_moe_per_class{suffix}"
    pd.DataFrame([{k: v for k, v in r.items() if k != "_footer"}
                  for r in rows]).to_csv(
        os.path.join(out_dir, f"{stem}.csv"), index=False)
    save_colored_table(rows, COL_HEADERS,
                       os.path.join(out_dir, f"{stem}.png"),
                       title=f"Per-class: Baseline vs TTA-MoE{suffix}")
    log.info(f"saved {stem}.png / .csv")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         required=True)
    parser.add_argument("--num_experts",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n_estimators", type=int,   default=300)
    parser.add_argument("--perturb",      default="gaussian",
                        choices=["gaussian", "mask"])
    parser.add_argument("--p_mask",       type=float, default=0.3)
    parser.add_argument("--noise_std",    type=float, default=0.1)
    parser.add_argument("--tta_epochs",   type=int,   default=30,
                        help="Adam epochs for aggregation weight learning")
    parser.add_argument("--tta_lr",       type=float, default=1e-4,
                        help="Adam learning rate for aggregation weights")
    parser.add_argument("--tta_batch",    type=int,   default=256,
                        help="Mini-batch size for TTA weight learning")
    parser.add_argument("--group_max_per_pseudo_class", type=int, default=5000,
                        help="Max pseudo-labeled samples per class for group-wise TTA")
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", ts)
    os.makedirs(out_dir, exist_ok=True)
    log = setup_logger(os.path.join(out_dir, "experiment.log"))
    log.info(f"Args: {vars(args)}")

    device = detect_device()
    log.info(f"Device: {device}")

    X, y, le, dataset_type = load_data(args.data)
    class_names = list(le.classes_)
    n_cls = len(class_names)
    log.info(f"Loaded {X.shape[0]:,} samples | {X.shape[1]} features | "
             f"{n_cls} classes | dataset={dataset_type}")

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, args.seed)
    log.info(f"Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}")

    col_means  = X_te.mean(axis=0)
    col_stds   = X_te.std(axis=0) + 1e-8
    perturb_fn = make_perturb_fn(
        args.perturb, args.p_mask, args.noise_std, col_means, col_stds)
    log.info(f"Perturbation: mode={args.perturb}  "
             f"p_mask={args.p_mask}  noise_std={args.noise_std}")

    # ── caps & training ───────────────────────────────────────────────────────
    caps = build_inclusive_caps(y_tr, n_cls, args.num_experts, log)
    group_names, class_to_group, _ = build_support_groups(
        y_tr, n_cls, class_names, log)

    baseline     = train_baseline(X_tr, y_tr, X_va, y_va, n_cls,
                                  args.n_estimators, device, args.seed, log)
    y_pred_base  = np.asarray(baseline.predict(X_te), dtype=int)
    save_pseudo_group_diagnostics(y_te, y_pred_base, class_to_group,
                                  group_names, class_names, out_dir, log)

    models = train_inclusive_experts(X_tr, y_tr, X_va, y_va, caps, n_cls,
                                     args.n_estimators, device, args.seed, log)

    # ── oracle ────────────────────────────────────────────────────────────────
    log.info("Oracle routing …")
    y_pred_oracle = oracle_predict(models, X_te, y_te, log)

    # ── test distributions ────────────────────────────────────────────────────
    log.info("\n═══ Building test distributions ═══")
    distributions = build_test_distributions(X_te, y_te, n_cls, args.seed, log)

    # ── TTA: per-distribution weight learning ─────────────────────────────────
    tta_results = {}
    group_results = {}
    for dist_name, (X_dist, _) in distributions.items():
        log.info(f"\n═══ TTA weight learning: {dist_name} "
                 f"(n={len(X_dist):,}) ═══")
        w = learn_aggregation_weights(
            models, X_dist, n_cls, perturb_fn,
            args.tta_epochs, args.tta_lr, args.tta_batch, args.seed, log)
        # always predict on full natural test set for fair comparison
        y_pred = predict_with_weights(models, X_te, n_cls, w)
        tta_results[dist_name] = {"w": w, "y_pred": y_pred}

        pseudo_dist = np.asarray(baseline.predict(X_dist), dtype=int)
        log.info(f"\nGroup-wise TTA weight learning: {dist_name} "
                 f"(pseudo-balanced)")
        W = learn_groupwise_weights(
            models, X_dist, pseudo_dist, class_to_group, group_names,
            n_cls, perturb_fn, args.tta_epochs, args.tta_lr, args.tta_batch,
            args.seed, log, args.group_max_per_pseudo_class)
        y_pred_g = predict_with_group_weights(
            models, X_te, n_cls, W, y_pred_base, class_to_group)
        group_results[dist_name] = {"W": W, "y_pred": y_pred_g}

    # ── plots ─────────────────────────────────────────────────────────────────
    log.info("\n═══ (A) Performance ═══")
    plot_A(y_te, y_pred_base, y_pred_oracle, tta_results, class_names, out_dir, log)

    log.info("\n═══ (B) Expert behavior ═══")
    plot_B(models, caps, X_te, y_te, tta_results, n_cls, class_names,
           perturb_fn, args.seed, out_dir, log)

    log.info("\n═══ (C) Learned weights ═══")
    plot_C(tta_results, caps, out_dir, log)
    save_group_weights(group_results, group_names, caps, out_dir, log)

    for dist_name, res in tta_results.items():
        log.info(f"\n═══ Per-class results: {dist_name} ═══")
        save_per_class_results(y_te, y_pred_base, res["y_pred"], class_names,
                               out_dir, log, suffix=f"_{dist_name}")

    for dist_name, res in group_results.items():
        log.info(f"\nPer-class results: groupwise_{dist_name}")
        save_per_class_results(y_te, y_pred_base, res["y_pred"], class_names,
                               out_dir, log, suffix=f"_groupwise_{dist_name}")

    log.info(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()

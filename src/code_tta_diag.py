#!/usr/bin/env python3
"""
code_tta_diag.py  —  Diagnostic: why exclusive-partition MoE + TTA fails.

Three pieces of evidence
------------------------
(A) Performance gap
      Oracle routing  : owning expert selected by true class (upper bound).
      JS-TTA routing  : per-sample stability-weighted aggregation.
      Oracle >> TTA   →  expert quality is fine; routing is the bottleneck.

(B) Expert behavior table  ← MAIN STORY
      For representative true classes, shows what each expert predicts for
      the original input x AND a perturbed view x', with what confidence.
      Key observation: non-owning experts predict a WRONG class with nearly
      the same confidence and stability as the owning expert predicts correctly.
      TTA stability cannot tell them apart → routing is effectively random.

(C) Stability gap distribution
      Per-sample: stability_owner vs max(stability_non_owners).
      If the distribution of (stability_owner − max_non_owner) peaks near 0,
      TTA weight allocation is indistinguishable from uniform random routing.

Usage
-----
  python src/code_tta_diag.py --data data/cic2017_proc.pkl
"""
import argparse
import logging
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split

PARTITIONS_2017 = [
    ("Benign",     ["benign"]),
    ("DoS",        ["dos-"]),
    ("DDoS_Scan",  ["ddos", "portscan"]),
    ("BruteForce", ["ftp-patator", "ssh-patator"]),
    ("WebBot",     ["web-attack", "bot"]),
    ("Tail",       []),
]
PARTITIONS_2018 = [
    ("Benign",       ["normal"]),
    ("DoS",          ["dos-attacks"]),
    ("DDoS",         ["ddos-attack", "ddos-attacks"]),
    ("BruteForce",   ["ftp-bruteforce", "ssh-bruteforce"]),
    ("Bot",          ["bot"]),
    ("Infiltration", ["infilteration"]),
    ("Tail",         []),
]
TAIL_IR_THRESHOLD = 5_000

SWEEP_LEVELS = {
    "gaussian": [0.01, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00],
    "mask":     [0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
}


@dataclass
class ExpertConfig:
    name: str
    global_class_ids: list
    is_binary: bool


# ── logging ───────────────────────────────────────────────────────────────────

def setup_logger(path):
    log = logging.getLogger("tta_diag")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in [logging.StreamHandler(), logging.FileHandler(path)]:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
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


# ── expert config ─────────────────────────────────────────────────────────────

def _tail_ids(y):
    counts = np.bincount(y)
    ir = counts.max() / np.maximum(counts, 1)
    return {int(i) for i in np.where(ir >= TAIL_IR_THRESHOLD)[0]}


def build_expert_configs(class_names, y, dataset_type):
    parts = PARTITIONS_2017 if "2017" in dataset_type else PARTITIONS_2018
    tids = _tail_ids(y)
    cid_to_part = {}
    for cid, cname in enumerate(class_names):
        if cid in tids:
            cid_to_part[cid] = "Tail"; continue
        cn = cname.lower().strip()
        assigned = "Other"
        for pname, pats in parts:
            if pname == "Tail": continue
            if any(cn.startswith(p) or p in cn for p in pats):
                assigned = pname; break
        cid_to_part[cid] = assigned
    configs = []
    for pname in [p for p, _ in parts] + ["Other"]:
        gids = sorted(c for c, p in cid_to_part.items() if p == pname)
        if gids:
            configs.append(ExpertConfig(pname, gids, len(gids) == 1))
    return configs


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _xgb_params(n_cls, n_est, device, seed):
    p = dict(n_estimators=n_est, max_depth=6, learning_rate=0.05,
             subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
             min_child_weight=1, tree_method="hist",
             early_stopping_rounds=20, random_state=seed, n_jobs=4)
    if device == "cuda": p["device"] = "cuda"
    if n_cls == 2:
        p.update(objective="binary:logistic", eval_metric="logloss")
    else:
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


def make_perturb_fn(mode, p_mask, noise_std, col_means, col_stds):
    """
    Returns a perturbation function  f(X, rng) → X_perturbed.

    mode="gaussian"
        Add zero-mean Gaussian noise scaled per-feature:
        noise_std × col_std_d  for each dimension d.

    mode="mask"
        Replace each feature independently with probability p_mask
        with the column mean (BERT-style masked-feature imputation).
    """
    if mode == "gaussian":
        scale = (col_stds * noise_std).astype(np.float32)   # [D]
        def fn(X, rng):
            return X + (rng.standard_normal(X.shape).astype(np.float32) * scale)
    elif mode == "mask":
        means = col_means.astype(np.float32)                 # [D]
        def fn(X, rng):
            B, D = X.shape
            mask = rng.random((B, D)) < p_mask
            return np.where(mask, means, X)
    else:
        raise ValueError(f"Unknown perturb mode '{mode}'. "
                         "Choose from: gaussian, mask")
    return fn


def zero_pad(local_p, cfg, n_global):
    N = len(local_p)
    g = np.zeros((N, n_global), dtype=np.float32)
    if cfg.is_binary:
        g[:, cfg.global_class_ids[0]] = local_p[:, 1]
    else:
        for lid, gid in enumerate(cfg.global_class_ids):
            g[:, gid] = local_p[:, lid]
    return g


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


def train_all_experts(X_tr, y_tr, X_va, y_va, configs, n_est, device, seed, log):
    t0 = time.time()
    models = []
    for cfg in configs:
        if cfg.is_binary:
            y_b_tr = np.isin(y_tr, cfg.global_class_ids).astype(int)
            y_b_va = np.isin(y_va, cfg.global_class_ids).astype(int)
            m = xgb.XGBClassifier(**_xgb_params(2, n_est, device, seed))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_tr, y_b_tr, sample_weight=_balanced_w(y_b_tr),
                      eval_set=[(X_va, y_b_va)], verbose=False)
        else:
            gids = cfg.global_class_ids
            g2l = {g: l for l, g in enumerate(gids)}
            mk_tr = np.isin(y_tr, gids)
            mk_va = np.isin(y_va, gids)
            Xe_tr, ye_tr = X_tr[mk_tr], np.array([g2l[g] for g in y_tr[mk_tr]])
            Xe_va, ye_va = X_va[mk_va], np.array([g2l[g] for g in y_va[mk_va]])
            m = xgb.XGBClassifier(**_xgb_params(len(gids), n_est, device, seed))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xe_tr, ye_tr, sample_weight=_balanced_w(ye_tr),
                      eval_set=[(Xe_va, ye_va)], verbose=False)
        models.append(m)
        log.info(f"  [{cfg.name}] done | classes={cfg.global_class_ids}")
    log.info(f"All experts in {time.time() - t0:.1f}s")
    return models


# ── oracle + JS-TTA ───────────────────────────────────────────────────────────

def oracle_predict(models, configs, X_te, y_te, n_global):
    preds = np.full(len(y_te), -1, dtype=int)
    for m, cfg in zip(models, configs):
        mask = np.isin(y_te, cfg.global_class_ids)
        if not mask.any(): continue
        gp = zero_pad(_predict_proba(m, X_te[mask]), cfg, n_global)
        preds[mask] = np.argmax(gp, axis=1)
    return preds


def _js_batch(p, q, eps=1e-10):
    p, q = np.clip(p, eps, 1), np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m), axis=1) +
                  np.sum(q * np.log(q / m), axis=1))


def js_tta_predict(models, configs, X_te, n_global, n_views, perturb_fn, seed):
    rng = np.random.default_rng(seed)
    N, E = len(X_te), len(models)
    base = [_predict_proba(m, X_te) for m in models]
    js_sum = np.zeros((N, E), dtype=np.float64)
    for _ in range(n_views):
        Xa = perturb_fn(X_te, rng)
        for e, m in enumerate(models):
            js_sum[:, e] += _js_batch(base[e], _predict_proba(m, Xa))
    stab = 1.0 / (1.0 + js_sum / n_views)
    w = stab / stab.sum(axis=1, keepdims=True)
    gp = np.zeros((N, n_global), dtype=np.float32)
    for e, (cfg, lp) in enumerate(zip(configs, base)):
        gp += w[:, e:e + 1] * zero_pad(lp, cfg, n_global)
    return np.argmax(gp, axis=1), stab, base


# ── (A) performance comparison ────────────────────────────────────────────────

def plot_A_f1_comparison(y_te, y_pred_base, y_pred_oracle, y_pred_tta,
                         class_names, out_dir, log):
    n_cls = len(class_names)

    def f1s(y_pred):
        _, _, f, s = precision_recall_fscore_support(
            y_te, y_pred, labels=np.arange(n_cls), zero_division=0)
        return f, s

    f_b, s = f1s(y_pred_base)
    f_o, _ = f1s(y_pred_oracle)
    f_t, _ = f1s(y_pred_tta)
    order   = [i for i in np.argsort(-s) if s[i] > 0]
    labels  = [f"{class_names[i]}\n(n={s[i]:,})" for i in order]

    x, w = np.arange(len(order)), 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.95), 5))
    ax.bar(x - w, [f_b[i] for i in order], w, label="Baseline",      color="#888888")
    ax.bar(x,     [f_o[i] for i in order], w, label="Oracle routing", color="#2196F3")
    ax.bar(x + w, [f_t[i] for i in order], w, label="JS-TTA routing", color="#F44336")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("(A) F1 per class: Baseline vs Oracle routing vs JS-TTA routing\n"
                 "Oracle = owning expert selected by true label  (routing upper bound)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A_f1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    mb = f1_score(y_te, y_pred_base,   average="macro", zero_division=0)
    mo = f1_score(y_te, y_pred_oracle, average="macro", zero_division=0)
    mt = f1_score(y_te, y_pred_tta,    average="macro", zero_division=0)
    log.info(f"(A) Macro F1 — Baseline={mb:.4f}  Oracle={mo:.4f}  JS-TTA={mt:.4f}")
    log.info(f"    Oracle−TTA gap = {mo - mt:.4f}  "
             f"(routing failure, not expert quality)")

    pd.DataFrame({
        "class": [class_names[i] for i in order],
        "support": [int(s[i]) for i in order],
        "f1_baseline": [float(f_b[i]) for i in order],
        "f1_oracle":   [float(f_o[i]) for i in order],
        "f1_tta":      [float(f_t[i]) for i in order],
    }).assign(oracle_minus_tta=lambda d: d.f1_oracle - d.f1_tta
    ).to_csv(os.path.join(out_dir, "A_f1_comparison.csv"), index=False)
    log.info("(A) saved A_f1_comparison.png / .csv")


# ── (B) expert behavior table ─────────────────────────────────────────────────

def _top_pred_name(predictions, n_cls, class_names):
    """Most common predicted global class among a set of predictions."""
    return class_names[np.bincount(predictions, minlength=n_cls).argmax()]


def compute_behavior(models, configs, X_te, y_te, stab, n_global, class_names, perturb_fn, seed):
    """
    For each (true_class, expert) pair compute:
      pred_x   : most common global prediction on x (original)
      conf_x   : mean max-local-prob on x
      pred_xa  : most common global prediction on x' (one perturbed view)
      conf_xa  : mean max-local-prob on x'
      stability: mean JS stability (from stab matrix already computed)
      tta_weight: mean TTA weight assigned to this expert on these samples

    Returns a DataFrame sorted by true_class support descending.
    """
    rng = np.random.default_rng(seed + 999)
    Xa  = perturb_fn(X_te, rng)                    # one representative perturbed view
    n_cls = len(class_names)
    E     = len(configs)

    # per-expert predictions & confidence on x and x'
    pred_x_all  = []   # [E][N]
    conf_x_all  = []   # [E][N]
    pred_xa_all = []   # [E][N]
    conf_xa_all = []   # [E][N]
    for m, cfg in zip(models, configs):
        lp  = _predict_proba(m, X_te)
        lpa = _predict_proba(m, Xa)
        pred_x_all.append(np.argmax(zero_pad(lp,  cfg, n_global), axis=1))
        pred_xa_all.append(np.argmax(zero_pad(lpa, cfg, n_global), axis=1))
        conf_x_all.append(lp.max(axis=1))
        conf_xa_all.append(lpa.max(axis=1))

    # TTA weights (row-normalised stability) — reuse stab from js_tta_predict
    tta_w = stab / stab.sum(axis=1, keepdims=True)   # [N, E]

    counts = np.bincount(y_te, minlength=n_cls)
    rows = []
    for c in np.argsort(-counts):
        if counts[c] == 0: continue
        mask = y_te == c
        for e, cfg in enumerate(configs):
            is_owner = c in cfg.global_class_ids
            px   = pred_x_all[e][mask]
            pxa  = pred_xa_all[e][mask]
            cx   = conf_x_all[e][mask]
            cxa  = conf_xa_all[e][mask]
            stab_e = stab[mask, e]
            w_e    = tta_w[mask, e]
            rows.append({
                "true_class": class_names[c],
                "support":    int(counts[c]),
                "expert":     cfg.name,
                "owner":      is_owner,
                "pred_x":     _top_pred_name(px,  n_cls, class_names),
                "conf_x":     float(cx.mean()),
                "pred_xa":    _top_pred_name(pxa, n_cls, class_names),
                "conf_xa":    float(cxa.mean()),
                "stability":  float(stab_e.mean()),
                "tta_weight": float(w_e.mean()),
            })
    return pd.DataFrame(rows)


def _rep_classes_per_partition(configs, class_names, y_te):
    """
    Pick one representative class per partition:
    the class with the most test samples in that partition.
    """
    counts = np.bincount(y_te, minlength=len(class_names))
    rep = []
    for cfg in configs:
        best = max(cfg.global_class_ids, key=lambda c: counts[c])
        rep.append(class_names[best])
    return rep


def plot_B_expert_behavior(behavior_df, rep_classes, configs, out_dir, log):
    """
    For each representative true class: draw a color-coded table showing
    per-expert pred(x), conf(x), pred(x'), conf(x'), stability, TTA weight.

    Color coding per row:
      owning expert  → steel blue background   (✓)
      non-owning     → red tint scaled by conf(x): 0.9→light, 0.999→deep red  (✗)

    The visual message: non-owning expert rows are as red (high-conf wrong) as
    the owning expert row is blue (high-conf correct), stability and TTA weight
    are nearly identical → TTA cannot discriminate.
    """
    # internal keys (no special chars) → display headers
    COLS  = ["expert", "pred_x", "conf_x", "pred_xa", "conf_xa",
             "stability", "tta_weight"]
    HEADS = ["Expert", "Pred (x)", "Conf(x)", "Pred (x')", "Conf(x')",
             "Stability", "TTA weight"]

    n_rep = len(rep_classes)
    fig, axes = plt.subplots(n_rep, 1,
                             figsize=(13, 2.0 * n_rep + 1.2),
                             gridspec_kw={"hspace": 0.6})
    if n_rep == 1:
        axes = [axes]

    OWNER_BG    = "#c6dcf7"   # blue
    OWNER_TEXT  = "#0a3d6b"
    WRONG_LIGHT = "#fff0f0"   # near-white red (confidence ~0.8)
    WRONG_DEEP  = "#ffaaaa"   # deep red       (confidence ~0.999)
    HEADER_BG   = "#404040"
    HEADER_FG   = "white"

    E = len(configs)

    for ax, cls_name in zip(axes, rep_classes):
        ax.axis("off")
        sub = behavior_df[behavior_df["true_class"] == cls_name].copy()
        if sub.empty:
            continue

        support = int(sub["support"].iloc[0])
        cell_text   = []
        cell_colors = []

        for _, row in sub.iterrows():
            is_own = row["owner"]
            conf   = row["conf_x"]
            texts = [
                ("✓ " if is_own else "✗ ") + row["expert"],
                row["pred_x"],
                f"{row['conf_x']:.4f}",
                row["pred_xa"],
                f"{row['conf_xa']:.4f}",
                f"{row['stability']:.4f}",
                f"{row['tta_weight']:.4f}",
            ]
            cell_text.append(texts)

            if is_own:
                colors = [OWNER_BG] * len(COLS)
            else:
                # blend between light-red (conf=0.8) and deep-red (conf=1.0)
                t = max(0.0, min(1.0, (conf - 0.80) / 0.20))
                r = int(255)
                g = int(240 - t * 100)   # 240 → 140
                b = int(240 - t * 100)
                hex_col = f"#{r:02x}{g:02x}{b:02x}"
                colors = [hex_col] * len(COLS)

            cell_colors.append(colors)

        tbl = ax.table(
            cellText=cell_text,
            colLabels=HEADS,
            cellColours=cell_colors,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.auto_set_column_width(list(range(len(COLS))))

        # header row styling
        for col_idx in range(len(COLS)):
            cell = tbl[0, col_idx]
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight="bold")

        # owning expert row: bold text
        for row_idx, (_, row) in enumerate(sub.iterrows(), start=1):
            if row["owner"]:
                for col_idx in range(len(COLS)):
                    tbl[row_idx, col_idx].set_text_props(
                        color=OWNER_TEXT, fontweight="bold")

        ax.set_title(
            f'True class: "{cls_name}"  (test n={support:,})  '
            f'— owning expert highlighted in blue\n'
            f'Non-owning experts (red) predict wrong classes '
            f'with nearly identical confidence & stability → TTA weight ≈ 1/E',
            fontsize=9, loc="left", pad=4,
        )

    fig.suptitle(
        "(B) Expert behavior on x and x' for representative true classes\n"
        "If non-owning experts were uncertain on OOD inputs, "
        "their rows would be low-confidence. They are not.",
        fontsize=10, y=1.0,
    )
    plt.savefig(os.path.join(out_dir, "B_expert_behavior.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    behavior_df.to_csv(os.path.join(out_dir, "B_expert_behavior.csv"), index=False)
    log.info("(B) saved B_expert_behavior.png / .csv")

    # log representative rows for quick inspection
    log.info("(B) Expert behavior for representative classes:")
    for cls_name in rep_classes:
        log.info(f"\n  True class: {cls_name}")
        sub = behavior_df[behavior_df["true_class"] == cls_name]
        for _, r in sub.iterrows():
            marker = "✓" if r["owner"] else "✗"
            log.info(f"    [{marker}] {r['expert']:12s} | "
                     f"pred(x)={r['pred_x']:30s} conf={r['conf_x']:.4f} | "
                     f"pred(x')={r['pred_xa']:30s} conf={r['conf_xa']:.4f} | "
                     f"stab={r['stability']:.4f}  w={r['tta_weight']:.4f}")


# ── (C) stability gap distribution ────────────────────────────────────────────

def plot_C_stability_gap(stab, y_te, configs, class_names, out_dir, log):
    """
    Per sample: Δ = stability_owner − max(stability_non_owners).

    If TTA can route correctly, the owning expert should be clearly more stable:
    Δ >> 0 for most samples.

    If Δ ≈ 0: TTA assigns near-equal weight to all experts → routing is random.
    If Δ < 0: TTA actively prefers a wrong expert (worst case, seen for BruteForce).

    Shown globally and broken out by true class (support-sorted).
    """
    N, E = stab.shape
    n_cls = len(class_names)
    counts = np.bincount(y_te, minlength=n_cls)

    # build owner index per sample
    owner_idx = np.full(N, -1, dtype=int)
    for e, cfg in enumerate(configs):
        mask = np.isin(y_te, cfg.global_class_ids)
        owner_idx[mask] = e

    stab_owner    = stab[np.arange(N), owner_idx]
    # max stability among non-owning experts
    stab_nonowner = np.array([
        np.max([stab[i, e] for e in range(E) if e != owner_idx[i]])
        for i in range(N)
    ])
    delta = stab_owner - stab_nonowner   # positive = TTA prefers owner

    # ── global histogram ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.linspace(-0.3, 0.3, 61)
    ax.hist(delta, bins=bins, color="#888", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="Δ = 0 (random routing)")
    ax.axvline(delta.mean(), color="orange", lw=1.5,
               label=f"mean Δ = {delta.mean():.4f}")
    ax.set_xlabel("stability_owner − max(stability_non_owners)")
    ax.set_ylabel("# samples")
    ax.set_title("(C) Stability gap: owning vs best non-owning expert\n"
                 "Δ ≈ 0 → TTA weight is near-uniform → routing is random")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    pct_neg = (delta < 0).mean() * 100
    ax.text(0.02, 0.97,
            f"{pct_neg:.1f}% of samples: non-owning expert is MORE stable\n"
            f"(TTA actively routes to wrong expert)",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))

    # ── per-class mean Δ bar chart ─────────────────────────────────────────────
    ax2 = axes[1]
    order   = [i for i in np.argsort(-counts) if counts[i] > 0]
    mean_d  = [delta[y_te == c].mean() for c in order]
    colors  = ["#2196F3" if d > 0.01 else
               "#F44336" if d < 0    else "#FFC107"
               for d in mean_d]
    bars = ax2.barh([class_names[c] for c in order], mean_d, color=colors)
    ax2.axvline(0, color="black", lw=1.0)
    ax2.set_xlabel("mean(stability_owner − max stability_non_owner)")
    ax2.set_title("Per-class mean stability gap\n"
                  "Red bars: TTA prefers the WRONG expert on average")
    ax2.grid(axis="x", alpha=0.3)

    # annotate owner expert name on each bar
    for c, d, bar in zip(order, mean_d, bars.patches):
        owner_name = next(
            cfg.name for cfg in configs if c in cfg.global_class_ids)
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"  owner={owner_name}", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "C_stability_gap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"(C) Global mean Δ = {delta.mean():.4f}  "
             f"({pct_neg:.1f}% samples have Δ < 0)")
    log.info("(C) Per-class mean stability gap:")
    for c in order:
        d = delta[y_te == c].mean()
        owner_name = next(cfg.name for cfg in configs if c in cfg.global_class_ids)
        flag = "← TTA prefers wrong expert" if d < 0 else ""
        log.info(f"  {class_names[c]:35s}  Δ={d:+.4f}  owner={owner_name}  {flag}")

    pd.DataFrame({
        "class": [class_names[c] for c in order],
        "support": [int(counts[c]) for c in order],
        "mean_delta": [float(delta[y_te == c].mean()) for c in order],
        "pct_delta_negative": [float((delta[y_te == c] < 0).mean() * 100) for c in order],
        "owner_expert": [
            next(cfg.name for cfg in configs if c in cfg.global_class_ids)
            for c in order],
    }).to_csv(os.path.join(out_dir, "C_stability_gap.csv"), index=False)
    log.info("(C) saved C_stability_gap.png / .csv")


# ── (D) noise-sensitivity sweep ──────────────────────────────────────────────

def sweep_noise_sensitivity(models, configs, X_te, y_te, n_global, class_names,
                             rep_classes, col_means, col_stds, n_trials, sweep_sample, log):
    """
    For each (mode, level, rep_class, expert): collect pred(x'), conf(x'), flip_rate
    averaged over n_trials random perturbations.  Returns a per-row DataFrame
    matching the structure of compute_behavior so plot_D can draw B-style tables.
    """
    n_cls = len(class_names)

    if len(X_te) > sweep_sample:
        idx = np.random.default_rng(0).choice(len(X_te), sweep_sample, replace=False)
        Xs, ys = X_te[idx], y_te[idx]
    else:
        Xs, ys = X_te, y_te

    counts = np.bincount(ys, minlength=n_cls)

    log.info(f"  Precomputing base predictions on {len(Xs):,} samples …")
    base_preds, base_confs, base_proba_local = [], [], []
    for m, cfg in zip(models, configs):
        lp = _predict_proba(m, Xs)
        base_preds.append(np.argmax(zero_pad(lp, cfg, n_global), axis=1))
        base_confs.append(lp.max(axis=1))
        base_proba_local.append(lp)   # full local-space proba for JS computation

    rows = []
    for mode in ["gaussian", "mask"]:
        levels = SWEEP_LEVELS[mode]
        log.info(f"  [{mode}] {len(levels)} levels × {n_trials} trials …")
        for level in levels:
            # accumulate per-trial perturbed predictions, confidences, and full probas
            trial_preds  = {cfg.name: [] for cfg in configs}
            trial_confs  = {cfg.name: [] for cfg in configs}
            trial_probas = {cfg.name: [] for cfg in configs}
            for trial in range(n_trials):
                rng = np.random.default_rng(trial * 1009 + abs(hash(mode)) % 997)
                if mode == "gaussian":
                    pfn = make_perturb_fn(mode, 0.0, level, col_means, col_stds)
                else:
                    pfn = make_perturb_fn(mode, level, 0.0, col_means, col_stds)
                Xa = pfn(Xs, rng)
                for e, (m, cfg) in enumerate(zip(models, configs)):
                    lpa = _predict_proba(m, Xa)
                    trial_preds[cfg.name].append(np.argmax(zero_pad(lpa, cfg, n_global), axis=1))
                    trial_confs[cfg.name].append(lpa.max(axis=1))
                    trial_probas[cfg.name].append(lpa)

            # compute per-sample JS stability and TTA weights for this level
            js_means = {}
            for e, cfg in enumerate(configs):
                js_sum = np.zeros(len(Xs), dtype=np.float64)
                for t in range(n_trials):
                    js_sum += _js_batch(base_proba_local[e], trial_probas[cfg.name][t])
                js_means[cfg.name] = js_sum / n_trials
            stab_arr = np.stack(
                [1.0 / (1.0 + js_means[cfg.name]) for cfg in configs], axis=1)  # [Ns, E]
            tta_w_arr = stab_arr / stab_arr.sum(axis=1, keepdims=True)           # [Ns, E]

            for cls_name in rep_classes:
                c = class_names.index(cls_name)
                mask_c = ys == c
                if not mask_c.any():
                    continue
                for e, cfg in enumerate(configs):
                    is_owner = c in cfg.global_class_ids

                    pred_x = class_names[
                        np.bincount(base_preds[e][mask_c], minlength=n_cls).argmax()]
                    conf_x = float(base_confs[e][mask_c].mean())

                    all_xa = np.concatenate(
                        [trial_preds[cfg.name][t][mask_c] for t in range(n_trials)])
                    pred_xa = class_names[np.bincount(all_xa, minlength=n_cls).argmax()]
                    conf_xa = float(np.mean(
                        [trial_confs[cfg.name][t][mask_c].mean() for t in range(n_trials)]))
                    flip_rate = float(np.mean(
                        [(trial_preds[cfg.name][t][mask_c] != base_preds[e][mask_c]).mean()
                         for t in range(n_trials)]))

                    rows.append({
                        "mode":       mode,
                        "level":      level,
                        "true_class": cls_name,
                        "support":    int(counts[c]),
                        "expert":     cfg.name,
                        "owner":      is_owner,
                        "pred_x":     pred_x,
                        "conf_x":     conf_x,
                        "pred_xa":    pred_xa,
                        "conf_xa":    conf_xa,
                        "flip_rate":  flip_rate,
                        "stability":  float(stab_arr[mask_c, e].mean()),
                        "tta_weight": float(tta_w_arr[mask_c, e].mean()),
                    })

            avg = np.mean([
                (trial_preds[cfg.name][t] != base_preds[e]).mean()
                for e, cfg in enumerate(configs) for t in range(n_trials)])
            log.info(f"    {mode} level={level:.2f}  avg_flip={avg:.4f}")

    return pd.DataFrame(rows)


def plot_B_sweep_per_level(sweep_df, configs, rep_classes, out_dir, log):
    """
    For each (mode, level) in sweep_df, save a standalone B-style PNG.
    Columns: Expert | Pred(x) | Conf(x) | Pred(x') | Conf(x') | Flip%
    Color scheme identical to plot_B_expert_behavior.
    """
    COLS  = ["expert", "pred_x", "conf_x", "pred_xa", "conf_xa",
             "stability", "tta_weight", "flip_rate"]
    HEADS = ["Expert", "Pred (x)", "Conf(x)", "Pred (x')", "Conf(x')",
             "Stability", "TTA weight", "Flip%"]
    OWNER_BG   = "#c6dcf7"
    OWNER_TEXT = "#0a3d6b"
    HEADER_BG  = "#404040"
    HEADER_FG  = "white"
    param_name = {"gaussian": "noise_std", "mask": "p_mask"}

    for mode in ["gaussian", "mask"]:
        mode_sub = sweep_df[sweep_df["mode"] == mode]
        for level in sorted(mode_sub["level"].unique()):
            level_sub = mode_sub[mode_sub["level"] == level]
            n_rep = len(rep_classes)

            fig, axes = plt.subplots(n_rep, 1,
                                     figsize=(14, 2.1 * n_rep + 1.2),
                                     gridspec_kw={"hspace": 0.6})
            if n_rep == 1:
                axes = [axes]

            for ax, cls_name in zip(axes, rep_classes):
                ax.axis("off")
                sub = level_sub[level_sub["true_class"] == cls_name]
                if sub.empty:
                    continue
                support = int(sub["support"].iloc[0])

                cell_text, cell_colors = [], []
                for _, row in sub.iterrows():
                    is_own = bool(row["owner"])
                    conf   = float(row["conf_xa"])
                    texts = [
                        ("✓ " if is_own else "✗ ") + row["expert"],
                        row["pred_x"],
                        f"{row['conf_x']:.4f}",
                        row["pred_xa"],
                        f"{row['conf_xa']:.4f}",
                        f"{row['stability']:.4f}",
                        f"{row['tta_weight']:.4f}",
                        f"{row['flip_rate']:.1%}",
                    ]
                    cell_text.append(texts)
                    if is_own:
                        colors = [OWNER_BG] * len(COLS)
                    else:
                        t = max(0.0, min(1.0, (conf - 0.80) / 0.20))
                        g = int(240 - t * 100)
                        colors = [f"#ff{g:02x}{g:02x}"] * len(COLS)
                    cell_colors.append(colors)

                tbl = ax.table(cellText=cell_text, colLabels=HEADS,
                               cellColours=cell_colors,
                               loc="center", cellLoc="center")
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8.5)
                tbl.auto_set_column_width(list(range(len(COLS))))

                for col_idx in range(len(COLS)):
                    cell = tbl[0, col_idx]
                    cell.set_facecolor(HEADER_BG)
                    cell.set_text_props(color=HEADER_FG, fontweight="bold")
                for row_idx, (_, row) in enumerate(sub.iterrows(), start=1):
                    if row["owner"]:
                        for col_idx in range(len(COLS)):
                            tbl[row_idx, col_idx].set_text_props(
                                color=OWNER_TEXT, fontweight="bold")

                ax.set_title(
                    f'True class: "{cls_name}"  (n={support:,})  '
                    f'{param_name[mode]}={level:.2f}'
                    f'  — Blue=owning expert  Red=non-owning (deeper=more confident)',
                    fontsize=9, loc="left", pad=4,
                )

            fig.suptitle(
                f"(B) Expert behavior at {param_name[mode]}={level:.2f}\n"
                f"Blue=owning expert  Red=non-owning — key observation: "
                f"non-owning experts predict wrong with nearly identical confidence & flip rate",
                fontsize=10, y=1.0,
            )
            safe_lv = f"{level:.2f}".replace(".", "_")
            out_path = os.path.join(out_dir, f"B_sweep_{mode}_{safe_lv}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"(B) saved B_sweep_{mode}_{safe_lv}.png")


def plot_D_noise_sweep(sweep_df, configs, rep_classes, n_trials, out_dir, log):
    """
    B-style sweep table.  Rows: for each expert, three sub-rows:
      pred(x') / conf(x') / flip%
    Columns: noise levels.  One value per cell — no multi-line cramming.
    """
    OWNER_BG   = "#c6dcf7"
    OWNER_TEXT = "#0a3d6b"
    HEADER_BG  = "#404040"
    HEADER_FG  = "white"
    param_name = {"gaussian": "noise_std", "mask": "p_mask"}

    for mode in ["gaussian", "mask"]:
        mode_sub = sweep_df[sweep_df["mode"] == mode]
        levels   = sorted(mode_sub["level"].unique())
        col_hdrs = [f"{param_name[mode]}={lv:.2f}" for lv in levels]
        n_rep    = len(rep_classes)

        fig, axes = plt.subplots(
            n_rep, 1,
            figsize=(max(14, len(levels) * 2.2), 5.5 * n_rep + 1.2),
            gridspec_kw={"hspace": 0.65},
        )
        if n_rep == 1:
            axes = [axes]

        for ax, cls_name in zip(axes, rep_classes):
            ax.axis("off")
            cls_sub = mode_sub[mode_sub["true_class"] == cls_name]
            if cls_sub.empty:
                continue
            support = int(cls_sub["support"].iloc[0])

            cell_text   = []
            cell_colors = []
            row_labels  = []

            for cfg in configs:
                exp_sub = cls_sub[cls_sub["expert"] == cfg.name]
                if exp_sub.empty:
                    continue
                is_owner = bool(exp_sub["owner"].iloc[0])
                prefix   = "✓ " if is_owner else "✗ "

                pred_row, conf_row, stab_row, tta_row, flip_row = [], [], [], [], []
                pred_col, conf_col, stab_col, tta_col, flip_col = [], [], [], [], []

                for level in levels:
                    r = exp_sub[exp_sub["level"] == level]
                    if r.empty:
                        for lst in [pred_row, conf_row, stab_row, tta_row, flip_row]:
                            lst.append("")
                        for lst in [pred_col, conf_col, stab_col, tta_col, flip_col]:
                            lst.append("#ffffff")
                        continue
                    r = r.iloc[0]
                    pred_row.append(r["pred_xa"])
                    conf_row.append(f"{r['conf_xa']:.4f}")
                    stab_row.append(f"{r['stability']:.4f}")
                    tta_row.append(f"{r['tta_weight']:.4f}")
                    flip_row.append(f"{r['flip_rate']:.1%}")

                    if is_owner:
                        c = OWNER_BG
                    else:
                        t = max(0.0, min(1.0, (float(r["conf_xa"]) - 0.80) / 0.20))
                        g = int(240 - t * 100)
                        c = f"#ff{g:02x}{g:02x}"
                    for lst in [pred_col, conf_col, stab_col, tta_col, flip_col]:
                        lst.append(c)

                row_labels.extend([
                    f"{prefix}{cfg.name}  pred(x')",
                    f"{prefix}{cfg.name}  conf(x')",
                    f"{prefix}{cfg.name}  stability",
                    f"{prefix}{cfg.name}  TTA weight",
                    f"{prefix}{cfg.name}  flip%",
                ])
                cell_text.extend([pred_row, conf_row, stab_row, tta_row, flip_row])
                cell_colors.extend([pred_col, conf_col, stab_col, tta_col, flip_col])

            tbl = ax.table(
                cellText=cell_text, rowLabels=row_labels,
                colLabels=col_hdrs, cellColours=cell_colors,
                loc="center", cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.auto_set_column_width(list(range(len(col_hdrs))))

            for col_idx in range(len(col_hdrs)):
                cell = tbl[0, col_idx]
                cell.set_facecolor(HEADER_BG)
                cell.set_text_props(color=HEADER_FG, fontweight="bold")

            # bold text for owning expert sub-rows (5 sub-rows per expert)
            data_row = 1
            for cfg in configs:
                exp_sub = cls_sub[cls_sub["expert"] == cfg.name]
                if exp_sub.empty:
                    continue
                is_owner = bool(exp_sub["owner"].iloc[0])
                for _ in range(5):
                    if is_owner:
                        for col_idx in range(len(col_hdrs)):
                            tbl[data_row, col_idx].set_text_props(
                                color=OWNER_TEXT, fontweight="bold")
                    data_row += 1

            ax.set_title(
                f'True class: "{cls_name}"  (n={support:,})\n'
                f'Blue=owning expert  Red=non-owning (deeper=more confident)  '
                f'(mean over {n_trials} trials)',
                fontsize=8, loc="left", pad=4,
            )

        fig.suptitle(
            f"(D) Expert behavior across {mode} noise levels\n"
            f"Each expert: 5 rows — pred(x') / conf(x') / stability / TTA weight / flip%",
            fontsize=10, y=1.0,
        )
        out_path = os.path.join(out_dir, f"D_sweep_{mode}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"(D) saved D_sweep_{mode}.png")

    sweep_df.to_csv(os.path.join(out_dir, "D_noise_sweep.csv"), index=False)
    log.info("(D) saved D_noise_sweep.csv")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         required=True)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n_estimators", type=int,   default=300)
    parser.add_argument("--n_views",      type=int,   default=5)
    parser.add_argument("--perturb",      default="gaussian",
                        choices=["gaussian", "mask"],
                        help="Perturbation mode for TTA views "
                             "(gaussian: scaled noise, mask: column-mean imputation)")
    parser.add_argument("--p_mask",       type=float, default=0.3,
                        help="Masking probability (mask mode)")
    parser.add_argument("--noise_std",    type=float, default=0.1,
                        help="Noise magnitude as fraction of per-feature std "
                             "(gaussian mode only)")
    parser.add_argument("--n_sweep_trials", type=int, default=10,
                        help="Number of random trials per (method, level) in sweep (D)")
    parser.add_argument("--sweep_sample",   type=int, default=5_000,
                        help="Max test samples used in sweep (subsampled for speed)")
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

    configs = build_expert_configs(class_names, y, dataset_type)
    log.info(f"Expert partition ({len(configs)} experts):")
    for c in configs:
        log.info(f"  [{c.name:12s}] {[class_names[i] for i in c.global_class_ids]}")

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, args.seed)
    log.info(f"Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}")

    # ── perturbation function ─────────────────────────────────────────────────
    col_means = X_te.mean(axis=0)
    col_stds  = X_te.std(axis=0) + 1e-8    # avoid div-by-zero for constant features
    perturb_fn = make_perturb_fn(
        args.perturb, args.p_mask, args.noise_std, col_means, col_stds)
    log.info(f"Perturbation: mode={args.perturb}  p_mask={args.p_mask}  "
             f"noise_std={args.noise_std}")

    # ── train ─────────────────────────────────────────────────────────────────
    baseline = train_baseline(X_tr, y_tr, X_va, y_va, n_cls,
                              args.n_estimators, device, args.seed, log)
    y_pred_base = np.asarray(baseline.predict(X_te), dtype=int)

    models = train_all_experts(X_tr, y_tr, X_va, y_va, configs,
                               args.n_estimators, device, args.seed, log)

    # ── oracle ────────────────────────────────────────────────────────────────
    log.info("Oracle routing …")
    y_pred_oracle = oracle_predict(models, configs, X_te, y_te, n_cls)

    # ── JS-TTA + stability matrix ─────────────────────────────────────────────
    log.info(f"JS-TTA (n_views={args.n_views}, perturb={args.perturb}) …")
    t0 = time.time()
    y_pred_tta, stab, base_probs = js_tta_predict(
        models, configs, X_te, n_cls, args.n_views, perturb_fn, args.seed)
    log.info(f"JS-TTA done in {time.time() - t0:.1f}s")

    # ── plots ─────────────────────────────────────────────────────────────────
    log.info("\n═══ (A) Performance ═══")
    plot_A_f1_comparison(y_te, y_pred_base, y_pred_oracle, y_pred_tta,
                         class_names, out_dir, log)

    log.info("\n═══ (B) Expert behavior on x and x' ═══")
    behavior_df = compute_behavior(
        models, configs, X_te, y_te, stab, n_cls, class_names,
        perturb_fn, args.seed)
    rep_classes = _rep_classes_per_partition(configs, class_names, y_te)
    plot_B_expert_behavior(behavior_df, rep_classes, configs, out_dir, log)

    log.info("\n═══ (C) Stability gap distribution ═══")
    plot_C_stability_gap(stab, y_te, configs, class_names, out_dir, log)

    log.info("\n═══ (D) Noise-sensitivity sweep ═══")
    sweep_df = sweep_noise_sensitivity(
        models, configs, X_te, y_te, n_cls, class_names,
        rep_classes, col_means, col_stds, args.n_sweep_trials, args.sweep_sample, log)
    plot_D_noise_sweep(sweep_df, configs, rep_classes, args.n_sweep_trials, out_dir, log)

    log.info("\n═══ (B) per noise level ═══")
    plot_B_sweep_per_level(sweep_df, configs, rep_classes, out_dir, log)

    log.info(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()

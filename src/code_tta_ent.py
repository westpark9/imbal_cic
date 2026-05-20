#!/usr/bin/env python3
"""
code_tta_ent.py — Exclusive-partition MoE + entropy-based stability TTA.

Expert design: identical to code_tta_js.py (exclusive partition, binary benign
  detector, zero-padded local→global probability mapping).

TTA — per-sample entropy stability:
  For each test sample x and K VIME views x'_1..K:
    ent_e(x) = (1/K) Σ_k H(p_e(x'_k))        (mean entropy across augmented views)
    stability_e(x) = 1 / (1 + ent_e(x))
    w_e(x)         = stability_e(x) / Σ_e' stability_e'(x)
  Final: argmax(Σ_e w_e(x) · zero_pad(p_e(x)))

  Entropy measures *prediction confidence* across augmented views.
  An expert that is consistently confident (low entropy) gets high weight,
  regardless of whether its confident prediction is CORRECT.

Comparison with code_tta_js:
  JS  — rewards consistency: "does the prediction CHANGE across views?"
  Ent — rewards confidence:  "is the prediction CERTAIN across views?"
  The key question: can a wrong expert be confidently stable?
  If yes, entropy weights will fail; JS weights may still discriminate.

Baseline: XGBoost with global balanced sample weights.

Usage:
  python src/code_tta_ent.py --data cic2017_proc.pkl --model 2
"""
import argparse
import gc
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
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

BLUE, RED, YELLOW, GRAY, WHITE = "#cce5ff", "#ffcccc", "#fff9cc", "#f2f2f2", "#ffffff"
EPS = 0.001

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
TAIL_IR_THRESHOLD = 5000


@dataclass
class ExpertConfig:
    name: str
    global_class_ids: list
    is_binary: bool


# ── logging ─────────────────────────────────────────────────────────────────

def setup_logger(path):
    log = logging.getLogger("tta_ent")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in [logging.StreamHandler(), logging.FileHandler(path)]:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


# ── device ───────────────────────────────────────────────────────────────────

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
    tids  = _tail_ids(y)

    cid_to_part = {}
    for cid, cname in enumerate(class_names):
        if cid in tids:
            cid_to_part[cid] = "Tail"
            continue
        cn = cname.lower().strip()
        assigned = "Other"
        for pname, pats in parts:
            if pname == "Tail":
                continue
            if any(cn.startswith(p) or p in cn for p in pats):
                assigned = pname
                break
        cid_to_part[cid] = assigned

    configs = []
    seen_parts = [p for p, _ in parts] + ["Other"]
    for pname in seen_parts:
        gids = sorted(c for c, p in cid_to_part.items() if p == pname)
        if not gids:
            continue
        configs.append(ExpertConfig(
            name=pname,
            global_class_ids=gids,
            is_binary=(len(gids) == 1),
        ))
    return configs, tids


# ── XGBoost helpers ───────────────────────────────────────────────────────────

def _xgb_params(n_cls, n_est, device, seed):
    p = dict(n_estimators=n_est, max_depth=6, learning_rate=0.05,
             subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
             min_child_weight=1, tree_method="hist",
             early_stopping_rounds=20, random_state=seed, n_jobs=4)
    if device == "cuda":
        p["device"] = "cuda"
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


# ── VIME perturbation ─────────────────────────────────────────────────────────

def vime_perturb(X, p_mask, rng):
    B, D = X.shape
    mask = rng.random((B, D)) < p_mask
    ref  = rng.integers(0, B, (B, D))
    return np.where(mask, X[ref, np.arange(D)], X)


# ── baseline ──────────────────────────────────────────────────────────────────

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


# ── exclusive expert training ─────────────────────────────────────────────────

def train_expert(X_tr, y_tr, X_va, y_va, cfg, n_est, device, seed, log):
    t0 = time.time()
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
        g2l  = {g: l for l, g in enumerate(gids)}
        mk_tr = np.isin(y_tr, gids)
        mk_va = np.isin(y_va, gids)
        Xe_tr = X_tr[mk_tr]
        ye_tr = np.array([g2l[g] for g in y_tr[mk_tr]])
        Xe_va = X_va[mk_va]
        ye_va = np.array([g2l[g] for g in y_va[mk_va]])
        if len(Xe_tr) < 20:
            log.warning(f"  [{cfg.name}] only {len(Xe_tr)} training samples — may overfit")
        m = xgb.XGBClassifier(**_xgb_params(len(gids), n_est, device, seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xe_tr, ye_tr, sample_weight=_balanced_w(ye_tr),
                  eval_set=[(Xe_va, ye_va)], verbose=False)
    log.info(f"  [{cfg.name}] {time.time() - t0:.1f}s | "
             f"classes={cfg.global_class_ids} | binary={cfg.is_binary}")
    return m


def train_all_experts(X_tr, y_tr, X_va, y_va, configs, n_est, device, seed, log):
    return [train_expert(X_tr, y_tr, X_va, y_va, c, n_est, device, seed, log)
            for c in configs]


# ── local → global probability mapping ───────────────────────────────────────

def zero_pad(local_p, cfg, n_global):
    N = len(local_p)
    g = np.zeros((N, n_global), dtype=np.float32)
    if cfg.is_binary:
        g[:, cfg.global_class_ids[0]] = local_p[:, 1]
    else:
        for lid, gid in enumerate(cfg.global_class_ids):
            g[:, gid] = local_p[:, lid]
    return g


# ── entropy stability ─────────────────────────────────────────────────────────

def _entropy_batch(p, eps=1e-10):
    """Per-sample Shannon entropy. [N,C] → [N]."""
    p = np.clip(p, eps, 1)
    return -np.sum(p * np.log(p), axis=1)


def compute_ent_weights(models, configs, X_b, n_views, p_mask, rng):
    """
    Per-sample per-expert weights via normalized mean entropy across K augmented views.

    ent_e(x)      = (1/K) Σ_k H(p_e(x'_k)) / log(n_local_classes_e)
    stability_e   = 1 / (1 + ent_e(x))
    weights       = row-normalised stability  [B, E]

    Entropy is normalised by log(n_local_classes) so experts with different
    class-space sizes are comparable. Without this, a 4-class expert always
    has higher raw entropy than a 2-class expert even at equal uncertainty,
    giving it systematically lower stability weights.

    Returns:
        weights   [B, E]
        base_local list of [B, C_e] local probs for ORIGINAL x (for aggregation)
    """
    B, E = len(X_b), len(models)
    base = [_predict_proba(m, X_b, chunk=B) for m in models]
    ent  = np.zeros((B, E), dtype=np.float64)

    # max-entropy normaliser per expert: log(n_local_classes)
    h_max = np.array([
        np.log(2 if c.is_binary else len(c.global_class_ids))
        for c in configs
    ], dtype=np.float64)                        # [E]

    for _ in range(n_views):
        Xa = vime_perturb(X_b, p_mask, rng)
        for e, m in enumerate(models):
            raw_h = _entropy_batch(_predict_proba(m, Xa, chunk=B))  # [B]
            ent[:, e] += raw_h / h_max[e]       # normalised to [0, 1]

    stab = 1.0 / (1.0 + ent / n_views)         # [B, E]  higher = more confident
    return stab / stab.sum(axis=1, keepdims=True), base


# ── TTA prediction ────────────────────────────────────────────────────────────

def tta_predict(models, configs, X_te, n_global, n_views, p_mask, seed,
                batch_size=50_000, log=None):
    """
    Per-sample adaptive aggregation via entropy stability.
    Returns: predictions [N], weight_matrix [N, E]
    """
    rng = np.random.default_rng(seed)
    preds, ws = [], []

    for s in range(0, len(X_te), batch_size):
        Xb      = X_te[s:s + batch_size]
        w, base = compute_ent_weights(models, configs, Xb, n_views, p_mask, rng)
        ws.append(w)

        gp = np.zeros((len(Xb), n_global), dtype=np.float32)
        for e, (c, lp) in enumerate(zip(configs, base)):
            gp += w[:, e:e + 1] * zero_pad(lp, c, n_global)
        preds.append(np.argmax(gp, axis=1))

    if log is not None:
        log.info("TTA weight stats (mean per expert across all test samples):")
        mean_w = np.concatenate(ws, axis=0).mean(axis=0)
        for c, mw in zip(configs, mean_w):
            log.info(f"  [{c.name}]  mean_w={mw:.4f}")

    return np.concatenate(preds), np.concatenate(ws, axis=0)


# ── output helpers ────────────────────────────────────────────────────────────

def sorted_report(y_true, y_pred, class_names):
    n = len(class_names)
    counts = np.bincount(y_true, minlength=n)
    order  = np.argsort(-counts)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n), zero_division=0)
    w = max(len(c) for c in class_names) + 2
    lines = [f"{'':>{w}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}", ""]
    for ci in order:
        if counts[ci] == 0:
            continue
        lines.append(f"{class_names[ci]:>{w}}  {p[ci]:9.4f}  {r[ci]:9.4f}  {f[ci]:9.4f}  {s[ci]:>9,}")
    lines.append("")
    for avg, lbl in [("macro", "macro avg"), ("weighted", "weighted avg")]:
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        lines.append(f"{lbl:>{w}}  {pa:9.4f}  {ra:9.4f}  {fa:9.4f}  {len(y_true):>9,}")
    return "\n".join(lines)


def save_colored_table(rows, col_headers, path, title=""):
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
    n_r, n_c = len(rows), len(col_headers)
    fig, ax = plt.subplots(figsize=(max(14, n_c * 1.3), max(4, n_r * 0.35)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_headers,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(n_c)))
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report_rows(y_true, y_pred_b, y_pred_m, class_names):
    n = len(class_names)
    counts = np.bincount(y_true, minlength=n)
    order  = np.argsort(-counts)
    p_b, r_b, f_b, _ = precision_recall_fscore_support(
        y_true, y_pred_b, labels=np.arange(n), zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred_m, labels=np.arange(n), zero_division=0)
    cols = ["class", "support", "prec(B)", "prec(M)",
            "recall(B)", "recall(M)", "f1(B)", "f1(M)"]
    rows = []
    for ci in order:
        if counts[ci] == 0:
            continue
        rows.append({"class": class_names[ci], "support": int(counts[ci]),
                     "prec(B)": float(p_b[ci]), "prec(M)": float(p_m[ci]),
                     "recall(B)": float(r_b[ci]), "recall(M)": float(r_m[ci]),
                     "f1(B)": float(f_b[ci]), "f1(M)": float(f_m[ci])})
    for avg, lbl in [("macro", "macro avg"), ("weighted", "weighted avg")]:
        pB, rB, fB, _ = precision_recall_fscore_support(y_true, y_pred_b, average=avg, zero_division=0)
        pM, rM, fM, _ = precision_recall_fscore_support(y_true, y_pred_m, average=avg, zero_division=0)
        rows.append({"class": lbl, "support": len(y_true),
                     "prec(B)": float(pB), "prec(M)": float(pM),
                     "recall(B)": float(rB), "recall(M)": float(rM),
                     "f1(B)": float(fB), "f1(M)": float(fM),
                     "_footer": True})
    return rows, cols


def save_weight_heatmap(weight_matrix, configs, y_te, class_names, out_dir, log):
    """
    Heatmap: rows = true class (by support desc), cols = expert,
             cell = mean entropy-stability weight for that true class.

    If entropy rewards confidence-over-correctness, wrong experts with
    small local class spaces (low entropy by construction) may dominate.
    This heatmap makes that visible.
    """
    E   = len(configs)
    n_c = len(class_names)
    counts = np.bincount(y_te, minlength=n_c)
    order  = np.argsort(-counts)

    mean_w = np.zeros((n_c, E), dtype=np.float64)
    for ci in range(n_c):
        mask = y_te == ci
        if mask.sum() > 0:
            mean_w[ci, :] = weight_matrix[mask].mean(axis=0)

    mean_w_ordered = mean_w[order, :]

    fig, ax = plt.subplots(figsize=(max(8, E * 1.8), max(5, n_c * 0.55 + 2)))
    im = ax.imshow(mean_w_ordered, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(E))
    ax.set_xticklabels([f"{c.name}\n{c.global_class_ids}" for c in configs],
                       fontsize=7)
    ax.set_yticks(range(n_c))
    ax.set_yticklabels(
        [f"{class_names[order[i]]}  (n={counts[order[i]]:,})" for i in range(n_c)],
        fontsize=7)
    ax.set_xlabel("Expert (owned global class IDs)", fontsize=9)
    ax.set_ylabel("True class (sorted by test support ↓)", fontsize=9)
    ax.set_title("Mean entropy-stability weight per true class × expert", fontsize=11)

    for ci in range(n_c):
        for e in range(E):
            v = mean_w_ordered[ci, e]
            ax.text(e, ci, f"{v:.3f}", ha="center", va="center",
                    fontsize=6, color="white" if v > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01).set_label("mean weight", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "weight_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Weight heatmap → {path}")

    df = pd.DataFrame(mean_w_ordered,
                      index=[class_names[order[i]] for i in range(n_c)],
                      columns=[c.name for c in configs])
    df.index.name = "true_class"
    df.to_csv(os.path.join(out_dir, "weight_heatmap.csv"))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exclusive-partition MoE + entropy-stability TTA for IDS")
    parser.add_argument("--data",         required=True)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n_estimators", type=int,   default=300)
    parser.add_argument("--n_views",      type=int,   default=5,
                        help="VIME augmented views for entropy averaging (default 5)")
    parser.add_argument("--p_mask",       type=float, default=0.3,
                        help="VIME feature-mask probability (default 0.3; 0.1 was too small for XGBoost)")
    parser.add_argument("--batch_size",   type=int,   default=4096)
    parser.add_argument("--model",        type=int,   default=2, choices=[0, 1, 2],
                        help="0=baseline only  1=MoE only  2=both")
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
    n_cls       = len(class_names)
    log.info(f"Loaded {X.shape[0]:,} samples | {X.shape[1]} features | "
             f"{n_cls} classes | dataset={dataset_type}")

    configs, tail_ids = build_expert_configs(class_names, y, dataset_type)
    log.info(f"Tail IDs (IR≥{TAIL_IR_THRESHOLD:,}): {sorted(tail_ids)}")
    log.info(f"Expert partition ({len(configs)} experts):")
    for c in configs:
        names = [class_names[i] for i in c.global_class_ids]
        log.info(f"  [{c.name:12s}] binary={c.is_binary} | {names}")

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, args.seed)
    log.info(f"Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}")

    # ── baseline ──────────────────────────────────────────────────────────────
    y_pred_b = None
    if args.model in (0, 2):
        baseline = train_baseline(
            X_tr, y_tr, X_va, y_va, n_cls, args.n_estimators, device, args.seed, log)
        y_pred_b = np.asarray(baseline.predict(X_te), dtype=int)
        log.info("\n── Baseline ──")
        log.info("\n" + sorted_report(y_te, y_pred_b, class_names))
        del baseline; gc.collect()

    # ── exclusive MoE + entropy TTA ───────────────────────────────────────────
    y_pred_m = None
    if args.model in (1, 2):
        t0 = time.time()
        models = train_all_experts(
            X_tr, y_tr, X_va, y_va, configs,
            args.n_estimators, device, args.seed, log)
        log.info(f"All experts trained in {time.time() - t0:.1f}s")

        t1 = time.time()
        log.info(f"Entropy-stability TTA (normalized): n_views={args.n_views}, p_mask={args.p_mask}")
        y_pred_m, weight_matrix = tta_predict(
            models, configs, X_te, n_cls,
            args.n_views, args.p_mask, args.seed,
            args.batch_size, log)
        log.info(f"TTA done in {time.time() - t1:.1f}s")

        save_weight_heatmap(weight_matrix, configs, y_te, class_names, out_dir, log)

        log.info("\n── Exclusive MoE + Entropy-TTA ──")
        log.info("\n" + sorted_report(y_te, y_pred_m, class_names))

    # ── comparison table ──────────────────────────────────────────────────────
    if args.model == 2 and y_pred_b is not None and y_pred_m is not None:
        rows, col_headers = build_report_rows(y_te, y_pred_b, y_pred_m, class_names)

        png = os.path.join(out_dir, "baseline_vs_moe_per_class.png")
        save_colored_table(rows, col_headers, png,
                           title=f"Exclusive MoE + Entropy-TTA | seed={args.seed}")
        log.info(f"PNG → {png}")

        df = pd.DataFrame(rows)
        df["delta_f1"] = df["f1(M)"] - df["f1(B)"]
        csv = os.path.join(out_dir, "baseline_vs_moe_per_class.csv")
        df.to_csv(csv, index=False)
        log.info(f"CSV → {csv}")

    log.info(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()

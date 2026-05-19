#!/usr/bin/env python3
"""
code_tta.py — SADE-style TTA on taxonomy-based MoE for IDS imbalanced classification.

Expert design : CIC-IDS2017 attack family taxonomy (6 experts), same as code_mati.py.
                Each expert trains on ALL data with balanced sample weights;
                focus classes receive a focus_weight multiplier.
                All experts output full n_classes probability vectors.

Test-time agg.: SADE (NeurIPS 2022) prediction stability maximisation.
                Two VIME-perturbed views per sample.
                Maximise  S = (1/B) Σ_b ŷ¹_b · ŷ²_b
                Theorem 1 (SADE): S ∝ I(Ŷ;Y) − H(Ŷ)
                → maximising stability promotes correct (high MI) and
                  confident (low entropy) ensemble predictions.
                All E experts optimised freely — no fixed tail weight.

Baseline       : standard XGBoost with balanced sample weights.

Usage:
  python src/code_tta.py --data cic2017_proc.pkl --model 2
"""
import argparse
import gc
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
import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ─── colour constants (CLAUDE.md spec) ────────────────────────────────────────
BLUE   = "#cce5ff"
RED    = "#ffcccc"
YELLOW = "#fff9cc"
GRAY   = "#f2f2f2"
WHITE  = "#ffffff"
EPS    = 0.001

# ─── CIC-IDS2017 family taxonomy (주석 처리) ──────────────────────────────────
FAMILIES_2017: dict[str, list[str]] = {
    "Benign":     ["benign"],
    "DoS":        ["dos-"],
    "DDoS_Scan":  ["ddos", "portscan"],
    "BruteForce": ["ftp-patator", "ssh-patator"],
    "WebBot":     ["web-attack", "bot"],
    "Tail":       [],
}

# ─── CIC-IDS2018 family taxonomy ──────────────────────────────────────────────
# 정규화된 레이블 기준 (normalize_label: lowercase / space→hyphen / benign→normal)
# Tail (IR ≥ 5000): ddos-attack-loic-udp, brute-force--web, brute-force--xss, sql-injection
FAMILIES_2018: dict[str, list[str]] = {
    "Benign":       ["normal"],
    "DoS":          ["dos-attacks"],
    "DDoS":         ["ddos-attack", "ddos-attacks"],
    "BruteForce":   ["ftp-bruteforce", "ssh-bruteforce"],
    "Bot":          ["bot"],
    "Infiltration": ["infilteration"],
    "Tail":         [],
}


def get_families(dataset_type: str) -> dict[str, list[str]]:
    if dataset_type == "cic2018":
        return FAMILIES_2018
    elif dataset_type == "cic2017":
        return FAMILIES_2017
    raise ValueError(
        f"Unknown dataset_type='{dataset_type}'. "
        "Add a FAMILIES_<dataset> dict and register it here."
    )
    


TAIL_IR_THRESHOLD = 5000


# ─── logging ──────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("sade_tta")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in [logging.StreamHandler(), logging.FileHandler(log_path)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ─── device ───────────────────────────────────────────────────────────────────

def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ─── data ─────────────────────────────────────────────────────────────────────

def load_data(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=int)
    le = d["label_encoder"]
    dataset_type = d.get("dataset_type", "cic2017")
    return X, y, le, dataset_type


def split_data(X, y, seed: int):
    """60 / 20 / 20 stratified random split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ─── family assignment ────────────────────────────────────────────────────────

def compute_tail_ids(y: np.ndarray, threshold: int = TAIL_IR_THRESHOLD) -> set:
    counts = np.bincount(y)
    max_n = int(counts.max())
    ir = max_n / np.maximum(counts, 1)
    return {int(i) for i in np.where(ir >= threshold)[0]}


def assign_family(class_name: str, class_id: int, tail_ids: set,
                  families: dict) -> str:
    if class_id in tail_ids:
        return "Tail"
    n = class_name.lower().strip()
    for family, patterns in families.items():
        if family == "Tail":
            continue
        if any(n.startswith(pat) for pat in patterns):
            return family
    return "Other"


# ─── math helpers ─────────────────────────────────────────────────────────────

def softmax(w: np.ndarray) -> np.ndarray:
    e = np.exp(w - np.max(w))
    return e / e.sum()


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.maximum(np.bincount(y), 1)
    n, k = len(y), len(counts)
    return (n / (k * counts[y])).astype(np.float32)


# ─── VIME-style tabular perturbation ──────────────────────────────────────────

def vime_perturb(X: np.ndarray, p_mask: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Replace each feature independently with a column-random value with prob p_mask."""
    B, D = X.shape
    mask = rng.random((B, D)) < p_mask
    ref_rows = rng.integers(0, B, size=(B, D))
    X_ref = X[ref_rows, np.arange(D)]
    return np.where(mask, X_ref, X)


# ─── XGBoost helpers ──────────────────────────────────────────────────────────

def xgb_params(n_classes: int, n_estimators: int, device: str, seed: int) -> dict:
    p = dict(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        tree_method="hist",
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=4,
    )
    if device == "cuda":
        p["device"] = "cuda"
    if n_classes == 2:
        p.update(objective="binary:logistic", eval_metric="logloss")
    else:
        p.update(objective="multi:softprob",
                 num_class=n_classes, eval_metric="mlogloss")
    return p


def predict_proba(model, X: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    out = []
    for s in range(0, len(X), chunk):
        p = model.predict_proba(X[s:s + chunk])
        out.append(np.asarray(p, dtype=np.float32))
    return np.concatenate(out, axis=0)


# ─── Baseline ─────────────────────────────────────────────────────────────────

def train_baseline(X_tr, y_tr, X_va, y_va,
                   n_classes, n_estimators, device, seed, logger):
    logger.info("Training baseline XGBoost …")
    t0 = time.time()
    w = balanced_weights(y_tr)
    model = xgb.XGBClassifier(**xgb_params(n_classes, n_estimators, device, seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr, sample_weight=w,
                  eval_set=[(X_va, y_va)], verbose=False)
    logger.info(f"Baseline ready in {time.time() - t0:.1f}s")
    return model


# ─── Family experts (identical to code_mati.py) ───────────────────────────────

def train_expert(X_tr, y_tr, X_va, y_va,
                 focus_ids: list[int],
                 n_classes: int,
                 n_estimators: int, device: str, seed: int,
                 focus_weight: float, logger, name: str):
    """Train on ALL data; focus classes receive an additional sample-weight multiplier."""
    t0 = time.time()
    w = balanced_weights(y_tr)
    w[np.isin(y_tr, focus_ids)] *= focus_weight

    model = xgb.XGBClassifier(**xgb_params(n_classes, n_estimators, device, seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr, sample_weight=w,
                  eval_set=[(X_va, y_va)], verbose=False)

    logger.info(f"  [{name}] trained in {time.time() - t0:.1f}s | focus classes: {focus_ids}")
    return model


def train_all_experts(X_tr, y_tr, X_va, y_va,
                      class_names, n_classes, n_estimators,
                      device, seed, logger,
                      tail_ids: set, focus_weight: float,
                      families: dict):
    models, fam_names = [], []
    for fam in families:
        focus_ids = [i for i, c in enumerate(class_names)
                     if assign_family(c, i, tail_ids, families) == fam]
        if not focus_ids:
            logger.warning(f"Family '{fam}' has no matched classes — skipped.")
            continue
        logger.info(f"Training expert '{fam}': {[class_names[i] for i in focus_ids]}")
        m = train_expert(
            X_tr, y_tr, X_va, y_va,
            focus_ids, n_classes, n_estimators, device, seed,
            focus_weight, logger, fam,
        )
        models.append(m)
        fam_names.append(fam)
    return models, fam_names


# ─── SADE TTA ─────────────────────────────────────────────────────────────────

def sade_tta(models, X_te: np.ndarray,
             n_epochs: int, lr: float, p_mask: float,
             batch_size: int, seed: int, logger,
             early_stop_thr: float = 0.02) -> np.ndarray:
    """
    SADE (NeurIPS 2022) prediction stability TTA for tabular classification.

    Objective (maximise):
        S = (1/B) Σ_b  ŷ¹_b · ŷ²_b
    where ŷ^m = Σ_e w_e · v^m_e  (weighted ensemble on VIME view m).

    Gradient w.r.t. w_e  (before softmax chain rule):
        g_e = ∂S/∂w_e = (1/B) Σ_b [v¹_{e,b} · ŷ²_b + ŷ¹_b · v²_{e,b}]

    Gradient w.r.t. softmax logit j  (chain rule):
        ∂S/∂logit_j = w_j · (g_j − Σ_k w_k g_k)

    Adam gradient ascent on S  (≡ descent on −S).
    Early stop when any weight collapses below early_stop_thr.
    """
    E = len(models)
    rng = np.random.default_rng(seed)
    logits = np.zeros(E, dtype=np.float64)

    m_adam = np.zeros(E)
    v_adam = np.zeros(E)
    t_step = 0
    β1, β2, ε_adam = 0.9, 0.999, 1e-8

    N = len(X_te)
    logger.info(
        f"SADE TTA: {E} experts, {n_epochs} epochs, "
        f"lr={lr}, p_mask={p_mask}, batch_size={batch_size}"
    )

    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        epoch_S = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_b = X_te[perm[start:end]]
            B = end - start

            x1 = vime_perturb(X_b, p_mask, rng)
            x2 = vime_perturb(X_b, p_mask, rng)

            # [E, B, C] expert probability vectors for each view
            v1 = np.stack([predict_proba(m, x1, chunk=B) for m in models])
            v2 = np.stack([predict_proba(m, x2, chunk=B) for m in models])

            w = softmax(logits)

            # Ensemble predictions [B, C]
            y1 = np.einsum("e,ebc->bc", w, v1)
            y2 = np.einsum("e,ebc->bc", w, v2)

            # Prediction stability S = (1/B) Σ_b ŷ¹_b · ŷ²_b
            S = float(np.mean(np.sum(y1 * y2, axis=1)))
            epoch_S += S
            n_batches += 1

            # g_e = ∂S/∂w_e
            dS_dw = (1.0 / B) * (
                np.einsum("ebc,bc->e", v1, y2) +
                np.einsum("bc,ebc->e", y1, v2)
            )

            # ∂S/∂logit_j = w_j(g_j - Σ_k w_k g_k)
            grad_logits = w * (dS_dw - np.dot(w, dS_dw))

            # Ascend on S ≡ descend on −S → negate gradient for Adam
            neg_grad = -grad_logits
            t_step += 1
            m_adam = β1 * m_adam + (1 - β1) * neg_grad
            v_adam = β2 * v_adam + (1 - β2) * neg_grad ** 2
            m_hat = m_adam / (1 - β1 ** t_step)
            v_hat = v_adam / (1 - β2 ** t_step)
            logits -= lr * m_hat / (np.sqrt(v_hat) + ε_adam)

        w_now = softmax(logits)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info(
                f"  epoch {epoch + 1:3d}/{n_epochs} | "
                f"S={epoch_S / max(n_batches, 1):.6f} | "
                f"w={w_now.round(3)}"
            )

        if early_stop_thr > 0 and np.any(w_now <= early_stop_thr):
            logger.info(
                f"  Early stop epoch {epoch + 1}: "
                f"min_w={w_now.min():.4f} ≤ {early_stop_thr}"
            )
            break

    return softmax(logits)


def moe_predict(models, weights: np.ndarray,
                X: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    probs = np.stack([predict_proba(m, X, chunk) for m in models])
    ensemble = np.einsum("e,enc->nc", weights, probs)
    return np.argmax(ensemble, axis=1)


# ─── output helpers ───────────────────────────────────────────────────────────

def sorted_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: list[str]) -> str:
    """sklearn classification_report 대체 — test support 내림차순 정렬."""
    n_cls = len(class_names)
    counts = np.bincount(y_true, minlength=n_cls)
    order = np.argsort(-counts)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_cls), zero_division=0
    )
    name_w = max(len(c) for c in class_names) + 2
    lines = [
        f"{'':>{name_w}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}",
        "",
    ]
    for ci in order:
        if counts[ci] == 0:
            continue
        lines.append(
            f"{class_names[ci]:>{name_w}}  {p[ci]:>9.4f}  {r[ci]:>9.4f}"
            f"  {f[ci]:>9.4f}  {int(s[ci]):>9,}"
        )
    lines.append("")
    for avg_key, avg_label in [("macro", "macro avg"), ("weighted", "weighted avg")]:
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg_key, zero_division=0
        )
        lines.append(
            f"{avg_label:>{name_w}}  {pa:>9.4f}  {ra:>9.4f}"
            f"  {fa:>9.4f}  {len(y_true):>9,}"
        )
    return "\n".join(lines)


def save_expert_dist(models: list, fam_names: list[str], weights: np.ndarray,
                     X_te: np.ndarray, y_te: np.ndarray,
                     class_names: list[str], out_dir: str,
                     logger) -> None:
    """
    Expert prediction distribution 히트맵.
    - x축: expert  (열 합계 = 100%)
    - y축: class   (행 레이블에 test support 표기)
    - 셀:  pct%(count)  — pct = count / N_te * 100
    - 색:  log10(pct) 스케일 (소수 클래스도 가시화)
    - 로그: expert별 top-3 예측 클래스
    - 저장: expert_pred_dist.png / .csv
    """
    n_cls = len(class_names)
    N_te  = len(X_te)
    E     = len(models)
    counts_te = np.bincount(y_te, minlength=n_cls)
    cls_order = np.argsort(-counts_te)          # test support 내림차순

    # pred_counts[c, e] = expert e가 class c로 예측한 test sample 수
    pred_counts = np.zeros((n_cls, E), dtype=np.int64)
    for e, model in enumerate(models):
        preds = np.argmax(predict_proba(model, X_te), axis=1)
        pred_counts[:, e] = np.bincount(preds, minlength=n_cls)

    # 행을 support 내림차순으로 재정렬
    pc   = pred_counts[cls_order, :]          # [n_cls, E]  counts
    pct  = pc / N_te * 100                    # [n_cls, E]  %  (열 합계 = 100%)

    # ── log ─────────────────────────────────────────────────────────────────
    logger.info("\n── Expert prediction distribution (top-3 per expert) ──")
    for e, fam in enumerate(fam_names):
        top3 = np.argsort(-pred_counts[:, e])[:3]
        detail = ", ".join(
            f"{class_names[i]}={pred_counts[i,e]/N_te*100:.2f}%"
            f"({pred_counts[i,e]:,})" for i in top3
        )
        logger.info(f"  [{fam:15s}]  SADE_w={weights[e]:.4f}  {detail}")

    # ── PNG ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(max(8, E * 2.0), max(6, n_cls * 0.62 + 2))
    )
    im = ax.imshow(pct, aspect="auto", cmap="Blues", vmin=0, vmax=100)

    # x축: expert 이름 + SADE weight (수평 표기)
    ax.set_xticks(range(E))
    ax.set_xticklabels(
        [f"{fam}\n(w={weights[e]:.4f})" for e, fam in enumerate(fam_names)],
        fontsize=8
    )

    # y축: class 이름 + test support count (수평 표기, 대각선 없음)
    sorted_cnt = counts_te[cls_order]
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(
        [f"{class_names[cls_order[i]]}  (n={sorted_cnt[i]:,})"
         for i in range(n_cls)],
        fontsize=8
    )

    ax.set_xlabel("Expert  (SADE weight)", fontsize=9)
    ax.set_ylabel("Predicted class  (sorted by test support ↓)", fontsize=9)
    ax.set_title("Expert argmax prediction distribution on test set", fontsize=11)

    # 셀 annotation: count > 0 이면 전부 표기  →  "X.XX%\n(count)"
    for ci in range(n_cls):
        for e in range(E):
            cnt = int(pc[ci, e])
            if cnt == 0:
                continue
            p = pct[ci, e]
            txt = f"{p:.2f}%\n({cnt:,})"
            fg  = "white" if p > 30 else "black"
            ax.text(e, ci, txt, ha="center", va="center",
                    fontsize=6, color=fg, linespacing=1.3)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("% of test samples", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "expert_pred_dist.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Expert dist PNG → {png_path}")

    # ── CSV ─────────────────────────────────────────────────────────────────
    row_labels = [
        f"{class_names[cls_order[i]]}(n={sorted_cnt[i]:,})"
        for i in range(n_cls)
    ]
    records = {}
    for e, fam in enumerate(fam_names):
        records[f"{fam}(w={weights[e]:.4f})"] = [
            f"{pct[ci,e]:.2f}%({int(pc[ci,e]):,})" for ci in range(n_cls)
        ]
    df_out = pd.DataFrame(records, index=row_labels)
    df_out.index.name = "class(test_n)"
    csv_path = os.path.join(out_dir, "expert_pred_dist.csv")
    df_out.to_csv(csv_path)
    logger.info(f"Expert dist CSV → {csv_path}")


def save_colored_table(rows: list[dict], col_headers: list[str],
                       path: str, title: str = "") -> None:
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
                    colors.append(
                        BLUE if m > b + EPS else RED if m < b - EPS else YELLOW
                    )
                else:
                    colors.append(WHITE)
            elif row.get("_footer") or col == "support":
                colors.append(GRAY)
            else:
                colors.append(WHITE)
        cell_text.append(texts)
        cell_colors.append(colors)

    n_rows, n_cols = len(rows), len(col_headers)
    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.3), max(4, n_rows * 0.35)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text, colLabels=col_headers,
        cellColours=cell_colors, loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(n_cols)))
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report_rows(y_true, y_pred_b, y_pred_m, class_names):
    n_cls = len(class_names)
    labels = np.arange(n_cls)
    counts = np.bincount(y_true, minlength=n_cls)
    order = np.argsort(-counts)

    p_b, r_b, f_b, _ = precision_recall_fscore_support(
        y_true, y_pred_b, labels=labels, zero_division=0
    )
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred_m, labels=labels, zero_division=0
    )

    col_headers = [
        "class", "support",
        "prec(B)", "prec(M)",
        "recall(B)", "recall(M)",
        "f1(B)", "f1(M)",
    ]

    rows = []
    for ci in order:
        if counts[ci] == 0:
            continue
        rows.append({
            "class":     class_names[ci],
            "support":   int(counts[ci]),
            "prec(B)":   float(p_b[ci]),
            "prec(M)":   float(p_m[ci]),
            "recall(B)": float(r_b[ci]),
            "recall(M)": float(r_m[ci]),
            "f1(B)":     float(f_b[ci]),
            "f1(M)":     float(f_m[ci]),
        })

    for avg_key, avg_label in [("macro", "macro avg"), ("weighted", "weighted avg")]:
        pB, rB, fB, _ = precision_recall_fscore_support(
            y_true, y_pred_b, average=avg_key, zero_division=0
        )
        pM, rM, fM, _ = precision_recall_fscore_support(
            y_true, y_pred_m, average=avg_key, zero_division=0
        )
        rows.append({
            "class":     avg_label,
            "support":   int(len(y_true)),
            "prec(B)":   float(pB),
            "prec(M)":   float(pM),
            "recall(B)": float(rB),
            "recall(M)": float(rM),
            "f1(B)":     float(fB),
            "f1(M)":     float(fM),
            "_footer":   True,
        })

    return rows, col_headers


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SADE-TTA IDS: taxonomy MoE + prediction-stability TTA"
    )
    parser.add_argument("--data", required=True,
                        help="Path to pre-processed .pkl (e.g. cic2017_proc.pkl)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n_estimators", type=int,   default=300,
                        help="Trees per XGBoost model")
    parser.add_argument("--focus_weight", type=float, default=10.0,
                        help="Sample-weight multiplier for focus classes")
    parser.add_argument("--tta_epochs",   type=int,   default=50)
    parser.add_argument("--tta_lr",       type=float, default=0.01)
    parser.add_argument("--tta_p_mask",   type=float, default=0.1,
                        help="VIME feature-mask probability")
    parser.add_argument("--tta_batch",    type=int,   default=4096)
    parser.add_argument("--tta_early_stop", type=float, default=0.02,
                        help="Stop TTA when any weight ≤ threshold (0 = disable)")
    parser.add_argument("--model", type=int, default=2, choices=[0, 1, 2],
                        help="0=baseline only  1=MoE+TTA only  2=both")
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", ts)
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logger(os.path.join(out_dir, "experiment.log"))
    logger.info(f"Args: {vars(args)}")

    device = detect_device()
    logger.info(f"Device: {device}")

    X, y, le, dataset_type = load_data(args.data)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    families = get_families(dataset_type)
    logger.info(
        f"Loaded {X.shape[0]:,} samples | "
        f"{X.shape[1]} features | {n_classes} classes | dataset={dataset_type}"
    )
    logger.info(f"Families: {list(families.keys())}")

    tail_ids = compute_tail_ids(y, TAIL_IR_THRESHOLD)
    logger.info(
        f"Tail IR threshold: {TAIL_IR_THRESHOLD:,} — "
        f"Tail class IDs: {sorted(tail_ids)}"
    )
    _counts_all = np.bincount(y, minlength=len(class_names))
    for i in np.argsort(-_counts_all):
        fam = assign_family(class_names[i], i, tail_ids, families)
        logger.info(f"  [{i:2d}] {class_names[i]:40s} family={fam:12s} n={int(_counts_all[i]):,}")

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, args.seed)
    logger.info(f"Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}")

    # ── baseline ──────────────────────────────────────────────────────────────
    y_pred_b = None
    if args.model in (0, 2):
        baseline = train_baseline(
            X_tr, y_tr, X_va, y_va,
            n_classes, args.n_estimators, device, args.seed, logger,
        )
        y_pred_b = np.asarray(baseline.predict(X_te), dtype=int)
        logger.info("\n── Baseline classification report ──")
        logger.info("\n" + sorted_classification_report(y_te, y_pred_b, class_names))
        del baseline
        gc.collect()

    # ── MoE + SADE TTA ────────────────────────────────────────────────────────
    y_pred_m = None
    if args.model in (1, 2):
        t_train = time.time()
        models, fam_names = train_all_experts(
            X_tr, y_tr, X_va, y_va,
            class_names, n_classes, args.n_estimators,
            device, args.seed, logger, tail_ids, args.focus_weight,
            families=families,
        )
        logger.info(
            f"All experts trained in {time.time() - t_train:.1f}s  "
            f"({len(models)} experts: {fam_names})"
        )

        t_tta = time.time()
        weights = sade_tta(
            models, X_te,
            n_epochs=args.tta_epochs, lr=args.tta_lr,
            p_mask=args.tta_p_mask, batch_size=args.tta_batch,
            seed=args.seed, logger=logger,
            early_stop_thr=args.tta_early_stop,
        )
        logger.info(f"SADE TTA done in {time.time() - t_tta:.1f}s")
        for fam, w_val in zip(fam_names, weights):
            logger.info(f"  Weight [{fam}] = {w_val:.4f}")

        save_expert_dist(models, fam_names, weights, X_te, y_te, class_names, out_dir, logger)

        y_pred_m = moe_predict(models, weights, X=X_te)
        logger.info("\n── MoE + SADE TTA classification report ──")
        logger.info("\n" + sorted_classification_report(y_te, y_pred_m, class_names))

    # ── comparison table ──────────────────────────────────────────────────────
    if args.model == 2 and y_pred_b is not None and y_pred_m is not None:
        rows, col_headers = build_report_rows(y_te, y_pred_b, y_pred_m, class_names)

        png_path = os.path.join(out_dir, "baseline_vs_moe_per_class.png")
        save_colored_table(
            rows, col_headers, png_path,
            title=f"SADE-TTA IDS | Baseline vs MoE+SADE-TTA  (seed={args.seed})",
        )
        logger.info(f"PNG table → {png_path}")

        df = pd.DataFrame(rows)
        df["delta_f1"] = df["f1(M)"] - df["f1(B)"]
        csv_path = os.path.join(out_dir, "baseline_vs_moe_per_class.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV       → {csv_path}")

    logger.info(f"\nResults directory: {out_dir}")


if __name__ == "__main__":
    main()

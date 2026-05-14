#!/usr/bin/env python3
"""
code_mati.py — MATI-inspired MoE for IDS imbalanced classification.

Expert division  : CIC-IDS2017 attack family taxonomy (5 families).
Expert training  : full training data + XGBoost sample_weight
                   (balanced base × FOCUS_WEIGHT multiplier on focus classes).
Test-time agg.   : VIME perturbation → two views per sample →
                   minimise Continuous Prediction Gap (MATI Eq. 6, adapted
                   for class-prob vectors) → Adam update on scalar weights w.
Baseline         : standard XGBoost with balanced sample weights.

Usage:
  python src/code_mati.py --data cic2017_proc.pkl --model 2
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

# ─── CIC-IDS2017 family taxonomy ──────────────────────────────────────────────
# Values are lowercase substrings matched against the already-normalised class
# names in cic2017_proc.pkl (spaces → dashes, lower-cased by preprocessing).
# Examples: 'dos-hulk', 'web-attack-?-xss', 'ftp-patator', 'benign'.
# "dos-" matches all DoS variants but NOT "ddos" (starts with 'dd').
# "web-attack" matches all three Web Attack sub-variants.
# "benign" matches nothing → "Other" → gets standard weight in every expert.
FAMILIES: dict[str, list[str]] = {
    "DoS":        ["dos-"],           # dos-hulk, dos-goldeneye, dos-slowloris, dos-slowhttptest
    "DDoS_Scan":  ["ddos", "portscan"],
    "BruteForce": ["ftp-patator", "ssh-patator"],
    "WebBot":     ["web-attack", "bot"],
    "Tail":       ["infiltration", "heartbleed"],
}

# Weight multiplier applied to focus-family classes on top of balanced weights
FOCUS_WEIGHT_DEFAULT = 10.0


# ─── logging ──────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("mati")
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
    return X, y, le


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

def assign_family(class_name: str) -> str:
    """Return FAMILIES key for class_name, or 'Other' (→ BENIGN / unmatched)."""
    n = class_name.lower().strip()
    for family, patterns in FAMILIES.items():
        if any(pat in n for pat in patterns):
            return family
    return "Other"


# ─── math helpers ─────────────────────────────────────────────────────────────

def softmax(w: np.ndarray) -> np.ndarray:
    e = np.exp(w - np.max(w))
    return e / e.sum()


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weight = n / (k · count_c).  Inverse-frequency rebalancing."""
    counts = np.maximum(np.bincount(y), 1)
    n, k = len(y), len(counts)
    return (n / (k * counts[y])).astype(np.float32)


# ─── VIME-style tabular perturbation ──────────────────────────────────────────

def vime_perturb(X: np.ndarray, p_mask: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Generate one perturbed view of X.
    Each feature-value is independently replaced with a random value drawn
    from the same column with probability p_mask.  Fully vectorised.
    """
    B, D = X.shape
    mask = rng.random((B, D)) < p_mask          # [B, D] bool
    ref_rows = rng.integers(0, B, size=(B, D))  # random row indices per cell
    X_ref = X[ref_rows, np.arange(D)]           # [B, D] random column samples
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
    """XGBoost predict_proba with chunking to control memory."""
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


# ─── Family experts ───────────────────────────────────────────────────────────

def train_expert(X_tr, y_tr, X_va, y_va,
                 focus_ids: list[int],
                 n_classes, n_estimators, focus_weight, device, seed,
                 logger, name: str):
    """
    XGBoost trained on ALL data; focus-family classes receive
    focus_weight × the base balanced weight.
    """
    t0 = time.time()
    w = balanced_weights(y_tr)
    focus_mask = np.isin(y_tr, focus_ids)
    w[focus_mask] *= focus_weight
    w /= w.mean()  # keep overall scale comparable across experts

    model = xgb.XGBClassifier(**xgb_params(n_classes, n_estimators, device, seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr, sample_weight=w,
                  eval_set=[(X_va, y_va)], verbose=False)
    logger.info(
        f"  [{name}] trained in {time.time() - t0:.1f}s | "
        f"focus classes: {focus_ids}"
    )
    return model


def train_all_experts(X_tr, y_tr, X_va, y_va,
                      class_names, n_classes, n_estimators,
                      focus_weight, device, seed, logger):
    """Return (models, family_names) lists, skipping families with no matches."""
    models, fam_names = [], []
    for fam in FAMILIES:
        focus_ids = [i for i, c in enumerate(class_names) if assign_family(c) == fam]
        if not focus_ids:
            logger.warning(f"Family '{fam}' has no matched classes — skipped.")
            continue
        logger.info(
            f"Training expert '{fam}': "
            f"{[class_names[i] for i in focus_ids]}"
        )
        m = train_expert(
            X_tr, y_tr, X_va, y_va,
            focus_ids, n_classes, n_estimators, focus_weight, device, seed,
            logger, fam,
        )
        models.append(m)
        fam_names.append(fam)
    return models, fam_names


# ─── Test-time aggregation ────────────────────────────────────────────────────

def tta_aggregate(models, X_te: np.ndarray,
                  n_epochs: int, lr: float, p_mask: float,
                  batch_size: int, seed: int, logger) -> np.ndarray:
    """
    MATI Algorithm 1 adapted for classification.

    Expert parameters are frozen.  Only the scalar weight vector w ∈ R^E is
    learned by minimising the Continuous Prediction Gap:

        S = (1/|B|) Σ_{x∈B} ‖ŷ¹ - ŷ²‖²,
        ŷ^v = Σ_e σ(w_e) · v^v_e,          v^v_e = Expert_e.predict_proba(x^v)

    Analytical gradient (no autograd needed — w is the only parameter):

        ∂S/∂w_k = (2/|B|) σ(w_k) Σ_{nc} diff_nc ((v¹_knc − ŷ¹_nc) − (v²_knc − ŷ²_nc))

    where diff = ŷ¹ − ŷ², derived from the Jacobian of softmax.

    Returns softmax(w) — the final expert aggregation weights.
    """
    E = len(models)
    rng = np.random.default_rng(seed)
    w = np.zeros(E, dtype=np.float64)  # softmax(0) = 1/E (uniform start)

    # Adam state
    m_adam = np.zeros(E)
    v_adam = np.zeros(E)
    t_step = 0
    β1, β2, ε_adam = 0.9, 0.999, 1e-8

    N = len(X_te)
    logger.info(
        f"TTA: {E} experts, {n_epochs} epochs, "
        f"lr={lr}, p_mask={p_mask}, batch_size={batch_size}"
    )

    early_stopped = False
    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_b = X_te[perm[start:end]]
            B = end - start

            # Two perturbed views of the mini-batch
            x1 = vime_perturb(X_b, p_mask, rng)
            x2 = vime_perturb(X_b, p_mask, rng)

            # Expert probabilities: [E, B, C]
            v1 = np.stack([predict_proba(m, x1, chunk=B) for m in models])
            v2 = np.stack([predict_proba(m, x2, chunk=B) for m in models])

            s = softmax(w)  # [E]

            # Weighted ensemble predictions [B, C]
            y1 = np.einsum("e,ebc->bc", s, v1)
            y2 = np.einsum("e,ebc->bc", s, v2)

            diff = y1 - y2                     # [B, C]
            loss = float(np.mean(diff ** 2))
            epoch_loss += loss
            n_batches += 1

            # Gradient: ∂S/∂w_k = (2/B)·s_k · Σ_{bc} diff_bc·(r1_kbc − r2_kbc)
            # where r1[k] = v1[k] − y1,  r2[k] = v2[k] − y2
            r1 = v1 - y1[None]  # [E, B, C]
            r2 = v2 - y2[None]  # [E, B, C]

            # Vectorised over k
            # contraction over b,c: Σ_{bc} diff_bc · (r1_kbc - r2_kbc) → [E]
            gap = np.einsum("bc,ebc->e", diff, r1 - r2)
            grad = (2.0 / B) * s * gap

            # Adam update
            t_step += 1
            m_adam = β1 * m_adam + (1 - β1) * grad
            v_adam = β2 * v_adam + (1 - β2) * grad ** 2
            m_hat  = m_adam / (1 - β1 ** t_step)
            v_hat  = v_adam / (1 - β2 ** t_step)
            w -= lr * m_hat / (np.sqrt(v_hat) + ε_adam)

            # Early stop: any softmax weight collapses to ≤ 0.05 (MATI §3.3)
            if np.min(softmax(w)) <= 0.05:
                logger.info(
                    f"  TTA early stop — epoch {epoch + 1}, "
                    f"batch {start}:{end}"
                )
                early_stopped = True
                break

        if epoch % 10 == 0 or epoch == n_epochs - 1 or early_stopped:
            s_now = softmax(w)
            logger.info(
                f"  epoch {epoch + 1:3d}/{n_epochs} | "
                f"loss={epoch_loss / max(n_batches, 1):.6f} | "
                f"w={s_now.round(3)}"
            )
        if early_stopped:
            break

    return softmax(w)


def moe_predict(models, weights: np.ndarray,
                X: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Weighted ensemble prediction; returns predicted class indices."""
    probs = np.stack(
        [predict_proba(m, X, chunk) for m in models], axis=0
    )  # [E, N, C]
    ensemble = np.einsum("e,enc->nc", weights, probs)
    return np.argmax(ensemble, axis=1)


# ─── output helpers ───────────────────────────────────────────────────────────

def save_colored_table(rows: list[dict], col_headers: list[str],
                       path: str, title: str = "") -> None:
    """Render and save a comparison table as PNG (CLAUDE.md spec)."""
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
    fig, ax = plt.subplots(
        figsize=(max(14, n_cols * 1.3), max(4, n_rows * 0.35))
    )
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
    """Return (rows, col_headers) sorted by descending support."""
    n_cls = len(class_names)
    labels = np.arange(n_cls)
    counts = np.bincount(y_true, minlength=n_cls)
    order  = np.argsort(-counts)

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
    parser = argparse.ArgumentParser(description="MATI-IDS: family MoE + TTA")
    parser.add_argument("--data", required=True,
                        help="Path to pre-processed .pkl (e.g. cic2017_proc.pkl)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300,
                        help="Trees per XGBoost model")
    parser.add_argument("--focus_weight", type=float, default=FOCUS_WEIGHT_DEFAULT,
                        help="Sample-weight multiplier for focus-family classes")
    parser.add_argument("--tta_epochs",  type=int,   default=50)
    parser.add_argument("--tta_lr",      type=float, default=0.01)
    parser.add_argument("--tta_p_mask",  type=float, default=0.1,
                        help="VIME feature-mask probability")
    parser.add_argument("--tta_batch",   type=int,   default=4096,
                        help="Mini-batch size for TTA gradient updates")
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

    # ── data ──────────────────────────────────────────────────────────────────
    X, y, le = load_data(args.data)
    class_names = list(le.classes_)
    n_classes   = len(class_names)
    logger.info(
        f"Loaded {X.shape[0]:,} samples | "
        f"{X.shape[1]} features | {n_classes} classes"
    )

    # Log family assignment for transparency
    for i, c in enumerate(class_names):
        fam = assign_family(c)
        cnt = int(np.sum(y == i))
        logger.info(f"  [{i:2d}] {c:40s} family={fam:12s} n={cnt:,}")

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, args.seed)
    logger.info(
        f"Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}"
    )

    # ── baseline ──────────────────────────────────────────────────────────────
    y_pred_b = None
    if args.model in (0, 2):
        baseline = train_baseline(
            X_tr, y_tr, X_va, y_va,
            n_classes, args.n_estimators, device, args.seed, logger,
        )
        y_pred_b = np.asarray(baseline.predict(X_te), dtype=int)
        logger.info("\n── Baseline classification report ──")
        logger.info(
            "\n" + classification_report(
                y_te, y_pred_b, target_names=class_names, zero_division=0
            )
        )
        del baseline
        gc.collect()

    # ── MoE + TTA ─────────────────────────────────────────────────────────────
    y_pred_m = None
    if args.model in (1, 2):
        t_train = time.time()
        models, fam_names = train_all_experts(
            X_tr, y_tr, X_va, y_va,
            class_names, n_classes, args.n_estimators,
            args.focus_weight, device, args.seed, logger,
        )
        logger.info(
            f"All experts trained in {time.time() - t_train:.1f}s  "
            f"({len(models)} experts: {fam_names})"
        )

        t_tta = time.time()
        weights = tta_aggregate(
            models, X_te,
            n_epochs=args.tta_epochs, lr=args.tta_lr,
            p_mask=args.tta_p_mask, batch_size=args.tta_batch,
            seed=args.seed, logger=logger,
        )
        logger.info(f"TTA done in {time.time() - t_tta:.1f}s")
        for fam, w_val in zip(fam_names, weights):
            logger.info(f"  Weight [{fam}] = {w_val:.4f}")

        y_pred_m = moe_predict(models, weights, X_te)
        logger.info("\n── MoE + TTA classification report ──")
        logger.info(
            "\n" + classification_report(
                y_te, y_pred_m, target_names=class_names, zero_division=0
            )
        )

    # ── comparison table ──────────────────────────────────────────────────────
    if args.model == 2 and y_pred_b is not None and y_pred_m is not None:
        rows, col_headers = build_report_rows(y_te, y_pred_b, y_pred_m, class_names)

        png_path = os.path.join(out_dir, "baseline_vs_moe_per_class.png")
        save_colored_table(
            rows, col_headers, png_path,
            title=f"MATI-IDS | Baseline vs MoE+TTA  (seed={args.seed})",
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

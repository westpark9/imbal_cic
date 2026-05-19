#!/usr/bin/env python3
"""
code_mati_repro.py  —  MATI adaptation for imbalanced multi-class classification.

Reference: "Mixture Experts with Test-Time Self-Supervised Aggregation for
            Tabular Imbalanced Regression"  arXiv 2506.07033

Adaptation decisions (regression → classification):
  SMOGN  → SMOTE   (discrete labels; Gaussian y-noise inapplicable)
  MSE    → mlogloss (XGBoost multi-class)
  GMM 1D signal: log1p(class_count[y_i]) per training sample
                 (sorted-frequency proxy for continuous y in the original)
  Expert outputs: local k-class probs zero-padded to global n_class vector

Pipeline
────────
  D_T ──[Stage-1 SMOTE (global)]──► D_S
  D_S ──[GMM, AIC selects N]──► D_S0 … D_S(N-1)   (hard, non-overlapping)
  D_Sn ──[Stage-2 SMOTE (region)]──► D_Sn_aug  →  Expert_n (XGBoost)

Test-time aggregation (MATI Algorithm 1)
─────────────────────────────────────────
  VIME perturbation → CPG loss → Adam → weights
  Early stop: any w_i ≤ 0.05

Three test sets: Normal / Balanced / Inverse
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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# ── colour constants ───────────────────────────────────────────────────────────
BLUE   = "#cce5ff"
RED    = "#ffcccc"
YELLOW = "#fff9cc"
GRAY   = "#f2f2f2"
WHITE  = "#ffffff"
EPS    = 0.001


# ── logging ────────────────────────────────────────────────────────────────────
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("mati_repro")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in [logging.StreamHandler(), logging.FileHandler(log_path)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ── device ─────────────────────────────────────────────────────────────────────
def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ── data ───────────────────────────────────────────────────────────────────────
def load_data(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=int)
    le = d["label_encoder"]
    return X, y, le


def split_data(X, y, seed: int):
    """60 / 20 / 20 stratified split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ── Stage-1 SMOTE (global) ─────────────────────────────────────────────────────
def stage1_smote(X_tr, y_tr, smote_floor: int, seed: int, logger) -> tuple:
    """
    Oversample classes below smote_floor to smote_floor samples.
    Classes already at or above smote_floor are untouched.
    """
    counts = np.bincount(y_tr)
    minority = {
        int(c): smote_floor
        for c in range(len(counts))
        if 0 < counts[c] < smote_floor
    }

    if not minority:
        logger.info("Stage-1 SMOTE: no classes below floor — skipped.")
        return X_tr, y_tr

    # k_neighbors must be < min minority class count
    min_minority_count = min(counts[c] for c in minority)
    k = min(5, int(min_minority_count) - 1)
    if k < 1:
        logger.warning("Stage-1 SMOTE: k_neighbors < 1 — skipped.")
        return X_tr, y_tr

    logger.info(
        f"Stage-1 SMOTE: floor={smote_floor}, k_neighbors={k}, "
        f"classes to oversample: {sorted(minority.keys())}"
    )
    t0 = time.time()
    sm = SMOTE(sampling_strategy=minority, k_neighbors=k, random_state=seed)
    X_s, y_s = sm.fit_resample(X_tr, y_tr)
    logger.info(
        f"  Stage-1 done in {time.time() - t0:.1f}s | "
        f"{len(y_tr):,} → {len(y_s):,} samples"
    )
    return X_s.astype(np.float32), y_s.astype(int)


# ── GMM expert assignment ──────────────────────────────────────────────────────
def _merge_single_class_components(class_comp: np.ndarray,
                                   log_cnt: np.ndarray,
                                   n_classes: int, logger) -> tuple:
    """
    Iteratively merge any component that owns only 1 class into its
    nearest neighbour (by mean log-count distance).  Re-indexes
    remaining components to 0..N_final-1.
    """
    from collections import defaultdict

    changed = True
    while changed:
        changed = False
        comp_to_cls: dict = defaultdict(list)
        for cls_id in range(n_classes):
            if class_comp[cls_id] >= 0:
                comp_to_cls[int(class_comp[cls_id])].append(cls_id)

        for comp, cls_list in list(comp_to_cls.items()):
            if len(cls_list) == 1:
                lone_cls    = cls_list[0]
                lone_signal = log_cnt[lone_cls]
                best_other, best_dist = None, np.inf
                for other, other_cls in comp_to_cls.items():
                    if other == comp:
                        continue
                    dist = abs(lone_signal - float(np.mean(log_cnt[other_cls])))
                    if dist < best_dist:
                        best_dist, best_other = dist, other
                if best_other is not None:
                    logger.info(
                        f"  Merge: Component {comp} (class {lone_cls}, "
                        f"log_cnt={lone_signal:.2f}) → Component {best_other} "
                        f"(dist={best_dist:.2f})"
                    )
                    class_comp[lone_cls] = best_other
                    changed = True
                    break   # restart after each merge

    # Re-index to 0..N_final-1
    unique = sorted({int(class_comp[c]) for c in range(n_classes) if class_comp[c] >= 0})
    remap  = {old: new for new, old in enumerate(unique)}
    new_comp = np.array(
        [remap[int(class_comp[c])] if class_comp[c] >= 0 else -1
         for c in range(n_classes)],
        dtype=int,
    )
    n_final = len(unique)
    logger.info(f"After merge: {n_final} final component(s)")
    for n in range(n_final):
        cls_in = [c for c in range(n_classes) if new_comp[c] == n]
        logger.info(
            f"  Final Component {n}: classes={cls_in} | "
            f"log_counts={[round(float(log_cnt[c]), 2) for c in cls_in]}"
        )
    return new_comp, n_final


def fit_gmm_and_assign(y_s, n_classes: int, max_components: int,
                       seed: int, logger) -> tuple:
    """
    Fit 1-D GMM at CLASS level on log1p(class_count).

    Why class-level (not sample-level):
      Each distinct log-count value produces its own infinitely tight Gaussian
      when GMM is fit to 1.7 M samples — every high-frequency class becomes a
      singleton component.  Fitting to n_classes points (≤ 15) instead lets
      AIC choose meaningful frequency-tier groupings.

    After AIC selection, any remaining single-class component is merged into
    its nearest multi-class neighbour so every expert trains on ≥ 2 classes.

    Returns (sample_assignments [N_samples], N_final, class_comp [n_classes]).
    """
    counts  = np.bincount(y_s, minlength=n_classes).astype(float)
    log_cnt = np.log1p(counts)                          # (n_classes,)

    present = np.where(counts > 0)[0]
    signal  = log_cnt[present].reshape(-1, 1)           # (n_present, 1)

    best_aic, best_gmm, best_n = np.inf, None, 1
    upper = min(len(present), max_components)
    for n in range(1, upper + 1):
        try:
            gmm = GaussianMixture(
                n_components=n, random_state=seed,
                n_init=10, max_iter=500, reg_covar=1e-3,
            )
            gmm.fit(signal)
            aic = gmm.aic(signal)
            if aic < best_aic:
                best_aic, best_gmm, best_n = aic, gmm, n
        except Exception as e:
            logger.warning(f"GMM n={n} failed: {e}")
            break

    # Class-level hard assignment
    preds      = best_gmm.predict(signal)               # (n_present,)
    class_comp = np.full(n_classes, -1, dtype=int)
    for i, cls_id in enumerate(present):
        class_comp[cls_id] = int(preds[i])

    logger.info(f"GMM (class-level): selected N={best_n} (AIC={best_aic:.1f})")
    for n in range(best_n):
        cls_in = [int(c) for c in range(n_classes) if class_comp[c] == n]
        logger.info(
            f"  Component {n}: classes={cls_in} | "
            f"log_counts={[round(float(log_cnt[c]), 2) for c in cls_in]}"
        )

    # Merge singleton components
    class_comp, n_final = _merge_single_class_components(
        class_comp, log_cnt, n_classes, logger
    )

    # Propagate class assignment to every sample
    sample_assignments = class_comp[y_s]
    return sample_assignments, n_final, class_comp


# ── Stage-2 SMOTE (per component) ─────────────────────────────────────────────
def stage2_smote(X_sn, y_sn, seed: int, logger, name: str) -> tuple:
    """Balance classes within one GMM component."""
    classes, counts = np.unique(y_sn, return_counts=True)
    if len(classes) < 2:
        # Single-class component: SMOTE inapplicable
        return X_sn, y_sn

    k = min(5, int(counts.min()) - 1)
    if k < 1:
        logger.warning(f"  [{name}] Stage-2 SMOTE: k_neighbors<1, skipped.")
        return X_sn, y_sn

    t0 = time.time()
    sm = SMOTE(k_neighbors=k, random_state=seed)
    X_aug, y_aug = sm.fit_resample(X_sn, y_sn)
    logger.info(
        f"  [{name}] Stage-2 SMOTE: {len(y_sn):,} → {len(y_aug):,} samples "
        f"in {time.time() - t0:.1f}s"
    )
    return X_aug.astype(np.float32), y_aug.astype(int)


# ── XGBoost helpers ────────────────────────────────────────────────────────────
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
        p.update(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
        )
    return p


# ── Expert training (local classes only) ──────────────────────────────────────
def train_expert_local(X_sn, y_sn, X_va, y_va,
                       n_estimators: int, device: str, seed: int,
                       logger, name: str):
    """
    Train XGBoost on the component's classes only (non-overlapping MATI design).
    Labels are remapped to 0..k-1 locally.
    Returns (model, local_classes) or (None, local_classes) if k < 2.
    """
    local_classes = sorted(int(c) for c in np.unique(y_sn))
    k = len(local_classes)

    if k < 2:
        logger.warning(
            f"  [{name}] only 1 class ({local_classes}) — skipping expert."
        )
        return None, local_classes

    label_map = {c: i for i, c in enumerate(local_classes)}
    y_local   = np.array([label_map[int(c)] for c in y_sn], dtype=int)

    # Val: filter to component classes and remap
    va_mask = np.isin(y_va, local_classes)
    if va_mask.sum() == 0:
        # Fallback: use a slice of training data for early stopping
        n_fb = min(1000, len(X_sn))
        X_va_loc = X_sn[:n_fb]
        y_va_loc = y_local[:n_fb]
    else:
        X_va_loc = X_va[va_mask]
        y_va_loc = np.array([label_map[int(c)] for c in y_va[va_mask]], dtype=int)

    t0 = time.time()
    model = xgb.XGBClassifier(**xgb_params(k, n_estimators, device, seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_sn, y_local,
            eval_set=[(X_va_loc, y_va_loc)],
            verbose=False,
        )
    logger.info(
        f"  [{name}] trained in {time.time() - t0:.1f}s | "
        f"k={k} local classes: {local_classes}"
    )
    return model, local_classes


def train_all_experts(X_s, y_s, assignments, N: int,
                      X_va, y_va, n_classes: int,
                      n_estimators: int, device: str, seed: int,
                      logger) -> tuple:
    """Stage-2 SMOTE + expert training for each GMM component."""
    models, comp_classes = [], []

    for n in range(N):
        mask = assignments == n
        X_sn, y_sn = X_s[mask], y_s[mask]
        name = f"Expert_{n}"

        logger.info(
            f"Component {n}: {len(y_sn):,} samples | "
            f"classes={sorted(int(c) for c in np.unique(y_sn))}"
        )
        X_aug, y_aug = stage2_smote(X_sn, y_sn, seed, logger, name)
        m, lc = train_expert_local(
            X_aug, y_aug, X_va, y_va,
            n_estimators, device, seed, logger, name,
        )
        if m is None:
            continue   # skip single-class component
        models.append(m)
        comp_classes.append(lc)

    return models, comp_classes


# ── Inference: local → global probability mapping ──────────────────────────────
def predict_proba_global(model, local_classes: list, n_classes: int,
                         X, chunk: int = 50_000) -> np.ndarray:
    """
    Run model.predict_proba (local k-class) and zero-pad to global n_class.
    Non-component class positions remain 0: the expert declares no opinion
    on classes outside its training set.
    """
    out = []
    for s in range(0, len(X), chunk):
        Xb = X[s : s + chunk]
        lp = np.asarray(model.predict_proba(Xb), dtype=np.float32)
        if lp.ndim == 1:          # binary edge case
            lp = np.column_stack([1 - lp, lp])
        gp = np.zeros((len(Xb), n_classes), dtype=np.float32)
        for li, gc in enumerate(local_classes):
            gp[:, gc] = lp[:, li]
        out.append(gp)
    return np.concatenate(out, axis=0)


def moe_predict(models, comp_classes, weights, n_classes: int,
                X, chunk: int = 50_000) -> np.ndarray:
    """Weighted ensemble of global probability vectors → argmax."""
    probs = np.stack([
        predict_proba_global(m, lc, n_classes, X, chunk)
        for m, lc in zip(models, comp_classes)
    ])  # [E, N, C]
    ensemble = np.einsum("e,enc->nc", weights, probs)
    return np.argmax(ensemble, axis=1)


# ── Test set construction (3 distributions) ────────────────────────────────────
def make_test_sets(X_te, y_te, seed: int, logger,
                   balanced_min: int = 100) -> list:
    """
    Return list of (X, y, name) triples:
      Normal   — original stratified distribution (no resampling)
      Balanced — balanced_min samples per class (replace=True for small classes)
      Inverse  — P(class c) ∝ 1/count_c at class level (rare classes heavy)

    Inverse bug fix:
      Old (wrong): sample ∝ 1/count_c at SAMPLE level
                   → total weight per class = count_c × (1/count_c) = 1
                   → every class gets equal total weight → uniform, not inverse
      New (correct): assign class-level probability ∝ 1/count_c,
                     then sample uniformly within each chosen class.
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(y_te)
    counts  = np.array([int(np.sum(y_te == c)) for c in classes])

    # Normal
    te_normal = (X_te, y_te, "Normal")

    # Balanced: balanced_min samples per class; replace=True for small classes
    bal_n   = max(balanced_min, int(counts.min()))
    idx_bal = np.concatenate([
        rng.choice(np.where(y_te == c)[0], bal_n, replace=True)
        for c in classes
    ])
    rng.shuffle(idx_bal)
    te_balanced = (X_te[idx_bal], y_te[idx_bal], "Balanced")

    # Inverse: class-level probability ∝ 1/count_c  (rare classes get high P)
    class_inv_prob = 1.0 / counts               # one value per class
    class_inv_prob /= class_inv_prob.sum()      # normalise over classes

    n_inv = len(y_te)
    # Step 1: pick n_inv classes according to inverse probability
    sampled_class_idx = rng.choice(len(classes), size=n_inv,
                                   replace=True, p=class_inv_prob)
    sampled_classes   = classes[sampled_class_idx]

    # Step 2: for each chosen class, pick one sample uniformly from that class
    cls_to_idx = {int(c): np.where(y_te == c)[0] for c in classes}
    # Group by class to avoid per-sample Python loops
    idx_inv = np.empty(n_inv, dtype=int)
    unique_sc, sc_counts = np.unique(sampled_classes, return_counts=True)
    pos = 0
    for cls, cnt in zip(unique_sc, sc_counts):
        pool = cls_to_idx[int(cls)]
        idx_inv[pos : pos + cnt] = rng.choice(pool, size=cnt, replace=True)
        pos += cnt
    rng.shuffle(idx_inv)
    te_inverse = (X_te[idx_inv], y_te[idx_inv], "Inverse")

    for _, y_t, nm in [te_normal, te_balanced, te_inverse]:
        per_class = np.bincount(y_t, minlength=int(classes.max()) + 1)[classes]
        logger.info(
            f"Test [{nm}]: {len(y_t):,} samples | "
            f"per-class counts: {per_class.tolist()}"
        )
    return [te_normal, te_balanced, te_inverse]


# ── VIME-style perturbation ────────────────────────────────────────────────────
def vime_perturb(X: np.ndarray, p_mask: float,
                 rng: np.random.Generator) -> np.ndarray:
    B, D = X.shape
    mask    = rng.random((B, D)) < p_mask
    ref_rows = rng.integers(0, B, size=(B, D))
    X_ref   = X[ref_rows, np.arange(D)]
    return np.where(mask, X_ref, X)


# ── TTA: MATI Algorithm 1 ──────────────────────────────────────────────────────
def tta_aggregate(models, comp_classes, n_classes: int,
                  X_te: np.ndarray,
                  n_epochs: int, lr: float, p_mask: float,
                  batch_size: int, seed: int, logger,
                  early_stop_thr: float = 0.05) -> np.ndarray:
    """
    MATI Algorithm 1 adapted for classification.

    All experts are free (no fixed-weight tail expert).
    CPG loss: S = (1/B) Σ ‖ŷ¹ - ŷ²‖²
    Gradient: ∂S/∂w_j = (2/B) · s_j · Σ_bc diff_bc · [(v¹_j-ȳ¹)-(v²_j-ȳ²)]_bc
    Early stop: any w_i ≤ early_stop_thr  (MATI paper criterion)
    """
    E   = len(models)
    rng = np.random.default_rng(seed)
    w   = np.zeros(E, dtype=np.float64)   # logits; uniform init via zeros

    m_adam = np.zeros(E)
    v_adam = np.zeros(E)
    t_step = 0
    β1, β2, ε_adam = 0.9, 0.999, 1e-8

    N = len(X_te)
    logger.info(
        f"TTA: E={E} experts | epochs={n_epochs} | lr={lr} | "
        f"p_mask={p_mask} | batch={batch_size} | early_stop_thr={early_stop_thr}"
    )

    def _softmax(v: np.ndarray) -> np.ndarray:
        e = np.exp(v - v.max())
        return e / e.sum()

    final_w = _softmax(w)
    for epoch in range(n_epochs):
        perm        = rng.permutation(N)
        epoch_loss  = 0.0
        n_batches   = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb  = X_te[perm[start:end]]
            B   = end - start

            x1 = vime_perturb(Xb, p_mask, rng)
            x2 = vime_perturb(Xb, p_mask, rng)

            # [E, B, C] probability tensors
            v1 = np.stack([
                predict_proba_global(m, lc, n_classes, x1, chunk=B)
                for m, lc in zip(models, comp_classes)
            ])
            v2 = np.stack([
                predict_proba_global(m, lc, n_classes, x2, chunk=B)
                for m, lc in zip(models, comp_classes)
            ])

            s   = _softmax(w)
            y1  = np.einsum("e,ebc->bc", s, v1)
            y2  = np.einsum("e,ebc->bc", s, v2)
            diff = y1 - y2                    # [B, C]

            loss = float(np.mean(diff ** 2))
            epoch_loss += loss
            n_batches  += 1

            # Gradient (chain rule through softmax)
            r1   = v1 - y1[None]             # [E, B, C]  residuals view-1
            r2   = v2 - y2[None]             # [E, B, C]  residuals view-2
            gap  = np.einsum("bc,ebc->e", diff, r1 - r2)
            grad = (2.0 / B) * s * gap

            t_step += 1
            m_adam = β1 * m_adam + (1 - β1) * grad
            v_adam = β2 * v_adam + (1 - β2) * grad ** 2
            m_hat  = m_adam / (1 - β1 ** t_step)
            v_hat  = v_adam / (1 - β2 ** t_step)
            w     -= lr * m_hat / (np.sqrt(v_hat) + ε_adam)

        final_w   = _softmax(w)
        avg_loss  = epoch_loss / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info(
                f"  epoch {epoch + 1:3d}/{n_epochs} | "
                f"loss={avg_loss:.6f} | w={final_w.round(3)}"
            )

        # MATI early-stop criterion
        if np.any(final_w <= early_stop_thr):
            logger.info(
                f"  Early stop epoch {epoch + 1}: "
                f"min_w={final_w.min():.4f} ≤ {early_stop_thr}"
            )
            break

    return final_w


# ── Baseline ───────────────────────────────────────────────────────────────────
def train_baseline(X_tr, y_tr, X_va, y_va,
                   n_classes: int, n_estimators: int,
                   device: str, seed: int, logger):
    logger.info("Training baseline XGBoost (balanced sample weights) …")
    t0 = time.time()
    counts = np.maximum(np.bincount(y_tr), 1)
    w = (len(y_tr) / (len(counts) * counts[y_tr])).astype(np.float32)
    model = xgb.XGBClassifier(**xgb_params(n_classes, n_estimators, device, seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr, sample_weight=w,
                  eval_set=[(X_va, y_va)], verbose=False)
    logger.info(f"Baseline ready in {time.time() - t0:.1f}s")
    return model


# ── Output helpers ─────────────────────────────────────────────────────────────
def save_colored_table(rows: list, col_headers: list,
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


def build_report_rows(y_true, y_pred_b, y_pred_m, class_names: list) -> tuple:
    n_cls  = len(class_names)
    labels = np.arange(n_cls)
    counts = np.bincount(y_true, minlength=n_cls)
    order  = np.argsort(-counts)

    p_b, r_b, f_b, _ = precision_recall_fscore_support(
        y_true, y_pred_b, labels=labels, zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred_m, labels=labels, zero_division=0)

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
            y_true, y_pred_b, average=avg_key, zero_division=0)
        pM, rM, fM, _ = precision_recall_fscore_support(
            y_true, y_pred_m, average=avg_key, zero_division=0)
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MATI-repro: GMM + SMOTE MoE + TTA for imbalanced classification"
    )
    parser.add_argument("--data", required=True,
                        help="Path to preprocessed .pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--smote_floor", type=int, default=5000,
                        help="Stage-1 SMOTE target: min samples per class")
    parser.add_argument("--gmm_max_components", type=int, default=8,
                        help="Upper bound on GMM components (AIC selects best)")
    parser.add_argument("--balanced_min", type=int, default=100,
                        help="Samples per class in Balanced test set")
    parser.add_argument("--tta_epochs",     type=int,   default=50)
    parser.add_argument("--tta_lr",         type=float, default=0.01)
    parser.add_argument("--tta_p_mask",     type=float, default=0.1)
    parser.add_argument("--tta_batch",      type=int,   default=4096)
    parser.add_argument("--tta_early_stop", type=float, default=0.05,
                        help="Stop TTA when any w_i ≤ this threshold")
    parser.add_argument("--model", type=int, default=2, choices=[0, 1, 2],
                        help="0=baseline only  1=MoE+TTA only  2=both")
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", ts)
    os.makedirs(out_dir, exist_ok=True)
    logger  = setup_logger(os.path.join(out_dir, "experiment.log"))
    logger.info(f"Args: {vars(args)}")

    device = detect_device()
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    X, y, le = load_data(args.data)
    class_names = list(le.classes_)
    n_classes   = len(class_names)
    logger.info(
        f"Loaded {X.shape[0]:,} samples | "
        f"{X.shape[1]} features | {n_classes} classes"
    )

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, args.seed)
    logger.info(
        f"Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}"
    )
    for i, c in enumerate(class_names):
        cnt = int(np.sum(y_tr == i))
        logger.info(f"  [{i:2d}] {c:40s} n_train={cnt:,}")

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline = None
    if args.model in (0, 2):
        baseline = train_baseline(
            X_tr, y_tr, X_va, y_va,
            n_classes, args.n_estimators, device, args.seed, logger,
        )

    # ── MoE pipeline ──────────────────────────────────────────────────────────
    models, comp_classes = None, None
    if args.model in (1, 2):
        # Stage-1: global SMOTE
        X_s, y_s = stage1_smote(
            X_tr, y_tr, args.smote_floor, args.seed, logger
        )

        # GMM expert assignment (class-level, with singleton merge)
        assignments, N, _ = fit_gmm_and_assign(
            y_s, n_classes, args.gmm_max_components, args.seed, logger
        )

        # Stage-2: regional SMOTE + expert training
        logger.info(f"Training {N} expert(s) …")
        t0 = time.time()
        models, comp_classes = train_all_experts(
            X_s, y_s, assignments, N,
            X_va, y_va, n_classes,
            args.n_estimators, device, args.seed, logger,
        )
        logger.info(
            f"{len(models)} expert(s) trained in {time.time() - t0:.1f}s"
        )
        del X_s, y_s, assignments
        gc.collect()

    # ── Test sets ─────────────────────────────────────────────────────────────
    test_sets = make_test_sets(X_te, y_te, args.seed, logger,
                               balanced_min=args.balanced_min)

    # ── Evaluate per test distribution ────────────────────────────────────────
    for X_t, y_t, set_name in test_sets:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Test distribution: {set_name}  ({len(y_t):,} samples)")

        y_pred_b = None
        if args.model in (0, 2) and baseline is not None:
            y_pred_b = np.asarray(baseline.predict(X_t), dtype=int)
            logger.info(f"\n── Baseline [{set_name}] ──")
            logger.info(
                "\n" + classification_report(
                    y_t, y_pred_b, target_names=class_names, zero_division=0
                )
            )

        y_pred_m = None
        if args.model in (1, 2) and models is not None:
            weights = tta_aggregate(
                models, comp_classes, n_classes,
                X_t,
                n_epochs=args.tta_epochs,
                lr=args.tta_lr,
                p_mask=args.tta_p_mask,
                batch_size=args.tta_batch,
                seed=args.seed,
                logger=logger,
                early_stop_thr=args.tta_early_stop,
            )
            for i, (w_val, lc) in enumerate(zip(weights, comp_classes)):
                logger.info(f"  Weight [Expert_{i} | classes={lc}] = {w_val:.4f}")

            y_pred_m = moe_predict(
                models, comp_classes, weights, n_classes, X_t
            )
            logger.info(f"\n── MoE+TTA [{set_name}] ──")
            logger.info(
                "\n" + classification_report(
                    y_t, y_pred_m, target_names=class_names, zero_division=0
                )
            )

        if args.model == 2 and y_pred_b is not None and y_pred_m is not None:
            rows, col_headers = build_report_rows(
                y_t, y_pred_b, y_pred_m, class_names
            )
            tag = set_name.lower()

            png_path = os.path.join(out_dir, f"baseline_vs_moe_{tag}.png")
            save_colored_table(
                rows, col_headers, png_path,
                title=(
                    f"MATI-repro | Baseline vs MoE+TTA [{set_name}]"
                    f"  (seed={args.seed})"
                ),
            )

            df = pd.DataFrame(rows)
            df["delta_f1"] = df["f1(M)"] - df["f1(B)"]
            csv_path = os.path.join(out_dir, f"baseline_vs_moe_{tag}.csv")
            df.to_csv(csv_path, index=False)

            logger.info(f"PNG → {png_path}")
            logger.info(f"CSV → {csv_path}")

    logger.info(f"\nResults directory: {out_dir}")


if __name__ == "__main__":
    main()

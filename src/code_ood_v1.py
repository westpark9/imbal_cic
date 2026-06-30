#!/usr/bin/env python3
"""
Hierarchical OOD-gated disjoint-expert MoE with a learned stability meta-router.

Design:
  * Expert 0 is a global binary benign/attack gate trained on the FULL train set.
  * Attack classes are partitioned into disjoint attack experts (DoS/DDoS,
    Scan/Brute, WebBot/rare). Benign is not assigned to any attack expert.
  * Each attack expert trains:
      1) a local XGBoost classifier over its owned attack classes
      2) an OOD threshold over that classifier (energy-based by default)
  * Test-time decision for input x:
      1) expert 0 decides benign vs attack; if benign -> predict benign, stop.
      2) otherwise each attack expert accepts/rejects x via its OOD gate.
      3) if all attack experts reject -> fall back to benign.
      4) among the accepted (in-distribution) experts, measure TTA stability on
         (x, x') and route with the chosen --select_mode. The default
         "meta_router" is a logistic model learned on validation over per-expert
         [energy margin, TTA stability, local confidence] -> P(correct owner),
         which lets the data decide how much (and which sign) to trust stability.

Example:
  python src/code_ood_v1.py --data data/cic2017_proc.pkl \
    --ood_gate energy --select_mode meta_router --n_estimators 300
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
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BLUE = "#cce5ff"
RED = "#ffcccc"
YELLOW = "#fff9cc"
GRAY = "#f2f2f2"
WHITE = "#ffffff"
EPS = 0.001


@dataclass
class Expert:
    name: str
    global_classes: list[int]
    local_model: object
    binary_gate: object | None
    threshold: float
    score_scale: float
    prior: np.ndarray


def setup_logger(path):
    log = logging.getLogger("ood_tta_gate")
    log.handlers.clear()
    log.setLevel(logging.INFO)
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


def load_data(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=int)
    feature_names = d.get("feature_names") or d.get("feature_columns")
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(X.shape[1])]
    return X, y, d["label_encoder"], list(feature_names), d.get("dataset_type", "cic2017")


def load_external_ood(path, expected_dim):
    with open(path, "rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["X"], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"External OOD X must be 2D, got shape={X.shape}")
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"External OOD feature dim mismatch: got {X.shape[1]}, expected {expected_dim}")
    return X


def split_data(X, y, seed):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def split_external_ood(X_ood, seed):
    X_ood_tr, X_ood_va = train_test_split(
        X_ood, test_size=0.25, random_state=seed)
    return X_ood_tr, X_ood_va


def maybe_subsample(X, max_samples, seed):
    if max_samples is None or max_samples <= 0 or len(X) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), int(max_samples), replace=False))
    return X[idx]


def _xgb_params(n_cls, n_est, device, seed, binary=False):
    p = dict(n_estimators=n_est, max_depth=6, learning_rate=0.05,
             subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
             min_child_weight=1, tree_method="hist",
             early_stopping_rounds=20, random_state=seed, n_jobs=4)
    if device == "cuda":
        p["device"] = "cuda"
    if binary:
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


def _predict_logits(model, X, n_cls, chunk=50_000):
    booster = model.get_booster()
    out = []
    for s in range(0, len(X), chunk):
        raw = booster.predict(xgb.DMatrix(X[s:s + chunk]), output_margin=True)
        out.append(raw.reshape(-1, n_cls).astype(np.float32))
    return np.concatenate(out)


def normalize_name(s):
    return str(s).strip().lower().replace(" ", "-").replace("_", "-")


def find_benign_id(class_names):
    for i, name in enumerate(class_names):
        n = normalize_name(name)
        if n in ("benign", "normal") or "benign" in n:
            return i
    raise ValueError("Could not identify benign/normal class")


def ids_by_keywords(class_names, keywords):
    ids = []
    for i, name in enumerate(class_names):
        n = normalize_name(name)
        if any(k in n for k in keywords):
            ids.append(i)
    return ids


def ids_by_exact_names(class_names, names):
    wanted = {normalize_name(name) for name in names}
    ids = []
    for i, name in enumerate(class_names):
        if normalize_name(name) in wanted:
            ids.append(i)
    return ids


def ids_by_custom_match(class_names, match_fn):
    ids = []
    for i, name in enumerate(class_names):
        if match_fn(normalize_name(name)):
            ids.append(i)
    return ids


def build_attack_partitions(class_names, benign_id, mode, log):
    attack_ids = [i for i in range(len(class_names)) if i != benign_id]
    if mode == "taxonomy_cic":
        specs = [
            ("DoS_DDoS", ["dos", "ddos"]),
            ("Scan_Brute", ["portscan", "patator"]),
            ("WebBot_Rare", ["bot", "web-attack", "infiltration", "heartbleed"]),
        ]
        used = set()
        parts = []
        for name, kws in specs:
            cls = [i for i in ids_by_keywords(class_names, kws)
                   if i in attack_ids and i not in used]
            used.update(cls)
            if cls:
                parts.append((name, cls))
        leftover = [i for i in attack_ids if i not in used]
        if leftover:
            parts.append(("Other_Attack", leftover))
    elif mode == "taxonomy_cic_websplit":
        used = set()
        parts = []
        hardcoded_specs = [
            ("DoS_DDoS", lambda n: n in {
                "ddos", "dos-goldeneye", "dos-hulk",
                "dos-slowhttptest", "dos-slowloris",
            }),
            ("Scan_Brute", lambda n: n in {
                "ftp-patator", "portscan", "ssh-patator",
            }),
            ("WebBot_Main", lambda n: (
                n == "bot"
                or (n.startswith("web-attack") and n.endswith("brute-force"))
                or (n.startswith("web-attack") and n.endswith("xss"))
            )),
            ("Rare_Micro", lambda n: (
                n in {"heartbleed", "infiltration"}
                or (n.startswith("web-attack") and n.endswith("sql-injection"))
            )),
        ]
        for name, match_fn in hardcoded_specs:
            cls = [i for i in ids_by_custom_match(class_names, match_fn)
                   if i in attack_ids and i not in used]
            used.update(cls)
            if cls:
                parts.append((name, cls))
        leftover = [i for i in attack_ids if i not in used]
        if leftover:
            parts.append(("Other_Attack", leftover))
    elif mode == "support_balanced":
        raise ValueError("support_balanced requires y_train; use taxonomy_cic for now")
    else:
        raise ValueError(f"Unknown partition mode: {mode}")

    log.info("Attack-only disjoint expert partition:")
    for name, cls in parts:
        labels = ", ".join(class_names[i] for i in cls)
        log.info(f"  {name}: {labels}")
    return parts


def train_baseline(X_tr, y_tr, X_va, y_va, n_cls, n_est, device, seed, log):
    log.info("Training global XGBoost baseline")
    m = xgb.XGBClassifier(**_xgb_params(n_cls, n_est, device, seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_tr, y_tr, sample_weight=_balanced_w(y_tr),
              eval_set=[(X_va, y_va)], verbose=False)
    return m


def train_binary_attack_gate(X_tr, y_tr, X_va, y_va, benign_id, n_est,
                             device, seed, log):
    """Expert 0: global benign(0) vs attack(1) classifier on the full train set."""
    log.info("Training expert0 binary benign/attack gate")
    yb_tr = (y_tr != benign_id).astype(int)
    yb_va = (y_va != benign_id).astype(int)
    m = xgb.XGBClassifier(**_xgb_params(2, n_est, device, seed, binary=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_tr, yb_tr, sample_weight=_balanced_w(yb_tr),
              eval_set=[(X_va, yb_va)], verbose=False)
    return m


def train_local_classifier(X_tr, y_tr, X_va, y_va, global_classes,
                           n_est, device, seed):
    cls_to_local = {c: i for i, c in enumerate(global_classes)}
    tr_mask = np.isin(y_tr, global_classes)
    va_mask = np.isin(y_va, global_classes)
    Xl_tr = X_tr[tr_mask]
    yl_tr = np.array([cls_to_local[c] for c in y_tr[tr_mask]], dtype=int)
    Xl_va = X_va[va_mask]
    yl_va = np.array([cls_to_local[c] for c in y_va[va_mask]], dtype=int)
    model = xgb.XGBClassifier(
        **_xgb_params(len(global_classes), n_est, device, seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xl_tr, yl_tr, sample_weight=_balanced_w(yl_tr),
                  eval_set=[(Xl_va, yl_va)], verbose=False)
    counts = np.maximum(np.bincount(yl_tr, minlength=len(global_classes)), 1)
    prior = counts / counts.sum()
    return model, prior.astype(np.float32)


def _sample_rows(X, n, rng):
    if X is None or len(X) == 0 or n <= 0:
        return None
    replace = len(X) < n
    return X[rng.choice(len(X), n, replace=replace)]


def train_oe_binary_gate(X_tr, y_tr, X_va, y_va, global_classes,
                         n_est, device, seed, neg_ratio,
                         X_ood_tr=None, X_ood_va=None, external_ood_ratio=0.0):
    rng = np.random.default_rng(seed)
    pos_tr = np.where(np.isin(y_tr, global_classes))[0]
    neg_tr = np.where(~np.isin(y_tr, global_classes))[0]
    n_neg = min(len(neg_tr), max(len(pos_tr), int(len(pos_tr) * neg_ratio)))
    neg_tr = rng.choice(neg_tr, n_neg, replace=False)
    n_ext = int(len(pos_tr) * external_ood_ratio)
    X_ext_tr = _sample_rows(X_ood_tr, n_ext, rng)
    X_gate_tr = [X_tr[pos_tr], X_tr[neg_tr]]
    y_gate_tr = [np.ones(len(pos_tr), dtype=int), np.zeros(len(neg_tr), dtype=int)]
    if X_ext_tr is not None:
        X_gate_tr.append(X_ext_tr)
        y_gate_tr.append(np.zeros(len(X_ext_tr), dtype=int))
    X_gate_tr = np.concatenate(X_gate_tr, axis=0)
    yb_tr = np.concatenate(y_gate_tr, axis=0)
    perm = rng.permutation(len(yb_tr))
    X_gate_tr, yb_tr = X_gate_tr[perm], yb_tr[perm]

    pos_va = np.where(np.isin(y_va, global_classes))[0]
    neg_va = np.where(~np.isin(y_va, global_classes))[0]
    n_neg_va = min(len(neg_va), max(len(pos_va), int(len(pos_va) * neg_ratio)))
    neg_va = rng.choice(neg_va, n_neg_va, replace=False)
    n_ext_va = int(len(pos_va) * external_ood_ratio)
    X_ext_va = _sample_rows(X_ood_va, n_ext_va, rng)
    X_gate_va = [X_va[pos_va], X_va[neg_va]]
    y_gate_va = [np.ones(len(pos_va), dtype=int), np.zeros(len(neg_va), dtype=int)]
    if X_ext_va is not None:
        X_gate_va.append(X_ext_va)
        y_gate_va.append(np.zeros(len(X_ext_va), dtype=int))
    X_gate_va = np.concatenate(X_gate_va, axis=0)
    yb_va = np.concatenate(y_gate_va, axis=0)
    perm = rng.permutation(len(yb_va))
    X_gate_va, yb_va = X_gate_va[perm], yb_va[perm]

    model = xgb.XGBClassifier(**_xgb_params(2, n_est, device, seed, binary=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_gate_tr, yb_tr, sample_weight=_balanced_w(yb_tr),
                  eval_set=[(X_gate_va, yb_va)], verbose=False)
    return model


def logsumexp(a, axis=1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def imood_adjust_from_proba(proba, prior, alpha):
    eps = 1e-8
    ratio = proba / (prior[None, :] + eps)
    return alpha * np.log(np.maximum(ratio.mean(axis=1), eps))


def expert_raw_scores(expert, X, gate, alpha):
    proba = _predict_proba(expert.local_model, X)
    if gate in ("energy", "imood_energy"):
        logits = _predict_logits(expert.local_model, X, len(expert.global_classes))
        score = logsumexp(logits, axis=1)
    elif gate in ("oe", "imood_oe"):
        if expert.binary_gate is None:
            raise ValueError("OE gate requested but binary gate is missing")
        score = _predict_proba(expert.binary_gate, X)[:, 1]
    else:
        raise ValueError(f"Unknown OOD gate: {gate}")
    if gate.startswith("imood_"):
        score = score + imood_adjust_from_proba(proba, expert.prior, alpha)
    return score.astype(np.float32), proba.astype(np.float32)


def calibrate_threshold(expert, X_va, y_va, gate, q, alpha):
    mask = np.isin(y_va, expert.global_classes)
    scores, _ = expert_raw_scores(expert, X_va[mask], gate, alpha)
    tau = float(np.quantile(scores, q))
    scale = float(np.std(scores) + 1e-6)
    return tau, scale


def calibrate_threshold_per_class(expert, X_va, y_va, gate, q, alpha):
    """Threshold = min over owned classes of each class's q-quantile.

    Protects minority owned classes (e.g. ssh-patator) from being clipped by a
    pooled quantile that a majority owned class (e.g. portscan) dominates.
    """
    per_class_tau, all_scores = [], []
    for c in expert.global_classes:
        cmask = y_va == c
        if not cmask.any():
            continue
        s, _ = expert_raw_scores(expert, X_va[cmask], gate, alpha)
        per_class_tau.append(float(np.quantile(s, q)))
        all_scores.append(s)
    if not all_scores:
        return calibrate_threshold(expert, X_va, y_va, gate, q, alpha)
    all_scores = np.concatenate(all_scores)
    tau = float(min(per_class_tau))
    scale = float(np.std(all_scores) + 1e-6)
    return tau, scale


def calibrate_threshold_hard_negative(expert, X_va, y_va, gate, q, alpha,
                                      benign_id, quantiles,
                                      benign_penalty, non_owned_penalty,
                                      min_owned_accept, min_owned_penalty):
    """Choose threshold with per-class ID floors and hard-negative penalties.

    Unlike `per_class_min`, this uses non-owned attack / benign validation rows
    as rejection negatives while protecting minority owned classes with a
    per-class accept-rate floor.
    """
    scores, _ = expert_raw_scores(expert, X_va, gate, alpha)
    owned_mask = np.isin(y_va, expert.global_classes)
    benign_mask = y_va == benign_id
    non_owned_attack_mask = (y_va != benign_id) & ~owned_mask
    owned_scores = scores[owned_mask]
    if len(owned_scores) == 0:
        return calibrate_threshold(expert, X_va, y_va, gate, q, alpha)

    class_score_map = {}
    for cls in expert.global_classes:
        cls_scores = scores[y_va == cls]
        if len(cls_scores):
            class_score_map[int(cls)] = cls_scores
    if not class_score_map:
        return calibrate_threshold(expert, X_va, y_va, gate, q, alpha)

    candidate_taus = {
        float(np.quantile(owned_scores, q)),
        float(min(np.quantile(v, q) for v in class_score_map.values())),
    }
    for qv in quantiles:
        candidate_taus.add(float(np.quantile(owned_scores, qv)))
        for cls_scores in class_score_map.values():
            candidate_taus.add(float(np.quantile(cls_scores, qv)))
    candidate_taus = sorted(candidate_taus)

    best = None
    for tau in candidate_taus:
        accept = scores >= tau
        per_class_accept = []
        floor_shortfall = 0.0
        for cls_scores in class_score_map.values():
            cls_accept = float((cls_scores >= tau).mean())
            per_class_accept.append(cls_accept)
            if cls_accept < min_owned_accept:
                floor_shortfall += (min_owned_accept - cls_accept)
        mean_owned_accept = float(np.mean(per_class_accept))
        benign_accept = float(accept[benign_mask].mean()) if benign_mask.any() else 0.0
        non_owned_accept = (
            float(accept[non_owned_attack_mask].mean()) if non_owned_attack_mask.any() else 0.0)
        objective = (
            mean_owned_accept
            - benign_penalty * benign_accept
            - non_owned_penalty * non_owned_accept
            - min_owned_penalty * floor_shortfall
        )
        row = {
            "threshold": float(tau),
            "objective": float(objective),
            "mean_owned_accept": mean_owned_accept,
            "benign_accept": benign_accept,
            "non_owned_accept": non_owned_accept,
            "floor_shortfall": float(floor_shortfall),
        }
        if best is None or row["objective"] > best["objective"]:
            best = row
    scale = float(np.std(owned_scores) + 1e-6)
    return float(best["threshold"]), scale


def parse_float_list(s):
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("Expected at least one float value")
    return vals


def tune_expert_thresholds(experts, X_va, y_va, benign_id, args, out_dir, log):
    quantiles = parse_float_list(args.sweep_quantiles)
    cls_to_expert = _expert_owner_map(experts)
    rows = []
    selected_rows = []
    for e, expert in enumerate(experts):
        scores, proba = expert_raw_scores(expert, X_va, args.ood_gate, args.imood_alpha)
        owned_mask = np.isin(y_va, expert.global_classes)
        benign_mask = y_va == benign_id
        non_owned_attack_mask = (y_va != benign_id) & ~owned_mask
        owned_scores = scores[owned_mask]
        if len(owned_scores) == 0:
            continue
        local_preds = local_pred_to_global(expert, proba.argmax(axis=1))
        best = None
        for q in quantiles:
            tau = float(np.quantile(owned_scores, q))
            accept = scores >= tau
            owned_accept = float(accept[owned_mask].mean()) if owned_mask.any() else np.nan
            benign_accept = float(accept[benign_mask].mean()) if benign_mask.any() else np.nan
            non_owned_accept = (
                float(accept[non_owned_attack_mask].mean()) if non_owned_attack_mask.any() else np.nan)
            owned_local_acc = float((local_preds[owned_mask & accept] == y_va[owned_mask & accept]).mean()) \
                if (owned_mask & accept).any() else np.nan
            objective = (
                owned_accept
                - args.sweep_benign_penalty * (0.0 if np.isnan(benign_accept) else benign_accept)
                - args.sweep_non_owned_penalty * (0.0 if np.isnan(non_owned_accept) else non_owned_accept)
            )
            if owned_accept < args.sweep_min_owned_accept:
                objective -= args.sweep_min_owned_penalty * (args.sweep_min_owned_accept - owned_accept)
            row = {
                "expert": expert.name,
                "quantile": q,
                "threshold": tau,
                "objective": float(objective),
                "owned_accept_rate": owned_accept,
                "benign_accept_rate": benign_accept,
                "non_owned_attack_accept_rate": non_owned_accept,
                "owned_local_acc_given_accept": owned_local_acc,
                "owned_support": int(owned_mask.sum()),
                "benign_support": int(benign_mask.sum()),
                "non_owned_attack_support": int(non_owned_attack_mask.sum()),
            }
            rows.append(row)
            if best is None or row["objective"] > best["objective"]:
                best = row
        old_threshold = expert.threshold
        expert.threshold = float(best["threshold"])
        expert.score_scale = float(np.std(owned_scores) + 1e-6)
        selected = dict(best)
        selected["old_threshold"] = float(old_threshold)
        selected["score_scale"] = float(expert.score_scale)
        selected_rows.append(selected)
        log.info(
            f"  tuned {expert.name}: q={best['quantile']:.4f}, "
            f"threshold {old_threshold:.4f}->{expert.threshold:.4f}, "
            f"owned_accept={best['owned_accept_rate']:.4f}, "
            f"benign_accept={best['benign_accept_rate']:.4f}, "
            f"non_owned_accept={best['non_owned_attack_accept_rate']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "expert_threshold_sweep.csv"), index=False)
    pd.DataFrame(selected_rows).to_csv(
        os.path.join(out_dir, "expert_threshold_selected.csv"), index=False)


def train_experts(X_tr, y_tr, X_va, y_va, partitions, n_est, gate_n_est, device,
                  seed, gate, q, alpha, neg_ratio, log, threshold_mode="fixed_quantile",
                  X_ood_tr=None, X_ood_va=None, external_ood_ratio=0.0,
                  benign_id=None, args=None):
    per_class = threshold_mode == "per_class_min"
    hard_negative = threshold_mode == "hard_negative"
    experts = []
    for e, (name, classes) in enumerate(partitions):
        log.info(f"Training expert {e}: {name} ({len(classes)} classes)")
        local, prior = train_local_classifier(
            X_tr, y_tr, X_va, y_va, classes, n_est, device, seed + e)
        binary = None
        if gate in ("oe", "imood_oe"):
            binary = train_oe_binary_gate(
                X_tr, y_tr, X_va, y_va, classes, gate_n_est, device,
                seed + 100 + e, neg_ratio, X_ood_tr, X_ood_va,
                external_ood_ratio)
        expert = Expert(name, classes, local, binary, 0.0, 1.0, prior)
        if hard_negative:
            tau, scale = calibrate_threshold_hard_negative(
                expert, X_va, y_va, gate, q, alpha, benign_id,
                parse_float_list(args.sweep_quantiles),
                args.sweep_benign_penalty, args.sweep_non_owned_penalty,
                args.sweep_min_owned_accept, args.sweep_min_owned_penalty)
        else:
            calib = calibrate_threshold_per_class if per_class else calibrate_threshold
            tau, scale = calib(expert, X_va, y_va, gate, q, alpha)
        expert.threshold = tau
        expert.score_scale = scale
        experts.append(expert)
        log.info(f"  threshold(q={q:.4f}, mode={threshold_mode})={tau:.4f}, "
                 f"score_scale={scale:.4f}")
    return experts


def build_feature_groups(feature_names):
    norm = [normalize_name(x) for x in feature_names]
    groups = {
        "volume": ["byte", "packet", "flow-bytes", "flow-packets", "length"],
        "timing": ["iat", "duration", "active", "idle"],
        "tcp": ["flag", "urg", "ack", "psh", "rst", "syn", "fin"],
    }
    out = {}
    for name, pats in groups.items():
        idx = [i for i, n in enumerate(norm) if any(p in n for p in pats)]
        if idx:
            out[name] = np.array(sorted(set(idx)), dtype=int)
    if not out:
        idx = np.arange(len(feature_names))
        out = {
            "g0": idx[0::3],
            "g1": idx[1::3],
            "g2": idx[2::3],
        }
    return {k: v for k, v in out.items() if len(v) > 0}


def perturb_batch(X, mode, rng, col_means, col_stds, p_mask, noise_std,
                  feature_groups):
    if mode == "gaussian":
        return X + rng.standard_normal(X.shape).astype(np.float32) * (
            col_stds * noise_std).astype(np.float32)
    if mode == "mask":
        mask = rng.random(X.shape) < p_mask
        return np.where(mask, col_means.astype(np.float32), X)
    if mode == "feature_group_mask":
        Xp = X.copy()
        keys = list(feature_groups)
        for i in range(len(Xp)):
            key = keys[int(rng.integers(0, len(keys)))]
            idx = feature_groups[key]
            Xp[i, idx] = col_means[idx]
        return Xp
    raise ValueError(f"Unknown perturb mode: {mode}")


def js_stability(p0, p1):
    eps = 1e-10
    m = 0.5 * (p0 + p1)
    kl0 = (p0 * (np.log(p0 + eps) - np.log(m + eps))).sum(axis=1)
    kl1 = (p1 * (np.log(p1 + eps) - np.log(m + eps))).sum(axis=1)
    jsd = 0.5 * (kl0 + kl1)
    return np.clip(1.0 - jsd / np.log(2.0), 0.0, 1.0).astype(np.float32)


def compute_stability(expert, X, p0, args, rng, col_means, col_stds, feature_groups):
    vals = []
    for _ in range(args.n_views):
        Xp = perturb_batch(X, args.perturb, rng, col_means, col_stds,
                           args.p_mask, args.noise_std, feature_groups)
        pp = _predict_proba(expert.local_model, Xp)
        vals.append(js_stability(p0, pp))
    return np.mean(vals, axis=0).astype(np.float32)


def local_pred_to_global(expert, local_pred):
    arr = np.asarray(expert.global_classes, dtype=int)
    return arr[local_pred]


def compute_expert_signals(experts, X, args, col_means, col_stds, feature_groups,
                           compute_stab):
    """Per-expert (energy margin, accept, stability, local pred/conf) for rows in X."""
    N, E = len(X), len(experts)
    out = {
        "scores": np.zeros((N, E), dtype=np.float32),
        "margins": np.zeros((N, E), dtype=np.float32),
        "accepted": np.zeros((N, E), dtype=bool),
        "stability": np.zeros((N, E), dtype=np.float32),
        "local_preds": np.zeros((N, E), dtype=int),
        "local_conf": np.zeros((N, E), dtype=np.float32),
    }
    if N == 0:
        return out
    rng = np.random.default_rng(args.seed + 999)
    for e, expert in enumerate(experts):
        s, p = expert_raw_scores(expert, X, args.ood_gate, args.imood_alpha)
        out["scores"][:, e] = s
        out["margins"][:, e] = (s - expert.threshold) / expert.score_scale
        out["accepted"][:, e] = s >= expert.threshold
        out["local_preds"][:, e] = p.argmax(axis=1)
        out["local_conf"][:, e] = p.max(axis=1)
        if compute_stab:
            out["stability"][:, e] = compute_stability(
                expert, X, p, args, rng, col_means, col_stds, feature_groups)
    return out


def _router_features(sig, j, cols):
    return np.stack([sig["margins"][j, cols],
                     sig["stability"][j, cols],
                     sig["local_conf"][j, cols]], axis=1)


def split_primary_and_fallback_experts(experts):
    """Use the last expert as a fallback branch when 4+ experts exist.

    This keeps the main routing competition among the earlier experts and lets
    the fallback expert act only after the primary experts all reject.
    """
    if len(experts) >= 4:
        return np.arange(len(experts) - 1, dtype=int), len(experts) - 1
    return np.arange(len(experts), dtype=int), None


def train_meta_router(experts, X_va, y_va, binary_gate, benign_id, args,
                      col_means, col_stds, feature_groups, log):
    """Logistic owner-vs-not router over [margin, stability, conf] of accepted experts.

    Trained on the same population the router sees at test time: validation samples
    the binary gate calls "attack", restricted to experts that pass the OOD gate.
    """
    proba_attack = _predict_proba(binary_gate, X_va)[:, 1]
    idx = np.where(proba_attack >= args.binary_threshold)[0]
    if len(idx) == 0:
        log.warning("Meta-router: no val samples pass the binary attack gate; skipping")
        return None
    sig = compute_expert_signals(experts, X_va[idx], args, col_means, col_stds,
                                 feature_groups, compute_stab=True)
    cls_to_expert = _expert_owner_map(experts)
    primary_experts, fallback_expert = split_primary_and_fallback_experts(experts)
    feats, labels = [], []
    for j, i in enumerate(idx):
        cand = primary_experts[sig["accepted"][j, primary_experts]]
        if len(cand) == 0:
            continue
        owner = cls_to_expert.get(int(y_va[i]), -1)
        if fallback_expert is not None and owner == fallback_expert:
            continue
        feats.append(_router_features(sig, j, cand))
        labels.append((cand == owner).astype(int))
    if not feats:
        log.warning("Meta-router: no accepted candidates on val; skipping")
        return None
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0)
    if len(set(labels.tolist())) < 2:
        log.warning("Meta-router: only one class among candidates; skipping")
        return None
    router = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
    router.fit(feats, labels)
    coef = router.named_steps["logisticregression"].coef_[0]
    log.info(
        f"Meta-router: {len(labels)} candidate rows (owner-positive={int(labels.sum())}); "
        f"standardized coef [margin={coef[0]:+.3f}, stability={coef[1]:+.3f}, "
        f"conf={coef[2]:+.3f}]")
    return router


def predict_ood_tta(experts, X, benign_id, binary_gate, meta_router, args,
                    col_means, col_stds, feature_groups, log):
    N, E = len(X), len(experts)
    primary_experts, fallback_expert = split_primary_and_fallback_experts(experts)
    proba_attack = _predict_proba(binary_gate, X)[:, 1]
    is_attack = proba_attack >= args.binary_threshold

    scores = np.zeros((N, E), dtype=np.float32)
    margins = np.zeros((N, E), dtype=np.float32)
    accepted = np.zeros((N, E), dtype=bool)
    stability = np.zeros((N, E), dtype=np.float32)
    local_preds = np.zeros((N, E), dtype=int)
    local_conf = np.zeros((N, E), dtype=np.float32)

    y_pred = np.full(N, benign_id, dtype=int)
    selected = np.full(N, -1, dtype=int)
    reason = np.full(N, "benign_binary", dtype=object)

    attack_idx = np.where(is_attack)[0]
    log.info(f"Binary gate: attack-predicted {is_attack.mean():.4f} "
             f"({len(attack_idx):,}/{N:,})")

    need_stab = args.select_mode in (
        "stability_tiebreak", "gate_stability", "meta_router")
    if len(attack_idx) > 0:
        sig = compute_expert_signals(
            experts, X[attack_idx], args, col_means, col_stds, feature_groups,
            compute_stab=need_stab)
        scores[attack_idx] = sig["scores"]
        margins[attack_idx] = sig["margins"]
        accepted[attack_idx] = sig["accepted"]
        stability[attack_idx] = sig["stability"]
        local_preds[attack_idx] = sig["local_preds"]
        local_conf[attack_idx] = sig["local_conf"]

        use_router = args.select_mode == "meta_router" and meta_router is not None
        for j, i in enumerate(attack_idx):
            active_primary = primary_experts[sig["accepted"][j, primary_experts]]
            if len(active_primary) == 0:
                best_primary_margin = float(np.max(sig["margins"][j, primary_experts])) \
                    if len(primary_experts) > 0 else float("-inf")
                fallback_margin = sig["margins"][j, fallback_expert] \
                    if fallback_expert is not None else float("-inf")
                fallback_ok = (
                    fallback_expert is not None
                    and sig["accepted"][j, fallback_expert]
                    and fallback_margin >= best_primary_margin + args.fallback_margin_gap
                )
                if fallback_ok:
                    chosen = int(fallback_expert)
                    reason[i] = "fallback_accept"
                    selected[i] = chosen
                    y_pred[i] = local_pred_to_global(experts[chosen], local_preds[i, chosen])
                else:
                    reason[i] = "all_reject"
                continue
            if len(active_primary) == 1:
                chosen = int(active_primary[0])
                reason[i] = "only_accept"
            elif use_router:
                p_owner = meta_router.predict_proba(
                    _router_features(sig, j, active_primary))[:, 1]
                chosen = int(active_primary[np.argmax(p_owner)])
                reason[i] = "meta_router"
            elif args.select_mode in ("gate_score", "meta_router"):
                chosen = int(active_primary[np.argmax(sig["margins"][j, active_primary])])
                reason[i] = "max_gate"
            elif args.select_mode == "stability_tiebreak":
                best = float(np.max(sig["margins"][j, active_primary]))
                cand = active_primary[
                    sig["margins"][j, active_primary] >= best - args.id_margin]
                chosen = int(cand[np.argmax(sig["stability"][j, cand])])
                reason[i] = "stability_tiebreak"
            elif args.select_mode == "gate_stability":
                combo = (sig["margins"][j, active_primary]
                         + args.stability_lambda * sig["stability"][j, active_primary])
                chosen = int(active_primary[np.argmax(combo)])
                reason[i] = "gate_stability"
            else:
                raise ValueError(f"Unknown select_mode: {args.select_mode}")
            selected[i] = chosen
            y_pred[i] = local_pred_to_global(experts[chosen], local_preds[i, chosen])

    diag = {
        "scores": scores,
        "margins": margins,
        "accepted": accepted,
        "stability": stability,
        "selected": selected,
        "reason": reason,
        "local_conf": local_conf,
        "is_attack": is_attack,
        "proba_attack": proba_attack,
    }
    log.info(f"Benign-by-binary-gate rate: {(reason == 'benign_binary').mean():.4f}")
    log.info(f"All-reject (attack-gated) rate: {(reason == 'all_reject').mean():.4f}")
    return y_pred, diag


def oracle_predict(experts, y_true, benign_id):
    cls_to_expert = {}
    for e, expert in enumerate(experts):
        for c in expert.global_classes:
            cls_to_expert[c] = e
    pred = np.full(len(y_true), benign_id, dtype=int)
    selected = np.full(len(y_true), -1, dtype=int)
    for i, y in enumerate(y_true):
        if y == benign_id:
            continue
        e = cls_to_expert.get(int(y), -1)
        if e >= 0:
            pred[i] = y
            selected[i] = e
    return pred, selected


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
    fig, ax = plt.subplots(figsize=(max(12, len(col_headers) * 1.2),
                                    max(4, len(rows) * 0.35)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_headers,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(col_headers))))
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cell_color(col, val, delta_cols, high_good, low_good):
    if not isinstance(val, (int, float)) or (isinstance(val, float) and np.isnan(val)):
        return WHITE
    if col in delta_cols:
        return BLUE if val > EPS else RED if val < -EPS else YELLOW
    if col in high_good:        # closer to 1.0 is better
        return BLUE if val >= 0.9 else YELLOW if val >= 0.7 else RED
    if col in low_good:         # closer to 0.0 is better
        return BLUE if val <= 0.02 else YELLOW if val <= 0.1 else RED
    return WHITE


def render_table_png(df, path, title="", delta_cols=(), high_good=(),
                     low_good=(), fmt="{:.3f}"):
    """Render a DataFrame as a color-coded PNG table (good=blue, bad=red)."""
    cols = list(df.columns)
    cell_text, cell_colors = [], []
    for _, row in df.iterrows():
        texts, colors = [], []
        for col in cols:
            v = row[col]
            if isinstance(v, float):
                texts.append("" if np.isnan(v) else fmt.format(v))
            else:
                texts.append(str(v))
            colors.append(_cell_color(col, v, delta_cols, high_good, low_good))
        cell_text.append(texts)
        cell_colors.append(colors)
    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 1.35),
                                    max(3, len(df) * 0.42)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _annot_heatmap(ax, M, row_labels, col_labels, owner_col, title,
                   cmap, vmin, vmax):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
        owner = owner_col[r]
        if owner is not None and owner >= 0:
            ax.add_patch(plt.Rectangle((owner - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="red", lw=2.0))
    return im


def save_gate_stability_dashboard(y_te, diag, experts, class_names, benign_id,
                                  out_dir, log):
    """One-glance heatmaps: per (true class x expert) OOD accept, selected,
    TTA stability, gate margin. Red box marks the owning expert — so you can see
    at a glance whether the owner accepts/selects and whether stability or margin
    actually separates the owner from non-owners."""
    counts = np.bincount(y_te, minlength=len(class_names))
    order = [c for c in np.argsort(-counts) if counts[c] > 0]
    E = len(experts)
    expert_names = [e.name for e in experts]
    cls_to_expert = _expert_owner_map(experts)
    accept = np.zeros((len(order), E), dtype=np.float32)
    select = np.zeros((len(order), E), dtype=np.float32)
    stab = np.zeros((len(order), E), dtype=np.float32)
    marg = np.zeros((len(order), E), dtype=np.float32)
    for r, c in enumerate(order):
        mask = y_te == c
        for e in range(E):
            accept[r, e] = diag["accepted"][mask, e].mean()
            select[r, e] = (diag["selected"][mask] == e).mean()
            stab[r, e] = diag["stability"][mask, e].mean()
            marg[r, e] = diag["margins"][mask, e].mean()
    owner_col = [cls_to_expert.get(int(c), -1) for c in order]
    row_labels = [f"{class_names[c]} ({int(counts[c])})" for c in order]

    fig, axes = plt.subplots(2, 2, figsize=(max(10, E * 1.8 + 5),
                                            max(7, len(order) * 0.55)))
    panels = [
        ("OOD accept rate", accept, "Blues", 0.0, 1.0),
        ("Selected rate", select, "Blues", 0.0, 1.0),
        ("Mean TTA stability", stab, "viridis", None, None),
        ("Mean gate margin (norm)", marg, "coolwarm", None, None),
    ]
    for ax, (t, M, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = _annot_heatmap(ax, M, row_labels, expert_names, owner_col, t,
                            cmap, vmin, vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("OOD gating & TTA stability per true class "
                 "(red box = owning expert)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(out_dir, "gate_stability_dashboard.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved gate_stability_dashboard.png")


def save_pipeline_funnel(y_te, y_pred, diag, experts, class_names, benign_id,
                         out_dir, log):
    """Stacked horizontal bars: for each attack class, the fraction lost at each
    pipeline stage (binary gate -> OOD reject -> wrong expert -> local error)
    vs correctly classified. Sums to 1 per class."""
    counts = np.bincount(y_te, minlength=len(class_names))
    cls_to_expert = _expert_owner_map(experts)
    attack_order = [c for c in np.argsort(-counts)
                    if counts[c] > 0 and c != benign_id]
    selected = diag["selected"]
    reason = diag["reason"]
    rows, labels = [], []
    for c in attack_order:
        mask = y_te == c
        n = int(mask.sum())
        owner = cls_to_expert.get(int(c), -1)
        benign_gate = float((reason[mask] == "benign_binary").mean())
        all_reject = float((reason[mask] == "all_reject").mean())
        routed = (selected[mask] >= 0)
        wrong_expert = float((routed & (selected[mask] != owner)).mean())
        owner_routed = routed & (selected[mask] == owner)
        correct = float((owner_routed & (y_pred[mask] == c)).mean())
        local_err = float((owner_routed & (y_pred[mask] != c)).mean())
        rows.append([correct, local_err, wrong_expert, all_reject, benign_gate])
        labels.append(f"{class_names[c]} ({n})")
    rows = np.array(rows, dtype=np.float32)
    seg_names = ["correct", "local error", "wrong expert",
                 "OOD all-reject", "benign gate miss"]
    seg_colors = ["#4caf50", "#ff9800", "#f44336", "#9e9e9e", "#212121"]
    fig, ax = plt.subplots(figsize=(11, max(3, len(labels) * 0.5)))
    left = np.zeros(len(labels), dtype=np.float32)
    yidx = np.arange(len(labels))
    for s in range(rows.shape[1]):
        ax.barh(yidx, rows[:, s], left=left, color=seg_colors[s],
                label=seg_names[s], edgecolor="white")
        left += rows[:, s]
    ax.set_yticks(yidx)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of true-class samples")
    ax.set_title("Per-class pipeline funnel (RECALL): where each true attack "
                 "class is lost")
    ax.legend(ncol=5, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pipeline_funnel.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Saved pipeline_funnel.png")


def save_precision_funnel(y_te, y_pred, benign_id, class_names, out_dir, log):
    """Stacked bars (PRECISION side): for each attack class the model PREDICTS,
    what fraction of those predictions are truly that class (correct) vs leaked
    from benign vs leaked from other attacks. Exposes precision collapse that the
    recall funnel hides (e.g. infiltration: recall 1.0 but predicted everywhere)."""
    counts = np.bincount(y_te, minlength=len(class_names))
    attack_order = [c for c in np.argsort(-counts)
                    if counts[c] > 0 and c != benign_id]
    rows, labels = [], []
    for c in attack_order:
        pmask = y_pred == c
        npred = int(pmask.sum())
        if npred == 0:
            rows.append([0.0, 0.0, 0.0])
            labels.append(f"{class_names[c]} (pred 0)")
            continue
        correct = float((y_te[pmask] == c).mean())
        from_benign = float((y_te[pmask] == benign_id).mean())
        from_other = max(0.0, 1.0 - correct - from_benign)
        rows.append([correct, from_other, from_benign])
        labels.append(f"{class_names[c]} (pred {npred})")
    rows = np.array(rows, dtype=np.float32)
    seg_names = ["correct (true class)", "false: other attack", "false: benign"]
    seg_colors = ["#4caf50", "#f44336", "#212121"]
    fig, ax = plt.subplots(figsize=(11, max(3, len(labels) * 0.5)))
    left = np.zeros(len(labels), dtype=np.float32)
    yidx = np.arange(len(labels))
    for s in range(rows.shape[1]):
        ax.barh(yidx, rows[:, s], left=left, color=seg_colors[s],
                label=seg_names[s], edgecolor="white")
        left += rows[:, s]
    ax.set_yticks(yidx)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of samples predicted as this class")
    ax.set_title("Per-class precision funnel: what contaminates each prediction")
    ax.legend(ncol=3, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_funnel.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Saved precision_funnel.png")


def save_per_class_results(y_te, y_base, y_moe, class_names, out_dir, log):
    n_cls = len(class_names)
    pb, rb, fb, sup = precision_recall_fscore_support(
        y_te, y_base, labels=np.arange(n_cls), zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_te, y_moe, labels=np.arange(n_cls), zero_division=0)
    order = [i for i in np.argsort(-sup) if sup[i] > 0]
    cols = ["class", "support", "prec(B)", "prec(M)", "recall(B)",
            "recall(M)", "f1(B)", "f1(M)", "delta_f1"]
    rows = []
    for i in order:
        rows.append({
            "class": class_names[i], "support": int(sup[i]),
            "prec(B)": float(pb[i]), "prec(M)": float(pm[i]),
            "recall(B)": float(rb[i]), "recall(M)": float(rm[i]),
            "f1(B)": float(fb[i]), "f1(M)": float(fm[i]),
            "delta_f1": float(fm[i] - fb[i]),
        })
    for avg_name in ["macro avg", "weighted avg"]:
        avg = avg_name.split()[0]
        rows.append({
            "class": avg_name, "_footer": True, "support": int(sup.sum()),
            "prec(B)": float(precision_score(y_te, y_base, average=avg, zero_division=0)),
            "prec(M)": float(precision_score(y_te, y_moe, average=avg, zero_division=0)),
            "recall(B)": float(recall_score(y_te, y_base, average=avg, zero_division=0)),
            "recall(M)": float(recall_score(y_te, y_moe, average=avg, zero_division=0)),
            "f1(B)": float(f1_score(y_te, y_base, average=avg, zero_division=0)),
            "f1(M)": float(f1_score(y_te, y_moe, average=avg, zero_division=0)),
        })
        rows[-1]["delta_f1"] = rows[-1]["f1(M)"] - rows[-1]["f1(B)"]
    pd.DataFrame([{k: v for k, v in r.items() if k != "_footer"}
                  for r in rows]).to_csv(
        os.path.join(out_dir, "baseline_vs_moe_per_class.csv"), index=False)
    save_colored_table(rows, cols, os.path.join(out_dir, "baseline_vs_moe_per_class.png"),
                       title="Baseline vs OOD-gated TTA MoE")
    log.info("Saved baseline_vs_moe_per_class.csv/png")


def _expert_owner_map(experts):
    cls_to_expert = {}
    for e, expert in enumerate(experts):
        for c in expert.global_classes:
            cls_to_expert[int(c)] = e
    return cls_to_expert


def _mean_or_nan(x):
    return float(np.mean(x)) if len(x) else np.nan


def save_bottleneck_results(y_te, y_base, y_pred, diag, experts, class_names,
                            benign_id, out_dir, log):
    n_cls = len(class_names)
    _, rb, fb, sup = precision_recall_fscore_support(
        y_te, y_base, labels=np.arange(n_cls), zero_division=0)
    _, rm, fm, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=np.arange(n_cls), zero_division=0)
    cls_to_expert = _expert_owner_map(experts)
    selected = diag["selected"]
    accepted = diag["accepted"]
    reasons = diag["reason"]
    margins = diag["margins"]
    stability = diag["stability"]
    local_conf = diag["local_conf"]

    rows = []
    for c, cname in enumerate(class_names):
        mask = y_te == c
        if not mask.any():
            continue
        row = {
            "class": cname,
            "support": int(mask.sum()),
            "baseline_recall": float(rb[c]),
            "moe_recall": float(rm[c]),
            "baseline_f1": float(fb[c]),
            "moe_f1": float(fm[c]),
            "delta_f1": float(fm[c] - fb[c]),
            "benign_gate_rate": float((reasons[mask] == "benign_binary").mean()),
            "all_reject_rate": float((reasons[mask] == "all_reject").mean()),
            "only_accept_rate": float((reasons[mask] == "only_accept").mean()),
            "multi_accept_rate": float((accepted[mask].sum(axis=1) > 1).mean()),
            "meta_router_rate": float((reasons[mask] == "meta_router").mean()),
            "stability_tiebreak_rate": float((reasons[mask] == "stability_tiebreak").mean()),
            "gate_stability_rate": float((reasons[mask] == "gate_stability").mean()),
            "max_gate_rate": float((reasons[mask] == "max_gate").mean()),
        }
        if c == benign_id:
            false_accept = selected[mask] >= 0
            row.update({
                "owned_expert": "benign_default",
                "owned_accept_rate": np.nan,
                "owned_selected_rate": np.nan,
                "wrong_expert_selected_rate": float(false_accept.mean()),
                "local_acc_given_owned_selected": np.nan,
                "selected_but_wrong_rate": float(false_accept.mean()),
                "mean_owned_margin": np.nan,
                "mean_owned_stability": np.nan,
                "mean_owned_local_conf": np.nan,
                "primary_bottleneck": "benign_false_accept" if false_accept.mean() > 0.001 else "ok",
            })
        else:
            owner = cls_to_expert.get(c, -1)
            owner_selected = selected[mask] == owner
            wrong_selected = (selected[mask] >= 0) & (selected[mask] != owner)
            selected_wrong = (selected[mask] >= 0) & (y_pred[mask] != y_te[mask])
            local_ok = y_pred[mask][owner_selected] == y_te[mask][owner_selected]
            owner_accept = accepted[mask, owner] if owner >= 0 else np.zeros(mask.sum(), dtype=bool)
            owner_margin = margins[mask, owner] if owner >= 0 else np.array([])
            owner_stability = stability[mask, owner] if owner >= 0 else np.array([])
            owner_conf = local_conf[mask, owner] if owner >= 0 else np.array([])
            local_acc = _mean_or_nan(local_ok)
            gate_gap = float(owner_selected.mean() - rm[c])
            if row["benign_gate_rate"] > 0.05:
                bottleneck = "binary_gate_miss"
            elif row["all_reject_rate"] > 0.05:
                bottleneck = "gate_reject"
            elif float(wrong_selected.mean()) > 0.01:
                bottleneck = "wrong_expert"
            elif not np.isnan(local_acc) and local_acc < 0.95:
                bottleneck = "local_classifier"
            elif gate_gap > 0.02:
                bottleneck = "selected_but_wrong"
            else:
                bottleneck = "ok"
            row.update({
                "owned_expert": "" if owner < 0 else experts[owner].name,
                "owned_accept_rate": float(owner_accept.mean()),
                "owned_selected_rate": float(owner_selected.mean()),
                "wrong_expert_selected_rate": float(wrong_selected.mean()),
                "local_acc_given_owned_selected": local_acc,
                "selected_but_wrong_rate": float(selected_wrong.mean()),
                "selected_minus_recall": gate_gap,
                "mean_owned_margin": _mean_or_nan(owner_margin),
                "mean_owned_stability": _mean_or_nan(owner_stability),
                "mean_owned_local_conf": _mean_or_nan(owner_conf),
                "primary_bottleneck": bottleneck,
            })
        rows.append(row)

    cols = [
        "class", "support", "owned_expert", "primary_bottleneck",
        "baseline_recall", "moe_recall", "baseline_f1", "moe_f1", "delta_f1",
        "benign_gate_rate", "all_reject_rate", "owned_accept_rate", "owned_selected_rate",
        "wrong_expert_selected_rate", "local_acc_given_owned_selected",
        "selected_but_wrong_rate", "selected_minus_recall",
        "multi_accept_rate", "meta_router_rate", "stability_tiebreak_rate",
        "mean_owned_margin", "mean_owned_stability", "mean_owned_local_conf",
    ]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["primary_bottleneck", "support"], ascending=[True, False])
    df.to_csv(os.path.join(out_dir, "bottleneck_by_class.csv"), index=False)
    png_cols = [c for c in [
        "class", "support", "primary_bottleneck", "baseline_f1", "moe_f1",
        "delta_f1", "benign_gate_rate", "all_reject_rate",
        "wrong_expert_selected_rate", "local_acc_given_owned_selected",
        "owned_selected_rate"] if c in df.columns]
    render_table_png(
        df[png_cols], os.path.join(out_dir, "bottleneck_by_class.png"),
        title="Per-class pipeline bottleneck (blue=good, red=bad)",
        delta_cols=("delta_f1",),
        high_good=("baseline_f1", "moe_f1", "local_acc_given_owned_selected",
                   "owned_selected_rate"),
        low_good=("benign_gate_rate", "all_reject_rate",
                  "wrong_expert_selected_rate"))

    summary_rows = []
    benign_mask = y_te == benign_id
    attack_mask = y_te != benign_id
    for e, expert in enumerate(experts):
        owned_mask = np.isin(y_te, expert.global_classes)
        owned_selected = selected[owned_mask] == e
        local_ok = y_pred[owned_mask][owned_selected] == y_te[owned_mask][owned_selected]
        non_owned_attack = attack_mask & ~np.isin(y_te, expert.global_classes)
        summary_rows.append({
            "expert": expert.name,
            "owned_classes": ",".join(class_names[c] for c in expert.global_classes),
            "threshold": float(expert.threshold),
            "owned_support": int(owned_mask.sum()),
            "owned_accept_rate": float(accepted[owned_mask, e].mean()) if owned_mask.any() else np.nan,
            "owned_selected_rate": float((selected[owned_mask] == e).mean()) if owned_mask.any() else np.nan,
            "local_acc_given_owned_selected": _mean_or_nan(local_ok),
            "owned_all_reject_rate": float((selected[owned_mask] < 0).mean()) if owned_mask.any() else np.nan,
            "benign_accept_rate": float(accepted[benign_mask, e].mean()) if benign_mask.any() else np.nan,
            "non_owned_attack_accept_rate": (
                float(accepted[non_owned_attack, e].mean()) if non_owned_attack.any() else np.nan),
            "mean_owned_margin": _mean_or_nan(margins[owned_mask, e]),
            "mean_benign_margin": _mean_or_nan(margins[benign_mask, e]),
            "mean_owned_stability": _mean_or_nan(stability[owned_mask, e]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "expert_summary.csv"), index=False)
    es_cols = [c for c in [
        "expert", "owned_support", "threshold", "owned_accept_rate",
        "owned_selected_rate", "local_acc_given_owned_selected",
        "benign_accept_rate", "non_owned_attack_accept_rate",
        "mean_owned_margin", "mean_benign_margin", "mean_owned_stability"]
        if c in summary_df.columns]
    render_table_png(
        summary_df[es_cols], os.path.join(out_dir, "expert_summary.png"),
        title="Per-expert health: classifier quality & OOD gate selectivity",
        high_good=("owned_accept_rate", "owned_selected_rate",
                   "local_acc_given_owned_selected"),
        low_good=("benign_accept_rate", "non_owned_attack_accept_rate"))
    log.info("Saved bottleneck_by_class.csv/png and expert_summary.csv/png")


def save_diagnostics(y_te, y_pred, diag, experts, class_names, benign_id, out_dir):
    rows = []
    for c, cname in enumerate(class_names):
        mask = y_te == c
        if not mask.any():
            continue
        row = {"true_class": cname, "support": int(mask.sum()),
               "benign_gate_rate": float((diag["reason"][mask] == "benign_binary").mean()),
               "all_reject_rate": float((diag["reason"][mask] == "all_reject").mean())}
        for e, expert in enumerate(experts):
            row[f"{expert.name}_accept_rate"] = float(diag["accepted"][mask, e].mean())
            row[f"{expert.name}_selected_rate"] = float((diag["selected"][mask] == e).mean())
            row[f"{expert.name}_margin"] = float(diag["margins"][mask, e].mean())
            row[f"{expert.name}_stability"] = float(diag["stability"][mask, e].mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "activation_matrix.csv"), index=False)

    sample_rows = []
    tail_ids = [i for i, n in enumerate(class_names)
                if i != benign_id and np.sum(y_te == i) <= 1000]
    tail_mask = np.isin(y_te, tail_ids)
    idx = np.where(tail_mask)[0][:5000]
    for i in idx:
        sample_rows.append({
            "true_class": class_names[int(y_te[i])],
            "pred_class": class_names[int(y_pred[i])],
            "selected_expert": "" if diag["selected"][i] < 0 else experts[int(diag["selected"][i])].name,
            "reason": diag["reason"][i],
            **{f"{ex.name}_accept": bool(diag["accepted"][i, e])
               for e, ex in enumerate(experts)},
            **{f"{ex.name}_margin": float(diag["margins"][i, e])
               for e, ex in enumerate(experts)},
            **{f"{ex.name}_stability": float(diag["stability"][i, e])
               for e, ex in enumerate(experts)},
        })
    pd.DataFrame(sample_rows).to_csv(os.path.join(out_dir, "per_sample_debug_tail.csv"), index=False)

    reason_counts = pd.Series(diag["reason"]).value_counts().rename_axis("reason").reset_index(name="count")
    reason_counts.to_csv(os.path.join(out_dir, "selection_reason.csv"), index=False)


def evaluate_sc_ood(experts, X_ood, args, out_dir, log, name="sc_ood"):
    if X_ood is None or len(X_ood) == 0:
        return
    scores = np.zeros((len(X_ood), len(experts)), dtype=np.float32)
    margins = np.zeros_like(scores)
    accepted = np.zeros_like(scores, dtype=bool)
    for e, expert in enumerate(experts):
        s, _ = expert_raw_scores(expert, X_ood, args.ood_gate, args.imood_alpha)
        scores[:, e] = s
        margins[:, e] = (s - expert.threshold) / expert.score_scale
        accepted[:, e] = s >= expert.threshold

    accept_counts = accepted.sum(axis=1)
    selected = np.full(len(X_ood), -1, dtype=int)
    has_accept = accept_counts > 0
    if has_accept.any():
        masked_margins = np.where(accepted, margins, -np.inf)
        selected[has_accept] = np.argmax(masked_margins[has_accept], axis=1)

    summary = {
        "dataset": name,
        "n_samples": int(len(X_ood)),
        "all_reject_rate": float((accept_counts == 0).mean()),
        "any_accept_rate": float(has_accept.mean()),
        "multi_accept_rate": float((accept_counts > 1).mean()),
        "mean_accept_count": float(accept_counts.mean()),
    }
    for e, expert in enumerate(experts):
        summary[f"{expert.name}_accept_rate"] = float(accepted[:, e].mean())
        summary[f"{expert.name}_selected_rate"] = float((selected == e).mean())
        summary[f"{expert.name}_mean_margin"] = float(margins[:, e].mean())
        summary[f"{expert.name}_p95_margin"] = float(np.quantile(margins[:, e], 0.95))
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "sc_ood_accept_summary.csv"), index=False)

    expert_rows = []
    for e, expert in enumerate(experts):
        expert_rows.append({
            "dataset": name,
            "expert": expert.name,
            "threshold": float(expert.threshold),
            "accept_rate": float(accepted[:, e].mean()),
            "selected_rate": float((selected == e).mean()),
            "mean_score": float(scores[:, e].mean()),
            "mean_margin": float(margins[:, e].mean()),
            "p50_margin": float(np.quantile(margins[:, e], 0.50)),
            "p90_margin": float(np.quantile(margins[:, e], 0.90)),
            "p95_margin": float(np.quantile(margins[:, e], 0.95)),
            "p99_margin": float(np.quantile(margins[:, e], 0.99)),
        })
    pd.DataFrame(expert_rows).to_csv(
        os.path.join(out_dir, "sc_ood_expert_accept.csv"), index=False)

    log.info(
        f"SC-OOD {name}: all_reject={summary['all_reject_rate']:.4f}, "
        f"any_accept={summary['any_accept_rate']:.4f}, "
        f"multi_accept={summary['multi_accept_rate']:.4f}")


def save_thresholds(experts, out_dir):
    rows = []
    for e, ex in enumerate(experts):
        for c, p in zip(ex.global_classes, ex.prior):
            rows.append({
                "expert": ex.name,
                "class_id": int(c),
                "threshold": float(ex.threshold),
                "score_scale": float(ex.score_scale),
                "local_prior": float(p),
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ood_thresholds.csv"), index=False)


def save_confusion_matrix(y_te, y_pred, class_names, out_dir, log, tag="moe"):
    """Save confusion matrix as CSV (counts) and PNG (row-normalized recall heatmap).

    Rows = true class (sorted by support desc), cols = predicted class.
    Only includes classes that appear in y_te or y_pred to keep the table compact.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    present = sorted(set(y_te.tolist()) | set(y_pred.tolist()))
    names = [class_names[i] for i in present]
    cm = sk_cm(y_te, y_pred, labels=present)

    # sort rows by support (descending)
    support = cm.sum(axis=1)
    order = np.argsort(-support)
    cm = cm[order][:, order]
    names = [names[i] for i in order]
    support = support[order]

    # --- CSV: raw counts ---
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    df_cm.insert(0, "true_class", names)
    df_cm.to_csv(os.path.join(out_dir, f"confusion_matrix_{tag}.csv"), index=False)

    # --- PNG: row-normalized (recall view) heatmap ---
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sum

    n = len(names)
    fig_w = max(10, n * 0.55)
    fig_h = max(8, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(
        [f"{nm} (n={s})" for nm, s in zip(names, support)], fontsize=7)
    ax.set_xlabel("Predicted class", fontsize=9)
    ax.set_ylabel("True class", fontsize=9)
    ax.set_title(f"Confusion matrix ({tag}, row-normalized recall view)", fontsize=10)

    # annotate cells with raw counts where nonzero
    for r in range(n):
        for c in range(n):
            cnt = int(cm[r, c])
            if cnt == 0:
                continue
            color = "white" if cm_norm[r, c] > 0.55 else "black"
            ax.text(c, r, str(cnt), ha="center", va="center",
                    fontsize=6, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confusion_matrix_{tag}.png / .csv")


def save_run_summary(y_te, y_base, y_moe, diag, experts, args, out_dir,
                     sc_ood_summary_path=None):
    rows = [{
        "metric": "baseline_macro_f1",
        "value": float(f1_score(y_te, y_base, average="macro", zero_division=0)),
    }, {
        "metric": "moe_macro_f1",
        "value": float(f1_score(y_te, y_moe, average="macro", zero_division=0)),
    }, {
        "metric": "baseline_weighted_f1",
        "value": float(f1_score(y_te, y_base, average="weighted", zero_division=0)),
    }, {
        "metric": "moe_weighted_f1",
        "value": float(f1_score(y_te, y_moe, average="weighted", zero_division=0)),
    }, {
        "metric": "test_binary_attack_rate",
        "value": float(diag["is_attack"].mean()),
    }, {
        "metric": "test_benign_by_binary_rate",
        "value": float((diag["reason"] == "benign_binary").mean()),
    }, {
        "metric": "test_all_reject_rate",
        "value": float((diag["reason"] == "all_reject").mean()),
    }, {
        "metric": "test_multi_accept_rate",
        "value": float((diag["accepted"].sum(axis=1) > 1).mean()),
    }, {
        "metric": "select_mode",
        "value": args.select_mode,
    }, {
        "metric": "threshold_mode",
        "value": args.threshold_mode,
    }, {
        "metric": "threshold_quantile",
        "value": float(args.threshold_quantile),
    }, {
        "metric": "external_ood_data",
        "value": "" if args.external_ood_data is None else args.external_ood_data,
    }, {
        "metric": "sc_ood_data",
        "value": "" if args.sc_ood_data is None else args.sc_ood_data,
    }]
    if sc_ood_summary_path and os.path.exists(sc_ood_summary_path):
        sc = pd.read_csv(sc_ood_summary_path).iloc[0].to_dict()
        for k in ["all_reject_rate", "any_accept_rate", "multi_accept_rate", "mean_accept_count"]:
            rows.append({"metric": f"sc_ood_{k}", "value": sc[k]})
    for ex in experts:
        rows.append({"metric": f"threshold_{ex.name}", "value": float(ex.threshold)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "run_summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--gate_n_estimators", type=int, default=200)
    parser.add_argument("--binary_n_estimators", type=int, default=300,
                        help="Trees for the expert0 binary benign/attack gate")
    parser.add_argument("--binary_threshold", type=float, default=0.1,
                        help="P(attack) threshold for expert0; below -> predict benign")
    parser.add_argument(
        "--partition",
        default="taxonomy_cic",
        choices=["taxonomy_cic", "taxonomy_cic_websplit"],
        help=("Attack expert partition. "
              "`taxonomy_cic_websplit` splits `WebBot_Rare` into "
              "`WebBot_Main` (bot/web attacks) and `Rare_Micro` "
              "(infiltration/heartbleed), with `sql-injection` kept in "
              "`WebBot_Main`."))
    parser.add_argument("--ood_gate", default="energy",
                        choices=["oe", "energy", "imood_oe", "imood_energy"])
    parser.add_argument("--threshold_mode", default="per_class_min",
                        choices=["fixed_quantile", "per_class_min",
                                 "expert_sweep", "hard_negative"])
    parser.add_argument("--threshold_quantile", type=float, default=0.05,
                        help="ID validation score lower quantile; accept if score >= threshold")
    parser.add_argument("--sweep_quantiles",
                        default="0.001,0.002,0.005,0.01,0.02,0.05",
                        help="Comma-separated owned-ID score quantiles for expert threshold sweep")
    parser.add_argument("--sweep_benign_penalty", type=float, default=3.0)
    parser.add_argument("--sweep_non_owned_penalty", type=float, default=1.0)
    parser.add_argument("--sweep_min_owned_accept", type=float, default=0.95)
    parser.add_argument("--sweep_min_owned_penalty", type=float, default=5.0)
    parser.add_argument("--imood_alpha", type=float, default=1.0)
    parser.add_argument("--oe_neg_ratio", type=float, default=2.0)
    parser.add_argument("--external_ood_data", default=None,
                        help="Optional pickle with X used as external negative samples for OE gates")
    parser.add_argument("--external_ood_ratio", type=float, default=1.0,
                        help="External OOD negatives per owned positive when --external_ood_data is set")
    parser.add_argument("--sc_ood_data", default=None,
                        help="Optional feature-aligned OOD pickle used only for accept-rate evaluation")
    parser.add_argument("--sc_ood_name", default="sc_ood")
    parser.add_argument("--sc_ood_max_samples", type=int, default=200_000)
    parser.add_argument("--select_mode", default="meta_router",
                        choices=["meta_router", "gate_score",
                                 "stability_tiebreak", "gate_stability"])
    parser.add_argument("--id_margin", type=float, default=0.25,
                        help="Normalized gate-score margin for stability tie-break candidates")
    parser.add_argument("--stability_lambda", type=float, default=0.2)
    parser.add_argument(
        "--fallback_margin_gap", type=float, default=0.0,
        help=("Require fallback expert normalized margin to exceed the best "
              "primary-expert margin by at least this amount before fallback "
              "classification is allowed."))
    parser.add_argument("--perturb", default="feature_group_mask",
                        choices=["feature_group_mask", "gaussian", "mask"])
    parser.add_argument("--n_views", type=int, default=5)
    parser.add_argument("--p_mask", type=float, default=0.3)
    parser.add_argument("--noise_std", type=float, default=0.05)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", ts)
    os.makedirs(out_dir, exist_ok=True)
    log = setup_logger(os.path.join(out_dir, "experiment.log"))
    log.info(f"Args: {vars(args)}")
    device = detect_device()
    log.info(f"Device: {device}")

    X, y, le, feature_names, dataset_type = load_data(args.data)
    class_names = list(le.classes_)
    n_cls = len(class_names)
    benign_id = find_benign_id(class_names)
    log.info(f"Loaded {len(y):,} samples | {X.shape[1]} features | dataset={dataset_type}")
    log.info(f"Benign class: {class_names[benign_id]} (id={benign_id})")

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, args.seed)
    log.info(f"Split: train={len(y_tr):,} val={len(y_va):,} test={len(y_te):,}")
    X_ood_tr = X_ood_va = None
    if args.external_ood_data:
        X_ood = load_external_ood(args.external_ood_data, X.shape[1])
        X_ood_tr, X_ood_va = split_external_ood(X_ood, args.seed)
        log.info(
            f"External OOD: {args.external_ood_data} | "
            f"train={len(X_ood_tr):,} val={len(X_ood_va):,} "
            f"ratio={args.external_ood_ratio:.3f}")
    X_sc_ood = None
    if args.sc_ood_data:
        X_sc_ood = load_external_ood(args.sc_ood_data, X.shape[1])
        X_sc_ood = maybe_subsample(X_sc_ood, args.sc_ood_max_samples, args.seed + 202)
        log.info(
            f"SC-OOD eval: {args.sc_ood_data} | "
            f"samples={len(X_sc_ood):,} | name={args.sc_ood_name}")
    col_means = X_tr.mean(axis=0).astype(np.float32)
    col_stds = (X_tr.std(axis=0) + 1e-8).astype(np.float32)
    feature_groups = build_feature_groups(feature_names)
    log.info(f"Perturbation feature groups: { {k: len(v) for k, v in feature_groups.items()} }")

    t0 = time.time()
    baseline = train_baseline(X_tr, y_tr, X_va, y_va, n_cls,
                              args.n_estimators, device, args.seed, log)
    y_base = np.asarray(baseline.predict(X_te), dtype=int)
    log.info(f"Baseline trained in {time.time() - t0:.1f}s")

    binary_gate = train_binary_attack_gate(
        X_tr, y_tr, X_va, y_va, benign_id, args.binary_n_estimators,
        device, args.seed + 7, log)

    partitions = build_attack_partitions(class_names, benign_id, args.partition, log)
    experts = train_experts(
        X_tr, y_tr, X_va, y_va, partitions, args.n_estimators,
        args.gate_n_estimators, device,
        args.seed, args.ood_gate, args.threshold_quantile, args.imood_alpha,
        args.oe_neg_ratio, log, args.threshold_mode, X_ood_tr, X_ood_va,
        args.external_ood_ratio if args.external_ood_data else 0.0,
        benign_id, args)
    if args.threshold_mode == "expert_sweep":
        log.info("Tuning expert-specific thresholds on validation split")
        tune_expert_thresholds(experts, X_va, y_va, benign_id, args, out_dir, log)

    meta_router = None
    if args.select_mode == "meta_router":
        log.info("Training stability meta-router on validation split")
        meta_router = train_meta_router(
            experts, X_va, y_va, binary_gate, benign_id, args,
            col_means, col_stds, feature_groups, log)
        if meta_router is None:
            log.warning("Meta-router unavailable; falling back to max-gate selection")

    y_oracle, _ = oracle_predict(experts, y_te, benign_id)
    log.info(f"Oracle attack-expert macro-F1: {f1_score(y_te, y_oracle, average='macro', zero_division=0):.4f}")

    log.info("Predicting OOD-gated TTA MoE")
    y_moe, diag = predict_ood_tta(experts, X_te, benign_id, binary_gate,
                                  meta_router, args, col_means, col_stds,
                                  feature_groups, log)
    log.info(f"Baseline macro-F1: {f1_score(y_te, y_base, average='macro', zero_division=0):.4f}")
    log.info(f"MoE macro-F1:      {f1_score(y_te, y_moe, average='macro', zero_division=0):.4f}")

    save_per_class_results(y_te, y_base, y_moe, class_names, out_dir, log)
    save_bottleneck_results(y_te, y_base, y_moe, diag, experts, class_names,
                            benign_id, out_dir, log)
    save_diagnostics(y_te, y_moe, diag, experts, class_names, benign_id, out_dir)
    save_confusion_matrix(y_te, y_moe, class_names, out_dir, log, tag="moe")
    save_confusion_matrix(y_te, y_base, class_names, out_dir, log, tag="baseline")
    save_gate_stability_dashboard(y_te, diag, experts, class_names, benign_id,
                                  out_dir, log)
    save_pipeline_funnel(y_te, y_moe, diag, experts, class_names, benign_id,
                         out_dir, log)
    save_precision_funnel(y_te, y_moe, benign_id, class_names, out_dir, log)
    save_thresholds(experts, out_dir)
    sc_ood_summary_path = None
    if X_sc_ood is not None:
        evaluate_sc_ood(experts, X_sc_ood, args, out_dir, log, args.sc_ood_name)
        sc_ood_summary_path = os.path.join(out_dir, "sc_ood_accept_summary.csv")
    save_run_summary(y_te, y_base, y_moe, diag, experts, args, out_dir,
                     sc_ood_summary_path)
    log.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()

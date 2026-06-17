#!/usr/bin/env python3
"""
OOD-aware stability-gated disjoint expert MoE.

Design:
  * Benign is not assigned to any expert.
  * Attack classes are partitioned into disjoint experts.
  * Each expert trains:
      1) a local XGBoost classifier over its owned attack classes
      2) an optional binary OE-style ID/OOD gate
  * Test-time decision:
      1) each expert independently accepts/rejects x
      2) if all experts reject, predict benign
      3) otherwise select one accepted expert
      4) TTA stability is used only after OOD gating

Example:
  python src/code_ood_tta_gate.py --data data/cic2017_proc.pkl \
    --ood_gate oe --select_mode stability_tiebreak --n_estimators 300
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
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

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
    feature_names = d.get("feature_names")
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(X.shape[1])]
    return X, y, d["label_encoder"], list(feature_names), d.get("dataset_type", "cic2017")


def split_data(X, y, seed):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


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


def train_oe_binary_gate(X_tr, y_tr, X_va, y_va, global_classes,
                         n_est, device, seed, neg_ratio):
    rng = np.random.default_rng(seed)
    pos_tr = np.where(np.isin(y_tr, global_classes))[0]
    neg_tr = np.where(~np.isin(y_tr, global_classes))[0]
    n_neg = min(len(neg_tr), max(len(pos_tr), int(len(pos_tr) * neg_ratio)))
    neg_tr = rng.choice(neg_tr, n_neg, replace=False)
    idx_tr = np.concatenate([pos_tr, neg_tr])
    rng.shuffle(idx_tr)
    yb_tr = np.isin(y_tr[idx_tr], global_classes).astype(int)

    pos_va = np.where(np.isin(y_va, global_classes))[0]
    neg_va = np.where(~np.isin(y_va, global_classes))[0]
    n_neg_va = min(len(neg_va), max(len(pos_va), int(len(pos_va) * neg_ratio)))
    neg_va = rng.choice(neg_va, n_neg_va, replace=False)
    idx_va = np.concatenate([pos_va, neg_va])
    rng.shuffle(idx_va)
    yb_va = np.isin(y_va[idx_va], global_classes).astype(int)

    model = xgb.XGBClassifier(**_xgb_params(2, n_est, device, seed, binary=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr[idx_tr], yb_tr, sample_weight=_balanced_w(yb_tr),
                  eval_set=[(X_va[idx_va], yb_va)], verbose=False)
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


def train_experts(X_tr, y_tr, X_va, y_va, partitions, n_est, gate_n_est, device,
                  seed, gate, q, alpha, neg_ratio, log):
    experts = []
    for e, (name, classes) in enumerate(partitions):
        log.info(f"Training expert {e}: {name} ({len(classes)} classes)")
        local, prior = train_local_classifier(
            X_tr, y_tr, X_va, y_va, classes, n_est, device, seed + e)
        binary = None
        if gate in ("oe", "imood_oe"):
            binary = train_oe_binary_gate(
                X_tr, y_tr, X_va, y_va, classes, gate_n_est, device,
                seed + 100 + e, neg_ratio)
        expert = Expert(name, classes, local, binary, 0.0, 1.0, prior)
        tau, scale = calibrate_threshold(expert, X_va, y_va, gate, q, alpha)
        expert.threshold = tau
        expert.score_scale = scale
        experts.append(expert)
        log.info(f"  threshold(q={q:.2f})={tau:.4f}, score_scale={scale:.4f}")
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


def predict_ood_tta(experts, X, benign_id, args, col_means, col_stds,
                    feature_groups, log):
    rng = np.random.default_rng(args.seed + 999)
    N, E = len(X), len(experts)
    scores = np.zeros((N, E), dtype=np.float32)
    margins = np.zeros((N, E), dtype=np.float32)
    accepted = np.zeros((N, E), dtype=bool)
    stability = np.zeros((N, E), dtype=np.float32)
    local_preds = np.zeros((N, E), dtype=int)
    local_conf = np.zeros((N, E), dtype=np.float32)

    for e, expert in enumerate(experts):
        s, p = expert_raw_scores(expert, X, args.ood_gate, args.imood_alpha)
        scores[:, e] = s
        margins[:, e] = (s - expert.threshold) / expert.score_scale
        accepted[:, e] = s >= expert.threshold
        local_preds[:, e] = p.argmax(axis=1)
        local_conf[:, e] = p.max(axis=1)
        if args.select_mode in ("stability_tiebreak", "gate_stability"):
            stability[:, e] = compute_stability(
                expert, X, p, args, rng, col_means, col_stds, feature_groups)

    y_pred = np.full(N, benign_id, dtype=int)
    selected = np.full(N, -1, dtype=int)
    reason = np.full(N, "all_reject", dtype=object)

    for i in range(N):
        active = np.where(accepted[i])[0]
        if len(active) == 0:
            continue
        if len(active) == 1:
            chosen = int(active[0])
            reason[i] = "only_accept"
        elif args.select_mode == "gate_score":
            chosen = int(active[np.argmax(margins[i, active])])
            reason[i] = "max_gate"
        elif args.select_mode == "stability_tiebreak":
            best = float(np.max(margins[i, active]))
            cand = active[margins[i, active] >= best - args.id_margin]
            chosen = int(cand[np.argmax(stability[i, cand])])
            reason[i] = "stability_tiebreak"
        elif args.select_mode == "gate_stability":
            combo = margins[i, active] + args.stability_lambda * stability[i, active]
            chosen = int(active[np.argmax(combo)])
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
    }
    log.info(f"All-reject rate: {(selected < 0).mean():.4f}")
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


def save_diagnostics(y_te, y_pred, diag, experts, class_names, benign_id, out_dir):
    rows = []
    for c, cname in enumerate(class_names):
        mask = y_te == c
        if not mask.any():
            continue
        row = {"true_class": cname, "support": int(mask.sum()),
               "all_reject_rate": float((diag["selected"][mask] < 0).mean())}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--gate_n_estimators", type=int, default=200)
    parser.add_argument("--partition", default="taxonomy_cic", choices=["taxonomy_cic"])
    parser.add_argument("--ood_gate", default="oe",
                        choices=["oe", "energy", "imood_oe", "imood_energy"])
    parser.add_argument("--threshold_quantile", type=float, default=0.05,
                        help="ID validation score lower quantile; accept if score >= threshold")
    parser.add_argument("--imood_alpha", type=float, default=1.0)
    parser.add_argument("--oe_neg_ratio", type=float, default=2.0)
    parser.add_argument("--select_mode", default="stability_tiebreak",
                        choices=["gate_score", "stability_tiebreak", "gate_stability"])
    parser.add_argument("--id_margin", type=float, default=0.25,
                        help="Normalized gate-score margin for stability tie-break candidates")
    parser.add_argument("--stability_lambda", type=float, default=0.2)
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
    col_means = X_tr.mean(axis=0).astype(np.float32)
    col_stds = (X_tr.std(axis=0) + 1e-8).astype(np.float32)
    feature_groups = build_feature_groups(feature_names)
    log.info(f"Perturbation feature groups: { {k: len(v) for k, v in feature_groups.items()} }")

    t0 = time.time()
    baseline = train_baseline(X_tr, y_tr, X_va, y_va, n_cls,
                              args.n_estimators, device, args.seed, log)
    y_base = np.asarray(baseline.predict(X_te), dtype=int)
    log.info(f"Baseline trained in {time.time() - t0:.1f}s")

    partitions = build_attack_partitions(class_names, benign_id, args.partition, log)
    experts = train_experts(
        X_tr, y_tr, X_va, y_va, partitions, args.n_estimators,
        args.gate_n_estimators, device,
        args.seed, args.ood_gate, args.threshold_quantile, args.imood_alpha,
        args.oe_neg_ratio, log)

    y_oracle, _ = oracle_predict(experts, y_te, benign_id)
    log.info(f"Oracle attack-expert macro-F1: {f1_score(y_te, y_oracle, average='macro', zero_division=0):.4f}")

    log.info("Predicting OOD-gated TTA MoE")
    y_moe, diag = predict_ood_tta(experts, X_te, benign_id, args,
                                  col_means, col_stds, feature_groups, log)
    log.info(f"Baseline macro-F1: {f1_score(y_te, y_base, average='macro', zero_division=0):.4f}")
    log.info(f"MoE macro-F1:      {f1_score(y_te, y_moe, average='macro', zero_division=0):.4f}")

    save_per_class_results(y_te, y_base, y_moe, class_names, out_dir, log)
    save_diagnostics(y_te, y_moe, diag, experts, class_names, benign_id, out_dir)
    save_thresholds(experts, out_dir)
    log.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()

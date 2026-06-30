#!/usr/bin/env python3
"""
Chronological v2 for CIC-IDS2017 OOD/TTA experiments.

Expected pickle:
  python scripts/preprocess_cic2017_chrono_v2.py \
    --data_dir data/cic2017 --output data/cic2017_chrono_v2.pkl

Run:
  python src/code_ood_v2.py --data data/cic2017_chrono_v2.pkl \
    --ood_gate energy --threshold_mode hard_negative

Evaluation label space:
  known/source classes from Mon-Thu are evaluated on a held-out known test set.
  Friday-only attack classes are evaluated as OOD rejection targets.

Routing:
  1. exp0 energy gate rejects rows unless binary attack confidence overrides it
  2. binary gate says benign -> benign
  3. binary gate says attack and a known expert accepts -> known class
  4. binary gate says attack and all known experts reject -> unknown routing
     outcome for OOD metrics.
"""
import argparse
import logging
import os
import pickle
import shutil
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.append(os.path.dirname(__file__))

from code_ood_v1 import (
    _predict_logits,
    _predict_proba,
    build_feature_groups,
    compute_expert_signals,
    detect_device,
    expert_raw_scores,
    find_benign_id,
    local_pred_to_global,
    logsumexp,
    parse_float_list,
    render_table_png,
    save_confusion_matrix,
    save_diagnostics,
    save_gate_stability_dashboard,
    save_pipeline_funnel,
    save_precision_funnel,
    save_run_summary,
    save_thresholds,
    setup_logger,
    train_baseline,
    train_binary_attack_gate,
    train_experts,
)


def make_unique_out_dir(root, prefix):
    base = os.path.join(root, prefix)
    candidate = base
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    os.makedirs(candidate)
    return candidate


def exp0_energy_scores(model, X, n_classes):
    logits = _predict_logits(model, X, n_classes)
    return logsumexp(logits, axis=1).astype(np.float32)


def calibrate_exp0_energy(model, X_val, y_val, n_classes, quantile):
    scores = exp0_energy_scores(model, X_val, n_classes)
    per_class_tau = []
    for cls in np.unique(y_val):
        cls_scores = scores[y_val == cls]
        if len(cls_scores):
            per_class_tau.append(float(np.quantile(cls_scores, quantile)))
    threshold = float(min(per_class_tau)) if per_class_tau else float(np.quantile(scores, quantile))
    scale = float(np.std(scores) + 1e-6)
    return threshold, scale


def load_chrono_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    required = [
        "X", "y", "feature_columns", "class_names",
        "train_indices", "val_indices",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required v2 pickle keys: {missing}")

    X = np.asarray(data["X"], dtype=np.float32)
    y_original = np.asarray(data["y"], dtype=int)
    original_names = list(data["class_names"])
    train_idx = np.asarray(data["train_indices"], dtype=int)
    val_idx = np.asarray(data["val_indices"], dtype=int)
    test_known_idx = np.asarray(data["test_known_indices"], dtype=int)
    test_ood_idx = np.asarray(data["test_ood_indices"], dtype=int)
    test_combined_idx = np.asarray(
        data.get("test_combined_indices", np.concatenate([test_known_idx, test_ood_idx])),
        dtype=int,
    )

    source_original_ids = sorted(np.unique(y_original[np.concatenate([train_idx, val_idx])]).tolist())
    benign_original_id = data.get("benign_original_id")
    if benign_original_id is None:
        benign_original_id = find_benign_id(original_names)
    benign_original_id = int(benign_original_id)
    if benign_original_id not in source_original_ids:
        raise ValueError("Benign/normal label must be present in source train/val split")

    original_to_source = {orig_id: i for i, orig_id in enumerate(source_original_ids)}
    source_to_original = np.array(source_original_ids, dtype=int)
    source_names = [original_names[i] for i in source_original_ids]
    benign_id = original_to_source[benign_original_id]

    def remap(labels, allow_unknown):
        out = np.empty(len(labels), dtype=int)
        for i, label in enumerate(labels):
            label = int(label)
            if label in original_to_source:
                out[i] = original_to_source[label]
            elif allow_unknown:
                out[i] = -1
            else:
                raise ValueError(
                    f"Source split contains label not in source mapping: "
                    f"{label} ({original_names[label]})"
                )
        return out

    y_train = remap(y_original[train_idx], allow_unknown=False)
    y_val = remap(y_original[val_idx], allow_unknown=False)
    y_test_known = remap(y_original[test_known_idx], allow_unknown=False)
    y_test_ood_source = remap(y_original[test_ood_idx], allow_unknown=True)
    y_test_combined_source = remap(y_original[test_combined_idx], allow_unknown=True)

    return {
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "X_test_known": X[test_known_idx],
        "X_test_ood": X[test_ood_idx],
        "X_test_combined": X[test_combined_idx],
        "y_train": y_train,
        "y_val": y_val,
        "y_test_known": y_test_known,
        "y_test_ood_source": y_test_ood_source,
        "y_test_ood_original": y_original[test_ood_idx],
        "y_test_combined_source": y_test_combined_source,
        "y_test_combined_original": y_original[test_combined_idx],
        "source_class_names": source_names,
        "original_class_names": original_names,
        "source_original_ids": source_original_ids,
        "original_to_source": original_to_source,
        "source_to_original": source_to_original,
        "benign_id": benign_id,
        "benign_original_id": benign_original_id,
        "feature_names": list(data["feature_columns"]),
        "metadata": data,
    }


def build_chrono_partitions(class_names, benign_id, log):
    dos_like = []
    credential_web_like = []
    rare_fallback = []
    for i, name in enumerate(class_names):
        if i == benign_id or name == "unknown-attack":
            continue
        norm = str(name).lower()
        if norm in {"heartbleed", "infiltration", "web-attack-sql-injection"}:
            rare_fallback.append(i)
        elif norm.startswith("dos-"):
            dos_like.append(i)
        elif "patator" in norm or norm.startswith("web-attack"):
            credential_web_like.append(i)

    partitions = []
    if dos_like:
        partitions.append(("DoS_Like_Known", dos_like))
    if credential_web_like:
        partitions.append(("Credential_Web_Like_Known", credential_web_like))
    if rare_fallback:
        partitions.append(("Rare_Fallback_Known", rare_fallback))

    covered = set(dos_like + credential_web_like + rare_fallback)
    leftovers = [
        i for i, name in enumerate(class_names)
        if i != benign_id and name != "unknown-attack" and i not in covered
    ]
    if leftovers:
        partitions.append(("Other_Source_Known", leftovers))

    log.info("Chronological v2 known-attack experts:")
    for name, classes in partitions:
        log.info(f"  {name}: {', '.join(class_names[c] for c in classes)}")
    return partitions


def map_source_predictions_to_original(y_pred_source, source_to_original):
    return source_to_original[np.asarray(y_pred_source, dtype=int)]


def split_primary_and_fallback_experts_v2(experts):
    if len(experts) >= 2 and "fallback" in str(experts[-1].name).lower():
        return np.arange(len(experts) - 1, dtype=int), len(experts) - 1
    return np.arange(len(experts), dtype=int), None


def choose_by_tta_confidence(sig, j, cols, args):
    combo = (
        sig["local_conf"][j, cols]
        + args.stability_lambda * sig["stability"][j, cols]
    )
    return int(cols[np.argmax(combo)])


def predict_ood_tta_v2(experts, X, benign_id, binary_gate, args,
                       col_means, col_stds, feature_groups, log):
    """v2 router: OOD gates only define candidates; TTA+confidence selects expert."""
    n_samples = len(X)
    n_experts = len(experts)
    primary_experts, fallback_expert = split_primary_and_fallback_experts_v2(experts)
    proba_attack = _predict_proba(binary_gate, X)[:, 1]
    is_attack = proba_attack >= args.binary_threshold

    scores = np.zeros((n_samples, n_experts), dtype=np.float32)
    margins = np.zeros((n_samples, n_experts), dtype=np.float32)
    accepted = np.zeros((n_samples, n_experts), dtype=bool)
    stability = np.zeros((n_samples, n_experts), dtype=np.float32)
    local_preds = np.zeros((n_samples, n_experts), dtype=int)
    local_conf = np.zeros((n_samples, n_experts), dtype=np.float32)

    y_pred = np.full(n_samples, benign_id, dtype=int)
    selected = np.full(n_samples, -1, dtype=int)
    reason = np.full(n_samples, "benign_binary", dtype=object)

    attack_idx = np.where(is_attack)[0]
    log.info(
        f"Binary gate: attack-predicted {is_attack.mean():.4f} "
        f"({len(attack_idx):,}/{n_samples:,})")

    if len(attack_idx) > 0:
        sig = compute_expert_signals(
            experts, X[attack_idx], args, col_means, col_stds, feature_groups,
            compute_stab=True)
        scores[attack_idx] = sig["scores"]
        margins[attack_idx] = sig["margins"]
        accepted[attack_idx] = sig["accepted"]
        stability[attack_idx] = sig["stability"]
        local_preds[attack_idx] = sig["local_preds"]
        local_conf[attack_idx] = sig["local_conf"]

        for j, i in enumerate(attack_idx):
            active_primary = primary_experts[sig["accepted"][j, primary_experts]]
            if len(active_primary) > 0:
                chosen = choose_by_tta_confidence(sig, j, active_primary, args)
                reason[i] = (
                    "only_accept"
                    if len(active_primary) == 1
                    else "tta_confidence"
                )
                selected[i] = chosen
                y_pred[i] = local_pred_to_global(experts[chosen], local_preds[j, chosen])
                continue

            best_primary_margin = (
                float(np.max(sig["margins"][j, primary_experts]))
                if len(primary_experts) > 0 else float("-inf")
            )
            fallback_margin = (
                sig["margins"][j, fallback_expert]
                if fallback_expert is not None else float("-inf")
            )
            fallback_ok = (
                fallback_expert is not None
                and sig["accepted"][j, fallback_expert]
                and fallback_margin >= best_primary_margin + args.fallback_margin_gap
            )
            if fallback_ok:
                chosen = int(fallback_expert)
                reason[i] = "fallback_accept"
                selected[i] = chosen
                y_pred[i] = local_pred_to_global(experts[chosen], local_preds[j, chosen])
            else:
                reason[i] = "all_reject"

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
    log.info("v2 expert selection: OOD gates filter candidates; TTA stability + confidence selects among candidates")
    return y_pred, diag


def predict_with_exp0_ood(exp0_model, exp0_threshold, exp0_scale, n_source_cls,
                          experts, X, benign_id, binary_gate, args,
                          col_means, col_stds, feature_groups, log):
    n_samples = len(X)
    n_experts = len(experts)
    exp0_scores = exp0_energy_scores(exp0_model, X, n_source_cls)
    exp0_margin = (exp0_scores - exp0_threshold) / exp0_scale
    exp0_in = exp0_scores >= exp0_threshold
    exp0_binary_proba_attack = _predict_proba(binary_gate, X)[:, 1].astype(np.float32)
    exp0_attack_override = exp0_binary_proba_attack >= args.exp0_attack_override_threshold
    route_downstream = exp0_in | exp0_attack_override

    y_pred = np.full(n_samples, benign_id, dtype=int)
    diag = {
        "scores": np.zeros((n_samples, n_experts), dtype=np.float32),
        "margins": np.zeros((n_samples, n_experts), dtype=np.float32),
        "accepted": np.zeros((n_samples, n_experts), dtype=bool),
        "stability": np.zeros((n_samples, n_experts), dtype=np.float32),
        "selected": np.full(n_samples, -1, dtype=int),
        "reason": np.full(n_samples, "exp0_ood", dtype=object),
        "local_conf": np.zeros((n_samples, n_experts), dtype=np.float32),
        "is_attack": np.zeros(n_samples, dtype=bool),
        "proba_attack": np.zeros(n_samples, dtype=np.float32),
        "exp0_energy_score": exp0_scores,
        "exp0_energy_margin": exp0_margin,
        "exp0_in": exp0_in,
        "exp0_attack_override": exp0_attack_override,
        "exp0_route_downstream": route_downstream,
        "exp0_binary_proba_attack": exp0_binary_proba_attack,
    }

    in_idx = np.where(route_downstream)[0]
    log.info(
        f"Exp0 energy gate: in-distribution {exp0_in.mean():.4f}; "
        f"binary override {exp0_attack_override.mean():.4f}; "
        f"downstream {route_downstream.mean():.4f} ({len(in_idx):,}/{n_samples:,}) | "
        f"energy_threshold={exp0_threshold:.4f}, "
        f"override_attack_threshold={args.exp0_attack_override_threshold:.4f}")
    if len(in_idx) == 0:
        return y_pred, diag

    y_sub, sub_diag = predict_ood_tta_v2(
        experts, X[in_idx], benign_id, binary_gate, args,
        col_means, col_stds, feature_groups, log)
    y_pred[in_idx] = y_sub
    for key in ["scores", "margins", "accepted", "stability", "local_conf"]:
        diag[key][in_idx] = sub_diag[key]
    for key in ["selected", "reason", "is_attack", "proba_attack"]:
        diag[key][in_idx] = sub_diag[key]
    return y_pred, diag


def diag_for_known_funnel(diag):
    out = dict(diag)
    reason = np.asarray(diag["reason"], dtype=object).copy()
    reason[reason == "exp0_ood"] = "all_reject"
    out["reason"] = reason
    return out


def retune_expert_thresholds_owned_benign_nonowned(experts, X_va, y_va,
                                                   benign_id, args, out_dir, log):
    """Tune expert OOD thresholds using owned, benign, and non-owned validation rows.

    The threshold is still calibrated without Friday leakage. Owned classes define
    the ID floor, benign rows penalize normal false accepts, and non-owned attack
    rows penalize cross-expert absorption.
    """
    quantiles = sorted(set(parse_float_list(args.sweep_quantiles)))
    rows = []
    selected_rows = []
    for expert in experts:
        scores, proba = expert_raw_scores(expert, X_va, args.ood_gate, args.imood_alpha)
        owned_mask = np.isin(y_va, expert.global_classes)
        benign_mask = y_va == benign_id
        non_owned_attack_mask = (y_va != benign_id) & ~owned_mask
        owned_scores = scores[owned_mask]
        if len(owned_scores) == 0:
            log.warning(f"  retune skipped {expert.name}: no owned validation rows")
            continue

        class_score_map = {
            int(cls): scores[y_va == cls]
            for cls in expert.global_classes
            if np.any(y_va == cls)
        }
        candidate_taus = set()
        for q in quantiles:
            candidate_taus.add(float(np.quantile(owned_scores, q)))
            for cls_scores in class_score_map.values():
                candidate_taus.add(float(np.quantile(cls_scores, q)))
        candidate_taus.add(float(expert.threshold))
        candidate_taus = sorted(candidate_taus)

        local_preds = local_pred_to_global(expert, proba.argmax(axis=1))
        best = None
        for tau in candidate_taus:
            accept = scores >= tau
            per_class_accept = []
            floor_shortfall = 0.0
            for cls, cls_scores in class_score_map.items():
                cls_accept = float((cls_scores >= tau).mean())
                per_class_accept.append(cls_accept)
                if cls_accept < args.sweep_min_owned_accept:
                    floor_shortfall += args.sweep_min_owned_accept - cls_accept
            mean_owned_accept = float(np.mean(per_class_accept)) if per_class_accept else 0.0
            min_owned_accept = float(np.min(per_class_accept)) if per_class_accept else 0.0
            owned_accept = float(accept[owned_mask].mean())
            benign_accept = float(accept[benign_mask].mean()) if benign_mask.any() else 0.0
            non_owned_accept = (
                float(accept[non_owned_attack_mask].mean())
                if non_owned_attack_mask.any() else 0.0
            )
            owned_local_acc = (
                float((local_preds[owned_mask & accept] == y_va[owned_mask & accept]).mean())
                if (owned_mask & accept).any() else np.nan
            )
            objective = (
                mean_owned_accept
                - args.sweep_benign_penalty * benign_accept
                - args.sweep_non_owned_penalty * non_owned_accept
                - args.sweep_min_owned_penalty * floor_shortfall
            )
            row = {
                "expert": expert.name,
                "threshold": float(tau),
                "objective": float(objective),
                "owned_accept_rate": owned_accept,
                "mean_owned_class_accept_rate": mean_owned_accept,
                "min_owned_class_accept_rate": min_owned_accept,
                "benign_accept_rate": benign_accept,
                "non_owned_attack_accept_rate": non_owned_accept,
                "owned_local_acc_given_accept": owned_local_acc,
                "floor_shortfall": float(floor_shortfall),
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
            f"  retuned {expert.name}: threshold {old_threshold:.4f}->{expert.threshold:.4f}, "
            f"owned={best['owned_accept_rate']:.4f}, "
            f"owned_class_mean={best['mean_owned_class_accept_rate']:.4f}, "
            f"owned_class_min={best['min_owned_class_accept_rate']:.4f}, "
            f"benign={best['benign_accept_rate']:.4f}, "
            f"non_owned={best['non_owned_attack_accept_rate']:.4f}")

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "expert_threshold_owned_benign_nonowned_sweep.csv"),
        index=False)
    pd.DataFrame(selected_rows).to_csv(
        os.path.join(out_dir, "expert_threshold_owned_benign_nonowned_selected.csv"),
        index=False)


def save_known_t_per_class(y_true, y_base, y_moe, class_names, out_dir):
    labels = np.arange(len(class_names))
    pb, rb, fb, sup = precision_recall_fscore_support(
        y_true, y_base, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_moe, labels=labels, zero_division=0)
    rows = []
    order = [i for i in np.argsort(-sup) if sup[i] > 0]
    for i in order:
        rows.append({
            "class": class_names[i],
            "support": int(sup[i]),
            "prec_baseline": float(pb[i]),
            "prec_moe": float(pm[i]),
            "recall_baseline": float(rb[i]),
            "recall_moe": float(rm[i]),
            "f1_baseline": float(fb[i]),
            "f1_moe": float(fm[i]),
            "delta_f1": float(fm[i] - fb[i]),
        })
    for average in ["macro", "weighted"]:
        rows.append({
            "class": f"{average} avg",
            "support": int(sup.sum()),
            "prec_baseline": float(precision_recall_fscore_support(
                y_true, y_base, labels=labels, average=average, zero_division=0)[0]),
            "prec_moe": float(precision_recall_fscore_support(
                y_true, y_moe, labels=labels, average=average, zero_division=0)[0]),
            "recall_baseline": float(precision_recall_fscore_support(
                y_true, y_base, labels=labels, average=average, zero_division=0)[1]),
            "recall_moe": float(precision_recall_fscore_support(
                y_true, y_moe, labels=labels, average=average, zero_division=0)[1]),
            "f1_baseline": float(f1_score(
                y_true, y_base, labels=labels, average=average, zero_division=0)),
            "f1_moe": float(f1_score(
                y_true, y_moe, labels=labels, average=average, zero_division=0)),
            "delta_f1": float(
                f1_score(y_true, y_moe, labels=labels, average=average, zero_division=0)
                - f1_score(y_true, y_base, labels=labels, average=average, zero_division=0)
            ),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "1a_known_t_per_class.csv"), index=False)
    render_table_png(
        df,
        os.path.join(out_dir, "1a_known_t_per_class.png"),
        title="Known Mon-Thu held-out classification",
        delta_cols=("delta_f1",),
        high_good=("prec_baseline", "prec_moe", "recall_baseline", "recall_moe", "f1_baseline", "f1_moe"),
    )


def save_ood_per_class_funnel(y_original, diag, original_names, source_original_ids,
                              benign_original_id, out_dir, log):
    source_set = set(int(x) for x in source_original_ids)
    counts = np.bincount(y_original, minlength=len(original_names))
    order = [c for c in np.argsort(-counts) if counts[c] > 0]
    rows, labels = [], []
    csv_rows = []
    for c in order:
        mask = y_original == c
        n = int(mask.sum())
        binary_benign = float((diag["reason"][mask] == "benign_binary").mean())
        exp0_ood = float((diag["reason"][mask] == "exp0_ood").mean())
        all_reject = float((diag["reason"][mask] == "all_reject").mean())
        known_accept = float((diag["selected"][mask] >= 0).mean())
        multi_accept = float((diag["accepted"][mask].sum(axis=1) > 1).mean())
        rows.append([exp0_ood, all_reject, known_accept, binary_benign])
        labels.append(f"{original_names[c]} ({n})")
        csv_rows.append({
            "class": original_names[c],
            "support": n,
            "seen_in_trainval": bool(c in source_set),
            "is_unseen_attack": bool(c not in source_set and c != benign_original_id),
            "exp0_ood_rate": exp0_ood,
            "ood_detect_all_reject_rate": all_reject,
            "ood_detect_total_rate": exp0_ood + all_reject,
            "known_expert_accept_rate": known_accept,
            "binary_benign_stop_rate": binary_benign,
            "multi_accept_rate": multi_accept,
        })
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, "2b_ood_per_class_funnel.csv"), index=False)

    rows = np.array(rows, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(11, max(3, len(labels) * 0.55)))
    left = np.zeros(len(labels), dtype=np.float32)
    yidx = np.arange(len(labels))
    seg_names = ["exp0 OOD", "expert all-reject", "known expert accept", "binary benign stop"]
    seg_colors = ["#2e7d32", "#81c784", "#f44336", "#212121"]
    for s in range(rows.shape[1]):
        ax.barh(yidx, rows[:, s], left=left, color=seg_colors[s],
                label=seg_names[s], edgecolor="white")
        left += rows[:, s]
    ax.set_yticks(yidx)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of Friday samples")
    ax.set_title("Friday OOD pipeline funnel")
    ax.legend(ncol=3, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2b_ood_per_class_funnel.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 2b_ood_per_class_funnel.csv/png")


def _annot_no_owner_heatmap(ax, matrix, row_labels, col_labels, title,
                            cmap, vmin=None, vmax=None):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    return im


def save_ood_gate_stability_dashboard(y_original, diag, experts,
                                      original_names, out_dir, log):
    counts = np.bincount(y_original, minlength=len(original_names))
    order = [c for c in np.argsort(-counts) if counts[c] > 0]
    expert_names = [expert.name for expert in experts]
    e_count = len(experts)
    accept = np.zeros((len(order), e_count), dtype=np.float32)
    selected = np.zeros((len(order), e_count), dtype=np.float32)
    stability = np.zeros((len(order), e_count), dtype=np.float32)
    margin = np.zeros((len(order), e_count), dtype=np.float32)
    rows = []
    for r, c in enumerate(order):
        mask = y_original == c
        for e, expert in enumerate(experts):
            accept[r, e] = float(diag["accepted"][mask, e].mean())
            selected[r, e] = float((diag["selected"][mask] == e).mean())
            stability[r, e] = float(diag["stability"][mask, e].mean())
            margin[r, e] = float(diag["margins"][mask, e].mean())
            rows.append({
                "class": original_names[c],
                "support": int(mask.sum()),
                "expert": expert.name,
                "accept_rate": accept[r, e],
                "selected_rate": selected[r, e],
                "mean_stability": stability[r, e],
                "mean_margin": margin[r, e],
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "2e_ood_gate_tta_dashboard.csv"), index=False)

    row_labels = [f"{original_names[c]} ({int(counts[c])})" for c in order]
    fig, axes = plt.subplots(2, 2, figsize=(max(10, e_count * 1.9 + 5),
                                            max(7, len(order) * 0.65)))
    panels = [
        ("OOD accept rate", accept, "Blues", 0.0, 1.0),
        ("Selected rate", selected, "Blues", 0.0, 1.0),
        ("Mean TTA stability", stability, "viridis", None, None),
        ("Mean gate margin", margin, "coolwarm", None, None),
    ]
    for ax, (title, matrix, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = _annot_no_owner_heatmap(
            ax, matrix, row_labels, expert_names, title, cmap, vmin, vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Friday OOD gate & TTA stability by original class", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(out_dir, "2e_ood_gate_tta_dashboard.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 2e_ood_gate_tta_dashboard.csv/png")


def save_ood_summary(y_original, diag, original_names, source_original_ids,
                     benign_original_id, out_dir):
    source_set = set(int(x) for x in source_original_ids)
    unseen_attack = np.array([
        (int(y) not in source_set) and int(y) != benign_original_id
        for y in y_original
    ], dtype=bool)
    target_benign = y_original == benign_original_id
    exp0_ood = diag["reason"] == "exp0_ood"
    all_reject = diag["reason"] == "all_reject"
    ood_detected = exp0_ood | (all_reject & diag["is_attack"])
    known_false_accept = unseen_attack & (diag["selected"] >= 0)

    rows = [{
        "metric": "unseen_attack_support",
        "value": int(unseen_attack.sum()),
    }, {
        "metric": "target_benign_support",
        "value": int(target_benign.sum()),
    }, {
        "metric": "ood_tpr_unseen_attack_all_reject",
        "value": float(all_reject[unseen_attack].mean()) if unseen_attack.any() else np.nan,
    }, {
        "metric": "ood_tpr_unseen_attack_exp0_ood",
        "value": float(exp0_ood[unseen_attack].mean()) if unseen_attack.any() else np.nan,
    }, {
        "metric": "ood_tpr_unseen_attack_total",
        "value": float(ood_detected[unseen_attack].mean()) if unseen_attack.any() else np.nan,
    }, {
        "metric": "ood_fpr_target_benign_all_reject",
        "value": float(all_reject[target_benign].mean()) if target_benign.any() else np.nan,
    }, {
        "metric": "ood_fpr_target_benign_exp0_ood",
        "value": float(exp0_ood[target_benign].mean()) if target_benign.any() else np.nan,
    }, {
        "metric": "ood_fpr_target_benign_total",
        "value": float(ood_detected[target_benign].mean()) if target_benign.any() else np.nan,
    }, {
        "metric": "known_expert_false_accept_unseen_attack",
        "value": float(known_false_accept[unseen_attack].mean()) if unseen_attack.any() else np.nan,
    }, {
        "metric": "binary_attack_rate_unseen_attack",
        "value": float(diag["is_attack"][unseen_attack].mean()) if unseen_attack.any() else np.nan,
    }, {
        "metric": "binary_attack_rate_target_benign",
        "value": float(diag["is_attack"][target_benign].mean()) if target_benign.any() else np.nan,
    }]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "2a_ood_summary.csv"), index=False)
    render_table_png(
        df,
        os.path.join(out_dir, "2a_ood_summary.png"),
        title="Friday OOD summary",
    )

def save_baseline_ood_absorption(y_ood_original, y_base_original,
                                 original_names, out_dir):
    rows = []
    for pred_id in sorted(np.unique(y_base_original).tolist()):
        mask = y_base_original == pred_id
        rows.append({
            "baseline_pred_class": original_names[int(pred_id)],
            "count": int(mask.sum()),
            "rate": float(mask.mean()),
        })
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df.to_csv(os.path.join(out_dir, "2c_ood_baseline_absorption.csv"), index=False)
    render_table_png(
        df,
        os.path.join(out_dir, "2c_ood_baseline_absorption.png"),
        title="Closed-set baseline absorption of Friday OOD",
        high_good=(),
        low_good=("rate",),
    )


def save_baseline_ood_absorption_by_class(y_ood_original, y_base_original,
                                          original_names, out_dir):
    true_ids = [int(i) for i in np.unique(y_ood_original)]
    pred_ids = [int(i) for i in np.unique(y_base_original)]
    rows = []
    matrix = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float32)
    count_matrix = np.zeros((len(true_ids), len(pred_ids)), dtype=int)
    for r, true_id in enumerate(true_ids):
        mask = y_ood_original == true_id
        support = int(mask.sum())
        for c, pred_id in enumerate(pred_ids):
            count = int((y_base_original[mask] == pred_id).sum())
            rate = float(count / support) if support else 0.0
            matrix[r, c] = rate
            count_matrix[r, c] = count
            rows.append({
                "true_ood_class": original_names[true_id],
                "baseline_pred_class": original_names[pred_id],
                "support": support,
                "count": count,
                "rate": rate,
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "2d_ood_baseline_absorption_by_class.csv"),
        index=False)

    row_labels = [f"{original_names[i]} ({int((y_ood_original == i).sum())})" for i in true_ids]
    col_labels = [original_names[i] for i in pred_ids]
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.1),
                                    max(3, len(row_labels) * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Closed-set baseline absorption by Friday OOD class", fontsize=11)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if count_matrix[r, c] > 0:
                ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2d_ood_baseline_absorption_by_class.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_source_val_per_class(y_true_source, y_base_source, y_moe_source,
                              source_class_names, out_dir):
    df_path = os.path.join(out_dir, "1v_known_v_per_class.csv")
    labels = np.arange(len(source_class_names))
    pb, rb, fb, sup = precision_recall_fscore_support(
        y_true_source, y_base_source, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true_source, y_moe_source, labels=labels, zero_division=0)
    rows = []
    order = [i for i in np.argsort(-sup) if sup[i] > 0]
    for i in order:
        rows.append({
            "class": source_class_names[i],
            "support": int(sup[i]),
            "prec_baseline": float(pb[i]),
            "prec_moe": float(pm[i]),
            "recall_baseline": float(rb[i]),
            "recall_moe": float(rm[i]),
            "f1_baseline": float(fb[i]),
            "f1_moe": float(fm[i]),
            "delta_f1": float(fm[i] - fb[i]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(df_path, index=False)
    render_table_png(
        df,
        os.path.join(out_dir, "1v_known_v_per_class.png"),
        title="Source Mon-Thu validation classification",
        delta_cols=("delta_f1",),
        high_good=("prec_baseline", "prec_moe", "recall_baseline", "recall_moe", "f1_baseline", "f1_moe"),
    )


def save_v2_metadata(bundle, args, out_dir):
    meta = bundle["metadata"]
    rows = []
    for key in ["scenario_name", "val_size", "seed"]:
        if key in meta:
            rows.append({"key": key, "value": meta[key]})
    rows.extend([
        {"key": "source_class_names", "value": "|".join(bundle["source_class_names"])},
        {"key": "original_class_names", "value": "|".join(bundle["original_class_names"])},
        {"key": "source_original_ids", "value": "|".join(map(str, bundle["source_original_ids"]))},
        {"key": "args", "value": repr(vars(args))},
    ])
    if "group_names" in meta and "group_roles" in meta:
        for name, role in zip(meta["group_names"], meta["group_roles"]):
            rows.append({"key": f"group:{role}", "value": name})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "v2_metadata.csv"), index=False)


def rename_if_exists(out_dir, old_name, new_name):
    old_path = os.path.join(out_dir, old_name)
    if os.path.exists(old_path):
        shutil.move(old_path, os.path.join(out_dir, new_name))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cic2017_chrono_v2.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--gate_n_estimators", type=int, default=200)
    parser.add_argument("--binary_n_estimators", type=int, default=300)
    parser.add_argument("--binary_threshold", type=float, default=0.1)
    parser.add_argument("--ood_gate", default="energy",
                        choices=["oe", "energy", "imood_oe", "imood_energy"])
    parser.add_argument("--threshold_mode", default="hard_negative",
                        choices=["fixed_quantile", "per_class_min",
                                 "expert_sweep", "hard_negative"])
    parser.add_argument("--threshold_quantile", type=float, default=0.05)
    parser.add_argument("--sweep_quantiles",
                        default="0.001,0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--sweep_benign_penalty", type=float, default=3.0)
    parser.add_argument("--sweep_non_owned_penalty", type=float, default=1.0)
    parser.add_argument("--sweep_min_owned_accept", type=float, default=0.95)
    parser.add_argument("--sweep_min_owned_penalty", type=float, default=5.0)
    parser.add_argument("--imood_alpha", type=float, default=1.0)
    parser.add_argument("--oe_neg_ratio", type=float, default=2.0)
    parser.add_argument("--id_margin", type=float, default=0.25)
    parser.add_argument("--stability_lambda", type=float, default=0.2)
    parser.add_argument("--fallback_margin_gap", type=float, default=0.0)
    parser.add_argument("--perturb", default="feature_group_mask",
                        choices=["feature_group_mask", "gaussian", "mask"])
    parser.add_argument("--n_views", type=int, default=5)
    parser.add_argument("--p_mask", type=float, default=0.3)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--disable_exp0_ood", action="store_true",
                        help="Disable the front exp0 energy OOD gate")
    parser.add_argument("--exp0_ood_quantile", type=float, default=0.05,
                        help="Per-class lower quantile for exp0 energy knownness threshold")
    parser.add_argument("--exp0_attack_override_threshold", type=float, default=0.5,
                        help="Route exp0-OOD rows downstream if binary P(attack) is at least this value")
    parser.add_argument("--skip_expert_threshold_retune", action="store_true",
                        help="Skip v2 owned/benign/non-owned expert OOD threshold retuning")
    parser.add_argument("--skip_source_val_eval", action="store_true",
                        help="Skip MoE prediction on Mon-Thu validation rows")
    return parser.parse_args()


def main():
    args = parse_args()
    args.external_ood_data = None
    args.sc_ood_data = None
    args.select_mode = "tta_confidence_v2"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = make_unique_out_dir("results", f"{ts}_pid{os.getpid()}_chrono_v2")
    log = setup_logger(os.path.join(out_dir, "experiment.log"))
    log.info(f"Args: {vars(args)}")
    logging.getLogger("ood_tta_gate").info("Running chronological original-label v2")

    device = detect_device()
    bundle = load_chrono_data(args.data)
    X_tr = bundle["X_train"]
    X_va = bundle["X_val"]
    X_test_known = bundle["X_test_known"]
    X_test_ood = bundle["X_test_ood"]
    y_tr = bundle["y_train"]
    y_va = bundle["y_val"]
    y_test_known = bundle["y_test_known"]
    y_test_ood_original = bundle["y_test_ood_original"]
    source_class_names = bundle["source_class_names"]
    original_class_names = bundle["original_class_names"]
    benign_id = bundle["benign_id"]
    benign_original_id = bundle["benign_original_id"]
    source_to_original = bundle["source_to_original"]
    n_source_cls = len(source_class_names)
    n_original_cls = len(original_class_names)

    log.info(f"Device: {device}")
    log.info(
        f"Loaded v2 data: train={len(y_tr):,} val={len(y_va):,} "
        f"test_known={len(y_test_known):,} test_ood={len(y_test_ood_original):,}")
    log.info(f"Source-train labels: {source_class_names}")
    log.info(f"Original eval labels: {original_class_names}")
    log.info(f"Benign source id={benign_id}, benign original id={benign_original_id}")

    col_means = X_tr.mean(axis=0).astype(np.float32)
    col_stds = (X_tr.std(axis=0) + 1e-8).astype(np.float32)
    feature_groups = build_feature_groups(bundle["feature_names"])
    log.info(f"Perturbation feature groups: { {k: len(v) for k, v in feature_groups.items()} }")

    t0 = time.time()
    baseline = train_baseline(X_tr, y_tr, X_va, y_va, n_source_cls,
                              args.n_estimators, device, args.seed, log)
    log.info(f"Baseline trained in {time.time() - t0:.1f}s")

    exp0_threshold = exp0_scale = None
    if not args.disable_exp0_ood:
        exp0_threshold, exp0_scale = calibrate_exp0_energy(
            baseline, X_va, y_va, n_source_cls, args.exp0_ood_quantile)
        log.info(
            f"Exp0 energy OOD threshold calibrated: "
            f"q={args.exp0_ood_quantile:.4f}, threshold={exp0_threshold:.4f}, "
            f"scale={exp0_scale:.4f}")

    binary_gate = train_binary_attack_gate(
        X_tr, y_tr, X_va, y_va, benign_id, args.binary_n_estimators,
        device, args.seed + 7, log)

    partitions = build_chrono_partitions(source_class_names, benign_id, log)
    experts = train_experts(
        X_tr, y_tr, X_va, y_va, partitions, args.n_estimators,
        args.gate_n_estimators, device, args.seed, args.ood_gate,
        args.threshold_quantile, args.imood_alpha, args.oe_neg_ratio,
        log, args.threshold_mode, None, None, 0.0, benign_id, args)

    if not args.skip_expert_threshold_retune:
        log.info("Retuning expert OOD thresholds on owned/benign/non-owned validation split")
        retune_expert_thresholds_owned_benign_nonowned(
            experts, X_va, y_va, benign_id, args, out_dir, log)

    log.info(
        "Using v2 router: expert OOD gates define candidates; "
        "TTA stability + local confidence select among accepted primary experts")

    if not args.skip_source_val_eval:
        log.info("Evaluating source Mon-Thu validation classification")
        y_base_val = np.asarray(baseline.predict(X_va), dtype=int)
        if args.disable_exp0_ood:
            y_moe_val, _ = predict_ood_tta_v2(
                experts, X_va, benign_id, binary_gate, args,
                col_means, col_stds, feature_groups, log)
        else:
            y_moe_val, _ = predict_with_exp0_ood(
                baseline, exp0_threshold, exp0_scale, n_source_cls,
                experts, X_va, benign_id, binary_gate, args,
                col_means, col_stds, feature_groups, log)
        save_source_val_per_class(
            y_va, y_base_val, y_moe_val, source_class_names, out_dir)

    log.info("Evaluating known Mon-Thu held-out classification")
    y_base_known = np.asarray(baseline.predict(X_test_known), dtype=int)
    if args.disable_exp0_ood:
        y_moe_known, diag_known = predict_ood_tta_v2(
            experts, X_test_known, benign_id, binary_gate, args,
            col_means, col_stds, feature_groups, log)
    else:
        y_moe_known, diag_known = predict_with_exp0_ood(
            baseline, exp0_threshold, exp0_scale, n_source_cls,
            experts, X_test_known, benign_id, binary_gate, args,
            col_means, col_stds, feature_groups, log)

    log.info("Evaluating Friday OOD rejection")
    y_base_ood_source = np.asarray(baseline.predict(X_test_ood), dtype=int)
    y_base_ood = map_source_predictions_to_original(y_base_ood_source, source_to_original)
    if args.disable_exp0_ood:
        y_moe_ood_raw, diag_ood = predict_ood_tta_v2(
            experts, X_test_ood, benign_id, binary_gate, args,
            col_means, col_stds, feature_groups, log)
    else:
        y_moe_ood_raw, diag_ood = predict_with_exp0_ood(
            baseline, exp0_threshold, exp0_scale, n_source_cls,
            experts, X_test_ood, benign_id, binary_gate, args,
            col_means, col_stds, feature_groups, log)

    source_labels = np.arange(n_source_cls)
    log.info(
        "Baseline known macro-F1: "
        f"{f1_score(y_test_known, y_base_known, labels=source_labels, average='macro', zero_division=0):.4f}")
    log.info(
        "MoE known macro-F1:      "
        f"{f1_score(y_test_known, y_moe_known, labels=source_labels, average='macro', zero_division=0):.4f}")
    log.info(
        "Friday OOD detect rate: "
        f"{float(np.isin(diag_ood['reason'], ['exp0_ood', 'all_reject']).mean()):.4f} "
        f"(exp0={float((diag_ood['reason'] == 'exp0_ood').mean()):.4f}, "
        f"expert_all_reject={float((diag_ood['reason'] == 'all_reject').mean()):.4f})")

    save_known_t_per_class(y_test_known, y_base_known, y_moe_known, source_class_names, out_dir)
    save_diagnostics(y_test_known, y_moe_known, diag_known, experts, source_class_names,
                     benign_id, out_dir)
    save_confusion_matrix(y_test_known, y_moe_known, source_class_names, out_dir, log, tag="moe_known")
    rename_if_exists(out_dir, "confusion_matrix_moe_known.csv", "1b_known_confusion_matrix_moe.csv")
    rename_if_exists(out_dir, "confusion_matrix_moe_known.png", "1b_known_confusion_matrix_moe.png")
    save_confusion_matrix(y_test_known, y_base_known, source_class_names, out_dir, log, tag="baseline_known")
    rename_if_exists(out_dir, "confusion_matrix_baseline_known.csv", "1b_known_confusion_matrix_baseline.csv")
    rename_if_exists(out_dir, "confusion_matrix_baseline_known.png", "1b_known_confusion_matrix_baseline.png")
    diag_known_funnel = diag_for_known_funnel(diag_known)
    save_pipeline_funnel(y_test_known, y_moe_known, diag_known_funnel, experts,
                         source_class_names, benign_id, out_dir, log)
    rename_if_exists(out_dir, "pipeline_funnel.png", "1c_known_recall_funnel.png")
    save_precision_funnel(y_test_known, y_moe_known, benign_id,
                          source_class_names, out_dir, log)
    rename_if_exists(out_dir, "precision_funnel.png", "1d_known_precision_funnel.png")
    save_gate_stability_dashboard(y_test_known, diag_known, experts,
                                  source_class_names, benign_id, out_dir, log)
    rename_if_exists(out_dir, "gate_stability_dashboard.png", "1e_known_gate_tta_dashboard.png")
    save_thresholds(experts, out_dir)
    save_run_summary(y_test_known, y_base_known, y_moe_known, diag_known, experts, args, out_dir, None)
    save_ood_summary(
        y_test_ood_original, diag_ood, original_class_names, bundle["source_original_ids"],
        benign_original_id, out_dir)
    save_ood_per_class_funnel(
        y_test_ood_original, diag_ood, original_class_names,
        bundle["source_original_ids"], benign_original_id, out_dir, log)
    save_ood_gate_stability_dashboard(
        y_test_ood_original, diag_ood, experts, original_class_names, out_dir, log)
    save_baseline_ood_absorption(
        y_test_ood_original, y_base_ood, original_class_names, out_dir)
    save_baseline_ood_absorption_by_class(
        y_test_ood_original, y_base_ood, original_class_names, out_dir)
    save_v2_metadata(bundle, args, out_dir)
    log.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()

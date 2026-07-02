#!/usr/bin/env python3
"""
CIC-IDS2017 MLP energy-based OOD sanity check.

This is the CIC/tabular counterpart of code_ood_resnet_cifar_svhn.py:
  1. train a single MLP closed-set classifier on source ID classes only,
  2. compute post-hoc energy from logits,
  3. evaluate held-out source known rows as ID and Friday unseen attacks as OOD.

Expected pickle:
  python scripts/preprocess_cic2017_chrono_v2.py \
    --data_dir data/cic2017 --output data/cic2017_chrono_v2.pkl

Run:
  python src/code_ood_mlp_cic.py \
    --data data/cic2017_chrono_v2.pkl \
    --epochs 80
"""
import argparse
import os
import pickle
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(__file__))

from code_ood_v1 import find_benign_id, render_table_png, setup_logger


def make_unique_out_dir(root, prefix):
    base = os.path.join(root, prefix)
    candidate = base
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    os.makedirs(candidate)
    return candidate


def detect_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_hidden_dims(value):
    dims = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            dims.append(int(part))
    if not dims:
        raise ValueError("--mlp_hidden must contain at least one dimension")
    return dims


class TabularMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IndexedTabularDataset(Dataset):
    def __init__(self, X, y_source, indices, mean, scale):
        self.X = X
        self.y_source = y_source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row_idx = int(self.indices[item])
        x = (self.X[row_idx].astype(np.float32, copy=False) - self.mean) / self.scale
        return x.astype(np.float32, copy=False), int(self.y_source[row_idx])


class ExternalOodDataset(Dataset):
    def __init__(self, X, indices, mean, scale):
        self.X = X
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row_idx = int(self.indices[item])
        x = (self.X[row_idx].astype(np.float32, copy=False) - self.mean) / self.scale
        return x.astype(np.float32, copy=False)


def make_loader(dataset, batch_size, shuffle, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )


def fit_scaler_in_chunks(X, indices, chunk_size, log):
    scaler = StandardScaler()
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(indices), chunk_size):
        batch_idx = indices[start:start + chunk_size]
        scaler.partial_fit(X[batch_idx].astype(np.float32, copy=False))
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    scale[scale < 1e-8] = 1.0
    log.info(f"StandardScaler fitted on {len(indices):,} train rows")
    return mean, scale


def remap_source_labels(y_original, train_idx, val_idx, original_names):
    source_original_ids = sorted(
        np.unique(y_original[np.concatenate([train_idx, val_idx])]).tolist())
    benign_original_id = find_benign_id(original_names)
    if benign_original_id not in source_original_ids:
        raise ValueError("Benign/normal label must be present in source train/val")
    original_to_source = {int(orig): i for i, orig in enumerate(source_original_ids)}
    source_to_original = np.asarray(source_original_ids, dtype=np.int64)
    y_source = np.full(len(y_original), -1, dtype=np.int64)
    for original_id, source_id in original_to_source.items():
        y_source[y_original == original_id] = source_id
    source_names = [original_names[i] for i in source_original_ids]
    benign_id = original_to_source[int(benign_original_id)]
    return y_source, source_names, source_to_original, int(benign_original_id), benign_id


def load_chrono_data(path, log):
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = [
        "X", "y", "class_names", "feature_columns",
        "train_indices", "val_indices",
        "test_known_indices", "test_ood_indices",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing required chrono pickle keys: {missing}")

    X = np.asarray(data["X"], dtype=np.float32)
    y_original = np.asarray(data["y"], dtype=np.int64)
    original_names = list(data["class_names"])
    train_idx = np.asarray(data["train_indices"], dtype=np.int64)
    val_idx = np.asarray(data["val_indices"], dtype=np.int64)
    test_known_idx = np.asarray(data["test_known_indices"], dtype=np.int64)
    test_ood_idx = np.asarray(data["test_ood_indices"], dtype=np.int64)

    y_source, source_names, source_to_original, benign_original_id, benign_id = (
        remap_source_labels(y_original, train_idx, val_idx, original_names))
    if np.any(y_source[train_idx] < 0) or np.any(y_source[val_idx] < 0):
        raise ValueError("Train/val contain labels outside source mapping")
    if np.any(y_source[test_known_idx] < 0):
        raise ValueError("Known test contains labels outside source mapping")

    log.info(
        f"Loaded CIC chrono data: train={len(train_idx):,} val={len(val_idx):,} "
        f"test_known={len(test_known_idx):,} friday={len(test_ood_idx):,}")
    log.info(f"Source classes: {source_names}")
    log.info(f"Original classes: {original_names}")
    return {
        "X": X,
        "y_original": y_original,
        "y_source": y_source,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_known_idx": test_known_idx,
        "test_ood_idx": test_ood_idx,
        "source_names": source_names,
        "original_names": original_names,
        "source_to_original": source_to_original,
        "benign_original_id": benign_original_id,
        "benign_id": benign_id,
        "feature_names": list(data["feature_columns"]),
        "metadata": data,
    }


def load_external_ood(path, expected_dim, max_samples, seed, log):
    if path is None or str(path).strip() == "":
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    X_ood = np.asarray(data["X"], dtype=np.float32)
    if X_ood.ndim != 2:
        raise ValueError(f"External OOD X must be 2D, got shape={X_ood.shape}")
    if X_ood.shape[1] != expected_dim:
        raise ValueError(
            f"External OOD feature dim mismatch: got {X_ood.shape[1]}, "
            f"expected {expected_dim}")
    indices = np.arange(len(X_ood), dtype=np.int64)
    if max_samples and max_samples > 0 and max_samples < len(indices):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, int(max_samples), replace=False))
    log.info(
        f"Loaded auxiliary OOD: {path} | rows={len(indices):,}/"
        f"{len(X_ood):,}, dim={X_ood.shape[1]}")
    return X_ood, indices


def maybe_subset_indices(indices, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, int(max_samples), replace=False))


def class_weights(y, n_classes, mode):
    if mode == "none":
        return None
    counts = np.maximum(np.bincount(y, minlength=n_classes), 1).astype(np.float32)
    weights = len(y) / (n_classes * counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights.astype(np.float32))


def negative_energy(logits, temperature):
    return -float(temperature) * torch.logsumexp(
        logits / float(temperature), dim=1)


def next_ood_batch(ood_loader, ood_iter):
    try:
        return next(ood_iter), ood_iter
    except StopIteration:
        ood_iter = iter(ood_loader)
        return next(ood_iter), ood_iter


def train_mlp(model, train_loader, val_loader, args, device, log, class_weight,
              ood_loader=None):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    weight = class_weight.to(device) if class_weight is not None else None

    best_state = None
    best_val_macro_f1 = -1.0
    best_epoch = 0
    stale = 0
    use_energy_margin = ood_loader is not None and args.energy_margin_weight > 0
    ood_iter = iter(ood_loader) if use_energy_margin else None
    if use_energy_margin:
        log.info(
            "Stage2 energy margin enabled: "
            f"aux_ood={args.external_ood_data}, "
            f"weight={args.energy_margin_weight}, "
            f"m_in={args.energy_margin_in}, m_out={args.energy_margin_out}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_margin_loss = 0.0
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce_loss = F.cross_entropy(logits, y, weight=weight)
            margin_loss = torch.zeros((), device=device)
            if use_energy_margin:
                x_ood, ood_iter = next_ood_batch(ood_loader, ood_iter)
                x_ood = x_ood.to(device, non_blocking=True)
                ood_logits = model(x_ood)
                energy_in = negative_energy(logits, args.temperature)
                energy_out = negative_energy(ood_logits, args.temperature)
                margin_loss = (
                    torch.relu(energy_in - args.energy_margin_in).pow(2).mean()
                    + torch.relu(args.energy_margin_out - energy_out).pow(2).mean()
                )
            loss = ce_loss + args.energy_margin_weight * margin_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            total_ce_loss += float(ce_loss.detach().cpu()) * len(y)
            total_margin_loss += float(margin_loss.detach().cpu()) * len(y)
            total += len(y)
            correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        scheduler.step()

        val_loss, val_acc, val_macro_f1 = evaluate_loss_acc_f1(
            model, val_loader, device, weight)
        train_acc = correct / max(total, 1)
        log.info(
            f"epoch={epoch:03d} train_loss={total_loss / max(total, 1):.4f} "
            f"ce={total_ce_loss / max(total, 1):.4f} "
            f"margin={total_margin_loss / max(total, 1):.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f} time={time.time() - t0:.1f}s")
        if val_macro_f1 > best_val_macro_f1 + 1e-6:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            log.info(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info(f"Best validation macro-F1={best_val_macro_f1:.4f} at epoch={best_epoch}")
    return model


def evaluate_loss_acc_f1(model, loader, device, weight):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weight, reduction="sum")
            pred = logits.argmax(dim=1)
            total_loss += float(loss.detach().cpu())
            total += len(y)
            correct += int((pred == y).sum().detach().cpu())
            preds.append(pred.detach().cpu().numpy().astype(np.int64))
            labels.append(y.detach().cpu().numpy().astype(np.int64))
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(labels, axis=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return total_loss / max(total, 1), correct / max(total, 1), macro_f1


def collect_scores_preds_labels(model, loader, device, temperature, need_preds):
    model.eval()
    scores_list = []
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            scores = float(temperature) * torch.logsumexp(
                logits / float(temperature), dim=1)
            scores_list.append(scores.detach().cpu().numpy().astype(np.float32))
            if need_preds:
                preds_list.append(
                    logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            labels_list.append(y.numpy().astype(np.int64))
            del x, y, logits, scores
    scores = np.concatenate(scores_list, axis=0)
    preds = np.concatenate(preds_list, axis=0) if need_preds else None
    labels = np.concatenate(labels_list, axis=0)
    return scores, preds, labels


def calibrate_energy_threshold(val_scores, y_val, quantile, strategy):
    if strategy == "global":
        threshold = float(np.quantile(val_scores, quantile))
    elif strategy == "per_class_min":
        per_class_tau = []
        for cls in np.unique(y_val):
            cls_scores = val_scores[y_val == cls]
            if len(cls_scores):
                per_class_tau.append(float(np.quantile(cls_scores, quantile)))
        threshold = float(min(per_class_tau)) if per_class_tau else float(
            np.quantile(val_scores, quantile))
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")
    scale = float(np.std(val_scores) + 1e-6)
    return threshold, scale


def ood_metrics(id_scores, unseen_scores, friday_benign_scores, threshold):
    y_true = np.concatenate([
        np.zeros(len(id_scores), dtype=np.int64),
        np.ones(len(unseen_scores), dtype=np.int64),
    ])
    ood_score = np.concatenate([-id_scores, -unseen_scores])
    auroc = float(roc_auc_score(y_true, ood_score))
    aupr = float(average_precision_score(y_true, ood_score))
    fpr, tpr, _ = roc_curve(y_true, ood_score)
    idx95 = int(np.argmin(np.abs(tpr - 0.95)))
    return {
        "auroc_unseen_attack": auroc,
        "aupr_unseen_attack": aupr,
        "fpr95_unseen_attack": float(fpr[idx95]),
        "id_retain_at_tau": float((id_scores >= threshold).mean()),
        "unseen_attack_detect_at_tau": float((unseen_scores < threshold).mean()),
        "unseen_attack_in_at_tau": float((unseen_scores >= threshold).mean()),
        "friday_benign_false_ood_at_tau": (
            float((friday_benign_scores < threshold).mean())
            if len(friday_benign_scores) else np.nan
        ),
        "friday_benign_retain_at_tau": (
            float((friday_benign_scores >= threshold).mean())
            if len(friday_benign_scores) else np.nan
        ),
    }


def save_classification(y_true, y_pred, class_names, out_dir):
    labels = np.arange(len(class_names))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    order = [int(i) for i in np.argsort(-sup) if sup[i] > 0]
    rows = []
    for cls in order:
        rows.append({
            "class": class_names[cls],
            "support": int(sup[cls]),
            "precision": float(prec[cls]),
            "recall": float(rec[cls]),
            "f1": float(f1[cls]),
        })
    for average in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0)
        rows.append({
            "class": f"{average} avg",
            "support": int(sup.sum()),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "id_classification.csv"), index=False)
    render_table_png(
        df, os.path.join(out_dir, "id_classification.png"),
        title="CIC source known MLP classification")

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8),
                                    max(8, len(class_names) * 0.65)))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("CIC source known MLP confusion matrix")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            if cm[r, c] > 0:
                ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center",
                        fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "id_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def save_ood_summary(metrics, threshold, scale, args, out_dir):
    rows = [
        {"metric": "energy_threshold", "value": float(threshold)},
        {"metric": "energy_scale", "value": float(scale)},
        {"metric": "temperature", "value": float(args.temperature)},
        {"metric": "threshold_strategy", "value": args.threshold_strategy},
        {"metric": "class_weight", "value": args.class_weight},
        {"metric": "external_ood_data", "value": args.external_ood_data or ""},
        {"metric": "energy_margin_weight", "value": float(args.energy_margin_weight)},
        {"metric": "energy_margin_in", "value": float(args.energy_margin_in)},
        {"metric": "energy_margin_out", "value": float(args.energy_margin_out)},
    ]
    rows.extend({"metric": key, "value": value} for key, value in metrics.items())
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ood_summary.csv"), index=False)
    render_table_png(
        df, os.path.join(out_dir, "ood_summary.png"),
        title="CIC Friday unseen attack energy OOD summary")


def save_energy_score_dump(val_scores, id_scores, friday_scores, threshold, out_dir):
    pd.DataFrame({
        "split": np.concatenate([
            np.repeat("source_val_id", len(val_scores)),
            np.repeat("source_test_known_id", len(id_scores)),
            np.repeat("friday_mixed", len(friday_scores)),
        ]),
        "energy": np.concatenate([val_scores, id_scores, friday_scores]),
    }).to_csv(os.path.join(out_dir, "energy_scores.csv"), index=False)


def save_energy_histogram(val_scores, id_scores, friday_scores,
                          friday_unseen_scores, friday_benign_scores,
                          threshold, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(val_scores, bins=120, alpha=0.35, density=True, label="source_val_id")
    ax.hist(id_scores, bins=120, alpha=0.35, density=True, label="source_test_known_id")
    ax.hist(friday_scores, bins=120, alpha=0.20, density=True, label="friday_mixed")
    if len(friday_benign_scores):
        ax.hist(friday_benign_scores, bins=120, alpha=0.30, density=True,
                label="friday_benign")
    if len(friday_unseen_scores):
        ax.hist(friday_unseen_scores, bins=120, alpha=0.30, density=True,
                label="friday_unseen_attack")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"tau={threshold:.4f}")
    ax.set_title("CIC MLP energy score distributions")
    ax.set_xlabel("energy = T logsumexp(logits / T); ID expected high")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_histogram.png"), dpi=150)
    plt.close(fig)


def save_ood_per_class(friday_original, friday_scores, threshold,
                       original_names, benign_original_id, source_original_ids,
                       out_dir):
    counts = np.bincount(friday_original, minlength=len(original_names))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]
    source_set = set(int(x) for x in source_original_ids)
    rows = []
    for cls in order:
        mask = friday_original == cls
        scores = friday_scores[mask]
        rows.append({
            "class": original_names[cls],
            "support": int(mask.sum()),
            "seen_in_source": bool(cls in source_set),
            "is_friday_unseen_attack": bool(cls not in source_set and cls != benign_original_id),
            "mean_energy": float(np.mean(scores)),
            "std_energy": float(np.std(scores)),
            "median_energy": float(np.median(scores)),
            "detect_ood_rate": float((scores < threshold).mean()),
            "in_distribution_rate": float((scores >= threshold).mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ood_per_class_energy.csv"), index=False)
    render_table_png(
        df, os.path.join(out_dir, "ood_per_class_energy.png"),
        title="CIC Friday per-class energy OOD")


def save_baseline_absorption(friday_original, friday_pred_source,
                             original_names, source_names, source_to_original,
                             benign_original_id, source_original_ids, out_dir):
    source_set = set(int(x) for x in source_original_ids)
    unseen_mask = np.array([
        int(y) not in source_set and int(y) != benign_original_id
        for y in friday_original
    ], dtype=bool)
    rows = []
    for true_id in sorted(np.unique(friday_original[unseen_mask]).tolist()):
        true_mask = unseen_mask & (friday_original == true_id)
        support = int(true_mask.sum())
        for pred_source in sorted(np.unique(friday_pred_source[true_mask]).tolist()):
            pred_original = int(source_to_original[int(pred_source)])
            count = int((friday_pred_source[true_mask] == pred_source).sum())
            rows.append({
                "true_unseen_class": original_names[int(true_id)],
                "baseline_pred_source_class": source_names[int(pred_source)],
                "baseline_pred_original_class": original_names[pred_original],
                "support": support,
                "count": count,
                "rate": float(count / max(support, 1)),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["true_unseen_class", "count"], ascending=[True, False])
    df.to_csv(os.path.join(out_dir, "baseline_absorption_unseen.csv"), index=False)


def save_threshold_tradeoff(id_scores, unseen_scores, friday_benign_scores, out_dir):
    rows = []
    for q in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90]:
        tau = float(np.quantile(id_scores, q))
        rows.append({
            "id_quantile_tau": q,
            "threshold": tau,
            "id_retain": float((id_scores >= tau).mean()),
            "unseen_attack_detect": float((unseen_scores < tau).mean()),
            "unseen_attack_in": float((unseen_scores >= tau).mean()),
            "friday_benign_false_ood": (
                float((friday_benign_scores < tau).mean())
                if len(friday_benign_scores) else np.nan
            ),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "threshold_tradeoff.csv"), index=False)


def save_metadata(bundle, args, out_dir):
    meta = bundle["metadata"]
    rows = [
        {"key": "args", "value": repr(vars(args))},
        {"key": "source_class_names", "value": "|".join(bundle["source_names"])},
        {"key": "original_class_names", "value": "|".join(bundle["original_names"])},
        {"key": "source_original_ids", "value": "|".join(
            map(str, bundle["source_to_original"].tolist()))},
    ]
    for key in ["scenario_name", "val_size", "test_known_size", "seed"]:
        if key in meta:
            rows.append({"key": key, "value": meta[key]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metadata.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cic2017_chrono_v2.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_strategy", default="per_class_min",
                        choices=["per_class_min", "global"])
    parser.add_argument("--ood_quantile", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10,
                        help="0 disables early stopping")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--test_batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_hidden", default="256,128,64")
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--class_weight", default="sqrt_balanced",
                        choices=["none", "balanced", "sqrt_balanced"])
    parser.add_argument("--external_ood_data",
                        default="data/cic2018_ood_for_cic2017.pkl",
                        help="Auxiliary OOD pickle for stage2 energy-margin training; empty disables it")
    parser.add_argument("--max_external_ood", type=int, default=500000)
    parser.add_argument("--energy_margin_weight", type=float, default=0.1,
                        help="0 disables stage2 energy-margin loss")
    parser.add_argument("--energy_margin_in", type=float, default=-8.0,
                        help="Original energy E=-logsumexp margin for ID; lower is more ID-like")
    parser.add_argument("--energy_margin_out", type=float, default=-4.0,
                        help="Original energy E=-logsumexp margin for auxiliary OOD")
    parser.add_argument("--scaler_chunk_size", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_val", type=int, default=0)
    parser.add_argument("--max_test_known", type=int, default=0)
    parser.add_argument("--max_friday", type=int, default=0)
    parser.add_argument("--skip_score_dump", action="store_true")
    parser.add_argument("--out_root", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = make_unique_out_dir(
        args.out_root, f"{ts}_pid{os.getpid()}_cic2017_mlp_energy")
    log = setup_logger(os.path.join(out_dir, "experiment.log"))
    log.info(f"Args: {vars(args)}")

    device = detect_device()
    log.info(f"Device: {device}")
    bundle = load_chrono_data(args.data, log)
    X = bundle["X"]
    y_source = bundle["y_source"]
    y_original = bundle["y_original"]
    train_idx = maybe_subset_indices(bundle["train_idx"], args.max_train, args.seed)
    val_idx = maybe_subset_indices(bundle["val_idx"], args.max_val, args.seed + 1)
    test_known_idx = maybe_subset_indices(
        bundle["test_known_idx"], args.max_test_known, args.seed + 2)
    friday_idx = maybe_subset_indices(
        bundle["test_ood_idx"], args.max_friday, args.seed + 3)
    source_names = bundle["source_names"]
    original_names = bundle["original_names"]
    n_classes = len(source_names)

    log.info(
        f"Using rows after optional subsampling: train={len(train_idx):,} "
        f"val={len(val_idx):,} test_known={len(test_known_idx):,} "
        f"friday={len(friday_idx):,}")
    mean, scale = fit_scaler_in_chunks(
        X, train_idx, args.scaler_chunk_size, log)
    external_ood = load_external_ood(
        args.external_ood_data, X.shape[1], args.max_external_ood,
        args.seed + 11, log)

    train_set = IndexedTabularDataset(X, y_source, train_idx, mean, scale)
    val_set = IndexedTabularDataset(X, y_source, val_idx, mean, scale)
    test_known_set = IndexedTabularDataset(X, y_source, test_known_idx, mean, scale)
    friday_y_for_loader = y_source.copy()
    friday_y_for_loader[friday_y_for_loader < 0] = 0
    friday_set = IndexedTabularDataset(X, friday_y_for_loader, friday_idx, mean, scale)
    ood_loader = None
    if external_ood is not None and args.energy_margin_weight > 0:
        X_external_ood, external_ood_idx = external_ood
        external_ood_set = ExternalOodDataset(
            X_external_ood, external_ood_idx, mean, scale)
        ood_loader = make_loader(
            external_ood_set, args.batch_size, True, args.workers)

    train_loader = make_loader(train_set, args.batch_size, True, args.workers)
    val_loader = make_loader(val_set, args.test_batch_size, False, args.workers)
    test_known_loader = make_loader(
        test_known_set, args.test_batch_size, False, args.workers)
    friday_loader = make_loader(friday_set, args.test_batch_size, False, args.workers)

    weights = class_weights(y_source[train_idx], n_classes, args.class_weight)
    if weights is not None:
        log.info(f"Class weights ({args.class_weight}): {weights.numpy().round(4).tolist()}")

    model = TabularMLP(
        input_dim=X.shape[1],
        num_classes=n_classes,
        hidden_dims=parse_hidden_dims(args.mlp_hidden),
        dropout=args.mlp_dropout,
    )
    t0 = time.time()
    model = train_mlp(
        model, train_loader, val_loader, args, device, log, weights, ood_loader)
    log.info(f"Training completed in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Collecting streaming energy scores for CIC ID/Friday splits")
    val_scores, _, y_val = collect_scores_preds_labels(
        model, val_loader, device, args.temperature, need_preds=False)
    id_scores, y_pred, y_id = collect_scores_preds_labels(
        model, test_known_loader, device, args.temperature, need_preds=True)
    friday_scores, friday_pred_source, _ = collect_scores_preds_labels(
        model, friday_loader, device, args.temperature, need_preds=True)

    threshold, scale_value = calibrate_energy_threshold(
        val_scores, y_val, args.ood_quantile, args.threshold_strategy)
    friday_original = y_original[friday_idx]
    source_original_ids = bundle["source_to_original"]
    source_set = set(int(x) for x in source_original_ids)
    unseen_attack_mask = np.array([
        int(y) not in source_set and int(y) != bundle["benign_original_id"]
        for y in friday_original
    ], dtype=bool)
    friday_benign_mask = friday_original == bundle["benign_original_id"]
    unseen_scores = friday_scores[unseen_attack_mask]
    friday_benign_scores = friday_scores[friday_benign_mask]

    metrics = ood_metrics(id_scores, unseen_scores, friday_benign_scores, threshold)
    metrics["id_macro_f1"] = float(f1_score(
        y_id, y_pred, average="macro", zero_division=0))
    metrics["id_weighted_f1"] = float(f1_score(
        y_id, y_pred, average="weighted", zero_division=0))
    metrics["id_accuracy"] = float((y_pred == y_id).mean())
    metrics["source_val_support"] = int(len(val_scores))
    metrics["source_test_known_support"] = int(len(id_scores))
    metrics["friday_support"] = int(len(friday_scores))
    metrics["friday_unseen_attack_support"] = int(unseen_attack_mask.sum())
    metrics["friday_benign_support"] = int(friday_benign_mask.sum())

    log.info(
        f"Energy threshold calibrated: strategy={args.threshold_strategy}, "
        f"q={args.ood_quantile:.4f}, threshold={threshold:.4f}, "
        f"scale={scale_value:.4f}")
    log.info(
        f"Energy gate: source known in-distribution={(id_scores >= threshold).mean():.4f}, "
        f"Friday unseen attack detected={(unseen_scores < threshold).mean():.4f}, "
        f"Friday benign false-OOD={(friday_benign_scores < threshold).mean():.4f}")
    log.info(
        f"CIC known test accuracy={metrics['id_accuracy']:.4f}, "
        f"macro-F1={metrics['id_macro_f1']:.4f}, "
        f"weighted-F1={metrics['id_weighted_f1']:.4f}")
    log.info(
        f"OOD AUROC={metrics['auroc_unseen_attack']:.4f}, "
        f"AUPR={metrics['aupr_unseen_attack']:.4f}, "
        f"FPR95={metrics['fpr95_unseen_attack']:.4f}, "
        f"unseen_detect@tau={metrics['unseen_attack_detect_at_tau']:.4f}")

    save_classification(y_id, y_pred, source_names, out_dir)
    save_ood_summary(metrics, threshold, scale_value, args, out_dir)
    save_ood_per_class(
        friday_original, friday_scores, threshold, original_names,
        bundle["benign_original_id"], source_original_ids, out_dir)
    save_baseline_absorption(
        friday_original, friday_pred_source, original_names, source_names,
        source_original_ids, bundle["benign_original_id"], source_original_ids,
        out_dir)
    save_threshold_tradeoff(id_scores, unseen_scores, friday_benign_scores, out_dir)
    save_energy_histogram(
        val_scores, id_scores, friday_scores, unseen_scores,
        friday_benign_scores, threshold, out_dir)
    if not args.skip_score_dump:
        save_energy_score_dump(val_scores, id_scores, friday_scores, threshold, out_dir)
    save_metadata(bundle, args, out_dir)
    log.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()

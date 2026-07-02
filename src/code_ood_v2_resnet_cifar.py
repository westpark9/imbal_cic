#!/usr/bin/env python3
"""
CIFAR10/SVHN ResNet18 reproduction of the code_ood_v2 routing idea.

Purpose:
  Check whether the v2 structure itself works when the base model/ID-OOD pair
  is known to support energy OOD reasonably well.

Scenario:
  - ID/source: CIFAR10 all 10 classes
  - OOD target: SVHN test
  - Train split: optional long-tail CIFAR10 subsampling
  - exp0..exp{k-2}: primary disjoint CIFAR10 family experts
  - exp{k-1}: rare fallback expert
  - routing: expert OOD candidates -> confidence + TTA stability -> fallback/all-reject
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

sys.path.append(os.path.dirname(__file__))

from code_ood_v1 import render_table_png, setup_logger


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
UNKNOWN_ID = -1


@dataclass
class Expert:
    name: str
    classes: list[int]
    model: nn.Module
    threshold: float
    score_scale: float


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


def build_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class LocalClassDataset(Dataset):
    def __init__(self, base, indices, global_classes):
        self.base = base
        self.indices = list(indices)
        self.global_classes = list(global_classes)
        self.to_local = {int(c): i for i, c in enumerate(self.global_classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x, y = self.base[self.indices[item]]
        return x, self.to_local[int(y)]


def maybe_subset_indices(indices, max_samples, seed):
    indices = np.asarray(indices, dtype=int)
    if max_samples is None or max_samples <= 0 or max_samples >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, int(max_samples), replace=False))


def make_long_tail_train_indices(labels, train_idx, imbalance_factor, min_per_class,
                                 seed, log):
    labels = np.asarray(labels)
    train_idx = np.asarray(train_idx, dtype=int)
    if imbalance_factor >= 1.0:
        return train_idx
    rng = np.random.default_rng(seed)
    classes = np.sort(np.unique(labels[train_idx]))
    max_count = max(int((labels[train_idx] == c).sum()) for c in classes)
    selected = []
    rows = []
    for rank, cls in enumerate(classes):
        cls_idx = train_idx[labels[train_idx] == cls]
        keep = int(round(max_count * (imbalance_factor ** (rank / max(len(classes) - 1, 1)))))
        keep = max(min_per_class, min(keep, len(cls_idx)))
        chosen = np.sort(rng.choice(cls_idx, keep, replace=False))
        selected.extend(chosen.tolist())
        rows.append({"class_id": int(cls), "train_support": int(keep)})
    selected = np.asarray(sorted(selected), dtype=int)
    log.info(
        f"Long-tail train enabled: imbalance_factor={imbalance_factor}, "
        f"rows={len(selected):,}/{len(train_idx):,}")
    for row in rows:
        log.info(f"  class={row['class_id']} train_support={row['train_support']}")
    return selected


def load_cifar10_svhn(args, log):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_aug_full = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform)
    train_eval_full = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=test_transform)
    test_id = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform)
    test_ood = datasets.SVHN(
        root=args.data_root, split="test", download=True, transform=test_transform)

    rng = np.random.default_rng(args.seed)
    labels = np.asarray(train_aug_full.targets)
    base_train_idx, val_idx = [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * args.val_ratio))
        val_idx.extend(cls_idx[:n_val].tolist())
        base_train_idx.extend(cls_idx[n_val:].tolist())
    base_train_idx = np.asarray(base_train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    train_idx = make_long_tail_train_indices(
        labels, base_train_idx, args.lt_imbalance_factor,
        args.lt_min_per_class, args.seed + 10, log)
    train_idx = maybe_subset_indices(train_idx, args.max_train, args.seed + 1)
    val_idx = maybe_subset_indices(val_idx, args.max_val, args.seed + 2)

    test_id = Subset(test_id, maybe_subset_indices(
        np.arange(len(test_id)), args.max_test, args.seed + 3).tolist())
    test_ood = Subset(test_ood, maybe_subset_indices(
        np.arange(len(test_ood)), args.max_ood, args.seed + 4).tolist())

    class_names = list(train_aug_full.classes)
    log.info(
        f"Loaded CIFAR10/SVHN v2: train={len(train_idx):,} val={len(val_idx):,} "
        f"test_id={len(test_id):,} svhn_ood={len(test_ood):,}")
    return train_aug_full, train_eval_full, train_idx, val_idx, test_id, test_ood, class_names


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


def train_resnet(model, train_loader, val_loader, args, device, log, name):
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            total += len(y)
            correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        scheduler.step()
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, device)
        train_acc = correct / max(total, 1)
        log.info(
            f"{name} epoch={epoch:03d} train_loss={total_loss / max(total, 1):.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} lr={scheduler.get_last_lr()[0]:.6f} "
            f"time={time.time() - t0:.1f}s")
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            log.info(f"{name} early stopping at epoch={epoch}; best_epoch={best_epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    log.info(f"{name} best validation accuracy={best_val_acc:.4f} at epoch={best_epoch}")
    return model


def evaluate_loss_acc(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total += len(y)
            correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
    return total_loss / max(total, 1), correct / max(total, 1)


def logits_to_energy(logits, temperature):
    return float(temperature) * torch.logsumexp(logits / float(temperature), dim=1)


def collect_scores_preds_labels(model, loader, device, temperature, need_preds=True):
    model.eval()
    scores_list, preds_list, labels_list, conf_list = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            proba = F.softmax(logits, dim=1)
            scores = logits_to_energy(logits, temperature)
            scores_list.append(scores.detach().cpu().numpy().astype(np.float32))
            if need_preds:
                conf, pred = proba.max(dim=1)
                preds_list.append(pred.detach().cpu().numpy().astype(np.int64))
                conf_list.append(conf.detach().cpu().numpy().astype(np.float32))
            labels_list.append(y.numpy().astype(np.int64))
    scores = np.concatenate(scores_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    preds = np.concatenate(preds_list, axis=0) if need_preds else None
    conf = np.concatenate(conf_list, axis=0) if need_preds else None
    return scores, preds, labels, conf


def calibrate_energy_threshold(scores, labels, quantile):
    per_class_tau = []
    for cls in np.unique(labels):
        cls_scores = scores[labels == cls]
        if len(cls_scores):
            per_class_tau.append(float(np.quantile(cls_scores, quantile)))
    threshold = float(min(per_class_tau)) if per_class_tau else float(
        np.quantile(scores, quantile))
    scale = float(np.std(scores) + 1e-6)
    return threshold, scale


def build_partitions(class_names, train_labels, args, log):
    vehicle = {0, 1, 8, 9}
    animal = {2, 3, 4, 5, 6, 7}
    counts = np.bincount(train_labels, minlength=len(class_names))
    tail_classes = set()
    if args.fallback_tail_classes > 0:
        order = [int(i) for i in np.argsort(counts)]
        tail_classes = set(order[:args.fallback_tail_classes])
    partitions = []
    vehicle_classes = sorted(vehicle - tail_classes)
    animal_classes = sorted(animal - tail_classes)
    if vehicle_classes:
        partitions.append(("Vehicle_Primary", vehicle_classes))
    if animal_classes:
        partitions.append(("Animal_Primary", animal_classes))
    if tail_classes:
        partitions.append(("Rare_Fallback", sorted(tail_classes)))
    covered = set(sum((classes for _, classes in partitions), []))
    leftovers = [i for i in range(len(class_names)) if i not in covered]
    if leftovers:
        partitions.append(("Other_Primary", leftovers))
    log.info("CIFAR v2 expert partitions:")
    for name, classes in partitions:
        log.info(
            f"  {name}: {', '.join(class_names[c] for c in classes)} "
            f"| train_support={int(counts[classes].sum())}")
    return partitions


def train_experts(train_aug_full, train_eval_full, train_idx, val_idx,
                  partitions, args, device, log):
    labels = np.asarray(train_aug_full.targets)
    experts = []
    for e, (name, classes) in enumerate(partitions):
        tr_local_idx = [int(i) for i in train_idx if int(labels[i]) in classes]
        va_local_idx = [int(i) for i in val_idx if int(labels[i]) in classes]
        if not tr_local_idx or not va_local_idx:
            raise ValueError(f"Expert {name} has empty train/val split")
        train_set = LocalClassDataset(train_aug_full, tr_local_idx, classes)
        val_set = LocalClassDataset(train_eval_full, va_local_idx, classes)
        train_loader = make_loader(train_set, args.batch_size, True, args.workers)
        val_loader = make_loader(val_set, args.test_batch_size, False, args.workers)
        model = build_resnet18(len(classes))
        model = train_resnet(model, train_loader, val_loader, args, device, log, name)
        val_scores, _, val_local_labels, _ = collect_scores_preds_labels(
            model, val_loader, device, args.temperature, need_preds=True)
        threshold, scale = calibrate_energy_threshold(
            val_scores, val_local_labels, args.ood_quantile)
        experts.append(Expert(name, list(classes), model, threshold, scale))
        log.info(
            f"{name} energy threshold={threshold:.4f}, scale={scale:.4f}, "
            f"val_accept={(val_scores >= threshold).mean():.4f}")
    return experts


def split_primary_fallback(experts):
    if experts and "fallback" in experts[-1].name.lower():
        return np.arange(len(experts) - 1, dtype=int), len(experts) - 1
    return np.arange(len(experts), dtype=int), None


def weak_tta(x):
    out = x.clone()
    if torch.rand(()) < 0.5:
        out = torch.flip(out, dims=[3])
    noise = torch.randn_like(out) * 0.01
    return out + noise


def compute_expert_signals(experts, x, args, device):
    n = len(x)
    e_count = len(experts)
    scores = np.zeros((n, e_count), dtype=np.float32)
    margins = np.zeros((n, e_count), dtype=np.float32)
    accepted = np.zeros((n, e_count), dtype=bool)
    local_preds = np.zeros((n, e_count), dtype=np.int64)
    global_preds = np.zeros((n, e_count), dtype=np.int64)
    local_conf = np.zeros((n, e_count), dtype=np.float32)
    stability = np.zeros((n, e_count), dtype=np.float32)

    with torch.no_grad():
        for e, expert in enumerate(experts):
            expert.model.eval()
            logits = expert.model(x)
            proba = F.softmax(logits, dim=1)
            conf, pred = proba.max(dim=1)
            energy = logits_to_energy(logits, args.temperature)
            scores[:, e] = energy.detach().cpu().numpy().astype(np.float32)
            margins[:, e] = ((energy - expert.threshold) / expert.score_scale
                             ).detach().cpu().numpy().astype(np.float32)
            accepted[:, e] = scores[:, e] >= expert.threshold
            local_preds[:, e] = pred.detach().cpu().numpy().astype(np.int64)
            local_conf[:, e] = conf.detach().cpu().numpy().astype(np.float32)
            global_map = np.asarray(expert.classes, dtype=np.int64)
            global_preds[:, e] = global_map[local_preds[:, e]]
            if args.n_views > 0:
                view_probs = []
                for _ in range(args.n_views):
                    view_logits = expert.model(weak_tta(x))
                    view_proba = F.softmax(view_logits, dim=1)
                    view_probs.append(view_proba.gather(1, pred[:, None]).squeeze(1))
                stability[:, e] = torch.stack(view_probs, dim=0).mean(
                    dim=0).detach().cpu().numpy().astype(np.float32)
    return {
        "scores": scores,
        "margins": margins,
        "accepted": accepted,
        "local_preds": local_preds,
        "global_preds": global_preds,
        "local_conf": local_conf,
        "stability": stability,
    }


def route_v2(experts, loader, args, device, log):
    """Expert-only v2 router.

    There is no global exp0 and no benign/attack binary gate in CIFAR.
    exp0..exp{k-2} are primary disjoint experts, and exp{k-1} can be a rare
    fallback expert. All experts first act as OOD gates; accepted primary
    experts are selected by local confidence + TTA stability.
    """
    primary_experts, fallback_expert = split_primary_fallback(experts)
    y_true_all = []
    y_pred_all = []
    reason_all = []
    selected_all = []
    accepted_all = []
    margins_all = []
    stability_all = []
    local_conf_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            n = len(y)
            y_pred = np.full(n, UNKNOWN_ID, dtype=np.int64)
            selected = np.full(n, -1, dtype=np.int64)
            reason = np.full(n, "all_reject", dtype=object)

            sig = compute_expert_signals(experts, x, args, device)
            accepted = sig["accepted"]
            margins = sig["margins"]
            stability = sig["stability"]
            local_conf = sig["local_conf"]

            for i in range(n):
                active_primary = primary_experts[accepted[i, primary_experts]]
                if len(active_primary) > 0:
                    combo = (local_conf[i, active_primary]
                             + args.stability_lambda
                             * stability[i, active_primary])
                    chosen = int(active_primary[np.argmax(combo)])
                    y_pred[i] = int(sig["global_preds"][i, chosen])
                    selected[i] = chosen
                    reason[i] = "only_accept" if len(active_primary) == 1 else "tta_confidence"
                    continue

                best_primary_margin = (
                    float(np.max(margins[i, primary_experts]))
                    if len(primary_experts) else float("-inf")
                )
                fallback_ok = False
                if fallback_expert is not None:
                    fallback_ok = (
                        accepted[i, fallback_expert]
                        and margins[i, fallback_expert]
                        >= best_primary_margin + args.fallback_margin_gap
                    )
                if fallback_ok:
                    chosen = int(fallback_expert)
                    y_pred[i] = int(sig["global_preds"][i, chosen])
                    selected[i] = chosen
                    reason[i] = "fallback_accept"

            y_true_all.append(y.numpy().astype(np.int64))
            y_pred_all.append(y_pred)
            reason_all.append(reason)
            selected_all.append(selected)
            accepted_all.append(accepted)
            margins_all.append(margins)
            stability_all.append(stability)
            local_conf_all.append(local_conf)

    diag = {
        "reason": np.concatenate(reason_all),
        "selected": np.concatenate(selected_all),
        "accepted": np.concatenate(accepted_all, axis=0),
        "margins": np.concatenate(margins_all, axis=0),
        "stability": np.concatenate(stability_all, axis=0),
        "local_conf": np.concatenate(local_conf_all, axis=0),
    }
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    max_primary_margin = np.max(diag["margins"][:, primary_experts], axis=1)
    diag["max_primary_margin"] = max_primary_margin.astype(np.float32)
    log.info(
        f"v2 expert-only route: all_reject={(diag['reason'] == 'all_reject').mean():.4f}, "
        f"fallback_accept={(diag['reason'] == 'fallback_accept').mean():.4f}, "
        f"known_accept={(diag['selected'] >= 0).mean():.4f}, "
        f"multi_accept={(diag['accepted'].sum(axis=1) > 1).mean():.4f}")
    return y_true, y_pred, diag

def ood_metrics(id_ood_score, ood_ood_score, ood_detect):
    y_true = np.concatenate([
        np.zeros(len(id_ood_score), dtype=np.int64),
        np.ones(len(ood_ood_score), dtype=np.int64),
    ])
    score = np.concatenate([id_ood_score, ood_ood_score])
    fpr, tpr, _ = roc_curve(y_true, score)
    idx95 = int(np.argmin(np.abs(tpr - 0.95)))
    return {
        "ood_auroc": float(roc_auc_score(y_true, score)),
        "ood_aupr": float(average_precision_score(y_true, score)),
        "ood_fpr95": float(fpr[idx95]),
        "ood_detect_rate": float(ood_detect.mean()),
    }


def save_known_classification(y_true, y_pred, class_names, out_dir):
    labels = np.arange(len(class_names))
    pred_for_metrics = y_pred.copy()
    pred_for_metrics[pred_for_metrics < 0] = len(class_names)
    metric_labels = np.arange(len(class_names) + 1)
    metric_names = class_names + ["unknown"]
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, pred_for_metrics, labels=metric_labels, zero_division=0)
    rows = []
    for i, name in enumerate(metric_names):
        rows.append({
            "class": name,
            "support": int(sup[i]),
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        })
    rows.append({
        "class": "macro_known avg",
        "support": int((y_true >= 0).sum()),
        "precision": float(precision_recall_fscore_support(
            y_true, pred_for_metrics, labels=labels,
            average="macro", zero_division=0)[0]),
        "recall": float(precision_recall_fscore_support(
            y_true, pred_for_metrics, labels=labels,
            average="macro", zero_division=0)[1]),
        "f1": float(f1_score(
            y_true, pred_for_metrics, labels=labels,
            average="macro", zero_division=0)),
    })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "1a_known_per_class.csv"), index=False)
    render_table_png(
        df, os.path.join(out_dir, "1a_known_per_class.png"),
        title="CIFAR v2 known classification")

    cm = confusion_matrix(y_true, pred_for_metrics, labels=metric_labels,
                          normalize="true")
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=8)
    ax.set_title("CIFAR v2 known confusion matrix")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            if cm[r, c] > 0:
                ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center",
                        fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1b_known_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def save_pipeline_funnel(y_true, diag, class_names, out_dir, prefix, title):
    if y_true.min() >= 0 and y_true.max() < len(class_names):
        order = list(range(len(class_names)))
        labels = class_names
    else:
        order = sorted(np.unique(y_true).tolist())
        labels = [f"svhn-{i}" for i in order]
    rows = []
    for cls, label in zip(order, labels):
        mask = y_true == cls
        if not mask.any():
            continue
        rows.append({
            "class": label,
            "support": int(mask.sum()),
            "all_reject_rate": float((diag["reason"][mask] == "all_reject").mean()),
            "primary_accept_rate": float(np.isin(
                diag["reason"][mask], ["only_accept", "tta_confidence"]).mean()),
            "fallback_accept_rate": float((diag["reason"][mask] == "fallback_accept").mean()),
            "multi_accept_rate": float((diag["accepted"][mask].sum(axis=1) > 1).mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{prefix}_pipeline_funnel.csv"), index=False)
    if len(df) == 0:
        return
    mat = df[["all_reject_rate", "primary_accept_rate", "fallback_accept_rate"]].values
    fig, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.45)))
    left = np.zeros(len(df))
    colors = ["#2e7d32", "#f44336", "#ff9800"]
    names = ["all reject", "primary accept", "fallback accept"]
    yidx = np.arange(len(df))
    for i in range(mat.shape[1]):
        ax.barh(yidx, mat[:, i], left=left, color=colors[i], label=names[i])
        left += mat[:, i]
    ax.set_yticks(yidx)
    ax.set_yticklabels([f"{r['class']} ({r['support']})" for _, r in df.iterrows()],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pipeline_funnel.png"), dpi=150)
    plt.close(fig)


def save_dashboard(y_true, diag, experts, class_names, out_dir, prefix, title):
    if y_true.min() >= 0 and y_true.max() < len(class_names):
        order = list(range(len(class_names)))
        row_names = class_names
    else:
        order = sorted(np.unique(y_true).tolist())
        row_names = [f"svhn-{i}" for i in order]
    expert_names = [expert.name for expert in experts]
    rows = []
    mats = {k: np.zeros((len(order), len(experts)), dtype=np.float32)
            for k in ["accept", "selected", "stability", "margin"]}
    for r, cls in enumerate(order):
        mask = y_true == cls
        for e, expert in enumerate(experts):
            mats["accept"][r, e] = float(diag["accepted"][mask, e].mean())
            mats["selected"][r, e] = float((diag["selected"][mask] == e).mean())
            mats["stability"][r, e] = float(diag["stability"][mask, e].mean())
            mats["margin"][r, e] = float(diag["margins"][mask, e].mean())
            rows.append({
                "class": row_names[r],
                "support": int(mask.sum()),
                "expert": expert.name,
                "accept_rate": mats["accept"][r, e],
                "selected_rate": mats["selected"][r, e],
                "mean_stability": mats["stability"][r, e],
                "mean_margin": mats["margin"][r, e],
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f"{prefix}_gate_tta_dashboard.csv"), index=False)
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(experts) * 2.2 + 4),
                                            max(7, len(order) * 0.55)))
    panels = [
        ("OOD accept rate", mats["accept"], "Blues", 0, 1),
        ("Selected rate", mats["selected"], "Blues", 0, 1),
        ("Mean TTA stability", mats["stability"], "viridis", None, None),
        ("Mean gate margin", mats["margin"], "coolwarm", None, None),
    ]
    ylabels = [f"{name} ({int((y_true == cls).sum())})"
               for name, cls in zip(row_names, order)]
    for ax, (panel_title, mat, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(expert_names)))
        ax.set_xticklabels(expert_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(panel_title, fontsize=10)
        for rr in range(mat.shape[0]):
            for cc in range(mat.shape[1]):
                ax.text(cc, rr, f"{mat[rr, cc]:.2f}", ha="center",
                        va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(out_dir, f"{prefix}_gate_tta_dashboard.png"),
                dpi=150)
    plt.close(fig)


def save_ood_summary(id_diag, ood_diag, experts, args, out_dir):
    id_ood_score = -id_diag["max_primary_margin"]
    ood_ood_score = -ood_diag["max_primary_margin"]
    ood_detect = ood_diag["reason"] == "all_reject"
    metrics = ood_metrics(id_ood_score, ood_ood_score, ood_detect)
    rows = [
        {"metric": "known_all_reject_rate", "value": float((id_diag["reason"] == "all_reject").mean())},
        {"metric": "known_fallback_accept_rate", "value": float((id_diag["reason"] == "fallback_accept").mean())},
        {"metric": "known_primary_accept_rate", "value": float(np.isin(id_diag["reason"], ["only_accept", "tta_confidence"]).mean())},
        {"metric": "known_multi_accept_rate", "value": float((id_diag["accepted"].sum(axis=1) > 1).mean())},
        {"metric": "svhn_all_reject_rate", "value": float((ood_diag["reason"] == "all_reject").mean())},
        {"metric": "svhn_fallback_accept_rate", "value": float((ood_diag["reason"] == "fallback_accept").mean())},
        {"metric": "svhn_known_accept_rate", "value": float((ood_diag["selected"] >= 0).mean())},
        {"metric": "svhn_multi_accept_rate", "value": float((ood_diag["accepted"].sum(axis=1) > 1).mean())},
    ]
    rows.extend({"metric": key, "value": float(value)} for key, value in metrics.items())
    for i, expert in enumerate(experts):
        rows.append({"metric": f"expert_{i}_name", "value": expert.name})
        rows.append({"metric": f"expert_{i}_classes", "value": "|".join(map(str, expert.classes))})
        rows.append({"metric": f"expert_{i}_threshold", "value": float(expert.threshold)})
    rows.append({"metric": "args", "value": repr(vars(args))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "2a_ood_summary.csv"), index=False)
    render_table_png(df, os.path.join(out_dir, "2a_ood_summary.png"),
                     title="CIFAR expert-only v2 SVHN OOD summary")


def save_max_margin_hist(id_diag, ood_diag, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(id_diag["max_primary_margin"], bins=80, alpha=0.35, density=True,
            label="cifar10_test_id")
    ax.hist(ood_diag["max_primary_margin"], bins=80, alpha=0.35, density=True,
            label="svhn_ood")
    ax.axvline(0.0, color="black", linestyle="--", label="accept boundary")
    ax.set_xlabel("max primary expert gate margin; ID expected high")
    ax.set_ylabel("density")
    ax.set_title("Expert gate max-margin distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2b_max_expert_margin_histogram.png"), dpi=150)
    plt.close(fig)


def save_baseline_absorption(y_ood, pred_ood, class_names, out_dir):
    rows = []
    names = class_names + ["unknown"]
    pred_for_count = pred_ood.copy()
    pred_for_count[pred_for_count < 0] = len(class_names)
    for pred in sorted(np.unique(pred_for_count).tolist()):
        rows.append({
            "v2_pred_class": names[int(pred)],
            "count": int((pred_for_count == pred).sum()),
            "rate": float((pred_for_count == pred).mean()),
        })
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df.to_csv(os.path.join(out_dir, "2c_baseline_absorption.csv"), index=False)
    render_table_png(df, os.path.join(out_dir, "2c_baseline_absorption.png"),
                     title="Expert-only v2 absorption of SVHN")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.expanduser("~/.cache/torchvision"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lt_imbalance_factor", type=float, default=0.1)
    parser.add_argument("--lt_min_per_class", type=int, default=50)
    parser.add_argument("--fallback_tail_classes", type=int, default=2)
    parser.add_argument("--ood_quantile", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8,
                        help="0 disables early stopping")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--stability_lambda", type=float, default=0.1)
    parser.add_argument("--fallback_margin_gap", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_val", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    parser.add_argument("--max_ood", type=int, default=0)
    parser.add_argument("--out_root", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = make_unique_out_dir(
        args.out_root, f"{ts}_pid{os.getpid()}_cifar10_svhn_resnet18_v2")
    log = setup_logger(os.path.join(out_dir, "experiment.log"))
    log.info(f"Args: {vars(args)}")
    device = detect_device()
    log.info(f"Device: {device}")

    train_aug_full, train_eval_full, train_idx, val_idx, test_id_set, test_ood_set, class_names = (
        load_cifar10_svhn(args, log))
    train_set = Subset(train_aug_full, train_idx.tolist())
    val_set = Subset(train_eval_full, val_idx.tolist())
    test_id_loader = make_loader(test_id_set, args.test_batch_size, False, args.workers)
    test_ood_loader = make_loader(test_ood_set, args.test_batch_size, False, args.workers)

    train_labels = np.asarray(train_aug_full.targets)[train_idx]
    partitions = build_partitions(class_names, train_labels, args, log)
    experts = train_experts(
        train_aug_full, train_eval_full, train_idx, val_idx,
        partitions, args, device, log)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Evaluating expert-only v2 router on CIFAR10 known test")
    y_known, pred_known, diag_known = route_v2(
        experts, test_id_loader, args, device, log)
    log.info("Evaluating expert-only v2 router on SVHN OOD test")
    y_ood, pred_ood, diag_ood = route_v2(
        experts, test_ood_loader, args, device, log)

    known_pred_for_f1 = pred_known.copy()
    known_pred_for_f1[known_pred_for_f1 < 0] = len(class_names)
    known_macro_f1 = f1_score(
        y_known, known_pred_for_f1, labels=np.arange(len(class_names)),
        average="macro", zero_division=0)
    known_acc = float((pred_known == y_known).mean())
    ood_detect = diag_ood["reason"] == "all_reject"
    log.info(
        f"Known accuracy={known_acc:.4f}, macro-F1={known_macro_f1:.4f}; "
        f"SVHN OOD detect={ood_detect.mean():.4f} "
        f"(all_reject={(diag_ood['reason'] == 'all_reject').mean():.4f}, "
        f"known_accept={(diag_ood['selected'] >= 0).mean():.4f})")

    save_known_classification(y_known, pred_known, class_names, out_dir)
    save_pipeline_funnel(
        y_known, diag_known, class_names, out_dir, "1c_known",
        "CIFAR known expert-only routing funnel")
    save_dashboard(
        y_known, diag_known, experts, class_names, out_dir, "1d_known",
        "CIFAR known expert gate/TTA dashboard")
    save_ood_summary(diag_known, diag_ood, experts, args, out_dir)
    save_max_margin_hist(diag_known, diag_ood, out_dir)
    save_pipeline_funnel(
        y_ood, diag_ood, class_names, out_dir, "2d_ood",
        "SVHN OOD expert-only routing funnel")
    save_dashboard(
        y_ood, diag_ood, experts, class_names, out_dir, "2e_ood",
        "SVHN OOD expert gate/TTA dashboard")
    save_baseline_absorption(y_ood, pred_ood, class_names, out_dir)
    log.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()

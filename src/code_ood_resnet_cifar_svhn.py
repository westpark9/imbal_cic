#!/usr/bin/env python3
"""
ResNet18 energy-based OOD sanity check on CIFAR-10 (ID) vs SVHN (OOD).

This script is intentionally close to the post-hoc setting of Energy OOD:
  1. train a strong CIFAR-10 classifier with cross-entropy only,
  2. compute energy from logits,
  3. evaluate CIFAR-10 test as ID and SVHN test as OOD.

Run:
  python src/code_ood_resnet_cifar_svhn.py \
    --data_root ~/.cache/torchvision \
    --epochs 120
"""
import argparse
import os
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

sys.path.append(os.path.dirname(__file__))

from code_ood_v1 import render_table_png, setup_logger


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


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


def maybe_subset(dataset, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), int(max_samples), replace=False))
    return Subset(dataset, idx.tolist())


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

    train_full = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform)
    val_eval_full = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=test_transform)
    test_id = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform)
    test_ood = datasets.SVHN(
        root=args.data_root, split="test", download=True, transform=test_transform)

    rng = np.random.default_rng(args.seed)
    labels = np.asarray(train_full.targets)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * args.val_ratio))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)

    if args.max_train and args.max_train > 0 and args.max_train < len(train_idx):
        train_idx = np.sort(rng.choice(train_idx, int(args.max_train), replace=False))
    if args.max_val and args.max_val > 0 and args.max_val < len(val_idx):
        val_idx = np.sort(rng.choice(val_idx, int(args.max_val), replace=False))

    train_set = Subset(train_full, train_idx.tolist())
    val_set = Subset(val_eval_full, val_idx.tolist())
    test_id = maybe_subset(test_id, args.max_test, args.seed + 1)
    test_ood = maybe_subset(test_ood, args.max_ood, args.seed + 2)

    log.info(
        f"Loaded CIFAR-10/SVHN: train={len(train_set):,} val={len(val_set):,} "
        f"test_id={len(test_id):,} test_ood={len(test_ood):,}")
    return train_set, val_set, test_id, test_ood, list(train_full.classes)


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


def train_resnet(model, train_loader, val_loader, args, device, log):
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
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
            f"epoch={epoch:03d} train_loss={total_loss / max(total, 1):.4f} "
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
            log.info(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info(f"Best validation accuracy={best_val_acc:.4f} at epoch={best_epoch}")
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
                preds_list.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            labels_list.append(y.numpy().astype(np.int64))
            del x, y, logits, scores
    scores = np.concatenate(scores_list, axis=0)
    preds = np.concatenate(preds_list, axis=0) if need_preds else None
    labels = np.concatenate(labels_list, axis=0)
    return scores, preds, labels


def calibrate_energy_threshold(val_scores, y_val, quantile):
    per_class_tau = []
    for cls in np.unique(y_val):
        cls_scores = val_scores[y_val == cls]
        if len(cls_scores):
            per_class_tau.append(float(np.quantile(cls_scores, quantile)))
    threshold = float(min(per_class_tau)) if per_class_tau else float(
        np.quantile(val_scores, quantile))
    scale = float(np.std(val_scores) + 1e-6)
    return threshold, scale


def ood_metrics(id_scores, ood_scores, threshold):
    y_true = np.concatenate([
        np.zeros(len(id_scores), dtype=np.int64),
        np.ones(len(ood_scores), dtype=np.int64),
    ])
    ood_score = np.concatenate([-id_scores, -ood_scores])
    auroc = float(roc_auc_score(y_true, ood_score))
    aupr = float(average_precision_score(y_true, ood_score))
    fpr, tpr, _ = roc_curve(y_true, ood_score)
    idx95 = int(np.argmin(np.abs(tpr - 0.95)))
    return {
        "auroc_ood": auroc,
        "aupr_ood": aupr,
        "fpr95_ood": float(fpr[idx95]),
        "id_retain_at_tau": float((id_scores >= threshold).mean()),
        "ood_detect_at_tau": float((ood_scores < threshold).mean()),
    }


def save_classification(y_true, y_pred, class_names, out_dir):
    labels = np.arange(len(class_names))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    for cls in labels:
        rows.append({
            "class": class_names[int(cls)],
            "support": int(sup[cls]),
            "precision": float(prec[cls]),
            "recall": float(rec[cls]),
            "f1": float(f1[cls]),
        })
    rows.append({
        "class": "macro avg",
        "support": int(sup.sum()),
        "precision": float(precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)[0]),
        "recall": float(precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)[1]),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    })
    rows.append({
        "class": "weighted avg",
        "support": int(sup.sum()),
        "precision": float(precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0)[0]),
        "recall": float(precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0)[1]),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "id_classification.csv"), index=False)
    render_table_png(
        df, os.path.join(out_dir, "id_classification.png"),
        title="CIFAR-10 ResNet18 classification")

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("CIFAR-10 ResNet18 confusion matrix")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "id_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def save_ood_summary(metrics, threshold, scale, args, out_dir):
    rows = [
        {"metric": "energy_threshold", "value": float(threshold)},
        {"metric": "energy_scale", "value": float(scale)},
        {"metric": "temperature", "value": float(args.temperature)},
    ]
    rows.extend({"metric": key, "value": float(value)}
                for key, value in metrics.items())
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ood_summary.csv"), index=False)
    render_table_png(
        df, os.path.join(out_dir, "ood_summary.png"),
        title="CIFAR-10 ResNet18 vs SVHN energy OOD summary")


def save_energy_outputs(train_scores, val_scores, id_scores, ood_scores,
                        threshold, out_dir):
    split_parts = [
        np.repeat("cifar10_val_id", len(val_scores)),
        np.repeat("cifar10_test_id", len(id_scores)),
        np.repeat("svhn_test_ood", len(ood_scores)),
    ]
    score_parts = [val_scores, id_scores, ood_scores]
    if train_scores is not None:
        split_parts.insert(0, np.repeat("cifar10_train_id", len(train_scores)))
        score_parts.insert(0, train_scores)
    pd.DataFrame({
        "split": np.concatenate(split_parts),
        "energy": np.concatenate(score_parts),
    }).to_csv(os.path.join(out_dir, "energy_scores.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_scores is not None:
        ax.hist(train_scores, bins=80, alpha=0.28, density=True, label="train_id")
    ax.hist(val_scores, bins=80, alpha=0.35, density=True, label="val_id")
    ax.hist(id_scores, bins=80, alpha=0.35, density=True, label="test_id")
    ax.hist(ood_scores, bins=80, alpha=0.35, density=True, label="svhn_ood")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"tau={threshold:.4f}")
    ax.set_title("Energy score distributions")
    ax.set_xlabel("energy = T logsumexp(logits / T); ID expected high")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_histogram.png"), dpi=150)
    plt.close(fig)


def save_threshold_tradeoff(id_scores, ood_scores, out_dir):
    rows = []
    for q in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90]:
        tau = float(np.quantile(id_scores, q))
        rows.append({
            "id_quantile_tau": q,
            "threshold": tau,
            "id_retain": float((id_scores >= tau).mean()),
            "ood_detect": float((ood_scores < tau).mean()),
            "ood_in": float((ood_scores >= tau).mean()),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "threshold_tradeoff.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.expanduser("~/.cache/torchvision"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--ood_quantile", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=0,
                        help="0 disables early stopping")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save_train_scores", action="store_true",
                        help="Also collect train energy scores; disabled by default to reduce memory pressure")
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
        args.out_root, f"{ts}_pid{os.getpid()}_cifar10_svhn_resnet18_energy")
    log = setup_logger(os.path.join(out_dir, "experiment.log"))
    log.info(f"Args: {vars(args)}")

    device = detect_device()
    log.info(f"Device: {device}")
    train_set, val_set, test_id_set, test_ood_set, class_names = load_cifar10_svhn(args, log)
    train_loader = make_loader(train_set, args.batch_size, True, args.workers)
    val_loader = make_loader(val_set, args.test_batch_size, False, args.workers)
    test_id_loader = make_loader(test_id_set, args.test_batch_size, False, args.workers)
    test_ood_loader = make_loader(test_ood_set, args.test_batch_size, False, args.workers)
    train_eval_loader = (
        make_loader(train_set, args.test_batch_size, False, args.workers)
        if args.save_train_scores else None
    )

    t0 = time.time()
    model = build_resnet18(len(class_names))
    model = train_resnet(model, train_loader, val_loader, args, device, log)
    log.info(f"Training completed in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Collecting streaming energy scores for ID/OOD splits")
    train_scores = None
    if train_eval_loader is not None:
        train_scores, _, _ = collect_scores_preds_labels(
            model, train_eval_loader, device, args.temperature, need_preds=False)
    val_scores, _, y_val = collect_scores_preds_labels(
        model, val_loader, device, args.temperature, need_preds=False)
    id_scores, y_pred, y_id = collect_scores_preds_labels(
        model, test_id_loader, device, args.temperature, need_preds=True)
    ood_scores, _, _ = collect_scores_preds_labels(
        model, test_ood_loader, device, args.temperature, need_preds=False)
    threshold, scale = calibrate_energy_threshold(
        val_scores, y_val, args.ood_quantile)

    id_macro_f1 = float(f1_score(y_id, y_pred, average="macro", zero_division=0))
    id_weighted_f1 = float(f1_score(y_id, y_pred, average="weighted", zero_division=0))
    metrics = ood_metrics(id_scores, ood_scores, threshold)
    metrics["id_macro_f1"] = id_macro_f1
    metrics["id_weighted_f1"] = id_weighted_f1
    metrics["id_accuracy"] = float((y_pred == y_id).mean())

    log.info(
        f"Energy threshold calibrated: q={args.ood_quantile:.4f}, "
        f"threshold={threshold:.4f}, scale={scale:.4f}")
    log.info(
        f"Energy gate: test_id in-distribution={(id_scores >= threshold).mean():.4f}, "
        f"svhn in-distribution={(ood_scores >= threshold).mean():.4f}")
    log.info(
        f"CIFAR-10 test accuracy={metrics['id_accuracy']:.4f}, "
        f"macro-F1={id_macro_f1:.4f}, weighted-F1={id_weighted_f1:.4f}")
    log.info(
        f"OOD AUROC={metrics['auroc_ood']:.4f}, AUPR={metrics['aupr_ood']:.4f}, "
        f"FPR95={metrics['fpr95_ood']:.4f}, "
        f"OOD_detect@tau={metrics['ood_detect_at_tau']:.4f}")

    save_classification(y_id, y_pred, class_names, out_dir)
    save_ood_summary(metrics, threshold, scale, args, out_dir)
    save_energy_outputs(train_scores, val_scores, id_scores, ood_scores, threshold, out_dir)
    save_threshold_tradeoff(id_scores, ood_scores, out_dir)
    log.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()

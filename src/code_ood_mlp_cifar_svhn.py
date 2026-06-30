#!/usr/bin/env python3
"""
MLP energy-based OOD sanity check on CIFAR-10 (ID) vs SVHN (OOD).

Goal:
  Verify whether a neural MLP baseline produces more useful energy scores for
  OOD detection than the earlier tree-based baseline.

Run:
  python src/code_ood_mlp_cifar_svhn.py \
    --data_root ~/.cache/torchvision \
    --ood_quantile 0.05
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(__file__))

from code_ood_v1 import detect_device, render_table_png, setup_logger


class TorchMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hidden
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifierWrapper:
    def __init__(self, input_dim, n_classes, hidden_dims, dropout, lr,
                 weight_decay, batch_size, epochs, patience, device, seed):
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.model = TorchMLP(
            input_dim, n_classes, self.hidden_dims, dropout).to(self.device)

    def fit(self, X, y, sample_weight=None, eval_set=None, verbose=False):
        torch.manual_seed(self.seed)
        Xs = self.scaler.fit_transform(
            np.asarray(X, dtype=np.float32)).astype(np.float32)
        y = np.asarray(y, dtype=np.int64)
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        sample_weight = np.asarray(sample_weight, dtype=np.float32)

        ds = TensorDataset(
            torch.from_numpy(Xs),
            torch.from_numpy(y),
            torch.from_numpy(sample_weight),
        )
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            num_workers=0, drop_last=False)
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        best_state = None
        best_loss = float("inf")
        stale = 0
        Xv = yv = None
        if eval_set:
            Xv_np, yv_np = eval_set[0]
            if len(Xv_np) > 0:
                Xv = torch.from_numpy(
                    self.scaler.transform(
                        np.asarray(Xv_np, dtype=np.float32)
                    ).astype(np.float32)
                ).to(self.device)
                yv = torch.from_numpy(
                    np.asarray(yv_np, dtype=np.int64)).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb, wb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss_each = F.cross_entropy(
                    self.model(xb), yb, reduction="none")
                loss = (loss_each * wb).mean()
                loss.backward()
                opt.step()

            val_loss = self._eval_loss(Xv, yv) if Xv is not None else float(
                loss.detach().cpu())
            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if verbose:
                print(f"epoch={epoch + 1} val_loss={val_loss:.6f}")
            if stale >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _eval_loss(self, Xv, yv):
        self.model.eval()
        with torch.no_grad():
            total = 0.0
            count = 0
            for start in range(0, len(Xv), self.batch_size * 4):
                logits = self.model(Xv[start:start + self.batch_size * 4])
                loss = F.cross_entropy(
                    logits, yv[start:start + self.batch_size * 4],
                    reduction="sum")
                total += float(loss.detach().cpu())
                count += len(logits)
        return total / max(count, 1)

    def predict_logits(self, X, chunk=50_000):
        if len(X) == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        Xs = self.scaler.transform(
            np.asarray(X, dtype=np.float32)).astype(np.float32)
        self.model.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(Xs), chunk):
                xb = torch.from_numpy(Xs[start:start + chunk]).to(self.device)
                outs.append(
                    self.model(xb).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    def predict_proba(self, X):
        logits = self.predict_logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def parse_hidden_dims(value):
    dims = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            dims.append(int(part))
    if not dims:
        raise ValueError("--mlp_hidden must contain at least one dimension")
    return dims


def balanced_weights(y):
    counts = np.maximum(np.bincount(y), 1)
    return (len(y) / (len(counts) * counts[y])).astype(np.float32)


def make_unique_out_dir(root, prefix):
    base = os.path.join(root, prefix)
    candidate = base
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    os.makedirs(candidate)
    return candidate


def dataset_to_numpy(dataset, batch_size, max_samples, seed):
    idx = None
    if max_samples and max_samples > 0 and max_samples < len(dataset):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(dataset), int(max_samples), replace=False))
        dataset = torch.utils.data.Subset(dataset, idx.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    X_batches = []
    y_batches = []
    for xb, yb in loader:
        X_batches.append(xb.view(len(xb), -1).cpu().numpy().astype(np.float32))
        y_batches.append(yb.cpu().numpy().astype(np.int64))
    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)
    return X, y


def load_cifar10_svhn(args, log):
    transform = transforms.ToTensor()
    cifar_train = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform)
    svhn_test = datasets.SVHN(
        root=args.data_root, split="test", download=True, transform=transform)

    X_train_full, y_train_full = dataset_to_numpy(
        cifar_train, args.io_batch_size, args.max_train, args.seed)
    X_test_id, y_test_id = dataset_to_numpy(
        cifar_test, args.io_batch_size, args.max_test, args.seed + 1)
    X_test_ood, y_test_ood = dataset_to_numpy(
        svhn_test, args.io_batch_size, args.max_ood, args.seed + 2)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_ratio,
        stratify=y_train_full,
        random_state=args.seed,
    )
    class_names = list(cifar_train.classes)
    log.info(
        f"Loaded CIFAR-10/SVHN: train={len(y_train):,} val={len(y_val):,} "
        f"test_id={len(y_test_id):,} test_ood={len(y_test_ood):,}")
    return X_train, y_train, X_val, y_val, X_test_id, y_test_id, X_test_ood, class_names


def train_mlp_model(X_tr, y_tr, X_va, y_va, n_cls, args, device, log, name):
    log.info(f"Training MLP {name}: n_cls={n_cls}, hidden={args.mlp_hidden}")
    model = MLPClassifierWrapper(
        input_dim=X_tr.shape[1],
        n_classes=n_cls,
        hidden_dims=parse_hidden_dims(args.mlp_hidden),
        dropout=args.mlp_dropout,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        batch_size=args.mlp_batch_size,
        epochs=args.mlp_epochs,
        patience=args.mlp_patience,
        device=device,
        seed=args.seed,
    )
    model.fit(
        X_tr, y_tr, sample_weight=balanced_weights(y_tr),
        eval_set=[(X_va, y_va)], verbose=False)
    return model


def energy_scores(model, X, temperature):
    logits = model.predict_logits(X)
    scaled = logits / float(temperature)
    return (float(temperature) * torch.logsumexp(
        torch.from_numpy(scaled), dim=1).numpy()).astype(np.float32)


def calibrate_energy_threshold(model, X_val, y_val, temperature, quantile):
    scores = energy_scores(model, X_val, temperature)
    per_class_tau = []
    for cls in np.unique(y_val):
        cls_scores = scores[y_val == cls]
        if len(cls_scores):
            per_class_tau.append(float(np.quantile(cls_scores, quantile)))
    threshold = float(min(per_class_tau)) if per_class_tau else float(
        np.quantile(scores, quantile))
    scale = float(np.std(scores) + 1e-6)
    return threshold, scale, scores


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
    fpr95 = float(fpr[idx95])
    id_retain = float((id_scores >= threshold).mean())
    ood_detect = float((ood_scores < threshold).mean())
    return {
        "auroc_ood": auroc,
        "aupr_ood": aupr,
        "fpr95_ood": fpr95,
        "id_retain_at_tau": id_retain,
        "ood_detect_at_tau": ood_detect,
    }


def save_energy_histogram(train_scores, val_scores, test_id_scores, test_ood_scores,
                          threshold, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(train_scores, bins=80, alpha=0.35, density=True, label="train_id")
    ax.hist(val_scores, bins=80, alpha=0.35, density=True, label="val_id")
    ax.hist(test_id_scores, bins=80, alpha=0.35, density=True, label="test_id")
    ax.hist(test_ood_scores, bins=80, alpha=0.35, density=True, label="test_ood_svhn")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"tau={threshold:.4f}")
    ax.set_title("Energy score distributions")
    ax.set_xlabel("energy")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_histogram.png"), dpi=150)
    plt.close(fig)


def save_id_classification(y_true, y_pred, class_names, out_dir):
    labels = np.arange(len(class_names))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    for i in np.argsort(-sup):
        if sup[i] == 0:
            continue
        rows.append({
            "class": class_names[i],
            "support": int(sup[i]),
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        })
    rows.append({
        "class": "macro avg",
        "support": int(sup.sum()),
        "precision": float(np.mean(prec)),
        "recall": float(np.mean(rec)),
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
        title="CIFAR-10 test classification")

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("CIFAR-10 confusion matrix (row-normalized)")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "id_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def save_ood_summary(metrics, threshold, scale, out_dir):
    rows = [{"metric": "energy_threshold", "value": float(threshold)},
            {"metric": "energy_scale", "value": float(scale)}]
    for key, value in metrics.items():
        rows.append({"metric": key, "value": float(value)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ood_summary.csv"), index=False)
    render_table_png(
        df, os.path.join(out_dir, "ood_summary.png"),
        title="CIFAR-10 vs SVHN energy OOD summary")


def save_score_dump(id_scores, ood_scores, out_dir):
    df = pd.DataFrame({
        "split": np.concatenate([
            np.repeat("cifar10_test_id", len(id_scores)),
            np.repeat("svhn_test_ood", len(ood_scores)),
        ]),
        "energy": np.concatenate([id_scores, ood_scores]),
    })
    df.to_csv(os.path.join(out_dir, "energy_scores.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.expanduser("~/.cache/torchvision"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--ood_quantile", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mlp_hidden", default="1024,512,256")
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_lr", type=float, default=3e-4)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_batch_size", type=int, default=512)
    parser.add_argument("--mlp_epochs", type=int, default=50)
    parser.add_argument("--mlp_patience", type=int, default=8)
    parser.add_argument("--io_batch_size", type=int, default=1024)
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    parser.add_argument("--max_ood", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = make_unique_out_dir(
        "results", f"{ts}_pid{os.getpid()}_cifar10_svhn_mlp_energy")
    log = setup_logger(os.path.join(out_dir, "experiment.log"))
    log.info(f"Args: {vars(args)}")

    device = detect_device()
    log.info(f"Device: {device}")
    t0 = time.time()
    X_tr, y_tr, X_va, y_va, X_te, y_te, X_ood, class_names = load_cifar10_svhn(args, log)
    log.info(f"Data prepared in {time.time() - t0:.1f}s")

    t0 = time.time()
    model = train_mlp_model(
        X_tr, y_tr, X_va, y_va, len(class_names), args, device, log, "baseline")
    log.info(f"Baseline trained in {time.time() - t0:.1f}s")

    threshold, scale, val_scores = calibrate_energy_threshold(
        model, X_va, y_va, args.temperature, args.ood_quantile)
    train_scores = energy_scores(model, X_tr, args.temperature)
    test_id_scores = energy_scores(model, X_te, args.temperature)
    test_ood_scores = energy_scores(model, X_ood, args.temperature)
    log.info(
        f"Energy threshold calibrated: q={args.ood_quantile:.4f}, "
        f"threshold={threshold:.4f}, scale={scale:.4f}")
    log.info(
        f"Energy gate: test_id in-distribution={(test_id_scores >= threshold).mean():.4f}, "
        f"svhn in-distribution={(test_ood_scores >= threshold).mean():.4f}")

    y_pred = np.asarray(model.predict(X_te), dtype=int)
    id_macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    id_weighted_f1 = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))
    metrics = ood_metrics(test_id_scores, test_ood_scores, threshold)
    metrics["id_macro_f1"] = id_macro_f1
    metrics["id_weighted_f1"] = id_weighted_f1
    log.info(
        f"CIFAR-10 test macro-F1={id_macro_f1:.4f}, "
        f"weighted-F1={id_weighted_f1:.4f}")
    log.info(
        f"OOD AUROC={metrics['auroc_ood']:.4f}, "
        f"AUPR={metrics['aupr_ood']:.4f}, "
        f"FPR95={metrics['fpr95_ood']:.4f}, "
        f"OOD_detect@tau={metrics['ood_detect_at_tau']:.4f}")

    save_id_classification(y_te, y_pred, class_names, out_dir)
    save_ood_summary(metrics, threshold, scale, out_dir)
    save_energy_histogram(
        train_scores, val_scores, test_id_scores, test_ood_scores, threshold, out_dir)
    save_score_dump(test_id_scores, test_ood_scores, out_dir)
    log.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()

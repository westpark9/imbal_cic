#!/usr/bin/env python3
"""
code_mat_mini_imagenet.py  –  FAR-MoE on Downsampled ImageNet 64×64
=====================================================================
Pipeline: ResNet50 (pretrained, fine-tuned on downsampled data)
          → 2048-dim features → CE-trained linear FAR-MoE ensemble.

Dataset: Downsampled ImageNet 64×64 (Chrabaszcz et al. 2017)
         Binary pickle batches; 1000 classes; labels 1-1000.
         Files: data/imagenet/train_data_batch_{1..5}, val_data

Key differences from code_mat_imagenet.py:
  - Reads pickle batches directly (no JPEG files, no tar extraction)
  - Labels remapped 1-1000 → 0-999
  - Stratified random train/val/test split (no LT annotation files)
  - --lt_sampling applies exponential LT subsampling for imbalance
  - --max_classes N subsets to N classes for quick smoke-tests
  - Transforms adapted for 64×64 source resolution

Usage
-----
  python src/code_mat_mini_imagenet.py                           # full run
  python src/code_mat_mini_imagenet.py --max_classes 100         # 100-class smoke test
  python src/code_mat_mini_imagenet.py --lt_sampling --max_classes 200
  python src/code_mat_mini_imagenet.py --backbone_epochs 0       # skip fine-tuning
"""

import argparse
import datetime
import logging
import os
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.models as tv_models
import torchvision.transforms as T

warnings.filterwarnings("ignore")


# ── Visualization (CLAUDE.md spec) ───────────────────────────────────────────
BLUE   = "#cce5ff"
RED    = "#ffcccc"
YELLOW = "#fff9cc"
GRAY   = "#f2f2f2"
WHITE  = "#ffffff"
EPS    = 0.001

COL_HEADERS = [
    "class", "support",
    "prec(B)", "prec(M)",
    "recall(B)", "recall(M)",
    "f1(B)", "f1(M)",
    "delta_f1",
]


def save_colored_table(rows: list, col_headers: list, path, title: str = ""):
    n_cols = len(col_headers)
    n_rows = len(rows)
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
                    colors.append(BLUE if m > b + EPS else (RED if m < b - EPS else YELLOW))
                else:
                    colors.append(WHITE)
            elif row.get("_footer") or col == "support":
                colors.append(GRAY)
            else:
                colors.append(WHITE)
        cell_text.append(texts)
        cell_colors.append(colors)

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.2), max(4, n_rows * 0.35)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_headers,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(n_cols)))
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Union-Find ────────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        self.p[self.find(a)] = self.find(b)

    def groups(self) -> List[List[int]]:
        d: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.p)):
            d[self.find(i)].append(i)
        return list(d.values())


# ── OOD detection ─────────────────────────────────────────────────────────────
class MahalanobisOOD:
    def __init__(self, max_fit_samples: int = 10000, random_state: int = 42):
        self.max_fit_samples = max_fit_samples
        self.random_state    = random_state
        self.mean_:      Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.var_:       Optional[np.ndarray] = None
        self.mode:       Optional[str]        = None

    def fit(self, X: np.ndarray) -> "MahalanobisOOD":
        X = np.asarray(X, dtype=np.float32)
        if len(X) > self.max_fit_samples:
            rng = np.random.default_rng(self.random_state)
            X   = X[rng.choice(len(X), size=self.max_fit_samples, replace=False)]
        try:
            lw = LedoitWolf().fit(X)
            self.mean_      = lw.location_.astype(np.float32)
            self.precision_ = lw.precision_.astype(np.float32)
            self.mode       = "full"
        except Exception:
            self.mean_ = np.mean(X, axis=0).astype(np.float32)
            self.var_  = (np.var(X, axis=0) + 1e-6).astype(np.float32)
            self.mode  = "diag"
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X    = np.asarray(X, dtype=np.float32)
        diff = X - self.mean_
        if self.mode == "full":
            dist2 = np.einsum("bi,ij,bj->b", diff, self.precision_, diff, optimize=True)
        else:
            dist2 = np.sum(diff ** 2 / self.var_, axis=1)
        return dist2.astype(np.float32)


# ── Expert ────────────────────────────────────────────────────────────────────
class Expert:
    """Single linear-head expert (CE-trained, weighted) with Mahalanobis OOD."""

    def __init__(
        self,
        name:           str,
        n_global:       int,
        feat_idx:       Optional[np.ndarray],
        target_cls:     Optional[List[int]],
        target_boost:   float,
        balanced:       bool,
        always_include: bool,
        device:         torch.device,
        seed:           int = 42,
    ):
        self.name           = name
        self.n_global       = n_global
        self.feat_idx       = feat_idx
        self.target_cls     = target_cls
        self.target_boost   = target_boost
        self.balanced       = balanced
        self.always_include = always_include
        self.device         = device
        self.seed           = seed

        self.head:          Optional[nn.Linear]      = None
        self.ood_model:     Optional[MahalanobisOOD] = None
        self.ood_feat_mean: Optional[np.ndarray]     = None
        self.ood_feat_std:  Optional[np.ndarray]     = None
        self.is_trained:    bool = False

    def _get_feat(self, X: np.ndarray) -> np.ndarray:
        return X if self.feat_idx is None else X[:, self.feat_idx]

    def _build_weights(self, y: np.ndarray) -> np.ndarray:
        w = np.ones(len(y), dtype=np.float32)
        if self.balanced:
            counts = np.maximum(np.bincount(y, minlength=self.n_global), 1)
            cw = len(y) / (self.n_global * counts)
            w  = cw[y].astype(np.float32)
        if self.target_cls is not None:
            w[np.isin(y, self.target_cls)] *= self.target_boost
        w /= w.mean()
        return w

    def fit(self, X: np.ndarray, y: np.ndarray, log: logging.Logger,
            epochs: int = 30, lr: float = 0.1, batch_size: int = 1024):
        Xf       = self._get_feat(X)
        w        = self._build_weights(y)
        feat_dim = Xf.shape[1]

        self.head = nn.Linear(feat_dim, self.n_global).to(self.device)
        opt   = optim.SGD(self.head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        X_t   = torch.tensor(Xf, dtype=torch.float32)
        y_t   = torch.tensor(y,  dtype=torch.long)
        w_t   = torch.tensor(w,  dtype=torch.float32)
        n     = len(y_t)

        self.head.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                Xb  = X_t[idx].to(self.device)
                yb  = y_t[idx].to(self.device)
                wb  = w_t[idx].to(self.device)
                opt.zero_grad()
                (F.cross_entropy(self.head(Xb), yb, reduction='none') * wb).mean().backward()
                opt.step()
            sched.step()

        self.ood_feat_mean = np.mean(Xf, axis=0).astype(np.float32)
        self.ood_feat_std  = (np.std(Xf, axis=0) + 1e-6).astype(np.float32)
        self.ood_model     = MahalanobisOOD(random_state=self.seed).fit(
            (Xf - self.ood_feat_mean) / self.ood_feat_std)
        self.is_trained = True

    def _proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self._get_feat(X)
        self.head.eval()
        with torch.no_grad():
            logits = self.head(torch.tensor(Xf, dtype=torch.float32).to(self.device))
            return F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

    def predict_proba_global(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.head is None:
            return np.full((len(X), self.n_global), 1.0 / self.n_global, dtype=np.float32)
        return self._proba(X)

    def ood_scores(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.full(len(X), 1e6, dtype=np.float32)
        Xf = self._get_feat(X)
        return self.ood_model.score_samples((Xf - self.ood_feat_mean) / self.ood_feat_std)

    def margin(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(len(X), dtype=np.float32)
        s = np.sort(self._proba(X), axis=1)
        return (s[:, -1] - s[:, -2]).astype(np.float32)

    def entropy(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.ones(len(X), dtype=np.float32)
        probs = np.clip(self._proba(X).astype(np.float64), 1e-12, 1.0)
        ent   = -np.sum(probs * np.log(probs), axis=1)
        if probs.shape[1] > 1:
            ent /= np.log(probs.shape[1])
        return ent.astype(np.float32)


# ── FAR-MoE ───────────────────────────────────────────────────────────────────
class FARMoE:
    def __init__(
        self,
        expert_epochs:  int          = 30,
        expert_lr:      float        = 0.1,
        expert_batch:   int          = 1024,
        conf_threshold: float        = 0.05,
        tail_n:         int          = 50,
        tail_boost:     float        = 5.0,
        ood_threshold:  float        = 3.0,
        router_C:       float        = 1.0,
        device:         torch.device = None,
        seed:           int          = 42,
    ):
        self.expert_epochs  = expert_epochs
        self.expert_lr      = expert_lr
        self.expert_batch   = expert_batch
        self.conf_threshold = conf_threshold
        self.tail_n         = tail_n
        self.tail_boost     = tail_boost
        self.ood_threshold  = ood_threshold
        self.router_C       = router_C
        self.device         = device or torch.device("cpu")
        self.seed           = seed

        self.experts_:        List[Expert]                       = []
        self.routers_:        List[Optional[LogisticRegression]] = []
        self.ood_val_stats_:  List[Tuple[float, float]]          = []

    def _find_conf_clusters(self, y_true, y_pred, n_classes):
        cm      = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_norm = cm / row_sum
        uf = UnionFind(n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm_norm[i, j] > self.conf_threshold:
                    uf.union(i, j)
        return [g for g in uf.groups() if len(g) >= 2]

    def fit(self, X_tr, y_tr, X_val, y_val, log) -> "FARMoE":
        n_classes = int(y_tr.max()) + 1
        n_feat    = X_tr.shape[1]
        half      = n_feat // 2

        log.info("  [Phase 1] Probe training (linear head, 10 epochs) ...")
        probe = nn.Linear(X_tr.shape[1], n_classes).to(self.device)
        p_opt = optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        p_sch = optim.lr_scheduler.CosineAnnealingLR(p_opt, T_max=10)
        X_t_p = torch.tensor(X_tr, dtype=torch.float32)
        y_t_p = torch.tensor(y_tr, dtype=torch.long)
        n_p   = len(y_t_p)
        probe.train()
        for _ in range(10):
            perm = torch.randperm(n_p)
            for i in range(0, n_p, 1024):
                idx = perm[i:i + 1024]
                p_opt.zero_grad()
                F.cross_entropy(probe(X_t_p[idx].to(self.device)),
                                y_t_p[idx].to(self.device)).backward()
                p_opt.step()
            p_sch.step()
        probe.eval()
        with torch.no_grad():
            y_val_pred = probe(
                torch.tensor(X_val, dtype=torch.float32).to(self.device)
            ).argmax(1).cpu().numpy()
        clusters = self._find_conf_clusters(y_val, y_val_pred, n_classes)
        log.info(f"  [Phase 1] {len(clusters)} cluster(s): {[len(c) for c in clusters]}")

        counts      = np.bincount(y_tr, minlength=n_classes)
        tail_cls    = [c for c in range(n_classes) if 0 < counts[c] < self.tail_n]
        log.info(f"  [Phase 2] Tail classes (< {self.tail_n}): {len(tail_cls)}")

        specs = [dict(name="anchor", feat_idx=None, target_cls=None,
                      target_boost=1.0, balanced=False, always_include=True)]
        for ci, cl in enumerate(clusters):
            specs.append(dict(name=f"conf_{ci}", feat_idx=None,
                              target_cls=list(cl), target_boost=5.0,
                              balanced=False, always_include=False))
        if tail_cls:
            specs.append(dict(name="tail", feat_idx=None,
                              target_cls=tail_cls, target_boost=self.tail_boost,
                              balanced=True, always_include=False))
        specs.append(dict(name="col_first",  feat_idx=np.arange(half),
                          target_cls=None, target_boost=1.0,
                          balanced=False, always_include=False))
        specs.append(dict(name="col_second", feat_idx=np.arange(half, n_feat),
                          target_cls=None, target_boost=1.0,
                          balanced=False, always_include=False))

        self.experts_ = []
        for sp in specs:
            exp = Expert(n_global=n_classes, device=self.device, seed=self.seed, **sp)
            log.info(f"  [Phase 3] Training '{sp['name']}' ...")
            exp.fit(X_tr, y_tr, log,
                    epochs=self.expert_epochs, lr=self.expert_lr, batch_size=self.expert_batch)
            self.experts_.append(exp)

        log.info("  [Phase 4] Fitting routers ...")
        self.routers_       = []
        self.ood_val_stats_ = []
        for exp in self.experts_:
            if not exp.is_trained:
                self.routers_.append(None)
                self.ood_val_stats_.append((0.0, 1.0))
                continue
            y_hat   = np.argmax(exp.predict_proba_global(X_val), axis=1)
            correct = (y_hat == y_val).astype(int)
            ood     = exp.ood_scores(X_val)
            om, os_ = float(np.mean(ood)), float(np.std(ood) + 1e-6)
            self.ood_val_stats_.append((om, os_))
            ood_z   = np.clip((ood - om) / os_, -8, 8)
            feats   = np.column_stack([exp.margin(X_val), exp.entropy(X_val), ood_z])
            if correct.sum() in (0, len(correct)):
                self.routers_.append(None)
            else:
                lr = LogisticRegression(C=self.router_C, max_iter=300,
                                        random_state=self.seed, n_jobs=-1)
                lr.fit(feats, correct)
                self.routers_.append(lr)
            log.info(f"    '{exp.name:12s}' val_acc={correct.mean():.4f} "
                     f"ood_mean={om:.1f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_classes = self.experts_[0].n_global
        N         = len(X)
        alpha     = np.zeros((len(self.experts_), N), dtype=np.float64)

        for ei, (exp, lr, (om, os_)) in enumerate(
                zip(self.experts_, self.routers_, self.ood_val_stats_)):
            if not exp.is_trained:
                continue
            ood_z = np.clip((exp.ood_scores(X) - om) / os_, -8, 8)
            if lr is None:
                a = np.full(N, 0.5)
            else:
                feats = np.column_stack([exp.margin(X), exp.entropy(X), ood_z])
                a     = lr.predict_proba(feats)[:, 1]
            if not exp.always_include:
                a = a * (ood_z <= self.ood_threshold)
            alpha[ei] = a

        s = alpha.sum(axis=0, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        alpha /= s

        out = np.zeros((N, n_classes), dtype=np.float64)
        for ei, exp in enumerate(self.experts_):
            out += exp.predict_proba_global(X).astype(np.float64) * alpha[ei, :, None]
        return out.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ── ResNet50 backbone ─────────────────────────────────────────────────────────
FEAT_DIM_RN50 = 2048

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 64×64 source images: resize up to 224 for ResNet50 compatibility
TRANSFORM_TRAIN = T.Compose([
    T.Resize(72),                          # slight upscale before crop
    T.RandomCrop(64),
    T.Resize(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TRANSFORM_EVAL = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_resnet50_backbone(freeze_backbone: bool = False) -> nn.Module:
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


# ── Downsampled ImageNet 64×64 dataset ───────────────────────────────────────
class MiniImageNetDataset(Dataset):
    """Dataset wrapping pre-loaded numpy arrays from the 64×64 pickle batches."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform):
        # data shape: (N, 12288) uint8, stored as R-channel-major (CHW flattened)
        self.data      = data      # (N, 12288) uint8
        self.labels    = labels    # (N,) int, 0-indexed
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        # Batch format: R-plane(64*64) | G-plane(64*64) | B-plane(64*64)
        img = self.data[i].reshape(3, 64, 64).transpose(1, 2, 0)  # HWC uint8
        img = Image.fromarray(img)
        return self.transform(img), int(self.labels[i])


def load_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single pickle batch; return (data uint8, labels 0-indexed)."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data   = d[b"data"] if b"data" in d else d["data"]
    labels = d[b"labels"] if b"labels" in d else d["labels"]
    labels = np.array(labels, dtype=np.int32) - 1   # 1-1000 → 0-999
    return data.astype(np.uint8), labels


def apply_lt_sampling(
    data: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    mu: float = 0.99,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample to create long-tail distribution.
    Class i keeps fraction mu^i of its samples (sorted by current count desc).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    counts  = np.bincount(labels, minlength=n_classes)
    order   = np.argsort(-counts)        # most frequent class first
    keep_idx = []
    for rank, cls in enumerate(order):
        idx  = np.where(labels == cls)[0]
        keep = max(1, int(len(idx) * (mu ** rank)))
        keep_idx.append(rng.choice(idx, size=keep, replace=False))
    keep_idx = np.concatenate(keep_idx)
    keep_idx = np.sort(keep_idx)
    return data[keep_idx], labels[keep_idx]


def load_mini_imagenet(
    args: argparse.Namespace,
    log:  logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    data_root = Path(args.data_root)
    rng       = np.random.default_rng(args.seed)

    # ── Load training batches ─────────────────────────────────────────────────
    log.info("  Loading training batches ...")
    tr_data_parts, tr_lbl_parts = [], []
    for i in range(1, 6):
        p = data_root / f"train_data_batch_{i}"
        if not p.exists():
            log.warning(f"  Batch {p} not found; skipping.")
            continue
        d, l = load_batch(p)
        tr_data_parts.append(d)
        tr_lbl_parts.append(l)
        log.info(f"  Loaded batch {i}: {len(l)} samples")
    tr_data = np.concatenate(tr_data_parts, axis=0)
    tr_lbls = np.concatenate(tr_lbl_parts, axis=0)

    # ── Load val batches (used as held-out test set) ───────────────────────────
    log.info("  Loading val batch ...")
    te_data, te_lbls = load_batch(data_root / "val_data")
    log.info(f"  Loaded val: {len(te_lbls)} samples")

    n_classes = 1000

    # ── Subset to max_classes ─────────────────────────────────────────────────
    if args.max_classes and args.max_classes < n_classes:
        keep_cls = np.arange(args.max_classes)
        tr_mask  = np.isin(tr_lbls, keep_cls)
        te_mask  = np.isin(te_lbls, keep_cls)
        tr_data, tr_lbls = tr_data[tr_mask], tr_lbls[tr_mask]
        te_data, te_lbls = te_data[te_mask], te_lbls[te_mask]
        n_classes = args.max_classes
        log.info(f"  Subsetted to {n_classes} classes: "
                 f"train={len(tr_lbls)} test={len(te_lbls)}")

    # ── Optional LT subsampling on training set ───────────────────────────────
    if args.lt_sampling:
        log.info(f"  Applying LT sampling (mu={args.lt_mu}) ...")
        tr_data, tr_lbls = apply_lt_sampling(tr_data, tr_lbls, n_classes,
                                             mu=args.lt_mu, rng=rng)
        counts = np.bincount(tr_lbls, minlength=n_classes)
        log.info(f"  After LT: train={len(tr_lbls)}  "
                 f"max={counts.max()} min={counts[counts>0].min()}")

    # ── Carve out validation split from training set ──────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, va_idx = next(sss.split(tr_data, tr_lbls))
    va_data, va_lbls = tr_data[va_idx], tr_lbls[va_idx]
    tr_data, tr_lbls = tr_data[tr_idx], tr_lbls[tr_idx]

    log.info(f"  Split: train={len(tr_lbls)} val={len(va_lbls)} test={len(te_lbls)}")
    counts = np.bincount(tr_lbls, minlength=n_classes)
    log.info(f"  Train: max={counts.max()} min={counts[counts>0].min()} "
             f"classes={len(counts[counts>0])}")

    # ── Backbone setup ────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    log.info(f"  Backbone device: {device}")

    nw = min(4, os.cpu_count() or 1)
    tr_ds = MiniImageNetDataset(tr_data, tr_lbls, TRANSFORM_TRAIN)
    va_ds = MiniImageNetDataset(va_data, va_lbls, TRANSFORM_EVAL)
    te_ds = MiniImageNetDataset(te_data, te_lbls, TRANSFORM_EVAL)
    tr_loader = DataLoader(tr_ds, batch_size=args.backbone_batch_size,
                           shuffle=True,  num_workers=nw, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.backbone_batch_size,
                           shuffle=False, num_workers=nw, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=args.backbone_batch_size,
                           shuffle=False, num_workers=nw, pin_memory=True)

    log.info(f"  Building ResNet50 backbone "
             f"(freeze={args.freeze_backbone}, ft_epochs={args.backbone_epochs}) ...")
    model = build_resnet50_backbone(freeze_backbone=args.freeze_backbone)

    if args.backbone_epochs > 0:
        t0    = time.time()
        model, head = finetune_backbone(model, tr_loader, device, n_classes,
                                        epochs=args.backbone_epochs,
                                        lr=args.backbone_lr, log=log)
        log.info(f"  Fine-tuning done in {time.time()-t0:.1f}s")

    # Use eval-transform loader for feature extraction from training set too
    tr_ds_ev    = MiniImageNetDataset(tr_data, tr_lbls, TRANSFORM_EVAL)
    tr_loader_ev = DataLoader(tr_ds_ev, batch_size=args.backbone_batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True)

    log.info("  Extracting features ...")
    X_tr,   y_tr   = extract_features_rn50(model, tr_loader_ev, device)
    X_val,  y_val  = extract_features_rn50(model, va_loader,    device)
    X_test, y_test = extract_features_rn50(model, te_loader,    device)

    log.info(f"  Feature dim={X_tr.shape[1]}  "
             f"train={len(y_tr)} val={len(y_val)} test={len(y_test)}")

    if args.backbone_epochs == 0:
        log.info("  Fitting linear probe on frozen features ...")
        head = fit_linear_probe(X_tr, y_tr, n_classes, device, log)

    log.info("  Computing linear-head baseline predictions ...")
    y_base_pred = eval_linear_head(head, X_test, device)
    return X_tr, y_tr, X_val, y_val, X_test, y_test, y_base_pred


# ── Backbone fine-tuning ──────────────────────────────────────────────────────
def finetune_backbone(
    model:        nn.Module,
    train_loader: DataLoader,
    device:       torch.device,
    n_classes:    int,
    epochs:       int,
    lr:           float,
    log:          logging.Logger,
) -> nn.Module:
    model.to(device)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    head = nn.Linear(FEAT_DIM_RN50, n_classes).to(device)
    model.train()
    head.train()
    for epoch in range(epochs):
        tot_loss = correct = total = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            feat = model(Xb)
            loss = criterion(head(feat), yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(yb)
            correct  += (head(feat).argmax(1) == yb).sum().item()
            total    += len(yb)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(f"    Fine-tune epoch {epoch+1:3d}/{epochs}: "
                     f"loss={tot_loss/total:.4f}  acc={correct/total:.4f}  "
                     f"lr={scheduler.get_last_lr()[0]:.6f}")
    return model, head


def extract_features_rn50(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats_list, lbls_list = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            feats_list.append(model(Xb.to(device)).cpu().numpy())
            lbls_list.append(yb.numpy())
    return (np.concatenate(feats_list, axis=0).astype(np.float32),
            np.concatenate(lbls_list, axis=0))


def fit_linear_probe(
    X_tr:      np.ndarray,
    y_tr:      np.ndarray,
    n_classes: int,
    device:    torch.device,
    log:       logging.Logger,
    epochs:    int = 30,
    lr:        float = 0.1,
    batch_size: int = 1024,
) -> nn.Linear:
    """Train a linear head on pre-extracted features with CE loss."""
    head      = nn.Linear(X_tr.shape[1], n_classes).to(device)
    optimizer = optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.long)
    n   = len(y_t)
    head.train()
    for epoch in range(epochs):
        perm  = torch.randperm(n)
        X_sh, y_sh = X_t[perm], y_t[perm]
        tot_loss = correct = 0
        for i in range(0, n, batch_size):
            Xb = X_sh[i:i + batch_size].to(device)
            yb = y_sh[i:i + batch_size].to(device)
            optimizer.zero_grad()
            loss = criterion(head(Xb), yb)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tot_loss += loss.item() * len(yb)
                correct  += (head(Xb).argmax(1) == yb).sum().item()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"    [probe] epoch {epoch+1:3d}/{epochs}: "
                     f"loss={tot_loss/n:.4f}  acc={correct/n:.4f}")
    return head


def eval_linear_head(
    head:       nn.Linear,
    X:          np.ndarray,
    device:     torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Apply linear head to pre-extracted features → class predictions."""
    head.eval()
    preds = []
    X_t   = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = head(X_t[i:i + batch_size].to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ── Experiment runner ─────────────────────────────────────────────────────────
def run_experiment(
    name:        str,
    X_tr:        np.ndarray, y_tr:   np.ndarray,
    X_val:       np.ndarray, y_val:  np.ndarray,
    X_test:      np.ndarray, y_test: np.ndarray,
    out_dir:     Path,
    args:        argparse.Namespace,
    log:         logging.Logger,
    y_base_pred: np.ndarray = None,
) -> dict:
    n_classes = int(np.concatenate([y_tr, y_val, y_test]).max()) + 1
    log.info(f"\n{'='*60}")
    log.info(f"Dataset : {name}  classes={n_classes}")
    log.info(f"Shapes  : train={X_tr.shape} val={X_val.shape} test={X_test.shape}")
    tr_counts = np.bincount(y_tr, minlength=n_classes)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # ── Baseline: ResNet50 linear head (CE-trained) ───────────────────────────
    log.info("[Baseline] ResNet50 linear head (CE-trained) ...")
    y_base = y_base_pred
    rep_b  = classification_report(y_test, y_base, output_dict=True, zero_division=0)
    log.info(f"[Baseline] macro-F1={rep_b['macro avg']['f1-score']:.4f}  "
             f"wt-F1={rep_b['weighted avg']['f1-score']:.4f}")

    # ── FAR-MoE ───────────────────────────────────────────────────────────────
    log.info("[FAR-MoE] Training ...")
    t0  = time.time()
    moe = FARMoE(
        expert_epochs=args.expert_epochs,
        expert_lr=args.expert_lr,
        expert_batch=args.expert_batch,
        conf_threshold=args.conf_threshold,
        tail_n=args.tail_n,
        tail_boost=args.tail_boost,
        ood_threshold=args.ood_threshold,
        router_C=args.router_C,
        device=device,
        seed=args.seed,
    )
    moe.fit(X_tr, y_tr, X_val, y_val, log)
    log.info(f"[FAR-MoE] {time.time()-t0:.1f}s")
    y_moe = moe.predict(X_test)
    rep_m = classification_report(y_test, y_moe, output_dict=True, zero_division=0)
    log.info(f"[FAR-MoE] macro-F1={rep_m['macro avg']['f1-score']:.4f}  "
             f"wt-F1={rep_m['weighted avg']['f1-score']:.4f}")

    delta = rep_m['macro avg']['f1-score'] - rep_b['macro avg']['f1-score']
    sign  = "▲" if delta > EPS else ("▼" if delta < -EPS else "–")
    log.info(f"[Delta] {delta:+.4f} {sign}")

    # ── Per-class table ───────────────────────────────────────────────────────
    all_cls = sorted(set(int(c) for c in np.unique(np.concatenate([y_tr, y_test]))))
    rows = []
    for c in all_cls:
        k  = str(c)
        bm = rep_b.get(k, {})
        dm = rep_m.get(k, {})
        rows.append({
            "class":     str(c),
            "support":   int(tr_counts[c]) if c < len(tr_counts) else 0,
            "prec(B)":   float(bm.get("precision", 0.0)),
            "prec(M)":   float(dm.get("precision", 0.0)),
            "recall(B)": float(bm.get("recall", 0.0)),
            "recall(M)": float(dm.get("recall", 0.0)),
            "f1(B)":     float(bm.get("f1-score", 0.0)),
            "f1(M)":     float(dm.get("f1-score", 0.0)),
            "delta_f1":  float(dm.get("f1-score", 0.0)) - float(bm.get("f1-score", 0.0)),
        })
    rows.sort(key=lambda r: r["support"], reverse=True)

    for avg in ("macro avg", "weighted avg"):
        bm = rep_b.get(avg, {})
        dm = rep_m.get(avg, {})
        rows.append({
            "class": avg, "support": int(bm.get("support", 0)),
            "prec(B)": float(bm.get("precision", 0.0)),
            "prec(M)": float(dm.get("precision", 0.0)),
            "recall(B)": float(bm.get("recall", 0.0)),
            "recall(M)": float(dm.get("recall", 0.0)),
            "f1(B)": float(bm.get("f1-score", 0.0)),
            "f1(M)": float(dm.get("f1-score", 0.0)),
            "delta_f1": float(dm.get("f1-score", 0.0)) - float(bm.get("f1-score", 0.0)),
            "_footer": True,
        })

    safe = name.replace("/", "_").replace(" ", "_")
    save_colored_table(rows, COL_HEADERS, out_dir / f"{safe}_comparison.png",
                       title=f"FAR-MoE vs Baseline — {name}")
    pd.DataFrame(rows).to_csv(out_dir / f"{safe}_comparison.csv", index=False)

    exp_rows = [{"name": e.name, "is_trained": e.is_trained,
                 "always_include": e.always_include,
                 "n_target_cls": len(e.target_cls) if e.target_cls else n_classes,
                 "feat_dim": e.feat_idx.shape[0] if e.feat_idx is not None else X_tr.shape[1]}
                for e in moe.experts_]
    pd.DataFrame(exp_rows).to_csv(out_dir / f"{safe}_experts.csv", index=False)

    return {
        "dataset": name, "n_classes": n_classes, "n_experts": len(moe.experts_),
        "macro_f1_base": rep_b['macro avg']['f1-score'],
        "macro_f1_moe":  rep_m['macro avg']['f1-score'],
        "delta_macro_f1": delta,
        "wt_f1_base":    rep_b['weighted avg']['f1-score'],
        "wt_f1_moe":     rep_m['weighted avg']['f1-score'],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FAR-MoE on Downsampled ImageNet 64×64")
    p.add_argument("--data_root",           default="data/imagenet",
                   help="Dir containing train_data_batch_* and val_data")
    # ── class subsetting ──────────────────────────────────────────────────────
    p.add_argument("--max_classes",         type=int,   default=None,
                   help="Use only first N classes (None = all 1000); good for quick tests")
    # ── LT sampling ───────────────────────────────────────────────────────────
    p.add_argument("--lt_sampling",         action="store_true",
                   help="Apply exponential LT subsampling to training set")
    p.add_argument("--lt_mu",               type=float, default=0.99,
                   help="LT decay factor mu (fraction kept = mu^rank). "
                        "0.99 ≈ mild, 0.95 ≈ strong")
    # ── backbone ──────────────────────────────────────────────────────────────
    p.add_argument("--backbone_epochs",     type=int,   default=5,
                   help="Fine-tuning epochs for ResNet50 (0 = frozen linear probe)")
    p.add_argument("--backbone_lr",         type=float, default=0.01)
    p.add_argument("--backbone_batch_size", type=int,   default=128)
    p.add_argument("--freeze_backbone",     action="store_true",
                   help="Freeze backbone weights; only update the classification head")
    # ── Expert / FAR-MoE ─────────────────────────────────────────────────────
    p.add_argument("--expert_epochs",       type=int,   default=30,
                   help="Epochs to train each expert linear head (default 30)")
    p.add_argument("--expert_lr",           type=float, default=0.1,
                   help="LR for expert SGD with cosine decay (default 0.1)")
    p.add_argument("--expert_batch",        type=int,   default=1024,
                   help="Batch size for expert training (default 1024)")
    p.add_argument("--conf_threshold",      type=float, default=0.05)
    p.add_argument("--tail_n",              type=int,   default=20)
    p.add_argument("--tail_boost",          type=float, default=5.0)
    p.add_argument("--ood_threshold",       type=float, default=3.0)
    p.add_argument("--router_C",            type=float, default=1.0)
    p.add_argument("--device",              default="cuda",
                   choices=["cpu", "cuda"])
    p.add_argument("--seed",                type=int,   default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"farmoemini_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "experiment.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger(__name__)
    log.info(f"Output dir : {out_dir}")
    log.info(f"Args       : {vars(args)}")

    X_tr, y_tr, X_val, y_val, X_test, y_test, y_base_pred = load_mini_imagenet(args, log)
    sr = run_experiment("MiniImageNet-LT", X_tr, y_tr, X_val, y_val,
                        X_test, y_test, out_dir, args, log, y_base_pred)

    log.info(f"\n{'='*60}")
    log.info(f"macro-F1  baseline={sr['macro_f1_base']:.4f}  "
             f"FAR-MoE={sr['macro_f1_moe']:.4f}  "
             f"delta={sr['delta_macro_f1']:+.4f}")
    pd.DataFrame([sr]).to_csv(out_dir / "summary.csv", index=False)
    log.info(f"Results → {out_dir}/")


if __name__ == "__main__":
    main()

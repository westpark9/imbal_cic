#!/usr/bin/env python3
"""
code_mat_cifar.py  –  FAR-MoE on CIFAR-10/100 Long-Tail (ResNet32 backbone)
=============================================================================
Pipeline: ResNet32 backbone (trained from scratch on LT data) → 64-dim features
          → CE-trained linear FAR-MoE ensemble.

Backbone: ResNet-32 (He et al., 2016 CIFAR variant)
  - 3 stages × 5 BasicBlocks, feature dim = 64 after global avg pool
  - Trained from scratch with SGD + cosine LR on the LT training split
  - Standard CIFAR augmentation: RandomCrop(32,pad=4) + RandomHorizontalFlip

Expert roster
-------------
  Anchor        : all classes, uniform weights          (always included)
  ConfMat_k     : one per confusion cluster from probe  (OOD-gated)
  Tail          : upweighted tail classes               (OOD-gated)
  ColSplit_0/1  : first-half / second-half feature dims (OOD-gated)

Usage
-----
  python src/code_mat_cifar.py                           # CIFAR-10+100, IR=100
  python src/code_mat_cifar.py --datasets cifar10        # CIFAR-10 only
  python src/code_mat_cifar.py --imbalance_ratio 50      # IR=50
  python src/code_mat_cifar.py --backbone_epochs 200 --device cuda
"""

import argparse
import datetime
import logging
import os
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

import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

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


# ── ResNet32 backbone (He et al. 2016, CIFAR variant) ────────────────────────
class _BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


def _make_stage(in_c: int, out_c: int, n_blocks: int, stride: int) -> nn.Sequential:
    layers = [_BasicBlock(in_c, out_c, stride)]
    for _ in range(1, n_blocks):
        layers.append(_BasicBlock(out_c, out_c, 1))
    return nn.Sequential(*layers)


class ResNet32(nn.Module):
    """CIFAR ResNet-32: 3 stages × 5 BasicBlocks + stem. Feature dim = 64."""
    FEAT_DIM = 64

    def __init__(self, num_classes: int = 10):
        super().__init__()
        n = 5  # (32-2) / 6 = 5 blocks per stage
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.layer1 = _make_stage(16, 16, n, stride=1)
        self.layer2 = _make_stage(16, 32, n, stride=2)
        self.layer3 = _make_stage(32, 64, n, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        feat = x.view(x.size(0), -1)
        return feat if return_features else self.fc(feat)


# ── Backbone training + feature extraction ────────────────────────────────────
class _NumpyImageDataset(Dataset):
    """Wraps (N, H, W, C) uint8 numpy arrays for DataLoader with transforms."""
    def __init__(self, imgs: np.ndarray, labels: Optional[np.ndarray], transform):
        self.imgs      = imgs
        self.labels    = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        img = self.transform(Image.fromarray(self.imgs[i]))
        if self.labels is not None:
            return img, int(self.labels[i])
        return img


def train_backbone(
    model:      nn.Module,
    X_imgs:     np.ndarray,  # (N, 32, 32, 3) uint8
    y:          np.ndarray,
    mean:       tuple,
    std:        tuple,
    device:     torch.device,
    epochs:     int,
    lr:         float,
    batch_size: int,
    log:        logging.Logger,
) -> nn.Module:
    """Train backbone on CIFAR LT training images with standard augmentation."""
    transform_tr = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    ds     = _NumpyImageDataset(X_imgs, y, transform_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=device.type == "cuda")

    model.to(device)
    optimizer  = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = correct = total = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(yb)
        scheduler.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            log.info(f"    Backbone epoch {epoch+1:3d}/{epochs}: "
                     f"loss={total_loss/total:.4f}  acc={correct/total:.4f}  "
                     f"lr={scheduler.get_last_lr()[0]:.5f}")
    return model


def extract_features(
    model:      nn.Module,
    X_imgs:     np.ndarray,  # (N, H, W, C) uint8
    mean:       tuple,
    std:        tuple,
    device:     torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Return (N, feat_dim) float32 penultimate-layer features."""
    transform_ev = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds     = _NumpyImageDataset(X_imgs, None, transform_ev)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=device.type == "cuda")
    model.eval()
    parts = []
    with torch.no_grad():
        for Xb in loader:
            parts.append(model(Xb.to(device), return_features=True).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def eval_fc_head(
    model:      nn.Module,
    X_feats:    np.ndarray,
    device:     torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Apply backbone FC head to pre-extracted features → class predictions."""
    model.eval()
    preds = []
    X_t   = torch.tensor(X_feats, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = model.fc(X_t[i:i + batch_size].to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ── Union-Find for confusion clustering ──────────────────────────────────────
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


# ── OOD detection (ported from code_mat.py) ──────────────────────────────────
class MahalanobisOOD:
    """Mahalanobis-distance OOD scorer using LedoitWolf covariance."""

    def __init__(self, max_fit_samples: int = 10000, random_state: int = 42):
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
        self.mean_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.mode: Optional[str] = None

    def fit(self, X: np.ndarray) -> "MahalanobisOOD":
        X = np.asarray(X, dtype=np.float32)
        if len(X) > self.max_fit_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), size=self.max_fit_samples, replace=False)
            X = X[idx]
        try:
            lw = LedoitWolf().fit(X)
            self.mean_ = lw.location_.astype(np.float32)
            self.precision_ = lw.precision_.astype(np.float32)
            self.mode = "full"
        except Exception:
            self.mean_ = np.mean(X, axis=0).astype(np.float32)
            self.var_ = (np.var(X, axis=0) + 1e-6).astype(np.float32)
            self.mode = "diag"
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.mode == "full":
            diff = X - self.mean_
            dist2 = np.einsum("bi,ij,bj->b", diff, self.precision_, diff, optimize=True)
        else:
            diff = X - self.mean_
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


# ── FAR-MoE for CIFAR ─────────────────────────────────────────────────────────
class FARMoECifar:
    """
    FAR-MoE adapted for image PCA features (CIFAR-10/100 long-tail).

    Differences from code_mat.py:
      - No network-traffic taxonomy; uses confusion-matrix clusters + column splits
      - Column-split experts replace volume/timing/packet/tcp feature views
      - LT tail expert upweights classes below tail_n train-sample threshold
    """

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

        self.experts_:       List[Expert]                       = []
        self.routers_:       List[Optional[LogisticRegression]] = []
        self.ood_val_stats_: List[Tuple[float, float]]          = []

    def _find_conf_clusters(self, y_true: np.ndarray, y_pred: np.ndarray,
                             n_classes: int) -> List[List[int]]:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_norm = cm / row_sum
        uf = UnionFind(n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm_norm[i, j] > self.conf_threshold:
                    uf.union(i, j)
        return [g for g in uf.groups() if len(g) >= 2]

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            log: logging.Logger) -> "FARMoECifar":
        n_classes = int(y_tr.max()) + 1
        n_feat = X_tr.shape[1]
        half = n_feat // 2
        col_first  = np.arange(half, dtype=int)
        col_second = np.arange(half, n_feat, dtype=int)

        # ── Phase 1: Probe → confusion clusters ───────────────────────────────
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
        log.info(f"  [Phase 1] {len(clusters)} confusion cluster(s): "
                 f"sizes {[len(c) for c in clusters]}")

        # ── Phase 2: Identify tail classes ────────────────────────────────────
        counts = np.bincount(y_tr, minlength=n_classes)
        tail_classes = [c for c in range(n_classes) if 0 < counts[c] < self.tail_n]
        log.info(f"  [Phase 2] Tail classes (< {self.tail_n} train samples): "
                 f"{len(tail_classes)} classes")

        # ── Phase 3: Build and train experts ──────────────────────────────────
        expert_specs = []
        expert_specs.append(dict(name="anchor", feat_idx=None, target_cls=None,
                                  target_boost=1.0, balanced=False, always_include=True))
        for ci, cluster in enumerate(clusters):
            expert_specs.append(dict(name=f"conf_{ci}", feat_idx=None,
                                      target_cls=list(cluster), target_boost=5.0,
                                      balanced=False, always_include=False))
        if tail_classes:
            expert_specs.append(dict(name="tail", feat_idx=None,
                                      target_cls=tail_classes, target_boost=self.tail_boost,
                                      balanced=True, always_include=False))
        expert_specs.append(dict(name="col_first", feat_idx=col_first,
                                  target_cls=None, target_boost=1.0,
                                  balanced=False, always_include=False))
        expert_specs.append(dict(name="col_second", feat_idx=col_second,
                                  target_cls=None, target_boost=1.0,
                                  balanced=False, always_include=False))

        self.experts_ = []
        for spec in expert_specs:
            exp = Expert(n_global=n_classes, device=self.device, seed=self.seed, **spec)
            log.info(f"  [Phase 3] Training expert '{spec['name']}' ...")
            exp.fit(X_tr, y_tr, log,
                    epochs=self.expert_epochs, lr=self.expert_lr, batch_size=self.expert_batch)
            self.experts_.append(exp)
        log.info(f"  [Phase 3] {len(self.experts_)} experts trained.")

        # ── Phase 4: Fit competence routers on val ────────────────────────────
        log.info("  [Phase 4] Fitting competence routers on val ...")
        self.routers_ = []
        self.ood_val_stats_ = []

        for exp in self.experts_:
            if not exp.is_trained:
                self.routers_.append(None)
                self.ood_val_stats_.append((0.0, 1.0))
                continue

            y_hat = np.argmax(exp.predict_proba_global(X_val), axis=1)
            correct = (y_hat == y_val).astype(int)

            ood_scores = exp.ood_scores(X_val)
            ood_mean = float(np.mean(ood_scores))
            ood_std  = float(np.std(ood_scores) + 1e-6)
            self.ood_val_stats_.append((ood_mean, ood_std))
            ood_z = np.clip((ood_scores - ood_mean) / ood_std, -8, 8)

            margin  = exp.margin(X_val)
            entropy = exp.entropy(X_val)
            feats = np.column_stack([margin, entropy, ood_z])

            if correct.sum() == 0 or correct.sum() == len(correct):
                self.routers_.append(None)
            else:
                lr = LogisticRegression(C=self.router_C, max_iter=300,
                                        random_state=self.seed, n_jobs=-1)
                lr.fit(feats, correct)
                self.routers_.append(lr)

            log.info(f"    Expert '{exp.name:12s}': val acc={correct.mean():.4f}  "
                     f"OOD mean={ood_mean:.1f} std={ood_std:.1f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_classes = self.experts_[0].n_global
        N = len(X)
        alpha = np.zeros((len(self.experts_), N), dtype=np.float64)

        for ei, (exp, lr, (ood_mean, ood_std)) in enumerate(
                zip(self.experts_, self.routers_, self.ood_val_stats_)):
            if not exp.is_trained:
                continue

            ood_scores = exp.ood_scores(X)
            ood_z = np.clip((ood_scores - ood_mean) / ood_std, -8, 8)

            if lr is None:
                a = np.full(N, 0.5)
            else:
                margin  = exp.margin(X)
                entropy = exp.entropy(X)
                feats = np.column_stack([margin, entropy, ood_z])
                a = lr.predict_proba(feats)[:, 1]

            # OOD gate: zero out contribution for out-of-distribution samples
            if not exp.always_include:
                a = a * (ood_z <= self.ood_threshold).astype(np.float64)

            alpha[ei] = a

        # Normalize across experts per sample
        alpha_sum = alpha.sum(axis=0, keepdims=True)
        alpha_sum = np.where(alpha_sum == 0, 1.0, alpha_sum)
        alpha_norm = alpha / alpha_sum  # (E, N)

        # Weighted average of expert global probabilities
        proba_sum = np.zeros((N, n_classes), dtype=np.float64)
        for ei, exp in enumerate(self.experts_):
            p = exp.predict_proba_global(X).astype(np.float64)
            proba_sum += p * alpha_norm[ei, :, np.newaxis]

        return proba_sum.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ── Data loading with long-tail option ───────────────────────────────────────
def apply_long_tail(
    X: np.ndarray, y: np.ndarray,
    imbalance_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Exponential-decay subsampling to create LT distribution."""
    rng = np.random.default_rng(seed)
    classes = np.sort(np.unique(y))
    K = len(classes)
    counts = np.bincount(y, minlength=K)
    n_max = int(counts.max())

    keep_idx = []
    for i, c in enumerate(classes):
        # n_i = n_max * IR^(-i/(K-1))
        n_target = max(int(n_max / (imbalance_ratio ** (i / max(K - 1, 1)))), 1)
        cls_idx = np.where(y == c)[0]
        n_take = min(n_target, len(cls_idx))
        keep_idx.append(rng.choice(cls_idx, size=n_take, replace=False))

    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


def load_cifar_lt(
    kind:             str,
    imbalance_ratio:  float,
    seed:             int,
    args:             argparse.Namespace,
    log:              logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, list]:
    """
    Load CIFAR-10/100, apply LT subsampling, train ResNet32 backbone,
    extract 64-dim features. Test set is the standard balanced CIFAR test split.
    """
    root   = str(Path(__file__).parent.parent / "data" / "cifar")
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    log.info(f"  Backbone device: {device}")

    if kind == "cifar10":
        train_ds = CIFAR10(root, train=True,  download=True)
        test_ds  = CIFAR10(root, train=False, download=True)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    elif kind == "cifar100":
        train_ds = CIFAR100(root, train=True,  download=True)
        test_ds  = CIFAR100(root, train=False, download=True)
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unknown dataset: {kind}")

    # Raw image arrays: (N, 32, 32, 3) uint8
    imgs_all = np.array(train_ds.data)
    y_all    = np.array(train_ds.targets)
    imgs_tst = np.array(test_ds.data)
    y_test   = np.array(test_ds.targets)

    # Apply LT to training pool only
    if imbalance_ratio > 1:
        imgs_all, y_all = apply_long_tail(imgs_all, y_all, imbalance_ratio, seed)
        c = np.bincount(y_all)
        log.info(f"  LT applied (IR={imbalance_ratio:.0f}): "
                 f"max={c.max()} min={c[c>0].min()} total={c.sum()}")

    # Stratified 75/25 train/val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_i, va_i = next(sss.split(imgs_all, y_all))
    imgs_tr, y_tr   = imgs_all[tr_i], y_all[tr_i]
    imgs_val, y_val = imgs_all[va_i], y_all[va_i]

    log.info(f"  {kind.upper()}: train={len(y_tr)} val={len(y_val)} test={len(y_test)}")

    # ── Train ResNet32 backbone ────────────────────────────────────────────────
    log.info(f"  Training ResNet32 backbone (epochs={args.backbone_epochs}, "
             f"lr={args.backbone_lr}, bs={args.backbone_batch_size}) ...")
    t0    = time.time()
    model = ResNet32(num_classes=len(train_ds.classes))
    model = train_backbone(model, imgs_tr, y_tr, mean, std, device,
                           epochs=args.backbone_epochs,
                           lr=args.backbone_lr,
                           batch_size=args.backbone_batch_size,
                           log=log)
    log.info(f"  Backbone trained in {time.time()-t0:.1f}s")

    # ── Extract features (no augmentation) ────────────────────────────────────
    log.info("  Extracting features ...")
    X_tr   = extract_features(model, imgs_tr,   mean, std, device)
    X_val  = extract_features(model, imgs_val,  mean, std, device)
    X_test = extract_features(model, imgs_tst,  mean, std, device)

    log.info(f"  Feature dim={X_tr.shape[1]}  "
             f"train={X_tr.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")

    log.info("  Computing linear-head baseline predictions ...")
    y_base_pred = eval_fc_head(model, X_test, device)
    return X_tr, y_tr, X_val, y_val, X_test, y_test, train_ds.classes, y_base_pred


# ── Experiment runner ─────────────────────────────────────────────────────────
def run_experiment(
    name: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    class_names: list,
    out_dir: Path,
    args: argparse.Namespace,
    log: logging.Logger,
    y_base_pred: np.ndarray = None,
) -> dict:
    n_classes = int(np.concatenate([y_tr, y_val, y_test]).max()) + 1
    log.info(f"\n{'='*60}")
    log.info(f"Dataset  : {name}")
    log.info(f"Train={X_tr.shape}  Val={X_val.shape}  Test={X_test.shape}")
    log.info(f"Classes  : {n_classes}")
    tr_counts = np.bincount(y_tr, minlength=n_classes)
    log.info(f"Train counts: max={tr_counts.max()} min={tr_counts[tr_counts>0].min()}")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # ── Baseline: ResNet32 FC head (CE-trained) ───────────────────────────────
    log.info(f"\n[Baseline] ResNet32 linear head (CE-trained) ...")
    y_base = y_base_pred
    rep_b  = classification_report(y_test, y_base, output_dict=True, zero_division=0)
    log.info(f"[Baseline] macro-F1={rep_b['macro avg']['f1-score']:.4f}  "
             f"wt-F1={rep_b['weighted avg']['f1-score']:.4f}")

    # ── FAR-MoE ───────────────────────────────────────────────────────────────
    log.info(f"\n[FAR-MoE] Training ...")
    t0 = time.time()
    moe = FARMoECifar(
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
    log.info(f"[FAR-MoE] Done in {time.time()-t0:.1f}s")

    y_moe = moe.predict(X_test)
    rep_m = classification_report(y_test, y_moe, output_dict=True, zero_division=0)
    log.info(f"[FAR-MoE] macro-F1={rep_m['macro avg']['f1-score']:.4f}  "
             f"wt-F1={rep_m['weighted avg']['f1-score']:.4f}")

    delta = rep_m['macro avg']['f1-score'] - rep_b['macro avg']['f1-score']
    sign = "▲" if delta > EPS else ("▼" if delta < -EPS else "–")
    log.info(f"[Delta] FAR-MoE − Baseline = {delta:+.4f} {sign}")

    # ── Per-class table (sorted by train support) ─────────────────────────────
    all_cls = sorted(set(int(c) for c in np.unique(np.concatenate([y_tr, y_test]))))

    def cname(c: int) -> str:
        return class_names[c] if c < len(class_names) else str(c)

    rows = []
    for c in all_cls:
        k = str(c)
        bm = rep_b.get(k, {})
        dm = rep_m.get(k, {})
        rows.append({
            "class":     cname(c),
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
            "f1(B)":   float(bm.get("f1-score", 0.0)),
            "f1(M)":   float(dm.get("f1-score", 0.0)),
            "delta_f1": float(dm.get("f1-score", 0.0)) - float(bm.get("f1-score", 0.0)),
            "_footer": True,
        })

    for r in rows:
        tag = "  [avg]" if r.get("_footer") else "      "
        log.info(f"{tag} {r['class']:30s} sup={r['support']:6d}  "
                 f"f1(B)={r['f1(B)']:.4f}  f1(M)={r['f1(M)']:.4f}  "
                 f"Δ={r['delta_f1']:+.4f}")

    safe = name.replace("/", "_").replace(" ", "_")
    save_colored_table(rows, COL_HEADERS, out_dir / f"{safe}_comparison.png",
                       title=f"FAR-MoE vs Baseline — {name}")
    pd.DataFrame(rows).to_csv(out_dir / f"{safe}_comparison.csv", index=False)

    # Expert summary
    exp_rows = []
    for exp in moe.experts_:
        exp_rows.append({
            "name": exp.name,
            "is_trained": exp.is_trained,
            "always_include": exp.always_include,
            "n_target_cls": len(exp.target_cls) if exp.target_cls else n_classes,
            "feat_dim": (exp.feat_idx.shape[0] if exp.feat_idx is not None
                         else X_tr.shape[1]),
        })
    pd.DataFrame(exp_rows).to_csv(out_dir / f"{safe}_experts.csv", index=False)

    return {
        "dataset":        name,
        "n_classes":      n_classes,
        "n_experts":      len(moe.experts_),
        "macro_f1_base":  rep_b['macro avg']['f1-score'],
        "macro_f1_moe":   rep_m['macro avg']['f1-score'],
        "delta_macro_f1": delta,
        "wt_f1_base":     rep_b['weighted avg']['f1-score'],
        "wt_f1_moe":      rep_m['weighted avg']['f1-score'],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FAR-MoE on CIFAR-10/100 Long-Tail (ResNet32 backbone)")
    p.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"],
                   choices=["cifar10", "cifar100"])
    p.add_argument("--imbalance_ratio",    type=float, default=100.0,
                   help="Long-tail imbalance ratio (1=balanced, 100=IR-100, default 100)")
    # ── backbone ─────────────────────────────────────────────────────────────
    p.add_argument("--backbone_epochs",    type=int,   default=200,
                   help="Epochs to train ResNet32 backbone (default 200)")
    p.add_argument("--backbone_lr",        type=float, default=0.1,
                   help="Initial LR for backbone SGD (cosine decay, default 0.1)")
    p.add_argument("--backbone_batch_size",type=int,   default=128,
                   help="Batch size for backbone training (default 128)")
    # ── Expert / FAR-MoE ─────────────────────────────────────────────────────
    p.add_argument("--expert_epochs",      type=int,   default=30,
                   help="Epochs to train each expert linear head (default 30)")
    p.add_argument("--expert_lr",          type=float, default=0.1,
                   help="LR for expert SGD with cosine decay (default 0.1)")
    p.add_argument("--expert_batch",       type=int,   default=1024,
                   help="Batch size for expert training (default 1024)")
    p.add_argument("--conf_threshold",     type=float, default=0.05)
    p.add_argument("--tail_n",             type=int,   default=50)
    p.add_argument("--tail_boost",         type=float, default=5.0)
    p.add_argument("--ood_threshold",      type=float, default=3.0)
    p.add_argument("--router_C",           type=float, default=1.0)
    p.add_argument("--device",             default="cuda",
                   choices=["cpu", "cuda"])
    p.add_argument("--seed",               type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"farmoecifar_{ts}"
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

    summary_rows = []
    for ds in args.datasets:
        log.info(f"\n{'='*60}\n"
                 f"Loading {ds.upper()} (backbone=ResNet32, IR={args.imbalance_ratio}) ...")
        X_tr, y_tr, X_val, y_val, X_test, y_test, cnames, y_base_pred = load_cifar_lt(
            ds, imbalance_ratio=args.imbalance_ratio,
            seed=args.seed, args=args, log=log)

        label = f"{ds.upper()}_LT{int(args.imbalance_ratio)}"
        sr = run_experiment(label, X_tr, y_tr, X_val, y_val, X_test, y_test,
                            cnames, out_dir, args, log, y_base_pred)
        summary_rows.append(sr)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'Dataset':28s} {'Experts':>7} {'F1-Base':>9} "
             f"{'F1-MoE':>9} {'Delta':>8}")
    for r in summary_rows:
        s = "▲" if r["delta_macro_f1"] > EPS else ("▼" if r["delta_macro_f1"] < -EPS else "–")
        log.info(f"{r['dataset']:28s} {r['n_experts']:>7d} "
                 f"{r['macro_f1_base']:>9.4f} {r['macro_f1_moe']:>9.4f} "
                 f"{r['delta_macro_f1']:>+8.4f} {s}")

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    log.info(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()

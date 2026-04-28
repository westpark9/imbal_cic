#!/usr/bin/env python3
"""
DCE-XGB: Dual-Balance Collaborative Experts with XGBoost
Adapted from: DCE (ICML 2025) - paper #06

Three XGBoost experts with different sample-weighting strategies:
  head     : log-inverse frequency (mild upweighting for majority classes)
  balanced : inverse frequency (standard balanced weighting)
  tail     : power-law upweighting (aggressive for rare classes)

Dynamic Expert Selector (DES):
  - Computes per-expert per-class recall on validation set -> affinity matrix
  - Assigns each class to its best-performing expert
  - Generates Gaussian pseudo-samples per class labeled with best expert index
  - Trains LogisticRegression router on pseudo-samples
  - At inference: soft routing weights alpha = router.predict_proba(x)
  - Final proba = sum_e(alpha_e * expert_e.predict_proba(x))

Usage:
  python code_dce.py --datasets cifar10lt unswnb15 cic2017
  python code_dce.py --datasets cic2017 --n_estimators 200 --subsample_train 500000
"""

import argparse
import datetime
import logging
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ─── Visualization constants (CLAUDE.md spec) ───────────────────────────────
BLUE   = "#cce5ff"
RED    = "#ffcccc"
YELLOW = "#fff9cc"
GRAY   = "#f2f2f2"
WHITE  = "#ffffff"
EPS    = 0.001


def save_colored_table(rows, col_headers, path, title=""):
    """Save colored PNG comparison table (CLAUDE.md visualization standard)."""
    n_cols = len(col_headers)
    n_rows = len(rows)
    cell_text = []
    cell_colors = []
    for row in rows:
        texts, colors = [], []
        for col in col_headers:
            val = row.get(col, "")
            texts.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            if col.endswith("(M)"):
                base_col = col.replace("(M)", "(B)")
                b_val = row.get(base_col)
                m_val = row.get(col)
                if isinstance(b_val, float) and isinstance(m_val, float):
                    if m_val > b_val + EPS:
                        colors.append(BLUE)
                    elif m_val < b_val - EPS:
                        colors.append(RED)
                    else:
                        colors.append(YELLOW)
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


# ─── DCE-XGB ────────────────────────────────────────────────────────────────

class DCE_XGB:
    """
    Dual-Balance Collaborative Experts with XGBoost backends.

    Three frequency-aware experts are trained on the same training data but
    with different sample-weighting strategies.  A LogisticRegression router
    (Dynamic Expert Selector) is fitted on Gaussian pseudo-samples drawn from
    per-class centroid/covariance estimates; its soft output probabilities are
    used as mixing weights at inference time.
    """

    EXPERT_NAMES = ["head", "balanced", "tail"]

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tail_power: float = 2.0,
        n_pseudo_per_class: int = 500,
        router_C: float = 1.0,
        router_max_iter: int = 1000,
        seed: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
        device: str = "cpu",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tail_power = tail_power
        self.n_pseudo_per_class = n_pseudo_per_class
        self.router_C = router_C
        self.router_max_iter = router_max_iter
        self.seed = seed
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.device = device

        self._experts: dict = {}
        self._router: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._affinity: np.ndarray | None = None   # [3, n_classes]
        self._class_to_expert: dict = {}            # class_int -> expert_idx
        self._classes: np.ndarray | None = None

    # ── private helpers ──────────────────────────────────────────────────────

    def _xgb_params(self) -> dict:
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric="mlogloss",
            random_state=self.seed,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
            device=self.device,
            verbosity=0,
        )

    def _sample_weights(self, y: np.ndarray, mode: str) -> np.ndarray:
        classes, counts = np.unique(y, return_counts=True)
        n = len(y)
        k = len(classes)
        if mode == "head":
            # log-inverse: gentle upweighting — head classes stay near 1
            cw = np.log1p(n / (k * counts))
            cw /= cw.min()
        elif mode == "balanced":
            cw = n / (k * counts.astype(float))
        else:  # tail
            cw = (n / (k * counts.astype(float))) ** self.tail_power
            cw = np.clip(cw, 1.0, 1000.0)
        wmap = dict(zip(classes, cw))
        return np.array([wmap[c] for c in y], dtype=np.float32)

    def _class_gaussian(self, X_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu, cov) for class samples X_c."""
        mu = X_c.mean(axis=0)
        n, d = X_c.shape
        if n >= d + 2:
            try:
                lw = LedoitWolf(assume_centered=False)
                lw.fit(X_c[:min(2000, n)])
                return mu, lw.covariance_
            except Exception:
                pass
        # diagonal fallback
        return mu, np.diag(X_c.var(axis=0) + 1e-6)

    def _build_pseudo_dataset(self, X: np.ndarray, y: np.ndarray):
        """Generate Gaussian pseudo-samples for router training."""
        rng = np.random.RandomState(self.seed)
        Xp_list, yp_list = [], []
        for c in self._classes:
            mask = y == c
            X_c = X[mask]
            mu, cov = self._class_gaussian(X_c)
            expert_idx = self._class_to_expert[c]
            try:
                samples = rng.multivariate_normal(mu, cov, self.n_pseudo_per_class)
            except np.linalg.LinAlgError:
                std = np.sqrt(np.maximum(np.diag(cov), 1e-8))
                samples = mu + rng.randn(self.n_pseudo_per_class, len(mu)) * std
            Xp_list.append(samples.astype(np.float32))
            yp_list.extend([expert_idx] * self.n_pseudo_per_class)
        return np.vstack(Xp_list), np.array(yp_list)

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "DCE_XGB":
        log = logging.getLogger(__name__)
        self._classes = np.unique(y_train)

        # Step 1: Train three experts
        log.info("  [DCE] Training three XGBoost experts...")
        for mode in self.EXPERT_NAMES:
            sw = self._sample_weights(y_train, mode)
            clf = xgb.XGBClassifier(**self._xgb_params())
            clf.fit(X_train, y_train, sample_weight=sw,
                    eval_set=[(X_val, y_val)], verbose=False)
            self._experts[mode] = clf
            log.info(f"    Expert '{mode}' done.")

        # Step 2: Affinity matrix — per-expert per-class recall on val
        log.info("  [DCE] Computing affinity matrix on val set...")
        n_cls = len(self._classes)
        aff = np.zeros((3, n_cls))
        for ei, mode in enumerate(self.EXPERT_NAMES):
            y_pred = self._experts[mode].predict(X_val)
            for ci, c in enumerate(self._classes):
                m = y_val == c
                if m.sum() > 0:
                    aff[ei, ci] = (y_pred[m] == c).mean()
        self._affinity = aff

        # Step 3: Assign each class to its best expert
        best = np.argmax(aff, axis=0)   # [n_classes]
        self._class_to_expert = dict(zip(self._classes, best))
        for ci, c in enumerate(self._classes):
            log.info(f"    class {c}: best={self.EXPERT_NAMES[best[ci]]} "
                     f"aff=[{aff[0,ci]:.3f}, {aff[1,ci]:.3f}, {aff[2,ci]:.3f}]")

        # Step 4: Pseudo-dataset + train DES router
        log.info("  [DCE] Building Gaussian pseudo-dataset for router...")
        Xp, yp = self._build_pseudo_dataset(X_train, y_train)
        self._scaler = StandardScaler()
        Xp_sc = self._scaler.fit_transform(Xp)

        log.info("  [DCE] Training Dynamic Expert Selector (LogisticRegression)...")
        self._router = LogisticRegression(
            C=self.router_C,
            max_iter=self.router_max_iter,
            solver="lbfgs",
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        self._router.fit(Xp_sc, yp)
        log.info("  [DCE] Router trained.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft-routed ensemble probability: sum_e( alpha_e * P_e(y|x) )."""
        X_sc = self._scaler.transform(X)
        alpha = self._router.predict_proba(X_sc)   # [N, n_experts_seen]
        router_classes = self._router.classes_      # expert indices actually seen

        # Align alpha columns to [head, balanced, tail] order
        alpha_aligned = np.zeros((len(X), 3), dtype=np.float32)
        for i, ec in enumerate(router_classes):
            alpha_aligned[:, int(ec)] = alpha[:, i]

        # Stack expert probabilities: [N, C, 3]
        expert_proba = np.stack(
            [self._experts[m].predict_proba(X) for m in self.EXPERT_NAMES],
            axis=2
        )
        # Weighted sum: einsum nce,ne -> nc
        return np.einsum("nce,ne->nc", expert_proba, alpha_aligned)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        # Map back to original class labels via expert's classes_
        clf0 = self._experts["head"]
        return clf0.classes_[idx]


# ─── Data loaders ────────────────────────────────────────────────────────────

def load_cifar10lt(imbalance_factor: int = 100, seed: int = 42,
                   pca_components: int = 128):
    """Generate CIFAR-10-LT training set; balanced test set; PCA features."""
    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required for CIFAR-10-LT. "
                          "Install with: pip install torchvision")
    import torchvision.datasets as dsets
    import torchvision.transforms as T

    log = logging.getLogger(__name__)
    transform = T.ToTensor()
    trainset = dsets.CIFAR10('/tmp/cifar10', train=True,  download=True, transform=transform)
    testset  = dsets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)

    X_full = trainset.data.reshape(len(trainset), -1).astype(np.float32) / 255.0
    y_full = np.array(trainset.targets)

    rng = np.random.RandomState(seed)
    n_classes, max_count = 10, 5000
    selected = []
    for c in range(n_classes):
        n_c = int(max_count * (imbalance_factor ** (-c / (n_classes - 1))))
        n_c = max(n_c, 1)
        idx = np.where(y_full == c)[0]
        idx = rng.choice(idx, size=min(n_c, len(idx)), replace=False)
        selected.extend(idx.tolist())
    rng.shuffle(selected)
    selected = np.array(selected)

    X_lt = X_full[selected]
    y_lt = y_full[selected]

    X_test_raw = testset.data.reshape(len(testset), -1).astype(np.float32) / 255.0
    y_test     = np.array(testset.targets)

    log.info(f"CIFAR-10-LT: {len(y_lt)} train samples, "
             f"IF={imbalance_factor}")
    for c in range(n_classes):
        log.info(f"  class {c}: {(y_lt==c).sum()}")

    log.info(f"Applying PCA({pca_components})...")
    pca = PCA(n_components=pca_components, random_state=seed)
    X_pca = pca.fit_transform(X_lt)
    X_test_pca = pca.transform(X_test_raw)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(sss.split(X_pca, y_lt))
    X_tr  = X_pca[tr_idx].astype(np.float32)
    y_tr  = y_lt[tr_idx]
    X_val = X_pca[val_idx].astype(np.float32)
    y_val = y_lt[val_idx]

    class_names = [f"class_{i}" for i in range(10)]
    return X_tr, y_tr, X_val, y_val, X_test_pca.astype(np.float32), y_test, class_names


def load_cifar100_tasks(
    n_balanced_tasks: int = 10,
    ir_medium: int = 50,
    ir_hard: int = 100,
    max_per_class: int = 500,
    seed: int = 42,
    pca_components: int = 256,
):
    """
    CIFAR-100 with superclass-based partial imbalance.

    Analogous to DCE's CORe50/CDDB-Hard where each "task" is a domain
    (art / clipart / product / realworld).  Here each task = one CIFAR-100
    superclass (20 superclasses × 5 fine classes = 100 classes).

    Regime assignment (randomised with seed, not by index range):
      10 superclasses  : balanced  (500 samples per fine class)
       5 superclasses  : IR=50     (exponential decay within superclass)
       5 superclasses  : IR=100    (exponential decay within superclass)

    Within each imbalanced superclass the 5 fine classes are shuffled
    randomly (seeded) before applying exponential decay, so no fine-class
    label systematically ends up at the bottom.
    """
    if not HAS_TORCHVISION:
        raise ImportError("torchvision required. pip install torchvision")
    import torchvision.datasets as dsets
    import torchvision.transforms as T

    log = logging.getLogger(__name__)

    # ── CIFAR-100 superclass → fine class label mapping ──────────────────────
    # Fine-class indices are the alphabetically-sorted CIFAR-100 labels (0-99).
    SUPERCLASSES = {
        "aquatic_mammals":              [4,  30, 55, 72, 95],
        "fish":                         [1,  32, 67, 73, 91],
        "flowers":                      [54, 62, 70, 82, 92],
        "food_containers":              [9,  10, 16, 28, 61],
        "fruit_and_vegetables":         [0,  51, 53, 57, 83],
        "household_electrical_devices": [22, 39, 40, 86, 87],
        "household_furniture":          [5,  20, 25, 84, 94],
        "insects":                      [6,   7, 14, 18, 24],
        "large_carnivores":             [3,  42, 43, 88, 97],
        "large_outdoor_man_made":       [12, 17, 37, 68, 76],
        "large_outdoor_natural":        [23, 33, 49, 60, 71],
        "large_omnivores_herbivores":   [15, 19, 21, 31, 38],
        "medium_mammals":               [34, 63, 64, 66, 75],
        "non_insect_invertebrates":     [26, 45, 77, 79, 99],
        "people":                       [2,  11, 35, 46, 98],
        "reptiles":                     [27, 29, 44, 78, 93],
        "small_mammals":                [36, 50, 65, 74, 80],
        "trees":                        [47, 52, 56, 59, 96],
        "vehicles_1":                   [8,  13, 48, 58, 90],
        "vehicles_2":                   [41, 69, 81, 85, 89],
    }
    superclass_names = list(SUPERCLASSES.keys())   # 20 superclasses
    n_super = len(superclass_names)                # 20
    n_medium = 5
    n_hard   = 5
    # n_balanced_tasks = 10  (default)

    # Randomly assign regimes to superclasses (reproducible)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_super)
    balanced_sc  = [superclass_names[i] for i in perm[:n_balanced_tasks]]
    medium_sc    = [superclass_names[i] for i in perm[n_balanced_tasks:n_balanced_tasks+n_medium]]
    hard_sc      = [superclass_names[i] for i in perm[n_balanced_tasks+n_medium:]]

    def regime_of(sc):
        if sc in balanced_sc: return "balanced",   1
        if sc in medium_sc:   return f"IR={ir_medium}", ir_medium
        return f"IR={ir_hard}", ir_hard

    # ── Load raw data ─────────────────────────────────────────────────────────
    transform = T.ToTensor()
    trainset = dsets.CIFAR100("/tmp/cifar100", train=True,  download=True, transform=transform)
    testset  = dsets.CIFAR100("/tmp/cifar100", train=False, download=True, transform=transform)

    X_full = trainset.data.reshape(len(trainset), -1).astype(np.float32) / 255.0
    y_full = np.array(trainset.targets)

    # ── Build imbalanced training set ─────────────────────────────────────────
    selected_idx = []
    task_meta = []   # (task_id, sc_name, regime, fine_classes, ir)

    for t, sc in enumerate(superclass_names):
        fine_classes  = list(SUPERCLASSES[sc])
        regime, ir    = regime_of(sc)
        n_cls_in_task = len(fine_classes)

        # Shuffle fine classes within the superclass before applying decay
        # so no single fine label is systematically the tail
        shuffled = rng.permutation(fine_classes).tolist()

        for rank, c in enumerate(shuffled):
            if ir == 1:
                n_c = max_per_class
            else:
                n_c = int(max_per_class * (ir ** (-rank / (n_cls_in_task - 1))))
                n_c = max(n_c, 1)
            idx = np.where(y_full == c)[0]
            idx = rng.choice(idx, size=min(n_c, len(idx)), replace=False)
            selected_idx.extend(idx.tolist())

        task_meta.append((t, sc, regime, fine_classes, ir))

    rng.shuffle(selected_idx)
    selected_idx = np.array(selected_idx)
    X_lt = X_full[selected_idx]
    y_lt = y_full[selected_idx]

    # ── Log task structure ────────────────────────────────────────────────────
    log.info("CIFAR-100 superclass task structure:")
    for t, sc, regime, fcs, ir in task_meta:
        cnts = [(y_lt == c).sum() for c in fcs]
        log.info(f"  Task {t:2d}  {sc:35s}  [{regime:12s}]  "
                 f"min={min(cnts):3d}  max={max(cnts):3d}  total={sum(cnts)}")
    log.info(f"  Total training samples : {len(y_lt)}")
    log.info(f"  Balanced superclasses  : {balanced_sc}")
    log.info(f"  IR={ir_medium} superclasses : {medium_sc}")
    log.info(f"  IR={ir_hard} superclasses : {hard_sc}")

    # ── Test set: balanced (100 samples/class, all 100 classes) ──────────────
    X_test_raw = testset.data.reshape(len(testset), -1).astype(np.float32) / 255.0
    y_test     = np.array(testset.targets)

    # ── PCA on training data only ─────────────────────────────────────────────
    log.info(f"Applying PCA({pca_components})...")
    pca = PCA(n_components=pca_components, random_state=seed)
    X_pca      = pca.fit_transform(X_lt).astype(np.float32)
    X_test_pca = pca.transform(X_test_raw).astype(np.float32)

    # ── 80/20 stratified train/val split ──────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(sss.split(X_pca, y_lt))
    X_tr  = X_pca[tr_idx];  y_tr  = y_lt[tr_idx]
    X_val = X_pca[val_idx]; y_val = y_lt[val_idx]

    # Fine-class names (alphabetically sorted, matching label indices 0-99)
    class_names = [
        "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle",
        "bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel",
        "can","castle","caterpillar","cattle","chair","chimpanzee","clock",
        "cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
        "dolphin","elephant","flatfish","forest","fox","girl","hamster",
        "house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion",
        "lizard","lobster","man","maple_tree","motorcycle","mountain","mouse",
        "mushroom","oak_tree","orange","orchid","otter","palm_tree","pear",
        "pickup_truck","pine_tree","plain","plate","poppy","porcupine",
        "possum","rabbit","raccoon","ray","road","rocket","rose","sea",
        "seal","shark","shrew","skunk","skyscraper","snail","snake",
        "spider","squirrel","streetcar","sunflower","sweet_pepper","table",
        "tank","telephone","television","tiger","tractor","train","trout",
        "tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm",
    ]
    return X_tr, y_tr, X_val, y_val, X_test_pca, y_test, class_names, task_meta


def load_pkl_dataset(pkl_path: str, seed: int = 42,
                     subsample_train: int | None = None):
    """Load preprocessed pkl (CIC-IDS2017 or UNSW-NB15)."""
    log = logging.getLogger(__name__)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"].astype(np.float32)
    y = data["y"]
    le = data["label_encoder"]
    class_names = list(le.classes_)

    log.info(f"Loaded {pkl_path}: X={X.shape}, classes={len(class_names)}")

    if "train_indices" in data and "test_indices" in data:
        tr_idx = np.array(data["train_indices"])
        te_idx = np.array(data["test_indices"])
        X_train_full = X[tr_idx]; y_train_full = y[tr_idx]
        X_test = X[te_idx];       y_test = y[te_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        t2, vi = next(sss.split(X_train_full, y_train_full))
        X_tr  = X_train_full[t2]; y_tr  = y_train_full[t2]
        X_val = X_train_full[vi]; y_val = y_train_full[vi]
    else:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
        tr_i, tmp_i = next(sss1.split(X, y))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        vi, te_i = next(sss2.split(X[tmp_i], y[tmp_i]))
        X_tr  = X[tr_i];          y_tr  = y[tr_i]
        X_val = X[tmp_i][vi];     y_val = y[tmp_i][vi]
        X_test = X[tmp_i][te_i];  y_test = y[tmp_i][te_i]

    if subsample_train and len(y_tr) > subsample_train:
        log.info(f"Subsampling train {len(y_tr)} -> {subsample_train}")
        sss3 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=subsample_train,
            random_state=seed
        )
        keep, _ = next(sss3.split(X_tr, y_tr))
        X_tr = X_tr[keep]; y_tr = y_tr[keep]

    return X_tr, y_tr, X_val, y_val, X_test, y_test, class_names


# ─── Experiment runner ───────────────────────────────────────────────────────

COL_HEADERS = [
    "class", "support",
    "prec(B)", "prec(M)",
    "recall(B)", "recall(M)",
    "f1(B)", "f1(M)",
    "delta_f1",
]


def run_experiment(dataset_name: str,
                   X_tr, y_tr, X_val, y_val, X_test, y_test,
                   class_names: list[str],
                   out_dir: Path,
                   args) -> tuple[dict, dict]:
    log = logging.getLogger(__name__)
    log.info(f"\n{'='*60}")
    log.info(f"Dataset : {dataset_name}")
    log.info(f"Train   : {X_tr.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    log.info(f"Classes : {len(class_names)}")
    for c in np.unique(y_tr):
        n = (y_tr == c).sum()
        nm = class_names[c] if c < len(class_names) else str(c)
        log.info(f"  {nm}: train={n}")

    base_xgb_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.colsample_bytree,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
        tree_method=args.tree_method,
        device=args.device,
        verbosity=0,
    )

    # ── Baseline ──
    log.info("\n[Baseline] Training XGBoost...")
    t0 = time.time()
    baseline = xgb.XGBClassifier(**base_xgb_params)
    baseline.fit(X_tr, y_tr,
                 eval_set=[(X_val, y_val)], verbose=False)
    log.info(f"[Baseline] Trained in {time.time()-t0:.1f}s")

    y_pred_base = baseline.predict(X_test)
    rep_base = classification_report(y_test, y_pred_base,
                                     output_dict=True, zero_division=0)
    log.info(f"[Baseline] macro-F1  = {rep_base['macro avg']['f1-score']:.4f}")
    log.info(f"[Baseline] wt-avg-F1 = {rep_base['weighted avg']['f1-score']:.4f}")

    # ── DCE-XGB ──
    log.info("\n[DCE-XGB] Training...")
    t0 = time.time()
    dce = DCE_XGB(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.colsample_bytree,
        tail_power=args.tail_power,
        n_pseudo_per_class=args.n_pseudo_per_class,
        router_C=args.router_C,
        seed=args.seed,
        tree_method=args.tree_method,
        device=args.device,
    )
    dce.fit(X_tr, y_tr, X_val, y_val)
    log.info(f"[DCE-XGB] Trained in {time.time()-t0:.1f}s")

    y_pred_dce = dce.predict(X_test)
    rep_dce = classification_report(y_test, y_pred_dce,
                                    output_dict=True, zero_division=0)
    log.info(f"[DCE-XGB] macro-F1  = {rep_dce['macro avg']['f1-score']:.4f}")
    log.info(f"[DCE-XGB] wt-avg-F1 = {rep_dce['weighted avg']['f1-score']:.4f}")

    delta_macro = (rep_dce['macro avg']['f1-score']
                   - rep_base['macro avg']['f1-score'])
    log.info(f"[Delta]   macro-F1 DCE - Base = {delta_macro:+.4f}")

    # ── Build comparison rows ──
    all_classes = sorted(np.unique(y_test))

    def cname(c):
        return class_names[c] if c < len(class_names) else str(c)

    rows = []
    for c in all_classes:
        k = str(c)
        bm = rep_base.get(k, {})
        dm = rep_dce.get(k, {})
        rows.append({
            "class":     cname(c),
            "support":   int(bm.get("support", 0)),
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
        bm = rep_base.get(avg, {})
        dm = rep_dce.get(avg, {})
        rows.append({
            "class":     avg,
            "support":   int(bm.get("support", 0)),
            "prec(B)":   float(bm.get("precision", 0.0)),
            "prec(M)":   float(dm.get("precision", 0.0)),
            "recall(B)": float(bm.get("recall", 0.0)),
            "recall(M)": float(dm.get("recall", 0.0)),
            "f1(B)":     float(bm.get("f1-score", 0.0)),
            "f1(M)":     float(dm.get("f1-score", 0.0)),
            "delta_f1":  float(dm.get("f1-score", 0.0)) - float(bm.get("f1-score", 0.0)),
            "_footer":   True,
        })

    # Log per-class comparison
    log.info("\nPer-class F1 comparison (Baseline vs DCE-XGB):")
    for r in rows:
        if r.get("_footer"):
            log.info(f"  [{r['class']:15s}] support={r['support']:7d}  "
                     f"f1(B)={r['f1(B)']:.4f}  f1(M)={r['f1(M)']:.4f}  "
                     f"delta={r['delta_f1']:+.4f}")
        else:
            log.info(f"  {r['class']:20s}  support={r['support']:7d}  "
                     f"f1(B)={r['f1(B)']:.4f}  f1(M)={r['f1(M)']:.4f}  "
                     f"delta={r['delta_f1']:+.4f}")

    # Save CSV
    csv_path = out_dir / f"{dataset_name}_baseline_vs_dce.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info(f"\nCSV saved: {csv_path}")

    # Save colored PNG
    png_path = out_dir / f"{dataset_name}_baseline_vs_dce.png"
    save_colored_table(rows, COL_HEADERS, png_path,
                       title=f"DCE-XGB vs Baseline — {dataset_name}")
    log.info(f"PNG saved: {png_path}")

    # Also save specialist activation info
    act_path = out_dir / f"{dataset_name}_expert_affinity.csv"
    aff_rows = []
    if dce._affinity is not None:
        for ci, c in enumerate(dce._classes):
            best_e = DCE_XGB.EXPERT_NAMES[int(np.argmax(dce._affinity[:, ci]))]
            aff_rows.append({
                "class": cname(int(c)),
                "train_support": int((y_tr == c).sum()),
                "best_expert": best_e,
                "aff_head": float(dce._affinity[0, ci]),
                "aff_balanced": float(dce._affinity[1, ci]),
                "aff_tail": float(dce._affinity[2, ci]),
            })
        pd.DataFrame(aff_rows).to_csv(act_path, index=False)
        log.info(f"Affinity CSV saved: {act_path}")

    return rep_base, rep_dce


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DCE-XGB Experiment Runner")
    parser.add_argument("--datasets", nargs="+",
                        default=["cifar10lt", "unswnb15", "cic2017"],
                        choices=["cifar10lt", "unswnb15", "cic2017", "cifar100tasks"])
    parser.add_argument("--cic_pkl",  default="data/cic2017_proc.pkl")
    parser.add_argument("--unsw_pkl", default="data/unswnb15_proc.pkl")

    # Model hyperparameters
    parser.add_argument("--n_estimators",    type=int,   default=200)
    parser.add_argument("--max_depth",       type=int,   default=6)
    parser.add_argument("--learning_rate",   type=float, default=0.1)
    parser.add_argument("--xgb_subsample",   type=float, default=0.8)
    parser.add_argument("--colsample_bytree",type=float, default=0.8)
    parser.add_argument("--tail_power",      type=float, default=2.0,
                        help="Exponent for tail expert sample weights")
    parser.add_argument("--n_pseudo_per_class", type=int, default=500,
                        help="Pseudo-samples per class for router training")
    parser.add_argument("--router_C",        type=float, default=1.0)

    # Data options
    parser.add_argument("--imbalance_factor", type=int, default=100,
                        help="CIFAR-10-LT imbalance factor")
    parser.add_argument("--subsample_train",  type=int, default=None,
                        help="Max training samples per dataset (None=all)")
    # CIFAR-100 task options (20 superclasses total)
    parser.add_argument("--c100_n_balanced",  type=int, default=10,
                        help="# balanced superclasses out of 20 (rest split equally into IR=medium/hard)")
    parser.add_argument("--c100_ir_medium",   type=int, default=50)
    parser.add_argument("--c100_ir_hard",     type=int, default=100)
    parser.add_argument("--c100_pca",         type=int, default=256)

    # Hardware
    parser.add_argument("--tree_method", default="hist",
                        choices=["hist", "approx", "exact"])
    parser.add_argument("--device",      default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Detect GPU
    if args.device == "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "cuda"
                args.tree_method = "hist"
                logging.info("GPU detected, using device=cuda")
        except ImportError:
            pass

    # Output directory
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"dce_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging — force=True overrides any handlers set by imported libs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "experiment.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger(__name__)
    log.info(f"Results dir : {out_dir}")
    log.info(f"Arguments   : {vars(args)}")
    log.info(f"XGBoost ver : {xgb.__version__}")

    summary = []

    for ds in args.datasets:
        log.info(f"\n{'#'*60}")
        log.info(f"# Dataset: {ds}")
        log.info(f"{'#'*60}")

        try:
            if ds == "cifar10lt":
                if not HAS_TORCHVISION:
                    log.warning("torchvision not found — skipping CIFAR-10-LT")
                    continue
                data = load_cifar10lt(
                    imbalance_factor=args.imbalance_factor,
                    seed=args.seed,
                )
                X_tr, y_tr, X_val, y_val, X_test, y_test, class_names = data

            elif ds == "cifar100tasks":
                if not HAS_TORCHVISION:
                    log.warning("torchvision not found — skipping CIFAR-100-Tasks")
                    continue
                data = load_cifar100_tasks(
                    n_balanced_tasks=args.c100_n_balanced,
                    ir_medium=args.c100_ir_medium,
                    ir_hard=args.c100_ir_hard,
                    seed=args.seed,
                    pca_components=args.c100_pca,
                )
                X_tr, y_tr, X_val, y_val, X_test, y_test, class_names, task_meta = data

                # Log task regime summary to experiment.log
                log.info("Task regimes (train split counts):")
                for t, sc, regime, fcs, ir in task_meta:
                    cnts = [(y_tr == c).sum() for c in fcs]
                    log.info(f"  Task {t:2d}  {sc:35s}  [{regime:12s}]  "
                             f"min={min(cnts):3d}  max={max(cnts):3d}  total={sum(cnts)}")

            elif ds == "unswnb15":
                pkl = Path(args.unsw_pkl)
                if not pkl.exists():
                    log.warning(f"UNSW pkl not found: {pkl} — skipping")
                    continue
                X_tr, y_tr, X_val, y_val, X_test, y_test, class_names = \
                    load_pkl_dataset(str(pkl), args.seed, args.subsample_train)

            elif ds == "cic2017":
                pkl = Path(args.cic_pkl)
                if not pkl.exists():
                    log.warning(f"CIC pkl not found: {pkl} — skipping")
                    continue
                X_tr, y_tr, X_val, y_val, X_test, y_test, class_names = \
                    load_pkl_dataset(str(pkl), args.seed, args.subsample_train)

            else:
                continue

            rep_base, rep_dce = run_experiment(
                ds, X_tr, y_tr, X_val, y_val, X_test, y_test,
                class_names, out_dir, args
            )
            summary.append({
                "dataset":          ds,
                "baseline_macro_f1": rep_base["macro avg"]["f1-score"],
                "dce_macro_f1":      rep_dce["macro avg"]["f1-score"],
                "delta":             rep_dce["macro avg"]["f1-score"]
                                     - rep_base["macro avg"]["f1-score"],
            })
        except Exception as exc:
            log.exception(f"Error in dataset {ds}: {exc}")

    # Final summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    for s in summary:
        sign = "+" if s["delta"] >= 0 else ""
        log.info(f"  {s['dataset']:12s}  baseline={s['baseline_macro_f1']:.4f}  "
                 f"dce={s['dce_macro_f1']:.4f}  delta={sign}{s['delta']:.4f}")

    if summary:
        df_sum = pd.DataFrame(summary)
        df_sum.to_csv(out_dir / "summary.csv", index=False)
        log.info(f"\nSummary CSV: {out_dir / 'summary.csv'}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()

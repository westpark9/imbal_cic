#!/usr/bin/env python3
import argparse
import pdb
import gc
import logging
import os
import pickle
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# -----------------------------
# Utilities
# -----------------------------

def infer_dataset_type(data_path, data=None):
    if data is not None and isinstance(data, dict) and "dataset_type" in data:
        return str(data["dataset_type"]).lower()
    name = os.path.basename(data_path).lower()
    if "nf-unsw" in name or "nf_unsw" in name or "nfunsw" in name:
        return "nfunswnb15"
    if "unsw" in name or "nb15" in name:
        return "unswnb15"
    if "2017" in name:
        return "cic2017"
    if "2018" in name:
        return "cic2018"
    raise ValueError(
        f"Cannot infer dataset_type from '{data_path}'. Include nfunswnb15/unswnb15/cic2017/cic2018 in filename or in pickle['dataset_type']."
    )


def get_xgb_device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = get_xgb_device()


def setup_logging(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("experiment")


def safe_train_test_split(X, y, test_size, random_state, stratify=True):
    try:
        return train_test_split(
            X, y, test_size=test_size, stratify=(y if stratify else None), random_state=random_state
        )
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def guarded_internal_split_with_weights(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    seed: int,
    logger,
    context: str,
    test_size: float = 0.1,
    max_tries: int = 12,
):
    """Create an internal train/validation split that is safe for rare classes.

    Guarantees, when possible:
    - train split retains all classes present in the subset
    - validation split contains at least 2 classes when the subset has >= 2 classes

    If this cannot be achieved (common for very small / tail subsets), it falls back to
    full-data training without an eval_set / early stopping.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w)

    unique_classes = np.sort(np.unique(y))
    n_unique = len(unique_classes)
    if n_unique < 2:
        return X, None, y, None, w, None, False

    counts = np.bincount(y)
    # If any class has <2 samples, a stratified internal split is inherently fragile.
    too_small_for_safe_holdout = np.any(counts[counts > 0] < 2) or len(y) < max(12, 2 * n_unique)
    trial_seeds = [seed + t for t in range(max_tries)]

    if not too_small_for_safe_holdout:
        for stratify in [True, False]:
            for rs in trial_seeds:
                try:
                    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
                        X,
                        y,
                        w,
                        test_size=test_size,
                        stratify=(y if stratify else None),
                        random_state=rs,
                    )
                except ValueError:
                    continue
                train_classes = np.sort(np.unique(y_tr))
                val_classes = np.sort(np.unique(y_va))
                if len(train_classes) == n_unique and len(val_classes) >= min(2, n_unique):
                    return X_tr, X_va, y_tr, y_va, w_tr, w_va, True

    logger.warning(
        f"[{context}] Could not form a safe internal validation split; "
        f"falling back to full-subset training without early stopping."
    )
    return X, None, y, None, w, None, False



def zscore_per_vector(values: np.ndarray, mean: float, std: float, clip: float = 8.0) -> np.ndarray:
    out = (values - mean) / max(std, 1e-6)
    return np.clip(out, -clip, clip)


def check_split_coverage(
    y_all: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_to_family: Optional[Dict[int, int]] = None,
) -> Tuple[bool, Dict[str, List[int]]]:
    """Coverage check for outer train/val/test splits.

    Stronger than a plain class-presence check:
    - training should contain every class with at least 2 total samples
    - validation should contain at least 2 classes
    - if family metadata is available, all sufficiently supported attack families
      should be represented in both train and validation
    """
    y_all = np.asarray(y_all, dtype=int)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    n_classes = int(np.max(y_all)) + 1
    counts_all = np.bincount(y_all, minlength=n_classes)
    counts_train = np.bincount(y_train, minlength=n_classes)
    counts_val = np.bincount(y_val, minlength=n_classes)

    must_be_in_train = [c for c in range(n_classes) if counts_all[c] >= 2]
    missing_train_classes = [c for c in must_be_in_train if counts_train[c] == 0]

    details = {
        "missing_train_classes": missing_train_classes,
        "missing_val_classes": [],
        "missing_train_families": [],
        "missing_val_families": [],
    }

    ok = len(missing_train_classes) == 0 and len(np.unique(y_val)) >= min(2, len(np.unique(y_all)))

    if class_to_family is not None:
        family_ids = sorted(set(class_to_family.values()))
        fam_counts_all = {fid: 0 for fid in family_ids}
        fam_counts_train = {fid: 0 for fid in family_ids}
        fam_counts_val = {fid: 0 for fid in family_ids}

        for c, n in enumerate(counts_all):
            fam_counts_all[class_to_family.get(c, 0)] += int(n)
        for c, n in enumerate(counts_train):
            fam_counts_train[class_to_family.get(c, 0)] += int(n)
        for c, n in enumerate(counts_val):
            fam_counts_val[class_to_family.get(c, 0)] += int(n)

        # Require every non-trivial attack family to appear in train, and every reasonably
        # supported attack family to appear in validation as well.
        must_train_fams = [fid for fid, n in fam_counts_all.items() if fid != 0 and n >= 2]
        must_val_fams = [fid for fid, n in fam_counts_all.items() if fid != 0 and n >= 10]

        missing_train_families = [fid for fid in must_train_fams if fam_counts_train.get(fid, 0) == 0]
        missing_val_families = [fid for fid in must_val_fams if fam_counts_val.get(fid, 0) == 0]

        details["missing_train_families"] = missing_train_families
        details["missing_val_families"] = missing_val_families

        ok = ok and (len(missing_train_families) == 0) and (len(missing_val_families) == 0)

    return ok, details



def row_softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / max(temperature, 1e-6)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    denom = np.sum(e, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return e / denom


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def get_feature_names(data: dict, n_features: int) -> List[str]:
    for key in ["feature_names", "columns", "feature_columns", "col_names"]:
        if key in data and data[key] is not None:
            vals = list(data[key])
            if len(vals) == n_features:
                return [str(v) for v in vals]
    return [f"f_{i}" for i in range(n_features)]


def compute_local_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    ent = -np.sum(probs * np.log(probs), axis=1)
    if probs.shape[1] > 1:
        ent = ent / np.log(probs.shape[1])
    return ent.astype(np.float32)


def compute_margin(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] == 1:
        return np.ones(probs.shape[0], dtype=np.float32)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    return (top1 - top2).astype(np.float32)


def compute_balanced_sample_weights(y_local: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_local)
    counts = np.maximum(counts, 1)
    n = len(y_local)
    k = len(counts)
    class_weights = n / (k * counts)
    return class_weights[y_local].astype(np.float32)


# -----------------------------
# Feature views
# -----------------------------

def build_feature_views(feature_names: List[str], logger) -> Dict[str, np.ndarray]:
    n_features = len(feature_names)
    norm_names = [normalize_name(x) for x in feature_names]

    def match_any(name: str, patterns: List[str]) -> bool:
        return any(p in name for p in patterns)

    volume_patterns = [
        "flowduration",
        "totfwdpkts",
        "totbwdpkts",
        "totalfwdpackets",
        "totalbackwardpackets",
        "flowbytess",
        "flowpacketss",
        "flowbytess",
        "flowpacketss",
        "subflowfwdbytes",
        "subflowbwdbytes",
        "subflowfwdpkts",
        "subflowbwdpkts",
        "flowbytes",
        "flowpackets",
        "downupratio",
    ]
    timing_patterns = ["iat", "activemean", "activestd", "activemax", "activemin", "idlemean", "idlestd", "idlemax", "idlemin"]
    packet_patterns = [
        "packetlength",
        "pktlen",
        "packetsize",
        "segmentsize",
        "fwdsegsize",
        "bwdsegsize",
        "fwdheaderlength",
        "bwdheaderlength",
        "totalfwdpacketlength",
        "totalbackwardpacketlength",
    ]
    tcp_patterns = [
        "flag",
        "window",
        "headerlength",
        "finflagcnt",
        "synflagcnt",
        "rstflagcnt",
        "pshflagcnt",
        "ackflagcnt",
        "urgflagcnt",
        "eceflagcnt",
        "cweflagcount",
        "fwdpshflags",
        "bwdpshflags",
        "fwdurgflags",
        "bwdu rgflags",
    ]

    views = {
        "all": np.arange(n_features, dtype=int),
        "volume": np.array([i for i, n in enumerate(norm_names) if match_any(n, volume_patterns)], dtype=int),
        "timing": np.array([i for i, n in enumerate(norm_names) if match_any(n, timing_patterns)], dtype=int),
        "packet": np.array([i for i, n in enumerate(norm_names) if match_any(n, packet_patterns)], dtype=int),
        "tcp": np.array([i for i, n in enumerate(norm_names) if match_any(n, tcp_patterns)], dtype=int),
    }

    named_features_available = any(not n.startswith("f_") for n in feature_names)
    if not named_features_available:
        logger.warning("Feature names not found; using deterministic column partitions for feature-view diversity.")
        idx = np.arange(n_features, dtype=int)
        views = {
            "all": idx,
            "volume": idx[0::4],
            "timing": idx[1::4],
            "packet": idx[2::4],
            "tcp": idx[3::4],
        }

    for k in ["volume", "timing", "packet", "tcp"]:
        if views[k].size == 0:
            logger.warning(f"Feature view '{k}' is empty; falling back to all features for that view.")
            views[k] = views["all"]

    return views


# -----------------------------
# Taxonomy helpers
# -----------------------------

def get_taxonomy_groups(dataset_type: str, class_names: List[str], logger) -> Tuple[Dict[str, List[int]], Dict[int, int], Dict[int, str]]:
    groups = {"Normal": [], "Group1": [], "Group2": [], "Group3": [], "Group4": []}

    if dataset_type in {"unswnb15", "nfunswnb15"}:
        type_groups = [
            ["DoS", "Worms"],
            ["Analysis", "Reconnaissance"],
            ["Exploits", "Fuzzers", "Generic"],
            ["Backdoor", "Shellcode"],
        ]
    elif dataset_type == "cic2017":
        # Family layout is chosen to better match attack semantics and routing behavior:
        #   Group1: volumetric DoS/DDoS
        #   Group2: credential / login attacks
        #   Group3: application-layer attacks
        #   Group4: recon / malware-like activity
        type_groups = [
            ["DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"],
            ["FTP-Patator", "SSH-Patator"],
            ["Web Attack  Brute Force", "Web Attack  Sql Injection", "Web Attack  XSS", "Heartbleed"],
            ["Bot", "Infiltration", "PortScan"],
        ]
    else:  # cic2018
        type_groups = [
            [
                "DDOS attack-HOIC",
                "DDOS attack-LOIC-UDP",
                "DDoS attacks-LOIC-HTTP",
                "DoS attacks-GoldenEye",
                "DoS attacks-Hulk",
                "DoS attacks-SlowHTTPTest",
                "DoS attacks-Slowloris",
            ],
            ["FTP-BruteForce", "SSH-Bruteforce"],
            ["Brute Force -Web", "SQL Injection", "Brute Force -XSS"],
            ["Bot", "Infilteration"],
        ]

    norm_type_groups = [[normalize_name(c) for c in group] for group in type_groups]
    class_to_family = {}
    family_id_to_name = {0: "Normal", 1: "Group1", 2: "Group2", 3: "Group3", 4: "Group4"}

    for idx, name in enumerate(class_names):
        n = normalize_name(name)
        if "normal" in n or "benign" in n:
            groups["Normal"].append(idx)
            class_to_family[idx] = 0
            continue

        matched = False
        for g_idx, norm_group in enumerate(norm_type_groups, start=1):
            if n in norm_group:
                groups[f"Group{g_idx}"].append(idx)
                class_to_family[idx] = g_idx
                matched = True
                break
        if not matched:
            logger.warning(f"Class '{name}' not mapped to a known attack family. Treating as Group4 fallback.")
            groups["Group4"].append(idx)
            class_to_family[idx] = 4

    return groups, class_to_family, family_id_to_name


# -----------------------------
# OOD model
# -----------------------------

class MahalanobisOOD:
    def __init__(self, max_fit_samples: int = 20000, random_state: int = 42):
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
        self.mean_ = None
        self.precision_ = None
        self.var_ = None
        self.mode = None

    def fit(self, X: np.ndarray):
        if X.shape[0] == 0:
            raise ValueError("Cannot fit OOD model on empty data")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] > self.max_fit_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0], size=self.max_fit_samples, replace=False)
            X = X[idx]

        try:
            lw = LedoitWolf().fit(X)
            self.mean_ = lw.location_.astype(np.float32)
            self.precision_ = lw.precision_.astype(np.float32)
            self.mode = "full"
        except Exception:
            self.mean_ = np.mean(X, axis=0).astype(np.float32)
            self.var_ = np.var(X, axis=0).astype(np.float32) + 1e-6
            self.mode = "diag"
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.mode == "full":
            diff = X - self.mean_
            dist2 = np.einsum("bi,ij,bj->b", diff, self.precision_, diff, optimize=True)
        elif self.mode == "diag":
            diff = X - self.mean_
            dist2 = np.sum((diff * diff) / self.var_, axis=1)
        else:
            raise RuntimeError("OOD model not fitted")
        return dist2.astype(np.float32)


# -----------------------------
# Expert definitions
# -----------------------------

@dataclass
class ExpertConfig:
    expert_id: int
    name: str
    target_classes: Optional[List[int]] = None  # None => global expert
    target_family_ids: List[int] = field(default_factory=list)
    feature_indices: Optional[np.ndarray] = None
    include_normal: bool = True
    extra_negative_classes: List[int] = field(default_factory=list)
    use_balanced_weights: bool = False
    target_boost: float = 1.0
    normal_weight_scale: float = 1.0
    extra_negative_weight_scale: float = 1.0
    always_include: bool = False
    kind: str = "specialist"


class ExpertXGB:
    def __init__(self, config: ExpertConfig, global_num_classes: int, device: str = "cpu", seed: int = 42):
        self.config = config
        self.global_num_classes = global_num_classes
        self.device = device
        self.seed = seed
        self.model = None
        self.local_classes = None
        self.is_trained = False
        self.ood_model = None
        self.ood_feature_mean = None
        self.ood_feature_std = None
        self.train_size = 0
        self.val_acc = None

    @property
    def name(self):
        return self.config.name

    @property
    def feature_indices(self):
        return self.config.feature_indices

    @property
    def assigned_classes(self):
        return self.local_classes.tolist() if self.local_classes is not None else []

    def _prepare_subset(self, X: np.ndarray, y: np.ndarray, normal_classes: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.target_classes is None:
            mask = np.ones(len(y), dtype=bool)
        else:
            keep = set(self.config.target_classes)
            if self.config.include_normal:
                keep.update(normal_classes)
            if self.config.extra_negative_classes:
                keep.update(self.config.extra_negative_classes)
            mask = np.isin(y, sorted(keep))
        X_sub = X[mask][:, self.feature_indices]
        y_sub = y[mask]
        return X_sub, y_sub, mask

    def fit(self, X: np.ndarray, y: np.ndarray, normal_classes: List[int], logger):
        X_sub, y_sub, _ = self._prepare_subset(X, y, normal_classes)
        self.train_size = len(y_sub)
        if len(y_sub) == 0:
            logger.warning(f"[{self.name}] Empty training subset; skipping expert.")
            return

        self.local_classes = np.sort(np.unique(y_sub))
        if len(self.local_classes) < 2:
            logger.warning(f"[{self.name}] Only one class in subset; skipping expert.")
            self.local_classes = None
            return

        mapper_g2l = {g: l for l, g in enumerate(self.local_classes)}
        y_local = np.array([mapper_g2l[v] for v in y_sub], dtype=int)

        weights = np.ones(len(y_local), dtype=np.float32)
        if self.config.use_balanced_weights:
            weights *= compute_balanced_sample_weights(y_local)

        if self.config.target_classes is not None:
            target_mask = np.isin(y_sub, self.config.target_classes)
            weights[target_mask] *= self.config.target_boost
            if self.config.include_normal and normal_classes:
                normal_mask = np.isin(y_sub, normal_classes)
                weights[normal_mask] *= self.config.normal_weight_scale
            if self.config.extra_negative_classes:
                extra_neg_mask = np.isin(y_sub, self.config.extra_negative_classes)
                weights[extra_neg_mask] *= self.config.extra_negative_weight_scale

        X_train_sub, X_val_sub, y_train_sub, y_val_sub, w_train_sub, w_val_sub, use_eval = guarded_internal_split_with_weights(
            X_sub, y_local, weights, seed=self.seed, logger=logger, context=self.name
        )

        n_local = len(self.local_classes)
        params = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
        )
        if use_eval:
            params["early_stopping_rounds"] = 20
        if self.device == "cuda":
            params["device"] = "cuda"

        if n_local == 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_local
            params["eval_metric"] = "mlogloss"

        self.model = xgb.XGBClassifier(**params)
        fit_kwargs = dict(sample_weight=w_train_sub, verbose=False)
        if use_eval and X_val_sub is not None and y_val_sub is not None:
            fit_kwargs["eval_set"] = [(X_val_sub, y_val_sub)]
            fit_kwargs["sample_weight_eval_set"] = [w_val_sub]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train_sub, y_train_sub, **fit_kwargs)

        if use_eval and X_val_sub is not None and y_val_sub is not None:
            probs_val = self._predict_proba_local(X_val_sub)
            pred_val = np.argmax(probs_val, axis=1)
            self.val_acc = float(np.mean(pred_val == y_val_sub))
        else:
            self.val_acc = float("nan")

        # OOD is computed on an expert-specific standardized feature space.
        self.ood_feature_mean = np.mean(X_train_sub, axis=0).astype(np.float32)
        self.ood_feature_std = (np.std(X_train_sub, axis=0) + 1e-6).astype(np.float32)
        X_train_ood = (X_train_sub - self.ood_feature_mean) / self.ood_feature_std
        self.ood_model = MahalanobisOOD(random_state=self.seed).fit(X_train_ood)
        self.is_trained = True

        del X_sub, y_sub, y_local, X_train_sub, y_train_sub, w_train_sub
        if X_val_sub is not None:
            del X_val_sub, y_val_sub, w_val_sub
        gc.collect()

    def _predict_proba_local(self, X_feat: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(X_feat)
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs]).astype(np.float32)
        return probs

    def extract_outputs(self, X: np.ndarray, batch_size: int = 50000) -> Dict[str, np.ndarray]:
        N = X.shape[0]
        if not self.is_trained or self.model is None or self.local_classes is None:
            uniform = np.ones((N, self.global_num_classes), dtype=np.float32) / self.global_num_classes
            return {
                "global_proba": uniform,
                "pred": np.argmax(uniform, axis=1),
                "margin": np.zeros(N, dtype=np.float32),
                "entropy": np.ones(N, dtype=np.float32),
                "ood": np.full(N, 1e3, dtype=np.float32),
            }

        global_probs = np.zeros((N, self.global_num_classes), dtype=np.float32)
        margin_all = np.zeros(N, dtype=np.float32)
        ent_all = np.zeros(N, dtype=np.float32)
        ood_all = np.zeros(N, dtype=np.float32)

        feat_idx = self.feature_indices
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch_feat = X[start:end][:, feat_idx]
            probs_local = self._predict_proba_local(X_batch_feat)
            for l_idx, g_idx in enumerate(self.local_classes):
                global_probs[start:end, g_idx] = probs_local[:, l_idx]
            margin_all[start:end] = compute_margin(probs_local)
            ent_all[start:end] = compute_local_entropy(probs_local)
            X_batch_ood = (X_batch_feat - self.ood_feature_mean) / self.ood_feature_std
            ood_all[start:end] = self.ood_model.score_samples(X_batch_ood)

        pred = np.argmax(global_probs, axis=1)
        return {
            "global_proba": global_probs,
            "pred": pred,
            "margin": margin_all,
            "entropy": ent_all,
            "ood": ood_all,
        }


class FamilyRouter:
    def __init__(self, num_families: int, device: str = "cpu", seed: int = 42):
        self.num_families = num_families
        self.device = device
        self.seed = seed
        self.model = None
        self.is_trained = False
        self.local_families = None
        self.constant_probs = None

    def fit(self, X: np.ndarray, y_family: np.ndarray):
        y_family = np.asarray(y_family, dtype=int)
        unique_families = np.sort(np.unique(y_family))
        self.local_families = unique_families.astype(int)

        prior = np.zeros(self.num_families, dtype=np.float32)
        counts = np.bincount(y_family, minlength=self.num_families).astype(np.float32)
        if counts.sum() > 0:
            prior = counts / counts.sum()

        if len(unique_families) < 2:
            self.constant_probs = prior
            self.is_trained = False
            return

        mapper = {g: l for l, g in enumerate(unique_families)}
        y_local = np.array([mapper[v] for v in y_family], dtype=int)
        weights = compute_balanced_sample_weights(y_local)
        X_tr, X_va, y_tr, y_va, w_tr, w_va, use_eval = guarded_internal_split_with_weights(
            X, y_local, weights, seed=self.seed, logger=logging.getLogger("experiment"), context="FamilyRouter"
        )

        params = dict(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
        )
        if use_eval:
            params["early_stopping_rounds"] = 20
        if self.device == "cuda":
            params["device"] = "cuda"

        if len(unique_families) == 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "multi:softprob"
            params["eval_metric"] = "mlogloss"
            params["num_class"] = len(unique_families)

        self.model = xgb.XGBClassifier(**params)
        fit_kwargs = dict(sample_weight=w_tr, verbose=False)
        if use_eval and X_va is not None and y_va is not None:
            fit_kwargs["eval_set"] = [(X_va, y_va)]
            fit_kwargs["sample_weight_eval_set"] = [w_va]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_tr, y_tr, **fit_kwargs)
        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.constant_probs is not None:
            return np.tile(self.constant_probs[None, :], (X.shape[0], 1)).astype(np.float32)
        if not self.is_trained or self.model is None or self.local_families is None:
            return np.ones((X.shape[0], self.num_families), dtype=np.float32) / self.num_families

        probs_local = self.model.predict_proba(X)
        probs_local = np.asarray(probs_local, dtype=np.float32)
        if probs_local.ndim == 1:
            probs_local = np.column_stack([1.0 - probs_local, probs_local]).astype(np.float32)

        probs_global = np.zeros((X.shape[0], self.num_families), dtype=np.float32)
        for l_idx, g_idx in enumerate(self.local_families):
            probs_global[:, int(g_idx)] = probs_local[:, l_idx]

        row_sums = probs_global.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 1e-12, None)
        probs_global = probs_global / row_sums
        return probs_global.astype(np.float32)


@dataclass
class CompetenceModel:
    margin_mean: float = 0.0
    margin_std: float = 1.0
    entropy_mean: float = 0.0
    entropy_std: float = 1.0
    ood_mean: float = 0.0
    ood_std: float = 1.0
    clf: Optional[LogisticRegression] = None
    constant_prob: Optional[float] = None
    val_accuracy: Optional[float] = None

    def predict_correct_prob(self, prior: np.ndarray, margin: np.ndarray, entropy: np.ndarray, ood: np.ndarray) -> np.ndarray:
        margin_z = zscore_per_vector(margin, self.margin_mean, self.margin_std)
        entropy_z = zscore_per_vector(entropy, self.entropy_mean, self.entropy_std)
        ood_z = zscore_per_vector(ood, self.ood_mean, self.ood_std)
        X = np.column_stack([prior, margin_z, entropy_z, ood_z]).astype(np.float64)
        if self.clf is not None:
            return self.clf.predict_proba(X)[:, 1].astype(np.float32)
        return np.full(len(prior), float(self.constant_prob), dtype=np.float32)


# -----------------------------
# Main MoE model
# -----------------------------

class ReliabilityRoutedMoE:
    def __init__(
        self,
        num_classes: int,
        dataset_type: str,
        class_names: List[str],
        feature_names: List[str],
        device: str = "cpu",
        seed: int = 42,
        top_k: int = 3,
        temperature: float = 0.7,
        random_experts: int = 3,
        ood_z_threshold: float = 3.0,
    ):
        self.num_classes = num_classes
        self.dataset_type = dataset_type
        self.class_names = list(class_names)
        self.feature_names = list(feature_names)
        self.device = device
        self.seed = seed
        # top_k means the number of *additional specialists* beyond always-included experts.
        self.top_k = max(0, int(top_k))
        self.temperature = temperature
        self.random_experts = max(0, random_experts)
        self.ood_z_threshold = float(ood_z_threshold)

        self.groups = None
        self.class_to_family = None
        self.family_id_to_name = None
        self.feature_views = None
        self.family_router = None
        self.experts: List[ExpertXGB] = []
        self.competence_models: Dict[int, CompetenceModel] = {}
        self.anchor_idx = None
        self.always_include_indices: List[int] = []

    # ---- expert construction
    def build_and_train(self, X_train: np.ndarray, y_train: np.ndarray, logger):
        self.groups, self.class_to_family, self.family_id_to_name = get_taxonomy_groups(
            self.dataset_type, self.class_names, logger
        )
        self.feature_views = build_feature_views(self.feature_names, logger)

        normal_classes = self.groups["Normal"]
        family_labels = np.array([self.class_to_family[int(v)] for v in y_train], dtype=int)
        self.family_router = FamilyRouter(num_families=len(self.family_id_to_name), device=self.device, seed=self.seed)
        logger.info("Training coarse family router...")
        self.family_router.fit(X_train, family_labels)

        expert_configs = self._build_expert_configs(y_train, logger)
        self.experts = []
        self.always_include_indices = []
        for cfg in expert_configs:
            expert = ExpertXGB(cfg, global_num_classes=self.num_classes, device=self.device, seed=self.seed)
            logger.info(
                f"Training expert [{cfg.expert_id:02d}] {cfg.name} | feature_dim={len(cfg.feature_indices)} | target_classes={cfg.target_classes} | extra_negatives={cfg.extra_negative_classes}"
            )
            expert.fit(X_train, y_train, normal_classes=normal_classes, logger=logger)
            if expert.is_trained:
                self.experts.append(expert)
                if cfg.always_include:
                    self.always_include_indices.append(len(self.experts) - 1)
                if cfg.kind == "anchor":
                    self.anchor_idx = len(self.experts) - 1
            else:
                logger.warning(f"Skipped expert {cfg.name} because training was unsuccessful.")

        if self.anchor_idx is None:
            raise RuntimeError("Anchor expert was not trained successfully.")

        logger.info("Final trained experts:")
        for idx, ex in enumerate(self.experts):
            logger.info(
                f"  [{idx:02d}] {ex.name:24s} | local_classes={[self.class_names[c] for c in ex.assigned_classes]} | train_size={ex.train_size} | val_acc={ex.val_acc:.4f}"
            )
        logger.info(
            f"Routing semantics: always include {len(self.always_include_indices)} anchor/generalist experts, "
            f"plus up to {self.top_k} additional specialists per sample after OOD filtering."
        )

    def _build_expert_configs(self, y_train: np.ndarray, logger) -> List[ExpertConfig]:
        counts = np.bincount(y_train, minlength=self.num_classes)
        normal_classes = self.groups["Normal"]
        attack_classes = [c for c in range(self.num_classes) if c not in normal_classes]
        attack_counts = [(c, counts[c]) for c in attack_classes]
        attack_counts = sorted(attack_counts, key=lambda x: x[1])
        n_tail = min(max(2, len(attack_counts) // 3), max(2, len(attack_counts))) if attack_counts else 0
        tail_classes = [c for c, _ in attack_counts[:n_tail]]

        family_class_map = {
            fid: [c for c in self.groups.get(f"Group{fid}", [])]
            for fid in [1, 2, 3, 4]
        }
        family_attack_counts = {
            fid: int(np.sum([counts[c] for c in cls])) for fid, cls in family_class_map.items()
        }

        def choose_hard_negative_classes(target_family_ids: List[int], max_families: int = 2) -> List[int]:
            other_families = [fid for fid in [1, 2, 3, 4] if fid not in target_family_ids and len(family_class_map.get(fid, [])) > 0]
            ranked = sorted(other_families, key=lambda fid: family_attack_counts.get(fid, 0), reverse=True)
            chosen = ranked[:max_families]
            classes = []
            for fid in chosen:
                classes.extend(family_class_map.get(fid, []))
            return sorted(set(classes))

        logger.info("Taxonomy groups:")
        for gname, idxs in self.groups.items():
            logger.info(f"  {gname:8s}: {[self.class_names[i] for i in idxs]}")
        logger.info(f"Tail classes: {[self.class_names[c] for c in tail_classes]}")

        cfgs: List[ExpertConfig] = []
        next_id = 0

        # Anchor global expert
        cfgs.append(
            ExpertConfig(
                expert_id=next_id,
                name="Anchor_Global",
                target_classes=None,
                target_family_ids=[1, 2, 3, 4],
                feature_indices=self.feature_views["all"],
                include_normal=True,
                extra_negative_classes=[],
                use_balanced_weights=False,
                target_boost=1.0,
                normal_weight_scale=1.0,
                extra_negative_weight_scale=1.0,
                always_include=True,
                kind="anchor",
            )
        )
        next_id += 1

        # Balanced generalist
        cfgs.append(
            ExpertConfig(
                expert_id=next_id,
                name="Balanced_Global",
                target_classes=None,
                target_family_ids=[1, 2, 3, 4],
                feature_indices=self.feature_views["all"],
                include_normal=True,
                extra_negative_classes=[],
                use_balanced_weights=True,
                target_boost=1.0,
                normal_weight_scale=1.0,
                extra_negative_weight_scale=1.0,
                always_include=True,
                kind="generalist",
            )
        )
        next_id += 1

        if tail_classes:
            tail_family_ids = sorted({self.class_to_family[c] for c in tail_classes if self.class_to_family[c] != 0})
            tail_hard_negs = choose_hard_negative_classes(tail_family_ids, max_families=2)
            cfgs.append(
                ExpertConfig(
                    expert_id=next_id,
                    name="Tail_Expert",
                    target_classes=tail_classes,
                    target_family_ids=tail_family_ids,
                    feature_indices=self.feature_views["all"],
                    include_normal=True,
                    extra_negative_classes=tail_hard_negs,
                    use_balanced_weights=True,
                    target_boost=2.0,
                    normal_weight_scale=0.4,
                    extra_negative_weight_scale=0.8,
                    always_include=False,
                    kind="tail",
                )
            )
            next_id += 1

        # Taxonomy specialists with family-specific feature views and explicit hard negatives
        family_to_view = {
            "Group1": "volume",
            "Group2": "timing",
            "Group3": "packet",
            "Group4": "tcp",
        }
        for family_name in ["Group1", "Group2", "Group3", "Group4"]:
            target = self.groups[family_name]
            if not target:
                continue
            family_id = int(family_name[-1])
            hard_neg_classes = choose_hard_negative_classes([family_id], max_families=2)
            cfgs.append(
                ExpertConfig(
                    expert_id=next_id,
                    name=f"Spec_{family_name}_{family_to_view[family_name]}",
                    target_classes=target,
                    target_family_ids=[family_id],
                    feature_indices=self.feature_views[family_to_view[family_name]],
                    include_normal=True,
                    extra_negative_classes=hard_neg_classes,
                    use_balanced_weights=True,
                    target_boost=1.6,
                    normal_weight_scale=0.45,
                    extra_negative_weight_scale=0.85,
                    always_include=False,
                    kind="family",
                )
            )
            next_id += 1

        # Random specialists: random attack subsets + random views + one hard-negative family
        rng = np.random.default_rng(self.seed)
        view_keys = ["volume", "timing", "packet", "tcp", "all"]
        if len(attack_classes) >= 2:
            for i in range(self.random_experts):
                subset_size = int(rng.integers(low=2, high=max(3, len(attack_classes) + 1)))
                subset_size = min(subset_size, len(attack_classes))
                rand_classes = sorted(rng.choice(attack_classes, size=subset_size, replace=False).tolist())
                rand_families = sorted({self.class_to_family[c] for c in rand_classes if self.class_to_family[c] != 0})
                hard_neg_classes = choose_hard_negative_classes(rand_families or [1], max_families=1)
                view = str(rng.choice(view_keys))
                cfgs.append(
                    ExpertConfig(
                        expert_id=next_id,
                        name=f"Rand_{i+1}_{view}",
                        target_classes=rand_classes,
                        target_family_ids=rand_families,
                        feature_indices=self.feature_views[view],
                        include_normal=True,
                        extra_negative_classes=hard_neg_classes,
                        use_balanced_weights=True,
                        target_boost=1.25,
                        normal_weight_scale=0.6,
                        extra_negative_weight_scale=0.9,
                        always_include=False,
                        kind="random",
                    )
                )
                next_id += 1

        return cfgs

    # ---- routing / competence
    def fit_router(self, X_val: np.ndarray, y_val: np.ndarray, logger, max_meta_samples: int = 200000):
        logger.info("Fitting expert-specific competence models...")
        family_probs = self.family_router.predict_proba(X_val)
        expert_outputs = [ex.extract_outputs(X_val) for ex in self.experts]

        self.competence_models = {}
        rng = np.random.default_rng(self.seed)
        for j, ex in enumerate(self.experts):
            outputs = expert_outputs[j]
            prior = self._compute_expert_prior(ex, family_probs)
            margin = outputs["margin"]
            entropy = outputs["entropy"]
            ood = outputs["ood"]
            labels = (outputs["pred"] == y_val).astype(int)

            cm = CompetenceModel(
                margin_mean=float(np.mean(margin)),
                margin_std=float(np.std(margin) + 1e-6),
                entropy_mean=float(np.mean(entropy)),
                entropy_std=float(np.std(entropy) + 1e-6),
                ood_mean=float(np.mean(ood)),
                ood_std=float(np.std(ood) + 1e-6),
                val_accuracy=float(np.mean(labels)),
            )

            margin_z = zscore_per_vector(margin, cm.margin_mean, cm.margin_std)
            entropy_z = zscore_per_vector(entropy, cm.entropy_mean, cm.entropy_std)
            ood_z = zscore_per_vector(ood, cm.ood_mean, cm.ood_std)
            X_meta = np.column_stack([prior, margin_z, entropy_z, ood_z]).astype(np.float64)
            y_meta = labels.astype(int)

            if len(np.unique(y_meta)) < 2:
                cm.constant_prob = float(np.mean(y_meta))
                self.competence_models[j] = cm
                logger.info(
                    f"  [{ex.name}] constant competence={cm.constant_prob:.4f} (degenerate target)."
                )
                continue

            if len(y_meta) > max_meta_samples:
                idx0 = np.where(y_meta == 0)[0]
                idx1 = np.where(y_meta == 1)[0]
                n0 = min(len(idx0), max_meta_samples // 2)
                n1 = min(len(idx1), max_meta_samples - n0)
                keep = np.concatenate(
                    [rng.choice(idx0, size=n0, replace=False), rng.choice(idx1, size=n1, replace=False)]
                )
                rng.shuffle(keep)
                X_fit = X_meta[keep]
                y_fit = y_meta[keep]
            else:
                X_fit = X_meta
                y_fit = y_meta

            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_fit, y_fit)
            cm.clf = clf
            self.competence_models[j] = cm
            coef = clf.coef_[0]
            logger.info(
                f"  [{ex.name}] val_acc={cm.val_accuracy:.4f} | coef(prior,margin,entropy,ood)=({coef[0]:.3f}, {coef[1]:.3f}, {coef[2]:.3f}, {coef[3]:.3f})"
            )

    def _compute_expert_prior(self, expert: ExpertXGB, family_probs: np.ndarray) -> np.ndarray:
        target_families = list(expert.config.target_family_ids)
        if target_families:
            prior = np.sum(family_probs[:, target_families], axis=1)
        else:
            prior = 1.0 - family_probs[:, 0]
        return np.asarray(prior, dtype=np.float32)

    def _collect_routing_state(self, X: np.ndarray, batch_size: int = 50000):
        family_probs = self.family_router.predict_proba(X)
        expert_outputs = [ex.extract_outputs(X, batch_size=batch_size) for ex in self.experts]
        N = X.shape[0]
        num_e = len(self.experts)
        competence = np.zeros((N, num_e), dtype=np.float32)
        probs_list = []
        ood_z_all = np.zeros((N, num_e), dtype=np.float32)

        for j, ex in enumerate(self.experts):
            out = expert_outputs[j]
            probs_list.append(out["global_proba"])
            cm = self.competence_models[j]
            prior = self._compute_expert_prior(ex, family_probs)
            competence[:, j] = cm.predict_correct_prob(prior, out["margin"], out["entropy"], out["ood"])
            ood_z_all[:, j] = zscore_per_vector(out["ood"], cm.ood_mean, cm.ood_std)
        return competence, probs_list, ood_z_all

    def predict(self, X: np.ndarray, batch_size: int = 50000, return_info: bool = False):
        competence, probs_list, ood_z_all = self._collect_routing_state(X, batch_size=batch_size)
        N = X.shape[0]
        num_e = len(self.experts)

        # always include anchor/generalists; shortlist up to self.top_k additional specialists by competence
        candidate_mask = np.zeros((N, num_e), dtype=bool)
        candidate_mask[:, self.always_include_indices] = True

        for j, ex in enumerate(self.experts):
            if j in self.always_include_indices:
                continue
            # hard OOD mask only for specialists
            candidate_mask[:, j] = ood_z_all[:, j] <= self.ood_z_threshold

        # choose top specialists among valid candidates
        scores_for_rank = competence.copy()
        for j in range(num_e):
            if j in self.always_include_indices:
                scores_for_rank[:, j] = -1e9
            invalid = ~candidate_mask[:, j]
            scores_for_rank[invalid, j] = -1e9

        n_extra = self.top_k
        if n_extra > 0:
            ranked = np.argsort(scores_for_rank, axis=1)[:, ::-1]
            rows = np.arange(N)
            for r in range(min(n_extra, ranked.shape[1])):
                cols = ranked[:, r]
                vals = scores_for_rank[rows, cols]
                valid_rows = vals > -1e8
                candidate_mask[rows[valid_rows], cols[valid_rows]] = True

        # competence probabilities -> logits for sharper soft weighting
        p = np.clip(competence, 1e-5, 1.0 - 1e-5)
        comp_logits = np.log(p / (1.0 - p))
        comp_logits[~candidate_mask] = -1e9
        weights = row_softmax(comp_logits, temperature=self.temperature)

        final_proba = np.zeros((N, self.num_classes), dtype=np.float32)
        for j in range(num_e):
            final_proba += weights[:, [j]] * probs_list[j]

        preds = np.argmax(final_proba, axis=1)
        if return_info:
            selected = np.argmax(weights, axis=1)
            return preds, {
                "weights": weights,
                "competence": competence,
                "selected_expert": selected,
                "candidate_mask": candidate_mask,
            }
        return preds


# -----------------------------
# Splitting
# -----------------------------

def make_splits(data: dict, X: np.ndarray, y: np.ndarray, seed: int, logger, class_to_family: Optional[Dict[int, int]] = None):
    group_keys = ["scenario_ids", "group_ids", "groups", "fold_ids", "file_groups"]
    ts_keys = ["timestamps", "timestamp", "time", "times"]
    dataset_type = str(data.get("dataset_type", "")).lower()

    def _log_coverage(prefix, y_tr, y_va, y_te):
        ok, details = check_split_coverage(y, y_tr, y_va, y_te, class_to_family=class_to_family)
        if ok:
            logger.info(f"{prefix} split coverage check passed.")
        else:
            logger.warning(
                f"{prefix} split coverage issue | "
                f"missing_train_classes={details['missing_train_classes']} | "
                f"missing_train_families={details['missing_train_families']} | "
                f"missing_val_families={details['missing_val_families']}"
            )
        return ok

    explicit_split_keys = ["train_indices", "val_indices", "test_indices"]
    if all(key in data and data[key] is not None for key in explicit_split_keys):
        logger.info("Using explicit train/val/test indices from pickle.")
        train_idx = np.asarray(data["train_indices"], dtype=int)
        val_idx = np.asarray(data["val_indices"], dtype=int)
        test_idx = np.asarray(data["test_indices"], dtype=int)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        _log_coverage("Explicit-index", y_train, y_val, y_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    if dataset_type == "unswnb15":
        logger.info("UNSW-NB15 detected: skipping file-aware split.")

        if "train_indices" in data and "test_indices" in data:
            logger.info("Using predefined train/test split from pickle for UNSW-NB15.")
            trainval_idx = np.asarray(data["train_indices"], dtype=int)
            test_idx = np.asarray(data["test_indices"], dtype=int)

            X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            X_train = X_val = y_train = y_val = None
            for trial in range(10):
                rs = seed + trial
                X_train, X_val, y_train, y_val = safe_train_test_split(
                    X_trainval, y_trainval, test_size=0.2, random_state=rs, stratify=True
                )
                if _log_coverage("UNSW predefined", y_train, y_val, y_test):
                    return X_train, X_val, X_test, y_train, y_val, y_test

            logger.warning("UNSW predefined split did not pass coverage check; returning last attempted split.")
            return X_train, X_val, X_test, y_train, y_val, y_test

    if dataset_type == "nfunswnb15":
        logger.info("NF-UNSW-NB15 detected: using stratified random split.")

    for key in group_keys:
        if key in data and data[key] is not None and len(data[key]) == len(y):
            groups = np.asarray(data[key])
            n_unique_groups = len(np.unique(groups))
            if n_unique_groups < 2:
                logger.warning(
                    f"Skipping group-aware split via pickle['{key}'] because only "
                    f"{n_unique_groups} unique group is available."
                )
                continue

            logger.info(f"Using file-aware split via pickle['{key}']. ({n_unique_groups} unique groups)")
            for trial in range(10):
                rs = seed + trial
                gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                trainval_idx, test_idx = next(gss1.split(X, y, groups))
                X_trainval, y_trainval, groups_trainval = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
                if len(np.unique(groups_trainval)) < 2:
                    logger.warning(
                        f"Trial {trial + 1}: train/val candidate has fewer than 2 unique groups; retrying."
                    )
                    continue
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                train_idx_rel, val_idx_rel = next(gss2.split(X_trainval, y_trainval, groups_trainval))
                X_train, y_train = X_trainval[train_idx_rel], y_trainval[train_idx_rel]
                X_val, y_val = X_trainval[val_idx_rel], y_trainval[val_idx_rel]
                X_test, y_test = X[test_idx], y[test_idx]
                if _log_coverage("Group-aware", y_train, y_val, y_test):
                    return X_train, X_val, X_test, y_train, y_val, y_test
            logger.warning("Unable to find a file-aware split with adequate class coverage; falling back.")
            break

    # for key in ts_keys:
    #     if key in data and data[key] is not None and len(data[key]) == len(y):
    #         logger.info(f"Using chronological blocked split via pickle['{key}'].")
    #         ts = np.asarray(data[key])
    #         order = np.argsort(ts)
    #         X_sorted, y_sorted = X[order], y[order]
    #         n = len(y_sorted)
    #         n_train = int(0.64 * n)
    #         n_val = int(0.16 * n)
    #         X_train = X_sorted[:n_train]
    #         y_train = y_sorted[:n_train]
    #         X_val = X_sorted[n_train : n_train + n_val]
    #         y_val = y_sorted[n_train : n_train + n_val]
    #         X_test = X_sorted[n_train + n_val :]
    #         y_test = y_sorted[n_train + n_val :]
    #         if _log_coverage("Chronological", y_train, y_val, y_test):
    #             return X_train, X_val, X_test, y_train, y_val, y_test
    #         logger.warning("Chronological split did not preserve sufficient training/validation class coverage; falling back.")
    #         break

    logger.warning(
        "No valid scenario/group/timestamp split with adequate class/family coverage found. Falling back to stratified random split. For CICIDS papers, blocked or scenario-aware splits are preferred."
    )
    X_train = X_val = X_test = y_train = y_val = y_test = None
    for trial in range(10):
        rs = seed + trial
        X_temp, X_test, y_temp, y_test = safe_train_test_split(X, y, test_size=0.2, random_state=rs, stratify=True)
        X_train, X_val, y_train, y_val = safe_train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=rs, stratify=True
        )
        if _log_coverage("Stratified-random", y_train, y_val, y_test):
            return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------
# Reporting helpers
# -----------------------------

def summarize_selection(info: dict, experts: List[ExpertXGB]) -> pd.DataFrame:
    selected = info["selected_expert"]
    weights = info["weights"]
    rows = []
    for j, ex in enumerate(experts):
        rows.append(
            {
                "expert_idx": j,
                "expert_name": ex.name,
                "selected_rate": float(np.mean(selected == j)),
                "avg_weight": float(np.mean(weights[:, j])),
                "val_acc": ex.val_acc,
                "assigned_classes": ", ".join(str(c) for c in ex.assigned_classes),
            }
        )
    return pd.DataFrame(rows)


def log_split_distribution(logger, split_name: str, y_split: np.ndarray, class_names: List[str]):
    y_split = np.asarray(y_split, dtype=int)
    counts = np.bincount(y_split, minlength=len(class_names))
    total = max(int(counts.sum()), 1)
    logger.info(f"{split_name} distribution (n={total}):")
    for class_id, count in enumerate(counts):
        if count == 0:
            continue
        pct = 100.0 * float(count) / float(total)
        logger.info(f"  [{class_id:02d}] {class_names[class_id]:30s}: {count:8d} ({pct:6.2f}%)")


# -----------------------------
# main
# -----------------------------

def main():
    start_time = time.time()
    gc.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to pickle data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--top_k", type=int, default=3, help="Number of additional specialists beyond the always-included anchor/generalist experts")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--random_experts", type=int, default=3)
    parser.add_argument("--ood_z_threshold", type=float, default=3.0)
    parser.add_argument(
        "--model",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="0: baseline only, 1: proposed only, 2: both",
    )
    args = parser.parse_args()

    exp_dir = f"results/logs_tailguard_v9_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(exp_dir)
    logger.info(f"XGBoost device: {DEVICE}")

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    dataset_type = infer_dataset_type(args.data, data)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)
    feature_names = get_feature_names(data, X.shape[1])
    class_names = data["label_encoder"].classes_ if "label_encoder" in data else [f"Class_{i}" for i in range(len(np.unique(y)))]
    class_names = [str(x) for x in class_names]
    num_classes = len(np.unique(y))

    logger.info(f"Dataset type: {dataset_type}")
    logger.info(f"X shape: {X.shape}, num_classes: {num_classes}")
    logger.info(f"Class names: {class_names}")

    _, class_to_family, family_id_to_name = get_taxonomy_groups(dataset_type, class_names, logger)
    logger.info(f"Family IDs: {family_id_to_name}")

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
        data, X, y, args.seed, logger, class_to_family=class_to_family
    )
    logger.info("\n=== Split Summary Before Training ===")
    log_split_distribution(logger, "train", y_train, class_names)
    log_split_distribution(logger, "val", y_val, class_names)
    log_split_distribution(logger, "test", y_test, class_names)
    logger.info("Entering pdb before training. Use `c` to continue.")
    pdb.set_trace()
    del data, X, y
    gc.collect()

    models = {}

    # Baseline: single global XGBoost
    if args.model in (0, 2):
        logger.info("\n=== Training baseline single XGBoost ===")
        baseline_cfg = ExpertConfig(
            expert_id=0,
            name="Baseline_Global",
            target_classes=None,
            target_family_ids=[1, 2, 3, 4],
            feature_indices=np.arange(X_train.shape[1], dtype=int),
            include_normal=True,
            use_balanced_weights=False,
            target_boost=1.0,
            normal_weight_scale=1.0,
            always_include=True,
            kind="anchor",
        )
        baseline = ExpertXGB(baseline_cfg, global_num_classes=num_classes, device=DEVICE, seed=args.seed)
        groups, _, _ = get_taxonomy_groups(dataset_type, class_names, logger)
        baseline.fit(X_train, y_train, normal_classes=groups["Normal"], logger=logger)
        models["baseline"] = baseline

    if args.model in (1, 2):
        logger.info("\n=== Training FAR-MoE ===")
        moe = ReliabilityRoutedMoE(
            num_classes=num_classes,
            dataset_type=dataset_type,
            class_names=class_names,
            feature_names=feature_names,
            device=DEVICE,
            seed=args.seed,
            top_k=args.top_k,
            temperature=args.temperature,
            random_experts=args.random_experts,
            ood_z_threshold=args.ood_z_threshold,
        )
        moe.build_and_train(X_train, y_train, logger)
        moe.fit_router(X_val, y_val, logger)
        models["moe"] = moe

    logger.info("\n" + "=" * 72)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 72)

    if "baseline" in models:
        t0 = time.time()
        base_out = models["baseline"].extract_outputs(X_test, batch_size=args.batch_size)
        base_preds = base_out["pred"]
        acc = accuracy_score(y_test, base_preds)
        logger.info(f"\n--- Baseline Single XGBoost ---")
        logger.info(f"Accuracy: {acc:.4f} | eval time: {time.time() - t0:.2f}s")
        logger.info("\n" + classification_report(y_test, base_preds, labels=np.arange(num_classes), target_names=class_names, digits=4, zero_division=0))

    if "moe" in models:
        t0 = time.time()
        moe_preds, route_info = models["moe"].predict(X_test, batch_size=args.batch_size, return_info=True)
        acc = accuracy_score(y_test, moe_preds)
        logger.info(f"\n--- FAR-MoE ---")
        logger.info(f"Accuracy: {acc:.4f} | eval time: {time.time() - t0:.2f}s")
        logger.info("\n" + classification_report(y_test, moe_preds, labels=np.arange(num_classes), target_names=class_names, digits=4, zero_division=0))

        selection_df = summarize_selection(route_info, models["moe"].experts)
        selection_path = os.path.join(exp_dir, "expert_selection_summary.csv")
        selection_df.to_csv(selection_path, index=False)
        logger.info(f"Saved expert selection summary to {selection_path}")

        # per-class summary for baseline vs moe when both are present
        if "baseline" in models:
            unique_classes = np.arange(num_classes)
            p_b, r_b, f1_b, _ = precision_recall_fscore_support(
                y_test, base_preds, labels=unique_classes, zero_division=0
            )
            p_m, r_m, f1_m, _ = precision_recall_fscore_support(
                y_test, moe_preds, labels=unique_classes, zero_division=0
            )
            df = pd.DataFrame(
                {
                    "class_id": unique_classes,
                    "class_name": class_names,
                    "baseline_precision": p_b,
                    "baseline_recall": r_b,
                    "baseline_f1": f1_b,
                    "moe_precision": p_m,
                    "moe_recall": r_m,
                    "moe_f1": f1_m,
                    "delta_f1": f1_m - f1_b,
                }
            )
            per_class_path = os.path.join(exp_dir, "baseline_vs_moe_per_class.csv")
            df.to_csv(per_class_path, index=False)
            logger.info(f"Saved per-class comparison to {per_class_path}")

    logger.info(f"\nTotal elapsed time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()

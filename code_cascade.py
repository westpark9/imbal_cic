#!/usr/bin/env python3
"""
code_cascade.py  –  2-Stage Hard Cascade Classifier
=====================================================
code_mat.py 의 MoE 구조를 제거하고, 2단계 하드 캐스케이드로 교체.

변경 사항:
  1. --random_split 플래그: split_tags / timestamps 무시 → stratified random split 강제
     → val confusion matrix가 에피소드 배정에 의존하지 않도록 함
  2. MoE (FamilyRouter, CompetenceModel, ReliabilityRoutedMoE) 제거
  3. CascadeClassifier 추가:
       Stage 1 : 전체 클래스 단일 XGBoost (baseline 동일)
       Stage 2 : confusion matrix 기반 focus 클래스 + normal 전문 XGBoost
       라우팅   : Stage 1 예측이 focus 클래스 → Stage 2 로 교체
                  나머지 샘플은 Stage 1 예측 그대로 사용
"""

import argparse
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────
# Utilities  (code_mat.py 와 동일)
# ──────────────────────────────────────────────

def infer_dataset_type(data_path, data=None):
    if data is not None and isinstance(data, dict) and "dataset_type" in data:
        return str(data["dataset_type"]).lower()
    name = os.path.basename(data_path).lower()
    if "unsw" in name or "nb15" in name:
        return "unswnb15"
    if "2017" in name:
        return "cic2017"
    if "2018" in name:
        return "cic2018"
    raise ValueError(f"Cannot infer dataset_type from '{data_path}'.")


def detect_dataset_type_from_dir(data_dir: str) -> str:
    folder_name = os.path.basename(data_dir.rstrip("/\\")).lower()
    if "nf-unsw" in folder_name or "nf_unsw" in folder_name or "nfunsw" in folder_name:
        return "nfunswnb15"
    if "unsw" in folder_name or "nb15" in folder_name:
        return "unswnb15"
    if "2017" in folder_name:
        return "cic2017"
    if "2018" in folder_name:
        return "cic2018"
    raise ValueError("Folder name doesn't contain '2017', '2018', or 'unsw/nb15'")


def find_label_column(df: pd.DataFrame, dataset_type: Optional[str] = None) -> Optional[str]:
    if dataset_type == "nfunswnb15":
        if "Attack" in df.columns:
            return "Attack"
        for col in df.columns:
            if col.strip().lower() == "attack":
                return col
        return None
    if dataset_type == "unswnb15":
        if "attack_cat" in df.columns:
            return "attack_cat"
        for col in df.columns:
            if col.strip().lower() == "attack_cat":
                return col
        return None
    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    return None


def remove_dataset_columns(df: pd.DataFrame, dataset_type: str):
    columns_to_remove = {
        "cic2017": ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"],
        "cic2018": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"],
        "unswnb15": ["id", "rate", "label"],
        "nfunswnb15": ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"],
    }
    for col_to_remove in columns_to_remove.get(dataset_type, []):
        for col in df.columns:
            if col.strip().lower() == col_to_remove.lower():
                df.drop(columns=[col], inplace=True)
                break


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype("int32")
    return df


def preprocess_single_csv_frame(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    df = df.copy()
    label_col = find_label_column(df, dataset_type)
    if dataset_type in ["unswnb15", "nfunswnb15"] and label_col:
        df[label_col] = df[label_col].fillna("normal")
        df[label_col] = df[label_col].replace(["-", " -", "- "], "normal")
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    if label_col:
        if dataset_type in ["cic2017", "cic2018"]:
            header_mask = df[label_col].astype(str).str.strip().str.lower() == "label"
        else:
            header_mask = df[label_col].astype(str).str.strip().str.lower().isin(["attack_cat", "attack"])
        if header_mask.any():
            df = df.loc[~header_mask].copy()
    target_columns = [col for col in df.columns if col != label_col]
    object_cols = list(df[target_columns].select_dtypes(include=["object", "category"]).columns)
    for col in target_columns:
        if col in object_cols:
            df[col] = df[col].replace(["-", " -", "- "], "Unknown")
            df[col] = df[col].fillna("Unknown")
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_df = df[target_columns].select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        keep_mask = ~df[numeric_df.columns].isna().any(axis=1)
        df = df.loc[keep_mask].copy()
    return df.reset_index(drop=True)


def ratio_counts_round(n: int) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    n_train = int(round(0.6 * n))
    n_val = int(round(0.2 * n))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = max(0, n - n_train)
    return n_train, n_val, n_test


def normalize_labels_for_split(series: pd.Series) -> np.ndarray:
    out = series.astype(str).str.lower().str.strip()
    out = out.str.replace(" ", "-", regex=False)
    out = out.str.replace("–", "-", regex=False)
    out = out.replace({"benign": "normal", "-": "normal"})
    return out.to_numpy(dtype=object)


def assign_normal_ratio_split(labels_norm: np.ndarray) -> np.ndarray:
    n = len(labels_norm)
    split = np.full(n, "train", dtype=object)
    normal_idx = np.where(labels_norm == "normal")[0]
    n_norm = len(normal_idx)
    if n_norm == 0:
        return split
    n_tr, n_va, n_te = ratio_counts_round(n_norm)
    a = 0
    b = min(n_norm, a + n_tr)
    c = min(n_norm, b + n_va)
    split[normal_idx[a:b]] = "train"
    split[normal_idx[b:c]] = "val"
    split[normal_idx[c:]] = "test"
    return split


def apply_attack_episode_ratio_split(local_split: np.ndarray, labels_norm: np.ndarray) -> np.ndarray:
    split = local_split.copy()
    n = len(labels_norm)
    if n == 0:
        return split
    episodes = []
    i = 0
    while i < n:
        if labels_norm[i] == "normal":
            i += 1
            continue
        j = i + 1
        while j < n and labels_norm[j] == labels_norm[i]:
            j += 1
        episodes.append((i, j, labels_norm[i], j - i))
        i = j
    if not episodes:
        return split
    labels_order = sorted(set(ep[2] for ep in episodes))
    for attack_label in labels_order:
        eps = [ep for ep in episodes if ep[2] == attack_label]
        short_eps = []
        for s, e, _, length in eps:
            if length >= 5:
                n_tr, n_va, n_te = ratio_counts_round(length)
                a = s
                b = min(e, a + n_tr)
                c = min(e, b + n_va)
                split[a:b] = "train"
                split[b:c] = "val"
                split[c:e] = "test"
            else:
                short_eps.append((s, e, length))
        if short_eps:
            total = int(sum(x[2] for x in short_eps))
            t_tr, t_va, t_te = ratio_counts_round(total)
            a_tr = a_va = a_te = 0
            for s, e, length in short_eps:
                rem = {"train": t_tr - a_tr, "val": t_va - a_va, "test": t_te - a_te}
                if rem["train"] > 0:
                    chosen = "train"
                elif rem["val"] > 0:
                    chosen = "val"
                elif rem["test"] > 0:
                    chosen = "test"
                else:
                    chosen = "test"
                split[s:e] = chosen
                if chosen == "train":
                    a_tr += length
                elif chosen == "val":
                    a_va += length
                else:
                    a_te += length
    return split


def load_and_preprocess_from_csv_dir(data_dir: str, logger):
    dataset_type = detect_dataset_type_from_dir(data_dir)
    csv_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    if dataset_type not in ["cic2017", "cic2018"]:
        raise ValueError(f"CSV file-wise split is intended for CIC2017/2018, got {dataset_type}.")

    logger.info("CSV split config: attack=episode 6:2:2, normal=chronological 6:2:2")
    all_frames, source_file_ids, split_tags, timestamps, per_file_counts = [], [], [], [], []

    for file_idx, file_name in enumerate(csv_files):
        file_path = os.path.join(data_dir, file_name)
        logger.info(f"Loading [{file_idx + 1}/{len(csv_files)}]: {file_name}")
        df = pd.read_csv(file_path, low_memory=False)
        df = downcast_dtypes(df)
        remove_dataset_columns(df, dataset_type)
        df = preprocess_single_csv_frame(df, dataset_type)
        label_col = find_label_column(df, dataset_type)
        if not label_col or len(df) == 0:
            logger.warning(f"Skipping {file_name}: no valid rows or label column.")
            continue
        n_rows = len(df)
        labels_norm = normalize_labels_for_split(df[label_col])
        local_split = assign_normal_ratio_split(labels_norm)
        local_split = apply_attack_episode_ratio_split(local_split, labels_norm)
        train_n = int(np.sum(local_split == "train"))
        val_n = int(np.sum(local_split == "val"))
        test_n = int(np.sum(local_split == "test"))
        per_file_counts.append((file_name, train_n, val_n, test_n))
        all_frames.append(df)
        source_file_ids.append(np.full(n_rows, file_idx, dtype=np.int32))
        split_tags.append(local_split)
        timestamps.append(np.arange(n_rows, dtype=np.int64) + (file_idx * 10 ** 9))

    if not all_frames:
        raise ValueError("No usable rows found across CSV files.")
    for file_name, tr_n, va_n, te_n in per_file_counts:
        logger.info(f"Per-file split rows | {file_name}: train={tr_n}, val={va_n}, test={te_n}")

    combined_df = pd.concat(all_frames, ignore_index=True)
    source_file_ids = np.concatenate(source_file_ids)
    split_tags = np.concatenate(split_tags)
    timestamps = np.concatenate(timestamps)

    label_col = find_label_column(combined_df, dataset_type)
    feature_columns = [c for c in combined_df.columns if c not in [label_col, "Label_encoded"]]

    combined_df[label_col] = combined_df[label_col].astype(str).str.lower().str.strip()
    combined_df[label_col] = combined_df[label_col].str.replace(" ", "-", regex=False)
    combined_df[label_col] = combined_df[label_col].str.replace("–", "-", regex=False)
    combined_df[label_col] = combined_df[label_col].replace({"benign": "normal"})

    cat_cols = list(combined_df[feature_columns].select_dtypes(include=["object", "category"]).columns)
    if cat_cols:
        logger.info(f"Frequency-encoding categorical columns: {cat_cols}")
        for col in cat_cols:
            freq_enc = combined_df[col].value_counts(normalize=True)
            combined_df[col] = combined_df[col].map(freq_enc).fillna(0.0).astype("float32")

    numeric_cols = combined_df[feature_columns].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        combined_df[numeric_cols] = combined_df[numeric_cols].clip(lower=-1e12, upper=1e12).fillna(0)

    for col in feature_columns:
        if col in combined_df.columns and not pd.api.types.is_numeric_dtype(combined_df[col]):
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").fillna(0)

    var_ser = combined_df[feature_columns].var()
    zero_var_cols = var_ser[var_ser == 0].index.tolist()
    if zero_var_cols:
        logger.info(f"Removed {len(zero_var_cols)} zero-variance feature columns.")
        feature_columns = [c for c in feature_columns if c not in zero_var_cols]

    label_encoder = LabelEncoder()
    combined_df["Label_encoded"] = label_encoder.fit_transform(combined_df[label_col])

    X = combined_df[feature_columns].values.astype(np.float32)
    y = combined_df["Label_encoded"].values.astype("int32")

    return {
        "X": X,
        "y": y,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "dataset_type": dataset_type,
        "split_tags": split_tags,
        "source_file_id": source_file_ids,
        "timestamps": timestamps,
        "source_files": csv_files,
    }


def split_by_preassigned_tags(X: np.ndarray, y: np.ndarray, split_tags: np.ndarray):
    split_tags = np.asarray(split_tags)
    train_mask = split_tags == "train"
    val_mask = split_tags == "val"
    test_mask = split_tags == "test"
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Preassigned split tags must include train/val/test samples.")
    return X[train_mask], X[val_mask], X[test_mask], y[train_mask], y[val_mask], y[test_mask]


def log_split_class_coverage(y_all, y_train, y_val, y_test, class_names, logger):
    all_classes = set(np.unique(y_all).tolist())
    train_classes = set(np.unique(y_train).tolist())
    val_classes = set(np.unique(y_val).tolist())
    test_classes = set(np.unique(y_test).tolist())
    missing_train = sorted(all_classes - train_classes)
    missing_val = sorted(all_classes - val_classes)
    missing_test = sorted(all_classes - test_classes)
    logger.info(
        f"Split class coverage | missing_in_train={missing_train} | "
        f"missing_in_val={missing_val} | missing_in_test={missing_test}"
    )
    if missing_train:
        missing_names = [class_names[i] for i in missing_train if i < len(class_names)]
        logger.warning(f"Classes absent from train (cannot be learned): {missing_names}")


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


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def get_feature_names(data: dict, n_features: int) -> List[str]:
    for key in ["feature_names", "columns", "feature_columns", "col_names"]:
        if key in data and data[key] is not None:
            vals = list(data[key])
            if len(vals) == n_features:
                return [str(v) for v in vals]
    return [f"f_{i}" for i in range(n_features)]


def compute_balanced_sample_weights(y_local: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_local)
    counts = np.maximum(counts, 1)
    n = len(y_local)
    k = len(counts)
    class_weights = n / (k * counts)
    return class_weights[y_local].astype(np.float32)


# ──────────────────────────────────────────────
# make_splits  (code_mat.py 와 동일)
# ──────────────────────────────────────────────

def make_splits(data: dict, X: np.ndarray, y: np.ndarray, seed: int, logger):
    group_keys = ["scenario_ids", "group_ids", "groups", "fold_ids"]
    ts_keys = ["timestamps", "timestamp", "time", "times"]

    for key in group_keys:
        if key in data and data[key] is not None and len(data[key]) == len(y):
            logger.info(f"Using group-aware split via pickle['{key}'].")
            groups = np.asarray(data[key])
            for trial in range(10):
                rs = seed + trial
                gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                trainval_idx, test_idx = next(gss1.split(X, y, groups))
                X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
                groups_tv = groups[trainval_idx]
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                train_idx_rel, val_idx_rel = next(gss2.split(X_trainval, y_trainval, groups_tv))
                X_train, y_train = X_trainval[train_idx_rel], y_trainval[train_idx_rel]
                X_val, y_val = X_trainval[val_idx_rel], y_trainval[val_idx_rel]
                X_test, y_test = X[test_idx], y[test_idx]
                if len(set(np.unique(y_train)) & set(np.unique(y_val))) >= 2:
                    return X_train, X_val, X_test, y_train, y_val, y_test
            break

    for key in ts_keys:
        if key in data and data[key] is not None and len(data[key]) == len(y):
            logger.info(f"Using chronological blocked split via pickle['{key}'].")
            ts = np.asarray(data[key])
            order = np.argsort(ts)
            X_sorted, y_sorted = X[order], y[order]
            n = len(y_sorted)
            n_train = int(0.64 * n)
            n_val = int(0.16 * n)
            return (
                X_sorted[:n_train], X_sorted[n_train:n_train + n_val], X_sorted[n_train + n_val:],
                y_sorted[:n_train], y_sorted[n_train:n_train + n_val], y_sorted[n_train + n_val:],
            )

    logger.info("Using stratified random split (6:2:2).")
    for trial in range(10):
        rs = seed + trial
        X_temp, X_test, y_temp, y_test = safe_train_test_split(X, y, test_size=0.2, random_state=rs)
        X_train, X_val, y_train, y_val = safe_train_test_split(X_temp, y_temp, test_size=0.25, random_state=rs)
        if len(np.unique(y_train)) >= 2 and len(np.unique(y_val)) >= 2:
            return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────
# ExpertXGB  (code_mat.py 와 동일, 라우팅 관련 불필요 코드 제외)
# ──────────────────────────────────────────────

@dataclass
class ExpertConfig:
    expert_id: int
    name: str
    target_classes: Optional[List[int]] = None
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


def guarded_internal_split_with_weights(X, y, w, seed, logger, context, test_size=0.1, max_tries=12):
    X, y, w = np.asarray(X), np.asarray(y), np.asarray(w)
    unique_classes = np.sort(np.unique(y))
    n_unique = len(unique_classes)
    if n_unique < 2:
        return X, None, y, None, w, None, False

    counts = np.bincount(y)
    too_small = np.any(counts[counts > 0] < 2) or len(y) < max(12, 2 * n_unique)
    if not too_small:
        for stratify in [True, False]:
            for rs in [seed + t for t in range(max_tries)]:
                try:
                    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
                        X, y, w, test_size=test_size,
                        stratify=(y if stratify else None), random_state=rs,
                    )
                except ValueError:
                    continue
                if len(np.unique(y_tr)) == n_unique and len(np.unique(y_va)) >= min(2, n_unique):
                    return X_tr, X_va, y_tr, y_va, w_tr, w_va, True

    logger.warning(f"[{context}] Cannot form safe internal val split; training on full subset without early stopping.")
    return X, None, y, None, w, None, False


class ExpertXGB:
    def __init__(self, config: ExpertConfig, global_num_classes: int, device: str = "cpu", seed: int = 42):
        self.config = config
        self.global_num_classes = global_num_classes
        self.device = device
        self.seed = seed
        self.model = None
        self.local_classes = None
        self.is_trained = False
        self.train_size = 0
        self.val_acc = None

    @property
    def name(self):
        return self.config.name

    def _prepare_subset(self, X, y, normal_classes):
        if self.config.target_classes is None:
            mask = np.ones(len(y), dtype=bool)
        else:
            keep = set(self.config.target_classes)
            if self.config.include_normal:
                keep.update(normal_classes)
            if self.config.extra_negative_classes:
                keep.update(self.config.extra_negative_classes)
            mask = np.isin(y, sorted(keep))
        X_sub = X[mask][:, self.config.feature_indices]
        y_sub = y[mask]
        return X_sub, y_sub, mask

    def fit(self, X, y, normal_classes, logger):
        X_sub, y_sub, _ = self._prepare_subset(X, y, normal_classes)
        self.train_size = len(y_sub)
        if len(y_sub) == 0:
            logger.warning(f"[{self.name}] Empty training subset; skipping.")
            return
        self.local_classes = np.sort(np.unique(y_sub))
        if len(self.local_classes) < 2:
            logger.warning(f"[{self.name}] Only one class; skipping.")
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
                extra_mask = np.isin(y_sub, self.config.extra_negative_classes)
                weights[extra_mask] *= self.config.extra_negative_weight_scale

        X_tr, X_va, y_tr, y_va, w_tr, w_va, use_eval = guarded_internal_split_with_weights(
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
        fit_kwargs = dict(sample_weight=w_tr, verbose=False)
        if use_eval and X_va is not None:
            fit_kwargs["eval_set"] = [(X_va, y_va)]
            fit_kwargs["sample_weight_eval_set"] = [w_va]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_tr, y_tr, **fit_kwargs)

        if use_eval and X_va is not None:
            probs_val = self._predict_proba_local(X_va)
            self.val_acc = float(np.mean(np.argmax(probs_val, axis=1) == y_va))
        else:
            self.val_acc = float("nan")

        self.is_trained = True
        del X_sub, y_sub, y_local, X_tr, y_tr, w_tr
        if X_va is not None:
            del X_va, y_va, w_va
        gc.collect()

    def _predict_proba_local(self, X_feat):
        probs = self.model.predict_proba(X_feat)
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs]).astype(np.float32)
        return probs

    def predict_global(self, X, batch_size=50000):
        """전체 클래스 차원의 확률 행렬 반환 (local→global 매핑 포함)."""
        N = X.shape[0]
        if not self.is_trained or self.model is None or self.local_classes is None:
            uniform = np.ones((N, self.global_num_classes), dtype=np.float32) / self.global_num_classes
            return uniform

        global_probs = np.zeros((N, self.global_num_classes), dtype=np.float32)
        feat_idx = self.config.feature_indices
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            probs_local = self._predict_proba_local(X[start:end][:, feat_idx])
            for l_idx, g_idx in enumerate(self.local_classes):
                global_probs[start:end, g_idx] = probs_local[:, l_idx]
        return global_probs

    def predict(self, X, batch_size=50000):
        """Global class ID 예측값 반환."""
        global_probs = self.predict_global(X, batch_size)
        return np.argmax(global_probs, axis=1)


# ──────────────────────────────────────────────
# Confusion matrix helpers
# ──────────────────────────────────────────────

def build_confusion_artifacts(y_true, y_pred, class_names, normal_class_ids,
                              top_k_pairs=8, low_recall_threshold=None):
    """
    Returns
    -------
    cm_df            : full confusion matrix DataFrame
    pairs_df         : sorted attack↔attack confusion pairs
    focus_classes    : sorted list of class IDs selected for Stage 2
                       (union of confusion-pair classes + low-recall classes)
    cm_attack_df     : attack-only confusion matrix DataFrame
    recall_per_class : dict {class_id: recall}
    low_recall_classes: list of class IDs added via low-recall threshold
    """
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    normal_set = set(int(x) for x in normal_class_ids)

    # ── attack↔attack confusion pairs ───────────────────────────────
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or int(cm[i, j]) <= 0:
                continue
            if i in normal_set or j in normal_set:
                continue
            rows.append({
                "true_class_id": int(i), "true_class_name": class_names[i],
                "pred_class_id": int(j), "pred_class_name": class_names[j],
                "count": int(cm[i, j]),
            })
    pairs_df = (
        pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
        if rows else pd.DataFrame(columns=["true_class_id", "true_class_name", "pred_class_id", "pred_class_name", "count"])
    )

    focus_from_pairs: set = set()
    if len(pairs_df) > 0 and top_k_pairs > 0:
        top_df = pairs_df.head(int(top_k_pairs))
        focus_from_pairs = (
            set(top_df["true_class_id"].astype(int).tolist())
            | set(top_df["pred_class_id"].astype(int).tolist())
        )

    # ── per-class recall (attack→normal misclassification 감지) ─────
    _, recall_arr, _, support_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    recall_per_class = {int(i): float(recall_arr[i]) for i in range(len(class_names))}

    low_recall_classes: List[int] = []
    if low_recall_threshold is not None:
        for i in range(len(class_names)):
            if i in normal_set:
                continue
            if int(support_arr[i]) > 0 and recall_arr[i] < float(low_recall_threshold):
                low_recall_classes.append(int(i))

    focus_classes = sorted(focus_from_pairs | set(low_recall_classes))

    # ── attack-only confusion matrix ─────────────────────────────────
    attack_ids = [i for i in range(len(class_names)) if i not in normal_set]
    attack_names = [class_names[i] for i in attack_ids]
    if attack_ids:
        cm_attack = cm[np.ix_(attack_ids, attack_ids)].astype(np.int64)
        cm_attack_df = pd.DataFrame(cm_attack, index=attack_names, columns=attack_names)
    else:
        cm_attack_df = pd.DataFrame()

    return cm_df, pairs_df, focus_classes, cm_attack_df, recall_per_class, low_recall_classes


def save_confusion_heatmap(cm_df, out_path, title, logger):
    if cm_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable, skipping heatmap: {e}")
        return
    n = len(cm_df)
    fig_w = max(8.0, min(28.0, 0.5 * n + 2.0))
    fig_h = max(6.0, min(24.0, 0.5 * n + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vals = cm_df.values.astype(np.float64)
    vmax = float(np.max(vals)) if vals.size > 0 else 1.0
    im = ax.imshow(vals, cmap="Blues", aspect="auto", vmin=0.0, vmax=max(1.0, vmax))
    ax.set_title(title)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cm_df.columns.tolist(), rotation=90, fontsize=7)
    ax.set_yticklabels(cm_df.index.tolist(), fontsize=7)
    threshold = 0.6 * max(1.0, vmax)
    for i in range(n):
        for j in range(n):
            v = int(round(vals[i, j]))
            color = "white" if vals[i, j] >= threshold else "black"
            ax.text(j, i, f"{v}", ha="center", va="center", color=color, fontsize=6)
    fig.colorbar(im, ax=ax).set_label("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────
# Stage2MoE  ← 핵심 신규 코드
# ──────────────────────────────────────────────

class Stage2MoE:
    """
    Stage 2 Mixture-of-Experts (normal 제외).

    구조:
      - focus 클래스마다 전담 ExpertXGB 1개 학습
      - 각 expert: target_classes=[해당 클래스], extra_negative_classes=[나머지 focus 클래스]
        include_normal=False → normal 샘플을 훈련 데이터에서 완전히 제거
      - 예측: 모든 expert의 predict_global() 출력을 평균 → focus 클래스 내 argmax

    Stage 2 에 도달한 샘플은 Stage 1 이 이미 focus 클래스 중 하나로 예측한 샘플이므로,
    Stage 2 는 focus 클래스 간 세밀한 구분만 담당.
    normal 을 제외함으로써 극심한 클래스 불균형(normal 45만 vs xss 130)을 회피.
    """

    def __init__(self, focus_classes: List[int], class_names: List[str],
                 num_classes: int, device: str = "cpu", seed: int = 42):
        self.focus_classes = sorted(focus_classes)
        self.class_names = list(class_names)
        self.num_classes = num_classes
        self.device = device
        self.seed = seed
        self.experts: Dict[int, ExpertXGB] = {}   # class_id → expert
        self.is_trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            normal_classes: List[int], logger,
            min_samples: int = 10, target_boost: float = 5.0):
        """
        Parameters
        ----------
        min_samples : focus 클래스 샘플이 이보다 적으면 해당 expert 를 건너뜀.
        target_boost: target 클래스 손실 가중치 배수.
        """
        n_features = X_train.shape[1]
        self.experts = {}

        for class_id in self.focus_classes:
            class_name = self.class_names[class_id]
            n_class = int(np.sum(y_train == class_id))
            if n_class < min_samples:
                logger.warning(
                    f"[Stage2MoE] Skip expert for '{class_name}': "
                    f"{n_class} samples < min_samples={min_samples}"
                )
                continue

            # 이 expert 의 negatives = 나머지 focus 클래스들 (normal 제외)
            extra_neg = [c for c in self.focus_classes if c != class_id]

            cfg = ExpertConfig(
                expert_id=class_id,
                name=f"S2E_{class_name}",
                target_classes=[class_id],
                target_family_ids=[],
                feature_indices=np.arange(n_features, dtype=int),
                include_normal=False,                    # normal 완전 제외
                extra_negative_classes=extra_neg,        # 나머지 focus 클래스
                use_balanced_weights=True,
                target_boost=target_boost,
                normal_weight_scale=0.0,                 # include_normal=False 이므로 무의미
                extra_negative_weight_scale=1.0,
                always_include=False,
                kind="specialist",
            )
            # normal_classes 를 빈 리스트로 넘겨야 _prepare_subset 이 normal 을 추가하지 않음
            # (include_normal=False 로 이미 처리되지만, 명시적으로 빈 리스트 전달)
            expert = ExpertXGB(cfg, global_num_classes=self.num_classes,
                               device=self.device, seed=self.seed)
            expert.fit(X_train, y_train, normal_classes=[], logger=logger)

            if expert.is_trained:
                self.experts[class_id] = expert
                local_cls = [self.class_names[c] for c in expert.local_classes]
                logger.info(
                    f"[Stage2MoE] Expert '{class_name}': "
                    f"train_size={expert.train_size:,}, val_acc={expert.val_acc:.4f}, "
                    f"local_classes={local_cls}"
                )
            else:
                logger.warning(f"[Stage2MoE] Expert '{class_name}': training failed.")

        self.is_trained = len(self.experts) > 0
        logger.info(
            f"[Stage2MoE] {len(self.experts)}/{len(self.focus_classes)} experts trained. "
            f"Trained for: {[self.class_names[c] for c in sorted(self.experts)]}"
        )

    def predict(self, X: np.ndarray, batch_size: int = 50000) -> np.ndarray:
        """
        각 expert 가 자기 target 클래스에 부여한 확률만 직접 비교 → argmax.

        평균 투표 방식의 문제점:
          - 8개 expert 중 7개가 잘못된 클래스를 지지하면 전문가 신호가 희석됨.
          - 예: xss expert p(xss)=0.8 vs 7명이 각자 p(자기)=0.5 → 평균에서 xss 패배.

        max-target-prob 방식:
          - 각 expert 는 오직 "내 target 클래스 확률" 하나만 경쟁에 올림.
          - xss expert p(xss)=0.8 vs brute-force expert p(brute-force)=0.5 → xss 승리.
          - Expert 전문화 신호가 그대로 전달됨.

        Returns global class IDs (shape (N,)).
        """
        N = len(X)
        if not self.is_trained or not self.experts:
            return np.full(N, self.focus_classes[0] if self.focus_classes else 0, dtype=np.int32)

        focus_arr = np.array(self.focus_classes, dtype=np.int32)
        # target_probs[n, i] = expert for focus_classes[i] 가 샘플 n 에 대해
        #                       자기 target 클래스에 부여한 확률 (expert 없으면 0)
        target_probs = np.zeros((N, len(focus_arr)), dtype=np.float64)

        for i, class_id in enumerate(focus_arr.tolist()):
            if class_id in self.experts:
                g_probs = self.experts[class_id].predict_global(X, batch_size=batch_size)
                target_probs[:, i] = g_probs[:, class_id]
            # expert 없으면 0 유지 → 해당 클래스 선택 안 됨

        local_idx = np.argmax(target_probs, axis=1)     # (N,)
        return focus_arr[local_idx]                      # global class IDs


# ──────────────────────────────────────────────
# CascadeClassifier  ← 핵심 신규 코드
# ──────────────────────────────────────────────

class CascadeClassifier:
    """
    2단계 하드 캐스케이드 분류기.

    Stage 1 : 전체 클래스 단일 XGBoost (baseline 동일).
    Stage 2 : confusion matrix로 선정된 focus 클래스 + normal 전용 XGBoost
              (balanced weights, target_boost 적용).

    라우팅 규칙 (하드, 확정적):
      - Stage 1 예측 ∈ focus_classes  →  Stage 2 의 예측으로 교체
      - Stage 1 예측 ∉ focus_classes  →  Stage 1 예측 그대로 사용

    MoE 대비 장점:
      - 라우팅 시 정답 클래스를 알 필요 없음 (Stage 1 예측이 곧 라우터)
      - 소프트 앙상블이 없어 Stage 2 신호가 희석되지 않음
      - 구조가 단순하여 실패 원인 분석이 용이

    한계:
      - Stage 1이 focus 클래스를 normal 로 잘못 예측한 샘플은
        Stage 2 에 도달하지 못함 (임계값 튜닝으로 별도 보완 가능)
    """

    def __init__(self, num_classes: int, class_names: List[str],
                 device: str = "cpu", seed: int = 42):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.device = device
        self.seed = seed
        self.stage1: Optional[ExpertXGB] = None
        self.stage2: Optional[Stage2MoE] = None     # MoE: per-class expert, no normal
        self.focus_classes: List[int] = []
        self.normal_classes: List[int] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            normal_classes: List[int], focus_classes: List[int], logger):
        self.normal_classes = list(normal_classes)
        self.focus_classes = sorted(set(focus_classes))
        n_features = X_train.shape[1]

        # ── Stage 1: 전체 클래스 글로벌 모델 ──────────────────────────
        s1_cfg = ExpertConfig(
            expert_id=0,
            name="Stage1_Global",
            target_classes=None,                          # 전체 클래스
            target_family_ids=[],
            feature_indices=np.arange(n_features, dtype=int),
            include_normal=True,
            extra_negative_classes=[],
            use_balanced_weights=False,
            target_boost=1.0,
            normal_weight_scale=1.0,
            extra_negative_weight_scale=1.0,
            always_include=True,
            kind="anchor",
        )
        self.stage1 = ExpertXGB(s1_cfg, global_num_classes=self.num_classes,
                                device=self.device, seed=self.seed)
        logger.info(f"[Cascade] Training Stage 1: global ({n_features} features, {self.num_classes} classes)")
        self.stage1.fit(X_train, y_train, normal_classes=normal_classes, logger=logger)
        logger.info(f"[Cascade] Stage 1 internal val_acc={self.stage1.val_acc:.4f}")

        # ── Stage 2: per-class MoE (normal 제외) ──────────────────────
        if len(self.focus_classes) >= 2:
            focus_names = [self.class_names[c] for c in self.focus_classes]
            logger.info(f"[Cascade] Training Stage 2 MoE: focus={focus_names}, NO normal")
            self.stage2 = Stage2MoE(
                focus_classes=self.focus_classes,
                class_names=self.class_names,
                num_classes=self.num_classes,
                device=self.device,
                seed=self.seed,
            )
            self.stage2.fit(X_train, y_train, normal_classes=normal_classes, logger=logger)
        else:
            logger.warning("[Cascade] focus_classes < 2; Stage 2 skipped. Cascade = single-stage.")

    def predict(self, X: np.ndarray, batch_size: int = 50000,
                soft_route_threshold: float = 0.0) -> Tuple[np.ndarray, dict]:
        """
        Parameters
        ----------
        soft_route_threshold : float
            0.0 = 비활성화 (기존 hard routing 만).
            > 0 = Stage 1 이 non-focus 로 예측했더라도, Stage 1 의 focus 클래스 최대
                  확률이 이 값을 초과하면 Stage 2 로 추가 라우팅 (soft routing).
                  예: 0.1 → focus 클래스 중 하나에 10% 이상 확률이 있는 샘플 포함.
                  bot/xss 처럼 Stage 1 이 normal 로 오분류한 샘플을 구제하는 목적.

        Returns
        -------
        final_preds : np.ndarray  (global class IDs, shape (N,))
        info        : dict
            s1_preds        : Stage 1 predictions (argmax)
            s1_global_probs : Stage 1 전체 확률 행렬 (N × num_classes)
            route_mask      : bool (hard + soft routing 합산)
            hard_route_mask : bool (Stage 1 예측 ∈ focus)
            soft_route_mask : bool (soft threshold 에 의한 추가 라우팅)
            s2_preds        : Stage 2 예측 (라우팅된 위치, 나머지 -1)
            changed_mask    : Stage 2 가 Stage 1 예측을 바꾼 위치
        """
        N = len(X)

        # ── Stage 1: 전체 확률 행렬 계산 ────────────────────────────
        s1_global_probs = self.stage1.predict_global(X, batch_size=batch_size)  # (N, C)
        s1_preds = np.argmax(s1_global_probs, axis=1).astype(np.int32)
        final_preds = s1_preds.copy()

        hard_route_mask = np.zeros(N, dtype=bool)
        soft_route_mask = np.zeros(N, dtype=bool)
        s2_preds_full = None
        changed_mask = np.zeros(N, dtype=bool)

        if (self.stage2 is not None and self.stage2.is_trained
                and len(self.focus_classes) > 0):

            focus_arr = np.array(self.focus_classes, dtype=np.int32)

            # ── Hard routing: Stage 1 이 focus 클래스로 예측한 샘플 ─
            hard_route_mask = np.isin(s1_preds, focus_arr)

            # ── Soft routing: Stage 1 이 non-focus 로 예측했지만
            #                  focus 클래스 최대 확률 > threshold 인 샘플 ─
            if soft_route_threshold > 0.0:
                focus_max_prob = s1_global_probs[:, focus_arr].max(axis=1)  # (N,)
                soft_route_mask = (~hard_route_mask) & (focus_max_prob > soft_route_threshold)

            route_mask = hard_route_mask | soft_route_mask

            if route_mask.any():
                X_routed = X[route_mask]
                s2_preds_routed = self.stage2.predict(X_routed, batch_size=batch_size)
                final_preds[route_mask] = s2_preds_routed
                s2_preds_full = np.full(N, -1, dtype=np.int32)
                s2_preds_full[route_mask] = s2_preds_routed
                changed_mask[route_mask] = s2_preds_routed != s1_preds[route_mask]
        else:
            route_mask = hard_route_mask  # all-False

        return final_preds, {
            "s1_preds": s1_preds,
            "s1_global_probs": s1_global_probs,
            "route_mask": route_mask,
            "hard_route_mask": hard_route_mask,
            "soft_route_mask": soft_route_mask,
            "s2_preds": s2_preds_full,
            "changed_mask": changed_mask,
        }


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    start_time = time.time()
    gc.enable()

    parser = argparse.ArgumentParser(description="2-Stage Hard Cascade Classifier")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Raw CSV directory (CIC2017/2018). 기본은 episode split 사용.")
    parser.add_argument("--random_split", action="store_true",
                        help="split_tags / timestamps 무시 → stratified random split 강제. "
                             "val confusion matrix 가 에피소드 배정에 의존하지 않도록 함.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--top_conf_pairs", type=int, default=8,
                        help="Stage 2 focus 클래스 선정에 쓸 top-K confusion 쌍 수")
    parser.add_argument("--low_recall_threshold", type=float, default=0.9,
                        help="이 값 미만의 recall 을 가진 attack 클래스를 "
                             "focus 클래스로 추가 (attack→normal 오분류 보완). "
                             "None 을 원하면 0.0 이하로 설정.")
    parser.add_argument("--s2_target_boost", type=float, default=5.0,
                        help="Stage2MoE 각 expert 의 target 클래스 손실 가중치 배수")
    parser.add_argument("--s2_min_samples", type=int, default=10,
                        help="Stage2MoE expert 학습에 필요한 최소 샘플 수")
    parser.add_argument("--soft_route_threshold", type=float, default=0.0,
                        help="Stage 1 이 non-focus 클래스로 예측했더라도, "
                             "focus 클래스 최대 확률이 이 값 초과이면 Stage 2 로 추가 라우팅. "
                             "0.0=비활성화. 추천 시작값: 0.1")
    parser.add_argument("--model", type=int, default=2, choices=[0, 1, 2],
                        help="0: baseline only, 1: cascade only, 2: both")
    args = parser.parse_args()

    exp_dir = f"results/logs_cascade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(exp_dir)
    logger.info(f"XGBoost device: {DEVICE}")
    logger.info(f"random_split={args.random_split}")

    if not args.data and not args.data_dir:
        raise ValueError("--data 또는 --data_dir 중 하나를 제공하세요.")

    if args.data_dir:
        data = load_and_preprocess_from_csv_dir(args.data_dir, logger)
    else:
        with open(args.data, "rb") as f:
            data = pickle.load(f)

    dataset_type = str(data.get("dataset_type", infer_dataset_type(args.data or "", data))).lower()

    # ── --random_split: split_tags 와 timestamps 를 무시 ────────────
    if args.random_split:
        for key in ["split_tags", "timestamps"]:
            if key in data and data[key] is not None:
                logger.info(f"--random_split: '{key}' 제거 → stratified random split 사용")
                data[key] = None

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)

    feature_names = get_feature_names(data, X.shape[1])
    class_names = [str(x) for x in (
        data["label_encoder"].classes_ if "label_encoder" in data
        else [f"Class_{i}" for i in range(len(np.unique(y)))]
    )]
    num_classes = len(class_names)

    logger.info(f"Dataset type: {dataset_type}")
    logger.info(f"X shape: {X.shape}, num_classes: {num_classes}")
    logger.info(f"Class names: {class_names}")

    # ── normal 클래스 ID 식별 ────────────────────────────────────────
    normal_class_ids = [
        i for i, name in enumerate(class_names)
        if "normal" in normalize_name(name) or "benign" in normalize_name(name)
    ]
    logger.info(f"Normal class IDs: {normal_class_ids} ({[class_names[i] for i in normal_class_ids]})")

    # ── 데이터 분할 ──────────────────────────────────────────────────
    if ("split_tags" in data and data["split_tags"] is not None
            and len(data["split_tags"]) == len(y)):
        logger.info("Using preassigned file-wise split from split_tags.")
        X_train, X_val, X_test, y_train, y_val, y_test = split_by_preassigned_tags(
            X, y, data["split_tags"]
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
            data, X, y, args.seed, logger
        )
    log_split_class_coverage(y, y_train, y_val, y_test, class_names, logger)
    del data, X, y
    gc.collect()

    logger.info(f"Split sizes | train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")

    # ── Val confusion matrix probe ───────────────────────────────────
    # Stage 2 focus 클래스를 결정하기 위해 val set 에서 confusion matrix 계산.
    # Stage 1 과 동일 구조의 모델을 probe 로 먼저 학습.
    focus_classes: List[int] = []
    probe_model: Optional[ExpertXGB] = None

    logger.info("\n=== Baseline Probe: val confusion matrix 계산 ===")
    probe_cfg = ExpertConfig(
        expert_id=0,
        name="Probe_Global",
        target_classes=None,
        target_family_ids=[],
        feature_indices=np.arange(X_train.shape[1], dtype=int),
        include_normal=True,
        use_balanced_weights=False,
        target_boost=1.0,
        normal_weight_scale=1.0,
        extra_negative_weight_scale=1.0,
        always_include=True,
        kind="anchor",
    )
    probe_model = ExpertXGB(probe_cfg, global_num_classes=num_classes, device=DEVICE, seed=args.seed)
    probe_model.fit(X_train, y_train, normal_classes=normal_class_ids, logger=logger)

    if probe_model.is_trained:
        val_preds = probe_model.predict(X_val, batch_size=args.batch_size)

        # low_recall_threshold: 0.0 이하이면 None 으로 취급 (기능 비활성화)
        effective_lrt = args.low_recall_threshold if args.low_recall_threshold > 0.0 else None

        cm_df, conf_pairs_df, focus_classes, cm_attack_df, recall_per_class, low_recall_classes = \
            build_confusion_artifacts(
                y_true=y_val,
                y_pred=val_preds,
                class_names=class_names,
                normal_class_ids=normal_class_ids,
                top_k_pairs=args.top_conf_pairs,
                low_recall_threshold=effective_lrt,
            )

        # 저장
        cm_df.to_csv(os.path.join(exp_dir, "probe_confusion_matrix_val.csv"), index=True)
        conf_pairs_df.to_csv(os.path.join(exp_dir, "probe_confusion_pairs_val.csv"), index=False)
        cm_attack_df.to_csv(os.path.join(exp_dir, "probe_confusion_attack_val.csv"), index=True)
        save_confusion_heatmap(
            cm_attack_df,
            os.path.join(exp_dir, "probe_confusion_attack_val.png"),
            title="Probe Confusion (Attack-Only, Val Set)",
            logger=logger,
        )

        logger.info("Top confusion pairs (val):")
        for _, r in conf_pairs_df.head(min(10, len(conf_pairs_df))).iterrows():
            logger.info(f"  {r['true_class_name']} -> {r['pred_class_name']} | count={int(r['count'])}")

        # per-class recall 로그 (attack 클래스만)
        normal_set = set(normal_class_ids)
        logger.info(f"Per-class recall (attack, val set) [threshold={effective_lrt}]:")
        for i in range(len(class_names)):
            if i in normal_set:
                continue
            tag = " ← [low-recall 추가]" if i in low_recall_classes else ""
            logger.info(f"  {class_names[i]:40s} recall={recall_per_class[i]:.4f}{tag}")

        # focus 클래스 선정 요약
        focus_from_pairs = set()
        if len(conf_pairs_df) > 0 and args.top_conf_pairs > 0:
            top_df = conf_pairs_df.head(args.top_conf_pairs)
            focus_from_pairs = (
                set(top_df["true_class_id"].astype(int).tolist())
                | set(top_df["pred_class_id"].astype(int).tolist())
            )
        logger.info(f"Focus from confusion pairs: {sorted(focus_from_pairs)} "
                    f"= {[class_names[c] for c in sorted(focus_from_pairs)]}")
        logger.info(f"Focus from low recall     : {sorted(low_recall_classes)} "
                    f"= {[class_names[c] for c in low_recall_classes]}")
        logger.info(f"Stage 2 focus class IDs   : {focus_classes}")
        logger.info(f"Stage 2 focus class names : {[class_names[c] for c in focus_classes]}")
    else:
        logger.warning("Probe training failed; Stage 2 will be skipped.")

    # ── 모델 학습 ────────────────────────────────────────────────────
    baseline: Optional[ExpertXGB] = None
    cascade: Optional[CascadeClassifier] = None

    # Baseline (Stage 1 과 동일 구조; probe 재사용)
    if args.model in (0, 2):
        logger.info("\n=== Baseline: single XGBoost ===")
        baseline = probe_model  # probe 와 완전히 동일한 모델
        logger.info("Probe model 재사용 (baseline = probe).")

    # Cascade
    if args.model in (1, 2):
        logger.info("\n=== Cascade Classifier ===")
        cascade = CascadeClassifier(
            num_classes=num_classes,
            class_names=class_names,
            device=DEVICE,
            seed=args.seed,
        )
        # Stage 1 은 baseline 과 동일 → probe 모델을 Stage 1 으로 재사용
        # Stage 2 만 별도 학습
        cascade.stage1 = probe_model
        cascade.normal_classes = normal_class_ids
        cascade.focus_classes = sorted(set(focus_classes))

        if len(cascade.focus_classes) >= 2:
            focus_names = [class_names[c] for c in cascade.focus_classes]
            logger.info(f"[Cascade] Stage 2 MoE focus={focus_names}, NO normal, "
                        f"target_boost={args.s2_target_boost}, min_samples={args.s2_min_samples}")
            cascade.stage2 = Stage2MoE(
                focus_classes=cascade.focus_classes,
                class_names=class_names,
                num_classes=num_classes,
                device=DEVICE,
                seed=args.seed,
            )
            cascade.stage2.fit(
                X_train, y_train,
                normal_classes=normal_class_ids,
                logger=logger,
                min_samples=args.s2_min_samples,
                target_boost=args.s2_target_boost,
            )
        else:
            logger.warning("[Cascade] focus_classes < 2 → Stage 2 없이 단일 Stage 1 동일.")

    # ── 평가 ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 72)

    base_preds = None
    if baseline is not None and baseline.is_trained:
        t0 = time.time()
        base_preds = baseline.predict(X_test, batch_size=args.batch_size)
        acc = accuracy_score(y_test, base_preds)
        logger.info(f"\n--- Baseline Single XGBoost ---")
        logger.info(f"Accuracy: {acc:.4f} | eval time: {time.time() - t0:.2f}s")
        logger.info("\n" + classification_report(
            y_test, base_preds,
            labels=np.arange(num_classes), target_names=class_names,
            digits=4, zero_division=0,
        ))

    cascade_preds = None
    if cascade is not None:
        t0 = time.time()
        cascade_preds, route_info = cascade.predict(
            X_test, batch_size=args.batch_size,
            soft_route_threshold=args.soft_route_threshold,
        )
        acc = accuracy_score(y_test, cascade_preds)
        n_hard = int(route_info["hard_route_mask"].sum())
        n_soft = int(route_info["soft_route_mask"].sum())
        n_routed = int(route_info["route_mask"].sum())
        n_changed = int(route_info["changed_mask"].sum())
        logger.info(f"\n--- Cascade Classifier ---")
        logger.info(f"Accuracy: {acc:.4f} | eval time: {time.time() - t0:.2f}s")
        logger.info(
            f"Routing | hard={n_hard:,}  soft={n_soft:,}  "
            f"total={n_routed:,}/{len(y_test):,} ({100*n_routed/len(y_test):.2f}%)"
        )
        logger.info(
            f"Routing | Stage 2 changed prediction: {n_changed:,}/{n_routed:,} "
            f"({100*n_changed/max(n_routed,1):.2f}% of routed)"
        )

        # focus 클래스별 라우팅 분석
        if n_routed > 0:
            logger.info("Routing analysis per focus class:")
            s1_preds_all = route_info["s1_preds"]
            final_preds_all = cascade_preds
            for fc in cascade.focus_classes:
                # Hard-routed (Stage 1 predicted fc)
                hard_fc = (s1_preds_all == fc)
                hard_n = int(hard_fc.sum())
                hard_correct = int((y_test[hard_fc] == fc).sum()) if hard_n > 0 else 0
                # True fc samples in test
                true_n = int((y_test == fc).sum())
                # How many true fc reached Stage 2 via hard route
                true_hard_reached = int(((s1_preds_all == fc) & (y_test == fc)).sum())
                # How many true fc reached Stage 2 via soft route
                soft_mask = route_info["soft_route_mask"]
                true_soft_reached = int((soft_mask & (y_test == fc)).sum()) if n_soft > 0 else 0
                # Final predictions as fc
                pred_as_fc = int((final_preds_all == fc).sum())
                logger.info(
                    f"  {class_names[fc]:35s} "
                    f"true={true_n} | "
                    f"hard_routed={hard_n}(precision={100*hard_correct/max(hard_n,1):.1f}%) | "
                    f"true_reached=hard:{true_hard_reached}+soft:{true_soft_reached} | "
                    f"final_pred_as={pred_as_fc}"
                )

        logger.info("\n" + classification_report(
            y_test, cascade_preds,
            labels=np.arange(num_classes), target_names=class_names,
            digits=4, zero_division=0,
        ))

        # 라우팅 요약 저장
        route_df = pd.DataFrame({
            "metric": [
                "n_test", "n_routed_to_stage2", "route_rate_pct",
                "n_changed_by_stage2", "change_rate_pct",
            ],
            "value": [
                len(y_test), n_routed, round(100 * n_routed / len(y_test), 4),
                n_changed, round(100 * n_changed / max(n_routed, 1), 4),
            ],
        })
        route_df.to_csv(os.path.join(exp_dir, "cascade_routing_summary.csv"), index=False)

    # ── per-class 비교 (baseline vs cascade) ────────────────────────
    if base_preds is not None and cascade_preds is not None:
        unique_classes = np.arange(num_classes)
        p_b, r_b, f1_b, _ = precision_recall_fscore_support(y_test, base_preds, labels=unique_classes, zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_test, cascade_preds, labels=unique_classes, zero_division=0)
        df = pd.DataFrame({
            "class_id": unique_classes,
            "class_name": class_names,
            "baseline_precision": p_b, "baseline_recall": r_b, "baseline_f1": f1_b,
            "cascade_precision": p_c, "cascade_recall": r_c, "cascade_f1": f1_c,
            "delta_f1": f1_c - f1_b,
            "is_focus_class": [int(i in (cascade.focus_classes if cascade else [])) for i in unique_classes],
        })
        per_class_path = os.path.join(exp_dir, "baseline_vs_cascade_per_class.csv")
        df.to_csv(per_class_path, index=False)
        logger.info(f"Saved per-class comparison to {per_class_path}")

        # focus 클래스 요약 로그
        focus_df = df[df["is_focus_class"] == 1]
        if not focus_df.empty:
            logger.info("\nFocus class delta_f1 summary:")
            for _, row in focus_df.iterrows():
                direction = "↑" if row["delta_f1"] > 0 else ("↓" if row["delta_f1"] < 0 else "=")
                logger.info(
                    f"  {row['class_name']:40s} "
                    f"baseline_f1={row['baseline_f1']:.4f} → cascade_f1={row['cascade_f1']:.4f} "
                    f"delta={row['delta_f1']:+.4f} {direction}"
                )

    logger.info(f"\nTotal elapsed time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()

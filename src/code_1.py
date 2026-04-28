#!/usr/bin/env python3
"""
code_1.py — Confidence-Gated Binary Specialist MoE

═══════════════════════════════════════════════════════════════════════════════
전체 구조
═══════════════════════════════════════════════════════════════════════════════

[배경: 기존 MoE의 실패 원인]
  1. Multi-class expert 내부에도 imbalance가 그대로 존재
       → tail class의 gradient가 다수 클래스에 묻혀 학습 불충분
  2. Router/Gate가 tail class를 잘못 라우팅
       → 희귀 클래스일수록 router도 못 배움 → 오라우팅 → 성능 저하
  3. expert 이름만 specialist, 실질적 전문화 부재

[핵심 아이디어: Binary Specialist + Self-Gating]
  Tail class T에 대해 "T vs. 나머지 전체" 이진 분류기를 훈련한다.
  - Positive: class T 샘플 전부
  - Negative: 나머지 모든 클래스에서 neg_ratio × |pos|개 다운샘플
  - SMOTE: positive 증강 (|pos_train| >= 6 인 경우)

  Gate 함수 = Specialist 자신의 출력 P(T | x)
    → 별도 router 없음
    → x가 달라질 때마다 gate 값이 달라지므로 input-dependent gating 만족
    → MoE의 정의 충족: 서로 다른 x에 대해 서로 다른 expert가 활성화됨

[MoE 구조 다이어그램]

  X (test sample)
   ├─► Baseline XGBoost ──────────────────────────────► P_base[0..C-1]
   ├─► BinarySpecialist_T1 ──► P(T1|x)  ─► Gate_T1  ─►  score[T1]
   ├─► BinarySpecialist_T2 ──► P(T2|x)  ─► Gate_T2  ─►  score[T2]
   └─► BinarySpecialist_Tk ──► P(Tk|x)  ─► Gate_Tk  ─►  score[Tk]

  score[majority c] = P_base[c]
  score[tail T]     = β_T × P_spec(T|x) + (1 - β_T) × P_base[T]
  prediction        = argmax_c score[c]

  β_T ∈ [0,1]: val set에서 class T의 F1을 최대화하는 값으로 per-class tuning.
    β_T=0 → baseline만 사용 (specialist 무시)
    β_T=1 → specialist만 사용

[Tail class 선정 기준 (Union)]
  - 기준A (샘플 수): train count < COUNT_THRESHOLD  (default 1000)
  - 기준B (성능):   val recall < RECALL_THRESHOLD   (default 0.85)
  두 조건 중 하나라도 해당하면 tail class로 지정.
  기준B는 baseline 훈련 후 val 추론 결과를 사용.

[학습 순서]
  1. make_splits: train / val / test 분할
  2. BaselineXGB.fit(X_train, y_train)
  3. select_tail_classes(y_train, X_val, y_val, baseline)
  4. BinarySpecialist.fit(X_train, y_train)  ← 각 tail class별
  5. tune_betas(X_val, y_val)               ← per-class β 최적화
  6. predict(X_test)                        ← score 결합 후 argmax
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import gc
import logging
import os
import pickle
import re
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티 (code_final.py 계승)
# ─────────────────────────────────────────────────────────────────────────────

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
    raise ValueError(
        f"Cannot infer dataset_type from '{data_path}'. "
        "Include unswnb15/cic2017/cic2018 in filename or in pickle['dataset_type']."
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


def safe_train_test_split(X, y, test_size, random_state, stratify=True):
    try:
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=(y if stratify else None),
            random_state=random_state,
        )
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def check_split_coverage(
    y_all: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> bool:
    """각 split에 모든 클래스가 최소 1개 이상 존재하는지 확인."""
    all_classes = set(np.unique(y_all))
    train_classes = set(np.unique(y_train))
    # val/test에 없는 클래스는 허용, train에 없는 클래스가 문제
    return all_classes == train_classes


def make_splits(data: dict, X: np.ndarray, y: np.ndarray, seed: int, logger):
    """
    Group / Timestamp / Stratified-random 순서로 split 시도.
    code_final.py의 make_splits와 동일한 우선순위 로직.
    """
    group_keys = ["scenario_ids", "group_ids", "groups", "fold_ids"]
    ts_keys = ["timestamps", "timestamp", "time", "times"]

    # Group-aware split
    for key in group_keys:
        if key in data and data[key] is not None and len(data[key]) == len(y):
            logger.info(f"Using group-aware split via pickle['{key}'].")
            groups = np.asarray(data[key])
            for trial in range(10):
                rs = seed + trial
                gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                trainval_idx, test_idx = next(gss1.split(X, y, groups))
                X_tv, y_tv, g_tv = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                tr_rel, va_rel = next(gss2.split(X_tv, y_tv, g_tv))
                X_train, y_train = X_tv[tr_rel], y_tv[tr_rel]
                X_val, y_val = X_tv[va_rel], y_tv[va_rel]
                X_test, y_test = X[test_idx], y[test_idx]
                if check_split_coverage(y, y_train, y_val, y_test):
                    return X_train, X_val, X_test, y_train, y_val, y_test
            logger.warning("Group-aware split coverage insufficient; falling back.")
            break

    # Chronological split
    for key in ts_keys:
        if key in data and data[key] is not None and len(data[key]) == len(y):
            logger.info(f"Using chronological blocked split via pickle['{key}'].")
            ts = np.asarray(data[key])
            order = np.argsort(ts)
            X_s, y_s = X[order], y[order]
            n = len(y_s)
            n_train = int(0.64 * n)
            n_val = int(0.16 * n)
            X_train, y_train = X_s[:n_train], y_s[:n_train]
            X_val, y_val = X_s[n_train:n_train + n_val], y_s[n_train:n_train + n_val]
            X_test, y_test = X_s[n_train + n_val:], y_s[n_train + n_val:]
            if check_split_coverage(y, y_train, y_val, y_test):
                return X_train, X_val, X_test, y_train, y_val, y_test
            logger.warning("Chronological split coverage insufficient; falling back.")
            break

    # Stratified-random fallback
    logger.warning(
        "No group/timestamp split available. "
        "Falling back to stratified-random split. "
        "Note: temporal leakage risk exists for CICIDS datasets."
    )
    for trial in range(10):
        rs = seed + trial
        X_temp, X_test, y_temp, y_test = safe_train_test_split(
            X, y, test_size=0.2, random_state=rs, stratify=True
        )
        X_train, X_val, y_train, y_val = safe_train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=rs, stratify=True
        )
        if check_split_coverage(y, y_train, y_val, y_test):
            return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_taxonomy_groups(
    dataset_type: str, class_names: List[str], logger
) -> Tuple[Dict[str, List[int]], Dict[int, int], Dict[int, str]]:
    """클래스 이름을 공격 family로 분류. Normal 클래스 식별에 사용."""
    groups = {"Normal": [], "Group1": [], "Group2": [], "Group3": [], "Group4": []}
    if dataset_type == "unswnb15":
        type_groups = [
            ["DoS", "Worms"],
            ["Analysis", "Reconnaissance"],
            ["Exploits", "Fuzzers", "Generic"],
            ["Backdoor", "Shellcode"],
        ]
    elif dataset_type == "cic2017":
        type_groups = [
            ["DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"],
            ["FTP-Patator", "SSH-Patator"],
            ["Web Attack  Brute Force", "Web Attack  Sql Injection", "Web Attack  XSS", "Heartbleed"],
            ["Bot", "Infiltration", "PortScan"],
        ]
    else:  # cic2018
        type_groups = [
            ["DDOS attack-HOIC", "DDOS attack-LOIC-UDP", "DDoS attacks-LOIC-HTTP",
             "DoS attacks-GoldenEye", "DoS attacks-Hulk",
             "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"],
            ["FTP-BruteForce", "SSH-Bruteforce"],
            ["Brute Force -Web", "SQL Injection", "Brute Force -XSS"],
            ["Bot", "Infilteration"],
        ]

    norm_type_groups = [[normalize_name(c) for c in group] for group in type_groups]
    class_to_family: Dict[int, int] = {}
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
            logger.warning(f"Class '{name}' not mapped; treating as Group4 fallback.")
            groups["Group4"].append(idx)
            class_to_family[idx] = 4

    return groups, class_to_family, family_id_to_name


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Full multi-class XGBoost (Anchor Expert)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineXGB:
    """
    단일 XGBoost 다중 분류기.
    Anchor Expert 역할: 모든 클래스에 대한 기본 확률 P_base[c] 제공.
    class_weight="balanced" 없이 훈련 → 다수 클래스 정확도 보존.
    """

    def __init__(self, num_classes: int, device: str = "cpu", seed: int = 42):
        self.num_classes = num_classes
        self.device = device
        self.seed = seed
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, logger):
        n_cls = self.num_classes
        params = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
            early_stopping_rounds=30,
        )
        if self.device == "cuda":
            params["device"] = "cuda"

        if n_cls == 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_cls
            params["eval_metric"] = "mlogloss"

        self.model = xgb.XGBClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        self.is_trained = True
        val_preds = np.argmax(self.predict_proba(X_val), axis=1)
        acc = accuracy_score(y_val, val_preds)
        logger.info(f"  [Baseline] val_acc={acc:.4f} | best_iteration={self.model.best_iteration}")

    def predict_proba(self, X: np.ndarray, batch_size: int = 50000) -> np.ndarray:
        """Returns (N, C) probability matrix."""
        N = X.shape[0]
        out = np.zeros((N, self.num_classes), dtype=np.float32)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            probs = self.model.predict_proba(X[start:end])
            probs = np.asarray(probs, dtype=np.float32)
            if probs.ndim == 1:
                probs = np.column_stack([1.0 - probs, probs])
            out[start:end] = probs
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Binary Specialist: Tail class T vs. 나머지 전체
# ─────────────────────────────────────────────────────────────────────────────

class BinarySpecialist:
    """
    단일 tail class T에 특화된 이진 분류기.

    [훈련 데이터 구성]
      Positive : y_train == T  (전부 사용)
      Negative : y_train != T  에서 neg_ratio × |pos|개 다운샘플
                 → 클래스별 균등 샘플링으로 특정 다수 클래스가 지배하지 않게 함
      SMOTE    : |pos_train| >= SMOTE_MIN_SAMPLES 인 경우 positive 증강

    [게이팅 역할]
      predict_confidence(x) = P(T | x) ∈ [0, 1]
      → x가 달라질 때마다 다른 값 → input-dependent gate
      → 높으면 "이 샘플은 내가 담당", 낮으면 baseline이 자동으로 지배
    """

    SMOTE_MIN_SAMPLES = 6    # SMOTE 최소 positive 샘플 수
    NEG_STRATIFY_BINS = 10   # negative 클래스별 균등 샘플 최대 bin 수

    def __init__(
        self,
        target_class: int,
        class_name: str,
        device: str = "cpu",
        seed: int = 42,
        neg_ratio: int = 50,
        use_smote: bool = True,
    ):
        self.target_class = target_class
        self.class_name = class_name
        self.device = device
        self.seed = seed
        self.neg_ratio = neg_ratio
        self.use_smote = use_smote
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False
        self.n_pos_train = 0
        self.n_neg_train = 0
        self.smote_applied = False
        self.best_iteration = None

    # ── 내부 helper ──────────────────────────────────────────────────────────

    def _sample_negatives(
        self, X_neg: np.ndarray, y_neg: np.ndarray, n_target: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Negative 클래스별 균등 샘플링.
        특정 다수 클래스(예: Benign)가 negative를 독점하지 않도록 함.
        """
        unique_neg_cls = np.unique(y_neg)
        per_class_quota = max(1, n_target // len(unique_neg_cls))
        selected_idx = []
        for c in unique_neg_cls:
            idx = np.where(y_neg == c)[0]
            take = min(len(idx), per_class_quota)
            selected_idx.append(rng.choice(idx, size=take, replace=False))
        selected_idx = np.concatenate(selected_idx)
        # 부족하면 나머지에서 추가
        if len(selected_idx) < n_target:
            remaining = np.setdiff1d(np.arange(len(y_neg)), selected_idx)
            extra = min(len(remaining), n_target - len(selected_idx))
            if extra > 0:
                selected_idx = np.concatenate(
                    [selected_idx, rng.choice(remaining, size=extra, replace=False)]
                )
        # 초과하면 랜덤 서브샘플
        if len(selected_idx) > n_target:
            selected_idx = rng.choice(selected_idx, size=n_target, replace=False)
        return X_neg[selected_idx], y_neg[selected_idx]

    def _binary_internal_split(
        self, X: np.ndarray, y_bin: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Early stopping을 위한 내부 train/val 분할 (10% val).
        positive가 너무 적으면 분할하지 않고 False 반환.
        """
        n_pos = int(np.sum(y_bin == 1))
        if n_pos < 10:
            return X, None, y_bin, None, False
        seed_val = int(rng.integers(0, 2**31))
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y_bin, test_size=0.1, stratify=y_bin, random_state=seed_val
            )
            # val에 positive가 최소 1개는 있어야 함
            if np.sum(y_va == 1) < 1:
                return X, None, y_bin, None, False
            return X_tr, X_va, y_tr, y_va, True
        except ValueError:
            return X, None, y_bin, None, False

    # ── 훈련 ─────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, logger):
        rng = np.random.default_rng(self.seed)

        pos_idx = np.where(y_train == self.target_class)[0]
        neg_idx = np.where(y_train != self.target_class)[0]
        n_pos = len(pos_idx)
        n_neg_target = min(len(neg_idx), n_pos * self.neg_ratio)

        if n_pos == 0:
            logger.warning(f"  [Specialist:{self.class_name}] No positive samples; skipping.")
            return
        if n_pos < 2:
            logger.warning(f"  [Specialist:{self.class_name}] Only {n_pos} positive sample(s); skipping.")
            return

        X_pos = X_train[pos_idx]
        y_neg_full = y_train[neg_idx]
        X_neg_sampled, _ = self._sample_negatives(
            X_train[neg_idx], y_neg_full, n_neg_target, rng
        )

        # 이진 레이블 결합
        X_combined = np.vstack([X_pos, X_neg_sampled])
        y_combined = np.concatenate([
            np.ones(len(X_pos), dtype=np.int32),
            np.zeros(len(X_neg_sampled), dtype=np.int32),
        ])

        # 내부 train/val 분할 (early stopping용)
        X_tr, X_va, y_tr, y_va, use_eval = self._binary_internal_split(X_combined, y_combined, rng)

        # SMOTE: positive를 증강
        n_pos_tr = int(np.sum(y_tr == 1))
        if self.use_smote and n_pos_tr >= self.SMOTE_MIN_SAMPLES:
            k = min(5, n_pos_tr - 1)
            try:
                smote = SMOTE(
                    sampling_strategy="minority",
                    k_neighbors=k,
                    random_state=int(rng.integers(0, 2**31)),
                )
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                self.smote_applied = True
            except Exception as e:
                logger.warning(f"  [Specialist:{self.class_name}] SMOTE failed ({e}); skipping SMOTE.")
                self.smote_applied = False
        else:
            self.smote_applied = False

        self.n_pos_train = n_pos_tr
        self.n_neg_train = int(np.sum(y_tr == 0))

        # scale_pos_weight: SMOTE 후 남은 imbalance 보정
        n_pos_after = int(np.sum(y_tr == 1))
        n_neg_after = int(np.sum(y_tr == 0))
        spw = max(1.0, n_neg_after / max(n_pos_after, 1))

        params = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="aucpr",    # imbalanced binary에 logloss보다 적합
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
        )
        if use_eval:
            params["early_stopping_rounds"] = 30
        if self.device == "cuda":
            params["device"] = "cuda"

        self.model = xgb.XGBClassifier(**params)
        fit_kwargs: dict = {"verbose": False}
        if use_eval and X_va is not None:
            fit_kwargs["eval_set"] = [(X_va, y_va)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_tr, y_tr, **fit_kwargs)

        self.is_trained = True
        self.best_iteration = getattr(self.model, "best_iteration", None)
        logger.info(
            f"  [Specialist:{self.class_name}] "
            f"n_pos={n_pos} | n_neg_sampled={len(X_neg_sampled)} | "
            f"smote={self.smote_applied} | "
            f"spw={spw:.1f} | best_iter={self.best_iteration}"
        )

    # ── 추론 ─────────────────────────────────────────────────────────────────

    def predict_confidence(self, X: np.ndarray, batch_size: int = 50000) -> np.ndarray:
        """
        P(target_class | x) 반환. shape=(N,).
        이 값이 gate: 높을수록 이 specialist가 강하게 활성화됨.
        """
        if not self.is_trained or self.model is None:
            return np.zeros(X.shape[0], dtype=np.float32)
        N = X.shape[0]
        conf = np.zeros(N, dtype=np.float32)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            probs = self.model.predict_proba(X[start:end])
            probs = np.asarray(probs, dtype=np.float32)
            if probs.ndim == 1:
                conf[start:end] = probs
            else:
                conf[start:end] = probs[:, 1]
        return conf


# ─────────────────────────────────────────────────────────────────────────────
# Confidence-Gated MoE (메인 모델)
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceGatedMoE:
    """
    Confidence-Gated Binary Specialist MoE.

    [Gate 설계 원리]
      각 binary specialist E_T는 P(T | x)를 출력.
      이 확률값 자체가 gate 함수 G_T(x).

      G_T(x)가 높다 → E_T가 이 샘플을 T라고 확신 → score[T] 상승
      G_T(x)가 낮다 → E_T가 이 샘플은 T가 아니라고 판단 → score[T]에 영향 미미

      β_T(≈1): specialist를 강하게 신뢰 (val에서 recall 향상)
      β_T(≈0): baseline을 유지 (false positive 억제)

    [β_T 결정 방식]
      val에서 class T에 대한 F1을 최대화하는 β_T를 그리드서치.
      val positive 샘플이 MIN_VAL_POS개 미만이면 DEFAULT_BETA 사용.
    """

    BETA_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    MIN_VAL_POS = 3        # β tuning 최소 val positive 수
    DEFAULT_BETA = 0.8     # val positive 부족 시 기본 β

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        normal_classes: List[int],
        device: str = "cpu",
        seed: int = 42,
        count_threshold: int = 1000,
        recall_threshold: float = 0.85,
        neg_ratio: int = 50,
        use_smote: bool = True,
    ):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.normal_classes = list(normal_classes)
        self.device = device
        self.seed = seed
        self.count_threshold = count_threshold
        self.recall_threshold = recall_threshold
        self.neg_ratio = neg_ratio
        self.use_smote = use_smote

        self.baseline: Optional[BaselineXGB] = None
        self.specialists: Dict[int, BinarySpecialist] = {}
        self.tail_classes: List[int] = []
        self.betas: Dict[int, float] = {}

    # ── Tail class 선정 ───────────────────────────────────────────────────────

    def select_tail_classes(
        self,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        logger,
    ) -> List[int]:
        """
        기준A: train count < count_threshold
        기준B: baseline val recall < recall_threshold
        → Union (두 조건 중 하나라도 해당)
        normal class는 specialist 대상 제외.
        """
        counts = np.bincount(y_train, minlength=self.num_classes)
        attack_classes = [c for c in range(self.num_classes) if c not in self.normal_classes]

        # 기준A
        tail_by_count = {c for c in attack_classes if counts[c] < self.count_threshold}

        # 기준B: baseline val recall
        assert self.baseline is not None and self.baseline.is_trained, \
            "select_tail_classes는 baseline 훈련 이후에 호출해야 합니다."
        val_probs = self.baseline.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)

        tail_by_recall = set()
        for c in attack_classes:
            mask = y_val == c
            if mask.sum() == 0:
                continue
            recall = float(np.mean(val_preds[mask] == c))
            if recall < self.recall_threshold:
                tail_by_recall.add(c)

        tail = sorted(tail_by_count | tail_by_recall)

        logger.info(f"Tail class 선정 결과:")
        logger.info(f"  기준A (count < {self.count_threshold}): "
                    f"{[self.class_names[c] for c in sorted(tail_by_count)]}")
        logger.info(f"  기준B (val recall < {self.recall_threshold}): "
                    f"{[self.class_names[c] for c in sorted(tail_by_recall)]}")
        logger.info(f"  최종 tail classes ({len(tail)}개): "
                    f"{[self.class_names[c] for c in tail]}")
        for c in tail:
            mask = y_val == c
            n_val_pos = int(mask.sum())
            recall = float(np.mean(val_preds[mask] == c)) if n_val_pos > 0 else float("nan")
            logger.info(f"    {self.class_names[c]:30s} | "
                        f"train_n={counts[c]:5d} | val_recall={recall:.3f}")

        self.tail_classes = tail
        return tail

    # ── 훈련 ─────────────────────────────────────────────────────────────────

    def fit_baseline(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        logger,
    ):
        logger.info("=== [1/4] Baseline XGBoost 훈련 ===")
        self.baseline = BaselineXGB(self.num_classes, self.device, self.seed)
        self.baseline.fit(X_train, y_train, X_val, y_val, logger)

    def fit_specialists(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        logger,
    ):
        assert self.tail_classes, "fit_specialists 전에 select_tail_classes를 호출하세요."
        logger.info(f"=== [3/4] Binary Specialist 훈련 ({len(self.tail_classes)}개) ===")
        for c in self.tail_classes:
            specialist = BinarySpecialist(
                target_class=c,
                class_name=self.class_names[c],
                device=self.device,
                seed=self.seed + c,         # 클래스별 다른 seed
                neg_ratio=self.neg_ratio,
                use_smote=self.use_smote,
            )
            specialist.fit(X_train, y_train, logger)
            if specialist.is_trained:
                self.specialists[c] = specialist

    # ── β tuning ─────────────────────────────────────────────────────────────

    def tune_betas(
        self,
        X_val: np.ndarray, y_val: np.ndarray,
        logger,
    ):
        """
        Val에서 각 tail class T의 F1을 최대화하는 β_T를 결정.

        score[T] = β × P_spec(T|x) + (1-β) × P_base[T](x)
        → β를 grid search하여 val F1@T 최대화
        """
        logger.info("=== [4/4] β per-class tuning ===")
        assert self.baseline is not None and self.baseline.is_trained

        base_probs_val = self.baseline.predict_proba(X_val)  # (N_val, C)

        for c, specialist in self.specialists.items():
            if not specialist.is_trained:
                self.betas[c] = 0.0
                continue

            p_spec_val = specialist.predict_confidence(X_val)  # (N_val,)
            n_pos = int(np.sum(y_val == c))

            if n_pos < self.MIN_VAL_POS:
                self.betas[c] = self.DEFAULT_BETA
                logger.info(
                    f"  {self.class_names[c]:30s} | val_pos={n_pos} < {self.MIN_VAL_POS} "
                    f"→ β=DEFAULT({self.DEFAULT_BETA})"
                )
                continue

            best_beta, best_f1 = 0.0, -1.0
            for beta in self.BETA_GRID:
                # score 계산
                scores = base_probs_val.copy()
                scores[:, c] = beta * p_spec_val + (1.0 - beta) * base_probs_val[:, c]
                preds = np.argmax(scores, axis=1)
                f1 = f1_score(y_val, preds, labels=[c], average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_beta = float(beta)

            self.betas[c] = best_beta
            logger.info(
                f"  {self.class_names[c]:30s} | val_pos={n_pos:4d} "
                f"→ β={best_beta:.2f} (val F1={best_f1:.4f})"
            )

    # ── 추론 ─────────────────────────────────────────────────────────────────

    def predict(
        self, X: np.ndarray, batch_size: int = 50000
    ) -> Tuple[np.ndarray, Dict]:
        """
        [추론 흐름]
          1. Baseline으로 P_base[c] 계산
          2. 각 tail specialist로 P_spec(T|x) 계산
          3. score 결합: score[T] = β_T * P_spec + (1-β_T) * P_base[T]
          4. argmax → final prediction

        Returns
        -------
        preds : (N,) int
        info  : dict with 'base_probs', 'specialist_conf', 'betas'
        """
        assert self.baseline is not None and self.baseline.is_trained

        base_probs = self.baseline.predict_proba(X, batch_size=batch_size)  # (N, C)
        final_scores = base_probs.copy()

        spec_conf: Dict[int, np.ndarray] = {}
        for c, specialist in self.specialists.items():
            if not specialist.is_trained:
                continue
            beta = self.betas.get(c, 0.0)
            if beta == 0.0:
                continue
            p_spec = specialist.predict_confidence(X, batch_size=batch_size)
            spec_conf[c] = p_spec
            final_scores[:, c] = beta * p_spec + (1.0 - beta) * base_probs[:, c]

        preds = np.argmax(final_scores, axis=1).astype(np.int32)
        info = {
            "base_probs": base_probs,
            "specialist_conf": spec_conf,
            "betas": dict(self.betas),
            "final_scores": final_scores,
        }
        return preds, info


# ─────────────────────────────────────────────────────────────────────────────
# 결과 리포팅 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def per_class_comparison(
    y_test: np.ndarray,
    base_preds: np.ndarray,
    moe_preds: np.ndarray,
    num_classes: int,
    class_names: List[str],
    tail_classes: List[int],
    betas: Dict[int, float],
) -> pd.DataFrame:
    labels = np.arange(num_classes)
    p_b, r_b, f1_b, _ = precision_recall_fscore_support(y_test, base_preds, labels=labels, zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, moe_preds, labels=labels, zero_division=0)
    return pd.DataFrame({
        "class_id":    labels,
        "class_name":  class_names,
        "is_tail":     [c in tail_classes for c in labels],
        "beta":        [betas.get(c, float("nan")) for c in labels],
        "base_prec":   p_b, "base_recall": r_b, "base_f1": f1_b,
        "moe_prec":    p_m, "moe_recall":  r_m, "moe_f1":  f1_m,
        "delta_f1":    f1_m - f1_b,
    })


def specialist_activation_summary(
    y_test: np.ndarray,
    info: Dict,
    class_names: List[str],
    tail_classes: List[int],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """각 specialist가 test set에서 얼마나 자주 활성화됐는지 요약."""
    rows = []
    spec_conf = info["specialist_conf"]
    betas = info["betas"]
    for c in tail_classes:
        if c not in spec_conf:
            continue
        conf = spec_conf[c]
        beta = betas.get(c, 0.0)
        fire_rate = float(np.mean(conf > threshold))
        fire_on_true = float(np.mean(conf[y_test == c] > threshold)) if np.any(y_test == c) else float("nan")
        rows.append({
            "class_name":    class_names[c],
            "beta":          beta,
            "fire_rate_all": fire_rate,
            "fire_rate_true": fire_on_true,
            "mean_conf_all":  float(np.mean(conf)),
            "mean_conf_true": float(np.mean(conf[y_test == c])) if np.any(y_test == c) else float("nan"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    gc.enable()

    parser = argparse.ArgumentParser(
        description="Confidence-Gated Binary Specialist MoE"
    )
    parser.add_argument("--data",              type=str,   required=True)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--batch_size",        type=int,   default=20000)
    parser.add_argument("--count_threshold",   type=int,   default=1000,
                        help="Tail class: train sample count 기준 (default: 1000)")
    parser.add_argument("--recall_threshold",  type=float, default=0.85,
                        help="Tail class: baseline val recall 기준 (default: 0.85)")
    parser.add_argument("--neg_ratio",         type=int,   default=50,
                        help="Specialist negative sampling 배율 (default: 50)")
    parser.add_argument("--no_smote",          action="store_true",
                        help="SMOTE 비활성화")
    parser.add_argument("--model",             type=int,   default=2,
                        choices=[0, 1, 2],
                        help="0: baseline only  1: MoE only  2: both (default)")
    args = parser.parse_args()

    exp_dir = f"results/code1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(exp_dir)
    logger.info(f"XGBoost device: {DEVICE}")
    logger.info(f"Args: {vars(args)}")

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    dataset_type = infer_dataset_type(args.data, data)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)

    feature_names = get_feature_names(data, X.shape[1])
    le = data.get("label_encoder", None)
    class_names = [str(c) for c in (le.classes_ if le is not None else range(len(np.unique(y))))]
    num_classes = len(np.unique(y))

    logger.info(f"Dataset: {dataset_type} | X={X.shape} | classes={num_classes}")
    logger.info(f"Class distribution:\n" + "\n".join(
        f"  [{i:2d}] {class_names[i]:35s} {int(np.sum(y == i)):7d}"
        for i in range(num_classes)
    ))

    # ── Taxonomy (normal class 식별용) ────────────────────────────────────────
    groups, class_to_family, _ = get_taxonomy_groups(dataset_type, class_names, logger)
    normal_classes = groups["Normal"]
    logger.info(f"Normal classes: {[class_names[c] for c in normal_classes]}")

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
        data, X, y, args.seed, logger
    )
    del data, X, y
    gc.collect()

    logger.info(f"Split sizes — train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    # ── 모델 훈련 ─────────────────────────────────────────────────────────────
    moe = ConfidenceGatedMoE(
        num_classes=num_classes,
        class_names=class_names,
        normal_classes=normal_classes,
        device=DEVICE,
        seed=args.seed,
        count_threshold=args.count_threshold,
        recall_threshold=args.recall_threshold,
        neg_ratio=args.neg_ratio,
        use_smote=not args.no_smote,
    )

    # Step 1: Baseline
    moe.fit_baseline(X_train, y_train, X_val, y_val, logger)

    if args.model in (1, 2):
        # Step 2: Tail class 선정
        logger.info("=== [2/4] Tail class 선정 ===")
        moe.select_tail_classes(y_train, X_val, y_val, logger)

        # Step 3: Specialist 훈련
        moe.fit_specialists(X_train, y_train, logger)

        # Step 4: β tuning
        moe.tune_betas(X_val, y_val, logger)

    # ── 평가 ──────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("FINAL EVALUATION — TEST SET")
    logger.info("=" * 72)

    # Baseline 예측
    t0 = time.time()
    base_probs_test = moe.baseline.predict_proba(X_test, batch_size=args.batch_size)
    base_preds = np.argmax(base_probs_test, axis=1)
    base_acc = accuracy_score(y_test, base_preds)
    base_time = time.time() - t0

    if args.model in (1, 2) and moe.specialists:
        # MoE 예측
        t0 = time.time()
        moe_preds, info = moe.predict(X_test, batch_size=args.batch_size)
        moe_acc = accuracy_score(y_test, moe_preds)
        moe_time = time.time() - t0

        # Accuracy 비교 (한 번만, 표 밖)
        logger.info(f"\n  Accuracy :  Baseline = {base_acc:.4f} ({base_time:.1f}s)   MoE = {moe_acc:.4f} ({moe_time:.1f}s)")

        # Per-class 통합 비교표 (샘플수 내림차순)
        labels = np.arange(num_classes)
        p_b, r_b, f1_b, sup = precision_recall_fscore_support(
            y_test, base_preds, labels=labels, zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            y_test, moe_preds, labels=labels, zero_division=0)

        order = np.argsort(sup)[::-1]
        col_w = max(len(n) for n in class_names)
        hdr = (f"{'class':<{col_w}}"
               f"  {'prec(B)':>8} {'prec(M)':>8}"
               f"  {'recall(B)':>9} {'recall(M)':>9}"
               f"  {'f1(B)':>7} {'f1(M)':>7}"
               f"  {'support':>9}")
        sep = "-" * len(hdr)
        tbl = ["\n", hdr, sep]
        for i in order:
            tbl.append(
                f"{class_names[i]:<{col_w}}"
                f"  {p_b[i]:8.4f} {p_m[i]:8.4f}"
                f"  {r_b[i]:9.4f} {r_m[i]:9.4f}"
                f"  {f1_b[i]:7.4f} {f1_m[i]:7.4f}"
                f"  {int(sup[i]):9d}"
            )
        tbl.append(sep)
        tbl.append(
            f"{'macro avg':<{col_w}}"
            f"  {p_b.mean():8.4f} {p_m.mean():8.4f}"
            f"  {r_b.mean():9.4f} {r_m.mean():9.4f}"
            f"  {f1_b.mean():7.4f} {f1_m.mean():7.4f}"
            f"  {int(sup.sum()):9d}"
        )
        w = sup / sup.sum()
        tbl.append(
            f"{'weighted avg':<{col_w}}"
            f"  {(p_b*w).sum():8.4f} {(p_m*w).sum():8.4f}"
            f"  {(r_b*w).sum():9.4f} {(r_m*w).sum():9.4f}"
            f"  {(f1_b*w).sum():7.4f} {(f1_m*w).sum():7.4f}"
            f"  {int(sup.sum()):9d}"
        )
        logger.info("\n".join(tbl))

        # Per-class 비교표 CSV
        cmp_df = per_class_comparison(
            y_test, base_preds, moe_preds,
            num_classes, class_names, moe.tail_classes, moe.betas,
        )
        cmp_path = os.path.join(exp_dir, "baseline_vs_moe_per_class.csv")
        cmp_df.to_csv(cmp_path, index=False)

        # Specialist 활성화 요약
        act_df = specialist_activation_summary(
            y_test, info, class_names, moe.tail_classes
        )
        act_path = os.path.join(exp_dir, "specialist_activation.csv")
        act_df.to_csv(act_path, index=False)
        logger.info(f"\n--- Specialist 활성화율 (threshold=0.5) ---")
        for _, row in act_df.iterrows():
            logger.info(
                f"  {row['class_name']:33s} | "
                f"fire_all={row['fire_rate_all']:.3f} | "
                f"fire_true={row['fire_rate_true']:.3f} | "
                f"mean_conf_true={row['mean_conf_true']:.3f}"
            )
        logger.info(f"\nSaved: {cmp_path}")
        logger.info(f"Saved: {act_path}")

    else:
        logger.info(f"\n--- Baseline XGBoost ---")
        logger.info(f"Accuracy: {base_acc:.4f}  ({base_time:.1f}s)")
        logger.info("\n" + classification_report(
            y_test, base_preds,
            labels=np.arange(num_classes), target_names=class_names,
            digits=4, zero_division=0,
        ))

    logger.info(f"\nTotal elapsed: {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()

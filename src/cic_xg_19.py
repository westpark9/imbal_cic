#!/usr/bin/env python3
"""
XGBoost Mixture-of-Experts with XGBoost Router - cic_xg_19.py

주요 기능:
1. Router: multi:softprob objective 사용, soft probability 기반 expert 선택
2. SMOTE 적용 (Expert 학습용)
3. Uncertainty 기반 Router 입력
4. Type-based Expert Assignment (Benign은 Router가 처리, Attack은 Expert가 처리)
5. 취약 클래스에 대한 Cluster Expert 지원
6. 라우터가 선택한 클래스에 따라 expert가 직접 분류 (weight 주는 형태가 아님)
7. Extra expert와 일반 expert가 모두 해당 클래스를 처리할 수 있고 판단이 다르면, 더 높은 확률로 예측한 클래스 선택
"""

import os
import time
import json
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import joblib
import pickle
import cupy as cp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_curve,
    auc,
)
from scipy.stats import entropy
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Device resolver
# ---------------------------------------------------------

def resolve_xgb_device():
    if not torch.cuda.is_available():
        print("[Device] PyTorch: CUDA not available → cpu 사용")
        return "cpu"
    try:
        model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            device="cuda",
            tree_method="hist",
            objective="multi:softprob",
            num_class=2,
        )
        model.fit(
            np.zeros((4,4), dtype=np.float32),
            np.zeros(4, dtype=np.int32)
        )
        return "cuda"
    except Exception as e:
        return "cpu"

DEFAULT_DEVICE = resolve_xgb_device()
print(f"[Device] Final selected device = {DEFAULT_DEVICE}")

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

def setup_logging(exp_dir: str):
    log_file = os.path.join(exp_dir, "experiment.log")
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
# ---------------------------------------------------------
# Utility Functions: Uncertainty & Loss
# ---------------------------------------------------------

def calculate_normalized_entropy(probs: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    probs_smooth = probs + epsilon
    probs_smooth = probs_smooth / probs_smooth.sum(axis=1, keepdims=True)
    num_classes = probs.shape[1]
    entropies = -np.sum(probs_smooth * np.log(probs_smooth + epsilon), axis=1)
    normalized_entropies = entropies / (np.log(num_classes) + epsilon)
    return normalized_entropies

def softmax_numpy(x):
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# ---------------------------------------------------------
# Expert
# ---------------------------------------------------------

class SampleExpertXGBoostClassifier:
    """XGBoost expert (전체 label space 공유) - 기존 유지"""

    def __init__(
        self,
        num_classes: int,
        n_estimators: int = 100,
        random_state: int = 42,
        device: str = None,
    ):
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.local_to_global = None
        self.global_to_local = None
        self.local_num_classes = None
        self.xgb_model = None
        self.trained = False
        # Use passed device if provided, otherwise fall back to resolved default
        self.device = device if device is not None else DEFAULT_DEVICE

    def fit(self, features: np.ndarray, labels: np.ndarray, expert_idx: int = None):
        if len(features) == 0:
            print("       WARNING: no samples for this expert, skip training")
            return
        
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            print(f"       WARNING: only {len(unique_classes)} class(es), skip training")
            return
        
        self.local_to_global = np.sort(unique_classes)
        self.local_num_classes = len(self.local_to_global)
        self.global_to_local = {g: l for l, g in enumerate(self.local_to_global)}
        labels_local = np.array([self.global_to_local[g] for g in labels])
        
        xgb_kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=6,
            learning_rate=0.3,
            subsample=1,
            colsample_bytree=1,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            objective="multi:softprob",
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            num_class=self.local_num_classes,
            device=DEFAULT_DEVICE,
            early_stopping_rounds=15,
        )

        try:
            self.xgb_model = xgb.XGBClassifier(**xgb_kwargs)
        except Exception:
            print("Fall back to cpu")
            xgb_kwargs["device"] = "cpu"
            self.xgb_model = xgb.XGBClassifier(**xgb_kwargs)
        
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels_local, test_size=0.2, random_state=self.random_state, stratify=labels_local
        )
        
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        self.trained = True

    def predict_proba(self, features: np.ndarray, global_num_classes: int) -> np.ndarray:
        num_samples = features.shape[0]

        if not self.trained or self.xgb_model is None or not hasattr(self.xgb_model, "get_booster"):
            return np.ones((num_samples, global_num_classes), dtype=float) / float(global_num_classes)

        try:
            if self.device == "cuda":
                booster = self.xgb_model.get_booster()
                features_gpu = cp.asarray(features, dtype=cp.float32)
                proba_local = booster.inplace_predict(features_gpu)
                proba_local = cp.asnumpy(proba_local)
                proba_local = proba_local.reshape(num_samples, self.local_num_classes)
            else:
                proba_local = self.xgb_model.predict_proba(features)
            
            out = np.zeros((num_samples, global_num_classes), dtype=float)
            for local_idx, global_idx in enumerate(self.local_to_global):
                out[:, global_idx] = proba_local[:, local_idx]
            return out
        except Exception as e:
            print(f"       predict_proba failed: {e}")
            return np.ones((num_samples, global_num_classes), dtype=float) / float(global_num_classes)

# ---------------------------------------------------------
# XGBoost Router
# ---------------------------------------------------------

class XGBoostRouter:
    def __init__(
        self,
        num_experts: int,
        num_classes: int,
        n_estimators: int = 100,
        random_state: int = 42,
        device: str = None,
    ):
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.xgb_model = None
        self.trained = False
        self.device = DEFAULT_DEVICE
        self.num_router_classes = num_classes



    def fit(self, X: np.ndarray, router_labels: np.ndarray, expert_uncertainties: np.ndarray = None, sample_weights: np.array=None):
        N = X.shape[0]
        
        if expert_uncertainties is not None:
            X_input = np.hstack([X, expert_uncertainties])
            print("   Training XGBoost router (multi:softprob, Original Feat + Uncertainty)...")
        else:
            X_input = X
            print("   Training XGBoost router (multi:softprob, Original Feat)...")
            
        print(f"     Train samples: {N:,}, Features: {X_input.shape[1]}")
        
        unique_labels, counts = np.unique(router_labels, return_counts=True)
        logger = logging.getLogger("experiment")
        logger.info(f"     Router training label distribution:")
        for label, count in zip(unique_labels, counts):
            pct = count / N * 100.0
            logger.info(f"       Label {label}: {count:,} ({pct:.2f}%)")

        xgb_kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=6,
            learning_rate=0.3,
            subsample=1,
            colsample_bytree=1,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="mlogloss", 
            objective="multi:softprob",  
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            num_class=self.num_router_classes,
            device=DEFAULT_DEVICE,
        )

        try:
            self.xgb_model = xgb.XGBClassifier(**xgb_kwargs)
        except Exception:
            print("Fall back to cpu")
            xgb_kwargs["device"] = "cpu"
            self.xgb_model = xgb.XGBClassifier(**xgb_kwargs)

        if sample_weights is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                X_input, router_labels, sample_weights,
                test_size=0.2,
                random_state=self.random_state,
                stratify=router_labels,
            )
            self.xgb_model.fit(
                X_train, y_train,
                sample_weight=sw_train, 
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[sw_val],
                verbose=False,
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_input, router_labels,
                test_size=0.2,
                random_state=self.random_state,
                stratify=router_labels,
            )
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        best_iter = self.xgb_model.best_iteration if hasattr(self.xgb_model, 'best_iteration') else self.n_estimators
        logging.getLogger("experiment").info(f"   Router (multi:softprob) trained. Best Iter: {best_iter}")
        self.trained = True

    def predict_proba(self, X: np.ndarray, expert_uncertainties: np.ndarray = None) -> np.ndarray:
        if not self.trained or self.xgb_model is None:
            N = X.shape[0]
            return np.ones((N, self.num_router_classes), dtype=np.float32) / float(self.num_router_classes)

        if expert_uncertainties is not None:
            X_input = np.hstack([X, expert_uncertainties])
        else:
            X_input = X

        if self.device == "cuda":
            booster = self.xgb_model.get_booster()
            X_input_gpu = cp.asarray(X_input, dtype=cp.float32)
            probs = booster.inplace_predict(X_input_gpu)
            probs = cp.asnumpy(probs)
            probs = probs.reshape(X_input.shape[0], self.num_router_classes)
        else:
            probs = self.xgb_model.predict_proba(X_input)
        
        return probs.astype(np.float32)

# ---------------------------------------------------------
# 메인 실험 클래스
# ---------------------------------------------------------

class MetaRouterExperiment:
    def __init__(
        self,
        num_experts: int = 4,
        n_estimators: int = 100,
        router_n_estimators: int = 100,
        device: str = None,
        smote_threshold: int = 1000,
        router_target: str = "expert",
    ):
        self.num_experts = num_experts
        self.n_estimators = n_estimators
        self.expert_n_estimators = n_estimators
        self.router_n_estimators = router_n_estimators
        self.baseline_n_estimators = n_estimators
        self.device = DEFAULT_DEVICE
        self.smote_threshold = smote_threshold
        self.router_class_counts_before_smote = None
        self.router_num_samples_before_smote = None
        self.expert_classes = None
        self.expert_uncertainties_cache = None
        self.benign_class_indices = None
        self.num_router_classes = None
        self.benign_router_index = None
        self.router_target = router_target

        print("XGBoost Mixture-of-Experts with XGBoost Router")
        print(f"   Number of experts: {num_experts}")
        print(f"   Device        : {self.device}")

    def _save_baseline_model(self, model, directory: str):
        import pickle
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "baseline_xgb.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.getLogger("experiment").info(f"Saved baseline model to: {path}")

    def _load_baseline_model(self, directory: str):
        import pickle
        path = os.path.join(directory, "baseline_xgb.pkl")
        if not os.path.exists(path):
            logging.getLogger("experiment").info(
                f"No baseline_xgb.pkl found in {directory}"
            )
            return None
        with open(path, "rb") as f:
            model = pickle.load(f)
        logging.getLogger("experiment").info(f"Loaded baseline model from: {path}")
        return model

    def _save_ensemble_models(self, expert_classifiers, directory: str, extra_expert=None):
        import pickle
        os.makedirs(directory, exist_ok=True)

        experts_path = os.path.join(directory, "experts.pkl")
        with open(experts_path, "wb") as f:
            pickle.dump(expert_classifiers, f)

        router_path = os.path.join(directory, "xgb_router.pkl")
        with open(router_path, "wb") as f:
            pickle.dump(self.meta_router, f)

        extra_expert_path = None
        if extra_expert is not None:
            extra_expert_path = os.path.join(directory, "extra_expert.pkl")
            with open(extra_expert_path, "wb") as f:
                pickle.dump(extra_expert, f)

        config = {
            "num_experts": self.num_experts,
            "default_n_estimators": self.n_estimators,
            "baseline_n_estimators": self.baseline_n_estimators,
            "expert_n_estimators": self.expert_n_estimators,
            "router_n_estimators": self.router_n_estimators,
            "num_classes": self.num_classes,
            "expert_classes": self.expert_classes,
            "has_extra_expert": extra_expert is not None,
        }
        config_path = os.path.join(directory, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        logger = logging.getLogger("experiment")
        logger.info(f"Saved experts to: {experts_path}")
        logger.info(f"Saved XGBoost router to: {router_path}")
        if extra_expert is not None:
            logger.info(f"Saved extra expert to: {extra_expert_path}")
        logger.info(f"Saved model config to: {config_path}")

    def _load_ensemble_models(self, directory: str):
        import pickle
        logger = logging.getLogger("experiment")

        experts_path = os.path.join(directory, "experts.pkl")
        router_path = os.path.join(directory, "xgb_router.pkl")

        if not (os.path.exists(experts_path) and os.path.exists(router_path)):
            logger.info(
                f"No experts.pkl or xgb_router.pkl found in {directory}"
            )
            return None, None

        with open(experts_path, "rb") as f:
            expert_classifiers = pickle.load(f)

        with open(router_path, "rb") as f:
            meta_router = pickle.load(f)

        logger.info(f"Loaded experts from: {experts_path}")
        logger.info(f"Loaded XGBoost router from: {router_path}")

        return expert_classifiers, meta_router

    def set_random_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    def load_preprocessed_data(self, pkl_path: str):

        print(f"   Loading preprocessed data from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        X = data["X"]
        y = data["y"]
        label_encoder = data["label_encoder"]
        feature_columns = data["feature_columns"]
        file_groups = data.get("file_groups", None)
        dataset_type = data.get("dataset_type", "unknown")
        has_predefined_split = data.get("has_predefined_split", False)
        train_indices = data.get("train_indices", None)
        test_indices = data.get("test_indices", None)
        num_classes = data.get("num_classes", len(np.unique(y)))

        print(f"     Dataset type: {dataset_type}")
        print(f"     Total samples: {len(X):,}")
        print(f"     Features: {len(feature_columns)}")
        print(f"     Classes: {num_classes}")

        if has_predefined_split and train_indices is not None and test_indices is not None:
            print(f"     Predefined split: Train={len(train_indices):,}, Test={len(test_indices):,}")

        return (
            X,
            y,
            label_encoder,
            feature_columns,
            file_groups,
            dataset_type,
            has_predefined_split,
            train_indices,
            test_indices,
            num_classes,
        )

    def setup_dataset(self, seed: int, data_path: str):
        self.set_random_seed(seed)
        (
            X,
            y,
            label_encoder_loaded,
            self.feature_columns,
            self.file_groups,
            self.dataset_type,
            has_split,
            train_idx,
            test_idx,
            _,
        ) = self.load_preprocessed_data(data_path)

        if label_encoder_loaded is not None:
            print("   Using original label_encoder from pickle file")
            self.label_encoder = label_encoder_loaded
            if isinstance(y[0], str) or (hasattr(y, 'dtype') and y.dtype == 'object'):
                print("   Encoding labels to integers using original label_encoder")
                y = label_encoder_loaded.transform(y)
            else:
                print("   Labels already encoded, using as-is")
            self.num_classes = len(label_encoder_loaded.classes_)
        else:
            print("   No label_encoder found, creating new one")
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoder = le
            self.num_classes = len(le.classes_)
        print(f"   Unique classes: {self.num_classes}, labels: {np.unique(y)}")

        benign_tokens = {"benign", "normal", "benign_traffic", "normal_traffic"}
        self.benign_class_indices = [
            idx for idx, name in enumerate(self.label_encoder.classes_)
            if name.lower() in benign_tokens
        ]
        self.benign_router_index = None
        if self.benign_class_indices:
            print(f"   Benign/Normal classes routed only by router: {self.benign_class_indices}")

        if self.router_target == 'expert':
            self.num_router_classes = self.num_experts
            print(f"   Router will predict expert assignment (0~{self.num_experts-1})")
        else:  # 'class'
            self.num_router_classes = self.num_classes
            print(f"   Router will classify all {self.num_router_classes} classes directly (no special BENIGN class).")

        if has_split and train_idx is not None and test_idx is not None:
            print("     Using predefined split...")
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
        else:
            print("     Temporal split (8:2) by class index order...")
            train_indices = []
            test_indices = []
            for cid in np.unique(y):
                mask = (y == cid)
                idx = np.where(mask)[0]
                n_train = int(len(idx) * 0.8)
                train_indices.extend(idx[:n_train])
                test_indices.extend(idx[n_train:])
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

        # SMOTE
        print("     Applying SMOTE for tail classes...")
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_count_dict = {int(c): int(n) for c, n in zip(unique_classes, class_counts)}
        
        tail_threshold = self.smote_threshold
        tail_classes = [cid for cid, count in class_count_dict.items() if count < tail_threshold]
        
        if len(tail_classes) > 0:
            print(f"     Found {len(tail_classes)} tail classes (count < {tail_threshold}): {tail_classes}")
            print(f"     Tail class counts: {[class_count_dict[c] for c in tail_classes]}")
            
            sampling_strategy = {cid: tail_threshold for cid in tail_classes}
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
            
            print(f"     Applying SMOTE: {sampling_strategy}")
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            unique_after, counts_after = np.unique(y_train, return_counts=True)
            print(f"     After SMOTE - Train samples: {len(X_train):,}")
            print(f"     Class distribution after SMOTE:")
            for cid, count in zip(unique_after, counts_after):
                class_name = self.label_encoder.inverse_transform([int(cid)])[0] if hasattr(self, 'label_encoder') else f"Class_{int(cid)}"
                print(f"       {class_name} (class {int(cid)}): {int(count):,}")
        else:
            print(f"     No tail classes found (all classes have >= {tail_threshold} samples). Skipping SMOTE.")

        print("     Applying StandardScaler...")

        if self.dataset_type == "unswnb15":
            cat_cols = {"proto", "state", "service"}
            col_to_idx = {col: i for i, col in enumerate(self.feature_columns)}

            norm_indices = [
                idx for col, idx in col_to_idx.items()
                if col not in cat_cols
            ]

            print(f"     UNSW-NB15: excluding categorical columns from scaling: "
                  f"{[c for c in self.feature_columns if c in cat_cols]}")

            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()

            if len(norm_indices) > 0:
                X_train_scaled[:, norm_indices] = scaler.fit_transform(X_train[:, norm_indices])
                X_test_scaled[:, norm_indices] = scaler.transform(X_test[:, norm_indices])
            else:
                print("     WARNING: no columns selected for scaling on UNSW-NB15.")

            X_train = X_train_scaled
            X_test = X_test_scaled
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"     Train: {len(X_train):,}, Test: {len(X_test):,}")

    def train_baseline(self, seed: int):
        print("   Training baseline XGBoost...")
        self.set_random_seed(seed)

        tree_method = "hist"
        
        xgb_kwargs = dict(
            n_estimators=self.baseline_n_estimators,
            max_depth=6,
            learning_rate=0.3,
            subsample=1,
            colsample_bytree=1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            objective="multi:softprob",
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method=tree_method,
            num_class=self.num_classes,
            early_stopping_rounds=15,  # 주석처리
        )
        xgb_kwargs["device"] = DEFAULT_DEVICE

        try:
            model = xgb.XGBClassifier(**xgb_kwargs)
        except Exception:
            print("Fall back to cpu")
            xgb_kwargs["device"] = "cpu"
            model = xgb.XGBClassifier(**xgb_kwargs)
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=seed, stratify=self.y_train
        )
        
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else self.baseline_n_estimators
        logger = logging.getLogger("experiment")
        if best_iteration < self.baseline_n_estimators:
            logger.info(f"   Baseline early stopped at iteration {best_iteration}/{self.baseline_n_estimators}")
        else:
            logger.info(f"   Baseline completed all {self.baseline_n_estimators} iterations (no early stopping)")
        
        return model

    def evaluate_baseline(self, model):
        preds = model.predict(self.X_test)
        try:
            y_proba = model.predict_proba(self.X_test)
        except:
            y_proba = None
        return self._evaluate_predictions(preds, y_proba=y_proba)

    def assign_samples_to_experts_type(self):
        
        if self.dataset_type not in ["unswnb15", "cic2017", "cic2018"]:
            raise ValueError(
                f"Type assignment is only supported for UNSW-NB15, CIC2017, and CIC2018. "
                f"Current dataset: {self.dataset_type}"
            )

        y = self.y_train
        E = self.num_experts

        if not hasattr(self, "label_encoder"):
            raise ValueError("label_encoder is required for type assignment")

        class_names = self.label_encoder.classes_
        name_to_id = {name: idx for idx, name in enumerate(class_names)}

        # ------------------------------
        # dataset별 attack type 그룹 정의
        # ------------------------------
        if self.dataset_type == "unswnb15":
            type_groups = [
                ["DoS", "Worms"],
                ["Analysis", "Reconnaissance"],
                ["Exploits", "Fuzzers", "Generic"],
                ["Backdoor", "Shellcode"],
            ]
        elif self.dataset_type == "cic2017":
            type_groups = [
                ["DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris", "PortScan"],
                ["FTP-Patator", "SSH-Patator"],
                ["Web Attack � Brute Force", "Web Attack � Sql Injection", "Web Attack � XSS"],
                ["Bot", "Heartbleed", "Infiltration"],
            ]
        else:  # "cic2018"
            type_groups = [
                ["DDOS attack-HOIC", "DDOS attack-LOIC-UDP", "DDoS attacks-LOIC-HTTP",
                 "DoS attacks-GoldenEye", "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"],
                ["FTP-BruteForce", "SSH-Bruteforce"],
                ["Bot", "Infilteration"],
                ["Brute Force -Web", "SQL Injection", "Brute Force -XSS"],
            ]

        if len(type_groups) > E:
            print(f"   WARNING: num_experts={E} < defined type groups={len(type_groups)}. "
                  f"Only first {E} groups will be used.")
            type_groups = type_groups[:E]

        self.expert_classes = {}
        class_to_expert = {}
        benign_set = set(self.benign_class_indices or [])

        for e, group in enumerate(type_groups):
            class_ids = []
            for name in group:
                if name not in name_to_id:
                    print(f"   WARNING: class name '{name}' not found in label encoder. Skipping.")
                    continue
                cid = int(name_to_id[name])
                if cid in benign_set:
                    continue
                class_ids.append(cid)

            if not class_ids:
                print(f"   WARNING: Expert {e} has no valid classes in type group {group}.")
            self.expert_classes[e] = sorted(class_ids)
            for cid in class_ids:
                class_to_expert[cid] = e

        self.class_to_expert = class_to_expert

        # 샘플별 expert_assignments 생성 (매핑되지 않은 클래스는 -1로 둔다)
        expert_assignments = np.full_like(y, -1, dtype=np.int64)

        for cid, e in class_to_expert.items():
            mask = (y == cid)
            expert_assignments[mask] = e

        print("   Type-based assignment (fixed class groups):")
        for e in range(E):
            classes_e = self.expert_classes.get(e, [])
            if hasattr(self, "label_encoder"):
                cls_str = ", ".join(
                    [f"{self.label_encoder.classes_[c]}" for c in classes_e]
                )
            else:
                cls_str = ", ".join([f"class{c}" for c in classes_e])

            N_e = int((expert_assignments == e).sum())
            print(f"     Expert {e}: {len(classes_e)} classes, {N_e:,} samples -> {cls_str}")

        # 매핑되지 않은 클래스(주로 BENIGN/NORMAL)가 남아 있는지 출력
        unassigned = int((expert_assignments == -1).sum())
        if unassigned > 0:
            print(f"   Unassigned samples (routed only by router): {unassigned:,}")

        return expert_assignments

    def train_experts(self, expert_assignments: np.ndarray, seed: int):
        self.set_random_seed(seed)
        E = self.num_experts
        experts = []

        benign_indices = set(self.benign_class_indices or [])

        print("   Training experts...")
        for e in range(E):
            print(f"   === Expert {e} ===")
            mask = (expert_assignments == e)
            X_sub = self.X_train[mask]
            y_sub = self.y_train[mask]

            if benign_indices:
                benign_mask = np.isin(y_sub, list(benign_indices))
                if benign_mask.any():
                    print(f"      Removing {benign_mask.sum():,} BENIGN/NORMAL samples from expert {e} training set")
                    X_sub = X_sub[~benign_mask]
                    y_sub = y_sub[~benign_mask]

            expert = SampleExpertXGBoostClassifier(
                num_classes=self.num_classes,
                n_estimators=self.expert_n_estimators,
                random_state=seed,
                device=self.device,
            )
            expert.fit(X_sub, y_sub, expert_idx=e)

            if hasattr(expert, "trained") and expert.trained and len(X_sub) > 0:
                probs_train = expert.predict_proba(X_sub, self.num_classes)
                preds_train = np.argmax(probs_train, axis=1)
                print(f"     Expert {e} (TRAINING):")
                self._print_detailed_metrics(preds_train, y_sub, f"Expert {e}")

            experts.append(expert)

        return experts
  

    def compute_expert_outputs(
        self,
        expert_classifiers,
        X: np.ndarray,
        description: str = "train",
        batch_size: int = 10000,
    ):
        
        N = len(X)
        E = self.num_experts
        C = self.num_classes

        print(f"   Computing expert outputs on {description} data...")
        print(f"     Samples: {N:,}, Experts: {E}, Classes: {C}")
        num_batches = (N + batch_size - 1) // batch_size

        probs_all = np.zeros((N, E, C), dtype=np.float32)

        pbar = tqdm(
            range(num_batches),
            desc=f"   Expert outputs ({description})",
            unit="batch",
            ncols=100,
        )

        for b in pbar:
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            X_batch = X[start:end]

            batch_size_now = end - start

            for e, expert in enumerate(expert_classifiers):
                prob = expert.predict_proba(X_batch, C).astype(np.float32)
                probs_all[start:end, e, :] = prob

            pbar.set_postfix(size=batch_size_now)

        print(f"   Expert probs shape:  {probs_all.shape} (dtype={probs_all.dtype})")
        return probs_all

    def prepare_router_labels(self, y_train: np.ndarray, expert_assignments: np.ndarray = None) -> np.ndarray:
        
        if expert_assignments is not None:
            router_labels = expert_assignments.copy()
        else:
            if self.expert_classes is None:
                raise ValueError("expert_classes is not set. Call assign_samples_to_experts_* first.")
            
            router_labels = np.zeros_like(y_train, dtype=np.int64)
            
            for e in range(self.num_experts):
                classes_e = self.expert_classes[e]
                for c in classes_e:
                    mask = (y_train == c)
                    router_labels[mask] = e
        
        counts = np.bincount(router_labels, minlength=self.num_experts)
        total = float(len(y_train))
        print("   Router label distribution (expert assignment):")
        for e in range(self.num_experts):
            classes_e = self.expert_classes[e]
            cls_str = ", ".join([f"{c}" for c in classes_e])
            print(f"     Expert {e}: {counts[e]:,} ({counts[e] / total * 100:.1f}%) - classes [{cls_str}]")
        
        return router_labels

    def calculate_expert_uncertainties(self, expert_classifiers, X: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        N = len(X)
        E = self.num_experts
        
        print(f"   Calculating expert uncertainties (batch_size={batch_size})...")
        
        expert_uncertainties = np.zeros((N, E), dtype=np.float32)
        num_batches = (N + batch_size - 1) // batch_size
        
        for e, expert in enumerate(expert_classifiers):
            if not expert.trained or expert.xgb_model is None:
                expert_uncertainties[:, e] = 1.0
                continue
            
            uncertainties = []
            for b in range(num_batches):
                start = b * batch_size
                end = min((b + 1) * batch_size, N)
                X_batch = X[start:end]
                
                probs_batch = expert.predict_proba(X_batch, self.num_classes)  # (batch_size, C)
                
                batch_uncertainties = calculate_normalized_entropy(probs_batch)
                uncertainties.append(batch_uncertainties)
            
            expert_uncertainties[:, e] = np.concatenate(uncertainties)
        
        print(f"   Expert uncertainties shape: {expert_uncertainties.shape}")
        print(f"   Mean uncertainty per expert: {expert_uncertainties.mean(axis=0)}")
        
        return expert_uncertainties

    def train_meta_router(self, router_labels, sample_weights=None, router_target=None, expert_uncertainties=None):
        if router_target is not None:
            self.router_target = router_target

        # Sample Weight (Sqrt)
        u, c = np.unique(router_labels, return_counts=True)
        max_c = c.max()
        cw = {lbl: np.sqrt(max_c/cnt) for lbl, cnt in zip(u, c)}
        # sw = np.array([cw[l] for l in router_labels], dtype=np.float32)
        # print(f"   Router Sample Weights (Sqrt): Min={sw.min():.2f}, Max={sw.max():.2f}")

        # Create router with explicit keyword arguments to avoid positional mistakes
        self.meta_router = XGBoostRouter(
            num_experts=self.num_experts,
            num_classes=self.num_router_classes,
            n_estimators=self.router_n_estimators,
            device=self.device,
        )
        # If caller provided sample_weights, pass them along; otherwise None
        self.meta_router.fit(self.X_train, router_labels, expert_uncertainties, sample_weights=sample_weights)
        return self.meta_router

    def evaluate_simple_ensemble(self, expert_classifiers, batch_size: int = 1024):
        probs_test = self.compute_expert_outputs(
            expert_classifiers,
            self.X_test,
            description="test",
            batch_size=batch_size,
        )

        N = self.X_test.shape[0]
        E = self.num_experts
        C = self.num_classes

        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        print("   Evaluating simple ensemble (average) on test data...")
        print(f"     Test samples: {N:,}, Experts: {E}, Classes: {C}")
        
        print("\n   Individual Expert Performance (TEST - only trained classes):")
        for e in range(E):
            expert = expert_classifiers[e]
            probs_e = probs_test[:, e, :]
            preds_e = np.argmax(probs_e, axis=1)
            
            if not (hasattr(expert, 'trained') and expert.trained):
                print(f"     Expert {e}: WARNING - Expert not trained, skipping evaluation")
                continue
            
            if hasattr(expert, 'local_to_global') and expert.local_to_global is not None:
                trained_classes = expert.local_to_global
                mask = np.isin(self.y_test, trained_classes)
                y_test_filtered = self.y_test[mask]
                preds_e_filtered = preds_e[mask]
                
                if len(y_test_filtered) > 0:
                    print(f"     Expert {e} trained classes: {trained_classes.tolist()}")
                    self._print_detailed_metrics(
                        preds_e_filtered,
                        y_test_filtered,
                        f"Expert {e}",
                        num_classes=len(trained_classes),
                        is_router=False,
                    )
                else:
                    print(f"     Expert {e}: No test samples for trained classes {trained_classes.tolist()}")
            else:
                print(f"     Expert {e}: Evaluating on all classes (no local_to_global mapping)")
                self._print_detailed_metrics(
                    preds_e,
                    self.y_test,
                    f"Expert {e}",
                    num_classes=self.num_classes,
                    is_router=False,
                )

        final_probs = probs_test.mean(axis=1)
        predictions = np.argmax(final_probs, axis=1)
        
        print("\n   Simple Ensemble (Average) Performance (TEST):")
        self._print_detailed_metrics(predictions, self.y_test, "Simple Ensemble", num_classes=self.num_classes, is_router=False, y_proba=final_probs)

        print(f"\n   Evaluation completed: {len(predictions):,} predictions")
        return self._evaluate_predictions(predictions, y_proba=final_probs)

    def evaluate_meta_ensemble(self, experts, batch_size=2048):
        print("\n" + "="*60)
        print(f"TEST EVALUATION (Router Target: {self.router_target.upper()})")
        print("="*60)

        # 1. Expert Outputs
        expert_probs = self.compute_expert_outputs(experts, self.X_test, batch_size) # (N, E, C)
        
        # 2. Router Outputs
        unc_test = self.calculate_expert_uncertainties(experts, self.X_test, batch_size)
        router_probs = self.meta_router.predict_proba(self.X_test, unc_test) # (N, Router_Out)
        
        N, E, C = expert_probs.shape
        final_probs = np.zeros((N, C), dtype=np.float32)
        
        benign_set = set(self.benign_class_indices or [])
        c2e = self.class_to_expert # Class ID -> Expert ID Mapping

        used_experts = np.full(N, -1, dtype=int)

        # 3. Combine Logic (Soft Probability 기반)
        for i in range(N):
            # --- [CASE A] Router predicts EXPERT ID ---
            if self.router_target == 'expert':
                # 라우터가 선택한 Expert의 예측만 사용 (weight 주는 형태가 아님)
                chosen_expert = int(np.argmax(router_probs[i]))
                final_probs[i] = expert_probs[i, chosen_expert]
                used_experts[i] = chosen_expert

            # --- [CASE B] Router predicts CLASS ID ---
            else:
                pred_class = int(np.argmax(router_probs[i]))
                
                # 1) Benign으로 예측했으면? -> Router 확률 사용
                if pred_class in benign_set:
                    final_probs[i] = router_probs[i]
                    used_experts[i] = -1 # Router Handled
                
                # 2) Attack Class로 예측했으면? -> 담당 Expert가 직접 분류
                else:
                    target_expert = c2e.get(pred_class, -1)
                    if target_expert != -1:
                        # Expert의 예측만 사용 (weight 주는 형태가 아님)
                        final_probs[i] = expert_probs[i, target_expert]
                        used_experts[i] = target_expert
                    else:
                        # 매핑 안된 클래스 (거의 없음) -> Router 예측값 사용
                        final_probs[i] = router_probs[i]
                        used_experts[i] = -1

        # Chart Logic (Expert Usage)
        if hasattr(self, "label_encoder"): cnames = self.label_encoder.classes_
        else: cnames = [f"C{i}" for i in range(C)]
        
        logger = logging.getLogger("experiment")
        logger.info("\nClass-wise Expert Usage:")
        header = f"{'Class':<20}" + "".join([f"  {'Exp'+str(e):>6}" for e in range(E)]) + f"  {'ACC':>6}"
        logger.info(header)
        
        for cid, cname in enumerate(cnames):
            mask = (self.y_test == cid)
            if mask.sum() == 0: continue
            row = f"{cname:<20}"
            gt_e = c2e.get(cid, -1)
            for e in range(E):
                cnt = (used_experts[mask] == e).sum()
                mark = "*" if e == gt_e else " "
                row += f"  {mark}{cnt:>5}"
            # Expert 할당 정확도: 올바른 Expert로 할당된 비율
            if gt_e != -1:
                acc = (used_experts[mask] == gt_e).mean()
            else:
                # 담당 Expert가 없는 경우 (예: BENIGN)는 Router가 처리(-1)해야 정확
                acc = (used_experts[mask] == -1).mean()
            row += f"  {acc:>6.4f}"
            logger.info(row)

        metrics = self._evaluate_predictions(final_preds, y_proba=final_probs)

        if self.router_target == 'class':
            per_class_acc = metrics[2]
            vulnerable = self._log_vulnerable_class_predictions(
                final_probs,
                final_preds,
                per_class_acc,
                threshold=0.5,
            )
            self._build_additional_expert(vulnerable)
            raise SystemExit("Early stop after vulnerable class handling and additional expert build.")

        return metrics

    def _log_vulnerable_class_predictions(
        self,
        final_probs: np.ndarray,
        final_preds: np.ndarray,
        per_class_acc: np.ndarray,
        threshold: float = 1.0,
        source_name: str = "FINAL",
    ):
        logger = logging.getLogger("experiment")
        cnames = self.label_encoder.classes_ if hasattr(self, "label_encoder") else [f"C{i}" for i in range(self.num_classes)]

        vulnerable = [cid for cid, acc in enumerate(per_class_acc) if acc <= threshold]
        if not vulnerable:
            logger.info(f"No vulnerable classes found under accuracy threshold {threshold}. (source={source_name})")
            return []

        logger.info(f"Vulnerable classes (acc <= {threshold}) from {source_name}: {[cnames[c] for c in vulnerable]}")
        for cid in vulnerable:
            mask = (self.y_test == cid)
            if mask.sum() == 0:
                logger.info(f"{cnames[cid]} | no test samples; skip logging.")
                continue

            probs_for_class = final_probs[mask]  # (n_samples, C)
            mean_probs = probs_for_class.mean(axis=0)
            order = np.argsort(-mean_probs)

            parts = []
            for pid in order:
                if mean_probs[pid] <= 0:
                    continue
                parts.append(f"{cnames[pid]}({mean_probs[pid]:.2f})")
            logger.info(f"{cnames[cid]} | " + ", ".join(parts))

        return vulnerable

    def _build_additional_expert(self, vulnerable_classes):
        logger = logging.getLogger("experiment")
        if not vulnerable_classes:
            logger.info("No additional expert created (no vulnerable classes).")
            return None

        mask = np.isin(self.y_train, vulnerable_classes)
        X_sub = self.X_train[mask]
        y_sub = self.y_train[mask]
        if X_sub.shape[0] == 0:
            logger.info("No training samples for vulnerable classes; skip additional expert.")
            return None

        class_names = None
        if hasattr(self, "label_encoder"):
            class_names = [self.label_encoder.classes_[c] for c in vulnerable_classes]
            logger.info(f"Additional expert classes (names): {class_names}")
        logger.info(f"Additional expert classes (ids): {vulnerable_classes}")
        logger.info(f"Training samples for additional expert: {X_sub.shape[0]}")

        extra_expert = SampleExpertXGBoostClassifier(
            num_classes=self.num_classes,
            n_estimators=self.expert_n_estimators,
            random_state=getattr(self, "seed", 42),
            device=self.device,
        )
        extra_expert.fit(X_sub, y_sub, expert_idx=self.num_experts)
        logger.info(f"Additional expert trained; classes covered: {np.unique(y_sub).tolist()}")
        self.extra_expert = extra_expert
        return extra_expert

    # ------------------------------
    # New routing/cluster helpers
    # ------------------------------

    def _build_cluster_expert(self, class_ids):
        logger = logging.getLogger("experiment")
        if not class_ids:
            logger.info("No cluster expert built (no cluster classes).")
            return None
        mask = np.isin(self.y_train, class_ids)
        X_sub = self.X_train[mask]
        y_sub = self.y_train[mask]
        if X_sub.shape[0] == 0:
            logger.info("Cluster expert skipped (no training samples for cluster classes).")
            return None

        uniq, cnt = np.unique(y_sub, return_counts=True)
        cnames = self.label_encoder.classes_ if hasattr(self, "label_encoder") else [f"C{i}" for i in range(self.num_classes)]
        logger.info(f"Cluster expert training data (no 2nd SMOTE applied):")
        for cid, count in zip(uniq, cnt):
            cname = cnames[int(cid)] if int(cid) < len(cnames) else f"Class_{int(cid)}"
            logger.info(f"  {cname} (class {int(cid)}): {int(count):,} samples")

        if hasattr(self, "label_encoder"):
            cname = [self.label_encoder.classes_[c] for c in class_ids]
            logger.info(f"Cluster expert classes (names): {cname}")
        logger.info(f"Cluster expert classes (ids): {class_ids}, samples: {X_sub.shape[0]}")
        cluster_expert = SampleExpertXGBoostClassifier(
            num_classes=self.num_classes,
            n_estimators=self.expert_n_estimators,
            random_state=getattr(self, "seed", 42),
            device=self.device,
        )
        cluster_expert.fit(X_sub, y_sub, expert_idx=self.num_experts)
        return cluster_expert

    def _combine_router_outputs(
        self,
        expert_probs,
        router_probs,
        class_to_expert,
        benign_set=None,
        extra_expert_probs=None,
        cluster_class_ids=None,
    ):
        benign_set = set(benign_set or [])
        cluster_set = set(cluster_class_ids or [])
        N, E, C = expert_probs.shape
        final_probs = np.zeros((N, C), dtype=np.float32)
        used_experts = np.full(N, -1, dtype=int)

        # 3. Combine Logic: 라우터가 선택한 클래스에 따라 expert가 직접 분류
        for i in range(N):
            r_pred = int(np.argmax(router_probs[i]))  # Router's top prediction index
            
            # --- [CASE A] Router predicts EXPERT ID ---
            if self.router_target == 'expert':
                chosen_expert = int(r_pred)
                # Chosen Expert의 예측값 사용
                final_probs[i] = expert_probs[i, chosen_expert]
                used_experts[i] = chosen_expert

            # --- [CASE B] Router predicts CLASS ID ---
            else:
                pred_class = int(r_pred)
                
                # 1) Benign으로 예측했으면? -> Router 확률 사용
                if pred_class in benign_set:
                    # Router의 Class Probability를 사용
                    final_probs[i] = router_probs[i]
                    used_experts[i] = -1  # Router Handled
                
                # 2) Attack Class로 예측했으면? -> 담당 Expert가 직접 분류
                else:
                    target_expert = class_to_expert.get(pred_class, -1)
                    
                    # 담당 Expert가 있으면 그 Expert의 예측 사용 (weight 주는 형태가 아님)
                    if target_expert != -1:
                        base_probs = expert_probs[i, target_expert]
                        base_pred = int(np.argmax(base_probs))
                        base_prob = base_probs[base_pred]
                        base_used = target_expert
                    else:
                        # 매핑 안된 클래스 (거의 없음) -> Router 예측값 사용
                        base_probs = router_probs[i]
                        base_pred = int(np.argmax(base_probs))
                        base_prob = base_probs[base_pred]
                        base_used = -1
                    
                    # Extra Expert가 있고, 예측한 클래스가 취약 클래스라면 Extra Expert도 고려
                    if extra_expert_probs is not None and pred_class in cluster_set:
                        extra_probs = extra_expert_probs[i]
                        extra_pred = int(np.argmax(extra_probs))
                        extra_prob = extra_probs[extra_pred]
                        
                        # 두 판단이 다르면 더 높은 확률로 예측한 클래스 선택
                        if base_pred != extra_pred:
                            if extra_prob > base_prob:
                                # Extra Expert의 예측이 더 높은 확률
                                final_probs[i] = extra_probs
                                used_experts[i] = -2  # Extra Expert 사용
                            else:
                                # 일반 Expert의 예측이 더 높은 확률
                                final_probs[i] = base_probs
                                used_experts[i] = base_used
                        else:
                            # 두 판단이 같으면 일반 Expert 사용
                            final_probs[i] = base_probs
                            used_experts[i] = base_used
                    else:
                        # Extra Expert가 없거나 취약 클래스가 아니면 일반 Expert만 사용
                        final_probs[i] = base_probs
                        used_experts[i] = base_used

        return final_probs, used_experts
    def _analyze_extra_expert_effectiveness(
        self, used_experts, final_probs, extra_probs, expert_probs, router_probs,
        cluster_class_ids, class_to_expert, benign_set
    ):
        """Extra expert 사용 효과 분석"""
        logger = logging.getLogger("experiment")
        logger.info("\n" + "="*60)
        logger.info("Extra Expert Effectiveness Analysis")
        logger.info("="*60)
        
        cluster_set = set(cluster_class_ids)
        cnames = self.label_encoder.classes_ if hasattr(self, "label_encoder") else [f"C{i}" for i in range(self.num_classes)]
        
        # Extra expert를 사용한 샘플들
        extra_mask = (used_experts == -2)
        extra_count = extra_mask.sum()
        logger.info(f"Total samples using Extra Expert: {extra_count:,} ({extra_count/len(used_experts)*100:.2f}%)")
        
        if extra_count == 0:
            logger.info("No samples used Extra Expert. Skipping analysis.")
            return
        
        final_preds = np.argmax(final_probs, axis=1)
        
        extra_final_preds = final_preds[extra_mask]
        extra_y_true = self.y_test[extra_mask]
        extra_acc = (extra_final_preds == extra_y_true).mean()
        logger.info(f"Accuracy on samples using Extra Expert: {extra_acc:.4f}")
        
        logger.info("\nPer-Class Analysis (samples using Extra Expert):")
        for cid in cluster_class_ids:
            class_mask = (self.y_test == cid)
            extra_class_mask = extra_mask & class_mask
            if extra_class_mask.sum() > 0:
                # Extra expert를 사용한 해당 클래스 샘플들의 성능
                extra_class_preds = np.argmax(final_probs[extra_class_mask], axis=1)
                extra_class_acc = (extra_class_preds == cid).mean()
                
                # Base를 사용했을 때의 예상 성능 (비교용)
                base_class_mask = (~extra_mask) & class_mask
                if base_class_mask.sum() > 0:
                    base_class_preds = np.argmax(final_probs[base_class_mask], axis=1)
                    base_class_acc = (base_class_preds == cid).mean()
                else:
                    base_class_acc = 0.0
                
                logger.info(f"  {cnames[cid]}:")
                logger.info(f"    Extra Expert used: {extra_class_mask.sum()} samples, ACC: {extra_class_acc:.4f}")
                logger.info(f"    Base used: {base_class_mask.sum()} samples, ACC: {base_class_acc:.4f}")
                
                # Router가 해당 클래스를 예측한 경우 분석
                # Router가 클래스 X를 예측한 모든 샘플 (실제 클래스와 무관)
                router_predicted_mask = (np.argmax(router_probs, axis=1) == cid)
                router_predicted_count = router_predicted_mask.sum()
                
                if router_predicted_count > 0:
                    # Router가 예측한 샘플들 중 실제로 해당 클래스인 샘플
                    router_correct_mask = router_predicted_mask & class_mask
                    router_correct_count = router_correct_mask.sum()
                    
                    # Router가 예측한 샘플들 중 최종 예측이 맞은 샘플 (Router만 사용)
                    router_only_correct = (final_preds[router_predicted_mask] == self.y_test[router_predicted_mask]).sum()
                    
                    # Router가 예측한 샘플들 중 Extra expert를 사용한 샘플
                    router_extra_mask = router_predicted_mask & extra_mask
                    router_extra_count = router_extra_mask.sum()
                    if router_extra_count > 0:
                        router_extra_correct = (final_preds[router_extra_mask] == self.y_test[router_extra_mask]).sum()
                    else:
                        router_extra_correct = 0
                    
                    # Router가 예측한 샘플들 중 Base를 사용한 샘플
                    router_base_mask = router_predicted_mask & (~extra_mask)
                    router_base_count = router_base_mask.sum()
                    if router_base_count > 0:
                        router_base_correct = (final_preds[router_base_mask] == self.y_test[router_base_mask]).sum()
                    else:
                        router_base_correct = 0
                    
                    logger.info(f"    Router predicted {cnames[cid]}: {router_predicted_count} samples")
                    logger.info(f"      - Actual {cnames[cid]} samples: {router_correct_count} (out of {class_mask.sum()} total {cnames[cid]} samples)")
                    logger.info(f"      - Correct predictions: {router_only_correct} (using Router+Base)")
                    logger.info(f"        * Base used: {router_base_count} samples, Correct: {router_base_correct}")
                    logger.info(f"        * Extra Expert used: {router_extra_count} samples, Correct: {router_extra_correct}")

    def _log_expert_usage(self, used_experts, final_probs, class_to_expert, extra_used=False):
        """클래스별 어떤 expert가 사용되었는지 로깅"""
        E = self.num_experts
        cnames = self.label_encoder.classes_ if hasattr(self, "label_encoder") else [f"C{i}" for i in range(final_probs.shape[1])]
        logger = logging.getLogger("experiment")
        logger.info("\nClass-wise Expert Usage:")
        header = f"{'Class':<20}" + "".join([f"  {'Exp'+str(e):>6}" for e in range(E)])
        if extra_used:
            header += "  Extra "
        header += f"  {'ACC':>6}"
        logger.info(header)

        for cid, cname in enumerate(cnames):
            mask = (self.y_test == cid)
            if mask.sum() == 0:
                continue
            row = f"{cname:<20}"
            gt_e = class_to_expert.get(cid, -1)
            for e in range(E):
                cnt = (used_experts[mask] == e).sum()
                mark = "*" if e == gt_e else " "
                row += f"  {mark}{cnt:>5}"
            if extra_used:
                cnt_extra = (used_experts[mask] == -2).sum()
                row += f"  {cnt_extra:>6}"
            # Expert 할당 정확도: 올바른 Expert로 할당된 비율
            if gt_e != -1:
                acc = (used_experts[mask] == gt_e).mean()
            else:
                # 담당 Expert가 없는 경우 (예: BENIGN)는 Router가 처리(-1)해야 정확
                acc = (used_experts[mask] == -1).mean()
            row += f"  {acc:>6.4f}"
            logger.info(row)

    def evaluate_meta_ensemble_v2(self, experts, batch_size=2048, vulnerable_threshold=0.5):
        print("\n" + "="*60)
        print(f"TEST EVALUATION (Router Target: {self.router_target.upper()})")
        print("="*60)

        expert_probs = self.compute_expert_outputs(experts, self.X_test, batch_size)
        unc_test = self.calculate_expert_uncertainties(experts, self.X_test, batch_size)
        router_probs = self.meta_router.predict_proba(self.X_test, unc_test)

        c2e = self.class_to_expert
        benign_set = set(self.benign_class_indices or [])

        # expert 모드에서는 추가 클러스터 로직 없이 결합만 수행
        if self.router_target != 'class':
            final_probs, used_experts = self._combine_router_outputs(
                expert_probs, router_probs, c2e, benign_set=benign_set
            )
            self._log_expert_usage(used_experts, final_probs, c2e, extra_used=False)
            final_preds = np.argmax(final_probs, axis=1)
            return self._evaluate_predictions(final_preds, y_proba=final_probs)

        # Router per-class ACC 기준으로 취약 클래스 결정
        router_preds = np.argmax(router_probs, axis=1)
        router_metrics = self._evaluate_predictions(router_preds, y_proba=router_probs)
        router_per_class_acc = router_metrics[2]
        
        # Router per-class accuracy만 로깅 (전체 성능은 _evaluate_predictions에서 이미 출력됨)
        logger = logging.getLogger("experiment")
        cnames = self.label_encoder.classes_ if hasattr(self, "label_encoder") else [f"C{i}" for i in range(self.num_classes)]
        logger.info("\nRouter Per-Class Accuracy (for vulnerable class identification):")
        for cid, acc in enumerate(router_per_class_acc):
            logger.info(f"  {cnames[cid]}: {acc:.4f}")

        vulnerable = self._log_vulnerable_class_predictions(
            router_probs,
            router_preds,
            router_per_class_acc,
            threshold=vulnerable_threshold,
            source_name="ROUTER",
        )
        # 취약 클래스들을 직접 사용 (Router의 per-class accuracy가 낮은 클래스들)
        cluster_class_ids = vulnerable
        extra_expert = self._build_cluster_expert(cluster_class_ids)
        # Extra expert를 인스턴스 변수로 저장 (나중에 저장할 때 사용)
        self.extra_expert = extra_expert
        extra_probs = None
        extra_used_flag = False
        if extra_expert is not None:
            extra_probs = extra_expert.predict_proba(self.X_test, self.num_classes)
            extra_used_flag = True
            
            # Extra expert의 개별 성능 평가 (취약 클래스에 대해서만)
            logger = logging.getLogger("experiment")
            logger.info("\n" + "="*60)
            logger.info("Extra Expert (Cluster Expert) Performance Analysis")
            logger.info("="*60)
            
            # Extra Expert 성능이 낮을 수 있는 이유 설명
            cnames = self.label_encoder.classes_ if hasattr(self, "label_encoder") else [f"C{i}" for i in range(self.num_classes)]
            logger.info(f"Extra Expert trained on {len(cluster_class_ids)} classes: {[cnames[c] for c in cluster_class_ids]}")
            logger.info(f"Note: Extra Expert learns multiple vulnerable classes together, which is more challenging than individual experts.")
            logger.info(f"      For example, Expert1 learned only 2 classes (Analysis, Reconnaissance) and achieved 0.5925 accuracy on Analysis.")
            logger.info(f"      Extra Expert learns {len(cluster_class_ids)} classes together, which may explain lower per-class accuracy.")
            logger.info(f"      However, Extra Expert's role is to rescue samples when Router makes incorrect predictions.")
            
            # 취약 클래스에 대해서만 평가
            vulnerable_mask = np.isin(self.y_test, cluster_class_ids)
            if vulnerable_mask.sum() > 0:
                extra_preds = np.argmax(extra_probs, axis=1)
                extra_preds_vuln = extra_preds[vulnerable_mask]
                extra_y_true_vuln = self.y_test[vulnerable_mask]
                extra_probs_vuln = extra_probs[vulnerable_mask]
                
                # 취약 클래스들에 대한 상세 성능 출력 (기존 expert처럼)
                logger.info(f"\nEvaluating Extra Expert on vulnerable classes only: {[cnames[c] for c in cluster_class_ids]}")
                logger.info(f"Test samples for vulnerable classes: {vulnerable_mask.sum():,}")
                self._print_detailed_metrics(
                    extra_preds_vuln,
                    extra_y_true_vuln,
                    name="Extra Expert (Vulnerable Classes Only)",
                    num_classes=self.num_classes,
                    class_names=cnames,
                    is_router=False,
                    y_proba=extra_probs_vuln,
                )

        # 최종 결합
        final_probs, used_experts = self._combine_router_outputs(
            expert_probs,
            router_probs,
            c2e,
            benign_set=benign_set,
            extra_expert_probs=extra_probs,
            cluster_class_ids=cluster_class_ids,
        )
        
        # Bot 클래스 디버깅 로그 추가
        cnames = self.label_encoder.classes_ if hasattr(self, "label_encoder") else [f"C{i}" for i in range(self.num_classes)]
        bot_class_id = None
        for cid, cname in enumerate(cnames):
            if cname == "Bot":
                bot_class_id = cid
                break
        
        if bot_class_id is not None:
            logger = logging.getLogger("experiment")
            logger.info("\n" + "="*60)
            logger.info("Bot Class Debugging Analysis")
            logger.info("="*60)
            
            bot_mask = (self.y_test == bot_class_id)
            bot_indices = np.where(bot_mask)[0]
            
            if len(bot_indices) > 0:
                # Router가 Bot에 부여한 확률 분포
                router_bot_probs = router_probs[bot_mask, bot_class_id]
                router_preds_bot = np.argmax(router_probs[bot_mask], axis=1)
                router_correct = (router_preds_bot == bot_class_id).sum()
                
                logger.info(f"Bot class ID: {bot_class_id}")
                logger.info(f"Total Bot samples: {len(bot_indices)}")
                logger.info(f"Router predicted Bot correctly: {router_correct} / {len(bot_indices)} ({router_correct/len(bot_indices)*100:.2f}%)")
                logger.info(f"Router Bot probability - Mean: {router_bot_probs.mean():.4f}, Min: {router_bot_probs.min():.4f}, Max: {router_bot_probs.max():.4f}, Median: {np.median(router_bot_probs):.4f}")
                
                # Router가 예측한 클래스 분포
                router_pred_classes, router_pred_counts = np.unique(router_preds_bot, return_counts=True)
                logger.info(f"\nRouter predictions for Bot samples:")
                for pred_cid, count in zip(router_pred_classes, router_pred_counts):
                    pred_cname = cnames[pred_cid] if pred_cid < len(cnames) else f"Class_{pred_cid}"
                    logger.info(f"  {pred_cname}: {count} samples ({count/len(bot_indices)*100:.2f}%)")
                
                # Expert 3가 Bot에 부여한 확률 분포
                expert3_bot_probs = expert_probs[bot_mask, 3, bot_class_id]
                expert3_preds_bot = np.argmax(expert_probs[bot_mask, 3], axis=1)
                expert3_correct = (expert3_preds_bot == bot_class_id).sum()
                
                logger.info(f"\nExpert 3 Bot probability - Mean: {expert3_bot_probs.mean():.4f}, Min: {expert3_bot_probs.min():.4f}, Max: {expert3_bot_probs.max():.4f}, Median: {np.median(expert3_bot_probs):.4f}")
                logger.info(f"Expert 3 predicted Bot correctly: {expert3_correct} / {len(bot_indices)} ({expert3_correct/len(bot_indices)*100:.2f}%)")
                
                # 최종 예측 분포
                final_preds_bot = np.argmax(final_probs[bot_mask], axis=1)
                final_correct = (final_preds_bot == bot_class_id).sum()
                
                logger.info(f"\nFinal predictions for Bot samples:")
                logger.info(f"Final predicted Bot correctly: {final_correct} / {len(bot_indices)} ({final_correct/len(bot_indices)*100:.2f}%)")
                
                final_pred_classes, final_pred_counts = np.unique(final_preds_bot, return_counts=True)
                for pred_cid, count in zip(final_pred_classes, final_pred_counts):
                    pred_cname = cnames[pred_cid] if pred_cid < len(cnames) else f"Class_{pred_cid}"
                    logger.info(f"  {pred_cname}: {count} samples ({count/len(bot_indices)*100:.2f}%)")
                
                # 최종 확률 분포
                final_bot_probs = final_probs[bot_mask, bot_class_id]
                logger.info(f"\nFinal Bot probability - Mean: {final_bot_probs.mean():.4f}, Min: {final_bot_probs.min():.4f}, Max: {final_bot_probs.max():.4f}, Median: {np.median(final_bot_probs):.4f}")
                
                # 샘플별 상세 분석 (처음 10개)
                logger.info(f"\nSample-by-sample analysis (first 10 Bot samples):")
                for idx in range(min(10, len(bot_indices))):
                    sample_idx = bot_indices[idx]
                    router_pred = int(np.argmax(router_probs[sample_idx]))
                    router_bot_prob = router_probs[sample_idx, bot_class_id]
                    expert3_bot_prob = expert_probs[sample_idx, 3, bot_class_id]
                    final_pred = int(np.argmax(final_probs[sample_idx]))
                    final_bot_prob = final_probs[sample_idx, bot_class_id]
                    used_expert = used_experts[sample_idx]
                    
                    router_pred_name = cnames[router_pred] if router_pred < len(cnames) else f"Class_{router_pred}"
                    final_pred_name = cnames[final_pred] if final_pred < len(cnames) else f"Class_{final_pred}"
                    
                    logger.info(f"  Sample {idx}: Router={router_pred_name}({router_bot_prob:.4f}), Expert3={expert3_bot_prob:.4f}, Final={final_pred_name}({final_bot_prob:.4f}), UsedExpert={used_expert}")
        
        self._log_expert_usage(used_experts, final_probs, c2e, extra_used=extra_used_flag)
        
        # Extra expert 사용 효과 분석
        if extra_used_flag and extra_probs is not None:
            self._analyze_extra_expert_effectiveness(
                used_experts, final_probs, extra_probs, expert_probs, router_probs, 
                cluster_class_ids, c2e, benign_set
            )
        
        final_preds = np.argmax(final_probs, axis=1)
        return self._evaluate_predictions(final_preds, y_proba=final_probs)

    def _print_detailed_metrics(self, predictions: np.ndarray, y_true: np.ndarray, name: str = "", num_classes: int = None, class_names: list = None, is_router: bool = False, y_proba: np.ndarray = None):
        
        if num_classes is None:
            num_classes = self.num_classes
        
        accuracy = accuracy_score(y_true, predictions)
        
        unique_labels = np.unique(np.concatenate([y_true, predictions]))
        unique_labels = np.sort(unique_labels)
        
        if class_names is None:
            if is_router:
                class_names = [f"Expert_{i}" for i in range(num_classes)]
            else:
                class_names = self.label_encoder.classes_ if hasattr(self, 'label_encoder') else [f"Class_{i}" for i in range(num_classes)]
        
        actual_class_names = [class_names[int(label)] if int(label) < len(class_names) else f"Class_{int(label)}" for label in unique_labels]
        
        from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
        
        report = classification_report(
            y_true,
            predictions,
            labels=unique_labels,
            target_names=actual_class_names,
            output_dict=True,
            zero_division=0,
        )
        
        per_class_acc = []
        per_class_f1 = []
        per_class_precision = []
        per_class_recall = []
        per_class_support = []
        per_class_pr_auc = []
        per_class_tp = []  # TP 저장
        per_class_tn = []  # TN 저장
        per_class_fp = []  # FP 저장
        per_class_fn = []  # FN 저장
        
        for idx, label in enumerate(unique_labels):
            cid = int(label)
            mask = (y_true == cid)
            if mask.sum() > 0:
                acc_c = (predictions[mask] == cid).mean()
            else:
                acc_c = 0.0
            per_class_acc.append(acc_c)
            
            # TP, TN, FP, FN 계산
            y_true_binary = (y_true == cid).astype(int)
            tp = int(np.sum((y_true_binary == 1) & (predictions == cid)))
            fp = int(np.sum((y_true_binary == 0) & (predictions == cid)))
            fn = int(np.sum((y_true_binary == 1) & (predictions != cid)))
            tn = int(np.sum((y_true_binary == 0) & (predictions != cid)))
            
            per_class_tp.append(tp)
            per_class_tn.append(tn)
            per_class_fp.append(fp)
            per_class_fn.append(fn)
            
            cname = actual_class_names[idx]
            if cname in report:
                precision = report[cname]["precision"]
                recall = report[cname]["recall"]
                f1 = report[cname]["f1-score"]
                support = int(report[cname]["support"])
            else:
                precision = precision_score(y_true == cid, predictions == cid, zero_division=0)
                recall = recall_score(y_true == cid, predictions == cid, zero_division=0)
                f1 = f1_score(y_true == cid, predictions == cid, zero_division=0)
                support = int(mask.sum())
            
            per_class_f1.append(f1)
            per_class_precision.append(precision)
            per_class_recall.append(recall)
            per_class_support.append(support)
            
            # PR-AUC 계산 (y_proba가 제공된 경우)
            if y_proba is not None and y_proba.shape[1] > cid:
                try:
                    y_proba_binary = y_proba[:, cid]
                    
                    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true_binary, y_proba_binary)
                    
                    # PR-AUC 계산 전 검증
                    # 만약 실제 양성 샘플이 없으면 (y_true_binary가 모두 0) PR-AUC는 정의되지 않음
                    if np.sum(y_true_binary) == 0:
                        pr_auc = 0.0
                        logger = logging.getLogger("experiment")
                        logger.warning(f"Class {cname}: No positive samples (TP={tp}, FP={fp}, FN={fn}, TN={tn}), PR-AUC set to 0.0")
                    # precision_curve나 recall_curve가 비정상적인 경우 체크
                    elif len(precision_curve) == 0 or len(recall_curve) == 0:
                        pr_auc = 0.0
                        logger = logging.getLogger("experiment")
                        logger.warning(f"Class {cname}: Empty precision/recall curves (TP={tp}, FP={fp}, FN={fn}, TN={tn}), PR-AUC set to 0.0")
                    else:
                        pr_auc = auc(recall_curve, precision_curve)
                        
                        # PR-AUC가 0인 경우 디버깅 정보 출력
                        if pr_auc == 0.0 and precision > 0.0 and recall > 0.0:
                            logger = logging.getLogger("experiment")
                            logger.warning(f"Class {cname}: PR-AUC is 0.0 despite good P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
                            logger.warning(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
                            logger.warning(f"  Precision curve length: {len(precision_curve)}, Recall curve length: {len(recall_curve)}")
                            logger.warning(f"  Precision curve: {precision_curve[:5]}...{precision_curve[-5:]}")
                            logger.warning(f"  Recall curve: {recall_curve[:5]}...{recall_curve[-5:]}")
                            logger.warning(f"  Precision curve range: [{np.min(precision_curve):.4f}, {np.max(precision_curve):.4f}]")
                            logger.warning(f"  Recall curve range: [{np.min(recall_curve):.4f}, {np.max(recall_curve):.4f}]")
                            logger.warning(f"  y_proba_binary range: [{np.min(y_proba_binary):.6f}, {np.max(y_proba_binary):.6f}]")
                            logger.warning(f"  y_proba_binary mean: {np.mean(y_proba_binary):.6f}")
                        
                        # P, R이 0이면 PR-AUC도 0으로 설정
                        if precision == 0.0 and recall == 0.0:
                            pr_auc = 0.0
                    
                    per_class_pr_auc.append(float(pr_auc))
                except Exception as e:
                    logger = logging.getLogger("experiment")
                    logger.error(f"Class {cname}: PR-AUC calculation failed: {str(e)}")
                    per_class_pr_auc.append(0.0)
            else:
                per_class_pr_auc.append(0.0)
        
        macro_accuracy = float(np.mean(per_class_acc)) if per_class_acc else 0.0
        macro_precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
        macro_recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
        macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
        macro_pr_auc = float(np.mean(per_class_pr_auc)) if per_class_pr_auc and len(per_class_pr_auc) > 0 else 0.0
        
        logger = logging.getLogger("experiment")
        logger.info(f"{name}: Accuracy: {accuracy:.4f}")
        logger.info(f"{name}: Macro Accuracy: {macro_accuracy:.4f}")
        if y_proba is not None:
            logger.info(f"{name}: Macro PR-AUC: {macro_pr_auc:.4f}")
        logger.info(f"{name}: Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")
        
        # TP, TN, FP, FN 디버깅 출력 (단일 표만 유지)
        header_name = "Expert" if is_router else "Class"
        logger.info("=" * 100)
        logger.info(f"{name} - Class-wise TP, TN, FP, FN Debugging Information:")
        logger.info("=" * 100)
        logger.info(f"{header_name:<30} {'TP':<10} {'TN':<10} {'FP':<10} {'FN':<10}"
                    f"{'ACC':<10} {'P':<10} {'R':<10} {'F1':<10}{'#Samples':<10}")
        logger.info("-" * 100)
        for idx, label in enumerate(unique_labels):
            cname = actual_class_names[idx]
            logger.info(
                f"{cname:<30} {per_class_tp[idx]:<10} {per_class_tn[idx]:<10} {per_class_fp[idx]:<10} {per_class_fn[idx]:<10} "
                f"{per_class_acc[idx]:<10.4f} {per_class_precision[idx]:<10.4f} {per_class_recall[idx]:<10.4f} {per_class_f1[idx]:<10.4f} {per_class_support[idx]:<10}"
            )
        logger.info("=" * 100)

    def _evaluate_predictions(self, predictions, y_proba=None):
        """
        Return exactly the following:
        (
            accuracy,
            cm,
            per_class_acc,
            per_class_precision,
            per_class_recall,
            per_class_f1,
            per_class_pr_auc,
            per_class_support,
            macro_accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            macro_pr_auc
        )
        """

        logger = logging.getLogger("experiment")
        y_true = self.y_test

        # -----------------------------------------
        # Overall accuracy
        # -----------------------------------------
        accuracy = accuracy_score(y_true, predictions)

        # -----------------------------------------
        # Confusion matrix
        # -----------------------------------------
        cm = confusion_matrix(y_true, predictions, labels=np.arange(self.num_classes))

        # -----------------------------------------
        # Per-class metrics using classification_report
        # -----------------------------------------
        report = classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        per_class_support = []
        per_class_acc = []

        for cid in range(self.num_classes):
            key = str(cid)
            if key in report:
                # precision / recall / f1 / support
                precision_c = report[key]["precision"]
                recall_c = report[key]["recall"]
                f1_c = report[key]["f1-score"]
                support_c = int(report[key]["support"])
            else:
                precision_c = 0.0
                recall_c = 0.0
                f1_c = 0.0
                support_c = 0

            # per-class accuracy: 해당 클래스의 샘플 중 올바르게 예측된 비율
            # 즉, 해당 클래스의 recall과 동일 (TP / (TP + FN))
            tp = cm[cid, cid]
            fn = cm[cid, :].sum() - tp
            # 해당 클래스의 샘플 수
            class_samples = tp + fn
            acc_c = tp / class_samples if class_samples > 0 else 0.0

            per_class_acc.append(acc_c)
            per_class_precision.append(precision_c)
            per_class_recall.append(recall_c)
            per_class_f1.append(f1_c)
            per_class_support.append(support_c)

        per_class_precision = np.array(per_class_precision, dtype=float)
        per_class_recall = np.array(per_class_recall, dtype=float)
        per_class_f1 = np.array(per_class_f1, dtype=float)
        per_class_support = np.array(per_class_support, dtype=int)
        per_class_acc = np.array(per_class_acc, dtype=float)


        # -----------------------------------------
        # Macro metrics
        # -----------------------------------------
        macro_accuracy = float(per_class_acc.mean())
        macro_precision = float(per_class_precision.mean())
        macro_recall = float(per_class_recall.mean())
        macro_f1 = float(per_class_f1.mean())


        # -----------------------------------------
        # Per-class PR-AUC
        # -----------------------------------------
        per_class_pr_auc = np.zeros(self.num_classes, dtype=float)

        if y_proba is not None:
            for cid in range(self.num_classes):
                y_true_bin = (y_true == cid).astype(int)
                y_score = y_proba[:, cid]

                if y_true_bin.sum() > 0 and (1 - y_true_bin).sum() > 0:
                    precision_c, recall_c, _ = precision_recall_curve(y_true_bin, y_score)
                    per_class_pr_auc[cid] = auc(recall_c, precision_c)
                else:
                    per_class_pr_auc[cid] = 0.0

        macro_pr_auc = float(per_class_pr_auc.mean())

        # -----------------------------------------
        # Log
        # -----------------------------------------
        logger.info(f"Accuracy         : {accuracy:.4f}")
        logger.info(f"Macro Accuracy   : {macro_accuracy:.4f}")
        logger.info(f"Macro Precision  : {macro_precision:.4f}")
        logger.info(f"Macro Recall     : {macro_recall:.4f}")
        logger.info(f"Macro F1         : {macro_f1:.4f}")
        logger.info(f"Macro PR-AUC     : {macro_pr_auc:.4f}")

        return (
            float(accuracy),

            cm,

            per_class_acc,
            per_class_precision,
            per_class_recall,
            per_class_f1,
            per_class_pr_auc,
            per_class_support,

            float(macro_accuracy),
            float(macro_precision),
            float(macro_recall),
            float(macro_f1),
            float(macro_pr_auc),
        )

    def plot_confusion_matrix(self, cm, class_names, title, save_path, cmap="Blues"):
        plt.figure(figsize=(20, 20))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_per_class_metrics(
        self,
        baseline_acc,
        baseline_f1,
        ensemble_acc,
        ensemble_f1,
        class_names,
        save_path,
    ):
        plt.figure(figsize=(20, 8))

        x = np.arange(len(class_names))
        width = 0.35

        plt.subplot(1, 2, 1)
        labels = []
        all_acc_values = []

        if baseline_acc is not None and len(baseline_acc) > 0:
            plt.bar(x - width / 2, baseline_acc, width, label="Baseline XGBoost", alpha=0.8)
            labels.append("Baseline XGBoost")
            all_acc_values.extend(baseline_acc)

        if ensemble_acc is not None and len(ensemble_acc) > 0:
            plt.bar(x + width / 2, ensemble_acc, width, label="Meta-Ensemble", alpha=0.8)
            labels.append("Meta-Ensemble")
            all_acc_values.extend(ensemble_acc)

        plt.title("Per-Class Accuracy")
        plt.xticks(x, class_names, rotation=45, ha="right")
        if all_acc_values:
            y_min = max(0.0, min(all_acc_values) - 0.05)
            y_max = min(1.0, max(all_acc_values) + 0.05)
            plt.ylim(y_min, y_max)
        if labels:
            plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        all_f1_values = []

        if baseline_f1 is not None and len(baseline_f1) > 0:
            plt.bar(x - width / 2, baseline_f1, width, label="Baseline XGBoost", alpha=0.8)
            all_f1_values.extend(baseline_f1)

        if ensemble_f1 is not None and len(ensemble_f1) > 0:
            plt.bar(x + width / 2, ensemble_f1, width, label="Meta-Ensemble", alpha=0.8)
            all_f1_values.extend(ensemble_f1)

        plt.title("Per-Class F1 Score")
        plt.xticks(x, class_names, rotation=45, ha="right")
        if all_f1_values:
            y_min = max(0.0, min(all_f1_values) - 0.05)
            y_max = min(1.0, max(all_f1_values) + 0.05)
            plt.ylim(y_min, y_max)
        if labels:
            plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def save_detailed_performance_metrics(
        self,
        baseline_acc,
        baseline_f1,
        baseline_precision,
        baseline_recall,
        baseline_support,
        baseline_pr_auc,
        ensemble_acc,
        ensemble_f1,
        ensemble_precision,
        ensemble_recall,
        ensemble_support,
        ensemble_pr_auc,
        class_names,
        save_path,
    ):
        baseline_acc = baseline_acc if baseline_acc is not None and len(baseline_acc) > 0 else None
        baseline_f1 = baseline_f1 if baseline_f1 is not None and len(baseline_f1) > 0 else None
        baseline_precision = baseline_precision if baseline_precision is not None and len(baseline_precision) > 0 else None
        baseline_recall = baseline_recall if baseline_recall is not None and len(baseline_recall) > 0 else None
        baseline_support = baseline_support if baseline_support is not None and len(baseline_support) > 0 else None
        baseline_pr_auc = baseline_pr_auc if baseline_pr_auc is not None and len(baseline_pr_auc) > 0 else None

        ensemble_acc = ensemble_acc if ensemble_acc is not None and len(ensemble_acc) > 0 else None
        ensemble_f1 = ensemble_f1 if ensemble_f1 is not None and len(ensemble_f1) > 0 else None
        ensemble_precision = ensemble_precision if ensemble_precision is not None and len(ensemble_precision) > 0 else None
        ensemble_recall = ensemble_recall if ensemble_recall is not None and len(ensemble_recall) > 0 else None
        ensemble_support = ensemble_support if ensemble_support is not None and len(ensemble_support) > 0 else None
        ensemble_pr_auc = ensemble_pr_auc if ensemble_pr_auc is not None and len(ensemble_pr_auc) > 0 else None

        data_dict = {"Class": class_names}

        if baseline_acc is not None:
            data_dict["Baseline_Accuracy"] = baseline_acc
            data_dict["Baseline_F1"] = baseline_f1
            data_dict["Baseline_Precision"] = baseline_precision
            data_dict["Baseline_Recall"] = baseline_recall
            data_dict["Baseline_Support"] = baseline_support
            if baseline_pr_auc is not None:
                data_dict["Baseline_PR_AUC"] = baseline_pr_auc

        if ensemble_acc is not None:
            data_dict["Ensemble_Accuracy"] = ensemble_acc
            data_dict["Ensemble_F1"] = ensemble_f1
            data_dict["Ensemble_Precision"] = ensemble_precision
            data_dict["Ensemble_Recall"] = ensemble_recall
            data_dict["Ensemble_Support"] = ensemble_support
            if ensemble_pr_auc is not None:
                data_dict["Ensemble_PR_AUC"] = ensemble_pr_auc

        df = pd.DataFrame(data_dict)
        df.to_csv(save_path, index=False)

        logger = logging.getLogger("experiment")
        logger.info(f"Detailed performance metrics saved to: {save_path}")
        logger.info(f"Saved detailed metrics for {len(class_names)} classes")

    def save_detailed_results_log(self, results: dict, exp_dir: str):
        logger = logging.getLogger("experiment")

        logger.info("=" * 80)
        logger.info("EXPERIMENT RESULTS SUMMARY (Meta-Router XGBoost Experts)")
        logger.info("=" * 80)

        logger.info(f"Seed: {results['seed']}")
        logger.info(f"Runtime: {results['runtime_minutes']:.1f} minutes")
        logger.info(f"Timestamp: {results['timestamp']}")
        logger.info(f"Results saved to: {exp_dir}")

        # Expert 클래스 배정 정보
        if self.expert_classes is not None:
            logger.info("\nExpert Class Assignment:")
            for e in range(self.num_experts):
                classes_e = self.expert_classes[e]
                cls_str = ", ".join([f"{c}" for c in classes_e])
                logger.info(f"  Expert {e}: classes [{cls_str}]")

        # Train/Test 분포
        logger.info("\nTrain/Test Data Distribution:")
        logger.info("Train Set:")
        unique_train, counts_train = np.unique(self.y_train, return_counts=True)
        total_train = len(self.y_train)
        for class_id, count in zip(unique_train, counts_train):
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            percentage = (count / total_train) * 100.0
            logger.info(f"  {class_name:25}: {count:7d} ({percentage:.2f}%)")
        logger.info(f"  Total Train Samples: {total_train}")

        logger.info("Test Set:")
        unique_test, counts_test = np.unique(self.y_test, return_counts=True)
        total_test = len(self.y_test)
        for class_id, count in zip(unique_test, counts_test):
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            percentage = (count / total_test) * 100.0
            logger.info(f"  {class_name:25}: {count:7d} ({percentage:.2f}%)")
        logger.info(f"  Total Test Samples: {total_test}")

        class_names = self.label_encoder.classes_

        # Baseline
        if results.get("baseline_accuracy") is not None:
            logger.info("\nBaseline XGBoost Performance:")
            logger.info(f"  Accuracy: {results['baseline_accuracy']:.4f}")
            logger.info(f"  Macro Accuracy: {results['baseline_macro_accuracy']:.4f}")
            if results.get("baseline_macro_pr_auc") is not None:
                logger.info(f"  Macro PR-AUC: {results['baseline_macro_pr_auc']:.4f}")

            logger.info("\nBaseline Per-Class Performance:")
            if results.get("baseline_per_class_pr_auc") is not None:
                logger.info(f"{'Class':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'PR-AUC':<10} {'Support':<8}")
                logger.info("-" * 80)
                for i, class_name in enumerate(class_names):
                    accuracy = results["baseline_per_class_acc"][i]
                    precision = results["baseline_per_class_precision"][i]
                    recall = results["baseline_per_class_recall"][i]
                    f1 = results["baseline_per_class_f1"][i]
                    pr_auc = results["baseline_per_class_pr_auc"][i]
                    support = results["baseline_per_class_support"][i]
                    logger.info(
                        f"  {class_name:25} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {pr_auc:<10.4f} {support:<8d}"
                    )
            else:
                logger.info(f"{'Class':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
                logger.info("-" * 70)
                for i, class_name in enumerate(class_names):
                    accuracy = results["baseline_per_class_acc"][i]
                    precision = results["baseline_per_class_precision"][i]
                    recall = results["baseline_per_class_recall"][i]
                    f1 = results["baseline_per_class_f1"][i]
                    support = results["baseline_per_class_support"][i]
                    logger.info(
                        f"  {class_name:25} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<8d}"
                    )

        # Ensemble
        if results.get("ensemble_accuracy") is not None:
            logger.info("\nMeta-Ensemble Performance:")
            logger.info(f"  Accuracy: {results['ensemble_accuracy']:.4f}")
            logger.info(f"  Macro Accuracy: {results['ensemble_macro_accuracy']:.4f}")
            if results.get("ensemble_macro_pr_auc") is not None:
                logger.info(f"  Macro PR-AUC: {results['ensemble_macro_pr_auc']:.4f}")

            logger.info("\nEnsemble Per-Class Performance:")
            if results.get("ensemble_per_class_pr_auc") is not None:
                logger.info(f"{'Class':<25} {'Accuracy':<10}{'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'PR-AUC':<10} {'Support':<8}")
                logger.info("-" * 80)
                for i, class_name in enumerate(class_names):
                    accuracy = results["ensemble_per_class_acc"][i]
                    precision = results["ensemble_per_class_precision"][i]
                    recall = results["ensemble_per_class_recall"][i]
                    f1 = results["ensemble_per_class_f1"][i]
                    pr_auc = results["ensemble_per_class_pr_auc"][i]
                    support = results["ensemble_per_class_support"][i]
                    logger.info(
                        f"  {class_name:25} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {pr_auc:<10.4f} {support:<8d}"
                    )
            else:
                logger.info(f"{'Class':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
                logger.info("-" * 70)
                for i, class_name in enumerate(class_names):
                    accuracy = results["ensemble_per_class_acc"][i]
                    precision = results["ensemble_per_class_precision"][i]
                    recall = results["ensemble_per_class_recall"][i]
                    f1 = results["ensemble_per_class_f1"][i]
                    support = results["ensemble_per_class_support"][i]
                    logger.info(
                        f"  {class_name:25} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<8d}"
                    )

            if results.get("improvement") is not None:
                logger.info(f"\nEnsemble Improvement: {results['improvement']:+.4f}")

        logger.info("=" * 80)

    def run_experiment(
        self,
        seed: int,
        data_path: str,
        epochs: int = None,
        models=None,
        save_models: bool = False,
        load_models_dir: str = None,
        assignment_mode: str = "random",
        use_router: bool = True,
        router_target: str = None,
    ):
        """
        전체 파이프라인 실행:
        1) 데이터 로드 및 전처리
        2) Expert 할당 및 학습 (원본 trainset 사용)
        3) Router 학습 (재가공된 label 사용)
        4) 평가
        """
        if router_target is not None:
            self.router_target = router_target
        if models is None:
            models = ["baseline", "ensemble"]

        start_time = time.time()
        # seed를 인스턴스에 저장해 추가 expert 등에서 재사용
        self.seed = seed

        exp_dir = f"meta_router_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)
        logger = setup_logging(exp_dir)

        logger.info(f"Starting experiment with seed {seed}")
        cuda_available = torch.cuda.is_available() if torch.cuda.is_available() else False
        logger.info(f"Compute device: {self.device} (CUDA available: {cuda_available})")
        logger.info(f"Models to train/evaluate: {models}")
        logger.info(f"Data path: {data_path}")
        if load_models_dir is not None:
            logger.info(f"Loading models from directory: {load_models_dir}")
        logger.info(f"Will save models: {save_models}")

        self.setup_dataset(seed, data_path)

        baseline_results = None
        ensemble_results = None

        baseline_acc = baseline_macro_acc = None
        baseline_macro_precision = baseline_macro_recall = baseline_macro_f1 = None
        baseline_per_class_acc = []
        baseline_per_class_f1 = []
        baseline_cm = None
        baseline_per_class_precision = []
        baseline_per_class_recall = []
        baseline_per_class_support = []
        baseline_per_class_pr_auc = None
        baseline_macro_pr_auc = None

        ensemble_acc = ensemble_macro_acc = None
        ensemble_macro_precision = ensemble_macro_recall = ensemble_macro_f1 = None
        ensemble_per_class_acc = []
        ensemble_per_class_f1 = []
        ensemble_cm = None
        ensemble_per_class_precision = []
        ensemble_per_class_recall = []
        ensemble_per_class_support = []
        ensemble_per_class_pr_auc = None
        ensemble_macro_pr_auc = None

        baseline_model = None
        expert_classifiers = None

        # 1) Baseline
        if "baseline" in models:
            if load_models_dir is not None:
                baseline_model = self._load_baseline_model(load_models_dir)

            if baseline_model is None:
                logger.info("Training baseline model...")
                baseline_model = self.train_baseline(seed)

            baseline_results = self.evaluate_baseline(baseline_model)
            (
                baseline_acc,

                baseline_cm,

                baseline_per_class_acc,
                baseline_per_class_precision,
                baseline_per_class_recall,
                baseline_per_class_f1,
                baseline_per_class_pr_auc,
                baseline_per_class_support,

                baseline_macro_acc,
                baseline_macro_precision,
                baseline_macro_recall,
                baseline_macro_f1,
                baseline_macro_pr_auc,
            ) = baseline_results
            logger.info(f"Baseline accuracy: {baseline_acc:.4f}")

        # 2) Expert + Meta-router
        if "ensemble" in models:
            loaded_router = None
            if load_models_dir is not None:
                if use_router:
                    expert_classifiers, loaded_router = self._load_ensemble_models(load_models_dir)
                    if loaded_router is not None:
                        self.meta_router = loaded_router
                else:
                    
                    experts_path = os.path.join(load_models_dir, "experts.pkl")
                    if os.path.exists(experts_path):
                        with open(experts_path, "rb") as f:
                            expert_classifiers = pickle.load(f)
                        logger.info(f"Loaded experts (no router) from: {experts_path}")
                    else:
                        expert_classifiers = None

            if expert_classifiers is None or (use_router and not hasattr(self, "meta_router")):
                if use_router:
                    logger.info("Running expert + meta-router training pipeline...")
                else:
                    logger.info("Running expert training pipeline (no router, simple average)...")
                    
                # assignment_mode는 이제 type만 지원
                if assignment_mode != "type":
                    raise ValueError("Only 'type' assignment_mode is supported in this version.")
                
                logger.info("Using type assignment (attack-type based, BENIGN/NORMAL handled only by router)")
                expert_assignments = self.assign_samples_to_experts_type()

                logger.info("Step 1) Training experts...")
                # Expert 학습 (원본 trainset 사용)
                expert_classifiers = self.train_experts(expert_assignments, seed)
                logger.info("Step 1 완료: experts 학습 완료")
                
                if use_router:
                    # (N, E, C): train set 에 대해 모든 expert의 예측 확률
                    probs_train = self.compute_expert_outputs(
                        expert_classifiers,
                        self.X_train,
                        description="train (for teacher router)",
                        batch_size=10000,
                    )
                    y_train = self.y_train
                    N, E, C = probs_train.shape

                    if self.router_target == 'expert':
                        # Expert ID를 라벨로 사용 (Type-based assignment 그대로)
                        router_labels = self.prepare_router_labels(self.y_train, expert_assignments)
                    else: # 'class'
                        # Class ID를 라벨로 사용 (Original Labels)
                        router_labels = self.y_train.copy()
                  
                    # 선택 사항: SMOTE 이전 샘플 수 기반으로 sample weight를 줄 수도 있음
                    # 여기서는 단순히 weight 없이 사용
                    sample_weights = None

                    # Router 입력용 uncertainty (train set 기준)
                    # 이미 있는 함수/로직을 재사용
                    expert_uncertainties_train = self.calculate_expert_uncertainties(
                        expert_classifiers,
                        self.X_train,
                        batch_size=10000,
                    )
                    self.expert_uncertainties_cache = expert_uncertainties_train

                    

                    # 실제 Router 학습
                    self.meta_router = self.train_meta_router(
                        router_labels=router_labels,
                        expert_uncertainties=expert_uncertainties_train,
                        router_target=self.router_target,
                        # sample_weights=sample_weights,
                    )


            else:
                if use_router:
                    logger.info("Using loaded experts and meta-router without retraining.")
                else:
                    logger.info("Using loaded experts without retraining (no router).")

            if use_router:
                ensemble_results = self.evaluate_meta_ensemble_v2(
                    expert_classifiers,
                    batch_size=2048,
                )
            else:
                ensemble_results = self.evaluate_simple_ensemble(
                    expert_classifiers,
                    batch_size=2048,
                )
            (
                ensemble_acc,
 

                ensemble_cm,

                ensemble_per_class_acc,
                ensemble_per_class_precision,
                ensemble_per_class_recall,
                ensemble_per_class_f1,
                ensemble_per_class_pr_auc,
                ensemble_per_class_support,

                ensemble_macro_acc,
                ensemble_macro_precision,
                ensemble_macro_recall,
                ensemble_macro_f1,
                ensemble_macro_pr_auc,
            ) = ensemble_results

        # 모델 저장
        if save_models:
            if "baseline" in models and baseline_model is not None:
                self._save_baseline_model(baseline_model, exp_dir)
            if "ensemble" in models and expert_classifiers is not None:
                if use_router and hasattr(self, "meta_router"):
                    extra_expert = getattr(self, "extra_expert", None)
                    self._save_ensemble_models(expert_classifiers, exp_dir, extra_expert=extra_expert)
                elif not use_router:
                    
                    os.makedirs(exp_dir, exist_ok=True)
                    experts_path = os.path.join(exp_dir, "experts.pkl")
                    with open(experts_path, "wb") as f:
                        pickle.dump(expert_classifiers, f)
                    logger.info(f"Saved experts (no router) to: {experts_path}")

        elapsed_minutes = (time.time() - start_time) / 60.0
        improvement = (
            float(ensemble_acc - baseline_acc)
            if (baseline_acc is not None and ensemble_acc is not None)
            else None
        )

        class_names = self.label_encoder.classes_

        # 혼동 행렬 그림
        if "baseline" in models and baseline_cm is not None:
            self.plot_confusion_matrix(
                baseline_cm,
                class_names,
                "Baseline XGBoost Confusion Matrix",
                os.path.join(exp_dir, "baseline_confusion_matrix.png"),
            )
        if "ensemble" in models and ensemble_cm is not None:
            self.plot_confusion_matrix(
                ensemble_cm,
                class_names,
                "Meta-Ensemble Confusion Matrix",
                os.path.join(exp_dir, "ensemble_confusion_matrix.png"),
            )

        # per-class metrics bar plot
        self.plot_per_class_metrics(
            baseline_per_class_acc if "baseline" in models and baseline_results is not None else None,
            baseline_per_class_f1 if "baseline" in models and baseline_results is not None else None,
            ensemble_per_class_acc if "ensemble" in models and ensemble_results is not None else None,
            ensemble_per_class_f1 if "ensemble" in models and ensemble_results is not None else None,
            class_names,
            os.path.join(exp_dir, "per_class_metrics.png"),
        )

        # detailed_performance_metrics.csv
        self.save_detailed_performance_metrics(
            baseline_per_class_acc if "baseline" in models and baseline_results is not None else None,
            baseline_per_class_f1 if "baseline" in models and baseline_results is not None else None,
            baseline_per_class_precision if "baseline" in models and baseline_results is not None else None,
            baseline_per_class_recall if "baseline" in models and baseline_results is not None else None,
            baseline_per_class_support if "baseline" in models and baseline_results is not None else None,
            baseline_per_class_pr_auc if "baseline" in models and baseline_results is not None else None,
            ensemble_per_class_acc if "ensemble" in models and ensemble_results is not None else None,
            ensemble_per_class_f1 if "ensemble" in models and ensemble_results is not None else None,
            ensemble_per_class_precision if "ensemble" in models and ensemble_results is not None else None,
            ensemble_per_class_recall if "ensemble" in models and ensemble_results is not None else None,
            ensemble_per_class_support if "ensemble" in models and ensemble_results is not None else None,
            ensemble_per_class_pr_auc if "ensemble" in models and ensemble_results is not None else None,
            class_names,
            os.path.join(exp_dir, "detailed_performance_metrics.csv"),
        )

        # JSON-friendly 변환
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(o) for o in obj]
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            else:
                return obj

        results = {
            "seed": int(seed),
            "runtime_minutes": float(elapsed_minutes),
            "timestamp": datetime.now().isoformat(),

            # ------------------------------
            # Baseline (None-safe)
            # ------------------------------
            "baseline_accuracy": float(baseline_acc) if baseline_acc is not None else None,

            "baseline_macro_accuracy": float(baseline_macro_acc) if baseline_macro_acc is not None else None,
            "baseline_macro_precision": float(baseline_macro_precision) if baseline_macro_precision is not None else None,
            "baseline_macro_recall": float(baseline_macro_recall) if baseline_macro_recall is not None else None,
            "baseline_macro_f1": float(baseline_macro_f1) if baseline_macro_f1 is not None else None,
            "baseline_macro_pr_auc": float(baseline_macro_pr_auc) if baseline_macro_pr_auc is not None else None,

            "baseline_per_class_acc": convert_numpy_types(baseline_per_class_acc),
            "baseline_per_class_precision": convert_numpy_types(baseline_per_class_precision),
            "baseline_per_class_recall": convert_numpy_types(baseline_per_class_recall),
            "baseline_per_class_f1": convert_numpy_types(baseline_per_class_f1),
            "baseline_per_class_pr_auc": convert_numpy_types(baseline_per_class_pr_auc) if baseline_per_class_pr_auc is not None else None,
            "baseline_per_class_support": convert_numpy_types(baseline_per_class_support),

            # ------------------------------
            # Ensemble (None-safe)
            # ------------------------------
            "ensemble_accuracy": float(ensemble_acc) if ensemble_acc is not None else None,

            "ensemble_macro_accuracy": float(ensemble_macro_acc) if ensemble_macro_acc is not None else None,
            "ensemble_macro_precision": float(ensemble_macro_precision) if ensemble_macro_precision is not None else None,
            "ensemble_macro_recall": float(ensemble_macro_recall) if ensemble_macro_recall is not None else None,
            "ensemble_macro_f1": float(ensemble_macro_f1) if ensemble_macro_f1 is not None else None,
            "ensemble_macro_pr_auc": float(ensemble_macro_pr_auc) if ensemble_macro_pr_auc is not None else None,

            "ensemble_per_class_acc": convert_numpy_types(ensemble_per_class_acc),
            "ensemble_per_class_precision": convert_numpy_types(ensemble_per_class_precision),
            "ensemble_per_class_recall": convert_numpy_types(ensemble_per_class_recall),
            "ensemble_per_class_f1": convert_numpy_types(ensemble_per_class_f1),
            "ensemble_per_class_pr_auc": convert_numpy_types(ensemble_per_class_pr_auc) if ensemble_per_class_pr_auc is not None else None,
            "ensemble_per_class_support": convert_numpy_types(ensemble_per_class_support),

            # ------------------------------
            # Improvement (None-safe)
            # ------------------------------
            "improvement": float(improvement) if improvement is not None else None,
        }       


        # 상세 결과 로그
        self.save_detailed_results_log(results, exp_dir)
        logger.info(f"Experiment completed. Results saved to: {exp_dir}")

        # 콘솔 요약
        print("\nResults summary:")
        if baseline_acc is not None:
            print(f"  Baseline accuracy : {baseline_acc:.4f}")
        if ensemble_acc is not None:
            print(f"  Ensemble accuracy : {ensemble_acc:.4f}")
        if improvement is not None:
            print(f"  Improvement       : {improvement:+.4f}")
        print(f"  Runtime           : {elapsed_minutes:.2f} minutes")

        return {
            "results": results,
            "baseline_metrics": baseline_results,
            "ensemble_metrics": ensemble_results,
        }

# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Mixture-of-Experts with Meta-Learning Router "
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed .pkl file (must contain X, y, label_encoder, ...)",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=4,
        help="Number of experts (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="Number of random seed runs to execute (default: 1).",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of trees for each expert XGBoost model (default: 200).",
    )
    parser.add_argument(
        "--router_n_estimators",
        type=int,
        default=200,
        help="Number of trees for each expert XGBoost model (default: 200).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "ensemble"],
        default=["baseline", "ensemble"],
        help="Which models to train/evaluate (default: baseline ensemble)",
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Save trained baseline / experts / meta-router into the experiment directory",
    )
    parser.add_argument(
        "--load_models_dir",
        type=str,
        default=None,
        help="Directory containing saved models to load instead of training",
    )
    parser.add_argument(
        "--assignment_mode",
        type=str,
        default="type",
    )
    parser.add_argument(
        "--use_router",
        dest="use_router",
        action="store_true",
        default=True,
        help="Use meta-router for expert weighting (default: True).",
    )
    parser.add_argument(
        "--no_router",
        dest="use_router",
        action="store_false",
        help="Disable meta-router and use simple average of expert predictions.",
    )
    parser.add_argument(
        "--smote_threshold",
        type=int,
        default=1000,
        help="Tail-class sample count threshold for applying SMOTE (default: 1000).",
    )
    parser.add_argument("--router_target",
        type=str,
        choices=["expert", "class"],
        default="expert",)


    args = parser.parse_args()

    if args.num_seeds < 1:
        raise ValueError("--num_seeds must be at least 1.")

    if args.num_seeds == 1:
        seeds = [int(args.seed)]
    else:
        rng = np.random.default_rng(args.seed)
        seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=args.num_seeds, dtype=np.uint32).tolist()]

    assignment_mode = args.assignment_mode

    print(f"\nPlanned runs: {len(seeds)} seed(s) -> {seeds}")
    print(f"Assignment mode: {assignment_mode}")
    print(f"Using router: {args.use_router}")

    all_results = []
    start_time = time.time()

    for idx, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 80)
        print(f"Running experiment {idx}/{len(seeds)} with seed={seed}")
        print("=" * 80)

        experiment = MetaRouterExperiment(
            num_experts=args.num_experts,
            n_estimators=args.n_estimators,
            smote_threshold=args.smote_threshold,
            router_target=args.router_target,
        )
        result = experiment.run_experiment(
            seed=seed,
            data_path=args.data,
            epochs=None,
            models=args.models,
            save_models=args.save_models,
            load_models_dir=args.load_models_dir,
            assignment_mode=assignment_mode,
            use_router=args.use_router,
            router_target=args.router_target,
        )
        all_results.append(result["results"])

    # 모든 시드 완료 후 집계 및 요약 출력
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("MULTI-SEED SUMMARY")
        print("=" * 80)
        
        baseline_accs = [r["baseline_accuracy"] for r in all_results if r["baseline_accuracy"] is not None]
        baseline_macro_accs = [r["baseline_macro_accuracy"] for r in all_results if r["baseline_macro_accuracy"] is not None]
        ensemble_accs = [r["ensemble_accuracy"] for r in all_results if r["ensemble_accuracy"] is not None]
        ensemble_macro_accs = [r["ensemble_macro_accuracy"] for r in all_results if r["ensemble_macro_accuracy"] is not None]
        improvements = [r["improvement"] for r in all_results if r["improvement"] is not None]
        
        if baseline_accs:
            print(f"\nBaseline Accuracy:")
            print(f"  Mean: {np.mean(baseline_accs):.4f} ± {np.std(baseline_accs):.4f}")
            print(f"  Range: [{np.min(baseline_accs):.4f}, {np.max(baseline_accs):.4f}]")
        
        if baseline_macro_accs:
            print(f"\nBaseline Macro Accuracy:")
            print(f"  Mean: {np.mean(baseline_macro_accs):.4f} ± {np.std(baseline_macro_accs):.4f}")
            print(f"  Range: [{np.min(baseline_macro_accs):.4f}, {np.max(baseline_macro_accs):.4f}]")
        
        if ensemble_accs:
            print(f"\nEnsemble Accuracy:")
            print(f"  Mean: {np.mean(ensemble_accs):.4f} ± {np.std(ensemble_accs):.4f}")
            print(f"  Range: [{np.min(ensemble_accs):.4f}, {np.max(ensemble_accs):.4f}]")
        
        if ensemble_macro_accs:
            print(f"\nEnsemble Macro Accuracy:")
            print(f"  Mean: {np.mean(ensemble_macro_accs):.4f} ± {np.std(ensemble_macro_accs):.4f}")
            print(f"  Range: [{np.min(ensemble_macro_accs):.4f}, {np.max(ensemble_macro_accs):.4f}]")
        
        if improvements:
            print(f"\nImprovement (Ensemble - Baseline):")
            print(f"  Mean: {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")
            print(f"  Range: [{np.min(improvements):+.4f}, {np.max(improvements):+.4f}]")
        
        total_runtime = (time.time() - start_time) / 60.0
        print(f"\nTotal Runtime: {total_runtime:.2f} minutes")
        print(f"Average per seed: {total_runtime / len(seeds):.2f} minutes")
        
        # JSON 파일로 저장
        summary_dir = f"multi_seed_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(summary_dir, exist_ok=True)
        
        summary = {
            "seeds": [r["seed"] for r in all_results],
            "num_seeds": len(all_results),
            "baseline_accuracy": {
                "mean": float(np.mean(baseline_accs)) if baseline_accs else None,
                "std": float(np.std(baseline_accs)) if baseline_accs else None,
                "min": float(np.min(baseline_accs)) if baseline_accs else None,
                "max": float(np.max(baseline_accs)) if baseline_accs else None,
            },
            "baseline_macro_accuracy": {
                "mean": float(np.mean(baseline_macro_accs)) if baseline_macro_accs else None,
                "std": float(np.std(baseline_macro_accs)) if baseline_macro_accs else None,
                "min": float(np.min(baseline_macro_accs)) if baseline_macro_accs else None,
                "max": float(np.max(baseline_macro_accs)) if baseline_macro_accs else None,
            },
            "ensemble_accuracy": {
                "mean": float(np.mean(ensemble_accs)) if ensemble_accs else None,
                "std": float(np.std(ensemble_accs)) if ensemble_accs else None,
                "min": float(np.min(ensemble_accs)) if ensemble_accs else None,
                "max": float(np.max(ensemble_accs)) if ensemble_accs else None,
            },
            "ensemble_macro_accuracy": {
                "mean": float(np.mean(ensemble_macro_accs)) if ensemble_macro_accs else None,
                "std": float(np.std(ensemble_macro_accs)) if ensemble_macro_accs else None,
                "min": float(np.min(ensemble_macro_accs)) if ensemble_macro_accs else None,
                "max": float(np.max(ensemble_macro_accs)) if ensemble_macro_accs else None,
            },
            "improvement": {
                "mean": float(np.mean(improvements)) if improvements else None,
                "std": float(np.std(improvements)) if improvements else None,
                "min": float(np.min(improvements)) if improvements else None,
                "max": float(np.max(improvements)) if improvements else None,
            },
            "total_runtime_minutes": float(total_runtime),
            "average_runtime_per_seed_minutes": float(total_runtime / len(seeds)),
            "all_results": all_results,
            "timestamp": datetime.now().isoformat(),
        }
        
        summary_path = os.path.join(summary_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary saved to: {summary_path}")
        print("=" * 80)


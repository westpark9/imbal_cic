#!/usr/bin/env python3
"""
XGBoost vs XGBoost Ensemble w/ different data distribution
Simple frequency-based expert assignment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm import tqdm
import warnings
import time
import json
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import logging
import xgboost as xgb

# Windows multiprocessing 문제 해결
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

# 로깅 설정
def setup_logging(exp_dir):
    """실험 디렉토리에 로그 파일 설정"""
    log_file = os.path.join(exp_dir, "experiment.log")
    
    # 로거 설정
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class CICIDS2017Dataset(Dataset):
    """CICIDS-2017 데이터셋을 위한 커스텀 Dataset 클래스"""
    
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return features, label


class SimpleExpertXGBoostClassifier:
    """XGBoost-based expert classifier for all classes with sample count limits"""

    def __init__(self, expert_classes, sample_limit, random_state=42):
        self.expert_classes = expert_classes  # 이 expert가 담당하는 클래스들 (모든 클래스)
        self.sample_limit = sample_limit  # 샘플 수 상한
        self.num_classes = len(expert_classes)
        self.random_state = random_state
        
        # XGBoost 모델 생성
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='mlogloss',
            objective='multi:softprob',
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist'
        )
        
        # 클래스 매핑 (전역 클래스 -> 로컬 클래스)
        self.class_mapping = {global_class: local_idx for local_idx, global_class in enumerate(expert_classes)}
        self.reverse_mapping = {local_idx: global_class for local_idx, global_class in enumerate(expert_classes)}

    def fit(self, features, labels):
        """XGBoost 모델 훈련 (샘플 수 제한 적용)"""
        # 샘플 수 제한을 적용한 데이터 생성
        limited_features, limited_labels = self._apply_sample_limit(features, labels)
        
        if len(limited_labels) == 0:
            print(f"       WARNING: Expert has no training samples after applying sample limit!")
            return
        
        # 전역 클래스를 로컬 클래스로 변환
        local_labels = []
        valid_indices = []
        
        for i, label in enumerate(limited_labels):
            if label in self.class_mapping:
                local_labels.append(self.class_mapping[label])
                valid_indices.append(i)
        
        if len(local_labels) == 0:
            print(f"       WARNING: Expert has no valid training samples!")
            return
        
        # 유효한 샘플만 사용
        local_labels = np.array(local_labels)
        valid_features = limited_features[valid_indices]
        
        # 클래스 수 확인 및 조정
        unique_classes = np.unique(local_labels)
        actual_num_classes = len(unique_classes)
        
        if actual_num_classes < 2:
            print(f"       WARNING: Expert has only {actual_num_classes} class(es), skipping training")
            return
        
        # XGBoost 모델 재생성 (올바른 클래스 수로)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss',
            objective='multi:softprob',
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            num_class=actual_num_classes  # 실제 클래스 수로 설정
        )
        
        # XGBoost 훈련
        self.xgb_model.fit(valid_features, local_labels)
    
    def _apply_sample_limit(self, features, labels):
        """샘플 수 제한을 적용하여 데이터 생성"""
        limited_features = []
        limited_labels = []
        
        # 클래스별로 샘플 수 제한 적용
        for class_id in self.expert_classes:
            class_mask = (labels == class_id)
            class_features = features[class_mask]
            class_labels = labels[class_mask]
            
            # 샘플 수 제한 적용
            if len(class_features) > self.sample_limit:
                # 무작위로 샘플 선택
                np.random.seed(self.random_state)
                selected_indices = np.random.choice(
                    len(class_features), 
                    size=self.sample_limit, 
                    replace=False
                )
                class_features = class_features[selected_indices]
                class_labels = class_labels[selected_indices]
            
            limited_features.append(class_features)
            limited_labels.append(class_labels)
        
        # 결합
        if limited_features:
            limited_features = np.vstack(limited_features)
            limited_labels = np.hstack(limited_labels)
        else:
            limited_features = np.array([])
            limited_labels = np.array([])
        
        return limited_features, limited_labels

    def predict(self, features):
        """예측 클래스 반환"""
        if len(self.expert_classes) == 0 or not hasattr(self.xgb_model, 'get_booster'):
            return np.zeros(features.shape[0], dtype=int)
        
        try:
            # XGBoost 예측
            local_pred = self.xgb_model.predict(features)
            
            # 로컬 클래스를 전역 클래스로 변환
            global_pred = np.array([self.reverse_mapping.get(pred, self.expert_classes[0]) for pred in local_pred])
            
            return global_pred
        except:
            # 예측 실패 시 무작위 예측 반환
            return np.random.randint(0, len(self.expert_classes), features.shape[0])


class SimpleExperiment:
    """
    Simple frequency-based expert assignment experiment
    CICIDS-2017 Network Intrusion Detection Dataset Version with XGBoost Experts
    """

    def __init__(self, num_experts=4):
        self.num_experts = num_experts

        print("Simple Frequency-Based Expert Assignment Experiment (XGBoost Experts)")
        print(f"   Dataset: CICIDS-2017 Network Intrusion Detection")
        print(f"   Number of experts: {num_experts}")
        print(f"   Expert type: XGBoost")
        print(f"   Assignment: Based on class sample frequency")

    def set_random_seed(self, seed):
        """Set all random seeds for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)

    def load_cicids2017_data(self, data_dir):
        print("   Loading CICIDS-2017 dataset...")
        
        # 데이터 파일 목록 (CICIDS-2017 전체)
        data_files = [
            'Monday-WorkingHours.pcap_ISCX.csv',
            'Tuesday-WorkingHours.pcap_ISCX.csv',
            'Wednesday-workingHours.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv'
        ]
        
        all_csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        for csv_file in all_csv_files:
            if csv_file not in data_files:
                print(f"     Adding missing file: {csv_file}")
                data_files.append(csv_file)
        
        all_data = []
        
        for file_name in data_files:
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)

                    label_column = None
                    for col in df.columns:
                        if col.strip() == 'Label':
                            label_column = col
                            break
                    
                    if label_column is not None:
                        # 정상 트래픽과 공격 트래픽 분리
                        df[label_column] = df[label_column].str.strip()
                        
                        df = df.dropna()
                        
                        df = df.replace([np.inf, -np.inf], np.nan)
                        df = df.dropna()
                        
                        if len(df) > 0:
                            label_counts = df[label_column].value_counts()
                        
                        all_data.append(df)
                    else:
                        print(f"       Warning: {file_name} has no 'Label' column")
                        print(f"       Available columns: {list(df.columns)}")
                        
                except Exception as e:
                    print(f"     Error loading {file_name}: {e}")
            else:
                print(f"     Warning: {file_path} not found")
        
        if not all_data:
            raise ValueError("No valid data files found!")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"     Total samples: {len(combined_df)}")
        
        label_column = None
        for col in combined_df.columns:
            if col.strip() == 'Label':
                label_column = col
                break
        
        if label_column is None:
            raise ValueError("Label column not found in combined data!")
        
        label_encoder = LabelEncoder()
        combined_df['Label_encoded'] = label_encoder.fit_transform(combined_df[label_column])
        
        feature_columns = [col for col in combined_df.columns if col not in [label_column, 'Label_encoded']]
        X_df = combined_df[feature_columns].copy()
        y = combined_df['Label_encoded'].values
        
        # 특성 전처리 및 스케일링
        print(f"     Feature preprocessing and scaling...")
        
        # 1-1. 중복 feature 제거
        if 'Destination Port' in X_df.columns:
            X_df = X_df.drop(columns=['Destination Port'])
        if 'Fwd Header Length' in X_df.columns:
            X_df = X_df.drop(columns=['Fwd Header Length'])
        
        # 1-2. 특정 열의 -1 값을 0으로 처리
        win_bytes_cols = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward']
        for col in win_bytes_cols:
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(-1, 0)
        
        # 1-3. StandardScaler 적용
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        
        # 최종 NaN 검증 및 처리
        nan_count = np.isnan(X_scaled).sum()
        if nan_count > 0:
            print(f"       WARNING: Found {nan_count} NaN values after scaling, replacing with 0")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        else:
            print(f"       - No NaN values found after scaling")
        
        # inf 값 처리
        inf_count = np.isinf(X_scaled).sum()
        if inf_count > 0:
            print(f"       WARNING: Found {inf_count} inf values, replacing with 0")
            X_scaled = np.nan_to_num(X_scaled, posinf=0.0, neginf=0.0)

        # 최종 결과
        X_final = X_scaled
        
        # 클래스별 샘플 수 확인
        unique_labels, counts = np.unique(y, return_counts=True)
        
        return X_final, y, label_encoder, None, feature_columns

    def setup_dataset(self, seed, data_dir):
        """CICIDS-2017 데이터셋 설정"""
        self.set_random_seed(seed)
        
        # 데이터 로드
        X, y, self.label_encoder, _, self.feature_columns = self.load_cicids2017_data(data_dir)
        
        # 클래스 수 설정
        self.num_classes = len(self.label_encoder.classes_)
        # 훈련/테스트 분할 (8:2) - 클래스별 개별 분할 후 결합
        print(f"     Splitting data into train/test (8:2 ratio) by class...")
        
        # 클래스별로 개별 분할 후 결합
        all_X_train = []
        all_X_test = []
        all_y_train = []
        all_y_test = []
        
        unique_classes = np.unique(y)
        for class_id in unique_classes:
            class_mask = (y == class_id)
            class_X = X[class_mask]
            class_y = y[class_mask]
            
            if len(class_X) > 1:  # 최소 2개 샘플이 있어야 분할 가능
                X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
                    class_X, class_y, test_size=0.2, random_state=seed
                )
                
                all_X_train.append(X_train_class)
                all_X_test.append(X_test_class)
                all_y_train.append(y_train_class)
                all_y_test.append(y_test_class)
            
        
        # 결합
        X_train = np.vstack(all_X_train) if all_X_train else np.array([])
        X_test = np.vstack(all_X_test) if all_X_test else np.array([])
        y_train = np.hstack(all_y_train) if all_y_train else np.array([])
        y_test = np.hstack(all_y_test) if all_y_test else np.array([])
        
        # 클래스별 분포 확인
        unique_classes = np.unique(y)
        total_train = len(X_train)
        total_test = len(X_test)

        
        # 데이터 저장
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # 데이터셋 객체 생성
        self.train_subset = CICIDS2017Dataset(X_train, y_train)
        self.test_subset = CICIDS2017Dataset(X_test, y_test)
        

    def get_frequency_based_groups(self, seed):
        """모든 전문가가 모든 클래스를 담당하되 샘플 수만 다르게 제한하는 방식"""
        print("   Creating frequency-based expert groups with sample count limits...")
        
        # 훈련 데이터에서 클래스별 샘플 수 계산
        unique_labels, counts = np.unique(self.y_train, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))
        
        # 클래스들을 샘플 수 기준으로 정렬 (내림차순)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_ids = [item[0] for item in sorted_classes]
        class_sample_counts = [item[1] for item in sorted_classes]
        
        print(f"     Total classes: {len(class_ids)}")
        print(f"     Sample count range: {min(class_sample_counts)} - {max(class_sample_counts)}")
        
        # 클래스별 샘플 수를 내림차순으로 정렬
        sorted_class_samples = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        sample_counts = [item[1] for item in sorted_class_samples]
        
        # 4개 구간으로 나누어 각 구간의 경계값 계산
        num_classes = len(sample_counts)
        quartile_indices = []
        for i in range(1, self.num_experts + 1):
            # 1/4, 2/4, 3/4, 4/4 지점의 인덱스 계산
            idx = int((i * num_classes) / self.num_experts) - 1
            idx = max(0, min(idx, num_classes - 1))  # 범위 내로 제한
            quartile_indices.append(idx)
        
        # 각 구간의 경계값을 전문가의 샘플 수 제한으로 설정
        sample_limits = []
        for expert_idx in range(self.num_experts):
            if expert_idx < len(quartile_indices):
                limit = sample_counts[quartile_indices[expert_idx]]
            else:
                # 전문가 수가 구간 수보다 많은 경우, 마지막 구간 사용
                limit = sample_counts[quartile_indices[-1]]
            sample_limits.append(limit)
        
        print(f"     Expert sample limits (based on quartile boundaries): {sample_limits}")
        print(f"     Quartile indices: {quartile_indices}")
        
        # 각 구간별 클래스 분포 출력
        print("     Quartile-based sample limit assignment:")
        for expert_idx, (limit, quartile_idx) in enumerate(zip(sample_limits, quartile_indices)):
            print(f"       Expert {expert_idx}: sample limit = {limit} (quartile {expert_idx + 1}/{self.num_experts})")
            
            # 해당 구간에 속하는 클래스들 찾기
            if expert_idx == 0:
                # 첫 번째 구간: 0부터 quartile_idx까지
                start_idx = 0
                end_idx = quartile_idx + 1
            else:
                # 나머지 구간: 이전 quartile_idx + 1부터 현재 quartile_idx까지
                start_idx = quartile_indices[expert_idx - 1] + 1
                end_idx = quartile_idx + 1
            
            for i in range(start_idx, end_idx):
                if i < len(sorted_class_samples):
                    class_id, count = sorted_class_samples[i]
                    class_name = self.label_encoder.inverse_transform([class_id])[0]
                    print(f"         · class {class_id} ({class_name}): {count} samples")
        
        # 모든 전문가가 모든 클래스를 담당 (샘플 수만 제한)
        expert_groups = []
        for expert_idx in range(self.num_experts):
            expert_classes = class_ids.copy()  # 모든 클래스 포함
            expert_groups.append(expert_classes)
        
        # 결과 출력 - 각 전문가별 샘플 수 제한 정보
        print("     Expert groups composition (all experts have all classes):")
        for expert_idx, class_list in enumerate(expert_groups):
            print(f"       Expert {expert_idx}: {len(class_list)} classes, sample limit: {sample_limits[expert_idx]}")
            
            # 각 클래스별 샘플 수 제한 정보
            for cid in class_list:
                original_count = class_counts[cid]
                limited_count = min(original_count, sample_limits[expert_idx])
                class_name = self.label_encoder.inverse_transform([cid])[0]
                print(f"         · class {cid} ({class_name}): {limited_count}/{original_count} samples")
        
        # 빈 expert 검증
        empty_experts = [i for i, group in enumerate(expert_groups) if len(group) == 0]
        if empty_experts:
            print(f"       WARNING: Experts {empty_experts} have no classes assigned!")

        return expert_groups

    def train_baseline(self, seed):
        """베이스라인 XGBoost 모델 학습"""
        print("   Training baseline XGBoost model...")

        self.set_random_seed(seed)

        # XGBoost 모델 생성
        baseline_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            eval_metric='mlogloss',
            objective='multi:softprob',
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist'
        )
        
        # XGBoost 훈련
        print("     Training XGBoost...")
        baseline_model.fit(self.X_train, self.y_train)
        
        # XGBoost는 epochs 개념이 없으므로 단순한 히스토리 반환
        training_history = {
            'trained': True
        }
        
        return baseline_model, training_history

    def train_expert_ensemble(self, expert_groups, seed):
        self.set_random_seed(seed + 300)
        
        # 샘플 수 제한 계산 (순서 기반)
        unique_labels, counts = np.unique(self.y_train, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))
        
        # 클래스별 샘플 수를 내림차순으로 정렬
        sorted_class_samples = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        sample_counts = [item[1] for item in sorted_class_samples]
        
        # 4개 구간으로 나누어 각 구간의 경계값 계산
        num_classes = len(sample_counts)
        quartile_indices = []
        for i in range(1, self.num_experts + 1):
            # 1/4, 2/4, 3/4, 4/4 지점의 인덱스 계산
            idx = int((i * num_classes) / self.num_experts) - 1
            idx = max(0, min(idx, num_classes - 1))  # 범위 내로 제한
            quartile_indices.append(idx)
        
        # 각 구간의 경계값을 전문가의 샘플 수 제한으로 설정
        sample_limits = []
        for expert_idx in range(self.num_experts):
            if expert_idx < len(quartile_indices):
                limit = sample_counts[quartile_indices[expert_idx]]
            else:
                # 전문가 수가 구간 수보다 많은 경우, 마지막 구간 사용
                limit = sample_counts[quartile_indices[-1]]
            sample_limits.append(limit)
        
        # XGBoost Expert Classifiers 생성
        expert_classifiers = []
        for expert_idx, group in enumerate(expert_groups):
            expert_classifier = SimpleExpertXGBoostClassifier(
                expert_classes=group, 
                sample_limit=sample_limits[expert_idx],
                random_state=seed + expert_idx
            )
            expert_classifiers.append(expert_classifier)

        # XGBoost Experts 훈련
        print("   Training XGBoost experts...")
        for expert_idx, (expert_classifier, expert_classes) in enumerate(
            zip(expert_classifiers, expert_groups)
        ):
            print(f"     Training XGBoost Expert {expert_idx} (sample limit: {sample_limits[expert_idx]})...")
            expert_classifier.fit(self.X_train, self.y_train)
        
        # XGBoost는 epochs 개념이 없으므로 단순한 히스토리 반환
        training_history = {
            'trained': True
        }
        
        expert_history = {}
        for expert_idx in range(len(expert_classifiers)):
            expert_history[expert_idx] = {
                'trained': True,
                'sample_limit': sample_limits[expert_idx]
            }
        
        return expert_classifiers, training_history, expert_history


    def compute_macro_accuracy(self, y_true, y_pred, num_classes):
        """Macro accuracy 계산"""
        accs = []
        for cls in range(num_classes):
            cls_idx = (y_true == cls)
            if cls_idx.sum() == 0:
                continue
            acc = np.mean(y_pred[cls_idx] == y_true[cls_idx])
            accs.append(acc)
        return np.mean(accs) if accs else 0.0

    def compute_per_class_metrics(self, predictions, targets, num_classes):
        """클래스별 정확도와 F1 점수를 계산합니다."""
        per_class_acc = []
        per_class_f1 = []
        
        for class_idx in range(num_classes):
            # 해당 클래스에 대한 정확도
            class_mask = (targets == class_idx)
            if class_mask.sum() > 0:
                acc = (predictions[class_mask] == class_idx).astype(float).mean()
            else:
                acc = 0.0
            per_class_acc.append(acc)
            
            # 해당 클래스에 대한 F1 점수
            f1 = f1_score(targets == class_idx, predictions == class_idx, zero_division=0)
            per_class_f1.append(f1)
        
        return per_class_acc, per_class_f1

    def plot_confusion_matrix(self, cm, class_names, title, save_path, cmap="Blues"):
        """혼동 행렬을 시각화합니다."""
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_per_class_metrics(self, baseline_acc, baseline_f1, ensemble_acc, ensemble_f1, class_names, save_path):
        """클래스별 성능 지표를 시각화합니다."""
        plt.figure(figsize=(20, 8))
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.subplot(1, 2, 1)
        labels = []
        all_acc_values = []
        if baseline_acc is not None and len(baseline_acc) > 0:
            plt.bar(x - width/2, baseline_acc, width, label='Baseline XGBoost', alpha=0.8)
            labels.append('Baseline XGBoost')
            all_acc_values.extend(baseline_acc)
        if ensemble_acc is not None and len(ensemble_acc) > 0:
            plt.bar(x + width/2, ensemble_acc, width, label='Expert Ensemble (XGBoost)', alpha=0.8)
            labels.append('Expert Ensemble (XGBoost)')
            all_acc_values.extend(ensemble_acc)
        plt.title('Per-Class Accuracy')
        plt.xticks(x, class_names, rotation=45, ha='right')
        if all_acc_values:
            y_min = max(0, min(all_acc_values) - 0.05)
            y_max = min(1, max(all_acc_values) + 0.05)
            plt.ylim(y_min, y_max)
        if labels:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        all_f1_values = []
        if baseline_f1 is not None and len(baseline_f1) > 0:
            plt.bar(x - width/2, baseline_f1, width, label='Baseline XGBoost', alpha=0.8)
            all_f1_values.extend(baseline_f1)
        if ensemble_f1 is not None and len(ensemble_f1) > 0:
            plt.bar(x + width/2, ensemble_f1, width, label='Expert Ensemble (XGBoost)', alpha=0.8)
            all_f1_values.extend(ensemble_f1)
        plt.title('Per-Class F1 Score')
        plt.xticks(x, class_names, rotation=45, ha='right')
        if all_f1_values:
            y_min = max(0, min(all_f1_values) - 0.05)
            y_max = min(1, max(all_f1_values) + 0.05)
            plt.ylim(y_min, y_max)
        if labels:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_performance_metrics_two_models(self, baseline_acc, baseline_f1, ensemble_acc, ensemble_f1, class_names, save_path):
        """2개 모델의 성능 지표를 CSV 파일로 저장합니다."""
        # 빈 배열인 경우 None으로 변환
        baseline_acc = baseline_acc if baseline_acc and len(baseline_acc) > 0 else None
        baseline_f1 = baseline_f1 if baseline_f1 and len(baseline_f1) > 0 else None
        ensemble_acc = ensemble_acc if ensemble_acc and len(ensemble_acc) > 0 else None
        ensemble_f1 = ensemble_f1 if ensemble_f1 and len(ensemble_f1) > 0 else None
        
        # 데이터 딕셔너리 생성
        data_dict = {'Class': class_names}
        
        if baseline_acc is not None:
            data_dict['Baseline_XGBoost_Accuracy'] = baseline_acc
            data_dict['Baseline_XGBoost_F1'] = baseline_f1
        if ensemble_acc is not None:
            data_dict['Expert_Ensemble_XGBoost_Accuracy'] = ensemble_acc
            data_dict['Expert_Ensemble_XGBoost_F1'] = ensemble_f1
        
        df = pd.DataFrame(data_dict)
        df.to_csv(save_path, index=False)
        
        # 로그에 저장 완료 메시지 추가
        logger = logging.getLogger('experiment')
        logger.info(f"Per-class metrics saved to: {save_path}")
        logger.info(f"Saved metrics for {len(class_names)} classes")

    def save_detailed_performance_metrics(self, baseline_acc, baseline_f1, baseline_precision, baseline_recall, baseline_support,
                                        ensemble_acc, ensemble_f1, ensemble_precision, ensemble_recall, ensemble_support,
                                        class_names, save_path):
        """상세한 classification report를 CSV 파일로 저장합니다."""
        # 빈 배열인 경우 None으로 변환
        baseline_acc = baseline_acc if baseline_acc and len(baseline_acc) > 0 else None
        baseline_f1 = baseline_f1 if baseline_f1 and len(baseline_f1) > 0 else None
        baseline_precision = baseline_precision if baseline_precision and len(baseline_precision) > 0 else None
        baseline_recall = baseline_recall if baseline_recall and len(baseline_recall) > 0 else None
        baseline_support = baseline_support if baseline_support and len(baseline_support) > 0 else None
        
        ensemble_acc = ensemble_acc if ensemble_acc and len(ensemble_acc) > 0 else None
        ensemble_f1 = ensemble_f1 if ensemble_f1 and len(ensemble_f1) > 0 else None
        ensemble_precision = ensemble_precision if ensemble_precision and len(ensemble_precision) > 0 else None
        ensemble_recall = ensemble_recall if ensemble_recall and len(ensemble_recall) > 0 else None
        ensemble_support = ensemble_support if ensemble_support and len(ensemble_support) > 0 else None
        
        # 데이터 딕셔너리 생성
        data_dict = {'Class': class_names}
        
        if baseline_acc is not None:
            data_dict['Baseline_Accuracy'] = baseline_acc
            data_dict['Baseline_F1'] = baseline_f1
            data_dict['Baseline_Precision'] = baseline_precision
            data_dict['Baseline_Recall'] = baseline_recall
            data_dict['Baseline_Support'] = baseline_support
            
        if ensemble_acc is not None:
            data_dict['Ensemble_Accuracy'] = ensemble_acc
            data_dict['Ensemble_F1'] = ensemble_f1
            data_dict['Ensemble_Precision'] = ensemble_precision
            data_dict['Ensemble_Recall'] = ensemble_recall
            data_dict['Ensemble_Support'] = ensemble_support
        
        df = pd.DataFrame(data_dict)
        df.to_csv(save_path, index=False)
        
        # 로그에 저장 완료 메시지 추가
        logger = logging.getLogger('experiment')
        logger.info(f"Detailed performance metrics saved to: {save_path}")
        logger.info(f"Saved detailed metrics for {len(class_names)} classes")

    def save_detailed_results_log(self, results, exp_dir):
        """전체 실험 결과를 로그 파일에 상세히 저장합니다."""
        logger = logging.getLogger('experiment')
        
        logger.info("=" * 80)
        logger.info("EXPERIMENT RESULTS SUMMARY (Simple XGBoost Experts)")
        logger.info("=" * 80)
        
        # 기본 정보
        logger.info(f"Seed: {results['seed']}")
        logger.info(f"Runtime: {results['runtime_minutes']:.1f} minutes")
        logger.info(f"Timestamp: {results['timestamp']}")
        logger.info(f"Results saved to: {exp_dir}")
        
        # Train/Test 데이터 분포
        logger.info("\nTrain/Test Data Distribution:")
        logger.info("Train Set:")
        unique_train, counts_train = np.unique(self.y_train, return_counts=True)
        total_train = len(self.y_train)
        for class_id, count in zip(unique_train, counts_train):
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            percentage = (count / total_train) * 100
            logger.info(f"  {class_name:25s}: {count:7d} ({(percentage):.2f}%)")
        logger.info(f"  Total Train Samples: {total_train}")
        
        logger.info("Test Set:")
        unique_test, counts_test = np.unique(self.y_test, return_counts=True)
        total_test = len(self.y_test)
        for class_id, count in zip(unique_test, counts_test):
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            percentage = (count / total_test) * 100
            logger.info(f"  {class_name:25s}: {count:7d} ({(percentage):.2f}%)")
        logger.info(f"  Total Test Samples: {total_test}")
        

        # 전문가 그룹 정보
        if results['expert_groups']:
            logger.info("\nExpert Groups Composition:")
            class_names = self.label_encoder.classes_
            for expert_idx, group in enumerate(results['expert_groups']):
                logger.info(f"  Expert {expert_idx}:")
                for class_id in group:
                    class_name = class_names[class_id]
                    # 해당 클래스의 훈련 샘플 수 계산
                    class_count = np.sum(self.y_train == class_id)
                    logger.info(f"    - {class_name} (ID: {class_id}, Train samples: {class_count})")
        
        
        if results:
            # Baseline 모델 성능 지표 로깅
            if 'baseline_accuracy' in results:
                logger.info("\nBaseline XGBoost Performance:")
                logger.info(f"  Accuracy: {results['baseline_accuracy']:.4f}")
                logger.info(f"  Macro Accuracy: {results['baseline_macro_accuracy']:.4f}")
                
                # 클래스별 성능 지표
                if 'baseline_per_class_acc' in results and 'baseline_per_class_f1' in results:
                    logger.info("\nBaseline Per-Class Performance:")
                    class_names = self.label_encoder.classes_
                    logger.info(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
                    logger.info("-" * 70)
                    for i, class_name in enumerate(class_names):
                        acc = results['baseline_per_class_acc'][i] if i < len(results['baseline_per_class_acc']) else 0.0
                        f1 = results['baseline_per_class_f1'][i] if i < len(results['baseline_per_class_f1']) else 0.0
                        precision = results['baseline_per_class_precision'][i] if i < len(results['baseline_per_class_precision']) else 0.0
                        recall = results['baseline_per_class_recall'][i] if i < len(results['baseline_per_class_recall']) else 0.0
                        support = results['baseline_per_class_support'][i] if i < len(results['baseline_per_class_support']) else 0
                        logger.info(f"  {class_name:25s} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<8d}")
            
            # Ensemble 모델 성능 지표 로깅
            if 'ensemble_accuracy' in results:
                logger.info("\nExpert Ensemble Performance:")
                logger.info(f"  Accuracy: {results['ensemble_accuracy']:.4f}")
                logger.info(f"  Macro Accuracy: {results['ensemble_macro_accuracy']:.4f}")
                
                # 클래스별 성능 지표
                if 'ensemble_per_class_acc' in results and 'ensemble_per_class_f1' in results:
                    logger.info("\nEnsemble Per-Class Performance:")
                    class_names = self.label_encoder.classes_
                    logger.info(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
                    logger.info("-" * 70)
                    for i, class_name in enumerate(class_names):
                        acc = results['ensemble_per_class_acc'][i] if i < len(results['ensemble_per_class_acc']) else 0.0
                        f1 = results['ensemble_per_class_f1'][i] if i < len(results['ensemble_per_class_f1']) else 0.0
                        precision = results['ensemble_per_class_precision'][i] if i < len(results['ensemble_per_class_precision']) else 0.0
                        recall = results['ensemble_per_class_recall'][i] if i < len(results['ensemble_per_class_recall']) else 0.0
                        support = results['ensemble_per_class_support'][i] if i < len(results['ensemble_per_class_support']) else 0
                        logger.info(f"  {class_name:25s} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<8d}")
                
                # 개선도 표시
                if 'improvement_ensemble' in results:
                    improvement = results['improvement_ensemble']
                    logger.info(f"\nEnsemble Improvement: {improvement:+.4f}")
        
        logger.info("=" * 80)

    def evaluate_model(self, model):
        """단일 모델 평가 및 클래스별 성능 지표 계산"""
        # XGBoost 예측
        predictions = model.predict(self.X_test)

        # 전체 정확도
        accuracy = (predictions == self.y_test).mean()
        
        # Micro/Macro 정확도 계산
        micro_accuracy = accuracy_score(self.y_test, predictions)
        macro_accuracy = self.compute_macro_accuracy(self.y_test, predictions, self.num_classes)
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            predictions, self.y_test, self.num_classes
        )
        
        # 상세한 classification report 계산
        class_names = self.label_encoder.classes_
        report = classification_report(self.y_test, predictions, target_names=class_names, output_dict=True, zero_division=0)
        
        # precision, recall, f1, support 추출
        per_class_precision = []
        per_class_recall = []
        per_class_f1_detailed = []
        per_class_support = []
        
        for i, class_name in enumerate(class_names):
            # 클래스 이름으로 직접 찾기 (target_names 사용 시)
            if class_name in report:
                per_class_precision.append(report[class_name]['precision'])
                per_class_recall.append(report[class_name]['recall'])
                per_class_f1_detailed.append(report[class_name]['f1-score'])
                per_class_support.append(int(report[class_name]['support']))
            elif str(i) in report:
                per_class_precision.append(report[str(i)]['precision'])
                per_class_recall.append(report[str(i)]['recall'])
                per_class_f1_detailed.append(report[str(i)]['f1-score'])
                per_class_support.append(int(report[str(i)]['support']))
            else:
                # 클래스를 찾을 수 없는 경우 0으로 설정
                per_class_precision.append(0.0)
                per_class_recall.append(0.0)
                per_class_f1_detailed.append(0.0)
                per_class_support.append(0)
        
        # 혼동 행렬
        cm = confusion_matrix(self.y_test, predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm, per_class_precision, per_class_recall, per_class_f1_detailed, per_class_support

    def evaluate_expert_ensemble(self, expert_classifiers, expert_groups):
        """전문가 앙상블 평가 및 클래스별 성능 지표 계산"""
        print("   Evaluating expert ensemble (all experts predict)...")
        
        # 모든 전문가의 예측을 수집
        all_expert_predictions = []
        for expert_idx, expert_classifier in enumerate(expert_classifiers):
            print(f"     Expert {expert_idx} predicting...")
            expert_pred = expert_classifier.predict(self.X_test)
            all_expert_predictions.append(expert_pred)
        
        all_expert_predictions = np.array(all_expert_predictions)  # shape: (num_experts, num_samples)
        
        # 앙상블 예측 방법 1: 투표 방식 (Majority Voting)
        predictions_voting = []
        for sample_idx in range(len(self.X_test)):
            # 각 샘플에 대해 모든 전문가의 예측을 수집
            sample_predictions = all_expert_predictions[:, sample_idx]
            
            # 가장 많이 예측된 클래스 선택
            unique, counts = np.unique(sample_predictions, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            predictions_voting.append(majority_class)
        
        predictions_voting = np.array(predictions_voting)
        
        # Majority voting 사용
        predictions = predictions_voting
        
        print(f"     Using majority voting for ensemble prediction")

        # 전체 정확도
        accuracy = (predictions == self.y_test).mean()
        
        # Micro/Macro 정확도 계산
        micro_accuracy = accuracy_score(self.y_test, predictions)
        macro_accuracy = self.compute_macro_accuracy(self.y_test, predictions, self.num_classes)
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            predictions, self.y_test, self.num_classes
        )
        
        # 상세한 classification report 계산
        class_names = self.label_encoder.classes_
        report = classification_report(self.y_test, predictions, target_names=class_names, output_dict=True, zero_division=0)
        
        # precision, recall, f1, support 추출
        per_class_precision = []
        per_class_recall = []
        per_class_f1_detailed = []
        per_class_support = []
        
        for i, class_name in enumerate(class_names):
            # 클래스 이름으로 직접 찾기 (target_names 사용 시)
            if class_name in report:
                per_class_precision.append(report[class_name]['precision'])
                per_class_recall.append(report[class_name]['recall'])
                per_class_f1_detailed.append(report[class_name]['f1-score'])
                per_class_support.append(int(report[class_name]['support']))
            elif str(i) in report:
                per_class_precision.append(report[str(i)]['precision'])
                per_class_recall.append(report[str(i)]['recall'])
                per_class_f1_detailed.append(report[str(i)]['f1-score'])
                per_class_support.append(int(report[str(i)]['support']))
            else:
                # 클래스를 찾을 수 없는 경우 0으로 설정
                per_class_precision.append(0.0)
                per_class_recall.append(0.0)
                per_class_f1_detailed.append(0.0)
                per_class_support.append(0)
        
        # 혼동 행렬
        cm = confusion_matrix(self.y_test, predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm, per_class_precision, per_class_recall, per_class_f1_detailed, per_class_support

    def run_single_experiment(self, seed, data_dir, models=['baseline', 'ensemble']):
        """단일 실험 실행 및 결과 저장"""
        # 실험 시작 (터미널 출력 제거)

        start_time = time.time()

        # 실험 디렉토리 생성
        exp_dir = f"experiment_results_simple_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # 로깅 설정
        logger = setup_logging(exp_dir)
        logger.info(f"Starting experiment with seed {seed}")
        logger.info(f"Models to train: {models}")
        logger.info(f"Data directory: {data_dir}")

        # 데이터셋 설정
        self.setup_dataset(seed, data_dir)

        # 모델별 학습 및 평가 (선택적)
        baseline_model = None
        baseline_acc = 0.0
        baseline_macro_acc = 0.0
        baseline_per_class_acc = []
        baseline_per_class_f1 = []
        baseline_per_class_precision = []
        baseline_per_class_recall = []
        baseline_per_class_support = []
        baseline_cm = None
        
        expert_classifiers = None
        expert_groups = None
        ensemble_acc = 0.0
        ensemble_macro_acc = 0.0
        ensemble_per_class_acc = []
        ensemble_per_class_f1 = []
        ensemble_per_class_precision = []
        ensemble_per_class_recall = []
        ensemble_per_class_support = []
        ensemble_cm = None
        
        # 훈련 과정 추적을 위한 변수들
        training_history = {}
        expert_history = {}

        # 베이스라인 모델
        if 'baseline' in models:
            baseline_model, baseline_history = self.train_baseline(seed)
            baseline_acc, baseline_per_class_acc, baseline_per_class_f1, baseline_cm, baseline_per_class_precision, baseline_per_class_recall, _, baseline_per_class_support = self.evaluate_model(baseline_model)
            baseline_macro_acc = self.compute_macro_accuracy(self.y_test, baseline_model.predict(self.X_test), self.num_classes)
            training_history['Baseline'] = baseline_history

        # 앙상블 모델
        if 'ensemble' in models:
            expert_groups = self.get_frequency_based_groups(seed)
            expert_classifiers, ensemble_history, expert_history = self.train_expert_ensemble(
                expert_groups, seed
            )
            # 앙상블 평가 (ground truth 독립적)
            ensemble_results = self.evaluate_expert_ensemble(expert_classifiers, expert_groups)
            ensemble_acc, ensemble_per_class_acc, ensemble_per_class_f1, ensemble_cm, ensemble_per_class_precision, ensemble_per_class_recall, _, ensemble_per_class_support = ensemble_results
            
            # 앙상블의 macro accuracy 계산 (이미 평가된 결과를 사용, 중복 호출 방지)
            # 모든 전문가의 예측을 다시 수행하여 macro accuracy 계산
            all_expert_predictions = []
            for expert_classifier in expert_classifiers:
                expert_pred = expert_classifier.predict(self.X_test)
                all_expert_predictions.append(expert_pred)
            all_expert_predictions = np.array(all_expert_predictions)
            
            # 투표 방식으로 최종 예측
            ensemble_predictions_for_macro = []
            for sample_idx in range(len(self.X_test)):
                sample_predictions = all_expert_predictions[:, sample_idx]
                unique, counts = np.unique(sample_predictions, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                ensemble_predictions_for_macro.append(majority_class)
            ensemble_predictions_for_macro = np.array(ensemble_predictions_for_macro)
            
            ensemble_macro_acc = self.compute_macro_accuracy(self.y_test, ensemble_predictions_for_macro, self.num_classes)
            training_history['Ensemble'] = ensemble_history

        elapsed_time = (time.time() - start_time) / 60
        improvement_ensemble = ensemble_acc - baseline_acc

        # 결과 시각화 및 저장
        class_names = self.label_encoder.classes_
        
        # 혼동 행렬 시각화 - 유효한 모델만
        if 'baseline' in models and baseline_cm is not None:
            self.plot_confusion_matrix(
                baseline_cm, class_names,
                "Baseline XGBoost Confusion Matrix",
                os.path.join(exp_dir, "baseline_confusion_matrix.png")
            )
        if 'ensemble' in models and ensemble_cm is not None:
            self.plot_confusion_matrix(
                ensemble_cm, class_names,
                "Expert Ensemble (XGBoost) Confusion Matrix",
                os.path.join(exp_dir, "ensemble_confusion_matrix.png")
            )
        
        # 클래스별 성능 지표 시각화 (실행된 모델들만 표시)
        self.plot_per_class_metrics(
            baseline_per_class_acc if 'baseline' in models else None,
            baseline_per_class_f1 if 'baseline' in models else None,
            ensemble_per_class_acc if 'ensemble' in models else None,
            ensemble_per_class_f1 if 'ensemble' in models else None,
            class_names,
            os.path.join(exp_dir, "per_class_metrics.png")
        )
        
        # 성능 지표 CSV 저장 (상세한 classification report 포함)
        self.save_detailed_performance_metrics(
            baseline_per_class_acc if 'baseline' in models else None,
            baseline_per_class_f1 if 'baseline' in models else None,
            baseline_per_class_precision if 'baseline' in models else None,
            baseline_per_class_recall if 'baseline' in models else None,
            baseline_per_class_support if 'baseline' in models else None,
            ensemble_per_class_acc if 'ensemble' in models else None,
            ensemble_per_class_f1 if 'ensemble' in models else None,
            ensemble_per_class_precision if 'ensemble' in models else None,
            ensemble_per_class_recall if 'ensemble' in models else None,
            ensemble_per_class_support if 'ensemble' in models else None,
            class_names,
            os.path.join(exp_dir, "detailed_performance_metrics.csv")
        )

        # JSON 직렬화를 위해 numpy 타입을 Python 기본 타입으로 변환
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            else:
                return obj

        results = {
            "seed": int(seed),
            "baseline_accuracy": float(baseline_acc),
            "baseline_macro_accuracy": float(baseline_macro_acc),
            "ensemble_accuracy": float(ensemble_acc),
            "ensemble_macro_accuracy": float(ensemble_macro_acc),
            "improvement_ensemble": float(improvement_ensemble),
            "runtime_minutes": float(elapsed_time),
            "expert_groups": convert_numpy_types(expert_groups),
            "timestamp": datetime.now().isoformat(),
            "baseline_per_class_acc": convert_numpy_types(baseline_per_class_acc),
            "baseline_per_class_f1": convert_numpy_types(baseline_per_class_f1),
            "baseline_per_class_precision": convert_numpy_types(baseline_per_class_precision),
            "baseline_per_class_recall": convert_numpy_types(baseline_per_class_recall),
            "baseline_per_class_support": convert_numpy_types(baseline_per_class_support),
            "ensemble_per_class_acc": convert_numpy_types(ensemble_per_class_acc),
            "ensemble_per_class_f1": convert_numpy_types(ensemble_per_class_f1),
            "ensemble_per_class_precision": convert_numpy_types(ensemble_per_class_precision),
            "ensemble_per_class_recall": convert_numpy_types(ensemble_per_class_recall),
            "ensemble_per_class_support": convert_numpy_types(ensemble_per_class_support)
        }

        # 결과는 로그 파일에만 저장 (터미널 출력 제거)
        
        # 상세 결과를 로그 파일에 저장
        self.save_detailed_results_log(results, exp_dir)
        logger.info(f"Experiment completed. Results saved to: {exp_dir}")

        return results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Simple Frequency-Based Expert Assignment Experiment - CICIDS-2017 Version with XGBoost Experts"
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "random", "both"],
        default="both",
        help="Experiment mode: fixed seed, random seeds, or both",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Fixed seed for reproducible experiments"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials for random seed experiments",
    )
    parser.add_argument(
        "--num_experts", type=int, default=4, help="Number of expert networks"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./MachineLearningCVE",
        help="Directory containing CICIDS-2017 CSV files",
    )
    parser.add_argument(
        "--models", 
        nargs='+', 
        choices=['baseline', 'ensemble'], 
        default=['baseline', 'ensemble'],
        help="Models to train and evaluate (default: all models)"
    )

    args = parser.parse_args()

    # Initialize experiment
    experiment = SimpleExperiment(
        num_experts=args.num_experts
    )

    if args.mode == "fixed":
        result = experiment.run_single_experiment(args.seed, args.data_dir, args.models)

        # Save single result
        with open(f"simple_experiment_xgboost_fixed_seed{args.seed}.json", "w") as f:
            json.dump(result, f, indent=2)

    elif args.mode == "random":
        random_seeds = [random.randint(1, 10000) for _ in range(args.trials)]
        # Note: run_multiple_experiments would need to be implemented similar to the original

    elif args.mode == "both":
        # Fixed seed experiment
        fixed_result = experiment.run_single_experiment(args.seed, args.data_dir, args.models)

        # Save combined results
        combined_results = {
            "fixed_seed_result": fixed_result,
            "timestamp": datetime.now().isoformat(),
        }

        with open(
            f"simple_experiment_xgboost_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w",
        ) as f:
            json.dump(combined_results, f, indent=2)


if __name__ == "__main__":
    main()

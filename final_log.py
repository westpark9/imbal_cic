#!/usr/bin/env python3
"""
Final Comprehensive Experiment: Clustering-Based Expert Ensemble vs Baseline
CICIDS-2017 Network Intrusion Detection Dataset Version

Usage:
    python final3.py --mode fixed --seed 42 --epochs 50        # Fixed seed with 50 epochs
    python final3.py --mode random --trials 3 --epochs 100     # Random seeds with 100 epochs
    python final3.py --mode both --seed 42 --trials 3 --epochs 75 # Both modes with 75 epochs
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import xgboost as xgb

# Windows multiprocessing 문제 해결
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")


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


class SharedBackbone(nn.Module):
    """Shared backbone network for feature extraction - MLP version for tabular data"""

    def __init__(self, input_dim, feature_dim=256):
        super(SharedBackbone, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x):
        return self.feature_extractor(x)


class ExpertClassifier(nn.Module):
    """Individual expert classifier for a subset of classes"""

    def __init__(self, feature_dim, num_classes):
        super(ExpertClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)


class FallbackRouter(nn.Module):
    """Fallback router using soft routing for robust ensemble"""

    def __init__(self, num_experts=4, feature_dim=256):
        super(FallbackRouter, self).__init__()
        self.num_experts = num_experts
        self.feature_dim = feature_dim

        self.router = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, features):
        routing_weights = self.router(features)
        return routing_weights


class BaselineModel(nn.Module):
    """Simple baseline model for comparison - MLP version for tabular data"""

    def __init__(self, input_dim, num_classes):
        super(BaselineModel, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class FinalExperiment:
    """
    Final comprehensive experiment class supporting both fixed and random seed modes
    CICIDS-2017 Network Intrusion Detection Dataset Version
    """

    def __init__(self, num_experts=4):
        self.num_experts = num_experts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Final Clustering-Based Expert Ensemble Experiment")
        print(f"   Dataset: CICIDS-2017 Network Intrusion Detection")
        print(f"   Number of experts: {num_experts}")
        print(f"   Using original data distribution (no artificial balancing)")
        print(f"   Device: {self.device}")

    def set_random_seed(self, seed):
        """Set all random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        # print(f"     Available CSV files in {data_dir}: {all_csv_files}")
        
        for csv_file in all_csv_files:
            if csv_file not in data_files:
                print(f"     Adding missing file: {csv_file}")
                data_files.append(csv_file)
        
        all_data = []
        
        for file_name in data_files:
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                # print(f"     Loading {file_name}...")
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
                            # print(f"       Classes in {file_name}: {list(label_counts.index)}")
                            # print(f"       Sample counts: {dict(label_counts)}")
                        
                        all_data.append(df)
                        # print(f"       Loaded {len(df)} samples (including all classes)")
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
        X_df = combined_df[feature_columns].copy()  # DataFrame으로 유지
        y = combined_df['Label_encoded'].values
        
        # 특성 전처리 및 스케일링
        print(f"     Feature preprocessing and scaling...")
        
        # 1-1. 중복 feature 제거 ('Fwd Header Length' 중복)
        if 'Fwd Header Length' in X_df.columns:
            X_df = X_df.drop(columns=['Fwd Header Length'])
        
        # 1-2. 특정 열의 -1 값을 0으로 처리 (KMeans는 NaN 허용 안함)
        win_bytes_cols = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward']
        for col in win_bytes_cols:
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(-1, 0)
        
        # 1-3. 각 feature별 통계 분석 및 스케일링
        X_scaled = X_df.copy()
        log_scaled_features = []
        standard_scaled_features = []
        
        # 플래그/카운트 특성들 (StandardScaler 사용) 
        flag_features = [
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
            'Down/Up Ratio', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
            'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
        ]
        
        actual_flag_features = []
        for flag in flag_features:
            for col in X_df.columns:
                if col.strip() == flag.strip(): 
                    actual_flag_features.append(col)
                    break

        
        for col in X_df.columns:
            if X_df[col].dtype in ['int64', 'float64']:
                valid_data = X_df[col].dropna()
                if len(valid_data) > 0:
                    mean_val = valid_data.mean()
                    std_val = valid_data.std()
                    
                    # print(f"       - {col}: mean={mean_val:.2f}, std={std_val:.2f}")
                    
                    col_data = X_df[col].fillna(0)
                    max_val = col_data.max()
                    min_val = col_data.min()
                    
                    # 1. 플래그/카운트 특성은 StandardScaler 사용
                    if col in actual_flag_features:
                        scaler = StandardScaler()
                        X_scaled[col] = scaler.fit_transform(X_df[[col]].fillna(0))
                        standard_scaled_features.append(col)
                        # print(f"       - {col}: Using standard scaling (flag/count feature)")
                    else:
                        # 2. 나머지 특성은 로그 스케일링 (음수값 처리 포함)
                        if min_val < 0:
                            # 음수값이 있으면 shift 후 로그 스케일링
                            shift_value = abs(min_val) + 1
                            shifted_data = col_data + shift_value
                            test_log = np.log1p(shifted_data)
                            inf_count = np.isinf(test_log).sum()
                            nan_count = np.isnan(test_log).sum()
                            
                            if inf_count == 0 and nan_count == 0:
                                X_scaled[col] = test_log
                                log_scaled_features.append(col)
                                print(f"       - {col}: Using log scaling with shift (shift={shift_value:.2e})")
                            else:
                                # inf/nan 발생하면 StandardScaler 사용
                                scaler = StandardScaler()
                                X_scaled[col] = scaler.fit_transform(X_df[[col]].fillna(0))
                                standard_scaled_features.append(col)
                                print(f"       - {col}: Using standard scaling (log with shift caused {inf_count} inf, {nan_count} nan)")
                        else:
                            # 음수값이 없으면 직접 로그 스케일링
                            test_log = np.log1p(col_data)
                            inf_count = np.isinf(test_log).sum()
                            nan_count = np.isnan(test_log).sum()
                            
                            if inf_count == 0 and nan_count == 0:
                                X_scaled[col] = test_log
                                log_scaled_features.append(col)
                                print(f"       - {col}: Using log scaling (no shift)")
                            else:
                                # inf/nan 발생하면 StandardScaler 사용
                                scaler = StandardScaler()
                                X_scaled[col] = scaler.fit_transform(X_df[[col]].fillna(0))
                                standard_scaled_features.append(col)
                                print(f"       - {col}: Using standard scaling (log caused {inf_count} inf, {nan_count} nan)")
        
        # 최종 NaN 검증 및 처리 (KMeans 호환성)
        nan_count = X_scaled.isnull().sum().sum()
        if nan_count > 0:
            print(f"       WARNING: Found {nan_count} NaN values after scaling, replacing with 0")
            X_scaled = X_scaled.fillna(0)
        else:
            print(f"       - No NaN values found after scaling")
        
        print(f"     Scaling summary:")
        print(f"       - Log-scaled features ({len(log_scaled_features)}): {log_scaled_features[:3]}{'...' if len(log_scaled_features) > 3 else ''}")
        print(f"       - Standard-scaled features ({len(standard_scaled_features)}): {standard_scaled_features[:3]}{'...' if len(standard_scaled_features) > 3 else ''}")
        
        # 최종 결과
        X_final = X_scaled.values
        
        # 클래스별 샘플 수 확인
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"     Classes: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            class_name = label_encoder.inverse_transform([label])[0]
            print(f"       {class_name}: {count}")
        
        return X_final, y, label_encoder, scaler, feature_columns

    def setup_dataset(self, seed, data_dir):
        """CICIDS-2017 데이터셋 설정 및 불균형 데이터 생성"""
        self.set_random_seed(seed)
        
        # 데이터 로드
        X, y, self.label_encoder, self.scaler, self.feature_columns = self.load_cicids2017_data(data_dir)
        
        # 훈련/테스트 분할 (7:3) - 원본 불균형 분포 유지
        print(f"     Splitting data into train/test (7:3 ratio)...")
        
        # 일반적인 train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        
        # 클래스별 분포 확인
        print(f"     Class distribution verification:")
        unique_classes = np.unique(y)
        total_train = len(X_train)
        total_test = len(X_test)
        
        # for class_id in unique_classes:
        #     train_count = np.sum(y_train == class_id)
        #     test_count = np.sum(y_test == class_id)
        #     total_count = train_count + test_count
            
        #     train_ratio = train_count / total_count if total_count > 0 else 0
        #     test_ratio = test_count / total_count if total_count > 0 else 0
            
        #     print(f"       - Class {class_id}: Train={train_count} ({train_ratio:.1%}), Test={test_count} ({test_ratio:.1%})")
        
        # print(f"     Final split: Train={total_train} samples, Test={total_test} samples")
        # print(f"     Overall ratio: Train={total_train/(total_train+total_test):.1%}, Test={total_test/(total_train+total_test):.1%}")
        
        # 원본 데이터 그대로 사용 (불균형 데이터셋 생성 제거)
        self.train_subset = CICIDS2017Dataset(X_train, y_train)
        self.test_dataset = CICIDS2017Dataset(X_test, y_test)
        
        # 클래스 수와 전체 데이터 저장 (clustering에서 사용)
        self.num_classes = len(np.unique(y))
        self.y_all = y  # 전체 데이터의 레이블 저장
        
        print(f"   Dataset: {len(self.train_subset)} train, {len(self.test_dataset)} test")
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {self.num_classes}")

    # _create_imbalanced_dataset 함수 제거 - 원본 데이터 그대로 사용

    def get_clustering_groups(self, seed):
        """클러스터링 기반 전문가 그룹 생성"""
        print("   Executing 6-step clustering pipeline...")

        # Step 1: Backbone warmup for feature extraction
        self.set_random_seed(seed + 10)
        
        input_dim = len(self.feature_columns)
        warmup_model = BaselineModel(input_dim, self.num_classes).to(self.device)
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=True, num_workers=0, persistent_workers=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            warmup_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )

        warmup_model.train()
        print("     Warming up backbone for feature extraction...")
        
        # Warmup 에포크에 대한 프로그레스 바 (final_cifar10.py와 일치)
        warmup_pbar = tqdm(range(15), desc="Backbone Warmup", unit="epoch")
        
        for epoch in warmup_pbar:
            total_loss = 0.0
            num_batches = 0
            
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = warmup_model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            warmup_pbar.set_postfix_str(f"Loss: {avg_loss:.4f}")
        
        warmup_pbar.close()

        # Step 2: Extract class embeddings
        feature_extractor = SharedBackbone(input_dim, feature_dim=256).to(self.device)
        
        # Load warmed-up weights (첫 번째 레이어만)
        feature_extractor.feature_extractor[0].weight.data = warmup_model.classifier[0].weight.data
        feature_extractor.feature_extractor[0].bias.data = warmup_model.classifier[0].bias.data

        feature_extractor.eval()

        # Collect class features
        class_features = [[] for _ in range(self.num_classes)]
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
        )

        with torch.no_grad():
            for data, targets in train_loader:
                data = data.to(self.device)
                features = feature_extractor(data).cpu().numpy()
                
                # inf 값 체크 및 처리
                if np.isinf(features).any():
                    print(f"       WARNING: inf values found in features, replacing with zeros")
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                for i, target in enumerate(targets):
                    class_features[target.item()].append(features[i])

        # Calculate class centroids
        class_centroids = []
        for class_id in range(self.num_classes):
            if len(class_features[class_id]) > 0:
                centroid = np.mean(class_features[class_id], axis=0)
                # inf/NaN 체크 및 처리
                if np.isnan(centroid).any() or np.isinf(centroid).any():
                    print(f"       WARNING: inf/NaN in class {class_id} centroid, replacing with zeros")
                    centroid = np.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0)
                class_centroids.append(centroid)
            else:
                centroid = np.random.normal(0, 0.1, 256)
                class_centroids.append(centroid)

        class_centroids = np.array(class_centroids)
        print(f"       DEBUG: class_centroids shape: {class_centroids.shape}, NaN count: {np.isnan(class_centroids).sum()}")

        # Step 3: Clustering with [frequency + embeddings]
        # 원본 데이터의 클래스별 샘플 수 계산
        unique_labels, counts = np.unique(self.y_all, return_counts=True)
        samples_array = np.array(counts)
        print(f"       DEBUG: samples_array: {samples_array}")
        print(f"       DEBUG: samples_array mean: {np.mean(samples_array)}, std: {np.std(samples_array)}")
        
        frequency_zscore = (samples_array - np.mean(samples_array)) / np.std(samples_array)
        print(f"       DEBUG: frequency_zscore: {frequency_zscore}, NaN count: {np.isnan(frequency_zscore).sum()}")

        frequency_weight = 10
        frequency_features = np.repeat(
            frequency_zscore.reshape(-1, 1), frequency_weight, axis=1
        )
        print(f"       DEBUG: frequency_features shape: {frequency_features.shape}, NaN count: {np.isnan(frequency_features).sum()}")

        combined_features = np.concatenate(
            [frequency_features, class_centroids], axis=1
        )
        print(f"       DEBUG: combined_features shape: {combined_features.shape}, NaN count: {np.isnan(combined_features).sum()}")

        scaler = StandardScaler()
        combined_features_scaled = scaler.fit_transform(combined_features)
        print(f"       DEBUG: combined_features_scaled shape: {combined_features_scaled.shape}, NaN count: {np.isnan(combined_features_scaled).sum()}")
        
        # KMeans NaN 검증
        if np.isnan(combined_features_scaled).any():
            print(f"       WARNING: NaN found in clustering features, replacing with zeros")
            combined_features_scaled = np.nan_to_num(combined_features_scaled, nan=0.0)

        kmeans = KMeans(n_clusters=self.num_experts, random_state=seed + 200, n_init=20)
        cluster_labels = kmeans.fit_predict(combined_features_scaled)

        # Group classes by cluster
        clustered_groups = [[] for _ in range(self.num_experts)]
        for class_id, cluster_id in enumerate(cluster_labels):
            clustered_groups[cluster_id].append(class_id)

        # Print per-expert class membership and sample counts
        try:
            # class_id -> sample count in training set (from stored train subset)
            train_labels = self.train_subset.labels.cpu().numpy()
            class_ids, class_counts = np.unique(train_labels, return_counts=True)
            class_id_to_count = {int(cid): int(cnt) for cid, cnt in zip(class_ids, class_counts)}
        except Exception:
            class_id_to_count = {}

        # Optional: class id to name mapping if label_encoder is available
        class_id_to_name = None
        if hasattr(self, 'label_encoder') and getattr(self, 'label_encoder') is not None:
            try:
                class_id_to_name = {idx: name for idx, name in enumerate(self.label_encoder.classes_)}
            except Exception:
                class_id_to_name = None

        print("       Expert clusters composition:")
        for expert_idx, class_list in enumerate(clustered_groups):
            # Build (class_id, count, name)
            details = []
            total_samples = 0
            for cid in class_list:
                cnt = class_id_to_count.get(int(cid), 0)
                total_samples += cnt
                name = class_id_to_name.get(int(cid), str(cid)) if class_id_to_name else str(cid)
                details.append((cid, cnt, name))
            # Sort by count desc
            details.sort(key=lambda x: x[1], reverse=True)

            print(f"         - Expert {expert_idx}: {len(class_list)} classes, {total_samples} samples")
            for cid, cnt, name in details:
                print(f"             · class {cid} ({name}): {cnt}")

        return clustered_groups

    def train_baseline(self, seed, epochs=100):
        """베이스라인 모델 학습"""
        print("   Training baseline model...")

        self.set_random_seed(seed)

        input_dim = len(self.feature_columns)
        model = BaselineModel(input_dim, self.num_classes).to(self.device)
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=True, num_workers=0, persistent_workers=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 전체 에포크에 대한 프로그레스 바
        epoch_pbar = tqdm(range(epochs), desc="Training Baseline", unit="epoch")
        
        for epoch in epoch_pbar:
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            # 에포크별 진행 상황 업데이트 (시간 정보만)
            avg_loss = total_loss / num_batches
            epoch_pbar.set_postfix_str(f"Loss: {avg_loss:.4f}")
        
        epoch_pbar.close()
        return model

    def train_expert_ensemble(self, expert_groups, seed, epochs=100):
        """클러스터링 기반 전문가 앙상블 학습"""
        print("   Training clustering-based expert ensemble...")

        self.set_random_seed(seed + 300)

        # Initialize models
        input_dim = len(self.feature_columns)
        shared_backbone = SharedBackbone(input_dim, feature_dim=256).to(self.device)
        expert_classifiers = []
        for group in expert_groups:
            expert_classifier = ExpertClassifier(
                feature_dim=256, num_classes=len(group)
            ).to(self.device)
            expert_classifiers.append(expert_classifier)

        router = FallbackRouter(num_experts=self.num_experts, feature_dim=256).to(
            self.device
        )

        # Combine all parameters
        all_params = list(shared_backbone.parameters())
        for expert_classifier in expert_classifiers:
            all_params.extend(list(expert_classifier.parameters()))
        all_params.extend(list(router.parameters()))

        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=True, num_workers=0, persistent_workers=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(all_params, lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 전체 에포크에 대한 프로그레스 바
        epoch_pbar = tqdm(range(epochs), desc="Training Ensemble", unit="epoch")
        
        for epoch in epoch_pbar:
            shared_backbone.train()
            router.train()
            for expert_classifier in expert_classifiers:
                expert_classifier.train()

            total_loss = 0.0
            total_cls_loss = 0.0
            total_router_loss = 0.0
            num_batches = 0
            
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                features = shared_backbone(data)
                routing_weights = router(features)

                # Expert assignments
                expert_assignments = torch.zeros(
                    targets.size(0), dtype=torch.long, device=self.device
                )
                for i, target in enumerate(targets):
                    for expert_idx, expert_classes in enumerate(expert_groups):
                        if target.item() in expert_classes:
                            expert_assignments[i] = expert_idx
                            break

                # Expert predictions
                final_logits = torch.zeros(data.size(0), self.num_classes, device=self.device)

                for expert_idx, (expert_classifier, expert_classes) in enumerate(
                    zip(expert_classifiers, expert_groups)
                ):
                    expert_pred = expert_classifier(features)

                    for i, global_class in enumerate(expert_classes):
                        final_logits[:, global_class] += (
                            routing_weights[:, expert_idx] * expert_pred[:, i]
                        )

                # Loss computation
                classification_loss = criterion(final_logits, targets)
                router_loss = criterion(routing_weights, expert_assignments)

                total_loss_val = classification_loss + 0.1 * router_loss
                total_loss_val.backward()
                optimizer.step()
                
                total_loss += total_loss_val.item()
                total_cls_loss += classification_loss.item()
                total_router_loss += router_loss.item()
                num_batches += 1

            scheduler.step()
            
            # 에포크별 진행 상황 업데이트 (시간 정보만)
            avg_total_loss = total_loss / num_batches
            epoch_pbar.set_postfix_str(f"Loss: {avg_total_loss:.4f}")
        
        epoch_pbar.close()
        return shared_backbone, expert_classifiers, router

    def train_xgboost(self, seed, epochs=100):
        """Train XGBoost model for comparison"""
        if xgb is None:
            print("   XGBoost not available. Skipping XGBoost training.")
            return None
            
        print("   Training XGBoost model...")
        
        self.set_random_seed(seed + 400)  # Offset seed for XGBoost
        
        # Get training data
        train_loader = DataLoader(
            self.train_subset, batch_size=len(self.train_subset), shuffle=False, num_workers=0
        )
        
        # Extract data
        for data, targets in train_loader:
            X_train = data.numpy()
            y_train = targets.numpy()
            break
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': epochs,  # epochs 파라미터 반영
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': seed + 400,
            'n_jobs': 1,  # Windows multiprocessing issue 해결
            'verbosity': 0
        }
        
        # Train XGBoost with progress bar
        print("     Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**xgb_params)
        
        # Progress bar를 위한 callback
        from tqdm import tqdm
        import time
        
        # 간단한 progress bar 시뮬레이션
        with tqdm(total=100, desc="Training XGBoost", unit="%") as pbar:
            xgb_model.fit(X_train, y_train)
            pbar.update(100)

        return xgb_model

    def evaluate_xgboost(self, xgb_model):
        """XGBoost 모델 평가 및 클래스별 성능 지표 계산"""
        if xgb_model is None:
            # XGBoost가 없을 때 더미 결과 반환
            num_classes = len(self.label_encoder.classes_)
            return 0.0, [0.0] * num_classes, [0.0] * num_classes, np.zeros((num_classes, num_classes))
            
        print("     Evaluating XGBoost model...")
        
        test_loader = DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=0
        )
        
        # Extract test data
        for data, targets in test_loader:
            X_test = data.numpy()
            y_test = targets.numpy()
            break
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        
        # 전체 정확도
        accuracy = 100.0 * (y_pred == y_test).mean()
        
        # 클래스별 성능 지표
        num_classes = len(np.unique(y_test))
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            y_pred, y_test, num_classes
        )
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        
        return accuracy, per_class_acc, per_class_f1, cm

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
        plt.figure(figsize=(15, 6))
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width/2, baseline_acc, width, label='Baseline')
        plt.bar(x + width/2, ensemble_acc, width, label='Ensemble')
        plt.title('Per-Class Accuracy')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.bar(x - width/2, baseline_f1, width, label='Baseline')
        plt.bar(x + width/2, ensemble_f1, width, label='Ensemble')
        plt.title('Per-Class F1 Score')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save_performance_metrics(self, baseline_acc, baseline_f1, ensemble_acc, ensemble_f1, class_names, save_path):
        """성능 지표를 CSV 파일로 저장합니다."""
        df = pd.DataFrame({
            'Class': class_names,
            'Baseline_Accuracy': baseline_acc,
            'Baseline_F1': baseline_f1,
            'Ensemble_Accuracy': ensemble_acc,
            'Ensemble_F1': ensemble_f1
        })
        df.to_csv(save_path, index=False)

    def plot_per_class_metrics_three_models(self, baseline_acc, baseline_f1, ensemble_acc, ensemble_f1, xgb_acc, xgb_f1, class_names, save_path):
        """3개 모델의 클래스별 성능 지표를 시각화합니다."""
        plt.figure(figsize=(20, 8))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width, baseline_acc, width, label='Baseline MLP', alpha=0.8)
        plt.bar(x, ensemble_acc, width, label='Expert Ensemble', alpha=0.8)
        plt.bar(x + width, xgb_acc, width, label='XGBoost', alpha=0.8)
        plt.title('Per-Class Accuracy Comparison', fontsize=14)
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(x - width, baseline_f1, width, label='Baseline MLP', alpha=0.8)
        plt.bar(x, ensemble_f1, width, label='Expert Ensemble', alpha=0.8)
        plt.bar(x + width, xgb_f1, width, label='XGBoost', alpha=0.8)
        plt.title('Per-Class F1 Score Comparison', fontsize=14)
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_performance_metrics_three_models(self, baseline_acc, baseline_f1, ensemble_acc, ensemble_f1, xgb_acc, xgb_f1, class_names, save_path):
        """3개 모델의 성능 지표를 CSV 파일로 저장합니다."""
        df = pd.DataFrame({
            'Class': class_names,
            'Baseline_MLP_Accuracy': baseline_acc,
            'Baseline_MLP_F1': baseline_f1,
            'Expert_Ensemble_Accuracy': ensemble_acc,
            'Expert_Ensemble_F1': ensemble_f1,
            'XGBoost_Accuracy': xgb_acc,
            'XGBoost_F1': xgb_f1
        })
        df.to_csv(save_path, index=False)

    def plot_training_curves(self, training_history, save_path, title_suffix=""):
        """훈련 과정의 accuracy와 F1 score 곡선을 그립니다."""
        if not training_history:
            return
            
        # Accuracy 곡선
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        for model_name, history in training_history.items():
            if 'accuracy' in history:
                plt.plot(history['epochs'], history['accuracy'], 
                        label=f'{model_name} Accuracy', marker='o', markersize=3)
        plt.title(f'Training Accuracy Curves{title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1 Score 곡선
        plt.subplot(1, 2, 2)
        for model_name, history in training_history.items():
            if 'f1_score' in history:
                plt.plot(history['epochs'], history['f1_score'], 
                        label=f'{model_name} F1 Score', marker='s', markersize=3)
        plt.title(f'Training F1 Score Curves{title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_expert_performance(self, expert_history, save_path):
        """각 expert의 성능을 CSV로 저장합니다."""
        if not expert_history:
            return
            
        # 모든 expert의 데이터를 하나의 DataFrame으로 결합
        all_data = []
        for expert_idx, history in expert_history.items():
            for epoch, acc, f1 in zip(history['epochs'], history['accuracy'], history['f1_score']):
                all_data.append({
                    'Expert': expert_idx,
                    'Epoch': epoch,
                    'Accuracy': acc,
                    'F1_Score': f1
                })
        
        df = pd.DataFrame(all_data)
        df.to_csv(save_path, index=False)

    def evaluate_model(self, model):
        """단일 모델 평가 및 클래스별 성능 지표 계산"""
        test_loader = DataLoader(
            self.test_dataset, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
        )

        all_predictions = []
        all_targets = []

        model.eval()
        print("     Evaluating baseline model...")
        
        with torch.no_grad():
            # 평가 진행 상황을 보여주는 프로그레스 바
            eval_pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
            
            for data, targets in eval_pbar:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 진행 상황 업데이트
                eval_pbar.set_postfix({
                    'Samples': len(all_predictions)
                })
        
        eval_pbar.close()

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 전체 정확도
        accuracy = 100.0 * (all_predictions == all_targets).mean()
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            all_predictions, all_targets, self.num_classes
        )
        
        # 혼동 행렬
        cm = confusion_matrix(all_targets, all_predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm

    def evaluate_expert_ensemble(
        self, shared_backbone, expert_classifiers, router, expert_groups
    ):
        """전문가 앙상블 평가 및 클래스별 성능 지표 계산"""
        test_loader = DataLoader(
            self.test_dataset, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
        )

        all_predictions = []
        all_targets = []

        shared_backbone.eval()
        router.eval()
        for expert_classifier in expert_classifiers:
            expert_classifier.eval()

        print("     Evaluating expert ensemble...")
        
        with torch.no_grad():
            # 평가 진행 상황을 보여주는 프로그레스 바
            eval_pbar = tqdm(test_loader, desc="Evaluating Ensemble", unit="batch")
            
            for data, targets in eval_pbar:
                data, targets = data.to(self.device), targets.to(self.device)

                features = shared_backbone(data)
                routing_weights = router(features)

                final_logits = torch.zeros(data.size(0), self.num_classes, device=self.device)

                for expert_idx, (expert_classifier, expert_classes) in enumerate(
                    zip(expert_classifiers, expert_groups)
                ):
                    expert_pred = expert_classifier(features)

                    for i, global_class in enumerate(expert_classes):
                        final_logits[:, global_class] += (
                            routing_weights[:, expert_idx] * expert_pred[:, i]
                        )

                _, predicted = final_logits.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        eval_pbar.close()

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 전체 정확도
        accuracy = 100.0 * (all_predictions == all_targets).mean()
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            all_predictions, all_targets, self.num_classes
        )
        
        # 혼동 행렬
        cm = confusion_matrix(all_targets, all_predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm

    def run_single_experiment(self, seed, data_dir, epochs=100, models=['baseline', 'ensemble', 'xgboost']):
        """단일 실험 실행 및 결과 저장"""
        print(f"\nExperiment with seed {seed}")
        print("=" * 50)

        start_time = time.time()

        # 실험 디렉토리 생성
        exp_dir = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)

        # 데이터셋 설정
        self.setup_dataset(seed, data_dir)

        # 모델별 학습 및 평가 (선택적)
        baseline_model = None
        baseline_acc = 0.0
        baseline_per_class_acc = []
        baseline_per_class_f1 = []
        baseline_cm = None
        
        shared_backbone = None
        expert_classifiers = None
        router = None
        clustering_groups = None
        ensemble_acc = 0.0
        ensemble_per_class_acc = []
        ensemble_per_class_f1 = []
        ensemble_cm = None
        
        xgb_model = None
        xgb_acc = 0.0
        xgb_per_class_acc = []
        xgb_per_class_f1 = []
        xgb_cm = None

        # 베이스라인 모델
        if 'baseline' in models:
            baseline_model = self.train_baseline(seed, epochs)
            baseline_acc, baseline_per_class_acc, baseline_per_class_f1, baseline_cm = self.evaluate_model(baseline_model)

        # 앙상블 모델
        if 'ensemble' in models:
            clustering_groups = self.get_clustering_groups(seed)
            shared_backbone, expert_classifiers, router = self.train_expert_ensemble(
                clustering_groups, seed, epochs
            )
            ensemble_acc, ensemble_per_class_acc, ensemble_per_class_f1, ensemble_cm = self.evaluate_expert_ensemble(
                shared_backbone, expert_classifiers, router, clustering_groups
            )

        # XGBoost 모델
        if 'xgboost' in models:
            xgb_model = self.train_xgboost(seed, epochs)
            xgb_acc, xgb_per_class_acc, xgb_per_class_f1, xgb_cm = self.evaluate_xgboost(xgb_model)

        elapsed_time = (time.time() - start_time) / 60
        improvement_ensemble = ensemble_acc - baseline_acc
        improvement_xgb = xgb_acc - baseline_acc

        # 결과 시각화 및 저장
        class_names = self.label_encoder.classes_
        
        # 혼동 행렬 시각화 (3개 모델) - 유효한 모델만
        if 'baseline' in models and baseline_cm is not None:
            self.plot_confusion_matrix(
                baseline_cm, class_names,
                "Baseline MLP Confusion Matrix",
                os.path.join(exp_dir, "baseline_confusion_matrix.png")
            )
        if 'ensemble' in models and ensemble_cm is not None:
            self.plot_confusion_matrix(
                ensemble_cm, class_names,
                "Expert Ensemble Confusion Matrix",
                os.path.join(exp_dir, "ensemble_confusion_matrix.png")
            )
        if 'xgboost' in models and xgb_cm is not None:
            self.plot_confusion_matrix(
                xgb_cm, class_names,
                "XGBoost Confusion Matrix",
                os.path.join(exp_dir, "xgboost_confusion_matrix.png")
            )
        
        # 클래스별 성능 지표 시각화 (선택된 모델들만)
        if len(models) > 1:  # 2개 이상의 모델이 있을 때만 비교 차트 생성
            self.plot_per_class_metrics_three_models(
                baseline_per_class_acc, baseline_per_class_f1,
                ensemble_per_class_acc, ensemble_per_class_f1,
                xgb_per_class_acc, xgb_per_class_f1,
                class_names,
                os.path.join(exp_dir, "per_class_metrics_three_models.png")
            )
            
            # 성능 지표 CSV 저장 (3개 모델)
            self.save_performance_metrics_three_models(
                baseline_per_class_acc, baseline_per_class_f1,
                ensemble_per_class_acc, ensemble_per_class_f1,
                xgb_per_class_acc, xgb_per_class_f1,
                class_names,
                os.path.join(exp_dir, "performance_metrics_three_models.csv")
            )
        else:  # 단일 모델인 경우
            if 'baseline' in models:
                self.plot_per_class_metrics(
                    baseline_per_class_acc, baseline_per_class_f1,
                    baseline_per_class_acc, baseline_per_class_f1,  # 동일한 값으로 단일 모델 표시
                    class_names,
                    os.path.join(exp_dir, "per_class_metrics_baseline.png")
                )
            elif 'ensemble' in models:
                self.plot_per_class_metrics(
                    ensemble_per_class_acc, ensemble_per_class_f1,
                    ensemble_per_class_acc, ensemble_per_class_f1,
                    class_names,
                    os.path.join(exp_dir, "per_class_metrics_ensemble.png")
                )
            elif 'xgboost' in models:
                self.plot_per_class_metrics(
                    xgb_per_class_acc, xgb_per_class_f1,
                    xgb_per_class_acc, xgb_per_class_f1,
                    class_names,
                    os.path.join(exp_dir, "per_class_metrics_xgboost.png")
                )

        results = {
            "seed": seed,
            "baseline_accuracy": baseline_acc,
            "ensemble_accuracy": ensemble_acc,
            "xgboost_accuracy": xgb_acc,
            "improvement_ensemble": improvement_ensemble,
            "improvement_xgb": improvement_xgb,
            "runtime_minutes": elapsed_time,
            "clustering_groups": clustering_groups,
            "timestamp": datetime.now().isoformat(),
            "baseline_per_class_acc": baseline_per_class_acc,
            "baseline_per_class_f1": baseline_per_class_f1,
            "ensemble_per_class_acc": ensemble_per_class_acc,
            "ensemble_per_class_f1": ensemble_per_class_f1,
            "xgboost_per_class_acc": xgb_per_class_acc,
            "xgboost_per_class_f1": xgb_per_class_f1
        }

        print(f"Results:")
        if 'baseline' in models:
            print(f"   Baseline MLP:                 {baseline_acc:.2f}%")
        if 'ensemble' in models:
            print(f"   Expert Ensemble:              {ensemble_acc:.2f}%")
        if 'xgboost' in models:
            print(f"   XGBoost:                      {xgb_acc:.2f}%")
        
        if 'baseline' in models and 'ensemble' in models:
            print(f"   Ensemble Improvement:         {improvement_ensemble:+.2f}%p")
        if 'baseline' in models and 'xgboost' in models:
            print(f"   XGBoost Improvement:          {improvement_xgb:+.2f}%p")
            
        print(f"   Runtime:                      {elapsed_time:.1f} minutes")
        print(f"   Results saved to:             {exp_dir}")

        return results

    def run_multiple_experiments(self, seeds, data_dir, epochs=100, models=['baseline', 'ensemble', 'xgboost']):
        """Run multiple experiments with different seeds"""
        print(f"\nRunning {len(seeds)} experiments with seeds: {seeds}")
        print("=" * 80)

        all_results = []

        for i, seed in enumerate(seeds):
            print(f"\nTrial {i+1}/{len(seeds)}")
            result = self.run_single_experiment(seed, data_dir, epochs, models)
            all_results.append(result)

        # Calculate statistics
        baseline_accs = [r["baseline_accuracy"] for r in all_results]
        ensemble_accs = [r["ensemble_accuracy"] for r in all_results]
        xgb_accs = [r["xgboost_accuracy"] for r in all_results]
        improvements_ensemble = [r["improvement_ensemble"] for r in all_results]
        improvements_xgb = [r["improvement_xgb"] for r in all_results]

        baseline_mean = np.mean(baseline_accs)
        baseline_std = np.std(baseline_accs)
        ensemble_mean = np.mean(ensemble_accs)
        ensemble_std = np.std(ensemble_accs)
        xgb_mean = np.mean(xgb_accs)
        xgb_std = np.std(xgb_accs)
        improvement_ensemble_mean = np.mean(improvements_ensemble)
        improvement_ensemble_std = np.std(improvements_ensemble)
        improvement_xgb_mean = np.mean(improvements_xgb)
        improvement_xgb_std = np.std(improvements_xgb)

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SUMMARY ({len(seeds)} trials)")
        print(f"{'='*80}")
        print(f"Baseline MLP:                 {baseline_mean:.2f}% ± {baseline_std:.2f}%")
        print(f"Expert Ensemble:              {ensemble_mean:.2f}% ± {ensemble_std:.2f}%")
        print(f"XGBoost:                      {xgb_mean:.2f}% ± {xgb_std:.2f}%")
        print(f"Ensemble Improvement:         {improvement_ensemble_mean:+.2f}%p ± {improvement_ensemble_std:.2f}%p")
        print(f"XGBoost Improvement:          {improvement_xgb_mean:+.2f}%p ± {improvement_xgb_std:.2f}%p")

        # Statistical significance check
        print(f"\nStatistical Significance:")
        if improvement_ensemble_mean > 2 * improvement_ensemble_std and improvement_ensemble_mean > 0:
            print(f"Ensemble: SIGNIFICANT IMPROVEMENT (>2σ confidence)")
        elif improvement_ensemble_mean > 0:
            print(f"Ensemble: POSITIVE IMPROVEMENT")
        else:
            print(f"Ensemble: NO SIGNIFICANT IMPROVEMENT")
            
        if improvement_xgb_mean > 2 * improvement_xgb_std and improvement_xgb_mean > 0:
            print(f"XGBoost: SIGNIFICANT IMPROVEMENT (>2σ confidence)")
        elif improvement_xgb_mean > 0:
            print(f"XGBoost: POSITIVE IMPROVEMENT")
        else:
            print(f"XGBoost: NO SIGNIFICANT IMPROVEMENT")

        # Save detailed results
        summary_results = {
            "experiment_type": "multiple_trials",
            "num_trials": len(seeds),
            "seeds": seeds,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "ensemble_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "xgboost_mean": xgb_mean,
            "xgboost_std": xgb_std,
            "improvement_ensemble_mean": improvement_ensemble_mean,
            "improvement_ensemble_std": improvement_ensemble_std,
            "improvement_xgb_mean": improvement_xgb_mean,
            "improvement_xgb_std": improvement_xgb_std,
            "individual_results": all_results,
            "timestamp": datetime.now().isoformat(),
        }

        results_file = (
            f"final_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(summary_results, f, indent=2)

        print(f"\nDetailed results saved to: {results_file}")

        return summary_results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Final Clustering-Based Expert Ensemble Experiment - CICIDS-2017 Version"
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
        "--epochs",
        type=int,
        default=50,  # 기본값을 50으로 변경
        help="Number of training epochs for models",
    )
    parser.add_argument(
        "--models", 
        nargs='+', 
        choices=['baseline', 'ensemble', 'xgboost'], 
        default=['baseline', 'ensemble', 'xgboost'],
        help="Models to train and evaluate (default: all models)"
    )

    args = parser.parse_args()

    # Initialize experiment
    experiment = FinalExperiment(
        num_experts=args.num_experts
    )

    if args.mode == "fixed":
        print(f"\nRunning FIXED SEED experiment (seed={args.seed})")
        result = experiment.run_single_experiment(args.seed, args.data_dir, args.epochs, args.models)

        # Save single result
        with open(f"final_experiment_fixed_seed{args.seed}.json", "w") as f:
            json.dump(result, f, indent=2)

    elif args.mode == "random":
        print(f"\nRunning RANDOM SEED experiments ({args.trials} trials)")
        random_seeds = [random.randint(1, 10000) for _ in range(args.trials)]
        summary = experiment.run_multiple_experiments(random_seeds, args.data_dir, args.epochs, args.models)

    elif args.mode == "both":
        print(f"\nRunning BOTH modes:")
        print(f"1. Fixed seed experiment (seed={args.seed})")
        print(f"2. Random seed experiments ({args.trials} trials)")

        # Fixed seed experiment
        print(f"\n" + "=" * 80)
        print(f"FIXED SEED EXPERIMENT")
        print(f"=" * 80)
        fixed_result = experiment.run_single_experiment(args.seed, args.data_dir, args.epochs, args.models)

        # Random seed experiments
        print(f"\n" + "=" * 80)
        print(f"RANDOM SEED EXPERIMENTS")
        print(f"=" * 80)
        random_seeds = [random.randint(1, 10000) for _ in range(args.trials)]
        random_summary = experiment.run_multiple_experiments(random_seeds, args.data_dir, args.epochs, args.models)

        # Compare results
        print(f"\n" + "=" * 80)
        print(f"COMPARISON: FIXED vs RANDOM SEEDS")
        print(f"=" * 80)
        print(f"Fixed seed - Ensemble:       {fixed_result['improvement_ensemble']:+.2f}%p")
        print(f"Fixed seed - XGBoost:        {fixed_result['improvement_xgb']:+.2f}%p")
        print(f"Random seeds - Ensemble:     {random_summary['improvement_ensemble_mean']:+.2f}%p ± {random_summary['improvement_ensemble_std']:.2f}%p")
        print(f"Random seeds - XGBoost:      {random_summary['improvement_xgb_mean']:+.2f}%p ± {random_summary['improvement_xgb_std']:.2f}%p")

        # Save combined results
        combined_results = {
            "fixed_seed_result": fixed_result,
            "random_seed_summary": random_summary,
            "timestamp": datetime.now().isoformat(),
        }

        with open(
            f"final_experiment_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w",
        ) as f:
            json.dump(combined_results, f, indent=2)


if __name__ == "__main__":
    main()


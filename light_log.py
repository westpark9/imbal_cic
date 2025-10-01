

###########################

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
from sklearn.decomposition import PCA
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
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
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
        
        # 최종 NaN 검증 및 처리 (KMeans 호환성)
        nan_count = X_scaled.isnull().sum().sum()
        if nan_count > 0:
            print(f"       WARNING: Found {nan_count} NaN values after scaling, replacing with 0")
            X_scaled = X_scaled.fillna(0)
        else:
            print(f"       - No NaN values found after scaling")
        
        # inf 값 처리
        inf_count = np.isinf(X_scaled).sum().sum()
        if inf_count > 0:
            print(f"       WARNING: Found {inf_count} inf values, replacing with 0")
            X_scaled = X_scaled.replace([np.inf, -np.inf], 0)

        # 최종 결과
        X_final = X_scaled.values
        
        # 클래스별 샘플 수 확인
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"     Classes: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            class_name = label_encoder.inverse_transform([label])[0]
            print(f"       {class_name}: {count}")
        
        return X_final, y, label_encoder, None, feature_columns

    def setup_dataset(self, seed, data_dir):
        """CICIDS-2017 데이터셋 설정 및 불균형 데이터 생성"""
        self.set_random_seed(seed)
        
        # 데이터 로드
        X, y, self.label_encoder, _, self.feature_columns = self.load_cicids2017_data(data_dir)
        
        # 클래스 수 설정
        self.num_classes = len(self.label_encoder.classes_)
        print(f"     Number of classes: {self.num_classes}")
        
        # 훈련/테스트 분할 (7:3) - 원본 불균형 분포 유지
        print(f"     Splitting data into train/test (7:3 ratio)...")
        
        # 일반적인 train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
     
        
        # 데이터셋 객체 생성
        self.train_subset = CICIDS2017Dataset(X_train, y_train)
        self.test_subset = CICIDS2017Dataset(X_test, y_test)
        
        print(f"   Dataset: {len(self.train_subset)} train, {len(self.test_subset)} test")
       

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
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = warmup_model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
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

        # Step 3: Clustering with [frequency (from TRAIN) + class embeddings]
        train_labels = self.train_subset.labels.cpu().numpy()
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        samples_array = counts.astype(np.float32)
        std_sa = np.std(samples_array)
        if std_sa == 0:
            frequency_zscore = np.zeros_like(samples_array)
        else:
            frequency_zscore = (samples_array - np.mean(samples_array)) / std_sa

        # Option 1: Frequency만 사용 (가장 확실한 방법)
        use_frequency_only = False
        
        if use_frequency_only:
            # Frequency-only 모드: 더 강력한 변환으로 극단값 완화
            
            # 세제곱근 변환 (큰 값들의 차이를 더 크게 줄임)
            cube_root_samples = np.power(samples_array, 1/3)
            
            # Min-Max 정규화 (0-1 범위)
            min_val = cube_root_samples.min()
            max_val = cube_root_samples.max()
            normalized_samples = (cube_root_samples - min_val) / (max_val - min_val)
            
            # 0-1 범위를 0-10 범위로 확장 (KMeans가 구분하기 쉽도록)
            scaled_features = normalized_samples * 10.0
            
            frequency_features = np.repeat(scaled_features.reshape(-1, 1), 20, axis=1)
            combined_features = frequency_features
            
            print(f"       Using frequency-only clustering with cube-root scaling (20 dimensions)")
            print(f"       Original samples range: [{samples_array.min():.0f}, {samples_array.max():.0f}]")
            print(f"       Cube-root samples range: [{cube_root_samples.min():.3f}, {cube_root_samples.max():.3f}]")
            print(f"       Normalized range: [{normalized_samples.min():.3f}, {normalized_samples.max():.3f}]")
            print(f"       Final scaled range: [{scaled_features.min():.3f}, {scaled_features.max():.3f}]")
            
            # 각 클래스의 변환된 값 출력 (디버깅용)
            print(f"       Transformed values per class:")
            for i, (class_id, count) in enumerate(zip(unique_labels, counts)):
                class_name = self.label_encoder.inverse_transform([class_id])[0]
                print(f"         Class {class_id} ({class_name}): {count} -> {scaled_features[i]:.3f}")
        else:
            # Option 2: Frequency + class centroids (기본 모드)
            frequency_weight = 50
            frequency_features = np.repeat(
                frequency_zscore.reshape(-1, 1), frequency_weight, axis=1
            )
            combined_features = np.concatenate([frequency_features, class_centroids], axis=1)
            print(f"       Using frequency + class centroids clustering ({frequency_weight} freq dims + {class_centroids.shape[1]} feature dims)")
        
        print(f"       Combined features shape: {combined_features.shape}")
        if not use_frequency_only:
            print(f"       Frequency weight applied: {frequency_weight}")
            print(f"       Frequency range: [{frequency_features.min():.3f}, {frequency_features.max():.3f}]")
            print(f"       Class centroids range: [{class_centroids.min():.3f}, {class_centroids.max():.3f}]")
        else:
            print(f"       Final frequency range: [{frequency_features.min():.3f}, {frequency_features.max():.3f}]")
        
        # standard scaler 생략
        
        # KMeans NaN 검증
        if np.isnan(combined_features).any():
            print(f"       WARNING: NaN found in clustering features, replacing with zeros")
            combined_features = np.nan_to_num(combined_features, nan=0.0)

        kmeans = KMeans(n_clusters=self.num_experts, random_state=seed + 200, n_init=20)
        cluster_labels = kmeans.fit_predict(combined_features)

        # Group classes by cluster
        clustered_groups = [[] for _ in range(self.num_experts)]
        for class_id, cluster_id in enumerate(cluster_labels):
            clustered_groups[cluster_id].append(class_id)
        

        # Print per-expert class membership and sample counts
        train_labels = self.train_subset.labels.cpu().numpy()
        class_ids, class_counts = np.unique(train_labels, return_counts=True)
        class_id_to_count = {int(cid): int(cnt) for cid, cnt in zip(class_ids, class_counts)}

        print("       Expert clusters composition:")
        for expert_idx, class_list in enumerate(clustered_groups):
            total_samples = sum(class_id_to_count.get(int(cid), 0) for cid in class_list)
            print(f"         - Expert {expert_idx}: {len(class_list)} classes, {total_samples} samples")
            if len(class_list) == 0:
                print(f"             WARNING: Expert {expert_idx} has no classes assigned!")
            else:
                for cid in sorted(class_list, key=lambda x: class_id_to_count.get(int(x), 0), reverse=True):
                    cnt = class_id_to_count.get(int(cid), 0)
                    class_name = self.label_encoder.inverse_transform([cid])[0]
                    print(f"             · class {cid} ({class_name}): {cnt}")
        
        # 클러스터링 결과 검증
        empty_experts = [i for i, group in enumerate(clustered_groups) if len(group) == 0]
        if empty_experts:
            print(f"       WARNING: Experts {empty_experts} have no classes assigned!")
            print(f"       This will cause these experts to always return 0 accuracy.")


        return clustered_groups

    def _evaluate_model_epoch(self, model):
        """단일 모델의 epoch별 성능을 평가합니다."""
        test_loader = DataLoader(
            self.test_subset, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
        )

        all_predictions = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 정확도 계산
        accuracy = (all_predictions == all_targets).mean()
        
        # F1 점수 계산
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return accuracy, f1

    def _evaluate_ensemble_epoch(self, shared_backbone, expert_classifiers, router, expert_groups):
        """앙상블 모델의 epoch별 성능을 평가합니다."""
        test_loader = DataLoader(
            self.test_subset, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
        )

        all_predictions = []
        all_targets = []

        shared_backbone.eval()
        router.eval()
        for expert_classifier in expert_classifiers:
            expert_classifier.eval()

        with torch.no_grad():
            for data, targets in test_loader:
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

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 정확도 계산
        accuracy = (all_predictions == all_targets).mean()
        
        # F1 점수 계산
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return accuracy, f1

    def _evaluate_expert_epoch(self, shared_backbone, expert_classifier, expert_classes):
        """개별 expert의 epoch별 성능을 평가합니다."""
        test_loader = DataLoader(
            self.test_subset, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
        )

        all_predictions = []
        all_targets = []
        expert_mask = []

        shared_backbone.eval()
        expert_classifier.eval()

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                features = shared_backbone(data)
                expert_pred = expert_classifier(features)
                _, predicted = expert_pred.max(1)
                
                # 해당 expert가 담당하는 클래스에 속하는 샘플만 평가
                for i, target in enumerate(targets):
                    if target.item() in expert_classes:
                        # Expert 내부 클래스 인덱스로 변환
                        local_class_idx = expert_classes.index(target.item())
                        all_predictions.append(predicted[i].cpu().numpy())
                        all_targets.append(local_class_idx)
                        expert_mask.append(True)
                    else:
                        expert_mask.append(False)

        if len(all_predictions) == 0:
            return 0.0, 0.0

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 정확도 계산
        accuracy = (all_predictions == all_targets).mean()
        
        # F1 점수 계산
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return accuracy, f1

    def _evaluate_xgboost_epoch(self, xgb_model):
        """XGBoost 모델의 epoch별 성능을 평가합니다."""
        test_loader = DataLoader(
            self.test_subset, batch_size=len(self.test_subset), shuffle=False, num_workers=0
        )
        
        # Extract test data
        for data, targets in test_loader:
            X_test = data.numpy()
            y_test = targets.numpy()
            break
        
        # 예측
        y_pred = xgb_model.predict(X_test)
        
        # 정확도 계산
        accuracy = (y_pred == y_test).mean()
        
        # F1 점수 계산
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return accuracy, f1

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

        # 훈련 과정 추적을 위한 변수들
        training_history = {
            'epochs': [],
            'accuracy': [],
            'f1_score': []
        }

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
            
            # 5에폭마다 성능 평가
            if (epoch + 1) % 5 == 0 or epoch == 0:
                acc, f1 = self._evaluate_model_epoch(model)
                training_history['epochs'].append(epoch + 1)
                training_history['accuracy'].append(acc)
                training_history['f1_score'].append(f1)
            
            # 에포크별 진행 상황 업데이트
            avg_loss = total_loss / num_batches
            epoch_pbar.set_postfix_str(f"Loss: {avg_loss:.4f}")
        
        epoch_pbar.close()
        return model, training_history

    def train_expert_ensemble(self, expert_groups, seed, epochs=100):
        """클러스터링 기반 전문가 앙상블 학습"""
        print("   Training clustering-based expert ensemble...")

        self.set_random_seed(seed + 300)

        # Initialize models
        input_dim = len(self.feature_columns)
        shared_backbone = SharedBackbone(input_dim, feature_dim=256).to(self.device)
        expert_classifiers = []
        for expert_idx, group in enumerate(expert_groups):
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

        # 훈련 과정 추적을 위한 변수들
        training_history = {
            'epochs': [],
            'accuracy': [],
            'f1_score': []
        }
        
        # 각 expert별 성능 추적
        expert_history = {}
        for expert_idx in range(len(expert_classifiers)):
            expert_history[expert_idx] = {
                'epochs': [],
                'accuracy': [],
                'f1_score': []
            }

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

                # Router weight 디버깅 (첫 번째 배치의 처음 5개 샘플만)
                if num_batches == 0:  # 첫 번째 배치에서만 출력
                    print(f"       DEBUG Router Training: routing_weights shape: {routing_weights.shape}")
                    print(f"       DEBUG Router Training: routing_weights (first 5 samples):")
                    for i in range(min(5, routing_weights.size(0))):
                        weights = routing_weights[i].detach().cpu().numpy()
                        target_class = targets[i].item()
                        class_name = self.label_encoder.inverse_transform([target_class])[0]
                        max_expert = np.argmax(weights)
                        
                        # 해당 클래스가 실제로 할당된 Expert 찾기
                        assigned_expert = None
                        for expert_idx, expert_classes in enumerate(expert_groups):
                            if target_class in expert_classes:
                                assigned_expert = expert_idx
                                break
                        
                        print(f"         Sample {i}: Class {target_class} ({class_name}) -> Router: Expert {max_expert}, Assigned: Expert {assigned_expert} (weights: {weights})")
                    print(f"       DEBUG Router Training: routing_weights range: {routing_weights.min().item():.4f}-{routing_weights.max().item():.4f}")

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
            
            # 5에폭마다 성능 평가
            if (epoch + 1) % 5 == 0 or epoch == 0:
                # 전체 앙상블 성능 평가
                acc, f1 = self._evaluate_ensemble_epoch(shared_backbone, expert_classifiers, router, expert_groups)
                training_history['epochs'].append(epoch + 1)
                training_history['accuracy'].append(acc)
                training_history['f1_score'].append(f1)
                
                # 각 expert별 성능 평가
                for expert_idx in range(len(expert_classifiers)):
                    expert_acc, expert_f1 = self._evaluate_expert_epoch(shared_backbone, expert_classifiers[expert_idx], expert_groups[expert_idx])
                    expert_history[expert_idx]['epochs'].append(epoch + 1)
                    expert_history[expert_idx]['accuracy'].append(expert_acc)
                    expert_history[expert_idx]['f1_score'].append(expert_f1)
            
            # 에포크별 진행 상황 업데이트
            avg_total_loss = total_loss / num_batches
            epoch_pbar.set_postfix_str(f"Loss: {avg_total_loss:.4f}")
        
        epoch_pbar.close()
        return shared_backbone, expert_classifiers, router, training_history, expert_history

    def train_xgboost(self, seed, epochs=100):
        """Train XGBoost model for comparison"""
        if xgb is None:
            print("   XGBoost not available. Skipping XGBoost training.")
            return None, None
            
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
        
        # 훈련 과정 추적을 위한 변수들
        training_history = {
            'epochs': [],
            'accuracy': [],
            'f1_score': []
        }
        
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

        # 5에폭마다 성능 평가 (XGBoost는 내부적으로 n_estimators를 사용하므로 시뮬레이션)
        for epoch in range(0, epochs, 5):
            if epoch == 0:
                # 초기 모델로 평가
                temp_model = xgb.XGBClassifier(**{k: v for k, v in xgb_params.items() if k != 'n_estimators'})
                temp_model.n_estimators = 1
                temp_model.fit(X_train, y_train)
            else:
                # 부분적으로 훈련된 모델로 평가
                temp_model = xgb.XGBClassifier(**{k: v for k, v in xgb_params.items() if k != 'n_estimators'})
                temp_model.n_estimators = epoch
                temp_model.fit(X_train, y_train)
            
            acc, f1 = self._evaluate_xgboost_epoch(temp_model)
            training_history['epochs'].append(epoch + 1)
            training_history['accuracy'].append(acc)
            training_history['f1_score'].append(f1)

        return xgb_model, training_history

    def evaluate_xgboost(self, xgb_model):
        """XGBoost 모델 평가 및 클래스별 성능 지표 계산"""
        if xgb_model is None:
            # XGBoost가 없을 때 더미 결과 반환
            num_classes = len(self.label_encoder.classes_)
            return 0.0, [0.0] * num_classes, [0.0] * num_classes, np.zeros((num_classes, num_classes))
            
        print("     Evaluating XGBoost model...")
        
        test_loader = DataLoader(
            self.test_subset, batch_size=len(self.test_subset), shuffle=False, num_workers=0
        )
        
        # Extract test data
        for data, targets in test_loader:
            X_test = data.numpy()
            y_test = targets.numpy()
            break
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        
        # 전체 정확도
        accuracy = (y_pred == y_test).mean()
        
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

    def plot_per_class_metrics(self, baseline_acc, baseline_f1, ensemble_acc, ensemble_f1, xgb_acc, xgb_f1, class_names, save_path):
        """클래스별 성능 지표를 시각화합니다."""
        plt.figure(figsize=(20, 8))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.subplot(1, 2, 1)
        bar_positions = []
        labels = []
        if baseline_acc is not None and len(baseline_acc) > 0:
            plt.bar(x - width, baseline_acc, width, label='Baseline MLP', alpha=0.8)
            labels.append('Baseline MLP')
        if ensemble_acc is not None and len(ensemble_acc) > 0:
            plt.bar(x, ensemble_acc, width, label='Expert Ensemble', alpha=0.8)
            labels.append('Expert Ensemble')
        if xgb_acc is not None and len(xgb_acc) > 0:
            plt.bar(x + width, xgb_acc, width, label='XGBoost', alpha=0.8)
            labels.append('XGBoost')
        plt.title('Per-Class Accuracy')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.ylim(0.0, 1.1)
        if labels:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if baseline_f1 is not None and len(baseline_f1) > 0:
            plt.bar(x - width, baseline_f1, width, label='Baseline MLP', alpha=0.8)
        if ensemble_f1 is not None and len(ensemble_f1) > 0:
            plt.bar(x, ensemble_f1, width, label='Expert Ensemble', alpha=0.8)
        if xgb_f1 is not None and len(xgb_f1) > 0:
            plt.bar(x + width, xgb_f1, width, label='XGBoost', alpha=0.8)
        plt.title('Per-Class F1 Score')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.ylim(0.0, 1.1)
        if labels:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

    def plot_training_curves(self, training_history, save_path, title_suffix=""):
        """훈련 과정의 accuracy와 F1 score 곡선을 그립니다."""
        if not training_history:
            return
            
        # Accuracy 곡선
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        for model_name, history in training_history.items():
            if 'accuracy' in history and history['accuracy']:  # XGBoost 결과 확인
                plt.plot(history['epochs'], history['accuracy'], 
                        label=f'{model_name} Accuracy', marker='o', markersize=3)
        plt.title(f'Training Accuracy Curves{title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0.7, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1 Score 곡선
        plt.subplot(1, 2, 2)
        for model_name, history in training_history.items():
            if 'f1_score' in history and history['f1_score']:  # XGBoost 결과 확인
                plt.plot(history['epochs'], history['f1_score'], 
                        label=f'{model_name} F1 Score', marker='s', markersize=3)
        plt.title(f'Training F1 Score Curves{title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.ylim(0.7, 1.1)
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
        
        # 에폭별로 그룹화하여 저장
        all_epochs = set()
        for expert_idx, history in expert_history.items():
            all_epochs.update(history['epochs'])
        
        for epoch in sorted(all_epochs):
            for expert_idx, history in expert_history.items():
                if epoch in history['epochs']:
                    epoch_idx = history['epochs'].index(epoch)
                    acc = history['accuracy'][epoch_idx]
                    f1 = history['f1_score'][epoch_idx]
                    all_data.append({
                        'Expert': expert_idx,
                        'Epoch': epoch,
                        'Accuracy': acc,
                        'F1_Score': f1
                    })
        
        df = pd.DataFrame(all_data)
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
        # 빈 배열인 경우 None으로 변환
        baseline_acc = baseline_acc if baseline_acc and len(baseline_acc) > 0 else None
        baseline_f1 = baseline_f1 if baseline_f1 and len(baseline_f1) > 0 else None
        ensemble_acc = ensemble_acc if ensemble_acc and len(ensemble_acc) > 0 else None
        ensemble_f1 = ensemble_f1 if ensemble_f1 and len(ensemble_f1) > 0 else None
        xgb_acc = xgb_acc if xgb_acc and len(xgb_acc) > 0 else None
        xgb_f1 = xgb_f1 if xgb_f1 and len(xgb_f1) > 0 else None
        
        # 데이터 딕셔너리 생성
        data_dict = {'Class': class_names}
        
        if baseline_acc is not None:
            data_dict['Baseline_MLP_Accuracy'] = baseline_acc
            data_dict['Baseline_MLP_F1'] = baseline_f1
        if ensemble_acc is not None:
            data_dict['Expert_Ensemble_Accuracy'] = ensemble_acc
            data_dict['Expert_Ensemble_F1'] = ensemble_f1
        if xgb_acc is not None:
            data_dict['XGBoost_Accuracy'] = xgb_acc
            data_dict['XGBoost_F1'] = xgb_f1
        
        df = pd.DataFrame(data_dict)
        df.to_csv(save_path, index=False)

    def evaluate_model(self, model):
        """단일 모델 평가 및 클래스별 성능 지표 계산"""
        test_loader = DataLoader(
            self.test_subset, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
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
        accuracy = (all_predictions == all_targets).mean()
        
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
            self.test_subset, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
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
            
            batch_count = 0
            for data, targets in eval_pbar:
                data, targets = data.to(self.device), targets.to(self.device)

                features = shared_backbone(data)
                routing_weights = router(features)

                # Router weight 디버깅 (첫 번째 배치의 처음 5개 샘플만)
                if batch_count == 0:  # 첫 번째 배치에서만 출력
                    print(f"       DEBUG Router Evaluation: routing_weights shape: {routing_weights.shape}")
                    print(f"       DEBUG Router Evaluation: routing_weights (first 5 samples):")
                    for i in range(min(5, routing_weights.size(0))):
                        weights = routing_weights[i].detach().cpu().numpy()
                        target_class = targets[i].item()
                        class_name = self.label_encoder.inverse_transform([target_class])[0]
                        max_expert = np.argmax(weights)
                        
                        # 해당 클래스가 실제로 할당된 Expert 찾기
                        assigned_expert = None
                        for expert_idx, expert_classes in enumerate(expert_groups):
                            if target_class in expert_classes:
                                assigned_expert = expert_idx
                                break
                        
                        print(f"         Sample {i}: Class {target_class} ({class_name}) -> Router: Expert {max_expert}, Assigned: Expert {assigned_expert} (weights: {weights})")
                    print(f"       DEBUG Router Evaluation: routing_weights range: {routing_weights.min().item():.4f}-{routing_weights.max().item():.4f}")
                    
                    # 클래스별 Expert 할당 통계
                    print(f"       DEBUG Router Evaluation: Class-to-Expert mapping (first batch):")
                    class_expert_mapping = {}
                    for i in range(routing_weights.size(0)):
                        target_class = targets[i].item()
                        max_expert = np.argmax(routing_weights[i].detach().cpu().numpy())
                        if target_class not in class_expert_mapping:
                            class_expert_mapping[target_class] = []
                        class_expert_mapping[target_class].append(max_expert)
                    
                    for class_id, expert_assignments in class_expert_mapping.items():
                        class_name = self.label_encoder.inverse_transform([class_id])[0]
                        expert_counts = {expert: expert_assignments.count(expert) for expert in range(4)}
                        print(f"         Class {class_id} ({class_name}): {expert_counts}")

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
                
                batch_count += 1
        
        eval_pbar.close()

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 전체 정확도
        accuracy = (all_predictions == all_targets).mean()
        
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

        # 훈련 과정 추적을 위한 변수들
        training_history = {}
        expert_history = {}

        # 베이스라인 모델
        if 'baseline' in models:
            baseline_model, baseline_history = self.train_baseline(seed, epochs)
            baseline_acc, baseline_per_class_acc, baseline_per_class_f1, baseline_cm = self.evaluate_model(baseline_model)
            training_history['Baseline'] = baseline_history

        # 앙상블 모델
        if 'ensemble' in models:
            clustering_groups = self.get_clustering_groups(seed)
            shared_backbone, expert_classifiers, router, ensemble_history, expert_history = self.train_expert_ensemble(
                clustering_groups, seed, epochs
            )
            ensemble_acc, ensemble_per_class_acc, ensemble_per_class_f1, ensemble_cm = self.evaluate_expert_ensemble(
                shared_backbone, expert_classifiers, router, clustering_groups
            )
            training_history['Ensemble'] = ensemble_history

        # XGBoost 모델
        if 'xgboost' in models:
            xgb_model, xgb_history = self.train_xgboost(seed, epochs)
            xgb_acc, xgb_per_class_acc, xgb_per_class_f1, xgb_cm = self.evaluate_xgboost(xgb_model)
            training_history['XGBoost'] = xgb_history

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
        
        # 훈련 과정 시각화
        if training_history:
            self.plot_training_curves(
                training_history, 
                os.path.join(exp_dir, "training_curves.png"),
                f" (Seed {seed})"
            )

        # Expert 성능 저장 (앙상블 모델이 있을 때만) - 훈련 과정에서만 사용
        if expert_history:
            self.save_expert_performance(
                expert_history,
                os.path.join(exp_dir, "expert_performance.csv")
            )

        # 클래스별 성능 지표 시각화 (실행된 모델들만 표시)
        self.plot_per_class_metrics(
            baseline_per_class_acc if 'baseline' in models else None,
            baseline_per_class_f1 if 'baseline' in models else None,
            ensemble_per_class_acc if 'ensemble' in models else None,
            ensemble_per_class_f1 if 'ensemble' in models else None,
            xgb_per_class_acc if 'xgboost' in models else None,
            xgb_per_class_f1 if 'xgboost' in models else None,
            class_names,
            os.path.join(exp_dir, "per_class_metrics.png")
        )
        
        # 성능 지표 CSV 저장
        self.save_performance_metrics_three_models(
            baseline_per_class_acc if 'baseline' in models else None,
            baseline_per_class_f1 if 'baseline' in models else None,
            ensemble_per_class_acc if 'ensemble' in models else None,
            ensemble_per_class_f1 if 'ensemble' in models else None,
            xgb_per_class_acc if 'xgboost' in models else None,
            xgb_per_class_f1 if 'xgboost' in models else None,
            class_names,
            os.path.join(exp_dir, "performance_metrics.csv")
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
            print(f"   Baseline MLP:                 {baseline_acc:.4f}")
        if 'ensemble' in models:
            print(f"   Expert Ensemble:              {ensemble_acc:.4f}")
        if 'xgboost' in models:
            print(f"   XGBoost:                      {xgb_acc:.4f}")
        
        if 'baseline' in models and 'ensemble' in models:
            print(f"   Ensemble Improvement:         {improvement_ensemble:+.4f}")
        if 'baseline' in models and 'xgboost' in models:
            print(f"   XGBoost Improvement:          {improvement_xgb:+.4f}")
            
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



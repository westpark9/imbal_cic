#!/usr/bin/env python3
"""
Final Comprehensive Experiment: Clustering-Based Expert Ensemble vs Baseline
CICIDS-2017 Network Intrusion Detection Dataset Version
############################
Usage:
    python final_experiment.py --mode fixed --seed 42           # Fixed seed for reproducibility
    python final_experiment.py --mode random --trials 3        # Random seeds for robust evaluation
    python final_experiment.py --mode both --seed 42 --trials 3 # Both modes
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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

    def __init__(self, num_experts=4, imbalance_factor=100):
        self.num_experts = num_experts
        self.imbalance_factor = imbalance_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Final Clustering-Based Expert Ensemble Experiment")
        print(f"   Dataset: CICIDS-2017 Network Intrusion Detection")
        print(f"   Number of experts: {num_experts}")
        print(f"   Imbalance factor: {imbalance_factor}")
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
        """CICIDS-2017 데이터를 로드하고 전처리합니다"""
        print("   Loading CICIDS-2017 dataset...")
        
        # 데이터 파일 목록
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
        
        all_data = []
        
        for file_name in data_files:
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                print(f"     Loading {file_name}...")
                try:
                    # CSV 파일 로드
                    df = pd.read_csv(file_path)
                    
              
                    
                    # 'Label' 컬럼 찾기 (공백 제거 후 비교)
                    label_column = None
                    for col in df.columns:
                        if col.strip() == 'Label':
                            label_column = col
                            break
                    
                    if label_column is not None:
                        
                        # 정상 트래픽과 공격 트래픽 분리
                        df[label_column] = df[label_column].str.strip()
                        df = df[df[label_column] != 'BENIGN']  # 정상 트래픽 제외
                        
                        # NaN 값 처리
                        df = df.dropna()
                        
                        # 무한대 값 처리
                        df = df.replace([np.inf, -np.inf], np.nan)
                        df = df.dropna()
                        
                        all_data.append(df)
                        print(f"       Loaded {len(df)} samples")
                    else:
                        print(f"       Warning: {file_name} has no 'Label' column")
                        print(f"       Available columns: {list(df.columns)}")
                        
                except Exception as e:
                    print(f"     Error loading {file_name}: {e}")
            else:
                print(f"     Warning: {file_path} not found")
        
        if not all_data:
            raise ValueError("No valid data files found!")
        
        # 모든 데이터 합치기
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"     Total samples: {len(combined_df)}")
        
        # 레이블 컬럼 찾기 (첫 번째 데이터프레임에서)
        label_column = None
        for col in combined_df.columns:
            if col.strip() == 'Label':
                label_column = col
                break
        
        if label_column is None:
            raise ValueError("Label column not found in combined data!")
        
        # 레이블 인코딩
        label_encoder = LabelEncoder()
        combined_df['Label_encoded'] = label_encoder.fit_transform(combined_df[label_column])
        
        # 특성과 레이블 분리
        feature_columns = [col for col in combined_df.columns if col not in [label_column, 'Label_encoded']]
        X = combined_df[feature_columns].values
        y = combined_df['Label_encoded'].values
        
        # 특성 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 클래스별 샘플 수 확인
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"     Classes: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            class_name = label_encoder.inverse_transform([label])[0]
            print(f"       {class_name}: {count}")
        
        return X_scaled, y, label_encoder, scaler, feature_columns

    def setup_dataset(self, seed, data_dir):
        """CICIDS-2017 데이터셋 설정 및 불균형 데이터 생성"""
        self.set_random_seed(seed)
        
        # 데이터 로드
        X, y, self.label_encoder, self.scaler, self.feature_columns = self.load_cicids2017_data(data_dir)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # 불균형 데이터셋 생성
        X_train_imbalanced, y_train_imbalanced, self.samples_per_class = self._create_imbalanced_dataset(
            X_train, y_train
        )
        
        # 데이터셋 생성
        self.train_subset = CICIDS2017Dataset(X_train_imbalanced, y_train_imbalanced)
        self.test_dataset = CICIDS2017Dataset(X_test, y_test)
        
        # 클래스 수 저장
        self.num_classes = len(np.unique(y))
        
        print(f"   Dataset: {len(self.train_subset)} train, {len(self.test_dataset)} test")
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {self.num_classes}")

    def _create_imbalanced_dataset(self, X, y):
        """불균형 데이터셋 생성 (long-tail distribution)"""
        unique_labels, counts = np.unique(y, return_counts=True)
        
        max_samples = np.max(counts)
        min_samples = max_samples // self.imbalance_factor
        
        samples_per_class = []
        selected_indices = []
        
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            # long-tail distribution 적용
            ratio = i / (len(unique_labels) - 1)
            num_samples = int(max_samples * (min_samples / max_samples) ** ratio)
            num_samples = max(num_samples, min_samples)
            
            # 해당 클래스의 인덱스 찾기
            label_indices = np.where(y == label)[0]
            
            if len(label_indices) >= num_samples:
                selected = np.random.choice(label_indices, num_samples, replace=False)
            else:
                selected = label_indices
            
            selected_indices.extend(selected)
            samples_per_class.append(num_samples)
        
        return X[selected_indices], y[selected_indices], samples_per_class

    def get_clustering_groups(self, seed):
        """클러스터링 기반 전문가 그룹 생성"""
        print("   Executing 6-step clustering pipeline...")

        # Step 1: Backbone warmup for feature extraction
        self.set_random_seed(seed + 10)
        
        input_dim = len(self.feature_columns)
        warmup_model = BaselineModel(input_dim, self.num_classes).to(self.device)
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=True, num_workers=0
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(warmup_model.parameters(), lr=0.001, weight_decay=1e-4)

        warmup_model.train()
        print("     Warming up backbone for feature extraction...")
        
        # Warmup 에포크에 대한 프로그레스 바
        warmup_pbar = tqdm(range(10), desc="Backbone Warmup", unit="epoch")
        
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
            self.train_subset, batch_size=128, shuffle=False, num_workers=0
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
                class_centroids.append(centroid)
            else:
                centroid = np.random.normal(0, 0.1, 256)
                class_centroids.append(centroid)

        class_centroids = np.array(class_centroids)

        # Step 3: Clustering with [frequency + embeddings]
        samples_array = np.array(self.samples_per_class)
        frequency_zscore = (samples_array - np.mean(samples_array)) / np.std(samples_array)

        frequency_weight = 10
        frequency_features = np.repeat(
            frequency_zscore.reshape(-1, 1), frequency_weight, axis=1
        )

        combined_features = np.concatenate(
            [frequency_features, class_centroids], axis=1
        )

        scaler = StandardScaler()
        combined_features_scaled = scaler.fit_transform(combined_features)

        kmeans = KMeans(n_clusters=self.num_experts, random_state=seed + 200, n_init=20)
        cluster_labels = kmeans.fit_predict(combined_features_scaled)

        # Group classes by cluster
        clustered_groups = [[] for _ in range(self.num_experts)]
        for class_id, cluster_id in enumerate(cluster_labels):
            clustered_groups[cluster_id].append(class_id)

        return clustered_groups

    def train_baseline(self, seed, epochs=100):
        """베이스라인 모델 학습"""
        print("   Training baseline model...")

        self.set_random_seed(seed)

        input_dim = len(self.feature_columns)
        model = BaselineModel(input_dim, self.num_classes).to(self.device)
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=True, num_workers=0
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
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
            self.train_subset, batch_size=128, shuffle=True, num_workers=0
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(all_params, lr=0.001, weight_decay=1e-4)
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

    def evaluate_model(self, model):
        """단일 모델 평가 및 클래스별 성능 지표 계산"""
        test_loader = DataLoader(
            self.test_dataset, batch_size=128, shuffle=False, num_workers=0
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
            self.test_dataset, batch_size=128, shuffle=False, num_workers=0
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

    def run_single_experiment(self, seed, data_dir):
        """단일 실험 실행 및 결과 저장"""
        print(f"\nExperiment with seed {seed}")
        print("=" * 50)

        start_time = time.time()

        # 실험 디렉토리 생성
        exp_dir = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)

        # 데이터셋 설정
        self.setup_dataset(seed, data_dir)

        # 베이스라인 모델 학습 및 평가
        baseline_model = self.train_baseline(seed)
        baseline_acc, baseline_per_class_acc, baseline_per_class_f1, baseline_cm = self.evaluate_model(baseline_model)

        # 클러스터링 그룹 생성 및 앙상블 학습
        clustering_groups = self.get_clustering_groups(seed)
        shared_backbone, expert_classifiers, router = self.train_expert_ensemble(
            clustering_groups, seed
        )
        ensemble_acc, ensemble_per_class_acc, ensemble_per_class_f1, ensemble_cm = self.evaluate_expert_ensemble(
            shared_backbone, expert_classifiers, router, clustering_groups
        )

        elapsed_time = (time.time() - start_time) / 60
        improvement = ensemble_acc - baseline_acc

        # 결과 시각화 및 저장
        class_names = self.label_encoder.classes_
        
        # 혼동 행렬 시각화
        self.plot_confusion_matrix(
            baseline_cm, class_names,
            "Baseline Model Confusion Matrix",
            os.path.join(exp_dir, "baseline_confusion_matrix.png")
        )
        self.plot_confusion_matrix(
            ensemble_cm, class_names,
            "Ensemble Model Confusion Matrix",
            os.path.join(exp_dir, "ensemble_confusion_matrix.png")
        )
        
        # 클래스별 성능 지표 시각화
        self.plot_per_class_metrics(
            baseline_per_class_acc, baseline_per_class_f1,
            ensemble_per_class_acc, ensemble_per_class_f1,
            class_names,
            os.path.join(exp_dir, "per_class_metrics.png")
        )
        
        # 성능 지표 CSV 저장
        self.save_performance_metrics(
            baseline_per_class_acc, baseline_per_class_f1,
            ensemble_per_class_acc, ensemble_per_class_f1,
            class_names,
            os.path.join(exp_dir, "performance_metrics.csv")
        )

        results = {
            "seed": seed,
            "baseline_accuracy": baseline_acc,
            "ensemble_accuracy": ensemble_acc,
            "improvement": improvement,
            "runtime_minutes": elapsed_time,
            "clustering_groups": clustering_groups,
            "timestamp": datetime.now().isoformat(),
            "baseline_per_class_acc": baseline_per_class_acc,
            "baseline_per_class_f1": baseline_per_class_f1,
            "ensemble_per_class_acc": ensemble_per_class_acc,
            "ensemble_per_class_f1": ensemble_per_class_f1
        }

        print(f"Results:")
        print(f"   Baseline:                     {baseline_acc:.2f}%")
        print(f"   Clustering-Based Ensemble:    {ensemble_acc:.2f}%")
        print(f"   Improvement:                  {improvement:+.2f}%p")
        print(f"   Runtime:                      {elapsed_time:.1f} minutes")
        print(f"   Results saved to:             {exp_dir}")

        return results

    def run_multiple_experiments(self, seeds, data_dir):
        """Run multiple experiments with different seeds"""
        print(f"\nRunning {len(seeds)} experiments with seeds: {seeds}")
        print("=" * 80)

        all_results = []

        for i, seed in enumerate(seeds):
            print(f"\nTrial {i+1}/{len(seeds)}")
            result = self.run_single_experiment(seed, data_dir)
            all_results.append(result)

        # Calculate statistics
        baseline_accs = [r["baseline_accuracy"] for r in all_results]
        ensemble_accs = [r["ensemble_accuracy"] for r in all_results]
        improvements = [r["improvement"] for r in all_results]

        baseline_mean = np.mean(baseline_accs)
        baseline_std = np.std(baseline_accs)
        ensemble_mean = np.mean(ensemble_accs)
        ensemble_std = np.std(ensemble_accs)
        improvement_mean = np.mean(improvements)
        improvement_std = np.std(improvements)

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SUMMARY ({len(seeds)} trials)")
        print(f"{'='*80}")
        print(
            f"Baseline:                     {baseline_mean:.2f}% ± {baseline_std:.2f}%"
        )
        print(
            f"Clustering-Based Ensemble:    {ensemble_mean:.2f}% ± {ensemble_std:.2f}%"
        )
        print(
            f"Average Improvement:          {improvement_mean:+.2f}%p ± {improvement_std:.2f}%p"
        )

        # Statistical significance check
        if improvement_mean > 2 * improvement_std and improvement_mean > 0:
            print(f"Result: SIGNIFICANT IMPROVEMENT (>2σ confidence)")
        elif improvement_mean > 0:
            print(f"Result: POSITIVE IMPROVEMENT")
        else:
            print(f"Result: NO SIGNIFICANT IMPROVEMENT")

        # Save detailed results
        summary_results = {
            "experiment_type": "multiple_trials",
            "num_trials": len(seeds),
            "seeds": seeds,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "ensemble_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "improvement_mean": improvement_mean,
            "improvement_std": improvement_std,
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
        "--imbalance_factor",
        type=int,
        default=100,
        help="Imbalance factor for long-tail distribution",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./MachineLearningCVE",
        help="Directory containing CICIDS-2017 CSV files",
    )

    args = parser.parse_args()

    # Initialize experiment
    experiment = FinalExperiment(
        num_experts=args.num_experts, imbalance_factor=args.imbalance_factor
    )

    if args.mode == "fixed":
        print(f"\nRunning FIXED SEED experiment (seed={args.seed})")
        result = experiment.run_single_experiment(args.seed, args.data_dir)

        # Save single result
        with open(f"final_experiment_fixed_seed{args.seed}.json", "w") as f:
            json.dump(result, f, indent=2)

    elif args.mode == "random":
        print(f"\nRunning RANDOM SEED experiments ({args.trials} trials)")
        random_seeds = [random.randint(1, 10000) for _ in range(args.trials)]
        summary = experiment.run_multiple_experiments(random_seeds, args.data_dir)

    elif args.mode == "both":
        print(f"\nRunning BOTH modes:")
        print(f"1. Fixed seed experiment (seed={args.seed})")
        print(f"2. Random seed experiments ({args.trials} trials)")

        # Fixed seed experiment
        print(f"\n" + "=" * 80)
        print(f"FIXED SEED EXPERIMENT")
        print(f"=" * 80)
        fixed_result = experiment.run_single_experiment(args.seed, args.data_dir)

        # Random seed experiments
        print(f"\n" + "=" * 80)
        print(f"RANDOM SEED EXPERIMENTS")
        print(f"=" * 80)
        random_seeds = [random.randint(1, 10000) for _ in range(args.trials)]
        random_summary = experiment.run_multiple_experiments(random_seeds, args.data_dir)

        # Compare results
        print(f"\n" + "=" * 80)
        print(f"COMPARISON: FIXED vs RANDOM SEEDS")
        print(f"=" * 80)
        print(f"Fixed seed (reproducible):    {fixed_result['improvement']:+.2f}%p")
        print(
            f"Random seeds (robust):        {random_summary['improvement_mean']:+.2f}%p ± {random_summary['improvement_std']:.2f}%p"
        )

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


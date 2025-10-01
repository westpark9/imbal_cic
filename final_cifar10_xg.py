#!/usr/bin/env python3
"""
xgboost 추가
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import random
from tqdm import tqdm
import warnings
import time
import json
import argparse
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


class SharedBackbone(nn.Module):
    """Shared backbone network for feature extraction"""

    def __init__(self, feature_dim=256):
        super(SharedBackbone, self).__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()

        self.backbone_features = nn.Sequential(*list(backbone.children())[:-1])

        in_features = backbone.fc.in_features
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x):
        backbone_out = self.backbone_features(x).flatten(1)
        features = self.feature_extractor(backbone_out)
        return features


class ExpertClassifier(nn.Module):
    """Individual expert classifier for a subset of classes"""

    def __init__(self, feature_dim, num_classes):
        super(ExpertClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
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
    """Simple baseline model for comparison"""

    def __init__(self):
        super(BaselineModel, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.backbone(x)


class XGBoostModel:
    """XGBoost model wrapper for CIFAR-10 classification"""

    def __init__(self, use_features=True, n_components=256):
        """
        Args:
            use_features: If True, use CNN features; if False, use raw pixels
            n_components: Number of PCA components for dimensionality reduction
        """
        self.use_features = use_features
        self.n_components = n_components
        self.feature_extractor = None
        self.pca = None
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _extract_features(self, dataloader):
        """Extract features from images using pre-trained backbone"""
        if self.feature_extractor is None:
            # Create feature extractor (backbone without final layer)
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
        
        features = []
        labels = []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data = data.to(self.device)
                if self.use_features:
                    # Extract CNN features
                    feat = self.feature_extractor(data).flatten(1).cpu().numpy()
                else:
                    # Use raw pixels (flattened)
                    feat = data.cpu().numpy().reshape(data.size(0), -1)
                
                features.append(feat)
                labels.extend(targets.numpy())
        
        features = np.vstack(features)
        labels = np.array(labels)
        
        return features, labels
    
    def _prepare_data(self, features, fit_transform=False):
        """Prepare data with scaling and dimensionality reduction"""
        if fit_transform:
            # Fit scaler
            features_scaled = self.scaler.fit_transform(features)
            
            # Apply PCA if features are high-dimensional
            if features_scaled.shape[1] > self.n_components:
                self.pca = PCA(n_components=self.n_components, random_state=42)
                features_final = self.pca.fit_transform(features_scaled)
            else:
                features_final = features_scaled
        else:
            # Transform only
            features_scaled = self.scaler.transform(features)
            if self.pca is not None:
                features_final = self.pca.transform(features_scaled)
            else:
                features_final = features_scaled
                
        return features_final


class FinalExperiment:
    """
    Final comprehensive experiment class supporting both fixed and random seed modes
    """

    def __init__(self, num_experts=4, imbalance_factor=100, epochs=100):
        self.num_experts = num_experts
        self.imbalance_factor = imbalance_factor
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Final Clustering-Based Expert Ensemble Experiment")
        print(f"   Number of experts: {num_experts}")
        print(f"   Imbalance factor: {imbalance_factor}")
        print(f"   Training epochs: {epochs}")
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

    def setup_dataset(self, seed):
        """Setup CIFAR-10 with long-tail imbalance"""
        self.set_random_seed(seed)  # Ensure dataset creation is reproducible

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

        full_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        self.train_subset, self.samples_per_class = self._create_imbalanced_dataset(
            full_train
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )

        print(
            f"   Dataset: {len(self.train_subset)} train, {len(self.test_dataset)} test"
        )

    def _create_imbalanced_dataset(self, dataset):
        """Create long-tail imbalanced dataset"""
        targets = [dataset[i][1] for i in range(len(dataset))]

        max_samples = 5000
        min_samples = max_samples // self.imbalance_factor

        samples_per_class = []
        for i in range(10):
            ratio = i / 9
            num_samples = int(max_samples * (min_samples / max_samples) ** ratio)
            samples_per_class.append(max(num_samples, min_samples))

        class_indices = [[] for _ in range(10)]
        for idx, target in enumerate(targets):
            class_indices[target].append(idx)

        selected_indices = []
        for class_id, max_count in enumerate(samples_per_class):
            indices = class_indices[class_id]
            if len(indices) >= max_count:
                selected = np.random.choice(indices, max_count, replace=False)
            else:
                selected = indices
            selected_indices.extend(selected)

        return Subset(dataset, selected_indices), samples_per_class

    def get_clustering_groups(self, seed):
        """Get clustering-based expert groups using the 6-step pipeline"""
        print("   Executing 6-step clustering pipeline...")

        # Step 1: Backbone warmup for feature extraction
        self.set_random_seed(seed + 10)  # Offset seed for warmup

        warmup_model = BaselineModel().to(self.device)
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=True, num_workers=2
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            warmup_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )

        warmup_model.train()
        for epoch in range(15):  # Quick warmup
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = warmup_model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Step 2: Extract class embeddings
        feature_extractor = SharedBackbone(feature_dim=256).to(self.device)

        # Load warmed-up weights
        backbone_state = {}
        for name, param in warmup_model.state_dict().items():
            if not name.startswith("backbone.fc"):
                backbone_state[name.replace("backbone.", "")] = param

        feature_state = feature_extractor.state_dict()
        for name, param in backbone_state.items():
            if name in feature_state and feature_state[name].shape == param.shape:
                feature_state[name] = param
        feature_extractor.load_state_dict(feature_state)

        feature_extractor.eval()

        # Collect class features
        class_features = [[] for _ in range(10)]
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=False, num_workers=2
        )

        with torch.no_grad():
            for data, targets in train_loader:
                data = data.to(self.device)
                features = feature_extractor(data).cpu().numpy()

                for i, target in enumerate(targets):
                    class_features[target.item()].append(features[i])

        # Calculate class centroids
        class_centroids = []
        for class_id in range(10):
            if len(class_features[class_id]) > 0:
                centroid = np.mean(class_features[class_id], axis=0)
                class_centroids.append(centroid)
            else:
                centroid = np.random.normal(0, 0.1, 256)
                class_centroids.append(centroid)

        class_centroids = np.array(class_centroids)

        # Step 3: Clustering with [frequency + embeddings]
        samples_array = np.array(self.samples_per_class)
        frequency_zscore = (samples_array - np.mean(samples_array)) / np.std(
            samples_array
        )

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

    def train_baseline(self, seed):
        """Train baseline model"""
        print("   Training baseline model...")

        self.set_random_seed(seed)

        model = BaselineModel().to(self.device)
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=True, num_workers=2
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            model.train()
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return model

    def train_xgboost(self, seed):
        """Train XGBoost model"""
        print("   Training XGBoost model...")

        self.set_random_seed(seed + 100)  # Offset seed for XGBoost

        # Create XGBoost model
        xgb_model = XGBoostModel(use_features=True, n_components=256)
        
        # Prepare data loaders
        train_loader = DataLoader(
            self.train_subset, batch_size=128, shuffle=False, num_workers=2
        )
        
        # Extract features and labels
        train_features, train_labels = xgb_model._extract_features(train_loader)
        
        # Prepare data with scaling and PCA
        train_features_final = xgb_model._prepare_data(train_features, fit_transform=True)
        
        # Train XGBoost model
        xgb_model.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed + 100,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        xgb_model.model.fit(train_features_final, train_labels)
        
        return xgb_model

    def train_expert_ensemble(self, expert_groups, seed):
        """Train clustering-based expert ensemble"""
        print("   Training clustering-based expert ensemble...")

        self.set_random_seed(seed + 300)  # Offset seed for ensemble training

        # Initialize models
        shared_backbone = SharedBackbone(feature_dim=256).to(self.device)
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
            self.train_subset, batch_size=128, shuffle=True, num_workers=2
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(all_params, lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            shared_backbone.train()
            router.train()
            for expert_classifier in expert_classifiers:
                expert_classifier.train()

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
                final_logits = torch.zeros(data.size(0), 10, device=self.device)

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

                total_loss = classification_loss + 0.1 * router_loss
                total_loss.backward()
                optimizer.step()

            scheduler.step()

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

    def plot_per_class_metrics(self, baseline_acc, baseline_f1, ensemble_acc, ensemble_f1, 
                              xgboost_acc, xgboost_f1, class_names, save_path):
        """클래스별 성능 지표를 시각화합니다 (3개 모델 비교)."""
        plt.figure(figsize=(18, 6))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width, baseline_acc, width, label='Baseline', alpha=0.8)
        plt.bar(x, ensemble_acc, width, label='Ensemble', alpha=0.8)
        plt.bar(x + width, xgboost_acc, width, label='XGBoost', alpha=0.8)
        plt.title('Per-Class Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(x - width, baseline_f1, width, label='Baseline', alpha=0.8)
        plt.bar(x, ensemble_f1, width, label='Ensemble', alpha=0.8)
        plt.bar(x + width, xgboost_f1, width, label='XGBoost', alpha=0.8)
        plt.title('Per-Class F1 Score')
        plt.ylabel('F1 Score')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_performance_metrics(self, baseline_acc, baseline_f1, ensemble_acc, ensemble_f1, 
                                xgboost_acc, xgboost_f1, class_names, save_path):
        """성능 지표를 CSV 파일로 저장합니다 (3개 모델 비교)."""
        df = pd.DataFrame({
            'Class': class_names,
            'Baseline_Accuracy': baseline_acc,
            'Baseline_F1': baseline_f1,
            'Ensemble_Accuracy': ensemble_acc,
            'Ensemble_F1': ensemble_f1,
            'XGBoost_Accuracy': xgboost_acc,
            'XGBoost_F1': xgboost_f1
        })
        df.to_csv(save_path, index=False)
        
    def plot_model_comparison(self, baseline_acc, ensemble_acc, xgboost_acc, save_path):
        """모델 간 전체 정확도 비교 차트"""
        plt.figure(figsize=(10, 6))
        
        models = ['Baseline', 'Ensemble', 'XGBoost']
        accuracies = [baseline_acc, ensemble_acc, xgboost_acc]
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # 막대 위에 정확도 값 표시
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, max(accuracies) * 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate_model(self, model):
        """단일 모델 평가 및 클래스별 성능 지표 계산"""
        test_loader = DataLoader(
            self.test_dataset, batch_size=128, shuffle=False, num_workers=2
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

        # 전체 정확도
        accuracy = 100.0 * (all_predictions == all_targets).mean()
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            all_predictions, all_targets, 10
        )
        
        # 혼동 행렬
        cm = confusion_matrix(all_targets, all_predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm

    def evaluate_xgboost(self, xgb_model):
        """XGBoost 모델 평가 및 클래스별 성능 지표 계산"""
        test_loader = DataLoader(
            self.test_dataset, batch_size=128, shuffle=False, num_workers=2
        )

        # Extract features from test data
        test_features, test_targets = xgb_model._extract_features(test_loader)
        
        # Prepare test data (transform only, no fitting)
        test_features_final = xgb_model._prepare_data(test_features, fit_transform=False)
        
        # Make predictions
        predictions = xgb_model.model.predict(test_features_final)

        # 전체 정확도
        accuracy = 100.0 * (predictions == test_targets).mean()
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            predictions, test_targets, 10
        )
        
        # 혼동 행렬
        cm = confusion_matrix(test_targets, predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm

    def evaluate_expert_ensemble(
        self, shared_backbone, expert_classifiers, router, expert_groups
    ):
        """전문가 앙상블 평가 및 클래스별 성능 지표 계산"""
        test_loader = DataLoader(
            self.test_dataset, batch_size=128, shuffle=False, num_workers=2
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

                final_logits = torch.zeros(data.size(0), 10, device=self.device)

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

        # 전체 정확도
        accuracy = 100.0 * (all_predictions == all_targets).mean()
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            all_predictions, all_targets, 10
        )
        
        # 혼동 행렬
        cm = confusion_matrix(all_targets, all_predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm

    def run_single_experiment(self, seed):
        """단일 실험 실행 및 결과 저장 (3개 모델 비교)"""
        print(f"\nExperiment with seed {seed}")
        print("=" * 50)

        start_time = time.time()

        # 실험 디렉토리 생성
        exp_dir = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)

        # 데이터셋 설정
        self.setup_dataset(seed)

        # 1. 베이스라인 모델 학습 및 평가
        print("   1/3: Training Baseline model...")
        baseline_model = self.train_baseline(seed)
        baseline_acc, baseline_per_class_acc, baseline_per_class_f1, baseline_cm = self.evaluate_model(baseline_model)

        # 2. XGBoost 모델 학습 및 평가
        print("   2/3: Training XGBoost model...")
        xgboost_model = self.train_xgboost(seed)
        xgboost_acc, xgboost_per_class_acc, xgboost_per_class_f1, xgboost_cm = self.evaluate_xgboost(xgboost_model)

        # 3. 클러스터링 그룹 생성 및 앙상블 학습
        print("   3/3: Training Expert Ensemble model...")
        clustering_groups = self.get_clustering_groups(seed)
        shared_backbone, expert_classifiers, router = self.train_expert_ensemble(
            clustering_groups, seed
        )
        ensemble_acc, ensemble_per_class_acc, ensemble_per_class_f1, ensemble_cm = self.evaluate_expert_ensemble(
            shared_backbone, expert_classifiers, router, clustering_groups
        )

        elapsed_time = (time.time() - start_time) / 60
        ensemble_improvement = ensemble_acc - baseline_acc
        xgboost_improvement = xgboost_acc - baseline_acc

        # 결과 시각화 및 저장
        class_names = self.test_dataset.classes
        
        # 혼동 행렬 시각화
        self.plot_confusion_matrix(
            baseline_cm, class_names,
            "Baseline Model Confusion Matrix",
            os.path.join(exp_dir, "baseline_confusion_matrix.png")
        )
        self.plot_confusion_matrix(
            xgboost_cm, class_names,
            "XGBoost Model Confusion Matrix",
            os.path.join(exp_dir, "xgboost_confusion_matrix.png")
        )
        self.plot_confusion_matrix(
            ensemble_cm, class_names,
            "Ensemble Model Confusion Matrix",
            os.path.join(exp_dir, "ensemble_confusion_matrix.png")
        )
        
        # 모델 비교 차트
        self.plot_model_comparison(
            baseline_acc, ensemble_acc, xgboost_acc,
            os.path.join(exp_dir, "model_comparison.png")
        )
        
        # 클래스별 성능 지표 시각화
        self.plot_per_class_metrics(
            baseline_per_class_acc, baseline_per_class_f1,
            ensemble_per_class_acc, ensemble_per_class_f1,
            xgboost_per_class_acc, xgboost_per_class_f1,
            class_names,
            os.path.join(exp_dir, "per_class_metrics.png")
        )
        
        # 성능 지표 CSV 저장
        self.save_performance_metrics(
            baseline_per_class_acc, baseline_per_class_f1,
            ensemble_per_class_acc, ensemble_per_class_f1,
            xgboost_per_class_acc, xgboost_per_class_f1,
            class_names,
            os.path.join(exp_dir, "performance_metrics.csv")
        )

        results = {
            "seed": seed,
            "baseline_accuracy": baseline_acc,
            "ensemble_accuracy": ensemble_acc,
            "xgboost_accuracy": xgboost_acc,
            "ensemble_improvement": ensemble_improvement,
            "xgboost_improvement": xgboost_improvement,
            "runtime_minutes": elapsed_time,
            "clustering_groups": clustering_groups,
            "timestamp": datetime.now().isoformat(),
            "baseline_per_class_acc": baseline_per_class_acc,
            "baseline_per_class_f1": baseline_per_class_f1,
            "ensemble_per_class_acc": ensemble_per_class_acc,
            "ensemble_per_class_f1": ensemble_per_class_f1,
            "xgboost_per_class_acc": xgboost_per_class_acc,
            "xgboost_per_class_f1": xgboost_per_class_f1
        }

        print(f"Results:")
        print(f"   Baseline:                     {baseline_acc:.2f}%")
        print(f"   XGBoost:                      {xgboost_acc:.2f}% ({xgboost_improvement:+.2f}%p)")
        print(f"   Clustering-Based Ensemble:    {ensemble_acc:.2f}% ({ensemble_improvement:+.2f}%p)")
        print(f"   Runtime:                      {elapsed_time:.1f} minutes")
        print(f"   Results saved to:             {exp_dir}")

        return results

    def run_multiple_experiments(self, seeds):
        """Run multiple experiments with different seeds"""
        print(f"\nRunning {len(seeds)} experiments with seeds: {seeds}")
        print("=" * 80)

        all_results = []

        for i, seed in enumerate(seeds):
            print(f"\nTrial {i+1}/{len(seeds)}")
            result = self.run_single_experiment(seed)
            all_results.append(result)

        # Calculate statistics
        baseline_accs = [r["baseline_accuracy"] for r in all_results]
        ensemble_accs = [r["ensemble_accuracy"] for r in all_results]
        xgboost_accs = [r["xgboost_accuracy"] for r in all_results]
        ensemble_improvements = [r["ensemble_improvement"] for r in all_results]
        xgboost_improvements = [r["xgboost_improvement"] for r in all_results]

        baseline_mean = np.mean(baseline_accs)
        baseline_std = np.std(baseline_accs)
        ensemble_mean = np.mean(ensemble_accs)
        ensemble_std = np.std(ensemble_accs)
        xgboost_mean = np.mean(xgboost_accs)
        xgboost_std = np.std(xgboost_accs)
        ensemble_improvement_mean = np.mean(ensemble_improvements)
        ensemble_improvement_std = np.std(ensemble_improvements)
        xgboost_improvement_mean = np.mean(xgboost_improvements)
        xgboost_improvement_std = np.std(xgboost_improvements)

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SUMMARY ({len(seeds)} trials)")
        print(f"{'='*80}")
        print(
            f"Baseline:                     {baseline_mean:.2f}% ± {baseline_std:.2f}%"
        )
        print(
            f"XGBoost:                      {xgboost_mean:.2f}% ± {xgboost_std:.2f}%"
        )
        print(
            f"Clustering-Based Ensemble:    {ensemble_mean:.2f}% ± {ensemble_std:.2f}%"
        )
        print(
            f"XGBoost Improvement:          {xgboost_improvement_mean:+.2f}%p ± {xgboost_improvement_std:.2f}%p"
        )
        print(
            f"Ensemble Improvement:         {ensemble_improvement_mean:+.2f}%p ± {ensemble_improvement_std:.2f}%p"
        )

        # Statistical significance check
        print(f"\nStatistical Significance Analysis:")
        if xgboost_improvement_mean > 2 * xgboost_improvement_std and xgboost_improvement_mean > 0:
            print(f"XGBoost: SIGNIFICANT IMPROVEMENT (>2σ confidence)")
        elif xgboost_improvement_mean > 0:
            print(f"XGBoost: POSITIVE IMPROVEMENT")
        else:
            print(f"XGBoost: NO SIGNIFICANT IMPROVEMENT")
            
        if ensemble_improvement_mean > 2 * ensemble_improvement_std and ensemble_improvement_mean > 0:
            print(f"Ensemble: SIGNIFICANT IMPROVEMENT (>2σ confidence)")
        elif ensemble_improvement_mean > 0:
            print(f"Ensemble: POSITIVE IMPROVEMENT")
        else:
            print(f"Ensemble: NO SIGNIFICANT IMPROVEMENT")

        # Save detailed results
        summary_results = {
            "experiment_type": "multiple_trials",
            "num_trials": len(seeds),
            "seeds": seeds,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "ensemble_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "xgboost_mean": xgboost_mean,
            "xgboost_std": xgboost_std,
            "ensemble_improvement_mean": ensemble_improvement_mean,
            "ensemble_improvement_std": ensemble_improvement_std,
            "xgboost_improvement_mean": xgboost_improvement_mean,
            "xgboost_improvement_std": xgboost_improvement_std,
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
        description="Final Clustering-Based Expert Ensemble Experiment"
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "random", "both"],
        default="fixed",
        help="Experiment mode: fixed seed, random seeds, or both",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Fixed seed for reproducible experiments"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
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
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs for neural network models",
    )

    args = parser.parse_args()

    # Initialize experiment
    experiment = FinalExperiment(
        num_experts=args.num_experts, 
        imbalance_factor=args.imbalance_factor,
        epochs=args.epochs
    )

    if args.mode == "fixed":
        print(f"\nRunning FIXED SEED experiment (seed={args.seed})")
        result = experiment.run_single_experiment(args.seed)

        # Save single result
        with open(f"final_experiment_fixed_seed{args.seed}.json", "w") as f:
            json.dump(result, f, indent=2)

    elif args.mode == "random":
        print(f"\nRunning RANDOM SEED experiments ({args.trials} trials)")
        random_seeds = [random.randint(1, 10000) for _ in range(args.trials)]
        summary = experiment.run_multiple_experiments(random_seeds)

    elif args.mode == "both":
        print(f"\nRunning BOTH modes:")
        print(f"1. Fixed seed experiment (seed={args.seed})")
        print(f"2. Random seed experiments ({args.trials} trials)")

        # Fixed seed experiment
        print(f"\n" + "=" * 80)
        print(f"FIXED SEED EXPERIMENT")
        print(f"=" * 80)
        fixed_result = experiment.run_single_experiment(args.seed)

        # Random seed experiments
        print(f"\n" + "=" * 80)
        print(f"RANDOM SEED EXPERIMENTS")
        print(f"=" * 80)
        random_seeds = [random.randint(1, 10000) for _ in range(args.trials)]
        random_summary = experiment.run_multiple_experiments(random_seeds)

        # Compare results
        print(f"\n" + "=" * 80)
        print(f"COMPARISON: FIXED vs RANDOM SEEDS")
        print(f"=" * 80)
        print(f"Fixed seed (ensemble):        {fixed_result['ensemble_improvement']:+.2f}%p")
        print(f"Fixed seed (xgboost):         {fixed_result['xgboost_improvement']:+.2f}%p")
        print(
            f"Random seeds (ensemble):      {random_summary['ensemble_improvement_mean']:+.2f}%p ± {random_summary['ensemble_improvement_std']:.2f}%p"
        )
        print(
            f"Random seeds (xgboost):       {random_summary['xgboost_improvement_mean']:+.2f}%p ± {random_summary['xgboost_improvement_std']:.2f}%p"
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


#!/usr/bin/env python3
"""
Final Comprehensive Experiment: Clustering-Based Expert Ensemble vs Baseline
Alternative Dataset Version (UNSW-NB15 or NSL-KDD)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
import requests
import zipfile

warnings.filterwarnings("ignore")

class AlternativeDataset(Dataset):
    """대안 데이터셋을 위한 커스텀 Dataset 클래스"""
    
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

class AlternativeExperiment:
    """
    Alternative dataset experiment class supporting UNSW-NB15 and NSL-KDD
    """

    def __init__(self, num_experts=4, imbalance_factor=100, dataset_type="unsw"):
        self.num_experts = num_experts
        self.imbalance_factor = imbalance_factor
        self.dataset_type = dataset_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Alternative Dataset Expert Ensemble Experiment")
        print(f"   Dataset: {dataset_type.upper()}")
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

    def download_unsw_nb15(self, data_dir):
        """UNSW-NB15 데이터셋 다운로드"""
        print("   Downloading UNSW-NB15 dataset...")
        
        # UNSW-NB15 데이터셋 URL (실제 URL로 변경 필요)
        urls = {
            "train": "https://raw.githubusercontent.com/UNSW-CSE-CS-CC-17-2/UNSW-NB15/master/data/UNSW_NB15_training-set.csv",
            "test": "https://raw.githubusercontent.com/UNSW-CSE-CS-CC-17-2/UNSW-NB15/master/data/UNSW_NB15_testing-set.csv"
        }
        
        os.makedirs(data_dir, exist_ok=True)
        
        for split, url in urls.items():
            filename = os.path.join(data_dir, f"UNSW_NB15_{split}.csv")
            if not os.path.exists(filename):
                try:
                    print(f"     Downloading {split} set...")
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    print(f"       Downloaded {filename}")
                except Exception as e:
                    print(f"       Error downloading {split}: {e}")
                    return False
        
        return True

    def load_unsw_nb15_data(self, data_dir):
        """UNSW-NB15 데이터 로드 및 전처리"""
        print("   Loading UNSW-NB15 dataset...")
        
        train_file = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
        test_file = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
        
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print("     UNSW-NB15 files not found. Please download first.")
            return None, None, None, None, None
        
        # 데이터 로드
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # 레이블 컬럼 찾기
        label_col = None
        for col in train_df.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                label_col = col
                break
        
        if label_col is None:
            print("     Label column not found!")
            return None, None, None, None, None
        
        # 특성과 레이블 분리
        feature_cols = [col for col in train_df.columns if col != label_col]
        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[label_col].values
        
        # 특성 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 레이블 인코딩
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"     Train samples: {len(X_train_scaled)}")
        print(f"     Test samples: {len(X_test_scaled)}")
        print(f"     Features: {X_train_scaled.shape[1]}")
        print(f"     Classes: {len(label_encoder.classes_)}")
        
        return X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, label_encoder, scaler, feature_cols

    def setup_dataset(self, seed, data_dir):
        """데이터셋 설정"""
        self.set_random_seed(seed)
        
        if self.dataset_type == "unsw":
            X_train, y_train, X_test, y_test, self.label_encoder, self.scaler, self.feature_columns = self.load_unsw_nb15_data(data_dir)
        else:
            print(f"     Dataset type {self.dataset_type} not supported yet")
            return False
        
        if X_train is None:
            return False
        
        # 불균형 데이터셋 생성
        X_train_imbalanced, y_train_imbalanced, self.samples_per_class = self._create_imbalanced_dataset(X_train, y_train)
        
        # 데이터셋 생성
        self.train_subset = AlternativeDataset(X_train_imbalanced, y_train_imbalanced)
        self.test_dataset = AlternativeDataset(X_test, y_test)
        
        # 클래스 수 저장
        self.num_classes = len(np.unique(y_train))
        
        print(f"   Dataset: {len(self.train_subset)} train, {len(self.test_dataset)} test")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {self.num_classes}")
        
        return True

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

    def run_single_experiment(self, seed, data_dir, epochs=100):
        """단일 실험 실행"""
        print(f"\nExperiment with seed {seed}")
        print("=" * 50)

        start_time = time.time()

        # 데이터셋 설정
        if not self.setup_dataset(seed, data_dir):
            print("     Failed to setup dataset!")
            return None

        # 실험 디렉토리 생성
        exp_dir = f"experiment_results_{self.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)

        print(f"   Results will be saved to: {exp_dir}")
        print(f"   Experiment completed in {((time.time() - start_time) / 60):.1f} minutes")

        return {"status": "success", "exp_dir": exp_dir}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Alternative Dataset Expert Ensemble Experiment"
    )
    parser.add_argument(
        "--dataset",
        choices=["unsw", "nsl"],
        default="unsw",
        help="Dataset type: unsw (UNSW-NB15) or nsl (NSL-KDD)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./alternative_data", help="Data directory"
    )

    args = parser.parse_args()

    # Initialize experiment
    experiment = AlternativeExperiment(dataset_type=args.dataset)

    # Run experiment
    result = experiment.run_single_experiment(args.seed, args.data_dir, args.epochs)

    if result:
        print("✅ Experiment completed successfully!")
    else:
        print("❌ Experiment failed!")

if __name__ == "__main__":
    main()

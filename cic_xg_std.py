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
from sklearn.metrics import f1_score, confusion_matrix
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
    """XGBoost-based expert classifier for a subset of classes"""

    def __init__(self, expert_classes, random_state=42):
        self.expert_classes = expert_classes  # 이 expert가 담당하는 클래스들
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
        """XGBoost 모델 훈련"""
        # 전역 클래스를 로컬 클래스로 변환
        local_labels = []
        valid_indices = []
        
        for i, label in enumerate(labels):
            if label in self.class_mapping:
                local_labels.append(self.class_mapping[label])
                valid_indices.append(i)
        
        if len(local_labels) == 0:
            print(f"       WARNING: Expert {self.expert_classes} has no training samples!")
            return
        
        # 유효한 샘플만 사용
        local_labels = np.array(local_labels)
        valid_features = features[valid_indices]
        
        # 클래스 수 확인 및 조정
        unique_classes = np.unique(local_labels)
        actual_num_classes = len(unique_classes)
        
        if actual_num_classes < 2:
            print(f"       WARNING: Expert {self.expert_classes} has only {actual_num_classes} class(es), skipping training")
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
        print(f"       Expert {self.expert_classes} trained with {len(valid_features)} samples, {actual_num_classes} classes")

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
        print(f"     Classes: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            class_name = label_encoder.inverse_transform([label])[0]
            print(f"       {class_name}: {count}")
        
        return X_final, y, label_encoder, None, feature_columns

    def setup_dataset(self, seed, data_dir):
        """CICIDS-2017 데이터셋 설정"""
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
        
        # 클래스별 분포 확인
        print(f"     Class distribution verification:")
        unique_classes = np.unique(y)
        total_train = len(X_train)
        total_test = len(X_test)

        print(f"     Final split: Train={total_train} samples, Test={total_test} samples")
        print(f"     Overall ratio: Train={total_train/(total_train+total_test):.1%}, Test={total_test/(total_train+total_test):.1%}")
        
        # 데이터 저장
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # 데이터셋 객체 생성
        self.train_subset = CICIDS2017Dataset(X_train, y_train)
        self.test_subset = CICIDS2017Dataset(X_test, y_test)
        
        print(f"   Dataset: {len(X_train)} train, {len(X_test)} test")

    def get_frequency_based_groups(self, seed):
        """클래스별 샘플 수 기준으로 전문가 그룹 생성"""
        print("   Creating frequency-based expert groups...")
        
        # 훈련 데이터에서 클래스별 샘플 수 계산
        unique_labels, counts = np.unique(self.y_train, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))
        
        # 클래스들을 샘플 수 기준으로 정렬 (내림차순)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_ids = [item[0] for item in sorted_classes]
        class_sample_counts = [item[1] for item in sorted_classes]
        
        print(f"     Class sample counts (sorted): {class_sample_counts}")
        
        # num_experts에 따라 분할점 계산
        # 예: 3개 expert면 1/3, 2/3 지점에서 분할
        split_points = []
        for i in range(1, self.num_experts):
            split_idx = int(len(class_ids) * i / self.num_experts)
            split_points.append(split_idx)
        
        print(f"     Split points: {split_points}")
        
        # 각 expert에게 클래스 할당
        expert_groups = [[] for _ in range(self.num_experts)]
        start_idx = 0
        
        for expert_idx in range(self.num_experts):
            if expert_idx == self.num_experts - 1:
                # 마지막 expert는 남은 모든 클래스
                end_idx = len(class_ids)
            else:
                end_idx = split_points[expert_idx]
            
            expert_groups[expert_idx] = class_ids[start_idx:end_idx]
            start_idx = end_idx
        
        # 결과 출력
        print("     Expert groups composition:")
        for expert_idx, class_list in enumerate(expert_groups):
            total_samples = sum(class_counts[cid] for cid in class_list)
            print(f"       Expert {expert_idx}: {len(class_list)} classes, {total_samples} samples")
            if len(class_list) == 0:
                print(f"         WARNING: Expert {expert_idx} has no classes assigned!")
            else:
                for cid in class_list:
                    cnt = class_counts[cid]
                    class_name = self.label_encoder.inverse_transform([cid])[0]
                    print(f"         · class {cid} ({class_name}): {cnt}")
        
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
        
        # 훈련 완료 후 성능 평가
        print("     Evaluating baseline performance...")
        baseline_acc, baseline_f1 = self._evaluate_baseline(baseline_model)
        

        training_history = {
            'accuracy': baseline_acc,
            'f1_score': baseline_f1
        }
        
        return baseline_model, training_history

    def train_expert_ensemble(self, expert_groups, seed):
        """빈도 기반 전문가 앙상블 학습 (XGBoost Experts)"""
        print("   Training frequency-based expert ensemble with XGBoost...")

        self.set_random_seed(seed + 300)
        
        # XGBoost Expert Classifiers 생성
        expert_classifiers = []
        for expert_idx, group in enumerate(expert_groups):
            expert_classifier = SimpleExpertXGBoostClassifier(
                expert_classes=group, random_state=seed + expert_idx
            )
            expert_classifiers.append(expert_classifier)
            print(f"       Expert {expert_idx} created with {len(group)} classes: {group}")

        # XGBoost Experts 훈련
        print("   Training XGBoost experts...")
        for expert_idx, (expert_classifier, expert_classes) in enumerate(
            zip(expert_classifiers, expert_groups)
        ):
            print(f"     Training XGBoost Expert {expert_idx}...")
            expert_classifier.fit(self.X_train, self.y_train)
        
        # 훈련 완료 후 최종 성능 평가
        print("   Evaluating final performance...")
        final_acc, final_f1 = self._evaluate_ensemble(expert_classifiers, expert_groups)
        
       
        training_history = {
            'accuracy': final_acc,
            'f1_score': final_f1
        }
        
        expert_history = {}
        for expert_idx in range(len(expert_classifiers)):
            expert_acc, expert_f1 = self._evaluate_expert(expert_classifiers[expert_idx], expert_groups[expert_idx])
            expert_history[expert_idx] = {
                'accuracy': expert_acc,
                'f1_score': expert_f1
            }
        
        return expert_classifiers, training_history, expert_history

    def _evaluate_baseline(self, baseline_model):
        """베이스라인 모델 성능 평가"""
        predictions = baseline_model.predict(self.X_test)

        # 정확도 계산
        accuracy = (predictions == self.y_test).mean()
        
        # F1 점수 계산
        f1 = f1_score(self.y_test, predictions, average='weighted', zero_division=0)
        
        return accuracy, f1

    def _evaluate_ensemble(self, expert_classifiers, expert_groups):
        """앙상블 모델 성능 평가"""
        predictions = []
        
        # 각 테스트 샘플에 대해 해당하는 expert 사용
        for i, (features, target) in enumerate(zip(self.X_test, self.y_test)):
            # 해당 클래스를 담당하는 expert 찾기
            expert_idx = None
            for idx, expert_classes in enumerate(expert_groups):
                if target in expert_classes:
                    expert_idx = idx
                    break
            
            if expert_idx is not None:
                # 해당 expert로 예측
                expert_pred = expert_classifiers[expert_idx].predict([features])
                predictions.append(expert_pred[0])
            else:
                # 담당 expert가 없는 경우 첫 번째 expert 사용
                expert_pred = expert_classifiers[0].predict([features])
                predictions.append(expert_pred[0])
        
        predictions = np.array(predictions)

        # 정확도 계산
        accuracy = (predictions == self.y_test).mean()
        
        # F1 점수 계산
        f1 = f1_score(self.y_test, predictions, average='weighted', zero_division=0)
        
        return accuracy, f1

    def _evaluate_expert(self, expert_classifier, expert_classes):
        """개별 expert 성능 평가"""
        # 해당 expert가 담당하는 클래스의 테스트 샘플만 필터링
        expert_mask = np.isin(self.y_test, expert_classes)
        
        if not expert_mask.any():
            return 0.0, 0.0

        X_expert = self.X_test[expert_mask]
        y_expert = self.y_test[expert_mask]
        
        # 예측
        predictions = expert_classifier.predict(X_expert)

        # 정확도 계산
        accuracy = (predictions == y_expert).mean()
        
        # F1 점수 계산
        f1 = f1_score(y_expert, predictions, average='weighted', zero_division=0)
        
        return accuracy, f1

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
        
        # 모델 성능
        if results['baseline_accuracy'] > 0:
            logger.info(f"Baseline XGBoost Accuracy: {results['baseline_accuracy']:.4f}")
        if results['ensemble_accuracy'] > 0:
            logger.info(f"Expert Ensemble (XGBoost) Accuracy: {results['ensemble_accuracy']:.4f}")
        if results['improvement_ensemble'] != 0:
            logger.info(f"Ensemble Improvement: {results['improvement_ensemble']:+.4f}")
        
        # 전문가 그룹 정보
        if results['expert_groups']:
            logger.info("\nExpert Groups:")
            for expert_idx, group in enumerate(results['expert_groups']):
                logger.info(f"  Expert {expert_idx}: {group}")
        
        # 클래스별 성능 (상세)
        if results.get('baseline_per_class_acc') and results.get('ensemble_per_class_acc'):
            logger.info("\nPer-Class Performance Details:")
            class_names = self.label_encoder.classes_
            
            for i, class_name in enumerate(class_names):
                baseline_acc = results['baseline_per_class_acc'][i] if i < len(results['baseline_per_class_acc']) else 0.0
                baseline_f1 = results['baseline_per_class_f1'][i] if i < len(results['baseline_per_class_f1']) else 0.0
                ensemble_acc = results['ensemble_per_class_acc'][i] if i < len(results['ensemble_per_class_acc']) else 0.0
                ensemble_f1 = results['ensemble_per_class_f1'][i] if i < len(results['ensemble_per_class_f1']) else 0.0
                
                logger.info(f"  {class_name}:")
                logger.info(f"    Baseline - Acc: {baseline_acc:.4f}, F1: {baseline_f1:.4f}")
                logger.info(f"    Ensemble (XGBoost) - Acc: {ensemble_acc:.4f}, F1: {ensemble_f1:.4f}")
                if baseline_acc > 0 and ensemble_acc > 0:
                    acc_improvement = ensemble_acc - baseline_acc
                    f1_improvement = ensemble_f1 - baseline_f1
                    logger.info(f"    Improvement - Acc: {acc_improvement:+.4f}, F1: {f1_improvement:+.4f}")
        
        logger.info("=" * 80)

    def evaluate_model(self, model):
        """단일 모델 평가 및 클래스별 성능 지표 계산"""
        print("     Evaluating baseline model...")
        
        # XGBoost 예측
        predictions = model.predict(self.X_test)

        # 전체 정확도
        accuracy = (predictions == self.y_test).mean()
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            predictions, self.y_test, self.num_classes
        )
        
        # 혼동 행렬
        cm = confusion_matrix(self.y_test, predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm

    def evaluate_expert_ensemble(self, expert_classifiers, expert_groups):
        """전문가 앙상블 평가 및 클래스별 성능 지표 계산"""
        print("     Evaluating expert ensemble (XGBoost)...")
        
        predictions = []
        
        # 각 테스트 샘플에 대해 해당하는 expert 사용
        for i, (features, target) in enumerate(zip(self.X_test, self.y_test)):
            # 해당 클래스를 담당하는 expert 찾기
            expert_idx = None
            for idx, expert_classes in enumerate(expert_groups):
                if target in expert_classes:
                    expert_idx = idx
                    break
            
            if expert_idx is not None:
                # 해당 expert로 예측
                expert_pred = expert_classifiers[expert_idx].predict([features])
                predictions.append(expert_pred[0])
            else:
                # 담당 expert가 없는 경우 첫 번째 expert 사용
                expert_pred = expert_classifiers[0].predict([features])
                predictions.append(expert_pred[0])
        
        predictions = np.array(predictions)

        # 전체 정확도
        accuracy = (predictions == self.y_test).mean()
        
        # 클래스별 성능 지표
        per_class_acc, per_class_f1 = self.compute_per_class_metrics(
            predictions, self.y_test, self.num_classes
        )
        
        # 혼동 행렬
        cm = confusion_matrix(self.y_test, predictions)
        
        return accuracy, per_class_acc, per_class_f1, cm

    def run_single_experiment(self, seed, data_dir, models=['baseline', 'ensemble']):
        """단일 실험 실행 및 결과 저장"""
        print(f"\nExperiment with seed {seed}")
        print("=" * 50)

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
        baseline_per_class_acc = []
        baseline_per_class_f1 = []
        baseline_cm = None
        
        expert_classifiers = None
        expert_groups = None
        ensemble_acc = 0.0
        ensemble_per_class_acc = []
        ensemble_per_class_f1 = []
        ensemble_cm = None
        
        # 훈련 과정 추적을 위한 변수들
        training_history = {}
        expert_history = {}

        # 베이스라인 모델
        if 'baseline' in models:
            baseline_model, baseline_history = self.train_baseline(seed)
            baseline_acc, baseline_per_class_acc, baseline_per_class_f1, baseline_cm = self.evaluate_model(baseline_model)
            training_history['Baseline'] = baseline_history

        # 앙상블 모델
        if 'ensemble' in models:
            expert_groups = self.get_frequency_based_groups(seed)
            expert_classifiers, ensemble_history, expert_history = self.train_expert_ensemble(
                expert_groups, seed
            )
            ensemble_acc, ensemble_per_class_acc, ensemble_per_class_f1, ensemble_cm = self.evaluate_expert_ensemble(
                expert_classifiers, expert_groups
            )
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
        
        # 성능 지표 CSV 저장
        self.save_performance_metrics_two_models(
            baseline_per_class_acc if 'baseline' in models else None,
            baseline_per_class_f1 if 'baseline' in models else None,
            ensemble_per_class_acc if 'ensemble' in models else None,
            ensemble_per_class_f1 if 'ensemble' in models else None,
            class_names,
            os.path.join(exp_dir, "performance_metrics.csv")
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
            "ensemble_accuracy": float(ensemble_acc),
            "improvement_ensemble": float(improvement_ensemble),
            "runtime_minutes": float(elapsed_time),
            "expert_groups": convert_numpy_types(expert_groups),
            "timestamp": datetime.now().isoformat(),
            "baseline_per_class_acc": convert_numpy_types(baseline_per_class_acc),
            "baseline_per_class_f1": convert_numpy_types(baseline_per_class_f1),
            "ensemble_per_class_acc": convert_numpy_types(ensemble_per_class_acc),
            "ensemble_per_class_f1": convert_numpy_types(ensemble_per_class_f1)
        }

        print(f"Results:")
        if 'baseline' in models:
            print(f"   Baseline XGBoost:             {baseline_acc:.4f}")
        if 'ensemble' in models:
            print(f"   Expert Ensemble (XGBoost):     {ensemble_acc:.4f}")
        
        if 'baseline' in models and 'ensemble' in models:
            print(f"   Ensemble Improvement:         {improvement_ensemble:+.4f}")
            
        print(f"   Runtime:                      {elapsed_time:.1f} minutes")
        print(f"   Results saved to:             {exp_dir}")
        
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
        print(f"\nRunning FIXED SEED experiment (seed={args.seed})")
        result = experiment.run_single_experiment(args.seed, args.data_dir, args.models)

        # Save single result
        with open(f"simple_experiment_xgboost_fixed_seed{args.seed}.json", "w") as f:
            json.dump(result, f, indent=2)

    elif args.mode == "random":
        print(f"\nRunning RANDOM SEED experiments ({args.trials} trials)")
        random_seeds = [random.randint(1, 10000) for _ in range(args.trials)]
        # Note: run_multiple_experiments would need to be implemented similar to the original

    elif args.mode == "both":
        print(f"\nRunning BOTH modes:")
        print(f"1. Fixed seed experiment (seed={args.seed})")
        print(f"2. Random seed experiments ({args.trials} trials)")

        # Fixed seed experiment
        print(f"\n" + "=" * 80)
        print(f"FIXED SEED EXPERIMENT")
        print(f"=" * 80)
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


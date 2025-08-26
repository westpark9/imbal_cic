#!/usr/bin/env python3
"""
Final Comprehensive Experiment: Clustering-Based Expert Ensemble vs Baseline
Supports both fixed seed (reproducible) and random seed (robust evaluation) modes.

Usage:
    python final_experiment.py --mode fixed --seed 42           # Fixed seed for reproducibility
    python final_experiment.py --mode random --trials 3        # Random seeds for robust evaluation
    python final_experiment.py --mode both --seed 42 --trials 3 # Both modes
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


class FinalExperiment:
    """
    Final comprehensive experiment class supporting both fixed and random seed modes
    """

    def __init__(self, num_experts=4, imbalance_factor=100):
        self.num_experts = num_experts
        self.imbalance_factor = imbalance_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Final Clustering-Based Expert Ensemble Experiment")
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

    def train_baseline(self, seed, epochs=100):
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
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

    def train_expert_ensemble(self, expert_groups, seed, epochs=100):
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
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
        """단일 실험 실행 및 결과 저장"""
        print(f"\nExperiment with seed {seed}")
        print("=" * 50)

        start_time = time.time()

        # 실험 디렉토리 생성
        exp_dir = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)

        # 데이터셋 설정
        self.setup_dataset(seed)

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
        class_names = self.test_dataset.classes
        
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
        description="Final Clustering-Based Expert Ensemble Experiment"
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

    args = parser.parse_args()

    # Initialize experiment
    experiment = FinalExperiment(
        num_experts=args.num_experts, imbalance_factor=args.imbalance_factor
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


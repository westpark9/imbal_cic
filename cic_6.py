#!/usr/bin/env python3
"""
TAILGUARD Ver.6.1: Explicit Taxonomy-Aware MoE (Modified)
- Pure Data Distribution (No Upsampling/Downsampling)
- Dynamic Ensemble Selection via Logistic Regression (Meta-Classifier)
"""

import os
import time
import argparse
import logging
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch

def infer_dataset_type(data_path, data=None):
    """파일명 또는 pkl 내 dataset_type에서 추론"""
    if data and 'dataset_type' in data:
        return data['dataset_type']
    name = os.path.basename(data_path).lower()
    if 'unsw' in name or 'nb15' in name:
        return 'unswnb15'
    if '2017' in name:
        return 'cic2017'
    if '2018' in name:
        return 'cic2018'
    raise ValueError(f"Cannot infer dataset_type from '{data_path}'. Use unswnb15/cic2017/cic2018 in filename.")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()

def setup_logging(exp_dir):
    log_file = os.path.join(exp_dir, "experiment.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger("experiment")

def calculate_energy(logits, T=1.0):
    logits = np.clip(logits.astype(np.float64), -700, 700)
    return (-T * logsumexp(logits / T, axis=1)).astype(np.float32)

def calculate_entropy(probs):
    probs = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return (-np.sum(probs * np.log(probs), axis=1)).astype(np.float32)

def batch_iterator(X, batch_size=20000):
    for i in range(0, X.shape[0], batch_size):
        yield X[i : i + batch_size]

class ExpertXGB:
    def __init__(self, expert_id, global_num_classes, expert_name="Expert", device="cpu", seed=42):
        self.id = expert_id
        self.name = expert_name
        self.global_num_classes = global_num_classes
        self.device = device
        self.seed = seed 
        self.model = None 
        self.local_classes = None     
        self.is_trained = False
        self.assigned_classes = [] 

    def fit_expert(self, X, y, expert_type="Global", target_classes=None, normal_classes=None):
        if expert_type == "Global":
            self.assigned_classes = np.unique(y).tolist()
            X_tr, y_tr = X, y
            logger = logging.getLogger("experiment")
            logger.info(f"      [{self.name}] Training with original data shape: {X_tr.shape}")
        else:
            self.assigned_classes = normal_classes + target_classes
            logger = logging.getLogger("experiment")
            logger.info(f"      [{self.name}] Filtering original data (No augmentation/subsampling)...")
            
            # 인위적인 샘플링 없이 할당된 클래스만 원본에서 그대로 추출
            mask = np.isin(y, self.assigned_classes)
            X_tr = X[mask]
            y_tr = y[mask]

            shuffle_idx = np.random.permutation(len(y_tr))
            X_tr = X_tr[shuffle_idx]
            y_tr = y_tr[shuffle_idx]
            logger.info(f"      [{self.name}] Filtered original data shape: {X_tr.shape}")

        if len(X_tr) == 0:
            self.is_trained = False
            return

        self.local_classes = np.sort(np.unique(y_tr))
        mapper_g2l = {g: l for l, g in enumerate(self.local_classes)}
        y_local = np.array([mapper_g2l[v] for v in y_tr], dtype=int)
        
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, objective="multi:softprob",
            num_class=len(self.local_classes), tree_method="hist", device=self.device,
            random_state=self.seed, n_jobs=4, early_stopping_rounds=10
        )

        try:
            X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
                X_tr, y_local, test_size=0.1, stratify=y_local, random_state=self.seed
            )
        except ValueError:
            X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
                X_tr, y_local, test_size=0.1, random_state=self.seed
            )

        self.model.fit(X_train_sub, y_train_sub, eval_set=[(X_val_sub, y_val_sub)], verbose=False)
        self.is_trained = True
        del X_tr, y_tr, X_train_sub, X_val_sub, y_train_sub, y_val_sub; gc.collect()

    def predict_proba_batch(self, X, batch_size=50000):
        N = X.shape[0]
        if not self.is_trained or self.model is None:
            return np.ones((N, self.global_num_classes), dtype=np.float32) / self.global_num_classes
        results = []
        for X_batch in batch_iterator(X, batch_size):
            try:
                pred_local = self.model.predict_proba(X_batch)
                pred_global = np.zeros((X_batch.shape[0], self.global_num_classes), dtype=np.float32)
                for l_idx, g_idx in enumerate(self.local_classes): pred_global[:, g_idx] = pred_local[:, l_idx]
                results.append(pred_global)
            except:
                results.append(np.ones((X_batch.shape[0], self.global_num_classes), dtype=np.float32) / self.global_num_classes)
        return np.vstack(results)

    def predict_logits_batch(self, X, batch_size=50000):
        N = X.shape[0]
        default_logit = -10.0 
        if not self.is_trained or self.model is None:
            return np.full((N, self.global_num_classes), default_logit, dtype=np.float32)
        results = []
        for X_batch in batch_iterator(X, batch_size):
            try:
                logits_local = self.model.predict(X_batch, output_margin=True)
                logits_global = np.full((X_batch.shape[0], self.global_num_classes), default_logit, dtype=np.float32)
                for l_idx, g_idx in enumerate(self.local_classes): logits_global[:, g_idx] = logits_local[:, l_idx]
                results.append(logits_global)
            except:
                results.append(np.full((X_batch.shape[0], self.global_num_classes), default_logit, dtype=np.float32))
        return np.vstack(results)

class TaxonomyDrivenMoE:
    def __init__(self, num_classes=None, dataset_type="cic2017", device="cpu", seed=42):
        self.num_classes = num_classes
        self.dataset_type = dataset_type
        self.device = device
        self.seed = seed
        self.experts = []
        self.norm_params = {}
        self.metrics_cache = {}
        self.meta_clf = None

    def get_taxonomy_groups(self, class_names):
        groups = {'Normal': [], 'Group1': [], 'Group2': [], 'Group3': [], 'Group4': []}
        
        def normalize(s): return s.lower().replace("", "").replace("-", "").replace(" ", "")
        
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
                ["Web Attack  Brute Force", "Web Attack  Sql Injection", "Web Attack  XSS"],
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
            
        norm_type_groups = [[normalize(c) for c in group] for group in type_groups]

        for idx, name in enumerate(class_names):
            norm_name = normalize(name)
            if 'normal' in norm_name or 'benign' in norm_name:
                groups['Normal'].append(idx)
                continue
                
            matched = False
            for g_idx, norm_group in enumerate(norm_type_groups):
                if norm_name in norm_group:
                    groups[f'Group{g_idx+1}'].append(idx)
                    matched = True
                    break
            if not matched:
                logging.getLogger("experiment").warning(f"Class '{name}' was not mapped to any group. Treating as Normal.")
                groups['Normal'].append(idx)
                
        return groups

    def train_single(self, X_train, y_train):
        exp_global = ExpertXGB(0, self.num_classes, expert_name="Baseline", device=self.device, seed=self.seed)
        exp_global.fit_expert(X_train, y_train, expert_type="Global")
        self.experts.append(exp_global)

    def build_experts(self, X_train, y_train, logger, class_names):
        groups = self.get_taxonomy_groups(class_names)
        normal_classes = groups['Normal']
        
        logger.info(f"\n  [Phase 1] Strict Taxonomy Mapping for {self.dataset_type}...")
        for group_name, classes in groups.items():
            names = [class_names[c] for c in classes]
            logger.info(f"    [{group_name:12s}] : {names}")

        logger.info("\n  [Phase 2] Training Domain Specialists...")
        
        exp_global = ExpertXGB(0, self.num_classes, expert_name="Global_Router", device=self.device, seed=self.seed)
        exp_global.fit_expert(X_train, y_train, expert_type="Global")
        self.experts.append(exp_global)
        
        specialist_configs = [
            (1, "Spec_Group1", groups['Group1']),
            (2, "Spec_Group2", groups['Group2']),
            (3, "Spec_Group3", groups['Group3']),
            (4, "Spec_Group4", groups['Group4'])
        ]
        
        for exp_id, name, target_group in specialist_configs:
            if target_group:
                exp = ExpertXGB(exp_id, self.num_classes, expert_name=name, device=self.device, seed=self.seed)
                exp.fit_expert(
                    X_train, y_train, expert_type="Specialist", 
                    target_classes=target_group, normal_classes=normal_classes,
                )
                self.experts.append(exp)

    def extract_metrics_batch(self, X, batch_size=20000):
        N, num_exp = X.shape[0], len(self.experts)
        preds_all = np.zeros((N, num_exp), dtype=int)
        conf_all, ent_all, eng_all = np.zeros((N, num_exp), dtype=np.float32), np.zeros((N, num_exp), dtype=np.float32), np.zeros((N, num_exp), dtype=np.float32)
        
        for i, expert in enumerate(self.experts):
            if not expert.is_trained: continue
            probs = expert.predict_proba_batch(X, batch_size=batch_size)
            logits = expert.predict_logits_batch(X, batch_size=batch_size)
            
            preds_all[:, i] = np.argmax(probs, axis=1)
            conf_all[:, i] = np.max(probs, axis=1)
            ent_all[:, i] = calculate_entropy(probs) / (np.log(self.num_classes))
            eng_all[:, i] = calculate_energy(logits)
            del probs, logits; gc.collect()
            
        return preds_all, conf_all, ent_all, eng_all

    def fit_optimal_combination(self, X_val, y_val, logger):
        logger.info("\n  [Phase 3] Meta-Optimization: Logistic Regression for Dynamic Ensemble Selection...")
        preds, conf, ent, eng = self.extract_metrics_batch(X_val)
        
        conf = np.nan_to_num(conf, nan=0.0, posinf=1e10, neginf=-1e10)
        ent = np.nan_to_num(ent, nan=0.0, posinf=1.0, neginf=0.0)
        eng = np.nan_to_num(eng, nan=0.0, posinf=0.0, neginf=-1e10)
        
        self.norm_params = {
            'conf_mean': np.mean(conf), 'conf_std': np.std(conf) + 1e-9,
            'ent_mean': np.mean(ent), 'ent_std': np.std(ent) + 1e-9,
            'eng_mean': np.mean(eng), 'eng_std': np.std(eng) + 1e-9,
        }
        
        conf_norm = (conf - self.norm_params['conf_mean']) / self.norm_params['conf_std']
        ent_norm = (ent - self.norm_params['ent_mean']) / self.norm_params['ent_std']
        eng_norm = (eng - self.norm_params['eng_mean']) / self.norm_params['eng_std']

        N, num_experts = preds.shape
        X_meta, y_meta = [], []
        
        for j in range(num_experts):
            features_j = np.column_stack((conf_norm[:, j], ent_norm[:, j], eng_norm[:, j]))
            labels_j = (preds[:, j] == y_val).astype(int)
            X_meta.append(features_j)
            y_meta.append(labels_j)
            
        X_meta = np.vstack(X_meta)
        y_meta = np.concatenate(y_meta)
        
        self.meta_clf = LogisticRegression(class_weight='balanced', random_state=self.seed)
        self.meta_clf.fit(X_meta, y_meta)
        
        coef = self.meta_clf.coef_[0]
        logger.info(f"  => Learned Coefficients: Conf(a)={coef[0]:.4f}, Ent(b)={coef[1]:.4f}, Eng(r)={coef[2]:.4f}")
        
        best_expert_indices = self._select_best_expert(conf_norm, ent_norm, eng_norm)
        best_acc = np.mean(preds[np.arange(N), best_expert_indices] == y_val)
        logger.info(f"  => Validation Acc via LR DES: {best_acc:.4f}")

    def _select_best_expert(self, conf_norm, ent_norm, eng_norm):
        N, num_experts = conf_norm.shape
        expert_reliability = np.zeros((N, num_experts))
        
        for j in range(num_experts):
            features_j = np.column_stack((conf_norm[:, j], ent_norm[:, j], eng_norm[:, j]))
            prob_correct = self.meta_clf.predict_proba(features_j)[:, 1]
            expert_reliability[:, j] = prob_correct
            
        return np.argmax(expert_reliability, axis=1)

    def predict_optimal(self, X_test):
        if len(self.experts) == 1:
            return np.argmax(self.experts[0].predict_proba_batch(X_test), axis=1)
            
        preds, conf, ent, eng = self.extract_metrics_batch(X_test)
        
        conf_norm = (conf - self.norm_params['conf_mean']) / self.norm_params['conf_std']
        ent_norm = (ent - self.norm_params['ent_mean']) / self.norm_params['ent_std']
        eng_norm = (eng - self.norm_params['eng_mean']) / self.norm_params['eng_std']
        
        best_expert_indices = self._select_best_expert(conf_norm, ent_norm, eng_norm)
        return preds[np.arange(X_test.shape[0]), best_expert_indices]

def main():
    start_time = time.time()
    gc.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .pkl data")
    parser.add_argument("--model", "-m", type=int, default=2, choices=[0, 1, 2],
                        help="0: baseline only, 1: proposed only, 2: both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20000)
    args = parser.parse_args()

    exp_dir = f"logs_exact_taxonomy_moe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logging(exp_dir)
    
    with open(args.data, "rb") as f: data = pickle.load(f)
    dataset_type = infer_dataset_type(args.data, data)
    logger.info(f"Dataset type: {dataset_type}")
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)
    X = np.clip(X, -1e12, 1e12).astype(np.float32)
    class_names = data['label_encoder'].classes_ if 'label_encoder' in data else [f"Class_{i}" for i in range(len(np.unique(y)))]
    num_classes = len(np.unique(y))
    del data; gc.collect()
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=args.seed)
    del X, y, X_temp, y_temp; gc.collect()

    models_dict = {}

    run_baseline = args.model in (0, 2)
    run_proposed = args.model in (1, 2)

    if run_baseline:
        t0 = time.time()
        logger.info("\n=== [Model S] Training Baseline ===")
        moe_s = TaxonomyDrivenMoE(num_classes=num_classes, dataset_type=dataset_type, device=DEVICE, seed=args.seed)
        moe_s.train_single(X_train, y_train)
        models_dict['S'] = moe_s
        logger.info(f"  [Model S] Training time: {time.time() - t0:.2f}s")

    if run_proposed:
        t0 = time.time()
        logger.info(f"\n=== [Model T] Training Explicit Taxonomy MoE (Dataset: {dataset_type}) ===")
        moe_t = TaxonomyDrivenMoE(num_classes=num_classes, dataset_type=dataset_type, device=DEVICE, seed=args.seed)
        moe_t.build_experts(X_train, y_train, logger, class_names)
        moe_t.fit_optimal_combination(X_val, y_val, logger)
        models_dict['T'] = moe_t
        logger.info(f"  [Model T] Training time: {time.time() - t0:.2f}s")
    
    del X_train, y_train, X_val, y_val; gc.collect()

    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)
    
    if run_baseline:
        t0 = time.time()
        base_preds = models_dict['S'].predict_optimal(X_test)
        acc_s = accuracy_score(y_test, base_preds)
        logger.info(f"\n--- [Model S] Baseline (Global Router Only) ---")
        logger.info(f"Accuracy: {acc_s:.4f} (eval time: {time.time() - t0:.2f}s)")
        logger.info("\n" + classification_report(y_test, base_preds, target_names=class_names, digits=4, zero_division=0))

    if run_proposed:
        t0 = time.time()
        prop_preds = models_dict['T'].predict_optimal(X_test)
        acc_t = accuracy_score(y_test, prop_preds)
        logger.info(f"\n--- [Model T] Explicit Taxonomy MoE ---")
        logger.info(f"Accuracy: {acc_t:.4f} (eval time: {time.time() - t0:.2f}s)")
        logger.info("\n" + classification_report(y_test, prop_preds, target_names=class_names, digits=4, zero_division=0))

    if not models_dict:
        logger.error("No model trained. Check --model argument.")
        return

    logger.info("\nConstructing Expert Metrics Matrix...")
    unique_classes = np.arange(num_classes)
    matrix_rows = []
    
    model_keys = sorted(models_dict.keys())
    for m_key in model_keys:
        model_obj = models_dict[m_key]
        model_expert_metrics = {}
        preds_all, conf_all, ent_all, eng_all = model_obj.extract_metrics_batch(X_test, batch_size=args.batch_size)
        
        for e_idx, expert in enumerate(model_obj.experts):
            preds = preds_all[:, e_idx]
            p_vec, r_vec, f1_vec, _ = precision_recall_fscore_support(y_test, preds, labels=unique_classes, zero_division=0)
            
            avg_conf, avg_ent, avg_eng = {}, {}, {}
            for c in unique_classes:
                mask = (y_test == c)
                if np.sum(mask) > 0:
                    avg_conf[c] = np.mean(conf_all[mask, e_idx])
                    avg_ent[c] = np.mean(ent_all[mask, e_idx])
                    avg_eng[c] = np.mean(eng_all[mask, e_idx])
                else:
                    avg_conf[c] = avg_ent[c] = avg_eng[c] = 0.0
            
            model_expert_metrics[e_idx] = {
                'p': p_vec, 'r': r_vec, 'f1': f1_vec,
                'conf': avg_conf, 'ent': avg_ent, 'eng': avg_eng,
                'assigned': expert.assigned_classes,
                'name': expert.name
            }
        model_obj.metrics_cache = model_expert_metrics

    ref_model = 'T' if 'T' in models_dict else 'S'
    num_experts_ref = len(models_dict[ref_model].experts)
    
    for c_idx in unique_classes:
        for e_idx in range(num_experts_ref):
            expert_name_ref = models_dict[ref_model].metrics_cache[e_idx]['name']
            row = {
                "Class_ID": c_idx,
                "Class_Name": class_names[c_idx] if c_idx < len(class_names) else f"Class_{c_idx}",
                "Expert_ID": e_idx,
                "Expert_Name": expert_name_ref
            }
            
            for m_key in model_keys:
                model_obj = models_dict[m_key]
                if m_key == 'S' and e_idx > 0 and len(model_obj.experts) == 1:
                    for k in ['P','R','F1','Conf','Ent','Eng']: row[f"{m_key}_{k}"] = "-"
                else:
                    actual_e_idx = 0 if (m_key == 'S' and len(model_obj.experts) == 1) else e_idx
                    if actual_e_idx >= len(model_obj.experts):
                        for k in ['P','R','F1','Conf','Ent','Eng']: row[f"{m_key}_{k}"] = "-"
                    else:
                        metrics = model_obj.metrics_cache[actual_e_idx]
                        assigned_mark = "(NA)" if c_idx not in metrics['assigned'] else ""
                        row[f"{m_key}_P"] = f"{metrics['p'][c_idx]:.2f}{assigned_mark}"
                        row[f"{m_key}_R"] = f"{metrics['r'][c_idx]:.2f}"
                        row[f"{m_key}_F1"] = f"{metrics['f1'][c_idx]:.2f}"
                        row[f"{m_key}_Conf"] = f"{metrics['conf'][c_idx]:.2f}"
                        row[f"{m_key}_Ent"] = f"{metrics['ent'][c_idx]:.2f}"
                        row[f"{m_key}_Eng"] = f"{metrics['eng'][c_idx]:.2f}"
            
            matrix_rows.append(row)

    df = pd.DataFrame(matrix_rows)
    metrics_cols = [f"{m}_{k}" for m in model_keys for k in ['P','R','F1','Conf','Ent','Eng']]
    df = df[["Class_ID", "Class_Name", "Expert_ID", "Expert_Name"] + metrics_cols]
    
    save_path = os.path.join(exp_dir, "exact_taxonomy_metrics.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"Saved: {save_path}")

    logger.info(f"\nTotal elapsed time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
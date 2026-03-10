#!/usr/bin/env python3
"""
CIC-IDS / UNSW-NB15 데이터셋 전처리 스크립트 (0306 수정본)
- [Fix] 빈 문자열 치환 버그 수정 (레이블 글자 쪼개짐 방지)
- [Fix] Data Leakage 방지 (Train 셋 기준으로만 Frequency Encoding 수행)
- [Fix] Infinity 처리 최적화 (불필요한 string 변환 제거로 메모리/정밀도 확보)
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def detect_dataset_type(data_dir):
    folder_name = os.path.basename(data_dir).lower()
    if 'unsw' in folder_name or 'nb15' in folder_name:
        print(f"  Dataset: UNSW-NB15")
        return 'unswnb15'
    elif '2017' in folder_name:
        print(f"  Dataset: CIC-IDS2017")
        return 'cic2017'
    elif '2018' in folder_name:
        print(f"  Dataset: CIC-IDS2018")
        return 'cic2018'
    else:
        raise ValueError("Folder name doesn't contain '2017', '2018', or 'unsw/nb15'")

def find_label_column(df, dataset_type=None):
    if dataset_type == 'unswnb15':
        if 'attack_cat' in df.columns: return 'attack_cat'
        for col in df.columns:
            if col.strip().lower() == 'attack_cat': return col
        return None
    
    for col in df.columns:
        if col.strip().lower() == 'label': return col
    return None

def remove_dataset_columns(X_df, dataset_type):
    columns_to_remove = {
        'cic2017': ['Destination Port'],
        'cic2018': ['Dst Port', 'Timestamp', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP'],
        'unswnb15': ['id', 'rate', 'label'] 
    }
    removed_cols = []
    for col_to_remove in columns_to_remove.get(dataset_type, []):
        for col in X_df.columns:
            if col.strip().lower() == col_to_remove.lower():
                X_df.drop(columns=[col], inplace=True)
                removed_cols.append(col)
                break
    if removed_cols:
        print(f"      Removed columns: {removed_cols}")

def downcast_dtypes(df):
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    if len(float_cols) > 0: df[float_cols] = df[float_cols].astype('float32')
    if len(int_cols) > 0: df[int_cols] = df[int_cols].astype('int32')
    return df

def preprocess_data(df, dataset_type):
    df = df.copy()
    print(f"      Processing {len(df)} rows...")
    
    if dataset_type == 'unswnb15':
        df = df.replace(['-', ' -', '- '], np.nan)
    
    label_col = find_label_column(df, dataset_type)
    if label_col and dataset_type in ['cic2017', 'cic2018']:
        header_mask = df[label_col].astype(str).str.strip() == 'Label'
        if header_mask.any():
            df = df[~header_mask].copy()
            print(f"      Removed {header_mask.sum()} 'Label' header row(s)")
    
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"      Removed duplicated columns")

    target_columns = [col for col in df.columns if col != label_col]
    unsw_nominal_cols = set(['proto', 'state', 'service']) if dataset_type == 'unswnb15' else set()

    for col in target_columns:
        if dataset_type == 'unswnb15' and col in unsw_nominal_cols:
            continue
        
        # [수정 1] 불필요한 string 변환 제거. pd.to_numeric이 'Infinity'를 알아서 np.inf로 변환함.
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 변환 불가능한 문자열(NaN으로 바뀐 놈들) 행 제거
    numeric_df = df[target_columns].select_dtypes(include=[np.number])
    cols_to_check = [c for c in numeric_df.columns if c not in unsw_nominal_cols]
        
    initial_len = len(df)
    df = df.dropna(subset=cols_to_check).copy()
    if len(df) < initial_len:
        print(f"      Removed {initial_len - len(df)} rows with invalid non-numeric values")

    # [수정 2] 무한대(np.inf)를 안전하게 처리
    safe_cap = 1e12
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if label_col:
        numeric_cols = [col for col in numeric_cols if col != label_col]
    
    if len(numeric_cols) > 0:
        # np.isinf를 찾아 safe_cap으로 클리핑
        df[numeric_cols] = df[numeric_cols].clip(lower=-safe_cap, upper=safe_cap)
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def preprocess_cicids_dataset(data_dir, output_path):
    print(f"\n{'='*80}\nNetwork Intrusion Dataset Preprocessing (0306 Fixed)\n{'='*80}")
    
    dataset_type = detect_dataset_type(data_dir)
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files: raise ValueError(f"No CSV files found in {data_dir}")
    
    all_data = []
    file_groups = []
    
    for file_idx, file_name in enumerate(csv_files):
        print(f"\n[{file_idx+1}/{len(csv_files)}] Loading: {file_name}")
        try:
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path, low_memory=False)
            df = downcast_dtypes(df)
            
            remove_dataset_columns(df, dataset_type)
            df = preprocess_data(df, dataset_type)
            
            label_col = find_label_column(df, dataset_type)
            if label_col:
                # [수정 3] '' -> '-' 치환 버그를 ' ' -> '-' 로 변경하여 정상 작동하게 함
                df[label_col] = df[label_col].astype(str).str.lower().str.strip()
                df[label_col] = df[label_col].str.replace(' ', '-', regex=False)
                df[label_col] = df[label_col].str.replace('–', '-', regex=False)
                
                if dataset_type == 'unswnb15':
                    if 'training' in file_name.lower(): file_group = np.full(len(df), 0, dtype=np.int8)
                    elif 'testing' in file_name.lower(): file_group = np.full(len(df), 1, dtype=np.int8)
                    else: file_group = np.full(len(df), file_idx, dtype=np.int8)
                else:
                    file_group = np.full(len(df), file_idx, dtype=np.int8)
                
                all_data.append(df)
                file_groups.append(file_group)
            else:
                print(f"      WARNING: No Label column found, skipping.")
        except Exception as e:
            print(f"      ERROR: {e}")
            
    print(f"\n{'='*80}\nCombining and Encoding...")
    combined_df = pd.concat(all_data, ignore_index=True)
    file_groups = np.concatenate(file_groups)
    combined_df = downcast_dtypes(combined_df)
    
    # [수정 4] Data Leakage 방지: Train 셋에서만 Frequency Encoding 계산 후 전체 매핑
    if dataset_type == 'unswnb15':
        cat_cols = ['proto', 'state', 'service']
        train_mask = (file_groups == 0) # Train 데이터만 추출
        for col in cat_cols:
            if col in combined_df.columns:
                print(f"Applying Frequency Encoding to '{col}' (Fitted on Train set only)...")
                # Train 셋 기준으로 매핑 테이블 생성
                freq_encoding = combined_df.loc[train_mask, col].value_counts(normalize=True)
                # Test 셋에 적용 (Train에 없던 카테고리는 0으로 처리)
                combined_df[col] = combined_df[col].map(freq_encoding).fillna(0.0).astype('float32')

    label_col = find_label_column(combined_df, dataset_type)
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    combined_df['Label_encoded'] = label_encoder.fit_transform(combined_df[label_col])
    
    unique_labels, counts = np.unique(combined_df['Label_encoded'].values, return_counts=True)
    print(f"\nClass distribution ({len(unique_labels)} classes):")
    for label_id, count in zip(unique_labels, counts):
        class_name = label_encoder.inverse_transform([label_id])[0]
        pct = (count / len(combined_df)) * 100
        print(f"  {class_name:30s}: {count:8d} ({pct:6.2f}%)")
    
    feature_columns = [col for col in combined_df.columns if col not in [label_col, 'Label_encoded']]
    X = combined_df[feature_columns].values
    y = combined_df['Label_encoded'].values.astype('int32')
    
    for col_idx, col in enumerate(feature_columns):
        if col in ['Init_Win_bytes_forward', 'Init_Win_bytes_backward']:
            X[:, col_idx] = np.where(X[:, col_idx] == -1, 0, X[:, col_idx])
            
    print(f"\nFeature matrix shape: {X.shape}, dtype: {X.dtype}")
    
    print(f"\n{'='*80}\nSaving to pickle...")
    save_dict = {
        'X': X, 'y': y, 'label_encoder': label_encoder,
        'feature_columns': feature_columns, 'dataset_type': dataset_type,
        'class_names': label_encoder.classes_.tolist(), 'file_groups': file_groups
    }
    if dataset_type == 'unswnb15':
        save_dict['train_indices'] = np.where(file_groups == 0)[0]
        save_dict['test_indices'] = np.where(file_groups == 1)[0]
        
    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)\nDone.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()
    
    if args.output is None:
        ds_name = os.path.basename(args.data_dir.rstrip('/\\'))
        args.output = f"../{ds_name}_preprocessed_0306.pkl"
        
    preprocess_cicids_dataset(args.data_dir, args.output)

if __name__ == "__main__":
    main()
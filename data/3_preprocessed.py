#!/usr/bin/env python3
"""
CIC-IDS / UNSW-NB15 데이터셋 전처리 스크립트 (Fixed Version)

기본 전처리 항목 (순서):
  1. 식별자 컬럼 삭제: Flow ID, Src/Dst IP, Src/Dst Port, Timestamp (또는 id)
  2. NaN 포함 feature/샘플 삭제 (수치형 컬럼 기준)
 10. 중복 행 삭제
  4. 중복 컬럼 삭제
  8. 반복 헤더 행 제거
  5. 클래스 이름 정제 (소문자, 공백→하이픈)

  수치형 아닌 컬럼 동적 감지 후:
  - string feature → 숫자 변환 (Frequency Encoding)
  - 6. 무한대(Infinity) 처리, 7. 잘못된 값(Init Win Byts=-1 → 0)
  9. 상수(zero-variance) 컬럼 삭제
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
    """식별자/비예측 컬럼 삭제. Flow ID, IP, Port, Timestamp 등."""
    columns_to_remove = {
        'cic2017': ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Timestamp'],
        'cic2018': ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp'],
        # UNSW: id=식별자, rate=분류에 불필요, label=0/1 정상구분(attack_cat 사용)
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
    
    label_col = find_label_column(df, dataset_type)
    
    # [수정포인트 1] UNSW-NB15의 attack_cat 내 결측치(NaN) 및 '-' 기호는 정상(normal) 트래픽을 의미함
    if dataset_type == 'unswnb15' and label_col:
        df[label_col] = df[label_col].fillna('normal')
        df[label_col] = df[label_col].replace(['-', ' -', '- '], 'normal')
    
    # 4. 중복 컬럼 삭제
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"      Removed duplicated columns")
    
    # 8. 반복 헤더 행 제거
    if label_col:
        if dataset_type in ['cic2017', 'cic2018']:
            header_mask = df[label_col].astype(str).str.strip().str.lower() == 'label'
        else:
            header_mask = df[label_col].astype(str).str.strip().str.lower() == 'attack_cat'
        if header_mask.any():
            df = df[~header_mask].copy()
            print(f"      Removed {header_mask.sum()} header row(s)")

    target_columns = [col for col in df.columns if col != label_col]
    
    # [수정포인트 2] 확실한 문자열(Categorical) 컬럼 감지
    object_cols = list(df[target_columns].select_dtypes(include=['object', 'category']).columns)
    if object_cols:
        print(f"      Non-numeric columns (will encode later): {object_cols}")

    for col in target_columns:
        if col in object_cols:
            # 문자열 피처의 결측치 및 '-' 기호를 'Unknown'으로 안전하게 매핑 (NaN 방지)
            df[col] = df[col].replace(['-', ' -', '- '], 'Unknown')
            df[col] = df[col].fillna('Unknown')
            continue
            
        # 숫자형이어야 하는데 object 형식으로 꼬인 경우만 강제 변환
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. NaN 포함 행 제거 (수치형 컬럼들만 검사하여 애먼 문자열 데이터 삭제 방지)
    numeric_df = df[target_columns].select_dtypes(include=[np.number])
    cols_to_check = list(numeric_df.columns)
    initial_len = len(df)
    df = df.dropna(subset=cols_to_check).copy()
    if len(df) < initial_len:
        print(f"      Removed {initial_len - len(df)} rows with NaN")
    
    return df

def preprocess_cicids_dataset(data_dir, output_path):
    print(f"\n{'='*80}\nNetwork Intrusion Dataset Preprocessing (Fixed)\n{'='*80}")
    
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
    
    label_col = find_label_column(combined_df, dataset_type)
    
    # # 10. 중복 행 삭제
    # before_dup = len(combined_df)
    # dup_mask = ~combined_df.duplicated()
    # combined_df = combined_df.loc[dup_mask].reset_index(drop=True)
    # file_groups = file_groups[dup_mask]
    # if before_dup - len(combined_df) > 0:
    #     print(f"      Removed {before_dup - len(combined_df):,} duplicate rows")
    
    # 4. 중복 컬럼 (combined 기준)
    if combined_df.columns.duplicated().any():
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        print(f"      Removed duplicated columns")
    
    # 8. 반복 헤더 행 (combined에서 혹시 남은 경우)
    if label_col:
        if dataset_type in ['cic2017', 'cic2018']:
            header_mask = combined_df[label_col].astype(str).str.strip().str.lower() == 'label'
        else:
            header_mask = combined_df[label_col].astype(str).str.strip().str.lower() == 'attack_cat'
        if header_mask.any():
            combined_df = combined_df.loc[~header_mask].reset_index(drop=True)
            file_groups = file_groups[~header_mask.values]
            print(f"      Removed {header_mask.sum()} header row(s)")
    
    feature_columns = [c for c in combined_df.columns if c not in [label_col, 'Label_encoded']]
    
    # 5. 클래스 이름 정제
    combined_df[label_col] = combined_df[label_col].astype(str).str.lower().str.strip()
    combined_df[label_col] = combined_df[label_col].str.replace(' ', '-', regex=False)
    combined_df[label_col] = combined_df[label_col].str.replace('–', '-', regex=False)
    
    # [수정포인트 3] 안전한 수치형 변환 (Frequency Encoding)
    cat_cols = list(combined_df[feature_columns].select_dtypes(include=['object', 'category']).columns)
    if cat_cols:
        print(f"      Non-numeric columns (Frequency Encoding): {cat_cols}")
        fit_mask = (file_groups == 0) if dataset_type == 'unswnb15' else np.ones(len(combined_df), dtype=bool)
        for col in cat_cols:
            freq_encoding = combined_df.loc[fit_mask, col].value_counts(normalize=True)
            combined_df[col] = combined_df[col].map(freq_encoding).fillna(0.0).astype('float32')
    
    # 6. 무한대 처리, 7. 잘못된 값 처리
    numeric_cols = combined_df[feature_columns].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        safe_cap = 1e12
        combined_df[numeric_cols] = combined_df[numeric_cols].clip(lower=-safe_cap, upper=safe_cap)
        combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0)
    
    # 9. 상수(zero-variance) 컬럼 삭제
    for col in feature_columns:
        if col in combined_df.columns and not pd.api.types.is_numeric_dtype(combined_df[col]):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
    var_ser = combined_df[feature_columns].var()
    zero_var_cols = var_ser[var_ser == 0].index.tolist()
    if zero_var_cols:
        feature_columns = [c for c in feature_columns if c not in zero_var_cols]
        print(f"      Removed {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")

    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    combined_df['Label_encoded'] = label_encoder.fit_transform(combined_df[label_col])
    
    unique_labels, counts = np.unique(combined_df['Label_encoded'].values, return_counts=True)
    print(f"\nClass distribution ({len(unique_labels)} classes):")
    for label_id, count in zip(unique_labels, counts):
        class_name = label_encoder.inverse_transform([label_id])[0]
        pct = (count / len(combined_df)) * 100
        print(f"  {class_name:30s}: {count:8d} ({pct:6.2f}%)")
    
    X = combined_df[feature_columns].values
    y = combined_df['Label_encoded'].values.astype('int32')
    
    init_win_cols = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward',
                     'Init Fwd Win Byts', 'Init Bwd Win Byts']
    for col_idx, col in enumerate(feature_columns):
        if col in init_win_cols:
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
        args.output = f"../{ds_name}_preprocessed.pkl"
        
    preprocess_cicids_dataset(args.data_dir, args.output)

if __name__ == "__main__":
    main()
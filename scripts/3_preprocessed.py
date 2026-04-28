#!/usr/bin/env python3
"""
Preprocess CIC-IDS / UNSW-NB15 style CSV datasets into a pickle file.

Key points:
1. Remove identifier / leakage-prone columns
2. Clean repeated headers and NaNs
3. Frequency-encode categorical features
4. Store split metadata when it is meaningful
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def detect_dataset_type(data_dir):
    folder_name = os.path.basename(data_dir).lower()
    if "nf-unsw" in folder_name or "nf_unsw" in folder_name or "nfunsw" in folder_name:
        print("  Dataset: NF-UNSW-NB15")
        return "nfunswnb15"
    if "unsw" in folder_name or "nb15" in folder_name:
        print("  Dataset: UNSW-NB15")
        return "unswnb15"
    if "2017" in folder_name:
        print("  Dataset: CIC-IDS2017")
        return "cic2017"
    if "2018" in folder_name:
        print("  Dataset: CIC-IDS2018")
        return "cic2018"
    raise ValueError("Folder name doesn't contain '2017', '2018', or 'unsw/nb15'")


def find_label_column(df, dataset_type=None):
    if dataset_type == "nfunswnb15":
        if "Attack" in df.columns:
            return "Attack"
        for col in df.columns:
            if col.strip().lower() == "attack":
                return col
        return None

    if dataset_type == "unswnb15":
        if "attack_cat" in df.columns:
            return "attack_cat"
        for col in df.columns:
            if col.strip().lower() == "attack_cat":
                return col
        return None

    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    return None


def find_timestamp_column(df, dataset_type=None):
    candidates = ["Timestamp", "timestamp", "Time", "time", "Datetime", "datetime", "Date", "date"]
    for cand in candidates:
        for col in df.columns:
            if col.strip().lower() == cand.lower():
                return col
    return None


def parse_timestamp_series(ts_series):
    ts = pd.to_datetime(ts_series, errors="coerce")
    if ts.isna().all():
        return np.arange(len(ts_series), dtype=np.int64), False

    valid_min = ts.dropna().min()
    ts = ts.fillna(valid_min)
    return ts.astype("int64").to_numpy(), True


def remove_dataset_columns(df, dataset_type):
    columns_to_remove = {
        "cic2017": ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"],
        "cic2018": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"],
        "unswnb15": ["id", "rate", "label"],
        "nfunswnb15": ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"],
    }
    removed_cols = []
    for col_to_remove in columns_to_remove.get(dataset_type, []):
        for col in df.columns:
            if col.strip().lower() == col_to_remove.lower():
                df.drop(columns=[col], inplace=True)
                removed_cols.append(col)
                break
    if removed_cols:
        print(f"      Removed columns: {removed_cols}")


def downcast_dtypes(df):
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype("int32")
    return df


def preprocess_data(df, dataset_type, timestamp_col=None):
    df = df.copy()
    print(f"      Processing {len(df)} rows...")

    timestamps = None
    if timestamp_col is not None and timestamp_col in df.columns:
        timestamps = df[timestamp_col].copy()

    label_col = find_label_column(df, dataset_type)

    if dataset_type in ["unswnb15", "nfunswnb15"] and label_col:
        df[label_col] = df[label_col].fillna("normal")
        df[label_col] = df[label_col].replace(["-", " -", "- "], "normal")

    if df.columns.duplicated().any():
        dedup_mask = ~df.columns.duplicated()
        df = df.loc[:, dedup_mask]
        print("      Removed duplicated columns")
        if timestamp_col is not None and timestamp_col not in df.columns:
            timestamp_col = find_timestamp_column(df, dataset_type)

    if label_col:
        if dataset_type in ["cic2017", "cic2018"]:
            header_mask = df[label_col].astype(str).str.strip().str.lower() == "label"
        else:
            header_mask = df[label_col].astype(str).str.strip().str.lower().isin(["attack_cat", "attack"])

        if header_mask.any():
            keep_mask = ~header_mask
            df = df.loc[keep_mask].copy()
            if timestamps is not None:
                timestamps = timestamps.loc[keep_mask].copy()
            print(f"      Removed {header_mask.sum()} header row(s)")

    target_columns = [col for col in df.columns if col != label_col]
    object_cols = list(df[target_columns].select_dtypes(include=["object", "category"]).columns)
    if object_cols:
        print(f"      Non-numeric columns (will encode later): {object_cols}")

    for col in target_columns:
        if col in object_cols:
            df[col] = df[col].replace(["-", " -", "- "], "Unknown")
            df[col] = df[col].fillna("Unknown")
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_df = df[target_columns].select_dtypes(include=[np.number])
    cols_to_check = list(numeric_df.columns)
    initial_len = len(df)
    if len(cols_to_check) > 0:
        keep_mask = ~df[cols_to_check].isna().any(axis=1)
        df = df.loc[keep_mask].copy()
        if timestamps is not None:
            timestamps = timestamps.loc[keep_mask].copy()

    if len(df) < initial_len:
        print(f"      Removed {initial_len - len(df)} rows with NaN")

    df = df.reset_index(drop=True)
    if timestamps is not None:
        timestamps = timestamps.reset_index(drop=True)
    return df, timestamps


def preprocess_cicids_dataset(data_dir, output_path):
    print(f"\n{'=' * 80}\nNetwork Intrusion Dataset Preprocessing (Fixed)\n{'=' * 80}")

    dataset_type = detect_dataset_type(data_dir)
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    all_data = []
    file_groups = []
    all_timestamps = []
    has_real_timestamps = False

    for file_idx, file_name in enumerate(csv_files):
        print(f"\n[{file_idx + 1}/{len(csv_files)}] Loading: {file_name}")
        try:
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path, low_memory=False)
            df = downcast_dtypes(df)

            timestamp_col = find_timestamp_column(df, dataset_type)
            if timestamp_col:
                print(f"      Timestamp column detected: {timestamp_col}")
            else:
                print("      Timestamp column not found in this file.")

            remove_dataset_columns(df, dataset_type)
            df, timestamps = preprocess_data(df, dataset_type, timestamp_col=timestamp_col)

            label_col = find_label_column(df, dataset_type)
            if not label_col:
                print("      WARNING: No Label column found, skipping.")
                continue

            if dataset_type == "unswnb15":
                if "training" in file_name.lower():
                    file_group = np.full(len(df), 0, dtype=np.int16)
                elif "testing" in file_name.lower():
                    file_group = np.full(len(df), 1, dtype=np.int16)
                else:
                    file_group = np.full(len(df), file_idx, dtype=np.int16)
            else:
                file_group = np.full(len(df), file_idx, dtype=np.int16)

            if timestamps is None:
                timestamps = pd.Series(np.arange(len(df), dtype=np.int64))

            ts_int, parsed_ok = parse_timestamp_series(timestamps)
            if parsed_ok:
                print("      Parsed timestamps successfully")
                has_real_timestamps = True
            else:
                print("      Failed to parse timestamps; using row-order fallback")

            all_data.append(df)
            file_groups.append(file_group)
            all_timestamps.append(ts_int)
        except Exception as e:
            print(f"      ERROR: {e}")

    print(f"\n{'=' * 80}\nCombining and Encoding...")
    combined_df = pd.concat(all_data, ignore_index=True)
    file_groups = np.concatenate(file_groups)
    timestamps = np.concatenate(all_timestamps)

    combined_df = downcast_dtypes(combined_df)
    label_col = find_label_column(combined_df, dataset_type)

    if combined_df.columns.duplicated().any():
        dedup_mask = ~combined_df.columns.duplicated()
        combined_df = combined_df.loc[:, dedup_mask]
        print("      Removed duplicated columns")

    if label_col:
        if dataset_type in ["cic2017", "cic2018"]:
            header_mask = combined_df[label_col].astype(str).str.strip().str.lower() == "label"
        else:
            header_mask = combined_df[label_col].astype(str).str.strip().str.lower().isin(["attack_cat", "attack"])

        if header_mask.any():
            keep_mask = ~header_mask.values
            combined_df = combined_df.loc[keep_mask].reset_index(drop=True)
            file_groups = file_groups[keep_mask]
            timestamps = timestamps[keep_mask]
            print(f"      Removed {header_mask.sum()} header row(s)")

    feature_columns = [c for c in combined_df.columns if c not in [label_col, "Label_encoded"]]

    combined_df[label_col] = combined_df[label_col].astype(str).str.lower().str.strip()
    combined_df[label_col] = combined_df[label_col].str.replace(" ", "-", regex=False)
    combined_df[label_col] = combined_df[label_col].str.replace("–", "-", regex=False)
    combined_df[label_col] = combined_df[label_col].replace({"benign": "normal"})

    cat_cols = list(combined_df[feature_columns].select_dtypes(include=["object", "category"]).columns)
    if cat_cols:
        print(f"      Non-numeric columns (Frequency Encoding): {cat_cols}")
        fit_mask = (file_groups == 0) if dataset_type == "unswnb15" else np.ones(len(combined_df), dtype=bool)
        for col in cat_cols:
            freq_encoding = combined_df.loc[fit_mask, col].value_counts(normalize=True)
            combined_df[col] = combined_df[col].map(freq_encoding).fillna(0.0).astype("float32")

    numeric_cols = combined_df[feature_columns].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        safe_cap = 1e12
        combined_df[numeric_cols] = combined_df[numeric_cols].clip(lower=-safe_cap, upper=safe_cap)
        combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0)

    for col in feature_columns:
        if col in combined_df.columns and not pd.api.types.is_numeric_dtype(combined_df[col]):
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").fillna(0)

    var_ser = combined_df[feature_columns].var()
    zero_var_cols = var_ser[var_ser == 0].index.tolist()
    if zero_var_cols:
        feature_columns = [c for c in feature_columns if c not in zero_var_cols]
        print(f"      Removed {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")

    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    combined_df["Label_encoded"] = label_encoder.fit_transform(combined_df[label_col])

    unique_labels, counts = np.unique(combined_df["Label_encoded"].values, return_counts=True)
    print(f"\nClass distribution ({len(unique_labels)} classes):")
    for label_id, count in zip(unique_labels, counts):
        class_name = label_encoder.inverse_transform([label_id])[0]
        pct = (count / len(combined_df)) * 100
        print(f"  {class_name:30s}: {count:8d} ({pct:6.2f}%)")

    X = combined_df[feature_columns].values
    y = combined_df["Label_encoded"].values.astype("int32")

    init_win_cols = [
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "Init Fwd Win Byts",
        "Init Bwd Win Byts",
    ]
    for col_idx, col in enumerate(feature_columns):
        if col in init_win_cols:
            X[:, col_idx] = np.where(X[:, col_idx] == -1, 0, X[:, col_idx])

    print(f"\nFeature matrix shape: {X.shape}, dtype: {X.dtype}")
    if has_real_timestamps:
        print(f"Timestamps shape: {timestamps.shape}, dtype: {timestamps.dtype}")
        if len(timestamps) != len(y):
            raise ValueError(f"Length mismatch: len(timestamps)={len(timestamps)} vs len(y)={len(y)}")

    print(f"\n{'=' * 80}\nSaving to pickle...")
    save_dict = {
        "X": X,
        "y": y,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "dataset_type": dataset_type,
        "class_names": label_encoder.classes_.tolist(),
        "file_groups": file_groups,
    }
    if dataset_type == "unswnb15":
        save_dict["train_indices"] = np.where(file_groups == 0)[0]
        save_dict["test_indices"] = np.where(file_groups == 1)[0]
    elif len(np.unique(file_groups)) > 1:
        save_dict["groups"] = file_groups

    if has_real_timestamps:
        save_dict["timestamps"] = timestamps

    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    if args.output is None:
        ds_name = os.path.basename(args.data_dir.rstrip("/\\"))
        args.output = f"../{ds_name}_preprocessed_t.pkl"

    preprocess_cicids_dataset(args.data_dir, args.output)


if __name__ == "__main__":
    main()

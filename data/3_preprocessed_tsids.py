#!/usr/bin/env python3
"""
TS-IDS-aligned preprocessing for flat CIC-IDS / UNSW style datasets.

This keeps the original flat feature representation used in this repo, while
adopting the TS-IDS split semantics as closely as possible for flat features:
1. For NF datasets, create TS-IDS-style 5-fold CV train/val/test assignments
2. Fit categorical encodings using only the non-test portion of the active fold
3. Save explicit split indices so downstream code can reuse the same test set
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


TSIDS_SPLIT_RATIO = (0.6, 0.2, 0.2)
TSIDS_SEED = 2022
TSIDS_NUM_FOLDS = 5


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


def find_aux_binary_label_column(df, dataset_type=None):
    if dataset_type == "nfunswnb15":
        for preferred in ["Label", "label"]:
            if preferred in df.columns:
                return preferred
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


def normalize_string_labels(series, benign_to_normal=True):
    out = series.astype(str).str.lower().str.strip()
    out = out.str.replace(" ", "-", regex=False)
    out = out.str.replace("??, ", "-", regex=False)
    out = out.replace({"-": "normal"})
    if benign_to_normal:
        out = out.replace({"benign": "normal"})
    return out


def remove_dataset_columns(df, dataset_type, keep_aux_binary_label=False):
    columns_to_remove = {
        "cic2017": ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"],
        "cic2018": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"],
        "unswnb15": ["id", "rate", "label"],
        "nfunswnb15": ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
    }
    if not keep_aux_binary_label:
        columns_to_remove["nfunswnb15"] = columns_to_remove["nfunswnb15"] + ["Label"]

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


def get_tvt_indices(labels, tvt_ratio=TSIDS_SPLIT_RATIO, seed=TSIDS_SEED):
    labels = np.asarray(labels)
    if abs(sum(tvt_ratio) - 1.0) > 1e-9:
        raise ValueError("Incorrect train/val/test ratio")

    val_ratio = tvt_ratio[1]
    test_ratio = tvt_ratio[2]
    n_samples = len(labels)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    all_idx = np.arange(n_samples, dtype=np.int32)

    train_idx, test_idx = train_test_split(
        all_idx, test_size=n_test, shuffle=True, stratify=labels, random_state=seed
    )
    remain_labels = labels[train_idx]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=n_val, shuffle=True, stratify=remain_labels, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def build_tvt_series(n_samples, train_idx, val_idx, test_idx):
    tvt = np.full(n_samples, "Other", dtype=object)
    tvt[test_idx] = "test"
    tvt[val_idx] = "val"
    tvt[train_idx] = "train"
    return tvt


def preprocess_data(df, dataset_type, timestamp_col=None):
    df = df.copy()
    print(f"      Processing {len(df)} rows...")

    timestamps = None
    if timestamp_col is not None and timestamp_col in df.columns:
        timestamps = df[timestamp_col].copy()

    label_col = find_label_column(df, dataset_type)
    aux_label_col = find_aux_binary_label_column(df, dataset_type)

    for col in [label_col, aux_label_col]:
        if dataset_type in ["unswnb15", "nfunswnb15"] and col:
            df[col] = df[col].fillna("normal")
            df[col] = df[col].replace(["-", " -", "- "], "normal")

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

    target_columns = [col for col in df.columns if col not in [label_col, aux_label_col]]
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


def preprocess_cicids_dataset(data_dir, output_path, active_fold=0, num_folds=TSIDS_NUM_FOLDS):
    print(f"\n{'=' * 80}\nTS-IDS Aligned Network Dataset Preprocessing\n{'=' * 80}")

    dataset_type = detect_dataset_type(data_dir)
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    all_data = []
    file_groups = []
    all_timestamps = []
    has_real_timestamps = False
    keep_aux_binary_label = dataset_type == "nfunswnb15"

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

            remove_dataset_columns(df, dataset_type, keep_aux_binary_label=keep_aux_binary_label)
            df, timestamps = preprocess_data(df, dataset_type, timestamp_col=timestamp_col)

            label_col = find_label_column(df, dataset_type)
            if not label_col:
                print("      WARNING: No label column found, skipping.")
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
    aux_label_col = find_aux_binary_label_column(combined_df, dataset_type)

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

    combined_df[label_col] = normalize_string_labels(combined_df[label_col], benign_to_normal=True)
    if aux_label_col:
        combined_df[aux_label_col] = normalize_string_labels(combined_df[aux_label_col], benign_to_normal=False)

    split_metadata = {}
    fit_mask = np.ones(len(combined_df), dtype=bool)

    if dataset_type == "nfunswnb15":
        if not 0 <= active_fold < num_folds:
            raise ValueError(f"active_fold must be in [0, {num_folds - 1}]")

        for fold in range(num_folds):
            fold_seed = TSIDS_SEED + fold
            attack_train_idx, attack_val_idx, attack_test_idx = get_tvt_indices(
                combined_df[label_col].values, tvt_ratio=TSIDS_SPLIT_RATIO, seed=fold_seed
            )
            attack_tvt_name = f"Attack_tvt_fold_{fold}"
            combined_df[attack_tvt_name] = build_tvt_series(
                len(combined_df), attack_train_idx, attack_val_idx, attack_test_idx
            )
            split_metadata[f"attack_train_indices_fold_{fold}"] = attack_train_idx
            split_metadata[f"attack_val_indices_fold_{fold}"] = attack_val_idx
            split_metadata[f"attack_test_indices_fold_{fold}"] = attack_test_idx
            split_metadata[attack_tvt_name] = combined_df[attack_tvt_name].values

            if fold == active_fold:
                combined_df["Attack_tvt"] = combined_df[attack_tvt_name].values
                split_metadata["attack_train_indices"] = attack_train_idx
                split_metadata["attack_val_indices"] = attack_val_idx
                split_metadata["attack_test_indices"] = attack_test_idx
                split_metadata["attack_tvt"] = combined_df["Attack_tvt"].values
                split_metadata["train_indices"] = attack_train_idx
                split_metadata["val_indices"] = attack_val_idx
                split_metadata["test_indices"] = attack_test_idx
                fit_mask = combined_df["Attack_tvt"].values != "test"

        if aux_label_col:
            for fold in range(num_folds):
                fold_seed = TSIDS_SEED + fold
                label_train_idx, label_val_idx, label_test_idx = get_tvt_indices(
                    combined_df[aux_label_col].values, tvt_ratio=TSIDS_SPLIT_RATIO, seed=fold_seed
                )
                label_tvt_name = f"Label_tvt_fold_{fold}"
                combined_df[label_tvt_name] = build_tvt_series(
                    len(combined_df), label_train_idx, label_val_idx, label_test_idx
                )
                split_metadata[f"label_train_indices_fold_{fold}"] = label_train_idx
                split_metadata[f"label_val_indices_fold_{fold}"] = label_val_idx
                split_metadata[f"label_test_indices_fold_{fold}"] = label_test_idx
                split_metadata[label_tvt_name] = combined_df[label_tvt_name].values

                if fold == active_fold:
                    combined_df["Label_tvt"] = combined_df[label_tvt_name].values
                    split_metadata["label_train_indices"] = label_train_idx
                    split_metadata["label_val_indices"] = label_val_idx
                    split_metadata["label_test_indices"] = label_test_idx
                    split_metadata["label_tvt"] = combined_df["Label_tvt"].values

    feature_columns = [
        c
        for c in combined_df.columns
        if c not in [label_col, aux_label_col, "Label_encoded", "Attack_tvt", "Label_tvt"]
        and not c.startswith("Attack_tvt_fold_")
        and not c.startswith("Label_tvt_fold_")
    ]

    cat_cols = list(combined_df[feature_columns].select_dtypes(include=["object", "category"]).columns)
    if cat_cols:
        print(f"      Non-numeric columns (Frequency Encoding, fit on non-test split): {cat_cols}")
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

    if dataset_type == "nfunswnb15":
        test_dist = combined_df.iloc[split_metadata["test_indices"]][label_col].value_counts(normalize=True).sort_index()
        print(f"\nTS-IDS Attack test distribution (active fold={active_fold}):")
        for class_name, pct in test_dist.items():
            print(f"  {class_name:30s}: {pct * 100:6.2f}%")

    X = combined_df[feature_columns].values
    y = combined_df["Label_encoded"].values.astype("int32")

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
        "split_strategy": "tsids_cv_get_tvt" if dataset_type == "nfunswnb15" else "original",
        "split_seed": TSIDS_SEED if dataset_type == "nfunswnb15" else None,
        "split_ratio": TSIDS_SPLIT_RATIO if dataset_type == "nfunswnb15" else None,
        "num_folds": num_folds if dataset_type == "nfunswnb15" else None,
        "active_fold": active_fold if dataset_type == "nfunswnb15" else None,
    }
    if dataset_type == "unswnb15":
        save_dict["train_indices"] = np.where(file_groups == 0)[0]
        save_dict["test_indices"] = np.where(file_groups == 1)[0]
    elif len(np.unique(file_groups)) > 1:
        save_dict["groups"] = file_groups

    if has_real_timestamps:
        save_dict["timestamps"] = timestamps

    if aux_label_col:
        save_dict["binary_label_name"] = aux_label_col
        save_dict["binary_label_values"] = combined_df[aux_label_col].values

    save_dict.update(split_metadata)

    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--fold", type=int, default=0, help="Active TS-IDS CV fold to materialize")
    parser.add_argument("--num_folds", type=int, default=TSIDS_NUM_FOLDS, help="Number of TS-IDS CV folds")
    args = parser.parse_args()

    if args.output is None:
        ds_name = os.path.basename(args.data_dir.rstrip("/\\"))
        args.output = f"../{ds_name}_preprocessed_tsids.pkl"

    preprocess_cicids_dataset(args.data_dir, args.output, active_fold=args.fold, num_folds=args.num_folds)


if __name__ == "__main__":
    main()

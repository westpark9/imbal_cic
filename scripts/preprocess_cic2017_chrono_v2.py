#!/usr/bin/env python3
"""
Build a chronological CIC-IDS2017 pickle for OOD/TTA v2.

Scenario:
  known/source = Monday + Tuesday + Wednesday + Thursday
  OOD target   = Friday

Source rows are split into train/val/test_known. Friday rows are held out as
test_ood. This lets v2 measure known classification, known tail classification
from the per-class table, and Friday OOD rejection separately.

The output keeps original multiclass labels plus explicit split indices and
file/day metadata. The v2 runner evaluates predictions against the original
CIC classes and reports target-day OOD reject diagnostics separately.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


SOURCE_DAY_TOKENS = ("monday", "tuesday", "wednesday", "thursday")
OOD_DAY_TOKENS = ("friday",)


def find_label_column(df):
    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    raise ValueError("Could not find CIC-IDS2017 Label column")


def remove_leakage_columns(df):
    remove_names = {"flow id", "source ip", "source port", "destination ip", "timestamp"}
    removed = []
    for col in list(df.columns):
        if col.strip().lower() in remove_names:
            df.drop(columns=[col], inplace=True)
            removed.append(col)
    return removed


def normalize_label(s):
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("–", "-")
        .replace("\u2013", "-")
    )


def normalize_web_label(s):
    label = normalize_label(s)
    label = label.replace("web-attack-\ufffd-", "web-attack-")
    label = label.replace("web-attack---", "web-attack-")
    label = label.replace("web-attack--", "web-attack-")
    return {"benign": "normal"}.get(label, label)


def read_one_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    label_col = find_label_column(df)
    header_mask = df[label_col].astype(str).str.strip().str.lower() == "label"
    if header_mask.any():
        df = df.loc[~header_mask].copy()
    remove_leakage_columns(df)
    label_col = find_label_column(df)
    df[label_col] = df[label_col].map(normalize_web_label)

    feature_columns = [c for c in df.columns if c != label_col]
    for col in feature_columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    return df, label_col


def split_role(file_name):
    lower = file_name.lower()
    if any(token in lower for token in SOURCE_DAY_TOKENS):
        return "source"
    if any(token in lower for token in OOD_DAY_TOKENS):
        return "ood"
    raise ValueError(f"Could not assign source/target role for {file_name}")


def build_chrono_pickle(data_dir, output_path, val_size, test_known_size, seed):
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    frames = []
    file_groups = []
    group_names = []
    roles = []
    for group_id, file_name in enumerate(csv_files):
        path = os.path.join(data_dir, file_name)
        role = split_role(file_name)
        print(f"[{group_id}] {role:6s} {file_name}")
        df, label_col = read_one_csv(path)
        frames.append(df)
        file_groups.append(np.full(len(df), group_id, dtype=np.int16))
        group_names.append(file_name)
        roles.append(role)

    combined = pd.concat(frames, ignore_index=True)
    file_groups = np.concatenate(file_groups)
    feature_columns = [c for c in combined.columns if c != label_col]

    combined[feature_columns] = combined[feature_columns].clip(-1e12, 1e12).fillna(0)
    zero_var = combined[feature_columns].var() == 0
    if zero_var.any():
        removed = zero_var[zero_var].index.tolist()
        feature_columns = [c for c in feature_columns if c not in removed]
        print(f"Removed {len(removed)} zero-variance columns")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(combined[label_col]).astype(np.int32)
    X = combined[feature_columns].to_numpy(dtype=np.float32)

    source_groups = np.array([i for i, role in enumerate(roles) if role == "source"], dtype=np.int16)
    ood_groups = np.array([i for i, role in enumerate(roles) if role == "ood"], dtype=np.int16)
    source_idx = np.where(np.isin(file_groups, source_groups))[0]
    test_ood_idx = np.where(np.isin(file_groups, ood_groups))[0]

    trainval_idx, test_known_idx = train_test_split(
        source_idx,
        test_size=test_known_size,
        random_state=seed,
        stratify=y[source_idx],
    )
    relative_val_size = val_size / (1.0 - test_known_size)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y[trainval_idx],
    )
    train_idx = np.sort(train_idx).astype(np.int64)
    val_idx = np.sort(val_idx).astype(np.int64)
    test_known_idx = np.sort(test_known_idx).astype(np.int64)
    test_ood_idx = np.sort(test_ood_idx).astype(np.int64)
    test_combined_idx = np.sort(np.concatenate([test_known_idx, test_ood_idx])).astype(np.int64)

    source_labels = set(np.unique(y[source_idx]).tolist())
    ood_labels = set(np.unique(y[test_ood_idx]).tolist())
    class_names = label_encoder.classes_.tolist()
    benign_ids = [i for i, name in enumerate(class_names) if name in ("normal", "benign")]
    if len(benign_ids) != 1:
        raise ValueError(f"Expected one benign/normal label, found {benign_ids}")
    benign_id = int(benign_ids[0])
    unknown_ood_ids = sorted(ood_labels - source_labels - {benign_id})

    save_dict = {
        "X": X,
        "y": y,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "dataset_type": "cic2017_chrono_v2",
        "class_names": class_names,
        "file_groups": file_groups.astype(np.int16),
        "group_names": group_names,
        "group_roles": roles,
        "source_groups": source_groups,
        "ood_groups": ood_groups,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_known_indices": test_known_idx,
        "test_ood_indices": test_ood_idx,
        "test_combined_indices": test_combined_idx,
        "test_indices": test_combined_idx,
        "known_label_ids": np.array(sorted(source_labels), dtype=np.int32),
        "unknown_ood_label_ids": np.array(unknown_ood_ids, dtype=np.int32),
        "benign_original_id": benign_id,
        "scenario_name": "chrono_mon_to_thu_known_friday_ood",
        "val_size": float(val_size),
        "test_known_size": float(test_known_size),
        "seed": int(seed),
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nSaved:", output_path)
    print("Shape:", X.shape)
    print("Train/val/test_known/test_ood:", len(train_idx), len(val_idx), len(test_known_idx), len(test_ood_idx))
    print("Known source labels:", [class_names[i] for i in sorted(source_labels)])
    print("Unknown Friday OOD labels:", [class_names[i] for i in unknown_ood_ids])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/cic2017")
    parser.add_argument("--output", default="data/cic2017_chrono_v2.pkl")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_known_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_chrono_pickle(args.data_dir, args.output, args.val_size, args.test_known_size, args.seed)


if __name__ == "__main__":
    main()

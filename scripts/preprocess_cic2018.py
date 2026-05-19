#!/usr/bin/env python3
"""
Preprocess CIC-IDS2018 raw CSVs → cic2018_proc.pkl  (code_tta.py 호환)

Usage:
    python scripts/preprocess_cic2018.py
    python scripts/preprocess_cic2018.py --data_dir data/cic2018 --output data/cic2018_proc.pkl

Output pkl keys: X, y, label_encoder, feature_columns, dataset_type, class_names, file_groups
Label normalization: lowercase / strip / space→hyphen / "benign"→"normal"
Leakage columns dropped: Timestamp, Src IP, Src Port, Dst IP, Flow ID
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_DROP_COLS = {"timestamp", "src ip", "src port", "dst ip", "flow id"}


def load_csv(path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, low_memory=False)
    drop = [c for c in df.columns if c.strip().lower() in _DROP_COLS]
    if drop:
        df.drop(columns=drop, inplace=True)
    label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
    if label_col is None:
        raise ValueError(f"'Label' column not found in {path}")
    # remove header-repeat rows (some CIC CSVs embed column names mid-file)
    mask = df[label_col].astype(str).str.strip().str.lower() != "label"
    return df[mask].copy(), label_col


def normalize_label(s: str) -> str:
    s = str(s).lower().strip().replace(" ", "-").replace("–", "-")
    return "normal" if s == "benign" else s


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS2018 CSVs → pkl")
    parser.add_argument("--data_dir", default="data/cic2018",
                        help="Directory containing CIC-IDS2018 CSV files")
    parser.add_argument("--output", default="data/cic2018_proc.pkl",
                        help="Output pickle file path")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {args.data_dir}")
    print(f"Found {len(files)} CSV files in {args.data_dir}\n")

    frames, fgroups = [], []
    for i, fp in enumerate(files):
        print(f"  [{i + 1}/{len(files)}] {os.path.basename(fp)}", end="", flush=True)
        df, label_col = load_csv(fp)
        frames.append(df)
        fgroups.append(np.full(len(df), i, dtype=np.int16))
        print(f"  →  {len(df):,} rows")

    print("\nCombining all files …")
    df = pd.concat(frames, ignore_index=True)
    file_groups = np.concatenate(fgroups)
    del frames, fgroups

    label_col = next(c for c in df.columns if c.strip().lower() == "label")
    df[label_col] = df[label_col].apply(normalize_label)

    feature_cols = [c for c in df.columns if c != label_col]

    # coerce non-numeric feature columns
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with any NaN in features (sync file_groups)
    nan_mask = df[feature_cols].isna().any(axis=1)
    if nan_mask.any():
        keep = ~nan_mask
        df = df[keep].reset_index(drop=True)
        file_groups = file_groups[keep.values]
        print(f"Dropped {nan_mask.sum():,} NaN rows  →  {len(df):,} remain")

    # clip extreme values
    df[feature_cols] = df[feature_cols].clip(-1e12, 1e12)

    # Init Win Bytes: sentinel -1 → 0
    for c in ["Init Fwd Win Byts", "Init Bwd Win Byts"]:
        if c in df.columns:
            df[c] = df[c].where(df[c] != -1, 0)

    # drop zero-variance features
    var = df[feature_cols].var()
    zero_var = var[var == 0].index.tolist()
    if zero_var:
        feature_cols = [c for c in feature_cols if c not in zero_var]
        print(f"Dropped {len(zero_var)} zero-variance columns: {zero_var}")

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col]).astype(np.int32)
    X = df[feature_cols].values.astype(np.float32)

    print(f"\nFeature matrix : {X.shape}  dtype={X.dtype}")
    print(f"Classes        : {len(le.classes_)}\n")
    counts = np.bincount(y)
    max_n = int(counts.max())
    for cls_id, cls_name in enumerate(le.classes_):
        ir = max_n / max(int(counts[cls_id]), 1)
        print(f"  {cls_name:45s}  n={counts[cls_id]:>10,}  IR={ir:>9.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out = {
        "X": X,
        "y": y,
        "label_encoder": le,
        "feature_columns": feature_cols,
        "dataset_type": "cic2018",
        "class_names": le.classes_.tolist(),
        "file_groups": file_groups,
    }
    with open(args.output, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"\nSaved  →  {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

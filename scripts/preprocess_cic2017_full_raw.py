#!/usr/bin/env python3
"""Build a full-class, split-agnostic CIC-IDS2017 pickle with a per-row
chronology proxy.

Companion to preprocess_cic2017_chrono_v2.py, not a replacement: chrono_v2
bakes in ONE split (Mon-Thu known / Friday OOD, then a random stratified
train/val/test_known split within Mon-Thu -- see that script's own
train_test_split calls, it is NOT chronologically ordered despite the name).
This script keeps all 15 classes, does no source/OOD role split, and defers
every split decision to the experiment (same "scaling/splits are deferred"
convention as scripts/preprocess_nfv3_energy_suite.py).

Chronology caveat: this CIC-IDS2017 CSV release (data/cic2017/*.csv, the
78-feature ISCX ML-CSV variant) has NO Timestamp/Flow ID/IP/Port columns at
all (checked directly against the raw files) -- unlike the NF-v3 suite, there
is no real per-row time to sort by. What this script stores as `time_proxy`
is (file_order, in-file row order), i.e. the numeric day-session prefix
already in each filename (1_Monday ... 8_Friday-Afternoon-DDos) for
coarse ordering, and each row's position within that file for fine ordering.
This assumes CICFlowMeter wrote rows in flow-completion order within a
capture session, which is the standard assumption for this dataset but is
NOT a verified timestamp -- treat time_proxy as an ordinal, not a duration.

Usage:
  python scripts/preprocess_cic2017_full_raw.py \
      --data-dir data/cic2017 --output data/cic2017_full_raw.pkl
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FILE_ORDER_SPAN = 10_000_000  # > any single session file's row count


def find_label_column(df):
    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    raise ValueError("Could not find CIC-IDS2017 Label column")


def remove_leakage_columns(df):
    remove_names = {"flow id", "source ip", "source port", "destination ip", "timestamp"}
    for col in list(df.columns):
        if col.strip().lower() in remove_names:
            df.drop(columns=[col], inplace=True)


def normalize_web_label(s):
    label = (
        str(s).strip().lower().replace(" ", "-").replace("–", "-").replace("–", "-")
    )
    label = label.replace("web-attack-�-", "web-attack-")
    label = label.replace("web-attack---", "web-attack-")
    label = label.replace("web-attack--", "web-attack-")
    return {"benign": "normal"}.get(label, label)


def read_one_csv(path, file_order):
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
    time_proxy = file_order * FILE_ORDER_SPAN + np.arange(len(df), dtype=np.int64)
    return df, label_col, time_proxy


def build_pickle(data_dir, output_path):
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    frames, time_proxies, file_names = [], [], []
    for file_order, file_name in enumerate(csv_files):
        path = os.path.join(data_dir, file_name)
        df, label_col, time_proxy = read_one_csv(path, file_order)
        print(f"[{file_order}] {file_name}: {len(df):,} rows after cleanup")
        frames.append(df)
        time_proxies.append(time_proxy)
        file_names.append(file_name)

    combined = pd.concat(frames, ignore_index=True)
    time_proxy = np.concatenate(time_proxies)
    feature_columns = [c for c in combined.columns if c != label_col]

    combined[feature_columns] = combined[feature_columns].clip(-1e12, 1e12).fillna(0)
    zero_var = combined[feature_columns].var() == 0
    if zero_var.any():
        removed = zero_var[zero_var].index.tolist()
        feature_columns = [c for c in feature_columns if c not in removed]
        print(f"Removed {len(removed)} zero-variance columns: {removed}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(combined[label_col]).astype(np.int32)
    X = combined[feature_columns].to_numpy(dtype=np.float32)
    class_names = label_encoder.classes_.tolist()

    print(f"\nClasses ({len(class_names)}): {class_names}")
    for i, name in enumerate(class_names):
        print(f"  {name:28s} {int((y == i).sum()):>10,}")

    save_dict = {
        "X": X,
        "y": y,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "feature_columns": feature_columns,
        "dataset_type": "cic2017_full_raw",
        "time_proxy": time_proxy.astype(np.int64),
        "time_proxy_note": (
            "Ordinal (file_order, in-file row order), NOT a real timestamp -- "
            "see module docstring. Use only for split ordering, never as a feature."
        ),
        "source_file_order": file_names,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved: {output_path}")
    print(f"Shape: {X.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join("data", "cic2017"))
    parser.add_argument("--output", default=os.path.join("data", "cic2017_full_raw.pkl"))
    args = parser.parse_args()
    build_pickle(args.data_dir, args.output)


if __name__ == "__main__":
    main()

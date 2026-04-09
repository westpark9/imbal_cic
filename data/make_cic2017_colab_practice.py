#!/usr/bin/env python3
"""Create an under-sampled CIC-IDS2017 practice dataset for Colab labs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]

LABEL_REPLACEMENTS = {
    "web-attack-�-brute-force": "web-attack-brute-force",
    "web-attack-�-xss": "web-attack-xss",
    "web-attack-�-sql-injection": "web-attack-sql-injection",
    "web-attack-brute force": "web-attack-brute-force",
    "web-attack-sql injection": "web-attack-sql-injection",
}

TARGET_LABELS = {
    "benign",
    "web-attack-brute-force",
    "web-attack-xss",
    "web-attack-sql-injection",
    "dos-slowloris",
    "dos-slowhttptest",
    "dos-goldeneye",
}


def find_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    raise ValueError("Label column not found.")


def remove_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = []
    for col in df.columns:
        if any(col.strip().lower() == target.lower() for target in DROP_COLUMNS):
            continue
        keep_cols.append(col)
    return df.loc[:, keep_cols].copy()


def preprocess_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    label_col = find_label_column(df)
    header_mask = df[label_col].astype(str).str.strip().str.lower() == "label"
    if header_mask.any():
        df = df.loc[~header_mask].copy()

    target_columns = [col for col in df.columns if col != label_col]
    object_cols = set(df[target_columns].select_dtypes(include=["object", "category"]).columns)

    for col in target_columns:
        if col in object_cols:
            df[col] = df[col].replace(["-", " -", "- "], "Unknown")
            df[col] = df[col].fillna("Unknown")
        elif df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = list(df[target_columns].select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        df = df.loc[~df[numeric_cols].isna().any(axis=1)].copy()

    df[label_col] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "-", regex=False)
        .str.replace("–", "-", regex=False)
        .replace({"normal": "benign"})
        .replace(LABEL_REPLACEMENTS)
    )

    return df.reset_index(drop=True)


def load_cic2017(data_dir: Path) -> pd.DataFrame:
    frames = []
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, low_memory=False)
        df = remove_known_columns(df)
        df = preprocess_frame(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    label_col = find_label_column(combined)
    feature_cols = [c for c in combined.columns if c != label_col]

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(combined[col]):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
        else:
            combined[col] = combined[col].astype(str).fillna("Unknown")

    numeric_cols = list(combined[feature_cols].select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        combined[numeric_cols] = combined[numeric_cols].replace([np.inf, -np.inf], np.nan)
        combined = combined.dropna(subset=numeric_cols).reset_index(drop=True)

    return combined


def undersample_benign(
    df: pd.DataFrame,
    benign_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    label_col = find_label_column(df)
    benign_mask = df[label_col] == "benign"
    benign_df = df.loc[benign_mask].copy()
    attack_df = df.loc[~benign_mask].copy()

    if benign_df.empty or attack_df.empty:
        raise ValueError("Both benign and attack samples must exist.")

    target_benign = int(round(len(attack_df) * benign_ratio / max(1e-8, 1.0 - benign_ratio)))
    target_benign = max(1, min(target_benign, len(benign_df)))

    benign_sampled = benign_df.sample(n=target_benign, random_state=random_state)
    practice_df = pd.concat([attack_df, benign_sampled], ignore_index=True)
    practice_df = practice_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return practice_df


def filter_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_col = find_label_column(df)
    filtered = df.loc[df[label_col].isin(TARGET_LABELS)].copy()
    if filtered.empty:
        raise ValueError("No rows remain after filtering to the target web-service labels.")
    return filtered.reset_index(drop=True)


def summarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_col = find_label_column(df)
    counts = df[label_col].value_counts().sort_values(ascending=False)
    summary = counts.rename_axis("label").reset_index(name="count")
    summary["ratio"] = summary["count"] / summary["count"].sum()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/cic2017")
    parser.add_argument("--output-dir", default="data/cic2017_colab_practice")
    parser.add_argument("--benign-ratio", type=float, default=0.50)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.benign_ratio < 1.0:
        raise ValueError("--benign-ratio must be in (0, 1).")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be in (0, 1).")

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_cic2017(input_dir)
    filtered_df = filter_target_labels(full_df)
    practice_df = undersample_benign(filtered_df, args.benign_ratio, args.random_state)

    label_col = find_label_column(practice_df)
    train_df, test_df = train_test_split(
        practice_df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=practice_df[label_col],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = output_dir / "cic2017_practice_train.csv"
    test_path = output_dir / "cic2017_practice_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summarize_labels(full_df).to_csv(output_dir / "label_distribution_full.csv", index=False)
    summarize_labels(filtered_df).to_csv(output_dir / "label_distribution_filtered.csv", index=False)
    summarize_labels(practice_df).to_csv(output_dir / "label_distribution_practice.csv", index=False)
    summarize_labels(train_df).to_csv(output_dir / "label_distribution_train.csv", index=False)
    summarize_labels(test_df).to_csv(output_dir / "label_distribution_test.csv", index=False)

    metadata = {
        "source_dir": str(input_dir.relative_to(repo_root) if input_dir.is_relative_to(repo_root) else input_dir),
        "output_dir": str(output_dir.relative_to(repo_root) if output_dir.is_relative_to(repo_root) else output_dir),
        "benign_ratio": args.benign_ratio,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "target_labels": sorted(TARGET_LABELS),
        "filtered_rows": int(len(filtered_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "columns": train_df.columns.tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved train CSV: {train_path}")
    print(f"Saved test CSV:  {test_path}")
    print(f"Practice rows:   {len(practice_df):,}")


if __name__ == "__main__":
    main()

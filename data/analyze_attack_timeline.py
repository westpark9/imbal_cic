#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


def detect_dataset_type_from_dir(data_dir: str) -> str:
    folder_name = os.path.basename(data_dir.rstrip("/\\")).lower()
    if "nf-unsw" in folder_name or "nf_unsw" in folder_name or "nfunsw" in folder_name:
        return "nfunswnb15"
    if "unsw" in folder_name or "nb15" in folder_name:
        return "unswnb15"
    if "2017" in folder_name:
        return "cic2017"
    if "2018" in folder_name:
        return "cic2018"
    return "unknown"


def find_label_column(columns: List[str], dataset_type: Optional[str] = None) -> Optional[str]:
    if dataset_type == "nfunswnb15":
        if "Attack" in columns:
            return "Attack"
        for col in columns:
            if col.strip().lower() == "attack":
                return col
        return None

    if dataset_type == "unswnb15":
        if "attack_cat" in columns:
            return "attack_cat"
        for col in columns:
            if col.strip().lower() == "attack_cat":
                return col
        return None

    for col in columns:
        if col.strip().lower() == "label":
            return col
    return None


def normalize_label_series(series: pd.Series, dataset_type: str) -> pd.Series:
    out = series.astype(str).str.lower().str.strip()
    out = out.str.replace("–", "-", regex=False)
    out = out.str.replace(" ", "-", regex=False)
    out = out.replace({"-": "normal", "benign": "normal", "nan": "normal"})

    if dataset_type in ["unswnb15", "nfunswnb15"]:
        out = out.replace({"attack_cat": "normal", "attack": "normal"})
    else:
        out = out.replace({"label": "normal"})
    return out


def load_timeline_from_csv_dir(data_dir: str, dataset_type: str) -> pd.DataFrame:
    csv_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    rows = []
    global_idx = 0

    for file_id, file_name in enumerate(csv_files):
        file_path = os.path.join(data_dir, file_name)
        header_df = pd.read_csv(file_path, nrows=0, low_memory=False)
        label_col = find_label_column(list(header_df.columns), dataset_type)
        if label_col is None:
            print(f"[WARN] Skipping {file_name}: label column not found.")
            continue

        labels = pd.read_csv(file_path, usecols=[label_col], low_memory=False)[label_col]
        labels = normalize_label_series(labels, dataset_type)

        for local_row, label in enumerate(labels.tolist()):
            rows.append((global_idx, file_id, file_name, local_row, label))
            global_idx += 1

    if not rows:
        raise ValueError("No rows could be loaded from CSV files.")

    return pd.DataFrame(
        rows,
        columns=["global_index", "file_id", "file_name", "file_row_index", "label"],
    )


def build_episodes(df: pd.DataFrame, break_on_file_boundary: bool = True) -> pd.DataFrame:
    labels = df["label"].to_numpy()
    file_ids = df["file_id"].to_numpy()
    gidx = df["global_index"].to_numpy()
    frow = df["file_row_index"].to_numpy()
    fnames = df["file_name"].to_numpy()

    starts = [0]
    for i in range(1, len(df)):
        boundary = labels[i] != labels[i - 1]
        if break_on_file_boundary and file_ids[i] != file_ids[i - 1]:
            boundary = True
        if boundary:
            starts.append(i)
    starts.append(len(df))

    records = []
    for ep_id in range(len(starts) - 1):
        s = starts[ep_id]
        e = starts[ep_id + 1] - 1
        records.append(
            {
                "episode_id": ep_id,
                "label": labels[s],
                "start_global_index": int(gidx[s]),
                "end_global_index": int(gidx[e]),
                "length_samples": int(e - s + 1),
                "start_file_id": int(file_ids[s]),
                "end_file_id": int(file_ids[e]),
                "start_file_name": str(fnames[s]),
                "end_file_name": str(fnames[e]),
                "start_file_row_index": int(frow[s]),
                "end_file_row_index": int(frow[e]),
            }
        )
    return pd.DataFrame(records)


def summarize_classes(episodes: pd.DataFrame, normal_labels: set) -> pd.DataFrame:
    class_totals = episodes.groupby("label", as_index=False)["length_samples"].sum()
    class_totals = class_totals.rename(columns={"length_samples": "total_samples"})

    episode_counts = episodes.groupby("label", as_index=False)["episode_id"].count()
    episode_counts = episode_counts.rename(columns={"episode_id": "num_episodes"})

    agg = (
        episodes.groupby("label", as_index=False)
        .agg(
            mean_episode_len=("length_samples", "mean"),
            median_episode_len=("length_samples", "median"),
            max_episode_len=("length_samples", "max"),
            min_start_idx=("start_global_index", "min"),
            max_end_idx=("end_global_index", "max"),
            num_files_touched=("start_file_id", "nunique"),
        )
        .copy()
    )
    out = class_totals.merge(episode_counts, on="label").merge(agg, on="label")
    out["is_normal"] = out["label"].isin(normal_labels)
    out = out.sort_values(["is_normal", "total_samples"], ascending=[True, False]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attack timeline continuity from CSV order (file order + row order)."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CSV files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: results/timeline_analysis_<timestamp>",
    )
    parser.add_argument(
        "--allow_cross_file_merge",
        action="store_true",
        help="If set, consecutive same labels across file boundaries are merged into one episode.",
    )
    args = parser.parse_args()

    dataset_type = detect_dataset_type_from_dir(args.data_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"results/timeline_analysis_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    timeline_df = load_timeline_from_csv_dir(args.data_dir, dataset_type)
    episodes = build_episodes(
        timeline_df,
        break_on_file_boundary=(not args.allow_cross_file_merge),
    )

    normal_labels = {"normal", "benign"}
    class_summary = summarize_classes(episodes, normal_labels=normal_labels)
    attack_summary = class_summary[~class_summary["is_normal"]].copy()
    attack_episodes = episodes[~episodes["label"].isin(normal_labels)].copy()

    timeline_path = os.path.join(output_dir, "timeline_rows.csv")
    episodes_path = os.path.join(output_dir, "episodes_all.csv")
    attack_episodes_path = os.path.join(output_dir, "episodes_attack_only.csv")
    summary_path = os.path.join(output_dir, "class_summary_all.csv")
    attack_summary_path = os.path.join(output_dir, "class_summary_attack_only.csv")
    by_file_path = os.path.join(output_dir, "file_label_counts.csv")

    timeline_df.to_csv(timeline_path, index=False)
    episodes.to_csv(episodes_path, index=False)
    attack_episodes.to_csv(attack_episodes_path, index=False)
    class_summary.to_csv(summary_path, index=False)
    attack_summary.to_csv(attack_summary_path, index=False)

    file_label_counts = (
        timeline_df.groupby(["file_id", "file_name", "label"], as_index=False)["global_index"]
        .count()
        .rename(columns={"global_index": "num_samples"})
    )
    file_label_counts.to_csv(by_file_path, index=False)

    print("=" * 80)
    print("Timeline analysis complete")
    print("=" * 80)
    print(f"Dataset type: {dataset_type}")
    print(f"Total rows: {len(timeline_df):,}")
    print(f"Total episodes (all labels): {len(episodes):,}")
    print(f"Total attack episodes: {len(attack_episodes):,}")
    print(f"Output dir: {output_dir}")
    print("- timeline rows      :", timeline_path)
    print("- all episodes       :", episodes_path)
    print("- attack episodes    :", attack_episodes_path)
    print("- class summary all  :", summary_path)
    print("- class summary atk  :", attack_summary_path)
    print("- file-label counts  :", by_file_path)
    print("=" * 80)


if __name__ == "__main__":
    main()


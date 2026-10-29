#!/usr/bin/env python3
"""Audit exact first/last timestamps and daily class counts in NetFlow-v3 CSVs."""

import argparse
import json
import os
from collections import Counter

import pandas as pd

from preprocess_nfv3_energy_suite import DATASETS, canonical_family


def iso_milliseconds(value):
    return pd.to_datetime(value, unit="ms", utc=True).isoformat()


def audit_file(path, chunksize):
    counts = Counter()
    first, last = {}, {}
    daily = Counter()
    for chunk in pd.read_csv(
        path, usecols=["FLOW_START_MILLISECONDS", "Attack"],
        chunksize=chunksize, low_memory=False,
    ):
        timestamps = pd.to_numeric(
            chunk["FLOW_START_MILLISECONDS"], errors="coerce"
        )
        families = chunk["Attack"].map(canonical_family)
        valid = timestamps.notna()
        frame = pd.DataFrame({
            "timestamp": timestamps[valid].astype("int64"),
            "family": families[valid],
        })
        frame["day"] = frame["timestamp"] // 86_400_000
        for family, group in frame.groupby("family"):
            counts[family] += len(group)
            minimum = int(group["timestamp"].min())
            maximum = int(group["timestamp"].max())
            first[family] = min(first.get(family, minimum), minimum)
            last[family] = max(last.get(family, maximum), maximum)
        for (family, day), count in frame.groupby(["family", "day"]).size().items():
            daily[(family, int(day))] += int(count)
    result = {}
    for family in sorted(counts):
        day_counts = sorted(
            (day, count) for (name, day), count in daily.items() if name == family
        )
        result[family] = {
            "count": counts[family],
            "first_utc": iso_milliseconds(first[family]),
            "last_utc": iso_milliseconds(last[family]),
            "daily_counts": {
                str(pd.to_datetime(day, unit="D", utc=True).date()): count
                for day, count in day_counts
            },
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/nfv3_temporal_audit.json")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    output = {"timestamp_timezone": "UTC", "datasets": {}}
    for dataset, filename in DATASETS.items():
        path = os.path.join(args.data_dir, filename)
        print(f"Auditing {dataset}: {path}")
        output["datasets"][dataset] = audit_file(path, args.chunksize)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

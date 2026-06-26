#!/usr/bin/env python3
"""Prepare feature-aligned auxiliary OOD data for code_ood_tta.py.

This creates a pickle that can be passed to:
  python src/code_ood_tta.py --external_ood_data data/cic2018_as_ood_for_cic2017.pkl

The source dataset must already be preprocessed into a dict with X and
feature_columns/feature_names. Columns are aligned to the ID dataset schema by
normalized feature names plus CICFlowMeter 2017/2018 aliases. Extra source
columns are ignored; missing ID columns are filled from the ID mean or zero.
"""
import argparse
import json
import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd


def feature_names(d, n_features):
    names = d.get("feature_names") or d.get("feature_columns")
    if names is None:
        names = [f"f_{i}" for i in range(n_features)]
    return list(names)


def canonical_feature_name(name):
    s = str(name).strip().lower()
    s = re.sub(r"\.1$", "", s)
    s = s.replace("/", " ")
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    replacements = [
        ("destination port", "dst port"),
        ("total fwd packets", "tot fwd pkts"),
        ("total backward packets", "tot bwd pkts"),
        ("total length of fwd packets", "totlen fwd pkts"),
        ("total length of bwd packets", "totlen bwd pkts"),
        ("packet", "pkt"),
        ("packets", "pkts"),
        ("bytes", "byts"),
        ("length", "len"),
        ("count", "cnt"),
        ("average", "avg"),
        ("variance", "var"),
        ("forward", "fwd"),
        ("backward", "bwd"),
        ("header length", "header len"),
        ("iat total", "iat tot"),
        ("init win bytes forward", "init fwd win byts"),
        ("init win bytes backward", "init bwd win byts"),
        ("act data pkt fwd", "fwd act data pkts"),
        ("min seg size forward", "fwd seg size min"),
        ("avg fwd segment size", "fwd seg size avg"),
        ("avg bwd segment size", "bwd seg size avg"),
        ("flow bytes s", "flow byts s"),
        ("flow packets s", "flow pkts s"),
        ("subflow fwd bytes", "subflow fwd byts"),
        ("subflow bwd bytes", "subflow bwd byts"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    reorder_aliases = {
        "min pkt len": "pkt len min",
        "max pkt len": "pkt len max",
        "pkt len variance": "pkt len var",
        "avg pkt size": "pkt size avg",
        "init win byts fwd": "init fwd win byts",
        "init win byts bwd": "init bwd win byts",
        "min seg size fwd": "fwd seg size min",
    }
    s = reorder_aliases.get(s, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sample_indices(n, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, int(max_samples), replace=False))


def build_alignment(id_names, src_names):
    src_by_key = {}
    duplicates = {}
    for i, name in enumerate(src_names):
        key = canonical_feature_name(name)
        if key in src_by_key:
            duplicates.setdefault(key, [src_by_key[key]]).append(i)
            continue
        src_by_key[key] = i

    rows = []
    src_indices = []
    missing = []
    for i, name in enumerate(id_names):
        key = canonical_feature_name(name)
        src_i = src_by_key.get(key)
        if src_i is None:
            missing.append(i)
        src_indices.append(src_i)
        rows.append({
            "id_index": i,
            "id_feature": name,
            "canonical": key,
            "source_index": "" if src_i is None else int(src_i),
            "source_feature": "" if src_i is None else src_names[src_i],
            "status": "missing" if src_i is None else "matched",
        })
    return src_indices, rows, duplicates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-data", required=True,
                        help="ID preprocessed pickle, e.g. data/cic2017_proc.pkl")
    parser.add_argument("--source-data", required=True,
                        help="Auxiliary OOD preprocessed pickle, e.g. data/cic2018_proc.pkl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=500_000,
                        help="Subsample source OOD rows; <=0 keeps all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fill", choices=["id_mean", "zero"], default="id_mean",
                        help="How to fill ID columns missing from source")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print alignment report without writing the output pickle")
    args = parser.parse_args()

    id_data = load_pickle(args.id_data)
    src_data = load_pickle(args.source_data)
    X_id = np.asarray(id_data["X"])
    X_src = np.asarray(src_data["X"])
    id_names = feature_names(id_data, X_id.shape[1])
    src_names = feature_names(src_data, X_src.shape[1])

    src_indices, rows, duplicates = build_alignment(id_names, src_names)
    matched = sum(i is not None for i in src_indices)
    missing = len(src_indices) - matched
    report = {
        "id_data": args.id_data,
        "source_data": args.source_data,
        "id_shape": list(X_id.shape),
        "source_shape": list(X_src.shape),
        "matched_features": matched,
        "missing_features": missing,
        "source_duplicate_canonical_names": {k: v for k, v in duplicates.items()},
    }
    print(json.dumps(report, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    if args.dry_run:
        return

    idx = sample_indices(len(X_src), args.max_samples, args.seed)
    X_src = X_src[idx]
    X_out = np.empty((len(X_src), len(id_names)), dtype=np.float32)
    fill_values = np.zeros(len(id_names), dtype=np.float32)
    if args.fill == "id_mean":
        fill_values = np.nan_to_num(X_id.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    for j, src_j in enumerate(src_indices):
        if src_j is None:
            X_out[:, j] = fill_values[j]
        else:
            X_out[:, j] = X_src[:, src_j].astype(np.float32)
    X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    y_src = src_data.get("y")
    if y_src is not None:
        y_src = np.asarray(y_src)[idx]

    out = {
        "X": X_out,
        "y": y_src,
        "feature_names": list(id_names),
        "feature_columns": list(id_names),
        "dataset_type": "external_ood",
        "source_dataset_type": src_data.get("dataset_type", "unknown"),
        "source_path": args.source_data,
        "id_reference_path": args.id_data,
        "source_class_names": src_data.get("class_names"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alignment": rows,
        "alignment_summary": report,
        "sample_indices": idx,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {args.out} with X shape={X_out.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build four NetFlow-v3 open-set benchmarks and clean external OE pools.

The UQ NetFlow-v3 release contains four datasets with an identical 53-column
schema.  This script reads them into one capped, auditable pool and records a
separate protocol for each target dataset:

  * benign and non-held-out target attacks are known ID;
  * selected target-only attack families are the final OOD test set;
  * only semantically non-overlapping families from the other datasets are OE.

No virtual/near-OOD samples are generated.  Absolute timestamps, addresses,
and ports are retained only long enough to make temporal splits and are not
model features.  Scaling is deliberately deferred to the experiment, which
must fit its scaler on the selected target's training indices only.

Example:
  python scripts/preprocess_nfv3_energy_suite.py \
    --data-dir data --output data/nfv3_energy_suite.pkl
"""

import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


DATASETS = {
    "unsw_nb15": "NF-UNSW-NB15-v3.csv",
    "ton_iot": "NF-ToN-IoT-v3.csv",
    "bot_iot": "NF-BoT-IoT-v3.csv",
    "cse_cic_ids2018": "NF-CICIDS2018-v3.csv",
}

# These families have a sufficiently distinct meaning from the common NIDS
# families (DoS, scanning, brute force, web attacks, and backdoors) to serve as
# conservative far-OE candidates.  A target's held-out families are removed
# from its own auxiliary set even when they appear here.
CONSERVATIVE_NOVEL_FAMILIES = {
    "fuzzers", "generic", "shellcode", "worms",
    "mitm", "ransomware", "theft",
}

PROTOCOLS = {
    "unsw_nb15": {
        "unknown_families": {"fuzzers", "generic", "shellcode", "worms"},
    },
    "ton_iot": {
        "unknown_families": {"mitm", "ransomware"},
    },
    "bot_iot": {
        "unknown_families": {"theft"},
    },
    "cse_cic_ids2018": {
        "unknown_families": {"bot", "infiltration", "web_attacks"},
    },
}

NON_FEATURE_COLUMNS = {
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    # DNS_QUERY_ID is a transaction identifier, not a behavioural quantity.
    "DNS_QUERY_ID", "Label", "Attack",
}


def canonical_family(value):
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    # NF-CSE-CIC-IDS2018-v3 retains scenario-level labels.  Collapse them to
    # the parent categories documented by UQ before applying overlap rules.
    if text.startswith("ddos_attack") or text.startswith("ddos_attacks"):
        return "ddos"
    if text.startswith("dos_attack") or text.startswith("dos_attacks"):
        return "dos"
    if text in {"ftp_bruteforce", "ssh_bruteforce"}:
        return "brute_force"
    if text in {"brute_force_web", "brute_force_xss", "sql_injection"}:
        return "web_attacks"
    aliases = {
        "benign": "benign",
        "normal": "benign",
        "bruteforce": "brute_force",
        "brute_force": "brute_force",
        "web_attack": "web_attacks",
        "web_attacks": "web_attacks",
        "infilteration": "infiltration",
        "man_in_the_middle": "mitm",
    }
    return aliases.get(text, text)


def inspect_schema(paths):
    reference = None
    for dataset, path in paths.items():
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if "Attack" not in columns or "FLOW_START_MILLISECONDS" not in columns:
            raise ValueError(f"{path} is not a labelled NetFlow-v3 CSV")
        if reference is None:
            reference = columns
        elif columns != reference:
            raise ValueError(
                f"NetFlow-v3 schema/order mismatch for {dataset}: "
                f"expected {reference}, got {columns}"
            )
    features = [column for column in reference if column not in NON_FEATURE_COLUMNS]
    if len(reference) != 55 or len(features) != 46:
        raise ValueError(
            f"Expected 55 CSV columns and 46 leakage-filtered features; "
            f"got {len(reference)} and {len(features)}"
        )
    return reference, features


class StratifiedReservoir:
    """Uniformly cap each (dataset, attack family) while streaming huge CSVs."""

    def __init__(self, maximum, seed):
        self.maximum = int(maximum)
        self.rng = np.random.default_rng(seed)
        self.values = {}
        self.timestamps = {}
        self.priorities = {}
        self.seen = Counter()
        self.uncapped_values = defaultdict(list)
        self.uncapped_timestamps = defaultdict(list)

    def add(self, dataset, family, timestamps, values):
        key = (dataset, family)
        self.seen[key] += len(values)
        if self.maximum <= 0:
            # Avoid quadratic repeated concatenation for the 66M-row uncapped
            # suite. Concatenate each family exactly once in finish().
            self.uncapped_values[key].append(values)
            self.uncapped_timestamps[key].append(timestamps)
            return
        priorities = self.rng.random(len(values))
        if key in self.values:
            values = np.concatenate([self.values[key], values])
            timestamps = np.concatenate([self.timestamps[key], timestamps])
            priorities = np.concatenate([self.priorities[key], priorities])
        if self.maximum > 0 and len(values) > self.maximum:
            keep = np.argpartition(priorities, self.maximum - 1)[:self.maximum]
            values = values[keep]
            timestamps = timestamps[keep]
            priorities = priorities[keep]
        self.values[key] = values
        self.timestamps[key] = timestamps
        self.priorities[key] = priorities

    def finish(self):
        values, datasets, families, timestamps = [], [], [], []
        keys = self.uncapped_values if self.maximum <= 0 else self.values
        for key in sorted(keys):
            dataset, family = key
            if self.maximum <= 0:
                raw_rows = np.concatenate(self.uncapped_values[key])
                raw_times = np.concatenate(self.uncapped_timestamps[key])
            else:
                raw_rows = self.values[key]
                raw_times = self.timestamps[key]
            order = np.argsort(raw_times, kind="stable")
            rows = raw_rows[order]
            times = raw_times[order]
            values.append(rows)
            datasets.append(np.full(len(rows), dataset, dtype=object))
            families.append(np.full(len(rows), family, dtype=object))
            timestamps.append(times)
        return (
            np.concatenate(values).astype(np.float32, copy=False),
            np.concatenate(datasets),
            np.concatenate(families),
            np.concatenate(timestamps).astype(np.int64, copy=False),
        )


def stream_pool(paths, features, chunksize, maximum, clip, seed):
    reservoir = StratifiedReservoir(maximum, seed)
    invalid_rows = Counter()
    nonfinite_feature_rows_imputed = Counter()
    observed = defaultdict(set)
    usecols = ["FLOW_START_MILLISECONDS", "Label", "Attack", *features]
    for dataset, path in paths.items():
        print(f"Reading {dataset}: {path}")
        for chunk_number, chunk in enumerate(pd.read_csv(
            path, usecols=usecols, chunksize=chunksize, low_memory=False,
        ), start=1):
            families = chunk["Attack"].map(canonical_family).to_numpy(dtype=object)
            # Attack is authoritative, but reject inconsistent binary labels.
            binary = pd.to_numeric(chunk["Label"], errors="coerce").to_numpy()
            timestamps = pd.to_numeric(
                chunk["FLOW_START_MILLISECONDS"], errors="coerce"
            ).to_numpy()
            numeric = chunk[features].apply(pd.to_numeric, errors="coerce")
            numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
            nonfinite = numeric.isna().any(axis=1).to_numpy()
            nonfinite_feature_rows_imputed[dataset] += int(nonfinite.sum())
            # NetFlow rate fields are undefined for zero-duration flows.  UQ's
            # CSVs encode these as NaN/inf, disproportionately affecting rare
            # families (e.g. BoT-IoT Theft).  Zero is the neutral, finite rate
            # convention and preserves the flow instead of selection-biasing it.
            numeric.fillna(0.0, inplace=True)
            valid = (
                np.isfinite(timestamps)
                & np.isfinite(binary)
                & np.asarray([bool(value) for value in families])
            )
            expected_attack = families != "benign"
            valid &= (binary.astype(bool) == expected_attack)
            invalid_rows[dataset] += int((~valid).sum())
            if valid.any():
                X = numeric.loc[valid].clip(-clip, clip).to_numpy(
                    dtype=np.float32, copy=True
                )
                y_family = families[valid]
                ts = timestamps[valid].astype(np.int64)
                observed[dataset].update(np.unique(y_family).tolist())
                for family in np.unique(y_family):
                    mask = y_family == family
                    reservoir.add(dataset, family, ts[mask], X[mask])
            if chunk_number % 20 == 0:
                print(f"  chunks={chunk_number:,} seen={sum(reservoir.seen.values()):,}")
    X, dataset_names, families, timestamps = reservoir.finish()
    return {
        "X": X,
        "dataset_names": dataset_names,
        "families": families,
        "timestamps": timestamps,
        "observed": {key: sorted(value) for key, value in observed.items()},
        "seen": {f"{key[0]}::{key[1]}": value for key, value in reservoir.seen.items()},
        "invalid_rows": dict(invalid_rows),
        "nonfinite_feature_rows_imputed": dict(nonfinite_feature_rows_imputed),
    }


def temporal_class_split(indices, families, timestamps, train_fraction, val_fraction):
    train, val, test = [], [], []
    for family in sorted(np.unique(families[indices])):
        family_indices = indices[families[indices] == family]
        order = np.argsort(timestamps[family_indices], kind="stable")
        ordered = family_indices[order]
        n_train = max(1, int(len(ordered) * train_fraction))
        n_val = max(1, int(len(ordered) * val_fraction))
        if n_train + n_val >= len(ordered):
            raise ValueError(
                f"Family {family} has only {len(ordered)} retained rows; "
                "increase --max-per-dataset-family or inspect the raw label"
            )
        train.extend(ordered[:n_train])
        val.extend(ordered[n_train:n_train + n_val])
        test.extend(ordered[n_train + n_val:])
    return {
        "train_indices": np.sort(np.asarray(train, dtype=np.int64)),
        "val_indices": np.sort(np.asarray(val, dtype=np.int64)),
        "test_known_indices": np.sort(np.asarray(test, dtype=np.int64)),
    }


def make_protocols(pool, train_fraction, val_fraction):
    datasets = pool["dataset_names"]
    families = pool["families"]
    timestamps = pool["timestamps"]
    protocols = {}
    for target, definition in PROTOCOLS.items():
        target_mask = datasets == target
        target_families = set(np.unique(families[target_mask]).tolist())
        unknown = set(definition["unknown_families"])
        missing = sorted(unknown - target_families)
        if missing:
            raise ValueError(f"Target {target} is missing configured OOD families: {missing}")
        known = target_families - unknown
        if "benign" not in known:
            raise ValueError(f"Target {target} has no benign ID family")
        known_idx = np.flatnonzero(target_mask & np.isin(families, sorted(known)))
        split = temporal_class_split(
            known_idx, families, timestamps, train_fraction, val_fraction
        )
        train_values = pool["X"][split["train_indices"]]
        variable_features = np.flatnonzero(np.ptp(train_values, axis=0) > 0).astype(np.int32)
        if not len(variable_features):
            raise ValueError(f"Target {target} has no variable training features")
        ood_idx = np.flatnonzero(target_mask & np.isin(families, sorted(unknown)))
        allowed_aux = CONSERVATIVE_NOVEL_FAMILIES - unknown - known
        aux_idx = np.flatnonzero(
            (~target_mask) & np.isin(families, sorted(allowed_aux))
        )
        if not len(aux_idx):
            raise ValueError(f"Target {target} has no clean external OE rows")
        protocols[target] = {
            **split,
            "test_ood_indices": ood_idx.astype(np.int64),
            "aux_indices": aux_idx.astype(np.int64),
            "known_families": sorted(known),
            "unknown_families": sorted(unknown),
            "allowed_aux_families": sorted(allowed_aux),
            "aux_source_datasets": sorted(set(datasets[aux_idx].tolist())),
            "feature_indices": variable_features,
            "constant_feature_indices": np.flatnonzero(
                np.ptp(train_values, axis=0) == 0
            ).astype(np.int32),
            "split_method": "within-family chronological after seeded reservoir sampling",
        }
    return protocols


def atomic_pickle(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def counts_for(indices, datasets, families):
    return dict(sorted(Counter(
        f"{datasets[index]}::{families[index]}" for index in indices
    ).items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/nfv3_energy_suite.pkl")
    parser.add_argument("--report", default="data/nfv3_energy_suite.json")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-per-dataset-family", type=int, default=100_000)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--clip", type=float, default=1e12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 < args.train_fraction < 1 and 0 < args.val_fraction < 1):
        raise ValueError("train/validation fractions must be in (0, 1)")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below 1")
    paths = {key: os.path.join(args.data_dir, name) for key, name in DATASETS.items()}
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing NetFlow-v3 files: {missing}")
    raw_columns, features = inspect_schema(paths)
    print(f"Verified common schema: raw={len(raw_columns)} model_features={len(features)}")
    pool = stream_pool(
        paths, features, args.chunksize, args.max_per_dataset_family,
        args.clip, args.seed,
    )
    protocols = make_protocols(pool, args.train_fraction, args.val_fraction)
    output = {
        "X": pool["X"],
        "dataset_names": pool["dataset_names"],
        "families": pool["families"],
        "timestamps": pool["timestamps"],
        "feature_columns": features,
        "feature_names": features,
        "dataset_type": "uq_netflow_v3_energy_open_set_suite",
        "protocols": protocols,
        "preprocessing": {
            "scaled": False,
            "clip": args.clip,
            "removed_columns": sorted(NON_FEATURE_COLUMNS - {"Label", "Attack"}),
            "max_per_dataset_family": args.max_per_dataset_family,
            "rows_seen_by_dataset_family": pool["seen"],
            "invalid_rows_dropped": pool["invalid_rows"],
            "nonfinite_feature_rows_imputed_with_zero": pool[
                "nonfinite_feature_rows_imputed"
            ],
            "conservative_novel_families": sorted(CONSERVATIVE_NOVEL_FAMILIES),
            "no_virtual_outliers": True,
        },
        "seed": args.seed,
    }
    atomic_pickle(output, args.output)

    report = {
        "output": args.output,
        "shape": list(pool["X"].shape),
        "feature_columns": features,
        "observed_families": pool["observed"],
        "raw_counts": pool["seen"],
        "invalid_rows_dropped": pool["invalid_rows"],
        "nonfinite_feature_rows_imputed_with_zero": pool[
            "nonfinite_feature_rows_imputed"
        ],
        "protocols": {},
    }
    for target, protocol in protocols.items():
        report["protocols"][target] = {
            key: protocol[key] for key in (
                "known_families", "unknown_families", "allowed_aux_families",
                "aux_source_datasets", "split_method",
            )
        }
        report["protocols"][target]["model_features"] = [
            features[index] for index in protocol["feature_indices"]
        ]
        report["protocols"][target]["constant_features_removed"] = [
            features[index] for index in protocol["constant_feature_indices"]
        ]
        report["protocols"][target]["sizes"] = {
            "train": len(protocol["train_indices"]),
            "validation": len(protocol["val_indices"]),
            "known_test": len(protocol["test_known_indices"]),
            "ood_test": len(protocol["test_ood_indices"]),
            "auxiliary_oe": len(protocol["aux_indices"]),
        }
        report["protocols"][target]["ood_counts"] = counts_for(
            protocol["test_ood_indices"], pool["dataset_names"], pool["families"]
        )
        report["protocols"][target]["aux_counts"] = counts_for(
            protocol["aux_indices"], pool["dataset_names"], pool["families"]
        )
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved {args.output}: X={pool['X'].shape}")
    print(f"Saved audit report: {args.report}")


if __name__ == "__main__":
    main()

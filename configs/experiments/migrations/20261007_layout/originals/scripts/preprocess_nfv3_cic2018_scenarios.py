#!/usr/bin/env python3
"""Replace CIC2018's parent-capped rows with scenario-preserving samples.

The existing NetFlow-v3 suite remains the source for the three external
datasets.  NF-CICIDS2018-v3 is reread because its original ``Attack`` values
were previously collapsed to parent families before sampling.  Rows are
uniformly capped per parent family as before, while retaining both the raw
scenario used for temporal splitting and the parent classifier label.
"""

import argparse
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_nfv3_energy_suite import (  # noqa: E402
    NON_FEATURE_COLUMNS,
    atomic_pickle,
    canonical_family,
)


TARGET = "cse_cic_ids2018"


def scenario_name(value):
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return "benign" if text in {"benign", "normal"} else text


class ScenarioReservoir:
    """Keep raw scenarios while uniformly capping each parent family."""

    def __init__(self, maximum, seed):
        self.maximum = int(maximum)
        self.rng = np.random.default_rng(seed)
        self.rows = {}
        self.times = {}
        self.priorities = {}
        self.scenarios = {}
        self.seen = Counter()
        self.uncapped_rows = defaultdict(list)
        self.uncapped_times = defaultdict(list)
        self.uncapped_scenarios = defaultdict(list)

    def add(self, scenario, family, rows, times):
        self.seen[scenario] += len(rows)
        key = family
        scenario_values = np.full(len(rows), scenario, dtype=object)
        if self.maximum <= 0:
            # Preserve every raw row without repeatedly copying a growing
            # family array for each CSV chunk.
            self.uncapped_rows[key].append(rows)
            self.uncapped_times[key].append(times)
            self.uncapped_scenarios[key].append(scenario_values)
            return
        priority = self.rng.random(len(rows))
        if key in self.rows:
            rows = np.concatenate((self.rows[key], rows))
            times = np.concatenate((self.times[key], times))
            scenario_values = np.concatenate((self.scenarios[key], scenario_values))
            priority = np.concatenate((self.priorities[key], priority))
        if self.maximum > 0 and len(rows) > self.maximum:
            keep = np.argpartition(priority, self.maximum - 1)[:self.maximum]
            rows, times = rows[keep], times[keep]
            scenario_values, priority = scenario_values[keep], priority[keep]
        self.rows[key] = rows
        self.times[key] = times
        self.scenarios[key] = scenario_values
        self.priorities[key] = priority

    def finish(self):
        rows, families, scenarios, times = [], [], [], []
        keys = self.uncapped_rows if self.maximum <= 0 else self.rows
        for family in sorted(keys):
            if self.maximum <= 0:
                raw_rows = np.concatenate(self.uncapped_rows[family])
                raw_times = np.concatenate(self.uncapped_times[family])
                raw_scenarios = np.concatenate(self.uncapped_scenarios[family])
            else:
                raw_rows = self.rows[family]
                raw_times = self.times[family]
                raw_scenarios = self.scenarios[family]
            order = np.argsort(raw_times, kind="stable")
            values = raw_rows[order]
            rows.append(values)
            times.append(raw_times[order])
            scenarios.append(raw_scenarios[order])
            families.append(np.full(len(values), family, dtype=object))
        return (
            np.concatenate(rows).astype(np.float32, copy=False),
            np.concatenate(families),
            np.concatenate(scenarios),
            np.concatenate(times).astype(np.int64, copy=False),
        )


def read_cic(path, features, chunksize, maximum, clip, seed):
    usecols = ["FLOW_START_MILLISECONDS", "Label", "Attack", *features]
    reservoir = ScenarioReservoir(maximum, seed)
    invalid = 0
    imputed = 0
    for chunk_no, chunk in enumerate(pd.read_csv(
        path, usecols=usecols, chunksize=chunksize, low_memory=False,
    ), start=1):
        scenarios = chunk["Attack"].map(scenario_name).to_numpy(dtype=object)
        families = chunk["Attack"].map(canonical_family).to_numpy(dtype=object)
        labels = pd.to_numeric(chunk["Label"], errors="coerce").to_numpy()
        times = pd.to_numeric(
            chunk["FLOW_START_MILLISECONDS"], errors="coerce"
        ).to_numpy()
        numeric = chunk[features].apply(pd.to_numeric, errors="coerce")
        numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        imputed += int(numeric.isna().any(axis=1).sum())
        numeric.fillna(0.0, inplace=True)
        valid = np.isfinite(labels) & np.isfinite(times)
        valid &= labels.astype(bool) == (families != "benign")
        invalid += int((~valid).sum())
        if valid.any():
            X = numeric.loc[valid].clip(-clip, clip).to_numpy(dtype=np.float32)
            valid_scenarios = scenarios[valid]
            valid_families = families[valid]
            valid_times = times[valid].astype(np.int64)
            for scenario in np.unique(valid_scenarios):
                mask = valid_scenarios == scenario
                parent = np.unique(valid_families[mask])
                if len(parent) != 1:
                    raise ValueError(f"Scenario {scenario} maps to parents {parent.tolist()}")
                reservoir.add(scenario, parent[0], X[mask], valid_times[mask])
        if chunk_no % 20 == 0:
            print(f"  chunks={chunk_no:,} rows_seen={sum(reservoir.seen.values()):,}")
    X, families, scenarios, times = reservoir.finish()
    return X, families, scenarios, times, reservoir.seen, invalid, imputed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-suite", default="data/nfv3_energy_suite.pkl")
    parser.add_argument("--cic-csv", default="data/NF-CICIDS2018-v3.csv")
    parser.add_argument("--output", default="data/nfv3_energy_suite_cic2018_scenarios.pkl")
    parser.add_argument("--report", default="data/nfv3_energy_suite_cic2018_scenarios.json")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-per-parent-family", type=int, default=100_000)
    parser.add_argument("--clip", type=float, default=1e12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.base_suite, "rb") as handle:
        base = pickle.load(handle)
    features = list(base["feature_columns"])
    expected = [
        name for name in pd.read_csv(args.cic_csv, nrows=0).columns
        if name not in NON_FEATURE_COLUMNS
    ]
    if features != expected:
        raise ValueError("CIC2018 feature schema does not match the base suite")
    print(f"Reading CIC2018 scenarios from {args.cic_csv}")
    X_cic, fam_cic, scen_cic, time_cic, seen, invalid, imputed = read_cic(
        args.cic_csv, features, args.chunksize, args.max_per_parent_family,
        args.clip, args.seed,
    )
    base_datasets = np.asarray(base["dataset_names"], dtype=object)
    external = base_datasets != TARGET
    X = np.concatenate((np.asarray(base["X"])[external], X_cic))
    datasets = np.concatenate((
        base_datasets[external], np.full(len(X_cic), TARGET, dtype=object)
    ))
    families = np.concatenate((np.asarray(base["families"])[external], fam_cic))
    # External scenarios are not used for the CIC target split. Their canonical
    # family is retained as an explicit placeholder rather than fabricated raw labels.
    scenarios = np.concatenate((np.asarray(base["families"])[external], scen_cic))
    timestamps = np.concatenate((np.asarray(base["timestamps"])[external], time_cic))
    output = {
        "X": X, "dataset_names": datasets, "families": families,
        "attack_scenarios": scenarios, "timestamps": timestamps,
        "feature_columns": features, "feature_names": features,
        "dataset_type": "uq_netflow_v3_cic2018_scenario_split_suite",
        "preprocessing": {
            "base_suite": args.base_suite,
            "base_max_per_dataset_family": base.get("preprocessing", {}).get(
                "max_per_dataset_family"
            ),
            "cic2018_replaced_from_raw": args.cic_csv,
            "max_per_parent_family": args.max_per_parent_family,
            "cic2018_rows_seen_by_scenario": dict(sorted(seen.items())),
            "invalid_rows_dropped": invalid,
            "nonfinite_feature_rows_imputed_with_zero": imputed,
            "scaled": False, "clip": args.clip,
        },
        "seed": args.seed,
    }
    atomic_pickle(output, args.output)
    report = {
        "output": args.output, "shape": list(X.shape),
        "cic2018_retained": int(len(X_cic)),
        "cic2018_raw_counts": dict(sorted(seen.items())),
        "cic2018_retained_counts": dict(sorted(Counter(scen_cic).items())),
        "scenario_to_parent": dict(sorted({
            str(s): str(f) for s, f in zip(scen_cic, fam_cic)
        }.items())),
        "invalid_rows_dropped": invalid,
        "nonfinite_feature_rows_imputed_with_zero": imputed,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved {args.output}: X={X.shape}, CIC2018={len(X_cic):,}")
    print(f"Saved audit report: {args.report}")


if __name__ == "__main__":
    main()

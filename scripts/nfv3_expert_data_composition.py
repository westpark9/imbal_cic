#!/usr/bin/env python3
"""Read-only NF-v3 data-composition planner for s44 (also matches s43 up to
the point s44 adds the benign expert).

Replays s44_nfv3_resolved_expert_pipeline's exact data-prep steps -- same
functions, same seed offsets, same call order -- up to the point per-expert
model training would start, and exports a clean per-expert / per-class
train:val:test row-count table. No model trains, no GPU needed (pure numpy
indexing over the suite pickle), and s44's own file is never touched (it's
frozen per CLAUDE.md M2) -- this only imports its module-level functions,
the same pattern s36 already uses on s35 (`import s35... as base`).

Because `cap_by_family`/`subset_indices` are deterministic given the same
input array + seed, calling them here with the identical arguments s44 uses
reproduces s44's real per-expert row counts exactly (not an estimate).

Two use cases:
  * Point this at an s44 arg configuration BEFORE running it, to know the
    exact composition a real run will have (e.g. to size a SOTA-comparison
    script's sampling budget against it) without paying for the full
    training run.
  * Regenerate the composition summary for an s44 run that already
    finished, by passing that run's exact args (copy them from its
    metadata.json / experiment.log "Args:" line).

Accepts the exact same flags as s44_nfv3_resolved_expert_pipeline.py (it
reuses s44's own argparse parser), so whatever args a real s44 run used,
this script's output matches that run's actual composition.

Usage:
  python scripts/nfv3_expert_data_composition.py \
      --data data/nfv3_energy_suite_uncapped_scenarios.pkl \
      --target cse_cic_ids2018 --unseen bot
"""

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import s44_nfv3_resolved_expert_pipeline as s44  # noqa: E402


def class_counts(indices, families):
    if not len(indices):
        return {}
    values, counts = np.unique(families[indices], return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def add_split_rows(rows, scope, role, source, indices_by_split, families):
    counts_by_split = {
        split_name: class_counts(indices, families)
        for split_name, indices in indices_by_split.items()
    }
    classes = sorted(set().union(*counts_by_split.values())) if counts_by_split else []
    for cls in classes:
        rows.append({
            "scope": scope, "role": role, "source_dataset": source, "class": cls,
            "train": counts_by_split.get("train", {}).get(cls, 0),
            "val": counts_by_split.get("val", {}).get(cls, 0),
            "test": counts_by_split.get("test", {}).get(cls, 0),
        })


def add_audit_rows(rows, scope, audit_records, role_by_decision):
    for record in audit_records:
        role = role_by_decision.get(record["decision"])
        if role is None:
            continue
        rows.append({
            "scope": scope, "role": role, "source_dataset": record["source"],
            "class": record["family"], "train": record["aux_train_selected"],
            "val": record["aux_validation_selected"], "test": 0,
        })


def main():
    args = s44.parse_args()
    s44.set_seed(args.seed)
    print(f"Args: {vars(args)}")

    suite = s44.load_pickle(args.data)
    datasets = np.asarray(suite["dataset_names"], dtype=object)
    families = np.asarray(suite["families"], dtype=object)
    scenarios = np.asarray(suite["attack_scenarios"], dtype=object)
    timestamps = np.asarray(suite["timestamps"], dtype=np.int64)

    target_indices = np.flatnonzero(datasets == args.target)
    target_classes = sorted(np.unique(families[target_indices]).tolist())
    target_attack_classes = [name for name in target_classes if name != "benign"]
    experts = s44.load_expert_groups(args.expert_config, args.target)
    s44.validate_expert_partition(experts, target_attack_classes)

    unseen = s44.normalize_unseen(args.unseen)
    if not unseen:
        unseen = list(s44.DEFAULT_UNSEEN[args.target])
    active_experts = {
        group: [name for name in owned if name not in unseen]
        for group, owned in experts.items()
    }
    active_experts = {g: o for g, o in active_experts.items() if o}
    known_classes = [name for name in target_attack_classes if name not in unseen]
    print(f"target={args.target} unseen={unseen} known_classes={known_classes}")
    print(f"experts={active_experts}")

    # --- Same split/cap sequence as s44_nfv3_resolved_expert_pipeline.main() ---
    target_attack_indices = target_indices[families[target_indices] != "benign"]
    split, _ = s44.scenario_chronological_split(target_attack_indices, scenarios, timestamps)
    known_split = {
        role: idx[np.isin(families[idx], known_classes)] for role, idx in split.items()
    }
    unseen_test_raw = split["test"][np.isin(families[split["test"]], unseen)]
    known_test = s44.cap_by_family(known_split["test"], families, args.test_cap_per_class, args.seed + 11)
    unseen_test = s44.cap_by_family(unseen_test_raw, families, args.test_cap_per_class, args.seed + 12)

    benign_pool = target_indices[families[target_indices] == "benign"]
    benign_split, _ = s44.scenario_chronological_split(benign_pool, scenarios, timestamps)
    benign_test = s44.subset_indices(benign_split["test"], args.test_cap_per_class, args.seed + 16)

    aux_groups, _ = s44.external_aux_groups(
        datasets, families, args.target, args.aux_cap_per_source_family,
        args.aux_val_fraction, args.seed + 30,
    )

    role_by_decision = {
        "TARGET KNOWN OOD": "target_known_OOD",
        "EXTERNAL AUX OOD": "external_AUX_OOD",
    }
    rows = []

    # --- overall target composition (raw train/val pool, final capped test) ---
    add_split_rows(
        rows, "TARGET_OVERALL", "known_attack", args.target,
        {"train": known_split["train"], "val": known_split["val"], "test": known_test},
        families,
    )
    add_split_rows(
        rows, "TARGET_OVERALL", "unseen_attack", args.target,
        {
            "train": split["train"][np.isin(families[split["train"]], unseen)],
            "val": split["val"][np.isin(families[split["val"]], unseen)],
            "test": unseen_test,
        },
        families,
    )
    add_split_rows(
        rows, "TARGET_OVERALL", "benign", args.target,
        {"train": benign_split["train"], "val": benign_split["val"], "test": benign_test},
        families,
    )

    # --- per attack-expert composition (ID + OOD roles) ---
    for expert_number, (group, active_owned) in enumerate(active_experts.items()):
        configured_owned = experts[group]
        owned_train_raw = known_split["train"][np.isin(families[known_split["train"]], active_owned)]
        owned_val_raw = known_split["val"][np.isin(families[known_split["val"]], active_owned)]
        owned_val = s44.cap_by_family(
            owned_val_raw, families, args.val_cap_per_class, args.seed + 90 + expert_number,
        )
        owned_train = s44.subset_indices(
            owned_train_raw, args.max_id_train_per_expert, args.seed + 100 + expert_number,
        )
        add_split_rows(
            rows, group, "ID", args.target,
            {"train": owned_train, "val": owned_val}, families,
        )

        _, _, external_aux_audit = s44.external_aux_for_expert(
            group, configured_owned, aux_groups, enabled=not args.disable_external_aux,
        )
        _, _, target_ood_audit = s44.target_known_ood_for_expert(
            group, active_owned, configured_owned, known_split, families,
            args.target_ood_cap_per_class, args.val_cap_per_class,
            args.seed + 130 + expert_number * 10, enabled=not args.disable_target_ood,
        )
        add_audit_rows(rows, group, target_ood_audit, role_by_decision)
        add_audit_rows(rows, group, external_aux_audit, role_by_decision)

    # --- benign expert composition (s44-only addition over s43) ---
    benign_owned = ["benign"]
    benign_owned_train = s44.subset_indices(benign_split["train"], args.benign_train_cap, args.seed + 200)
    benign_owned_val = s44.cap_by_family(benign_split["val"], families, args.val_cap_per_class, args.seed + 201)
    add_split_rows(
        rows, "benign_expert", "ID", args.target,
        {"train": benign_owned_train, "val": benign_owned_val}, families,
    )
    _, _, benign_external_aux_audit = s44.external_aux_for_expert(
        "benign", benign_owned, aux_groups, enabled=not args.disable_external_aux,
    )
    _, _, benign_target_ood_audit = s44.target_known_ood_for_expert(
        "benign", benign_owned, benign_owned, known_split, families,
        args.target_ood_cap_per_class, args.val_cap_per_class,
        args.seed + 230, enabled=not args.disable_target_ood,
    )
    add_audit_rows(rows, "benign_expert", benign_target_ood_audit, role_by_decision)
    add_audit_rows(rows, "benign_expert", benign_external_aux_audit, role_by_decision)

    table = pd.DataFrame(rows).sort_values(["scope", "role", "source_dataset", "class"])
    out_path = os.path.join(
        REPO_ROOT, "scripts", "results",
        f"nfv3_expert_data_composition_{args.target}_unseen_{'-'.join(unseen)}.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\n{table.to_string(index=False)}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

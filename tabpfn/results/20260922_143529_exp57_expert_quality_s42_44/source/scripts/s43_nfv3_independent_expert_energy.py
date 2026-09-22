#!/usr/bin/env python3
"""s43: Independent NF-v3 expert Energy OOD and classification experiment.

This experiment intentionally stops before final routing.  Every expert is
trained independently with:

  ID  = target-train classes owned by that expert
  OOD = target-train non-owned known attacks
        + external NF-v3 attacks that do not semantically match owned classes

Only attack experts are trained.  At test time each expert independently
answers IN/OUT with its Energy score.  If it answers IN, its local attack-class
prediction is evaluated.  Multi-IN and all-OUT samples are counted per class
(5b) but never resolved into a final pipeline prediction in s43.  Benign is
excluded from training and calibration; a capped benign pool is scored as a
diagnostic-only OOD role to inform the future benign/unseen rule.

Unseen target classes are configurable.  If --unseen is omitted, a non-tail
middle-frequency attack is held out so tail classes remain known and their
classification change can be compared with a Global XGBoost baseline.

Output artifacts (CSV + PNG pairs unless noted):
  0a dataset/class/expert protocol map          0b chronological split audit
  1b per-expert OOD composition audit (ablation exclusions recorded)
  2a energy fine-tune history
  3a OOD before/after per role (+benign_diagnostic role, never trained)
  3b OOD after fine-tune                        3c energy histograms (+benign)
  4a per-class XGB vs gated expert PRF with macro/tail/weighted footers
  4b per-class recall funnel: correct_accepted / correct_rejected(gate) /
     wrong_accepted / wrong_rejected(local), oracle-gate local recall,
     dominant_recall_loss attribution, FP sources incl. unseen/benign
  4c accepted-only confusion per expert (true class x local prediction)
  5a expert IN/OUT decision summary (+benign false-IN)
  5b per-class gate overlap: all-OUT / owner-only / non-owner-single /
     multi-IN rates (counted only, never resolved)
  5c per non-owner false-accept on a known class: does the true owner's own
     energy already score lower on that same sample (resolvable_by_argmin_rate,
     a threshold/calibration miss an argmin-over-accepting-experts resolution
     rule would fix for free) or not (genuine feature-space collision)

Ablation knobs: --disable-external-aux and --disable-target-ood drop one OOD
training source each (tag suffix _noextaux/_notgtood) so the external-AUX
contribution is testable one knob at a time.
"""

import argparse
import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from exp_utils import (
    FeatureDataset,
    IndexedDataset,
    TabularMLP,
    auto_margins,
    detect_device,
    energy,
    finetune_energy,
    fit_scaler,
    labels_for,
    load_pickle,
    make_loader,
    make_output_dir,
    ood_metrics,
    parse_hidden_dims,
    pretrain,
    render_table_png,
    scenario_chronological_split,
    set_seed,
    setup_logger,
    sqrt_balanced_weights,
    subset_indices,
    variable_feature_mask,
)


DEFAULT_EXPERT_CONFIG = os.path.join(REPO_ROOT, "configs", "nfv3_experts.json")

# These classes remain known in the default experiment.  The lists reflect
# the rare/future groups used by the preceding NF-v3 experiments.
TAIL_CLASSES = {
    "cse_cic_ids2018": ["bot", "infiltration", "web_attacks"],
    "bot_iot": ["theft"],
    "ton_iot": ["backdoor", "mitm", "ransomware"],
    "unsw_nb15": ["analysis", "backdoor", "shellcode", "worms"],
}

# Middle-frequency, non-tail defaults from the uncapped NF-v3 audit.
DEFAULT_UNSEEN = {
    "cse_cic_ids2018": ["dos"],
    "bot_iot": ["ddos"],
    "ton_iot": ["password"],
    "unsw_nb15": ["generic"],
}

SEMANTIC_ALIASES = {
    "brute_force": "credential_attack",
    "password": "credential_attack",
    "reconnaissance": "reconnaissance",
    "scanning": "reconnaissance",
    "analysis": "web_attack",
    "injection": "web_attack",
    "web_attacks": "web_attack",
    "xss": "web_attack",
}


def semantic_family(name):
    name = str(name)
    return SEMANTIC_ALIASES.get(name, name)


def safe_tag(value):
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    )


def normalize_unseen(values):
    if not values:
        return []
    result = []
    for value in values:
        result.extend(
            part.strip() for part in str(value).split(",") if part.strip()
        )
    return list(dict.fromkeys(result))


def expert_help_epilog():
    lines = [
        "Experiment scope:",
        "  * Each expert independently learns owned ID vs attack-only OOD.",
        "  * OOD train = target non-owned known attacks + external attack AUX.",
        "  * OOD test is reported separately for target non-owned known and unseen.",
        "  * Benign is excluded from training/validation; a capped benign",
        "    pool is only SCORED as a diagnostic role (3a/3c/4c/5a/5b).",
        "  * IN -> run that expert's local classifier.",
        "  * Multi-IN and all-OUT are counted per class (5b) but never",
        "    resolved into a final pipeline prediction in s43.",
        "  * Global XGBoost is a closed-set classification baseline only.",
        "  * --disable-external-aux / --disable-target-ood ablate one OOD",
        "    training source per run (one-knob).",
        "",
        "Attack experts:",
    ]
    try:
        with open(DEFAULT_EXPERT_CONFIG, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        for target, groups in config.items():
            lines.append(f"  [{target}]")
            for group, classes in groups.items():
                lines.append(f"    {group:<18} = {', '.join(classes)}")
            lines.append(
                f"    default unseen     = {', '.join(DEFAULT_UNSEEN[target])}"
            )
            lines.append(
                f"    kept-known tail    = {', '.join(TAIL_CLASSES[target])}"
            )
            lines.append("")
    except (OSError, ValueError, KeyError):
        lines.append(
            f"  Default config could not be read: {DEFAULT_EXPERT_CONFIG}"
        )
        lines.append("")
    lines.extend([
        "OOD rule for expert K:",
        "  Train: target-train non-owned known attacks + external attack AUX.",
        "  Validation: target-val non-owned known attacks + external attack AUX.",
        "  Test: target-test non-owned known attacks and target-test unseen.",
        "  Target unseen is never used for training, validation, or calibration.",
        "  External families semantically matching K's full ownership are excluded.",
        "  Benign is always excluded.",
        "",
        "Examples:",
        "  # Default non-tail middle-frequency unseen",
        "  python src/s43_nfv3_independent_expert_energy.py "
        "--target cse_cic_ids2018",
        "",
        "  # One or more custom unseen classes",
        "  python src/s43_nfv3_independent_expert_energy.py "
        "--target cse_cic_ids2018 \\",
        "      --unseen ddos dos",
        "",
        "  # Comma-separated form is also accepted",
        "  python src/s43_nfv3_independent_expert_energy.py "
        "--target ton_iot \\",
        "      --unseen injection,xss",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "s43: independent expert Energy OOD + local classification; "
            "no final router or unknown/benign resolution."
        ),
        epilog=expert_help_epilog(),
    )
    parser.add_argument(
        "-h", "--help", "--h", action="help",
        help="Show options, expert ownership, default unseen, and OOD rules",
    )
    parser.add_argument(
        "--data", default="data/nfv3_energy_suite_uncapped_scenarios.pkl"
    )
    parser.add_argument(
        "--target", choices=sorted(DEFAULT_UNSEEN),
        default="cse_cic_ids2018",
    )
    parser.add_argument(
        "--unseen", nargs="*", default=None,
        help=(
            "Target classes held out from all training. Accepts multiple values "
            "or comma-separated names. Default: target-specific non-tail middle "
            "class shown in --help."
        ),
    )
    parser.add_argument(
        "--expert-config", default=DEFAULT_EXPERT_CONFIG,
        help="JSON mapping target -> attack expert -> owned classes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dims", default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--pretrain-patience", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--m-in", type=float, default=None)
    parser.add_argument("--m-out", type=float, default=None)
    parser.add_argument("--margin-gap-std", type=float, default=3.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--id-quantile", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--aux-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--max-id-train-per-expert", type=int, default=1_000_000,
        help="Natural random cap over each expert's pooled owned ID train rows",
    )
    parser.add_argument(
        "--max-global-train", type=int, default=2_000_000,
        help="Natural random cap for the Global XGBoost train set",
    )
    parser.add_argument(
        "--test-cap-per-class", type=int, default=100_000,
        help="Maximum known/unseen test rows retained per target class; 0=all",
    )
    parser.add_argument(
        "--val-cap-per-class", type=int, default=50_000,
        help=(
            "Maximum target validation rows retained per class for both owned "
            "ID and non-owned known OOD; 0=all"
        ),
    )
    parser.add_argument(
        "--target-ood-cap-per-class", type=int, default=100_000,
        help=(
            "Maximum target-train non-owned known OOD rows retained per class "
            "for each expert; 0=all"
        ),
    )
    parser.add_argument(
        "--aux-cap-per-source-family", type=int, default=20_000,
        help=(
            "Maximum external rows retained per dataset-family before the "
            "AUX train/validation split; 0=all"
        ),
    )
    parser.add_argument("--aux-val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--disable-external-aux", action="store_true",
        help=(
            "Ablation: train expert OOD without external attack AUX "
            "(target non-owned known attacks only); tag suffix _noextaux"
        ),
    )
    parser.add_argument(
        "--disable-target-ood", action="store_true",
        help=(
            "Ablation: train expert OOD without target non-owned known "
            "attacks (external attack AUX only); tag suffix _notgtood"
        ),
    )
    parser.add_argument(
        "--benign-diag-cap", type=int, default=100_000,
        help=(
            "Target benign rows scored per expert as a diagnostic-only OOD "
            "role; never used for training or calibration; 0=all"
        ),
    )
    parser.add_argument("--hist-max-per-role", type=int, default=50_000)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=8)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument(
        "--xgb-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--out-root", default="results")
    return parser.parse_args()


def load_expert_groups(path, target):
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if target not in config:
        raise ValueError(
            f"Expert config has no target {target!r}; available={sorted(config)}"
        )
    attack_groups = {
        str(group): [str(name) for name in names]
        for group, names in config[target].items()
    }
    if not attack_groups or any(not names for names in attack_groups.values()):
        raise ValueError(f"Invalid empty expert group in {path}: {attack_groups}")
    return attack_groups


def validate_expert_partition(experts, target_classes):
    assigned = [name for names in experts.values() for name in names]
    if sorted(assigned) != sorted(target_classes):
        raise ValueError(
            "Experts must partition every target class exactly once; "
            f"target={sorted(target_classes)}, assigned={sorted(assigned)}"
        )
    if len(assigned) != len(set(assigned)):
        raise ValueError(f"Duplicate class ownership in experts: {experts}")


def cap_by_family(indices, families, maximum, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if maximum <= 0:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    selected = []
    for family in sorted(np.unique(families[indices])):
        candidates = indices[families[indices] == family]
        if len(candidates) > maximum:
            candidates = rng.choice(candidates, maximum, replace=False)
        selected.extend(candidates.tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def external_aux_groups(
    datasets, families, target, cap_per_group, val_fraction, seed,
):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--aux-val-fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    groups = {}
    records = []
    for source in sorted(set(datasets.tolist()) - {target}):
        source_indices = np.flatnonzero(datasets == source)
        for family in sorted(np.unique(families[source_indices])):
            candidates = source_indices[families[source_indices] == family]
            available = len(candidates)
            if cap_per_group > 0 and len(candidates) > cap_per_group:
                candidates = rng.choice(
                    candidates, cap_per_group, replace=False
                )
            candidates = np.asarray(candidates, dtype=np.int64)
            rng.shuffle(candidates)
            n_val = max(1, int(round(len(candidates) * val_fraction)))
            n_val = min(n_val, len(candidates) - 1)
            if n_val <= 0:
                raise ValueError(
                    f"External AUX group is too small: {source}::{family}"
                )
            groups[(str(source), str(family))] = {
                "train": np.sort(candidates[n_val:]),
                "val": np.sort(candidates[:n_val]),
                "available": available,
            }
            records.append({
                "source": str(source), "family": str(family),
                "semantic_family": semantic_family(family),
                "available": available, "selected_total": len(candidates),
                "aux_train": len(candidates) - n_val, "aux_validation": n_val,
            })
    return groups, pd.DataFrame(records)


def external_aux_for_expert(
    group_name, configured_owned, aux_groups, enabled=True,
):
    excluded_semantics = {
        semantic_family(name) for name in configured_owned
    }
    train, validation, audit = [], [], []
    for (source, family), group in sorted(aux_groups.items()):
        semantic_overlap = semantic_family(family) in excluded_semantics
        benign_disabled = family == "benign"
        included = enabled and not semantic_overlap and not benign_disabled
        if not enabled:
            decision = "EXCLUDED: --disable-external-aux"
        elif semantic_overlap:
            decision = "EXCLUDED: owned semantic"
        elif benign_disabled:
            decision = "EXCLUDED: benign policy"
        else:
            decision = "EXTERNAL AUX OOD"
        if included:
            train.extend(group["train"].tolist())
            validation.extend(group["val"].tolist())
        audit.append({
            "expert": group_name,
            "configured_owned_classes": ",".join(configured_owned),
            "excluded_semantics": ",".join(sorted(excluded_semantics)),
            "source": source, "family": family,
            "semantic_family": semantic_family(family),
            "included_as_aux_ood": included,
            "decision": decision,
            "available": int(group["available"]),
            "aux_train_selected": int(len(group["train"]) if included else 0),
            "aux_validation_selected": int(
                len(group["val"]) if included else 0
            ),
        })
    if enabled and (not train or not validation):
        raise ValueError(
            f"Semantic exclusions left no AUX data for expert {group_name}"
        )
    return (
        np.sort(np.asarray(train, dtype=np.int64)),
        np.sort(np.asarray(validation, dtype=np.int64)),
        audit,
    )


def target_known_ood_for_expert(
    group_name, active_owned, configured_owned, known_split, families,
    train_cap, val_cap, seed, enabled=True,
):
    train_candidates = known_split["train"][
        ~np.isin(families[known_split["train"]], active_owned)
    ]
    val_candidates = known_split["val"][
        ~np.isin(families[known_split["val"]], active_owned)
    ]
    if not enabled:
        train_candidates = train_candidates[:0]
        val_candidates = val_candidates[:0]
    train = cap_by_family(
        train_candidates, families, train_cap, seed
    )
    validation = cap_by_family(
        val_candidates, families, val_cap, seed + 1
    )
    audit = []
    known_classes = sorted(np.unique(
        np.concatenate([
            families[known_split["train"]],
            families[known_split["val"]],
        ])
    ))
    for family in known_classes:
        owned = family in active_owned
        family_train = known_split["train"][
            families[known_split["train"]] == family
        ]
        family_val = known_split["val"][
            families[known_split["val"]] == family
        ]
        audit.append({
            "expert": group_name,
            "configured_owned_classes": ",".join(configured_owned),
            "excluded_semantics": "",
            "source": "TARGET known attack",
            "family": str(family),
            "semantic_family": semantic_family(family),
            "included_as_aux_ood": (not owned) and enabled,
            "decision": (
                "EXCLUDED: owned ID" if owned
                else (
                    "TARGET KNOWN OOD" if enabled
                    else "EXCLUDED: --disable-target-ood"
                )
            ),
            "available": int(len(family_train) + len(family_val)),
            "aux_train_selected": int(np.sum(families[train] == family)),
            "aux_validation_selected": int(
                np.sum(families[validation] == family)
            ),
        })
    return train, validation, audit


def xgb_model(n_classes, args):
    device = args.xgb_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    parameters = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        min_child_weight=args.xgb_min_child_weight,
        reg_lambda=args.xgb_reg_lambda,
        tree_method="hist", device=device, random_state=args.seed,
        n_jobs=args.xgb_n_jobs,
    )
    if n_classes == 2:
        parameters.update(objective="binary:logistic", eval_metric="logloss")
    else:
        parameters.update(
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss",
        )
    return xgb.XGBClassifier(**parameters)


def train_global_xgb(X, train_indices, train_labels, args, log):
    n_classes = int(train_labels.max()) + 1
    counts = np.maximum(
        np.bincount(train_labels, minlength=n_classes), 1
    )
    class_weight = np.sqrt(len(train_labels) / (n_classes * counts))
    model = xgb_model(n_classes, args)
    start = time.time()
    model.fit(
        X[train_indices], train_labels,
        sample_weight=class_weight[train_labels], verbose=False,
    )
    log.info("Global XGBoost trained in %.1fs", time.time() - start)
    return model


def collect_expert_outputs(
    model, loader, device, temperature, local_class_count,
):
    model.eval()
    energies, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            features = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits = model(features.to(device, non_blocking=True))
            energies.append(energy(logits, temperature).cpu().numpy())
            predictions.append(
                logits[:, :local_class_count].argmax(dim=1).cpu().numpy()
            )
    return {
        "energy": np.concatenate(energies).astype(np.float32),
        "prediction": np.concatenate(predictions).astype(np.int32),
    }


def sampled(values, maximum, seed):
    values = np.asarray(values)
    if maximum <= 0 or len(values) <= maximum:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), maximum, replace=False)]


def _draw_table_separators(axis, table, row_starts, n_columns):
    axis.figure.canvas.draw()
    for row_start, width, color in row_starts:
        row = row_start + 1  # header is row 0
        left = table[(row, 0)].get_x()
        right_cell = table[(row, n_columns - 1)]
        right = right_cell.get_x() + right_cell.get_width()
        y = table[(row, 0)].get_y() + table[(row, 0)].get_height()
        axis.plot(
            [left, right], [y, y], transform=axis.transAxes,
            color=color, linewidth=width, solid_capstyle="butt", zorder=20,
        )


def build_dataset_class_map(
    datasets, families, config_path, target, unseen,
):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    rows = []
    dataset_order = [target] + [
        name for name in sorted(config) if name != target
    ]
    for dataset_name in dataset_order:
        dataset_indices = np.flatnonzero(datasets == dataset_name)
        counts = {
            str(name): int(count)
            for name, count in zip(
                *np.unique(families[dataset_indices], return_counts=True)
            )
        }
        ownership = {
            class_name: expert
            for expert, names in config[dataset_name].items()
            for class_name in names
        }
        for class_name, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        ):
            if class_name == "benign":
                expert = "not evaluated"
            else:
                expert = ownership[class_name]
            if class_name == "benign":
                role = "excluded benign"
            elif dataset_name != target:
                role = "external AUX source"
            elif class_name in unseen:
                role = "target unseen OOD"
            else:
                role = "target known: owner ID / non-owner OOD"
            rows.append({
                "dataset": dataset_name, "class": class_name,
                "sample_count": count, "expert": expert, "role": role,
                "is_tail": (
                    class_name in TAIL_CLASSES.get(dataset_name, [])
                ),
            })
    return pd.DataFrame(rows)


def save_dataset_class_map_png(frame, path):
    columns = ["dataset", "class", "sample_count", "expert", "role", "is_tail"]
    display = frame[columns].copy()
    display["sample_count"] = display["sample_count"].map(
        lambda value: f"{int(value):,}"
    )
    display["is_tail"] = display["is_tail"].map(
        lambda value: "tail" if value else ""
    )
    fig_height = max(9, 0.34 * len(display) + 1.8)
    fig, axis = plt.subplots(figsize=(16, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=display.values, colLabels=columns,
        cellLoc="left", colLoc="center", loc="center",
        colWidths=[0.18, 0.16, 0.13, 0.18, 0.27, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#243b5a")
        table[(0, column)].get_text().set_color("white")
        table[(0, column)].get_text().set_weight("bold")
    palette = [
        "#dcecff", "#e7dcf7", "#ffe3bd", "#d9f0df",
        "#f7d9df", "#d8eef2",
    ]
    expert_colors = {}
    row_starts = []
    previous_dataset = None
    for row_index, row in enumerate(frame.itertuples(index=False)):
        if row.dataset != previous_dataset:
            row_starts.append((row_index, 2.8, "#182433"))
            previous_dataset = row.dataset
        key = (row.dataset, row.expert)
        if row.expert == "not evaluated":
            color = "#e5e7eb"
        else:
            if key not in expert_colors:
                expert_colors[key] = palette[
                    len(expert_colors) % len(palette)
                ]
            color = expert_colors[key]
        table[(row_index + 1, 3)].set_facecolor(color)
        if "unseen" in row.role:
            table[(row_index + 1, 4)].set_facecolor("#ffd6d6")
        elif "known attack" in row.role:
            table[(row_index + 1, 4)].set_facecolor("#d9f3df")
        table[(row_index + 1, 2)].get_text().set_ha("right")
        table[(row_index + 1, 2)].get_text().set_fontfamily("DejaVu Sans Mono")
        for column in range(len(columns)):
            table[(row_index + 1, column)].set_edgecolor("#aeb7c3")
    _draw_table_separators(axis, table, row_starts, len(columns))
    axis.set_title(
        "NF-v3 class counts (descending within dataset) and expert ownership",
        fontsize=15, pad=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_aux_composition_png(frame, path):
    columns = [
        "expert", "source", "family", "semantic_family", "available",
        "aux_train_selected", "aux_validation_selected", "decision",
    ]
    display = frame[columns].copy()
    for column in [
        "available", "aux_train_selected", "aux_validation_selected"
    ]:
        display[column] = display[column].map(lambda value: f"{int(value):,}")
    fig_height = max(10, 0.28 * len(display) + 2.0)
    fig, axis = plt.subplots(figsize=(18, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=display.values, colLabels=columns,
        cellLoc="left", colLoc="center", loc="center",
        colWidths=[0.12, 0.15, 0.14, 0.14, 0.10, 0.12, 0.13, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#243b5a")
        table[(0, column)].get_text().set_color("white")
        table[(0, column)].get_text().set_weight("bold")
    row_starts = []
    previous_expert, previous_source = None, None
    for row_index, row in enumerate(frame.itertuples(index=False)):
        if row.expert != previous_expert:
            row_starts.append((row_index, 3.4, "#111827"))
            previous_expert, previous_source = row.expert, row.source
        elif row.source != previous_source:
            row_starts.append((row_index, 2.2, "#4b5563"))
            previous_source = row.source
        decision_cell = table[(row_index + 1, 7)]
        if str(row.decision).endswith("OOD"):
            decision_cell.set_facecolor("#bfe8c8")
        elif row.decision in {
            "EXCLUDED: owned semantic", "EXCLUDED: owned ID"
        }:
            decision_cell.set_facecolor("#f4b8b8")
        else:
            decision_cell.set_facecolor("#e5e7eb")
        for column in [4, 5, 6]:
            table[(row_index + 1, column)].get_text().set_ha("right")
            table[(row_index + 1, column)].get_text().set_fontfamily(
                "DejaVu Sans Mono"
            )
        for column in range(len(columns)):
            table[(row_index + 1, column)].set_edgecolor("#b5bec9")
    _draw_table_separators(axis, table, row_starts, len(columns))
    axis.set_title(
        "Expert-specific OOD composition "
        "(red=owned excluded, green=OOD, gray=benign excluded)",
        fontsize=15, pad=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_expert_histogram_grid(
    expert_results, stage, ood_frame, args, out_dir,
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    axes = axes.ravel()
    roles = [
        ("owned", "#377eb8", "owned attack ID", "bar"),
        ("non_owned", "#ff7f00", "non-owned known attack", "line"),
        ("unseen", "#e41a1c", "target unseen", "line"),
        ("external_aux", "#984ea3", "external attack AUX val", "line"),
        (
            "target_known_ood_val", "#4daf4a",
            "target non-owned known val", "line",
        ),
        ("benign_diag", "#636363", "target benign (diagnostic)", "line"),
    ]
    for expert_index, (group, result) in enumerate(expert_results.items()):
        axis = axes[expert_index]
        output = result["stage_outputs"][stage]
        for role_index, (key, color, label, kind) in enumerate(roles):
            values = sampled(
                output[key], args.hist_max_per_role,
                args.seed + 300 + expert_index * 10 + role_index,
            )
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            numeric_values = values.astype(np.float64)
            lower = float(numeric_values.min())
            upper = float(numeric_values.max())
            padding = max(1.0, abs(lower), abs(upper)) * 1e-6
            if upper <= lower:
                lower -= padding
                upper += padding
            bins = np.linspace(lower, upper, 71, dtype=np.float64)
            axis.hist(
                values, bins=bins, density=True,
                alpha=0.42 if kind == "bar" else 1.0,
                histtype="bar" if kind == "bar" else "step",
                linewidth=1.6, color=color, label=label,
            )
        threshold = output["threshold"]
        axis.axvline(
            threshold, color="black", linestyle="--", linewidth=1.4,
            label=f"ID q{int(args.id_quantile * 100)}={threshold:.2f}",
        )
        pooled = ood_frame[
            (ood_frame["expert"] == group)
            & (ood_frame["stage"] == stage)
            & (ood_frame["ood_role"] == "pooled_attack_out")
        ].iloc[0]
        axis.set_title(
            f"{group} ({', '.join(result['active_owned'])}) | "
            f"FPR95={pooled.fpr95:.3f}"
        )
        axis.set_xlabel("Energy E(x); larger = more OOD-like")
        axis.set_ylabel("Density")
        axis.legend(fontsize=7)
    for axis in axes[len(expert_results):]:
        axis.axis("off")
    stage_title = (
        "Before attack-OOD Energy fine-tuning"
        if stage == "pretrained"
        else "After attack-OOD Energy fine-tuning"
    )
    fig.suptitle(f"Independent attack expert OOD | {stage_title}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = "pretrained" if stage == "pretrained" else "finetuned"
    fig.savefig(
        os.path.join(out_dir, f"3c_energy_hist_2x2_{suffix}.png"),
        dpi=180, bbox_inches="tight",
    )
    plt.close(fig)


def binary_class_metrics(y_true, y_pred, class_id):
    true_positive = y_true == class_id
    predicted_positive = y_pred == class_id
    tp = int(np.sum(true_positive & predicted_positive))
    fp = int(np.sum(~true_positive & predicted_positive))
    fn = int(np.sum(true_positive & ~predicted_positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1, int(true_positive.sum())


def accepted_confusion_rows(
    expert_results, known_classes, y_known_test, families, unseen_test, unseen,
):
    """Long-form confusion over gate-accepted samples only.

    True rows cover target known classes, each unseen class (``unseen::x``)
    and the never-trained benign diagnostic pool; columns are the local
    predictions of the accepting expert mapped to global class names."""
    rows = []
    for group, result in expert_results.items():
        final = result["stage_outputs"]["energy_finetuned"]
        pred_names = [
            known_classes[index] for index in result["local_to_global"]
        ]

        def add(true_name, true_role, accepted_mask, local_prediction):
            support_total = int(len(accepted_mask))
            local_prediction = local_prediction[accepted_mask]
            for local_id, count in zip(
                *np.unique(local_prediction, return_counts=True)
            ):
                rows.append({
                    "expert": group,
                    "true_class": true_name,
                    "true_role": true_role,
                    "predicted_class": pred_names[int(local_id)],
                    "count": int(count),
                    "true_class_test_support": support_total,
                    "share_of_test_support": float(count / support_total),
                })

        for class_id, class_name in enumerate(known_classes):
            true_mask = y_known_test == class_id
            add(
                class_name, "target known",
                result["accepted"][true_mask],
                final["known_local_prediction"][true_mask],
            )
        for unseen_name in unseen:
            unseen_mask = families[unseen_test] == unseen_name
            add(
                f"unseen::{unseen_name}", "target unseen",
                result["unseen_accepted"][unseen_mask],
                final["unseen_local_prediction"][unseen_mask],
            )
        if len(result["benign_accepted"]):
            add(
                "benign_diagnostic", "benign diagnostic",
                result["benign_accepted"],
                final["benign_local_prediction"],
            )
    return pd.DataFrame(rows)


def save_accepted_confusion_png(
    confusion_frame, expert_results, known_classes, unseen, has_benign, path,
):
    true_order = list(known_classes)
    true_order.extend(f"unseen::{name}" for name in unseen)
    if has_benign:
        true_order.append("benign_diagnostic")
    n_experts = len(expert_results)
    n_columns = 2
    n_rows = int(np.ceil(n_experts / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(7.5 * n_columns, 0.62 * len(true_order) * n_rows + 2.5),
        squeeze=False,
    )
    axes = axes.ravel()
    for axis_index, (group, result) in enumerate(expert_results.items()):
        axis = axes[axis_index]
        pred_names = [
            known_classes[index] for index in result["local_to_global"]
        ]
        share = np.zeros((len(true_order), len(pred_names)))
        counts = np.zeros_like(share, dtype=np.int64)
        subset = confusion_frame[confusion_frame["expert"] == group]
        for true_class, predicted_class, count, row_share in subset[
            ["true_class", "predicted_class", "count", "share_of_test_support"]
        ].itertuples(index=False, name=None):
            row_index = true_order.index(true_class)
            column_index = pred_names.index(predicted_class)
            counts[row_index, column_index] = count
            share[row_index, column_index] = row_share
        axis.imshow(
            share, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0,
        )
        axis.set_xticks(range(len(pred_names)))
        axis.set_xticklabels(pred_names, rotation=30, ha="right", fontsize=8)
        axis.set_yticks(range(len(true_order)))
        axis.set_yticklabels(true_order, fontsize=8)
        owned = set(result["active_owned"])
        for row_index, true_name in enumerate(true_order):
            for column_index in range(len(pred_names)):
                axis.text(
                    column_index, row_index,
                    f"{counts[row_index, column_index]:,}\n"
                    f"{share[row_index, column_index]:.3f}",
                    ha="center", va="center", fontsize=6.5,
                )
            if true_name in owned:
                owner_column = pred_names.index(true_name)
                axis.add_patch(plt.Rectangle(
                    (owner_column - 0.5, row_index - 0.5), 1, 1,
                    fill=False, edgecolor="#1f6feb", linewidth=2.0,
                ))
        axis.set_title(
            f"{group} ({', '.join(result['active_owned'])})", fontsize=10
        )
        axis.set_xlabel("local prediction of accepted samples")
    for axis in axes[n_experts:]:
        axis.axis("off")
    fig.suptitle(
        "Accepted-only confusion: count and share of each true class' test "
        "support entering this expert (blue box = owner cell; off-owner mass "
        "explains precision loss)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def multi_in_energy_rows(
    expert_order, energy_known, accept_known, threshold_by_expert,
    known_classes, y_known_test, active_experts,
):
    """Per non-owner false-accept on a known attack class, compare the true
    owner's energy against the intruding expert's energy on the SAME sample.

    This tells apart two causes of a bad non_owned_known_attack FPR95: a
    threshold-only miss, where the owner is still more confident (lower
    energy) and an argmin-over-accepting-experts resolution rule would
    already route the sample correctly (resolvable_by_argmin_rate), versus a
    genuine feature-space collision, where the owner is not even more
    confident than the intruder and no per-sample energy signal can fix it.
    """
    owner_of_class = {
        class_name: next(
            name for name, owned in active_experts.items()
            if class_name in owned
        )
        for class_name in known_classes
    }
    rows = []
    for class_id, class_name in enumerate(known_classes):
        owner = owner_of_class[class_name]
        owner_index = expert_order.index(owner)
        true_mask = y_known_test == class_id
        owner_energy = energy_known[owner_index][true_mask]
        owner_accepted = accept_known[owner_index][true_mask]
        for intruder_index, intruder in enumerate(expert_order):
            if intruder == owner:
                continue
            false_accept = accept_known[intruder_index][true_mask]
            if not false_accept.any():
                continue
            intruder_energy = energy_known[intruder_index][true_mask]
            owner_lower = owner_energy < intruder_energy
            rows.append({
                "true_class": class_name,
                "owner_expert": owner,
                "intruding_expert": intruder,
                "false_accept_count": int(false_accept.sum()),
                "true_class_test_support": int(true_mask.sum()),
                "owner_also_in_rate": float(
                    owner_accepted[false_accept].mean()
                ),
                "owner_lower_energy_rate": float(
                    owner_lower[false_accept].mean()
                ),
                "resolvable_by_argmin_rate": float(
                    (owner_accepted & owner_lower)[false_accept].mean()
                ),
                "mean_owner_energy": float(owner_energy[false_accept].mean()),
                "owner_threshold": float(threshold_by_expert[owner_index]),
                "mean_intruder_energy": float(
                    intruder_energy[false_accept].mean()
                ),
                "intruder_threshold": float(
                    threshold_by_expert[intruder_index]
                ),
            })
    columns = [
        "true_class", "owner_expert", "intruding_expert",
        "false_accept_count", "true_class_test_support",
        "owner_also_in_rate", "owner_lower_energy_rate",
        "resolvable_by_argmin_rate", "mean_owner_energy",
        "owner_threshold", "mean_intruder_energy", "intruder_threshold",
    ]
    return pd.DataFrame(rows, columns=columns)


def main():
    args = parse_args()
    set_seed(args.seed)
    suite = load_pickle(args.data)
    required = {
        "X", "dataset_names", "families", "attack_scenarios", "timestamps"
    }
    missing = sorted(required - set(suite))
    if missing:
        raise ValueError(f"Missing suite fields: {missing}")

    datasets = np.asarray(suite["dataset_names"], dtype=object)
    families = np.asarray(suite["families"], dtype=object)
    scenarios = np.asarray(suite["attack_scenarios"], dtype=object)
    timestamps = np.asarray(suite["timestamps"], dtype=np.int64)
    target_indices = np.flatnonzero(datasets == args.target)
    if not len(target_indices):
        raise ValueError(f"Target {args.target!r} is absent from {args.data}")
    target_classes = sorted(np.unique(families[target_indices]).tolist())
    target_attack_classes = [
        name for name in target_classes if name != "benign"
    ]
    experts = load_expert_groups(args.expert_config, args.target)
    validate_expert_partition(experts, target_attack_classes)

    unseen = normalize_unseen(args.unseen)
    used_default_unseen = not unseen
    if used_default_unseen:
        unseen = list(DEFAULT_UNSEEN[args.target])
    missing_unseen = sorted(set(unseen) - set(target_classes))
    if missing_unseen:
        raise ValueError(
            f"Unseen classes absent from target: {missing_unseen}; "
            f"available={target_classes}"
        )
    if "benign" in unseen:
        raise ValueError("benign cannot be held out as unseen in s43")

    tail_classes = sorted(
        set(TAIL_CLASSES.get(args.target, [])) & set(target_attack_classes)
    )
    tail_removed = sorted(set(unseen) & set(tail_classes))
    active_experts = {
        group: [name for name in owned if name not in unseen]
        for group, owned in experts.items()
    }
    active_experts = {
        group: owned for group, owned in active_experts.items() if owned
    }
    known_classes = [
        name for name in target_attack_classes if name not in unseen
    ]
    if len(known_classes) < 2:
        raise ValueError(f"Too few known classes after unseen={unseen}")

    if args.disable_external_aux and args.disable_target_ood:
        raise ValueError(
            "Both OOD training sources are disabled; keep at least one of "
            "external AUX and target non-owned known attacks"
        )
    tag = (
        f"nfv3_{safe_tag(args.target)}_independent_expert_energy_"
        f"unseen_{'-'.join(safe_tag(name) for name in unseen)}"
    )
    if args.disable_external_aux:
        tag += "_noextaux"
    if args.disable_target_ood:
        tag += "_notgtood"
    out_dir = make_output_dir(args.out_root, tag)
    log = setup_logger(os.path.join(out_dir, "experiment.log"))
    log.info("Args: %s", vars(args))
    log.info(
        "Target=%s unseen=%s default_unseen=%s known_tail=%s experts=%s",
        args.target, unseen, used_default_unseen, tail_classes, active_experts,
    )
    if tail_removed:
        log.warning(
            "Custom unseen removes tail classes=%s; those classes cannot be "
            "used to measure known-tail classification improvement.",
            tail_removed,
        )

    protocol_table = build_dataset_class_map(
        datasets, families, args.expert_config, args.target, unseen
    )
    protocol_table.to_csv(
        os.path.join(out_dir, "0a_dataset_class_expert_map.csv"), index=False
    )
    save_dataset_class_map_png(
        protocol_table,
        os.path.join(out_dir, "0a_dataset_class_expert_map.png"),
    )

    target_attack_indices = target_indices[
        families[target_indices] != "benign"
    ]
    split, split_audit = scenario_chronological_split(
        target_attack_indices, scenarios, timestamps
    )
    split_audit.to_csv(
        os.path.join(out_dir, "0b_split_audit.csv"), index=False
    )
    render_table_png(
        split_audit, os.path.join(out_dir, "0b_split_audit.png"),
        title="Scenario-wise chronological 60/20/20 split",
    )
    known_split = {
        role: indices[np.isin(families[indices], known_classes)]
        for role, indices in split.items()
    }
    unseen_test = split["test"][np.isin(families[split["test"]], unseen)]
    if not len(unseen_test):
        raise ValueError(f"No unseen test rows after split: {unseen}")
    known_test = cap_by_family(
        known_split["test"], families, args.test_cap_per_class, args.seed + 11
    )
    unseen_test = cap_by_family(
        unseen_test, families, args.test_cap_per_class, args.seed + 12
    )
    # Diagnostic-only benign pool: scored by every expert, never trained on,
    # never used for thresholds.  Informs the future benign/unseen rule.
    benign_pool = target_indices[families[target_indices] == "benign"]
    benign_diag = subset_indices(
        benign_pool, args.benign_diag_cap, args.seed + 13
    )

    X = np.asarray(suite["X"], dtype=np.float32)
    variable = variable_feature_mask(X, known_split["train"])
    if not np.any(variable):
        raise ValueError("No variable feature remains in known target train")
    log.info(
        "Rows attack_train=%s attack_val=%s attack_test=%s unseen_test=%s "
        "benign_diag=%s features=%s variable_features=%s",
        f"{len(known_split['train']):,}", f"{len(known_split['val']):,}",
        f"{len(known_test):,}", f"{len(unseen_test):,}",
        f"{len(benign_diag):,}", X.shape[1], int(variable.sum()),
    )

    global_train = subset_indices(
        known_split["train"], args.max_global_train, args.seed + 20
    )
    y_global_train = labels_for(global_train, families, known_classes)
    y_known_test = labels_for(known_test, families, known_classes)
    global_model = train_global_xgb(
        X, global_train, y_global_train, args, log
    )
    global_prediction = global_model.predict(X[known_test]).astype(np.int64)
    global_precision, global_recall, global_f1, global_support = (
        precision_recall_fscore_support(
            y_known_test, global_prediction,
            labels=np.arange(len(known_classes)), zero_division=0,
        )
    )

    aux_groups, _ = external_aux_groups(
        datasets, families, args.target,
        args.aux_cap_per_source_family, args.aux_val_fraction,
        args.seed + 30,
    )

    device = detect_device()
    log.info("Torch device=%s", device)
    all_aux_audit = []
    all_histories = []
    ood_rows = []
    expert_results = {}
    class_to_known_id = {
        name: class_id for class_id, name in enumerate(known_classes)
    }

    for expert_number, (group, active_owned) in enumerate(
        active_experts.items()
    ):
        configured_owned = experts[group]
        log.info(
            "Expert=%s active_owned=%s configured_owned=%s",
            group, active_owned, configured_owned,
        )
        owned_train = known_split["train"][
            np.isin(families[known_split["train"]], active_owned)
        ]
        owned_val = known_split["val"][
            np.isin(families[known_split["val"]], active_owned)
        ]
        owned_val = cap_by_family(
            owned_val, families, args.val_cap_per_class,
            args.seed + 90 + expert_number,
        )
        owned_train = subset_indices(
            owned_train, args.max_id_train_per_expert,
            args.seed + 100 + expert_number,
        )
        if not len(owned_train) or not len(owned_val):
            raise ValueError(
                f"Expert {group} has empty ID train/validation rows"
            )
        local_mapping = {
            name: class_id for class_id, name in enumerate(active_owned)
        }
        y_train = np.asarray(
            [local_mapping[str(name)] for name in families[owned_train]],
            dtype=np.int64,
        )
        y_val = np.asarray(
            [local_mapping[str(name)] for name in families[owned_val]],
            dtype=np.int64,
        )
        external_aux_train, external_aux_val, external_aux_audit = (
            external_aux_for_expert(
                group, configured_owned, aux_groups,
                enabled=not args.disable_external_aux,
            )
        )
        target_ood_train, target_ood_val, target_ood_audit = (
            target_known_ood_for_expert(
                group, active_owned, configured_owned, known_split, families,
                args.target_ood_cap_per_class, args.val_cap_per_class,
                args.seed + 130 + expert_number * 10,
                enabled=not args.disable_target_ood,
            )
        )
        ood_train = np.sort(np.concatenate([
            target_ood_train, external_aux_train,
        ]))
        ood_val = np.sort(np.concatenate([
            target_ood_val, external_aux_val,
        ]))
        if not len(ood_train):
            raise ValueError(f"Expert {group} has no OOD training rows")
        all_aux_audit.extend(target_ood_audit)
        all_aux_audit.extend(external_aux_audit)
        mean, scale = fit_scaler(X, owned_train)
        train_loader = make_loader(
            IndexedDataset(X, owned_train, y_train, mean, scale),
            args.batch_size, True, args.workers,
        )
        val_loader = make_loader(
            IndexedDataset(X, owned_val, y_val, mean, scale),
            args.test_batch_size, False, args.workers,
        )
        ood_train_loader = make_loader(
            FeatureDataset(X, ood_train, mean, scale),
            args.aux_batch_size, True, args.workers,
        )
        target_ood_val_loader = (
            make_loader(
                FeatureDataset(X, target_ood_val, mean, scale),
                args.test_batch_size, False, args.workers,
            )
            if len(target_ood_val) else None
        )
        external_aux_val_loader = (
            make_loader(
                FeatureDataset(X, external_aux_val, mean, scale),
                args.test_batch_size, False, args.workers,
            )
            if len(external_aux_val) else None
        )
        known_test_loader = make_loader(
            FeatureDataset(X, known_test, mean, scale),
            args.test_batch_size, False, args.workers,
        )
        unseen_loader = make_loader(
            FeatureDataset(X, unseen_test, mean, scale),
            args.test_batch_size, False, args.workers,
        )
        benign_loader = (
            make_loader(
                FeatureDataset(X, benign_diag, mean, scale),
                args.test_batch_size, False, args.workers,
            )
            if len(benign_diag) else None
        )

        # A singleton expert uses a two-logit Energy head.  Only the first logit
        # is a real local class; the reserve logit prevents zero-gradient
        # one-class cross-entropy during ID pretraining.
        output_dim = max(2, len(active_owned))
        model = TabularMLP(
            X.shape[1], parse_hidden_dims(args.hidden_dims),
            output_dim, args.dropout,
        ).to(device)
        class_weights = sqrt_balanced_weights(
            y_train, output_dim
        ).to(device)
        best_pretrain_f1 = pretrain(
            model, train_loader, val_loader, class_weights,
            args, device, log,
        )
        stage_models = {"pretrained": copy.deepcopy(model)}
        auto_in, auto_out, energy_std = auto_margins(
            model, val_loader, device, args.temperature,
            args.margin_gap_std,
        )
        m_in = auto_in if args.m_in is None else args.m_in
        m_out = auto_out if args.m_out is None else args.m_out
        if m_out <= m_in:
            raise ValueError(
                f"Expert {group}: m_out must exceed m_in, "
                f"got {m_in} and {m_out}"
        )
        history = finetune_energy(
            model, train_loader, ood_train_loader, val_loader,
            class_weights, m_in, m_out, args, device, log,
        )
        history.insert(0, "expert", group)
        all_histories.append(history)
        stage_models["energy_finetuned"] = copy.deepcopy(model)
        log.info(
            "Expert=%s ID_train=%s target_known_OOD_train=%s "
            "external_AUX_train=%s OOD_val=%s "
            "pretrain_F1=%.4f margins=(%.4f,%.4f) energy_std=%.4f",
            group, f"{len(owned_train):,}", f"{len(target_ood_train):,}",
            f"{len(external_aux_train):,}", f"{len(ood_val):,}",
            best_pretrain_f1,
            m_in, m_out, energy_std,
        )

        stage_outputs = {}
        for stage, stage_model in stage_models.items():
            stage_model = stage_model.to(device)
            val_output = collect_expert_outputs(
                stage_model, val_loader, device, args.temperature,
                len(active_owned),
            )
            known_output = collect_expert_outputs(
                stage_model, known_test_loader, device, args.temperature,
                len(active_owned),
            )
            unseen_output = collect_expert_outputs(
                stage_model, unseen_loader, device, args.temperature,
                len(active_owned),
            )
            if target_ood_val_loader is None:
                target_ood_val_output = {
                    "energy": np.empty(0, dtype=np.float32),
                    "prediction": np.empty(0, dtype=np.int32),
                }
            else:
                target_ood_val_output = collect_expert_outputs(
                    stage_model, target_ood_val_loader, device,
                    args.temperature, len(active_owned),
                )
            empty_output = {
                "energy": np.empty(0, dtype=np.float32),
                "prediction": np.empty(0, dtype=np.int32),
            }
            external_aux_output = (
                collect_expert_outputs(
                    stage_model, external_aux_val_loader, device,
                    args.temperature, len(active_owned),
                )
                if external_aux_val_loader is not None else dict(empty_output)
            )
            benign_output = (
                collect_expert_outputs(
                    stage_model, benign_loader, device, args.temperature,
                    len(active_owned),
                )
                if benign_loader is not None else dict(empty_output)
            )
            threshold = float(np.quantile(
                val_output["energy"], args.id_quantile
            ))
            owned_mask = np.isin(families[known_test], active_owned)
            role_energies = {
                "non_owned_known_attack": known_output["energy"][~owned_mask],
                "unseen": unseen_output["energy"],
                "target_non_owned_validation": (
                    target_ood_val_output["energy"]
                ),
                "external_aux_validation": external_aux_output["energy"],
                "benign_diagnostic": benign_output["energy"],
            }
            pooled_attack_out = np.concatenate([
                role_energies["non_owned_known_attack"],
                role_energies["unseen"],
            ])
            role_energies["pooled_attack_out"] = pooled_attack_out
            owned_energy = known_output["energy"][owned_mask]
            for role, ood_energy_values in role_energies.items():
                if not len(ood_energy_values):
                    continue
                metrics = ood_metrics(
                    owned_energy, ood_energy_values, threshold
                )
                ood_rows.append({
                    "expert": group, "stage": stage,
                    "active_owned_classes": ",".join(active_owned),
                    "configured_owned_classes": ",".join(configured_owned),
                    "ood_role": role,
                    "id_support": len(owned_energy),
                    "ood_support": len(ood_energy_values),
                    "threshold": threshold, **metrics,
                })
            for unseen_name in unseen:
                unseen_mask = families[unseen_test] == unseen_name
                metrics = ood_metrics(
                    owned_energy,
                    unseen_output["energy"][unseen_mask],
                    threshold,
                )
                ood_rows.append({
                    "expert": group, "stage": stage,
                    "active_owned_classes": ",".join(active_owned),
                    "configured_owned_classes": ",".join(configured_owned),
                    "ood_role": f"unseen::{unseen_name}",
                    "id_support": len(owned_energy),
                    "ood_support": int(unseen_mask.sum()),
                    "threshold": threshold, **metrics,
                })
            stage_outputs[stage] = {
                "threshold": threshold,
                "known_energy": known_output["energy"],
                "known_local_prediction": known_output["prediction"],
                "unseen_energy": unseen_output["energy"],
                "unseen_local_prediction": unseen_output["prediction"],
                "benign_energy": benign_output["energy"],
                "benign_local_prediction": benign_output["prediction"],
                "target_known_ood_validation_energy": (
                    target_ood_val_output["energy"]
                ),
                "external_aux_validation_energy": (
                    external_aux_output["energy"]
                ),
                "owned_mask": owned_mask,
                "owned": known_output["energy"][owned_mask],
                "non_owned": known_output["energy"][~owned_mask],
                "unseen": unseen_output["energy"],
                "target_known_ood_val": target_ood_val_output["energy"],
                "external_aux": external_aux_output["energy"],
                "benign_diag": benign_output["energy"],
            }

        final_output = stage_outputs["energy_finetuned"]
        accepted = final_output["known_energy"] <= final_output["threshold"]
        mapped_prediction = np.full(len(known_test), -1, dtype=np.int64)
        local_to_global = np.asarray(
            [class_to_known_id[name] for name in active_owned],
            dtype=np.int64,
        )
        mapped_prediction[accepted] = local_to_global[
            final_output["known_local_prediction"][accepted]
        ]
        expert_results[group] = {
            "active_owned": active_owned,
            "configured_owned": configured_owned,
            "accepted": accepted,
            "mapped_prediction": mapped_prediction,
            "local_to_global": local_to_global,
            "unseen_accepted": (
                final_output["unseen_energy"] <= final_output["threshold"]
            ),
            "benign_accepted": (
                final_output["benign_energy"] <= final_output["threshold"]
            ),
            "singleton": len(active_owned) == 1,
            "stage_outputs": stage_outputs,
        }

    aux_audit_frame = pd.DataFrame(all_aux_audit)
    aux_audit_frame.to_csv(
        os.path.join(out_dir, "1b_expert_aux_composition.csv"), index=False
    )
    save_aux_composition_png(
        aux_audit_frame,
        os.path.join(out_dir, "1b_expert_aux_composition.png"),
    )
    history_frame = pd.concat(all_histories, ignore_index=True)
    history_frame.to_csv(
        os.path.join(out_dir, "2a_expert_finetune_history.csv"), index=False
    )
    render_table_png(
        history_frame,
        os.path.join(out_dir, "2a_expert_finetune_history.png"),
        title="Energy fine-tuning history for independent experts",
        high_good=("val_macro_f1", "val_accuracy"),
        low_good=("loss", "ce_loss", "energy_margin_loss"),
    )

    ood_frame = pd.DataFrame(ood_rows)
    ood_frame.to_csv(
        os.path.join(out_dir, "3a_expert_ood_all_stages.csv"), index=False
    )
    metric_columns = [
        "expert", "active_owned_classes", "ood_role",
        "auroc", "aupr_id", "fpr95", "id_retain_at_tau",
        "ood_detect_at_tau",
    ]
    before = ood_frame[
        ood_frame["stage"] == "pretrained"
    ][metric_columns].rename(columns={
        "auroc": "pre_auroc", "aupr_id": "pre_aupr_id",
        "fpr95": "pre_fpr95", "id_retain_at_tau": "pre_id_retain",
        "ood_detect_at_tau": "pre_ood_detect",
    })
    after = ood_frame[
        ood_frame["stage"] == "energy_finetuned"
    ][metric_columns].rename(columns={
        "auroc": "post_auroc", "aupr_id": "post_aupr_id",
        "fpr95": "post_fpr95", "id_retain_at_tau": "post_id_retain",
        "ood_detect_at_tau": "post_ood_detect",
    })
    ood_comparison = before.merge(
        after,
        on=["expert", "active_owned_classes", "ood_role"],
        validate="one_to_one",
    )
    ood_comparison["auroc_delta"] = (
        ood_comparison["post_auroc"] - ood_comparison["pre_auroc"]
    )
    ood_comparison["fpr95_delta"] = (
        ood_comparison["post_fpr95"] - ood_comparison["pre_fpr95"]
    )
    ood_comparison["ood_detect_delta"] = (
        ood_comparison["post_ood_detect"]
        - ood_comparison["pre_ood_detect"]
    )
    ood_comparison.to_csv(
        os.path.join(out_dir, "3a_expert_ood_before_after.csv"),
        index=False,
    )
    render_table_png(
        ood_comparison,
        os.path.join(out_dir, "3a_expert_ood_before_after.png"),
        title=(
            "PRIMARY: expert OOD before vs after target-known + external-AUX "
            "Energy training"
        ),
        high_good=(
            "pre_auroc", "post_auroc", "post_aupr_id",
            "post_ood_detect", "auroc_delta", "ood_detect_delta",
        ),
        low_good=("pre_fpr95", "post_fpr95", "fpr95_delta"),
    )
    final_ood_frame = ood_frame[
        ood_frame["stage"] == "energy_finetuned"
    ].reset_index(drop=True)
    final_ood_frame.to_csv(
        os.path.join(out_dir, "3b_expert_ood_finetuned.csv"), index=False
    )
    render_table_png(
        final_ood_frame,
        os.path.join(out_dir, "3b_expert_ood_finetuned.png"),
        title=(
            "Independent expert Energy OOD after fine-tuning "
            "(lower FPR95, higher AUROC/detection)"
        ),
        high_good=("auroc", "aupr_id", "ood_detect_at_tau"),
        low_good=("fpr95",),
    )
    save_expert_histogram_grid(
        expert_results, "pretrained", ood_frame, args, out_dir
    )
    save_expert_histogram_grid(
        expert_results, "energy_finetuned", ood_frame, args, out_dir
    )

    comparison_rows = []
    for class_id, class_name in enumerate(known_classes):
        group = next(
            name for name, owned in active_experts.items()
            if class_name in owned
        )
        result = expert_results[group]
        expert_precision, expert_recall, expert_f1, support = (
            binary_class_metrics(
                y_known_test, result["mapped_prediction"], class_id
            )
        )
        true_class = y_known_test == class_id
        per_expert_in_rates = {
            other_group: float(
                other_result["accepted"][true_class].mean()
            )
            for other_group, other_result in expert_results.items()
        }
        other_in_rates = [
            rate for other_group, rate in per_expert_in_rates.items()
            if other_group != group
        ]
        row = {
            "class": class_name,
            "owner_expert": group,
            "is_tail": class_name in tail_classes,
            "singleton_expert": result["singleton"],
            "support": support,
            "xgb_precision": float(global_precision[class_id]),
            "xgb_recall": float(global_recall[class_id]),
            "xgb_f1": float(global_f1[class_id]),
            "expert_precision": expert_precision,
            "expert_recall": expert_recall,
            "expert_f1": expert_f1,
            "f1_delta": expert_f1 - float(global_f1[class_id]),
            "owner_expert_in_rate": per_expert_in_rates[group],
            "other_experts_out_rate": (
                float(np.mean([1.0 - rate for rate in other_in_rates]))
                if other_in_rates else float("nan")
            ),
        }
        for other_group, rate in per_expert_in_rates.items():
            row[f"in_rate__{safe_tag(other_group)}"] = rate
        comparison_rows.append(row)
    comparison_frame = pd.DataFrame(comparison_rows).sort_values(
        ["support", "class"], ascending=[False, True]
    ).reset_index(drop=True)
    metric_columns = [
        "xgb_precision", "xgb_recall", "xgb_f1",
        "expert_precision", "expert_recall", "expert_f1", "f1_delta",
    ]
    rate_columns = [
        column for column in comparison_frame.columns
        if column.startswith("in_rate__")
        or column in ("owner_expert_in_rate", "other_experts_out_rate")
    ]

    def aggregate_row(frame, label, weights=None):
        row = {
            "class": label, "owner_expert": "", "is_tail": "",
            "singleton_expert": "", "support": int(frame["support"].sum()),
        }
        for column in metric_columns:
            row[column] = float(np.average(
                frame[column].to_numpy(dtype=np.float64), weights=weights,
            ))
        for column in rate_columns:
            row[column] = float("nan")
        return row

    tail_frame = comparison_frame[comparison_frame["is_tail"] == True]  # noqa: E712
    aggregate_f1 = {
        "xgb_macro_f1": float(comparison_frame["xgb_f1"].mean()),
        "expert_macro_f1": float(comparison_frame["expert_f1"].mean()),
        "xgb_tail_macro_f1": (
            float(tail_frame["xgb_f1"].mean()) if len(tail_frame) else None
        ),
        "expert_tail_macro_f1": (
            float(tail_frame["expert_f1"].mean()) if len(tail_frame) else None
        ),
    }
    footer_rows = [aggregate_row(comparison_frame, "macro_avg")]
    if len(tail_frame):
        footer_rows.append(aggregate_row(tail_frame, "tail_macro_avg"))
    footer_rows.append(aggregate_row(
        comparison_frame, "weighted_avg",
        weights=comparison_frame["support"].to_numpy(dtype=np.float64),
    ))
    comparison_frame = pd.concat(
        [comparison_frame, pd.DataFrame(footer_rows)], ignore_index=True
    )
    comparison_rate_labels = {
        "owner_expert_in_rate": "owner_expert_in_rate (↑)",
        "other_experts_out_rate": "other_experts_out_rate (↑)",
    }
    for group in expert_results:
        column = f"in_rate__{safe_tag(group)}"
        comparison_rate_labels[column] = (
            f"{column} (owner↑/non-owner↓)"
        )
    comparison_frame = comparison_frame.rename(
        columns=comparison_rate_labels
    )
    comparison_frame.to_csv(
        os.path.join(out_dir, "4a_xgb_vs_gated_expert_class_prf.csv"),
        index=False,
    )
    render_table_png(
        comparison_frame,
        os.path.join(out_dir, "4a_xgb_vs_gated_expert_class_prf.png"),
        title=(
            "SECONDARY: Global XGBoost vs independent Energy-gated expert "
            "(no final routing)"
        ),
        high_good=(
            "xgb_precision", "xgb_recall", "xgb_f1",
            "expert_precision", "expert_recall", "expert_f1", "f1_delta",
            "owner_expert_in_rate (↑)", "other_experts_out_rate (↑)",
        ),
    )

    confusion_frame = accepted_confusion_rows(
        expert_results, known_classes, y_known_test, families,
        unseen_test, unseen,
    )
    confusion_frame.to_csv(
        os.path.join(out_dir, "4c_expert_accepted_confusion.csv"), index=False
    )
    save_accepted_confusion_png(
        confusion_frame, expert_results, known_classes, unseen,
        bool(len(benign_diag)),
        os.path.join(out_dir, "4c_expert_accepted_confusion.png"),
    )

    funnel_rows = []
    for class_id, class_name in enumerate(known_classes):
        group = next(
            name for name, owned in active_experts.items()
            if class_name in owned
        )
        result = expert_results[group]
        final = result["stage_outputs"]["energy_finetuned"]
        true_mask = y_known_test == class_id
        accepted = result["accepted"][true_mask]
        oracle_prediction = result["local_to_global"][
            final["known_local_prediction"][true_mask]
        ]
        correct = oracle_prediction == class_id
        recall_loss_gate = float((correct & ~accepted).mean())
        recall_loss_local = float((~correct).mean())
        owner_fp = confusion_frame[
            (confusion_frame["expert"] == group)
            & (confusion_frame["predicted_class"] == class_name)
            & (confusion_frame["true_class"] != class_name)
        ]
        known_fp = owner_fp[
            owner_fp["true_role"] == "target known"
        ].sort_values("count", ascending=False)
        funnel_rows.append({
            "class": class_name,
            "owner_expert": group,
            "is_tail": class_name in tail_classes,
            "support": int(true_mask.sum()),
            "xgb_recall": float(global_recall[class_id]),
            "oracle_local_recall": float(correct.mean()),
            "gated_recall": float((correct & accepted).mean()),
            "recall_lost_to_gate": recall_loss_gate,
            "wrong_accepted": float((~correct & accepted).mean()),
            "wrong_rejected": float((~correct & ~accepted).mean()),
            "dominant_recall_loss": (
                "none"
                if max(recall_loss_gate, recall_loss_local) <= 0.001
                else (
                    "gate" if recall_loss_gate >= recall_loss_local
                    else "local_classifier"
                )
            ),
            "fp_from_known_count": int(known_fp["count"].sum()),
            "top_fp_true_class": (
                known_fp["true_class"].iloc[0] if len(known_fp) else ""
            ),
            "top_fp_count": (
                int(known_fp["count"].iloc[0]) if len(known_fp) else 0
            ),
            "fp_from_unseen_count": int(
                owner_fp[owner_fp["true_role"] == "target unseen"]["count"]
                .sum()
            ),
            "fp_from_benign_count": int(
                owner_fp[owner_fp["true_role"] == "benign diagnostic"]["count"]
                .sum()
            ),
        })
    funnel_frame = pd.DataFrame(funnel_rows).sort_values(
        ["support", "class"], ascending=[False, True]
    ).reset_index(drop=True)
    funnel_frame.to_csv(
        os.path.join(out_dir, "4b_class_recall_funnel.csv"), index=False
    )
    render_table_png(
        funnel_frame,
        os.path.join(out_dir, "4b_class_recall_funnel.png"),
        title=(
            "ATTRIBUTION: per-class recall funnel — gated_recall + "
            "recall_lost_to_gate + wrong_accepted + wrong_rejected = 1; "
            "oracle_local_recall = local classifier ceiling with a perfect "
            "gate; FP columns attribute precision loss"
        ),
        high_good=("xgb_recall", "oracle_local_recall", "gated_recall"),
        low_good=(
            "recall_lost_to_gate", "wrong_accepted", "wrong_rejected",
        ),
    )

    expert_summary_rows = []
    for group, result in expert_results.items():
        output = result["stage_outputs"]["energy_finetuned"]
        threshold = output["threshold"]
        owned_mask = output["owned_mask"]
        expert_summary_rows.append({
            "expert": group,
            "active_owned_classes": ",".join(result["active_owned"]),
            "configured_owned_classes": ",".join(result["configured_owned"]),
            "singleton_expert": result["singleton"],
            "owned_attack_in_rate": float(
                result["accepted"][owned_mask].mean()
            ),
            "non_owned_attack_false_in_rate": (
                float(result["accepted"][~owned_mask].mean())
                if np.any(~owned_mask) else float("nan")
            ),
            "target_non_owned_validation_false_in_rate": float(
                np.mean(
                    output["target_known_ood_validation_energy"] <= threshold
                )
                if len(output["target_known_ood_validation_energy"])
                else float("nan")
            ),
            "unseen_false_in_rate": float(
                (output["unseen_energy"] <= threshold).mean()
            ),
            "external_aux_validation_false_in_rate": float(
                (output["external_aux_validation_energy"] <= threshold).mean()
                if len(output["external_aux_validation_energy"])
                else float("nan")
            ),
            "benign_false_in_rate": float(
                (output["benign_energy"] <= threshold).mean()
                if len(output["benign_energy"]) else float("nan")
            ),
        })
    expert_summary = pd.DataFrame(expert_summary_rows)
    expert_summary = expert_summary.rename(columns={
        "owned_attack_in_rate": "owned_attack_in_rate (↑)",
        "non_owned_attack_false_in_rate": (
            "non_owned_attack_false_in_rate (↓)"
        ),
        "target_non_owned_validation_false_in_rate": (
            "target_non_owned_validation_false_in_rate (↓)"
        ),
        "unseen_false_in_rate": "unseen_false_in_rate (↓)",
        "external_aux_validation_false_in_rate": (
            "external_aux_validation_false_in_rate (↓)"
        ),
        "benign_false_in_rate": "benign_false_in_rate (diagnostic)",
    })
    expert_summary.to_csv(
        os.path.join(out_dir, "5a_expert_decision_summary.csv"), index=False
    )
    render_table_png(
        expert_summary,
        os.path.join(out_dir, "5a_expert_decision_summary.png"),
        title=(
            "Independent attack-expert IN/OUT rates; no multi-IN/all-OUT counts"
        ),
        high_good=("owned_attack_in_rate (↑)",),
        low_good=(
            "non_owned_attack_false_in_rate (↓)",
            "target_non_owned_validation_false_in_rate (↓)",
            "unseen_false_in_rate (↓)",
            "external_aux_validation_false_in_rate (↓)",
        ),
    )

    expert_order = list(expert_results)
    accept_known = np.stack(
        [expert_results[name]["accepted"] for name in expert_order]
    )
    accept_unseen = np.stack(
        [expert_results[name]["unseen_accepted"] for name in expert_order]
    )
    accept_benign = (
        np.stack(
            [expert_results[name]["benign_accepted"] for name in expert_order]
        )
        if len(benign_diag) else np.zeros((len(expert_order), 0), dtype=bool)
    )

    def overlap_row(true_name, role, owner, accepted_matrix):
        in_counts = accepted_matrix.sum(axis=0)
        row = {
            "true_class": true_name, "role": role,
            "owner_expert": owner if owner else "",
            "support": int(accepted_matrix.shape[1]),
            "all_out_rate": float((in_counts == 0).mean()),
            "multi_in_rate": float((in_counts >= 2).mean()),
            "mean_experts_in": float(in_counts.mean()),
        }
        if owner:
            owner_accepted = accepted_matrix[expert_order.index(owner)]
            row["owner_only_in_rate"] = float(
                ((in_counts == 1) & owner_accepted).mean()
            )
            row["single_non_owner_in_rate"] = float(
                ((in_counts == 1) & ~owner_accepted).mean()
            )
        else:
            row["owner_only_in_rate"] = float("nan")
            row["single_non_owner_in_rate"] = float((in_counts == 1).mean())
        return row

    overlap_rows = []
    for class_id, class_name in enumerate(known_classes):
        owner = next(
            name for name, owned in active_experts.items()
            if class_name in owned
        )
        overlap_rows.append(overlap_row(
            class_name, "target known", owner,
            accept_known[:, y_known_test == class_id],
        ))
    for unseen_name in unseen:
        overlap_rows.append(overlap_row(
            f"unseen::{unseen_name}", "target unseen", None,
            accept_unseen[:, families[unseen_test] == unseen_name],
        ))
    if len(benign_diag):
        overlap_rows.append(overlap_row(
            "benign_diagnostic", "benign diagnostic", None, accept_benign,
        ))
    overlap_frame = pd.DataFrame(overlap_rows)[[
        "true_class", "role", "owner_expert", "support",
        "all_out_rate", "owner_only_in_rate", "single_non_owner_in_rate",
        "multi_in_rate", "mean_experts_in",
    ]]
    overlap_frame.to_csv(
        os.path.join(out_dir, "5b_class_gate_overlap.csv"), index=False
    )
    render_table_png(
        overlap_frame,
        os.path.join(out_dir, "5b_class_gate_overlap.png"),
        title=(
            "Gate overlap per true class (counted, never resolved): known "
            "classes want owner_only_in high; unseen/benign want all_out "
            "high; multi-IN shows how far 4a is from a resolvable pipeline"
        ),
    )

    energy_known = np.stack([
        expert_results[name]["stage_outputs"]["energy_finetuned"][
            "known_energy"
        ]
        for name in expert_order
    ])
    threshold_by_expert = np.array([
        expert_results[name]["stage_outputs"]["energy_finetuned"][
            "threshold"
        ]
        for name in expert_order
    ])
    resolution_frame = multi_in_energy_rows(
        expert_order, energy_known, accept_known, threshold_by_expert,
        known_classes, y_known_test, active_experts,
    ).sort_values(
        "false_accept_count", ascending=False
    ).reset_index(drop=True)
    resolution_frame.to_csv(
        os.path.join(out_dir, "5c_multi_in_energy_resolution.csv"),
        index=False,
    )
    if resolution_frame.empty:
        log.info(
            "5c: no non-owner false accepts; writing an empty schema-only "
            "table"
        )
    resolution_png_frame = (
        resolution_frame if not resolution_frame.empty else pd.DataFrame([{
            "status": "No non-owner false accepts on known test classes"
        }])
    )
    render_table_png(
        resolution_png_frame,
        os.path.join(out_dir, "5c_multi_in_energy_resolution.png"),
        title=(
            "No non-owner false accepts on known test classes"
            if resolution_frame.empty else
            "Per non-owner false-accept on a known class: owner_also_in_rate "
            "= true multi-IN vs owner gate already rejecting it; "
            "resolvable_by_argmin_rate = owner IN and owner's energy already "
            "lower (fixable by argmin-over-accepting-experts resolution); "
            "low rate = genuine feature-space collision, not a threshold miss"
        ),
        high_good=("owner_also_in_rate", "resolvable_by_argmin_rate"),
    )

    with open(
        os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "title": (
                "Independent attack-expert Energy OOD and gated classification"
            ),
            "args": vars(args),
            "target": args.target,
            "unseen_classes": unseen,
            "used_default_unseen": used_default_unseen,
            "known_classes": known_classes,
            "tail_classes": tail_classes,
            "tail_removed_by_custom_unseen": tail_removed,
            "experts_configured": experts,
            "experts_active": active_experts,
            "aux_policy": (
                "OOD train = target-train non-owned known attacks plus "
                "external attack AUX; external semantic matches to the "
                "expert's full configured ownership are excluded; benign and "
                "target unseen are excluded from all OOD training"
            ),
            "evaluation_scope": (
                "attack experts only; target-test non-owned known attacks and "
                "target-test unseen are separate OOD roles; benign is excluded; "
                "per-expert OOD and per-class gated classification only; "
                "multi-IN/all-OUT final resolution is intentionally absent"
            ),
            "singleton_policy": (
                "two-logit Energy head; IN maps deterministically to the "
                "single active owned class"
            ),
            "global_xgb_known_attack_accuracy": float(
                accuracy_score(y_known_test, global_prediction)
            ),
            "global_xgb_known_attack_macro_f1": float(f1_score(
                y_known_test, global_prediction,
                labels=np.arange(len(known_classes)),
                average="macro", zero_division=0,
            )),
            "aggregate_f1_xgb_vs_gated_expert": aggregate_f1,
            "ood_training_ablation": {
                "external_aux_disabled": bool(args.disable_external_aux),
                "target_known_ood_disabled": bool(args.disable_target_ood),
            },
            "benign_diagnostic_rows": int(len(benign_diag)),
        }, handle, indent=2)
    log.info("Results: %s", out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()

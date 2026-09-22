#!/usr/bin/env python3
"""Read-only cross-run summary of s44_nfv3_resolved_expert_pipeline sweeps.

s44 answers one config (one target dataset, one unseen-class choice, one aux
composition) per run. The open question from the 2026-08-04 lab discussion
was whether holding out a "middle" class as unseen trades off *against* the
known/other-expert classes the same way regardless of which class is chosen,
or whether the winner (better unseen recall vs. better known-class F1) flips
depending on which class is picked and which OOD-training source (target
non-owned known attacks only, vs. + external AUX) is used. Eyeballing that
across a dozen separate experiment.log files is exactly the kind of thing
that should be one table instead.

This script never trains anything and never touches s44 itself (frozen per
CLAUDE.md M2) -- it only globs finished results/*_resolved_expert_pipeline_*
run directories, parses each run's "Args:" log line and its
4a_xgb_vs_gated_expert_class_prf.csv / "Resolved accuracy:" log line, and
pivots target+external vs. target-only (and external-only, where a run has
it) side by side per unseen-class choice.

Usage:
  python scripts/s44_unseen_tradeoff_summary.py
  python scripts/s44_unseen_tradeoff_summary.py --results-root results --out-root results/scripts_analysis
"""

import argparse
import ast
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import s44_nfv3_resolved_expert_pipeline as s44  # noqa: E402
from exp_utils import render_table_png  # noqa: E402

# ood_role names in 3b_expert_ood_finetuned.csv -> the role key + the color
# s44's own 3c histogram already uses for that role's line (non_owned=#ff7f00,
# unseen=#e41a1c, benign_diag=#636363; see save_expert_histogram_grid roles=).
FPR95_ROLES = {
    "non_owned_known_attack": ("non_owned", "#ff7f00"),
    "unseen": ("unseen", "#e41a1c"),
    "benign_diagnostic": ("benign", "#636363"),
}

ARGS_RE = re.compile(r"INFO\s+Args:\s+(\{.*\})\s*$", re.MULTILINE)
RESOLVED_RE = re.compile(
    r"Resolved accuracy:\s*known=([\d.]+)\s+unseen=([\d.]+)\s+benign=([\d.]+)"
)


def variant_of(run_args):
    disable_ext = bool(run_args.get("disable_external_aux"))
    disable_tgt = bool(run_args.get("disable_target_ood"))
    if disable_ext and disable_tgt:
        return None  # invalid combination, s44 itself refuses this
    if disable_ext:
        return "target_only"
    if disable_tgt:
        return "external_only"
    return "target_external"


def parse_run(run_dir):
    log_path = os.path.join(run_dir, "experiment.log")
    prf_path = os.path.join(run_dir, "4a_xgb_vs_gated_expert_class_prf.csv")
    if not (os.path.isfile(log_path) and os.path.isfile(prf_path)):
        return None
    with open(log_path, "r", encoding="utf-8") as handle:
        log_text = handle.read()
    args_match = ARGS_RE.search(log_text)
    resolved_match = RESOLVED_RE.findall(log_text)
    if not args_match or not resolved_match:
        return None
    run_args = ast.literal_eval(args_match.group(1))
    variant = variant_of(run_args)
    if variant is None:
        return None
    known_acc, unseen_acc, benign_acc = (float(v) for v in resolved_match[-1])

    prf = pd.read_csv(prf_path)
    macro_row = prf[prf["class"] == "macro_avg"]
    tail_row = prf[prf["class"] == "tail_macro_avg"]
    if macro_row.empty:
        return None
    known_f1_expert = float(macro_row["expert_f1"].iloc[0])
    known_f1_xgb = float(macro_row["xgb_f1"].iloc[0])
    tail_f1_expert = (
        float(tail_row["expert_f1"].iloc[0]) if not tail_row.empty else float("nan")
    )

    unseen = tuple(sorted(str(name) for name in run_args["unseen"]))
    target = run_args["target"]

    return {
        "target": target,
        "unseen": unseen,
        "variant": variant,
        "out_dir": os.path.basename(run_dir),
        "known_macro_f1_expert": known_f1_expert,
        "known_macro_f1_xgb": known_f1_xgb,
        "tail_macro_f1_expert": tail_f1_expert,
        "known_resolved_acc": known_acc,
        "unseen_resolved_acc": unseen_acc,
        "benign_resolved_acc": benign_acc,
    }


def load_fpr95_by_expert(run_dir):
    """Per-expert {non_owned, unseen, benign} FPR95 straight from 3b -- the
    3c histogram title only shows pooled_attack_out (non_owned+unseen
    combined), which conflates the two roles this is meant to separate."""
    path = os.path.join(run_dir, "3b_expert_ood_finetuned.csv")
    if not os.path.isfile(path):
        return {}
    frame = pd.read_csv(path)
    result = {}
    for expert, group in frame.groupby("expert"):
        per_role = {}
        for ood_role, (key, _color) in FPR95_ROLES.items():
            match = group[group["ood_role"] == ood_role]
            if not match.empty:
                per_role[key] = float(match["fpr95"].iloc[0])
        result[str(expert)] = per_role
    return result


# Anchor for each of the up to 4 expert subplots inside an embedded 3c image
# (plt.subplots(2, 2).ravel() order: top-left, top-right, bottom-left,
# bottom-right). Anchored to each quadrant's own top/bottom-RIGHT corner,
# opposite the legend box s44 always draws at each subplot's top-left.
QUADRANT_ANCHORS = [
    {"x": 0.46, "y": 0.93, "ha": "right", "va": "top"},
    {"x": 0.97, "y": 0.93, "ha": "right", "va": "top"},
    {"x": 0.46, "y": 0.45, "ha": "right", "va": "top"},
    {"x": 0.97, "y": 0.45, "ha": "right", "va": "top"},
]


def build_fpr95_grid(cases, results_root, out_path, cols=5):
    rows = -(-len(cases) // cols)  # ceil
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.8, rows * 4.1), squeeze=False,
    )
    axes = axes.ravel()
    role_order = ["non_owned", "unseen", "benign"]
    role_color = {key: color for _role, (key, color) in FPR95_ROLES.items()}
    for axis, case in zip(axes, cases):
        image_path = os.path.join(
            results_root, case["out_dir"], "3c_energy_hist_2x2_finetuned.png",
        )
        if os.path.isfile(image_path):
            axis.imshow(mpimg.imread(image_path))
        axis.axis("off")
        axis.set_title(
            f"{case['target']} | unseen={case['unseen']}", fontsize=9,
        )
        for expert, anchor in zip(case["expert_order"], QUADRANT_ANCHORS):
            per_role = case["fpr95_by_expert"].get(expert, {})
            for offset, role in enumerate(role_order):
                value = per_role.get(role)
                if value is None:
                    continue
                axis.text(
                    anchor["x"], anchor["y"] - offset * 0.075, f"{value:.3f}",
                    transform=axis.transAxes, color=role_color[role],
                    fontsize=11, fontweight="bold",
                    ha=anchor["ha"], va=anchor["va"],
                    bbox=dict(
                        facecolor="white", alpha=0.75, edgecolor="none", pad=1.0,
                    ),
                )
    for axis in axes[len(cases):]:
        axis.axis("off")
    fig.suptitle(
        "s44 sweep: per-expert FPR95, placed at each expert's own histogram "
        "(target+external variant) -- orange=non-owned known attack, "
        "red=target unseen, gray=target benign",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_fpr95_long_table(cases):
    rows = []
    for case in cases:
        for expert, per_role in case["fpr95_by_expert"].items():
            rows.append({
                "target": case["target"], "unseen": case["unseen"],
                "expert": expert,
                "non_owned_known_attack_fpr95": per_role.get("non_owned"),
                "unseen_fpr95": per_role.get("unseen"),
                "benign_fpr95": per_role.get("benign"),
            })
    return pd.DataFrame(rows)


def active_expert_order(expert_config_path, target, unseen):
    """Same active_experts filtering/order s44's main() uses when building
    expert_results, which is the order save_expert_histogram_grid fills the
    2x2 subplot grid in (top-left, top-right, bottom-left, bottom-right)."""
    experts = s44.load_expert_groups(expert_config_path, target)
    unseen_set = set(unseen)
    return [
        group for group, owned in experts.items()
        if any(name not in unseen_set for name in owned)
    ]


def owner_expert_of(expert_config_path, target, unseen):
    experts = s44.load_expert_groups(expert_config_path, target)
    unseen_set = set(unseen)
    for group, owned in experts.items():
        if unseen_set <= set(owned):
            return group, unseen_set == set(owned)
    return "unknown", False


def build_long_table(records, expert_config_path):
    rows = []
    for record in records:
        owner, fully_removed = owner_expert_of(
            expert_config_path, record["target"], record["unseen"]
        )
        rows.append({
            "target": record["target"],
            "unseen": "+".join(record["unseen"]),
            "owner_expert": owner,
            "owner_expert_fully_removed": fully_removed,
            "aux_variant": record["variant"],
            "known_macro_f1_expert": record["known_macro_f1_expert"],
            "known_macro_f1_xgb_baseline": record["known_macro_f1_xgb"],
            "tail_macro_f1_expert": record["tail_macro_f1_expert"],
            "known_resolved_acc": record["known_resolved_acc"],
            "unseen_resolved_acc": record["unseen_resolved_acc"],
            "benign_resolved_acc": record["benign_resolved_acc"],
            "out_dir": record["out_dir"],
        })
    variant_order = {"target_external": 0, "target_only": 1, "external_only": 2}
    frame = pd.DataFrame(rows)
    frame["_variant_order"] = frame["aux_variant"].map(variant_order)
    frame = frame.sort_values(
        ["target", "unseen", "_variant_order"]
    ).drop(columns="_variant_order").reset_index(drop=True)
    return frame


PIVOT_COLUMNS = [
    "target", "unseen", "owner_expert", "owner_expert_fully_removed",
    "known_f1__target_external", "known_f1__target_only",
    "unseen_acc__target_external", "unseen_acc__target_only",
    "delta_known_f1__target_only_vs_external",
    "delta_unseen_acc__target_only_vs_external",
]


def build_pivot_table(long_frame):
    pivot_rows = []
    group_cols = ["target", "unseen", "owner_expert", "owner_expert_fully_removed"]
    for key, group in long_frame.groupby(group_cols):
        target, unseen, owner, fully_removed = key
        by_variant = group.set_index("aux_variant")
        row = {
            "target": target, "unseen": unseen, "owner_expert": owner,
            "owner_expert_fully_removed": fully_removed,
        }
        for variant in ("target_external", "target_only"):
            if variant in by_variant.index:
                row[f"known_f1__{variant}"] = by_variant.loc[
                    variant, "known_macro_f1_expert"
                ]
                row[f"unseen_acc__{variant}"] = by_variant.loc[
                    variant, "unseen_resolved_acc"
                ]
            else:
                row[f"known_f1__{variant}"] = float("nan")
                row[f"unseen_acc__{variant}"] = float("nan")
        row["delta_known_f1__target_only_vs_external"] = (
            row["known_f1__target_only"] - row["known_f1__target_external"]
        )
        row["delta_unseen_acc__target_only_vs_external"] = (
            row["unseen_acc__target_only"] - row["unseen_acc__target_external"]
        )
        pivot_rows.append(row)
    frame = pd.DataFrame(pivot_rows, columns=PIVOT_COLUMNS).sort_values(
        ["target", "unseen"]
    ).reset_index(drop=True)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument(
        "--out-root", default=os.path.join(REPO_ROOT, "results", "scripts_analysis")
    )
    parser.add_argument(
        "--expert-config",
        default=os.path.join(REPO_ROOT, "configs", "nfv3_experts.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.results_root, "*_resolved_expert_pipeline_unseen_*")
    candidate_dirs = sorted(glob.glob(pattern))

    parsed = {}
    skipped = []
    for run_dir in candidate_dirs:
        record = parse_run(run_dir)
        if record is None:
            skipped.append(os.path.basename(run_dir))
            continue
        key = (record["target"], record["unseen"], record["variant"])
        # Runs are timestamp-prefixed dir names -> lexicographic max = latest.
        if key not in parsed or run_dir > os.path.join(
            args.results_root, parsed[key]["out_dir"]
        ):
            parsed[key] = record

    if not parsed:
        raise SystemExit(f"No complete s44 runs found under {args.results_root}")

    long_frame = build_long_table(list(parsed.values()), args.expert_config)
    pivot_frame = build_pivot_table(long_frame)

    os.makedirs(args.out_root, exist_ok=True)
    long_csv = os.path.join(args.out_root, "s44_unseen_tradeoff_long.csv")
    long_png = os.path.join(args.out_root, "s44_unseen_tradeoff_long.png")
    pivot_csv = os.path.join(args.out_root, "s44_unseen_tradeoff_pivot.csv")
    pivot_png = os.path.join(args.out_root, "s44_unseen_tradeoff_pivot.png")

    long_frame.to_csv(long_csv, index=False)
    render_table_png(
        long_frame, long_png,
        title="s44 sweep: every run's resolved (known, unseen, benign) accuracy "
              "and known/tail expert macro-F1",
        delta_cols=(),
    )
    pivot_frame.to_csv(pivot_csv, index=False)
    render_table_png(
        pivot_frame, pivot_png,
        title="s44 unseen-class tradeoff: target-only vs. target+external AUX "
              "(blue=target-only better, red=worse)",
        delta_cols=(
            "delta_known_f1__target_only_vs_external",
            "delta_unseen_acc__target_only_vs_external",
        ),
    )

    cases = []
    for record in parsed.values():
        if record["variant"] != "target_external":
            continue
        run_dir = os.path.join(args.results_root, record["out_dir"])
        cases.append({
            "target": record["target"],
            "unseen": "+".join(record["unseen"]),
            "out_dir": record["out_dir"],
            "fpr95_by_expert": load_fpr95_by_expert(run_dir),
            "expert_order": active_expert_order(
                args.expert_config, record["target"], record["unseen"],
            ),
        })
    cases.sort(key=lambda case: (case["target"], case["unseen"]))

    fpr95_grid_png = os.path.join(args.out_root, "s44_unseen_fpr95_grid.png")
    fpr95_csv = os.path.join(args.out_root, "s44_unseen_fpr95_by_expert.csv")
    if cases:
        build_fpr95_grid(cases, args.results_root, fpr95_grid_png)
        build_fpr95_long_table(cases).to_csv(fpr95_csv, index=False)
    else:
        print("No target_external-variant runs found; skipped the FPR95 grid.")

    print(f"Parsed {len(parsed)} runs ({len(skipped)} skipped/incomplete).")
    if skipped:
        print("Skipped:", ", ".join(skipped))
    print(f"\n{pivot_frame.to_string(index=False)}")
    print(f"\nWrote {long_csv}\nWrote {long_png}\nWrote {pivot_csv}\nWrote {pivot_png}")
    if cases:
        print(f"Wrote {fpr95_grid_png}\nWrote {fpr95_csv}")


if __name__ == "__main__":
    main()

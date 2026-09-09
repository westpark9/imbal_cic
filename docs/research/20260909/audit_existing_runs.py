"""Read the frozen 100k/balanced/n=4 runs; write diagnostic CSVs, without fitting.

Run from any directory: python docs/research/20260909/audit_existing_runs.py
Positive gain below means weighted NLL gain, not a corrected classification.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN_NAMES = [
    "20260904_010654_nfv3_cic2018_exp31_c0alloc",
    "20260904_010911_nfv3_cic2018_exp31_c0alloc",
    "20260903_235255_nfv3_cic2018_exp31_c0alloc",
    "20260904_021924_nfv3_cic2018_exp31_c0alloc",
    "20260903_235413_nfv3_cic2018_exp31_c0alloc",
    "20260904_022152_nfv3_cic2018_exp31_c0alloc",
    "20260904_025507_nfv3_cic2018_exp31_c0alloc",
]


def main():
    summaries, labels, transitions, manifests = [], [], [], []
    for name in RUN_NAMES:
        config = ROOT / "tabpfn/results" / name / "args.json"
        args = json.loads(config.read_text())
        if not (
            args.get("target_dataset") == "cic2018"
            and args.get("global_context_size") == 100000
            and args.get("c0_attack_alloc") == "balanced"
            and args.get("c0_benign_share") == 0.75
            and args.get("n_estimators") == 4
            and args.get("c0_dedup") == "none"
            and args.get("c0_conflict") == "keep"
            and args.get("test_cap_per_class") == 0
            and args.get("seed") in range(42, 49)
        ):
            raise ValueError(f"Unexpected configuration for {name}")
        run = config.parent
        metrics = pd.read_csv(run / "per_class_metrics.csv")
        def f1(method, cls):
            return float(metrics.loc[(metrics.method == method) & (metrics["class"] == cls), "f1"].iloc[0])
        with np.load(run / "system_dump.npz") as dump:
            y = dump["y_true"]
            baseline, final = dump["y_glob"], dump["final"]
            accepted = dump["accepted"].astype(bool)
            names = dump["class_names"].astype(str)
            good0, good1 = baseline == y, final == y
            changed = baseline != final
            tau = float(dump["tau_pre"])
            for c, name in enumerate(names):
                mask = y == c
                transitions.append(dict(run=run.name, seed=args["seed"], cls=name,
                    rows=int(mask.sum()), accepted=int((mask & accepted).sum()),
                    prediction_changed=int((mask & changed).sum()),
                    helpful=int((mask & ~good0 & good1).sum()),
                    harmful=int((mask & good0 & ~good1).sum()),
                    wrong_to_wrong=int((mask & changed & ~good0 & ~good1).sum())))
            row = dict(run=run.name, seed=args["seed"], test_rows=len(y),
                       tau_pre=tau, accepted=int(accepted.sum()),
                       prediction_changed=int(changed.sum()),
                       accepted_same_label=int((accepted & ~changed).sum()),
                       helpful=int((~good0 & good1).sum()), harmful=int((good0 & ~good1).sum()))
        for cls in ["macro_avg", "tail_avg", "infiltration", "web_attacks"]:
            row["global_" + cls] = f1("global_tabpfn", cls)
            row["system_" + cls] = f1("racepfn_system", cls)
            row["delta_" + cls] = row["system_" + cls] - row["global_" + cls]
        split = pd.read_csv(run / "0c_split_manifest.csv")
        cal = split.loc[split["split"] == "D_cal"].copy()
        cal["capped"] = cal["rows"].clip(upper=args["cal_cap_per_class"])
        row["cal_rows_before_hash_mask"] = int(cal["capped"].sum())
        row["cal_benign_fraction_before_hash_mask"] = float(
            cal.loc[cal["class"] == "benign", "capped"].sum() / cal["capped"].sum())
        row["cal_rows_after_hash_mask"] = int(pd.read_csv(run / "4d_calibration.csv")["cal_rows_used"].iloc[0])
        with np.load(run / "route_gain.npz") as route:
            gain, passed, yroute = route["gain"], route["b_oof"].astype(bool), route["y"]
            positive = gain > 0
            row["positive_gain_pairs"] = int(positive.sum())
            row["positive_gain_survival_after_oof"] = float((positive & passed).sum() / max(positive.sum(), 1))
            for c, name in enumerate(names):
                mask = yroute == c
                pos = positive[mask]
                kept = pos & passed[mask]
                labels.append(dict(run=run.name, seed=args["seed"], cls=name,
                    route_rows=int(mask.sum()), experts=gain.shape[1],
                    positive_gain_pairs=int(pos.sum()), surviving_pairs=int(kept.sum()),
                    positive_gain_survival=float(kept.sum() / max(pos.sum(), 1))))
        summaries.append(row)
        manifests.append(dict(run=run.name, args=args))
    summary = pd.DataFrame(summaries).sort_values("seed")
    summary.to_csv(OUT / "latest_routing_audit.csv", index=False)
    pd.DataFrame(labels).to_csv(OUT / "scorer_target_survival_by_class.csv", index=False)
    pd.DataFrame(transitions).to_csv(OUT / "latest_transitions_by_class.csv", index=False)
    (OUT / "audited_run_args.json").write_text(json.dumps(manifests, indent=2, ensure_ascii=False) + "\n")
    print(summary[["seed", "global_macro_avg", "delta_macro_avg", "tau_pre", "accepted", "prediction_changed", "helpful", "harmful", "positive_gain_survival_after_oof"]].to_string(index=False))
    print("Mean global macro/tail:", summary[["global_macro_avg", "global_tail_avg"]].mean().to_dict())
    print("Mean system delta macro/tail:", summary[["delta_macro_avg", "delta_tail_avg"]].mean().to_dict())


if __name__ == "__main__":
    main()

"""Extract recorded XGB settings for the 0911 audit; no data/model loading."""
import ast
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FIELDS = ("max_depth", "n_estimators", "learning_rate", "subsample", "colsample_bytree")
EXPECTED = (8, 300, 0.05, 0.8, 0.8)
rows = []


def record(path, args, exp, scope, prefix="", reused=""):
    row = {
        "experiment": exp,
        "scope": scope,
        "run": path.parent.name,
        "source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": args["seed"],
        **{key: args[prefix + key] for key in FIELDS},
        "full_prediction_source": reused,
    }
    assert tuple(row[key] for key in FIELDS) == EXPECTED, row
    rows.append(row)


for exp, run in (
    (40, "20260911_160256_3299866_nfv3_cic2018_exp40_dedup_xgb"),
    (41, "20260911_165635_3341259_nfv3_cic2018_exp41_dedup_dataset_xgb"),
):
    path = ROOT / "results" / run / "experiment.log"
    line = next(line for line in path.read_text().splitlines() if "Args:" in line)
    args = ast.literal_eval(line.split("Args:", 1)[1].strip())
    record(path, args, exp, "full_train")

for exp in (40, 41):
    paths = sorted((ROOT / "tabpfn/results").glob(f"20260911_*exp{exp}_*/args.json"))
    for path in paths:
        args = json.loads(path.read_text())
        if not args.get("skip_xgboost", False):
            record(path, args, exp, "C0", "xgb_")

# The current 0911 HTML also incorporates EXP42/44, run on 0914/15.
for exp in (42, 44):
    for date in ("20260914", "20260915"):
        paths = sorted((ROOT / "tabpfn/results").glob(f"{date}_*exp{exp}_*/args.json"))
        for path in paths:
            args = json.loads(path.read_text())
            methods = set(args.get("methods", "").split(","))
            if args.get("test_cap_per_class") or not methods.intersection({"xgb_c0", "xgb_full"}):
                continue
            reused = args.get("xgb_full_run", "")
            if exp == 42 and "xgb_full" in methods:
                source = Path(reused)
                if not source.is_absolute():
                    source = ROOT / source
                full = next(r for r in rows if r["scope"] == "full_train" and r["run"] == source.name)
                assert full["max_depth"] == args["xgb_max_depth"]
                reused = str(source.relative_to(ROOT))
            scope = "C0; full reused" if exp == 42 else "C0 and full shared constructor"
            record(path, args, exp, scope, "xgb_", reused)

counts = {str(exp): sum(r["experiment"] == exp for r in rows) for exp in (40, 41, 42, 44)}
assert counts == {"40": 5, "41": 13, "42": 5, "44": 2}, counts
with (OUT / "0911_xgb_depth_audit.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "scope": "0911 EXP40/41 and EXP42/44 incorporated into the updated report",
    "record_count": len(rows),
    "records_by_experiment": counts,
    "matched_settings": dict(zip(FIELDS, EXPECTED)),
    "full_train_exp40_exp41_seeds": [42],
    "C0_exp41_seeds": [42, 43, 44],
    "evidence_limit": "Recorded arguments and constructor wiring; not fitted tree-depth inspection.",
    "records": rows,
}
(OUT / "0911_xgb_depth_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))

#!/usr/bin/env python3
"""Audit conflicting labels on the official CIC-IDS2018 CICFlowMeter CSV files.

The audit intentionally uses the 78 common CICFlowMeter statistical fields,
excluding ``Timestamp`` and ``Label``.  It does not reuse the project's
NF-v3-derived matrices, and it keeps the CSV value representation after
trimming outer whitespace.  A SHA-256 key is generated from every selected
field; representative groups are then re-read from the source CSV to expose
the actual values used in the slide evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


METADATA_COLUMNS = {"Timestamp", "Label", "Flow ID", "Src IP", "Src Port", "Dst IP"}
DISPLAY_COLUMNS = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Byts/s", "Flow Pkts/s", "Idle Min",
]

# The published NF-CICIDS2018-v3 benchmark reports these seven target
# families.  The audit keeps the original CSV values, but evaluates a
# contradiction at the same target granularity used by the experiments.
NFV3_FAMILY = {
    "Benign": "benign",
    "DDOS attack-HOIC": "ddos",
    "DDOS attack-LOIC-UDP": "ddos",
    "DDoS attacks-LOIC-HTTP": "ddos",
    "Bot": "bot",
    "DoS attacks-GoldenEye": "dos",
    "DoS attacks-Hulk": "dos",
    "DoS attacks-SlowHTTPTest": "dos",
    "DoS attacks-Slowloris": "dos",
    "FTP-BruteForce": "bf",
    "SSH-Bruteforce": "bf",
    "Infilteration": "inf",
    "Brute Force -Web": "web",
    "Brute Force -XSS": "web",
    "SQL Injection": "web",
}
NFV3_FAMILY_ORDER = ["benign", "ddos", "bot", "dos", "bf", "inf", "web"]


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def family_case() -> str:
    clauses = " ".join(
        f"WHEN trim(\"Label\") = {sql_quote(label)} THEN {sql_quote(family)}"
        for label, family in NFV3_FAMILY.items()
    )
    return f"CASE {clauses} ELSE NULL END"


def source_relation(files: list[Path], columns: list[str]) -> str:
    """Select only shared fields so the one wider original CSV is harmless."""
    select = ", ".join([*(identifier(column) for column in columns), "filename"])
    parts = []
    for path in files:
        options = "header=true, all_varchar=true, filename=true, strict_mode=false, null_padding=true"
        # DuckDB cannot parallel-read this malformed source file.  Other files
        # retain parallel parsing; no row is skipped in either branch.
        if path.name.startswith("3_Thuesday-20-02-2018"):
            options += ", parallel=false"
        parts.append(f"SELECT {select} FROM read_csv({sql_quote(str(path))}, {options})")
    return "(" + " UNION ALL ".join(parts) + ")"


def headers(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return [item.strip() for item in next(csv.reader(handle))]


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-db", type=Path, default=Path("/tmp/cic2018_original_csv_audit.duckdb"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--top-groups", type=int, default=12)
    args = parser.parse_args()

    files = sorted(args.csv_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files under {args.csv_dir}")
    common = set(headers(files[0]))
    for path in files[1:]:
        common &= set(headers(path))
    feature_columns = sorted(common - METADATA_COLUMNS)
    missing_display = set(DISPLAY_COLUMNS) - set(feature_columns)
    if missing_display:
        raise SystemExit(f"Expected display columns are missing: {sorted(missing_display)}")
    if len(feature_columns) != 78:
        raise SystemExit(f"Expected 78 shared statistical fields, got {len(feature_columns)}")

    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise SystemExit("Install duckdb first: python -m pip install --user duckdb") from exc

    args.work_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.work_db))
    con.execute(f"SET threads TO {int(args.threads)}")
    con.execute("SET preserve_insertion_order TO false")
    con.execute("DROP TABLE IF EXISTS flow_rows")
    con.execute("DROP TABLE IF EXISTS conflict_keys")
    con.execute("DROP TABLE IF EXISTS family_conflict_keys")

    fields = ", ".join(
        f"coalesce(trim({identifier(column)}), '<NULL>')" for column in feature_columns
    )
    relation = source_relation(files, [*feature_columns, "Label"])
    print(f"Scanning {len(files)} files with {len(feature_columns)} common feature fields", flush=True)
    con.execute(
        "CREATE TABLE flow_rows AS "
        f"SELECT trim(\"Label\") AS label, {family_case()} AS nfv3_family, "
        f"sha256(concat_ws(chr(31), {fields})) AS feature_sha256 "
        # A repeated header is embedded as a data row in part of the released
        # CSV.  ``Label`` is not an attack class and is excluded explicitly.
        f"FROM {relation} WHERE trim(\"Label\") <> '' AND trim(\"Label\") <> 'Label'"
    )
    print("Grouping exact feature keys", flush=True)
    con.execute(
        "CREATE TABLE conflict_keys AS "
        "SELECT feature_sha256, count(*) AS rows, count(DISTINCT label) AS labels "
        "FROM flow_rows GROUP BY feature_sha256 HAVING count(DISTINCT label) > 1"
    )
    con.execute(
        "CREATE TABLE family_conflict_keys AS "
        "SELECT feature_sha256, count(*) AS rows, count(DISTINCT nfv3_family) AS families "
        "FROM flow_rows GROUP BY feature_sha256 HAVING count(DISTINCT nfv3_family) > 1"
    )

    total_rows = con.execute("SELECT count(*) FROM flow_rows").fetchone()[0]
    group_count, conflict_rows = con.execute(
        "SELECT count(*), coalesce(sum(rows), 0) FROM conflict_keys"
    ).fetchone()
    label_totals = dict(con.execute(
        "SELECT label, count(*) FROM flow_rows GROUP BY label"
    ).fetchall())
    label_conflicts = dict(con.execute(
        "SELECT r.label, count(*) FROM flow_rows r "
        "JOIN conflict_keys c USING(feature_sha256) GROUP BY r.label"
    ).fetchall())
    per_label = []
    for label, rows in label_totals.items():
        conflict = int(label_conflicts.get(label, 0))
        per_label.append({
            "label": label,
            "rows": int(rows),
            "conflicting_rows": conflict,
            "conflicting_fraction": conflict / rows if rows else 0.0,
        })
    per_label.sort(key=lambda row: (-row["conflicting_fraction"], -row["conflicting_rows"], row["label"]))

    unknown_families = con.execute(
        "SELECT DISTINCT label FROM flow_rows WHERE nfv3_family IS NULL ORDER BY label"
    ).fetchall()
    if unknown_families:
        raise RuntimeError(f"Unmapped CIC2018 labels: {[row[0] for row in unknown_families]}")
    family_totals = dict(con.execute(
        "SELECT nfv3_family, count(*) FROM flow_rows GROUP BY nfv3_family"
    ).fetchall())
    family_conflicts = dict(con.execute(
        "SELECT r.nfv3_family, count(*) FROM flow_rows r "
        "JOIN family_conflict_keys c USING(feature_sha256) GROUP BY r.nfv3_family"
    ).fetchall())
    per_family = []
    for family in NFV3_FAMILY_ORDER:
        rows = int(family_totals.get(family, 0))
        conflict = int(family_conflicts.get(family, 0))
        per_family.append({
            "family": family,
            "rows": rows,
            "conflicting_rows": conflict,
            "conflicting_fraction": conflict / rows if rows else 0.0,
        })
    family_group_count, family_conflict_rows = con.execute(
        "SELECT count(*), coalesce(sum(rows), 0) FROM family_conflict_keys"
    ).fetchone()

    top_rows = con.execute(
        "SELECT c.feature_sha256, c.rows, c.labels, "
        "string_agg(DISTINCT r.label, ' | ' ORDER BY r.label) AS label_set "
        "FROM conflict_keys c JOIN flow_rows r USING(feature_sha256) "
        "GROUP BY c.feature_sha256, c.rows, c.labels ORDER BY c.rows DESC, c.feature_sha256 LIMIT ?",
        [args.top_groups],
    ).fetchall()

    examples = []
    display_select = ", ".join(identifier(column) for column in DISPLAY_COLUMNS)
    for group_index, (key, rows, labels, label_set) in enumerate(top_rows):
        records = con.execute(
            "SELECT label, count(*) AS rows FROM flow_rows WHERE feature_sha256=? "
            "GROUP BY label ORDER BY label", [key]
        ).fetchall()
        columns = ["filename", "label", *DISPLAY_COLUMNS]
        raw_samples = []
        # Only the largest group needs literal source rows for the deck.  A
        # window over every label and all ten CSVs for each top group is much
        # slower than this bounded representative lookup.
        if group_index == 0:
            for label, _ in records:
                sample_query = (
                    f"SELECT filename, trim(\"Label\") AS label, {display_select} FROM {relation} "
                    f"WHERE sha256(concat_ws(chr(31), {fields}))=? AND trim(\"Label\")=? LIMIT 1"
                )
                sample = con.execute(sample_query, [key, label]).fetchone()
                if sample is not None:
                    raw_samples.append(dict(zip(columns, sample)))
        examples.append({
            "feature_sha256": key,
            "rows": int(rows),
            "labels": int(labels),
            "label_set": label_set,
            "per_label_rows": [{"label": label, "rows": int(count)} for label, count in records],
            "raw_samples": raw_samples,
        })

    result = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "csv_dir": str(args.csv_dir),
            "files": [path.name for path in files],
            "features": feature_columns,
            "feature_count": len(feature_columns),
            "excluded_columns": sorted(METADATA_COLUMNS),
            "equality": "Every selected raw CSV field is outer-whitespace-trimmed and concatenated in a fixed column order before SHA-256 grouping.",
        },
        "totals": {
            "rows": int(total_rows),
            "conflicting_feature_groups": int(group_count),
            "rows_in_conflicting_groups": int(conflict_rows),
            "fraction_rows_in_conflicting_groups": conflict_rows / total_rows if total_rows else 0.0,
        },
        "nfv3_family_audit": {
            "families": NFV3_FAMILY_ORDER,
            "criterion": "The same 78-field vector occurs with more than one NF-v3 target family.",
            "conflicting_feature_groups": int(family_group_count),
            "rows_in_conflicting_groups": int(family_conflict_rows),
            "fraction_rows_in_conflicting_groups": family_conflict_rows / total_rows if total_rows else 0.0,
            "per_family": per_family,
        },
        "per_label": per_label,
        "representative_conflicting_groups": examples,
    }
    write_json(args.output, result)
    print(json.dumps(result["totals"], ensure_ascii=False), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

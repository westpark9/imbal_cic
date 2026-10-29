#!/usr/bin/env python3
"""Measure same-feature/different-label groups in the current NF-ToN-IoT-v3 input.

This is intentionally distinct from the CIC-IDS2018 original-CSV audit.  It
uses exactly the 46 input columns listed in the saved ToN-IoT manifest, and
therefore describes the project's current NF-v3 tabular input, not the
underlying ToN-IoT PCAP or all source CSV columns.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label-column', default='Attack')
    parser.add_argument('--work-db', type=Path, default=Path('/tmp/toniot_nfv3_conflict_audit.duckdb'))
    parser.add_argument('--threads', type=int, default=12)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    features = manifest['feature_names']
    label_column = args.label_column
    with args.csv.open(newline='', encoding='utf-8', errors='replace') as handle:
        header = set(next(csv.reader(handle)))
    missing = sorted(set([*features, label_column]) - header)
    if missing:
        raise SystemExit(f'Missing expected columns: {missing}')
    if len(features) != 46:
        raise SystemExit(f'Expected the saved 46 ToN-IoT model features, received {len(features)}')
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise SystemExit('Install duckdb first: python -m pip install --user duckdb') from exc

    args.work_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.work_db))
    con.execute(f'SET threads TO {args.threads}')
    con.execute('SET preserve_insertion_order TO false')
    con.execute('DROP TABLE IF EXISTS feature_rows')
    con.execute('DROP TABLE IF EXISTS conflict_keys')
    fields = ', '.join(f"coalesce(trim({identifier(column)}), '<NULL>')" for column in features)
    csv_sql = str(args.csv).replace("'", "''")
    source = f"read_csv('{csv_sql}', header=true, all_varchar=true)"
    print(f'Scanning {args.csv.name}: {len(features)} selected model fields', flush=True)
    con.execute(
        'CREATE TABLE feature_rows AS '
        f"SELECT trim({identifier(label_column)}) AS label, sha256(concat_ws(chr(31), {fields})) AS feature_sha256 "
        f"FROM {source} WHERE trim({identifier(label_column)}) <> ''"
    )
    print('Grouping exact selected-feature keys', flush=True)
    con.execute(
        'CREATE TABLE conflict_keys AS '
        'SELECT feature_sha256, count(*) AS rows, count(DISTINCT label) AS labels '
        'FROM feature_rows GROUP BY feature_sha256 HAVING count(DISTINCT label) > 1'
    )
    rows = con.execute('SELECT count(*) FROM feature_rows').fetchone()[0]
    groups, conflicting_rows = con.execute('SELECT count(*), coalesce(sum(rows), 0) FROM conflict_keys').fetchone()
    totals = dict(con.execute('SELECT label, count(*) FROM feature_rows GROUP BY label').fetchall())
    conflicts = dict(con.execute(
        'SELECT r.label, count(*) FROM feature_rows r JOIN conflict_keys c USING(feature_sha256) GROUP BY r.label'
    ).fetchall())
    per_label = [
        dict(label=label, rows=int(count), conflicting_rows=int(conflicts.get(label, 0)),
             conflicting_fraction=conflicts.get(label, 0) / count if count else 0.0)
        for label, count in totals.items()
    ]
    per_label.sort(key=lambda item: (-item['conflicting_rows'], item['label']))
    result = {
        'schema': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'input': {
            'csv': str(args.csv),
            'manifest': str(args.manifest),
            'feature_count': len(features),
            'features': features,
            'label_column': label_column,
            'equality': 'Every selected raw CSV field is outer-whitespace-trimmed and concatenated in the saved model-feature order before SHA-256 grouping.',
        },
        'totals': {
            'rows': int(rows),
            'conflicting_feature_groups': int(groups),
            'rows_in_conflicting_groups': int(conflicting_rows),
            'fraction_rows_in_conflicting_groups': conflicting_rows / rows if rows else 0.0,
        },
        'per_label': per_label,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['totals'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

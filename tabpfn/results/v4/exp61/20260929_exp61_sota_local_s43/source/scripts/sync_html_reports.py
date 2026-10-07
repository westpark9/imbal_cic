#!/usr/bin/env python3
"""Synchronize local report HTML into its index; does not publish online artifacts."""

import argparse
import base64
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / 'lablog/html_report/post_tabpfn.html'


def sync(index: Path, check: bool) -> int:
    text = index.read_text(encoding='utf-8')
    match = re.search(r'const REPORTS = (.*?);\n', text)
    if match is None:
        raise ValueError(f'No REPORTS JSON array found in {index}')
    reports = json.loads(match.group(1))
    changed = []
    count = 0
    ids = [report['id'] for report in reports]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate report IDs in index')
    for report in reports:
        if report.get('type') != 'iframe':
            continue
        source = index.parent / (report['id'] + '.html')
        encoded = base64.b64encode(source.read_bytes()).decode('ascii')
        count += 1
        if report.get('b64') != encoded:
            changed.append(report['id'])
            report['b64'] = encoded
    if not changed:
        print(f'{count} embedded reports are up to date: {index}')
        return 0
    if check:
        print('Outdated embedded reports: ' + ', '.join(changed))
        return 1
    updated = (text[:match.start(1)] + json.dumps(reports, ensure_ascii=False)
               + text[match.end(1):])
    index.write_text(updated, encoding='utf-8')
    print('Updated embedded reports: ' + ', '.join(changed))
    print(f'Upload this HTML to the existing online artifact: {index}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX,
                        help='Local report index, with source reports in the same folder')
    parser.add_argument('--check', action='store_true',
                        help='Check only; exit 1 if an embedded report is outdated')
    args = parser.parse_args()
    try:
        return sync(args.index.resolve(), args.check)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())

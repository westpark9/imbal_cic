#!/usr/bin/env python3
"""Version the experiment record (user direction 2026-10-02, item 4).

Run directories under tabpfn/results/ and results/ are never moved (scripts reference them by path). Instead a
manifest tabpfn/results/VERSIONS.json assigns every run to a research-direction version by its date (and tag
overrides), an index tabpfn/results/INDEX.md is regenerated, and tabpfn/results/versions/vN_<label>/ holds symlinks
to the runs of each version.

    python scripts/results_versions.py index                 # (re)build INDEX.md and the symlink views
    python scripts/results_versions.py new --label "..." [--start YYYY-MM-DD]   # open a new version (direction change)
    python scripts/results_versions.py assign RUN_DIR vN     # pin one run to a version regardless of its date
Run this after each campaign; the index is the record of which runs belong to which direction.
"""
import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOTS = [ROOT / 'tabpfn/results', ROOT / 'results']
MANIFEST = ROOT / 'tabpfn/results/VERSIONS.json'
INDEX = ROOT / 'tabpfn/results/INDEX.md'
VIEWS = ROOT / 'tabpfn/results/versions'
RUN = re.compile(r'^(?P<date>\d{8})(?:_(?P<time>\d{6}))?(?:_(?P<pid>\d+))?_(?P<rest>.+)$')
DEFAULT = dict(versions=[
    dict(id='v1', label='tabpfn_intro_and_tailguard', start='2026-08-01', note='TabPFN memory study, ensembles, energy-gate expert track'),
    dict(id='v2', label='racepfn_structure_c0_sota_smoke', start='2026-08-26', note='residual experts, scorer/verifier, C0 composition campaign, SOTA PFN ports, routing correction'),
    dict(id='v3', label='data_quality_audit', start='2026-09-11', note='EXP40–44 contradiction/duplication/leakage audit'),
    dict(id='v4', label='conflict_free_expert_sv', start='2026-09-16', note='EXP47–63 on conflict-free splits: expert capability, S/V, K sweep, SOTA and cost'),
    dict(id='v5', label='context_optimisation_temporal', start='2026-10-01', note='EXP64–68: context composition feasibility, prompt-tuned contexts, temporal-arrival protocol'),
], pins={})


def load():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    MANIFEST.write_text(json.dumps(DEFAULT, ensure_ascii=False, indent=2) + '\n'); return json.loads(MANIFEST.read_text())


def save(m):
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2) + '\n')


def runs():
    out = []
    for root in RESULT_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir() or p.name in ('versions', 'archive') or p.name.startswith('.'):
                continue
            m = RUN.match(p.name)
            if m:
                d = m.group('date'); rest = m.group('rest')
            else:
                m2 = re.search(r'(\d{8})', p.name); d = m2.group(1) if m2 else None; rest = p.name
            tag = re.search(r'(exp\d+[a-z0-9]*)', rest); tag = tag.group(1) if tag else rest.split('_')[-1]
            status = 'complete' if any((p / f).exists() for f in ['COMPLETE.json', 'results.json', 'summary.csv', 'status.json']) else 'partial'
            args_line = None
            for log in list(p.glob('*.log'))[:3]:
                try:
                    first = log.read_text(errors='ignore').splitlines()[:5]
                except OSError:
                    continue
                a = next((l for l in first if l.startswith('Args:')), None)
                if a:
                    args_line = a[:160]; break
            out.append(dict(path=str(p.relative_to(ROOT)), date=f'{d[:4]}-{d[4:6]}-{d[6:]}' if d else None, tag=tag, status=status, args=args_line,
                            mtime=time.strftime('%Y-%m-%d', time.localtime(p.stat().st_mtime))))
    return out


def version_of(run, m):
    if run['path'] in m['pins']:
        return m['pins'][run['path']]
    d = run['date'] or run['mtime']
    chosen = None
    for v in sorted(m['versions'], key=lambda v: v['start']):
        if d >= v['start']:
            chosen = v['id']
    return chosen or m['versions'][0]['id']


def index(m):
    allruns = runs(); by = {v['id']: [] for v in m['versions']}
    for r in allruns:
        by[version_of(r, m)].append(r)
    lines = ['# Experiment run index', f'Generated {time.strftime("%Y-%m-%d %H:%M")} by scripts/results_versions.py · runs are never moved; versions are a view.', '']
    VIEWS.mkdir(exist_ok=True)
    for v in m['versions']:
        rs = by[v['id']]; lines += [f"## {v['id']} · {v['label']} (from {v['start']}) — {len(rs)} runs", v.get('note', ''), '', '| date | tag | status | run | args |', '|---|---|---|---|---|']
        view = VIEWS / f"{v['id']}_{v['label']}"; view.mkdir(exist_ok=True)
        for r in rs:
            lines.append(f"| {r['date'] or r['mtime']} | {r['tag']} | {r['status']} | `{r['path']}` | {(r['args'] or '')[:100].replace('|', '/')} |")
            link = view / Path(r['path']).name
            if not link.exists():
                try:
                    link.symlink_to(ROOT / r['path'])
                except OSError:
                    pass
        lines.append('')
    INDEX.write_text('\n'.join(lines))
    print(f'{len(allruns)} runs indexed →', INDEX.relative_to(ROOT), '; views in', VIEWS.relative_to(ROOT))
    for v in m['versions']:
        print(f"  {v['id']} {v['label']}: {len(by[v['id']])} runs")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('index')
    p_new = sub.add_parser('new'); p_new.add_argument('--label', required=True); p_new.add_argument('--start', default=date.today().isoformat()); p_new.add_argument('--note', default='')
    p_as = sub.add_parser('assign'); p_as.add_argument('run'); p_as.add_argument('version')
    a = ap.parse_args(argv); m = load()
    if a.cmd == 'new':
        vid = f"v{len(m['versions']) + 1}"; m['versions'].append(dict(id=vid, label=re.sub(r'[^A-Za-z0-9가-힣]+', '_', a.label).strip('_'), start=a.start, note=a.note)); save(m); print('opened', vid)
    elif a.cmd == 'assign':
        m['pins'][a.run] = a.version; save(m); print('pinned', a.run, '→', a.version)
    index(m)


if __name__ == '__main__':
    main(sys.argv[1:])

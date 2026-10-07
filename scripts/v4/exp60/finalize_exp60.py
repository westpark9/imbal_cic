#!/usr/bin/env python3
"""Collect completed EXP60 datasets and write a local text readout (no HTML)."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

ROOT = repo_root(__file__)
sys.path.insert(0, str(ROOT / 'tabpfn/scripts'))
from exp59_residual_membership_oracle import write


def render(root, complete):
    parts = ['# EXP60 — 상대사례 무작위 / 가까운 context', '',
             'seed 43 · K=4 · 공통 anchor와 core, 총크기, 클래스별 개수를 두 신규 조건 간 고정.',
             '기존 context 대비 변화에는 block 80/20 재배분도 포함된다. 선택 규칙의 효과는 무작위와 가까운 조건 사이에서 비교한다.', '']
    for ds in complete:
        out = root / f'{ds}_s43'
        summary = pd.read_csv(out / 'summary.csv').set_index('model')
        capability = pd.read_csv(out / 'capability_summary.csv').set_index(['model', 'scope'])
        classes = pd.read_csv(out / 'class_metrics.csv').set_index(['model', 'scope', 'class_name'])
        parts += [f'## {ds}', '', '| 평가 | 기존 | 무작위 상대사례 | 가까운 상대사례 |', '|---|---:|---:|---:|']
        for suffix in ['residual_oracle', 'input_nearest']:
            vals = [summary.loc[a+'_'+suffix, 'macro_f1'] for a in ['designed', 'counter_random', 'counter_near']]
            parts.append('| '+suffix+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
        parts += ['', 'Global Macro-F1: '+f'{summary.loc["global", "macro_f1"]:.4f}', '',
                  '| Expert | 범위 | 기존 교정 / 훼손 | 무작위 교정 / 훼손 | 가까운 교정 / 훼손 |', '|---|---|---:|---:|---:|']
        for k in range(1, 5):
            for scope in ['residual', 'observable']:
                cells = []
                for arm in ['designed', 'counter_random', 'counter_near']:
                    row = capability.loc[(f'{arm}_e{k}', scope)]
                    cells.append(f'{int(row.fixed):,} / {int(row.harmed):,}')
                parts.append(f'| e{k} | {scope} | '+' | '.join(cells)+' |')
        parts += ['', '### 클래스별 차이: 가까운 − 무작위', '',
                  '아래는 두 신규 조건의 같은 샘플에서 비교한 결과다. 교정·훼손 합계만으로 채택하지 않고 TP/FP와 클래스별 F1을 함께 확인한다.', '',
                  '| Expert | 범위 | 클래스 | 무작위 F1 | 가까운 F1 | ΔF1 | 무작위 TP / FP | 가까운 TP / FP |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        for k in range(1, 5):
            for scope in ['residual', 'observable']:
                a = classes.loc[(f'counter_random_e{k}', scope)]
                b = classes.loc[(f'counter_near_e{k}', scope)]
                for cl in a.index:
                    x, y = a.loc[cl], b.loc[cl]
                    if not x.support and not x.FP and not y.FP:
                        continue
                    if x.support:
                        values = f'{x.f1:.4f} | {y.f1:.4f} | {y.f1-x.f1:+.4f}'
                    else:
                        values = '— | — | —'
                    parts.append(f'| e{k} | {scope} | {cl} | {values} | {int(x.TP):,} / {int(x.FP):,} | {int(y.TP):,} / {int(y.FP):,} |')
        parts += ['', f'분석 원본: `{out.relative_to(ROOT)}`', '']
    target = ROOT / 'docs/research/20260929/exp60_results.md'
    target.write_text('\n'.join(parts)+'\n')
    return str(target)


def run(root):
    completed = []
    state = root / 'readout_status.json'
    while True:
        for ds in ['cic2018', 'toniot']:
            if ds in completed or not (root/f'{ds}_s43/COMPLETE.json').exists():
                continue
            write(state, dict(state='analyzing', dataset=ds, completed=completed))
            subprocess.run([sys.executable, str(script_path('exp60_counterexamples.py')),
                            '--root',str(root),'--stage','analyze','--dataset',ds],check=True,cwd=ROOT)
            completed.append(ds)
            report = render(root, completed)
            write(state,dict(state='complete' if len(completed)==2 else 'partial_complete',completed=completed,
                             report=report,html_modified=False,updated_epoch=time.time()))
        if len(completed)==2:
            return
        status_path=root/'status.json'
        if status_path.exists() and read_record(status_path).get('state')=='needs_recovery':
            write(state,dict(state='experiment_needs_recovery',completed=completed,updated_epoch=time.time()))
            return
        time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root.resolve())

#!/usr/bin/env python3
"""Consolidate the experiment record into LaTeX (user direction 2026-10-02, item 3).

Reads the saved result files (no recomputation) and writes booktabs tables to docs/latex/results/*.tex plus a
standalone, compilable record docs/latex/experiment_record.tex that \\input{}s them. The paper (docs/slides/main.tex)
can \\input the same table files. Numbers: 4 decimals for F1/accuracy, thousands separators for counts.

    python scripts/build_latex_results.py            # write tables + source manifest
    python scripts/build_latex_results.py --compile  # also run latexmk on the record
"""
import argparse
import glob
import hashlib
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/latex'
RES = OUT / 'results'
DOC = ROOT / 'docs/research/20260930'
OURS_RUN = ROOT / 'tabpfn/results/20260930_exp63_k_sweep_s43'  # Fixed representative setting; never choose a test maximum
OURS_K = 4
LABEL = dict(benign='Benign', bot='Bot', brute_force='Brute force', ddos='DDoS', dos='DoS', infiltration='Infiltration', web_attacks='Web attack',
             backdoor='Backdoor', injection='Injection', mitm='MitM', password='Password', ransomware='Ransomware', scanning='Scanning', xss='XSS')
METHOD = dict(global_raw='Global TabPFN-v3 (C0)', xgb='XGBoost (same 100k)', boostpfn='BoostPFN', localpfn='LoCalPFN (FT)', distpfn='DistPFN (v3)', xgb_full='XGBoost (full train)')
DS = dict(cic2018='CIC-IDS2018', toniot='ToN-IoT')
SHORT = dict(global_raw='Global TabPFN-v3', xgb='XGBoost (100k)', xgb_full='XGBoost (full)', localpfn='LoCalPFN')


def tex_escape(s):
    return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')


def table(cols, rows, caption, label, align=None, note=None, bold_max_cols=(), wide=False):
    align = align or ('l' + 'r' * (len(cols) - 1))
    # wide: shrink to the line width (only for tables whose natural width exceeds it; resizebox would enlarge narrow ones)
    out = ['\\begin{table}[H]', '\\centering', '\\small', f'\\caption{{{caption}}}', f'\\label{{{label}}}'] + (['\\resizebox{\\linewidth}{!}{%'] if wide else []) + [f'\\begin{{tabular}}{{{align}}}', '\\toprule',
           ' & '.join(tex_escape(c) for c in cols) + ' \\\\', '\\midrule']
    maxes = {c: max(float(str(r[c]).replace(',', '').rstrip('%')) for r in rows if _num(r[c])) for c in bold_max_cols}
    for r in rows:
        cells = []
        for c in range(len(cols)):
            v = r[c]; s = tex_escape(v)
            if c in maxes and _num(v) and abs(float(str(v).replace(',', '').rstrip('%')) - maxes[c]) < 1e-12:
                s = f'\\textbf{{{s}}}'
            cells.append(s)
        out.append(' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}' + ('}' if wide else '')]
    if note:
        out.append(f'\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\footnotesize {note}}}')
    out.append('\\end{table}')
    return '\n'.join(out) + '\n'


def _num(v):
    try:
        float(str(v).replace(',', '').rstrip('%')); return True
    except ValueError:
        return False


def f4(x):
    return f'{float(x):.4f}'


def n(x):
    return f'{int(x):,}'


def latest(pattern):
    paths = [p for p in sorted(glob.glob(str(ROOT / pattern))) if Path(p, 'results.json').exists()]
    return json.loads(Path(paths[-1], 'results.json').read_text()) if paths else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    sota = json.loads((DOC / 'exp61_report_data.json').read_text())
    cap = json.loads((DOC / 'exp63_capability_report_data.json').read_text())
    written = {}

    # T1 overall SOTA + proposed K=4 S+V (same source as scripts/build_exp61_report.py: EXP63 K sweep, S1V0 policy, full test)
    rows = []
    for m in sota['methods']:
        a, b = sota['datasets']['cic2018']['results'][m], sota['datasets']['toniot']['results'][m]
        rows.append([METHOD[m], f4(a['macro_f1']), f'{100*a["accuracy"]:.2f}', f4(b['macro_f1']), f'{100*b["accuracy"]:.2f}', '100,000' if m != 'xgb_full' else '8,666,430 / 4,669,119'])
    k6 = {ds: cap['datasets'][ds]['banks'][str(OURS_K)]['policy'] for ds in DS}
    rows.append([f'Proposed: experts (K={OURS_K}) + S/V', f4(k6['cic2018']['macro_f1']), f'{100*k6["cic2018"]["accuracy"]:.2f}', f4(k6['toniot']['macro_f1']), f'{100*k6["toniot"]["accuracy"]:.2f}', '100,000 + expert/route pools'])
    written['tab_sota_overall'] = table(['Method', 'CIC macro-F1', 'CIC acc. (%)', 'ToN macro-F1', 'ToN acc. (%)', 'Training rows (CIC / ToN)'], rows,
        'Macro-F1 and accuracy on the conflict-free test splits (seed 43). The first five methods use exactly the same 100,000 training IDs as the Global context C0; full-train XGBoost and the proposed model use additional training data and are not same-budget comparisons.',
        'tab:sota_overall', align='lrrrrl', note='Test rows: CIC-IDS2018 3,042,473; ToN-IoT 2,271,723. Source: EXP61 (SOTA), EXP63 K sweep (proposed, K=4: previously used representative configuration, not selected as the test maximum, single seed).', wide=True)

    # T2 cost
    rows = []
    for m in sota['methods']:
        a, b = sota['datasets']['cic2018']['results'][m], sota['datasets']['toniot']['results'][m]
        rows.append([METHOD[m], f'{a["fit_seconds"]:,.1f}', f'{a["predict_seconds"]:,.1f}', f'{b["fit_seconds"]:,.1f}', f'{b["predict_seconds"]:,.1f}', f'{a["peak_gpu_sampled_gib"]:.2f} / {b["peak_gpu_sampled_gib"]:.2f}'])
    written['tab_sota_cost'] = table(['Method', 'CIC fit (s)', 'CIC predict (s)', 'ToN fit (s)', 'ToN predict (s)', 'Peak GPU GiB (CIC / ToN)'], rows,
        'Measured cost on one RTX 4090 (24\\,GB) with up to two jobs in parallel: elapsed times include resource contention and are not standalone latencies. LoCalPFN fit includes validation; its prediction includes kNN retrieval. DistPFN shares the Global fit and prediction.',
        'tab:sota_cost', align='lrrrrc', wide=True)

    # T3 per-class F1 per dataset (+ proposed K=4 S+V from the K-sweep run's class_metrics.csv, arm s1v0, full test)
    for ds, title in DS.items():
        d = sota['datasets'][ds]; order = sorted(range(len(d['class_order'])), key=lambda i: -d['results']['global_raw']['classes'][i]['support'])
        with (OURS_RUN / f'{ds}_k{OURS_K}' / 'class_metrics.csv').open() as fh:
            sv = {r['class']: r for r in csv.DictReader(fh) if r['arm'] == 's1v0' and r['split'] == 'full_test'}
        assert all(abs(float(sv[c]['support']) - d['results']['global_raw']['classes'][i]['support']) < 1 for i, c in enumerate(d['class_order']))
        rows = []
        for i in order:
            c = d['class_order'][i]; r = [LABEL[c], n(d['results']['global_raw']['classes'][i]['support'])]
            r += [f4(d['results'][m]['classes'][i]['f1']) for m in sota['methods']] + [f4(float(sv[c]['system_f1']))]
            rows.append(r)
        written[f'tab_classwise_{ds}'] = table(['Class', 'Test rows'] + [SHORT.get(m, METHOD[m].split(' (')[0]) for m in sota['methods']] + [f'Proposed K={OURS_K}'], rows,
            f'Per-class F1 on {title} (seed 43, test rows in descending order). Proposed = experts (K={OURS_K}) + scorer/verifier (S1V0 policy), EXP63 K sweep.', f'tab:classwise_{ds}',
            align='lr' + 'r' * (len(sota['methods']) + 1), wide=True)

    # T4 S/V by K
    rows = []
    for ds, title in DS.items():
        g = cap['datasets'][ds]['banks']['4']['models']['full_test']['global']['summary']['macro_f1']
        rows.append([title, 'Global', f4(g), '0.00', '0.00', '0', '0', '0'])
        for k in cap['ks']:
            p = cap['datasets'][ds]['banks'][str(k)]['policy']
            rows.append([title, f'K = {k}', f4(p['macro_f1']), f'{100*p["proposed"]/p["rows"]:.2f}', f'{100*p["accepted"]/p["rows"]:.2f}', n(p['changed']), n(p['helpful']), n(p['harmful'])])
    written['tab_sv_by_k'] = table(['Dataset', 'Bank', 'Macro-F1', 'Call rate (%)', 'Accept rate (%)', 'Changed', 'Corrected', 'Damaged'], rows,
        'Scorer/verifier (S1V0) on the seed-43 banks with K = 2, 4, 6, 8 (EXP62/63). Calls and acceptances are logical counts obtained by applying the validation-selected policy to stored expert predictions. CIC-IDS2018 keeps the Global prediction at every K; ToN-IoT calls one expert per row ; Scanning is a prominent improvement. Total context rows differ across K, so this is not a fixed-memory comparison.',
        'tab:sv_by_k', align='llrrrrrr')

    # Direct expert competence: identical complete test population for each predictor.
    for ds, title in DS.items():
        models = cap['datasets'][ds]['banks'][str(OURS_K)]['models']['full_test']
        keyed = {m: {r['class']: r for r in v['classes']} for m, v in models.items()}
        classes = sorted(keyed['global'], key=lambda c: -keyed['global'][c]['support'])
        rows = [[LABEL[c], n(keyed['global'][c]['support'])] + [f4(keyed[m][c]['f1']) for m in models] for c in classes]
        written[f'tab_experts_{ds}'] = table(['Class', 'Test rows', 'Global', 'Expert 1', 'Expert 2', 'Expert 3', 'Expert 4'], rows,
            f'Direct expert evaluation on {title}, K=4. All experts predict the same complete test set; class F1 includes false positives from all other classes. This is not label-assisted assignment or an online routing result.', f'tab:experts_{ds}')
        # Full precision/recall retained without a cross-method maximum highlight.
        rows = []
        for c in classes:
            for m in models:
                z = keyed[m][c]
                rows.append([LABEL[c], m, n(z['support']), f4(z['precision']), f4(z['recall']), f4(z['f1'])])
        # Separate optional detail files are paper inputs, not all included in the meeting record.
        lines = [r'\begin{longtable}{llrrrr}', r'\caption{Expert precision, recall and F1: ' + title + r', K=4.}\\', r'\toprule', r'Class & Predictor & Test rows & Precision & Recall & F1 \\', r'\midrule\endhead']
        lines += [' & '.join(tex_escape(v) for v in row) + r' \\' for row in rows]
        lines += [r'\bottomrule', r'\end{longtable}']
        written[f'tab_expert_detail_{ds}'] = '\n'.join(lines) + '\n'

    for name, body in written.items():
        (RES / f'{name}.tex').write_text(body)
    sources = [DOC / 'exp61_report_data.json', DOC / 'exp63_capability_report_data.json']
    sources += [OURS_RUN / f'{ds}_k{OURS_K}/class_metrics.csv' for ds in DS]
    manifest = dict(research_revision='r06_sparse_label_ids', status='exploratory_evidence; sequential protocol not run', representative_k=OURS_K,
                    sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
                    tables=[dict(path=str((RES/f'{k}.tex').relative_to(ROOT)),sha256=hashlib.sha256((RES/f'{k}.tex').read_bytes()).hexdigest()) for k in written],
                    excluded_from_current_claim=['EXP60 withdrawn by user', 'EXP64-67 separate screening, not adopted automatically', 'EXP68 scenario not adopted by user'])
    (OUT / 'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('wrote',len(written),'tables; human-authored narrative preserved')
    if args.compile:
        subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-quiet','experiment_record.tex'],cwd=OUT,check=True)


if __name__ == '__main__':
    main()

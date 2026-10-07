#!/usr/bin/env python3
"""Consolidate the experiment record into LaTeX (user direction 2026-10-02, item 3).

Reads the saved result files (no recomputation) and writes booktabs tables to docs/latex/results/*.tex plus a
standalone, compilable record docs/latex/experiment_record.tex that \\input{}s them. The paper (docs/slides/main.tex)
can \\input the same table files. Numbers: 4 decimals for F1/accuracy, thousands separators for counts.

    python scripts/build_latex_results.py            # write tables + record
    python scripts/build_latex_results.py --compile  # also run latexmk on the record
"""
import argparse
import glob
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
OURS_RUN = ROOT / 'tabpfn/results/20260930_exp63_k_sweep_s43'  # proposed model source, same as scripts/build_exp61_report.py
OURS_K = 6
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
    merged = json.loads((DOC / 'merged_expert_report_data.json').read_text())
    written = {}

    # T1 overall SOTA + proposed K=6 S+V (same source as scripts/build_exp61_report.py: EXP63 K sweep, S1V0 policy, full test)
    rows = []
    for m in sota['methods']:
        a, b = sota['datasets']['cic2018']['results'][m], sota['datasets']['toniot']['results'][m]
        rows.append([METHOD[m], f4(a['macro_f1']), f'{100*a["accuracy"]:.2f}', f4(b['macro_f1']), f'{100*b["accuracy"]:.2f}', '100,000' if m != 'xgb_full' else '8,666,430 / 4,669,119'])
    k6 = {ds: cap['datasets'][ds]['banks'][str(OURS_K)]['policy'] for ds in DS}
    rows.append([f'Proposed: experts (K={OURS_K}) + S/V', f4(k6['cic2018']['macro_f1']), f'{100*k6["cic2018"]["accuracy"]:.2f}', f4(k6['toniot']['macro_f1']), f'{100*k6["toniot"]["accuracy"]:.2f}', '100,000 + expert/route pools'])
    written['tab_sota_overall'] = table(['Method', 'CIC macro-F1', 'CIC acc. (%)', 'ToN macro-F1', 'ToN acc. (%)', 'Training rows (CIC / ToN)'], rows,
        'Macro-F1 and accuracy on the conflict-free test splits (seed 43). The first five methods use exactly the same 100,000 training IDs as the Global context C0; full-train XGBoost and the proposed model use additional training data and are not same-budget comparisons.',
        'tab:sota_overall', align='lrrrrl', bold_max_cols=(1, 2, 3, 4), note='Bold: column maximum. Test rows: CIC-IDS2018 3,042,473; ToN-IoT 2,271,723. Source: EXP61 (SOTA), EXP63 K sweep (proposed, K=6: the best ToN macro-F1 of K in 2, 4, 6, 8, chosen on test, single seed).', wide=True)

    # T2 cost
    rows = []
    for m in sota['methods']:
        a, b = sota['datasets']['cic2018']['results'][m], sota['datasets']['toniot']['results'][m]
        rows.append([METHOD[m], f'{a["fit_seconds"]:,.1f}', f'{a["predict_seconds"]:,.1f}', f'{b["fit_seconds"]:,.1f}', f'{b["predict_seconds"]:,.1f}', f'{a["peak_gpu_sampled_gib"]:.2f} / {b["peak_gpu_sampled_gib"]:.2f}'])
    written['tab_sota_cost'] = table(['Method', 'CIC fit (s)', 'CIC predict (s)', 'ToN fit (s)', 'ToN predict (s)', 'Peak GPU GiB (CIC / ToN)'], rows,
        'Measured cost on one RTX 4090 (24\\,GB) with up to two jobs in parallel: elapsed times include resource contention and are not standalone latencies. LoCalPFN fit includes validation; its prediction includes kNN retrieval. DistPFN shares the Global fit and prediction.',
        'tab:sota_cost', align='lrrrrc', wide=True)

    # T3 per-class F1 per dataset (+ proposed K=6 S+V from the K-sweep run's class_metrics.csv, arm s1v0, full test)
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
            f'Per-class F1 on {title} (seed 43, test rows in descending order). Proposed = experts (K={OURS_K}) + scorer/verifier (S1V0 policy), EXP63 K sweep; bold marks the row maximum.', f'tab:classwise_{ds}',
            align='lr' + 'r' * (len(sota['methods']) + 1), bold_max_cols=tuple(range(2, len(sota['methods']) + 3)), wide=True)

    # T4 S/V by K
    rows = []
    for ds, title in DS.items():
        g = cap['datasets'][ds]['banks']['4']['models']['full_test']['global']['summary']['macro_f1']
        rows.append([title, 'Global', f4(g), '0.00', '0.00', '0', '0', '0'])
        for k in cap['ks']:
            p = cap['datasets'][ds]['banks'][str(k)]['policy']
            rows.append([title, f'K = {k}', f4(p['macro_f1']), f'{100*p["proposed"]/p["rows"]:.2f}', f'{100*p["accepted"]/p["rows"]:.2f}', n(p['changed']), n(p['helpful']), n(p['harmful'])])
    written['tab_sv_by_k'] = table(['Dataset', 'Bank', 'Macro-F1', 'Call rate (%)', 'Accept rate (%)', 'Changed', 'Corrected', 'Damaged'], rows,
        'Scorer/verifier (S1V0) on the seed-43 banks with K = 2, 4, 6, 8 (EXP62/63). Calls and acceptances are logical counts obtained by applying the validation-selected policy to stored expert predictions. CIC-IDS2018 keeps the Global prediction at every K; ToN-IoT calls one expert per row and the gain is almost entirely Scanning.',
        'tab:sv_by_k', align='llrrrrrr')

    # T5 residual-assignment oracle per class (EXP59, K=4) — Global vs oracle
    rows = []
    for ds, title in DS.items():
        pc = pd.read_csv(ROOT / f'tabpfn/results/20260928_exp59_residual_oracle_s43/{ds}_s43/per_class.csv')
        g = pc[pc.model == 'global'].set_index('class'); o = pc[pc.model == 'designed_residual_oracle'].set_index('class')
        for c in g.sort_values('support', ascending=False).index:
            rows.append([title, LABEL[c], n(g.loc[c, 'support']), f4(g.loc[c, 'precision']), f4(g.loc[c, 'recall']), f4(g.loc[c, 'f1']), f4(o.loc[c, 'precision']), f4(o.loc[c, 'recall']), f4(o.loc[c, 'f1'])])
    written['tab_residual_oracle'] = table(['Dataset', 'Class', 'Test rows', 'G P', 'G R', 'G F1', 'Oracle P', 'Oracle R', 'Oracle F1'], rows,
        'Label-assisted residual-assignment oracle (EXP59, K=4): each test row is assigned to the expert of its nearest training residual cluster (the residual uses the test label) and that expert\\textquotesingle s prediction is scored as is. It is a diagnostic, not an attainable bound; for the Benign/Infiltration pair the assignment leaks the label, and for MitM the oracle is worse than Global.',
        'tab:residual_oracle', align='llrrrrrrr', wide=True)

    # T6 feasibility of the four follow-up approaches (10-01)
    e64 = latest('tabpfn/results/*_toniot_exp64_benign_diversity_s43'); e66 = latest('tabpfn/results/*_toniot_exp66_supcon_s43')
    e65 = latest('tabpfn/results/*_cic2018_exp65_lito_lite_web_attacks_s43')
    e67 = {}
    for p in sorted(glob.glob(str(ROOT / 'tabpfn/results/*_exp67_prompt_tuned_s4*'))):
        if Path(p, 'results.json').exists():
            r = json.loads(Path(p, 'results.json').read_text())
            if r['args'].get('full_test'):
                e67[(r['dataset'], r['args']['seed'], r['args']['query_sampling'], r['context_rows'])] = r['results']
    rows = []
    if e64:
        v = e64['results']; rows.append(['1 Benign diversity (ToN)', 'C0 benign rows re-selected (time / k-center / inverse density / hard)', f'{v["random"]["macro_f1"]:.3f} $\\rightarrow$ {min(x["macro_f1"] for x in v.values()):.3f}--{max(x["macro_f1"] for x in v.values()):.3f} (subsample)', 'no effect'])
    if e65:
        a = e65['results']['arms']; gw = sota['datasets']['cic2018']['results']['global_raw']['classes'][6]['f1']
        rows.append(['2 LITO-style synthesis (CIC web)', f'224 real rows $\\rightarrow$ 2,000 synthetic, {e65["results"]["authenticated"]} self-authenticated', f'macro {f4(sota["datasets"]["cic2018"]["results"]["global_raw"]["macro_f1"])} $\\rightarrow$ synthetic {f4(a["c0+syn"]["macro_f1"])}; real duplicates {f4(a["c0+dup"]["macro_f1"])} (web F1 {gw:.3f} $\\rightarrow$ {a["c0+dup"]["classes"][6]["f1"]:.3f})', 'synthesis rejected; tail ratio is a knob'])
    if e66:
        v = e66['results']; rows.append(['3 SupCon prototype embedding (ToN)', '16-d encoder trained on the route pool, appended to features', f'{v["raw"]["macro_f1"]:.3f} $\\rightarrow$ {v["raw+emb"]["macro_f1"]:.3f} (raw+emb), {v["emb"]["macro_f1"]:.3f} (emb) (subsample)', 'no effect'])
    if e67:
        ton = {k[1]: v for k, v in e67.items() if k[0] == 'toniot' and k[2] == 'natural' and k[3] == 3000}
        cic = [v for k, v in e67.items() if k[0] == 'cic2018' and k[2] == 'natural']
        bal = [v for k, v in e67.items() if k[0] == 'toniot' and k[2] == 'balanced']
        txt = f'ToN 3,000-row context: {f4(ton[43]["tuned"]["macro_f1"])} / {f4(ton[44]["tuned"]["macro_f1"])} (seeds 43/44)' + (f'; CIC 2,100-row: {f4(cic[0]["tuned"]["macro_f1"])}' if cic else '') + (f'; balanced-query objective {f4(bal[0]["tuned"]["macro_f1"])}' if bal else '')
        rows.append(['4 Context optimisation (ICD / prompt tuning)', 'context features optimised by gradient through the frozen TabPFN, natural-ratio train queries, 300 steps', txt, 'exceeds the 100k random C0 (ToN 0.6796, CIC 0.7824)'])
    if rows:
        written['tab_feasibility'] = table(['Approach', 'Change', 'Result (full test unless noted)', 'Verdict'], rows,
            'Feasibility of the four follow-up approaches (2026-10-01, seed 43, single runs). Subsample = stratified screening subset with 20k rows per class; it inflates tail precision and is used only for relative comparison.',
            'tab:feasibility', align='p{3.4cm}p{4.6cm}p{5.2cm}p{2.6cm}', wide=True)

    # T7 temporal arrival (EXP68) when available
    e68 = latest('tabpfn/results/*_toniot_exp68_temporal_arrival_s43')
    if e68 and len(e68['results'].get('tabpfn_context', [])) == e68['args']['windows']:
        R = e68['results']; rows = []
        for w in range(e68['args']['windows']):
            rows.append([f'{w+1}', ', '.join(LABEL[c] for c in R['tabpfn_frozen'][w]['present_classes'] if c != 'benign'), n(R['tabpfn_frozen'][w]['rows'])] + [f4(R[a][w]['macro_f1_present']) for a in ['tabpfn_frozen', 'tabpfn_context', 'xgb_frozen', 'xgb_retrain']] + [n(R['tabpfn_context'][w]['labels_used'])])
        written['tab_temporal_arrival'] = table(['Period', 'Attack families present', 'Eval rows', 'TabPFN frozen', 'TabPFN + context', 'XGB frozen', 'XGB retrained', 'Labels'], rows,
            'Temporal-arrival protocol on ToN-IoT (EXP68, seed 43): test rows in time order, five equal periods; in each period the first 10\\% of every family\\textquotesingle s rows may be labelled (at most 50 per class) and the rest is evaluated. Macro-F1 over the classes present in the period. TabPFN + context appends the labelled rows to C0 without any weight update; XGB retrained refits on C0 plus the accumulated labels.',
            'tab:temporal_arrival', align='llrrrrrr', bold_max_cols=(3, 4, 5, 6), wide=True)

    for name, body in written.items():
        (RES / f'{name}.tex').write_text(body)
    record = ['\\documentclass[11pt]{article}', '\\usepackage[margin=2.2cm]{geometry}', '\\usepackage{booktabs}', '\\usepackage{graphicx}', '\\usepackage{float}', '\\usepackage{textcomp}', '\\usepackage{hyperref}',
              '\\title{Experiment record: context-composed TabPFN for long-tailed NetFlow intrusion detection}', '\\author{Generated from saved runs by \\texttt{scripts/build\\_latex\\_results.py}}', f'\\date{{{time.strftime("%Y-%m-%d")}}}',
              '\\begin{document}', '\\maketitle',
              '\\section{Claim and research question}',
              'A frozen tabular foundation model (TabPFN-v3) adapts to newly appearing attack traffic by changing its context rows rather than its weights. On the same 100k training rows its macro-F1 matches XGBoost (Table~\\ref{tab:sota_overall}); the research question is what to put into a context budget that is far smaller than the available training data so that tail attacks are recovered, and how adaptation cost (labels, time) compares with retraining. The target is tail recovery and benign false positives, not all-class macro-F1.',
              '\\section{Data and protocol}',
              'NetFlow-v3 CIC-IDS2018 and ToN-IoT, chronological splits per attack scenario, exact-duplicate label conflicts removed (2026-09-15/16 fixed splits). Test rows: CIC-IDS2018 3,042,473; ToN-IoT 2,271,723. All numbers are seed 43 unless stated; test is a previously observed development holdout, so no result below is a final claim.',
              '\\section{Baselines on the cleaned data (EXP61)}', '\\input{results/tab_sota_overall}', '\\input{results/tab_sota_cost}', '\\input{results/tab_classwise_cic2018}', '\\input{results/tab_classwise_toniot}',
              '\\section{Residual experts and scorer/verifier (EXP59, EXP62, EXP63)}', '\\input{results/tab_residual_oracle}', '\\input{results/tab_sv_by_k}',
              '\\section{Where the benign confusion comes from (2026-10-01 diagnosis)}',
              'Nearest-neighbour distances of Global\\textquotesingle s errors to the training pool (standardised 46-dimensional L2; a typical benign row is 0.01 from training benign): Scanning misses lie 0.60 from training Scanning and 2.92 from benign (a context prior problem, fixed by an expert with a Scanning block plus the verifier: F1 0.039 to 0.90); Infiltration false positives lie 0.11 / 0.17 from both (overlapping region, a feature limit shared by every method); Ransomware false positives lie 0.12 from training Ransomware and 2.84 from benign (near-duplicate label contradictions); MitM false positives lie about 10 from every training class (a benign block that only exists in the test period, i.e. drift).',
              '\\section{Context composition feasibility (EXP64--67, 2026-10-01)}'] + (['\\input{results/tab_feasibility}'] if 'tab_feasibility' in written else []) + [
              '\\section{Temporal-arrival protocol (EXP68)}'] + (['\\input{results/tab_temporal_arrival}'] if 'tab_temporal_arrival' in written else ['Running; see \\texttt{lablog/report/0930.md}.']) + [
              '\\section{Decision log (branch points)}', '\\begin{itemize}',
              '\\item 2026-09-15: data-audit experiments (EXP40--44) closed; contradictions are reported as per-class ceilings, not removed from the headline benchmark; focus moved to improving the context-composed model.',
              '\\item 2026-09-22: residual oracle redefined (label-assisted assignment, designated expert scored as is); union-of-experts oracles are no longer cited as expert capability.',
              '\\item 2026-09-29: class-level specialist substitution without row routing and the open-set leg were rejected by the author; the model structure (global $\\rightarrow$ scorer $\\rightarrow$ top-1 expert $\\rightarrow$ verifier) is fixed.',
              '\\item 2026-09-30: EXP60 (random vs. nearby counterexample contexts) withdrawn from the evidence base; EXP61 SOTA on cleaned data and EXP62/63 S/V completed; three report tabs merged.',
              '\\item 2026-10-01: the earlier A/B interpretation (``experts are good in their own region but cannot separate similar classes\\textquotesingle\\textquotesingle) was retracted in favour of direct same-population evaluation; four follow-up approaches screened, only context optimisation produced new information.',
              '\\item 2026-10-02: the deck and the protocol were reframed claim-first: adaptation without retraining, tail recovery as the target, temporal-arrival evaluation.',
              '\\end{itemize}', '\\end{document}']
    (OUT / 'experiment_record.tex').write_text('\n'.join(record) + '\n')
    (OUT / 'README.md').write_text('Generated by scripts/build_latex_results.py from saved runs. Re-run after new results; tables in results/ can be \\input by the paper.\n')
    print('wrote', len(written), 'tables to', RES, 'and', OUT / 'experiment_record.tex')
    if args.compile:
        r = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-quiet', 'experiment_record.tex'], cwd=OUT, capture_output=True, text=True)
        print('latexmk exit', r.returncode, (OUT / 'experiment_record.pdf').exists())
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-2000:])


if __name__ == '__main__':
    main()

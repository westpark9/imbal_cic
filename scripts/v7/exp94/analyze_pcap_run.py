"""Summarise an EXP94 PCAP run next to the CSV reference: curves, n=20 class F1, 3-layer audit -> JSON + figures."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
COLORS = {'xgb': '#1769aa', 'tabpfn': '#b71c1c'}
LABEL = {'xgb': 'XGBoost', 'tabpfn': 'TabPFN'}
VIEW_TITLE = {'csv': 'CSV (CICFlowMeter)', 'flowstats-v1': 'PCAP flow statistics', 'netfound-v2-small': 'PCAP netFound (PCA 64)'}


def load(p: Path):
    return json.loads(p.read_text())


def aggregate(results):
    cells, classes = defaultdict(list), defaultdict(list)
    for r in results:
        cells[(r['regime'], r['method'], r['budget'])].append(r['metrics']['macro_f1'])
        for c in r['metrics']['classes']:
            classes[(r['regime'], r['method'], r['budget'], c['name'])].append((c['f1'], c['support']))
    macro = [dict(regime=k[0], method=k[1], budget=k[2], seeds=len(v), mean=float(np.mean(v)), sample_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
             for k, v in sorted(cells.items())]
    cls = [dict(regime=k[0], method=k[1], budget=k[2], name=k[3], mean=float(np.mean([x for x, _ in v])),
                sample_sd=float(np.std([x for x, _ in v], ddof=1)) if len(v) > 1 else 0.0, support=int(v[0][1])) for k, v in sorted(classes.items())]
    by_seed = {(r['regime'], r['method'], r['budget'], r['seed']): r['metrics']['macro_f1'] for r in results}
    pd = defaultdict(list)
    for (regime, method, budget, seed), v in by_seed.items():
        if method == 'tabpfn' and (regime, 'xgb', budget, seed) in by_seed:
            pd[(regime, budget)].append(v - by_seed[(regime, 'xgb', budget, seed)])
    paired = [dict(regime=k[0], budget=k[1], mean_delta=float(np.mean(v))) for k, v in sorted(pd.items())]
    return macro, cls, paired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--csv-summary', type=Path, default=ROOT / 'docs/lablog/evidence/20261103/exp94_csv_summary.json')
    ap.add_argument('--summary', type=Path, default=ROOT / 'docs/lablog/evidence/20261104/exp94_pcap_summary.json')
    ap.add_argument('--fig-dir', type=Path, default=ROOT / 'docs/lablog/figures')
    ap.add_argument('--prefix', default='exp94_pcap')
    a = ap.parse_args()
    views = {}
    for vdir in sorted((a.run / 'evaluations').glob('*/fewshot')):
        results = [load(j) for j in sorted(vdir.glob('jobs/*/result.json'))]
        macro, cls, paired = aggregate(results)
        views[vdir.parent.name] = dict(completed_jobs=len(results), macro_f1=macro, class_f1=cls, tabpfn_minus_xgb=paired)
    csv = load(a.csv_summary) if a.csv_summary.exists() else None
    if csv:
        views['csv'] = dict(completed_jobs=csv['completed_jobs'], macro_f1=csv['macro_f1'], class_f1=csv['class_f1'], tabpfn_minus_xgb=csv['tabpfn_minus_xgb'], source=str(a.csv_summary.relative_to(ROOT)))
    audit = load(a.run / 'audit3.json') if (a.run / 'audit3.json').exists() else None
    summary = dict(schema=1, run=str(a.run.relative_to(ROOT)) if a.run.is_relative_to(ROOT) else str(a.run),
                   corpus_manifest={k: v for k, v in load(a.run / 'corpus/manifest.json').items() if k in ('captures', 'label_status_counts', 'source_counts', 'split', 'raw_stats')},
                   label_coverage=load(a.run / 'label_coverage.json'),
                   projection=load(a.run / 'encoders/netfound-v2-small/projection.json') if (a.run / 'encoders/netfound-v2-small/projection.json').exists() else None,
                   audit3={k: v for k, v in (audit or {}).items() if k != 'layers'} | {'layers': {n: {kk: (vv if kk != 'full_labelled' and kk != 'test' else {x: y for x, y in vv.items() if x != 'examples'}) for kk, vv in l.items()} for n, l in (audit or {}).get('layers', {}).items()}},
                   views=views)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = [v for v in ('csv', 'flowstats-v1', 'netfound-v2-small') if v in views]
    regimes = [('with_benign', 'Normal + attack'), ('attack_only', 'Attack families only')]
    a.fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(order), figsize=(4.2 * len(order), 6.4), sharey='row')
    axes = np.atleast_2d(axes)
    for col, view in enumerate(order):
        for row, (regime, title) in enumerate(regimes):
            ax = axes[row, col]
            for method in ('xgb', 'tabpfn'):
                pts = [m for m in views[view]['macro_f1'] if m['regime'] == regime and m['method'] == method]
                if pts:
                    ax.errorbar([p['budget'] for p in pts], [100 * p['mean'] for p in pts], yerr=[100 * p['sample_sd'] for p in pts],
                                color=COLORS[method], marker='o', linewidth=2, capsize=3, label=LABEL[method])
            ax.set_xscale('log'); ax.set_xticks([2, 3, 5, 10, 20, 50]); ax.set_xticklabels(['2', '3', '5', '10', '20', '50']); ax.grid(alpha=.3)
            ax.set_title(f'{VIEW_TITLE.get(view, view)}\n{title}', fontsize=10)
            if row == 1: ax.set_xlabel('Labels per class (n)')
            if col == 0: ax.set_ylabel('Macro-F1 (%)')
    axes[0, -1].legend(loc='lower right')
    fig.tight_layout(); fig.savefig(a.fig_dir / f'{a.prefix}_macro_f1.pdf', bbox_inches='tight'); plt.close(fig)
    # n=20 class F1, with_benign, grouped by view (TabPFN solid, XGB hatched)
    names = sorted({c['name'] for v in order for c in views[v]['class_f1'] if c['budget'] == 20 and c['regime'] == 'with_benign'},
                   key=lambda n: -max(c['support'] for v in order for c in views[v]['class_f1'] if c['name'] == n and c['budget'] == 20 and c['regime'] == 'with_benign'))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(names)); w = .8 / (2 * len(order))
    vcol = {'csv': '#7f7f7f', 'flowstats-v1': '#2ca02c', 'netfound-v2-small': '#b71c1c'}
    k = 0
    for view in order:
        for method, hatch in (('xgb', '//'), ('tabpfn', None)):
            vals = []
            for n in names:
                c = next((c for c in views[view]['class_f1'] if c['name'] == n and c['budget'] == 20 and c['regime'] == 'with_benign' and c['method'] == method), None)
                vals.append(100 * c['mean'] if c else 0)
            ax.bar(x + (k - len(order) + .5) * w, vals, w, color=vcol.get(view, '#333'), hatch=hatch, edgecolor='white', label=f'{VIEW_TITLE.get(view, view)} · {LABEL[method]}')
            k += 1
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel('Class F1 (%)'); ax.set_title('n = 20, normal + attack'); ax.grid(axis='y', alpha=.3)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    fig.savefig(a.fig_dir / f'{a.prefix}_class_f1_n20.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'wrote {a.summary}; views {order}')
    for view in order:
        for m in views[view]['macro_f1']:
            if m['budget'] in (2, 20, 50): print(view, m['regime'], m['method'], m['budget'], round(100 * m['mean'], 2))
    if audit:
        for n, l in audit['layers'].items():
            print(n, 'full cross', l['full_labelled']['cross_label_groups'], l['full_labelled']['cross_label_rows'], 'test', l['test']['cross_label_groups'], 'dup', round(l['train_to_test_exact_duplicate_fraction'], 4), {r: round(o['macro_f1'], 4) for r, o in l['oracle_ceiling'].items()})


if __name__ == '__main__':
    main()

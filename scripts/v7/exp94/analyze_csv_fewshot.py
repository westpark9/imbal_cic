"""Summarise an EXP94 CSV run: learning curves, n=20 class F1, audits -> JSON + figures."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
COLORS = {'xgb': '#1769aa', 'tabpfn': '#b71c1c'}
LABEL = {'xgb': 'XGBoost', 'tabpfn': 'TabPFN'}


def load(p: Path):
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--summary', type=Path, default=ROOT / 'docs/lablog/evidence/20261103/exp94_csv_summary.json')
    ap.add_argument('--fig-dir', type=Path, default=ROOT / 'docs/lablog/figures')
    a = ap.parse_args()
    jobs = sorted((a.run / 'fewshot/jobs').glob('*/result.json'))
    results = [load(j) for j in jobs]
    cells, classes = defaultdict(list), defaultdict(list)
    for r in results:
        cells[(r['regime'], r['method'], r['budget'])].append(r['metrics']['macro_f1'])
        for c in r['metrics']['classes']:
            classes[(r['regime'], r['method'], r['budget'], c['name'])].append((c['f1'], c['support']))
    macro = [dict(regime=k[0], method=k[1], budget=k[2], seeds=len(v), mean=float(np.mean(v)),
                  sample_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in sorted(cells.items())]
    class_f1 = [dict(regime=k[0], method=k[1], budget=k[2], name=k[3], seeds=len(v), mean=float(np.mean([x for x, _ in v])),
                     sample_sd=float(np.std([x for x, _ in v], ddof=1)) if len(v) > 1 else 0.0, support=int(v[0][1]))
                for k, v in sorted(classes.items())]
    paired = []
    by_seed = {(r['regime'], r['method'], r['budget'], r['seed']): r['metrics']['macro_f1'] for r in results}
    for (regime, method, budget, seed), v in by_seed.items():
        if method == 'tabpfn' and (regime, 'xgb', budget, seed) in by_seed:
            paired.append(dict(regime=regime, budget=budget, seed=seed, delta=v - by_seed[(regime, 'xgb', budget, seed)]))
    pd = defaultdict(list)
    for p in paired: pd[(p['regime'], p['budget'])].append(p['delta'])
    paired_mean = [dict(regime=k[0], budget=k[1], mean_delta=float(np.mean(v)), seeds=len(v)) for k, v in sorted(pd.items())]
    summary = dict(schema=1, run=str(a.run.relative_to(ROOT)) if a.run.is_relative_to(ROOT) else str(a.run),
                   completed_jobs=len(results), dataset_manifest=load(a.run / 'dataset_manifest.json'),
                   label_coverage=load(a.run / 'label_coverage.json'), audit=load(a.run / 'audit.json'),
                   macro_f1=macro, class_f1=class_f1, tabpfn_minus_xgb=paired_mean,
                   timing={f'{k[0]}/{k[1]}': dict(fit_mean=float(np.mean([r['fit_seconds'] for r in results if (r['regime'], r['method']) == k])),
                                   infer_mean=float(np.mean([r['inference_seconds'] for r in results if (r['regime'], r['method']) == k])))
                           for k in sorted({(r['regime'], r['method']) for r in results})})
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    regimes = [('with_benign', 'Normal + attack classification'), ('attack_only', 'Attack-family classification')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (regime, title) in zip(axes, regimes):
        for method in ('xgb', 'tabpfn'):
            pts = [m for m in macro if m['regime'] == regime and m['method'] == method]
            if not pts: continue
            ax.errorbar([p['budget'] for p in pts], [100 * p['mean'] for p in pts], yerr=[100 * p['sample_sd'] for p in pts],
                        color=COLORS[method], marker='o', linewidth=2, capsize=3, label=LABEL[method])
        ax.set_xscale('log'); ax.set_xticks([2, 3, 5, 10, 20, 50]); ax.set_xticklabels(['2', '3', '5', '10', '20', '50'])
        ax.set_title(title); ax.set_xlabel('Labels per class (n)'); ax.grid(alpha=.3)
    axes[0].set_ylabel('Macro-F1 (%)'); axes[1].legend(loc='lower right')
    a.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig_dir / 'exp94_csv_macro_f1.pdf', bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (regime, title) in zip(axes, regimes):
        rows = [c for c in class_f1 if c['regime'] == regime and c['budget'] == 20]
        names = sorted({c['name'] for c in rows}, key=lambda n: -next(c['support'] for c in rows if c['name'] == n))
        x = np.arange(len(names)); w = .38
        for k, method in enumerate(('xgb', 'tabpfn')):
            vals = [next((c for c in rows if c['name'] == n and c['method'] == method), None) for n in names]
            ax.bar(x + (k - .5) * w, [100 * (v['mean'] if v else 0) for v in vals], w, yerr=[100 * (v['sample_sd'] if v else 0) for v in vals],
                   capsize=2.5, color=COLORS[method], label=LABEL[method])
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title(f'{title}, n = 20'); ax.grid(axis='y', alpha=.3)
    axes[0].set_ylabel('Class F1 (%)'); axes[1].legend(loc='lower right')
    fig.savefig(a.fig_dir / 'exp94_csv_class_f1_n20.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'{len(results)} jobs; wrote {a.summary} and figures')
    for m in macro:
        if m['budget'] in (2, 20, 50): print(m)


if __name__ == '__main__':
    main()

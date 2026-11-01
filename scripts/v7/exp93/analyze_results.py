#!/usr/bin/env python3
"""Summarize the completed EXP93 PCAP few-shot benchmark and make report figures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'scripts/common/experiment_paths.py').is_file())
DEFAULT_RUN = ROOT / 'tabpfn/results/v7/exp93/20261029_192036_cic2018_pcap_expanded_59b3028b'


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def records(run: Path) -> list[dict[str, Any]]:
    values = [load(path) for path in sorted(run.glob('evaluations/netfound-v2-small/fewshot/jobs/*/result.json'))]
    if len(values) != 120:
        raise ValueError(f'Expected 120 result files, found {len(values)}')
    return values


def mean_sd(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), sample_sd=float(a.std(ddof=1)), minimum=float(a.min()), maximum=float(a.max()))


def summarize(run: Path, values: list[dict[str, Any]]) -> dict[str, Any]:
    config = load(run / 'protocol_snapshot.json')
    manifest = load(run / 'corpus/manifest.json')
    coverage = load(run / 'label_coverage.json')
    projection = load(run / 'encoders/netfound-v2-small/projection.json')
    audit = load(run / 'encoders/netfound-v2-small/audit.json')
    run_meta = load(run / 'RUN_META.json')
    budgets, seeds = config['fewshot']['budgets'], config['fewshot']['seeds']
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        grouped[(value['regime'], value['method'], int(value['budget']))].append(value)
    macro, classes, timing, paired = [], [], [], []
    for regime in config['fewshot']['regimes']:
        for budget in budgets:
            per_method = {}
            for method in config['fewshot']['methods']:
                group = sorted(grouped[(regime, method, budget)], key=lambda item: item['seed'])
                if [item['seed'] for item in group] != seeds:
                    raise ValueError(f'Missing seed in {regime}/{method}/n={budget}')
                metric = mean_sd([item['metrics']['macro_f1'] for item in group])
                macro.append(dict(regime=regime, method=method, budget=budget, seeds=seeds, **metric))
                timing.append(dict(regime=regime, method=method, budget=budget,
                                   fit_seconds=mean_sd([item['fit_seconds'] for item in group]),
                                   inference_seconds=mean_sd([item['inference_seconds'] for item in group])))
                per_method[method] = group
                by_class: dict[str, list[float]] = defaultdict(list)
                for item in group:
                    for entry in item['metrics']['classes']:
                        by_class[entry['name']].append(float(entry['f1']))
                for name, scores in by_class.items():
                    classes.append(dict(regime=regime, method=method, budget=budget, name=name,
                                        support=next(entry['support'] for entry in group[0]['metrics']['classes'] if entry['name'] == name),
                                        **mean_sd(scores)))
            left, right = per_method['xgb'], per_method['tabpfn']
            differences = [right[i]['metrics']['macro_f1'] - left[i]['metrics']['macro_f1'] for i in range(len(seeds))]
            paired.append(dict(regime=regime, budget=budget, comparison='tabpfn_minus_xgb', **mean_sd(differences)))
    input_examples = audit['identical_input_feature_cross_label_examples']
    projected_examples = audit['identical_projected_feature_cross_label_examples']
    return dict(
        schema=1,
        analysis_date='2026-11-01',
        run=dict(path=str(run.relative_to(ROOT)), status=run_meta['status'], started_utc=run_meta['started'],
                 finished_utc=run_meta['finished'], returncode=run_meta['returncode'], git_head=run_meta['git_head']),
        protocol=dict(class_order=config['class_order'], budgets=budgets, seeds=seeds,
                      regimes=config['fewshot']['regimes'], methods=config['fewshot']['methods'],
                      primary_regime='with_benign', adapter='netfound-v2-small'),
        corpus=dict(label_status_counts=manifest['label_status_counts'], source_counts=manifest['source_counts'],
                    sampled_counts=manifest['sampled_counts'], label_schema=manifest['evaluation_label_schema'],
                    coverage=coverage),
        projection=projection,
        audit=dict(rows=audit['rows'], labelled_rows=audit['labelled_rows'], unresolved_rows=audit['unresolved_rows'],
                   group_overlap=audit['group_overlap'], finite_vectors=audit['finite_vectors'],
                   identical_input_feature_groups=audit['identical_input_feature_groups'],
                   identical_input_feature_cross_label_groups=audit['identical_input_feature_cross_label_groups'],
                   identical_input_feature_cross_label_rows=sum(item['rows'] for item in input_examples),
                   identical_projected_feature_groups=audit['identical_projected_feature_groups'],
                   identical_projected_feature_cross_label_groups=audit['identical_projected_feature_cross_label_groups'],
                   identical_projected_feature_cross_label_rows=sum(item['rows'] for item in projected_examples)),
        macro_f1=macro, class_f1=classes, timing=timing, paired_differences=paired,
        ceiling_audit=load(run / 'evaluations/netfound-v2-small/fewshot/ceiling_audit.json')['saturation_candidates'])


def macro_lookup(summary: dict[str, Any], regime: str, method: str) -> list[dict[str, Any]]:
    values = [item for item in summary['macro_f1'] if item['regime'] == regime and item['method'] == method]
    return sorted(values, key=lambda item: item['budget'])


def draw_macro(summary: dict[str, Any], destination: Path) -> None:
    colors = {'xgb': '#1769aa', 'tabpfn': '#b71c1c'}
    labels = {'xgb': 'XGBoost', 'tabpfn': 'TabPFN'}
    regimes = [('with_benign', 'Normal + attack classification'), ('attack_only', 'Attack-family classification')]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65), sharey=True, layout='constrained')
    for ax, (regime, title) in zip(axes, regimes):
        for method in ('xgb', 'tabpfn'):
            values = macro_lookup(summary, regime, method)
            x = list(range(len(values)))
            y = [100 * item['mean'] for item in values]
            e = [100 * item['sample_sd'] for item in values]
            ax.errorbar(x, y, yerr=e, color=colors[method], marker='o', linewidth=2.0,
                        markersize=5.5, capsize=3, label=labels[method])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Labels per class (n)')
        ax.set_xticks(range(6), ['2', '3', '5', '10', '20', '50'])
        ax.set_ylim(35, 103)
        ax.grid(axis='y', alpha=.22)
    axes[0].set_ylabel('Macro-F1 (%)')
    axes[1].legend(loc='lower right', frameon=False)
    fig.savefig(destination, bbox_inches='tight')
    plt.close(fig)


def draw_class_n20(summary: dict[str, Any], destination: Path) -> None:
    colors = {'xgb': '#1769aa', 'tabpfn': '#b71c1c'}
    titles = [('with_benign', 'Normal + attack, n = 20'), ('attack_only', 'Attack families only, n = 20')]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=True, layout='constrained')
    for ax, (regime, title) in zip(axes, titles):
        values = [item for item in summary['class_f1'] if item['regime'] == regime and item['budget'] == 20]
        names = [name for name in summary['protocol']['class_order']
                 if any(item['name'] == name for item in values)]
        x = np.arange(len(names)); width = .37
        for offset, method in [(-width / 2, 'xgb'), (width / 2, 'tabpfn')]:
            entries = {item['name']: item for item in values if item['method'] == method}
            y = [100 * entries[name]['mean'] for name in names]
            e = [100 * entries[name]['sample_sd'] for name in names]
            ax.bar(x + offset, y, width, yerr=e, capsize=2.5, color=colors[method],
                   label='XGBoost' if method == 'xgb' else 'TabPFN')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, names)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=.22)
    axes[0].set_ylabel('Class F1 (%)')
    axes[1].legend(loc='lower right', frameon=False)
    fig.savefig(destination, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'docs/lablog/evidence/20261029/exp93_summary.json')
    parser.add_argument('--macro-figure', type=Path, default=ROOT / 'docs/lablog/figures/exp93_macro_f1.pdf')
    parser.add_argument('--class-figure', type=Path, default=ROOT / 'docs/lablog/figures/exp93_class_f1_n20.pdf')
    args = parser.parse_args()
    summary = summarize(args.run.resolve(), records(args.run.resolve()))
    for path in (args.evidence, args.macro_figure, args.class_figure):
        path.parent.mkdir(parents=True, exist_ok=True)
    dump(args.evidence, summary)
    draw_macro(summary, args.macro_figure)
    draw_class_n20(summary, args.class_figure)
    print(json.dumps(dict(evidence=str(args.evidence), macro_figure=str(args.macro_figure),
                          class_figure=str(args.class_figure), jobs=120), ensure_ascii=False))


if __name__ == '__main__':
    main()

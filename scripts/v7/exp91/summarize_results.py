#!/usr/bin/env python3
"""Validate saved EXP91 confusion matrices and render report artifacts, without fitting."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents
             if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import repo_root
ROOT = repo_root(__file__)

import argparse
import csv
from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    read = lambda p: json.loads(p.read_text())
    complete = read(run/'COMPLETE.json')
    date = datetime.fromisoformat(complete['at']).strftime('%Y%m%d')
    summaries = read(run/'summary.json')
    assert complete['conditions'] == len(summaries) == 10
    rows, class_rows, sources = [], [], {}
    def record(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for f in ['summary.json', 'COMPLETE.json', 'RUN_META.json', 'report_error.txt']:
        record(run/f)
    for name, models in summaries.items():
        job = run/'jobs'/name
        md, cfg = read(job/'input_manifest.json'), read(job/'config.json')
        saved, finished = read(job/'summary.json'), read(job/'COMPLETE.json')
        audit = read(job/'b_audit.json')
        for f in ['input_manifest.json', 'config.json', 'summary.json', 'COMPLETE.json',
                  'a_audit.json', 'b_audit.json']:
            record(job/f)
        assert models == saved and finished['status'] == 'complete'
        assert set(models) == {'a', 'b'} and md['seed'] == 43
        assert cfg['input_sha256'] == md['npz_sha256']
        assert cfg['label_rows'] == sum(md['label_counts'])
        assert cfg['unlabelled_rows'] == md['unlabelled_rows'] == 10000
        assert cfg['test_rows'] == md['test_rows'] == sum(md['test_counts'])
        benign = md['benign_class']
        assert md['label_counts'][benign] == 750
        assert all(c == md['n'] for i, c in enumerate(md['label_counts']) if i != benign)
        for model, result in models.items():
            cm = np.asarray(result['metrics']['confusion_matrix'], dtype=np.int64)
            counts = np.asarray(md['test_counts'])
            assert cm.shape == (len(counts), len(counts)) and (cm >= 0).all()
            np.testing.assert_array_equal(cm.sum(axis=1), counts)
            denominator = cm.sum(axis=0) + cm.sum(axis=1)
            f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(len(counts)), where=denominator != 0)
            macro = np.delete(f1, benign).mean()
            fpr = 1 - cm[benign, benign]/counts[benign]
            np.testing.assert_allclose(macro, result['metrics']['attack_macro_f1'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(fpr, result['metrics']['benign_fpr'], rtol=0, atol=1e-12)
            assert result['export_reload_verified']
            if model == 'b':
                assert result['selected_experts'] == len(audit['selected_steps'])
            rows.append(dict(dataset=md['dataset'], n=md['n'], seed=md['seed'], model=model,
                attack_macro_f1_percent=100*macro, benign_fpr_percent=100*fpr,
                fit_seconds=result['fit_seconds'], inference_seconds=result['inference_seconds'],
                selected_experts=result['selected_experts'],
                train_test_feature_groups=cfg['train_test_feature_group_overlap']))
            for c in np.argsort(-counts, kind='stable'):
                class_rows.append(dict(dataset=md['dataset'], n=md['n'], model=model,
                    class_name=md['class_names'][c], test_rows=int(counts[c]), f1_percent=100*f1[c]))
    rows.sort(key=lambda r: (r['dataset'], r['n'], r['model']))
    data = ROOT/'docs/lablog/evidence'/date
    figures = ROOT/'docs/lablog/figures'/date
    evidence = ROOT/'docs/lablog/evidence'/date
    for directory in [data, figures, evidence]:
        directory.mkdir(parents=True, exist_ok=True)
    for filename, values in [('exp91_results.csv', rows), ('exp91_class_f1.csv', class_rows)]:
        with (data/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(values)
    header = r'Data & $n$ & A F1 (\%) & B F1 (\%) & A FPR (\%) & B FPR (\%) & B experts \\'
    table = [r'\begingroup\small', r'\begin{longtable}{lrrrrrr}', r'\toprule', header,
             r'\midrule\endfirsthead', r'\toprule', header, r'\midrule\endhead']
    for dataset, title in [('cic2018', 'CIC2018'), ('toniot', 'ToN-IoT')]:
        for n in [2, 3, 5, 10, 20]:
            a, b = [r for r in rows if r['dataset'] == dataset and r['n'] == n]
            table.append(f"{title} & {n} & {a['attack_macro_f1_percent']:.2f} & {b['attack_macro_f1_percent']:.2f} & "
                         f"{a['benign_fpr_percent']:.3f} & {b['benign_fpr_percent']:.3f} & {b['selected_experts']} " + r'\\')
        table.append(r'\midrule')
    table += [r'\end{longtable}', r'\endgroup']
    (data/'exp91_results_table.tex').write_text('\n'.join(table)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6), constrained_layout=True)
    for col, (dataset, title) in enumerate([('cic2018', 'CIC2018'), ('toniot', 'ToN-IoT')]):
        for model, color, marker, line in [('a', '#c23a37', 'o', '-'), ('b', '#2866a8', 's', '--')]:
            values = [r for r in rows if r['dataset'] == dataset and r['model'] == model]
            for axis, metric in zip(axes[:, col], ['attack_macro_f1_percent', 'benign_fpr_percent']):
                axis.plot([r['n'] for r in values], [r[metric] for r in values],
                          color=color, marker=marker, linestyle=line, linewidth=1.8,
                          label='A: teacher-student' if model == 'a' else 'B: pair residual')
                axis.set_xticks([2, 3, 5, 10, 20]); axis.grid(alpha=.22)
                axis.set_xlabel('Labels per attack class (n)')
        axes[0, col].set_title(title)
        axes[0, col].set_ylim(0, 80); axes[1, col].set_ylim(0, 11)
        axes[0, col].set_ylabel('Attack macro-F1 (%)')
        axes[1, col].set_ylabel('Benign false-positive rate (%)')
        axes[0, col].legend(handlelength=3.8, fontsize=9, loc='lower right')
    fig.savefig(figures/'exp91_remote_results.pdf')
    plt.close(fig)
    meta = read(run/'RUN_META.json')
    validation = dict(checked_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        run=str(run.relative_to(ROOT)), original_start=meta['started'], original_finish=meta['finished'],
        conditions=10, model_evaluations=20, passed=True,
        checks=['Parent and job summaries agree', 'Confusion matrix supports match full test counts',
                'Macro-F1 and benign FPR recomputed to 1e-12', 'Label budgets and input hashes agree',
                'Export/reload success recorded for all models', 'Expert counts match selection audit'],
        limitation='Saved confusion matrices verified; ignored row-level predictions were not transferred or rerun',
        a_higher_attack_f1_conditions=int(sum(rows[i]['attack_macro_f1_percent'] > rows[i+1]['attack_macro_f1_percent']
                                         for i in range(0, len(rows), 2))),
        b_with_experts_conditions=sum(r['model'] == 'b' and r['selected_experts'] > 0 for r in rows),
        raw_sources_sha256=sources)
    (evidence/'exp91_completed_run_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k != 'raw_sources_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

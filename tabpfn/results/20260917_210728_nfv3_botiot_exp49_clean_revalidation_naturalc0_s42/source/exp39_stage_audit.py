#!/usr/bin/env python3
"""Read frozen EXP39 predictions to separate ranking, verifier, and final policy.

The top1 and post_gt0 rows bypass the calibrated pre threshold and are diagnostic
only. They do not change a fitted model, threshold, or saved test prediction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(run, cache, out):
    if not (run / 'COMPLETE.json').exists():
        raise ValueError('A completed EXP39 run is required')
    args = json.loads((run / 'args.json').read_text())
    selections = json.loads((run / 'selected_thresholds.json').read_text())
    meta = json.loads((cache / 'COMPLETE.json').read_text())
    summary, per_class = [], []
    for split, prefix, positions in [
        ('cal_select', 'cal', 'cal_select_positions.npy'),
        ('cal_confirm', 'cal', 'cal_confirm_positions.npy'),
        ('full_test', 'eval', None),
    ]:
        ix = np.load(run / positions) if positions else slice(None)
        y = np.load(cache / f'{prefix}_y.npy', mmap_mode='r')[ix]
        glob = np.load(cache / f'{prefix}_p0.npy', mmap_mode='r')[ix].argmax(1)
        bank = np.zeros(len(y), bool)
        for k in range(1, meta['n_experts'] + 1):
            bank |= np.load(cache / f'{prefix}_p{k}.npy', mmap_mode='r')[ix].argmax(1) == y
        bank &= glob != y
        for arm in args['arms'].split(','):
            with np.load(run / f'{prefix}_{arm}_scores.npz') as score:
                pred, pre, post = (score[key][ix] for key in ['candidate', 'pre', 'post'])
            policy = selections[arm]
            final_take = np.zeros(len(y), bool) if policy['tau_pre'] is None else (
                (pre > float(policy['tau_pre'])) & (post > policy['tau_post']))
            for stage, take in [('top1_no_gates', np.ones(len(y), bool)),
                                ('post_gt0_no_pre_gate', post > 0),
                                ('original_final_policy', final_take)]:
                helpful = take & (glob != y) & (pred == y)
                harmful = take & (glob == y) & (pred != y)
                row = {'split': split, 'arm': arm, 'stage': stage,
                       'rows': len(y), 'bank_correctable': int(bank.sum()),
                       'helpful': int(helpful.sum()), 'harmful': int(harmful.sum()),
                       'changed': int((take & (pred != glob)).sum()),
                       'correctable_recovery': float(helpful.sum() / max(bank.sum(), 1)),
                       'diagnostic_only': stage != 'original_final_policy',
                       'uses_all_expert_post_information': arm == 'dense_decision_raw'}
                summary.append(row)
                for c, name in enumerate(meta['class_names']):
                    mask = y == c
                    per_class.append({'split': split, 'arm': arm, 'stage': stage,
                                      'class': name, 'rows': int(mask.sum()),
                                      'bank_correctable': int((bank & mask).sum()),
                                      'helpful': int((helpful & mask).sum()),
                                      'harmful': int((harmful & mask).sum())})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / 'stage_counts.csv', index=False)
    pd.DataFrame(per_class).to_csv(out / 'stage_counts_by_class.csv', index=False)
    (out / 'scope.json').write_text(json.dumps({
        'run': str(run), 'cache': str(cache), 'models_refitted': False,
        'thresholds_changed': False, 'original_test_policy_changed': False,
        'note': 'Retrospective stage diagnostics; no threshold selection on confirmation or test.'
    }, indent=2) + '\n')
    print(f'Wrote {len(summary)} stage rows to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.cache, args.out)

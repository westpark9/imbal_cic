#!/usr/bin/env python3
"""Audit a completed EXP39 run; finer-grid diagnostics use calibration only.

Does not alter the frozen experiment, its selected policy, or its test results.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tabpfn' / 'scripts'))
from nfv3_v3_exp39_decision_routing import confusion, metrics_from_cm, population_weights


def audit(run, cache, out):
    if not (run / 'COMPLETE.json').exists():
        raise ValueError('Experiment is not complete')
    cfg = json.loads((run / 'args.json').read_text())
    meta = json.loads((cache / 'COMPLETE.json').read_text())
    names = meta['class_names']; C = len(names)
    tail = [names.index(c) for c in meta['tail_classes']]
    protected = [names.index(c) for c in cfg['protected_classes'].split(',')]
    benign = names.index('benign')
    ix = np.load(run / 'cal_select_positions.npy')
    y = np.load(cache / 'cal_y.npy', mmap_mode='r')[ix]
    glob = np.load(cache / 'cal_p0.npy', mmap_mode='r')[ix].argmax(1)
    prior = np.asarray(meta['train_counts'], float); prior /= prior.sum()
    weights = population_weights(y, prior, C)
    base_cm = confusion(y, glob, C, weights)
    base = metrics_from_cm(base_cm, tail, benign)
    all_rows, summaries, class_rows = [], [], []
    for arm in cfg['arms'].split(','):
        original = pd.read_csv(run / f'4a_{arm}_threshold_grid.csv')
        original = original[~original.reason.fillna('').str.startswith('selected:')]
        with np.load(run / f'cal_{arm}_scores.npz') as scores:
            pre, post, candidate = (scores[k][ix] for k in ['pre', 'post', 'candidate'])
        disagreement = candidate != glob
        best = original.sort_values('delta_macro_f1', ascending=False).iloc[0]
        accepted = (pre > float(best.tau_pre)) & (post > best.tau_post)
        for c, name in enumerate(names):
            mask = (y == c) & accepted
            class_rows.append({'arm': arm, 'class': name,
                'helpful': int((mask & (glob != y) & (candidate == y)).sum()),
                'harmful': int((mask & (glob == y) & (candidate != y)).sum())})
        summary = {'arm': arm, 'original_grid_rows': len(original),
            'original_feasible': int(original.feasible.sum()),
            'original_positive_macro': int((original.delta_macro_f1 > 0).sum()),
            'original_best_delta_macro': float(best.delta_macro_f1),
            'original_best_delta_tail': float(best.delta_tail_f1),
            'original_best_delta_fpr': float(best.delta_benign_fpr),
            'original_best_helpful': int(best.helpful), 'original_best_harmful': int(best.harmful),
            'failure_reasons': json.dumps(original.reason.fillna('').value_counts().to_dict()),
            'max_original_post_threshold': float(original.tau_post.max()),
            'disagreement_rows': int(disagreement.sum())}
        if 'decision' not in arm:
            summaries.append(summary); continue
        # Add high-score boundaries among changed-label candidates. Exact tie
        # handling stays 'score > threshold' as in EXP39; no test observations.
        qq = [0., .25, .5, .75, .9, .95, .975, .99, .995, .999, 1.]
        pre_grid = np.unique(np.r_[original.tau_pre.to_numpy(),
                                   np.quantile(pre[disagreement], qq)])
        post_grid = np.unique(np.r_[original.tau_post.to_numpy(),
                                    np.maximum(0., np.quantile(post[disagreement], qq))])
        dy, dg, de, dw = y[disagreement], glob[disagreement], candidate[disagreement], weights[disagreement]
        ds, dq = pre[disagreement], post[disagreement]
        fine = []
        for tp in pre_grid:
            proposal = float(weights[pre > tp].sum() / weights.sum())
            for tq in post_grid:
                take = (ds > tp) & (dq > tq)
                cm = (base_cm + confusion(dy[take], de[take], C, dw[take])
                       - confusion(dy[take], dg[take], C, dw[take]))
                m = metrics_from_cm(cm, tail, benign)
                dm, dt = m['macro_f1'] - base['macro_f1'], m['tail_f1'] - base['tail_f1']
                df = m['benign_fpr'] - base['benign_fpr']
                h = int(((dg != dy) & (de == dy) & take).sum())
                bad = int(((dg == dy) & (de != dy) & take).sum())
                reason = []
                if proposal > cfg['max_proposal'] + 1e-12: reason.append('proposal')
                if df > cfg['benign_fpr_increase'] + 1e-12: reason.append('benign_fpr')
                if dt < -1e-12: reason.append('tail_f1')
                if any(m['f1'][c] - base['f1'][c] < -cfg['protected_f1_drop'] - 1e-12 for c in protected):
                    reason.append('protected_class_f1')
                if h + bad < cfg['min_decided']: reason.append('support')
                fine.append({'arm': arm, 'tau_pre': tp, 'tau_post': tq,
                    'delta_macro_f1': dm, 'delta_tail_f1': dt, 'delta_benign_fpr': df,
                    'helpful': h, 'harmful': bad, 'changed': int(take.sum()),
                    'feasible': not reason, 'reason': ','.join(reason)})
        fine = pd.DataFrame(fine)
        all_rows.append(fine)
        good = fine[fine.feasible & (fine.delta_macro_f1 > 0)]
        summary['finer_feasible_positive'] = len(good)
        if len(good):
            row = good.sort_values('delta_macro_f1', ascending=False).iloc[0]
            for k in ['delta_macro_f1', 'delta_tail_f1', 'delta_benign_fpr', 'helpful', 'harmful', 'tau_pre', 'tau_post']:
                summary['finer_best_' + k] = row[k]
        summaries.append(summary)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out / 'calibration_grid_audit.csv', index=False)
    pd.DataFrame(class_rows).to_csv(out / 'original_best_candidate_by_class.csv', index=False)
    pd.concat(all_rows).to_csv(out / 'finer_calibration_candidates.csv', index=False)
    (out / 'scope.json').write_text(json.dumps({
        'run': str(run), 'cache': str(cache), 'calibration_only': True,
        'original_test_policy_changed': False,
        'note': 'Post-run diagnostic; finer candidates have not been applied to confirmation or test.'}, indent=2))
    print(pd.DataFrame(summaries).drop(columns='failure_reasons').to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); audit(a.run, a.cache, a.out)

"""Read-only accounting for calibration gates and CIC2018 expert contexts."""
import json
from pathlib import Path

import numpy as np


def explain(cic, read_csv):
    root = Path(cic['launch']['run_dir'])
    cache, diag = root / 'frozen_cache', root / 'diagnostics'
    base = Path(cic['meta']['source_run'])
    names = cic['meta']['class_names']
    result = {'run': cic['run'], 'class_names': names,
              'contexts': read_csv(base / '2a_expert_contexts.csv'),
              'clusters': read_csv(base / '1b_cluster_composition.csv'),
              'k_selection': read_csv(base / '2d_k_selection.csv'),
              'pruning': read_csv(base / '2e_pruning.csv'),
              'residual_stats': read_csv(base / '1a_residual_stats.csv'),
              'cal_grid': []}
    def transition(y, glob, candidate, take):
        cc = int((take & (glob == y) & (candidate == y)).sum())
        h = int((take & (glob != y) & (candidate == y)).sum())
        d = int((take & (glob == y) & (candidate != y)).sum())
        ww = int((take & (glob != y) & (candidate != y)).sum())
        assert cc + h + d + ww == int(take.sum())
        return {'rows': cc+h+d+ww, 'both_correct': cc, 'helpful': h, 'harmful': d,
                'both_wrong': ww, 'changed': int((take & (glob != candidate)).sum())}
    with np.load(diag / 'baseline_cal_scores.npz') as score:
        y = np.load(cache / 'cal_y.npy')
        glob = np.load(cache / 'cal_p0.npy').argmax(1)
        mask = np.load(cache / 'cal_baseline_mask.npy')
        candidate = score['candidate']
        pre = score['score'].max(1)
        for r in cic['grid']:
            called = mask & (pre > r['tau_pre'])
            accepted = called & (score['g_lower'] > r['tau_post'])
            a = transition(y, glob, candidate, accepted)
            assert (a['rows'], a['helpful'], a['harmful']) == (r['accepted'], r['helpful'], r['harmful'])
            result['cal_grid'].append({**r, 'called_transitions': transition(y, glob, candidate, called),
                                       'accepted_transitions': a,
                                       'rejected_transitions': transition(y, glob, candidate, called & ~accepted)})
    args = json.loads((base / 'args.json').read_text())
    result['pre_quantiles'] = [float(x) for x in args['tau_pre_quantiles'].split(',')]
    result['post_grid'] = [float(x) for x in args['tau_post_grid'].split(',')]
    result['threshold_limits'] = {k: args[k] for k in ['cal_min_decided', 'cal_min_accepted', 'cal_max_proposal',
                                                     'cal_harmful_frac', 'cal_benign_fpr_increase']}
    with np.load(diag / 'residual_inputs.npz') as source, np.load(base / 'context_rows.npz') as contexts:
        ids, y = source['row_id'], source['y']
        glob = source['p0'].argmax(1)
        order = ids.argsort(); sorted_ids = ids[order]
        result['global_context_rows'] = len(contexts['C0'])
        result['clip'] = float(source['r_max'])
        result['pool_rows'] = len(ids)
        result['block_stats'] = []
        for k in range(1, 9):
            block_ids = contexts[f'expert{k}_block']
            positions = order[np.searchsorted(sorted_ids, block_ids)]
            assert np.array_equal(ids[positions], block_ids)
            wrong = positions[glob[positions] != y[positions]]
            result['block_stats'].append({'expert': k, 'rows': len(positions), 'global_wrong': len(wrong),
                'global_correct': len(positions)-len(wrong),
                'wrong_by_class': dict(zip(names, np.bincount(y[wrong], minlength=len(names)).tolist())),
                'mean_residual': float(source['residual'][positions].mean())})
    y = np.load(cache / 'eval_y.npy', mmap_mode='r')
    glob = np.load(cache / 'eval_p0.npy', mmap_mode='r').argmax(1)
    benign_correct = (y == names.index('benign')) & (glob == y)
    result['benign_harm_destinations'] = {}
    for k in range(1, 9):
        candidate = np.load(cache / f'eval_p{k}.npy', mmap_mode='r').argmax(1)
        counts = np.bincount(candidate[benign_correct & (candidate != y)], minlength=len(names))
        expected = next(r['harmful'] for r in cic['quality'] if r['model'] == f'expert{k}' and r['class'] == 'benign')
        assert int(counts.sum()) == expected
        result['benign_harm_destinations'][f'expert{k}'] = dict(zip(names, counts.tolist()))
    example = min((r for r in result['cal_grid'] if r['tau_post'] == 0), key=lambda r: abs(r['tau_pre']-.611271))
    assert example['called_transitions']['rows'] == 62428
    assert example['accepted_transitions'] == {'rows': 42435, 'both_correct': 42417, 'helpful': 6,
                                              'harmful': 3, 'both_wrong': 9, 'changed': 9}
    return result

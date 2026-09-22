#!/usr/bin/env python3
"""Five-condition frozen evaluation; no expert fitting and no label-based repair.

1 global; 2 fixed true-class assignment oracle; 3 each expert on the full test;
4 saved scorer with matched calls, no verifier; 5 saved scorer+verifier.
Class mapping and all fixed-FPR thresholds are written before loading test y.
The test is the previously used development holdout, not a fresh final test.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def designated_mapping(block):
    dominant = block.argmax(1)
    mapping = np.full(block.shape[1], -1, dtype=int)
    for c in range(block.shape[1]):
        candidates = np.flatnonzero(dominant == c)
        if len(candidates):
            mapping[c] = candidates[np.argmax(block[candidates, c])]
    return dominant, mapping


def assignment_oracle(y, global_pred, experts, mapping):
    """y selects the fixed owner ONLY. Incorrect expert answers remain incorrect."""
    owner = mapping[y]
    called = owner >= 0
    pred = global_pred.copy()
    pred[called] = experts[np.flatnonzero(called), owner[called]]
    return pred, called


def scorer_policies(global_pred, scores, selection):
    calls = np.zeros(len(global_pred), dtype=bool)
    if selection['tau_pre'] is not None:
        calls = scores['pre'] > float(selection['tau_pre'])
    accepted = np.zeros_like(calls)
    if selection['tau_post'] is not None:
        accepted = calls & (scores['post'] > float(selection['tau_post']))
    return (np.where(calls, scores['candidate'], global_pred),
            np.where(accepted, scores['candidate'], global_pred), calls, accepted)


def fpr_threshold(scores, positive, weights, target):
    """Strict score > cutoff; tied negative scores are excluded conservatively."""
    neg = ~positive
    if not neg.any() or weights[neg].sum() <= 0:
        raise ValueError('Calibration needs negative support')
    order = np.argsort(scores[neg], kind='stable')
    negative_scores = scores[neg][order]
    cumulative = np.cumsum(weights[neg][order], dtype=np.float64)
    index = np.searchsorted(cumulative, (1 - target) * cumulative[-1], side='left')
    return float(negative_scores[min(int(index), len(order) - 1)])


def argmax(path, chunk=250_000):
    p = np.load(path, mmap_mode='r')
    pred = np.empty(len(p), dtype=np.int16)
    for start in range(0, len(p), chunk):
        pred[start:start + chunk] = p[start:start + chunk].argmax(1)
    return pred


def ratio(a, b):
    return float(a / b) if b else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', required=True, type=Path)
    ap.add_argument('--policy-dir', required=True, type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--arm', default='s1v0')
    ap.add_argument('--fpr-targets', default='0.001,0.01')
    args = ap.parse_args()
    start = time.monotonic()
    out, cache, policy = args.out_dir, args.cache_dir.resolve(), args.policy_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'experiment_source.py')
    meta = json.loads((cache / 'COMPLETE.json').read_text())
    identity = json.loads((cache / 'identity.json').read_text())
    source_args = json.loads((policy / 'args.json').read_text())
    assert source_args['cache_identity'] == identity, 'Policy/cache identity mismatch'
    assert Path(source_args['cache_dir']).resolve() == cache
    names, K = meta['class_names'], int(meta['n_experts'])
    C, ben = len(names), names.index('benign')
    tail = [names.index(c) for c in meta['tail_classes']]
    targets = [float(x) for x in args.fpr_targets.split(',')]
    assert all(0 < t < 1 for t in targets)
    context_path = Path(meta['source_run']) / '2a_expert_contexts.csv'
    ctx = pd.read_csv(context_path)
    ctx = ctx[ctx.expert != 'anchor(shared)'].copy()
    ctx['expert'] = ctx.expert.astype(int)
    ctx = ctx.sort_values('expert')
    assert list(ctx.expert) == list(range(1, K + 1))
    block = ctx[names].to_numpy(float)
    dominant, mapping = designated_mapping(block)
    selection = json.loads((policy / 'selected_thresholds.json').read_text())[args.arm]
    dump(out / 'protocol.json', {
        'cache': str(cache), 'policy': str(policy), 'arm': args.arm, 'cache_identity': identity,
        'class_names': names, 'n_experts': K, 'test_role': meta['test_role'],
        'mapping_rule': 'block dominant class; largest target-class block; lowest ID breaks ties',
        'assignment_oracle': 'owner by true class; raw expert output; no correctness fallback',
        'unassigned_classes': 'global by predeclared rule',
        'scorer_only': 'same candidate and tau_pre as saved S+V; remove post gate only',
        'selection': selection, 'fixed_fpr_targets': targets,
        'fixed_fpr_calibration': 'saved cal_select positions, negative weights from train class prior',
        'threshold_comparison': 'strict >; ties conservatively excluded; no test threshold adjustment',
        'average_precision': 'sklearn average_precision_score; not trapezoidal PR integration',
        'test_roc_metrics': 'descriptive recall at empirical test FPR budget; no interpolation or deployment threshold selection',
        'expert_training': False, 'fresh_independent_test': False,
        'source_sha256': sha256(__file__), 'contexts_sha256': sha256(context_path),
        'policy_scores_sha256': sha256(policy / f'eval_{args.arm}_scores.npz'),
        'cal_select_sha256': sha256(policy / 'cal_select_positions.npy'),
    })
    pd.DataFrame([{'class': c, 'expert': int(mapping[i]) + 1 if mapping[i] >= 0 else 0,
                   'block_rows': int(block[mapping[i], i]) if mapping[i] >= 0 else 0}
                  for i, c in enumerate(names)]).to_csv(out / '0a_assignment_map.csv', index=False)

    # Freeze all thresholds using calibration only, before opening eval_y.
    cal_y_all = np.load(cache / 'cal_y.npy')
    select = np.load(policy / 'cal_select_positions.npy')
    confirm = np.load(policy / 'cal_confirm_positions.npy')
    assert not np.intersect1d(select, confirm).size
    cal_y = cal_y_all[select]
    counts = np.bincount(cal_y, minlength=C)
    assert (counts > 0).all()
    prior = np.asarray(meta['train_counts'], float)
    prior /= prior.sum()
    weights = prior[cal_y] / counts[cal_y]
    threshold_rows = []
    for k in range(K + 1):
        cal_p = np.asarray(np.load(cache / f'cal_p{k}.npy', mmap_mode='r')[select])
        assert np.isfinite(cal_p).all()
        for c, name in enumerate(names):
            pos = cal_y == c
            for target in targets:
                threshold = fpr_threshold(cal_p[:, c], pos, weights, target)
                pred = cal_p[:, c] > threshold
                achieved = float(weights[pred & ~pos].sum() / weights[~pos].sum())
                assert achieved <= target + 1e-12
                threshold_rows.append({'model': 'global' if k == 0 else f'expert{k}',
                                       'model_index': k, 'class': name, 'class_index': c,
                                       'target_cal_fpr': target, 'threshold': threshold,
                                       'cal_fpr_weighted': achieved,
                                       'cal_recall': float(pred[pos].mean()),
                                       'cal_positive_rows': int(pos.sum()),
                                       'cal_negative_rows': int((~pos).sum())})
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(out / '0b_fixed_fpr_thresholds.csv', index=False)
    print(f'Protocol and {len(threshold_rows)} calibration thresholds frozen in {time.monotonic()-start:.1f}s', flush=True)

    y = np.load(cache / 'eval_y.npy').astype(np.int16)
    g = argmax(cache / 'eval_p0.npy')
    E = np.stack([argmax(cache / f'eval_p{k+1}.npy') for k in range(K)], axis=1)
    N = len(y)
    assert len(g) == N and E.shape == (N, K)
    with np.load(policy / f'eval_{args.arm}_scores.npz') as z:
        scores = {k: z[k] for k in z.files}
    assert all(v.shape == (N,) for v in scores.values())
    assert (scores['winner'] >= 0).all() and (scores['winner'] < K).all()
    np.testing.assert_array_equal(scores['candidate'], E[np.arange(N), scores['winner']])
    s_only, sv, called, accepted = scorer_policies(g, scores, selection)
    np.testing.assert_array_equal(sv, np.load(policy / f'{args.arm}_final.npy'))
    oracle, assigned = assignment_oracle(y, g, E, mapping)
    all_rows = np.ones(N, bool)
    no_rows = np.zeros(N, bool)
    policies = [('global', 1, g, no_rows, no_rows),
                ('assignment_oracle', 2, oracle, assigned, assigned)]
    policies += [(f'expert{k+1}_always', 3, E[:, k], all_rows, all_rows) for k in range(K)]
    policies += [('scorer_only', 4, s_only, called, called), ('scorer_verifier', 5, sv, called, accepted)]
    summary, per_class, matrices = [], [], []
    g_ok = g == y
    for label, condition, pred, calls, accept in policies:
        cm = np.bincount(y.astype(np.int64) * C + pred, minlength=C*C).reshape(C, C)
        tp = np.diag(cm)
        support, predicted = cm.sum(1), cm.sum(0)
        f1 = np.divide(2*tp, support+predicted, out=np.zeros(C), where=support+predicted > 0)
        ok = pred == y
        h, d = (~g_ok & ok), (g_ok & ~ok)
        summary.append({'condition': condition, 'policy': label, 'rows': N,
                        'macro_f1': float(f1.mean()), 'accuracy': float(ok.mean()),
                        'tail_f1': float(f1[tail].mean()), 'benign_f1': float(f1[ben]),
                        'calls': int(calls.sum()), 'accepted': int(accept.sum()),
                        'changed': int((pred != g).sum()), 'H': int(h.sum()), 'D': int(d.sum()),
                        'net_correction': int(h.sum())-int(d.sum())})
        for c, name in enumerate(names):
            mask = y == c
            fp = int(predicted[c]-tp[c])
            per_class.append({'condition': condition, 'policy': label, 'class': name,
                              'support': int(support[c]), 'TP': int(tp[c]), 'FP': fp,
                              'FN': int(support[c]-tp[c]), 'TN': int(N-support[c]-fp),
                              'precision': ratio(tp[c], predicted[c]), 'recall': ratio(tp[c], support[c]),
                              'f1': float(f1[c]), 'fpr': ratio(fp, N-support[c]),
                              'H': int(h[mask].sum()), 'D': int(d[mask].sum()),
                              'calls': int(calls[mask].sum()), 'accepted': int(accept[mask].sum())})
            for j in range(C):
                matrices.append({'policy': label, 'true_class': name, 'predicted_class': names[j], 'rows': int(cm[c,j])})
        if label in ('assignment_oracle', 'scorer_only', 'scorer_verifier'):
            np.save(out / f'{label}_pred.npy', pred)
    summary = pd.DataFrame(summary)
    per_class = pd.DataFrame(per_class)
    summary.to_csv(out / '1a_summary.csv', index=False)
    per_class.to_csv(out / '1b_per_class.csv', index=False)
    pd.DataFrame(matrices).to_csv(out / '1c_confusion.csv', index=False)
    previous = pd.read_csv(policy / 'summary.csv')
    for arm, label in [('global','global'), (args.arm,'scorer_verifier')]:
        old = previous[(previous.split == 'full_test') & (previous.arm == arm)].iloc[0]
        new = summary[summary.policy == label].iloc[0]
        assert abs(old.macro_f1-new.macro_f1) < 1e-12
        assert int(old.helpful) == int(new.H) and int(old.harmful) == int(new.D)
    print(summary[['policy','macro_f1','calls','H','D']].to_string(index=False), flush=True)

    # Supplemental routing funnel: this observes predictions, not an extra oracle policy.
    e_ok = E == y[:, None]
    bank_correctable = ~g_ok & e_ok.any(1)
    candidate_H = ~g_ok & (scores['candidate'] == y)
    candidate_D = g_ok & (scores['candidate'] != y)
    funnel = []
    for c, name in [(-1, 'ALL')] + list(enumerate(names)):
        mask = all_rows if c < 0 else y == c
        bank = int((mask & bank_correctable).sum())
        ch = int((mask & called & candidate_H).sum())
        cd = int((mask & called & candidate_D).sum())
        ah = int((mask & accepted & candidate_H).sum())
        ad = int((mask & accepted & candidate_D).sum())
        funnel.append({'class': name, 'bank_correctable': bank,
                       'candidate_H_before_call_gate': int((mask & candidate_H).sum()),
                       'candidate_D_before_call_gate': int((mask & candidate_D).sum()),
                       'called_H': ch, 'called_D': cd, 'accepted_H': ah, 'accepted_D': ad,
                       'scorer_recovery': ratio(ch, bank), 'verifier_help_retention': ratio(ah,ch),
                       'verifier_harm_pass': ratio(ad,cd)})
    pd.DataFrame(funnel).to_csv(out / '2a_routing_funnel.csv', index=False)
    regions = np.load(cache / 'eval_distance.npy', mmap_mode='r').argmin(1)
    region_rows = []
    for k in range(K):
        mask = regions == k
        for j, pred in [(0, g)]+[(j+1,E[:,j]) for j in range(K)]:
            region_rows.append({'region': k+1, 'model': 'global' if j == 0 else f'expert{j}',
                                'rows': int(mask.sum()), 'accuracy': float((pred[mask]==y[mask]).mean()),
                                'H': int((mask & ~g_ok & (pred==y)).sum()),
                                'D': int((mask & g_ok & (pred!=y)).sum())})
    pd.DataFrame(region_rows).to_csv(out / '2b_region_matrix.csv', index=False)
    specialization = []
    for k in range(K):
        c = dominant[k]
        mask = y == c
        own = e_ok[mask,k].mean()
        others = np.delete(e_ok[mask].mean(0), k)
        row = per_class[(per_class.policy==f'expert{k+1}_always') & (per_class['class']==names[c])].iloc[0]
        specialization.append({'expert': k+1, 'class': names[c], 'block_share': block[k,c]/block[k].sum(),
                               'global_recall': float(g_ok[mask].mean()), 'own_recall': float(own),
                               'best_other_recall': float(others.max()), 'H': int(row.H), 'D': int(row.D),
                               'outside_FP': int(row.FP), 'outside_FPR': float(row.fpr),
                               'correctable_global_error_fraction': ratio(row.H,(mask & ~g_ok).sum()),
                               'global_correct_damage_fraction': ratio(row.D,(mask & g_ok).sum()),
                               'unique_global_corrections': int((~g_ok & e_ok[:,k] & (e_ok.sum(1)==1)).sum())})
    pd.DataFrame(specialization).to_csv(out / '3a_specialization.csv', index=False)
    print(f'Five conditions and H/D diagnostics complete at {time.monotonic()-start:.1f}s; evaluating probability ranking', flush=True)

    operating, ap_rows, roc_rows = [], [], []
    for k in range(K + 1):
        p = np.load(cache / f'eval_p{k}.npy', mmap_mode='r')
        for c, name in enumerate(names):
            score = np.asarray(p[:,c]).copy()
            assert np.isfinite(score).all()
            positive = y == c
            ap_rows.append({'model': 'global' if k==0 else f'expert{k}', 'class': name,
                            'test_average_precision': float(average_precision_score(positive, score)),
                            'test_prevalence': float(positive.mean())})
            fpr, tpr, _ = roc_curve(positive, score, drop_intermediate=False)
            for target in targets:
                eligible = fpr <= target
                best = np.flatnonzero(eligible)[np.argmax(tpr[eligible])]
                roc_rows.append({'model': 'global' if k==0 else f'expert{k}', 'class': name,
                                 'test_fpr_budget': target, 'achieved_test_fpr': float(fpr[best]),
                                 'test_descriptive_recall_at_fpr': float(tpr[best])})
            for t in threshold_rows:
                if t['model_index'] != k or t['class_index'] != c:
                    continue
                pred = score > t['threshold']
                tp, fp = int((pred & positive).sum()), int((pred & ~positive).sum())
                operating.append({**t, 'test_TP': tp, 'test_FP': fp,
                                  'test_recall': ratio(tp, positive.sum()),
                                  'test_fpr': ratio(fp,(~positive).sum()), 'test_precision': ratio(tp,tp+fp)})
        print(f'Probability diagnostics model {k}/{K}: {time.monotonic()-start:.1f}s', flush=True)
    pd.DataFrame(operating).to_csv(out / '4a_fixed_fpr.csv', index=False)
    pd.DataFrame(ap_rows).to_csv(out / '4b_average_precision.csv', index=False)
    pd.DataFrame(roc_rows).to_csv(out / '4c_test_roc_metrics.csv', index=False)
    dump(out / 'COMPLETE.json', {'seconds': time.monotonic()-start, 'rows': N, 'n_experts': K,
                               'dataset': meta.get('dataset') or meta['source_config']['target_dataset'],
                               'cache_identity': identity, 'test_role': meta['test_role'],
                               'source_policy_reproduced_exactly': True,
                               'macro_f1': dict(zip(summary.policy,summary.macro_f1))})
    print(f'DONE {time.monotonic()-start:.1f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()

"""Read EXP47 traces; report opportunities, gate attrition, and expert quality."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from cic2018_conflict_clean import write_json
from nfv3_v3_exp31_c0alloc import build_pair_post


def counts(y, glob, candidate, bank, masks, names):
    rows = []
    for stage, take in masks.items():
        pred = np.where(take, candidate, glob)
        for c, name in enumerate(names):
            m = y == c
            possible = m & bank & (glob != y)
            h = int((m & (glob != y) & (pred == y)).sum())
            rows.append({'stage': stage, 'class': name, 'rows': int(m.sum()),
                         'global_errors': int((m & (glob != y)).sum()),
                         'bank_correctable': int(possible.sum()),
                         'taken': int((take & m).sum()), 'helpful': h,
                         'harmful': int((m & (glob == y) & (pred != y)).sum()),
                         'changed': int((m & (pred != glob)).sum()),
                         'correctable_recovery': h / max(int(possible.sum()), 1)})
    return rows


def run(root):
    root = Path(root); diag = root / 'diagnostics'; cache = root / 'frozen_cache'
    meta = json.loads((cache / 'COMPLETE.json').read_text())
    names, K = meta['class_names'], meta['n_experts']
    base = Path(meta['source_run'])
    def array(split, key): return np.load(cache / f'{split}_{key}.npy', mmap_mode='r')
    # Residual mass versus actual global errors.
    with np.load(diag / 'residual_inputs.npz') as r:
        y, p = r['y'], r['p0']
        frame = pd.DataFrame({'class': np.asarray(names)[y], 'global_pred': np.asarray(names)[p.argmax(1)],
                              'wrong': p.argmax(1) != y, 'residual': r['residual'],
                              'clipped_residual': r['clipped_residual'], 'clipped': r['residual'] > r['r_max']})
        frame.groupby(['class', 'global_pred']).agg(rows=('wrong', 'size'), wrong=('wrong', 'sum'),
            residual_mass=('residual', 'sum'), clipped_mass=('clipped_residual', 'sum'),
            clipped_rows=('clipped', 'sum')).reset_index().to_csv(diag / 'residual_audit.csv', index=False)
    with np.load(diag / 'scorer_targets.npz') as t:
        y, G, b, U = t['y'], t['gain'], t['b_oof'], t['utility']
        glob = array('route', 'p0').argmax(1)
        rows = []
        for k in range(K):
            pred = array('route', f'p{k + 1}').argmax(1)
            helpful = (glob != y) & (pred == y)
            for c, name in enumerate(names):
                m = y == c
                rows.append({'expert': k + 1, 'class': name, 'rows': int(m.sum()),
                             'helpful_available': int((m & helpful).sum()),
                             'positive_gain': int((m & (G[:, k] > 0)).sum()),
                             'helpful_and_positive_gain': int((m & helpful & (G[:, k] > 0)).sum()),
                             'helpful_passes_oof': int((m & helpful & (b[:, k] > 0)).sum()),
                             'scorer_positive': int((m & (U[:, k] > 0)).sum()),
                             'helpful_scorer_positive': int((m & helpful & (U[:, k] > 0)).sum())})
        pd.DataFrame(rows).to_csv(diag / 'scorer_target_audit.csv', index=False)
    verifier = joblib.load(diag / 'baseline_verifier.joblib')
    cal = np.load(diag / 'baseline_cal_scores.npz')
    tau_pre, tau_post, q_corr = [float(cal[k]) for k in ('tau_pre', 'tau_post', 'q_corr')]
    metrics, stage_rows, evidence = [], [], {}
    for split in ['cal', 'eval']:
        y = array(split, 'y'); p0 = array(split, 'p0'); glob = p0.argmax(1)
        score = cal['score'] if split == 'cal' else array(split, 'baseline_scorer')
        top = score.argmax(1); pre = score.max(1)
        candidate = glob.copy(); bank = np.zeros(len(y), bool); q_hat = np.zeros(len(y))
        affinity0, distance = array(split, 'affinity_0'), array(split, 'distance')
        qk = np.load(cache / 'qk.npy')
        for k in range(K + 1):
            pk = array(split, f'p{k}'); pred = pk.argmax(1)
            precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(len(names)), zero_division=0)
            for c, name in enumerate(names):
                m = y == c
                metrics.append({'split': split, 'model': 'global' if k == 0 else f'expert{k}', 'class': name,
                                'precision': precision[c], 'recall': recall[c], 'f1': f1[c], 'support': int(support[c]),
                                'helpful': int((m & (glob != y) & (pred == y)).sum()),
                                'harmful': int((m & (glob == y) & (pred != y)).sum())})
            if k == 0: continue
            bank |= pred == y
            selected = np.flatnonzero(top == k - 1)
            candidate[selected] = pred[selected]
            ak = array(split, f'affinity_{k}')
            for start in range(0, len(selected), 100_000):
                ix = selected[start:start + 100_000]
                hp = build_pair_post(p0[ix], pk[ix], affinity0[ix], ak[ix], distance[ix, k - 1], qk[k - 1])
                q_hat[ix] = verifier.predict(hp)
        if split == 'cal':
            # Use original per-query cal predictions and verifier scores for original gates.
            evidence['cal_dense_vs_original_candidate_changes'] = int((candidate != cal['candidate']).sum())
            candidate, q_hat = cal['candidate'], cal['q_hat']
        lower = q_hat - q_corr
        masks = {'top1_no_gates': np.ones(len(y), bool), 'pre_only': pre > tau_pre,
                 'raw_verifier_gt0_diagnostic': q_hat > 0,
                 'post_only': lower > tau_post, 'both_gates_recomputed': (pre > tau_pre) & (lower > tau_post)}
        rows = counts(y, glob, candidate, bank, masks, names)
        stage_rows.extend([{**r, 'split': split} for r in rows])
        np.savez_compressed(diag / f'baseline_{split}_dense_candidate_scores.npz', candidate=candidate,
                            pre=pre, q_hat=q_hat, g_lower=lower)
        if split == 'eval':
            with np.load(base / 'system_dump.npz') as native:
                replay = np.where(masks['both_gates_recomputed'], candidate, glob)
                evidence['dense_diagnostic_vs_native_final_label_changes'] = int((replay != native['final']).sum())
                actual_rows = counts(y, glob, native['final'], bank, {'native_final_policy': np.ones(len(y), bool)}, names)
                for row in actual_rows:
                    c = names.index(row['class'])
                    row['taken'] = int(((y == c) & native['accepted']).sum())
                stage_rows.extend([{**r, 'split': split} for r in actual_rows])
    pd.DataFrame(metrics).to_csv(diag / 'expert_quality.csv', index=False)
    pd.DataFrame(stage_rows).to_csv(diag / 'baseline_routing_stage_counts.csv', index=False)
    write_json(diag / 'audit_scope.json', {**evidence, 'new_thresholds_selected': False,
        'cal_role': 'baseline selection/training diagnostics; independent EXP39 confirm results reported separately',
        'dense_gate_results': 'diagnostic; native_final_policy is actual EXP31 result'})
    write_json(diag / 'COMPLETE.json', {'metrics_rows': len(metrics), 'stage_rows': len(stage_rows)})

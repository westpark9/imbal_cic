"""Observation-only dumps from EXP31; full expert evaluation uses its dense flag."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from cic2018_conflict_clean import sha256, write_json


class Trace:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = self.root / 'frozen_cache'
        self.diag = self.root / 'diagnostics'
        self.cache.mkdir(exist_ok=True); self.diag.mkdir(exist_ok=True)
        self.started = time.monotonic()
        self.meta = {}

    def save(self, name, a):
        np.save(self.cache / (name + '.npy'), a, allow_pickle=False)

    def split_features(self, split, p0, z, d2, a0, aff_k):
        self.save(split + '_p0', p0)
        self.save(split + '_z', z)
        self.save(split + '_distance', np.sqrt(d2))
        self.save(split + '_affinity_0', a0)
        for k, aff in enumerate(aff_k, 1):
            self.save(f'{split}_affinity_{k}', aff.score(z))

    def record(self, stage, v):
        write_json(self.root / 'trace_progress.json', {'stage': stage, 'elapsed_seconds': time.monotonic() - self.started})
        print('EXP47 TRACE: ' + stage, flush=True)
        if stage == 'inputs':
            self.meta = {'class_names': v['class_names'], 'tail_classes': v['tail_classes'],
                         'protected_classes': ['brute_force', 'ddos', 'dos'],
                         'train_counts': np.bincount(v['y_train'], minlength=v['n_classes']).tolist(),
                         'clean_manifest': v['args'].clean_manifest}
            for split, idx, X, y in [('route', v['route_idx'], v['X_route'], v['y_route']),
                                      ('cal', v['cal_idx'], v['X_cal'], v['y_cal']),
                                      ('eval', v['eval_idx'], v['X_eval'], v['y_eval'])]:
                for key, arr in [('ids', idx), ('X', X), ('y', y),
                                 ('time', v['ts_all'][idx]), ('scenario', v['scen_all'][idx]),
                                 ('hash', pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy())]:
                    self.save(split + '_' + key, arr)
            self.save('cal_mask', ~v['cal_route_dup'])  # EXP39 strict route-overlap mask.
            self.save('cal_baseline_mask', v['cal_sel_mask'])
            np.savez_compressed(self.diag / 'context_membership_initial.npz', C0=v['g_idx'], anchor=v['anchor_idx'],
                                phi_fit=v['phi_fit_idx'], expert=v['exp_idx'], route=v['route_idx'],
                                tune=v['tune_idx'], cal=v['cal_idx'], eval=v['eval_idx'])
        elif stage == 'residual':
            np.savez_compressed(self.diag / 'residual_inputs.npz', row_id=v['exp_idx'], y=v['y_exp'],
                                p0=v['p0_exp'], class_weight=v['w_bal'], residual=v['r_exp'],
                                clipped_residual=v['r_bar'], r_max=v['r_max'])
        elif stage == 'preprune':
            ids = {f'expert{k + 1}': v['exp_idx'][ex['block_rows']] for k, ex in enumerate(v['bank']['experts'])}
            np.savez_compressed(self.diag / 'context_membership_preprune.npz', **ids)
            np.savez_compressed(self.diag / 'preprune_tune.npz', row_id=v['tune_idx'], y=v['y_tune'],
                                p0=v['p0_tune'], pk=np.stack(v['pk_tune']),
                                mask=v['tune_sel_mask'], assign=v['assign_tune'])
        elif stage == 'route':
            self.split_features('route', v['p0_rt'], v['z_rt'], v['d2_rt'], v['a0_rt'], v['aff_k'])
            for k, pk in enumerate(v['pk_rt'], 1): self.save(f'route_p{k}', pk)
            self.save('qk', v['qk_mat'])
        elif stage == 'targets':
            np.savez_compressed(self.diag / 'scorer_targets.npz', y=v['y_route'], gain=v['G_rt'],
                                b_oof=v['b_oof'], utility=v['U_rt'], positive=v['sc_label'])
        elif stage == 'cal':
            self.split_features('cal', v['p0_cal'], v['z_cal'], v['d2_cal'], v['a0_cal'], v['aff_k'])
            np.savez_compressed(self.diag / 'baseline_cal_scores.npz', score=v['U_hat_cal'],
                                candidate=v['top1_pred_cal'], q_hat=v['q_hat_cal'],
                                g_lower=v['g_lower_cal'], gain=v['g_cal'],
                                tau_pre=v['tau_pre'], tau_post=v['tau_post'], q_corr=v['q_corr'])
            rows = []
            mask, y = v['cal_sel_mask'], v['y_cal']
            g, candidate = v['y_glob_cal'], v['top1_pred_cal']
            benign = y == v['benign_id']
            for tp in v['tau_pre_grid']:
                called = (v['u_max_cal'] > tp) & mask
                for to in v['tau_post_grid']:
                    acc = called & (v['g_lower_cal'] > to)
                    h = int((acc & (g != y) & (candidate == y)).sum())
                    d = int((acc & (g == y) & (candidate != y)).sum())
                    fpr = (int((acc & benign & (candidate != y)).sum()) - int((acc & benign & (g != y)).sum())) / max(int((benign & mask).sum()), 1)
                    args = v['args']; reasons = []
                    if called.sum() / mask.sum() > args.cal_max_proposal: reasons.append('proposal')
                    if d / max(h + d, 1) > args.cal_harmful_frac: reasons.append('harmful_fraction')
                    if fpr > args.cal_benign_fpr_increase: reasons.append('benign_fpr')
                    if acc.sum() < args.cal_min_accepted: reasons.append('accepted_support')
                    if h + d < args.cal_min_decided: reasons.append('decided_support')
                    rows.append({'tau_pre': tp, 'tau_post': to, 'helpful': h, 'harmful': d,
                                 'accepted': int(acc.sum()), 'net_gain': float(v['g_cal'][acc].sum()),
                                 'feasible': not reasons, 'reason': ','.join(reasons)})
            pd.DataFrame(rows).to_csv(self.diag / 'baseline_policy_rejection_reasons.csv', index=False)
        elif stage == 'eval':
            self.split_features('eval', v['p0_eval'], v['z_eval'], v['d2_eval'], v['a0_eval'], v['aff_k'])
            self.save('eval_baseline_scorer', v['U_hat_ev'])
        elif stage == 'dense':
            self.save(f"eval_p{v['k'] + 1}", v['pk'])
        elif stage == 'complete':
            import joblib
            # Additional calibration predictions occur AFTER the original policy/test evaluation.
            cal_original = np.load(self.diag / 'baseline_cal_scores.npz')
            top = cal_original['score'].argmax(1)
            dense_cal_disagreement = 0
            for k, ex in enumerate(v['experts']):
                pk = ex['corr'].correct(v['batched_proba'](ex['clf'], v['X_cal'], f'exp47/e{k + 1}/cal_all'), v['beta'], v['temp'])
                self.save(f'cal_p{k + 1}', pk)
                take = top == k
                dense_cal_disagreement += int((pk[take].argmax(1) != cal_original['candidate'][take]).sum())
            joblib.dump(v['scorer'], self.diag / 'baseline_scorer.joblib')
            joblib.dump(v['verifier'], self.diag / 'baseline_verifier.joblib')
            meta = {**self.meta, 'n_experts': v['K'], 'source_run': v['out_dir'],
                    'source_config': vars(v['args']), 'full_test_rows': len(v['y_eval']),
                    'cal_mask_rows': int(np.load(self.cache / 'cal_mask.npy').sum()),
                    'test_role': 'conflict-free subset of previously-used development holdout',
                    'prediction_source': 'same fitted EXP31 models; no reconstruction/refit',
                    'cal_all_vs_original_top1_label_changes': dense_cal_disagreement,
                    'seconds': time.monotonic() - self.started}
            for split in ['route', 'cal', 'eval']:
                n = len(np.load(self.cache / f'{split}_y.npy', mmap_mode='r'))
                for name in ['X', 'ids', 'time', 'scenario', 'hash', 'z', 'distance'] + [f'p{k}' for k in range(v['K'] + 1)] + [f'affinity_{k}' for k in range(v['K'] + 1)]:
                    a = np.load(self.cache / f'{split}_{name}.npy', mmap_mode='r')
                    if len(a) != n: raise ValueError(f'Row mismatch {split}/{name}')
            identity = {'schema': 'exp47_direct_v1', 'clean_manifest_sha256': sha256(v['args'].clean_manifest),
                        'source_args_sha256': sha256(Path(v['out_dir']) / 'args.json'),
                        'contexts_sha256': sha256(Path(v['out_dir']) / 'context_rows.npz'),
                        'trace_source_sha256': sha256(__file__)}
            write_json(self.cache / 'identity.json', identity)
            write_json(self.cache / 'COMPLETE.json', meta)


def instrument(source):
    """Only insert observation callbacks, with exact anchors checked before launch."""
    anchors = [
        ('    del X\n    core._PICKLE_CACHE.clear()', 'inputs', 'before', '    '),
        ('    timings["residual_clip_r_max"] = round(r_max, 6)', 'residual', 'before', '    '),
        ('    keep, prune_df = greedy_regime_prune(', 'preprune', 'before', '    '),
        ('    def quantile_verifier(seed):', 'route', 'before', '    '),
        ('    if sc_label.min() == sc_label.max():', 'targets', 'before', '    '),
        ('    del z_cal, d2_cal, a0_cal, p0_cal', 'cal', 'before', '    '),
        ('    timings["eval_scorer_s"] = round(time.time() - t0, 1)', 'eval', 'after', '    '),
        ('            dense_preds.append(pk.argmax(axis=1))', 'dense', 'after', '            '),
        ('    return out_dir', 'complete', 'before', '    '),
    ]
    for anchor, stage, where, indent in anchors:
        if source.count(anchor) != 1:
            raise ValueError(f'Ambiguous instrumentation anchor: {stage}')
        hook = indent + f'_exp47_trace.record("{stage}", locals())'
        replacement = hook + '\n' + anchor if where == 'before' else anchor + '\n' + hook
        source = source.replace(anchor, replacement, 1)
    return source

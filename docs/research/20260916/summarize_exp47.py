"""Reproduce the key EXP47 report evidence from saved predictions; no training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / 'tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42'
BASE = next((RUN / 'baseline').iterdir())
ROUTING = next((RUN / 'routing').iterdir())
DIAG, CACHE = RUN / 'diagnostics', RUN / 'frozen_cache'
names = json.loads((CACHE / 'COMPLETE.json').read_text())['class_names']


def array(split, key):
    return np.load(CACHE / f'{split}_{key}.npy', mmap_mode='r')


def main():
    result = {'run': str(RUN.relative_to(ROOT)), 'completion': json.loads((RUN / 'COMPLETE.json').read_text()),
              'new_models_trained': False, 'new_thresholds_selected': False}
    with np.load(BASE / 'system_dump.npz') as s:
        assert np.array_equal(s['final'], s['y_glob'])
        result['native'] = {'macro_f1': f1_score(s['y_true'], s['final'], labels=np.arange(len(names)), average='macro'),
                            'rows': len(s['y_true']), 'global_errors': int((s['y_true'] != s['y_glob']).sum()),
                            'calls': int((s['proposal'] != 0).sum()), 'accepted': int(s['accepted'].sum()),
                            'q_corr': float(s['q_corr'])}
    with np.load(DIAG / 'scorer_targets.npz') as t:
        y, positive = t['y'], t['utility'] > 0
        glob = array('route', 'p0').argmax(1)
        categories = {'both_correct': 0, 'helpful': 0, 'harmful': 0, 'both_wrong': 0}
        for k in range(positive.shape[1]):
            pred = array('route', f'p{k + 1}').argmax(1)
            for name, mask in [('both_correct', (glob == y) & (pred == y)), ('helpful', (glob != y) & (pred == y)),
                               ('harmful', (glob == y) & (pred != y)), ('both_wrong', (glob != y) & (pred != y))]:
                categories[name] += int((positive[:, k] & mask).sum())
        assert sum(categories.values()) == int(positive.sum())
        result['positive_scorer_training_pairs'] = {'total': int(positive.sum()), **categories,
            'fractions': {key: n / int(positive.sum()) for key, n in categories.items()}}
    target = pd.read_csv(DIAG / 'scorer_target_audit.csv')
    result['scorer_teacher_pairs'] = target[['helpful_available', 'helpful_passes_oof']].sum().astype(int).to_dict()
    residual = pd.read_csv(DIAG / 'residual_audit.csv').groupby('class').sum(numeric_only=True)
    residual['raw_mass_fraction'] = residual.residual_mass / residual.residual_mass.sum()
    residual['clipped_mass_fraction'] = residual.clipped_mass / residual.clipped_mass.sum()
    with np.load(DIAG / 'residual_inputs.npz') as r, np.load(BASE / 'context_rows.npz') as contexts:
        used = np.unique(np.concatenate([contexts[k] for k in contexts.files if k.startswith('expert') and k.endswith('_block')]))
        selected = np.isin(r['row_id'], used)
        wrong = r['p0'].argmax(1) != r['y']
        for c, name in enumerate(names):
            m = r['y'] == c
            residual.loc[name, 'context_rows'] = int((selected & m).sum())
            residual.loc[name, 'global_errors_in_context'] = int((selected & m & wrong).sum())
        result['residual_clip_value'] = float(r['r_max'])
    residual.to_csv(OUT / 'exp47_residual_context_evidence.csv')
    result['verifier'] = {}
    funnel = []
    for split in ['cal', 'eval']:
        with np.load(DIAG / f'baseline_{split}_dense_candidate_scores.npz') as s:
            y, glob = array(split, 'y'), array(split, 'p0').argmax(1)
            candidate, q, lower = s['candidate'], s['q_hat'], s['g_lower']
            mask = array('cal', 'baseline_mask') if split == 'cal' else np.ones(len(y), bool)
            h, d = (glob != y) & (candidate == y), (glob == y) & (candidate != y)
            hd = mask & (h | d)
            result['verifier'][split] = {
                'evaluation_rows': int(mask.sum()), 'candidate_helpful': int((mask & h).sum()),
                'candidate_harmful': int((mask & d).sum()), 'raw_positive_helpful': int((mask & h & (q > 0)).sum()),
                'raw_positive_harmful': int((mask & d & (q > 0)).sum()),
                'corrected_positive_helpful': int((mask & h & (lower > 0)).sum()),
                'corrected_positive_harmful': int((mask & d & (lower > 0)).sum()),
                'q_hat_auroc_helpful_vs_harmful': roc_auc_score(h[hd], q[hd]),
                'q_hat_helpful_median': float(np.median(q[mask & h])),
                'q_hat_harmful_median': float(np.median(q[mask & d]))}
            for stage, take in [('top1_no_gates', mask), ('raw_positive_diagnostic', mask & (q > 0)),
                                ('corrected_positive_diagnostic', mask & (lower > 0))]:
                funnel.append({'split': split, 'stage': stage, 'taken': int(take.sum()),
                               'helpful': int((take & h).sum()), 'harmful': int((take & d).sum())})
    pd.DataFrame(funnel).to_csv(OUT / 'exp47_verifier_funnel.csv', index=False)
    result['policy_candidates'] = {}
    for key, path in [('EXP31', DIAG / 'baseline_policy_rejection_reasons.csv'), ('EXP39_legacy', ROUTING / '4a_legacy_threshold_grid.csv')]:
        df = pd.read_csv(path)
        df = df[~df.reason.fillna('').str.startswith('selected:')]
        assert not df.feasible.any()
        result['policy_candidates'][key] = {'count': len(df), 'feasible': int(df.feasible.sum()),
            'support_rejections': int(df.reason.fillna('').str.contains('support').sum())}
    summary = pd.read_csv(ROUTING / 'summary.csv')
    result['bank_correctness_oracle'] = summary[summary.arm == 'correctness_oracle'].iloc[0].to_dict()
    result['audit_scope'] = json.loads((DIAG / 'audit_scope.json').read_text())
    (OUT / 'exp47_result_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Audit expert corrections and confusions using saved EXP59 predictions only.

R_k: frozen full-residual membership (uses test truth for diagnosis).
O_k: pre-existing nearest observable centroid membership (features/global probs).
No fitting, probability inference, resampling, threshold choice, or bank selection.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'tabpfn/results/20260928_exp59_residual_oracle_s43'
OUT = RESULTS / 'expert_capability'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confusion(y, pred, classes):
    return np.bincount(y.astype(np.int64) * classes + pred,
                       minlength=classes * classes).reshape(classes, classes)


def metrics(cm):
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = cm.diagonal()
    fp = predicted - tp
    fn = support - tp
    p = np.divide(tp, predicted, out=np.zeros(len(tp), float), where=predicted > 0)
    r = np.divide(tp, support, out=np.zeros(len(tp), float), where=support > 0)
    f = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp), float), where=2 * tp + fp + fn > 0)
    return [dict(support=int(support[i]), predicted=int(predicted[i]), TP=int(tp[i]),
                 FP=int(fp[i]), FN=int(fn[i]), precision=float(p[i]), recall=float(r[i]), f1=float(f[i]))
            for i in range(len(tp))]


def evaluate(y, g, e, names, mask):
    y, g, e = y[mask], g[mask], e[mask]
    C = len(names)
    gc, ec = confusion(y, g, C), confusion(y, e, C)
    fixed, harmed = (g != y) & (e == y), (g == y) & (e != y)
    # Mutually exclusive paired outcomes must conserve the evaluated population.
    both_correct = int(((g == y) & (e == y)).sum())
    both_wrong = int(((g != y) & (e != y)).sum())
    assert both_correct + both_wrong + fixed.sum() + harmed.sum() == len(y)
    assert ec.trace() - gc.trace() == fixed.sum() - harmed.sum()
    gm, em = metrics(gc), metrics(ec)
    classes = []
    for c, name in enumerate(names):
        sources = [dict(true_class=names[j], rows=int(ec[j].sum()),
                        expert=int(ec[j, c]), global_count=int(gc[j, c]),
                        new_harm=int(((y == j) & (e == c) & (g == y)).sum()))
                   for j in range(C) if j != c and ec[j, c] > 0]
        sources.sort(key=lambda x: (-x['expert'], x['true_class']))
        classes.append(dict(name=name, negative_rows=int(len(y) - ec[c].sum()),
                            global_metrics=gm[c], expert_metrics=em[c],
                            fixed=int((fixed & (y == c)).sum()),
                            harmed=int((harmed & (y == c)).sum()),
                            new_fp_harm=int((harmed & (e == c)).sum()), fp_sources=sources))
    pairs = []
    for a in range(C):
        for b in range(C):
            if a != b and (ec[a, b] or gc[a, b]):
                pairs.append(dict(true_class=names[a], predicted_class=names[b],
                                  rows=int(ec[a].sum()), expert=int(ec[a, b]),
                                  global_count=int(gc[a, b]),
                                  new_harm=int(((y == a) & (e == b) & (g == y)).sum())))
    pairs.sort(key=lambda x: (-x['expert'], x['true_class'], x['predicted_class']))
    return dict(rows=len(y), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                both_correct=both_correct, both_wrong=both_wrong,
                global_correct=int(gc.trace()), expert_correct=int(ec.trace()),
                classes=classes, confusion_pairs=pairs,
                global_confusion=gc.tolist(), expert_confusion=ec.tolist())


def analyze():
    OUT.mkdir(exist_ok=True)
    report = dict(date='2026-09-29', seed=43, K=4, arm='designed',
                  extra_inference=False, extra_training=False,
                  selection='All 4 experts and all classes; no test-based expert/class selection',
                  scopes={
                      'residual': 'Saved full residual argmin membership R_k; label-assisted diagnostic',
                      'observable': 'Saved fixed_regions.npz/test O_k; nearest frozen centroid in scaled z,p_global only; no new threshold or tuning'},
                  interpretation='Observable membership is one fixed input-similarity probe, not the trained scorer/verifier policy. Confusion sources ranked on test are descriptive, not a training selection rule.',
                  datasets={}, source_sha256={}, checks={})
    flat_classes, flat_summary, flat_pairs = [], [], []
    for ds in ['cic2018', 'toniot']:
        p = RESULTS / f'{ds}_s43'
        source = p / 'fresh_bank'
        design = json.loads((source / 'design.json').read_text())
        names = design['class_names']
        y = np.load(source / 'evaluation_identity.npz')['y']
        g = np.load(source / 'predictions/global_raw.npy')
        residual = np.load(p / 'residual_test_regions.npy')
        observable = np.load(source / 'fixed_regions.npz')['test']
        recorded = pd.read_csv(p / 'region_per_class.csv').query('arm == "designed"')
        full_recorded = pd.read_csv(p / 'per_class.csv').set_index(['model', 'class'])
        assert y.shape == g.shape == residual.shape == observable.shape
        assert set(residual) == set(observable) == set(range(4))
        files = [source / 'design.json', source / 'evaluation_identity.npz',
                 source / 'predictions/global_raw.npy', p / 'residual_test_regions.npy',
                 source / 'fixed_regions.npz']
        experts = []
        assembled = {key: np.empty_like(g) for key in ['residual', 'observable']}
        for k in range(1, 5):
            pred_path = source / f'predictions/designed_e{k}_raw.npy'
            e = np.load(pred_path)
            files.append(pred_path)
            for c, m in zip(names, metrics(confusion(y, e, len(names)))):
                ref = full_recorded.loc[(f'designed_e{k}', c)]
                for key in ['TP', 'FP', 'FN', 'support']:
                    assert m[key] == ref[key]
                np.testing.assert_allclose(m['f1'], ref.f1, atol=1e-12)
            views = {}
            for scope, membership in [('residual', residual), ('observable', observable)]:
                mask = membership == k - 1
                views[scope] = q = evaluate(y, g, e, names, mask)
                assembled[scope][mask] = e[mask]
                flat_summary.append(dict(dataset=ds, expert=k, scope=scope,
                                         **{key: q[key] for key in ['rows', 'fixed', 'harmed', 'global_correct', 'expert_correct']}))
                for row in q['classes']:
                    if scope == 'residual':
                        for model, prefix in [('global', 'global_metrics'), (f'e{k}', 'expert_metrics')]:
                            ref = recorded[(recorded.region == k) & (recorded['class'] == row['name']) & (recorded.model == model)].iloc[0]
                            for key in ['TP', 'FP', 'FN', 'support']:
                                assert row[prefix][key] == ref[key]
                            np.testing.assert_allclose(row[prefix]['f1'], ref.f1, atol=1e-12)
                    flat_classes.append(dict(dataset=ds, expert=k, scope=scope, class_name=row['name'],
                                             negative_rows=row['negative_rows'], fixed=row['fixed'], harmed=row['harmed'],
                                             new_fp_harm=row['new_fp_harm'],
                                             **{'global_' + a: b for a, b in row['global_metrics'].items()},
                                             **{'expert_' + a: b for a, b in row['expert_metrics'].items()}))
                for row in q['confusion_pairs']:
                    flat_pairs.append(dict(dataset=ds, expert_id=k, scope=scope, **row))
            experts.append(dict(expert=k, **views))
        for scope, path in [('residual', p / 'designed_residual_oracle_pred.npy'),
                            ('observable', source / 'predictions/designed_fixed_region.npy')]:
            assert np.array_equal(assembled[scope], np.load(path))
        report['datasets'][ds] = dict(class_names=names, rows=len(y), experts=experts)
        for path in files:
            report['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    report['checks'] = dict(raw_counts_match_existing_full_and_residual_results=True,
                            paired_outcomes_conserve_rows_and_correct_count=True,
                            assembled_views_match_saved_residual_and_input_assignment_predictions=True,
                            experts=8, expert_scope_class_rows=len(flat_classes))
    (OUT / 'capability.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    pd.DataFrame(flat_classes).to_csv(OUT / 'class_metrics.csv', index=False)
    pd.DataFrame(flat_summary).to_csv(OUT / 'expert_summary.csv', index=False)
    pd.DataFrame(flat_pairs).to_csv(OUT / 'confusion_pairs.csv', index=False)
    print(json.dumps(report['checks'], ensure_ascii=False))
    print(pd.DataFrame(flat_summary).to_string(index=False))


if __name__ == '__main__':
    analyze()

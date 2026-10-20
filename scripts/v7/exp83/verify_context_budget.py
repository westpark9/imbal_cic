#!/usr/bin/env python3
"""Independently verify released IDs, buffer histories, and full-test scores."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)

import argparse
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from exp61_utils import write, sha
from context_budget_sweep import BUDGETS, POLICIES, SOURCE, read, case_source
from finetune_final import PREVIOUS


def verify(run, partial=False):
    cfg = read(run / 'config.json')
    result = read(run / 'results.json')
    expected = {(d, n, m) for d in cfg['datasets'] for n in BUDGETS
                for m in ['xgb', 'pfn_accumulate', 'pfn_fifo', 'pfn_random', 'pfn_similar']}
    actual = {(j['dataset'], j['new_per_class'], j['method']) for j in result['jobs']}
    assert len(actual) == len(result['jobs'])
    if not partial:
        assert result['state'] == 'complete' and len(result['jobs']) == 60 and actual == expected
    else:
        assert actual <= expected
    assert read(run / 'preflight.json')['passed']
    checked = []
    for ds in cfg['datasets']:
        metadata = read(run / 'data' / ds / 'manifest.json')
        base = Path(metadata['source'])
        names = metadata['class_names']
        C = len(names)
        y = np.load(base / 'test_y.npy', mmap_mode='r')
        train_y = np.load(base / 'train_y.npy', mmap_mode='r')
        train_ids = np.load(base / 'train_ids.npy', mmap_mode='r')
        test_ids = np.load(base / 'test_ids.npy', mmap_mode='r')
        order = np.load(run / 'data' / ds / 'query_order.npy', mmap_mode='r')
        np.testing.assert_array_equal(order, np.random.default_rng(43).permutation(len(y)))
        assert metadata['query_order_sha256'] == sha(run / 'data' / ds / 'query_order.npy')
        frozen = read(SOURCE / 'jobs' / f'{ds}_pfn_pretrained' / 'results.json')['fits'][-1]['weights_sha256']
        initial = read(SOURCE / 'data' / ds / 'plans.json')[0]['incoming']
        previous = None
        for n in BUDGETS:
            data = run / f'new_{n}' / 'data' / ds
            plans = read(data / 'plans.json')
            meta = read(data / 'manifest.json')
            assert meta['plans_sha256'] == sha(data / 'plans.json')
            assert meta['source_plans_sha256'] == sha(meta['source_plans'])
            assert plans[0]['incoming'] == initial
            for stage, plan in enumerate(plans):
                assert plan['released'] == sum((p['incoming'] for p in plans[:stage + 1]), [])
                assert len(set(plan['released'])) == len(plan['released'])
                if stage:
                    labels, counts = np.unique(train_y[plan['incoming']], return_counts=True)
                    assert all(counts == n)
                    assert {names[c] for c in labels} == set(plan['new_classes'])
                if previous:
                    assert set(previous[stage]['incoming']) <= set(plan['incoming'])
            assert len(plans[-1]['released']) == 1000 + n * (4 if ds == 'cic2018' else 7)
            assert not np.intersect1d(train_ids[plans[-1]['released']], test_ids).size
            previous = plans
            for j in (j for j in result['jobs'] if j['dataset'] == ds and j['new_per_class'] == n):
                method = j['method']
                pred = np.load(j['predictions_path'], mmap_mode='r')
                assert pred.shape == y.shape and np.all((pred >= 0) & (pred < C))
                assert j['evaluation'] == 'final_full_test' and j['test_rows'] == len(y)
                assert j['test_order'] == 'seed43_mixed'
                assert j['query_order_sha256'] == metadata['query_order_sha256']
                assert j['initial_finetuning'] is False and j['initial_finetuning_seconds'] == 0
                cm = confusion_matrix(y, pred, labels=np.arange(C))
                np.testing.assert_array_equal(cm, j['metrics']['confusion'])
                den = cm.sum(0) + cm.sum(1)
                f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(C), where=den != 0)
                assert abs(f1.mean() - j['metrics']['macro_f1']) < 1e-12
                benign = names.index('benign')
                fpr = 1 - cm[benign, benign] / cm[benign].sum()
                assert abs(fpr - j['metrics']['benign_false_positive_rate']) < 1e-12
                for c, item in enumerate(j['metrics']['classes']):
                    assert item['name'] == names[c] and item['support'] == cm[c].sum()
                    assert abs(item['f1'] - f1[c]) < 1e-12
                assert abs(j['update_seconds'] - sum(f['update_seconds'] for f in j['fits'][1:])) < 1e-10
                assert j['label_budget'] == len(plans[-1]['released'])
                assert len(j['fits']) == len(plans)
                evicted = set()
                prior = None
                old_policy = None
                if n == 100 and method in ['pfn_fifo', 'pfn_random', 'pfn_similar']:
                    old_policy = read(PREVIOUS / 'jobs' / f'{ds}_{method}' / 'results.json')
                for stage, fit in enumerate(j['fits']):
                    ids = np.asarray(fit['context_indices'], dtype='int64')
                    removed = np.asarray(fit['removed_indices'], dtype='int64')
                    incoming = np.asarray(plans[stage]['incoming'], dtype='int64')
                    assert len(ids) == len(set(ids)) == fit['context_rows']
                    assert set(ids) <= set(plans[stage]['released'])
                    assert fit['finetuning_seconds'] == 0
                    assert abs(fit['update_seconds'] - sum(fit[k] for k in ['selection_seconds', 'construct_seconds', 'fit_seconds'])) < 1e-10
                    if method.startswith('pfn_'):
                        assert fit['weights_sha256'] == frozen
                    if old_policy is not None:
                        assert fit['context_indices'] == old_policy['fits'][stage]['context_indices']
                        assert fit['removed_indices'] == old_policy['fits'][stage]['removed_indices']
                    assert set(train_y[ids]) == set(train_y[plans[stage]['released']])
                    if method in ['xgb', 'pfn_accumulate']:
                        assert ids.tolist() == plans[stage]['released'] and len(removed) == 0
                    else:
                        assert len(ids) == 1000
                        assert not set(ids) & evicted
                        if not stage:
                            assert ids.tolist() == initial and len(removed) == 0
                        else:
                            assert len(removed) == len(incoming)
                            assert set(removed) <= set(prior)
                            assert set(ids) == (set(prior) - set(removed)) | set(incoming)
                            assert ids[-len(incoming):].tolist() == incoming.tolist()
                            for c in set(train_y[prior]):
                                assert np.count_nonzero(train_y[ids] == c) >= 1
                        evicted.update(removed.tolist())
                        np.testing.assert_array_equal(np.bincount(train_y[ids], minlength=C), fit['context_class_counts'])
                    prior = ids
                np.testing.assert_array_equal(np.bincount(train_y[prior], minlength=C), j['final_context_counts'])
                assert j['context_rows'] == len(prior)
                assert j['feature_disjoint_diagnostic']['rows'] == len(y) - meta['test_feature_overlap_with_released_rows']
                if j['cost_reused']:
                    assert sha(j['cost_source']) == j['cost_source_sha256']
                    assert read(j['cost_source'])['update_seconds'] == j['update_seconds']
                marker = read(run / f'new_{n}' / 'jobs' / f'{ds}_{method}' / 'COMPLETE.json')
                assert marker['predictions_sha256'] == sha(j['predictions_path'])
                checked.append(dict(dataset=ds, new_per_class=n, method=method, macro_f1=float(f1.mean()),
                                    predictions_sha256=marker['predictions_sha256']))
    for source, digest in read(run / 'source_hashes.json').items():
        assert sha(source) == digest
    report = dict(passed=True, complete=not partial, checked_jobs=len(checked), jobs=checked,
                  independent_full_test_metrics=True, identical_released_ids=True, nested_budgets=True,
                  fixed_buffer_capacity_verified=True, no_future_or_evicted_rows=True,
                  all_seen_classes_preserved=True, frozen_weights=True, cost_scope_verified=True,
                  budget_100_matches_original_context_policy=True,
                  mixed_query_order_verified=True, source_hashes_unchanged=True)
    write(run / ('partial_verification.json' if partial else 'verification.json'), report)
    print(json.dumps(dict(passed=True, complete=not partial, checked_jobs=len(checked))))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    verify(a.run, a.partial)

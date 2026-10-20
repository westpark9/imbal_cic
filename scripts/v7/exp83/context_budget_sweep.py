#!/usr/bin/env python3
"""Frozen TabPFN: cumulative context versus three bounded-buffer policies."""
from pathlib import Path
import sys

_root = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)

import argparse
import copy
import gc
import importlib.metadata
import json
import shutil
import time
import numpy as np
from exp61_utils import write, sha
from finetune_final import CHECKPOINT, weights_digest
from noft_dist_budget import load_source, SOURCE
from replay_models import make_base, timed
from replay_protocol import confusion_metrics
from run_context_policies import replace_context

SMALLER = ROOT / 'tabpfn/results/v7/exp78/20261016_005435_noft_dist_new100_50_20_10_s43_52588487'
EXPANDED = ROOT / 'tabpfn/results/v7/exp80/20261016_022614_new200_400_noft_s43_c6f6ea24'
MIXED = ROOT / 'tabpfn/results/v7/exp81/20261016_025421_mixed_query_final_s43_b6ba1d34'
BUDGETS = [10, 20, 50, 100, 200, 400]
POLICIES = ['fifo', 'random', 'similar']


def read(path):
    return json.loads(Path(path).read_text())


def progress(out, phase, **values):
    row = dict(phase=phase, updated_epoch=time.time(), **values)
    write(out / 'progress.json', row)
    print(json.dumps(row), flush=True)


def case_source(n):
    return SOURCE if n == 100 else (SMALLER if n < 100 else EXPANDED) / f'new_{n}'


def stripped_plans(path):
    return [{key: copy.deepcopy(plan[key]) for key in ['stage', 'incoming', 'released', 'new_classes']}
            for plan in read(path)]


def simulate(plans, arr, policy, seed=43):
    """Preflight on actual training IDs; never consumes query features or labels."""
    old = None
    history = []
    evicted = set()
    for stage, plan in enumerate(plans):
        incoming = np.asarray(plan['incoming'], dtype='int64')
        if stage == 0:
            ctx, removed = incoming, np.array([], dtype='int64')
        else:
            ctx, removed = replace_context(old, incoming, arr['train_y'], arr['train_X'], policy, seed + stage * 1009)
            assert len(removed) == len(incoming)
            assert set(removed) <= set(old)
            assert set(ctx) == (set(old) - set(removed)) | set(incoming)
        assert len(ctx) == len(set(ctx)) == 1000
        assert not (set(ctx) & evicted)
        assert set(ctx) <= set(plan['released'])
        assert set(arr['train_y'][ctx]) == set(arr['train_y'][plan['released']])
        evicted.update(removed.tolist())
        history.append(dict(context_indices=ctx.tolist(), removed_indices=removed.tolist()))
        old = ctx
    return history


def prepare(out, cfg):
    checks = []
    for ds in cfg['datasets']:
        meta, arr, _ = load_source(ds)
        root = out / 'data' / ds
        root.mkdir(parents=True, exist_ok=True)
        query_source = MIXED / 'data' / ds / 'query_order.npy'
        order = np.load(query_source, mmap_mode='r')
        np.testing.assert_array_equal(order, np.random.default_rng(43).permutation(len(arr['test_y'])))
        shutil.copy2(query_source, root / 'query_order.npy')
        write(root / 'manifest.json', dict(source=meta['source'], class_names=meta['class_names'],
              source_manifest=str(SOURCE / 'data' / ds / 'manifest.json'),
              test_rows=len(order), query_order_sha256=sha(query_source), query_order_labels_used=False))
        previous = None
        for n in BUDGETS:
            source = case_source(n)
            original_path = source / 'data' / ds / 'plans.json'
            plans = stripped_plans(original_path)
            assert plans[0]['incoming'] == stripped_plans(SOURCE / 'data' / ds / 'plans.json')[0]['incoming']
            for stage, plan in enumerate(plans):
                assert plan['stage'] == stage
                assert plan['released'] == sum([p['incoming'] for p in plans[:stage + 1]], [])
                assert len(set(plan['released'])) == len(plan['released'])
                if stage:
                    labels, counts = np.unique(arr['train_y'][plan['incoming']], return_counts=True)
                    assert all(counts == n)
                    assert {meta['class_names'][c] for c in labels} == set(plan['new_classes'])
                if previous:
                    assert set(previous[stage]['incoming']) <= set(plan['incoming'])
            assert len(plans[-1]['released']) == 1000 + n * (4 if ds == 'cic2018' else 7)
            assert not np.intersect1d(arr['train_ids'][plans[-1]['released']], arr['test_ids']).size
            dest = out / f'new_{n}' / 'data' / ds
            write(dest / 'plans.json', plans)
            for policy in POLICIES:
                history = simulate(plans, arr, policy)
                write(dest / f'{policy}_preflight.json', history)
            write(dest / 'manifest.json', dict(dataset=ds, new_per_class=n, initial_rows=1000,
                  final_released_rows=len(plans[-1]['released']), source_plans=str(original_path),
                  source_plans_sha256=sha(original_path), plans_sha256=sha(dest / 'plans.json'),
                  validation_used=False, train_test_row_ids_disjoint=True,
                  test_feature_overlap_with_released_rows=int(np.isin(arr['test_hash'], arr['train_hash'][plans[-1]['released']]).sum())))
            previous = plans
            checks.append(dict(dataset=ds, new_per_class=n, capacity=1000, stages=len(plans), policies=POLICIES))
    write(out / 'preflight.json', dict(passed=True, checks=checks, nested_budgets=True,
          class_coverage_protected=True, no_evicted_rows_reintroduced=True))


def evaluate(out, job, cfg, ds, n, method, model, arr, names, classes, order):
    total = len(order)
    pred = np.lib.format.open_memmap(job / 'predictions.npy', mode='w+', dtype='int16', shape=(total,))
    pred[:] = -1
    services = []
    for i, start in enumerate(range(0, total, cfg['batch'])):
        ids = order[start:start + cfg['batch']]
        p, elapsed = timed(lambda: model.predict_proba(np.asarray(arr['test_X'][ids])))
        assert p.shape == (len(ids), len(classes)) and np.isfinite(p).all()
        assert np.allclose(p.sum(1), 1, atol=1e-3)
        pred[ids] = classes[p.argmax(1)]
        services.append([start, start + len(ids), elapsed])
        if i % 80 == 0 or start + len(ids) == total:
            done = start + len(ids)
            progress(out, 'final_full_test', dataset=ds, new_per_class=n, method=method,
                     rows_done=done, rows_total=total, job_remaining_seconds=(total - done) * sum(s[2] for s in services) / done)
    pred.flush()
    assert np.all(pred >= 0)
    np.save(job / 'service_times.npy', np.asarray(services))
    return pred, sum(s[2] for s in services)


def complete(job, result):
    write(job / 'results.json', result)
    write(job / 'COMPLETE.json', dict(complete=True, predictions_sha256=sha(result['predictions_path'])))
    return result


def baseline(out, cfg, ds, n, method, arr, names, plans, order):
    """Preserve already measured cumulative update costs; align final inference."""
    import torch
    from tabpfn import TabPFNClassifier
    job = out / f'new_{n}' / 'jobs' / f'{ds}_{method}'
    job.mkdir(parents=True, exist_ok=True)
    if (job / 'COMPLETE.json').exists():
        return read(job / 'results.json')
    old_method = 'pfn_pretrained' if method == 'pfn_accumulate' else 'xgb'
    old_job = case_source(n) / 'jobs' / f'{ds}_{old_method}'
    result = read(old_job / 'results.json')
    for stage, fit in enumerate(result['fits']):
        assert fit['context_indices'] == plans[stage]['released']
        assert fit['finetuning_seconds'] == 0
    old_pred = np.load(old_job / 'predictions.npy', mmap_mode='r')
    classes = np.unique(arr['train_y'][plans[-1]['released']])
    if method == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(old_job / 'final_model.ubj')
        model.set_params(device='cuda:0', n_jobs=cfg['threads'])
        ids = order[:cfg['batch']]
        assert np.array_equal(classes[model.predict_proba(np.asarray(arr['test_X'][ids])).argmax(1)], old_pred[ids])
        pred = old_pred
        prediction_path = old_job / 'predictions.npy'
        inference_seconds = None
        reuse_reason = 'Saved row-wise XGB predictions; exact mixed-batch audit on 8192 rows'
        del model
    elif n == 100:
        mixed_job = MIXED / 'jobs' / f'small_{ds}_new_100_tabpfn'
        mixed = read(mixed_job / 'results.json')
        assert mixed['query_order_sha256'] == sha(out / 'data' / ds / 'query_order.npy')
        assert mixed['full_reinference']
        pred = np.load(mixed['predictions_path'], mmap_mode='r')
        prediction_path = Path(mixed['predictions_path'])
        inference_seconds = mixed['serving_seconds']
        reuse_reason = 'Existing complete mixed-order inference in EXP81'
    else:
        model = TabPFNClassifier.load_from_fit_state(old_job / 'final_model.tabpfn_fit', device='cuda:0')
        assert weights_digest(model.models_[0].state_dict()) == result['fits'][-1]['weights_sha256']
        np.testing.assert_array_equal(model.classes_, np.arange(len(classes)))
        pred, inference_seconds = evaluate(out, job, cfg, ds, n, method, model, arr, names, classes, order)
        prediction_path = job / 'predictions.npy'
        assert weights_digest(model.models_[0].state_dict()) == result['fits'][-1]['weights_sha256']
        reuse_reason = 'Saved cumulative fit state; complete mixed-order inference rerun'
        del model
    gc.collect()
    torch.cuda.empty_cache()
    result.update(dataset=ds, method=method, new_per_class=n, initial_finetuning=False,
                  test_order='seed43_mixed', query_order_sha256=sha(out / 'data' / ds / 'query_order.npy'),
                  prediction_storage_order='original', predictions_path=str(prediction_path),
                  metrics=confusion_metrics(arr['test_y'], np.asarray(pred), names),
                  context_rows=len(plans[-1]['released']), context_capacity=None,
                  final_context_counts=np.bincount(arr['train_y'][plans[-1]['released']], minlength=len(names)).tolist(),
                  cost_reused=True, cost_source=str(old_job / 'results.json'),
                  cost_source_sha256=sha(old_job / 'results.json'),
                  source_predictions_sha256=sha(old_job / 'predictions.npy'),
                  evaluation_inference_seconds=inference_seconds,
                  evaluation_reuse_reason=reuse_reason,
                  original_order_prediction_mismatches=int(np.count_nonzero(pred != old_pred)),
                  note='Cumulative baseline: original measured update costs, matched final evaluation, no IDS weight fine-tuning')
    mask = ~np.isin(arr['test_hash'], arr['train_hash'][plans[-1]['released']])
    result['feature_disjoint_diagnostic'] = confusion_metrics(arr['test_y'][mask], np.asarray(pred)[mask], names)
    return complete(job, result)


def buffer_job(out, cfg, ds, n, policy, arr, names, plans, order):
    import torch
    method = f'pfn_{policy}'
    job = out / f'new_{n}' / 'jobs' / f'{ds}_{method}'
    job.mkdir(parents=True, exist_ok=True)
    if (job / 'COMPLETE.json').exists():
        return read(job / 'results.json')
    model = None
    fits = []
    old = None
    expected = read(case_source(n) / 'jobs' / f'{ds}_pfn_pretrained' / 'results.json')['fits'][-1]['weights_sha256']
    preflight = read(out / f'new_{n}' / 'data' / ds / f'{policy}_preflight.json')
    for stage, plan in enumerate(plans):
        progress(out, 'context_update', dataset=ds, new_per_class=n, method=method, stage=stage)
        start = time.perf_counter()
        incoming = np.asarray(plan['incoming'], dtype='int64')
        if stage == 0:
            ctx, removed = incoming, np.array([], dtype='int64')
        else:
            ctx, removed = replace_context(old, incoming, arr['train_y'], arr['train_X'], policy, cfg['seed'] + stage * 1009)
        selection = time.perf_counter() - start
        classes = np.unique(arr['train_y'][ctx])
        labels = np.searchsorted(classes, arr['train_y'][ctx])
        construct = 0.
        if model is None:
            model, construct = timed(lambda: make_base('tabpfn', cfg, len(classes)))
        _, fit_seconds = timed(lambda: model.fit(arr['train_X'][ctx], labels))
        digest = weights_digest(model.models_[0].state_dict())
        assert digest == expected
        assert ctx.tolist() == preflight[stage]['context_indices']
        assert removed.tolist() == preflight[stage]['removed_indices']
        np.testing.assert_array_equal(model.classes_, np.arange(len(classes)))
        info = dict(stage=stage, context_indices=ctx.tolist(), removed_indices=removed.tolist(),
                    context_rows=len(ctx), released_rows=len(plan['released']),
                    context_class_counts=np.bincount(arr['train_y'][ctx], minlength=len(names)).tolist(),
                    removed_class_counts=np.bincount(arr['train_y'][removed], minlength=len(names)).tolist(),
                    classes=[names[c] for c in classes], selection_seconds=selection,
                    construct_seconds=construct, fit_seconds=fit_seconds, finetuning_seconds=0.,
                    update_seconds=selection + construct + fit_seconds, weights_sha256=digest)
        write(job / f'stage_{stage}' / 'fit_metrics.json', info)
        fits.append(info)
        old = ctx
    model.save_fit_state(job / 'final_model.tabpfn_fit')
    pred, inference = evaluate(out, job, cfg, ds, n, method, model, arr, names, classes, order)
    assert weights_digest(model.models_[0].state_dict()) == expected
    mask = ~np.isin(arr['test_hash'], arr['train_hash'][plans[-1]['released']])
    result = dict(dataset=ds, method=method, new_per_class=n, seed=cfg['seed'],
                  evaluation='final_full_test', final_stage=len(plans) - 1, test_full=True,
                  test_rows=len(order), test_order='seed43_mixed', batch=cfg['batch'],
                  query_order_sha256=sha(out / 'data' / ds / 'query_order.npy'),
                  prediction_storage_order='original', predictions_path=str(job / 'predictions.npy'),
                  metrics=confusion_metrics(arr['test_y'], np.asarray(pred), names), fits=fits,
                  feature_disjoint_diagnostic=confusion_metrics(arr['test_y'][mask], np.asarray(pred)[mask], names),
                  initial_seconds=fits[0]['update_seconds'], initial_finetuning=False,
                  initial_finetuning_seconds=0., update_seconds=sum(f['update_seconds'] for f in fits[1:]),
                  total_preparation_seconds=sum(f['update_seconds'] for f in fits),
                  serving_seconds=inference, evaluation_inference_seconds=inference,
                  cost_reused=False, label_budget=len(plans[-1]['released']),
                  context_rows=len(ctx), context_capacity=1000,
                  final_context_counts=fits[-1]['context_class_counts'], checkpoint=str(CHECKPOINT),
                  note='Frozen pretrained v3; 1000-row buffer; at least one example of every seen class; final-model full mixed test; no test tuning')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return complete(job, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(protocol='exp83.1', seed=43, threads=8, batch=8192,
               datasets=['cic2018', 'toniot'], new_budgets=BUDGETS, buffer_capacity=1000,
               policies=POLICIES, checkpoint=str(CHECKPOINT), seed_repetitions=1,
               initial_finetuning=False, html_publication=False, slides_publication=False,
               cost_scope='selection + construction + fit, stages after initial; inference excluded')
    write(out / 'config.json', cfg)
    deps = [Path(__file__), ROOT / 'configs/experiments/exp83_protocol.json',
            ROOT / 'scripts/v7/exp76/run_context_policies.py', ROOT / 'scripts/v7/exp77/finetune_final.py',
            ROOT / 'scripts/v7/exp78/noft_dist_budget.py', ROOT / 'scripts/v7/exp72/replay_models.py',
            ROOT / 'scripts/v7/exp72/replay_protocol.py', ROOT / 'scripts/v7/exp71/exp71_bounded_updates.py',
            ROOT / 'scripts/common/exp61_utils.py', ROOT / 'scripts/common/experiment_paths.py']
    hashes = {}
    for source in deps:
        dest = out / 'source_dependencies' / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[str(source)] = sha(source)
    hashes[str(CHECKPOINT)] = sha(CHECKPOINT)
    write(out / 'source_hashes.json', hashes)
    write(out / 'environment.json', {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'scikit-learn', 'tabpfn', 'xgboost']})
    progress(out, 'preflight')
    prepare(out, cfg)
    import torch
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(cfg['seed'])
    jobs = []
    started = time.time()
    # Show the shared 100-row setting first, then sweep the remaining budgets.
    for n in [100, 10, 20, 50, 200, 400]:
        for ds in cfg['datasets']:
            meta, arr, _ = load_source(ds)
            names = meta['class_names']
            plans = read(out / f'new_{n}' / 'data' / ds / 'plans.json')
            order = np.load(out / 'data' / ds / 'query_order.npy', mmap_mode='r')
            for method in ['xgb', 'pfn_accumulate', 'pfn_fifo', 'pfn_random', 'pfn_similar']:
                if method in ['xgb', 'pfn_accumulate']:
                    result = baseline(out, cfg, ds, n, method, arr, names, plans, order)
                else:
                    result = buffer_job(out, cfg, ds, n, method.removeprefix('pfn_'), arr, names, plans, order)
                jobs.append(result)
                write(out / 'results.json', dict(state='running', jobs=jobs, expected_jobs=60))
                progress(out, 'job_complete', dataset=ds, new_per_class=n, method=method,
                         completed_jobs=len(jobs), expected_jobs=60, elapsed_seconds=time.time() - started,
                         macro_f1=result['metrics']['macro_f1'], update_seconds=result['update_seconds'])
    assert all(sha(path) == digest for path, digest in hashes.items())
    write(out / 'results.json', dict(state='complete', jobs=jobs, expected_jobs=60))
    progress(out, 'complete', completed_jobs=len(jobs), expected_jobs=60, elapsed_seconds=time.time() - started)


if __name__ == '__main__':
    main()

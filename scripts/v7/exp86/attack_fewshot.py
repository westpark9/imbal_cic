"""EXP86: paired attack-only few-shot fits; full-test and attack-only metrics."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)

import argparse
import gc
import importlib.metadata
import json
import pickle
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from exp61_utils import write, sha
from exp69_class_arrival import CLEAN
from finetune_final import CHECKPOINT, weights_digest
from noft_dist_budget import SOURCE
from replay_models import make_base, timed, sync
from replay_protocol import confusion_metrics

BUDGETS = [1, 2, 5, 10, 20]
SEEDS = [43, 44, 45, 46, 47]
DATASETS = ['cic2018', 'toniot']


def read(p):
    return json.loads(Path(p).read_text())


def progress(out, phase, **kw):
    value = dict(phase=phase, updated_epoch=time.time(), **kw)
    write(out/'progress.json', value)
    print(json.dumps(value), flush=True)


def prepare(out):
    if (out/'prepared.json').exists():
        return
    source = ROOT/'data/nfv3_energy_suite_uncapped_scenarios.pkl'
    progress(out, 'loading_full_source', path=str(source))
    digest = sha(source)
    with source.open('rb') as f:
        suite = pickle.load(f)
    families = np.asarray(suite['families'])
    checks = []
    for ds in DATASETS:
        clean = ROOT/'data/derived'/CLEAN[ds]
        clean_meta = read(clean/'manifest.json')
        assert clean_meta['source']['sha256'] == digest
        old = read(SOURCE/'data'/ds/'manifest.json')
        base = Path(old['source'])
        names = old['class_names']
        train = np.load(clean/'train_idx.npy')
        test = np.load(clean/'test_idx.npy')
        np.testing.assert_array_equal(test, np.load(base/'test_ids.npy'))
        source_ids = np.load(clean/'source_idx.npy', mmap_mode='r')
        hashes = np.load(clean/'vector_hash.npy', mmap_mode='r')
        dest = out/'data'/ds
        dest.mkdir(parents=True, exist_ok=True)
        order = np.random.default_rng(43).permutation(len(test))
        np.save(dest/'query_order.npy', order)
        candidates = {c: train[families[train] == name] for c, name in enumerate(names) if name != 'benign'}
        for seed in SEEDS:
            picked = [np.random.default_rng(seed+c*1009).choice(pool, 20, replace=False) for c, pool in candidates.items()]
            ids = np.concatenate(picked).astype('int64')
            labels = np.concatenate([np.full(20, c, dtype='int16') for c in candidates])
            assert len(ids) == len(np.unique(ids)) and not np.isin(ids, test).any()
            np.testing.assert_array_equal(families[ids], np.array(names)[labels])
            d = dest/f'seed_{seed}';d.mkdir(exist_ok=True)
            X = np.nan_to_num(np.asarray(suite['X'][ids], dtype='float32'))
            assert np.isfinite(X).all()
            for key, arr in dict(X=X, y=labels, source_ids=ids, hashes=hashes[np.searchsorted(source_ids, ids)]).items():
                np.save(d/f'{key}.npy', arr)
            for n in BUDGETS:
                idx = np.concatenate([np.arange(j*20, j*20+n) for j in range(len(candidates))])
                assert np.all(np.unique(labels[idx], return_counts=True)[1] == n)
                np.save(d/f'n_{n}_indices.npy', idx)
        # Confirm the reused test tensor is the same deterministic raw feature transform.
        probe = np.random.default_rng(43).choice(len(test), 256, replace=False)
        np.testing.assert_array_equal(np.load(base/'test_X.npy', mmap_mode='r')[probe],
                                      np.nan_to_num(np.asarray(suite['X'][test[probe]], dtype='float32')))
        meta = dict(dataset=ds, class_names=names, test_base=str(base), source_pickle=str(source),
            source_pickle_sha256=digest, clean_manifest=str(clean/'manifest.json'),
            clean_manifest_sha256=sha(clean/'manifest.json'),
            candidate_counts={names[c]: len(pool) for c, pool in candidates.items()},
            test_counts=np.bincount(np.load(base/'test_y.npy'), minlength=len(names)).tolist(),
            test_rows=len(test), features=int(suite['X'].shape[1]), train_test_ids_disjoint=True,
            sampling='20 rows per class from FULL clean train, nested prefixes at each n; no initial bank',
            preprocessing='Existing row-local zero fill and fixed clipping; no learned full-train scaling. Estimator preprocessing fitted on supplied rows only.',
            query_order_sha256=sha(dest/'query_order.npy'),
            split_caveat='Previously inspected conflict-filtered fixed split; identical feature vectors can cross train/test. No new split or chronological replay.')
        write(dest/'manifest.json', meta)
        checks.append(meta)
    del suite, families
    gc.collect()
    write(out/'prepared.json', dict(passed=True, datasets=checks, nested=True, benign_train_rows=0))


def load_data(out, ds, seed):
    meta = read(out/'data'/ds/'manifest.json')
    d = out/'data'/ds/f'seed_{seed}'
    small = {k: np.load(d/f'{k}.npy') for k in ['X', 'y', 'source_ids', 'hashes']}
    base = Path(meta['test_base'])
    test = {k: np.load(base/f'test_{k}.npy', mmap_mode='r') for k in ['X', 'y', 'hash', 'ids']}
    order = np.load(out/'data'/ds/'query_order.npy', mmap_mode='r')
    return meta, small, test, order


def run_job(out, cfg, ds, seed, n, method):
    import torch
    job = out/'jobs'/f'{ds}_s{seed}_n{n}_{method}'
    if (job/'COMPLETE.json').exists():
        result = read(job/'results.json')
        assert sha(job/'predictions.npy') == read(job/'COMPLETE.json')['predictions_sha256']
        return result
    job.mkdir(parents=True, exist_ok=True)
    meta, small, test, order = load_data(out, ds, seed)
    idx = np.load(out/'data'/ds/f'seed_{seed}'/f'n_{n}_indices.npy')
    X = small['X'][idx]; y = small['y'][idx]; classes = np.unique(y)
    local_y = np.searchsorted(classes, y)
    np.save(job/'train_source_ids.npy', small['source_ids'][idx])
    assert len(classes) == len(meta['class_names']) - 1
    assert meta['class_names'].index('benign') not in classes
    settings = dict(cfg, seed=seed)
    torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats()
    progress(out, 'fit', dataset=ds, seed=seed, n=n, method=method)
    model, construct = timed(lambda: make_base('xgb' if method == 'xgb' else 'tabpfn', settings, len(classes)))
    _, fit = timed(lambda: model.fit(X, local_y))
    np.testing.assert_array_equal(model.classes_, np.arange(len(classes)))
    before = weights_digest(model.models_[0].state_dict()) if method == 'pfn' else None
    expected = read(SOURCE/'jobs'/f'{ds}_pfn_pretrained'/'results.json')['fits'][0]['weights_sha256']
    if before is not None:
        assert before == expected
    # A fixed tree configuration can become all stumps in the smallest budgets.
    tree_splits = None
    if method == 'xgb':
        frame = model.get_booster().trees_to_dataframe()
        tree_splits = int((frame.Feature != 'Leaf').sum())
        model.save_model(job/'model.ubj')
    total = len(order)
    pred = np.lib.format.open_memmap(job/'predictions.npy', mode='w+', dtype='int16', shape=(total,))
    pred[:] = -1
    services = []
    for start in range(0, total, cfg['batch']):
        rows = order[start:start+cfg['batch']]
        proba, elapsed = timed(lambda: model.predict_proba(np.asarray(test['X'][rows])))
        assert proba.shape == (len(rows), len(classes)) and np.isfinite(proba).all()
        assert np.allclose(proba.sum(1), 1, atol=1e-3)
        pred[rows] = classes[proba.argmax(1)]
        services.append([start, len(rows), elapsed])
        if start == 0 or start//cfg['batch'] % 100 == 0:
            done = start+len(rows)
            progress(out, 'predict', dataset=ds, seed=seed, n=n, method=method, rows_done=done, rows_total=total,
                     estimated_job_remaining_seconds=(total-done)*sum(x[2] for x in services)/done)
    pred.flush();np.save(job/'service_times.npy', np.asarray(services))
    assert np.isin(pred, classes).all()
    if before is not None:
        assert before == weights_digest(model.models_[0].state_dict())
    names = meta['class_names']; benign = names.index('benign')
    attack = np.asarray(test['y']) != benign
    phase_a = confusion_metrics(test['y'][attack], pred[attack], names, classes)
    phase_b = confusion_metrics(test['y'], np.asarray(pred), names)
    assert phase_b['benign_false_positive_rate'] == 1.
    disjoint = ~np.isin(test['hash'], small['hashes'][idx])
    filtered = attack & disjoint
    diagnostic = confusion_metrics(test['y'][filtered], pred[filtered], names, classes)
    result = dict(dataset=ds, seed=seed, n=n, method=method, context_rows=len(idx), class_names=names,
        train_source_ids=small['source_ids'][idx].tolist(), train_ids_sha256=sha(job/'train_source_ids.npy'),
        train_class_counts=np.bincount(y, minlength=len(names)).tolist(), phase_a=phase_a, phase_b=phase_b,
        feature_disjoint_attack=diagnostic, exact_feature_overlap_test_rows=int((~disjoint).sum()),
        exact_feature_overlap_attack_rows=int((attack & ~disjoint).sum()),
        training_unique_feature_rows=int(len(np.unique(X, axis=0))),
        construct_seconds=construct, fit_seconds=fit, preparation_seconds=construct+fit,
        inference_seconds=sum(x[2] for x in services), test_rows=total,
        weights_sha256=before, weights_unchanged=before is not None,
        initial_ft=False, later_ft=False, xgb_tree_splits=tree_splits,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30, checkpoint=str(CHECKPOINT),
        predictions_path=str(job/'predictions.npy'), query_order_sha256=meta['query_order_sha256'],
        cost_scope='Fresh construction + fit/cache preparation, not sequential update; excludes inference, saving, validation.',
        phase_b_role='Attack-only output space: benign FPR=1 by construction. Diagnostic, not benign detection efficacy.')
    write(job/'results.json', result)
    write(job/'COMPLETE.json', dict(predictions_sha256=sha(job/'predictions.npy')))
    del model, pred
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int)
    args = parser.parse_args();out = args.out.resolve();out.mkdir(parents=True, exist_ok=True)
    cfg = dict(protocol='exp86.1', datasets=DATASETS, seeds=SEEDS, budgets=BUDGETS,
               checkpoint=str(CHECKPOINT), batch=8192, threads=8, expected_jobs=100,
               report_date=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d'),
               authorization='2026-10-21 user: 그래. 야간작업 시작해라.', max_jobs=args.max_jobs)
    write(out/'config.json', cfg)
    for p in [Path(__file__), Path(__file__).with_name('report_fewshot.py'), ROOT/'configs/experiments/exp86_protocol.json',
              ROOT/'scripts/v7/exp71/exp71_bounded_updates.py',ROOT/'scripts/v7/exp72/replay_models.py',
              ROOT/'scripts/v7/exp72/replay_protocol.py',ROOT/'scripts/common/preprocess_nfv3_energy_suite.py']:
        target=out/'source_dependencies'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(out/'environment.json', {k: importlib.metadata.version(k) for k in ['torch','tabpfn','xgboost','numpy','scikit-learn']})
    import torch
    torch.set_num_threads(8)
    jobs=[];state='running';started=time.time()
    try:
        prepare(out)
        torch.cuda.init();sync()
        a=torch.ones((512,512),device='cuda');a=a@a;sync();del a
        write(out/'checkpoint.json', dict(path=str(CHECKPOINT), sha256=sha(CHECKPOINT), gpu=torch.cuda.get_device_name()))
        # First seed at both ends of n provides a representative timing pilot before the remaining sweep.
        schedule=[(ds,seed,n,method) for seed in SEEDS for n in [1,20,2,5,10] for ds in DATASETS for method in ['xgb','pfn']]
        if args.max_jobs:schedule=schedule[:args.max_jobs]
        for ds,seed,n,method in schedule:
            jobs.append(run_job(out,cfg,ds,seed,n,method))
            write(out/'results.json',dict(state=state,jobs=jobs,expected_jobs=len(schedule)))
            progress(out,'job_complete',dataset=ds,seed=seed,n=n,method=method,completed_jobs=len(jobs),expected_jobs=len(schedule),elapsed_seconds=time.time()-started)
            if len(jobs) in [4,8,20,40,60,80]:
                subprocess.run([sys.executable,str(Path(__file__).with_name('report_fewshot.py')),'--run',str(out)],check=True)
        state='pilot_complete' if args.max_jobs else 'complete'
        write(out/'results.json',dict(state=state,jobs=jobs,expected_jobs=len(schedule)))
        progress(out,state,completed_jobs=len(jobs),expected_jobs=len(schedule),elapsed_seconds=time.time()-started)
    except BaseException as exc:
        state='failed';write(out/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc(),completed_jobs=len(jobs)))
        write(out/'results.json',dict(state=state,jobs=jobs,expected_jobs=100))
        raise
    finally:
        rc=subprocess.run([sys.executable,str(Path(__file__).with_name('report_fewshot.py')),'--run',str(out)]).returncode
        if rc and state != 'failed':raise RuntimeError('Automatic report failed; inspect log')


if __name__ == '__main__':
    main()

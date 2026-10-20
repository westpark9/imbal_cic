#!/usr/bin/env python3
"""Paired attack-arrival streams: bounded rehearsal before cumulative replay.

No expert, scorer or verifier. Immutable stream IDs can later be reused with
--history-mode cumulative --stream-root EXISTING_RUN; the default is bounded.
"""
from pathlib import Path
import sys

_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)

import argparse
import csv
import gc
import hashlib
import importlib.metadata as im
import json
import os
import pickle
import shutil
import subprocess
import time
import traceback

import numpy as np
from exp61_utils import Cost, sha, write
from exp69_class_arrival import STAGES, CLEAN, array_digest, encode_labels, metrics, seen_classes, utc

ROOT = repo_root(__file__)
CASES = ['uniform', 'natural']
REFERENCE = ROOT / 'tabpfn/results/v7/exp70/20261002_exp70_class_arrival_100k_s43'


def apportion(total, weights, capacities, minimum=0):
    """Capped proportional quotas; enforce floors only where explicitly requested."""
    weights = np.asarray(weights, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.int64)
    if total < 0 or total > int(capacities.sum()) or np.any(capacities < 0):
        raise ValueError('Infeasible allocation')
    floor = np.minimum(capacities, minimum)
    if floor.sum() > total:
        raise ValueError('Budget cannot preserve requested class coverage')
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError('Invalid allocation weights')
    lo, hi = 0., 1.
    def continuous(scale):
        return np.minimum(capacities, np.maximum(floor, weights * scale))
    while continuous(hi).sum() < total - 1e-8:
        hi *= 2
        if hi > 1e20:
            raise ValueError('Positive quota requested for zero-weight classes')
    for _ in range(90):
        mid = (lo + hi) / 2
        if continuous(mid).sum() < total: lo = mid
        else: hi = mid
    raw = continuous(hi)
    result = np.maximum(floor, np.floor(raw + 1e-10).astype(np.int64))
    left = int(total - result.sum())
    if left < 0:
        raise ValueError('Numerical over-allocation')
    order = np.argsort(-(raw - result), kind='stable')
    for i in order:
        if left and result[i] < capacities[i]:
            result[i] += 1; left -= 1
    if left or result.sum() != total:
        raise ValueError('Allocation did not close')
    return result


def allocations(available, nominal):
    attacks = sorted(c for c in available if c != 'benign')
    uniform = {c: min(nominal, available[c]) for c in attacks}
    budget = sum(uniform.values())
    natural_values = apportion(budget, [available[c] for c in attacks],
                               [available[c] for c in attacks])
    natural = dict(zip(attacks, map(int, natural_values)))
    if min(natural.values()) < 1:
        raise ValueError('Natural allocation omits an introduced class; increase nominal budget explicitly')
    for values in (uniform, natural):
        values['benign'] = budget * 3
        if values['benign'] > available['benign']:
            raise ValueError('Insufficient benign labels')
    return {'uniform': uniform, 'natural': natural}


def select_history(previous_fit, y, names, case, available, cap, seed):
    """Shrink the previous fitted set only: dropped historical rows never return."""
    previous_fit = np.asarray(previous_fit, dtype=np.int64)
    if len(previous_fit) <= cap:
        return previous_fit.copy(), {'active': False, 'floor_adjusted_classes': []}
    present = [c for c in names if np.any(y[previous_fit] == names.index(c))]
    counts = np.array([np.sum(y[previous_fit] == names.index(c)) for c in present])
    weights = np.zeros(len(present), dtype=float)
    attacks = [c for c in present if c != 'benign']
    attack_weight = {c: 1. if case == 'uniform' else float(available[c]) for c in attacks}
    for i, c in enumerate(present):
        weights[i] = .75 if c == 'benign' else .25 * attack_weight[c] / sum(attack_weight.values())
    unfloored = apportion(cap, weights, counts, minimum=0)
    quotas = apportion(cap, weights, counts, minimum=1)
    selected = []
    for c, n in zip(present, quotas):
        candidates = previous_fit[y[previous_fit] == names.index(c)]
        # New random priorities at each shrink; no access to discarded rows.
        rng = np.random.default_rng(seed + names.index(c) * 1009)
        selected.extend(rng.choice(candidates, int(n), replace=False).tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return selected, {'active': True,
                     'floor_adjusted_classes': [c for c, a, b in zip(present, quotas, unfloored) if a > b],
                     'quotas': dict(zip(present, map(int, quotas))),
                     'without_floor_quotas': dict(zip(present, map(int, unfloored)))}


def make_plans(bank_y, names, dataset, allocation, available, seed, cap):
    plans = {}
    benign = np.flatnonzero(bank_y == names.index('benign'))[:allocation['uniform']['benign']]
    slices = np.array_split(benign, len(STAGES[dataset]))
    for case in CASES:
        previous = np.empty(0, dtype=np.int64); delivered = []
        plans[case] = []
        for stage, group in enumerate(STAGES[dataset]):
            incoming = [slices[stage]]
            for c in group:
                if c != 'benign':
                    incoming.append(np.flatnonzero(bank_y == names.index(c))[:allocation[case][c]])
            incoming = np.concatenate(incoming).astype(np.int64)
            history, note = select_history(previous, bank_y, names, case, available, cap, seed + stage * 100003)
            fit = np.concatenate([history, incoming])
            delivered.extend(incoming.tolist())
            if len(np.unique(delivered)) != len(delivered): raise ValueError('Stream repeats a row')
            if set(names[i] for i in np.unique(bank_y[fit])) != set(seen_classes(dataset, stage)):
                raise ValueError('Missing or future fit class')
            plans[case].append(dict(stage=stage, new_classes=[c for c in group if c != 'benign'] if stage else [],
                                    incoming=incoming.tolist(), history=history.tolist(), fit=fit.tolist(),
                                    delivered_rows=len(delivered), memory=note,
                                    fit_class_counts={c: int(np.sum(bank_y[fit] == names.index(c))) for c in seen_classes(dataset, stage)},
                                    incoming_class_counts={c: int(np.sum(bank_y[incoming] == names.index(c))) for c in seen_classes(dataset, stage)}))
            previous = fit
    return plans


def prepare(args):
    out = args.out.resolve()
    if (out / 'prepared.json').exists():
        p = json.loads((out / 'prepared.json').read_text())
        if (p['seed'], p['nominal_attack_labels'], p['history_cap']) != (args.seed, args.nominal, args.history_cap):
            raise ValueError('Preparation configuration mismatch')
        return
    write(out / 'status.json', dict(state='preparing', started=utc(), pid=os.getpid()))
    started = time.perf_counter()
    print('Loading original source for train-only stream preparation', flush=True)
    with args.data.open('rb') as f: suite = pickle.load(f)
    families = np.asarray(suite['families'])
    source_sha = sha(args.data)
    for ds in STAGES:
        clean = ROOT / 'data/derived' / CLEAN[ds]
        clean_meta = json.loads((clean / 'manifest.json').read_text())
        if source_sha != clean_meta['source']['sha256']: raise ValueError('Source mismatch')
        names = clean_meta['class_names']
        splits = {s: np.load(clean / f'{s}_idx.npy') for s in ['train', 'val', 'test']}
        for s in splits:
            if sha(clean / f'{s}_idx.npy') != clean_meta['artifacts_sha256'][f'{s}_idx.npy']:
                raise ValueError('Split changed')
        available = {c: int(np.sum(families[splits['train']] == c)) for c in names}
        alloc = allocations(available, args.nominal)
        picks = []; bank_y = []
        for i, c in enumerate(names):
            candidates = splits['train'][families[splits['train']] == c]
            n = max(a[c] for a in alloc.values())
            chosen = np.random.default_rng(args.seed + i * 1009).choice(candidates, n, replace=False)
            picks.extend(chosen.tolist()); bank_y.extend([i] * n)
        ids = np.asarray(picks, dtype=np.int64); y = np.asarray(bank_y, dtype=np.int16)
        if any(np.intersect1d(ids, splits[s]).size for s in ['val', 'test']):
            raise ValueError('Evaluation rows entered stream')
        dest = out / 'data' / ds; dest.mkdir(parents=True, exist_ok=True)
        source_ids = np.load(clean / 'source_idx.npy', mmap_mode='r')
        hashes = np.load(clean / 'vector_hash.npy', mmap_mode='r')
        for name, arr in {'train_ids': ids, 'train_y': y,
                          'train_X': np.nan_to_num(np.asarray(suite['X'][ids], dtype=np.float32)),
                          'train_hash': hashes[np.searchsorted(source_ids, ids)]}.items():
            np.save(dest / f'{name}.npy', arr)
        # Existing test arrays have complete fixed-test IDs; preserve them byte-for-byte.
        ref = REFERENCE / 'data' / ds
        ref_meta = json.loads((ref / 'manifest.json').read_text())
        if ref_meta['class_names'] != names or not np.array_equal(np.load(ref / 'test_ids.npy'), splits['test']):
            raise ValueError('Reference evaluation population mismatch')
        for name in ['test_X.npy', 'test_y.npy', 'test_ids.npy', 'test_hash.npy']:
            if sha(ref / name) != ref_meta['files'][name]: raise ValueError('Reference test bytes changed')
            shutil.copyfile(ref / name, dest / name)
        sample = np.random.default_rng(args.seed).choice(len(splits['test']), 128, replace=False)
        cached = np.load(dest / 'test_X.npy', mmap_mode='r')
        expected = np.nan_to_num(np.asarray(suite['X'][splits['test'][sample]], dtype=np.float32))
        if not np.array_equal(cached[sample], expected): raise ValueError('Reference feature mismatch')
        plans = make_plans(y, names, ds, alloc, available, args.seed, args.history_cap)
        write(dest / 'plans.json', plans)
        write(dest / 'manifest.json', dict(dataset=ds, class_names=names, available_train=available,
            allocation=alloc, final_stream_rows=sum(alloc['uniform'].values()),
            stages=STAGES[ds], first_appearance_utc=ref_meta['first_appearance_utc'],
            fixed_test_class_counts=ref_meta['fixed_test_class_counts'],
            clean_manifest_sha256=sha(clean / 'manifest.json'), source_sha256=source_sha,
            source_test_reference=str(ref.relative_to(ROOT)),
            plans_sha256=sha(dest / 'plans.json'), files={p.name: sha(p) for p in dest.glob('*.npy')}))
        print(ds, 'allocation', alloc, 'fit rows', {c: [len(x['fit']) for x in plans[c]] for c in CASES}, flush=True)
    del suite; gc.collect()
    write(out / 'prepared.json', dict(seed=args.seed, nominal_attack_labels=args.nominal,
        history_cap=args.history_cap, source_sha256=source_sha, cases=CASES,
        benign_policy='75% of final stream, equally released over all stages; identical across cases',
        natural_policy='attack-only clean-train proportions; no test frequencies',
        memory_policy='previous fitted rows only; 75% target benign, case-specific attack weights, capacity caps and minimum one per seen class',
        seconds=time.perf_counter()-started, completed=utc()))


def get_model(method, args, class_count):
    if method == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, min_child_weight=1, reg_lambda=1,
            objective='multi:softprob', num_class=class_count, eval_metric='mlogloss',
            tree_method='hist', device='cuda:0', n_jobs=args.threads, random_state=args.seed)
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device='cuda:0', model_path=str(args.checkpoint.resolve()),
        random_state=args.seed, n_estimators=4, auto_scale_n_estimators=False,
        fit_mode='fit_with_cache', keep_cache_on_device=False, n_preprocessing_jobs=1,
        inference_config={'SUBSAMPLE_SAMPLES': None})


def worker(args):
    import torch
    torch.set_num_threads(args.threads)
    out = args.out.resolve(); source = (args.stream_root or out).resolve()
    base = source / 'data' / args.dataset
    info = json.loads((base / 'manifest.json').read_text())
    plans = json.loads((base / 'plans.json').read_text())[args.case]
    names = info['class_names']
    a = {k: np.load(base / f'{k}.npy', mmap_mode='r') for k in
         ['train_ids', 'train_y', 'train_X', 'train_hash', 'test_ids', 'test_y', 'test_X', 'test_hash']}
    model = None; previous_fit = np.empty(0, dtype=np.int64); previous_report = None
    delivered = np.empty(0, dtype=np.int64); prior_pred_path = None; prior_eval_idx = None
    model_fit_count = 0
    for plan in plans:
        stage = plan['stage']; seen = seen_classes(args.dataset, stage)
        dest = out / 'jobs' / f'{args.dataset}_{args.case}_{args.method}' / f'stage_{stage}'
        incoming = np.asarray(plan['incoming'], dtype=np.int64)
        delivered = np.concatenate([delivered, incoming])
        lookup = np.full(len(names), -1, dtype=np.int16)
        for i, c in enumerate(seen): lookup[names.index(c)] = i
        eval_idx = np.flatnonzero(lookup[a['test_y']] >= 0)
        y = lookup[a['test_y'][eval_idx]]
        if (dest / 'COMPLETE.json').exists():
            previous_fit = np.load(dest / 'fit_bank_indices.npy')
            previous_report = json.loads((dest / 'metrics.json').read_text())
            prior_pred_path = dest / 'pred.npy'; prior_eval_idx = eval_idx
            continue
        if (out / 'STOP_AFTER_STAGE').exists(): return 75
        dest.mkdir(parents=True, exist_ok=True)
        cost = Cost(dest)
        success = False
        try:
            with cost.phase('select_and_materialize'):
                if args.history_mode == 'bounded':
                    history, note = select_history(previous_fit, a['train_y'], names, args.case,
                            info['available_train'], args.history_cap, args.seed + stage * 100003)
                    fit = np.concatenate([history, incoming])
                    if not np.array_equal(fit, np.asarray(plan['fit'])): raise ValueError('Replay disagrees with plan')
                else:
                    fit = delivered.copy(); history = fit[:-len(incoming)]
                    note = {'active': False, 'floor_adjusted_classes': []}
                X = np.asarray(a['train_X'][fit]); tr_y = lookup[a['train_y'][fit]]
                if set(tr_y.tolist()) != set(range(len(seen))): raise ValueError('Missing/future class')
            cold = args.method == 'tabpfn' and model_fit_count == 0
            with cost.phase('model_construct'):
                if args.method == 'xgb' or model is None: model = get_model(args.method, args, len(seen))
            with cost.phase('fit_context_or_retrain'):
                model.fit(X, tr_y)
                model_fit_count += 1
            update_seconds = sum(cost.seconds(n) for n in ['select_and_materialize', 'model_construct', 'fit_context_or_retrain'])
            np.save(dest / 'fit_bank_indices.npy', fit)
            np.save(dest / 'train_ids.npy', a['train_ids'][fit])
            np.save(dest / 'test_ids.npy', a['test_ids'][eval_idx])
            result = np.lib.format.open_memmap(dest / 'proba.npy', mode='w+', dtype='float32', shape=(len(y), len(seen)))
            pred = np.lib.format.open_memmap(dest / 'pred.npy', mode='w+', dtype='int16', shape=(len(y),))
            batch_times = []; start_inference = time.perf_counter()
            with cost.phase('predict'):
                for start in range(0, len(y), args.batch):
                    end = min(start + args.batch, len(y)); torch.cuda.synchronize(); tick = time.perf_counter()
                    p = model.predict_proba(np.asarray(a['test_X'][eval_idx[start:end]]))
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - tick
                    if p.shape != (end-start, len(seen)) or not np.isfinite(p).all(): raise ValueError('Invalid probabilities')
                    if not np.array_equal(model.classes_, np.arange(len(seen))): raise ValueError('Invalid label map')
                    result[start:end] = p; pred[start:end] = p.argmax(1)
                    batch_times.append({'rows': end-start, 'seconds': elapsed})
                    result.flush(); pred.flush()
                    rate = end / max(time.perf_counter()-start_inference, 1e-9)
                    write(dest / 'progress.json', dict(state='running', phase='predict', dataset=args.dataset,
                        case=args.case, method=args.method, stage=stage, rows_done=end, rows_total=len(y),
                        rows_per_second=rate, remaining_seconds=(len(y)-end)/rate, update_seconds=update_seconds,
                        pid=os.getpid(), updated=utc()))
                    if start == 0 or len(batch_times) % 20 == 0 or end == len(y):
                        print(args.dataset, args.case, args.method, stage, f'{end:,}/{len(y):,}', f'{rate:,.0f}/s', flush=True)
            report = metrics(y, pred, seen, plan['new_classes'], STAGES[args.dataset][0])
            overlap = np.isin(a['test_hash'][eval_idx], a['train_hash'][fit])
            report['without_context_vector_overlap'] = metrics(y[~overlap], pred[~overlap], seen, plan['new_classes'], STAGES[args.dataset][0])
            report['exact_feature_overlap_test_rows'] = int(overlap.sum())
            if stage:
                old_count = len(seen_classes(args.dataset, stage-1)); old_mask = y < old_count
                fixed_old = metrics(y[old_mask], pred[old_mask], seen, plan['new_classes'], STAGES[args.dataset][0])
                if not np.array_equal(eval_idx[old_mask], prior_eval_idx): raise ValueError('Old population changed')
                old_pred = np.load(prior_pred_path, mmap_mode='r')
                before_correct = old_pred == y[old_mask]; after_correct = pred[old_mask] == y[old_mask]
                report.update(retention_before_macro_f1=previous_report['macro_f1'],
                    retention_after_macro_f1=fixed_old['previous_class_macro_f1'],
                    retention_old_rows=int(old_mask.sum()),
                    old_corrected=int(np.sum(~before_correct & after_correct)),
                    old_damaged=int(np.sum(before_correct & ~after_correct)))
            else:
                report.update(retention_before_macro_f1=None, retention_after_macro_f1=None,
                              retention_old_rows=0, old_corrected=0, old_damaged=0)
            report.update(dataset=args.dataset, case=args.case, method=args.method, stage=stage,
                history_mode=args.history_mode, seed=args.seed, seen_classes=seen, new_classes=plan['new_classes'],
                history_rows=len(history), incoming_rows=len(incoming), context_rows=len(fit),
                delivered_rows=len(delivered), memory=note,
                fit_class_counts={c:int(np.sum(tr_y == i)) for i,c in enumerate(seen)},
                incoming_class_counts=plan['incoming_class_counts'],
                train_ids_digest=array_digest(a['train_ids'][fit]), test_ids_digest=array_digest(a['test_ids'][eval_idx]),
                stream_plan_sha256=sha(base / 'plans.json'), checkpoint_sha256=args.checkpoint_sha,
                update_seconds=update_seconds, select_seconds=cost.seconds('select_and_materialize'),
                construct_seconds=cost.seconds('model_construct'), fit_seconds=cost.seconds('fit_context_or_retrain'),
                predict_seconds=cost.seconds('predict'), first_batch_seconds=batch_times[0]['seconds'],
                first_batch_rows=batch_times[0]['rows'], batch_times=batch_times,
                pfn_cold_fit=cold, packages={n:im.version(n) for n in ['torch','tabpfn','xgboost','numpy']},
                gpu=torch.cuda.get_device_name(0), completed=utc())
            write(dest / 'metrics.json', report)
            previous_fit=fit; previous_report=report; prior_pred_path=dest/'pred.npy'; prior_eval_idx=eval_idx
            success=True
        finally:
            resources=cost.finish()
        if success:
            write(dest / 'COMPLETE.json', dict(state='complete', metrics_sha256=sha(dest/'metrics.json'),
                  peak_gpu_sampled_gib=resources['peak_gpu_sampled_gib'], completed=utc()))
    return 0


def aggregate(out, require_complete=False):
    rows=[]; classes=[]; reports=[]
    for ds in STAGES:
        for case in CASES:
            for stage in range(len(STAGES[ds])):
                paired={}
                for method in ['xgb','tabpfn']:
                    folder=out/'jobs'/f'{ds}_{case}_{method}'/f'stage_{stage}'
                    if not (folder/'COMPLETE.json').exists(): continue
                    marker=json.loads((folder/'COMPLETE.json').read_text())
                    if sha(folder/'metrics.json') != marker['metrics_sha256']: raise ValueError('Result modified')
                    r=json.loads((folder/'metrics.json').read_text());paired[method]=r;reports.append(r)
                    keys=['dataset','case','method','stage','history_mode','seed','history_rows','incoming_rows',
                          'context_rows','delivered_rows','rows','macro_f1','new_class_macro_f1','benign_false_alarm_rate',
                          'retention_before_macro_f1','retention_after_macro_f1','old_corrected','old_damaged',
                          'select_seconds','construct_seconds','fit_seconds','update_seconds','predict_seconds',
                          'first_batch_seconds','pfn_cold_fit','exact_feature_overlap_test_rows']
                    rows.append({k:r[k] for k in keys})
                    classes.extend(dict(dataset=ds,case=case,method=method,stage=stage,
                        context_rows=r['fit_class_counts'][c['name']],new_class=c['name'] in r['new_classes'],**c) for c in r['classes'])
                if len(paired)==2:
                    for k in ['train_ids_digest','test_ids_digest','seen_classes','seed','stream_plan_sha256','history_mode']:
                        if paired['xgb'][k]!=paired['tabpfn'][k]:raise ValueError(f'Unpaired comparison: {k}')
    for name,data in [('summary.csv',rows),('class_metrics.csv',classes)]:
        if data:
            tmp=out/(name+'.tmp')
            with tmp.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
            tmp.replace(out/name)
    if require_complete and len(rows)!=44:raise ValueError(f'Expected 44 completed stages, got {len(rows)}')
    write(out/'report_data.json',dict(results=reports, complete=len(rows)==44))
    return rows


def run(args):
    out=args.out.resolve(); source=(args.stream_root or out).resolve()
    prepared=json.loads((source/'prepared.json').read_text())
    if (prepared['seed'],prepared['history_cap'],prepared['nominal_attack_labels']) != (args.seed,args.history_cap,args.nominal):
        raise ValueError('Run/stream configuration mismatch')
    for ds in STAGES:
        folder=source/'data'/ds; meta=json.loads((folder/'manifest.json').read_text())
        for name,digest in meta['files'].items():
            if sha(folder/name)!=digest:raise ValueError(f'Input bytes changed: {ds}/{name}')
        if sha(folder/'plans.json')!=meta['plans_sha256']:raise ValueError('Plan changed')
    checkpoint_sha=sha(args.checkpoint)
    config=dict(seed=args.seed,history_cap=args.history_cap,nominal=args.nominal,history_mode=args.history_mode,
                stream_root=str(source),batch=args.batch,threads=args.threads,checkpoint_sha256=checkpoint_sha,
                prepared_sha256=sha(source/'prepared.json'))
    if (out/'request.json').exists():
        if json.loads((out/'request.json').read_text())!=config:raise ValueError('Resume config mismatch')
    else:write(out/'request.json',config)
    for p in [Path(__file__),ROOT/'scripts/common/exp61_utils.py',ROOT/'scripts/v7/exp69/exp69_class_arrival.py',
              ROOT/'scripts/common/experiment_paths.py']:
        dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha(dest)!=sha(p):raise ValueError('Source changed during run')
        if not dest.exists():shutil.copy2(p,dest)
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES='0')
    logs=out/'logs';logs.mkdir(exist_ok=True)
    for ds in STAGES:
        for case in CASES:
            for method in ['tabpfn','xgb']:
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--out',str(out),
                    '--stream-root',str(source),'--dataset',ds,'--case',case,'--method',method,
                    '--history-mode',args.history_mode,'--nominal',str(args.nominal),'--history-cap',str(args.history_cap),
                    '--seed',str(args.seed),'--batch',str(args.batch),'--threads',str(args.threads),
                    '--checkpoint',str(args.checkpoint.resolve()),'--checkpoint-sha',checkpoint_sha]
                log=logs/f'{ds}_{case}_{method}.log'
                print('Launching',ds,case,method,flush=True)
                with log.open('a') as f:
                    proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env)
                    while proc.poll() is None:
                        write(out/'status.json',dict(state='running',dataset=ds,case=case,method=method,
                              worker_pid=proc.pid,controller_pid=os.getpid(),log=str(log.relative_to(out)),updated=utc()))
                        time.sleep(3)
                aggregate(out)
                if proc.returncode:
                    write(out/'status.json',dict(state='failed',returncode=proc.returncode,dataset=ds,case=case,method=method))
                    return proc.returncode
    aggregate(out,require_complete=True)
    write(out/'COMPLETE.json',dict(state='complete',summary_sha256=sha(out/'summary.csv'),completed=utc()))
    write(out/'status.json',dict(state='complete',updated=utc(),completed_stages=44))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['all','prepare','run','worker','aggregate'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stream-root',type=Path)
    p.add_argument('--data',type=Path,default=ROOT/'data/nfv3_energy_suite_uncapped_scenarios.pkl')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt')
    p.add_argument('--checkpoint-sha',default='')
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--nominal',type=int,default=1024)
    p.add_argument('--history-cap',type=int,default=4096)
    p.add_argument('--history-mode',choices=['bounded','cumulative'],default='bounded')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--batch',type=int,default=32768)
    p.add_argument('--dataset',choices=list(STAGES))
    p.add_argument('--case',choices=CASES)
    p.add_argument('--method',choices=['xgb','tabpfn'])
    args=p.parse_args()
    if args.command=='all':
        if args.stream_root is None:prepare(args)
        return run(args)
    if args.command=='prepare':prepare(args);return 0
    if args.command=='run':return run(args)
    if args.command=='worker':return worker(args)
    if args.command=='aggregate':aggregate(args.out,require_complete=True);return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc();sys.exit(1)

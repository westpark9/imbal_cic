#!/usr/bin/env python3
"""Class-arrival replay: equal labeled rows, XGB refit vs frozen TabPFN context.

Prepared data and completed stages are portable. No test labels enter fit.
Run `prepare`, then `run`; `run` resumes only fully completed stages.
"""
import argparse
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp61_utils import Cost, sha, write

STAGES = {
    'cic2018': [
        ['benign', 'brute_force', 'dos'], ['ddos'], ['web_attacks'],
        ['infiltration'], ['bot']],
    'toniot': [
        ['benign', 'scanning', 'dos'], ['injection', 'ddos'], ['password'],
        ['xss'], ['ransomware', 'backdoor'], ['mitm']],
}
CLEAN = {
    'cic2018': 'cic2018_conflict_free_fixed_split_20260915_180000',
    'toniot': 'toniot_conflict_free_fixed_split_20260916_021730',
}
DATASET_KEY = {'cic2018': 'cse_cic_ids2018', 'toniot': 'ton_iot'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def seen_classes(dataset, stage):
    return [c for group in STAGES[dataset][:stage + 1] for c in group]


def encode_labels(labels, names):
    lookup = {name: i for i, name in enumerate(names)}
    return np.asarray([lookup[x] for x in labels], dtype=np.int16)


def select_context(y, names, dataset, stage, budget):
    """The stored per-class rows are a seeded random order; budgets are nested."""
    mask = []
    for name in seen_classes(dataset, stage):
        ids = np.flatnonzero(y == names.index(name))
        limit = len(ids) if name in STAGES[dataset][0] else budget
        if len(ids) < limit:
            raise ValueError(f'Insufficient labels: {name}, {len(ids)} < {limit}')
        mask.extend(ids[:limit])
    return np.asarray(mask, dtype=np.int64)


def metrics(y, pred, names, new_names, initial_names):
    c = len(names)
    if len(y) == 0:
        return None
    if np.any((y < 0) | (y >= c) | (pred < 0) | (pred >= c)):
        raise ValueError('Invalid class encoding')
    cm = np.bincount(y.astype(np.int64)*c + pred, minlength=c*c).reshape(c, c)
    support = cm.sum(1); tp = np.diag(cm); fp = cm.sum(0)-tp; fn = support-tp
    div = lambda a, b: np.divide(a, b, out=np.zeros(c, dtype=float), where=b != 0)
    p = div(tp, tp+fp); r = div(tp, support); f = div(2*tp, 2*tp+fp+fn)
    average = lambda group: float(np.mean([f[names.index(n)] for n in group])) if group else None
    b = names.index('benign')
    return dict(rows=int(len(y)), macro_f1=float(f.mean()), accuracy=float(tp.sum()/len(y)),
                new_class_macro_f1=average(new_names),
                initial_class_macro_f1=average(initial_names),
                previous_class_macro_f1=average([n for n in names if n not in new_names]),
                benign_false_alarm_rate=float(fn[b]/support[b]) if support[b] else None,
                confusion_matrix=cm.tolist(),
                classes=[dict(name=n, support=int(support[i]), tp=int(tp[i]), fp=int(fp[i]),
                              fn=int(fn[i]), precision=float(p[i]), recall=float(r[i]), f1=float(f[i]))
                         for i, n in enumerate(names)])


def prepare(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'prepared.json').exists():
        print('Prepared input exists; keeping immutable sample IDs', flush=True)
        return
    write(out/'status.json', dict(state='preparing', updated=utc(), pid=os.getpid()))
    source = Path(args.data).resolve()
    print(f'Loading source: {source}', flush=True)
    t = time.perf_counter()
    with source.open('rb') as fh:
        suite = pickle.load(fh)
    print(f'Loaded in {time.perf_counter()-t:.1f}s; keys={list(suite)}', flush=True)
    families = np.asarray(suite['families']); times = np.asarray(suite['timestamps'])
    summaries = {}
    for ds in STAGES:
        clean = Path(args.clean_root)/CLEAN[ds]
        manifest = json.loads((clean/'manifest.json').read_text())
        names = manifest['class_names']
        tr = np.load(clean/'train_idx.npy'); te = np.load(clean/'test_idx.npy')
        va = np.load(clean/'val_idx.npy')
        source_ids = np.load(clean/'source_idx.npy', mmap_mode='r')
        hashes = np.load(clean/'vector_hash.npy', mmap_mode='r')
        if not np.all(source_ids[1:] > source_ids[:-1]):
            raise ValueError('source_idx must be strictly sorted for hash lookup')
        for split, ids in [('train', tr), ('test', te), ('val', va)]:
            if sha(clean/f'{split}_idx.npy') != manifest['artifacts_sha256'][f'{split}_idx.npy']:
                raise ValueError(f'{ds} {split} manifest mismatch')
        dest = out/'data'/ds; dest.mkdir(parents=True, exist_ok=True)
        selections = []; first = {}; counts = {}
        # Selection depends on training labels, never test labels or predictions.
        train_labels = families[tr]
        for i, name in enumerate(names):
            candidate = tr[train_labels == name]
            n = (args.initial_benign if name == 'benign' else args.initial_attack) if name in STAGES[ds][0] else max(args.budgets)
            if len(candidate) < n:
                raise ValueError(f'{ds}/{name}: requested {n}, only {len(candidate)} train rows')
            chosen = np.random.default_rng(args.seed + i*1009).choice(candidate, n, replace=False)
            selections.append(chosen); counts[name] = dict(available_train=int(len(candidate)), saved=int(n))
            source_class = source_ids[families[source_ids] == name]
            first[name] = int(times[source_class].min())
        selected = np.concatenate(selections)
        if np.intersect1d(selected, te).size or np.intersect1d(selected, va).size:
            raise ValueError('Train/evaluation row leakage')
        for name, arr in dict(train_X=np.nan_to_num(np.asarray(suite['X'][selected], dtype=np.float32)),
                              train_y=encode_labels(families[selected], names), train_ids=selected,
                              train_time=times[selected], train_hash=hashes[np.searchsorted(source_ids, selected)]).items():
            np.save(dest/f'{name}.npy', arr)
        cache = ROOT/'tabpfn/results/20260930_exp62_sv_current_bank_s43'/ds/'cache'
        cache_ids = np.load(cache/'eval_ids.npy', mmap_mode='r') if (cache/'eval_ids.npy').exists() else None
        reuse = cache_ids is not None and np.array_equal(cache_ids, te)
        # Materialized arrays are independent of the source pickle and portable to A100.
        if reuse:
            shutil.copyfile(cache/'eval_X.npy', dest/'test_X.npy')
            check_rows = np.random.default_rng(args.seed).choice(len(te), 128, replace=False)
            cached = np.load(dest/'test_X.npy', mmap_mode='r')
            if not np.array_equal(cached[check_rows], np.nan_to_num(np.asarray(suite['X'][te[check_rows]], dtype=np.float32))):
                raise ValueError('Raw feature cache mismatch')
        else:
            np.save(dest/'test_X.npy', np.nan_to_num(np.asarray(suite['X'][te], dtype=np.float32)))
        for name, arr in dict(test_y=encode_labels(families[te], names), test_ids=te,
                              test_hash=hashes[np.searchsorted(source_ids, te)]).items():
            np.save(dest/f'{name}.npy', arr)
        prev = -1
        for group in STAGES[ds]:
            attack_dates = [first[n] for n in group if n != 'benign']
            if min(attack_dates) < prev:
                raise ValueError('Attack first-appearance order violated')
            prev = max(attack_dates)
        info = dict(class_names=names, stages=STAGES[ds], first_appearance_ms=first,
                    first_appearance_utc={n: datetime.fromtimestamp(v/1000, timezone.utc).isoformat() for n,v in first.items()},
                    training_counts=counts, test_rows=int(len(te)), clean_manifest=manifest,
                    raw_feature_cache_reused=bool(reuse), seed=args.seed,
                    files={p.name: sha(p) for p in dest.glob('*.npy')})
        write(dest/'manifest.json', info); summaries[ds] = info
        print(f'Prepared {ds}: train bank={len(selected):,}, test={len(te):,}', flush=True)
    source_sha = sha(source)
    if any(info['clean_manifest']['source']['sha256'] != source_sha for info in summaries.values()):
        raise ValueError('Raw source SHA does not match cleaned split provenance')
    write(out/'prepared.json', dict(created=utc(), seed=args.seed, budgets=args.budgets,
          initial_benign=args.initial_benign, initial_attack=args.initial_attack,
          source_sha256=source_sha, source_bytes=source.stat().st_size,
          seconds=time.perf_counter()-t, datasets=list(summaries)))
    del suite; gc.collect()


def make_model(method, args, c):
    if method == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, min_child_weight=1, reg_lambda=1,
            objective='multi:softprob', num_class=c, eval_metric='mlogloss',
            tree_method='hist', device='cuda:0', n_jobs=args.threads, random_state=args.seed)
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device='cuda:0', model_path=str(args.checkpoint.resolve()),
        random_state=args.seed, n_estimators=4, auto_scale_n_estimators=False,
        fit_mode='fit_with_cache', keep_cache_on_device=False,
        n_preprocessing_jobs=1, inference_config={'SUBSAMPLE_SAMPLES': None})


def verify_prepared(out):
    for ds in STAGES:
        directory = out/'data'/ds
        info = json.loads((directory/'manifest.json').read_text())
        for name, digest in info['files'].items():
            if sha(directory/name) != digest:
                raise ValueError(f'Prepared data changed: {ds}/{name}')


def worker(args):
    import torch
    import importlib.metadata as im
    torch.set_num_threads(args.threads)
    out=args.out.resolve(); ds=args.dataset; directory=out/'data'/ds
    info=json.loads((directory/'manifest.json').read_text()); all_names=info['class_names']
    prepared=json.loads((out/'prepared.json').read_text())
    if prepared['seed'] != args.seed or args.budget not in prepared['budgets']:
        raise ValueError('Seed/budget differs from preparation')
    arrays={name:np.load(directory/f'{name}.npy',mmap_mode='r') for name in
            ['train_X','train_y','train_ids','train_hash','test_X','test_y','test_ids','test_hash']}
    base=out/'jobs'/f'{ds}_{args.method}_n{args.budget}'; base.mkdir(parents=True,exist_ok=True)
    model=None; model_fit_count=0
    for stage, new in enumerate(STAGES[ds]):
        # Identical initial model/evaluation shared across budgets; never duplicated in totals.
        dest=(out/'jobs'/f'{ds}_{args.method}_initial' if stage==0 else base/f'stage_{stage}')
        if (dest/'COMPLETE.json').exists():
            continue
        if (out/'STOP_AFTER_STAGE').exists():
            print('Stopped at stage boundary by STOP_AFTER_STAGE',flush=True)
            return 75
        dest.mkdir(parents=True,exist_ok=True)
        names=seen_classes(ds,stage); c=len(names)
        sel=select_context(arrays['train_y'],all_names,ds,stage,args.budget)
        lookup=np.full(len(all_names),-1,dtype=np.int16)
        for i,n in enumerate(names):lookup[all_names.index(n)]=i
        tr_y=lookup[arrays['train_y'][sel]]
        eval_idx=np.flatnonzero(lookup[arrays['test_y']]>=0)
        y=lookup[arrays['test_y'][eval_idx]]
        if set(tr_y.tolist())!=set(range(c)) or np.any(tr_y<0):
            raise ValueError('Noncontiguous/future label in context')
        train_ids=np.asarray(arrays['train_ids'][sel]); test_ids=np.asarray(arrays['test_ids'][eval_idx])
        np.save(dest/'train_ids.npy',train_ids); np.save(dest/'test_ids.npy',test_ids)
        identity=dict(dataset=ds,method=args.method,stage=stage,new_classes=new if stage else [],
            seen_classes=names,labels_per_new_class=args.budget if stage else None,
            context_rows=len(sel),eval_rows=len(eval_idx),train_ids_digest=array_digest(train_ids),
            test_ids_digest=array_digest(test_ids),seed=args.seed,checkpoint_sha256=args.checkpoint_sha,
            training_class_counts={n:int((tr_y==i).sum()) for i,n in enumerate(names)},
            preprocessing_fit='current accumulated context only',status='running',started=utc(),
            gpu=torch.cuda.get_device_name(0),packages={n:im.version(n) for n in ['torch','tabpfn','xgboost','numpy','scikit-learn']})
        write(dest/'identity.json',identity)
        cost=Cost(dest); success=False
        try:
            with cost.phase('model_construct'):
                if args.method=='xgb' or model is None:
                    model=make_model(args.method,args,c)
                cold=model_fit_count==0 and args.method=='tabpfn'
            with cost.phase('fit_context_or_retrain'):
                model.fit(np.asarray(arrays['train_X'][sel]),tr_y)
                model_fit_count+=1
            # Fit may load checkpoint lazily. Warm fits and resumed cold fits are explicit.
            identity['pfn_model_previously_fitted']=bool(args.method=='tabpfn' and not cold)
            result=np.lib.format.open_memmap(dest/'proba.npy',mode='w+',dtype='float32',shape=(len(y),c))
            wall=time.perf_counter(); batch_times=[]
            with cost.phase('predict'):
                for start in range(0,len(y),args.batch):
                    end=min(start+args.batch,len(y)); torch.cuda.synchronize(); tick=time.perf_counter()
                    probability=model.predict_proba(np.asarray(arrays['test_X'][eval_idx[start:end]]))
                    torch.cuda.synchronize(); elapsed=time.perf_counter()-tick
                    if probability.shape!=(end-start,c) or not np.isfinite(probability).all():
                        raise ValueError('Invalid probabilities')
                    if not np.array_equal(model.classes_,np.arange(c)):
                        raise ValueError('Prediction class map mismatch')
                    result[start:end]=probability; result.flush()
                    batch_times.append(dict(rows=end-start,seconds=elapsed))
                    rate=end/max(time.perf_counter()-wall,1e-9)
                    write(dest/'progress.json',dict(state='running',phase='predict',rows_done=end,
                        rows_total=len(y),rows_per_second=rate,remaining_seconds=(len(y)-end)/rate,
                        fit_seconds=cost.seconds('fit_context_or_retrain'),pid=os.getpid(),updated=utc()))
                    if start==0 or end==len(y) or len(batch_times)%10==0:
                        print(f'{ds} {args.method} n={args.budget} stage={stage}: {end:,}/{len(y):,} {rate:,.0f} rows/s',flush=True)
            pred=np.asarray(result).argmax(1).astype(np.int16); np.save(dest/'pred.npy',pred)
            report=metrics(y,pred,names,new if stage else [],STAGES[ds][0])
            overlap=np.isin(arrays['test_hash'][eval_idx],arrays['train_hash'][sel])
            # Same-label duplicate vectors survived the historical split; disclose separately.
            report['exact_feature_overlap_test_rows']=int(overlap.sum())
            report['without_context_vector_overlap']=metrics(y[~overlap],pred[~overlap],names,new if stage else [],STAGES[ds][0])
            report.update(identity)
            report.update(status='complete',fit_seconds=cost.seconds('fit_context_or_retrain'),
                          predict_seconds=cost.seconds('predict'),batch_times=batch_times)
            if args.method=='xgb':model.save_model(dest/'model.ubj')
            write(dest/'metrics.json',report); success=True
        finally:
            costs=cost.finish()
        if success:
            write(dest/'identity.json',dict(identity,status='complete'))
            write(dest/'COMPLETE.json',dict(state='complete',finished=utc(),cost=costs,
                                           metrics_sha256=sha(dest/'metrics.json')))
        del result; gc.collect()
    return 0


def aggregate(out, budgets):
    rows=[]; classes=[]
    for ds in STAGES:
        for budget in budgets:
            for stage in range(len(STAGES[ds])):
                found={}
                for method in ['xgb','tabpfn']:
                    p=out/'jobs'/(f'{ds}_{method}_initial' if stage==0 else f'{ds}_{method}_n{budget}/stage_{stage}')
                    if not (p/'COMPLETE.json').exists():continue
                    value=json.loads((p/'metrics.json').read_text()); found[method]=value
                    row=dict(dataset=ds,budget=budget,method=method,stage=stage,initial_reused=stage==0,
                        **{k:value[k] for k in ['context_rows','eval_rows','macro_f1','new_class_macro_f1',
                        'previous_class_macro_f1','initial_class_macro_f1','benign_false_alarm_rate',
                        'fit_seconds','predict_seconds','exact_feature_overlap_test_rows']})
                    rows.append(row)
                    classes.extend(dict(dataset=ds,budget=budget,method=method,stage=stage,**cl) for cl in value['classes'])
                if len(found)==2:
                    for key in ['train_ids_digest','test_ids_digest','seen_classes','seed']:
                        if found['xgb'][key]!=found['tabpfn'][key]:raise ValueError(f'Unfair comparison: {key}')
    for name,values in [('summary.csv',rows),('class_metrics.csv',classes)]:
        if values:
            temp=out/(name+'.tmp')
            with temp.open('w') as fh:
                w=csv.DictWriter(fh,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
            temp.replace(out/name)
    return rows


def run(args):
    import subprocess
    out=args.out.resolve(); verify_prepared(out)
    prepared=json.loads((out/'prepared.json').read_text())
    if args.seed!=prepared['seed'] or any(b not in prepared['budgets'] for b in args.budgets):
        raise ValueError('Run configuration conflicts with prepared inputs')
    checkpoint_sha=sha(args.checkpoint)
    request=dict(seed=args.seed,budgets=args.budgets,batch=args.batch,threads=args.threads,
                 checkpoint_sha256=checkpoint_sha,started=utc(),stage_order=STAGES,
                 prepared_manifest_sha256=sha(out/'prepared.json'),
                 cost_policy='One GPU worker at a time; per-stage fit and inference measured separately')
    if (out/'request.json').exists():
        old=json.loads((out/'request.json').read_text())
        for key in ['seed','budgets','checkpoint_sha256','prepared_manifest_sha256','batch']:
            if old[key]!=request[key]:raise ValueError(f'Resume configuration changed: {key}')
    else:write(out/'request.json',request)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
             OPENBLAS_NUM_THREADS=str(args.threads),PYTHONUNBUFFERED='1',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES=args.gpu)
    # Budget 64 first on both datasets. Additional budgets keep identical initial IDs.
    for budget in args.budgets:
        for ds in STAGES:
            for method in ['tabpfn','xgb']:
                if (out/'STOP_AFTER_STAGE').exists():
                    write(out/'status.json',dict(state='paused',updated=utc()));return 75
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--out',str(out),
                    '--dataset',ds,'--method',method,'--budget',str(budget),'--seed',str(args.seed),
                    '--checkpoint',str(args.checkpoint.resolve()),'--checkpoint-sha',checkpoint_sha,
                    '--threads',str(args.threads),'--batch',str(args.batch)]
                log=logs/f'{ds}_{method}_n{budget}.log'
                print(f'Launching {ds}/{method}/{budget}: {log}',flush=True)
                with log.open('a') as fh:
                    proc=subprocess.Popen(command,stdout=fh,stderr=subprocess.STDOUT,env=env)
                    while proc.poll() is None:
                        write(out/'status.json',dict(state='running',dataset=ds,method=method,budget=budget,
                            worker_pid=proc.pid,controller_pid=os.getpid(),updated=utc(),log=str(log.relative_to(out))))
                        time.sleep(5)
                    code=proc.returncode
                aggregate(out,args.budgets)
                if code:
                    write(out/'status.json',dict(state='paused' if code==75 else 'failed',returncode=code,
                          dataset=ds,method=method,budget=budget,updated=utc()))
                    return code
    rows=aggregate(out,args.budgets)
    expected=sum(len(v) for v in STAGES.values())*2*len(args.budgets)
    if len(rows)!=expected:raise ValueError(f'Incomplete aggregation: {len(rows)} != {expected}')
    write(out/'status.json',dict(state='complete',updated=utc(),rows=len(rows)))
    write(out/'COMPLETE.json',dict(state='complete',finished=utc(),summary_sha256=sha(out/'summary.csv')))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','worker','aggregate'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/nfv3_energy_suite_uncapped_scenarios.pkl')
    p.add_argument('--clean-root',type=Path,default=ROOT/'data/derived')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt')
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--budgets',type=int,nargs='+',default=[64,16,256])
    p.add_argument('--initial-benign',type=int,default=10000)
    p.add_argument('--initial-attack',type=int,default=2000)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--batch',type=int,default=32768)
    p.add_argument('--gpu',default='0')
    p.add_argument('--dataset',choices=list(STAGES))
    p.add_argument('--method',choices=['tabpfn','xgb'])
    p.add_argument('--budget',type=int,default=64)
    p.add_argument('--checkpoint-sha')
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='worker':return worker(args)
    elif args.command=='run':return run(args)
    else:aggregate(args.out,args.budgets)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc();sys.exit(1)

#!/usr/bin/env python3
"""Measured-time replay of the approved EXP71 100k class-arrival pools.

Current authorized execution: XGBoost and TabPFN, both datasets, both cases.
The expensive SOTA adapters are retained separately and are NOT queued.
No report is automatically published by an experiment or by its completion.
"""
from replay_protocol import (ROOT, STAGES, partition_test, update_schedule,
                             dispatch_versions, fifo_response_times, confusion_metrics)
import argparse
import gc
import hashlib
import importlib.metadata as im
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import fcntl

import numpy as np
from exp61_utils import sha, write
from exp69_class_arrival import seen_classes, utc, array_digest
from replay_models import fit_stage, Predictor, timed

SOURCE=ROOT/'tabpfn/results/v7/exp71/20261014_103108_cumulative_100k_uniform_natural_s43_43a766d9'


def prepare(out,config):
    source=Path(config['source'])
    if (out/'PREPARED.json').exists():return
    for ds in config['datasets']:
        base=source/'data'/ds
        meta=json.loads((base/'manifest.json').read_text())
        for n,digest in meta['files'].items():
            if sha(base/n)!=digest:raise ValueError('Source file changed: '+str(base/n))
        if sha(base/'plans.json')!=meta['plans_sha256']:raise ValueError('Plans changed')
        y=np.load(base/'test_y.npy',mmap_mode='r')
        order,batches,bounds=partition_test(y,meta['class_names'],ds,config['seed'],config['batch'])
        dest=out/'data'/ds;dest.mkdir(parents=True,exist_ok=True)
        for n,v in dict(order=order,batches=batches,stage_bounds=bounds).items():np.save(dest/(n+'.npy'),v)
        write(dest/'manifest.json',dict(source=str(base),source_manifest_sha256=sha(base/'manifest.json'),
              class_names=meta['class_names'],rows=len(order),batches=len(batches),stage_bounds=bounds.tolist(),
              global_test_class_counts=meta['fixed_test_class_counts'],allocation=meta['allocation'],
              order_sha256=sha(dest/'order.npy'),test_rows_repeated=False,
              validation_policy='XGB/TabPFN fixed settings; no validation labels consumed'))
        print('PREPARED',ds,len(order),'rows',len(batches),'batches',flush=True)
    write(out/'PREPARED.json',dict(state='complete',completed=utc(),source=str(source)))


def stage_data(config,ds,case,stage):
    base=Path(config['source'])/'data'/ds
    meta=json.loads((base/'manifest.json').read_text())
    plan=json.loads((base/'plans.json').read_text())[case][stage]
    names=meta['class_names']; seen=seen_classes(ds,stage)
    indices=np.asarray(plan['fit'],dtype=np.int64)
    tr_y=np.load(base/'train_y.npy',mmap_mode='r')
    lookup=np.full(len(names),-1,dtype=np.int16)
    for j,c in enumerate(seen):lookup[names.index(c)]=j
    start=time.perf_counter()
    X=np.asarray(np.load(base/'train_X.npy',mmap_mode='r')[indices])
    y=lookup[tr_y[indices]]
    elapsed=time.perf_counter()-start
    if np.any(y<0) or set(np.unique(y))!=set(range(len(seen))):raise ValueError('Future/missing training class')
    # A train-only probe measures first-response cost without future test access.
    probe=X[:config['batch']]
    data=dict(Xtr=X,ytr=y,Xte=probe,yte=y[:len(probe)],Xva=X[:0],yva=y[:0],
              names=seen,synthetic=config.get('smoke',False))
    return data,dict(select_seconds=elapsed,train_indices=indices,
                     train_ids=np.load(base/'train_ids.npy',mmap_mode='r')[indices],
                     global_class_indices=[names.index(c) for c in seen])


def worker_fit(out,config,ds,case,method):
    import torch
    model=None
    job=out/'jobs'/f'{ds}_{case}_{method}'
    for stage in range(len(STAGES[ds])):
        dest=job/f'stage_{stage}'
        if (dest/'FIT_COMPLETE.json').exists():continue
        if (out/'STOP_AFTER_UNIT').exists():return 75
        data,identity=stage_data(config,ds,case,stage)
        print('FIT',ds,case,method,stage,len(data['ytr']),flush=True)
        # After resume, restore the previous PFN state so a followup update is
        # not silently charged an unrelated cold checkpoint startup.
        if method=='tabpfn' and model is None and stage:
            from tabpfn import TabPFNClassifier
            model=TabPFNClassifier.load_from_fit_state(job/f'stage_{stage-1}'/'model.tabpfn_fit',device='cuda:0')
        model,info=fit_stage(method,config,ds,data,dest,model)
        info.update(stage=stage,case=case,select_seconds=identity['select_seconds'],
                    update_seconds=info['update_seconds']+identity['select_seconds'],
                    train_ids_sha256=array_digest(identity['train_ids']),
                    global_class_indices=identity['global_class_indices'])
        np.save(dest/'train_ids.npy',identity['train_ids'])
        write(dest/'fit_metrics.json',info)
        write(dest/'FIT_COMPLETE.json',dict(state='complete',metrics_sha256=sha(dest/'fit_metrics.json'),completed=utc()))
        write(job/'progress.json',dict(phase='fit',stage=stage,stages=len(STAGES[ds]),updated=utc(),
              update_seconds=info['update_seconds'],pid=os.getpid()))
        del data;gc.collect()
    model=None;gc.collect();torch.cuda.empty_cache()
    return 0


def make_schedules(out,config,ds,case,method):
    job=out/'jobs'/f'{ds}_{case}_{method}'
    dest=out/'data'/ds
    batches=np.load(dest/'batches.npy'); bounds=np.load(dest/'stage_bounds.npy')
    fit=[json.loads((job/f'stage_{i}'/'fit_metrics.json').read_text()) for i in range(len(STAGES[ds]))]
    updates=[x['update_seconds'] for x in fit]
    schedules={'ideal':batches[:,0].astype(np.int16)}
    records={}
    for rate in config['rates']:
        ready=update_schedule(bounds,updates,rate)
        key=f'rate_{rate}'
        schedules[key]=dispatch_versions(batches,ready,rate)
        records[key]=dict(rate=rate,ready_seconds=ready.tolist(),
                         unfinished_at_last_arrival=[i for i,v in enumerate(ready) if v>bounds[-1,1]/rate])
    for key,v in schedules.items():np.save(job/f'versions_{key}.npy',v)
    write(job/'schedule.json',dict(updates_seconds=updates,rates=records,
          dispatch='Bind newest completed version when batch collection ends; retain it in FIFO queue',
          update_queue='One separate logical FIFO worker; queued builds use the snapshot at their label release',
          contention='No update/inference hardware contention is simulated'))
    return schedules


def worker_predict(out,config,ds,case,method):
    import torch
    job=out/'jobs'/f'{ds}_{case}_{method}'
    prep=out/'data'/ds; base=Path(config['source'])/'data'/ds
    meta=json.loads((prep/'manifest.json').read_text()); names=meta['class_names']
    schedules=make_schedules(out,config,ds,case,method)
    batches=np.load(prep/'batches.npy'); order=np.load(prep/'order.npy',mmap_mode='r')
    testX=np.load(base/'test_X.npy',mmap_mode='r')
    for version in range(len(STAGES[ds])):
        dest=job/f'stage_{version}'
        requested=np.flatnonzero(np.logical_or.reduce([v==version for v in schedules.values()]))
        if (dest/'PREDICT_COMPLETE.json').exists():continue
        if (out/'STOP_AFTER_UNIT').exists():return 75
        data,identity=stage_data(config,ds,case,version)
        predictor=Predictor(method,config,ds,data,dest)
        mapping=np.asarray(identity['global_class_indices'],dtype=np.int16)
        predpath=dest/'replay_pred.npy'; timepath=dest/'batch_seconds.npy'
        if predpath.exists() and timepath.exists():
            pred=np.load(predpath,mmap_mode='r+');seconds=np.load(timepath,mmap_mode='r+')
        else:
            pred=np.lib.format.open_memmap(predpath,mode='w+',dtype='int16',shape=(len(order),));pred[:]=-1;pred.flush()
            seconds=np.lib.format.open_memmap(timepath,mode='w+',dtype='float64',shape=(len(batches),));seconds[:]=np.nan;seconds.flush()
        start=time.perf_counter(); done_start=int(np.isfinite(seconds[requested]).sum()); count=0; rows_done=0
        for idx in requested:
            if np.isfinite(seconds[idx]):continue
            _,lo,hi=batches[idx]
            if (out/'STOP_AFTER_UNIT').exists():predictor.close();return 75
            # Timer includes gathering the observed batch and every model-specific
            # transformation, not result-file I/O or experiment checkpoint loads.
            p,elapsed=timed(lambda:predictor.predict(np.asarray(testX[order[lo:hi]])))
            if p.shape!=(hi-lo,len(mapping)) or not np.isfinite(p).all():raise ValueError('Invalid probabilities')
            if not np.allclose(p.sum(1),1,atol=1e-4):raise ValueError('Non-normalized probabilities')
            pred[lo:hi]=mapping[p.argmax(1)];pred.flush()
            seconds[idx]=elapsed;seconds.flush()
            count+=1;rows_done+=int(hi-lo)
            if count==1 or count%10==0 or idx==requested[-1]:
                elapsed_wall=time.perf_counter()-start;rate=rows_done/max(elapsed_wall,1e-9)
                remaining=len(requested)-done_start-count
                write(job/'progress.json',dict(phase='predict',version=version,versions=len(STAGES[ds]),
                     batches_done=done_start+count,batches_total=len(requested),
                     measured_batch_seconds=elapsed,rows_per_second=rate,
                     version_remaining_seconds=remaining*config['batch']/rate,
                     pid=os.getpid(),updated=utc()))
                print('PREDICT',ds,case,method,version,done_start+count,'/',len(requested),
                      f'{rate:.1f} rows/s',flush=True)
        predictor.close();del data;gc.collect();torch.cuda.empty_cache()
        if np.any(~np.isfinite(seconds[requested])):raise ValueError('Missing batch timings')
        write(dest/'PREDICT_COMPLETE.json',dict(state='complete',requested_batches=len(requested),
              requested_rows=int(np.sum(batches[requested,2]-batches[requested,1])),completed=utc()))
    summarize(out,config,ds,case,method)
    return 0


def summarize(out,config,ds,case,method):
    job=out/'jobs'/f'{ds}_{case}_{method}'; prep=out/'data'/ds
    base=Path(config['source'])/'data'/ds
    meta=json.loads((prep/'manifest.json').read_text());names=meta['class_names']
    order=np.load(prep/'order.npy',mmap_mode='r');batches=np.load(prep/'batches.npy');bounds=np.load(prep/'stage_bounds.npy')
    y=np.asarray(np.load(base/'test_y.npy',mmap_mode='r')[order])
    predby=[np.load(job/f'stage_{i}'/'replay_pred.npy',mmap_mode='r') for i in range(len(STAGES[ds]))]
    timesby=[np.load(job/f'stage_{i}'/'batch_seconds.npy',mmap_mode='r') for i in range(len(STAGES[ds]))]
    results={}
    for key in ['ideal',*[f'rate_{r}' for r in config['rates']]]:
        versions=np.load(job/f'versions_{key}.npy')
        pred=np.full(len(y),-1,dtype=np.int16);seconds=np.empty(len(batches))
        for j,(stage,lo,hi) in enumerate(batches):
            v=versions[j];pred[lo:hi]=predby[v][lo:hi];seconds[j]=timesby[v][j]
        if np.any(pred<0) or not np.isfinite(seconds).all():raise ValueError('Incomplete replay')
        report=confusion_metrics(y,pred,names)
        report['stages']=[]
        for stage,(lo,hi) in enumerate(bounds):
            allowed=[names.index(c) for c in seen_classes(ds,stage)]
            row=confusion_metrics(y[lo:hi],pred[lo:hi],names,allowed)
            row.update(stage=stage,new_classes=STAGES[ds][stage] if stage else [])
            report['stages'].append(row)
        report.update(evaluated_rows=len(y),each_test_row_once=True,predict_service_seconds=float(seconds.sum()))
        np.save(job/f'pred_{key}.npy',pred)
        if key!='ideal':
            rate=int(key.split('_')[1]);lat=fifo_response_times(batches,seconds,rate)
            row_counts=batches[:,2]-batches[:,1]
            # Exact row latency distribution (all rows, no averaging of percentiles).
            latency=np.empty(len(y),dtype=np.float64)
            for j,(_,lo,hi) in enumerate(batches):latency[lo:hi]=lat['finish'][j]-np.arange(lo+1,hi+1)/rate
            report['latency']=dict(mean_seconds=float(latency.mean()),p50_seconds=float(np.quantile(latency,.5)),
                 p95_seconds=float(np.quantile(latency,.95)),max_seconds=float(latency.max()),
                 queue_wait_max_seconds=float(lat['queue_wait'].max()),
                 drain_after_last_arrival_seconds=float(lat['finish'][-1]-len(y)/rate),
                 rows_using_outdated_version=int(row_counts[versions<batches[:,0]].sum()),
                 completed_by_end_of_arrivals=int(row_counts[lat['finish']<=len(y)/rate].sum()))
            np.savez_compressed(job/f'timeline_{key}.npz',version=versions,batches=batches,
                 service_seconds=seconds,finish=lat['finish'],queue_wait=lat['queue_wait'])
        results[key]=report
    write(job/'results.json',dict(dataset=ds,case=case,method=method,seed=config['seed'],results=results))
    write(job/'COMPLETE.json',dict(state='complete',completed=utc(),results_sha256=sha(job/'results.json')))


def aggregate(out,config):
    completed=[]
    for ds in config['datasets']:
        for case in config['cases']:
            for method in config['methods']:
                f=out/'jobs'/f'{ds}_{case}_{method}'/'results.json'
                if f.exists():completed.append(json.loads(f.read_text()))
    write(out/'results.json',dict(state='partial' if len(completed)<8 else 'complete',jobs=completed,
              note='Not published to HTML: user requested protocol only',updated=utc()))


def environment(config):
    env=dict(os.environ)
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[n]=str(config['threads'])
    env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    return env


def controller(out,config):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(out/'status.json',dict(state='preparing',pid=os.getpid(),started=utc()))
        prepare(out,config)
        jobs=[(ds,case,m) for m in config['methods'] for case in config['cases'] for ds in config['datasets']]
        failed=[];start=time.time()
        for ds,case,method in jobs:
            job=out/'jobs'/f'{ds}_{case}_{method}';job.mkdir(parents=True,exist_ok=True)
            for phase in ['fit','predict']:
                if (job/'COMPLETE.json').exists():continue
                attempt=len(list(job.glob(f'{phase}_attempt_*.log')))+1
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),phase,'--out',str(out),
                     '--dataset',ds,'--case',case,'--method',method]
                logpath=job/f'{phase}_attempt_{attempt}.log'
                with logpath.open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL,env=environment(config))
                    print('START',ds,case,method,phase,'PID',proc.pid,flush=True)
                    while proc.poll() is None:
                        detail=json.loads((job/'progress.json').read_text()) if (job/'progress.json').exists() else {}
                        write(out/'status.json',dict(state='running',job=job.name,phase=phase,
                              controller_pid=os.getpid(),worker_pid=proc.pid,progress=detail,
                              elapsed_seconds=time.time()-start,failed=failed,updated=utc()))
                        time.sleep(3)
                print('END',ds,case,method,phase,proc.returncode,flush=True)
                if proc.returncode:
                    failed.append(dict(job=job.name,phase=phase,returncode=proc.returncode,log=str(logpath)))
                    if proc.returncode==75:
                        write(out/'status.json',dict(state='paused',failed=failed,updated=utc()));return 75
                    break
            aggregate(out,config)
        write(out/'status.json',dict(state='complete' if not failed else 'completed_with_failures',
              failed=failed,elapsed_seconds=time.time()-start,completed=utc()))
        return 1 if failed else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','fit','predict','prepare'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--dataset',choices=list(STAGES));p.add_argument('--case',choices=['natural','uniform'])
    p.add_argument('--method',choices=['xgb','tabpfn'])
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():config=json.loads((out/'config.json').read_text())
    else:
        config=dict(protocol_revision='exp72.1',source=str(a.source.resolve()),seed=43,batch=1024,rates=[100,1000],
            threads=8,cost_repeats=3,methods=['xgb','tabpfn'],deferred_methods=['boostpfn','localpfn','distpfn'],
            datasets=['cic2018','toniot'],cases=['natural','uniform'],
            checkpoint=str(ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'),
            hardware_contention=False,html_results_authorized=False)
        write(out/'config.json',config)
        snapshot=out/'source';snapshot.mkdir(exist_ok=True)
        for f in Path(__file__).resolve().parent.glob('*.py'):shutil.copyfile(f,snapshot/f.name)
        import torch
        write(out/'environment.json',dict(python=sys.executable,platform=platform.platform(),
              packages={k:im.version(k) for k in ['torch','numpy','tabpfn','xgboost']},
              gpu=torch.cuda.get_device_name(0),checkpoint_sha256=sha(config['checkpoint']),
              source_files={f.name:sha(f) for f in snapshot.glob('*.py')}))
    os.environ.update(environment(config))
    try:
        if a.command=='run':return controller(out,config)
        if a.command=='prepare':prepare(out,config);return 0
        import torch
        torch.set_num_threads(config['threads'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if not all([a.dataset,a.case,a.method]):p.error('Worker requires dataset/case/method')
        fn=worker_fit if a.command=='fit' else worker_predict
        return fn(out,config,a.dataset,a.case,a.method)
    except BaseException:
        write(out/f'ERROR_{a.command}_{a.dataset}_{a.case}_{a.method}.json',dict(error=traceback.format_exc(),at=utc()))
        raise

if __name__=='__main__':raise SystemExit(main())

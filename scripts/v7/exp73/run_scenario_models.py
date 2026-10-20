#!/usr/bin/env python3
"""Current model + BoostPFN/LoCalPFN/DistPFN on the EXP72 replay protocol."""
from scenario_data import ROOT,SOURCE,STAGES,prepare,stage_data
import argparse,fcntl,gc,hashlib,importlib.metadata as im,json,os,platform,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from exp61_utils import write,sha
from exp69_class_arrival import utc,array_digest
from replay_models import fit_stage,Predictor,timed
from replay_protocol import dist_adjust_current_batch
import run_replay as replay
from current_model import fit_current,CurrentPredictor

BASELINE=ROOT/'tabpfn/results/v7/exp72/20261014_134735_replay_100k_xgb_tabpfn_s43_de8b4a78'


def progress(job,**kw):
    write(job/'progress.json',dict(pid=os.getpid(),updated=utc(),**kw))
    print(json.dumps(kw),flush=True)


def worker_fit(out,config,ds,case,method):
    import torch
    job=out/'jobs'/f'{ds}_{case}_{method}';job.mkdir(parents=True,exist_ok=True)
    for stage in range(len(STAGES[ds])):
        dest=job/f'stage_{stage}';dest.mkdir(exist_ok=True)
        if (dest/'READY.json').exists():continue
        if (out/'STOP_AFTER_UNIT').exists():return 75
        start=time.perf_counter();data,identity=stage_data(out,config,ds,case,stage);selection=time.perf_counter()-start
        progress(job,phase='fit',stage=stage,method=method,dataset=ds,case=case,train_rows=len(data['ytr']),validation_rows=len(data['yva']))
        if method=='ours':info=fit_current(data,config,ds,dest)
        elif method=='distpfn':
            source=Path(config['baseline'])/'jobs'/f'{ds}_{case}_tabpfn'/f'stage_{stage}'
            np.testing.assert_array_equal(np.load(source/'train_ids.npy'),identity['train_ids'])
            original=json.loads((source/'fit_metrics.json').read_text())
            prior,seconds=timed(lambda:np.bincount(data['ytr'],minlength=len(data['names']))/len(data['ytr']))
            np.save(dest/'prior.npy',prior)
            model=dest/'model.tabpfn_fit'
            if not model.exists():model.symlink_to(source/'model.tabpfn_fit')
            info=dict(original,update_seconds=original['update_seconds']+seconds,base_fit_reused=str(source),
                      prior_prepare_seconds=seconds,validation_rows=0,posterior_scope='current completed batch only')
            selection=0. # Identical source input preparation already timed.
        else:
            _,info=fit_stage(method,config,ds,data,dest)
            if method=='localpfn':
                serving=Predictor(method,config,ds,data,dest)
                info['serving_index_seconds']=serving.index_seconds
                info['update_seconds']+=serving.index_seconds
                serving.close()
        info.update(stage=stage,dataset=ds,case=case,method=method,update_seconds=info['update_seconds']+selection,
                    input_prepare_seconds=selection,train_ids_sha256=array_digest(identity['train_ids']),
                    validation_ids_sha256=array_digest(identity['validation_ids']),validation_available_rows=len(identity['validation_ids']),
                    validation_used_rows=len(identity['validation_ids']) if method in ('ours','localpfn') else 0,
                    global_class_indices=identity['global_class_indices'])
        np.save(dest/'train_ids.npy',identity['train_ids']);np.save(dest/'validation_ids.npy',identity['validation_ids'])
        np.savez(dest/'role_ids.npz',**identity['role_ids'])
        write(dest/'fit_metrics.json',info)
        write(dest/'READY.json',dict(state='complete',metrics_sha256=sha(dest/'fit_metrics.json'),completed=utc()))
        del data;gc.collect();torch.cuda.empty_cache()
    return 0


class DistPredictor(Predictor):
    def __init__(self,config,ds,data,dest):
        super().__init__('tabpfn',config,ds,data,dest);self.prior=np.load(dest/'prior.npy')
    def predict(self,X):return dist_adjust_current_batch(super().predict(X),self.prior).astype('float32')


def worker_predict(out,config,ds,case,method):
    import torch
    job=out/'jobs'/f'{ds}_{case}_{method}';prep=out/'data'/ds;base=Path(config['source'])/'data'/ds
    schedules=replay.make_schedules(out,config,ds,case,method)
    batches=np.load(prep/'batches.npy');order=np.load(prep/'order.npy',mmap_mode='r');X=np.load(base/'test_X.npy',mmap_mode='r')
    for version in range(len(STAGES[ds])):
        dest=job/f'stage_{version}';requested=np.flatnonzero(np.logical_or.reduce([v==version for v in schedules.values()]))
        if (dest/'PREDICT_COMPLETE.json').exists():continue
        if (out/'STOP_AFTER_UNIT').exists():return 75
        data,identity=stage_data(out,config,ds,case,version)
        if method=='ours':predictor=CurrentPredictor(dest,config)
        elif method=='distpfn':predictor=DistPredictor(config,ds,data,dest)
        else:predictor=Predictor(method,config,ds,data,dest)
        mapping=np.array(identity['global_class_indices'],dtype='int16')
        def mmap(name,dtype,shape,initial):
            path=dest/(name+'.npy')
            if path.exists():return np.load(path,mmap_mode='r+')
            a=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape);a[:]=initial;a.flush();return a
        pred=mmap('replay_pred','int16',(len(order),),-1);seconds=mmap('batch_seconds','float64',(len(batches),),np.nan)
        aux={}
        if method=='ours':
            for key in ('global_pred','calls','accepted','expert'):aux[key]=mmap(key,'int16',(len(order),),-1)
        started=time.perf_counter();done=0;rows_done=0
        for idx in requested:
            if np.isfinite(seconds[idx]):continue
            if (out/'STOP_AFTER_UNIT').exists():predictor.close();return 75
            _,lo,hi=batches[idx]
            p,elapsed=timed(lambda:predictor.predict(np.asarray(X[order[lo:hi]])))
            if p.shape!=(hi-lo,len(mapping)) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-4):raise ValueError('Invalid probabilities')
            pred[lo:hi]=mapping[p.argmax(1)];pred.flush()
            for key,a in aux.items():
                value=predictor.aux[key]
                a[lo:hi]=mapping[value] if key=='global_pred' else value;a.flush()
            seconds[idx]=elapsed;seconds.flush();done+=1;rows_done+=int(hi-lo)
            if done==1 or done%10==0 or idx==requested[-1]:
                rate=rows_done/max(time.perf_counter()-started,1e-9);remaining=int(np.isnan(seconds[requested]).sum())
                progress(job,phase='predict',version=version,batches_done=len(requested)-remaining,batches_total=len(requested),
                         last_service_seconds=elapsed,rows_per_second=rate,version_remaining_seconds=remaining*config['batch']/rate,
                         method=method,dataset=ds,case=case)
        predictor.close();del data;gc.collect();torch.cuda.empty_cache()
        write(dest/'PREDICT_COMPLETE.json',dict(state='complete',requested_batches=len(requested),completed=utc()))
    replay.summarize(out,config,ds,case,method)
    if method=='ours':summarize_policy(out,config,ds,case)
    return 0


def summarize_policy(out,config,ds,case):
    job=out/'jobs'/f'{ds}_{case}_ours';prep=out/'data'/ds
    result=json.loads((job/'results.json').read_text());batches=np.load(prep/'batches.npy');bounds=np.load(prep/'stage_bounds.npy')
    order=np.load(prep/'order.npy');y=np.load(Path(config['source'])/'data'/ds/'test_y.npy')[order]
    names=json.loads((prep/'manifest.json').read_text())['class_names']
    for policy,value in result['results'].items():
        versions=np.load(job/f'versions_{policy}.npy');final=np.load(job/f'pred_{policy}.npy')
        arrays={key:np.full(len(y),-1,dtype='int16') for key in ('global_pred','calls','accepted','expert')}
        for version in range(len(STAGES[ds])):
            source={key:np.load(job/f'stage_{version}'/(key+'.npy'),mmap_mode='r') for key in arrays}
            for b in np.flatnonzero(versions==version):
                _,lo,hi=batches[b]
                for key in arrays:arrays[key][lo:hi]=source[key][lo:hi]
        g=arrays['global_pred'];calls=arrays['calls'].astype(bool);accepted=arrays['accepted'].astype(bool)
        assert (g>=0).all() and (arrays['calls']>=0).all() and np.all(~accepted|calls)
        def stats(mask):
            return dict(rows=int(mask.sum()),called=int(calls[mask].sum()),accepted=int(accepted[mask].sum()),
              call_rate=float(calls[mask].mean()),accept_rate=float(accepted[mask].mean()),
              helpful=int(((final==y)&(g!=y)&mask).sum()),harmful=int(((final!=y)&(g==y)&mask).sum()),
              changed=int(((final!=g)&mask).sum()))
        value['policy']=stats(np.ones(len(y),bool))
        value['policy']['classes']=[dict(name=n,**stats(y==i)) for i,n in enumerate(names)]
        value['policy']['stages']=[dict(stage=i,**stats((np.arange(len(y))>=lo)&(np.arange(len(y))<hi))) for i,(lo,hi) in enumerate(bounds)]
        np.savez_compressed(job/f'policy_{policy}.npz',**arrays)
        value['global_same_version']=replay.confusion_metrics(y,g,names)
        assert value['policy']['helpful']-value['policy']['harmful']==int((final==y).sum()-(g==y).sum())
    write(job/'results.json',result);write(job/'COMPLETE.json',dict(state='complete',results_sha256=sha(job/'results.json'),completed=utc()))


def aggregate(out,config):
    jobs=[]
    for ds in config['datasets']:
        for case in config['cases']:
            for method in ['xgb','tabpfn',*config['methods']]:
                path=out/'jobs'/f'{ds}_{case}_{method}'
                if (path/'COMPLETE.json').exists():jobs.append(json.loads((path/'results.json').read_text()))
    write(out/'results.json',dict(state='complete' if len(jobs)==24 else 'partial',jobs=jobs,updated=utc(),html_publication='not authorized for this run'))


def controller(out,config):
    import torch
    lock=(out/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    prepare(out,config);(out/'jobs').mkdir(exist_ok=True)
    # Preserve identical earlier baselines rather than remeasuring a changed fit.
    for ds in config['datasets']:
        for case in config['cases']:
            for method in ['xgb','tabpfn']:
                target=out/'jobs'/f'{ds}_{case}_{method}'
                if not target.exists():target.symlink_to(Path(config['baseline'])/'jobs'/target.name,target_is_directory=True)
    start=time.time();failed=[]
    for method in config['methods']:
        for case in config['cases']:
            for ds in config['datasets']:
                job=out/'jobs'/f'{ds}_{case}_{method}';job.mkdir(exist_ok=True)
                if (job/'COMPLETE.json').exists():continue
                for phase in ('fit','predict'):
                    if (out/'STOP_AFTER_UNIT').exists():write(out/'status.json',dict(state='paused',failed=failed,updated=utc()));return 75
                    attempt=len(list(job.glob(f'{phase}_attempt_*.log')))+1;logpath=job/f'{phase}_attempt_{attempt}.log'
                    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),phase,'--out',str(out),'--dataset',ds,'--case',case,'--method',method]
                    with logpath.open('w') as log:
                        process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=replay.environment(config))
                        print('START',ds,case,method,phase,process.pid,flush=True)
                        while process.poll() is None:
                            item=dict(state='running',controller_pid=os.getpid(),worker_pid=process.pid,job=job.name,phase=phase,log=str(logpath),updated=utc(),elapsed_seconds=time.time()-start,failed=failed)
                            if (job/'progress.json').exists():item['progress']=json.loads((job/'progress.json').read_text())
                            write(out/'status.json',item);time.sleep(3)
                    print('END',ds,case,method,phase,process.returncode,flush=True)
                    if process.returncode:
                        failed.append(dict(job=job.name,phase=phase,returncode=process.returncode,log=str(logpath)))
                        if process.returncode==75:write(out/'status.json',dict(state='paused',failed=failed,updated=utc()));return 75
                        break
                aggregate(out,config)
    write(out/'status.json',dict(state='complete' if not failed else 'completed_with_failures',failed=failed,completed=utc(),elapsed_seconds=time.time()-start))
    return int(bool(failed))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','prepare','fit','predict'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',choices=list(STAGES));p.add_argument('--case',choices=['natural','uniform']);p.add_argument('--method',choices=['ours','distpfn','boostpfn','localpfn'])
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    path=out/'config.json'
    if path.exists():config=json.loads(path.read_text())
    else:
        config=dict(protocol_revision='exp73.1',source=str(SOURCE),baseline=str(BASELINE),seed=43,batch=1024,rates=[100,1000],threads=8,
            cost_repeats=3,prepare_batch=4096,experts=4,validation_cap=10000,
            methods=['ours','distpfn','boostpfn','localpfn'],datasets=['cic2018','toniot'],cases=['natural','uniform'],
            checkpoint=str(ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'),hardware_contention=False,
            validation_policy='User chose option 2: unchanged 100k training + separate existing validation; only observed classes',html_results_authorized=False)
        write(path,config)
        snapshot=out/'source';snapshot.mkdir(exist_ok=True)
        for folder in [Path(__file__).resolve().parent,ROOT/'scripts/v7/exp72']:
            for f in folder.glob('*.py'):
                dest=snapshot/folder.name/f.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(f,dest)
        import torch
        write(out/'environment.json',dict(python=sys.executable,platform=platform.platform(),gpu=torch.cuda.get_device_name(0),
              packages={k:im.version(k) for k in ['torch','numpy','tabpfn','xgboost']},checkpoint_sha256=sha(config['checkpoint']),
              source_files={str(f.relative_to(snapshot)):sha(f) for f in snapshot.rglob('*.py')}))
    os.environ.update(replay.environment(config))
    try:
        if args.command=='run':return controller(out,config)
        if args.command=='prepare':prepare(out,config);return 0
        import torch
        torch.set_num_threads(config['threads'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        fn=worker_fit if args.command=='fit' else worker_predict
        return fn(out,config,args.dataset,args.case,args.method)
    except BaseException:
        write(out/f'ERROR_{args.command}_{args.dataset}_{args.case}_{args.method}.json',dict(error=traceback.format_exc(),at=utc()));raise

if __name__=='__main__':raise SystemExit(main())

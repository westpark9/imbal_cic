#!/usr/bin/env python3
"""Inference-only final-state reevaluation of the saved 100k models."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse,gc,json,shutil,time
import numpy as np
from exp61_utils import write,sha
from scenario_data import stage_data,STAGES
from replay_models import Predictor,timed
from current_model import CurrentPredictor
from replay_protocol import confusion_metrics,dist_adjust_current_batch

SOURCE=ROOT/'tabpfn/results/v7/exp73/20261014_154250_replay_current_sota_100k_s43_b0f8a9dc'


def report(out,**kw):
    value=dict(updated_epoch=time.time(),**kw);write(out/'progress.json',value);print(json.dumps(value),flush=True)


def evaluate(out,cfg,ds,case,method):
    import torch
    key=f'{ds}_{case}_{method}';dest=out/'jobs'/key;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    stage=len(STAGES[ds])-1;source=SOURCE/'jobs'/key/f'stage_{stage}'
    data,identity=stage_data(SOURCE,cfg,ds,case,stage)
    names=json.loads((SOURCE/'data'/ds/'manifest.json').read_text())['class_names']
    base=Path(cfg['source'])/'data'/ds
    X=np.load(base/'test_X.npy',mmap_mode='r');y=np.load(base/'test_y.npy',mmap_mode='r')
    mapping=np.array(identity['global_class_indices'],dtype='int16');total=len(y)
    np.testing.assert_array_equal(np.load(source/'train_ids.npy'),identity['train_ids'])
    assert len(identity['train_ids'])==100000
    assert not np.intersect1d(identity['train_ids'],np.load(base/'test_ids.npy')).size
    files=[source/'fit_metrics.json',source/'train_ids.npy']
    if method=='xgb':files += [source/'model.ubj']
    elif method=='tabpfn':files += [source/'model.tabpfn_fit']
    elif method=='ours':files += [source/'state.joblib',source/'contexts.json',*sorted(source.glob('model_*.tabpfn_fit'))]
    elif method=='boostpfn':files += [source/'boost_state.npz',Path(cfg['checkpoint'])]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    report(out,dataset=ds,case=case,method=method,phase='restore',rows_total=total)
    predictor=CurrentPredictor(source,cfg) if method=='ours' else Predictor(method,cfg,ds,data,source)
    prior=np.bincount(data['ytr'],minlength=len(mapping))/len(data['ytr'])
    # An original 1024-row query reproduces saved final-version predictions.
    old_seconds=np.load(source/'batch_seconds.npy');idx=int(np.flatnonzero(np.isfinite(old_seconds))[0])
    batches=np.load(SOURCE/'data'/ds/'batches.npy');_,lo,hi=batches[idx]
    order=np.load(SOURCE/'data'/ds/'order.npy',mmap_mode='r')
    old=np.load(source/'replay_pred.npy',mmap_mode='r')
    check=predictor.predict(np.asarray(X[order[lo:hi]]))
    mismatches=int(np.count_nonzero(mapping[check.argmax(1)]!=old[lo:hi]))
    assert mismatches==0,(key,'restoration mismatches',mismatches)
    restore=dict(rows=int(hi-lo),batch_index=idx,mismatches=mismatches,source_hashes=hashes)
    if method=='tabpfn':
        dsource=SOURCE/'jobs'/f'{ds}_{case}_distpfn'/f'stage_{stage}'
        dold=np.load(dsource/'replay_pred.npy',mmap_mode='r')
        dm=int(np.count_nonzero(mapping[dist_adjust_current_batch(check,prior).argmax(1)]!=dold[lo:hi]))
        assert dm==0,('DistPFN restore',ds,case,dm)
        restore['dist_mismatches']=dm
    write(dest/'restoration.json',restore)
    def mmap(name,dtype,shape,fill):
        path=dest/(name+'.npy')
        if path.exists():return np.load(path,mmap_mode='r+')
        a=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape);a[:]=fill;a.flush();return a
    pred=mmap('predictions','int16',(total,),-1)
    count=(total+cfg['batch']-1)//cfg['batch'];seconds=mmap('service_times','float64',(count,4),np.nan)
    dist=mmap('dist_predictions','int16',(total,),-1) if method=='tabpfn' else None
    aux={k:mmap(k,'int16',(total,),-1) for k in ['global_pred','calls','accepted']} if method=='ours' else {}
    for i,a in enumerate(range(0,total,cfg['batch'])):
        if np.isfinite(seconds[i,2]):continue
        b=min(a+cfg['batch'],total)
        p,elapsed=timed(lambda:predictor.predict(np.asarray(X[a:b])))
        assert p.shape==(b-a,len(mapping)) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-3)
        pred[a:b]=mapping[p.argmax(1)];adjust=0.
        if dist is not None:
            q,adjust=timed(lambda:dist_adjust_current_batch(p,prior));dist[a:b]=mapping[q.argmax(1)];dist.flush()
        for k,v in aux.items():v[a:b]=mapping[predictor.aux[k]] if k=='global_pred' else predictor.aux[k];v.flush()
        pred.flush();seconds[i]=[a,b,elapsed,adjust];seconds.flush()
        if i%10==0 or b==total:
            done=np.isfinite(seconds[:,2]);rate=float(np.sum(seconds[done,1]-seconds[done,0])/np.sum(seconds[done,2]))
            report(out,dataset=ds,case=case,method=method,phase='final_test',rows_done=b,rows_total=total,
                rows_per_second=rate,remaining_seconds=(total-b)/rate)
    assert (pred>=0).all()
    fit=json.loads((source/'fit_metrics.json').read_text())
    result=dict(dataset=ds,case=case,method=method,seed=cfg['seed'],evaluation='final_full_test',test_order='original',
        batch=cfg['batch'],test_rows=total,training_rows=100000,final_stage=stage,
        source_model_dir=str(source.relative_to(ROOT)),source_fit_metrics=fit,source_hashes=hashes,
        metrics=confusion_metrics(y,np.asarray(pred),names),predictions_path=str(dest/'predictions.npy'),
        serving_seconds=float(seconds[:,2].sum()),training_rerun=False,restoration_mismatches=mismatches)
    if method=='ours':
        result['global_metrics']=confusion_metrics(y,np.asarray(aux['global_pred']),names)
        result['policy']=dict(called=int(aux['calls'].sum()),accepted=int(aux['accepted'].sum()),
            helpful=int(((pred==y)&(aux['global_pred']!=y)).sum()),harmful=int(((pred!=y)&(aux['global_pred']==y)).sum()))
    write(dest/'results.json',result)
    if dist is not None:
        d=dict(result,method='distpfn',metrics=confusion_metrics(y,np.asarray(dist),names),
               predictions_path=str(dest/'dist_predictions.npy'),serving_seconds=float(seconds[:,2:].sum()),
               correction_scope='current completed 8192-row query batch; no query labels',initial_finetuning=False)
        write(dest/'dist_results.json',d)
    write(dest/'COMPLETE.json',dict(complete=True,predictions_sha256=sha(dest/'predictions.npy')))
    predictor.close();del predictor,data;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'config.json').read_text());cfg.update(batch=8192,protocol='exp79.1',methods=['xgb','tabpfn','ours','boostpfn'],
        evaluation='final_full_test',training_rerun=False,source_models=str(SOURCE),seed_repetitions=1)
    write(out/'config.json',cfg)
    for source in [Path(__file__),ROOT/'scripts/v7/exp73/current_model.py',ROOT/'scripts/v7/exp72/replay_models.py',ROOT/'scripts/v7/exp72/replay_protocol.py',ROOT/'scripts/v7/exp73/scenario_data.py']:
        target=out/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    import torch
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    # Baselines first, then proposed, then the slower 50-learner ensemble.
    for method in cfg['methods']:
        for ds in cfg['datasets']:
            for case in cfg['cases']:
                evaluate(out,cfg,ds,case,method)
                jobs=[json.loads(p.read_text()) for p in sorted((out/'jobs').glob('*/*results.json'))]
                write(out/'results.json',dict(state='running',jobs=jobs))
    jobs=[json.loads(p.read_text()) for p in sorted((out/'jobs').glob('*/*results.json'))]
    assert len(jobs)==20
    write(out/'results.json',dict(state='complete',jobs=jobs))


if __name__=='__main__':main()

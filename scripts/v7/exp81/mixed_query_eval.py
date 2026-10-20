#!/usr/bin/env python3
"""Mixed-query final evaluation and paired posterior grouping diagnostic."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse,copy,gc,json,shutil,time
import numpy as np
from exp61_utils import write,sha
from replay_models import Predictor,timed
from current_model import CurrentPredictor
from scenario_data import stage_data,STAGES
from replay_protocol import confusion_metrics,dist_adjust_current_batch

SMALL=ROOT/'tabpfn/results/v7/exp77/20261015_230535_initialft10_finalfull_s43_2eb2237d'
OLD_BUDGET=ROOT/'tabpfn/results/v7/exp78/20261016_005435_noft_dist_new100_50_20_10_s43_52588487'
HISTORY=ROOT/'tabpfn/results/v7/exp73/20261014_154250_replay_current_sota_100k_s43_b0f8a9dc'
LARGE=ROOT/'tabpfn/results/v7/exp79/20261016_022105_saved100k_finalfull_s43_7f90866d'
EXPANDED=ROOT/'tabpfn/results/v7/exp80/20261016_022614_new200_400_noft_s43_c6f6ea24'


def progress(out,**kw):
    value=dict(updated_epoch=time.time(),**kw);write(out/'progress.json',value);print(json.dumps(value),flush=True)


def inputs(scope,ds,case,method):
    if scope=='large':
        cfg=json.loads((HISTORY/'config.json').read_text());cfg['batch']=8192
        stage=len(STAGES[ds])-1;data,identity=stage_data(HISTORY,cfg,ds,case,stage)
        base=Path(cfg['source'])/'data'/ds;source=HISTORY/'jobs'/f'{ds}_{case}_{method}'/f'stage_{stage}'
        mapping=np.array(identity['global_class_indices'],dtype='int16')
        refs=json.loads((LARGE/'results.json').read_text())['jobs']
        ref=next(j for j in refs if j['dataset']==ds and j['case']==case and j['method']==method)
        names=json.loads((base/'manifest.json').read_text())['class_names']
        model_file=source/('model.tabpfn_fit' if method=='tabpfn' else 'model.ubj')
    else:
        cfg=json.loads((SMALL/'config.json').read_text());meta=json.loads((SMALL/'data'/ds/'manifest.json').read_text());base=Path(meta['source']);names=meta['class_names']
        label='pfn_pretrained' if method=='tabpfn' else method;job=SMALL/'jobs'/f'{ds}_{label}'
        ref=json.loads((job/'results.json').read_text());ref['predictions_path']=str(job/'predictions.npy')
        ids=np.array(ref['fits'][-1]['context_indices']);ytr=np.load(base/'train_y.npy',mmap_mode='r')[ids];mapping=np.unique(ytr).astype('int16')
        data=dict(names=[names[c] for c in mapping],Xtr=np.load(base/'train_X.npy',mmap_mode='r')[ids],ytr=np.searchsorted(mapping,ytr))
        source=sorted(job.glob('stage_*'))[-1]
        model_file=job/('final_model.tabpfn_fit' if method=='tabpfn' else 'final_model.ubj')
    X=np.load(base/'test_X.npy',mmap_mode='r');y=np.load(base/'test_y.npy',mmap_mode='r')
    return cfg,data,names,mapping,X,y,source,model_file,ref


def order_for(out,ds,total):
    dest=out/'data'/ds;dest.mkdir(parents=True,exist_ok=True);path=dest/'query_order.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    order=np.random.default_rng(43).permutation(total).astype('int64');np.save(path,order)
    write(dest/'manifest.json',dict(seed=43,rows=total,query_order_sha256=sha(path),labels_used_to_order=False))
    return order


def pfn_pair(out,scope,ds,case):
    import torch
    from tabpfn import TabPFNClassifier
    dest=out/'jobs'/f'{scope}_{ds}_{case}_tabpfn';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    cfg,data,names,mapping,X,y,source,model_file,ref=inputs(scope,ds,case,'tabpfn')
    n=len(y);order=order_for(out,ds,n);batch=8192;C=len(mapping)
    progress(out,scope=scope,dataset=ds,case=case,method='tabpfn',phase='restore')
    model=TabPFNClassifier.load_from_fit_state(model_file,device='cuda:0');model_hash=sha(model_file)
    prior=np.bincount(data['ytr'],minlength=C)/len(data['ytr'])
    def mmap(name,dtype,shape,fill):
        path=dest/(name+'.npy')
        if path.exists():return np.load(path,mmap_mode='r+')
        a=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape);a[:]=fill;a.flush();return a
    probabilities=mmap('probabilities','float32',(n,C),np.nan)
    pred=mmap('predictions','int16',(n,),-1);dist=mmap('dist_predictions','int16',(n,),-1)
    seconds=mmap('service_times','float64',((n+batch-1)//batch,3),np.nan)
    for i,a in enumerate(range(0,n,batch)):
        if np.isfinite(seconds[i,2]):continue
        b=min(a+batch,n);ids=order[a:b]
        p,elapsed=timed(lambda:model.predict_proba(np.asarray(X[ids])))
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-3)
        q=dist_adjust_current_batch(p,prior)
        probabilities[ids]=p;pred[ids]=mapping[p.argmax(1)];dist[ids]=mapping[q.argmax(1)]
        probabilities.flush();pred.flush();dist.flush();seconds[i]=[a,b,elapsed];seconds.flush()
        if i%20==0 or b==n:
            done=np.isfinite(seconds[:,2]);rate=np.sum(seconds[done,1]-seconds[done,0])/np.sum(seconds[done,2])
            progress(out,scope=scope,dataset=ds,case=case,method='tabpfn',phase='mixed_test',rows_done=b,rows_total=n,remaining_seconds=float((n-b)/rate))
    assert np.isfinite(probabilities).all() and (pred>=0).all() and (dist>=0).all()
    grouped=mmap('dist_original_groups','int16',(n,),-1)
    for a in range(0,n,batch):
        b=min(a+batch,n);grouped[a:b]=mapping[dist_adjust_current_batch(np.asarray(probabilities[a:b]),prior).argmax(1)]
    grouped.flush()
    original=np.load(ref['predictions_path'],mmap_mode='r');mismatch=int(np.count_nonzero(pred!=original))
    base=dict(ref,scope=scope,case=case,method='pfn_pretrained' if scope=='small' else 'tabpfn',
        evaluation='final_full_test',test_order='seed43_mixed',prediction_storage_order='original',batch=batch,
        metrics=confusion_metrics(y,np.asarray(pred),names),predictions_path=str(dest/'predictions.npy'),
        full_reinference=True,training_rerun=False,original_order_prediction_mismatches=mismatch,
        source_model=str(model_file),source_model_sha256=model_hash,query_order_sha256=sha(out/'data'/ds/'query_order.npy'),
        probability_path=str(dest/'probabilities.npy'),class_mapping=mapping.tolist(),training_prior=prior.tolist(),
        source_result_scope='EXP77' if scope=='small' else 'EXP79',initial_finetuning=False)
    write(dest/'results.json',base)
    dj=dict(base,method='distpfn_pretrained' if scope=='small' else 'distpfn',metrics=confusion_metrics(y,np.asarray(dist),names),
        predictions_path=str(dest/'dist_predictions.npy'),correction_scope='current mixed query batch only',
        original_group_counterfactual=confusion_metrics(y,np.asarray(grouped),names),
        original_group_predictions_path=str(dest/'dist_original_groups.npy'))
    write(dest/'dist_results.json',dj);write(dest/'COMPLETE.json',dict(complete=True))
    del model;gc.collect();torch.cuda.empty_cache()


def static_model(out,scope,ds,case,method):
    import torch
    dest=out/'jobs'/f'{scope}_{ds}_{case}_{method}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    cfg,data,names,mapping,X,y,source,model_file,ref=inputs(scope,ds,case,method);n=len(y);order=order_for(out,ds,n)
    progress(out,scope=scope,dataset=ds,case=case,method=method,phase='mixed_audit')
    if method=='ours':model=CurrentPredictor(source,cfg);predict=model.predict
    elif method=='xgb' and scope=='small':
        from xgboost import XGBClassifier
        model=XGBClassifier();model.load_model(model_file);model.set_params(device='cuda:0',n_jobs=8);predict=model.predict_proba
    else:model=Predictor(method,cfg,ds,data,source);predict=model.predict
    original=np.load(ref['predictions_path'],mmap_mode='r');ids=order[:8192];p=predict(np.asarray(X[ids]))
    mismatches=int(np.count_nonzero(mapping[p.argmax(1)]!=original[ids]))
    result=dict(ref,scope=scope,case=case,test_order='seed43_mixed',prediction_storage_order='original',
        full_reinference=False,training_rerun=False,mixed_audit_rows=len(ids),mixed_audit_mismatches=mismatches,
        query_order_sha256=sha(out/'data'/ds/'query_order.npy'),
        reuse_reason='Row-wise predictor without batch posterior correction; mixed-query audit matched original predictions')
    if mismatches:
        pred=np.lib.format.open_memmap(dest/'predictions.npy',mode='w+',dtype='int16',shape=(n,));aux={}
        if method=='ours':aux={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype='int16',shape=(n,)) for k in ['global_pred','calls','accepted']}
        for a in range(0,n,8192):
            b=min(a+8192,n);ids=order[a:b];p=predict(np.asarray(X[ids]));pred[ids]=mapping[p.argmax(1)]
            for k,v in aux.items():v[ids]=mapping[model.aux[k]] if k=='global_pred' else model.aux[k]
            if a%(8192*20)==0:progress(out,scope=scope,dataset=ds,case=case,method=method,phase='fallback_full_test',rows_done=b,rows_total=n)
        pred.flush();result.update(full_reinference=True,reuse_reason=None,predictions_path=str(dest/'predictions.npy'),metrics=confusion_metrics(y,np.asarray(pred),names))
        if method=='ours':
            for v in aux.values():v.flush()
            result['global_metrics']=confusion_metrics(y,np.asarray(aux['global_pred']),names)
            result['policy']=dict(called=int(aux['calls'].sum()),accepted=int(aux['accepted'].sum()),helpful=int(((pred==y)&(aux['global_pred']!=y)).sum()),harmful=int(((pred!=y)&(aux['global_pred']==y)).sum()))
    write(dest/'results.json',result);write(dest/'COMPLETE.json',dict(complete=True))
    if hasattr(model,'close'):model.close()
    del model;gc.collect();torch.cuda.empty_cache()


def budget_audits(out):
    import torch
    from tabpfn import TabPFNClassifier
    audits=[]
    for root,levels in [(OLD_BUDGET,[10,20,50]),(EXPANDED,[200,400])]:
        jobs=json.loads((root/'results.json').read_text())['jobs']
        for ds in ['cic2018','toniot']:
            meta=json.loads((SMALL/'data'/ds/'manifest.json').read_text());base=Path(meta['source']);X=np.load(base/'test_X.npy',mmap_mode='r');order=order_for(out,ds,len(X))
            for n in levels:
                j=next(j for j in jobs if j['dataset']==ds and j['new_per_class']==n and j['method']=='pfn_pretrained')
                model=TabPFNClassifier.load_from_fit_state(Path(j['predictions_path']).parent/'final_model.tabpfn_fit',device='cuda:0')
                # These jobs encode the complete alphabetically ordered global classes.
                p=model.predict_proba(np.asarray(X[order[:8192]]));original=np.load(j['predictions_path'],mmap_mode='r')
                mismatches=int(np.count_nonzero(p.argmax(1)!=original[order[:8192]]))
                assert mismatches==0,('expanded/reduced budget order dependence',ds,n,mismatches)
                audits.append(dict(dataset=ds,new_per_class=n,rows=8192,mismatches=0,source_predictions=j['predictions_path']))
                del model;gc.collect();torch.cuda.empty_cache()
    write(out/'budget_order_audits.json',dict(passed=True,audits=audits))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
    write(out/'config.json',dict(seed=43,seed_repetitions=1,batch=8192,training_rerun=False,test_order='seed43_mixed',sources=[str(SMALL),str(LARGE),str(EXPANDED)]))
    progress(out,phase='waiting_for_gpu',after_run=str(EXPANDED))
    while True:
        meta=json.loads((EXPANDED/'RUN_META.json').read_text())
        if meta['status']=='finished':assert meta['returncode']==0,meta;break
        time.sleep(10)
    import torch
    torch.set_num_threads(8);torch.manual_seed(43)
    dest=out/'source';dest.mkdir(exist_ok=True);shutil.copy2(__file__,dest/Path(__file__).name)
    for scope in ['small','large']:
        for ds in ['cic2018','toniot']:
            for case in (['new_100'] if scope=='small' else ['natural','uniform']):
                pfn_pair(out,scope,ds,case)
                for method in ['xgb','ours','boostpfn']:static_model(out,scope,ds,case,method)
                jobs=[json.loads(p.read_text()) for p in sorted((out/'jobs').glob('*/*results.json'))]
                write(out/'results.json',dict(state='running',jobs=jobs))
    budget_audits(out)
    jobs=[json.loads(p.read_text()) for p in sorted((out/'jobs').glob('*/*results.json'))];assert len(jobs)==30
    write(out/'results.json',dict(state='complete',jobs=jobs))


if __name__=='__main__':main()

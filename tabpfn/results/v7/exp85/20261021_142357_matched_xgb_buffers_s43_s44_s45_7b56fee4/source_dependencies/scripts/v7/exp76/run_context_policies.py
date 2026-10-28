#!/usr/bin/env python3
"""Small-context accumulation, replacement, and fixed-context replay.

Uses the existing class-arrival sequence and ALL original fixed-test rows.
The new label budget is explicit, separate from the historical 100k experiment.
"""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)
import argparse
import gc
import hashlib
import json
import time
import numpy as np
from exp61_utils import write, Cost, sha
from exp71_bounded_updates import get_model
from replay_protocol import partition_test, STAGES, confusion_metrics, dist_adjust_current_batch
from replay_models import timed, local_request
import exp61_sota_worker as sota

SOURCE = ROOT/'tabpfn/results/v7/exp71/20261014_103108_cumulative_100k_uniform_natural_s43_43a766d9'
METHODS = ['xgb','pfn_accumulate','pfn_fifo','pfn_similar','pfn_random','pfn_fixed','boostpfn']


def select_initial(y, names, ds, budget, seed):
    rng = np.random.default_rng(seed)
    group = STAGES[ds][0]
    benign = int(round(budget*.75))
    remaining = budget-benign
    ids = []
    for i, name in enumerate(group):
        n = benign if name == 'benign' else remaining//2 + (remaining%2 if i == 1 else 0)
        pool = np.flatnonzero(y == names.index(name))
        ids.extend(rng.choice(pool, n, replace=False).tolist())
    return rng.permutation(ids).astype('int64')


def replace_context(old, new, y, X, policy, seed):
    """Evict current rows only, protecting at least one row of every old class.

    FIFO is acquisition order, not observed obsolescence. Similarity scaling is
    fitted on the old context and queries are the newly released train rows.
    """
    if len(new) >= len(old):
        raise ValueError('Incoming batch must leave room for old-class coverage')
    if policy == 'fifo': order = np.arange(len(old))
    elif policy == 'random': order = np.random.default_rng(seed).permutation(len(old))
    elif policy == 'similar':
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import pairwise_distances
        scaler = StandardScaler().fit(X[old])
        dist = pairwise_distances(scaler.transform(X[old]), scaler.transform(X[new]), metric='euclidean')
        order = np.argsort(dist.min(axis=1), kind='stable')
    else: raise ValueError(policy)
    counts = np.bincount(y[old], minlength=int(y.max())+1)
    remove = []
    for i in order:
        c = y[old[i]]
        if counts[c] <= 1: continue
        remove.append(int(i)); counts[c] -= 1
        if len(remove) == len(new): break
    if len(remove) != len(new): raise ValueError('Insufficient removable rows')
    keep = np.ones(len(old),bool);keep[remove] = False
    return np.r_[old[keep],new],old[remove]


def prepare(out, cfg, ds):
    base = SOURCE/'data'/ds
    meta = json.loads((base/'manifest.json').read_text());names=meta['class_names']
    arr={k:np.load(base/f'{k}.npy',mmap_mode='r') for k in ['train_X','train_y','train_ids','train_hash','test_X','test_y','test_ids','test_hash']}
    d=out/'data'/ds;d.mkdir(parents=True,exist_ok=True)
    y=arr['train_y'];rng=np.random.default_rng(cfg['seed'])
    initial=select_initial(y,names,ds,cfg['context_budget'],cfg['seed'])
    released=initial.copy();incoming=[initial];plans=[]
    for stage,group in enumerate(STAGES[ds]):
        if stage:
            ids=np.concatenate([rng.choice(np.flatnonzero(y==names.index(n)),cfg['new_per_class'],replace=False) for n in group])
            incoming.append(ids);released=np.r_[released,ids]
        plans.append(dict(stage=stage,new_classes=[] if stage==0 else group,released=released.tolist(),incoming=incoming[stage].tolist()))
    assert len(np.unique(released)) == len(released)
    assert not np.intersect1d(arr['train_ids'][released],arr['test_ids']).size
    order,batches,bounds=partition_test(arr['test_y'],names,ds,cfg['seed'],cfg['batch'])
    np.save(d/'order.npy',order);np.save(d/'batches.npy',batches);np.save(d/'bounds.npy',bounds)
    write(d/'plans.json',plans)
    write(d/'manifest.json',dict(source=str(base),source_manifest_sha256=sha(base/'manifest.json'),class_names=names,
        initial_rows=len(initial),new_per_class=cfg['new_per_class'],final_released_rows=len(released),test_rows=len(order),
        test_full=True,initial_benign_fraction=.75,initial_attack_allocation='equal between the two initial attack classes',
        plans_sha256=sha(d/'plans.json'),order_sha256=sha(d/'order.npy'),batch=cfg['batch'],seed=cfg['seed'],
        test_feature_overlap_with_released_rows=int(np.isin(arr['test_hash'],arr['train_hash'][released]).sum())))
    return arr,names,plans,order,batches,bounds


def metric(y,p,names):
    return confusion_metrics(np.asarray(y),np.asarray(p),names)


def run_method(out,cfg,ds,method,data):
    import torch
    arr,names,plans,order,batches,bounds=data
    job=out/'jobs'/f'{ds}_{method}';job.mkdir(parents=True,exist_ok=True)
    if (job/'COMPLETE.json').exists():return json.loads((job/'results.json').read_text())
    X=arr['train_X'];y=arr['train_y'];TX=arr['test_X'];TY=arr['test_y'];C=len(names)
    predictions=np.lib.format.open_memmap(job/'predictions.npy',mode='w+',dtype='int16',shape=(len(order),))
    distpred=np.lib.format.open_memmap(job/'dist_predictions.npy',mode='w+',dtype='int16',shape=(len(order),)) if method=='pfn_accumulate' else None
    model=None;old=None;fits=[];stages=[];services=[];distservices=[]
    for stage,plan in enumerate(plans):
        dest=job/f'stage_{stage}';dest.mkdir(exist_ok=True)
        print(json.dumps(dict(dataset=ds,method=method,stage=stage,phase='fit')),flush=True)
        write(out/'progress.json',dict(dataset=ds,method=method,stage=stage,phase='fit',updated_epoch=time.time()))
        removed=np.array([],dtype='int64');select_start=time.perf_counter()
        incoming=np.array(plan['incoming'],dtype='int64')
        if stage==0:ctx=incoming
        elif method in ('xgb','pfn_accumulate','boostpfn'):ctx=np.array(plan['released'],dtype='int64')
        elif method=='pfn_fixed':ctx=old
        else:ctx,removed=replace_context(old,incoming,y,X,method.removeprefix('pfn_'),cfg['seed']+stage*1009)
        selection_seconds=time.perf_counter()-select_start
        classes=np.unique(y[ctx]);lookup=np.full(C,-1,dtype='int16');lookup[classes]=np.arange(len(classes))
        cost=Cost(dest);fit_seconds=construct_seconds=0.;first_seconds=0.
        if stage==0 or method!='pfn_fixed':
            if method=='boostpfn':
                request=local_request(cfg,ds)
                boostdata=dict(Xtr=np.asarray(X[ctx]),ytr=lookup[y[ctx]],Xte=np.empty((0,X.shape[1]),dtype='float32'))
                with cost.phase('model_load'):base,model,split=sota.boost_factory(request)
                construct_seconds=cost.seconds('model_load')
                with cost.phase('fit'):
                    model.fit(torch.from_numpy(boostdata['Xtr']),torch.from_numpy(boostdata['ytr'].astype('int64')),torch.empty((0,X.shape[1])))
                fit_seconds=cost.seconds('fit')
                np.savez_compressed(dest/'boost_state.npz',sampled_idxs=np.asarray(model.sampled_idxs),alphas=np.asarray(model.alphas))
            else:
                # Reuse the loaded PFN weights across updates, as in EXP72.
                if model is None or method=='xgb':
                    with cost.phase('construct'):
                        model=get_model('xgb' if method=='xgb' else 'tabpfn',argparse.Namespace(threads=cfg['threads'],seed=cfg['seed'],checkpoint=Path(cfg['checkpoint'])),len(classes))
                    construct_seconds=cost.seconds('construct')
                with cost.phase('fit'):model.fit(X[ctx],lookup[y[ctx]])
                fit_seconds=cost.seconds('fit')
        def predict(A):
            if method=='boostpfn':return sota.replay_boost(base,model,split,boostdata['Xtr'],boostdata['ytr'],A,cfg['batch'])
            return model.predict_proba(A)
        lo,hi=bounds[stage]
        # All first-batch prediction work remains serving cost; readiness fit is explicit.
        prior=np.bincount(lookup[y[ctx]],minlength=len(classes))/len(ctx)
        stage_seconds=0.;stage_dist=0.
        for number,(_,a,b) in enumerate(batches[batches[:,0]==stage]):
            def serve():return predict(np.asarray(TX[order[a:b]]))
            p,seconds=timed(serve);stage_seconds+=seconds
            if number==0:first_seconds=seconds
            if p.shape!=(b-a,len(classes)) or not np.isfinite(p).all():raise ValueError('Invalid probabilities')
            predictions[a:b]=classes[p.argmax(1)];services.append([stage,int(a),int(b),seconds])
            if distpred is not None:
                adjusted,seconds2=timed(lambda:dist_adjust_current_batch(p,prior))
                distpred[a:b]=classes[adjusted.argmax(1)];stage_dist+=seconds2
                distservices.append([stage,int(a),int(b),seconds+seconds2])
            if number%100==0:write(out/'progress.json',dict(dataset=ds,method=method,stage=stage,phase='predict',rows_done=int(b),rows_total=len(order),updated_epoch=time.time()))
        resources=cost.finish();predictions.flush()
        info=dict(stage=stage,classes=[names[c] for c in classes],context_rows=len(ctx),released_rows=len(plan['released']),
            context_indices=ctx.tolist(),removed_indices=removed.tolist(),selection_seconds=selection_seconds,
            construct_seconds=construct_seconds,fit_seconds=fit_seconds,
            update_seconds=0. if stage and method=='pfn_fixed' else selection_seconds+construct_seconds+fit_seconds,
            first_batch_seconds=first_seconds,serving_seconds=stage_seconds,
            sampled_gpu_peak_gib=resources['peak_gpu_sampled_gib'],torch_peak_gib=resources['peak_torch_allocated_gib'],rss_peak_gib=resources['peak_rss_gib'])
        write(dest/'fit_metrics.json',info);fits.append(info)
        row=metric(TY[order[lo:hi]],np.asarray(predictions[lo:hi]),names);row.update(stage=stage,new_classes=plan['new_classes'])
        write(dest/'metrics.json',row);stages.append(row);old=ctx
    p=np.asarray(predictions);all_y=np.asarray(TY[order])
    totals=metric(all_y,p,names)
    # Disjoint-feature diagnostics use the same population for every policy.
    final_released=np.array(plans[-1]['released']);mask=~np.isin(arr['test_hash'][order],arr['train_hash'][final_released])
    result=dict(dataset=ds,method=method,seed=cfg['seed'],metrics=totals,stages=stages,fits=fits,
        update_seconds=sum(f['update_seconds'] for f in fits[1:]),initial_seconds=fits[0]['update_seconds'],
        serving_seconds=sum(s[3] for s in services),test_full=True,
        feature_disjoint_diagnostic=metric(all_y[mask],p[mask],names),
        note='Instant-ready class-arrival replay; measured update latency reported separately, not converted to delayed predictions')
    np.save(job/'service_times.npy',np.array(services));write(job/'results.json',result)
    if distpred is not None:
        distpred.flush();dist=dict(result);dist.update(method='distpfn',metrics=metric(all_y,np.asarray(distpred),names),
            feature_disjoint_diagnostic=metric(all_y[mask],np.asarray(distpred)[mask],names),
            serving_seconds=sum(s[3] for s in distservices),
            stages=[dict(metric(TY[order[lo:hi]],np.asarray(distpred[lo:hi]),names),stage=i,new_classes=plans[i]['new_classes']) for i,(lo,hi) in enumerate(bounds)],
            note='Same cumulative TabPFN backbone/context and preparation cost, with current-batch posterior adjustment; no future test distribution')
        write(job/'dist_results.json',dist)
    write(job/'COMPLETE.json',dict(complete=True,predictions_sha256=sha(job/'predictions.npy')))
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--context-budget',type=int,default=1000);p.add_argument('--new-per-class',type=int,default=100)
    p.add_argument('--datasets',nargs='+',default=['cic2018','toniot']);p.add_argument('--methods',nargs='+',default=METHODS)
    p.add_argument('--batch',type=int,default=8192);p.add_argument('--seed',type=int,default=43)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=dict(context_budget=a.context_budget,new_per_class=a.new_per_class,datasets=a.datasets,methods=a.methods,
        batch=a.batch,seed=a.seed,threads=8,checkpoint=str(ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'),
        source=str(SOURCE),protocol_revision='exp76.1',html_publication=False,backbone_finetuning=False,
        label_budget_note='100 is an experimental choice, not the user example 20; initial 75% benign, remaining two initial attacks equal')
    write(a.out/'config.json',cfg)
    import torch
    torch.set_num_threads(cfg['threads']);torch.manual_seed(a.seed)
    jobs=[]
    for ds in a.datasets:
        data=prepare(a.out,cfg,ds)
        for method in a.methods:
            jobs.append(run_method(a.out,cfg,ds,method,data))
            dp=a.out/'jobs'/f'{ds}_{method}'/'dist_results.json'
            if dp.exists():jobs.append(json.loads(dp.read_text()))
            write(a.out/'results.json',dict(state='running',jobs=jobs))
    write(a.out/'results.json',dict(state='complete',jobs=jobs))
    print('ALL COMPLETE',flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""No-FT DistPFN plus nested new-class label budgets, one fixed seed."""
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
from finetune_final import run_method,weights_digest
from replay_models import timed
from replay_protocol import confusion_metrics,dist_adjust_current_batch

SOURCE=ROOT/'tabpfn/results/v7/exp77/20261015_230535_initialft10_finalfull_s43_2eb2237d'
METHODS=['xgb','pfn_pretrained','pfn_ft_accumulate']


def load_source(ds):
    meta=json.loads((SOURCE/'data'/ds/'manifest.json').read_text())
    base=Path(meta['source'])
    arr={k:np.load(base/f'{k}.npy',mmap_mode='r') for k in ['train_X','train_y','train_hash','train_ids','test_X','test_y','test_hash','test_ids']}
    plans=json.loads((SOURCE/'data'/ds/'plans.json').read_text())
    return meta,arr,plans


def subset_plans(plans,y,n,seed):
    reduced=[];released=[]
    for s,old in enumerate(plans):
        p=copy.deepcopy(old);ids=np.asarray(old['incoming'],dtype=np.int64)
        if s:
            chosen=[]
            for c in np.unique(y[ids]):
                pool=ids[y[ids]==c]
                rng=np.random.default_rng(seed+s*104729+int(c)*1009)
                chosen.extend(rng.permutation(pool)[:n].tolist())
            # Retain original row ordering within the nested selected set.
            ids=ids[np.isin(ids,chosen)]
        p['incoming']=ids.tolist();released.extend(ids.tolist());p['released']=list(released)
        for key in ['ft_roles','ours_roles']:
            p[key]={r:[i for i in v if i in set(released)] for r,v in old[key].items()}
        p['new_per_class']=n;reduced.append(p)
    return reduced


def dist_from_saved(out,cfg,ds,meta,arr,plans):
    import torch
    from tabpfn import TabPFNClassifier
    dest=out/'dist_noft'/'jobs'/f'{ds}_distpfn_pretrained';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return json.loads((dest/'results.json').read_text())
    source=SOURCE/'jobs'/f'{ds}_pfn_pretrained'
    base=json.loads((source/'results.json').read_text())
    model=TabPFNClassifier.load_from_fit_state(source/'final_model.tabpfn_fit',device='cuda:0')
    before=weights_digest(model.models_[0].state_dict())
    assert before==base['fits'][-1]['weights_sha256']
    labels=arr['train_y'][plans[-1]['released']];classes=np.unique(labels)
    assert np.array_equal(model.classes_,np.arange(len(classes)))
    prior=np.bincount(np.searchsorted(classes,labels),minlength=len(classes))/len(labels)
    oldpred=np.load(source/'predictions.npy',mmap_mode='r');total=len(arr['test_y'])
    pred=np.lib.format.open_memmap(dest/'predictions.npy',mode='w+',dtype='int16',shape=(total,))
    times=[];mismatch=0
    for a in range(0,total,cfg['batch']):
        b=min(a+cfg['batch'],total)
        p,sec=timed(lambda:model.predict_proba(np.asarray(arr['test_X'][a:b])))
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-3)
        mismatch+=int(np.count_nonzero(classes[p.argmax(1)]!=oldpred[a:b]))
        q,adj=timed(lambda:dist_adjust_current_batch(p,prior))
        pred[a:b]=classes[q.argmax(1)];times.append([a,b,sec,adj])
        if (a//cfg['batch'])%50==0:
            progress=dict(phase='dist_noft',dataset=ds,rows_done=b,rows_total=total)
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert mismatch==0,(ds,'restored base differs',mismatch)
    assert before==weights_digest(model.models_[0].state_dict())
    pred.flush();np.save(dest/'service_times.npy',np.asarray(times))
    mask=~np.isin(arr['test_hash'],arr['train_hash'][plans[-1]['released']])
    result=copy.deepcopy(base)
    result.update(method='distpfn_pretrained',new_per_class=100,
        metrics=confusion_metrics(arr['test_y'],np.asarray(pred),meta['class_names']),
        feature_disjoint_diagnostic=confusion_metrics(arr['test_y'][mask],np.asarray(pred)[mask],meta['class_names']),
        serving_seconds=sum(t[2]+t[3] for t in times),adjustment_seconds=sum(t[3] for t in times),
        initial_finetuning_seconds=0.,source_result=str(source/'results.json'),
        predictions_path=str(dest/'predictions.npy'),preparation_cost_source=str(source/'results.json'),
        preparation_cost_reused=True,restored_base_prediction_mismatches=mismatch,
        weights_before_sha256=before,weights_after_sha256=before,
        note='Pretrained v3 + cumulative context + current-query-batch DistPFN correction; no FT; no query labels. Same context preparation as the saved EXP77 no-FT PFN. Posterior correction belongs to inference, not context-update time.')
    write(dest/'results.json',result);write(dest/'COMPLETE.json',dict(predictions_sha256=sha(dest/'predictions.npy')))
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'config.json').read_text())
    cfg.update(protocol='exp78.1',source=str(SOURCE),methods=METHODS,dist_methods=[],new_budgets=[100,50,20,10],
        seed_repetitions=1,initial_data_unchanged=True,initial_checkpoint_reused=True,
        budget_axis='newly introduced attack examples per class; initial 1000 fixed',
        displayed_cost='sum of stages after initial: context selection + construction + fit; excludes initial preparation and all inference')
    write(out/'config.json',cfg)
    import torch
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    files=[Path(__file__),ROOT/'scripts/v7/exp77/finetune_final.py',ROOT/'scripts/v7/exp72/replay_protocol.py',ROOT/'scripts/v7/exp72/replay_models.py',ROOT/'scripts/v7/exp71/exp71_bounded_updates.py']
    hashes={}
    for source in files:
        target=out/'source_dependencies'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);hashes[str(source.relative_to(ROOT))]=sha(source)
    write(out/'source_dependencies.json',hashes)
    old=json.loads((SOURCE/'results.json').read_text());jobs=[];dist=[]
    for ds in cfg['datasets']:
        meta,arr,plans=load_source(ds)
        dest=out/'data'/ds;dest.mkdir(parents=True,exist_ok=True)
        write(dest/'manifest.json',meta);write(dest/'plans.json',plans)
        dist.append(dist_from_saved(out,cfg,ds,meta,arr,plans))
        write(out/'dist_results.json',dict(state='complete' if len(dist)==2 else 'running',jobs=dist))
    for ds in cfg['datasets']:
        meta,arr,plans=load_source(ds)
        audit=json.loads((SOURCE/'finetuning'/ds/'initial'/'audit.json').read_text())
        checkpoint=SOURCE/'finetuning'/ds/'initial'/'weights.ckpt'
        for j in old['jobs']:
            if j['dataset']==ds and j['method'] in METHODS:
                item=copy.deepcopy(j);item.update(new_per_class=100,reused=True,
                    source_result=str(SOURCE/'jobs'/f"{ds}_{j['method']}"/'results.json'),
                    predictions_path=str(SOURCE/'jobs'/f"{ds}_{j['method']}"/'predictions.npy'))
                jobs.append(item)
        for n in [50,20,10]:
            case=out/f'new_{n}';case.mkdir(exist_ok=True)
            subset=subset_plans(plans,arr['train_y'],n,cfg['seed'])
            dest=case/'data'/ds;dest.mkdir(parents=True,exist_ok=True)
            write(dest/'plans.json',subset)
            write(dest/'manifest.json',dict(meta,new_per_class=n,final_released_rows=len(subset[-1]['released']),
                sampling='nested per-class permutation of EXP77 incoming rows; initial stage unchanged',
                new_plans_sha256=sha(dest/'plans.json')))
            write(case/'config.json',dict(cfg,new_per_class=n))
            for method in METHODS:
                progress=dict(phase='budget',dataset=ds,new_per_class=n,method=method)
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                job=run_method(case,cfg,ds,arr,meta['class_names'],subset,checkpoint,audit,method)
                item=copy.deepcopy(job);item.update(new_per_class=n,reused=False,
                    source_result=str(case/'jobs'/f'{ds}_{method}'/'results.json'),
                    predictions_path=str(case/'jobs'/f'{ds}_{method}'/'predictions.npy'))
                jobs.append(item)
                write(out/'results.json',dict(state='running',jobs=jobs,dist_jobs=dist))
    write(out/'results.json',dict(state='complete',jobs=jobs,dist_jobs=dist,expected_jobs=24,expected_dist_jobs=2))
    write(out/'progress.json',dict(phase='complete',jobs=len(jobs),dist_jobs=len(dist)))


if __name__=='__main__':main()

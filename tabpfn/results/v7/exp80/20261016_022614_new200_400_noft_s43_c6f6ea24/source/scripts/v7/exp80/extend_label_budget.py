#!/usr/bin/env python3
"""Extend new-class budgets to 200/400 with unchanged initial 1000 rows."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse,copy,json,shutil,time
import numpy as np
from exp61_utils import write,sha
from finetune_final import run_method,CHECKPOINT
from noft_dist_budget import load_source,SOURCE


def expanded_plans(original,y,n,seed):
    plans=[];released=[]
    for s,old in enumerate(original):
        p={k:copy.deepcopy(v) for k,v in old.items() if k not in ['ft_roles','ours_roles']}
        incoming=np.asarray(old['incoming'],dtype='int64')
        if s:
            extras=[]
            for c in np.unique(y[incoming]):
                existing=incoming[y[incoming]==c];assert len(existing)==100
                pool=np.flatnonzero((y==c)&~np.isin(np.arange(len(y)),incoming))
                assert len(pool)>=n-100,(s,c,n,len(pool))
                chosen=np.random.default_rng(seed+s*104729+int(c)*1009).permutation(pool)[:n-100]
                extras.extend(chosen.tolist())
            incoming=np.concatenate([incoming,np.asarray(extras,dtype='int64')])
        p['incoming']=incoming.tolist();released.extend(incoming.tolist());p['released']=list(released)
        p['new_per_class']=n;plans.append(p)
    return plans


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--after-run',type=Path);a=p.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'config.json').read_text());cfg.update(protocol='exp80.1',methods=['xgb','pfn_pretrained'],
        new_budgets=[200,400],dist_methods=[],initial_finetuning=False,seed_repetitions=1,source=str(SOURCE))
    write(out/'config.json',cfg)
    if a.after_run:
        write(out/'progress.json',dict(phase='waiting_for_gpu',after_run=str(a.after_run)))
        while True:
            meta=json.loads((a.after_run/'RUN_META.json').read_text())
            if meta['status']=='finished':
                assert meta['returncode']==0,meta
                break
            time.sleep(10)
    import torch
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    for source in [Path(__file__),ROOT/'scripts/v7/exp77/finetune_final.py',ROOT/'scripts/v7/exp71/exp71_bounded_updates.py']:
        dest=out/'source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    jobs=[]
    for ds in cfg['datasets']:
        meta,arr,original=load_source(ds)
        for n in cfg['new_budgets']:
            case=out/f'new_{n}';dest=case/'data'/ds;dest.mkdir(parents=True,exist_ok=True)
            plans=expanded_plans(original,arr['train_y'],n,cfg['seed'])
            assert len(plans[-1]['released'])==1000+n*({'cic2018':4,'toniot':7}[ds])
            assert len(set(plans[-1]['released']))==len(plans[-1]['released'])
            assert not np.intersect1d(arr['train_ids'][plans[-1]['released']],arr['test_ids']).size
            write(dest/'plans.json',plans)
            write(dest/'manifest.json',dict(meta,new_per_class=n,source_plans_sha256=sha(SOURCE/'data'/ds/'plans.json'),
                plans_sha256=sha(dest/'plans.json'),initial_data_unchanged=True,
                final_released_rows=len(plans[-1]['released']),validation_within_released_budget=False,validation_used=False))
            write(case/'config.json',dict(cfg,new_per_class=n))
            for method in cfg['methods']:
                write(out/'progress.json',dict(dataset=ds,new_per_class=n,method=method,phase='running',detail_progress=str(case/'progress.json')))
                print(json.dumps(dict(dataset=ds,new_per_class=n,method=method)),flush=True)
                job=run_method(case,cfg,ds,arr,meta['class_names'],plans,CHECKPOINT,None,method)
                job=dict(job,new_per_class=n,predictions_path=str(case/'jobs'/f'{ds}_{method}'/'predictions.npy'),
                    source_result=str(case/'jobs'/f'{ds}_{method}'/'results.json'))
                jobs.append(job);write(out/'results.json',dict(state='running',jobs=jobs))
    write(out/'results.json',dict(state='complete',jobs=jobs))


if __name__=='__main__':main()

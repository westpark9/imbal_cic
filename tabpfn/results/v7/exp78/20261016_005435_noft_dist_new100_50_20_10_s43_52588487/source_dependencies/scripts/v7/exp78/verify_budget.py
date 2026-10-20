#!/usr/bin/env python3
"""Verify saved predictions, nested label budgets, costs, and frozen weights."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,json
import numpy as np
from sklearn.metrics import confusion_matrix
from exp61_utils import write,sha


def verify(run,dist_only=False):
    cfg=json.loads((run/'config.json').read_text());source=Path(cfg['source'])
    dist=json.loads((run/'dist_results.json').read_text());assert dist['state']=='complete'
    result=None if dist_only else json.loads((run/'results.json').read_text())
    if result:
        assert result['state']=='complete' and len(result['jobs'])==24 and len(result['dist_jobs'])==2
        assert {(j['dataset'],j['method'],j['new_per_class']) for j in result['jobs']}=={
            (ds,m,n) for ds in cfg['datasets'] for m in cfg['methods'] for n in cfg['new_budgets']}
    checked=[]
    for ds in cfg['datasets']:
        meta=json.loads((source/'data'/ds/'manifest.json').read_text());data=Path(meta['source'])
        names=meta['class_names'];y=np.load(data/'test_y.npy',mmap_mode='r');train=np.load(data/'train_y.npy',mmap_mode='r')
        train_ids=np.load(data/'train_ids.npy',mmap_mode='r');test_ids=np.load(data/'test_ids.npy',mmap_mode='r')
        original=json.loads((source/'data'/ds/'plans.json').read_text())
        initial=json.loads((source/'finetuning'/ds/'initial'/'audit.json').read_text())
        base=json.loads((source/'jobs'/f'{ds}_pfn_pretrained'/'results.json').read_text())
        plans_by_budget={100:original}
        if result:
            for n in [50,20,10]:
                plans=json.loads((run/f'new_{n}'/'data'/ds/'plans.json').read_text());plans_by_budget[n]=plans
                subsetmeta=json.loads((run/f'new_{n}'/'data'/ds/'manifest.json').read_text())
                assert subsetmeta['plans_sha256']==sha(run/f'new_{n}'/'data'/ds/'plans.json')
                assert subsetmeta['source_plans_sha256']==sha(source/'data'/ds/'plans.json')
                matching=[j for j in result['jobs'] if j['dataset']==ds and j['new_per_class']==n]
                assert all(subsetmeta['test_feature_overlap_with_released_rows']==j['test_rows']-j['feature_disjoint_diagnostic']['rows'] for j in matching)
                assert plans[0]==dict(original[0],new_per_class=n)
                for s,plan in enumerate(plans):
                    assert set(plan['released'])==set(sum([p['incoming'] for p in plans[:s+1]],[]))
                    assert len(plan['released'])==len(set(plan['released']))
                    assert set(plan['incoming'])<=set(original[s]['incoming'])
                    if s:
                        labels,counts=np.unique(train[plan['incoming']],return_counts=True)
                        assert set(labels)==set(train[original[s]['incoming']]) and np.all(counts==n)
                assert len(plans[-1]['released'])==1000+n*({'cic2018':4,'toniot':7}[ds])
                assert not np.intersect1d(train_ids[plans[-1]['released']],test_ids).size
            for a,b in [(10,20),(20,50),(50,100)]:
                for pa,pb in zip(plans_by_budget[a],plans_by_budget[b]):assert set(pa['incoming'])<=set(pb['incoming'])
        jobs=[j for j in dist['jobs'] if j['dataset']==ds]
        if result:jobs += [j for j in result['jobs'] if j['dataset']==ds]
        for job in jobs:
            pred=np.load(job['predictions_path'],mmap_mode='r');method=job['method'];n=job['new_per_class']
            assert len(pred)==len(y)==job['test_rows'] and job['evaluation']=='final_full_test'
            cm=confusion_matrix(y,pred,labels=np.arange(len(names)));np.testing.assert_array_equal(cm,job['metrics']['confusion'])
            denom=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denom,out=np.zeros(len(names)),where=denom!=0)
            assert np.isclose(f1.mean(),job['metrics']['macro_f1'],rtol=0,atol=1e-12)
            assert job['label_budget']==len(plans_by_budget[n][-1]['released'])
            assert np.isclose(job['update_seconds'],sum(f['update_seconds'] for f in job['fits'][1:]))
            for s,fit in enumerate(job['fits']):
                assert fit['context_indices']==plans_by_budget[n][s]['released']
                if method=='pfn_ft_accumulate':assert fit['weights_sha256']==initial['after_sha256']
                if method in ['pfn_pretrained','distpfn_pretrained']:assert fit['weights_sha256']==base['fits'][0]['weights_sha256']
                if s:assert fit['finetuning_seconds']==0
            if method=='distpfn_pretrained':
                assert job['restored_base_prediction_mismatches']==0
                assert job['weights_before_sha256']==job['weights_after_sha256']==base['fits'][-1]['weights_sha256']
                assert job['initial_finetuning_seconds']==0
                assert job['update_seconds']==base['update_seconds'] and job['fits']==base['fits']
            checked.append(dict(dataset=ds,method=method,new_per_class=n,macro_f1=float(f1.mean()),predictions_sha256=sha(job['predictions_path'])))
    report=dict(passed=True,seed=43,seed_repetitions=1,jobs=checked,
        checks=['independent full-test confusion and F1','final-state evaluation',
        'DistPFN no FT and exact base-prediction match','cost is later-stage updates only',
        'initial 1000 unchanged','nested same-ID class budgets','all context updates preserve weights'])
    write(run/('dist_verification.json' if dist_only else 'verification.json'),report)
    print(json.dumps(dict(passed=True,jobs=len(checked))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--dist-only',action='store_true');a=p.parse_args();verify(a.run,a.dist_only)

#!/usr/bin/env python3
"""Independent prediction/label-budget/weight-update audit for EXP77."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,json
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from exp61_utils import write,sha


def verify(run):
    result=json.loads((run/'results.json').read_text());cfg=json.loads((run/'config.json').read_text())
    assert result['state']=='complete' and len(result['jobs'])==result['expected_jobs']
    verified=[]
    for ds in cfg['datasets']:
        meta=json.loads((run/'data'/ds/'manifest.json').read_text());source=Path(meta['source'])
        plans=json.loads((run/'data'/ds/'plans.json').read_text());names=meta['class_names']
        y=np.load(source/'test_y.npy');ty=np.load(source/'train_y.npy');th=np.load(source/'train_hash.npy')
        assert not np.intersect1d(np.load(source/'train_ids.npy')[plans[-1]['released']],np.load(source/'test_ids.npy')).size
        initial=json.loads((run/'finetuning'/ds/'initial'/'audit.json').read_text())
        assert initial['changed_tensors']>0 and initial['before_sha256']!=initial['after_sha256']
        assert initial['train_rows']+initial['validation_rows']==1000
        for plan in plans:
            for rolekey in ['ft_roles','ours_roles']:
                roles=plan[rolekey];assert sorted(sum(roles.values(),[]))==sorted(plan['released'])
                for i,r in enumerate(roles):
                    for q in list(roles)[i+1:]:assert not np.intersect1d(th[roles[r]],th[roles[q]]).size
        for job in [j for j in result['jobs'] if j['dataset']==ds]:
            method=job['method'];parent='pfn_ft_accumulate' if method=='distpfn_ft' else method
            d=run/'jobs'/f'{ds}_{parent}'
            pred=np.load(d/('dist_predictions.npy' if method=='distpfn_ft' else 'predictions.npy'))
            assert len(pred)==len(y)==job['test_rows']
            assert job['evaluation']=='final_full_test' and job['final_stage']==len(plans)-1
            value=f1_score(y,pred,labels=np.arange(len(names)),average='macro',zero_division=0)
            assert np.isclose(value,job['metrics']['macro_f1'],rtol=0,atol=1e-12)
            cm=confusion_matrix(y,pred,labels=np.arange(len(names)))
            assert cm.sum()==len(y)
            np.testing.assert_array_equal(cm,job['metrics']['confusion'])
            benign=names.index('benign')
            assert np.isclose(1-cm[benign,benign]/cm[benign].sum(),job['metrics']['benign_false_positive_rate'])
            assert job['label_budget']==len(plans[-1]['released'])==({'cic2018':1400,'toniot':1700}[ds])
            assert np.isclose(job['update_seconds'],sum(f['update_seconds'] for f in job['fits'][1:]))
            previous=set();discarded=set();previous_digest=initial['after_sha256']
            for s,fit in enumerate(job['fits']):
                ctx=np.asarray(fit['context_indices']);released=plans[s]['released']
                assert set(ctx).issubset(released) and len(np.unique(ctx))==len(ctx)
                assert not set(ctx)&discarded
                if method in ('pfn_ft_fifo','pfn_ft_similar','pfn_ft_random'):
                    assert len(ctx)==1000 and set(ty[ctx])==set(ty[released])
                    if s:
                        removed=set(fit['removed_indices']);incoming=set(plans[s]['incoming'])
                        assert len(removed)==len(incoming) and removed<=previous
                        assert set(ctx)==(previous-removed)|incoming
                if method=='pfn_ft_fixed':assert set(ctx)==set(plans[0]['incoming'])
                if method.startswith('pfn_ft_') or method=='distpfn_ft':
                    expected=initial['after_sha256']
                    if method=='pfn_ft_repeated' and s:
                        audit=json.loads((d/'finetuning'/f'stage_{s}'/'audit.json').read_text())
                        assert audit['before_sha256']==previous_digest and audit['changed_tensors']>0
                        expected=audit['after_sha256'];previous_digest=expected
                    assert fit['weights_sha256']==expected
                previous=set(ctx);discarded.update(fit['removed_indices'])
            if method=='pfn_ft_fixed':
                assert job['update_seconds']==0 and set(pred).issubset(set(ty[plans[0]['incoming']]))
            verified.append(dict(dataset=ds,method=method,test_rows=len(y),macro_f1=value,
                predictions_sha256=sha(d/('dist_predictions.npy' if method=='distpfn_ft' else 'predictions.npy'))))
    report=dict(passed=True,jobs=verified,checks=['final-state full-test predictions','independent sklearn macro-F1',
        'all validation labels within budget','feature-disjoint persistent training roles','initial weights changed',
        'context-only updates preserve weights','class-covering buffer eviction','fixed context cannot output future labels'])
    write(run/'verification.json',report);print(json.dumps(dict(passed=True,jobs=len(verified))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);verify(p.parse_args().run)

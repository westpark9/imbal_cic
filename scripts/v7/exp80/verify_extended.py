#!/usr/bin/env python3
"""Verify expanded sample budgets, identical IDs and frozen-weight updates."""
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


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
    cfg=json.loads((run/'config.json').read_text());source=Path(cfg['source']);result=json.loads((run/'results.json').read_text())
    assert result['state']=='complete' and len(result['jobs'])==8
    assert {(j['dataset'],j['method'],j['new_per_class']) for j in result['jobs']}=={(d,m,n) for d in cfg['datasets'] for m in cfg['methods'] for n in cfg['new_budgets']}
    checked=[]
    for ds in cfg['datasets']:
        meta=json.loads((source/'data'/ds/'manifest.json').read_text());base=Path(meta['source']);names=meta['class_names']
        train=np.load(base/'train_y.npy');train_ids=np.load(base/'train_ids.npy');test_ids=np.load(base/'test_ids.npy');y=np.load(base/'test_y.npy')
        original=json.loads((source/'data'/ds/'plans.json').read_text());plans_by_n={100:original}
        frozen=json.loads((source/'jobs'/f'{ds}_pfn_pretrained'/'results.json').read_text())['fits'][-1]['weights_sha256']
        for n in cfg['new_budgets']:
            plans=json.loads((run/f'new_{n}'/'data'/ds/'plans.json').read_text());plans_by_n[n]=plans
            metadata=json.loads((run/f'new_{n}'/'data'/ds/'manifest.json').read_text())
            assert metadata['plans_sha256']==metadata['new_plans_sha256']==sha(run/f'new_{n}'/'data'/ds/'plans.json')
            assert metadata['source_plans_sha256']==sha(source/'data'/ds/'plans.json')
            for s,plan in enumerate(plans):
                assert set(plan['released'])==set(sum([p['incoming'] for p in plans[:s+1]],[]))
                assert len(plan['released'])==len(set(plan['released']))
                if s:
                    labels,counts=np.unique(train[plan['incoming']],return_counts=True)
                    assert set(labels)==set(train[original[s]['incoming']]) and np.all(counts==n)
                else:
                    assert plan['incoming']==original[0]['incoming'] and plan['released']==original[0]['released']
            assert not np.intersect1d(train_ids[plans[-1]['released']],test_ids).size
            for j in [j for j in result['jobs'] if j['dataset']==ds and j['new_per_class']==n]:
                assert metadata['test_feature_overlap_with_released_rows']==j['test_rows']-j['feature_disjoint_diagnostic']['rows']
                assert j['label_budget']==1000+n*({'cic2018':4,'toniot':7}[ds])==len(plans[-1]['released'])
                pred=np.load(j['predictions_path']);assert len(pred)==len(y) and j['evaluation']=='final_full_test'
                cm=confusion_matrix(y,pred,labels=np.arange(len(names)));np.testing.assert_array_equal(cm,j['metrics']['confusion'])
                den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(len(names)),where=den!=0)
                assert abs(f1.mean()-j['metrics']['macro_f1'])<1e-12
                assert abs(j['update_seconds']-sum(f['update_seconds'] for f in j['fits'][1:]))<1e-10
                for s,f in enumerate(j['fits']):
                    assert f['context_indices']==plans[s]['released'] and f['finetuning_seconds']==0
                    if j['method']=='pfn_pretrained':assert f['weights_sha256']==frozen
                checked.append(dict(dataset=ds,new_per_class=n,method=j['method'],macro_f1=float(f1.mean()),predictions_sha256=sha(j['predictions_path'])))
        for lo,hi in [(100,200),(200,400)]:
            for a,b in zip(plans_by_n[lo],plans_by_n[hi]):assert set(a['incoming'])<=set(b['incoming'])
    write(run/'verification.json',dict(passed=True,jobs=checked,seed_repetitions=1,initial_1000_unchanged=True,nested_100_200_400=True,
        checks=['independent full-test confusion and F1','same IDs for both models','nested expanded budgets','frozen pretrained weights','later-stage update cost']))
    print(json.dumps(dict(passed=True,jobs=len(checked))))


if __name__=='__main__':main()

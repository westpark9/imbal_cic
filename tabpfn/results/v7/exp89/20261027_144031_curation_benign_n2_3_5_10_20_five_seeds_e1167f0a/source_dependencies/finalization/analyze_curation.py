"""Independently validate EXP89 inputs/predictions and export paired contrasts."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)

import argparse
import gc
import json
import pickle
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from exp61_utils import sha
from curation_benign import read,write,stamp

KEYS=['dataset','train_version','benign_training','seed','n','method']
MEASURES=['attack_macro_f1','all_macro_f1','benign_fpr','attack_detection_recall','attack_to_benign']


def independent_scores(cm,names):
    cm=np.asarray(cm)
    tp=np.diag(cm)
    denom=cm.sum(0)+cm.sum(1)
    f=np.divide(2*tp,denom,out=np.zeros(len(names)),where=denom!=0)
    b=names.index('benign')
    return f,float(np.delete(f,b).mean()),float(f.mean())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    cfg=read(run/'config.json')
    design=cfg['design']
    result=read(run/'results.json')
    assert result['state']=='complete' and len(result['jobs'])==result['expected_jobs']==400
    out=run/'analysis'
    out.mkdir(exist_ok=True)
    source=Path(read(run/'data/cic2018/manifest.json')['source_pickle'])
    assert sha(source)==read(run/'prepared.json')['source_sha256']
    with source.open('rb') as stream:
        suite=pickle.load(stream)
    families=np.asarray(suite['families'])
    input_checks=[]
    for ds in design['datasets']:
        d=run/'data'/ds
        meta=read(d/'manifest.json')
        names=meta['class_names']
        b=names.index('benign')
        test_ids=np.load(d/'test_ids.npy')
        ty=np.load(d/'test_y.npy')
        retained=np.load(d/'retained_mask.npy')
        clean_dir=Path(meta['clean_manifest']).parent
        np.testing.assert_array_equal(test_ids,np.load(clean_dir/'original_test_idx.npy'))
        np.testing.assert_array_equal(test_ids[retained],np.load(clean_dir/'test_idx.npy'))
        np.testing.assert_array_equal(families[test_ids],np.asarray(names)[ty])
        order=np.load(d/'query_order.npy')
        np.testing.assert_array_equal(order,np.random.default_rng(43).permutation(len(test_ids)))
        tx=np.load(d/'test_X.npy',mmap_mode='r')
        for start in range(0,len(test_ids),200000):
            np.testing.assert_array_equal(tx[start:start+200000],
                np.nan_to_num(np.asarray(suite['X'][test_ids[start:start+200000]],dtype='float32')))
        for version in design['train_versions']:
            pool=np.load(d/f'{version}_train_ids.npy')
            assert not np.isin(pool,test_ids).any()
            for seed in design['seeds']:
                s=d/version/f'seed_{seed}'
                ids=np.load(s/'source_ids.npy')
                x,y=np.load(s/'X.npy'),np.load(s/'y.npy')
                assert np.isin(ids,pool).all() and len(np.unique(ids))==len(ids)
                np.testing.assert_array_equal(families[ids],np.asarray(names)[y])
                np.testing.assert_array_equal(x,np.nan_to_num(np.asarray(suite['X'][ids],dtype='float32')))
                previous={False:set(),True:set()}
                for n in design['budgets']:
                    selections={}
                    for benign in design['benign_training']:
                        index=np.load(s/f'n{n}_benign{int(benign)}_indices.npy')
                        selected=ids[index]
                        labels=y[index]
                        expected=np.full(len(names),n)
                        if not benign:expected[b]=0
                        np.testing.assert_array_equal(np.bincount(labels,minlength=len(names)),expected)
                        assert previous[benign]<=set(selected)
                        previous[benign]=set(selected)
                        selections[benign]=(selected,labels)
                        for method in design['methods']:
                            job=run/'jobs'/f'{ds}_{version}_benign{int(benign)}_s{seed}_n{n}_{method}'
                            np.testing.assert_array_equal(selected,np.load(job/'train_source_ids.npy'))
                    np.testing.assert_array_equal(selections[False][0],
                        selections[True][0][selections[True][1]!=b])
        input_checks.append(dict(dataset=ds,original_test_rows=len(test_ids),clean_test_rows=int(retained.sum()),
            raw_source_features_labels=True,original_split_unchanged=True,train_test_id_disjoint=True,
            matched_models=True,benign_addition_preserves_attack_ids=True,nested_budgets=True,
            label_free_fixed_query_order=True))
    del suite,families
    gc.collect()
    metrics=[];per_class=[];checks=[];cost=[];continuity=[]
    for ds in design['datasets']:
        d=run/'data'/ds
        meta=read(d/'manifest.json')
        names=meta['class_names'];C=len(names);b=names.index('benign')
        ty=np.load(d/'test_y.npy');retained=np.load(d/'retained_mask.npy')
        for j in [j for j in result['jobs'] if j['dataset']==ds]:
            path=Path(j['predictions_path'])
            assert sha(path)==read(path.parent/'COMPLETE.json')['predictions_sha256']
            assert sha(path.parent/'train_source_ids.npy')==j['train_ids_sha256']
            assert j['query_order_sha256']==meta['query_order_sha256']
            p=np.load(path)
            assert p.shape==ty.shape and p.min()>=0 and p.max()<C
            if not j['benign_training']:assert not np.any(p==b)
            original=confusion_matrix(ty,p,labels=np.arange(C))
            clean=confusion_matrix(ty[retained],p[retained],labels=np.arange(C))
            for tv,cm in [('original',original),('clean',clean)]:
                for scope in ['all','attack']:
                    current=cm.copy()
                    if scope=='attack':current[b,:]=0
                    saved=j['metrics'][tv+'_'+scope]
                    np.testing.assert_array_equal(current,saved['confusion'])
                    f,attack_f,all_f=independent_scores(current,names)
                    assert abs(attack_f-saved['attack_macro_f1'])<1e-12
                    assert abs(all_f-saved['all_macro_f1'])<1e-12
                    np.testing.assert_allclose(f,[c['f1'] for c in saved['classes']],rtol=0,atol=1e-12)
                    support=current.sum(1);tp=current.diagonal();fp=current.sum(0)-tp;fn=support-tp
                    for c,entry in enumerate(saved['classes']):
                        assert [int(support[c]),int(tp[c]),int(fp[c]),int(fn[c])]==[entry[k] for k in ['support','TP','FP','FN']]
                        if support[c]:assert abs(entry['recall']-tp[c]/support[c])<1e-12
                        if tp[c]+fp[c]:assert abs(entry['precision']-tp[c]/(tp[c]+fp[c]))<1e-12
                    if scope=='all':
                        assert abs(saved['benign_fpr']-(1-current[b,b]/support[b]))<1e-12
                    else:assert saved['benign_fpr'] is None
                    assert saved['attack_to_benign']==int(np.delete(current[:,b],b).sum())
                    key={k:j[k] for k in KEYS}
                    key.update(test_version=tv,test_scope=scope)
                    metrics.append(dict(**key,rows=saved['rows'],**{m:saved[m] for m in MEASURES}))
                    per_class.extend(dict(**key,**c) for c in saved['classes'] if scope=='all' or c['name']!='benign')
            cost.append({k:j[k] for k in KEYS+['train_rows','construct_seconds','fit_seconds','preparation_seconds',
                                              'inference_seconds','peak_gpu_gib','xgb_splits']})
            if j['continuity']:
                continuity.append(dict(**{k:j[k] for k in KEYS},**j['continuity']))
            checks.append(dict(job=j['job_id'],prediction_hash=True,independent_confusion=True,
                               independent_f1=True,train_ids_hash=True,views=4))
            if len(checks)%20==0:print(f'Independently checked {len(checks)}/400 jobs',flush=True)
    frames={'metrics':pd.DataFrame(metrics),'per_class':pd.DataFrame(per_class),'cost':pd.DataFrame(cost),
            'continuity':pd.DataFrame(continuity),'inventory':pd.DataFrame(read(run/'inventory.json'))}
    frame=frames['metrics']
    contrasts=[]
    specifications=[('model','method','xgb','pfn'),('benign_training','benign_training',False,True),
                    ('train_curation','train_version','original','clean'),
                    ('test_curation','test_version','original','clean'),
                    ('benign_test_inclusion','test_scope','attack','all')]
    allkeys=KEYS+['test_version','test_scope']
    for contrast,axis,base,changed in specifications:
        keys=[k for k in allkeys if k!=axis]
        a=frame[frame[axis]==base].set_index(keys)
        z=frame[frame[axis]==changed].set_index(keys)
        assert a.index.is_unique and z.index.is_unique
        z=z.reindex(a.index)
        assert z['rows'].notna().all()
        for measure in MEASURES:
            if measure in ['all_macro_f1','benign_fpr'] and contrast=='benign_test_inclusion':continue
            delta=(z[measure]-a[measure]).rename('delta').reset_index()
            delta['contrast']=contrast;delta['measure']=measure
            delta['baseline']=str(base);delta['changed']=str(changed)
            contrasts.append(delta)
    frames['paired_contrasts']=pd.concat(contrasts,ignore_index=True)
    grouping=[k for k in allkeys if k!='seed']
    summary=frame.groupby(grouping)[MEASURES].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    frames['summary']=summary.reset_index()
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    validation=dict(passed=True,validated_at=stamp(),jobs=len(checks),views=len(metrics),
                    input_checks=input_checks,job_checks=checks,
                    zero_split_jobs=[x['job_id'] for x in result['jobs'] if x['method']=='xgb' and x['xgb_splits']==0],
                    csv_sha256={p.name:sha(p) for p in out.glob('*.csv')})
    write(out/'validation.json',validation)
    destination=run/'source_dependencies/finalization'
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,destination/Path(__file__).name)
    print(json.dumps(dict(passed=True,jobs=len(checks),views=len(metrics))),flush=True)


if __name__=='__main__':main()

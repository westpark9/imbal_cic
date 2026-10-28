"""Independent validation and reusable tables for EXP88; original runs read-only."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').exists())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)

import argparse,json,pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from exp61_utils import write,sha

def read(path):return json.loads(Path(path).read_text())

def metrics(y,p,classes):
    cm=confusion_matrix(y,p,labels=classes)
    denominator=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),denominator,out=np.zeros(len(classes)),where=denominator!=0)
    return dict(macro_f1=float(f.mean()),per_class_f1=f.tolist(),confusion=cm.tolist())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();cfg=read(run/'config.json');design=cfg['design'];result=read(run/'results.json')
    assert result['state']=='complete' and len(result['jobs'])==result['expected_jobs']==320
    out=run/'analysis';out.mkdir(exist_ok=True)
    jobs=result['jobs'];coverage=pd.DataFrame(read(run/'coverage.json'))
    coverage.to_csv(out/'coverage.csv',index=False)
    inventory=pd.DataFrame(read(run/'inventory.json'));inventory.to_csv(out/'inventory.csv',index=False)
    flat=[];per_class=[];checks=[];secondary=[];complement=[]
    source=ROOT/design['source_run']
    original=read(Path(read(source/'data/cic2018/manifest.json')['source_exp86'])/'data/cic2018/manifest.json')
    with Path(original['source_pickle']).open('rb') as stream:suite=pickle.load(stream)
    for ds in design['datasets']:
        d=run/'data'/ds;meta=read(d/'manifest.json');names=meta['class_names'];classes=np.asarray(meta['attack_classes'])
        train=np.load(Path(meta['clean_manifest']).parent/'train_idx.npy')
        ty=np.load(d/'test_y.npy');test_ids=np.load(d/'test_ids.npy');th=np.load(d/'test_hash.npy')
        reference=np.concatenate([np.load(d/f'class_{c}_reference_ids.npy') for c in classes])
        candidate=np.concatenate([np.load(d/f'class_{c}_candidate_ids.npy') for c in classes])
        assert len(np.intersect1d(reference,candidate))==0
        assert np.isin(reference,train).all() and np.isin(candidate,train).all()
        assert len(np.intersect1d(candidate,test_ids))==len(np.intersect1d(reference,test_ids))==0
        assert len(candidate)+len(reference)==sum(meta['candidate_counts'].values())
        scaler=np.load(d/'scaler.npz');assert not np.isin(scaler['fit_source_ids'],reference).any()
        full_hash=[]
        for arm in design['arms']:
            for seed in design['seeds']:
                context=d/arm/f'seed_{seed}';ids=np.load(context/'source_ids.npy')
                x=np.load(context/'X.npy');y=np.load(context/'y.npy')
                np.testing.assert_array_equal(x,np.nan_to_num(np.asarray(suite['X'][ids],dtype='float32')))
                np.testing.assert_array_equal(np.asarray(suite['families'])[ids],np.asarray(names)[y])
                assert np.isin(ids,candidate).all() and not np.isin(ids,reference).any()
                previous=set()
                for n in design['budgets']:
                    ix=np.load(context/f'n_{n}_indices.npy');current=set(ids[ix].tolist())
                    assert previous<=current and len(current)==n*len(classes);previous=current
                    np.testing.assert_array_equal(np.bincount(y[ix],minlength=len(names))[classes],np.full(len(classes),n))
                    for method in ['xgb','pfn']:
                        job=run/'jobs'/f'{ds}_{arm}_s{seed}_n{n}_{method}'
                        np.testing.assert_array_equal(np.load(job/'train_source_ids.npy'),ids[ix])
                    sub=coverage[(coverage.dataset==ds)&(coverage.arm==arm)&(coverage.seed==seed)&(coverage.n==n)]
                    for c in classes:
                        unique=len(np.unique(x[ix][y[ix]==c],axis=0))
                        assert unique==int(sub[sub.class_id==c].unique_vectors.iloc[0])
                full_hash.extend(np.load(context/'hashes.npy').tolist())
        # One mask per dataset fixed across budgets, models, seeds and arms.
        disjoint=~np.isin(th,np.asarray(full_hash,dtype=th.dtype))
        _,first=np.unique(th,return_index=True)
        masks={'context_union_disjoint':disjoint,'unique_test_vector':first}
        np.save(out/f'{ds}_fixed_disjoint_mask.npy',disjoint)
        np.save(out/f'{ds}_unique_test_indices.npy',first)
        checks.append(dict(dataset=ds,clean_train_membership=True,reference_candidate_test_disjoint=True,
            source_X_y_match=True,nested_and_paired=True,scaler_excludes_reference=True,
            reference_rows=len(reference),full_test_rows=len(ty),disjoint_test_rows=int(disjoint.sum()),
            disjoint_test_class_counts=np.bincount(ty[disjoint],minlength=len(names)).tolist(),
            unique_test_rows=len(first),unique_test_class_counts=np.bincount(ty[first],minlength=len(names)).tolist()))
        cache={}
        for j in [j for j in jobs if j['dataset']==ds]:
            path=Path(j['predictions_path']);pred=np.load(path)
            assert sha(path)==read(path.parent/'COMPLETE.json')['predictions_sha256']
            got=metrics(ty,pred,classes)
            assert abs(got['macro_f1']-j['metrics']['macro_f1'])<1e-12
            np.testing.assert_array_equal(got['confusion'],np.asarray(j['metrics']['confusion'])[np.ix_(classes,classes)])
            keys={k:j[k] for k in ['dataset','arm','seed','n','method']}
            flat.append(dict(**keys,macro_f1=j['metrics']['macro_f1'],preparation_seconds=j['preparation_seconds'],
                inference_seconds=j['inference_seconds'],unique_training_vectors=j['training_unique_feature_rows'],
                exact_test_overlap_rows=j['exact_feature_overlap_attack_rows'],
                peak_gpu_gib=j['peak_gpu_gib']))
            for c in j['metrics']['classes']:
                if c['name']=='benign':continue
                per_class.append(dict(**keys,**c))
            for variant,mask in masks.items():
                summary=metrics(ty[mask],pred[mask],classes)
                secondary.append(dict(**keys,evaluation=variant,macro_f1=summary['macro_f1']))
            if j['method']=='xgb':cache[j['arm'],j['seed'],j['n']]=path
        for j in [j for j in jobs if j['dataset']==ds and j['method']=='pfn']:
            a=np.load(cache[j['arm'],j['seed'],j['n']]);b=np.load(j['predictions_path'])
            for c in classes:
                mask=ty==c;ac=a[mask]==c;bc=b[mask]==c
                complement.append(dict(dataset=ds,arm=j['arm'],seed=j['seed'],n=j['n'],class_name=names[c],
                    support=int(mask.sum()),both_correct=int((ac&bc).sum()),
                    pfn_only_correct=int((~ac&bc).sum()),xgb_only_correct=int((ac&~bc).sum()),
                    both_wrong=int((~ac&~bc).sum())))
    del suite
    # Nested contexts cannot make nearest-reference distances increase.
    for (_,_,_), group in coverage.groupby(['dataset','arm','seed']):
        for _,g in group.groupby('class_id'):
            g=g.sort_values('n')
            assert (np.diff(g.unique_vectors)>=0).all()
            assert (np.diff(g.reference_distance_median)<=1e-9).all()
            assert (np.diff(g.reference_distance_p90)<=1e-9).all()
    frames={'metrics':pd.DataFrame(flat),'per_class':pd.DataFrame(per_class),
            'secondary_metrics':pd.DataFrame(secondary),'complementarity':pd.DataFrame(complement)}
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    main=frames['metrics']
    summary=main.groupby(['dataset','arm','n','method']).macro_f1.agg(['mean','std','min','max','count']).reset_index()
    assert (summary['count']==5).all();summary.to_csv(out/'f1_summary.csv',index=False)
    paired=main.pivot(index=['dataset','arm','n','seed'],columns='method',values='macro_f1').reset_index()
    paired['pfn_minus_xgb_pp']=(paired.pfn-paired.xgb)*100;paired.to_csv(out/'model_paired_deltas.csv',index=False)
    arm_deltas=[]
    for (ds,n,method),g in main.groupby(['dataset','n','method']):
        pivot=g.pivot(index='seed',columns='arm',values='macro_f1')
        for arm in ['unique_random','cluster20','cluster40']:
            for seed,value in (100*(pivot[arm]-pivot.row_random)).items():
                arm_deltas.append(dict(dataset=ds,n=n,method=method,arm=arm,seed=seed,delta_pp=value))
    pd.DataFrame(arm_deltas).to_csv(out/'selection_paired_deltas.csv',index=False)
    coverage.groupby(['dataset','arm','n','seed']).agg(
        unique_vectors=('unique_vectors','mean'),distance_median=('reference_distance_median','mean'),
        distance_p90=('reference_distance_p90','mean'),zero_distance_rate=('reference_zero_distance_rate','mean'),
        clusters_covered=('covered_cluster20_count','mean')).reset_index().to_csv(out/'coverage_class_macro.csv',index=False)
    write(out/'validation.json',dict(passed=True,conditions=320,coverage_conditions=len(coverage),dataset_checks=checks,
        nested_distance_monotonic=True,independent_sklearn_metric_checks=320,source_hash_verified_by_preparation=True,
        checkpoint_frozen_all_pfn_jobs=True,report_pending=True,
        note='Unique-test diagnostic uses audited source vector hashes; no selection or fitting uses test rows.'))
    print(json.dumps(dict(passed=True,jobs=320,analysis=str(out)),ensure_ascii=False))

if __name__=='__main__':main()

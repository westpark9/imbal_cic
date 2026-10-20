#!/usr/bin/env python3
"""Recompute saved-prediction metrics and audit matched stream/memory IDs."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from exp71_bounded_updates import aggregate, STAGES, CASES, seen_classes, sha, write


def verify(run, reference=None):
    aggregate(run,require_complete=True)
    config=json.loads((run/'request.json').read_text())
    mode=config['history_mode']
    assert mode in ['bounded','cumulative']
    source=Path(config['stream_root'])
    audited=[]
    for ds in STAGES:
        folder=source/'data'/ds
        meta=json.loads((folder/'manifest.json').read_text());names=meta['class_names']
        plan=json.loads((folder/'plans.json').read_text())
        train_ids=np.load(folder/'train_ids.npy');train_y=np.load(folder/'train_y.npy')
        test_ids=np.load(folder/'test_ids.npy');test_y=np.load(folder/'test_y.npy')
        assert not np.intersect1d(train_ids,test_ids).size
        for case in CASES:
            for method in ['xgb','tabpfn']:
                previous_fit=set();delivered=set();delivered_order=[];discarded=set();previous_pred=None;previous_y=None
                for stage in range(len(STAGES[ds])):
                    dest=run/'jobs'/f'{ds}_{case}_{method}'/f'stage_{stage}'
                    r=json.loads((dest/'metrics.json').read_text())
                    p=plan[case][stage];fit=np.load(dest/'fit_bank_indices.npy')
                    incoming=set(p['incoming'])
                    assert not incoming&delivered
                    delivered_order.extend(p['incoming'])
                    expected=p['fit'] if mode=='bounded' else delivered_order
                    assert np.array_equal(fit,np.asarray(expected))
                    assert np.array_equal(np.load(dest/'train_ids.npy'),train_ids[fit])
                    assert len(np.unique(fit))==len(fit)
                    history=set(fit)-incoming
                    if mode=='bounded':
                        assert history==set(p['history'])
                        assert history<=previous_fit and len(history)<=config['history_cap']
                        assert not set(fit)&discarded
                    else:
                        assert history==previous_fit==delivered
                        assert set(fit)==delivered|incoming
                    assert r['history_mode']==mode and r['history_rows']==len(history)
                    assert r['incoming_rows']==len(incoming) and r['context_rows']==len(fit)
                    delivered|=incoming;discarded|=previous_fit-history;previous_fit=set(fit)
                    seen=seen_classes(ds,stage);lookup=np.full(len(names),-1,dtype=np.int16)
                    for i,c in enumerate(seen):lookup[names.index(c)]=i
                    idx=np.flatnonzero(lookup[test_y]>=0);y=lookup[test_y[idx]]
                    assert np.array_equal(np.load(dest/'test_ids.npy'),test_ids[idx])
                    assert set(train_y[fit])=={names.index(c) for c in seen}
                    pred=np.load(dest/'pred.npy',mmap_mode='r');proba=np.load(dest/'proba.npy',mmap_mode='r')
                    assert len(pred)==len(y) and proba.shape==(len(y),len(seen))
                    for start in range(0,len(y),131072):
                        end=min(start+131072,len(y));block=proba[start:end]
                        assert np.isfinite(block).all() and np.array_equal(block.argmax(1),pred[start:end])
                        assert np.allclose(block.sum(1),1,rtol=1e-4,atol=1e-4)
                    cm=np.zeros((len(seen),len(seen)),dtype=np.int64)
                    np.add.at(cm,(y,pred),1)
                    assert np.array_equal(cm,np.asarray(r['confusion_matrix']))
                    tp=np.diag(cm);support=cm.sum(1);den=cm.sum(0)+support
                    f1=np.divide(2*tp,den,out=np.zeros(len(seen)),where=den!=0)
                    assert abs(float(f1.mean())-r['macro_f1'])<1e-12
                    assert abs(float(tp.sum()/cm.sum())-r['accuracy'])<1e-12
                    for i,c in enumerate(r['classes']):
                        assert c['name']==seen[i] and c['support']==int(support[i])
                        assert c['fp']==int(cm[:,i].sum()-tp[i]) and abs(c['f1']-f1[i])<1e-12
                        precision=float(tp[i]/cm[:,i].sum()) if cm[:,i].sum() else 0.
                        recall=float(tp[i]/support[i]) if support[i] else 0.
                        assert abs(c['precision']-precision)<1e-12 and abs(c['recall']-recall)<1e-12
                        assert c['tp']==int(tp[i]) and c['fn']==int(support[i]-tp[i])
                    b=seen.index('benign');fpr=(support[b]-tp[b])/support[b]
                    assert abs(fpr-r['benign_false_alarm_rate'])<1e-12
                    if stage:
                        old_count=len(seen_classes(ds,stage-1));old=y<old_count
                        new_indices=[seen.index(c) for c in r['new_classes']]
                        assert abs(float(f1[new_indices].mean())-r['new_class_macro_f1'])<1e-12
                        assert np.array_equal(y[old],previous_y)
                        old_cm=cm.copy();old_cm[old_count:,:]=0
                        denom=old_cm.sum(0)+old_cm.sum(1)
                        old_f=np.divide(2*np.diag(old_cm),denom,out=np.zeros(len(seen)),where=denom!=0)
                        assert abs(float(old_f[:old_count].mean())-r['retention_after_macro_f1'])<1e-12
                        before=previous_pred==y[old];after=pred[old]==y[old]
                        assert int(np.sum(~before&after))==r['old_corrected']
                        assert int(np.sum(before&~after))==r['old_damaged']
                    previous_pred=pred;previous_y=y
                    cost=json.loads((dest/'cost.json').read_text())
                    wanted={'select_and_materialize','model_construct','fit_context_or_retrain'}
                    update=sum(x['seconds'] for x in cost['phases'] if x['phase'] in wanted)
                    assert abs(update-r['update_seconds'])<1e-10
                    assert all(x['complete'] for x in cost['phases'])
                    audited.append(dict(dataset=ds,case=case,method=method,stage=stage,rows=len(y)))
    validation=dict(status='passed',history_mode=mode,stages=len(audited),evaluated_rows=sum(x['rows'] for x in audited),
        checks=['paired train/test IDs','no future fit labels',
                'bounded history / no discarded-row resurrection' if mode=='bounded' else 'all delivered history retained in arrival order',
                'probability finiteness/normalization/argmax',
                'full confusion matrices and class F1','benign false alarms',
                'matched old-population retention/corrections/damage','cost phase accounting'],
        summary_sha256=sha(run/'summary.csv'),audited=audited)
    write(run/'validation.json',validation)
    if reference is not None:
        assert mode=='cumulative'
        compare_reference(run,reference)
    print(json.dumps({k:v for k,v in validation.items() if k!='audited'},ensure_ascii=False,indent=2))


def compare_reference(run, reference):
    """Match policies on identical streams and compare the saved predictions."""
    current=json.loads((run/'request.json').read_text())
    baseline=json.loads((reference/'request.json').read_text())
    assert baseline['history_mode']=='bounded' and current['history_mode']=='cumulative'
    for k in ['seed','history_cap','nominal','batch','threads','checkpoint_sha256','prepared_sha256']:
        assert current[k]==baseline[k],k
    assert Path(current['stream_root']).resolve()==Path(baseline['stream_root']).resolve()
    assert json.loads((reference/'validation.json').read_text())['summary_sha256']==sha(reference/'summary.csv')
    recovery_file=run/'recovery.json'
    recovery=json.loads(recovery_file.read_text()) if recovery_file.exists() else None
    baseline_environment=json.loads((reference/'environment.json').read_text())
    environments={}
    for ds in STAGES:
        for case in CASES:
            for method in ['xgb','tabpfn']:
                job=f'{ds}_{case}_{method}'
                record=next(r for r in recovery['jobs'] if r['job']==job) if recovery else None
                env=json.loads((run/(record['environment_file'] if record else 'environment.json')).read_text())
                for key in ['packages','tabpfn_python_sources_sha256']:
                    assert env[key]==baseline_environment[key],(job,key)
                environments[job]=dict(
                    baseline_platform=baseline_environment['platform'],current_platform=env['platform'],
                    baseline_gpu=baseline_environment['gpu'],current_gpu=env['gpu'],
                    matched=all(env[k]==baseline_environment[k] for k in ['platform','gpu']))
    changes={job:env for job,env in environments.items() if not env['matched']}
    checks=[]
    for ds in STAGES:
        source=Path(current['stream_root'])/'data'/ds
        names=json.loads((source/'manifest.json').read_text())['class_names']
        test_y=np.load(source/'test_y.npy')
        for case in CASES:
            for method in ['xgb','tabpfn']:
                for stage in range(len(STAGES[ds])):
                    relative=Path('jobs')/f'{ds}_{case}_{method}'/f'stage_{stage}'
                    b=reference/relative;c=run/relative
                    br=json.loads((b/'metrics.json').read_text());cr=json.loads((c/'metrics.json').read_text())
                    for k in ['test_ids_digest','seen_classes','new_classes','incoming_class_counts',
                              'incoming_rows','delivered_rows','seed','checkpoint_sha256','stream_plan_sha256','packages','gpu']:
                        assert br[k]==cr[k],(relative,k)
                    bi=np.load(b/'train_ids.npy');ci=np.load(c/'train_ids.npy')
                    assert np.isin(bi,ci).all() and len(ci)==cr['delivered_rows']
                    assert np.array_equal(np.load(b/'test_ids.npy'),np.load(c/'test_ids.npy'))
                    if stage==0:assert np.array_equal(bi,ci)
                    lookup=np.full(len(names),-1,dtype=np.int16)
                    for i,n in enumerate(cr['seen_classes']):lookup[names.index(n)]=i
                    y=lookup[test_y];y=y[y>=0]
                    bp=np.load(b/'pred.npy',mmap_mode='r');cp=np.load(c/'pred.npy',mmap_mode='r')
                    bc=bp==y;cc=cp==y
                    checks.append(dict(dataset=ds,case=case,method=method,stage=stage,
                        system_environment_matched=environments[f'{ds}_{case}_{method}']['matched'],
                        bounded_rows=len(bi),cumulative_rows=len(ci),test_rows=len(y),
                        prediction_changes=int(np.sum(bp!=cp)),
                        corrected=int(np.sum(~bc&cc)),damaged=int(np.sum(bc&~cc))))
    write(run/'comparison_validation.json',dict(status='passed',reference=str(reference),
        reference_summary_sha256=sha(reference/'summary.csv'),current_summary_sha256=sha(run/'summary.csv'),
        checks=['identical incoming IDs via shared immutable plans','identical full test IDs',
                'same seed/checkpoint/model configuration/Python packages/TabPFN source/GPU model',
                'kernel and driver compared per worker; differences explicitly recorded',
                'bounded fit subset of cumulative fit',
                'identical initial fit IDs','policy corrections/damage on matched saved predictions'],
        system_environment_changes=changes,stages=checks))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    a=p.parse_args();verify(a.run.resolve(),a.reference.resolve() if a.reference else None)

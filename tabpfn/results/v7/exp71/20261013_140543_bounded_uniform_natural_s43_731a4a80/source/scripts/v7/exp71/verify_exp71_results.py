#!/usr/bin/env python3
"""Recompute saved-prediction metrics and audit matched stream/memory IDs."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from exp71_bounded_updates import aggregate, STAGES, CASES, seen_classes, sha, write


def verify(run):
    aggregate(run,require_complete=True)
    config=json.loads((run/'request.json').read_text())
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
                previous_fit=set();delivered=set();discarded=set();previous_pred=None;previous_y=None
                for stage in range(len(STAGES[ds])):
                    dest=run/'jobs'/f'{ds}_{case}_{method}'/f'stage_{stage}'
                    r=json.loads((dest/'metrics.json').read_text())
                    p=plan[case][stage];fit=np.load(dest/'fit_bank_indices.npy')
                    assert np.array_equal(fit,np.asarray(p['fit']))
                    assert np.array_equal(np.load(dest/'train_ids.npy'),train_ids[fit])
                    assert len(np.unique(fit))==len(fit)
                    history=set(p['history']);incoming=set(p['incoming'])
                    assert history<=previous_fit and len(history)<=config['history_cap']
                    assert not incoming&delivered and not set(fit)&discarded
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
    validation=dict(status='passed',stages=len(audited),evaluated_rows=sum(x['rows'] for x in audited),
        checks=['paired train/test IDs','no future fit labels','no discarded-row resurrection',
                'bounded history size','probability finiteness/normalization/argmax',
                'full confusion matrices and class F1','benign false alarms',
                'matched old-population retention/corrections/damage','cost phase accounting'],
        summary_sha256=sha(run/'summary.csv'),audited=audited)
    write(run/'validation.json',validation)
    print(json.dumps({k:v for k,v in validation.items() if k!='audited'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    verify(p.parse_args().run.resolve())

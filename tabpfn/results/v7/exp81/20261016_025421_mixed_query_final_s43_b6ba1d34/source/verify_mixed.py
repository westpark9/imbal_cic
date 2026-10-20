#!/usr/bin/env python3
"""Verify mixed-query predictions, paired correction, and retained model states."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,json,shutil
import numpy as np
from sklearn.metrics import confusion_matrix
from exp61_utils import write,sha
from mixed_query_eval import SMALL,LARGE,EXPANDED,OLD_BUDGET


def verify_metrics(y,pred,metrics,C):
    assert len(pred)==len(y) and np.all((pred>=0)&(pred<C))
    cm=confusion_matrix(y,pred,labels=np.arange(C))
    np.testing.assert_array_equal(cm,metrics['confusion'])
    den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*np.diag(cm),den,out=np.zeros(C),where=den!=0)
    assert abs(float(f1.mean())-metrics['macro_f1'])<1e-12
    assert metrics['rows']==len(y)
    return float(f1.mean())


def corrected(p,prior,mapping):
    p=np.asarray(p,dtype=np.float64)
    posterior=p.sum(0)/len(p)
    q=p*posterior[None,:]
    q=q/(prior+1e-8)[None,:]
    q=(q/q.sum(1)[:,None]).astype('float32')
    return mapping[np.argmax(q,axis=1)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run
    from finalize_mixed_metadata import finalize
    finalize(run)
    result=json.loads((run/'results.json').read_text());assert result['state']=='complete' and len(result['jobs'])==30
    jobs=result['jobs'];expected=set()
    for scope in ['small','large']:
        methods=['xgb','pfn_pretrained','ours','boostpfn','distpfn_pretrained'] if scope=='small' else ['xgb','tabpfn','ours','boostpfn','distpfn']
        for ds in ['cic2018','toniot']:
            for case in (['new_100'] if scope=='small' else ['natural','uniform']):
                expected.update((scope,ds,case,m) for m in methods)
    assert {(j['scope'],j['dataset'],j['case'],j['method']) for j in jobs}==expected
    data={};checked=[];hashes={};counterfactuals=[]
    additional=json.loads((run/'additional_model_provenance.json').read_text())
    for path,digest in additional['source_hashes'].items():
        assert sha(_root/path)==digest,path;hashes[path]=digest
    for ds in ['cic2018','toniot']:
        manifest=json.loads((SMALL/'data'/ds/'manifest.json').read_text())
        y=np.load(Path(manifest['source'])/'test_y.npy',mmap_mode='r');C=len(manifest['class_names'])
        order=np.load(run/'data'/ds/'query_order.npy',mmap_mode='r')
        np.testing.assert_array_equal(order,np.random.default_rng(43).permutation(len(y)))
        assert json.loads((run/'data'/ds/'manifest.json').read_text())['labels_used_to_order'] is False
        data[ds]=(y,C,order,sha(run/'data'/ds/'query_order.npy'))
    for j in jobs:
        y,C,order,order_sha=data[j['dataset']];pred=np.load(j['predictions_path'],mmap_mode='r')
        assert j['evaluation']=='final_full_test' and j['test_order']=='seed43_mixed' and j['prediction_storage_order']=='original'
        assert not j['training_rerun'] and j['query_order_sha256']==order_sha
        f1=verify_metrics(y,pred,j['metrics'],C)
        for path,digest in j.get('source_hashes',{}).items():
            if path not in hashes:assert sha(_root/path)==digest,path;hashes[path]=digest
        if j['method'] in ['tabpfn','pfn_pretrained']:
            assert j['full_reinference'] and not j['initial_finetuning']
            assert sha(j['source_model'])==j['source_model_sha256']
            hashes[j['source_model']]=j['source_model_sha256']
            probs=np.load(j['probability_path'],mmap_mode='r');mapping=np.array(j['class_mapping']);prior=np.array(j['training_prior'])
            assert probs.shape==(len(y),C) and np.isfinite(probs).all()
            np.testing.assert_allclose(probs.sum(1),1,atol=1e-3)
            np.testing.assert_array_equal(mapping[probs.argmax(1)],pred)
            dm='distpfn_pretrained' if j['scope']=='small' else 'distpfn'
            dj=next(q for q in jobs if (q['scope'],q['dataset'],q['case'],q['method'])==(j['scope'],j['dataset'],j['case'],dm))
            dp=np.load(dj['predictions_path'],mmap_mode='r');gp=np.load(dj['original_group_predictions_path'],mmap_mode='r')
            for start in range(0,len(y),8192):
                ids=order[start:start+8192]
                np.testing.assert_array_equal(corrected(probs[ids],prior,mapping),dp[ids])
                sl=slice(start,start+8192)
                np.testing.assert_array_equal(corrected(probs[sl],prior,mapping),gp[sl])
            group_f1=verify_metrics(y,gp,dj['original_group_counterfactual'],C)
            counterfactuals.append(dict(scope=j['scope'],dataset=j['dataset'],case=j['case'],
                mixed_macro_f1=dj['metrics']['macro_f1'],original_group_macro_f1=group_f1,
                prediction_changes=int(np.count_nonzero(dp!=gp)),probabilities_sha256=sha(j['probability_path'])))
        elif j['method'] in ['xgb','ours','boostpfn']:
            assert j['mixed_audit_rows']==8192
            if not j['full_reinference']:assert j['mixed_audit_mismatches']==0 and j['reuse_reason']
            else:assert j['mixed_audit_mismatches']>0
        checked.append(dict(scope=j['scope'],dataset=j['dataset'],case=j['case'],method=j['method'],macro_f1=f1,
            full_reinference=j['full_reinference'],predictions_sha256=sha(j['predictions_path'])))
    audit=json.loads((run/'budget_order_audits.json').read_text());assert audit['passed'] and len(audit['audits'])==10
    assert {(x['dataset'],x['new_per_class']) for x in audit['audits']}=={(d,n) for d in data for n in [10,20,50,200,400]}
    assert all(x['mismatches']==0 and x['rows']==8192 for x in audit['audits'])
    shutil.copy2(__file__,run/'source'/Path(__file__).name)
    write(run/'verification.json',dict(passed=True,jobs=checked,counterfactuals=counterfactuals,
        source_files_unchanged=len(hashes),source_hashes=hashes,budget_audit_passed=True,training_rerun=False,seed_repetitions=1,
        checks=['seeded label-independent complete query permutation','predictions scattered to original row IDs',
                'independent confusion and macro F1','paired DistPFN grouping on identical probabilities',
                'immutable saved models','static-model mixed-query audits','all reduced/expanded-budget mixed-query audits']))
    print(json.dumps(dict(passed=True,jobs=len(checked))))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Exact accuracy and macro-F1 ceilings for selection from a fixed prediction bank.

Test truth is deliberately available to this unattainable selector. It cannot
invent labels: every selected answer must occur in the allowed prediction bank.
No fitting, no HTML updates. Original five-condition artifacts stay immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from exp56_five_condition_evaluation import argmax, dump, scorer_policies, sha256
from exp56_label_free_revision import metrics


def candidate_masks(global_pred, experts, allowed_calls=None):
    Cmax=int(max(global_pred.max(),experts.max()))+1
    if Cmax>16:raise ValueError('Bitmask implementation supports at most 16 classes')
    masks=np.left_shift(np.uint32(1),global_pred.astype(np.uint32))
    for j in range(experts.shape[1]):
        bit=np.left_shift(np.uint32(1),experts[:,j].astype(np.uint32))
        masks|=bit if allowed_calls is None else np.where(allowed_calls,bit,0).astype(np.uint32)
    return masks


def exact_macro_f1_selector(y,masks,C):
    """Return a feasible oracle and an exact maximum over all row-wise selectors.

    Correcting a correctable row increases its true-class F1 and cannot decrease
    other classes' F1. Thus all coverable rows are correct in an optimum. TP and
    FN then become fixed. For remaining rows only the destination of FP varies.
    Their candidate-label masks define independent allocation groups.

    The remaining objective is sum_c 2 TP_c/(support_c+TP_c+FP_c), a convex,
    separable function of FP. At a global maximum, any movable group must be
    assigned to a class with greatest partial derivative; otherwise moving it
    raises the objective by the convex gradient inequality. A derivative tie
    involving positive TP also raises the objective after a finite move by
    strict convexity. Zero-TP ties do not affect it. Therefore some optimum
    assigns every group to its first available class in one total priority
    order. A subset DP searches all C! orders in O(C 2^C), without sampling.
    """
    y=np.asarray(y,int);masks=np.asarray(masks,np.uint32)
    if len(y)!=len(masks) or np.any(masks==0) or np.any(masks>=(1<<C)):
        raise ValueError('Invalid candidate masks')
    if np.any(y<0) or np.any(y>=C):raise ValueError('Invalid labels')
    covered=(masks&np.left_shift(np.uint32(1),y.astype(np.uint32)))!=0
    support=np.bincount(y,minlength=C);tp=np.bincount(y[covered],minlength=C)
    weights=np.bincount(masks[~covered].astype(int),minlength=1<<C).astype(np.int64)
    subset_sums=weights.copy()
    for c in range(C):
        bit=1<<c
        for mask in range(1<<C):
            if mask&bit:subset_sums[mask]+=subset_sums[mask^bit]
    dp=np.full(1<<C,-np.inf);dp[0]=0.;choice=np.full(1<<C,-1,int)
    for mask in range(1,1<<C):
        for c in range(C):
            if not mask&(1<<c):continue
            rest=mask^(1<<c);fp=int(subset_sums[mask]-subset_sums[rest])
            den=int(support[c]+tp[c]+fp)
            value=(2.*tp[c]/den if den else 0.)+dp[rest]
            if value>dp[mask]+1e-15:dp[mask]=value;choice[mask]=c
    order=[];mask=(1<<C)-1
    while mask:
        c=int(choice[mask]);order.append(c);mask^=1<<c
    pred=y.astype(np.int16).copy();todo=~covered
    for c in order:
        select=todo&((masks&(1<<c))!=0);pred[select]=c;todo[select]=False
    assert not todo.any()
    assert np.all((masks&np.left_shift(np.uint32(1),pred.astype(np.uint32)))!=0)
    cm=np.bincount(y*C+pred,minlength=C*C).reshape(C,C)
    den=cm.sum(0)+cm.sum(1)
    achieved=float(np.divide(2*np.diag(cm),den,out=np.zeros(C),where=den>0).mean())
    np.testing.assert_allclose(achieved,dp[-1]/C,atol=1e-13,rtol=0)
    # These per-class maxima need not be jointly attainable; keep them separate.
    forced_fp=np.array([weights[1<<c] for c in range(C)])
    den=support+tp+forced_fp
    class_upper=np.divide(2*tp,den,out=np.zeros(C),where=den>0)
    return pred,dict(accuracy_upper_bound=float(covered.mean()),macro_f1_upper_bound=achieved,
                     covered_rows=int(covered.sum()),uncovered_rows=int((~covered).sum()),
                     unavoidable_prediction_groups=int(np.count_nonzero(weights)),
                     false_positive_destination_priority=order,
                     class_recall_upper=np.divide(tp,support,out=np.zeros(C),where=support>0).tolist(),
                     class_f1_individual_upper=class_upper.tolist(),
                     optimization='exact subset DP over all class priority orders; fixed bank, row-wise choices')


def evaluate(source_root,out_root):
    out_root.mkdir(parents=True,exist_ok=False);started=time.time();combined=[]
    shutil.copy2(__file__,out_root/'experiment_source.py')
    for ds in ['cic2018','toniot']:
        source=source_root/ds;out=out_root/ds;out.mkdir()
        protocol=json.loads((source/'protocol.json').read_text());cache=Path(protocol['cache']);policy=Path(protocol['policy'])
        assert json.loads((cache/'identity.json').read_text())==protocol['cache_identity']
        assert sha256(policy/f'eval_{protocol["arm"]}_scores.npz')==protocol['policy_scores_sha256']
        names=protocol['class_names'];C=len(names);K=protocol['n_experts']
        meta=json.loads((cache/'COMPLETE.json').read_text());tail=[names.index(x) for x in meta['tail_classes']]
        y=np.load(cache/'eval_y.npy').astype(np.int16);g=argmax(cache/'eval_p0.npy')
        E=np.stack([argmax(cache/f'eval_p{k+1}.npy') for k in range(K)],axis=1)
        with np.load(policy/f'eval_{protocol["arm"]}_scores.npz') as z:scores={k:z[k] for k in z.files}
        sonly,sv,calls,accepted=scorer_policies(g,scores,protocol['selection'])
        np.testing.assert_array_equal(scores['candidate'],E[np.arange(len(y)),scores['winner']])
        np.testing.assert_array_equal(sv,np.load(policy/f'{protocol["arm"]}_final.npy'))
        summary=pd.read_csv(source/'1a_summary.csv').to_dict('records')
        per_class=pd.read_csv(source/'1b_per_class.csv').to_dict('records')
        confusion=pd.read_csv(source/'1c_confusion.csv').to_dict('records')
        records=[]
        for scope,gate in [('bank',None),('fixed_call_bank',calls)]:
            masks=candidate_masks(g,E,gate);pred,cert=exact_macro_f1_selector(y,masks,C)
            np.save(out/(scope+'_oracle_pred.npy'),pred)
            covered=(masks&(1<<y))!=0
            canonical=np.where(covered,y,g).astype(np.int16)
            logical_calls=np.ones(len(y),bool) if gate is None else gate
            row,pc,cm=metrics(y,pred,g,names,tail,scope+'_oracle',6 if scope=='bank' else 0,logical_calls,pred!=g)
            canon,_,_=metrics(y,canonical,g,names,tail,scope+'_global_fallback_oracle',0,logical_calls,canonical!=g)
            assert row['macro_f1']+1e-12>=canon['macro_f1']
            assert row['accuracy']==cert['accuracy_upper_bound']
            comparators=summary if scope=='bank' else [r for r in summary if r['policy'] in ['global','scorer_only','scorer_verifier']]
            assert all(row['macro_f1']+1e-12>=r['macro_f1'] and row['accuracy']+1e-12>=r['accuracy'] for r in comparators)
            for r in pc:
                c=names.index(r['class']);r['recall_upper_bound']=cert['class_recall_upper'][c]
                r['individual_f1_upper_not_joint']=cert['class_f1_individual_upper'][c]
            summary.append(row);per_class+=pc;confusion+=cm
            cert.update(scope=scope,old_global_fallback_macro_f1=canon['macro_f1'],old_global_fallback_accuracy=canon['accuracy'],
                        policy=row['policy'],K=K,C=C,bank_satisfies_K_less_than_C=K<C,
                        class_names=names,call_locations_fixed=gate is not None,
                        allows_unrestricted_expert_selection=True,cost_or_benign_FPR_constraints_enforced=False,
                        truth_use='per-sample answer selection and macro-F1 optimization; unattainable test-label-informed ceiling',
                        expert_quality_claim=False,test_role='same previously observed development holdout as the five conditions')
            dump(out/(scope+'_certificate.json'),cert);records.append(cert)
            combined.append(dict(dataset=ds,scope=scope,accuracy_upper_bound=cert['accuracy_upper_bound'],macro_f1_upper_bound=cert['macro_f1_upper_bound'],
                                 canonical_global_fallback_f1=canon['macro_f1'],uncovered_rows=cert['uncovered_rows']))
        new=pd.DataFrame(summary);old=pd.read_csv(source/'1a_summary.csv')
        pd.testing.assert_frame_equal(new.iloc[:len(old)][old.columns].reset_index(drop=True),old,check_dtype=False)
        new.to_csv(out/'six_condition_summary.csv',index=False)
        pd.DataFrame(per_class).sort_values(['support','condition'],ascending=[False,True]).to_csv(out/'six_condition_per_class.csv',index=False)
        pd.DataFrame(confusion).to_csv(out/'six_condition_confusion.csv',index=False)
        dump(out/'provenance.json',dict(source=str(source.resolve()),source_protocol_sha256=sha256(source/'protocol.json'),cache_identity=protocol['cache_identity'],
                                      prediction_bank_sha256=hashlib.sha256(g.tobytes()+E.tobytes()).hexdigest(),truth_sha256=sha256(cache/'eval_y.npy'),
                                      source_sha256=sha256(__file__),html_updates=False,new_training=False,original_five_conditions_unchanged=True))
        print(ds,pd.DataFrame(combined).query('dataset == @ds').to_string(index=False),flush=True)
    pd.DataFrame(combined).to_csv(out_root/'oracle_summary.csv',index=False)
    dump(out_root/'COMPLETE.json',dict(seconds=time.time()-started,datasets=['cic2018','toniot'],oracle_type='fixed prediction bank, exact accuracy and macro-F1 ceilings'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True);a=p.parse_args()
    evaluate(a.source_root,a.out_root)

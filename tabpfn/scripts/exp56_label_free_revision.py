#!/usr/bin/env python3
"""Recompute condition 2 without test truth; audit reported anomalies from frozen rows.

Original EXP56 results remain immutable. This revision reuses probability diagnostics,
recomputes all prediction metrics, and records provenance and exact source agreement.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from exp56_five_condition_evaluation import (argmax, assignment_oracle, designated_mapping,
                                            dump, fixed_region_assignment, scorer_policies, sha256)


def metrics(y, pred, g, names, tail, policy, condition, calls, accepted):
    C=len(names);cm=np.bincount(y.astype(np.int64)*C+pred,minlength=C*C).reshape(C,C)
    tp=np.diag(cm);sup=cm.sum(1);fp=cm.sum(0)-tp;fn=sup-tp
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(C),where=2*tp+fp+fn>0)
    H=(g!=y)&(pred==y);D=(g==y)&(pred!=y)
    row=dict(condition=condition,policy=policy,rows=len(y),macro_f1=float(f1.mean()),accuracy=float((y==pred).mean()),
             tail_f1=float(f1[tail].mean()),benign_f1=float(f1[names.index('benign')]),calls=int(calls.sum()),
             accepted=int(accepted.sum()),changed=int((g!=pred).sum()),H=int(H.sum()),D=int(D.sum()),net_correction=int(H.sum()-D.sum()))
    pc=[]
    for c,name in enumerate(names):
        mask=y==c
        pc.append(dict(condition=condition,policy=policy,**{'class':name},support=int(sup[c]),TP=int(tp[c]),FP=int(fp[c]),FN=int(fn[c]),TN=int(len(y)-sup[c]-fp[c]),
                       precision=float(tp[c]/max(tp[c]+fp[c],1)),recall=float(tp[c]/max(sup[c],1)),f1=float(f1[c]),fpr=float(fp[c]/max(len(y)-sup[c],1)),
                       H=int(H[mask].sum()),D=int(D[mask].sum()),calls=int(calls[mask].sum()),accepted=int(accepted[mask].sum())))
    return row,pc,[dict(policy=policy,true_class=names[c],predicted_class=names[j],rows=int(cm[c,j])) for c in range(C) for j in range(C)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True);a=p.parse_args()
    a.out_root.mkdir(exist_ok=False,parents=True);started=time.monotonic()
    for ds in ['cic2018','toniot']:
        source=a.source_root/ds;out=a.out_root/ds;out.mkdir()
        shutil.copy2(__file__,out/'revision_source.py')
        protocol=json.loads((source/'protocol.json').read_text());cache=Path(protocol['cache']);policy=Path(protocol['policy'])
        meta=json.loads((cache/'COMPLETE.json').read_text());names=meta['class_names'];C=len(names);K=meta['n_experts'];tail=[names.index(c) for c in meta['tail_classes']]
        assert json.loads((cache/'identity.json').read_text())==protocol['cache_identity']
        assert sha256(policy/f'eval_{protocol["arm"]}_scores.npz')==protocol['policy_scores_sha256']
        manifest_path=Path(meta['clean_manifest']);manifest=json.loads(manifest_path.read_text())
        assert sha256(manifest_path)==protocol['cache_identity']['clean_manifest_sha256']
        assert manifest['verification']['remaining_conflicting_vector_groups']==0
        g=argmax(cache/'eval_p0.npy');E=np.stack([argmax(cache/f'eval_p{k+1}.npy') for k in range(K)],axis=1)
        fixed,regions=fixed_region_assignment(E,np.load(cache/'eval_distance.npy',mmap_mode='r'))
        # Negative control: one constant per region, selected from training routing rows only.
        route_regions=np.load(cache/'route_distance.npy',mmap_mode='r').argmin(1);route_y=np.load(cache/'route_y.npy')
        constants=np.array([np.bincount(route_y[route_regions==k],minlength=C).argmax() if (route_regions==k).any() else np.argmax(meta['train_counts']) for k in range(K)])
        constant_pred=constants[regions]
        pd.DataFrame({'region':np.arange(K)+1,'constant_from_route_train':[names[c] for c in constants]}).to_csv(out/'0c_region_constant_control.csv',index=False)
        dump(out/'protocol.json',{**protocol,'condition_2':'observable fixed centroid, raw expert prediction; no truth input',
                                  'historical_assignment_oracle':'excluded; retained only in historical files and anomaly explanation',
                                  'source_results':str(source),'bank_satisfies_K_less_than_C':K<C,'new_expert_training':False,
                                  'source_revision_sha256':sha256(__file__),'eval_distances_sha256':sha256(cache/'eval_distance.npy')})
        # Test truth is opened only after fixed owners and training constants have been selected.
        y=np.load(cache/'eval_y.npy').astype(np.int16);N=len(y);assert N==manifest['split_rows']['test']
        with np.load(policy/f'eval_{protocol["arm"]}_scores.npz') as z:scores={k:z[k] for k in z.files}
        np.testing.assert_array_equal(scores['candidate'],E[np.arange(N),scores['winner']])
        sonly,sv,calls,accepted=scorer_policies(g,scores,protocol['selection'])
        for label,pred in [('scorer_only',sonly),('scorer_verifier',sv)]:np.testing.assert_array_equal(pred,np.load(source/f'{label}_pred.npy'))
        ctx=pd.read_csv(Path(meta['source_run'])/'2a_expert_contexts.csv');ctx=ctx[ctx.expert!='anchor(shared)'].copy();ctx.expert=ctx.expert.astype(int);ctx=ctx.sort_values('expert')
        _,mapping=designated_mapping(ctx[names].to_numpy());old,_=assignment_oracle(y,g,E,mapping)
        np.testing.assert_array_equal(old,np.load(source/'assignment_oracle_pred.npy'))
        all_rows=np.ones(N,bool);none=np.zeros(N,bool)
        policies=[('global',1,g,none,none),('fixed_region',2,fixed,all_rows,all_rows)]+[(f'expert{k+1}_always',3,E[:,k],all_rows,all_rows) for k in range(K)]+[('scorer_only',4,sonly,calls,calls),('scorer_verifier',5,sv,calls,accepted)]
        summary=[];pc=[];cm=[]
        for name,condition,pred,cl,ac in policies:
            r,rows,matrix=metrics(y,pred,g,names,tail,name,condition,cl,ac);summary.append(r);pc+=rows;cm+=matrix
        control,control_pc,_=metrics(y,constant_pred,g,names,tail,'region_constant',0,all_rows,all_rows)
        pd.DataFrame([control]).to_csv(out/'3b_constant_control_summary.csv',index=False);pd.DataFrame(control_pc).to_csv(out/'3c_constant_control_per_class.csv',index=False)
        # Region composition confounds raw accuracy: retain macro-F1 and train-constant control.
        region_rows=[]
        for k in range(K):
            mask=regions==k;local_y=y[mask];local_g=g[mask];present=np.flatnonzero(np.bincount(local_y,minlength=C))
            for model,pred in [('global',g),(f'expert{k+1}',E[:,k]),('train_region_constant',constant_pred)]:
                r,pr,_=metrics(local_y,pred[mask],local_g,names,tail,model,0,np.ones(mask.sum(),bool),np.ones(mask.sum(),bool))
                region_rows.append(dict(region=k+1,model=model,rows=int(mask.sum()),classes_present=len(present),accuracy=r['accuracy'],macro_f1_present=float(np.mean([pr[c]['f1'] for c in present]))))
        pd.DataFrame(region_rows).to_csv(out/'3d_region_constant_comparison.csv',index=False)
        new=pd.DataFrame(summary);old_summary=pd.read_csv(source/'1a_summary.csv').set_index('policy')
        for _,r in new.iterrows():
            if r.policy=='fixed_region':continue
            for col in ['macro_f1','accuracy','calls','accepted','changed','H','D']:assert abs(r[col]-old_summary.loc[r.policy,col])<1e-12
        new.to_csv(out/'1a_summary.csv',index=False);pd.DataFrame(pc).to_csv(out/'1b_per_class.csv',index=False);pd.DataFrame(cm).to_csv(out/'1c_confusion.csv',index=False)
        # Trace which predictor supplied each target-positive output, without any oracle repair.
        target='infiltration' if ds=='cic2018' else 'scanning';c=names.index(target);origin_rows=[]
        origins={'historical_class_oracle':mapping[y]+1,'scorer_only':np.where(calls,scores['winner']+1,0),'scorer_verifier':np.where(accepted,scores['winner']+1,0),'fixed_region':regions+1}
        for label,pred in [('historical_class_oracle',old),('scorer_only',sonly),('scorer_verifier',sv),('fixed_region',fixed)]:
            origin=origins[label]
            for k in range(K+1):
                m=(origin==k)&(pred==c)
                origin_rows.append(dict(policy=label,source_model='global' if k==0 else f'expert{k}',target_class=target,TP=int((m&(y==c)).sum()),FP=int((m&(y!=c)).sum())))
        pd.DataFrame(origin_rows).to_csv(out/'5a_target_prediction_sources.csv',index=False)
        # Preserve compatible probability and per-expert diagnostics with explicit reuse provenance.
        reused=['0a_assignment_map','0b_fixed_fpr_thresholds','2b_region_matrix','3a_specialization','4a_fixed_fpr','4b_average_precision','4c_test_roc_metrics']
        for stem in reused:shutil.copy2(source/(stem+'.csv'),out/(stem+'.csv'))
        grid=pd.read_csv(policy/f'4a_{protocol["arm"]}_threshold_grid.csv')
        audit=dict(dataset=ds,clean_manifest=str(manifest_path),clean_manifest_sha256=sha256(manifest_path),clean_rule=manifest['rule'],
                   split_rule=manifest['split_rule'],split_rows=manifest['split_rows'],rows_removed=manifest['rows_removed'],remaining_conflicting_vector_groups=0,
                   global_equals_s_only=bool(np.array_equal(g,sonly)),global_equals_s_v=bool(np.array_equal(g,sv)),
                   calibration_grid_reasons=grid.reason.fillna('feasible').value_counts().to_dict(),
                   frozen_prediction_match=True,probability_diagnostics_reused={s:sha256(source/(s+'.csv')) for s in reused},
                   historical_target=pd.read_csv(source/'1b_per_class.csv').query('`class` == @target').to_dict('records'))
        dump(out/'audit.json',audit)
        dump(out/'COMPLETE.json',dict(rows=N,n_experts=K,dataset=ds,cache_identity=protocol['cache_identity'],source_policy_reproduced_exactly=True,
                                     bank_satisfies_K_less_than_C=K<C,new_expert_training=False,macro_f1=dict(zip(new.policy,new.macro_f1)),constant_control_macro_f1=control['macro_f1']))
        print(ds,new[['policy','macro_f1']].to_string(index=False),'\nTarget prediction sources:\n',pd.DataFrame(origin_rows).query('policy == "scorer_verifier" and (TP > 0 or FP > 0)').to_string(index=False),flush=True)
    print(f'DONE in {time.monotonic()-started:.1f}s: {a.out_root}',flush=True)

if __name__=='__main__':main()

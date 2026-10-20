#!/usr/bin/env python3
"""Inspect the unchanged experts on held-out calibration rows, never test."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,gc,json
import numpy as np
from exp61_utils import write
from current_model import CurrentPredictor,infer,candidates
from finetune_final import proposed_data
from replay_protocol import confusion_metrics
from audit_proposed_ton import SOURCE


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);run=p.parse_args().run.resolve()
    import torch
    from tabpfn import TabPFNClassifier
    cfg=json.loads((run/'config.json').read_text());torch.set_num_threads(cfg['threads'])
    meta=json.loads((run/'data/toniot/manifest.json').read_text());base=Path(meta['source'])
    arr={k:np.load(base/(k+'.npy'),mmap_mode='r') for k in ['train_X','train_y']}
    plan=json.loads((run/'data/toniot/plans.json').read_text())[-1];data=proposed_data(arr,meta['class_names'],plan)
    dest=run/'rebuild/stage_5';names=data['names'];X=data['X_cal'];y=data['y_cal']
    wrapper=CurrentPredictor(dest,cfg);model=wrapper.models[0]
    # Resolve the very small original-vs-mixed argmax discrepancies using the
    # original query groups, without accessing their ground-truth test labels.
    original=np.load(SOURCE/'jobs/toniot_ours/predictions.npy',mmap_mode='r')
    mixed=np.load(run/'jobs/saved_proposed/predictions.npy',mmap_mode='r')
    mixed_prob=np.load(run/'jobs/saved_proposed/probabilities.npy',mmap_mode='r')
    changed=np.flatnonzero(original!=mixed);test_X=np.load(base/'test_X.npy',mmap_mode='r')
    discrepancies=[]
    for batch_start in np.unique((changed//8192)*8192):
        prob=np.asarray(model.predict_proba(np.asarray(test_X[batch_start:batch_start+8192])))
        ids=changed[(changed>=batch_start)&(changed<batch_start+8192)]
        for i in ids:
            oldp=prob[i-batch_start];newp=np.asarray(mixed_prob[i]);top=np.sort(newp)[-2:]
            discrepancies.append(dict(row=int(i),original_prediction=int(original[i]),mixed_prediction=int(mixed[i]),
                original_group_recomputed=int(oldp.argmax()),mixed_top_two_margin=float(top[1]-top[0]),
                max_probability_difference=float(np.max(np.abs(oldp-newp)))))
    write(run/'query_order_discrepancies.json',dict(changed_rows=len(changed),rows=discrepancies,
        original_group_reproduces_original=all(r['original_prediction']==r['original_group_recomputed'] for r in discrepancies),
        test_labels_used=False))
    p0=np.asarray(model.predict_proba(X));g=p0.argmax(1)
    z=wrapper.state['pca'].transform(infer(model,X,cfg['prepare_batch'],True)).astype('float32')
    contexts=json.loads((dest/'contexts.json').read_text());ps=[];stats=[]
    for k in range(4):
        expert=TabPFNClassifier.load_from_fit_state(dest/f'model_{k+1}.tabpfn_fit',device='cuda:0')
        pk=np.asarray(expert.predict_proba(X));ep=pk.argmax(1);ps.append(pk)
        stats.append(dict(expert=k+1,context_rows=len(contexts[k]['anchor_indices'])+len(contexts[k]['block_indices']),
            anchor_rows=len(contexts[k]['anchor_indices']),residual_rows=len(contexts[k]['block_indices']),
            changed=int((ep!=g).sum()),helpful=int(((g!=y)&(ep==y)).sum()),harmful=int(((g==y)&(ep!=y)).sum()),
            metrics=confusion_metrics(y,ep,names)))
        del expert;gc.collect();torch.cuda.empty_cache()
    pre,post,candidate,winner=candidates(wrapper.state,p0,z,ps)
    experts=np.array([p.argmax(1) for p in ps])
    potential=(g!=y)&(experts==y).any(0)
    write(run/'calibration_diagnostic.json',dict(rows=len(y),data_scope='original held-out D_cal only; no threshold adjustment',
        global_metrics=confusion_metrics(y,g,names),experts=stats,
        any_expert_can_correct_global_error=int(potential.sum()),
        scorer_selected_candidate_helpful=int(((g!=y)&(candidate==y)).sum()),
        scorer_selected_candidate_harmful=int(((g==y)&(candidate!=y)).sum()),
        positive_pre_rows=int((pre>0).sum()),positive_post_rows=int((post>0).sum()),
        final_policy=wrapper.state['thresholds'],
        oracle_note='Any-expert correction count uses calibration labels for diagnostic upper bound only; never a deployable routing score'))
    np.savez_compressed(run/'calibration_predictions.npz',labels=y,global_probabilities=p0,expert_probabilities=np.array(ps),
        pre=pre,post=post,candidate=candidate,winner=winner)
    wrapper.close()
    print(json.dumps(dict(experts=[{k:v for k,v in s.items() if k!='metrics'} for s in stats],
        any_expert_can_correct=int(potential.sum()),candidate_helpful=int(((g!=y)&(candidate==y)).sum()),candidate_harmful=int(((g==y)&(candidate!=y)).sum()))))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Rebuild the small-label ToN proposal and isolate its Global context."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse
import gc
import importlib.metadata
import json
import shutil
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from exp61_utils import sha, write
from finetune_final import load_data, proposed_data
from current_model import fit_current, CurrentPredictor
from replay_models import make_base, timed
from replay_protocol import confusion_metrics

SOURCE = _root/'tabpfn/results/v7/exp77/20261015_230535_initialft10_finalfull_s43_2eb2237d'
MIXED = _root/'tabpfn/results/v7/exp81/20261016_025421_mixed_query_final_s43_b6ba1d34'


def progress(out, phase, **kw):
    value = dict(phase=phase, updated_epoch=time.time(), **kw)
    write(out/'progress.json', value)
    print(json.dumps(value), flush=True)


def evaluate(out, label, model, arr, names, order, reference, context_rows):
    dest = out/'jobs'/label
    dest.mkdir(parents=True, exist_ok=True)
    predict = model.predict if isinstance(model, CurrentPredictor) else model.predict_proba
    n = len(order); C = len(names)
    pred = np.lib.format.open_memmap(dest/'predictions.npy', mode='w+', dtype='int16', shape=(n,))
    prob = np.lib.format.open_memmap(dest/'probabilities.npy', mode='w+', dtype='float32', shape=(n,C))
    calls = accepted = 0; times = []
    for i, start in enumerate(range(0,n,8192)):
        ids = order[start:start+8192]
        p, seconds = timed(lambda: predict(np.asarray(arr['test_X'][ids])))
        assert p.shape == (len(ids),C) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-3)
        pred[ids] = np.argmax(p,axis=1); prob[ids] = p
        times.append(seconds)
        if isinstance(model, CurrentPredictor):
            calls += int(model.aux['calls'].sum()); accepted += int(model.aux['accepted'].sum())
        if i%30 == 0 or start+len(ids) == n:
            done = start+len(ids)
            progress(out,label,rows_done=done,rows_total=n,remaining_seconds=(n-done)*sum(times)/done)
    pred.flush(); prob.flush()
    metrics = confusion_metrics(arr['test_y'],np.asarray(pred),names)
    # Independent implementation validates aggregate and class scores.
    cm = confusion_matrix(arr['test_y'],pred,labels=np.arange(C))
    np.testing.assert_array_equal(cm,np.asarray(metrics['confusion']))
    f1 = 2*np.diag(cm)/(cm.sum(0)+cm.sum(1))
    assert abs(float(f1.mean())-metrics['macro_f1']) < 1e-12
    previous = np.load(reference,mmap_mode='r')
    result = dict(condition=label,dataset='toniot',context_rows=context_rows,total_label_budget=1700,
        seed=43,initial_ids_finetuning=False,test_rows=n,test_order='EXP81 seed43 mixed',batch=8192,
        metrics=metrics,calls=calls,accepted=accepted,predictions_path=str(dest/'predictions.npy'),
        probabilities_path=str(dest/'probabilities.npy'),source_prediction_reference=str(reference),
        reference_prediction_mismatches=int(np.count_nonzero(pred!=previous)),
        inference_seconds=sum(times),full_reinference=True,independent_metrics_verified=True)
    write(dest/'results.json',result)
    return result


def compare_files(a,b):
    pa=np.load(a['predictions_path'],mmap_mode='r');pb=np.load(b['predictions_path'],mmap_mode='r')
    xa=np.load(a['probabilities_path'],mmap_mode='r');xb=np.load(b['probabilities_path'],mmap_mode='r')
    maximum=0.; absolute_sum=0.
    for start in range(0,len(pa),8192):
        d=np.abs(xa[start:start+8192]-xb[start:start+8192]);maximum=max(maximum,float(d.max()));absolute_sum+=float(d.sum(dtype=np.float64))
    return dict(a=a['condition'],b=b['condition'],prediction_mismatches=int(np.count_nonzero(pa!=pb)),
                max_probability_difference=maximum,mean_probability_difference=absolute_sum/xa.size)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=True)
    import torch
    from tabpfn import TabPFNClassifier
    cfg=json.loads((SOURCE/'config.json').read_text());torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    write(out/'config.json',dict(cfg,source_experiment=str(SOURCE),initial_finetuning=False,diagnostic='exp82'))
    source_job=SOURCE/'jobs/toniot_ours';saved=source_job/'stage_5'
    files=[SOURCE/'config.json',SOURCE/'data/toniot/plans.json',SOURCE/'data/toniot/manifest.json',
           source_job/'predictions.npy',saved/'model_0.tabpfn_fit',saved/'state.joblib',Path(cfg['checkpoint']),
           MIXED/'data/toniot/query_order.npy',SOURCE/'jobs/toniot_pfn_pretrained/final_model.tabpfn_fit',
           MIXED/'jobs/small_toniot_new_100_tabpfn/predictions.npy']
    dep_names=['scripts/v7/exp73/current_model.py','scripts/v7/exp73/scenario_data.py',
        'scripts/v7/exp77/finetune_final.py','scripts/v7/exp72/replay_models.py',
        'scripts/v7/exp71/exp71_bounded_updates.py','tabpfn/scripts/v4/exp51/nfv3_v3_exp51_scorer_target.py',
        'tabpfn/scripts/v2/exp38/nfv3_v3_exp38_frozen_cache.py']
    for rel in dep_names:
        p=_root/rel
        if not p.exists():
            matches=list((_root/'tabpfn/scripts').glob('v*/*/'+Path(rel).name));assert len(matches)==1;p=matches[0]
        files.append(p);dest=out/'source_dependencies'/p.relative_to(_root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    files += [Path(__file__),_root/'configs/experiments/exp82_protocol.json']
    hashes={str(p):sha(p) for p in files}
    write(out/'source_hashes.json',hashes)
    write(out/'environment.json',{name:importlib.metadata.version(name) for name in ['torch','numpy','scikit-learn','tabpfn','xgboost']})
    arr,names,plans=load_data(out,'toniot',43)
    assert plans==json.loads((SOURCE/'data/toniot/plans.json').read_text()),'Regenerated role plans differ'
    plan=plans[-1];C=len(names);ids=np.asarray(plan['released'])
    assert len(ids)==1700 and len(plan['ours_roles']['global'])==695
    np.testing.assert_array_equal(np.unique(arr['train_y'][ids]),np.arange(C))
    roles={role:dict(rows=len(index),class_counts=np.bincount(arr['train_y'][index],minlength=C).tolist()) for role,index in plan['ours_roles'].items()}
    order=np.load(MIXED/'data/toniot/query_order.npy',mmap_mode='r')
    np.testing.assert_array_equal(order,np.random.default_rng(43).permutation(len(arr['test_y'])))
    write(out/'data_audit.json',dict(source_manifest=json.loads((SOURCE/'data/toniot/manifest.json').read_text()),
        class_names=names,roles=roles,plans_identical=True,all_roles_class_complete=True,
        roles_feature_disjoint=True,train_test_row_ids_disjoint=True,query_order_sha256=sha(MIXED/'data/toniot/query_order.npy')))
    jobs=[];reference=source_job/'predictions.npy'
    progress(out,'restore_original_final')
    model=CurrentPredictor(saved,cfg)
    np.testing.assert_array_equal(model.models[0].classes_,np.arange(C))
    assert model.state['names']==names
    jobs.append(evaluate(out,'saved_proposed',model,arr,names,order,reference,695))
    model.close();del model;gc.collect();torch.cuda.empty_cache()
    stage_audits=[]
    for s,p in enumerate(plans):
        progress(out,'rebuild_proposed',stage=s,stages=len(plans))
        dest=out/'rebuild'/f'stage_{s}'
        info=fit_current(proposed_data(arr,names,p),cfg,'toniot',dest)
        write(dest/'proposed_fit.json',info)
        old=json.loads((source_job/f'stage_{s}/selection.json').read_text())
        new=json.loads((dest/'selection.json').read_text())
        stage_audits.append(dict(stage=s,global_rows=info['global_rows'],selection_identical=new==old,
                                selection=new,preparation_seconds=info['update_seconds']))
    write(out/'stage_audit.json',stage_audits)
    model=CurrentPredictor(out/'rebuild/stage_5',cfg)
    np.testing.assert_array_equal(model.models[0].classes_,np.arange(C))
    jobs.append(evaluate(out,'rebuilt_proposed',model,arr,names,order,reference,695))
    model.close();del model;gc.collect();torch.cuda.empty_cache()
    # No routing, embedding extraction, calibration or serialization before this
    # control predicts: it isolates precisely the proposed Global context.
    for label,index,ref in [('fresh_global_695',plan['ours_roles']['global'],reference),
                           ('fresh_full_1700',plan['released'],MIXED/'jobs/small_toniot_new_100_tabpfn/predictions.npy')]:
        progress(out,'fit_control',condition=label,context_rows=len(index))
        model=make_base('tabpfn',cfg,C)
        _,elapsed=timed(lambda:model.fit(np.asarray(arr['train_X'][index]),np.asarray(arr['train_y'][index])))
        np.testing.assert_array_equal(model.classes_,np.arange(C))
        j=evaluate(out,label,model,arr,names,order,ref,len(index));j['fit_seconds']=elapsed
        model.save_fit_state(out/'jobs'/label/'model.tabpfn_fit');write(out/'jobs'/label/'results.json',j);jobs.append(j)
        del model;gc.collect();torch.cuda.empty_cache()
    comparisons=[compare_files(jobs[0],jobs[1]),compare_files(jobs[1],jobs[2]),compare_files(jobs[2],jobs[3])]
    grid=pd.read_csv(out/'rebuild/stage_5/threshold_grid.csv')
    candidates=grid[~grid['reason'].fillna('').str.startswith('selected:')]
    threshold_audit=dict(calibration_rows=198,minimum_helpful_plus_harmful=30,
        candidate_threshold_pairs=len(candidates),maximum_helpful_plus_harmful=int((candidates.helpful+candidates.harmful).max()),
        candidate_pairs_failing_support=int(candidates['reason'].fillna('').str.contains('support').sum()),
        feasible_candidate_pairs=int(candidates.feasible.sum()),selected=stage_audits[-1]['selection'])
    assert all(sha(p)==digest for p,digest in hashes.items()),'Original source changed'
    write(out/'results.json',dict(state='complete',jobs=jobs,comparisons=comparisons,threshold_audit=threshold_audit,
        note='Diagnostic only: same seed, same labels, no hyperparameter or threshold changes'))
    write(out/'verification.json',dict(passed=True,independent_metrics=True,all_test_rows_evaluated=True,
        regenerated_plans_identical=True,source_hashes_unchanged=True,class_mapping_verified=True,
        stage_selections_identical=all(s['selection_identical'] for s in stage_audits),comparisons=comparisons))
    progress(out,'complete',results=[dict(condition=j['condition'],macro_f1=j['metrics']['macro_f1'],reference_mismatches=j['reference_prediction_mismatches']) for j in jobs])


if __name__=='__main__':main()

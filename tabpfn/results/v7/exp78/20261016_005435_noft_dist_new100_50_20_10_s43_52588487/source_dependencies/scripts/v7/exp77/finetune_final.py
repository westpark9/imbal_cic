#!/usr/bin/env python3
"""Initial IDS weight adaptation, context updates, and final-state evaluation."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)
import argparse
import gc
import hashlib
import json
import time
import traceback
import numpy as np
from exp61_utils import write, sha
from replay_models import timed, make_base, local_request
from replay_protocol import confusion_metrics, dist_adjust_current_batch
from scenario_data import assign_new_roles
from run_context_policies import replace_context

PREVIOUS = ROOT/'tabpfn/results/v7/exp76/20261015_212937_context1000_new100_s43_1c62d14c'
CHECKPOINT = ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'
METHODS = ['xgb','pfn_pretrained','pfn_ft_accumulate','pfn_ft_fifo','pfn_ft_similar','pfn_ft_random','pfn_ft_fixed','pfn_ft_repeated','ours','boostpfn']


def weights_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_data(out, ds, seed):
    old = PREVIOUS/'data'/ds
    manifest = json.loads((old/'manifest.json').read_text())
    base = Path(manifest['source'])
    arr = {k: np.load(base/f'{k}.npy', mmap_mode='r') for k in ['train_X','train_y','train_hash','train_ids','test_X','test_y','test_hash','test_ids']}
    plans = json.loads((old/'plans.json').read_text())
    # Persist feature-group-disjoint roles as new labeled rows arrive.
    known_ft = {}; known_ours = {}; ft = {'train': [], 'val': []}
    ours = {r: [] for r in ['global','expert','route','tune','cal']}
    for s, plan in enumerate(plans):
        ids = np.asarray(plan['incoming'], dtype='int64')
        for target, known, fractions in [(ft, known_ft, (.9,.1)), (ours, known_ours, (.4,.2,.2,.08,.12))]:
            split = assign_new_roles(ids, arr['train_y'], arr['train_hash'], known, seed+s*104729, tuple(target), fractions)
            for role, values in split.items(): target[role].extend(values.tolist())
        plan['ft_roles'] = {r: list(v) for r,v in ft.items()}
        plan['ours_roles'] = {r: list(v) for r,v in ours.items()}
        for roles in (plan['ft_roles'], plan['ours_roles']):
            assert sorted(sum(roles.values(), [])) == sorted(plan['released'])
            for i,r in enumerate(roles):
                assert set(arr['train_y'][roles[r]]) == set(arr['train_y'][plan['released']]), (ds,s,r)
                for q in list(roles)[i+1:]:
                    assert not np.intersect1d(arr['train_hash'][roles[r]],arr['train_hash'][roles[q]]).size
    assert not np.intersect1d(arr['train_ids'][plans[-1]['released']], arr['test_ids']).size
    dest = out/'data'/ds; dest.mkdir(parents=True, exist_ok=True)
    write(dest/'plans.json', plans)
    write(dest/'manifest.json', dict(manifest, evaluation='single final model on complete fixed test in original file order',
        source_plans_sha256=sha(old/'plans.json'), new_plans_sha256=sha(dest/'plans.json'),
        validation_within_released_budget=True, test_labels_used_for_training=False))
    return arr, manifest['class_names'], plans


def finetune(out, cfg, arr, plan, source):
    import torch
    from tabpfn.finetuning import FinetunedTabPFNClassifier
    from tabpfn.model_loading import save_tabpfn_model
    out.mkdir(parents=True, exist_ok=True)
    if (out/'audit.json').exists(): return out/'weights.ckpt', json.loads((out/'audit.json').read_text())
    tr = np.asarray(plan['ft_roles']['train']); va = np.asarray(plan['ft_roles']['val'])
    classes = np.unique(arr['train_y'][plan['released']])
    labels = lambda ids: np.searchsorted(classes, arr['train_y'][ids])
    ft = FinetunedTabPFNClassifier(device='cuda:0', epochs=cfg['epochs'], learning_rate=1e-5,
        weight_decay=.01, n_finetune_ctx_plus_query_samples=1024, finetune_ctx_query_split_ratio=.2,
        n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=4,
        early_stopping=False, early_stopping_patience=3, eval_metric='log_loss', random_state=cfg['seed'],
        use_lr_scheduler=False, save_checkpoint_interval=None, use_fixed_preprocessing_seed=False,
        extra_classifier_kwargs={'model_path': str(source), 'n_preprocessing_jobs': 1, 'auto_scale_n_estimators': False})
    before = torch.load(source, map_location='cpu', weights_only=False)['state_dict']
    torch.cuda.reset_peak_memory_stats()
    _, elapsed = timed(lambda: ft.fit(np.asarray(arr['train_X'][tr]), labels(tr),
        X_val=np.asarray(arr['train_X'][va]), y_val=labels(va), output_dir=out/'training'))
    save_tabpfn_model(ft.finetuned_estimator_, out/'weights.ckpt')
    after = torch.load(out/'weights.ckpt', map_location='cpu', weights_only=False)['state_dict']
    changed = [k for k in after if k in before and not torch.equal(after[k], before[k])]
    assert changed, 'Fixed-epoch fine-tuning did not change any parameter tensors'
    audit = dict(seconds=elapsed, source=str(source), checkpoint=str(out/'weights.ckpt'),
        before_sha256=weights_digest(before), after_sha256=weights_digest(after), changed_tensors=len(changed),
        total_tensors=len(after), train_indices=tr.tolist(), validation_indices=va.tolist(),
        train_rows=len(tr), validation_rows=len(va), classes=classes.tolist(),
        epochs_max=cfg['epochs'], learning_rate=1e-5, optimizer='AdamW', loss='cross_entropy',
        best_epoch_selection='none; fixed epoch budget; validation log_loss monitored only; no test access',
        peak_torch_gib=torch.cuda.max_memory_allocated()/2**30)
    write(out/'audit.json', audit)
    del ft, before, after; gc.collect(); torch.cuda.empty_cache()
    return out/'weights.ckpt', audit


def proposed_data(arr, names, plan):
    classes=np.unique(arr['train_y'][plan['released']])
    data={'names':[names[c] for c in classes]}
    for role,ids in plan['ours_roles'].items():
        ids=np.asarray(ids,dtype='int64');data['i_'+role]=ids
        data['X_'+role]=np.asarray(arr['train_X'][ids]);data['y_'+role]=np.searchsorted(classes,arr['train_y'][ids])
    ids=np.asarray(plan['released']);va=np.asarray(plan['ours_roles']['tune']+plan['ours_roles']['cal'])
    data.update(Xtr=np.asarray(arr['train_X'][ids]),ytr=np.searchsorted(classes,arr['train_y'][ids]),
        Xva=np.asarray(arr['train_X'][va]),yva=np.searchsorted(classes,arr['train_y'][va]))
    return data


def run_method(out,cfg,ds,arr,names,plans,initial_checkpoint,initial_audit,method):
    import torch
    from current_model import fit_current, CurrentPredictor
    import exp61_sota_worker as sota
    job=out/'jobs'/f'{ds}_{method}';job.mkdir(parents=True,exist_ok=True)
    if (job/'COMPLETE.json').exists():return json.loads((job/'results.json').read_text())
    X=arr['train_X'];y=arr['train_y'];model=None;old=None;fits=[];current_checkpoint=initial_checkpoint
    is_ft=method.startswith('pfn_ft_')
    expected_digest=initial_audit['after_sha256'] if is_ft else None
    for stage,plan in enumerate(plans):
        stageout=job/f'stage_{stage}';stageout.mkdir(exist_ok=True)
        progress=dict(dataset=ds,method=method,stage=stage,phase='fit',updated_epoch=time.time())
        print(json.dumps(progress),flush=True);write(out/'progress.json',progress)
        start=time.perf_counter();incoming=np.asarray(plan['incoming'],dtype='int64');removed=np.array([],dtype='int64')
        if not stage:ctx=incoming
        elif method=='pfn_ft_fixed':ctx=old
        elif method in ('pfn_ft_fifo','pfn_ft_similar','pfn_ft_random'):
            ctx,removed=replace_context(old,incoming,y,X,method.removeprefix('pfn_ft_'),cfg['seed']+stage*1009)
        else:ctx=np.asarray(plan['released'],dtype='int64')
        selection=time.perf_counter()-start
        classes=np.unique(y[ctx]);labels=np.searchsorted(classes,y[ctx]);ft_seconds=0.;fit_seconds=0.;construct=0.
        if is_ft and not stage:ft_seconds=initial_audit['seconds']
        if method=='pfn_ft_repeated' and stage:
            del model;model=None;gc.collect();torch.cuda.empty_cache()
            current_checkpoint,ft_audit=finetune(job/'finetuning'/f'stage_{stage}',cfg,arr,plan,current_checkpoint)
            expected_digest=ft_audit['after_sha256'];ft_seconds=ft_audit['seconds']
        if stage and method=='pfn_ft_fixed':selection=0.
        elif method=='ours':
            info=fit_current(proposed_data(arr,names,plan),cfg,ds,stageout)
            fit_seconds=info['update_seconds']
            info['protocol_adaptations']=['All five feature-disjoint roles share the same 1000/1400/1700 revealed-label budget; no external validation',
                'Original pretrained backbone; same EXP62 K=4 anchor/residual S1V0 architecture; no test tuning']
            write(stageout/'proposed_fit.json',info)
        elif method=='boostpfn':
            request=local_request(cfg,ds)
            (base,model,split),construct=timed(lambda:sota.boost_factory(request))
            boostX=np.asarray(X[ctx]);boosty=labels.astype('int64')
            _,fit_seconds=timed(lambda:model.fit(torch.from_numpy(boostX),torch.from_numpy(boosty),torch.empty((0,X.shape[1]))))
            np.savez_compressed(stageout/'boost_state.npz',sampled_idxs=np.asarray(model.sampled_idxs),alphas=np.asarray(model.alphas))
        else:
            if model is None or method=='xgb':
                model,construct=timed(lambda:make_base('xgb' if method=='xgb' else 'tabpfn',
                    dict(cfg,checkpoint=str(current_checkpoint if is_ft else CHECKPOINT)),len(classes)))
            _,fit_seconds=timed(lambda:model.fit(X[ctx],labels))
        digest=None
        if method.startswith('pfn_'):
            digest=weights_digest(model.models_[0].state_dict())
            if expected_digest is None:expected_digest=digest
            assert digest==expected_digest, (method,stage,'context update changed weights')
        info=dict(stage=stage,context_indices=ctx.tolist(),removed_indices=removed.tolist(),context_rows=len(ctx),
            released_rows=len(plan['released']),classes=[names[c] for c in classes],
            selection_seconds=selection,construct_seconds=construct,fit_seconds=fit_seconds,finetuning_seconds=ft_seconds,
            update_seconds=selection+construct+fit_seconds+ft_seconds,weights_sha256=digest)
        write(stageout/'fit_metrics.json',info);fits.append(info);old=ctx
    # Only the FINAL state predicts the COMPLETE test. No earlier-stage predictions are pooled.
    if method=='ours':model=CurrentPredictor(stageout,cfg)
    elif method=='xgb':model.save_model(job/'final_model.ubj')
    elif method.startswith('pfn_'):model.save_fit_state(job/'final_model.tabpfn_fit')
    def predict(A):
        if method=='ours':return model.predict(A)
        if method=='boostpfn':return sota.replay_boost(base,model,split,boostX,boosty,A,cfg['batch'])
        return model.predict_proba(A)
    total=len(arr['test_y']);pred=np.lib.format.open_memmap(job/'predictions.npy',mode='w+',dtype='int16',shape=(total,))
    dist_methods=cfg.get('dist_methods',['pfn_ft_accumulate'])
    distpred=np.lib.format.open_memmap(job/'dist_predictions.npy',mode='w+',dtype='int16',shape=(total,)) if method in dist_methods else None
    services=[];adjustment_seconds=0.;calls=accepted=0
    prior=np.bincount(labels,minlength=len(classes))/len(labels)
    for a in range(0,total,cfg['batch']):
        b=min(a+cfg['batch'],total)
        p,seconds=timed(lambda:predict(np.asarray(arr['test_X'][a:b])))
        assert p.shape==(b-a,len(classes)) and np.isfinite(p).all()
        assert np.allclose(p.sum(1),1,atol=1e-3)
        pred[a:b]=classes[p.argmax(1)];services.append([a,b,seconds])
        if method=='ours':calls+=int(model.aux['calls'].sum());accepted+=int(model.aux['accepted'].sum())
        if distpred is not None:
            adjusted,sec=timed(lambda:dist_adjust_current_batch(p,prior));adjustment_seconds+=sec
            distpred[a:b]=classes[adjusted.argmax(1)]
        if (a//cfg['batch'])%50==0:
            write(out/'progress.json',dict(dataset=ds,method=method,phase='final_test',rows_done=b,rows_total=total,updated_epoch=time.time()))
    pred.flush();np.save(job/'service_times.npy',np.asarray(services))
    mask=~np.isin(arr['test_hash'],arr['train_hash'][plans[-1]['released']])
    result=dict(dataset=ds,method=method,seed=cfg['seed'],evaluation='final_full_test',final_stage=len(plans)-1,
        test_rows=total,test_full=True,metrics=confusion_metrics(arr['test_y'],np.asarray(pred),names),fits=fits,
        feature_disjoint_diagnostic=confusion_metrics(arr['test_y'][mask],np.asarray(pred)[mask],names),
        initial_seconds=fits[0]['update_seconds'],initial_finetuning_seconds=initial_audit['seconds'] if is_ft else 0.,
        update_seconds=sum(f['update_seconds'] for f in fits[1:]),serving_seconds=sum(s[2] for s in services),
        total_preparation_seconds=sum(f['update_seconds'] for f in fits),calls=calls,accepted=accepted,
        checkpoint=str(current_checkpoint if is_ft else CHECKPOINT),label_budget=len(plans[-1]['released']),
        note='Original pretrained backbone for ours, role-held-out labels never used for IDS gradient training; FT comparators reuse identical initial fine-tune')
    write(job/'results.json',result)
    if distpred is not None:
        distpred.flush();dist=dict(result);dist.update(method='distpfn_ft' if is_ft else 'distpfn_pretrained',
            metrics=confusion_metrics(arr['test_y'],np.asarray(distpred),names),
            feature_disjoint_diagnostic=confusion_metrics(arr['test_y'][mask],np.asarray(distpred)[mask],names),
            serving_seconds=result['serving_seconds']+adjustment_seconds,
            note=f'Current-batch posterior adjustment on {method}; shared backbone/context/preparation; no query labels')
        write(job/'dist_results.json',dist)
    write(job/'COMPLETE.json',dict(complete=True,predictions_sha256=sha(job/'predictions.npy')))
    if method=='ours':model.close()
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def run_dataset(out,cfg,ds,arr,names,plans,checkpoint,audit):
    for method in cfg['methods']:
        try:
            run_method(out,cfg,ds,arr,names,plans,checkpoint,audit,method)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            write(out/'jobs'/f'{ds}_{method}'/'FAILED.json',dict(error=error,epoch=time.time()))
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        jobs=[]
        for p in sorted((out/'jobs').glob('*/results.json')):
            jobs.append(json.loads(p.read_text()));dist=p.with_name('dist_results.json')
            if dist.exists():jobs.append(json.loads(dist.read_text()))
        write(out/'results.json',dict(state='running',jobs=jobs))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',action='store_true');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--datasets',nargs='+',default=['cic2018','toniot']);p.add_argument('--methods',nargs='+',default=METHODS)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=dict(seed=43,threads=8,batch=8192,prepare_batch=4096,experts=4,epochs=a.epochs,
        checkpoint=str(CHECKPOINT),source=str(PREVIOUS),protocol='exp77.1',methods=a.methods,datasets=a.datasets,
        evaluation='final_full_test',initial_finetuning=True,html_publication=False)
    write(a.out/'config.json',cfg)
    import torch
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    for ds in a.datasets:
        arr,names,plans=load_data(a.out,ds,cfg['seed'])
        checkpoint,audit=finetune(a.out/'finetuning'/ds/'initial',cfg,arr,plans[0],CHECKPOINT)
        if a.profile:
            ids=np.asarray(plans[-1]['released']); classes=np.unique(arr['train_y'][ids])
            model=make_base('tabpfn',dict(cfg,checkpoint=str(checkpoint)),len(classes))
            _,fit_seconds=timed(lambda:model.fit(arr['train_X'][ids],np.searchsorted(classes,arr['train_y'][ids])))
            _,cold=timed(lambda:model.predict_proba(arr['test_X'][:cfg['batch']]))
            _,warm=timed(lambda:model.predict_proba(arr['test_X'][cfg['batch']:5*cfg['batch']]))
            write(a.out/f'{ds}_profile.json',dict(finetuning=audit,fit_seconds=fit_seconds,first_batch_seconds=cold,
                warm_rows=4*cfg['batch'],warm_seconds=warm,estimated_full_inference_seconds=warm/(4*cfg['batch'])*len(arr['test_y'])))
            del model;gc.collect();torch.cuda.empty_cache()
        else:
            run_dataset(a.out,cfg,ds,arr,names,plans,checkpoint,audit)
    if not a.profile:
        result=json.loads((a.out/'results.json').read_text())
        expected=len(a.datasets)*(len(a.methods)+int('pfn_ft_accumulate' in a.methods))
        result.update(state='complete' if len(result['jobs'])==expected else 'incomplete',expected_jobs=expected)
        write(a.out/'results.json',result)
        if result['state']!='complete':raise RuntimeError('Some methods did not complete; inspect FAILED.json')


if __name__=='__main__':main()

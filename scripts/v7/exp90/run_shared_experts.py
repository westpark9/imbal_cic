#!/usr/bin/env python3
"""EXP90: shared labels, grouped cross-fitting, expert ablations, automatic report."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)

import argparse
import gc
import importlib.metadata
import json
import pickle
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from sklearn.metrics import confusion_matrix
from exp61_utils import sha
from replay_models import timed
from shared_experts import Component, COMPONENTS, METHODS, folds, fit_combiner, combine, dependencies

PROTOCOL=ROOT/'configs/experiments/exp90_protocol.json'


def read(p): return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def stamp(): return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def progress(out,phase,**values):
    value=dict(at=stamp(),phase=phase,**values)
    write(out/'progress.json',value)
    with (out/'events.jsonl').open('a') as f: f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def condition_id(ds,regime,n,seed): return f'{ds}_{regime}_n{n}_s{seed}'


def conditions(cfg):
    d=cfg['design']
    return [(ds,r,n,s) for n in d['budgets'] for s in d['seeds'] for ds in d['datasets'] for r in d['benign_regimes']]


def full_condition(cfg,regime,n):
    d=cfg['design']['full_test']
    return regime==d['benign_regime'] and n in d['budgets']


def prepare(out,cfg):
    if (out/'PREPARED.json').exists(): return
    old=ROOT/cfg['design']['source_exp89']
    meta=read(old/'data/cic2018/manifest.json')
    raw=Path(meta['source_pickle'])
    progress(out,'source_verification',path=str(raw))
    assert sha(raw)==meta['source_sha256']
    with raw.open('rb') as f: suite=pickle.load(f)
    families=np.asarray(suite['families'])
    for ds in cfg['design']['datasets']:
        source=old/'data'/ds; dest=out/'data'/ds;dest.mkdir(parents=True,exist_ok=True)
        md=read(source/'manifest.json');names=md['class_names'];b=names.index('benign')
        full_ids=np.load(source/'test_ids.npy',mmap_mode='r')
        ty=np.load(source/'test_y.npy',mmap_mode='r')
        tx=np.load(source/'test_X.npy',mmap_mode='r')
        rng=np.random.default_rng(cfg['design']['panel']['seed'])
        panel=[]
        for c in range(len(names)):
            rows=np.flatnonzero(ty==c)
            cap=cfg['design']['panel']['benign_cap' if c==b else 'attack_class_cap']
            panel.extend(rng.choice(rows,min(cap,len(rows)),replace=False).tolist())
        panel=rng.permutation(np.asarray(panel,dtype=np.int64))
        np.save(dest/'panel_positions.npy',panel)
        np.save(dest/'panel_X.npy',np.asarray(tx[panel]))
        np.save(dest/'panel_y.npy',np.asarray(ty[panel]))
        cnt=np.bincount(ty[panel],minlength=len(names))
        weight=np.asarray(md['original_test_counts'],dtype=float)/cnt
        np.save(dest/'panel_weights.npy',weight[ty[panel]])
        pool=np.load(source/'original_train_ids.npy',mmap_mode='r')
        benign_pool=pool[families[pool]=='benign']
        for seed in cfg['design']['seeds']:
            prior=source/'original'/f'seed_{seed}'
            ids=np.load(prior/'source_ids.npy'); y=np.load(prior/'y.npy')
            existing=ids[y==b]
            # Oversample candidates slightly, then discard existing IDs, preserving RNG order.
            extra=np.random.default_rng(seed+900001).choice(benign_pool,800,replace=False)
            extra=extra[~np.isin(extra,existing)][:730]
            assert len(extra)==730
            fixed=np.concatenate([existing,extra])
            np.testing.assert_array_equal(families[fixed],np.repeat('benign',750))
            for regime in cfg['design']['benign_regimes']:
                for n in cfg['design']['budgets']:
                    chosen=[]
                    for c in range(len(names)):
                        chosen.extend((fixed if c==b and regime=='fixed750' else ids[y==c][:n]).tolist())
                    chosen=np.asarray(chosen,dtype=np.int64)
                    labels=np.array([names.index(str(x)) for x in families[chosen]],dtype=np.int16)
                    assert len(np.unique(chosen))==len(chosen)
                    assert not np.isin(chosen,full_ids).any()
                    expected=(750 if regime=='fixed750' else n)+(len(names)-1)*n
                    assert len(chosen)==expected
                    for c in range(len(names)):
                        if c!=b or regime=='matched':
                            np.testing.assert_array_equal(chosen[labels==c],ids[y==c][:n])
                    x=np.nan_to_num(np.asarray(suite['X'][chosen],dtype=np.float32))
                    job=out/'jobs'/condition_id(ds,regime,n,seed);job.mkdir(parents=True,exist_ok=True)
                    for key,value in dict(train_X=x,train_y=labels,train_ids=chosen).items():np.save(job/f'{key}.npy',value)
                    fold,group,fold_audit=folds(x,labels,n,seed)
                    np.save(job/'folds.npy',fold);np.save(job/'groups.npy',group)
                    write(job/'sampling.json',dict(dataset=ds,regime=regime,n=n,seed=seed,train_rows=len(chosen),
                        class_counts=np.bincount(labels,minlength=len(names)).tolist(),unique_feature_groups=len(np.unique(group)),
                        unique_groups_by_class=[len(np.unique(group[labels==c])) for c in range(len(names))],
                        fold_audit=fold_audit,train_ids_sha256=sha(job/'train_ids.npy'),
                        paired_labels=True,additional_validation_labels=0,train_test_row_ids_disjoint=True))
        write(dest/'manifest.json',dict(md,source_exp89=str(source),panel_rows=len(panel),panel_class_counts=cnt.tolist(),
            panel_positions_sha256=sha(dest/'panel_positions.npy'),panel_weight_per_class=weight.tolist(),
            outer_test_feature_disjoint=False))
        progress(out,'prepared_dataset',dataset=ds,panel_rows=len(panel),full_rows=len(ty))
        del pool,benign_pool,tx,ty;gc.collect()
    del suite,families;gc.collect()
    write(out/'PREPARED.json',dict(at=stamp(),conditions=len(conditions(cfg)),test_label_training_access=False))


def oof(out,cfg,ds,regime,n,seed):
    job=out/'jobs'/condition_id(ds,regime,n,seed)
    if (job/'selection.json').exists():return read(job/'selection.json')
    md=read(out/'data'/ds/'manifest.json');names=md['class_names'];C=len(names);b=names.index('benign')
    X=np.load(job/'train_X.npy');y=np.load(job/'train_y.npy');fold=np.load(job/'folds.npy');group=np.load(job/'groups.npy')
    arrays={};costs={};local=dict(cfg,seed=seed)
    for kind in COMPONENTS:
        path=job/'oof'/f'{kind}.npy';marker=path.with_suffix('.json')
        if marker.exists():
            info=read(marker);assert sha(path)==info['sha256'];arrays[kind]=np.load(path);costs[kind]=info;continue
        progress(out,'crossfit',condition=job.name,component=kind)
        output=2 if kind.endswith('_gate') else C-1 if kind.endswith('_attack') else C
        pred=np.full((len(y),output),np.nan,dtype=np.float32)
        timing=[];hashes=[]
        for k in np.unique(fold):
            tr=np.flatnonzero(fold!=k);va=np.flatnonzero(fold==k)
            assert not np.intersect1d(group[tr],group[va]).size
            model=Component(kind,local,names,b,md['feature_names']).fit(X[tr],y[tr])
            p,t=timed(lambda:model.predict(X[va]));pred[va]=p
            timing.append(dict(fold=int(k),fit_seconds=model.fit_seconds,inference_seconds=t))
            if model.before:hashes.append(model.before)
            model.close()
        assert np.isfinite(pred).all() and np.allclose(pred.sum(1),1,atol=1e-3)
        path.parent.mkdir(parents=True,exist_ok=True);np.save(path,pred)
        info=dict(component=kind,folds=timing,seconds=sum(x['fit_seconds']+x['inference_seconds'] for x in timing),
                  weights_sha256=sorted(set(hashes)),sha256=sha(path))
        write(marker,info);arrays[kind]=pred;costs[kind]=info
    state,t=timed(lambda:fit_combiner(arrays,y,b,C))
    state.update(combiner_fit_seconds=t,created_at=stamp(),train_ids_sha256=sha(job/'train_ids.npy'),
                 oof_costs=costs,test_predictions_started=False)
    write(job/'selection.json',state)
    return state


def infer_component(out,cfg,ds,regime,n,seed,kind,phase):
    job=out/'jobs'/condition_id(ds,regime,n,seed);dest=job/phase;dest.mkdir(exist_ok=True)
    path=dest/f'{kind}.npy';marker=dest/f'{kind}.json'
    if marker.exists():
        info=read(marker);assert sha(path)==info['probabilities_sha256'];return info
    md=read(out/'data'/ds/'manifest.json');names=md['class_names'];C=len(names);b=names.index('benign')
    X=np.load(job/'train_X.npy');y=np.load(job/'train_y.npy')
    local=dict(cfg,seed=seed)
    model=Component(kind,local,names,b,md['feature_names']).fit(X,y)
    source=Path(md['source_exp89'])
    tx=np.load((out/'data'/ds/'panel_X.npy') if phase=='panel' else source/'test_X.npy',mmap_mode='r')
    order=np.arange(len(tx)) if phase=='panel' else np.load(source/'query_order.npy',mmap_mode='r')
    output=len(model.output_classes)
    p=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(tx),output));p[:]=np.nan
    service=[];last=time.monotonic()
    progress(out,'predict_component',condition=job.name,component=kind,evaluation=phase,rows_total=len(tx))
    for start in range(0,len(tx),cfg['batch']):
        pos=order[start:start+cfg['batch']]
        value,t=timed(lambda:model.predict(np.asarray(tx[pos])))
        p[pos]=value;service.append(t)
        if time.monotonic()-last>=30:
            done=start+len(pos)
            progress(out,'predict_component',condition=job.name,component=kind,evaluation=phase,
                rows_done=done,rows_total=len(tx),component_eta_seconds=(len(tx)-done)*sum(service)/done)
            last=time.monotonic()
    p.flush()
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-3)
    before=model.before
    info=dict(component=kind,phase=phase,rows=len(tx),fit_seconds=model.fit_seconds,
        inference_seconds=sum(service),weights_sha256=before,probabilities_sha256=sha(path),at=stamp())
    np.save(dest/f'{kind}_service.npy',np.asarray(service));model.close()
    write(marker,info);del p,tx;gc.collect()
    return info


def score(cm,names):
    cm=np.asarray(cm,dtype=float);tp=np.diag(cm);support=cm.sum(1);den=support+cm.sum(0)
    f1=np.divide(2*tp,den,out=np.zeros(len(cm)),where=den>0)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(len(cm)),where=cm.sum(0)>0)
    recall=np.divide(tp,support,out=np.zeros(len(cm)),where=support>0)
    b=names.index('benign');attack=np.arange(len(cm))!=b
    return dict(confusion=cm.tolist(),rows=float(cm.sum()),attack_macro_f1=float(f1[attack].mean()),
        all_macro_f1=float(f1.mean()),benign_fpr=float(1-recall[b]) if support[b] else None,
        attack_detection_recall=float(1-cm[attack,b].sum()/support[attack].sum()),
        classes=[dict(name=names[c],support=float(support[c]),precision=float(precision[c]),
                      recall=float(recall[c]),f1=float(f1[c])) for c in range(len(names))])


def evaluate(out,cfg,ds,regime,n,seed,phase):
    job=out/'jobs'/condition_id(ds,regime,n,seed);dest=job/phase;marker=dest/'COMPLETE.json'
    if marker.exists():return read(dest/'metrics.json')
    state=read(job/'selection.json');selection_sha=sha(job/'selection.json')
    kinds=list(dict.fromkeys(['xgb0',f"xgb{state['chosen_xgb']}",'pfn','pfn_log','pfn_gate',
                              'pfn_attack','xgb_gate','xgb_attack','pfn16']))
    costs={kind:infer_component(out,cfg,ds,regime,n,seed,kind,phase) for kind in kinds}
    # The selector is frozen before any evaluation, including the panel.
    assert sha(job/'selection.json')==selection_sha
    md=read(out/'data'/ds/'manifest.json');names=md['class_names'];C=len(names);b=names.index('benign')
    source=Path(md['source_exp89'])
    y=np.load(out/'data'/ds/'panel_y.npy') if phase=='panel' else np.load(source/'test_y.npy',mmap_mode='r')
    weights=np.load(out/'data'/ds/'panel_weights.npy') if phase=='panel' else None
    arrays={k:np.load(dest/f'{k}.npy',mmap_mode='r') for k in kinds}
    predictions={m:np.lib.format.open_memmap(dest/f'pred_{m}.npy',mode='w+',dtype=np.int16,shape=(len(y),)) for m in METHODS}
    mats={m:np.zeros((C,C),dtype=float) for m in METHODS}
    corrections={m:dict(helpful=0.,harmful=0.,changed=0.) for m in METHODS}
    timings=[]
    oracle_correctable=0.;global_errors=0.
    for start in range(0,len(y),32768):
        stop=min(start+32768,len(y));p={k:np.asarray(v[start:stop]) for k,v in arrays.items()}
        combined,t=timed(lambda:combine(p,state,b,C));timings.append(t)
        truth=y[start:stop];w=weights[start:stop] if weights is not None else np.ones(stop-start)
        base=combined['pfn'].argmax(1)
        alternative=np.stack([combined[m].argmax(1) for m in ['xgb_tuned','pfn_log','hier_pfn']],axis=1)
        oracle_correctable+=float(w[(base!=truth)&(alternative==truth[:,None]).any(1)].sum())
        global_errors+=float(w[base!=truth].sum())
        for method,prob in combined.items():
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-3)
            pred=prob.argmax(1);predictions[method][start:stop]=pred
            mats[method]+=np.bincount(truth*C+pred,weights=w,minlength=C*C).reshape(C,C)
            corrections[method]['helpful']+=float(w[(pred==truth)&(base!=truth)].sum())
            corrections[method]['harmful']+=float(w[(pred!=truth)&(base==truth)].sum())
            corrections[method]['changed']+=float(w[pred!=base].sum())
    metrics=[]
    for method in METHODS:
        predictions[method].flush()
        # Independent sklearn implementation over saved hard predictions.
        independent=confusion_matrix(y,predictions[method],labels=np.arange(C),sample_weight=weights)
        np.testing.assert_allclose(independent,mats[method],rtol=1e-9,atol=1e-4)
        atk=mats[method].copy();atk[b]=0
        deps=dependencies(method,state)
        requires_xgb_selection=method in ['xgb_tuned','mean_xgb_pfn','mean_bank','convex_oof','gate_oof','anchored_gate']
        oof_deps=[f'xgb{i}' for i in range(4)] if requires_xgb_selection else []
        learned=method in ['convex_oof','gate_oof','anchored_gate']
        if learned:oof_deps+=['pfn','pfn_log','pfn_gate','pfn_attack']
        preparation_oof=sum(state['oof_costs'][k]['seconds'] for k in oof_deps)
        combiner_fit=state['convex_fit_seconds'] if method=='convex_oof' else state['gate_fit_seconds'] if method in ['gate_oof','anchored_gate'] else 0.
        info=dict(dataset=ds,regime=regime,n=n,seed=seed,phase=phase,method=method,
            evaluated_rows=len(y),train_rows=int(sum(read(job/'sampling.json')['class_counts'])),
            full_population_rows=md['original_test_rows'],normal_inclusive=score(mats[method],names),
            attack_only=score(atk,names),corrections=corrections[method],components=deps,
            inference_seconds=sum(costs[k]['inference_seconds'] for k in deps),
            fit_seconds=sum(costs[k]['fit_seconds'] for k in deps),
            oof_seconds=preparation_oof,combiner_fit_seconds=combiner_fit,
            preparation_seconds=preparation_oof+combiner_fit+sum(costs[k]['fit_seconds'] for k in deps),
            bank_all_methods_combination_seconds=sum(timings),
            predictions_sha256=sha(dest/f'pred_{method}.npy'))
        metrics.append(info)
    write(dest/'metrics.json',metrics)
    write(dest/'complementarity.json',dict(global_errors=global_errors,oracle_correctable_errors=oracle_correctable,
        oracle_correctable_fraction=oracle_correctable/max(global_errors,1),
        warning='Uses ground truth only for a diagnostic ceiling; never an implementable gate.'))
    write(marker,dict(at=stamp(),methods=len(metrics),rows=len(y),selection_sha256=selection_sha,
                       independent_confusion_verified=True,metrics_sha256=sha(dest/'metrics.json')))
    del arrays,predictions;gc.collect()
    return metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():cfg=read(out/'config.json')
    else:
        design=read(PROTOCOL)
        if args.smoke:
            design.update(datasets=['cic2018'],budgets=[2],seeds=[43],benign_regimes=['matched'],expected_conditions=1)
            design['full_test']['budgets']=[];design['full_test']['expected_conditions']=0
            design['panel'].update(benign_cap=150,attack_class_cap=100)
        cfg=dict(design=design,checkpoint=str(ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'),
                 threads=8,batch=8192,started_at=stamp(),smoke=args.smoke)
        # Smoke uses the same model estimator counts; only evaluation rows/conditions shrink.
        cfg.pop('smoke');cfg['is_smoke']=args.smoke
        write(out/'config.json',cfg)
    for src in [*Path(__file__).parent.glob('*.py'),PROTOCOL,ROOT/'scripts/v7/exp72/replay_models.py',
                ROOT/'scripts/v7/exp71/exp71_bounded_updates.py',ROOT/'scripts/common/lablog.py']:
        target=out/'source_dependencies'/src.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(src,target)
        elif sha(target)!=sha(src):raise RuntimeError(f'Source changed during resume: {src}')
    write(out/'environment.json',{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','tabpfn','xgboost']})
    write(out/'checkpoint.json',dict(path=cfg['checkpoint'],sha256=sha(cfg['checkpoint'])))
    import torch
    torch.set_num_threads(cfg['threads'])
    begun=time.perf_counter();completed=0
    try:
        prepare(out,cfg)
        if args.prepare_only:return
        # All selection precedes its evaluation; parameters are then immutable.
        for ds,r,n,s in conditions(cfg):
            oof(out,cfg,ds,r,n,s)
            evaluate(out,cfg,ds,r,n,s,'panel')
            completed+=1
            progress(out,'panel_condition_complete',completed=completed,total=len(conditions(cfg)),
                     elapsed_seconds=time.perf_counter()-begun)
        write(out/'PANELS_COMPLETE.json',dict(at=stamp(),conditions=completed))
        full=[x for x in conditions(cfg) if full_condition(cfg,x[1],x[2])]
        for i,(ds,r,n,s) in enumerate(full):
            evaluate(out,cfg,ds,r,n,s,'full')
            progress(out,'full_condition_complete',completed=i+1,total=len(full),elapsed_seconds=time.perf_counter()-begun)
        write(out/'MODELS_COMPLETE.json',dict(at=stamp(),panel_conditions=completed,full_conditions=len(full)))
        if not cfg['is_smoke']:
            subprocess.run([sys.executable,str(Path(__file__).with_name('report_shared_experts.py')),'--run',str(out)],check=True)
        write(out/'COMPLETE.json',dict(at=stamp(),panel_conditions=completed,full_conditions=len(full),
                                      seconds=time.perf_counter()-begun,report_written=not cfg['is_smoke']))
        progress(out,'complete',panel_conditions=completed,full_conditions=len(full))
    except BaseException as exc:
        write(out/'FAILURE.json',dict(at=stamp(),error=str(exc),traceback=traceback.format_exc()))
        report=Path(__file__).with_name('report_shared_experts.py')
        if report.exists() and not cfg['is_smoke']:
            subprocess.run([sys.executable,str(report),'--run',str(out),'--partial'],check=False)
        raise


if __name__=='__main__':main()

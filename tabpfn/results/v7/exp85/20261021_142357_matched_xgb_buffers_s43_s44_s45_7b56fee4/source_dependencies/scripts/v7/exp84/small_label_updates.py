#!/usr/bin/env python3
"""Fresh cumulative/1000-row updates, full test, and paired old-class panels."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)

import argparse
import gc
import importlib.metadata
import json
import resource
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from exp61_utils import write, sha
from finetune_final import CHECKPOINT, weights_digest
from noft_dist_budget import load_source, subset_plans, SOURCE
from run_context_policies import replace_context
from context_budget_sweep import simulate, evaluate, MIXED
from replay_models import make_base, timed, sync
from replay_protocol import confusion_metrics

BUDGETS = [5, 10, 20, 30, 50, 100]
METHODS = ['xgb', 'pfn_accumulate', 'pfn_fifo', 'pfn_random', 'pfn_similar']


def read(p):
    return json.loads(Path(p).read_text())


def progress(out, phase, **kw):
    d=dict(phase=phase, updated_epoch=time.time(), **kw)
    write(out/'progress.json', d)
    print(json.dumps(d), flush=True)


def prepare(out, cfg):
    checks=[]
    for ds in cfg['datasets']:
        meta, arr, original = load_source(ds)
        dest=out/'data'/ds
        dest.mkdir(parents=True, exist_ok=True)
        original_order=MIXED/'data'/ds/'query_order.npy'
        order=np.load(original_order,mmap_mode='r')
        assert len(order)==len(arr['test_y'])
        np.testing.assert_array_equal(order,np.random.default_rng(43).permutation(len(order)))
        shutil.copy2(original_order,dest/'query_order.npy')
        rng=np.random.default_rng(43)
        panel=np.concatenate([rng.choice(np.flatnonzero(arr['test_y']==c),min(2000,int((arr['test_y']==c).sum())),replace=False)
                              for c in range(len(meta['class_names']))])
        panel=np.random.default_rng(43).permutation(panel).astype('int64')
        np.save(dest/'panel_ids.npy',panel)
        write(dest/'manifest.json',dict(source=meta['source'],class_names=meta['class_names'],
             test_rows=len(order),query_order_sha256=sha(original_order),panel_rows=len(panel),
             panel_ids_sha256=sha(dest/'panel_ids.npy'),panel_role='Fixed offline diagnostic, future query features may occur; labels never passed to model',
             source_manifest=str(SOURCE/'data'/ds/'manifest.json')))
        previous=None
        for n in BUDGETS:
            full=subset_plans(original,arr['train_y'],n,43)
            plans=[{k:p[k] for k in ['stage','incoming','released','new_classes']} for p in full]
            assert plans[0]['incoming']==original[0]['incoming']
            for s, p in enumerate(plans):
                assert len(p['released'])==len(set(p['released']))
                if s:assert all(np.unique(arr['train_y'][p['incoming']],return_counts=True)[1]==n)
                if previous:assert set(previous[s]['incoming'])<=set(p['incoming'])
            if n in [10,20,50]:
                saved=read(ROOT/f'tabpfn/results/v7/exp78/20261016_005435_noft_dist_new100_50_20_10_s43_52588487/new_{n}/data/{ds}/plans.json')
                assert all(a['incoming']==b['incoming'] for a,b in zip(plans,saved))
            if n==100:assert all(a['incoming']==b['incoming'] for a,b in zip(plans,original))
            ids=plans[-1]['released']
            assert len(ids)==1000+n*(4 if ds=='cic2018' else 7)
            assert not np.intersect1d(arr['train_ids'][ids],arr['test_ids']).size
            case=out/f'new_{n}'/'data'/ds
            write(case/'plans.json',plans)
            for policy in ['fifo','random','similar']:
                write(case/f'{policy}_preflight.json',simulate(plans,arr,policy,43))
            checks.append(dict(dataset=ds,n=n,stages=len(plans),all_new_admitted=True,capacity=1000))
            previous=plans
    write(out/'preflight.json',dict(passed=True,checks=checks,conditions=60,
          nested_ids=True,prior_overlapping_ids_identical=True,train_test_row_ids_disjoint=True,
          note='Same-feature duplicates can remain; final feature-disjoint diagnostic retained'))


def predict_panel(model, arr, ids, classes, cfg):
    pred=np.empty(len(ids),dtype='int16');seconds=0.
    for start in range(0,len(ids),cfg['batch']):
        batch=ids[start:start+cfg['batch']]
        p,t=timed(lambda:model.predict_proba(np.asarray(arr['test_X'][batch])))
        assert p.shape==(len(batch),len(classes)) and np.isfinite(p).all()
        assert np.allclose(p.sum(1),1,atol=1e-3)
        pred[start:start+len(batch)]=classes[p.argmax(1)];seconds+=t
    return pred,seconds


def stage_metrics(y,pred,previous,classes,old_classes,initial_classes,names):
    known_mask=np.isin(y,classes);initial_mask=np.isin(y,initial_classes)
    d=dict(known=confusion_metrics(y[known_mask],pred[known_mask],names,classes),
           initial_only=confusion_metrics(y[initial_mask],pred[initial_mask],names,initial_classes))
    if previous is not None:
        mask=np.isin(y,old_classes)
        before=confusion_metrics(y[mask],previous[mask],names,old_classes)
        after=confusion_metrics(y[mask],pred[mask],names,old_classes)
        d.update(old_before=before,old_after=after,
                 old_macro_f1_change=after['macro_f1']-before['macro_f1'],old_panel_rows=int(mask.sum()))
    return d


def run_job(out,cfg,ds,n,method,meta,arr,plans):
    import torch
    assert method in METHODS or method in ['xgb_fifo','xgb_random','xgb_similar']
    is_xgb=method=='xgb' or method.startswith('xgb_')
    job=out/f'new_{n}'/'jobs'/f'{ds}_{method}';job.mkdir(parents=True,exist_ok=True)
    if (job/'COMPLETE.json').exists():return read(job/'results.json')
    names=meta['class_names'];panel=np.load(out/'data'/ds/'panel_ids.npy')
    panel_y=np.asarray(arr['test_y'][panel]);order=np.load(out/'data'/ds/'query_order.npy',mmap_mode='r')
    initial_classes=np.unique(arr['train_y'][plans[0]['incoming']])
    expected=read(SOURCE/'jobs'/f'{ds}_pfn_pretrained'/'results.json')['fits'][0]['weights_sha256']
    history=None if method in ['xgb','pfn_accumulate'] else read(out/f'new_{n}'/'data'/ds/f'{method[4:]}_preflight.json')
    model=None;old=None;previous=None;old_classes=None;fits=[];panels=[]
    torch.cuda.reset_peak_memory_stats();job_start=time.time()
    for s,plan in enumerate(plans):
        progress(out,'update',dataset=ds,new_per_class=n,method=method,stage=s+1)
        sync();start=time.perf_counter()
        incoming=np.asarray(plan['incoming'],dtype='int64')
        if s==0:ctx=incoming;removed=np.array([],dtype='int64')
        elif method in ['xgb','pfn_accumulate']:
            ctx=np.asarray(plan['released'],dtype='int64');removed=np.array([],dtype='int64')
        else:ctx,removed=replace_context(old,incoming,arr['train_y'],arr['train_X'],method[4:],cfg['seed']+s*1009)
        classes=np.unique(arr['train_y'][ctx]);labels=np.searchsorted(classes,arr['train_y'][ctx]);X=np.asarray(arr['train_X'][ctx])
        sync();selection=time.perf_counter()-start
        assert set(incoming)<=set(ctx)
        if history:
            assert ctx.tolist()==history[s]['context_indices'] and removed.tolist()==history[s]['removed_indices']
        if is_xgb or model is None:
            model,construct=timed(lambda:make_base('xgb' if is_xgb else 'tabpfn',cfg,len(classes)))
        else:construct=0.
        _,fit_seconds=timed(lambda:model.fit(X,labels))
        np.testing.assert_array_equal(model.classes_,np.arange(len(classes)))
        digest=None if is_xgb else weights_digest(model.models_[0].state_dict())
        if not is_xgb:assert digest==expected
        info=dict(stage=s,stage_number=s+1,context_indices=ctx.tolist(),removed_indices=removed.tolist(),
                  context_rows=len(ctx),released_rows=len(plan['released']),incoming_rows=len(incoming),
                  admitted_rows=len(incoming),rejected_rows=0,context_class_counts=np.bincount(arr['train_y'][ctx],minlength=len(names)).tolist(),
                  selection_seconds=selection,construct_seconds=construct,fit_seconds=fit_seconds,
                  update_seconds=selection+construct+fit_seconds,finetuning_seconds=0.,weights_sha256=digest,
                  classes=classes.tolist())
        stage_dir=job/f'stage_{s}';write(stage_dir/'fit_metrics.json',info);fits.append(info)
        start=time.perf_counter()
        if is_xgb:model.save_model(stage_dir/'model.ubj')
        else:model.save_fit_state(stage_dir/'model.tabpfn_fit')
        info['checkpoint_write_seconds']=time.perf_counter()-start
        pred,panel_seconds=predict_panel(model,arr,panel,classes,cfg)
        np.save(stage_dir/'panel_predictions.npy',pred)
        pm=stage_metrics(panel_y,pred,previous,classes,old_classes,initial_classes,names)
        pm.update(stage=s,stage_number=s+1,inference_seconds=panel_seconds,query_ids_sha256=sha(out/'data'/ds/'panel_ids.npy'))
        write(stage_dir/'panel_metrics.json',pm);panels.append(pm)
        write(stage_dir/'fit_metrics.json',info)
        previous=pred;old_classes=classes;old=ctx
    pred,inference=evaluate(out,job,cfg,ds,n,method,model,arr,names,classes,order)
    if not is_xgb:assert weights_digest(model.models_[0].state_dict())==expected
    metrics=confusion_metrics(arr['test_y'],np.asarray(pred),names)
    benign=names.index('benign');initial_attacks=[c for c in initial_classes if c!=benign]
    new_classes=[c for c in classes if c not in initial_classes]
    mask=~np.isin(arr['test_hash'],arr['train_hash'][plans[-1]['released']])
    result=dict(dataset=ds,method=method,new_per_class=n,seed=cfg['seed'],evaluation='final_full_test',
         test_full=True,test_rows=len(order),final_stage=len(plans)-1,batch=cfg['batch'],test_order='seed43_mixed',
         predictions_path=str(job/'predictions.npy'),query_order_sha256=sha(out/'data'/ds/'query_order.npy'),
         metrics=metrics,initial_attack_mean_f1=float(np.mean([metrics['classes'][c]['f1'] for c in initial_attacks])),
         new_attack_mean_f1=float(np.mean([metrics['classes'][c]['f1'] for c in new_classes])),
         initial_panel_final_change=panels[-1]['initial_only']['macro_f1']-panels[0]['initial_only']['macro_f1'],
         worst_update_old_change=min(p['old_macro_f1_change'] for p in panels[1:]),
         feature_disjoint_diagnostic=confusion_metrics(arr['test_y'][mask],np.asarray(pred)[mask],names),
         fits=fits,stage_panel_metrics=panels,initial_finetuning=False,initial_finetuning_seconds=0.,
         initial_seconds=fits[0]['update_seconds'],update_seconds=sum(f['update_seconds'] for f in fits[1:]),
         serving_seconds=inference,stage_panel_inference_seconds=sum(p['inference_seconds'] for p in panels),
         cost_reused=False,predictions_reused=False,label_budget=len(plans[-1]['released']),context_rows=len(ctx),
         final_context_counts=fits[-1]['context_class_counts'],checkpoint=str(CHECKPOINT),
         peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
         process_peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
         elapsed_seconds=time.time()-job_start,
         cost_scope='Sum of post-initial selection/construction/fit; excludes all inference, checkpoint writes, diagnostics',
         note='One seed, no IDS FT; original deletion rules; all incoming rows admitted; class coverage minimum one')
    write(job/'results.json',result)
    write(job/'COMPLETE.json',dict(complete=True,predictions_sha256=sha(job/'predictions.npy')))
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=dict(protocol='exp84.1',seed=43,threads=8,batch=8192,datasets=['cic2018','toniot'],new_budgets=BUDGETS,
             buffer_capacity=1000,checkpoint=str(CHECKPOINT),methods=METHODS,seed_repetitions=1,
             initial_finetuning=False,slides_publication=False,report_date=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d'))
    write(out/'config.json',cfg)
    deps=[Path(__file__),Path(__file__).with_name('build_small_label_record.py'),ROOT/'configs/experiments/exp84_protocol.json',
          ROOT/'scripts/v7/exp76/run_context_policies.py',ROOT/'scripts/v7/exp77/finetune_final.py',ROOT/'scripts/v7/exp78/noft_dist_budget.py',
          ROOT/'scripts/v7/exp83/context_budget_sweep.py',ROOT/'scripts/v7/exp72/replay_models.py',ROOT/'scripts/v7/exp72/replay_protocol.py',
          ROOT/'scripts/v7/exp71/exp71_bounded_updates.py',ROOT/'scripts/common/exp61_utils.py',ROOT/'scripts/common/experiment_paths.py']
    hashes={}
    for source in deps:
        dest=out/'source_dependencies'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);hashes[str(source)]=sha(source)
    hashes[str(CHECKPOINT)]=sha(CHECKPOINT);write(out/'source_hashes.json',hashes)
    write(out/'environment.json',{k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn','tabpfn','xgboost']})
    import torch
    torch.set_num_threads(8);torch.manual_seed(43)
    jobs=[];started=time.time();error=None
    try:
        prepare(out,cfg)
        if args.prepare_only:return
        torch.cuda.init();sync();t=time.perf_counter();a=torch.ones((512,512),device='cuda');_ = a@a;sync();del a,_
        write(out/'warmup.json',dict(seconds=time.perf_counter()-t,method='Single 512x512 CUDA matmul before jobs; not included in update cost',gpu=torch.cuda.get_device_name()))
        for n in BUDGETS:
            for ds in cfg['datasets']:
                meta,arr,_=load_source(ds);plans=read(out/f'new_{n}'/'data'/ds/'plans.json')
                for method in METHODS:
                    result=run_job(out,cfg,ds,n,method,meta,arr,plans);jobs.append(result)
                    write(out/'results.json',dict(state='running',jobs=jobs,expected_jobs=60))
                    progress(out,'job_complete',dataset=ds,new_per_class=n,method=method,completed_jobs=len(jobs),expected_jobs=60,
                             elapsed_seconds=time.time()-started,macro_f1=result['metrics']['macro_f1'],update_seconds=result['update_seconds'])
        assert all(sha(p)==h for p,h in hashes.items())
        write(out/'results.json',dict(state='complete',jobs=jobs,expected_jobs=60))
        progress(out,'complete',completed_jobs=60,expected_jobs=60,elapsed_seconds=time.time()-started)
    except BaseException as exc:
        error=exc;write(out/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc(),completed_jobs=len(jobs)))
        write(out/'results.json',dict(state='failed',jobs=jobs,expected_jobs=60,error=str(exc)))
        raise
    finally:
        if not args.prepare_only:
            rc=subprocess.run([sys.executable,str(Path(__file__).with_name('build_small_label_record.py')),'--run',str(out)],cwd=ROOT).returncode
            if rc and error is None:raise RuntimeError('Experiment finished but automatic report/validation failed; inspect stdout.log')


if __name__=='__main__':main()

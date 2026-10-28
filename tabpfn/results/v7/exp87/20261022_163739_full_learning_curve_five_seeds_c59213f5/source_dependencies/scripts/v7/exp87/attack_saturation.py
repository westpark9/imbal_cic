"""EXP87: attack-only learning curves, single-seed search then multiseed checks."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').exists())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)

import argparse,gc,hashlib,importlib.metadata,json,pickle,shutil,subprocess,time,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from exp61_utils import write,sha
from replay_models import make_base,timed,sync
from replay_protocol import confusion_metrics
from finetune_final import weights_digest

PROTOCOL=ROOT/'configs/experiments/exp87_protocol.json'
SEEDS=[43,44,45,46,47]
METHODS=['xgb','pfn']

def read(p):return json.loads(Path(p).read_text())

def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

def progress(out,phase,**kwargs):
    r=dict(phase=phase,updated_at=stamp(),**kwargs)
    write(out/'progress.json',r);print(json.dumps(r,ensure_ascii=False),flush=True)

def prepare(out,protocol):
    if (out/'prepared.json').exists():return
    source=ROOT/protocol['source_run']
    original_meta=read(source/'data/cic2018/manifest.json')
    raw=Path(original_meta['source_pickle'])
    progress(out,'loading_source',source=str(raw))
    assert sha(raw)==original_meta['source_pickle_sha256']
    with raw.open('rb') as f:suite=pickle.load(f)
    families=np.asarray(suite['families'])
    for ds in protocol['datasets']:
        previous=read(source/'data'/ds/'manifest.json');names=previous['class_names'];b=names.index('benign')
        clean=Path(previous['clean_manifest']).parent;train=np.load(clean/'train_idx.npy')
        clean_source=np.load(clean/'source_idx.npy',mmap_mode='r');hashes=np.load(clean/'vector_hash.npy',mmap_mode='r')
        base=Path(previous['test_base']);y=np.load(base/'test_y.npy',mmap_mode='r')
        rows=np.flatnonzero(y!=b);assert len(rows)==sum(previous['test_counts'])-previous['test_counts'][b]
        d=out/'data'/ds;d.mkdir(parents=True,exist_ok=True)
        for key in ['X','y','hash','ids']:
            arr=np.load(base/f'test_{key}.npy',mmap_mode='r')
            np.save(d/f'test_{key}.npy',np.asarray(arr[rows]));del arr
        old_order=np.load(source/'data'/ds/'query_order.npy')
        old_attack_order=old_order[np.asarray(y[old_order])!=b]
        order=np.searchsorted(rows,old_attack_order)
        assert np.array_equal(rows[order],old_attack_order) and len(np.unique(order))==len(rows)
        np.save(d/'query_order.npy',order);np.save(d/'original_test_positions.npy',rows)
        pool={c:train[families[train]==name] for c,name in enumerate(names) if c!=b}
        cap=protocol['budgets'][ds][-1];assert cap==min(map(len,pool.values()))
        for seed in SEEDS:
            oldd=source/'data'/ds/f'seed_{seed}'
            old_ids=np.load(oldd/'source_ids.npy');old_y=np.load(oldd/'y.npy')
            chosen=[];labels=[]
            for c,ids in pool.items():
                first=old_ids[old_y==c];assert len(first)==20 and np.isin(first,ids).all()
                remaining=ids[~np.isin(ids,first)]
                rng=np.random.default_rng(8700001+seed*1000003+c*1009)
                extra=rng.choice(remaining,cap-20,replace=False)
                chosen.append(np.concatenate([first,extra]));labels.append(np.full(cap,c,dtype='int16'))
            ids=np.concatenate(chosen);labels=np.concatenate(labels);idx=np.searchsorted(clean_source,ids)
            assert np.array_equal(clean_source[idx],ids) and len(np.unique(ids))==len(ids)
            np.testing.assert_array_equal(families[ids],np.array(names)[labels])
            assert not np.isin(ids,np.load(d/'test_ids.npy')).any()
            values=np.nan_to_num(np.asarray(suite['X'][ids],dtype='float32'));assert np.isfinite(values).all()
            dest=d/f'seed_{seed}';dest.mkdir(exist_ok=True)
            for k,v in dict(X=values,y=labels,source_ids=ids,hashes=hashes[idx]).items():np.save(dest/f'{k}.npy',v)
            for n in protocol['budgets'][ds]:
                sub=np.concatenate([np.arange(i*cap,i*cap+n) for i in range(len(pool))])
                np.save(dest/f'n_{n}_indices.npy',sub)
                if n<=20:
                    oldsub=np.load(oldd/f'n_{n}_indices.npy')
                    np.testing.assert_array_equal(ids[sub],old_ids[oldsub])
                    np.testing.assert_array_equal(values[sub],np.load(oldd/'X.npy')[oldsub])
        write(d/'manifest.json',dict(dataset=ds,class_names=names,benign=b,attack_classes=list(pool),
            features=int(suite['X'].shape[1]),candidate_counts={names[c]:len(v) for c,v in pool.items()},
            cap=cap,test_rows=len(rows),test_counts=np.bincount(y[rows],minlength=len(names)).tolist(),
            source_exp86=str(source),source_test_base=str(base),clean_manifest=previous['clean_manifest'],
            raw_source_sha256=previous['source_pickle_sha256'],
            query_order_sha256=sha(d/'query_order.npy'),original_positions_sha256=sha(d/'original_test_positions.npy'),
            nested=True,matched_between_models=True,exp86_first20_preserved=True,benign_inference_rows=0))
        progress(out,'dataset_prepared',dataset=ds,cap=cap,attack_test_rows=len(rows))
    del suite,families;gc.collect()
    write(out/'prepared.json',dict(completed_at=stamp(),passed=True,benign_train_rows=0,benign_inference_rows=0))

def run_job(out,cfg,ds,seed,n,method):
    import torch
    job=out/'jobs'/f'{ds}_s{seed}_n{n}_{method}';job.mkdir(parents=True,exist_ok=True)
    if (job/'COMPLETE.json').exists():
        assert sha(job/'predictions.npy')==read(job/'COMPLETE.json')['predictions_sha256']
        return read(job/'results.json')
    d=out/'data'/ds;meta=read(d/'manifest.json');names=meta['class_names'];dest=d/f'seed_{seed}'
    idx=np.load(dest/f'n_{n}_indices.npy');X=np.load(dest/'X.npy',mmap_mode='r')[idx];y=np.load(dest/'y.npy')[idx]
    ids=np.load(dest/'source_ids.npy')[idx];classes=np.asarray(meta['attack_classes']);local=np.searchsorted(classes,y)
    assert np.all(np.bincount(local,minlength=len(classes))==n)
    np.save(job/'train_source_ids.npy',ids)
    tx=np.load(d/'test_X.npy',mmap_mode='r');ty=np.load(d/'test_y.npy',mmap_mode='r');order=np.load(d/'query_order.npy')
    assert not np.any(ty==meta['benign'])
    settings=dict(cfg,seed=seed);torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats()
    progress(out,'fit',dataset=ds,seed=seed,n=n,method=method,context_rows=len(y))
    model,construct=timed(lambda:make_base('xgb' if method=='xgb' else 'tabpfn',settings,len(classes)))
    _,fit=timed(lambda:model.fit(X,local));np.testing.assert_array_equal(model.classes_,np.arange(len(classes)))
    before=weights_digest(model.models_[0].state_dict()) if method=='pfn' else None
    if before:
        expected=read(Path(meta['source_exp86'])/'jobs'/f'{ds}_s{seed}_n20_pfn'/'results.json')['weights_sha256']
        assert before==expected
    splits=None
    if method=='xgb':
        splits=int((model.get_booster().trees_to_dataframe().Feature!='Leaf').sum());assert splits>0
        model.save_model(job/'model.ubj')
    pred=np.lib.format.open_memmap(job/'predictions.npy',mode='w+',dtype='int16',shape=(len(ty),));pred[:]=-1
    services=[]
    for start in range(0,len(order),cfg['batch']):
        batch=order[start:start+cfg['batch']]
        proba,seconds=timed(lambda:model.predict_proba(np.asarray(tx[batch])))
        assert proba.shape==(len(batch),len(classes)) and np.isfinite(proba).all()
        assert np.allclose(proba.sum(1),1,atol=1e-3)
        pred[batch]=classes[proba.argmax(1)];services.append([start,len(batch),seconds])
        if start==0 or start//cfg['batch']%25==0:
            done=start+len(batch)
            progress(out,'predict',dataset=ds,seed=seed,n=n,method=method,rows_done=done,rows_total=len(order),
                estimated_job_remaining_seconds=(len(order)-done)*sum(x[2] for x in services)/done)
    pred.flush();assert np.isin(pred,classes).all()
    if before:assert before==weights_digest(model.models_[0].state_dict())
    metrics=confusion_metrics(ty,pred,names,classes)
    overlap=np.isin(np.load(d/'test_hash.npy',mmap_mode='r'),np.load(dest/'hashes.npy')[idx])
    anchor=None
    if n==20:
        old=Path(meta['source_exp86'])/'jobs'/f'{ds}_s{seed}_n20_{method}'
        prior_pred=np.load(old/'predictions.npy',mmap_mode='r')[np.load(d/'original_test_positions.npy')]
        previous=read(old/'results.json')
        anchor=dict(reference=str(old),prediction_disagreement_count=int((prior_pred!=pred).sum()),
            prediction_disagreement_rate=float((prior_pred!=pred).mean()),
            macro_f1_delta=metrics['macro_f1']-previous['phase_a']['macro_f1'],
            meaning='Same train/test rows; new batches contain attacks only. All EXP87 points rerun with this protocol.')
    result=dict(dataset=ds,seed=seed,n=n,method=method,context_rows=len(y),class_names=names,
        train_source_ids=ids.tolist(),train_ids_sha256=sha(job/'train_source_ids.npy'),train_class_counts=np.bincount(y,minlength=len(names)).tolist(),
        test_rows=len(ty),metrics=metrics,preparation_seconds=construct+fit,construct_seconds=construct,fit_seconds=fit,
        inference_seconds=sum(x[2] for x in services),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        weights_sha256=before,finetuning=False,xgb_tree_splits=splits,exact_feature_overlap_attack_rows=int(overlap.sum()),
        training_unique_feature_rows=len(np.unique(X,axis=0)),anchor_comparison=anchor,
        predictions_path=str(job/'predictions.npy'),query_order_sha256=meta['query_order_sha256'],
        completed_at=stamp(),cost_scope='Fresh construct+fit/cache and attack-only test inference; no sequential updates')
    np.save(job/'service_times.npy',np.asarray(services));write(job/'results.json',result)
    write(job/'COMPLETE.json',dict(predictions_sha256=sha(job/'predictions.npy'),completed_at=stamp()))
    del model,pred,tx,ty;gc.collect();torch.cuda.empty_cache()
    return result

def window_status(jobs,ds,budgets,seeds):
    if len(budgets)<3:return dict(complete=False,flat=False,budgets=budgets)
    selected=budgets[-3:];lookup={(j['dataset'],j['seed'],j['n'],j['method']):j for j in jobs};spans={};seed_spans={};means={}
    for method in METHODS:
        if any((ds,s,n,method) not in lookup for s in seeds for n in selected):
            return dict(complete=False,flat=False,budgets=selected)
        a=np.array([[lookup[ds,s,n,method]['metrics']['macro_f1'] for n in selected] for s in seeds])
        means[method]=a.mean(axis=0).tolist();spans[method]=float(np.ptp(a.mean(axis=0)))
        seed_spans[method]={str(s):float(np.ptp(a[i])) for i,s in enumerate(seeds)}
    return dict(complete=True,flat=all(x<=.01 for x in spans.values()),budgets=selected,seeds=seeds,
        mean_f1=means,mean_ranges=spans,per_seed_ranges=seed_spans,
        pfn_minus_xgb=(np.array(means['pfn'])-means['xgb']).tolist())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);protocol=read(PROTOCOL)
    previous=ROOT/protocol['source_run'];oldcfg=read(previous/'config.json')
    cfg=dict(protocol='exp87.1',checkpoint=oldcfg['checkpoint'],threads=8,batch=8192,
        report_date=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d'),started_at=stamp(),design=protocol)
    if (out/'config.json').exists():cfg=read(out/'config.json')
    write(out/'config.json',cfg)
    for file in [Path(__file__),Path(__file__).with_name('report_saturation.py'),PROTOCOL,
                 ROOT/'scripts/v7/exp71/exp71_bounded_updates.py',ROOT/'scripts/v7/exp72/replay_models.py',ROOT/'scripts/v7/exp72/replay_protocol.py']:
        dest=out/'source_dependencies'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    write(out/'environment.json',{name:importlib.metadata.version(name) for name in ['torch','tabpfn','xgboost','numpy','scikit-learn']})
    import torch
    torch.set_num_threads(8)
    jobs=[];decisions={};state='running';events=[]
    def save(stage):write(out/'results.json',dict(state=state,stage=stage,jobs=jobs,decisions=decisions,events=events,updated_at=stamp()))
    def report():
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('report_saturation.py')),'--run',str(out)],capture_output=True,text=True)
        if result.returncode:
            write(out/'REPORT_FAILURE.json',dict(at=stamp(),stderr=result.stderr[-5000:],stdout=result.stdout[-5000:]))
            print(result.stdout[-3000:]+result.stderr[-3000:],flush=True)
            return False
        (out/'REPORT_FAILURE.json').unlink(missing_ok=True);print(result.stdout,flush=True);return True
    def pair(ds,seed,n,stage):
        for method in METHODS:
            job=run_job(out,cfg,ds,seed,n,method)
            if not any(j['dataset']==ds and j['seed']==seed and j['n']==n and j['method']==method for j in jobs):jobs.append(job)
            save(stage)
        progress(out,'pair_complete',dataset=ds,seed=seed,n=n,stage=stage,completed_jobs=len(jobs))
    try:
        save('preparing');report();prepare(out,protocol)
        torch.cuda.init();sync();z=torch.ones((512,512),device='cuda');z=z@z;sync();del z
        write(out/'checkpoint.json',dict(path=cfg['checkpoint'],sha256=sha(Path(cfg['checkpoint'])),gpu=torch.cuda.get_device_name()))
        # Explore both datasets first, before spending extra seeds on the end region.
        active=set(protocol['datasets']);history={d:[] for d in active};previous_flat={d:False for d in active}
        for level in range(max(map(len,protocol['budgets'].values()))):
            for ds in protocol['datasets']:
                if ds not in active:continue
                grid=protocol['budgets'][ds]
                if level>=len(grid):continue
                n=grid[level];pair(ds,43,n,'pilot');history[ds].append(n)
                status=window_status(jobs,ds,history[ds],[43]);eligible=n>=80 and status['flat']
                confirmed=eligible and previous_flat[ds];atcap=n==grid[-1]
                decisions[ds]=dict(pilot_budgets=history[ds].copy(),pilot_window=status,pilot_plateau_confirmed=confirmed,
                    cap_reached=atcap,validation_state='pending')
                events.append(dict(at=stamp(),phase='pilot_decision',dataset=ds,n=n,eligible=eligible,confirmed=confirmed,at_cap=atcap,window=status))
                previous_flat[ds]=eligible
                if confirmed or atcap:active.remove(ds)
                save('pilot');report()
            if not active:break
        save('pilot_complete');report()
        for ds in protocol['datasets']:
            grid=protocol['budgets'][ds];selected=history[ds][-3:]
            decisions[ds]['validation_budgets']=selected.copy();decisions[ds]['validation_state']='running';save('validation')
            for seed in SEEDS[1:]:
                for n in selected:pair(ds,seed,n,'validation')
                report()
            validation=window_status(jobs,ds,selected,SEEDS)
            # If the selected end region does not reproduce, extend all seeds together.
            while not validation['flat'] and selected[-1]!=grid[-1]:
                n=grid[grid.index(selected[-1])+1];selected.append(n)
                for seed in SEEDS:pair(ds,seed,n,'validation_extension')
                validation=window_status(jobs,ds,selected,SEEDS)
                decisions[ds]['validation_budgets']=selected.copy();decisions[ds]['validation_window']=validation
                events.append(dict(at=stamp(),phase='validation_extension',dataset=ds,n=n,window=validation));save('validation_extension');report()
            decisions[ds].update(validation_state='complete',validation_budgets=selected.copy(),validation_window=validation,
                final_plateau_confirmed=validation['flat'],cap_reached=selected[-1]==grid[-1],
                conclusion='five_seed_mean_plateau' if validation['flat'] else 'label_cap_reached_without_plateau')
            events.append(dict(at=stamp(),phase='dataset_complete',dataset=ds,window=validation));save('validation');report()
        state='complete';save('complete');progress(out,'complete',completed_jobs=len(jobs),decisions=decisions)
    except BaseException as exc:
        state='failed';write(out/'FAILURE.json',dict(at=stamp(),error=str(exc),traceback=traceback.format_exc(),completed_jobs=len(jobs)))
        save('failed');raise
    finally:
        okay=report()
        if not okay and state=='complete':raise RuntimeError('Models complete, final report failed; see REPORT_FAILURE.json')

if __name__=='__main__':main()

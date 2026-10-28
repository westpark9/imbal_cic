"""EXP88: matched tiny-context sampling and independent reference coverage."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').exists())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)

import argparse, gc, importlib.metadata, json, pickle, shutil, time, traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
from exp61_utils import write, sha
from replay_models import make_base, timed
from replay_protocol import confusion_metrics
from finetune_final import weights_digest

PROTOCOL = ROOT/'configs/experiments/exp88_protocol.json'

def read(path):
    return json.loads(Path(path).read_text())

def stamp():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

def progress(out, phase, **values):
    record = dict(at=stamp(), phase=phase, **values)
    write(out/'progress.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)

def unique_vectors(x):
    x = np.ascontiguousarray(x, dtype='float32').copy()
    x[x == 0] = 0
    return np.unique(x.view(np.dtype((np.void, x.shape[1]*4))).ravel(),
                     return_index=True, return_inverse=True, return_counts=True)

def numeric_transform(x):
    return np.sign(x)*np.log1p(np.abs(x))

def geometry(x, num, cat, center, scale, vocab):
    numeric = np.clip((numeric_transform(x[:,num])-center)/scale, -6, 6)
    blocks = [numeric.astype('float32')]
    for j, levels in zip(cat, vocab):
        ix = np.searchsorted(levels, x[:,j])
        ok = (ix < len(levels)) & (levels[np.minimum(ix,len(levels)-1)] == x[:,j])
        ix = np.where(ok, ix, len(levels))
        block = np.zeros((len(x),len(levels)+1), dtype='float32')
        block[np.arange(len(x)),ix] = 1/np.sqrt(2)
        blocks.append(block)
    return np.concatenate(blocks, axis=1)/np.sqrt(x.shape[1])

def representatives(z, k, seed):
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(z), min(100000,len(z)), replace=False)
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                           batch_size=4096, max_iter=100, max_no_improvement=20)
    model.fit(z[fit_idx])
    closest = np.full(k, np.inf)
    reps = np.full(k, -1, dtype='int64')
    labels = np.empty(len(z),dtype='int16')
    for start in range(0,len(z),20000):
        dist = cdist(z[start:start+20000], model.cluster_centers_, 'sqeuclidean')
        group = dist.argmin(axis=1)
        labels[start:start+len(group)] = group
        for c in np.unique(group):
            members = np.flatnonzero(group == c)
            row = members[dist[members,c].argmin()]
            if dist[row,c] < closest[c]:
                closest[c], reps[c] = dist[row,c], start+row
    occupied = np.flatnonzero(reps>=0)
    # Repeated centers can occur when many flows are identical. Record actual
    # occupancy; never invent synthetic training rows or silently refill groups.
    if len(occupied)<20:
        raise RuntimeError(f'Only {len(occupied)} occupied clusters at requested k={k}')
    return reps[occupied], labels, model.cluster_centers_, fit_idx, occupied

def farthest_order(z, seed):
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(z)))]
    distance = cdist(z,z[[selected[0]]]).ravel()
    while len(selected)<20:
        distance[selected] = -1
        nxt = int(distance.argmax())
        selected.append(nxt)
        distance = np.minimum(distance, cdist(z,z[[nxt]]).ravel())
    return np.asarray(selected)

def prepare(out, cfg):
    if (out/'prepared.json').exists():
        return
    started = time.perf_counter()
    design = cfg['design']; source = ROOT/design['source_run']
    original = read(Path(read(source/'data/cic2018/manifest.json')['source_exp86'])/'data/cic2018/manifest.json')
    raw = Path(original['source_pickle'])
    progress(out,'verify_source',source=str(raw))
    assert sha(raw)==original['source_pickle_sha256']
    with raw.open('rb') as stream:
        suite=pickle.load(stream)
    families = np.asarray(suite['families'])
    coverage=[]; inventories=[]
    for ds_index, ds in enumerate(design['datasets']):
        meta=read(source/'data'/ds/'manifest.json')
        clean=Path(meta['clean_manifest']).parent
        names=meta['class_names']; classes=meta['attack_classes']
        train=np.load(clean/'train_idx.npy'); clean_source=np.load(clean/'source_idx.npy',mmap_mode='r')
        all_hashes=np.load(clean/'vector_hash.npy',mmap_mode='r')
        d=out/'data'/ds; d.mkdir(parents=True,exist_ok=True)
        for key in ['X','y','ids','hash']:
            link=d/f'test_{key}.npy'
            if not link.exists(): link.symlink_to(source/'data'/ds/link.name)
        link=d/'query_order.npy'
        if not link.exists(): link.symlink_to(source/'data'/ds/link.name)
        test_ids=np.load(d/'test_ids.npy'); assert not np.isin(train,test_ids).any()
        feature_names=read(clean/'manifest.json')['feature_names']
        assert len(feature_names)==46
        cat=[feature_names.index(f) for f in design['geometry']['categorical']]
        num=[i for i in range(46) if i not in cat]
        pools={}; references={}; scaler_rows=[]
        for c in classes:
            ids=train[families[train]==names[c]]
            rng=np.random.default_rng(design['reference']['seed']+ds_index*10007+c*1009)
            nr=min(design['reference']['maximum_per_class'], max(1,int(len(ids)*design['reference']['fraction'])))
            ref=rng.choice(ids,nr,replace=False); pool=ids[~np.isin(ids,ref)]
            assert len(pool)>=40
            pools[c],references[c]=pool,ref
            scaler_rows.append(rng.choice(pool,min(5000,len(pool)),replace=False))
            np.save(d/f'class_{c}_reference_ids.npy',ref)
            np.save(d/f'class_{c}_candidate_ids.npy',pool)
        # Equal class weight in each marginal quantile, including small web class.
        sizes=np.array([len(r) for r in scaler_rows])
        fit_ids=np.concatenate(scaler_rows)
        raw_fit=np.asarray(suite['X'][fit_ids],dtype='float32')
        transformed=numeric_transform(raw_fit[:,num])
        weights=np.concatenate([np.full(n,1/n) for n in sizes])
        quantiles=[]
        for j in range(len(num)):
            order=np.argsort(transformed[:,j],kind='stable'); w=weights[order]
            positions=(np.cumsum(w)-w/2)/w.sum()
            quantiles.append(np.interp([.25,.5,.75],positions,transformed[order,j]))
        q=np.asarray(quantiles).T; center=q[1]; scale=np.maximum(q[2]-q[0],.1)
        np.savez(d/'scaler.npz',fit_source_ids=fit_ids,numeric_indices=num,categorical_indices=cat,
                 center=center,scale=scale)
        write(d/'geometry.json',dict(feature_names=feature_names,numeric_indices=num,categorical_indices=cat,
                                    scaler_fit_rows=len(fit_ids),scaler_fit_counts=sizes.tolist()))
        contexts={(arm,seed):[] for arm in design['arms'] for seed in design['seeds']}
        for c in classes:
            begun=time.perf_counter(); ids=pools[c]; ref=references[c]
            progress(out,'prepare_class',dataset=ds,class_name=names[c],candidate_rows=len(ids),reference_rows=len(ref))
            x=np.nan_to_num(np.asarray(suite['X'][ids],dtype='float32')); x[x==0]=0
            xr=np.nan_to_num(np.asarray(suite['X'][ref],dtype='float32')); xr[xr==0]=0
            assert np.isfinite(x).all() and np.isfinite(xr).all()
            _,first,inverse,counts=unique_vectors(x)
            _,ref_first,_,_=unique_vectors(xr)
            vocab=[np.unique(x[:,j]) for j in cat]
            z=geometry(x,num,cat,center,scale,vocab); zr=geometry(xr,num,cat,center,scale,vocab)
            cluster={}
            for k in [20,40]:
                r,labels,centers,fit_ix,occupied=representatives(z,k,880101+ds_index*100003+c*101+k)
                cluster[k]=(r,labels,centers)
                np.savez(d/f'class_{c}_cluster{k}.npz',representative_source_ids=ids[r],
                         candidate_labels=labels,centers=centers,fit_source_ids=ids[fit_ix],occupied_groups=occupied)
            write(d/f'class_{c}_geometry.json',dict(vocabulary=[v.tolist() for v in vocab],dimension=z.shape[1],
                  candidate_unique_vectors=len(first),reference_unique_vectors=len(ref_first),
                  occupied_clusters={str(k):len(v[0]) for k,v in cluster.items()}))
            for seed in design['seeds']:
                seed_value=880003+ds_index*1000003+seed*10007+c*1009
                selected={
                    'row_random':np.random.default_rng(seed_value).choice(len(ids),20,replace=False),
                    'unique_random':first[np.random.default_rng(seed_value+101).choice(len(first),20,replace=False)]}
                for k in [20,40]:
                    reps=cluster[k][0]
                    selected[f'cluster{k}']=reps[farthest_order(z[reps],seed_value+203)]
                for arm,ix in selected.items():
                    assert len(np.unique(ids[ix]))==20 and not np.isin(ids[ix],ref).any()
                    contexts[arm,seed].append((ids[ix],np.full(20,c,dtype='int16')))
                    for n in design['budgets']:
                        picked=ix[:n]; distance=cdist(zr,z[picked]).min(axis=1); spread=pdist(z[picked])
                        unique_count=len(np.unique(inverse[picked])); covered=np.unique(cluster[20][1][picked])
                        group_counts=np.bincount(cluster[20][1],minlength=20)
                        row=dict(dataset=ds,class_name=names[c],class_id=c,arm=arm,seed=seed,n=n,
                            unique_vectors=unique_count,duplicate_rows=n-unique_count,
                            pair_distance_median=float(np.median(spread)),pair_distance_min=float(spread.min()),
                            reference_rows=len(ref),reference_unique_vectors=len(ref_first),
                            reference_distance_mean=float(distance.mean()),reference_distance_median=float(np.median(distance)),
                            reference_distance_p90=float(np.quantile(distance,.9)),
                            reference_unique_distance_median=float(np.median(distance[ref_first])),
                            reference_unique_distance_p90=float(np.quantile(distance[ref_first],.9)),
                            reference_zero_distance_rate=float((distance<1e-7).mean()),
                            covered_cluster20_count=len(covered),
                            covered_cluster20_candidate_mass=float(group_counts[covered].sum()/len(ids)))
                        coverage.append(row)
                        distances=d/'reference_distances'/f'{c}_{arm}_s{seed}_n{n}.npy'
                        distances.parent.mkdir(exist_ok=True);np.save(distances,distance.astype('float32'))
            inventories.append(dict(dataset=ds,class_name=names[c],class_id=c,original_train_rows=len(ids)+len(ref),
                candidate_rows=len(ids),candidate_unique_vectors=len(first),reference_rows=len(ref),
                reference_unique_vectors=len(ref_first),test_rows=meta['test_counts'][c],geometry_dimensions=z.shape[1],
                occupied_cluster20=len(cluster[20][0]),occupied_cluster40=len(cluster[40][0]),
                seconds=time.perf_counter()-begun))
            write(out/'coverage.json',coverage);write(out/'inventory.json',inventories)
            del x,xr,z,zr,inverse;gc.collect()
        for (arm,seed),parts in contexts.items():
            ids=np.concatenate([p[0] for p in parts]); labels=np.concatenate([p[1] for p in parts])
            dest=d/arm/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            pos=np.searchsorted(clean_source,ids)
            assert np.array_equal(clean_source[pos],ids)
            np.testing.assert_array_equal(families[ids],np.asarray(names)[labels])
            assert not np.isin(ids,test_ids).any()
            for key,value in dict(X=np.nan_to_num(np.asarray(suite['X'][ids],dtype='float32')),y=labels,
                                  source_ids=ids,hashes=all_hashes[pos]).items():
                np.save(dest/f'{key}.npy',value)
            for n in design['budgets']:
                np.save(dest/f'n_{n}_indices.npy',np.concatenate([np.arange(j*20,j*20+n) for j in range(len(classes))]))
        write(d/'manifest.json',dict(meta,source_run=str(source),reference_seed=design['reference']['seed'],
                                    selected_cap=20,candidate_counts_after_reference={names[c]:len(pools[c]) for c in classes}))
    del suite;gc.collect()
    write(out/'prepared.json',dict(passed=True,completed_at=stamp(),seconds=time.perf_counter()-started,
                                  coverage_rows=len(coverage),inventory_classes=len(inventories),source_sha256=sha(raw)))
    progress(out,'prepared',seconds=time.perf_counter()-started)

def run_job(out,cfg,ds,arm,seed,n,method):
    import torch
    job=out/'jobs'/f'{ds}_{arm}_s{seed}_n{n}_{method}';job.mkdir(parents=True,exist_ok=True)
    if (job/'COMPLETE.json').exists():
        assert sha(job/'predictions.npy')==read(job/'COMPLETE.json')['predictions_sha256']
        return read(job/'results.json')
    d=out/'data'/ds;meta=read(d/'manifest.json');names=meta['class_names'];classes=np.asarray(meta['attack_classes'])
    selected=d/arm/f'seed_{seed}';ix=np.load(selected/f'n_{n}_indices.npy')
    x=np.load(selected/'X.npy')[ix]; y=np.load(selected/'y.npy')[ix];ids=np.load(selected/'source_ids.npy')[ix]
    local=np.searchsorted(classes,y)
    np.testing.assert_array_equal(np.bincount(local,minlength=len(classes)),np.full(len(classes),n))
    np.save(job/'train_source_ids.npy',ids)
    tx=np.load(d/'test_X.npy',mmap_mode='r');ty=np.load(d/'test_y.npy',mmap_mode='r');order=np.load(d/'query_order.npy')
    assert not np.any(ty==meta['benign'])
    settings=dict(cfg,seed=seed);torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats()
    progress(out,'model',dataset=ds,arm=arm,seed=seed,n=n,method=method)
    model,construct=timed(lambda:make_base('xgb' if method=='xgb' else 'tabpfn',settings,len(classes)))
    _,fit=timed(lambda:model.fit(x,local));np.testing.assert_array_equal(model.classes_,np.arange(len(classes)))
    before=weights_digest(model.models_[0].state_dict()) if method=='pfn' else None
    if before:
        expected=read(Path(meta['source_exp86'])/'jobs'/f'{ds}_s{seed}_n20_pfn'/'results.json')['weights_sha256']
        assert before==expected
    else:
        model.save_model(job/'model.ubj')
    pred=np.lib.format.open_memmap(job/'predictions.npy',mode='w+',dtype='int16',shape=(len(ty),));pred[:]=-1
    services=[]
    for start in range(0,len(order),cfg['batch']):
        batch=order[start:start+cfg['batch']]
        prob,seconds=timed(lambda:model.predict_proba(np.asarray(tx[batch])))
        assert prob.shape==(len(batch),len(classes)) and np.isfinite(prob).all()
        assert np.allclose(prob.sum(1),1,atol=1e-3)
        pred[batch]=classes[prob.argmax(1)];services.append([start,len(batch),seconds])
    pred.flush();assert np.isin(pred,classes).all()
    if before: assert before==weights_digest(model.models_[0].state_dict())
    metrics=confusion_metrics(ty,pred,names,classes)
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(ty,pred,labels=np.arange(len(names)))
    np.testing.assert_array_equal(cm,metrics['confusion'])
    den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(len(names)),where=den!=0)
    assert abs(f[classes].mean()-metrics['macro_f1'])<1e-12
    overlap=np.isin(np.load(d/'test_hash.npy',mmap_mode='r'),np.load(selected/'hashes.npy')[ix])
    result=dict(dataset=ds,arm=arm,seed=seed,n=n,method=method,context_rows=len(y),class_names=names,
        train_ids_sha256=sha(job/'train_source_ids.npy'),train_class_counts=np.bincount(y,minlength=len(names)).tolist(),
        test_rows=len(ty),metrics=metrics,preparation_seconds=construct+fit,construct_seconds=construct,fit_seconds=fit,
        inference_seconds=sum(s[2] for s in services),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        weights_sha256=before,finetuning=False,training_unique_feature_rows=len(unique_vectors(x)[1]),
        exact_feature_overlap_attack_rows=int(overlap.sum()),independent_confusion_check=True,
        predictions_path=str(job/'predictions.npy'),query_order_sha256=meta['query_order_sha256'],completed_at=stamp())
    np.save(job/'service_times.npy',np.asarray(services));write(job/'results.json',result)
    write(job/'COMPLETE.json',dict(predictions_sha256=sha(job/'predictions.npy'),completed_at=stamp()))
    del model,pred,tx,ty;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    design=read(PROTOCOL);oldcfg=read(ROOT/design['source_run']/'config.json')
    cfg=dict(protocol='exp88.1',checkpoint=oldcfg['checkpoint'],threads=8,batch=8192,
             started_at=stamp(),design=design)
    if (out/'config.json').exists():cfg=read(out/'config.json')
    write(out/'config.json',cfg)
    sources=[Path(__file__),PROTOCOL,ROOT/'scripts/common/experiment_paths.py',
             ROOT/'scripts/v7/exp71/exp71_bounded_updates.py',ROOT/'scripts/v7/exp72/replay_models.py',
             ROOT/'scripts/v7/exp72/replay_protocol.py',ROOT/'scripts/v7/exp77/finetune_final.py',
             ROOT/'scripts/common/exp61_utils.py']
    for file in sources:
        dst=out/'source_dependencies'/file.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dst)
    write(out/'environment.json',{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','torch','tabpfn','xgboost']})
    write(out/'checkpoint.json',dict(path=cfg['checkpoint'],sha256=sha(Path(cfg['checkpoint']))))
    import torch
    torch.set_num_threads(cfg['threads'])
    jobs=[];begun=time.perf_counter()
    expected=len(design['datasets'])*len(design['arms'])*len(design['seeds'])*len(design['budgets'])*2
    try:
        with threadpool_limits(limits=8):prepare(out,cfg)
        # Interleave datasets and methods so progress yields comparable evidence.
        for arm in design['arms']:
            for n in design['budgets']:
                for ds in design['datasets']:
                    for seed in design['seeds']:
                        for method in ['xgb','pfn']:
                            jobs.append(run_job(out,cfg,ds,arm,seed,n,method))
                            write(out/'results.json',dict(state='running',jobs=jobs,expected_jobs=expected,updated_at=stamp()))
                            progress(out,'job_complete',completed=len(jobs),expected=expected,elapsed_seconds=time.perf_counter()-begun,
                                     last_f1=jobs[-1]['metrics']['macro_f1'])
        assert len(jobs)==expected
        write(out/'results.json',dict(state='complete',jobs=jobs,expected_jobs=expected,updated_at=stamp(),seconds=time.perf_counter()-begun))
        write(out/'COMPLETE.json',dict(completed_at=stamp(),jobs=len(jobs),results_sha256=sha(out/'results.json')))
    except BaseException as exc:
        write(out/'FAILURE.json',dict(at=stamp(),error=str(exc),traceback=traceback.format_exc(),completed_jobs=len(jobs)))
        raise

if __name__=='__main__':
    main()

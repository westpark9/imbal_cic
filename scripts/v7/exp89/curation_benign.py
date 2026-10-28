"""EXP89: original/clean train crossed with n benign examples, fixed full test."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents
             if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)

import argparse
from datetime import datetime
import gc
import importlib.metadata
import json
import pickle
import shutil
import subprocess
import time
import traceback
from zoneinfo import ZoneInfo
import numpy as np
from exp61_utils import sha
from replay_models import make_base, timed
from finetune_final import weights_digest

PROTOCOL = ROOT/'configs/experiments/exp89_protocol.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def stamp():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def progress(out, phase, **values):
    value = dict(at=stamp(), phase=phase, **values)
    write(out/'progress.json', value)
    with (out/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def score_cm(cm, names):
    cm = np.asarray(cm, dtype=np.int64)
    support = cm.sum(1)
    tp = cm.diagonal()
    fp, fn = cm.sum(0)-tp, support-tp
    ratio = lambda a,b: np.divide(a,b,out=np.zeros(len(names), dtype=float),where=b!=0)
    precision, recall, f1 = ratio(tp,tp+fp), ratio(tp,support), ratio(2*tp,2*tp+fp+fn)
    b = names.index('benign')
    attack = np.arange(len(names)) != b
    return dict(rows=int(cm.sum()), confusion=cm.tolist(),
                attack_macro_f1=float(f1[attack].mean()), all_macro_f1=float(f1.mean()),
                benign_fpr=float(1-recall[b]) if support[b] else None,
                attack_detection_recall=float(1-cm[attack,b].sum()/support[attack].sum()) if support[attack].sum() else None,
                attack_to_benign=int(cm[attack,b].sum()),
                classes=[dict(name=name, support=int(support[c]), TP=int(tp[c]), FP=int(fp[c]), FN=int(fn[c]),
                              precision=float(precision[c]), recall=float(recall[c]), f1=float(f1[c]))
                         for c,name in enumerate(names)])


def score_views(y, pred, retained, names):
    C = len(names)
    code = np.asarray(y,dtype=np.int64)*C+pred
    original = np.bincount(code,minlength=C*C).reshape(C,C)
    clean = np.bincount(code[retained],minlength=C*C).reshape(C,C)
    result = {}
    for version, cm in [('original',original),('clean',clean)]:
        result[version+'_all'] = score_cm(cm,names)
        attacks = cm.copy()
        attacks[names.index('benign'),:] = 0
        result[version+'_attack'] = score_cm(attacks,names)
    return result


def prepare(out, cfg):
    if (out/'prepared.json').exists():
        return
    design = cfg['design']
    source = ROOT/design['source_exp86']
    first = read(source/'data/cic2018/manifest.json')
    raw = Path(first['source_pickle'])
    progress(out,'verify_source',path=str(raw))
    digest = sha(raw)
    assert digest == first['source_pickle_sha256']
    with raw.open('rb') as stream:
        suite = pickle.load(stream)
    families = np.asarray(suite['families'])
    inventories = []
    for ds in design['datasets']:
        prior = read(source/'data'/ds/'manifest.json')
        clean_dir = Path(prior['clean_manifest']).parent
        clean_md = read(clean_dir/'manifest.json')
        assert clean_md['source']['sha256'] == digest
        names = prior['class_names']
        C = len(names)
        b = names.index('benign')
        d = out/'data'/ds
        d.mkdir(parents=True,exist_ok=True)
        train = {v:np.load(clean_dir/('original_train_idx.npy' if v=='original' else 'train_idx.npy'))
                 for v in design['train_versions']}
        ids = np.load(clean_dir/'original_test_idx.npy')
        clean_ids = np.load(clean_dir/'test_idx.npy')
        assert np.all(np.diff(ids)>0)
        assert np.isin(train['clean'],train['original']).all()
        assert not np.isin(train['original'],ids).any()
        retained = np.isin(ids,clean_ids,assume_unique=True)
        np.testing.assert_array_equal(ids[retained],clean_ids)
        np.testing.assert_array_equal(clean_ids,np.load(Path(prior['test_base'])/'test_ids.npy'))
        y = np.full(len(ids),-1,dtype='int16')
        for c,name in enumerate(names):
            y[families[ids]==name] = c
        assert y.min()>=0
        progress(out,'prepare_test',dataset=ds,original_rows=len(ids),clean_rows=len(clean_ids))
        tx = np.lib.format.open_memmap(d/'test_X.npy',mode='w+',dtype='float32',shape=(len(ids),46))
        for start in range(0,len(ids),200000):
            tx[start:start+200000] = np.nan_to_num(np.asarray(suite['X'][ids[start:start+200000]],dtype='float32'))
        tx.flush()
        assert np.isfinite(tx).all()
        old_x = np.load(Path(prior['test_base'])/'test_X.npy',mmap_mode='r')
        positions = np.flatnonzero(retained)
        for start in range(0,len(positions),200000):
            np.testing.assert_array_equal(tx[positions[start:start+200000]],old_x[start:start+200000])
        np.testing.assert_array_equal(y[retained],np.load(Path(prior['test_base'])/'test_y.npy'))
        for key,value in dict(test_ids=ids,test_y=y,retained_mask=retained,
                              query_order=np.random.default_rng(43).permutation(len(ids))).items():
            np.save(d/f'{key}.npy',value)
        candidate_counts = {}
        for version,pool_ids in train.items():
            # Save membership inventories without modifying the old cache.
            np.save(d/f'{version}_train_ids.npy',pool_ids)
            candidates = {c:pool_ids[families[pool_ids]==name] for c,name in enumerate(names)}
            candidate_counts[version] = {names[c]:len(pool) for c,pool in candidates.items()}
            assert sum(map(len,candidates.values())) == len(pool_ids)
            for seed in design['seeds']:
                selected = np.concatenate([np.random.default_rng(seed+c*1009).choice(pool,20,replace=False)
                                           for c,pool in candidates.items()])
                labels = np.repeat(np.arange(C,dtype='int16'),20)
                assert len(np.unique(selected))==20*C and not np.isin(selected,ids).any()
                np.testing.assert_array_equal(families[selected],np.asarray(names)[labels])
                s = d/version/f'seed_{seed}'
                s.mkdir(parents=True,exist_ok=True)
                for key,value in dict(X=np.nan_to_num(np.asarray(suite['X'][selected],dtype='float32')),
                                      y=labels,source_ids=selected).items():
                    np.save(s/f'{key}.npy',value)
                if version=='clean':
                    old = source/'data'/ds/f'seed_{seed}'
                    np.testing.assert_array_equal(selected[labels!=b],np.load(old/'source_ids.npy'))
                    np.testing.assert_array_equal(np.load(s/'X.npy')[labels!=b],np.load(old/'X.npy'))
                for n in design['budgets']:
                    for benign in design['benign_training']:
                        classes = [c for c in range(C) if benign or c!=b]
                        index = np.concatenate([np.arange(c*20,c*20+n) for c in classes])
                        np.save(s/f'n{n}_benign{int(benign)}_indices.npy',index)
        meta = dict(dataset=ds,class_names=names,benign=b,features=46,feature_names=clean_md['feature_names'],
                    original_test_rows=len(ids),clean_test_rows=int(retained.sum()),
                    original_test_counts=np.bincount(y,minlength=C).tolist(),
                    clean_test_counts=np.bincount(y[retained],minlength=C).tolist(),
                    candidate_counts=candidate_counts,source_pickle=str(raw),source_sha256=digest,
                    clean_manifest=prior['clean_manifest'],clean_manifest_sha256=sha(clean_dir/'manifest.json'),
                    retained_tensors_exactly_match_exp86=True,clean_attack_samples_exactly_match_exp86=True,
                    query_order_sha256=sha(d/'query_order.npy'),original_ids_sha256=sha(d/'test_ids.npy'))
        write(d/'manifest.json',meta)
        for c,name in enumerate(names):
            inventories.append(dict(dataset=ds,class_name=name,original_train=candidate_counts['original'][name],
                                    clean_train=candidate_counts['clean'][name],
                                    original_test=meta['original_test_counts'][c],clean_test=meta['clean_test_counts'][c]))
        write(out/'inventory.json',inventories)
        del tx,old_x,train,y,retained
        gc.collect()
    del suite,families
    gc.collect()
    write(out/'prepared.json',dict(completed_at=stamp(),passed=True,source_sha256=digest,
                                   paired_models=True,paired_benign_arms=True,nested=True))


def run_job(out,cfg,ds,version,benign,seed,n,method,completed):
    import torch
    name = f'{ds}_{version}_benign{int(benign)}_s{seed}_n{n}_{method}'
    job = out/'jobs'/name
    if (job/'COMPLETE.json').exists():
        assert sha(job/'predictions.npy') == read(job/'COMPLETE.json')['predictions_sha256']
        return read(job/'results.json')
    job.mkdir(parents=True,exist_ok=True)
    d = out/'data'/ds
    meta = read(d/'manifest.json')
    names = meta['class_names']
    s = d/version/f'seed_{seed}'
    index = np.load(s/f'n{n}_benign{int(benign)}_indices.npy')
    x,y,ids = (np.load(s/f'{key}.npy')[index] for key in ['X','y','source_ids'])
    classes = np.unique(y)
    np.testing.assert_array_equal(np.unique(y,return_counts=True)[1],np.full(len(classes),n))
    assert (meta['benign'] in classes) == benign
    np.save(job/'train_source_ids.npy',ids)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    progress(out,'fit',job=name,completed_jobs=completed,expected_jobs=cfg['design']['expected_jobs'])
    model,construct = timed(lambda:make_base('xgb' if method=='xgb' else 'tabpfn',dict(cfg,seed=seed),len(classes)))
    _,fit = timed(lambda:model.fit(x,np.searchsorted(classes,y)))
    np.testing.assert_array_equal(model.classes_,np.arange(len(classes)))
    before = weights_digest(model.models_[0].state_dict()) if method=='pfn' else None
    splits = None
    if method=='xgb':
        splits = int((model.get_booster().trees_to_dataframe().Feature!='Leaf').sum())
        model.save_model(job/'model.ubj')
    else:
        prior = ROOT/cfg['design']['source_exp86']/'jobs'/f'{ds}_s{seed}_n20_pfn'/'results.json'
        assert before == read(prior)['weights_sha256']
    tx = np.load(d/'test_X.npy',mmap_mode='r')
    ty = np.load(d/'test_y.npy',mmap_mode='r')
    order = np.load(d/'query_order.npy',mmap_mode='r')
    retained = np.load(d/'retained_mask.npy')
    pred = np.lib.format.open_memmap(job/'predictions.npy',mode='w+',dtype='int16',shape=(len(ty),))
    pred[:] = -1
    services = []
    last_progress = time.monotonic()
    for start in range(0,len(order),cfg['batch']):
        rows = order[start:start+cfg['batch']]
        probability,seconds = timed(lambda:model.predict_proba(np.asarray(tx[rows])))
        assert probability.shape==(len(rows),len(classes)) and np.isfinite(probability).all()
        assert np.allclose(probability.sum(1),1,atol=1e-3)
        pred[rows] = classes[probability.argmax(1)]
        services.append([start,len(rows),seconds])
        if time.monotonic()-last_progress>=30:
            progress(out,'predict',job=name,completed_jobs=completed,expected_jobs=cfg['design']['expected_jobs'],
                     rows_done=start+len(rows),rows_total=len(order))
            last_progress=time.monotonic()
    pred.flush()
    assert np.isin(pred,classes).all()
    if before:
        assert before == weights_digest(model.models_[0].state_dict())
    views = score_views(ty,pred,retained,names)
    continuity = None
    if version=='clean' and not benign and n!=3:
        previous = ROOT/cfg['design']['source_exp86']/'jobs'/f'{ds}_s{seed}_n{n}_{method}'
        old_pred = np.load(previous/'predictions.npy',mmap_mode='r')
        old = read(previous/'results.json')
        np.testing.assert_array_equal(ids,np.load(previous/'train_source_ids.npy'))
        continuity = dict(reference=str(previous),same_train_ids=True,
                          retained_prediction_disagreements=int(np.count_nonzero(pred[retained]!=old_pred)),
                          attack_macro_f1_delta=views['clean_attack']['attack_macro_f1']-old['phase_a']['macro_f1'],
                          note='Original full query set changes batch composition; both current test views use the same predictions.')
        if method=='xgb':
            assert continuity['retained_prediction_disagreements']==0
    result = dict(job_id=name,dataset=ds,train_version=version,benign_training=benign,seed=seed,n=n,method=method,
                  train_rows=len(ids),train_class_counts=np.bincount(y,minlength=len(names)).tolist(),
                  train_ids_sha256=sha(job/'train_source_ids.npy'),test_rows=len(ty),metrics=views,
                  construct_seconds=construct,fit_seconds=fit,preparation_seconds=construct+fit,
                  inference_seconds=sum(v[2] for v in services),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                  xgb_splits=splits,weights_sha256=before,finetuning=False,continuity=continuity,
                  query_order_sha256=meta['query_order_sha256'],predictions_path=str(job/'predictions.npy'),completed_at=stamp())
    np.save(job/'service_times.npy',np.asarray(services))
    write(job/'results.json',result)
    write(job/'COMPLETE.json',dict(completed_at=stamp(),predictions_sha256=sha(job/'predictions.npy')))
    del model,pred,tx,ty,order,retained
    gc.collect()
    torch.cuda.empty_cache()
    return result


def self_test():
    names=['benign','a','b']
    y=np.array([0,0,1,1,2,2])
    p=np.array([0,1,0,1,1,2])
    retained=np.array([True,False,True,True,True,True])
    scores=score_views(y,p,retained,names)
    assert scores['original_attack']['attack_to_benign']==1
    assert scores['original_all']['benign_fpr']==.5
    assert scores['clean_all']['benign_fpr']==0
    assert scores['original_attack']['classes'][1]['FP']==1
    assert scores['original_all']['classes'][1]['FP']==2
    assert scores['original_attack']['classes'][1]['FN']==1
    assert scores['original_all']['rows']==6 and scores['original_attack']['rows']==4
    print('Metric mask and benign-output checks passed.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out is None:
        parser.error('--out is required')
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    design=read(PROTOCOL)
    old=read(ROOT/design['source_exp86']/'config.json')
    cfg=dict(protocol='exp89.1',checkpoint=old['checkpoint'],threads=8,batch=8192,started_at=stamp(),design=design)
    if (out/'config.json').exists():
        cfg=read(out/'config.json')
    write(out/'config.json',cfg)
    sources=[*Path(__file__).parent.glob('*.py'),PROTOCOL,ROOT/'scripts/common/experiment_paths.py',
             ROOT/'scripts/common/lablog.py',ROOT/'scripts/v7/exp71/exp71_bounded_updates.py',
             ROOT/'scripts/v7/exp72/replay_models.py',ROOT/'scripts/v7/exp77/finetune_final.py']
    for source in sources:
        dest=out/'source_dependencies'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            shutil.copy2(source,dest)
    write(out/'environment.json',{name:importlib.metadata.version(name)
                                  for name in ['numpy','scipy','scikit-learn','torch','tabpfn','xgboost']})
    write(out/'checkpoint.json',dict(path=cfg['checkpoint'],sha256=sha(Path(cfg['checkpoint']))))
    import torch
    torch.set_num_threads(cfg['threads'])
    jobs=[]
    begun=time.perf_counter()
    try:
        prepare(out,cfg)
        for n in design['budgets']:
            for seed in design['seeds']:
                for ds in design['datasets']:
                    for version in design['train_versions']:
                        for benign in design['benign_training']:
                            for method in design['methods']:
                                jobs.append(run_job(out,cfg,ds,version,benign,seed,n,method,len(jobs)))
                                write(out/'results.json',dict(state='running',jobs=jobs,expected_jobs=design['expected_jobs'],updated_at=stamp()))
                                progress(out,'job_complete',completed_jobs=len(jobs),expected_jobs=design['expected_jobs'],
                                         elapsed_seconds=time.perf_counter()-begun,last_job=jobs[-1]['job_id'])
        assert len(jobs)==design['expected_jobs']
        write(out/'results.json',dict(state='complete',jobs=jobs,expected_jobs=design['expected_jobs'],updated_at=stamp(),
                                      elapsed_seconds=time.perf_counter()-begun))
        write(out/'MODELS_COMPLETE.json',dict(completed_at=stamp(),jobs=len(jobs),results_sha256=sha(out/'results.json')))
        for script in ['analyze_curation.py','report_curation.py']:
            progress(out,'finalizing',script=script,completed_jobs=len(jobs))
            subprocess.run([sys.executable,str(Path(__file__).with_name(script)),'--run',str(out)],check=True)
        write(out/'COMPLETE.json',dict(completed_at=stamp(),jobs=len(jobs),validation=read(out/'analysis/validation.json')['passed'],
                                      report=str(ROOT/'docs/lablog/reports'/f"{cfg['started_at'][:10].replace('-','')}_exp89.tex")))
        progress(out,'complete',completed_jobs=len(jobs),expected_jobs=design['expected_jobs'])
    except BaseException as exc:
        write(out/'FAILURE.json',dict(at=stamp(),error=str(exc),traceback=traceback.format_exc(),completed_jobs=len(jobs)))
        report=Path(__file__).with_name('report_curation.py')
        if report.exists():
            subprocess.run([sys.executable,str(report),'--run',str(out),'--partial'],check=False)
        raise


if __name__=='__main__':
    main()

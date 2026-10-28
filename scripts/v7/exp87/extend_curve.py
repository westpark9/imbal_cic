"""Authorized seed43 expansion beyond the balanced class-count limit."""
from pathlib import Path
import argparse,copy,gc,json,pickle,shutil,subprocess,sys,time,traceback
import numpy as np
from attack_saturation import ROOT,read,stamp,progress,run_job
from exp61_utils import write,sha


def prepare(out,source,plan):
    if (out/'prepared.json').exists():return
    oldcfg=read(source/'config.json')
    exp86=ROOT/oldcfg['design']['source_run']
    meta=read(exp86/'data/cic2018/manifest.json');raw=Path(meta['source_pickle'])
    progress(out,'loading_source',source=str(raw))
    assert sha(raw)==meta['source_pickle_sha256']
    with raw.open('rb') as handle:suite=pickle.load(handle)
    families=np.asarray(suite['families'])
    for ds,conditions in plan['conditions'].items():
        old=source/'data'/ds;manifest=read(old/'manifest.json');names=manifest['class_names']
        clean=Path(manifest['clean_manifest']).parent;train=np.load(clean/'train_idx.npy')
        clean_source=np.load(clean/'source_idx.npy',mmap_mode='r');hashes=np.load(clean/'vector_hash.npy',mmap_mode='r')
        original=old/'seed_43';oldids=np.load(original/'source_ids.npy');oldy=np.load(original/'y.npy')
        chosen=[];labels=[];blocks={};cursor=0
        for c in manifest['attack_classes']:
            pool=train[families[train]==names[c]];first=oldids[oldy==c]
            cap=conditions[-1]['class_counts'][names[c]]
            remaining=pool[~np.isin(pool,first)]
            extra=np.random.default_rng(8702001+c*1009).choice(remaining,cap-len(first),replace=False)
            chosen.append(np.concatenate([first,extra]));labels.append(np.full(cap,c,dtype='int16'))
            blocks[c]=(cursor,cap);cursor+=cap
        ids=np.concatenate(chosen);labels=np.concatenate(labels);idx=np.searchsorted(clean_source,ids)
        assert len(np.unique(ids))==len(ids) and np.array_equal(clean_source[idx],ids)
        np.testing.assert_array_equal(families[ids],np.array(names)[labels])
        assert not np.isin(ids,np.load(old/'test_ids.npy',mmap_mode='r')).any()
        values=np.nan_to_num(np.asarray(suite['X'][ids],dtype='float32'));assert np.isfinite(values).all()
        dest=out/'data'/ds;dest.mkdir(parents=True,exist_ok=True)
        for name in ['test_X.npy','test_y.npy','test_hash.npy','test_ids.npy','query_order.npy','original_test_positions.npy']:
            if not (dest/name).exists():(dest/name).symlink_to(old/name)
        seed=dest/'seed_43';seed.mkdir(exist_ok=True)
        for name,array in dict(X=values,y=labels,source_ids=ids,hashes=hashes[idx]).items():np.save(seed/f'{name}.npy',array)
        for condition in conditions:
            n=condition['target_n']
            sub=np.concatenate([np.arange(start,start+condition['class_counts'][names[c]]) for c,(start,cap) in blocks.items()])
            assert len(sub)==condition['total_rows']
            np.save(seed/f'n_{n}_indices.npy',sub)
        for c,(start,_) in blocks.items():
            orig=np.flatnonzero(oldy==c);np.testing.assert_array_equal(ids[start:start+len(orig)],oldids[orig])
            np.testing.assert_array_equal(values[start:start+len(orig)],np.load(original/'X.npy',mmap_mode='r')[orig])
        manifest.update(allocation='available_cap',conditions=conditions,source_balanced_run=str(source),nested=True)
        write(dest/'manifest.json',manifest)
        progress(out,'dataset_prepared',dataset=ds,total_rows=len(ids),class_counts=conditions[-1]['class_counts'])
    del suite,families;gc.collect()
    write(out/'prepared.json',dict(at=stamp(),passed=True,allocation='available_cap',benign_rows=0,seed=43))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--wait-for',required=True,type=Path)
    args=parser.parse_args();out=args.out.resolve();source=args.source.resolve();base=args.wait_for.resolve()
    plan=read(ROOT/'configs/experiments/proposals/20261022_exp87_beyond_balanced_cap.json')
    out.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(read(source/'config.json'));cfg.update(protocol='exp87.3',allocation='available_cap',
        started_at=stamp(),source_run=str(source),report_base=str(base),extension_plan=plan)
    write(out/'config.json',cfg)
    for file in [Path(__file__),Path(__file__).with_name('attack_saturation.py'),Path(__file__).with_name('extension_report.py')]:
        dest=out/'source_dependencies'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    for name in ['checkpoint.json','environment.json']:shutil.copy2(source/name,out/name)
    jobs=[];state='queued'
    def save(stage):write(out/'results.json',dict(state=state,stage=stage,jobs=jobs,updated_at=stamp(),plan=plan,report_base=str(base)))
    def report():
        p=subprocess.run([sys.executable,str(Path(__file__).with_name('report_saturation.py')),'--run',str(base),'--extension',str(out)],capture_output=True,text=True)
        if p.returncode:
            write(out/'REPORT_FAILURE.json',dict(at=stamp(),stderr=p.stderr[-8000:],stdout=p.stdout[-3000:]));print(p.stderr[-8000:],flush=True);return False
        (out/'REPORT_FAILURE.json').unlink(missing_ok=True);print(p.stdout,flush=True);return True
    save('waiting_for_fullcurve')
    try:
        while read(base/'results.json')['state']=='running':time.sleep(15)
        # The baseline reporter finishes before touching the shared report.
        while read(base/'RUN_META.json').get('status')=='running':time.sleep(5)
        state='running';save('preparing');report();prepare(out,source,plan)
        import torch
        torch.set_num_threads(cfg['threads'])
        for level in range(max(map(len,plan['conditions'].values()))):
            for ds,conditions in plan['conditions'].items():
                if level>=len(conditions):continue
                n=conditions[level]['target_n']
                for method in ['xgb','pfn']:
                    jobs.append(run_job(out,cfg,ds,43,n,method));save('extension')
                report()
        state='complete';save('complete');progress(out,'complete',completed_jobs=len(jobs))
    except BaseException as exc:
        state='failed';write(out/'FAILURE.json',dict(at=stamp(),error=str(exc),traceback=traceback.format_exc()));save('failed');raise
    finally:
        if not report() and state=='complete':raise RuntimeError('Report verification failed')


if __name__=='__main__':main()

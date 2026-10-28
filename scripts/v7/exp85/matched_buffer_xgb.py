#!/usr/bin/env python3
"""Retrain XGB on the exact per-stage contexts of completed EXP84 PFN runs."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse,importlib.metadata,shutil,subprocess,time,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from exp61_utils import write,sha
from small_label_updates import read,run_job,progress,BUDGETS
from noft_dist_budget import load_source

REFERENCE=ROOT/'tabpfn/results/v7/exp84/20261021_111613_small_labels_buffer_s44_s45_29cd3f40'
PROTOCOL=ROOT/'configs/experiments/exp85_protocol.json'


def report(out):
    subprocess.run([sys.executable,str(Path(__file__).with_name('build_matched_record.py')),'--run',str(out)],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--policies',nargs='+',choices=['random','fifo','similar'],default=['random'])
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    reference=read(REFERENCE/'results.json')
    assert reference['state']=='complete' and reference['validated_seeds']==[43,44,45]
    cfg=dict(protocol='exp85.1',research_version='v7',seeds=[43,44,45],datasets=['cic2018','toniot'],
             new_budgets=BUDGETS,policies=args.policies,methods=['xgb_'+p for p in args.policies],
             source_runs=reference['seed_runs'],reference_run=str(REFERENCE),buffer_capacity=1000,
             expected_jobs=36*len(args.policies),report_date=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d'),
             prediction_scope='Final model, full fixed test',pfn_rerun=False)
    if (out/'config.json').exists():
        old=read(out/'config.json');cfg['report_date']=old['report_date'];assert cfg==old
    write(out/'config.json',cfg)
    env={k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn','tabpfn','xgboost']}
    assert env==read(REFERENCE/'environment.json');write(out/'environment.json',env)
    deps=[Path(__file__),Path(__file__).with_name('build_matched_record.py'),PROTOCOL]
    deps += [Path(p) for p in read(REFERENCE/'source_hashes.json') if Path(p).suffix=='.py']
    source_hashes={}
    for p in dict.fromkeys(deps):
        dest=out/'source_dependencies'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==sha(p)
        else:shutil.copy2(p,dest)
        source_hashes[str(p)]=sha(p)
    write(out/'source_hashes.json',source_hashes)
    state=dict(state='preparing',expected_jobs=cfg['expected_jobs'],completed_jobs=0,validated_seeds=[],
               seed_runs={str(s):str(out/f'seed_{s}') for s in cfg['seeds']},started_epoch=time.time())
    write(out/'results.json',state);error=None
    try:
        checks=[]
        reference_hashes={}
        for seed in cfg['seeds']:
            src=Path(cfg['source_runs'][str(seed)]);dest=out/f'seed_{seed}'
            val=read(src/'validation.json');assert val['passed'] and val['completed']==60
            seed_cfg=read(src/'config.json');seed_cfg.update(protocol='exp85.1',methods=cfg['methods'],report_date=cfg['report_date'])
            write(dest/'config.json',seed_cfg)
            for ds in cfg['datasets']:
                d=dest/'data'/ds;d.mkdir(parents=True,exist_ok=True)
                for name in ['manifest.json','panel_ids.npy','query_order.npy']:
                    shutil.copy2(src/'data'/ds/name,d/name);assert sha(d/name)==sha(src/'data'/ds/name)
                for n in BUDGETS:
                    case=dest/f'new_{n}'/'data'/ds;case.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(src/f'new_{n}'/'data'/ds/'plans.json',case/'plans.json')
                    for policy in cfg['policies']:
                        pfn=src/f'new_{n}'/'jobs'/f'{ds}_pfn_{policy}'/'results.json'
                        r=read(pfn);assert r['seed']==seed and r['context_rows']==1000
                        history=[dict(context_indices=f['context_indices'],removed_indices=f['removed_indices']) for f in r['fits']]
                        assert history==read(src/f'new_{n}'/'data'/ds/f'{policy}_preflight.json')
                        write(case/f'{policy}_preflight.json',history)
                        reference_hashes[str(pfn)]=sha(pfn)
                        checks.append(dict(seed=seed,dataset=ds,n=n,policy=policy,stages=len(history),pfn_result=str(pfn)))
        write(out/'reference_hashes.json',reference_hashes)
        write(out/'preflight.json',dict(passed=True,conditions=len(checks),checks=checks,
            exact_same_row_ids_and_order=True,fixed_test_query_order_and_panel=True,
            selection_timing='Recompute deletion inside timer; assert exact saved PFN IDs; do not reuse prior timing'))
        state['state']='running';write(out/'results.json',state);report(out)
        import torch
        torch.set_num_threads(8);torch.cuda.init();torch.cuda.synchronize()
        start=time.perf_counter();a=torch.ones((512,512),device='cuda');b=a@a;torch.cuda.synchronize();del a,b
        write(out/'warmup.json',dict(seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),excluded=True))
        for seed in cfg['seeds']:
            seed_out=out/f'seed_{seed}';seed_cfg=read(seed_out/'config.json');torch.manual_seed(seed)
            jobs=[];state['active_seed']=seed;write(out/'results.json',state)
            for n in BUDGETS:
                for ds in cfg['datasets']:
                    meta,arr,_=load_source(ds);plans=read(seed_out/f'new_{n}'/'data'/ds/'plans.json')
                    for policy in cfg['policies']:
                        method='xgb_'+policy
                        progress(out,'running',seed=seed,dataset=ds,n=n,method=method,
                                 completed_jobs=state['completed_jobs'],expected_jobs=cfg['expected_jobs'])
                        result=run_job(seed_out,seed_cfg,ds,n,method,meta,arr,plans)
                        reference_path=Path(cfg['source_runs'][str(seed)])/f'new_{n}'/'jobs'/f'{ds}_pfn_{policy}'/'results.json'
                        original=read(reference_path)
                        assert all(a['context_indices']==b['context_indices'] and a['removed_indices']==b['removed_indices']
                                   for a,b in zip(result['fits'],original['fits']))
                        result.update(reference_pfn_result=str(reference_path),matched_policy=policy,
                                      matched_rows_all_stages=True,pfn_predictions_reused=True,
                                      note='Fresh XGB retraining on exact saved PFN buffer rows; all new rows admitted; no PFN rerun')
                        write(seed_out/f'new_{n}'/'jobs'/f'{ds}_{method}'/'results.json',result)
                        jobs.append(result);state['completed_jobs']+=1
                        write(seed_out/'results.json',dict(state='running',jobs=jobs,expected_jobs=12*len(cfg['policies'])))
                        write(out/'results.json',state)
            write(seed_out/'results.json',dict(state='complete',jobs=jobs,expected_jobs=12*len(cfg['policies'])))
            subprocess.run([sys.executable,str(ROOT/'scripts/v7/exp84/build_small_label_record.py'),
                            '--run',str(seed_out),'--validate-only'],cwd=ROOT,check=True)
            state['validated_seeds'].append(seed);write(out/'results.json',state);report(out)
        assert all(sha(p)==h for p,h in source_hashes.items())
        assert all(sha(p)==h for p,h in reference_hashes.items())
        state.update(state='complete',finished_epoch=time.time(),reference_preserved=True)
        write(out/'results.json',state);progress(out,'complete',completed_jobs=state['completed_jobs'],expected_jobs=cfg['expected_jobs'])
    except BaseException as exc:
        error=exc;state.update(state='failed',error=str(exc));write(out/'results.json',state)
        write(out/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        try:report(out)
        except Exception as exc:
            write(out/'REPORT_FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc()))
            if error is None:raise


if __name__=='__main__':main()

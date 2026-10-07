#!/usr/bin/env python3
"""Portable A100/RTX runner: cleaned CIC2018 + ToN, four SOTA methods and cost."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import argparse
from datetime import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from exp61_utils import sha,write

ROOT=repo_root(__file__)
METHODS=['xgb','distpfn','boostpfn','xgb_full','localpfn']


def environment(gpu,threads):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads),
        NUMEXPR_NUM_THREADS=str(threads),MALLOC_ARENA_MAX='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')


def snapshot(destination):
    paths=[]
    for directory in ['scripts','tabpfn/scripts','tabpfn/third_party/BoostPFN','tabpfn/third_party/LoCalPFN']:
        paths.extend((ROOT/directory).rglob('*.py'))
    paths.extend((ROOT/'tabpfn/configs/exp61').glob('*'))
    paths.append(ROOT/'tabpfn/configs/exp57_clean_split_reference.json')
    paths.append(ROOT/'configs/experiments/registry.json')
    hashes={}
    for p in paths:
        if not p.is_file():continue
        relative=p.relative_to(ROOT);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);hashes[str(relative)]=sha(target)
    write(destination.parent/'source_sha256.json',hashes)


def aggregate(root,request):
    import csv
    rows=[];classes=[]
    for dataset in request['datasets']:
        for method in request['methods']:
            path=root/f'{dataset}_{method}'/'results.json'
            if not (path.parent/'COMPLETE.json').exists():continue
            for entry in read_record(path):
                rows.append(dict(dataset=dataset,**{k:v for k,v in entry.items() if k!='classes'}))
                classes.extend(dict(dataset=dataset,method=entry['method'],**c) for c in entry['classes'])
    for name,data in [('summary.csv',rows),('class_metrics.csv',classes)]:
        if not data:continue
        with (root/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)


def run_parallel(root,request,status,completed,failed):
    """Keep independent jobs live, with durable status and serial OOM recovery."""
    pending=[(ds,method,False) for method in request['methods'] for ds in request['datasets']]
    # Start one long retrieval/FT job alongside the shorter baselines.
    first_local=next((job for job in pending if job[1]=='localpfn'),None)
    if first_local:pending.remove(first_local);pending.insert(0,first_local)
    active={};retry=[]
    while pending or active or retry:
        # A failed allocation is retried after concurrent jobs finish.
        if not pending and not active and retry:
            pending=retry;retry=[]
        limit=1 if pending and pending[0][2] else request.get('max_parallel',1)
        while pending and len(active)<limit:
            ds,method,serial=pending.pop(0);job=f'{ds}_{method}';out=root/job
            if (out/'COMPLETE.json').exists():completed.append(job);continue
            out.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u',str(script_path('exp61_sota_worker.py')),'--request',str(root/'request.json'),'--dataset',ds,'--method',method]
            attempt=len(list(out.glob('attempt_*.log')))+1;logpath=out/f'attempt_{attempt}.log'
            log=logpath.open('w');start=time.time()
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment(request['gpu'],request['cpu_threads']))
            active[job]=dict(proc=proc,log=log,out=out,attempt=attempt,start=start,cmd=cmd,ds=ds,method=method,serial=serial)
            print(f'START {job} pid={proc.pid} log={logpath}',flush=True)
        details=[]
        for job,item in list(active.items()):
            proc=item['proc'];out=item['out'];code=proc.poll()
            if code is None:
                path=out/'progress.json'
                details.append(dict(job=job,worker_pid=proc.pid,attempt=item['attempt'],elapsed_seconds=time.time()-item['start'],progress=read_record(path) if path.exists() else {}))
                continue
            item['log'].close()
            write(out/f'attempt_{item["attempt"]}.json',dict(returncode=code,wall_seconds=time.time()-item['start'],started_epoch=item['start'],command=item['cmd'],serial_retry=item['serial']))
            if code==0 and (out/'COMPLETE.json').exists():completed.append(job)
            else:
                error=(out/'ERROR.json').read_text() if (out/'ERROR.json').exists() else ''
                if not item['serial'] and ('out of memory' in error.lower() or code==-9):
                    retry.append((item['ds'],item['method'],True))
                    print(f'QUEUE SERIAL RETRY {job}',flush=True)
                else:failed.append(dict(job=job,returncode=code,log=str(out/f'attempt_{item["attempt"]}.log')))
            del active[job];aggregate(root,request)
            print(f'END {job} returncode={code}',flush=True)
        status('running',active=details,pending=[f'{ds}_{method}' for ds,method,_ in pending],serial_retries=[f'{ds}_{method}' for ds,method,_ in retry])
        if active:time.sleep(5)


def run(root):
    root=Path(root).resolve();request=read_record(root/'request.json')
    os.environ.update(environment(request['gpu'],request['cpu_threads']))
    started=time.time();completed=[];failed=[]
    # Duplicate launch of the same output cannot overwrite a running worker.
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This result directory already has an active controller')
        def status(state,**kw):write(root/'status.json',dict(state=state,controller_pid=os.getpid(),completed=completed,failed=failed,elapsed_seconds=time.time()-started,updated_epoch=time.time(),**kw))
        try:
            if not request['synthetic_smoke']:
                from exp57_prepare_inputs import ensure_clean_split
                ref=read_record(ROOT/'tabpfn/configs/exp57_clean_split_reference.json')
                status('verifying_source')
                if sha(request['data'])!=ref['source']['sha256']:raise ValueError('Original PKL hash mismatch')
                for ds in request['datasets']:
                    status('preparing_clean',dataset=ds)
                    verified=ensure_clean_split(request['data'],Path(request['clean_root'])/ref['datasets'][ds]['directory'],dict(source=ref['source'],dataset=ref['datasets'][ds]))
                    write(root/f'{ds}_input_verification.json',verified)
            if request['prepare_only']:status('prepared');return 0
            if request.get('max_parallel',1)>1:
                run_parallel(root,request,status,completed,failed)
                aggregate(root,request)
                status('complete' if not failed else 'completed_with_failures')
                return 0 if not failed else 1
            # Short jobs first; the expensive fine-tuning and retrieval jobs last.
            for method in request['methods']:
                for ds in request['datasets']:
                    job=f'{ds}_{method}';out=root/job
                    if (out/'COMPLETE.json').exists():completed.append(job);continue
                    out.mkdir(exist_ok=True)
                    cmd=[sys.executable,'-u',str(script_path('exp61_sota_worker.py')),'--request',str(root/'request.json'),'--dataset',ds,'--method',method]
                    attempt=len(list(out.glob('attempt_*.log')))+1;start=time.time()
                    with (out/f'attempt_{attempt}.log').open('w') as log:
                        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment(request['gpu'],request['cpu_threads']))
                        status('running',job=job,worker_pid=p.pid,attempt=attempt)
                        print(f'START {job} pid={p.pid} log={log.name}',flush=True)
                        while p.poll() is None:
                            time.sleep(10)
                            progress=out/'progress.json'
                            detail=read_record(progress) if progress.exists() else {}
                            status('running',job=job,worker_pid=p.pid,attempt=attempt,progress=detail)
                    write(out/f'attempt_{attempt}.json',dict(returncode=p.returncode,wall_seconds=time.time()-start,started_epoch=start,command=cmd))
                    if p.returncode==0 and (out/'COMPLETE.json').exists():completed.append(job)
                    else:failed.append(dict(job=job,returncode=p.returncode,log=str(out/f'attempt_{attempt}.log')))
                    aggregate(root,request)
                    print(f'END {job} returncode={p.returncode}',flush=True)
            aggregate(root,request)
            status('complete' if not failed else 'completed_with_failures')
            return 0 if not failed else 1
        except BaseException:
            status('failed',traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path);p.add_argument('--model-path',type=Path,default=ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt')
    p.add_argument('--root',type=Path,default=result_root(__file__)/'exp61_sota_clean_s43')
    p.add_argument('--clean-root',type=Path,default=ROOT/'data/derived')
    p.add_argument('--datasets',nargs='+',choices=['cic2018','toniot'],default=['cic2018','toniot'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--gpu',default='0');p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--max-parallel',type=int,default=1,help='Independent workers sharing the GPU; timing reflects contention')
    p.add_argument('--pfn-batch',type=int,default=125000);p.add_argument('--boost-batch',type=int,default=50000)
    p.add_argument('--local-batch',type=int,default=256);p.add_argument('--local-checkpoint-rows',type=int,default=50000)
    p.add_argument('--detach',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--synthetic-smoke',action='store_true',help='Tiny generated data; verifies execution only, never benchmark results')
    p.add_argument('--run-prepared',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.run_prepared:raise SystemExit(run(a.run_prepared))
    if any(v<1 for v in [a.cpu_threads,a.max_parallel,a.pfn_batch,a.boost_batch,a.local_batch,a.local_checkpoint_rows]):p.error('Thread, parallelism and batch settings must be positive')
    if not a.synthetic_smoke and (a.data is None or not a.data.is_file()):p.error('--data must point to the existing original PKL')
    os.environ.update(environment(a.gpu,a.cpu_threads))
    import torch
    if not torch.cuda.is_available():p.error('CUDA is required. Use scripts/setup_exp61_env.py and its prefix/bin/python')
    if torch.cuda.get_device_capability(0)[0]<8:p.error('A100/Ampere or newer GPU required')
    packages={name:importlib.metadata.version(name) for name in ['torch','tabpfn','numpy','pandas','scikit-learn','scipy','xgboost','faiss-cpu','tensorboard','psutil']}
    ref=read_record(ROOT/'tabpfn/configs/exp61/reference.json')
    boost=ROOT/'tabpfn/third_party/BoostPFN';local=ROOT/'tabpfn/third_party/LoCalPFN/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'
    required={}
    if 'distpfn' in a.methods:required[a.model_path.resolve()]=ref['v3_checkpoint_sha256']
    if 'boostpfn' in a.methods:required[boost/'models_diff/prior_diff_real_checkpoint_n_0_epoch_100.cpkt']=ref['v1_checkpoint_sha256']
    if 'localpfn' in a.methods:required[local]=ref['v1_checkpoint_sha256']
    for path,digest in required.items():
        if not path.is_file():p.error(f'Missing {path}. For v1 weights: bash tabpfn/third_party/fetch_checkpoints.sh')
        if sha(path)!=digest:p.error(f'Checkpoint hash mismatch: {path}')
    root=a.root.resolve()
    request=dict(root=str(root),data=str(a.data.resolve()) if a.data else None,checkpoint=str(a.model_path.resolve()),clean_root=str(a.clean_root.resolve()),
        boost_weights_root=str(boost),local_checkpoint=str(local),datasets=list(dict.fromkeys(a.datasets)),methods=list(dict.fromkeys(a.methods)),seed=43,
        gpu=a.gpu,cpu_threads=a.cpu_threads,pfn_batch=a.pfn_batch,boost_batch=a.boost_batch,local_batch=a.local_batch,local_checkpoint_rows=a.local_checkpoint_rows,
        boost_rounds=2 if a.synthetic_smoke else 50,boost_samples=50 if a.synthetic_smoke else 500,synthetic_smoke=a.synthetic_smoke,prepare_only=a.prepare_only)
    if a.max_parallel>1:request.update(max_parallel=a.max_parallel,execution_mode='concurrent_shared_gpu')
    if (root/'request.json').exists():
        prior=read_record(root/'request.json')
        if prior!=request:p.error('Existing root has different settings; use a new --root')
    else:
        root.mkdir(parents=True,exist_ok=True);snapshot(root/'source');write(root/'request.json',request)
        write(root/'environment.json',dict(python=sys.version,packages=packages,gpu=torch.cuda.get_device_name(0),vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30,cuda=torch.version.cuda,
            driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),cpu_threads=a.cpu_threads,
            checkpoint_hashes={str(k):v for k,v in required.items()},created_epoch=time.time()))
    command=[sys.executable,'-u',str(snapshot_path(root / 'source', 'scripts/run_exp61_sota.py')),'--run-prepared',str(root)]
    if a.detach:
        with (root/'controller.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=root/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment(a.gpu,a.cpu_threads))
        write(root/'launch.json',dict(pid=proc.pid,command=command,launched_epoch=time.time()))
        print(json.dumps(dict(pid=proc.pid,root=str(root),status=str(root/'status.json'),log=str(root/'controller.log')),indent=2))
    else:raise SystemExit(subprocess.call(command,cwd=root/'source',env=environment(a.gpu,a.cpu_threads)))


if __name__=='__main__':main()

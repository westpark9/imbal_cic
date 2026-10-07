#!/usr/bin/env python3
"""Durable two-worker K sweep; resume cached work and serialize OOM retries."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from exp61_utils import write
from run_exp61_sota import environment,snapshot

PROJECT=repo_root(__file__)
DEFAULT=PROJECT/'tabpfn/results/v4/exp63/20260930_exp63_k_sweep_s43'


def aggregate(root,request):
    for source,target in [('summary.csv','summary.csv'),('class_metrics.csv','class_metrics.csv'),('diagnostics/expert_summary.csv','expert_summary.csv'),('diagnostics/expert_class_metrics.csv','expert_class_metrics.csv')]:
        rows=[]
        for k in request['ks']:
            for ds in request['datasets']:
                out=root/f'{ds}_k{k}';path=out/source
                if not (out/'COMPLETE.json').exists() or not path.exists():continue
                with path.open() as f:
                    rows.extend(dict(dataset=ds,K=k,**{a:b for a,b in row.items() if a not in ['dataset','K']}) for row in csv.DictReader(f))
        if rows:
            tmp=root/(target+'.tmp')
            with tmp.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            tmp.replace(root/target)


def run(root):
    request=read_record(root/'request.json');active={};done=[];failed=[];retry=[];started=time.time()
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Reused K=4 diagnostics complete quickly and validate the full path
        # before new K=2 banks, followed by K=6 and K=8.
        order=([4] if 4 in request['ks'] else [])+[k for k in request['ks'] if k!=4]
        queue=[]
        for k in order:
            for ds in request['datasets']:
                job=f'{ds}_k{k}'
                if (root/job/'COMPLETE.json').exists():done.append(job)
                else:queue.append((ds,k,False))
        aggregate(root,request)
        while queue or active or retry:
            if not queue and not active and retry:queue=retry;retry=[]
            limit=1 if (queue and queue[0][2]) or any(v['serial'] for v in active.values()) else request['max_parallel']
            while queue and len(active)<limit:
                ds,k,serial=queue.pop(0);job=f'{ds}_k{k}';out=root/job;out.mkdir(exist_ok=True)
                attempt=len(list(out.glob('attempt_*.log')))+1;log=(out/f'attempt_{attempt}.log').open('w')
                cmd=[sys.executable,'-u',str(snapshot_path(root / 'source', 'tabpfn/scripts/exp63_k_sweep.py')),'--project',request['project'],'--root',str(root),'--dataset',ds,'--k',str(k),'--threads',str(request['threads']),'--batch',str(request['batch'])]
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment('0',request['threads']),cwd=request['project'])
                active[job]=dict(proc=proc,log=log,serial=serial,attempt=attempt,started=time.time(),dataset=ds,K=k,command=cmd)
                print(f'START {job} pid={proc.pid} attempt={attempt}',flush=True)
            details=[]
            for job,item in list(active.items()):
                proc=item['proc'];out=root/job
                if proc.poll() is None:
                    progress=out/'progress.json'
                    details.append(dict(job=job,pid=proc.pid,attempt=item['attempt'],elapsed_seconds=time.time()-item['started'],progress=read_record(progress) if progress.exists() else {}));continue
                item['log'].close();write(out/f'attempt_{item["attempt"]}.json',dict(returncode=proc.returncode,seconds=time.time()-item['started'],command=item['command']))
                if proc.returncode==0 and (out/'COMPLETE.json').exists():done.append(job)
                else:
                    error=(out/'ERROR.json').read_text() if (out/'ERROR.json').exists() else ''
                    if not item['serial'] and ('out of memory' in error.lower() or proc.returncode==-9):retry.append((item['dataset'],item['K'],True))
                    else:failed.append(dict(job=job,returncode=proc.returncode,log=str(out/f'attempt_{item["attempt"]}.log')))
                del active[job];aggregate(root,request)
                print(f'END {job} returncode={proc.returncode}',flush=True)
            state='running' if queue or active or retry else ('complete' if not failed else 'completed_with_failures')
            write(root/'status.json',dict(state=state,pid=os.getpid(),active=details,pending=[f'{ds}_k{k}' for ds,k,_ in queue],completed=done,failed=failed,serial_retries=[f'{ds}_k{k}' for ds,k,_ in retry],updated_epoch=time.time(),elapsed_seconds=time.time()-started))
            if active:time.sleep(5)
        if not failed:write(root/'COMPLETE.json',dict(completed=done,seconds=time.time()-started))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--ks',type=int,nargs='+',default=[2,4,6,8]);p.add_argument('--max-parallel',type=int,default=2);p.add_argument('--threads',type=int,default=6);p.add_argument('--run-prepared',type=Path);a=p.parse_args()
    if a.run_prepared:return run(a.run_prepared.resolve())
    if sorted(set(a.ks))!=sorted(a.ks) or not set(a.ks)<={2,4,6,8}:raise ValueError('K must be distinct members of 2,4,6,8')
    if not 1<=a.max_parallel<=2:raise ValueError('Use one or two workers on the local GPU')
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    # Prevent launching a duplicate controller before printing a misleading PID.
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This sweep already has a live controller')
    if not (root/'request.json').exists():
        snapshot(root/'source')
        write(root/'request.json',dict(project=str(PROJECT),root=str(root),seed=43,ks=a.ks,datasets=['cic2018','toniot'],policy='unchanged EXP62 S1V0 including validation threshold search',context='common anchor + residual-weighted clustering and original diversity-selected block',per_expert_block_cap=186000,global_anchor_and_splits_frozen=True,K4_reused=True,max_parallel=a.max_parallel,threads=a.threads,batch=65536,created_epoch=time.time(),metrics=['full-test and validation direct expert per-class P/R/F1 and FP/FN','global/fixed-route/S+V performance','logical calls/accepts/helpful/harmful','context composition and measured offline preparation cost'],test_role='previously observed development holdout; no test-based K or threshold selection'))
    else:
        request=read_record(root/'request.json')
        assert request['ks']==a.ks,'Existing request differs; use its original --ks or a new root'
    with (root/'controller.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(snapshot_path(root / 'source', 'scripts/run_exp63_local.py')),'--run-prepared',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment('0',a.threads),start_new_session=True,cwd=PROJECT)
    write(root/'launch.json',dict(pid=proc.pid,launched_epoch=time.time()));print(json.dumps(dict(pid=proc.pid,root=str(root)),indent=2))


if __name__=='__main__':main()

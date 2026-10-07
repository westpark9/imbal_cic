#!/usr/bin/env python3
"""Detached two-dataset S/V controller with immutable code snapshot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from exp61_utils import write
from run_exp61_sota import environment,snapshot

PROJECT=Path(__file__).resolve().parents[1]

def run(root):
    request=json.loads((root/'request.json').read_text());active={};done=[];failed=[];retry=[];started=time.time()
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue=[(d,False) for d in ['cic2018','toniot'] if not (root/d/'COMPLETE.json').exists()]
        done=[d for d in ['cic2018','toniot'] if (root/d/'COMPLETE.json').exists()]
        while queue or active or retry:
            if not queue and not active and retry:queue=retry;retry=[]
            limit=1 if queue and queue[0][1] else 2
            while queue and len(active)<limit:
                ds,serial=queue.pop(0);out=root/ds;out.mkdir(exist_ok=True);attempt=len(list(out.glob('attempt_*.log')))+1
                log=(out/f'attempt_{attempt}.log').open('w')
                cmd=[sys.executable,'-u',str(root/'source/tabpfn/scripts/exp62_sv_current_bank.py'),'--project',request['project'],'--root',str(root),'--dataset',ds,'--threads','6','--batch','65536']
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment('0',6),cwd=request['project'])
                active[ds]=dict(proc=proc,log=log,serial=serial,attempt=attempt,started=time.time())
                print(f'START {ds} pid={proc.pid} attempt={attempt}',flush=True)
            details=[]
            for ds,item in list(active.items()):
                p=item['proc'];out=root/ds
                if p.poll() is None:
                    progress=out/'progress.json';details.append(dict(dataset=ds,pid=p.pid,attempt=item['attempt'],progress=json.loads(progress.read_text()) if progress.exists() else {}));continue
                item['log'].close();write(out/f'attempt_{item["attempt"]}.json',dict(returncode=p.returncode,seconds=time.time()-item['started']))
                if p.returncode==0 and (out/'COMPLETE.json').exists():done.append(ds)
                else:
                    error=(out/'ERROR.json').read_text() if (out/'ERROR.json').exists() else ''
                    if not item['serial'] and ('out of memory' in error.lower() or p.returncode==-9):retry.append((ds,True))
                    else:failed.append(dict(dataset=ds,returncode=p.returncode))
                del active[ds];print(f'END {ds} returncode={p.returncode}',flush=True)
            state='running' if queue or active or retry else ('complete' if not failed else 'completed_with_failures')
            write(root/'status.json',dict(state=state,pid=os.getpid(),active=details,completed=done,failed=failed,serial_retries=retry,updated_epoch=time.time(),elapsed_seconds=time.time()-started))
            if active:time.sleep(5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=PROJECT/'tabpfn/results/20260930_exp62_sv_current_bank_s43');p.add_argument('--run-prepared',type=Path);a=p.parse_args()
    if a.run_prepared:return run(a.run_prepared.resolve())
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if not (root/'request.json').exists():
        snapshot(root/'source')
        write(root/'request.json',dict(project=str(PROJECT),root=str(root),seed=43,K=4,policy='s1v0',datasets=['cic2018','toniot'],reuse='frozen EXP59 context IDs, Global fitted state, test probabilities and PCA',new_work='route/cal probabilities, affinity descriptors, direct decision scorer, NLL verifier, validation-only thresholds',max_parallel=2,batch=65536,created_epoch=time.time()))
    with (root/'controller.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(root/'source/scripts/run_exp62_local.py'),'--run-prepared',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment('0',6),start_new_session=True,cwd=PROJECT)
    write(root/'launch.json',dict(pid=proc.pid,launched_epoch=time.time()));print(json.dumps(dict(pid=proc.pid,root=str(root)),indent=2))

if __name__=='__main__':main()

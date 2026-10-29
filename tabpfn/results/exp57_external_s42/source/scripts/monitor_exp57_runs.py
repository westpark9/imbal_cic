#!/usr/bin/env python3
"""Notify the local user and recover failed EXP57 jobs after the original GPU queue.

No chat/email integration: notifications are delivered to the current user's
desktop session and also saved to JSONL. Original source snapshots stay intact.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def write(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


class Monitor:
    def __init__(self,original,recoveries):
        self.original=original;self.recoveries=recoveries;self.seen=set();self.last=time.time();self.child=None
        self.env={**os.environ,'DBUS_SESSION_BUS_ADDRESS':f'unix:path=/run/user/{os.getuid()}/bus'}
        self.jobs=read(original/'protocol.json')['jobs'];self.state={'pid':os.getpid(),'stage':'watching_original','recovery_roots':[str(x) for x in recoveries]}
        self.external=read(original/'external_assignments.json',{}).get('jobs',[])
        self.save()

    def save(self,**extra):
        self.state.update(extra,updated_epoch=time.time());write(self.original/'monitor_status.json',self.state)

    def notify(self,key,title,body,urgency='normal'):
        if key in self.seen:return
        self.seen.add(key);event=dict(epoch=time.time(),key=key,title=title,body=body)
        try:
            p=subprocess.run(['notify-send','--app-name=EXP57','--urgency='+urgency,title,body],env=self.env,capture_output=True,text=True,timeout=10)
            event.update(notification_exit_code=p.returncode,notification_error=p.stderr)
        except Exception as e:event['notification_error']=str(e)
        with (self.original/'notifications.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
        print(json.dumps(event,ensure_ascii=False),flush=True)

    def inspect(self,root,job):
        path=root/job;rows=[]
        try:
            with (path/'summary.csv').open() as f:rows=list(csv.DictReader(f))
        except FileNotFoundError:pass
        models={r['model'] for r in rows if 'model' in r}
        for arm,kr in [('designed','현재 방식'),('matched_random','클래스별 행 수를 맞춘 무작위'),('balanced_random','같은 크기의 균형 추출')]:
            if arm+'_region_constant' in models:
                self.notify(str(path)+arm,'EXP57 bank 평가 완료',f'{job}: {kr} bank 평가를 완료했습니다. 전체 작업은 계속 진행합니다.')
        progress=read(path/'progress.json',{})
        self.save(active_job=job,active_root=str(root),progress=progress,model_summary_rows=len(rows))
        if time.time()-self.last>=1800:
            self.last=time.time();self.notify('heartbeat'+str(int(self.last)),'EXP57 중간 진행',f'{job}: {progress.get("phase","준비 중")} / {progress.get("model",progress.get("arm",""))}. 평가 요약 {len(rows)}/35개 저장.')

    def original_queue(self):
        while True:
            s=read(self.original/'status.json',{})
            for job in s.get('completed',[]):self.notify('original_done_'+job,'EXP57 데이터·seed 완료',job+'의 세 bank 평가가 완료됐습니다.')
            for job in s.get('failed',[]):self.notify('original_failed_'+job,'EXP57 종료·복구 대기',job+'가 비정상 종료됐습니다. '+('외부 GPU 실행으로 배정했습니다.' if job in self.external else '원래 GPU 큐 종료 후 작은 예측 배치와 단일 CPU 스레드로 재시도합니다.'))
            job=s.get('active_job')
            if job:self.inspect(self.original,job)
            if s.get('state') in ['complete','complete_with_errors','stopped']:break
            pid=s.get('controller_pid')
            if pid and not Path('/proc') .joinpath(str(pid)).exists():
                self.notify('controller_missing','EXP57 제어 프로세스 종료','완료되지 않은 작업을 복구 큐로 옮깁니다.');break
            time.sleep(20)

    def recover(self):
        self.save(stage='recovery')
        for job in self.jobs:
            if job in self.external:continue
            if (self.original/job/'COMPLETE.json').exists():continue
            for root in self.recoveries:
                if (root/job/'COMPLETE.json').exists():break
                job_config=read(root/job/'job.json');batch=job_config['args']['test_batch_size']
                self.notify(str(root)+job+'start','EXP57 복구 실행',f'{job}: 예측 배치 {batch:,}, CPU 스레드 1로 재실행합니다.')
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
                     'MALLOC_ARENA_MAX':'2','CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                     'PYTHONFAULTHANDLER':'1','PYTHONUNBUFFERED':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
                command=[sys.executable,'-u',str(root/'source/tabpfn/scripts/exp57_expert_quality_suite.py'),'--root',str(root),'--stage','worker','--job',job]
                with (root/job/'worker.log').open('a') as log:
                    self.child=subprocess.Popen(command,cwd=root/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    self.save(worker_pid=self.child.pid,active_job=job,active_root=str(root))
                    while self.child.poll() is None:self.inspect(root,job);time.sleep(20)
                    code=self.child.returncode
                write(root/job/'EXIT.json',dict(code=code,epoch=time.time(),batch_size=batch,cpu_threads=1))
                if code==0 and (root/job/'COMPLETE.json').exists():
                    self.notify(str(root)+job+'done','EXP57 복구 완료',job+'의 세 bank 평가가 완료됐습니다.');break
                self.notify(str(root)+job+'failed','EXP57 복구 실행 오류',f'{job}: 종료 코드 {code}. '+('다음 작은 배치로 재시도합니다.' if root!=self.recoveries[-1] else '추가 원인 확인이 필요합니다.'))

    def finish(self):
        import pandas as pd
        selected={};failed=[];pending=[];frames=[]
        for job in self.jobs:
            root=next((r for r in [self.original]+self.recoveries if (r/job/'COMPLETE.json').exists()),None)
            if root is None:
                (pending if job in self.external else failed).append(job);continue
            selected[job]=str(root/job);d=pd.read_csv(root/job/'summary.csv');d['job']=job;d['dataset']=job.rsplit('_s',1)[0];d['seed']=int(job.rsplit('_s',1)[1]);frames.append(d)
        if frames:
            table=pd.concat(frames);table.to_csv(self.original/'recovered_all_seed_results.csv',index=False)
            table.groupby(['dataset','model']).macro_f1.agg(['count','mean','std','min','max']).to_csv(self.original/'recovered_seed_summary.csv')
        self.save(stage='complete_with_errors' if failed else ('waiting_external' if pending else 'complete'),selected_result_roots=selected,failed=failed,pending_external=pending)
        self.notify('suite_finished','EXP57 로컬 큐 종료',f'{len(selected)}/6 데이터·seed 완료. 외부 결과 대기: {", ".join(pending) or "없음"}. '+('실패: '+', '.join(failed) if failed else '로컬 seed 요약을 저장했습니다.'))

    def stop(self,signum,frame):
        if self.child is not None and self.child.poll() is None:self.child.terminate()
        self.save(stage='stopped',signal=signum);raise SystemExit(128+signum)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',required=True,type=Path);p.add_argument('--recovery',required=True,type=Path,nargs='+');a=p.parse_args()
    m=Monitor(a.original.resolve(),[r.resolve() for r in a.recovery]);signal.signal(signal.SIGTERM,m.stop);signal.signal(signal.SIGINT,m.stop)
    m.notify('monitor_started','EXP57 진행 알림 시작','Bank·데이터·seed 완료, 오류·재시작, 30분 간격 진행 상황을 이 데스크톱으로 알립니다.')
    m.original_queue();m.recover();m.finish()

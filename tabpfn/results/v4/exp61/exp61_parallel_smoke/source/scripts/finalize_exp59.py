#!/usr/bin/env python3
"""Publish finished datasets as they arrive, then finish the combined report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tabpfn/scripts'))
from exp59_residual_membership_oracle import write,sha


def publish(root):
    state=root/'report_status.json'
    write(state,dict(state='building_report',pid=os.getpid()))
    subprocess.run([sys.executable,'scripts/build_exp59_report.py','--available','--integrate'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/sync_html_reports.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/sync_html_reports.py','--check'],cwd=ROOT,check=True)
    write(state,dict(state='browser_checks',pid=os.getpid()))
    subprocess.run([sys.executable,'scripts/check_exp59_report.py','--out',str(root/'report_qa')],cwd=ROOT,check=True)
    p=ROOT/'docs/research/20260928/exp59_report_manifest.json'
    report=json.loads(p.read_text());report['validation'].update(
        integrated_sync_check=True,browser_checks=json.loads((root/'report_qa/COMPLETE.json').read_text()))
    report['integrated_sha256']=sha(ROOT/'lablog/html_report/post_tabpfn.html')
    report['publishing']['local_html']='complete and browser-checked'
    write(p,report)
    return report


def finish(root):
    state=root/'report_status.json';started=time.time();last=None
    while True:
        paths=[]
        for ds in ['cic2018','toniot']:
            p=root/f'{ds}_s43/COMPLETE.json'
            if not p.exists():p=root/f'{ds}_s43/partial/COMPLETE.json'
            if p.exists():paths.append(p)
        signature=tuple((str(p),p.stat().st_mtime_ns) for p in paths)
        complete=all((root/f'{ds}_s43/COMPLETE.json').exists() for ds in ['cic2018','toniot'])
        if signature and signature!=last:
            report=publish(root);last=signature
            write(state,dict(state='complete' if complete else 'partial_published_waiting_for_experiments',
                pid=os.getpid(),updated_epoch=time.time(),total_seconds=time.time()-started,
                scope=report['scope'],local_report='lablog/html_report/scorer_verifier_target_0928.html',
                integrated_report='lablog/html_report/post_tabpfn.html',git_push='not performed',
                online_artifact='not updated: no connected Claude artifact editor'))
        elif not signature:
            write(state,dict(state='waiting_for_experiments',pid=os.getpid(),started_epoch=started))
        if complete:return
        status=json.loads((root/'status.json').read_text())
        if status['state']=='needs_recovery':raise RuntimeError(f'Experiment requires recovery: {status}')
        pid=status.get('controller_pid')
        if pid:
            try:os.kill(pid,0)
            except ProcessLookupError:raise RuntimeError('Controller exited before both banks completed')
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:finish(args.root.resolve())
    except BaseException:
        write(args.root/'report_status.json',dict(state='needs_recovery',traceback=traceback.format_exc()))
        raise

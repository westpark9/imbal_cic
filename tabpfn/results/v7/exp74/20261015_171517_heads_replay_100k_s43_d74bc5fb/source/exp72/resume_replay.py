#!/usr/bin/env python3
"""Resume a previously registered EXP72 run, preserving its run ID and audit."""
from replay_protocol import ROOT
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from exp61_utils import sha,write
from exp69_class_arrival import utc
from results_versions import Registry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reason',required=True)
    a=parser.parse_args();out=a.out.resolve()
    meta=json.loads((out/'RUN_META.json').read_text())
    if meta['experiment']!='exp72' or meta['version']!='v7':raise ValueError('Wrong registered experiment')
    status=json.loads((out/'status.json').read_text())
    if status['state']=='running':raise ValueError('Refusing to duplicate a running controller')
    command=[sys.executable,'-u',str(Path(__file__).resolve().with_name('run_replay.py')),'run','--out',str(out)]
    attempt=len(meta.get('resumes',[]))+1
    snapshot=out/'source_revisions'/f'resume_{attempt}'
    snapshot.mkdir(parents=True,exist_ok=True)
    for f in Path(__file__).resolve().parent.glob('*.py'):shutil.copyfile(f,snapshot/f.name)
    record=dict(attempt=attempt,reason=a.reason,started=utc(),command=command,
                source_files={f.name:sha(f) for f in snapshot.glob('*.py')})
    meta.setdefault('resumes',[]).append(record);meta['status']='running'
    meta.pop('finished',None);meta.pop('returncode',None)
    write(out/'RUN_META.json',meta)
    log=out/f'resume_{attempt}.log'
    with log.open('a') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        record['controller_pid']=p.pid;write(out/'RUN_META.json',meta)
        rc=p.wait()
    record.update(finished=utc(),returncode=rc)
    meta.update(status='finished' if rc==0 else 'interrupted' if rc==75 else 'failed',returncode=rc,finished=utc())
    write(out/'RUN_META.json',meta)
    reg=Registry(ROOT)
    with reg.lock():reg.index(reg.load())
    return rc

if __name__=='__main__':raise SystemExit(main())

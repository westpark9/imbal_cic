#!/usr/bin/env python3
"""Resume unchanged EXP60 workers, track liveness, and notify the local desktop."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = repo_root(__file__)
sys.path.insert(0, str(ROOT / 'tabpfn/scripts'))
from exp59_residual_membership_oracle import write
from finalize_exp60 import render


def read(path):
    return read_record(path) if path.exists() else {}


def notify(root, key, title, body, critical=False):
    path = root / 'notifications.json'
    history = read(path)
    if history.get(key, {}).get('delivered'):
        return
    try:
        result = subprocess.run(
            ['notify-send', '--app-name=상대사례 실험', '--urgency=' + ('critical' if critical else 'normal'),
             '--expire-time=0', '--print-id', title, body],
            capture_output=True, text=True, timeout=15,
        )
        history[key] = dict(delivered=result.returncode == 0, notification_id=result.stdout.strip(),
                            error=result.stderr.strip(), epoch=time.time(), title=title, body=body)
    except Exception as e:
        history[key] = dict(delivered=False, error=str(e), epoch=time.time())
    write(path, history)


def run(root):
    lock = (root / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    worker = script_path('exp60_counterexamples.py')
    recovery = read(root / 'recovery_history.json')
    sessions = recovery.setdefault('sessions', [])
    session = dict(started_epoch=time.time(), supervisor_pid=os.getpid(), previous_status=read(root/'status.json'),
                   worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(), attempts=[],
                   numerical_protocol_changed=False, completed_models_reused=True)
    sessions.append(session)
    write(root/'recovery_history.json', recovery)
    completed = []
    active = None
    try:
        for ds in ['cic2018', 'toniot']:
            out = root / f'{ds}_s43'
            out.mkdir(exist_ok=True)
            for attempt in range(1, 4):
                if (out/'COMPLETE.json').exists():
                    break
                env = {**os.environ, 'OMP_NUM_THREADS':'16', 'MKL_NUM_THREADS':'16',
                       'OPENBLAS_NUM_THREADS':'16', 'NUMEXPR_NUM_THREADS':'16',
                       'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'PYTHONUNBUFFERED':'1',
                       'PYTHONFAULTHANDLER':'1', 'CUDA_VISIBLE_DEVICES':'0'}
                record = dict(dataset=ds, attempt=attempt, started_epoch=time.time())
                session['attempts'].append(record)
                with (out/'worker.log').open('a') as log:
                    log.write(f'\nSUPERVISOR resume dataset={ds} attempt={attempt} time={time.time()}\n')
                    log.flush()
                    active = subprocess.Popen([sys.executable, '-u', str(worker), '--stage', 'worker',
                                               '--root', str(root), '--dataset', ds],
                                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                    record['worker_pid'] = active.pid
                    write(root/'recovery_history.json', recovery)
                    while active.poll() is None:
                        write(root/'status.json', dict(state='running', active_job=ds,
                              controller_pid=os.getpid(), worker_pid=active.pid, worker_alive=True,
                              attempt=attempt, completed=completed, heartbeat_epoch=time.time(),
                              started_epoch=session['started_epoch']))
                        time.sleep(10)
                    code = active.returncode
                    active = None
                record.update(exit_code=code, ended_epoch=time.time())
                write(root/'recovery_history.json', recovery)
                if code == 0:
                    if not (out/'COMPLETE.json').exists():
                        raise RuntimeError(f'{ds} exited without a completion marker')
                    break
                if code not in [-11, -6] or attempt == 3:
                    raise RuntimeError(f'{ds} worker exited {code}, attempt {attempt}')
                # Native crash only: fresh process, unchanged model and batch settings.
                write(root/'status.json', dict(state='recovering', active_job=ds,
                      controller_pid=os.getpid(), completed=completed, exit_code=code,
                      next_attempt=attempt+1, heartbeat_epoch=time.time()))
            completed.append(ds)
            report = render(root, completed)
            write(root/'readout_status.json', dict(state='complete' if len(completed)==2 else 'partial_complete',
                  completed=completed, report=report, html_modified=False, updated_epoch=time.time()))
            if len(completed) == 1:
                notify(root, ds, 'CIC2018 상대사례 실험 완료',
                       '무작위·가까운 상대사례 8개 expert 완료. ToN 실험을 이어서 실행합니다.')
        session['completed_epoch'] = time.time()
        write(root/'recovery_history.json', recovery)
        write(root/'status.json', dict(state='complete', completed=completed, worker_alive=False,
              seconds=time.time()-session['started_epoch'], completed_epoch=time.time()))
        notify(root, 'all_complete', '상대사례 실험 전체 완료',
               'CIC2018·ToN, 무작위·가까운 상대사례 16개 expert와 분석 완료\n'
               '결과: docs/research/20260929/exp60_results.md')
    except BaseException:
        if active is not None and active.poll() is None:
            active.terminate()
            active.wait(timeout=30)
        write(root/'status.json', dict(state='needs_recovery', completed=completed,
              controller_pid=os.getpid(), worker_alive=False, updated_epoch=time.time(),
              traceback=traceback.format_exc()))
        notify(root, f'failure_{os.getpid()}', '상대사례 실험 중단',
               '자동 복구 범위를 넘어 실험이 중단됐습니다. status.json과 worker.log 확인 필요', critical=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root.resolve())

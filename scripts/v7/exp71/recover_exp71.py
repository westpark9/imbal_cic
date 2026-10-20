#!/usr/bin/env python3
"""Recover EXP71 after reboot without overwriting the interrupted attempt."""
from pathlib import Path
import argparse
import importlib.metadata as im
import json
import platform
import shutil
import subprocess
import sys

from exp71_bounded_updates import ROOT, STAGES, CASES, sha, write, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    out, parent = args.out.resolve(), args.resume_from.resolve()
    if (out / 'request.json').exists():
        raise ValueError('Recovery must use a new managed run directory')
    request = json.loads((parent / 'request.json').read_text())
    assert request['history_mode'] == 'cumulative'
    for saved in (parent / 'source').rglob('*.py'):
        assert sha(saved) == sha(ROOT / saved.relative_to(parent / 'source')), saved
    previous_env = json.loads((parent / 'environment.json').read_text())
    packages = {name: im.version(name) for name in previous_env['packages']}
    assert packages == previous_env['packages'], 'Python packages changed'
    package_root = Path(previous_env['tabpfn_package_root'])
    for name, digest in previous_env['tabpfn_python_sources_sha256'].items():
        assert sha(package_root / name) == digest, name
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable'
    assert (torch.ones(4, device='cuda') + 1).sum().item() == 8
    current_env = dict(previous_env, recorded=utc(), platform=platform.platform(),
                       executable=sys.executable, python=sys.version,
                       gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                                    '--format=csv,noheader'], text=True).strip())
    write(out / 'environment.json', current_env)
    write(out / 'environment_before_reboot.json', previous_env)
    records = []
    for ds in STAGES:
        for case in CASES:
            for method in ['tabpfn', 'xgb']:
                job = f'{ds}_{case}_{method}'
                replay = ds == 'toniot' and case == 'natural'
                if not replay:
                    source = parent / 'jobs' / job
                    for stage in range(len(STAGES[ds])):
                        folder = source / f'stage_{stage}'
                        marker = json.loads((folder / 'COMPLETE.json').read_text())
                        assert marker['metrics_sha256'] == sha(folder / 'metrics.json'), folder
                        report = json.loads((folder / 'metrics.json').read_text())
                        assert report['history_mode'] == 'cumulative'
                        assert stage == 0 or not report['pfn_cold_fit']
                    shutil.copytree(source, out / 'jobs' / job)
                records.append(dict(job=job, action='replay_full_worker' if replay else 'reuse_complete_worker',
                                    source_run=None if replay else str(parent),
                                    environment_file='environment.json' if replay else 'environment_before_reboot.json'))
    write(out / 'recovery.json', dict(parent_run=str(parent), recorded=utc(), jobs=records,
          reason='Host reboot and missing NVIDIA kernel module; original attempt retained unchanged',
          replay_reason='Restart complete ToN natural workers to preserve initial cold / later warm update accounting',
          cross_policy_environment_matched=False,
          cost_note='ToN natural cumulative used kernel 7.0.0-38 / driver 580.178.04; bounded and reused jobs used kernel 7.0.0-31 / driver 580.173.02. Cross-policy timing differences are not attributable solely to history policy.'))
    dest = out / 'source' / Path(__file__).resolve().relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, dest)
    command = [sys.executable, str(ROOT / 'scripts/v7/exp71/exp71_bounded_updates.py'),
               'run', '--out', str(out), '--stream-root', request['stream_root'],
               '--history-mode', request['history_mode']]
    for key in ['seed', 'nominal', 'history_cap', 'threads', 'batch']:
        command += ['--' + key.replace('_', '-'), str(request[key])]
    print('Reusing 32 complete stages; replaying both ToN natural workers (12 stages).', flush=True)
    return subprocess.call(command)


if __name__ == '__main__':
    sys.exit(main())

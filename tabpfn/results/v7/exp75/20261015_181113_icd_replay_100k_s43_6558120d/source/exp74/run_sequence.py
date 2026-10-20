#!/usr/bin/env python3
"""Durable managed sequence: EXP74 first, then authorized EXP75."""
from run_adaptation import ROOT, write, utc
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--state', type=Path, required=True)
    a = p.parse_args()
    value = dict(state='running', started=utc(), order=['exp74', 'exp75'], runs=[])
    write(a.state, value)
    for experiment, script, method in [
        ('exp74', 'scripts/v7/exp74/run_adaptation.py', 'heads'),
        ('exp75', 'scripts/v7/exp75/run_context_adaptation.py', 'icd')]:
        command = [sys.executable, '-u', 'scripts/results_versions.py', 'run',
            '--version', 'v7', '--experiment', experiment, '--scope', 'tabpfn',
            '--label', f'{method}_replay_100k_s43', '--protocol', f'docs/research/20261015/{experiment}_protocol.md',
            '--', sys.executable, '-u', script, 'run', '--out', '{run_dir}', '--method', method]
        value.update(current=experiment, command=command, updated=utc()); write(a.state, value)
        result = subprocess.run(command, cwd=ROOT)
        value['runs'].append(dict(experiment=experiment, returncode=result.returncode, finished=utc()))
        if result.returncode:
            value.update(state='paused' if result.returncode == 75 else 'failed', updated=utc())
            write(a.state, value); return result.returncode
    value.update(state='complete', completed=utc()); write(a.state, value)
    return 0


if __name__ == '__main__': raise SystemExit(main())

"""Resume an existing EXP89 run and retain its interruption and process history."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


def stamp():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    root = next(p for p in Path(__file__).resolve().parents
                if (p / 'scripts/common/experiment_paths.py').is_file())
    with (run / 'resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'COMPLETE.json').exists():
            raise RuntimeError('This run is already complete.')
        environment = read(run / 'environment.json')
        actual = {name: importlib.metadata.version(name) for name in environment}
        if actual != environment:
            raise RuntimeError(f'Environment mismatch: {actual}')
        source_checks = {}
        for relative in ['scripts/v7/exp89/curation_benign.py',
                         'configs/experiments/exp89_protocol.json',
                         'scripts/common/experiment_paths.py',
                         'scripts/v7/exp71/exp71_bounded_updates.py',
                         'scripts/v7/exp72/replay_models.py',
                         'scripts/v7/exp77/finetune_final.py']:
            saved = run / 'source_dependencies' / relative
            current = root / relative
            source_checks[relative] = hashlib.sha256(current.read_bytes()).hexdigest()
            if current.read_bytes() != saved.read_bytes():
                raise RuntimeError(f'Original experiment source changed: {relative}')
        started = stamp()
        history = run / 'resumptions' / datetime.now().strftime('%Y%m%d_%H%M%S')
        history.mkdir(parents=True)
        for name in ['RUN_META.json', 'progress.json', 'results.json',
                     'environment.json', 'report_pipeline.json', 'FAILURE.json']:
            if (run / name).exists():
                shutil.copy2(run / name, history / name)
        for job in (run / 'jobs').iterdir():
            if job.is_dir() and not (job / 'COMPLETE.json').exists():
                shutil.copytree(job, history / 'incomplete_jobs' / job.name)
        if (run / 'FAILURE.json').exists():
            (run / 'FAILURE.json').unlink()
        completed = len(list((run / 'jobs').glob('*/COMPLETE.json')))
        entry = dict(resumed_at=started, status='running', completed_before=completed,
                     expected_jobs=read(run / 'config.json')['design']['expected_jobs'],
                     previous_progress=read(history / 'progress.json'),
                     boot_time=subprocess.check_output(['uptime', '-s'], text=True).strip(),
                     reason='User requested resumption after background execution stopped; prior process absent.',
                     supervisor_pid=os.getpid(), environment_unchanged=True,
                     source_sha256=source_checks, history_path=str(history),
                     resume_policy='Verify completed prediction hashes and reuse completed jobs; restart incomplete job.')
        command = [sys.executable, '-u', str(root / 'scripts/v7/exp89/curation_benign.py'),
                   '--out', str(run)]
        entry['command'] = command
        meta = read(run / 'RUN_META.json')
        meta.update(status='running', last_resumed_at=started, supervisor_pid=os.getpid())
        meta.setdefault('resumptions', []).append(entry)
        write(run / 'RUN_META.json', meta)
        write(history / 'resume.json', entry)
        code = 1
        try:
            with (run / 'stdout.log').open('a') as log:
                log.write('\n' + json.dumps(dict(phase='resume', **entry), ensure_ascii=False) + '\n')
                log.flush()
                process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, EXPERIMENT_RUN_DIR=str(run)))
                entry['worker_pid'] = process.pid
                meta['worker_pid'] = process.pid
                write(run / 'RUN_META.json', meta)
                write(history / 'resume.json', entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
                code = process.wait()
            entry.update(status='finished' if code == 0 else 'failed',
                         finished_at=stamp(), returncode=code)
        except BaseException as exc:
            entry.update(status='failed', finished_at=stamp(), error=repr(exc), returncode=code)
            raise
        finally:
            meta.update(status=entry['status'], finished=entry.get('finished_at'), returncode=code)
            write(history / 'resume.json', entry)
            write(run / 'RUN_META.json', meta)
            subprocess.run([sys.executable, str(root / 'scripts/results_versions.py'), 'index'],
                           cwd=root, check=False)
        return code


if __name__ == '__main__':
    raise SystemExit(main())

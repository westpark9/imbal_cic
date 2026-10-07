#!/usr/bin/env python3
"""EXP49: clean BoT-IoT, run original EXP31, then existing EXP39 legacy policy."""
import argparse
import ast
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import types

from nfv3_conflict_clean import install_clean_loader, sha256, write_json
from exp49_trace import Trace, instrument

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def verify_sources(root):
    for filename, expected in json.loads((root / 'source_identity.json').read_text()).items():
        if sha256(filename) != expected: raise RuntimeError(f'Source changed during run: {filename}')


def baseline(root, launch):
    import nfv3_v3_common as core
    from threadpoolctl import threadpool_limits
    manifest = str(Path(launch['clean_dir']) / 'manifest.json')
    install_clean_loader(core, manifest)
    args = json.loads((root / 'baseline_args.json').read_text())
    source_path = HERE / 'nfv3_v3_exp31_c0alloc.py'
    source = instrument(source_path.read_text())
    (root / 'source/exp31_instrumented.py').write_text(source)
    module = types.ModuleType('exp49_instrumented_baseline')
    module.__file__ = str(source_path)
    module._exp47_trace = Trace(root)
    sys.modules[module.__name__] = module
    exec(compile(source, str(root / 'source/exp31_instrumented.py'), 'exec'), module.__dict__)
    with threadpool_limits(16): out = module.run_exp29(types.SimpleNamespace(**args))
    write_json(root / 'BASELINE_COMPLETE.json', {'run_dir': out, 'completed_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat()})


def routing(root):
    source_path = HERE / 'nfv3_v3_exp39_decision_routing.py'
    source = source_path.read_text()
    assert source.count('_nfv3_cic2018_exp39_decision_routing') == 1
    source = source.replace('_nfv3_cic2018_exp39_decision_routing', '_nfv3_botiot_exp39_decision_routing')
    (root / 'source/exp39_output_name_only.py').write_text(source)
    exp = types.ModuleType('exp49_legacy_routing')
    exp.__file__ = str(source_path)
    sys.modules[exp.__name__] = exp
    exec(compile(source, str(source_path), 'exec'), exp.__dict__)
    sys.argv = [str(HERE / 'nfv3_v3_exp39_decision_routing.py'), '--cache-dir', str(root / 'frozen_cache'),
                '--out-root', str(root / 'routing'), '--arms', 'legacy', '--threads', '16', '--seed', '42',
                '--protected-classes', 'ddos,dos']
    exp.main()


def audit(root):
    from exp47_audit import run
    run(root)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from exp39_stage_audit import audit as stage_audit
    completed = list((root / 'routing').glob('*/COMPLETE.json'))
    if len(completed) != 1: raise ValueError('Expected one completed legacy routing run')
    stage_audit(completed[0].parent, root / 'frozen_cache', root / 'diagnostics/exp39_legacy')


def controller(root, launch):
    started = time.time()
    phase, child = 'initializing', None
    def stopped(signum, frame):
        raise InterruptedError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM, stopped); signal.signal(signal.SIGINT, stopped)
    try:
        clean = Path(launch['clean_dir'])
        if not (clean / 'COMPLETE.json').exists(): raise ValueError('Cleaning verification is not complete')
        verify_sources(root)
        for phase in ['baseline', 'routing', 'audit']:
            log = root / f'{phase}.log'
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-dir', str(root), '--stage', phase]
            with log.open('w') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                write_json(root / 'RUNNING.json', {'pid': os.getpid(), 'worker_pid': child.pid, 'phase': phase,
                           'started_epoch': started, 'updated_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), 'log': str(log)})
                print(f'STAGE {phase}: pid={child.pid}, log={log}', flush=True)
                code = child.wait()
            if code: raise RuntimeError(f'{phase} exited {code}; see {log}')
        write_json(root / 'COMPLETE.json', {'seconds': time.time() - started, 'completed_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), 'seed': 42})
        (root / 'RUNNING.json').unlink(missing_ok=True)
    except BaseException as exc:
        if child and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        status = 'STOPPED.json' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'ERROR.json'
        write_json(root / status, {'phase': phase, 'error': str(exc), 'traceback': traceback.format_exc(), 'seconds': time.time() - started})
        (root / 'RUNNING.json').unlink(missing_ok=True)
        raise


def prepare_run(root):
    launch = json.loads((root / 'launch.json').read_text())
    source_run = Path(launch['baseline_source_run'])
    args = json.loads((source_run / 'args.json').read_text())
    args.update(target_dataset='bot_iot', clean_manifest=str(Path(launch['clean_dir']) / 'manifest.json'),
                out_root=str(root / 'baseline'), resume_dir=str(root / 'resume'),
                models_dir=str(root / 'models'), dense_eval=True, seed=42)
    for key in ['out_root', 'resume_dir', 'models_dir']: Path(args[key]).mkdir(exist_ok=True)
    (root / 'routing').mkdir(exist_ok=True)
    write_json(root / 'baseline_args.json', args)
    sources = [Path(__file__).resolve(), HERE / 'nfv3_conflict_clean.py', HERE / 'cic2018_conflict_clean.py', HERE / 'exp49_trace.py', HERE / 'exp47_trace.py', HERE / 'exp47_audit.py',
               HERE / 'nfv3_v3_exp31_c0alloc.py', HERE / 'nfv3_v3_exp38_frozen_cache.py',
               HERE / 'nfv3_v3_exp39_decision_routing.py', HERE / 'nfv3_v3_common.py', ROOT / 'scripts/exp_utils.py',
               ROOT / 'scripts/exp39_stage_audit.py', source_run / 'args.json']
    (root / 'source').mkdir(exist_ok=True)
    identity = {}
    for path in sources:
        identity[str(path)] = sha256(path)
        shutil.copy2(path, root / 'source' / (path.name if path.name != 'args.json' else 'parent_args.json'))
    write_json(root / 'source_identity.json', identity)
    write_json(root / 'protocol.json', {'dataset': 'bot_iot', 'seed': 42,
        'parent_args': str(source_run / 'args.json'), 'clean_manifest': args['clean_manifest'],
        'training_configuration_changes': ['target_dataset: cic2018 -> bot_iot', 'dataset tail vocabulary: theft'], 'secondary_protected_classes': ['ddos', 'dos'], 'diagnostic_changes': ['dense_eval=True', 'nine observation callbacks', 'calibration all-expert predictions after baseline test'],
        'baseline_policy': 'EXP31 weighted-NLL constraints, original scorer and normgain verifier',
        'secondary_policy': 'existing EXP39 legacy only; shared macro-F1 calibration/confirmation protocol',
        'context_rule': '100k C0; benign share .75, attack balanced, original scenario-stratified role rules',
        'outer_split': 'original memberships intersected with clean keep indices',
        'old_predictions_reused': False})
    print(json.dumps({'run_dir': str(root), 'prepared': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'controller', 'baseline', 'routing', 'audit'], default='controller')
    args = parser.parse_args(); root = args.run_dir.resolve()
    if args.stage == 'prepare': return prepare_run(root)
    launch = json.loads((root / 'launch.json').read_text())
    verify_sources(root)
    if args.stage == 'controller': controller(root, launch)
    elif args.stage == 'baseline': baseline(root, launch)
    elif args.stage == 'routing': routing(root)
    else: audit(root)


if __name__ == '__main__': main()

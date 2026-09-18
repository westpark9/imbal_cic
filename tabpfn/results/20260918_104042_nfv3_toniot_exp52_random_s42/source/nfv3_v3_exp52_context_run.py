#!/usr/bin/env python3
"""EXP52 run controller: refit ToN global+experts with EXP52's --expert-select-mode
knob (confident/random vs the exp31 default residual), reusing the existing EXP48
clean ToN data (no re-cleaning). Baseline stage only trains; routing stage runs the
unmodified EXP39 legacy arm just to get its full-test expertK_always diagnostic
(no scorer/verifier comparison needed here) -- no audit stage, no calibration
funnel diagnostics; this experiment is about self-class F1 only."""
import argparse
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
from exp48_trace import Trace, instrument

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
    source_path = HERE / 'nfv3_v3_exp52_expert_context.py'
    source = instrument(source_path.read_text())
    (root / 'source/exp52_instrumented.py').write_text(source)
    module = types.ModuleType('exp52_instrumented_baseline')
    module.__file__ = str(source_path)
    module._exp47_trace = Trace(root)
    sys.modules[module.__name__] = module
    exec(compile(source, str(root / 'source/exp52_instrumented.py'), 'exec'), module.__dict__)
    with threadpool_limits(16): out = module.run_exp29(types.SimpleNamespace(**args))
    write_json(root / 'BASELINE_COMPLETE.json', {'run_dir': out, 'completed_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat()})


def routing(root):
    source_path = HERE / 'nfv3_v3_exp39_decision_routing.py'
    source = source_path.read_text()
    assert source.count('_nfv3_cic2018_exp39_decision_routing') == 1
    source = source.replace('_nfv3_cic2018_exp39_decision_routing', '_nfv3_toniot_exp52_decision_routing')
    (root / 'source/exp39_output_name_only.py').write_text(source)
    exp = types.ModuleType('exp52_legacy_routing')
    exp.__file__ = str(source_path)
    sys.modules[exp.__name__] = exp
    exec(compile(source, str(source_path), 'exec'), exp.__dict__)
    sys.argv = [str(HERE / 'nfv3_v3_exp39_decision_routing.py'), '--cache-dir', str(root / 'frozen_cache'),
                '--out-root', str(root / 'routing'), '--arms', 'legacy', '--threads', '16', '--seed', '42',
                '--protected-classes', 'ddos,dos']
    exp.main()


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
        for phase in ['baseline', 'routing']:
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
    args.update(target_dataset='ton_iot', clean_manifest=str(Path(launch['clean_dir']) / 'manifest.json'),
                out_root=str(root / 'baseline'), resume_dir=str(root / 'resume'),
                models_dir=str(root / 'models'), dense_eval=True, seed=42,
                expert_select_mode=launch['expert_select_mode'])
    for key in ['out_root', 'resume_dir', 'models_dir']: Path(args[key]).mkdir(exist_ok=True)
    (root / 'routing').mkdir(exist_ok=True)
    write_json(root / 'baseline_args.json', args)
    sources = [Path(__file__).resolve(), HERE / 'nfv3_conflict_clean.py', HERE / 'cic2018_conflict_clean.py', HERE / 'exp48_trace.py', HERE / 'exp47_trace.py',
               HERE / 'nfv3_v3_exp52_expert_context.py', HERE / 'nfv3_v3_exp38_frozen_cache.py',
               HERE / 'nfv3_v3_exp39_decision_routing.py', HERE / 'nfv3_v3_common.py', ROOT / 'scripts/exp_utils.py',
               source_run / 'args.json']
    (root / 'source').mkdir(exist_ok=True)
    identity = {}
    for path in sources:
        identity[str(path)] = sha256(path)
        shutil.copy2(path, root / 'source' / (path.name if path.name != 'args.json' else 'parent_args.json'))
    write_json(root / 'source_identity.json', identity)
    write_json(root / 'protocol.json', {'dataset': 'ton_iot', 'seed': 42,
        'parent_args': str(source_run / 'args.json'), 'clean_manifest': args['clean_manifest'],
        'training_configuration_changes': ['target_dataset: cic2018 -> ton_iot',
            f"expert_select_mode: residual (exp31 default) -> {launch['expert_select_mode']}"],
        'experiment_purpose': '09-18 self-class specialization paradox diagnosis: does confident-first or '
            'random within-cell expert block selection recover self-class F1, vs the exp31 default '
            'residual (hardest-first) selection that anti-correlated with the block\'s own dominant class?',
        'secondary_protected_classes': ['ddos', 'dos'],
        'diagnostic_changes': ['dense_eval=True (needed for expertK_always full-test diagnostic)'],
        'baseline_policy': 'EXP31 weighted-NLL constraints, unchanged',
        'secondary_policy': 'EXP39 legacy arm only, run to obtain frozen_cache expertK_always diagnostic; '
            'scorer/verifier comparison itself is out of scope for this experiment',
        'context_rule': 'same as EXP48 (100k C0, benign share .75, attack balanced); ONLY expert_select_mode changed',
        'outer_split': 'original memberships intersected with clean keep indices (reused from EXP48, not re-cleaned)',
        'old_predictions_reused': False})
    print(json.dumps({'run_dir': str(root), 'prepared': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'controller', 'baseline', 'routing'], default='controller')
    args = parser.parse_args(); root = args.run_dir.resolve()
    if args.stage == 'prepare': return prepare_run(root)
    launch = json.loads((root / 'launch.json').read_text())
    verify_sources(root)
    if args.stage == 'controller': controller(root, launch)
    elif args.stage == 'baseline': baseline(root, launch)
    else: routing(root)


if __name__ == '__main__': main()

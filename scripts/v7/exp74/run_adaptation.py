#!/usr/bin/env python3
"""EXP74 heads / EXP75 context distillation, on the unchanged EXP72 replay."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents
             if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
import traceback

import numpy as np
from exp61_utils import sha, write
from exp69_class_arrival import utc, array_digest
from scenario_data import stage_data, assign_new_roles, SOURCE
from replay_protocol import STAGES, confusion_metrics
from replay_models import timed
import run_replay as replay
from distribution_heads import embed, fit_heads, HeadsPredictor, HEAD_NAMES
from context_distillation import fit_context, ContextPredictor

PREPARED = ROOT/'tabpfn/results/v7/exp73/20261014_154250_replay_current_sota_100k_s43_b0f8a9dc'
BASELINE = ROOT/'tabpfn/results/v7/exp72/20261014_134735_replay_100k_xgb_tabpfn_s43_de8b4a78'


def defaults(method):
    return dict(method=method, protocol_revision='exp74.1' if method == 'heads' else 'exp75.1',
        methods=['heads'] if method == 'heads' else ['icd', 'icd_init'],
        source=str(SOURCE), prepared=str(PREPARED), baseline=str(BASELINE), seed=43,
        datasets=['cic2018', 'toniot'], cases=['natural', 'uniform'], batch=1024,
        rates=[100, 1000], threads=8, prepare_batch=1024, validation_cap=10000,
        checkpoint=str(ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'),
        head_hidden=128, head_lr=.001, head_epochs=100, head_patience=12, head_batch=512,
        context_budget=2048, icd_steps=300, icd_lr=.005, icd_query_batch=256,
        icd_eval_batch=512, icd_eval_every=50, html_publication=False,
        scientific_status='exploratory architecture adaptation; existing approved research version v7',
        same_unique_label_access='100k original train plus previously approved separate validation')


def progress(job, **values):
    value = dict(pid=os.getpid(), updated=utc(), updated_epoch=time.time(), **values)
    write(job/'progress.json', value); print(json.dumps(value), flush=True)


def prepare(out, config):
    if (out/'PREPARED.json').exists(): return
    source = Path(config['prepared'])
    target = out/'data'
    if not target.exists(): target.symlink_to(source/'data', target_is_directory=True)
    (out/'jobs').mkdir(exist_ok=True)
    manifests = {}
    for ds in config['datasets']:
        base = source/'data'/ds
        meta = json.loads((base/'manifest.json').read_text())
        assert sha(base/'plans.json') == meta['plans_sha256']
        assert sha(base/'order.npy') == meta['order_sha256']
        for name, value in meta['validation_files'].items():
            assert sha(base/f'validation_{name}.npy') == value
        manifests[ds] = sha(base/'manifest.json')
        for case in config['cases']:
            for method in ('xgb', 'tabpfn'):
                target = out/'jobs'/f'{ds}_{case}_{method}'
                if not target.exists():
                    target.symlink_to(Path(config['baseline'])/'jobs'/target.name, target_is_directory=True)
    write(out/'PREPARED.json', dict(source=str(source), manifests=manifests,
        train_budget=100000, test_rows_and_stage_order_unchanged=True,
        future_classes_used=False, complete=True))


def validation_split(out, ds, data, seed):
    # Use only currently available calibration rows; keep identical feature groups together.
    hashes = np.load(out/'data'/ds/'validation_hash.npy', mmap_mode='r')[data['i_cal']]
    split = assign_new_roles(np.arange(len(hashes)), data['y_cal'], hashes, {}, seed,
                             roles=('select', 'temperature'), fractions=(.5, .5))
    a, b = split['select'], split['temperature']
    if np.intersect1d(hashes[a], hashes[b]).size: raise AssertionError('Feature overlap')
    C = len(data['names'])
    for ids in (a, b):
        if set(np.unique(data['y_cal'][ids])) != set(range(C)):
            raise ValueError('Head selection/calibration misses a class')
    return a, b


def worker_fit(out, config, ds, case):
    import torch
    from tabpfn import TabPFNClassifier
    torch.set_num_threads(config['threads'])
    method = config['method']; job = out/'jobs'/f'{ds}_{case}_{method}'; job.mkdir(exist_ok=True)
    for stage in range(len(STAGES[ds])):
        dest = job/f'stage_{stage}'; dest.mkdir(exist_ok=True)
        if (dest/'FIT_COMPLETE.json').exists(): continue
        if (out/'STOP_AFTER_UNIT').exists(): return 75
        data, identity = stage_data(out, config, ds, case, stage)
        C = len(data['names'])
        progress(job, phase='fit', stage=stage, dataset=ds, case=case, method=method,
                 train_rows=len(data['ytr']), validation_rows=len(data['yva']))
        def log(**v): progress(job, phase='fit', stage=stage, dataset=ds, case=case, method=method, **v)
        if method == 'heads':
            src = Path(config['baseline'])/'jobs'/f'{ds}_{case}_tabpfn'/f'stage_{stage}'
            np.testing.assert_array_equal(np.load(src/'train_ids.npy'), identity['train_ids'])
            prior = json.loads((src/'fit_metrics.json').read_text())
            base = TabPFNClassifier.load_from_fit_state(src/'model.tabpfn_fit', device='cuda:0')
            # A shared full 100k context is retained, including every currently released train row.
            select, cal = validation_split(out, ds, data, config['seed']+1901)
            zfit, fit_seconds = timed(lambda: embed(base, data['X_tune'], config['prepare_batch']))
            zrest, cal_seconds = timed(lambda: embed(base, data['X_cal'], config['prepare_batch']))
            log(component='embeddings_complete', features=zfit.shape[1], seconds=fit_seconds+cal_seconds)
            training = fit_heads(zfit, data['y_tune'], zrest[select], data['y_cal'][select],
                zrest[cal], data['y_cal'][cal], np.bincount(data['ytr'], minlength=C), config, dest, log)
            np.savez(dest/'head_role_ids.npz', head_fit=identity['role_ids']['tune'],
                head_selection=identity['role_ids']['cal'][select],
                head_temperature=identity['role_ids']['cal'][cal])
            link = dest/'model.tabpfn_fit'
            if not link.exists(): link.symlink_to(src/'model.tabpfn_fit')
            extra = fit_seconds+cal_seconds+training['work_seconds']
            info = dict(update_seconds=prior['update_seconds']+extra,
                base_update_seconds=prior['update_seconds'], adaptation_seconds=extra,
                embedding_seconds=fit_seconds+cal_seconds, head_fit_seconds=training['work_seconds'],
                base_fit_reused=str(src), global_context_rows=len(data['ytr']),
                head_training_rows=len(data['y_tune']), validation_role_policy='tune=head fit; cal split for selection/temperature',
                target_prior_exponents=[1, 0, -1], prior_from='currently released train only')
            del base, zfit, zrest
        else:
            info = fit_context(data, config, dest, log)
            info['update_seconds'] = info['work_seconds']
            info.update(validation_role_policy='cal=checkpoint selection; no validation rows in training queries',
                        fit_mode='differentiable_input; one estimator; fixed standardized features',
                        baseline_difference='full-context TabPFN has four estimators and its standard preprocessing')
            # Untuned small context is evaluated under its own actual preparation schedule.
            init = out/'jobs'/f'{ds}_{case}_icd_init'/f'stage_{stage}'; init.mkdir(parents=True, exist_ok=True)
            link = init/'context.npz'
            if not link.exists(): link.symlink_to(dest/'context.npz')
            init_info = dict(info, method='icd_init', update_seconds=info['initial_prepare_seconds'])
            init_info.update(stage=stage, case=case, class_names=data['names'],
                global_class_indices=identity['global_class_indices'], train_rows=len(data['ytr']))
            write(init/'fit_metrics.json', init_info)
            np.save(init/'train_ids.npy', identity['train_ids'])
            write(init/'FIT_COMPLETE.json', dict(complete=True, reused_initialization=str(dest)))
        info.update(method=method, dataset=ds, case=case, stage=stage,
            train_rows=len(data['ytr']), validation_rows=len(data['yva']), class_names=data['names'],
            train_class_counts=np.bincount(data['ytr'], minlength=C).tolist(),
            global_class_indices=identity['global_class_indices'], train_ids_sha256=array_digest(identity['train_ids']))
        np.save(dest/'train_ids.npy', identity['train_ids'])
        write(dest/'fit_metrics.json', info)
        write(dest/'FIT_COMPLETE.json', dict(complete=True, metrics_sha256=sha(dest/'fit_metrics.json'), completed=utc()))
        del data; gc.collect(); torch.cuda.empty_cache()
    return 0


def predictor(dest, config, C, method):
    if method == 'heads':
        from tabpfn import TabPFNClassifier
        base = TabPFNClassifier.load_from_fit_state(dest/'model.tabpfn_fit', device='cuda:0')
        return HeadsPredictor(dest, base)
    return ContextPredictor(dest, config, C, 'initial' if method == 'icd_init' else 'tuned')


def memmap(path, shape, dtype, fill):
    if path.exists(): return np.load(path, mmap_mode='r+')
    array = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
    array[:] = fill; array.flush(); return array


def worker_predict(out, config, ds, case, method):
    import torch
    torch.set_num_threads(config['threads'])
    job = out/'jobs'/f'{ds}_{case}_{method}'; prep = out/'data'/ds
    schedules = replay.make_schedules(out, config, ds, case, method)
    batches = np.load(prep/'batches.npy'); order = np.load(prep/'order.npy', mmap_mode='r')
    X = np.load(Path(config['source'])/'data'/ds/'test_X.npy', mmap_mode='r')
    for version in range(len(STAGES[ds])):
        dest = job/f'stage_{version}'
        requested = np.flatnonzero(np.logical_or.reduce([v == version for v in schedules.values()]))
        if (dest/'PREDICT_COMPLETE.json').exists(): continue
        if (out/'STOP_AFTER_UNIT').exists(): return 75
        info = json.loads((dest/'fit_metrics.json').read_text())
        mapping = np.asarray(info['global_class_indices'], dtype='int16')
        model = predictor(dest, config, len(mapping), method)
        pred = memmap(dest/'replay_pred.npy', (len(order),), 'int16', -1)
        seconds = memmap(dest/'batch_seconds.npy', (len(batches),), 'float64', np.nan)
        auxiliary = {n: memmap(dest/(n+'.npy'), (len(order),), 'int16', -1)
                     for n in (HEAD_NAMES if method == 'heads' else ())}
        remaining = [int(i) for i in requested if not np.isfinite(seconds[i])]
        start = time.perf_counter(); rows = 0
        for count, idx in enumerate(remaining, 1):
            if (out/'STOP_AFTER_UNIT').exists(): model.close(); return 75
            _, lo, hi = batches[idx]
            (p, aux), elapsed = timed(lambda: model.predict_with_aux(np.asarray(X[order[lo:hi]])))
            if p.shape != (hi-lo, len(mapping)) or not np.isfinite(p).all():
                raise ValueError('Invalid predicted probabilities')
            np.testing.assert_allclose(p.sum(1), 1., atol=1e-4)
            pred[lo:hi] = mapping[p.argmax(1)]; pred.flush()
            for name, value in aux.items(): auxiliary[name][lo:hi] = mapping[value]; auxiliary[name].flush()
            # Timings are completion markers and are flushed after all predictions.
            seconds[idx] = elapsed; seconds.flush(); rows += int(hi-lo)
            if count == 1 or count % 20 == 0 or count == len(remaining):
                speed = rows/(time.perf_counter()-start)
                progress(job, phase='predict', version=version, method=method, dataset=ds, case=case,
                    batches_done=len(requested)-len(remaining)+count, batches_total=len(requested),
                    measured_batch_seconds=elapsed, rows_per_second=speed,
                    version_remaining_seconds=(len(remaining)-count)*config['batch']/speed)
        model.close(); del model; gc.collect(); torch.cuda.empty_cache()
        write(dest/'PREDICT_COMPLETE.json', dict(complete=True, requested_batches=len(requested), completed=utc()))
    replay.summarize(out, config, ds, case, method)
    if method == 'heads': summarize_heads(out, config, ds, case)
    return 0


def summarize_heads(out, config, ds, case):
    job = out/'jobs'/f'{ds}_{case}_heads'; prep = out/'data'/ds
    names = json.loads((prep/'manifest.json').read_text())['class_names']
    order = np.load(prep/'order.npy', mmap_mode='r'); batches = np.load(prep/'batches.npy')
    y = np.load(Path(config['source'])/'data'/ds/'test_y.npy', mmap_mode='r')[order]
    value = json.loads((job/'results.json').read_text())
    diagnostics = {}
    for name in HEAD_NAMES:
        pred = np.full(len(y), -1, dtype='int16')
        for version in range(len(STAGES[ds])):
            source = np.load(job/f'stage_{version}'/(name+'.npy'), mmap_mode='r')
            for _, lo, hi in batches[batches[:, 0] == version]: pred[lo:hi] = source[lo:hi]
        if np.any(pred < 0): raise ValueError('Missing head diagnostic predictions')
        diagnostics[name] = confusion_metrics(y, pred, names)
        np.save(job/f'pred_ideal_{name}.npy', pred)
    value['head_diagnostics_ideal'] = diagnostics
    value['head_diagnostics_note'] = 'Same shared representation and ideal version schedule; per-head serving costs not measured separately'
    write(job/'results.json', value)
    write(job/'COMPLETE.json', dict(state='complete', results_sha256=sha(job/'results.json'), completed=utc()))


def aggregate(out, config):
    methods = ['xgb', 'tabpfn', config['method']]
    if config['method'] == 'icd': methods.append('icd_init')
    jobs = []
    for ds in config['datasets']:
        for case in config['cases']:
            for method in methods:
                path = out/'jobs'/f'{ds}_{case}_{method}'
                if (path/'COMPLETE.json').exists(): jobs.append(json.loads((path/'results.json').read_text()))
    complete = len(jobs) == len(config['datasets'])*len(config['cases'])*len(methods)
    write(out/'results.json', dict(state='complete' if complete else 'partial', jobs=jobs,
                                  updated=utc(), html_publication=False))
    return complete


def controller(out, config):
    lock = (out/'controller.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(out, config); started = time.time()
    for case in config['cases']:
        for ds in config['datasets']:
            phases = [('fit', config['method']), ('predict', config['method'])]
            if config['method'] == 'icd': phases.append(('predict', 'icd_init'))
            for phase, method in phases:
                job = out/'jobs'/f'{ds}_{case}_{method}'; job.mkdir(exist_ok=True)
                if (job/'COMPLETE.json').exists(): continue
                if (out/'STOP_AFTER_UNIT').exists(): return 75
                attempt = len(list(job.glob(f'{phase}_attempt_*.log')))+1
                logpath = job/f'{phase}_attempt_{attempt}.log'
                command = [sys.executable, '-u', str(Path(__file__).resolve()), phase,
                           '--out', str(out), '--dataset', ds, '--case', case, '--method', method]
                with logpath.open('w') as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=replay.environment(config),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        state = dict(state='running', controller_pid=os.getpid(), worker_pid=process.pid,
                            dataset=ds, case=case, method=method, phase=phase, log=str(logpath),
                            updated=utc(), elapsed_seconds=time.time()-started)
                        if (job/'progress.json').exists(): state['progress'] = json.loads((job/'progress.json').read_text())
                        write(out/'status.json', state); time.sleep(3)
                if process.returncode:
                    write(out/'status.json', dict(state='paused' if process.returncode == 75 else 'failed',
                        returncode=process.returncode, log=str(logpath), updated=utc()))
                    return process.returncode
                aggregate(out, config)
    complete = aggregate(out, config)
    if not complete: raise RuntimeError('Controller finished without all expected results')
    verify_results(out, config)
    write_research_summary(out, config)
    write(out/'status.json', dict(state='complete', completed=utc(), elapsed_seconds=time.time()-started))
    return 0


def verify_results(out, config):
    from verify_replay import verify
    verify(out, require_all=True)
    value = json.loads((out/'verification.json').read_text())
    expected = len(config['methods'])*len(config['datasets'])*len(config['cases'])*(len(config['rates'])+1)
    assert value['verified_policies'] == expected
    value['expected_policies'] = expected
    for ds in config['datasets']:
        for case in config['cases']:
            for stage in range(len(STAGES[ds])):
                dest = out/'jobs'/f"{ds}_{case}_{config['method']}"/f'stage_{stage}'
                data, identity = stage_data(out, config, ds, case, stage)
                if config['method'] == 'heads':
                    roles = np.load(dest/'head_role_ids.npz')
                    np.testing.assert_array_equal(roles['head_fit'], identity['role_ids']['tune'])
                    np.testing.assert_array_equal(np.sort(np.r_[roles['head_selection'], roles['head_temperature']]),
                                                  np.sort(identity['role_ids']['cal']))
                    arrays = [identity['train_ids'], *[roles[k] for k in roles.files]]
                    for i, a in enumerate(arrays):
                        for b in arrays[i+1:]: assert not np.intersect1d(a, b).size
                else:
                    context = np.load(dest/'context.npz')
                    assert not np.intersect1d(context['context_indices'], context['query_indices']).size
                    np.testing.assert_array_equal(context['labels'], data['ytr'][context['context_indices']])
                    for key in ['initial', 'tuned']: assert np.isfinite(context[key]).all()
                del data
    value['adaptation_roles_verified'] = True
    write(out/'verification.json', value)


def write_research_summary(out, config):
    experiment = 'exp74' if config['method'] == 'heads' else 'exp75'
    result = json.loads((out/'results.json').read_text())
    lines = [f'# {experiment.upper()} · 분류 구조 개선 실험', '',
        '기존 순차 공격 도입 시나리오, 최종 train 10만 행, 별도 validation, seed 43을 유지했다. '
        '모든 test 행을 정책별로 한 번 평가했으며 실제 수집 시간이나 환경 drift를 재현한 실험은 아니다.', '',
        ('EXP74는 단일 TabPFN과 동일한 전체 context를 유지하고, 별도 validation 일부에서 분포 목표가 다른 '
         '세 head를 학습했다. 나머지 validation은 head 선택과 확률 보정에 분리해서 사용했다. '
         '기존 기준선은 같은 validation 라벨 접근 권한이 있으나 사용하지 않았으므로, '
         '추가 라벨 활용과 결합 구조의 기여를 동일시하지 않는다.' if experiment == 'exp74' else
         'EXP75는 2,048행 이하 context를 초기화한 뒤 고정 TabPFN을 통해 context feature만 최적화했다. '
         '클래스 균형 query 손실로 학습하고 validation 손실로 초기 상태를 포함한 checkpoint를 선택했다. '
         'untuned 대조군은 같은 context·단일 estimator·전처리를 사용한다. '
         '10만 context·4 estimators인 원 TabPFN과의 차이는 최적화 효과 하나로 해석하지 않는다.'), '',
        '| 데이터 | 배분 | 방법 | 지연 없음 Macro-F1 (%) | 1,000행/초 Macro-F1 (%) | 지연 없음 Benign FPR (%) |',
        '|---|---|---|---:|---:|---:|']
    for item in result['jobs']:
        a = item['results']['ideal']; b = item['results']['rate_1000']
        lines.append(f"| {item['dataset']} | {item['case']} | {item['method']} | {a['macro_f1']*100:.2f} | "
                     f"{b['macro_f1']*100:.2f} | {a['benign_false_positive_rate']*100:.3f} |")
    lines += ['', '위 수치는 자동 집계한 관찰 결과다. 클래스별 변화, 다른 클래스로부터 유입된 FP, '
              '단일 head 또는 초기 context 대조군을 함께 검토한 뒤 원인과 채택 여부를 판단한다. '
              '실행 성공을 성능 개선이나 논문 근거 채택으로 간주하지 않는다.', '',
              f'실행: `{out.relative_to(ROOT)}`',
              f'상세 결과: [{experiment} results.json]({os.path.relpath(out/"results.json", ROOT/"docs/lablog/evidence/20261015")})',
              f'독립 검증: [{experiment} verification.json]({os.path.relpath(out/"verification.json", ROOT/"docs/lablog/evidence/20261015")})',
              '', 'HTML·논문은 이번 실행 요청에 따라 자동 변경하지 않았다.']
    text = '\n'.join(lines)+'\n'
    from lablog import write_record
    write_record(ROOT/'docs/lablog/evidence/20261015'/f'{experiment}_results.md', text)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['run', 'prepare', 'fit', 'predict'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=['heads', 'icd', 'icd_init'], required=True)
    p.add_argument('--dataset', choices=list(STAGES)); p.add_argument('--case', choices=['natural', 'uniform'])
    args = p.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists(): config = json.loads((out/'config.json').read_text())
    else:
        config = defaults(args.method); write(out/'config.json', config)
        snapshot = out/'source'; snapshot.mkdir(exist_ok=True)
        for folder in [ROOT/'scripts/v7'/x for x in ('exp74', 'exp75', 'exp73', 'exp72')]:
            for source in folder.glob('*.py'):
                dest = snapshot/folder.name/source.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(source, dest)
        import torch
        write(out/'environment.json', dict(python=sys.executable, gpu=torch.cuda.get_device_name(0),
            packages={n: importlib.metadata.version(n) for n in ['torch', 'tabpfn', 'numpy', 'scikit-learn']},
            checkpoint_sha256=sha(config['checkpoint']), sources={str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*.py')}))
    if args.command == 'run': return controller(out, config)
    if args.command == 'prepare': prepare(out, config); return 0
    if not args.dataset or not args.case: p.error('--dataset and --case required for a worker')
    if args.command == 'fit': return worker_fit(out, config, args.dataset, args.case)
    return worker_predict(out, config, args.dataset, args.case, args.method)


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Repeat EXP84 with seeds 44/45 and retain the completed seed-43 reference."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)

import argparse
import importlib.metadata
import random
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from exp61_utils import write, sha
from small_label_updates import BUDGETS, METHODS, read, run_job, progress
from noft_dist_budget import load_source, subset_plans
from run_context_policies import select_initial
from replay_protocol import STAGES
from context_budget_sweep import simulate
from finetune_final import CHECKPOINT

BASE = ROOT/'tabpfn/results/v7/exp84/20261020_160101_small_labels_5_100_buffer_retention_s43_3204a372'
PROTOCOL = ROOT/'configs/experiments/exp84_multiseed_protocol.json'
SEEDS = [43, 44, 45]


def original_plans(ds, arr, names, seed):
    """Identical sampling algorithm to the historical 100-row incoming pools."""
    rng = np.random.default_rng(seed)
    initial = select_initial(arr['train_y'], names, ds, 1000, seed)
    released, plans = [], []
    for stage, group in enumerate(STAGES[ds]):
        incoming = initial if stage == 0 else np.concatenate([
            rng.choice(np.flatnonzero(arr['train_y'] == names.index(name)), 100, replace=False)
            for name in group])
        released.extend(incoming.tolist())
        plans.append(dict(stage=stage, incoming=incoming.tolist(), released=list(released),
                          new_classes=[] if stage == 0 else group, ft_roles={}, ours_roles={}))
    return plans


def prepare_seed(out, cfg):
    checks = []
    for ds in cfg['datasets']:
        meta, arr, historical = load_source(ds)
        # Seed 43 must reproduce the reference exactly, including within-stage order.
        replay = original_plans(ds, arr, meta['class_names'], 43)
        assert [p['incoming'] for p in replay] == [p['incoming'] for p in historical]
        original = original_plans(ds, arr, meta['class_names'], cfg['seed'])
        dest = out/'data'/ds
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['query_order.npy', 'panel_ids.npy']:
            shutil.copy2(BASE/'data'/ds/name, dest/name)
            assert sha(dest/name) == sha(BASE/'data'/ds/name)
        manifest = read(BASE/'data'/ds/'manifest.json')
        manifest.update(sampling_seed=cfg['seed'], evaluation_seed=43, reference_run=str(BASE),
                        fixed_test_across_seeds=True, fixed_panel_across_seeds=True)
        write(dest/'manifest.json', manifest)
        previous = None
        for n in BUDGETS:
            reference = read(BASE/f'new_{n}'/'data'/ds/'plans.json')
            replay_n = subset_plans(replay, arr['train_y'], n, 43)
            assert [p['incoming'] for p in replay_n] == [p['incoming'] for p in reference]
            full = subset_plans(original, arr['train_y'], n, cfg['seed'])
            plans = [{k: p[k] for k in ['stage', 'incoming', 'released', 'new_classes']} for p in full]
            for s, plan in enumerate(plans):
                assert len(plan['released']) == len(set(plan['released']))
                if s:
                    labels, counts = np.unique(arr['train_y'][plan['incoming']], return_counts=True)
                    assert set(labels) == {meta['class_names'].index(c) for c in STAGES[ds][s]}
                    assert np.all(counts == n)
                if previous:
                    assert set(previous[s]['incoming']) <= set(plan['incoming'])
            ids = plans[-1]['released']
            assert len(ids) == 1000+n*(4 if ds == 'cic2018' else 7)
            assert not np.intersect1d(arr['train_ids'][ids], arr['test_ids']).size
            if cfg['seed'] != 43:
                assert plans[0]['incoming'] != reference[0]['incoming']
                assert all(p['incoming'] != r['incoming'] for p, r in zip(plans[1:], reference[1:]))
            case = out/f'new_{n}'/'data'/ds
            write(case/'plans.json', plans)
            for policy in ['fifo', 'random', 'similar']:
                history = simulate(plans, arr, policy, cfg['seed'])
                write(case/f'{policy}_preflight.json', history)
                if cfg['seed'] == 43:
                    assert history == read(BASE/f'new_{n}'/'data'/ds/f'{policy}_preflight.json')
            checks.append(dict(dataset=ds, n=n, seed=cfg['seed'], initial_rows=1000,
                               initial_counts=sorted(np.unique(arr['train_y'][plans[0]['incoming']], return_counts=True)[1].tolist()),
                               final_released_rows=len(ids), all_new_admitted=True))
            previous = plans
    write(out/'preflight.json', dict(passed=True, seed=cfg['seed'], checks=checks,
          seed43_sampling_replayed_exactly=True, train_test_row_ids_disjoint=True,
          nested_budgets=True, fixed_evaluation_ids_and_query_order=True,
          buffer_all_rows_admitted=True, no_evicted_row_retrieved=True))


def report(out):
    subprocess.run([sys.executable, str(Path(__file__).with_name('build_multiseed_record.py')),
                    '--run', str(out)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(protocol='exp84.2-multiseed', research_version='v7', seeds=SEEDS, new_seeds=[44,45],
               reference_seed=43, reference_run=str(BASE), datasets=['cic2018','toniot'],
               new_budgets=BUDGETS, methods=METHODS, conditions_per_seed=60, expected_jobs=180,
               expected_new_jobs=120, buffer_capacity=1000, evaluation_seed=43,
               report_date=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d'))
    if (out/'config.json').exists():
        old_cfg = read(out/'config.json')
        cfg['report_date'] = old_cfg['report_date']
        assert cfg == old_cfg, 'Cannot resume with a changed protocol'
    write(out/'config.json', cfg)
    env = {k: importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn','tabpfn','xgboost']}
    assert env == read(BASE/'environment.json'), 'Reference seed environment mismatch'
    write(out/'environment.json', env)
    validation = read(BASE/'validation.json')
    assert validation['passed'] and validation['state'] == 'complete' and validation['completed'] == 60
    seeds = {'43': str(BASE), '44': str(out/'seed_44'), '45': str(out/'seed_45')}
    state = dict(state='preparing', seeds=SEEDS, seed_runs=seeds, expected_jobs=180,
                 expected_new_jobs=120, validated_seeds=[43], reused_seed=43,
                 started_epoch=time.time())
    write(out/'results.json', state)
    source_hashes = {}
    dependencies = list(Path(__file__).parent.glob('*.py')) + [PROTOCOL,
        ROOT/'scripts/common/lablog.py', ROOT/'scripts/common/lablog_format.py',
        ROOT/'scripts/common/experiment_paths.py', ROOT/'scripts/common/exp61_utils.py',
        ROOT/'scripts/v7/exp76/run_context_policies.py', ROOT/'scripts/v7/exp77/finetune_final.py',
        ROOT/'scripts/v7/exp78/noft_dist_budget.py', ROOT/'scripts/v7/exp83/context_budget_sweep.py',
        ROOT/'scripts/v7/exp72/replay_models.py', ROOT/'scripts/v7/exp72/replay_protocol.py',
        ROOT/'scripts/v7/exp71/exp71_bounded_updates.py']
    for source in dependencies:
        dest = out/'source_dependencies'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert sha(dest) == sha(source), f'Source changed since snapshot: {source}'
        else:
            shutil.copy2(source, dest)
        source_hashes[str(source)] = sha(source)
    source_hashes[str(CHECKPOINT)] = sha(CHECKPOINT)
    assert source_hashes[str(CHECKPOINT)] == read(BASE/'source_hashes.json')[str(CHECKPOINT)]
    write(out/'source_hashes.json', source_hashes)
    protected = [BASE/'results.json', BASE/'config.json', BASE/'validation.json',
                 ROOT/'docs/lablog/reports/20261020_exp84.tex']
    protected += list(BASE.glob('new_*/data/*/plans.json'))
    protected += list((ROOT/'docs/lablog/data').glob('exp84_*.csv'))
    protected = [p for p in protected if not p.name.startswith('exp84_multiseed')]
    protected += list((ROOT/'docs/lablog/figures').glob('exp84_*.pdf'))
    protected = [p for p in protected if not p.name.startswith('exp84_multiseed')]
    reference_hashes = {str(p): sha(p) for p in protected}
    write(out/'reference_preservation.json', reference_hashes)
    error = None
    try:
        for seed in cfg['new_seeds']:
            seed_out = Path(seeds[str(seed)])
            seed_cfg = read(BASE/'config.json')
            seed_cfg.update(protocol='exp84.2-multiseed', seed=seed, report_date=cfg['report_date'],
                            evaluation_seed=43, parent_run=str(out), seed_repetitions=3)
            write(seed_out/'config.json', seed_cfg)
            prepare_seed(seed_out, seed_cfg)
        for ds in cfg['datasets']:
            a = read(out/'seed_44/new_100/data'/ds/'plans.json')
            b = read(out/'seed_45/new_100/data'/ds/'plans.json')
            assert all(p['incoming'] != q['incoming'] for p,q in zip(a,b))
        write(out/'preflight.json', dict(passed=True, reference_seed_complete=True,
              same_packages_and_checkpoint=True, seed43_sampling_replayed_exactly=True,
              distinct_training_samples=True, fixed_test_panel_and_query_order=True, seeds=SEEDS))
        if args.prepare_only:
            progress(out,'preflight_complete',seeds=SEEDS)
            return
        state['state'] = 'running'
        write(out/'results.json', state)
        report(out)
        import torch
        torch.set_num_threads(8)
        torch.cuda.init()
        progress(out, 'running', completed_new_jobs=0, expected_new_jobs=120, seeds=SEEDS)
        for seed in cfg['new_seeds']:
            seed_out = Path(seeds[str(seed)])
            seed_cfg = read(seed_out/'config.json')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            torch.cuda.synchronize()
            warm_start = time.perf_counter()
            a = torch.ones((512,512), device='cuda'); b = a@a
            torch.cuda.synchronize(); del a,b
            write(seed_out/'warmup.json', dict(seconds=time.perf_counter()-warm_start,
                  method='512x512 CUDA matmul; excluded from reported update cost', gpu=torch.cuda.get_device_name()))
            jobs = []
            state.update(active_seed=seed)
            write(out/'results.json', state)
            for n in BUDGETS:
                for ds in cfg['datasets']:
                    meta, arr, _ = load_source(ds)
                    plans = read(seed_out/f'new_{n}'/'data'/ds/'plans.json')
                    for method in METHODS:
                        progress(out,'running',active_seed=seed,dataset=ds,new_per_class=n,method=method,
                                 completed_new_jobs=(seed-44)*60+len(jobs),expected_new_jobs=120)
                        result = run_job(seed_out,seed_cfg,ds,n,method,meta,arr,plans)
                        assert result['seed'] == seed
                        jobs.append(result)
                        write(seed_out/'results.json',dict(state='running',jobs=jobs,expected_jobs=60))
            write(seed_out/'results.json',dict(state='complete',jobs=jobs,expected_jobs=60))
            subprocess.run([sys.executable,str(Path(__file__).with_name('build_small_label_record.py')),
                            '--run',str(seed_out),'--validate-only'],cwd=ROOT,check=True)
            state['validated_seeds'].append(seed)
            write(out/'results.json',state)
            report(out)
        assert all(sha(p)==h for p,h in source_hashes.items()), 'Dependency changed during run'
        assert all(sha(p)==h for p,h in reference_hashes.items()), 'Seed-43 artifact changed'
        state.update(state='complete',finished_epoch=time.time(),reference_preserved=True,
                     validated_seeds=SEEDS,completed_jobs=180,completed_new_jobs=120)
        write(out/'results.json',state)
        progress(out,'complete',completed_new_jobs=120,expected_new_jobs=120,seeds=SEEDS)
    except BaseException as exc:
        error=exc
        state.update(state='failed',error=str(exc),failed_epoch=time.time())
        write(out/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc()))
        write(out/'results.json',state)
        # Validate and report completed jobs even when a later job failed.
        if state.get('active_seed'):
            seed_out=Path(seeds[str(state['active_seed'])])
            data=read(seed_out/'results.json') if (seed_out/'results.json').exists() else dict(jobs=[])
            data.update(state='failed',error=str(exc),expected_jobs=60)
            write(seed_out/'results.json',data)
            subprocess.run([sys.executable,str(Path(__file__).with_name('build_small_label_record.py')),
                            '--run',str(seed_out),'--validate-only'],cwd=ROOT,check=False)
        raise
    finally:
        if not args.prepare_only:
            try:
                report(out)
            except Exception as exc:
                write(out/'REPORT_FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc()))
                if error is None:
                    raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit the revised small-label design on real row IDs; never fit a model."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from experiment_paths import repo_root

ROOT = repo_root()


def existing_function(relative_path, name):
    """Load only the exact audited function, avoiding runner imports/side effects."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'np': np, 'copy': copy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name], hashlib.sha256(ast.get_source_segment(path.read_text(), node).encode()).hexdigest()


def main():
    config_path = ROOT / 'configs/experiments/proposals/20261020_context_admission.json'
    config = json.loads(config_path.read_text())
    assert config['status'] == 'authorized_for_execution'
    assert config['available_labels_per_new_class'] == [5, 10, 20, 30, 50, 100]
    subset, subset_sha = existing_function('scripts/v7/exp78/noft_dist_budget.py', 'subset_plans')
    replace, replace_sha = existing_function('scripts/v7/exp76/run_context_policies.py', 'replace_context')
    checks = []
    for ds in config['datasets']:
        base = ROOT / config['sources']['base_plans'] / ds
        meta = json.loads((base / 'manifest.json').read_text())
        source = Path(meta['source'])
        arr = {k: np.load(source / f'{k}.npy', mmap_mode='r')
               for k in ['train_X', 'train_y', 'train_ids', 'test_ids']}
        original = json.loads((base / 'plans.json').read_text())
        previous = None
        for n in config['available_labels_per_new_class']:
            plans = subset(original, arr['train_y'], n, config['seed'])
            assert plans[0]['incoming'] == original[0]['incoming']
            if n in [10, 20, 50]:
                saved = ROOT / config['sources']['overlap_plans'] / f'new_{n}/data/{ds}/plans.json'
                saved_plans = json.loads(saved.read_text())
                for a, b in zip(plans, saved_plans):
                    assert a['incoming'] == b['incoming'] and a['released'] == b['released']
            if n == 100:
                for a, b in zip(plans, original):
                    assert a['incoming'] == b['incoming'] and a['released'] == b['released']
            for s, plan in enumerate(plans):
                if previous:
                    assert set(previous[s]['incoming']) <= set(plan['incoming'])
                if s:
                    counts = np.unique(arr['train_y'][plan['incoming']], return_counts=True)[1]
                    assert all(counts == n)
            released = plans[-1]['released']
            assert len(released) == 1000 + n * (4 if ds == 'cic2018' else 7)
            assert len(released) == len(set(released))
            assert not np.intersect1d(arr['train_ids'][released], arr['test_ids']).size
            for policy in ['fifo', 'random', 'similar']:
                ctx = np.asarray(plans[0]['incoming'], dtype='int64')
                evicted = set()
                stages = []
                for s, plan in enumerate(plans):
                    incoming = np.asarray(plan['incoming'], dtype='int64')
                    if s:
                        old = ctx.copy()
                        ctx, removed = replace(old, incoming, arr['train_y'], arr['train_X'],
                                               policy, config['seed'] + s * 1009)
                        assert len(removed) == len(incoming)
                        assert set(removed) <= set(old)
                        assert set(ctx) == (set(old) - set(removed)) | set(incoming)
                        evicted.update(removed.tolist())
                    else:
                        removed = np.array([], dtype='int64')
                    assert len(ctx) == len(set(ctx)) == 1000
                    assert not set(ctx) & evicted
                    assert set(incoming) <= set(ctx)
                    assert set(ctx) <= set(plan['released'])
                    assert set(arr['train_y'][ctx]) == set(arr['train_y'][plan['released']])
                    counts = np.bincount(arr['train_y'][ctx], minlength=len(meta['class_names']))
                    stages.append(dict(stage=s+1, supplied=len(incoming), admitted=len(incoming),
                                       rejected=0, removed=len(removed), context_rows=len(ctx),
                                       cumulative_rows=len(plan['released']),
                                       context_counts=dict(zip(meta['class_names'], counts.tolist())),
                                       context_ids_sha256=hashlib.sha256(ctx.tobytes()).hexdigest()))
                checks.append(dict(dataset=ds, n=n, policy=policy, stages=stages))
            previous = plans
    dest = ROOT / 'docs/lablog/evidence/20261020/context_admission_feasibility.json'
    dest.write_text(json.dumps(dict(
        passed=True, revision=3, model_experiments_executed_by_this_audit=False,
        scope='Real training row selection and buffer feasibility only, not model quality or update timing',
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        exact_existing_functions={'subset_plans': subset_sha, 'replace_context': replace_sha},
        existing_10_20_50_100_plans_identical=True, all_supplied_new_rows_admitted=True,
        train_test_row_ids_disjoint=True, checks=checks), ensure_ascii=False, indent=2)+'\n')
    print(f'{len(checks)} real-row buffer schedules passed; 60 authorized model conditions; no model fitted by this audit')


if __name__ == '__main__':
    main()

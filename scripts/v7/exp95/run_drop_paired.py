"""EXP95: strictly paired few-shot comparison of deleting residual flows.

Every job starts from the training flows EXP94 selected.  Only residual flows are
replaced, so a difference from EXP94 comes from deletion, the replaced rows and,
in one arm, the refit PCA.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
for rel in ('scripts/common', 'scripts/v7/exp92', 'scripts/v7/exp93', 'scripts/v7/exp94', 'scripts/v7/exp95'):
    sys.path.insert(0, str(ROOT / rel))
from experiment_paths import resolve_path
from run_boundary_context import base_run, snapshot
from run_pcap_benchmark import _f1, _make_model, _predict_proba_batched, _timed, _weights_digest, read, write

import numpy as np

SOURCE_VIEW = {'flowstats-v1': 'flowstats-v1', 'netfound-exp94-pca': 'netfound-v2-small', 'netfound-refit-pca': 'netfound-v2-small'}


def arm_data(cfg: dict[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Vectors of the remaining rows plus the residual mask of all EXP94 rows."""
    first = resolve_path(cfg['first_run'], ROOT)
    tail = np.load(first / 'base/residual.npz')['tail']
    with np.load(first / 'encoders' / SOURCE_VIEW[arm] / 'dataset.npz') as z:
        X, y, split, ids, names = z['X'], z['y'], z['split'], z['flow_id'], z['class_names'].tolist()
    if arm == 'netfound-exp94-pca':
        with np.load(base_run(cfg) / 'encoders/netfound-v2-small/dataset.npz') as z:
            base_X, base_ids = z['X'], z['flow_id']
        if not np.array_equal(base_ids[~tail], ids):
            raise RuntimeError('remaining rows differ from EXP94 rows')
        X = base_X[~tail]
    return X, y, split, ids, names, tail


def paired_rows(old_ids: list[str], classes: list[int], max_budget: int, seed: int, y: np.ndarray, split: np.ndarray, where: dict[str, int]) -> tuple[dict[int, np.ndarray], int]:
    """EXP94 selection in order, with removed flows replaced at the same position."""
    chosen, replaced = {}, 0
    for pos, c in enumerate(classes):
        part = old_ids[pos * max_budget:(pos + 1) * max_budget]
        kept = [where.get(v, -1) for v in part]
        missing = [i for i, v in enumerate(kept) if v < 0]
        if missing:
            pool = np.flatnonzero((split == 'train') & (y == c))
            spare = np.setdiff1d(pool, np.asarray([v for v in kept if v >= 0]), assume_unique=False)
            draw = np.random.default_rng(seed + 1009 * c + 777).choice(spare, len(missing), replace=False)
            for i, v in zip(missing, draw.tolist()):
                kept[i] = v
            replaced += len(missing)
        chosen[c] = np.asarray(kept, dtype=np.int64)
    return chosen, replaced


def worker(cfg: dict[str, Any], out: Path, arm: str) -> None:
    few = snapshot(cfg)['fewshot']
    src = base_run(cfg)
    X, y, split, ids, names, _ = arm_data(cfg, arm)
    where = {str(v): i for i, v in enumerate(ids)}
    view = SOURCE_VIEW[arm]
    max_budget = max(few['budgets'])
    for regime in few['regimes']:
        classes = list(range(len(names))) if regime == 'with_benign' else list(range(1, len(names)))
        test = np.flatnonzero((split == 'test') & np.isin(y, classes))
        for seed in few['seeds']:
            old = read(src / 'evaluations' / view / 'fewshot/jobs' / f'{regime}_s{seed}_n{max_budget}_xgb' / 'result.json')['train_flow_ids']
            selected, replaced = paired_rows(old, classes, max_budget, seed, y, split, where)
            for budget in few['budgets']:
                chosen = np.concatenate([selected[c][:budget] for c in classes])
                local = {c: i for i, c in enumerate(classes)}
                y_train = np.asarray([local[int(v)] for v in y[chosen]], dtype=np.int64)
                for method in few['methods']:
                    job = out / 'evaluations' / arm / 'jobs' / f'{regime}_s{seed}_n{budget}_{method}'
                    if (job / 'result.json').exists():
                        continue
                    job.mkdir(parents=True, exist_ok=True)
                    model, construct = _timed(lambda: _make_model(method, seed, classes, few))
                    _, fit = _timed(lambda: model.fit(X[chosen], y_train))
                    frozen = _weights_digest(model) if method == 'tabpfn' else None
                    proba, infer = _timed(lambda: _predict_proba_batched(model, X[test], int(few['inference_batch_rows'])))
                    if proba.shape != (len(test), len(classes)) or not np.isfinite(proba).all():
                        raise RuntimeError('invalid probability matrix')
                    if frozen and frozen != _weights_digest(model):
                        raise RuntimeError('TabPFN backbone weights changed during inference')
                    pred = np.asarray(classes, dtype=np.int16)[proba.argmax(1)]
                    np.save(job / 'predictions.npy', pred)
                    old_budget = set(read(src / 'evaluations' / view / 'fewshot/jobs' / f'{regime}_s{seed}_n{budget}_{method}' / 'result.json')['train_flow_ids'])
                    write(job / 'result.json', dict(arm=arm, regime=regime, seed=seed, budget=budget, method=method, train_rows=len(chosen), train_flow_ids=[str(v) for v in ids[chosen]],
                                                    replaced_rows=int(sum(str(v) not in old_budget for v in ids[chosen])), replaced_rows_at_max_budget=replaced,
                                                    construct_seconds=construct, fit_seconds=fit, inference_seconds=infer, frozen_weights_sha256=frozen))
                    del model; gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    print('worker: done', arm, flush=True)


def f1s(yt: np.ndarray, pr: np.ndarray, classes: list[int], names: list[str]) -> dict[str, Any]:
    r = _f1(yt, pr, classes, names)
    return dict(macro=r['macro_f1'], classes={c['name']: c for c in r['classes']})


def report(cfg: dict[str, Any], out: Path) -> None:
    few = snapshot(cfg)['fewshot']; src = base_run(cfg); first = resolve_path(cfg['first_run'], ROOT)
    tail = np.load(first / 'base/residual.npz')['tail']
    with np.load(src / 'encoders/flowstats-v1/dataset.npz') as z:
        y, split = z['y'], z['split']
    names = json.loads((first / 'base/meta.json').read_text())['class_names']
    rows = []
    for arm in cfg['arms']:
        for regime in few['regimes']:
            classes = list(range(len(names))) if regime == 'with_benign' else list(range(1, len(names)))
            test_all = np.flatnonzero((split == 'test') & np.isin(y, classes))
            keep = ~tail[test_all]; yt = y[test_all][keep]
            for seed in few['seeds']:
                for budget in few['budgets']:
                    for method in few['methods']:
                        tag = f'{regime}_s{seed}_n{budget}_{method}'
                        p_old = np.load(src / 'evaluations' / SOURCE_VIEW[arm] / 'fewshot/jobs' / tag / 'predictions.npy')[keep]
                        job = out / 'evaluations' / arm / 'jobs' / tag
                        p_new = np.load(job / 'predictions.npy'); info = read(job / 'result.json')
                        if len(p_old) != len(p_new):
                            raise RuntimeError(f'prediction length mismatch: {tag}')
                        b, d = f1s(yt, p_old, classes, names), f1s(yt, p_new, classes, names)
                        row = dict(arm=arm, regime=regime, seed=seed, budget=budget, method=method, test_rows=len(yt), replaced_rows=info['replaced_rows'], train_rows=info['train_rows'],
                                   macro_exp94=b['macro'], macro_drop=d['macro'], agreement=float((p_old == p_new).mean()))
                        if regime == 'with_benign':
                            ben = yt == 0
                            row['benign_alarm_exp94'] = float((p_old[ben] != 0).mean()); row['benign_alarm_drop'] = float((p_new[ben] != 0).mean())
                        for c in b['classes']:
                            for k in ('f1', 'precision', 'recall'):
                                row[f'{k}_{c}_exp94'] = b['classes'][c][k]; row[f'{k}_{c}_drop'] = d['classes'][c][k]
                        rows.append(row)
    with (out / 'summary_pairs.csv').open('w', newline='') as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('arm', 'regime', 'seed', 'budget', 'method'), k))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['arm'], r['regime'], r['method'], r['budget'])].append(r)
    table = []
    for (arm, regime, method, budget), g in sorted(groups.items()):
        diff = np.asarray([x['macro_drop'] - x['macro_exp94'] for x in g])
        item = dict(arm=arm, regime=regime, method=method, budget=budget, seeds=len(g), exp94=float(np.mean([x['macro_exp94'] for x in g])), drop=float(np.mean([x['macro_drop'] for x in g])),
                    paired_gain=float(diff.mean()), paired_sd=float(diff.std(ddof=1)), seeds_improved=int((diff > 0).sum()), replaced_rows=float(np.mean([x['replaced_rows'] for x in g])), train_rows=g[0]['train_rows'],
                    agreement=float(np.mean([x['agreement'] for x in g])))
        for c in names:
            if f'f1_{c}_drop' in g[0]:
                item[f'f1_{c}'] = [float(np.mean([x[f'f1_{c}_exp94'] for x in g])), float(np.mean([x[f'f1_{c}_drop'] for x in g]))]
                item[f'precision_{c}'] = [float(np.mean([x[f'precision_{c}_exp94'] for x in g])), float(np.mean([x[f'precision_{c}_drop'] for x in g]))]
        if 'benign_alarm_drop' in g[0]:
            item['benign_alarm'] = [float(np.mean([x['benign_alarm_exp94'] for x in g])), float(np.mean([x['benign_alarm_drop'] for x in g]))]
        table.append(item)
    write(out / 'comparison.json', dict(jobs=len(rows), table=table))
    print('report: jobs', len(rows), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--stage', default='all', choices=['all', 'worker', 'report']); ap.add_argument('--arm')
    args = ap.parse_args()
    cfg = read(resolve_path(args.config, ROOT)); out = resolve_path(args.output, ROOT); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'worker':
        return worker(cfg, out, args.arm)
    if args.stage == 'all':
        write(out / 'protocol_snapshot.json', cfg)
        procs = []
        for arm in cfg['arms']:
            log = (out / f'worker_{arm}.log').open('a')
            procs.append(subprocess.Popen([sys.executable, __file__, '--config', args.config, '--output', str(out), '--stage', 'worker', '--arm', arm], stdout=log, stderr=subprocess.STDOUT))
        if any(p.wait() != 0 for p in procs):
            raise RuntimeError('a worker failed')
    report(cfg, out)
    write(out / 'status.json', dict(status='complete', stage=args.stage))


if __name__ == '__main__':
    main()

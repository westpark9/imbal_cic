"""EXP95: few-shot re-evaluation with residual flows deleted, paired with EXP94.

Remaining flows keep their EXP94 IDs, labels, splits, tokens, embeddings and
statistics.  Only PCA is refit, on the remaining train rows.  The EXP94 baseline
is recomputed from saved predictions on the same remaining test rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
for rel in ('scripts/common', 'scripts/v7/exp92', 'scripts/v7/exp93', 'scripts/v7/exp94', 'scripts/v7/exp95'):
    sys.path.insert(0, str(ROOT / rel))
from experiment_paths import resolve_path
from run_boundary_context import base_run, snapshot, stage_base
from run_pcap_benchmark import _f1, fewshot, read, write

import numpy as np


def prepare(cfg: dict[str, Any], out: Path) -> None:
    from sklearn.decomposition import PCA
    done = out / 'views.json'
    if done.exists():
        return
    stage_base(cfg, out)
    src = base_run(cfg)
    a = np.load(out / 'base/arrays.npz'); tail = np.load(out / 'base/residual.npz')['tail']
    y, split, ids = a['y'], a['split'], a['flow_id']
    names = read(out / 'base/meta.json')['class_names']
    idx = np.flatnonzero(~tail)
    t0 = time.time()
    with np.load(src / 'encoders/flowstats-v1/dataset.npz') as z:
        stats = z['X'][idx]
    emb = np.load(src / 'encoders/netfound-v2-small/embeddings.npy', mmap_mode='r')
    E = np.asarray(emb[idx], dtype=np.float32)
    train = split[idx] == 'train'
    pca = PCA(n_components=64, svd_solver='full', random_state=int(cfg['seed'])).fit(E[train])
    Z = pca.transform(E).astype(np.float32)
    if not (np.isfinite(Z).all() and np.isfinite(stats).all()):
        raise FloatingPointError('non-finite vectors')
    for view, X in (('flowstats-v1', stats), ('netfound-v2-small', Z)):
        target = out / 'encoders' / view
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / 'dataset.npz', X=X, y=y[idx], split=split[idx], flow_id=ids[idx], class_names=np.asarray(names))
    write(done, dict(rows=len(idx), removed=int(tail.sum()), train_rows=int(train.sum()), labelled_rows=int((y[idx] >= 0).sum()),
                     pca_dimensions=64, pca_fit_rows=int(train.sum()), pca_explained_variance=float(pca.explained_variance_ratio_.sum()), seconds=round(time.time() - t0)))
    print('prepare: rows', len(idx), 'pca variance', float(pca.explained_variance_ratio_.sum()), flush=True)


def worker(cfg: dict[str, Any], out: Path, view: str) -> None:
    config = snapshot(cfg)
    config['class_order'] = read(out / 'base/meta.json')['class_names']
    fewshot(config, out / 'encoders' / view, out, view)
    print('fewshot: done', view, flush=True)


def f1s(yt: np.ndarray, pr: np.ndarray, classes: list[int], names: list[str]) -> dict[str, Any]:
    r = _f1(yt, pr, classes, names)
    return dict(macro=r['macro_f1'], classes={c['name']: c['f1'] for c in r['classes']}, precision={c['name']: c['precision'] for c in r['classes']}, recall={c['name']: c['recall'] for c in r['classes']})


def report(cfg: dict[str, Any], out: Path) -> None:
    src = base_run(cfg); few = snapshot(cfg)['fewshot']
    a = np.load(out / 'base/arrays.npz'); tail = np.load(out / 'base/residual.npz')['tail']
    y, split, ids = a['y'], a['split'], a['flow_id']
    names = read(out / 'base/meta.json')['class_names']
    rows, seeds_all = [], few['seeds']
    used = defaultdict(list)
    for view in cfg['views']:
        for regime in few['regimes']:
            classes = list(range(len(names))) if regime == 'with_benign' else list(range(1, len(names)))
            test_all = np.flatnonzero((split == 'test') & np.isin(y, classes))
            keep = ~tail[test_all]
            yt = y[test_all][keep]
            with np.load(out / 'encoders' / view / 'dataset.npz') as z:
                dy, ds, di = z['y'], z['split'], z['flow_id']
            mine = (ds == 'test') & np.isin(dy, classes)
            if not np.array_equal(di[mine], ids[test_all][keep]):
                raise RuntimeError('test rows differ from EXP94 minus residual rows')
            index = {str(v): i for i, v in enumerate(ids)}
            for seed in seeds_all:
                for budget in few['budgets']:
                    for method in few['methods']:
                        tag = f'{regime}_s{seed}_n{budget}_{method}'
                        old = src / 'evaluations' / view / 'fewshot/jobs' / tag
                        new = out / 'evaluations' / view / 'fewshot/jobs' / tag
                        p_old = np.load(old / 'predictions.npy'); p_new = np.load(new / 'predictions.npy')
                        if len(p_old) != len(test_all) or len(p_new) != len(yt):
                            raise RuntimeError(f'prediction length mismatch: {tag}')
                        whole = f1s(y[test_all], p_old, classes, names)
                        base = f1s(yt, p_old[keep], classes, names); drop = f1s(yt, p_new, classes, names)
                        row = dict(view=view, regime=regime, seed=seed, budget=budget, method=method, test_rows=len(yt), exp94_test_rows=len(test_all),
                                   macro_exp94_all_rows=whole['macro'], macro_exp94_same_rows=base['macro'], macro_drop=drop['macro'])
                        if regime == 'with_benign':
                            b = yt == 0
                            row['benign_alarm_exp94'] = float((p_old[keep][b] != 0).mean()); row['benign_alarm_drop'] = float((p_new[b] != 0).mean())
                        for c in base['classes']:
                            row[f'f1_{c}_exp94'] = base['classes'][c]; row[f'f1_{c}_drop'] = drop['classes'][c]
                            row[f'precision_{c}_exp94'] = base['precision'][c]; row[f'precision_{c}_drop'] = drop['precision'][c]
                            row[f'recall_{c}_exp94'] = base['recall'][c]; row[f'recall_{c}_drop'] = drop['recall'][c]
                        rows.append(row)
                        if budget == max(few['budgets']):
                            train = read(old / 'result.json')['train_flow_ids']
                            used[(view, regime)].append(float(np.mean([tail[index[v]] for v in train])))
    with (out / 'summary_pairs.csv').open('w', newline='') as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('view', 'regime', 'seed', 'budget', 'method'), k))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    table = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r['view'], r['regime'], r['method'], r['budget'])].append(r)
    for (view, regime, method, budget), g in sorted(groups.items()):
        d = np.asarray([x['macro_drop'] - x['macro_exp94_same_rows'] for x in g])
        item = dict(view=view, regime=regime, method=method, budget=budget, seeds=len(g), exp94_all_rows=float(np.mean([x['macro_exp94_all_rows'] for x in g])),
                    exp94_same_rows=float(np.mean([x['macro_exp94_same_rows'] for x in g])), drop=float(np.mean([x['macro_drop'] for x in g])), paired_gain=float(d.mean()), paired_sd=float(d.std(ddof=1)), seeds_improved=int((d > 0).sum()))
        for c in names:
            if f'f1_{c}_drop' in g[0]:
                item[f'f1_{c}'] = [float(np.mean([x[f'f1_{c}_exp94'] for x in g])), float(np.mean([x[f'f1_{c}_drop'] for x in g]))]
        if 'benign_alarm_drop' in g[0]:
            item['benign_alarm'] = [float(np.mean([x['benign_alarm_exp94'] for x in g])), float(np.mean([x['benign_alarm_drop'] for x in g]))]
        table.append(item)
    write(out / 'comparison.json', dict(jobs=len(rows), test_rows={r['regime']: [r['exp94_test_rows'], r['test_rows']] for r in rows}, table=table,
                                        residual_share_of_exp94_n50_train_rows={f'{k[0]}|{k[1]}': float(np.mean(v)) for k, v in used.items()}))
    print('report: jobs', len(rows), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--stage', default='all', choices=['all', 'worker', 'report']); ap.add_argument('--view')
    args = ap.parse_args()
    cfg = read(resolve_path(args.config, ROOT)); out = resolve_path(args.output, ROOT); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'worker':
        return worker(cfg, out, args.view)
    if args.stage == 'all':
        write(out / 'protocol_snapshot.json', cfg)
        prepare(cfg, out)
        procs = []
        for view in cfg['views']:
            log = (out / f'worker_{view}.log').open('a')
            procs.append(subprocess.Popen([sys.executable, __file__, '--config', args.config, '--output', str(out), '--stage', 'worker', '--view', view], stdout=log, stderr=subprocess.STDOUT))
        if any(p.wait() != 0 for p in procs):
            raise RuntimeError('a few-shot worker failed')
    report(cfg, out)
    write(out / 'status.json', dict(status='complete', stage=args.stage))


if __name__ == '__main__':
    main()

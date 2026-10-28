#!/usr/bin/env python3
"""Build portable EXP91 NPZs from the existing, unfiltered NF-v3 suite.

No host-specific absolute paths or old result arrays are required. The source
suite and checkpoint are local assets excluded from Git. Sampling reproduces
EXP89 attacks and EXP90 fixed750 benign rows; original test overlap is retained
and explicitly declared as a historical diagnostic, not an independent split.
"""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents
             if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import repo_root
ROOT = repo_root(__file__)

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import pickle
import numpy as np
if __package__:
    from .common import row_groups
else:
    from common import row_groups

SOURCE = ROOT/'data/nfv3_energy_suite_uncapped_scenarios.pkl'
INPUTS = ROOT/'data/exp91/inputs'
CHECKPOINT = ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'
SOURCE_HASH = '07bf031ae6fd94380a59f9a199d615aea08c699816991a5d6b07248f5ef7ae61'
SOURCE_BYTES = 14694741242
DATASETS = {'cic2018': 'cse_cic_ids2018', 'toniot': 'ton_iot'}


def stamp():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def filename(dataset, n, seed, unlabelled):
    return f'{dataset}_original_fixed750_n{n}_s{seed}_u{unlabelled}_full.npz'


def choose_labels(pool, families, names, seed):
    """Exactly EXP89's 20-row class draws plus EXP90's 730 extra benign rows."""
    selected = []
    for c, name in enumerate(names):
        candidates = pool[families[pool] == name]
        if len(candidates) < 20:
            raise ValueError(f'Fewer than 20 training rows for {name}')
        rows = np.random.default_rng(seed+c*1009).choice(candidates, 20, replace=False)
        if name == 'benign':
            if len(candidates) < 800:
                raise ValueError('At least 800 benign candidates needed for the fixed750 sampling rule')
            extra = np.random.default_rng(seed+900001).choice(candidates, 800, replace=False)
            extra = extra[~np.isin(extra, rows)][:730]
            rows = np.concatenate([rows, extra])
        selected.append(rows)
    return selected


def choose_unlabelled(X, pool, reserved_ids, count, seed):
    """Uniform train-only draw; exclude all n<=20 labeled feature groups.

    No family labels are inspected. Duplicate rows inside U are preserved.
    A seeded permutation keeps accepted rows independent of processing chunks.
    """
    if count == 0:
        return np.empty(0, dtype=np.int64), 0
    rng = np.random.default_rng(seed+710003)
    blocked = row_groups(np.nan_to_num(np.asarray(X[reserved_ids], dtype=np.float32)))
    # Random permutation sampling is independent of labels, including tail labels.
    order = rng.permutation(pool)
    accepted, collected, rejected = [], 0, 0
    for start in range(0, len(order), max(count, 4096)):
        batch = order[start:start+max(count, 4096)]
        batch = batch[~np.isin(batch, reserved_ids)]
        features = np.nan_to_num(np.asarray(X[batch], dtype=np.float32))
        keep = ~np.isin(row_groups(features), blocked)
        rejected += int((~keep).sum())
        good = batch[keep][:count-collected]
        accepted.append(good)
        collected += len(good)
        if collected == count:
            return np.concatenate(accepted), rejected
    raise ValueError(f'Only {collected} eligible unlabeled rows for requested {count}')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=INPUTS)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--budgets', type=int, nargs='+', default=[2, 3, 5, 10, 20])
    p.add_argument('--seeds', type=int, nargs='+', default=[43])
    p.add_argument('--unlabelled', type=int, default=10000)
    p.add_argument('--check', action='store_true', help='Show exact required/generated paths without reading the suite')
    return p.parse_args()


def main():
    args = arguments()
    if not all(1 <= n <= 20 for n in args.budgets) or args.unlabelled < 0:
        raise ValueError('Budgets must be 1..20 and unlabelled count must be nonnegative')
    requested = [(d, n, s, args.output/filename(d, n, s, args.unlabelled))
                 for d in args.datasets for s in args.seeds for n in args.budgets]
    if args.check:
        print(json.dumps({'source': str(args.source), 'source_exists': args.source.is_file(),
            'checkpoint': str(CHECKPOINT), 'checkpoint_exists': CHECKPOINT.is_file(),
            'inputs': [{'path': str(p), 'ready': p.is_file() and p.with_suffix('.json').is_file()}
                       for _, _, _, p in requested],
            'git_excludes': ['data/', '*.npz', '*.ckpt'],
            'note': 'git pull transfers code only; copy the suite or prepared inputs and checkpoint separately.'},
            ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        missing = []
        for d, n, s, p in requested:
            if p.is_file() and p.with_suffix('.json').is_file():
                md = json.loads(p.with_suffix('.json').read_text())
                if sha(p) != md['npz_sha256'] or md['source_sha256'] != SOURCE_HASH:
                    raise ValueError(f'Existing input integrity mismatch: {p}; use a fresh output folder')
                print(f'Reuse {p}', flush=True)
            else:
                missing.append((d, n, s, p))
        if not missing:
            return
        if not args.source.is_file():
            raise FileNotFoundError(f'Git does not contain the dataset. Place the existing source at {args.source}, '
                                    'or copy the prepared data/exp91/inputs directory from the source machine.')
        print(f'[{stamp()}] Verify source: {args.source}', flush=True)
        if args.source.stat().st_size != SOURCE_BYTES or sha(args.source) != SOURCE_HASH:
            raise ValueError('Source differs from the EXP89/90 suite; refusing a silent dataset change')
        print(f'[{stamp()}] Load source (14.7 GB file; preparation requires substantial CPU RAM)', flush=True)
        with args.source.open('rb') as f:
            suite = pickle.load(f)  # trusted, existing research artifact, hash verified above
        from exp_utils import scenario_chronological_split
        X = suite['X']
        families = np.asarray(suite['families'])
        datasets = np.asarray(suite['dataset_names'])
        scenarios = np.asarray(suite['attack_scenarios'])
        timestamps = np.asarray(suite['timestamps'])
        source_root = ROOT/'tabpfn/results/v7/exp89/20261027_144031_curation_benign_n2_3_5_10_20_five_seeds_e1167f0a/data'
        for dataset in args.datasets:
            pending = [r for r in missing if r[0] == dataset]
            if not pending:
                continue
            print(f'[{stamp()}] Rebuild original split: {dataset}', flush=True)
            target = np.flatnonzero(datasets == DATASETS[dataset])
            names = sorted(np.unique(families[target]).tolist())
            splits, _ = scenario_chronological_split(target, scenarios, timestamps)
            pool, test = splits['train'], splits['test']
            prior = json.loads((source_root/dataset/'manifest.json').read_text())
            actual_counts = {name: int(np.sum(families[pool] == name)) for name in names}
            if (names != prior['class_names'] or len(test) != prior['original_test_rows']
                    or actual_counts != prior['candidate_counts']['original']):
                raise ValueError('Rebuilt split does not match the EXP89 inventory')
            # Same seed43, label-independent test ordering as EXP89/90.
            test = test[np.random.default_rng(43).permutation(len(test))]
            tx = np.nan_to_num(np.asarray(X[test], dtype=np.float32))
            ty = np.searchsorted(names, families[test]).astype(np.int64)
            for seed in args.seeds:
                jobs = [r for r in pending if r[2] == seed]
                if not jobs:
                    continue
                draws = choose_labels(pool, families, names, seed)
                reserved = np.concatenate(draws)
                uid, rejected = choose_unlabelled(X, pool, reserved, args.unlabelled, seed)
                xu = np.nan_to_num(np.asarray(X[uid], dtype=np.float32))
                for _, n, _, path in jobs:
                    lid = np.concatenate([rows if name == 'benign' else rows[:n]
                                          for name, rows in zip(names, draws)])
                    yl = np.searchsorted(names, families[lid]).astype(np.int64)
                    xl = np.nan_to_num(np.asarray(X[lid], dtype=np.float32))
                    if np.intersect1d(lid, uid).size or np.intersect1d(np.r_[lid, uid], test).size:
                        raise RuntimeError('Row identity leakage in prepared input')
                    print(f'[{stamp()}] Write {path.name}: L={len(lid)}, U={len(uid)}, test={len(test)}', flush=True)
                    tmp = path.with_name(path.name+'.tmp')
                    with tmp.open('wb') as stream:
                        np.savez_compressed(stream, X_l=xl, y_l=yl, X_u=xu, X_test=tx, y_test=ty,
                                            ids_l=lid, ids_u=uid, ids_test=test)
                    tmp.replace(path)
                    write(path.with_suffix('.json'), dict(created_at=stamp(), dataset=dataset, n=n, seed=seed,
                        source_sha256=SOURCE_HASH, npz_sha256=sha(path), bytes=path.stat().st_size,
                        class_names=names, benign_class=names.index('benign'),
                        label_counts=np.bincount(yl, minlength=len(names)).tolist(), unlabelled_rows=len(uid),
                        test_rows=len(test), test_counts=np.bincount(ty, minlength=len(names)).tolist(),
                        feature_names=list(suite['feature_columns']), categorical_columns=[0, 1],
                        unlabelled_sampling='train-only, no family labels; excludes all seed n<=20 label feature groups',
                        rejected_unlabelled_feature_copies=rejected,
                        historical_test_feature_overlap_allowed=True,
                        evaluation='full original test; previously inspected split, not independent confirmation',
                        label_sampling='EXP89 attack n prefixes; EXP90 fixed750 benign; no conflict-label filtering'))
            del tx, ty
        print(f'[{stamp()}] Inputs ready: {args.output}', flush=True)


if __name__ == '__main__':
    main()

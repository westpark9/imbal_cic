#!/usr/bin/env python3
"""Extend the approved EXP71 streams to an exact cumulative input budget."""
import argparse
import gc
import importlib.metadata as im
import importlib.util
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import exp71_bounded_updates as base

ROOT = base.ROOT
BASELINE = ROOT / 'tabpfn/results/v7/exp71/20261013_140543_bounded_uniform_natural_s43_731a4a80'


def allocations(available, total):
    if total <= 0 or total % 4:
        raise ValueError('Total must be positive and divisible by four')
    attacks = sorted(c for c in available if c != 'benign')
    capacities = [available[c] for c in attacks]
    result = {}
    for case, weights in [('uniform', [1] * len(attacks)), ('natural', capacities)]:
        counts = base.apportion(total // 4, weights, capacities)
        result[case] = dict(zip(attacks, map(int, counts)))
        result[case]['benign'] = total * 3 // 4
        if min(counts) < 1 or result[case]['benign'] > available['benign']:
            raise ValueError('Insufficient examples for the requested allocation')
        assert sum(result[case].values()) == total
    return result


def extend_class(candidates, old_ids, count, seed):
    """Retain every old bank row and append previously unused train rows."""
    if len(old_ids) > count or not np.isin(old_ids, candidates).all():
        raise ValueError('Old bank does not fit in the new train-only bank')
    unused = candidates[~np.isin(candidates, old_ids)]
    extra = np.random.default_rng(seed).choice(unused, count - len(old_ids), replace=False)
    return np.concatenate([old_ids, extra]).astype(np.int64)


def extend_benign(candidates, old_ids, count, stages, seed):
    """Preserve the old arrival stage of benign rows when expanding each stage."""
    all_ids = extend_class(candidates, old_ids, count, seed)
    old_parts = np.array_split(old_ids, stages)
    target_sizes = [len(p) for p in np.array_split(np.empty(count, dtype=np.int8), stages)]
    extra = all_ids[len(old_ids):]
    result = []; offset = 0
    for old, size in zip(old_parts, target_sizes):
        needed = size - len(old)
        if needed < 0:
            raise ValueError('A benign stage would shrink')
        result.append(np.concatenate([old, extra[offset:offset + needed]]))
        offset += needed
    assert offset == len(extra)
    return np.concatenate(result)


def prepare(args):
    out = args.out.resolve(); old = args.extend_from.resolve()
    config = dict(total=args.total, seed=args.seed, extend_from=str(old),
                  base_prepared_sha256=base.sha(old / 'prepared.json'))
    if (out / 'prepared.json').exists():
        if json.loads((out / 'scale_request.json').read_text()) != config:
            raise ValueError('Scale preparation configuration changed')
        return
    base.write(out / 'scale_request.json', config)
    base.write(out / 'status.json', dict(state='preparing', pid=os.getpid(), started=base.utc()))
    started = time.perf_counter()
    print('Preparing exact 100k cumulative streams; extending existing arrival IDs', flush=True)
    with args.data.open('rb') as f:
        suite = pickle.load(f)
    families = np.asarray(suite['families']); source_sha = base.sha(args.data)
    audits = []
    for ds in base.STAGES:
        clean = ROOT / 'data/derived' / base.CLEAN[ds]
        cm = json.loads((clean / 'manifest.json').read_text())
        if source_sha != cm['source']['sha256']:
            raise ValueError('Raw source hash changed')
        names = cm['class_names']
        splits = {s: np.load(clean / f'{s}_idx.npy') for s in ['train', 'val', 'test']}
        for s in splits:
            assert base.sha(clean / f'{s}_idx.npy') == cm['artifacts_sha256'][f'{s}_idx.npy']
        train_labels = families[splits['train']]
        available = {c: int(np.sum(train_labels == c)) for c in names}
        alloc = allocations(available, args.total)
        ref = old / 'data' / ds
        rm = json.loads((ref / 'manifest.json').read_text())
        assert rm['class_names'] == names and rm['available_train'] == available
        for f, digest in rm['files'].items():
            assert base.sha(ref / f) == digest, f
        assert base.sha(ref / 'plans.json') == rm['plans_sha256']
        old_ids = np.load(ref / 'train_ids.npy'); old_y = np.load(ref / 'train_y.npy')
        old_plans = json.loads((ref / 'plans.json').read_text())
        ids_parts = []; labels = []
        for i, c in enumerate(names):
            count = max(a[c] for a in alloc.values())
            candidates = splits['train'][train_labels == c]
            prior = old_ids[old_y == i]
            if c == 'benign':
                picked = extend_benign(candidates, prior, count, len(base.STAGES[ds]), args.seed + 1009*i)
            else:
                picked = extend_class(candidates, prior, count, args.seed + 1009*i)
            ids_parts.append(picked); labels.extend([i] * count)
        ids = np.concatenate(ids_parts); y = np.asarray(labels, dtype=np.int16)
        assert len(np.unique(ids)) == len(ids)
        assert all(not np.intersect1d(ids, splits[s]).size for s in ['val', 'test'])
        plans = base.make_plans(y, names, ds, alloc, available, args.seed, args.total)
        for case in base.CASES:
            assert len(plans[case][-1]['fit']) == args.total
            for stage, plan in enumerate(plans[case]):
                before = old_ids[old_plans[case][stage]['incoming']]
                after = ids[plan['incoming']]
                assert np.isin(before, after).all(), (ds, case, stage)
                assert plan['delivered_rows'] == len(plan['fit'])
                audits.append(dict(dataset=ds, case=case, stage=stage, old_incoming=len(before),
                                   new_incoming=len(after), cumulative=len(plan['fit']), old_ids_nested=True))
        dest = out / 'data' / ds; dest.mkdir(parents=True, exist_ok=True)
        source_ids = np.load(clean / 'source_idx.npy', mmap_mode='r')
        hashes = np.load(clean / 'vector_hash.npy', mmap_mode='r')
        for key, array in dict(train_ids=ids, train_y=y,
                train_X=np.nan_to_num(np.asarray(suite['X'][ids], dtype=np.float32)),
                train_hash=hashes[np.searchsorted(source_ids, ids)]).items():
            np.save(dest / f'{key}.npy', array)
        assert np.array_equal(np.load(ref / 'test_ids.npy'), splits['test'])
        for filename in ['test_X.npy', 'test_y.npy', 'test_ids.npy', 'test_hash.npy']:
            shutil.copyfile(ref / filename, dest / filename)
            assert base.sha(dest / filename) == rm['files'][filename]
        base.write(dest / 'plans.json', plans)
        base.write(dest / 'manifest.json', dict(dataset=ds, class_names=names,
            available_train=available, allocation=alloc, final_stream_rows=args.total,
            stages=base.STAGES[ds], first_appearance_utc=rm['first_appearance_utc'],
            fixed_test_class_counts=rm['fixed_test_class_counts'],
            clean_manifest_sha256=base.sha(clean / 'manifest.json'), source_sha256=source_sha,
            source_test_reference=str(ref.relative_to(ROOT)),
            plans_sha256=base.sha(dest / 'plans.json'), files={p.name: base.sha(p) for p in dest.glob('*.npy')}))
        print(ds, 'allocation', alloc, 'stage cumulative sizes',
              {c: [len(p['fit']) for p in plans[c]] for c in base.CASES}, flush=True)
    del suite; gc.collect()
    base.write(out / 'scale_validation.json', dict(status='passed', stages=audits,
        checks=['exact total and 75/25 split', 'same immutable full test', 'train-only unique IDs',
                'old incoming IDs nested in same arrival stage', 'no future class', 'no dropped history']))
    base.write(out / 'prepared.json', dict(seed=args.seed, nominal_attack_labels=0,
        history_cap=args.total, total=args.total, source_sha256=source_sha, cases=base.CASES,
        allocation_mode='exact_total_capped_uniform_or_attack_natural',
        benign_policy='75% total, equally released by stage; same IDs across cases',
        natural_policy='25% total allocated by clean train attack proportions; no tail floor',
        memory_policy='retain all delivered examples', seconds=time.perf_counter()-started, completed=base.utc()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--total', type=int, default=100000)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--extend-from', type=Path, default=BASELINE)
    p.add_argument('--data', type=Path, default=ROOT/'data/nfv3_energy_suite_uncapped_scenarios.pkl')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--batch', type=int, default=32768)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    # Query the environment without creating a parent-process CUDA context.
    package_root = Path(importlib.util.find_spec('tabpfn').origin).parent
    base.write(args.out/'environment.json', dict(platform=platform.platform(),
        packages={n: im.version(n) for n in ['torch','tabpfn','numpy','pandas','scikit-learn','xgboost']},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'], text=True).strip(),
        tabpfn_python_sources_sha256={str(f.relative_to(package_root)):base.sha(f) for f in sorted(package_root.rglob('*.py'))}))
    for src in [Path(__file__), ROOT/'docs/research/20261014/exp71_100k_protocol.md']:
        dst = args.out/'source'/src.resolve().relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and base.sha(dst) != base.sha(src):
            raise ValueError('Source changed since start')
        shutil.copy2(src, dst)
    prepare(args)
    args.stream_root = args.out; args.history_cap = args.total; args.nominal = 0
    args.history_mode = 'cumulative'
    return base.run(args)


if __name__ == '__main__':
    sys.exit(main())

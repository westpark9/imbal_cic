#!/usr/bin/env python3
"""Materialize the seed-43 context pools as .npy so context experiments (EXP64+) run without the 14.7 GB pkl.

Mirrors exp62_sv_current_bank.prepare(): same frozen helpers, same scenario-stratified partition,
same shared_context_ids.npz (C0, anchor) as the EXP59/62/63 runs. Output per dataset:
  data/derived/pools_s43/<ds>/{global_pool,c0,anchor}_{ids,X,y,time,scenario}.npy + META.json
global_pool = the whole D_global partition the 100k C0 was sampled from.
"""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import argparse
import gc
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = repo_root(__file__)
FROZEN = ROOT / 'tabpfn/results/v4/exp57/20260922_143529_exp57_expert_quality_s42_44/source/tabpfn/scripts'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', nargs='+', default=['toniot', 'cic2018'])
    ap.add_argument('--out', type=Path, default=ROOT / 'data/derived/pools_s43')
    args = ap.parse_args()
    print('Args:', vars(args), flush=True)
    sys.path.insert(0, str(FROZEN))
    suite = None
    for ds in args.datasets:
        t0 = time.time()
        base = ROOT / f'tabpfn/results/v4/exp59/20260928_exp59_residual_oracle_s43/{ds}_s43'
        source = base / 'fresh_bank'
        cfg = read_record(source / 'job.json')['args']
        design = read_record(source / 'design.json')
        names = design['class_names']; C = len(names)
        h = types.ModuleType('pools_helpers_' + ds); h.__file__ = str(FROZEN / 'nfv3_v3_exp31_c0alloc.py'); sys.modules[h.__name__] = h
        exec(compile((base / 'executed_fresh_prefix.py').read_text(), h.__file__, 'exec'), h.__dict__)
        if suite is None:
            print('loading', cfg['data'], flush=True)
            suite = h.core.load_pickle(cfg['data'])
            print(f'loaded in {time.time()-t0:.0f}s', flush=True)
        X = suite['X']; families = np.asarray(suite['families']); times = np.asarray(suite['timestamps']); scenarios = np.asarray(suite['attack_scenarios'])
        class_index = {n: i for i, n in enumerate(names)}

        def labels(ids):
            return np.asarray([class_index[n] for n in families[ids]], dtype='int16')

        def feats(ids):
            return np.nan_to_num(np.asarray(X[ids], dtype='float32'))

        clean = Path(cfg['clean_manifest']); clean = clean.parent if clean.is_file() else clean
        tr = np.load(clean / 'train_idx.npy')
        pools, _, _ = h.scenario_stratified_partition(tr, labels(tr), times[tr], scenarios[tr], cfg['context_frac'], cfg['expert_frac'], names)
        shared = np.load(source / 'shared_context_ids.npz')
        gpool = np.sort(np.asarray(pools['context']))
        assert np.isin(shared['global_context'], gpool).all(), 'C0 must lie inside the D_global partition'
        assert not np.intersect1d(gpool, np.load(clean / 'test_idx.npy')).size
        out = args.out / ds; out.mkdir(parents=True, exist_ok=True)
        sets = [('global_pool', gpool), ('c0', shared['global_context']), ('anchor', shared['anchor'])]
        for name, ids in sets:
            np.save(out / f'{name}_ids.npy', ids); np.save(out / f'{name}_X.npy', feats(ids)); np.save(out / f'{name}_y.npy', labels(ids))
            np.save(out / f'{name}_time.npy', times[ids]); np.save(out / f'{name}_scenario.npy', scenarios[ids].astype(str))
        meta = dict(dataset=ds, class_names=names, rows={n: int(len(v)) for n, v in sets},
                    class_counts={n: np.bincount(labels(v), minlength=C).tolist() for n, v in sets},
                    source=str(source.relative_to(ROOT)), clean_manifest=cfg['clean_manifest'],
                    partition='scenario_stratified context/expert/route = %s/%s' % (cfg['context_frac'], cfg['expert_frac']),
                    created_epoch=time.time(), seconds=time.time() - t0)
        (out / 'META.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
        print(ds, json.dumps(meta['rows']), 'c0 counts', meta['class_counts']['c0'], f'{time.time()-t0:.0f}s', flush=True)
        gc.collect()
    print('done', flush=True)


if __name__ == '__main__':
    main()

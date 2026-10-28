"""Evaluate frozen few-shot XGB models on CIC2018 before conflict filtering.

Only the evaluation population changes. Training is still globally curated;
this is not a replacement leakage-free benchmark or a new training protocol.
"""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import repo_root

import argparse
from datetime import datetime
import gc
import json
import pickle
import shutil
import time
from zoneinfo import ZoneInfo
import numpy as np
from xgboost import XGBClassifier
from audit_fewshot_validity import read, write, sha, one, metrics

ROOT = repo_root(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    curve = one('tabpfn/results/v7/exp87/*full_learning*/results.json').parent
    pilot = ROOT / read(curve / 'results.json')['source_run']
    md = read(pilot / 'data/cic2018/manifest.json')
    clean = Path(md['clean_manifest']).parent
    clean_md = read(clean / 'manifest.json')
    source = ROOT / 'data/nfv3_energy_suite_uncapped_scenarios.pkl'
    config = dict(started_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                  source=str(source), source_curve=str(curve), seeds=[43,44,45,46,47],
                  budgets=[2,20], method='frozen_saved_xgb', research_version='v7',
                  scope='Original attack-only outer test before conflict removal; same trained models',
                  caveat='Training samples remain selected from globally conflict-filtered pool; no causal claim about train curation',
                  permission='2026-10-27 user requested review of data difficulty, feature bias, and code')
    write(out / 'config.json', config)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('audit_fewshot_validity.py'), out / 'audit_fewshot_validity.py')
    assert sha(source) == clean_md['source']['sha256']
    print('loading original suite', flush=True)
    with source.open('rb') as stream:
        suite = pickle.load(stream)
    original = np.load(clean / 'original_test_idx.npy')
    names = md['class_names']
    families = np.asarray(suite['families'])[original]
    ids = original[families != 'benign']
    families = families[families != 'benign']
    y = np.array([names.index(x) for x in families], dtype=np.int16)
    X = np.nan_to_num(np.asarray(suite['X'][ids], dtype=np.float32))
    scenarios = np.asarray(suite['attack_scenarios'])[ids].copy()
    del suite
    gc.collect()
    classes = np.asarray(md['attack_classes'])
    retained = np.isin(ids, np.load(clean / 'test_idx.npy'))
    expected_ids = np.load(pilot / 'data/cic2018/test_ids.npy')
    np.testing.assert_array_equal(ids[retained], expected_ids)
    np.testing.assert_array_equal(X[retained], np.load(pilot / 'data/cic2018/test_X.npy'))
    np.testing.assert_array_equal(y[retained], np.load(pilot / 'data/cic2018/test_y.npy'))
    np.save(out / 'test_source_ids.npy', ids)
    np.save(out / 'test_y.npy', y)
    np.save(out / 'retained_mask.npy', retained)
    write(out / 'population.json', dict(original_attack_rows=len(ids), retained_rows=int(retained.sum()),
          removed_rows=int((~retained).sum()), class_names=names,
          original_counts=np.bincount(y, minlength=len(names)).tolist(),
          retained_counts=np.bincount(y[retained], minlength=len(names)).tolist(),
          source_sha256=clean_md['source']['sha256'], exact_retained_tensor_match=True))
    results = []
    for j in read(curve / 'results.json')['jobs']:
        if j['dataset'] != 'cic2018' or j['method'] != 'xgb' or j['n'] not in [2,20]:
            continue
        dest = out / 'jobs' / f"s{j['seed']}_n{j['n']}"
        dest.mkdir(parents=True)
        ppath = Path(j['predictions_path'])
        model = XGBClassifier()
        model.load_model(ppath.parent / 'model.ubj')
        model.set_params(device='cpu', n_jobs=8)
        p = classes[model.predict(X).astype(int)]
        np.testing.assert_array_equal(p[retained], np.load(ppath))
        np.save(dest / 'predictions.npy', p)
        score = {label: metrics(y[mask], p[mask], classes) for label, mask in
                 [('original', np.ones(len(ids), dtype=bool)), ('retained', retained), ('removed', ~retained)]}
        row = dict(seed=j['seed'], n=j['n'], frozen_model=str(ppath.parent / 'model.ubj'),
                   model_sha256=sha(ppath.parent / 'model.ubj'), **score,
                   scenarios=[dict(scenario=str(s), rows=int((scenarios==s).sum()),
                         correct=int(((scenarios==s) & (y==p)).sum())) for s in np.unique(scenarios)])
        write(dest / 'results.json', row)
        results.append(row)
        write(out / 'results.json', results)
        print('evaluated', j['seed'], j['n'], score['original']['macro_f1'], flush=True)
    write(out / 'COMPLETE.json', dict(finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
          jobs=len(results), elapsed_seconds=time.time()-started, paper_adoption='not_adopted_exploratory'))


if __name__ == '__main__':
    main()

"""Shared pieces for the context feasibility prototypes (EXP64–67). New module; nothing here edits older helpers.

Data sources (seed 43, conflict-free fixed splits):
  pools  : data/derived/pools_s43/<ds>/{global_pool,c0,anchor}_{X,y,ids,time,scenario}.npy  (scripts/materialize_nfv3_pools_s43.py)
  caches : tabpfn/results/20260930_exp63_k_sweep_s43/<ds>_k6/cache/{route,cal,eval}_{X,y,ids,time,scenario}.npy
Evaluation: stratified test subsample (all rows of small classes, capped large classes) for screening; --full-test for the winner.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POOLS = ROOT / 'data/derived/pools_s43'
CACHES = {'toniot': ROOT / 'tabpfn/results/20260930_exp63_k_sweep_s43/toniot_k6/cache',
          'cic2018': ROOT / 'tabpfn/results/20260930_exp63_k_sweep_s43/cic2018_k6/cache'}
CKPT = ROOT / 'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'
TAILS = {'toniot': ['scanning', 'mitm', 'ransomware'], 'cic2018': ['infiltration', 'web_attacks']}


def load_dataset(ds):
    cache = CACHES[ds]; pool = POOLS / ds
    names = json.load(open(cache / 'COMPLETE.json'))['class_names']
    meta = json.load(open(pool / 'META.json')); assert meta['class_names'] == names
    d = dict(names=names, C=len(names), benign=names.index('benign'), tails=[names.index(t) for t in TAILS[ds]],
             c0_X=np.load(pool / 'c0_X.npy'), c0_y=np.load(pool / 'c0_y.npy').astype(int), c0_ids=np.load(pool / 'c0_ids.npy'),
             pool_X=np.load(pool / 'global_pool_X.npy', mmap_mode='r'), pool_y=np.load(pool / 'global_pool_y.npy').astype(int),
             pool_ids=np.load(pool / 'global_pool_ids.npy'), pool_time=np.load(pool / 'global_pool_time.npy'), pool_scenario=np.load(pool / 'global_pool_scenario.npy'),
             route_X=np.load(cache / 'route_X.npy', mmap_mode='r'), route_y=np.load(cache / 'route_y.npy').astype(int),
             eval_X=np.load(cache / 'eval_X.npy', mmap_mode='r'), eval_y=np.load(cache / 'eval_y.npy').astype(int))
    return d


def standardizer(X_ref):
    mu = np.asarray(X_ref, dtype=np.float64).mean(0); sd = np.asarray(X_ref, dtype=np.float64).std(0) + 1e-6
    return lambda X: np.clip((np.asarray(X, dtype=np.float32) - mu) / sd, -10, 10).astype(np.float32)


def test_subsample(d, per_class, rng, full=False):
    y = d['eval_y']
    if full:
        idx = np.arange(len(y))
    else:
        idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())), replace=False) for c in range(d['C'])]))
    return idx, np.asarray(d['eval_X'][idx]), y[idx]


def class_table(y, pred, names):
    rows = []
    for i, n in enumerate(names):
        tp = int(((pred == i) & (y == i)).sum()); fp = int(((pred == i) & (y != i)).sum()); fn = int(((pred != i) & (y == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0; f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append(dict(cls=n, support=int((y == i).sum()), precision=p, recall=r, f1=f, TP=tp, FP=fp, FN=fn))
    return rows


def summarize(y, pred, d):
    rows = class_table(y, pred, d['names'])
    return dict(macro_f1=float(np.mean([r['f1'] for r in rows])), accuracy=float((pred == y).mean()),
                tail_f1=float(np.mean([rows[i]['f1'] for i in d['tails']])),
                benign_fpr=float(((pred == d['benign']) & (y != d['benign'])).sum() / max(1, (y != d['benign']).sum())),
                classes=rows)


def tabpfn_global(ctx_X, ctx_y, seed, n_estimators=4, device='cuda'):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(model_path=str(CKPT), device=device, n_estimators=n_estimators, random_state=seed,
                           ignore_pretraining_limits=True, fit_mode='fit_with_cache')
    t0 = time.time(); clf.fit(ctx_X, ctx_y); return clf, time.time() - t0


def predict(clf, X, batch=100000, proba=False):
    outs = []
    for i in range(0, len(X), batch):
        p = clf.predict_proba(X[i:i + batch]); outs.append(p if proba else p.argmax(1))
    return np.concatenate(outs)


def run_dir(ds, tag, seed):
    out = ROOT / f'tabpfn/results/{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}_{ds}_{tag}_s{seed}'
    out.mkdir(parents=True, exist_ok=True); return out


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')

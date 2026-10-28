"""Causal class-arrival replay, independent of model implementations.

Updates and inference have separate logical workers. A completed input batch
binds the newest ready model at arrival; inference is FIFO, with no drops.
No later model is substituted for a queued batch. Recorded service durations
include the complete batch prediction (including DistPFN's current-batch prior).
"""
from pathlib import Path
import sys

_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)

import numpy as np
from exp69_class_arrival import STAGES

ROOT = repo_root(__file__)


def partition_test(y, names, dataset, seed=43, batch_size=1024):
    """Each original test ID occurs once, at/after its class's arrival stage."""
    groups = STAGES[dataset]
    parts = [[] for _ in groups]
    for c, name in enumerate(names):
        first = next(i for i, g in enumerate(groups) if name in g)
        ids = np.flatnonzero(y == c)
        np.random.default_rng(seed + 1009*c).shuffle(ids)
        for stage, chunk in enumerate(np.array_split(ids, len(groups)-first), first):
            parts[stage].append(chunk)
    order, batches, stages = [], [], []
    offset = 0
    for stage, chunks in enumerate(parts):
        rows = np.concatenate(chunks)
        np.random.default_rng(seed + 104729*(stage+1)).shuffle(rows)
        start = offset
        order.append(rows)
        for lo in range(0, len(rows), batch_size):
            hi = min(len(rows), lo+batch_size)
            batches.append((stage, offset+lo, offset+hi))
        offset += len(rows)
        stages.append((start, offset))
    order = np.concatenate(order).astype(np.int64)
    assert len(order) == len(y) and len(np.unique(order)) == len(y)
    return order, np.asarray(batches, dtype=np.int64), np.asarray(stages, dtype=np.int64)


def update_schedule(stage_bounds, update_seconds, rate):
    """Version zero is ready before traffic. Later builds run in FIFO order."""
    if rate <= 0 or len(stage_bounds) != len(update_seconds):
        raise ValueError('Invalid rate or update list')
    ready = [0.]
    for stage in range(1, len(stage_bounds)):
        released = float(stage_bounds[stage][0])/rate
        ready.append(max(released, ready[-1])+float(update_seconds[stage]))
    return np.asarray(ready)


def dispatch_versions(batches, ready, rate):
    close_times = batches[:, 2]/float(rate)
    version = np.searchsorted(ready, close_times, side='right')-1
    # The batch may end exactly at the next stage boundary. A future stage's
    # version must never be used to classify this stage's completed batch.
    return np.minimum(version, batches[:, 0]).astype(np.int16)


def fifo_response_times(batches, service_seconds, rate):
    seconds = np.asarray(service_seconds, dtype=float)
    if len(seconds) != len(batches) or not np.isfinite(seconds).all() or np.any(seconds < 0):
        raise ValueError('Incomplete/invalid batch service timings')
    finish = np.empty(len(batches)); wait = np.empty(len(batches))
    previous = 0.
    for i, (_, lo, hi) in enumerate(batches):
        arrival = float(hi)/rate
        start = max(arrival, previous)
        wait[i] = start-arrival
        finish[i] = previous = start+seconds[i]
    # End-to-end average row latency includes batch collection, queue, compute.
    mean_row_arrival = (batches[:, 1]+1+batches[:, 2])/(2.*rate)
    return dict(finish=finish, queue_wait=wait,
                mean_row_latency=finish-mean_row_arrival,
                last_row_latency=finish-batches[:, 2]/rate)


def dist_adjust_current_batch(proba, train_prior):
    """Official DistPFN correction, using only this arrived batch, no labels."""
    p = np.asarray(proba, dtype=np.float64)
    adjusted = p*p.mean(axis=0)/(np.asarray(train_prior)+1e-8)
    return (adjusted/adjusted.sum(axis=1, keepdims=True)).astype(np.float32)


def confusion_metrics(y, pred, names, allowed=None):
    """Class precision includes every row, including false positives from others."""
    C = len(names)
    cm = np.bincount(np.asarray(y, dtype=np.int64)*C+pred, minlength=C*C).reshape(C, C)
    tp = np.diag(cm); support = cm.sum(1); fp = cm.sum(0)-tp; fn = support-tp
    divide = lambda a, b: np.divide(a, b, out=np.zeros(C, dtype=float), where=b!=0)
    p, r, f1 = divide(tp, tp+fp), divide(tp, support), divide(2*tp, 2*tp+fp+fn)
    selected = np.asarray(allowed if allowed is not None else np.flatnonzero(support), dtype=int)
    benign = names.index('benign')
    attack = np.arange(C) != benign
    return dict(rows=int(support.sum()), confusion=cm.tolist(),
                macro_f1=float(f1[selected].mean()) if len(selected) else None,
                benign_false_positive_rate=float((support[benign]-tp[benign])/support[benign]) if support[benign] else None,
                attack_to_benign=int(cm[attack, benign].sum()),
                wrong_attack_family=int(cm[np.ix_(attack, attack)].sum()-tp[attack].sum()),
                attack_detection_recall=float(1-cm[attack, benign].sum()/support[attack].sum()) if support[attack].sum() else None,
                classes=[dict(name=n, support=int(support[i]), precision=float(p[i]),
                              recall=float(r[i]) if support[i] else None,
                              f1=float(f1[i]) if support[i] or fp[i] else None,
                              TP=int(tp[i]), FP=int(fp[i]), FN=int(fn[i])) for i,n in enumerate(names)])

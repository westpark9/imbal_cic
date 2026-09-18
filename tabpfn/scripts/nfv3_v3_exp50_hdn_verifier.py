#!/usr/bin/env python3
"""EXP50: verifier-target ablation on an EXP39-format frozen cache (S0V0 vs S0V1).

Forked from EXP39 (nfv3_v3_exp39_decision_routing.py, unmodified) to isolate the
verifier learning target while holding the scorer and post-features fixed (S0):
- legacy (S0V0): existing OOF sign-classification scorer, weighted-NLL-gain
  quantile-regression verifier ("normgain"). Unmodified control arm.
- hdn (S0V1): same scorer, but the verifier is a 3-class classifier over
  {harmful, neutral, helpful} = sign(decision_gain), scored as
  P(helpful) - hdn_lambda * P(harmful). Isolates whether classifying the actual
  correction/harm/neutral outcome (instead of regressing NLL gain) finds more
  usable positive-score region under EXP47/48's near-empty raw-verifier-positive
  calibration counts. See docs/research/20260916/expert_acceptance_next_plan.md §1.

S1 (scorer target changed to correction-probability minus harm-risk) is a
follow-up not implemented here; only V0/V1 on the existing S0 scorer.
All other mechanics (threshold search, chronological cal split, evaluation) are
byte-identical to EXP39.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nfv3_v3_exp38_frozen_cache import (
    build_pair_pre, build_pair_post, sha256, write_json, save_array, uniq_rows,
)

SPECS = {
    'legacy': ('base', 'legacy', 'nll'),
    'post_z': ('z', 'legacy', 'nll'),
    'post_raw': ('raw', 'legacy', 'nll'),
    'decision_post_raw': ('raw', 'legacy', 'decision'),
    'decision_both_raw': ('raw', 'decision', 'decision'),
    'dense_decision_raw': ('raw', 'dense', 'decision'),  # diagnostic: bypass pre-call scorer
    'hdn': ('base', 'legacy', 'hdn'),  # S0V1: legacy scorer, H/D/N-classification verifier
}


def decision_gain(y, global_pred, expert_pred):
    return ((expert_pred == y).astype(np.float32)
            - (global_pred == y).astype(np.float32))


def confusion(y, pred, n_classes, weights=None):
    return np.bincount(y.astype(np.int64) * n_classes + pred,
                       weights=weights, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def metrics_from_cm(cm, tail_ids, benign_id):
    cm = np.asarray(cm, dtype=np.float64)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    tp = np.diag(cm)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(tp)),
                   where=support + predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    return {'macro_f1': float(f1.mean()), 'tail_f1': float(f1[tail_ids].mean()),
            'benign_fpr': float(1 - recall[benign_id]),
            'accuracy': float(tp.sum() / max(cm.sum(), 1)),
            'f1': f1, 'recall': recall, 'precision': precision, 'support': support}


def population_weights(y, prior, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError('Cannot standardize class risk with absent classes')
    return (len(y) * np.asarray(prior) / counts)[y]


def chronological_cal_split(y, scenario, timestamp, hashes, eligible, fraction):
    """Early selection / later confirmation within each observed class-scenario.

    Remove confirmation feature hashes present in selection; no fallback unmask.
    This is scenario-conditional chronology, not global calendar ordering.
    """
    choose, confirm = [], []
    frame = pd.DataFrame({'y': y, 'scenario': scenario})
    for members in frame.groupby(['y', 'scenario'], sort=True).indices.values():
        rows = members[eligible[members]]
        rows = rows[np.argsort(timestamp[rows], kind='stable')]
        if not len(rows):
            continue
        cut = max(1, int(np.floor(len(rows) * (1 - fraction))))
        if cut < len(rows) and timestamp[rows[cut - 1]] == timestamp[rows[cut]]:
            cut = int(np.searchsorted(timestamp[rows], timestamp[rows[cut]], side='right'))
        choose.extend(rows[:cut]); confirm.extend(rows[cut:])
    choose = np.sort(np.asarray(choose, dtype=np.int64))
    confirm = np.sort(np.asarray(confirm, dtype=np.int64))
    if len(confirm):
        confirm = confirm[~np.isin(hashes[confirm], np.unique(hashes[choose]))]
    return choose, confirm


def scalar_metrics(cm, tail_ids, benign_id):
    m = metrics_from_cm(cm, tail_ids, benign_id)
    return {key: m[key] for key in ['macro_f1', 'tail_f1', 'benign_fpr', 'accuracy']}


def select_thresholds(y, global_pred, candidate_pred, pre, post, weights,
                      n_classes, tail_ids, benign_id, protected_ids,
                      pre_quantiles, post_quantiles, fpr_limit, protected_drop,
                      max_proposal, min_changed):
    """Pure calibration function: no access to test labels or test predictions."""
    baseline_cm = confusion(y, global_pred, n_classes, weights)
    baseline = metrics_from_cm(baseline_cm, tail_ids, benign_id)
    tp_grid = np.unique(np.r_[-np.inf, 0., np.quantile(pre, pre_quantiles)])
    tq_grid = np.unique(np.r_[0., np.maximum(0., np.quantile(post, post_quantiles))])
    rows = []
    # Explicit feasible no-intervention candidate, not counted as a gain.
    best = {'tau_pre': None, 'tau_post': None, 'delta_macro_f1': 0.,
            'delta_tail_f1': 0., 'delta_benign_fpr': 0., 'accepted': 0,
            'changed': 0, 'helpful': 0, 'harmful': 0, 'feasible': True,
            'proposal_rate': 0., 'reason': 'global_fallback'}
    for tp in tp_grid:
        proposed = pre > tp
        rate = float(weights[proposed].sum() / weights.sum())
        for tq in tq_grid:
            accepted = proposed & (post > tq)
            pred = np.where(accepted, candidate_pred, global_pred)
            m = metrics_from_cm(confusion(y, pred, n_classes, weights), tail_ids, benign_id)
            helpful = int(((global_pred != y) & (pred == y)).sum())
            harmful = int(((global_pred == y) & (pred != y)).sum())
            changed = int((pred != global_pred).sum())
            df1 = m['f1'] - baseline['f1']
            delta_macro = m['macro_f1'] - baseline['macro_f1']
            delta_tail = m['tail_f1'] - baseline['tail_f1']
            delta_fpr = m['benign_fpr'] - baseline['benign_fpr']
            reason = []
            if rate > max_proposal + 1e-12: reason.append('proposal')
            if delta_fpr > fpr_limit + 1e-12: reason.append('benign_fpr')
            if delta_tail < -1e-12: reason.append('tail_f1')
            if any(df1[c] < -protected_drop - 1e-12 for c in protected_ids):
                reason.append('protected_class_f1')
            if helpful + harmful < min_changed: reason.append('support')
            row = {'tau_pre': float(tp) if np.isfinite(tp) else '-inf',
                   'tau_post': float(tq), 'delta_macro_f1': float(delta_macro),
                   'delta_tail_f1': float(delta_tail), 'delta_benign_fpr': float(delta_fpr),
                   'accepted': int(accepted.sum()), 'changed': changed,
                   'helpful': helpful, 'harmful': harmful, 'proposal_rate': rate,
                   'harmful_fraction': harmful / max(helpful + harmful, 1),
                   'feasible': not reason, 'reason': ','.join(reason)}
            rows.append(row)
            if not reason and delta_macro > best['delta_macro_f1'] + 1e-12:
                best = row.copy()
    rows.append({**best, 'reason': 'selected:' + best['reason']})
    return best, pd.DataFrame(rows)


def apply_thresholds(global_pred, candidate_pred, pre, post, selection):
    if selection['tau_pre'] is None:
        return global_pred.copy(), np.zeros(len(pre), bool), np.zeros(len(pre), bool)
    tp = float(selection['tau_pre'])
    proposed = pre > tp
    accepted = proposed & (post > selection['tau_post'])
    return np.where(accepted, candidate_pred, global_pred), proposed, accepted


class ConstantScore:
    def __init__(self, value): self.value = float(value)
    def predict(self, X): return np.full(len(X), self.value, dtype=np.float32)


class HDNScore:
    """Wraps a 3-class {harmful=0, neutral=1, helpful=2} classifier as a scalar
    post-score P(helpful) - lam * P(harmful), so it drops into the existing
    real-valued threshold search (select_thresholds/apply_thresholds) unchanged."""
    def __init__(self, model, lam):
        self.model, self.lam = model, float(lam)

    def predict(self, X):
        proba = self.model.predict_proba(X)
        return (proba[:, 2] - self.lam * proba[:, 0]).astype(np.float32)


def predict_score(model, X, classification=False, chunk=100000):
    out = []
    for start in range(0, len(X), chunk):
        block = X[start:start + chunk]
        if classification and not isinstance(model, ConstantScore):
            out.append(model.predict_proba(block)[:, 1])
        else:
            out.append(model.predict(block))
    return np.concatenate(out).astype(np.float32)


class FrozenCache:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'COMPLETE.json').read_text())
        self.K = self.meta['n_experts']
        self.names = self.meta['class_names']
        self.C = len(self.names)
        self.qk = np.load(self.path / 'qk.npy')
        self.arrays = {}

    def array(self, split, key):
        name = f'{split}_{key}'
        if name not in self.arrays:
            self.arrays[name] = np.load(self.path / (name + '.npy'), mmap_mode='r')
        return self.arrays[name]

    def pre(self, split, k, rows):
        return build_pair_pre(self.array(split, 'p0')[rows], self.array(split, 'z')[rows],
                              self.array(split, 'distance')[rows, k],
                              self.array(split, 'affinity_0')[rows], self.qk[k])

    def post(self, split, k, rows, kind):
        features = build_pair_post(
            self.array(split, 'p0')[rows], self.array(split, f'p{k + 1}')[rows],
            self.array(split, 'affinity_0')[rows],
            self.array(split, f'affinity_{k + 1}')[rows],
            self.array(split, 'distance')[rows, k], self.qk[k])
        if kind == 'z':
            features = np.concatenate([features, self.array(split, 'z')[rows]], axis=1)
        elif kind == 'raw':
            # XGB trees do not require scaling; only EXP38's train-independent
            # nan_to_num conversion is applied to the original 46 columns.
            features = np.concatenate([features, self.array(split, 'X')[rows]], axis=1)
        return features


def build_model(args, objective, seed):
    shared = dict(n_estimators=args.trees, max_depth=args.depth,
                  learning_rate=args.learning_rate, tree_method='hist',
                  n_jobs=args.threads, random_state=seed)
    if objective == 'sign':
        return xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', **shared)
    if objective == 'quantile':
        return xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=.25, **shared)
    if objective == 'hdn':
        return xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', **shared)
    return xgb.XGBRegressor(objective='reg:squarederror', **shared)


def fit_model(args, X, target, objective, seed, weights=None):
    if np.all(target == target[0]):
        return ConstantScore(target[0])
    model = build_model(args, objective, seed)
    model.fit(X, target, sample_weight=weights)
    return model


def train_policies(cache, args, out):
    y = cache.array('route', 'y')
    rows = np.arange(len(y))
    p0 = cache.array('route', 'p0')
    g0 = p0.argmax(1)
    counts = np.bincount(y, minlength=cache.C)
    if np.any(counts == 0): raise ValueError('Route set is missing classes')
    balance = len(y) / (cache.C * counts)
    # Source NLL weighting is reproduced separately from the new decision loss.
    train_counts = np.asarray(cache.meta['train_counts'])
    source_weights = (train_counts.sum() / (cache.C * train_counts)
                      ) ** cache.meta['source_config']['residual_gamma']
    nll, delta = [], []
    for k in range(cache.K):
        pk = cache.array('route', f'p{k + 1}')
        nll.append((np.log(np.clip(pk[rows, y], 1e-12, 1))
                    - np.log(np.clip(p0[rows, y], 1e-12, 1))).astype(np.float32))
        delta.append(decision_gain(y, g0, pk.argmax(1)))
    nll = np.stack(nll, axis=1); delta = np.stack(delta, axis=1)
    save_array(out / 'route_decision_gain.npy', delta)
    diagnostics = []
    for k in range(cache.K):
        for c in range(cache.C):
            mask = y == c
            diagnostics.append({'expert': k + 1, 'class': cache.names[c],
                'rows': int(mask.sum()), 'helpful': int((delta[mask, k] > 0).sum()),
                'harmful': int((delta[mask, k] < 0).sum()),
                'neutral': int((delta[mask, k] == 0).sum())})
    pd.DataFrame(diagnostics).to_csv(out / '1a_route_decision_targets.csv', index=False)
    hpre = np.concatenate([cache.pre('route', k, rows) for k in range(cache.K)])
    legacy_path = out / 'legacy_scorer.joblib'
    if legacy_path.exists():
        legacy_scorer = joblib.load(legacy_path)
    else:
        # Reproduce old two-fold teacher dependency only in the control arm.
        # All rows are outer train; reversed fold is documented, not called forward OOF.
        order = np.argsort(cache.array('route', 'time'), kind='stable')
        folds = (order[:len(order) // 2], order[len(order) // 2:])
        b = np.zeros_like(nll)
        for fit_i, pred_i in [(0, 1), (1, 0)]:
            Xfit = np.concatenate([cache.post('route', k, folds[fit_i], 'base')
                                   for k in range(cache.K)])
            target = np.concatenate([nll[folds[fit_i], k] * source_weights[y[folds[fit_i]]]
                                     for k in range(cache.K)])
            teacher = fit_model(args, Xfit, target, 'quantile', args.seed + 2600 + fit_i)
            del Xfit
            for k in range(cache.K):
                b[folds[pred_i], k] = predict_score(
                    teacher, cache.post('route', k, folds[pred_i], 'base')) > 0
            del teacher; gc.collect()
            print(f'legacy OOF fold {fit_i} complete', flush=True)
        save_array(out / 'legacy_b_oof.npy', b)
        target = np.concatenate([(b[:, k] * nll[:, k]) > 0 for k in range(cache.K)])
        weight = np.concatenate([np.log1p(np.abs(nll[:, k] * source_weights[y]))
                                 for k in range(cache.K)]) + 1e-3
        legacy_scorer = fit_model(args, hpre, target, 'sign', args.seed + 1700, weight)
        joblib.dump(legacy_scorer, legacy_path)
    direct_path = out / 'decision_scorer.joblib'
    if direct_path.exists():
        direct_scorer = joblib.load(direct_path)
    else:
        target = np.concatenate([delta[:, k] for k in range(cache.K)])
        direct_scorer = fit_model(args, hpre, target, 'mean', args.seed + 1700,
                                  np.tile(balance[y], cache.K))
        joblib.dump(direct_scorer, direct_path)
    del hpre; gc.collect()
    policies = {}
    for arm in args.arms.split(','):
        kind, scorer_kind, target_kind = SPECS[arm]
        verifier_path = out / f'verifier_{kind}_{target_kind}.joblib'
        if verifier_path.exists():
            verifier = joblib.load(verifier_path)
        else:
            hp = np.concatenate([cache.post('route', k, rows, kind) for k in range(cache.K)])
            if target_kind == 'hdn':
                # H(helpful)=2, N(neutral, both-correct or both-wrong)=1, D(harmful)=0.
                delta_flat = np.concatenate([delta[:, k] for k in range(cache.K)])
                hdn_label = np.where(delta_flat > 0, 2, np.where(delta_flat < 0, 0, 1)).astype(np.int64)
                weight = np.tile(balance[y], cache.K)  # same class-risk weighting as 'decision' arm
                base = fit_model(args, hp, hdn_label, 'hdn', args.seed + 1800, weight)
                verifier = base if isinstance(base, ConstantScore) else HDNScore(base, args.hdn_lambda)
            else:
                target = np.concatenate([(nll if target_kind == 'nll' else delta)[:, k]
                                         for k in range(cache.K)])
                verifier = fit_model(args, hp, target,
                                      'quantile' if target_kind == 'nll' else 'mean',
                                      args.seed + 1800,
                                      None if target_kind == 'nll' else np.tile(balance[y], cache.K))
            joblib.dump(verifier, verifier_path)
            del hp; gc.collect()
        policies[arm] = (legacy_scorer if scorer_kind == 'legacy' else direct_scorer, verifier)
        print(f'policy fitted: {arm}', flush=True)
    return policies


def policy_scores(cache, split, arm, policy, args, out):
    path = out / f'{split}_{arm}_scores.npz'
    if path.exists():
        with np.load(path) as z: return {k: z[k] for k in z.files}
    kind, scorer_kind, _ = SPECS[arm]
    scorer, verifier = policy
    # Exact feature hashes allow reusing identical inference, never label hashes.
    _, first, inverse = np.unique(cache.array(split, 'hash'), return_index=True, return_inverse=True)
    if scorer_kind == 'dense':
        # Diagnostic only: observe all expert outputs and rank with the verifier.
        # Never present this all-call policy as the proposed sparse deployment.
        scores = np.empty((len(first), cache.K), np.float32)
        for k in range(cache.K):
            for begin in range(0, len(first), args.predict_chunk):
                rr = first[begin:begin + args.predict_chunk]
                scores[begin:begin + len(rr), k] = predict_score(
                    verifier, cache.post(split, k, rr, kind))
        winner = scores.argmax(1)
        candidate = np.empty(len(first), np.int32)
        for k in range(cache.K):
            pos = np.flatnonzero(winner == k)
            candidate[pos] = cache.array(split, f'p{k + 1}')[first[pos]].argmax(1)
        result = {'pre': np.ones(len(inverse), np.float32),
                  'post': scores.max(1)[inverse], 'candidate': candidate[inverse],
                  'winner': winner[inverse].astype(np.int16)}
        with Path(str(path) + '.tmp').open('wb') as stream:
            np.savez_compressed(stream, **result)
        Path(str(path) + '.tmp').replace(path)
        return result
    score_path = out / f'{split}_{scorer_kind}_pre_unique.npy'
    if score_path.exists():
        scores = np.load(score_path)
    else:
        scores = np.empty((len(first), cache.K), np.float32)
        for k in range(cache.K):
            for begin in range(0, len(first), args.predict_chunk):
                rr = first[begin:begin + args.predict_chunk]
                scores[begin:begin + len(rr), k] = predict_score(
                    scorer, cache.pre(split, k, rr), classification=scorer_kind == 'legacy')
            print(f'{split} {scorer_kind} pre expert {k + 1}/{cache.K}', flush=True)
        save_array(score_path, scores)
    winner = scores.argmax(1)
    pre = scores.max(1)
    post = np.empty(len(first), np.float32)
    candidate = np.empty(len(first), np.int32)
    # A real inference adapter would call only this selected expert here.
    # Offline cache lookup permits exact comparisons with identical expert outputs.
    for k in range(cache.K):
        positions = np.flatnonzero(winner == k)
        for begin in range(0, len(positions), args.predict_chunk):
            pos = positions[begin:begin + args.predict_chunk]
            rr = first[pos]
            post[pos] = predict_score(verifier, cache.post(split, k, rr, kind))
            candidate[pos] = cache.array(split, f'p{k + 1}')[rr].argmax(1)
    result = {'pre': pre[inverse], 'post': post[inverse],
              'candidate': candidate[inverse], 'winner': winner[inverse].astype(np.int16)}
    temp = Path(str(path) + '.tmp')
    with temp.open('wb') as stream: np.savez_compressed(stream, **result)
    temp.replace(path)
    return result


def evaluation_rows(cache, split, arm, y, global_pred, final, proposed, accepted,
                    tail_ids, benign_id, subset=None):
    if subset is not None:
        y, global_pred, final = y[subset], global_pred[subset], final[subset]
        proposed, accepted = proposed[subset], accepted[subset]
    if len(y) == 0: return [], {'arm': arm, 'split': split, 'rows': 0}
    baseline = metrics_from_cm(confusion(y, global_pred, cache.C), tail_ids, benign_id)
    current = metrics_from_cm(confusion(y, final, cache.C), tail_ids, benign_id)
    summary = {'arm': arm, 'split': split, 'rows': len(y),
        **{k: current[k] for k in ['macro_f1', 'tail_f1', 'benign_fpr', 'accuracy']},
        'delta_macro_f1': current['macro_f1'] - baseline['macro_f1'],
        'delta_tail_f1': current['tail_f1'] - baseline['tail_f1'],
        'delta_benign_fpr': current['benign_fpr'] - baseline['benign_fpr'],
        'proposed': int(proposed.sum()), 'accepted': int(accepted.sum()),
        'logical_expert_calls': int(proposed.sum()) * (cache.K if arm == 'dense_decision_raw' else 1),
        'diagnostic_all_expert_policy': arm == 'dense_decision_raw',
        'changed': int((final != global_pred).sum()),
        'helpful': int(((global_pred != y) & (final == y)).sum()),
        'harmful': int(((global_pred == y) & (final != y)).sum())}
    rows = []
    for c, name in enumerate(cache.names):
        mask = y == c
        h = int((mask & (global_pred != y) & (final == y)).sum())
        d = int((mask & (global_pred == y) & (final != y)).sum())
        rows.append({'arm': arm, 'split': split, 'class': name,
            'support': int(mask.sum()), 'global_f1': baseline['f1'][c],
            'system_f1': current['f1'][c], 'delta_f1': current['f1'][c] - baseline['f1'][c],
            'precision': current['precision'][c], 'recall': current['recall'][c],
            'helpful': h, 'harmful': d, 'net_correction': (h - d) / max(mask.sum(), 1)})
    return rows, summary


def run(args):
    print('Args: ' + json.dumps(vars(args), sort_keys=True), flush=True)
    started = time.monotonic()
    cache = FrozenCache(args.cache_dir)
    requested = args.arms.split(',')
    if not set(requested) <= SPECS.keys(): raise ValueError('Unknown arm')
    out = Path(args.out_root) / (time.strftime('%Y%m%d_%H%M%S') +
                                f'_{os.getpid()}_exp50_hdn_verifier')
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'cache_identity': json.loads((cache.path / 'identity.json').read_text()),
              'script_sha256': sha256(__file__), 'test_role': cache.meta['test_role'],
              'calibration_prior_assumption': 'class-conditional transfer from masked calibration to historical train prior',
              'legacy_control_note': 'new shared calibration and shared learner hyperparameters; not exact EXP31 replay'}
    write_json(out / 'args.json', config)
    shutil.copy2(__file__, out / 'experiment_source.py')
    shutil.copy2(Path(__file__).with_name('nfv3_v3_exp38_frozen_cache.py'), out / 'cache_helpers_source.py')
    write_json(out / 'RUNNING.json', {'pid': os.getpid(), 'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    print('RUN_DIR: ' + str(out), flush=True)
    ycal = cache.array('cal', 'y')
    select, confirm = chronological_cal_split(
        ycal, cache.array('cal', 'scenario'), cache.array('cal', 'time'),
        cache.array('cal', 'hash'), cache.array('cal', 'mask'), args.confirm_fraction)
    save_array(out / 'cal_select_positions.npy', select)
    save_array(out / 'cal_confirm_positions.npy', confirm)
    split_rows = [{'split': split, 'class': name, 'rows': int((ycal[rr] == c).sum())}
                  for split, rr in [('select', select), ('confirm', confirm)]
                  for c, name in enumerate(cache.names)]
    pd.DataFrame(split_rows).to_csv(out / '0a_calibration_split.csv', index=False)
    if any((ycal[select] == c).sum() == 0 for c in range(cache.C)):
        raise ValueError('Selection split missing a class after strict masks')
    prior = np.asarray(cache.meta['train_counts'], float)
    prior /= prior.sum()
    weights = population_weights(ycal[select], prior, cache.C)
    tail_ids = [cache.names.index(n) for n in cache.meta['tail_classes']]
    benign_id = cache.names.index('benign')
    protected_ids = [cache.names.index(n) for n in args.protected_classes.split(',')]
    policies = train_policies(cache, args, out)
    per_class, summaries, thresholds = [], [], {}
    gcal = cache.array('cal', 'p0').argmax(1)
    # Finish ALL policy fitting and threshold selection before reading eval labels.
    for arm in requested:
        score = policy_scores(cache, 'cal', arm, policies[arm], args, out)
        best, grid = select_thresholds(
            ycal[select], gcal[select], score['candidate'][select],
            score['pre'][select], score['post'][select], weights, cache.C,
            tail_ids, benign_id, protected_ids,
            np.array([0., .3, .5, .65, .75, .8, .85, .9, .95, .98, .99]),
            np.array([0., .1, .25, .5, .75, .9, .95, .99]),
            args.benign_fpr_increase, args.protected_f1_drop,
            args.max_proposal, args.min_decided)
        grid.to_csv(out / f'4a_{arm}_threshold_grid.csv', index=False)
        thresholds[arm] = best
        final, proposed, accepted = apply_thresholds(gcal, score['candidate'], score['pre'], score['post'], best)
        rows, summary = evaluation_rows(cache, 'cal_confirm', arm, ycal, gcal, final,
                                         proposed, accepted, tail_ids, benign_id, confirm)
        per_class.extend(rows); summaries.append(summary)
        print(f'{arm} selected: {json.dumps(best)}', flush=True)
    write_json(out / 'selected_thresholds.json', thresholds)
    # Test is full-sized and only evaluated after all choices above are frozen.
    ytest = cache.array('eval', 'y')
    gtest = cache.array('eval', 'p0').argmax(1)
    npreds = np.stack([cache.array('eval', f'p{k + 1}').argmax(1) for k in range(cache.K)], axis=1)
    any_correct = (npreds == ytest[:, None]).any(1)
    oracle = np.where(any_correct, ytest, gtest)
    for name, pred in [('global', gtest), ('correctness_oracle', oracle)] + [
            (f'expert{k + 1}_always', npreds[:, k]) for k in range(cache.K)]:
        rows, summary = evaluation_rows(cache, 'full_test', name, ytest, gtest, pred,
            np.zeros(len(ytest), bool), pred != gtest, tail_ids, benign_id)
        per_class.extend(rows); summaries.append(summary)
    for arm in requested:
        score = policy_scores(cache, 'eval', arm, policies[arm], args, out)
        final, proposed, accepted = apply_thresholds(gtest, score['candidate'], score['pre'], score['post'], thresholds[arm])
        rows, summary = evaluation_rows(cache, 'full_test', arm, ytest, gtest, final,
                                         proposed, accepted, tail_ids, benign_id)
        per_class.extend(rows); summaries.append(summary)
        save_array(out / f'{arm}_final.npy', final.astype(np.int32))
        possible = (gtest != ytest) & any_correct
        proposal_diag = []
        for c, name in enumerate(cache.names):
            mask = possible & (ytest == c)
            recovered = mask & proposed & (score['candidate'] == ytest)
            proposal_diag.append({'class': name, 'correctable_global_errors': int(mask.sum()),
                'scorer_recovered': int(recovered.sum()),
                'scorer_recovery_rate': float(recovered.sum() / max(mask.sum(), 1)),
                'final_recovered': int((mask & (final == ytest)).sum())})
        pd.DataFrame(proposal_diag).to_csv(out / f'5a_{arm}_proposal_recovery.csv', index=False)
        print('RESULT: ' + json.dumps(summary), flush=True)
        pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
        pd.DataFrame(per_class).to_csv(out / 'per_class_metrics.csv', index=False)
    # Portable visualization of full-test deltas, plus complete tabular records.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = pd.DataFrame(summaries)
    shown = result[(result['split'] == 'full_test') & result['arm'].isin(requested)]
    fig, ax = plt.subplots(figsize=(10, 4))
    xx = np.arange(len(shown))
    ax.bar(xx - .18, shown.delta_macro_f1, .36, label='Delta macro-F1')
    ax.bar(xx + .18, shown.delta_tail_f1, .36, label='Delta tail-F1')
    ax.set_xticks(xx, shown.arm, rotation=15); ax.axhline(0, color='black', linewidth=.8)
    ax.legend(); ax.set_ylabel('Change from frozen global'); fig.tight_layout()
    fig.savefig(out / '6a_full_test_deltas.png', dpi=160); plt.close(fig)
    write_json(out / 'COMPLETE.json', {'seconds': time.monotonic() - started,
               'full_test_rows': len(ytest), 'arms': requested,
               'evaluated_as_development_holdout': True})
    (out / 'RUNNING.json').unlink()
    print('WROTE: ' + str(out), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--out-root', default='tabpfn/results')
    p.add_argument('--arms', default=','.join(SPECS))
    p.add_argument('--trees', type=int, default=300)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=.05)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--predict-chunk', type=int, default=100000)
    p.add_argument('--confirm-fraction', type=float, default=.3)
    p.add_argument('--benign-fpr-increase', type=float, default=.0005)
    p.add_argument('--protected-classes', default='brute_force,ddos,dos')
    p.add_argument('--protected-f1-drop', type=float, default=0.)
    p.add_argument('--max-proposal', type=float, default=1.)
    p.add_argument('--min-decided', type=int, default=30)
    p.add_argument('--hdn-lambda', type=float, default=1.,
                   help='hdn arm post-score = P(helpful) - hdn_lambda * P(harmful)')
    p.add_argument('--wait-for-cache', action='store_true')
    p.add_argument('--cache-wait-minutes', type=float, default=120.)
    args = p.parse_args()
    if not 0 < args.confirm_fraction < 1: p.error('confirm-fraction must be in (0,1)')
    if args.protected_f1_drop < 0 or args.benign_fpr_increase < 0:
        p.error('Non-degradation tolerances must be nonnegative')
    if args.wait_for_cache:
        print('Args: ' + json.dumps(vars(args), sort_keys=True), flush=True)
        deadline = time.monotonic() + 60 * args.cache_wait_minutes
        cache_path = Path(args.cache_dir)
        while not (cache_path / 'COMPLETE.json').exists():
            lock = cache_path / 'PREPARING.lock'
            if not lock.exists():
                raise RuntimeError('Cache incomplete and producer is not running; inspect EXP38 log')
            os.kill(int(lock.read_text()), 0)
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for EXP38 cache')
            print('Waiting for frozen cache: ' + str(cache_path), flush=True)
            time.sleep(30)
    with threadpool_limits(args.threads): run(args)


if __name__ == '__main__':
    main()

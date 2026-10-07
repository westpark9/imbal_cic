#!/usr/bin/env python3
"""EXP68 (feasibility) — temporal-arrival (prequential) protocol: attacks appear over time, the detector adapts with a
few labels per period, no retraining of the backbone.

Test rows are ordered by timestamp and cut into --windows equal-size periods. In each period the first --warmup-frac
of EACH class's rows (chronologically, i.e. the first flows of every family as it appears) may be labelled by an
analyst, up to --label-budget rows per class; the remaining rows of the period are evaluated. Labels accumulate. Arms (same evaluation rows):
  tabpfn_frozen   : the recorded 100k C0 Global, never updated (saved predictions)
  tabpfn_context  : C0 + accumulated labelled rows appended to the context (no weight update)
  xgb_frozen      : XGBoost trained once on the C0 rows (EXP61 settings)
  xgb_retrain     : XGBoost retrained each period on C0 + accumulated labelled rows
Per period: macro-F1 over classes present in the evaluated rows, per-class P/R/F1, benign FPR, labels used, fit/predict
seconds. Seed 43 single run; feasibility scale. Later extension: hold a family out of train (unseen/OOD arrival).
"""
import argparse
import json
import time

import numpy as np

from exp6x_context_common import load_dataset, class_table, tabpfn_global, predict, run_dir, dump, CACHES


def xgb_fit(X, y, C, seed, threads=6):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.05, subsample=.8, colsample_bytree=.8, min_child_weight=1, reg_lambda=1,
                          objective='multi:softprob', num_class=C, eval_metric='mlogloss', tree_method='hist', device='cuda:0', n_jobs=threads, random_state=seed)
    t0 = time.time(); m.fit(X, y); return m, time.time() - t0


def xgb_predict(m, X):
    outs = []
    for i in range(0, len(X), 500000):
        outs.append(np.asarray(m.predict(X[i:i + 500000])).astype(int))
    return np.concatenate(outs)


def window_metrics(y, pred, d):
    rows = class_table(y, pred, d['names']); present = [r for r in rows if r['support'] > 0]
    b = d['benign']
    return dict(rows=int(len(y)), present_classes=[r['cls'] for r in present],
                macro_f1_present=float(np.mean([r['f1'] for r in present])),
                tail_f1_present=float(np.mean([r['f1'] for r in present if r['cls'] in [d['names'][i] for i in d['tails']]])) if any(r['cls'] in [d['names'][i] for i in d['tails']] for r in present) else None,
                benign_fpr=float(((pred == b) & (y != b)).sum() / max(1, (y != b).sum())), accuracy=float((pred == y).mean()), classes=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='toniot', choices=['toniot', 'cic2018'])
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--windows', type=int, default=5)
    ap.add_argument('--warmup-frac', type=float, default=0.1)
    ap.add_argument('--label-budget', type=int, default=50, help='max labelled rows per class per period')
    ap.add_argument('--arms', default='tabpfn_frozen,tabpfn_context,xgb_frozen,xgb_retrain')
    ap.add_argument('--n-estimators', type=int, default=4)
    args = ap.parse_args(); print('Args:', vars(args), flush=True)
    rng = np.random.default_rng(args.seed); d = load_dataset(args.dataset); C = d['C']; names = d['names']
    out = run_dir(args.dataset, 'exp68_temporal_arrival', args.seed)
    cache = CACHES[args.dataset]
    t = np.load(cache / 'eval_time.npy'); order = np.argsort(t, kind='stable'); y_all = d['eval_y']
    p0 = np.load(cache / 'eval_p0.npy', mmap_mode='r')
    chunks = np.array_split(order, args.windows)
    plan = []
    for w, idx in enumerate(chunks):
        # per-class chronological warm-up: the first --warmup-frac of EACH class's rows in the period may be labelled
        # (the analyst labels the first flows of every family as it appears); the rest of the period is evaluated.
        warm_parts, lab_parts = [], []
        for c in range(C):
            rows_c = idx[y_all[idx] == c]
            if len(rows_c) == 0:
                continue
            n_warm = int(np.ceil(args.warmup_frac * len(rows_c))); warm_c = rows_c[:n_warm]; warm_parts.append(warm_c)
            lab_parts.append(rng.choice(warm_c, min(args.label_budget, len(warm_c)), replace=False))
        warm = np.concatenate(warm_parts); lab_rows = np.concatenate(lab_parts); ev = np.setdiff1d(idx, warm)
        plan.append(dict(window=w + 1, t_start=int(t[idx[0]]), t_end=int(t[idx[-1]]), warm_rows=int(len(warm)), eval_rows=int(len(ev)), labels=lab_rows, eval=np.sort(ev),
                         eval_counts=np.bincount(y_all[ev], minlength=C).tolist(), label_counts=np.bincount(y_all[lab_rows], minlength=C).tolist()))
        print(f'period {w+1}: eval {len(ev):,} rows, labels {len(lab_rows)} {dict(zip(names, plan[-1]["label_counts"]))}', flush=True)
    arms = args.arms.split(','); results = {a: [] for a in arms}; cost = {a: dict(fit_seconds=0.0, predict_seconds=0.0) for a in arms}
    xgb_frozen = None
    if 'xgb_frozen' in arms:
        xgb_frozen, fs = xgb_fit(d['c0_X'], d['c0_y'], C, args.seed); cost['xgb_frozen']['fit_seconds'] += fs
    acc_idx = np.array([], dtype=int)
    for p in plan:
        w = p['window']; ev = p['eval']; y_ev = y_all[ev]; X_ev = np.asarray(d['eval_X'][ev])
        acc_idx = np.concatenate([acc_idx, p['labels']])  # this period's warm-up labels are available for its evaluation rows; labels accumulate
        X_lab = np.asarray(d['eval_X'][np.sort(acc_idx)]); y_lab = y_all[np.sort(acc_idx)]
        for arm in arms:
            t0 = time.time()
            if arm == 'tabpfn_frozen':
                pred = np.asarray(p0[ev]).argmax(1); fs = 0.0
            elif arm == 'tabpfn_context':
                clf, fs = tabpfn_global(np.concatenate([d['c0_X'], X_lab]), np.concatenate([d['c0_y'], y_lab]), args.seed, args.n_estimators)
                t1 = time.time(); pred = predict(clf, X_ev); del clf
                import torch; torch.cuda.empty_cache()
            elif arm == 'xgb_frozen':
                fs = 0.0; pred = xgb_predict(xgb_frozen, X_ev)
            elif arm == 'xgb_retrain':
                m, fs = xgb_fit(np.concatenate([d['c0_X'], X_lab]), np.concatenate([d['c0_y'], y_lab]), C, args.seed); pred = xgb_predict(m, X_ev)
            else:
                raise ValueError(arm)
            ps = time.time() - t0 - fs
            m_ = window_metrics(y_ev, pred, d); m_.update(window=w, fit_seconds=fs, predict_seconds=ps, labels_used=int(len(acc_idx)), context_rows=int(len(d['c0_y']) + (len(acc_idx) if arm in ('tabpfn_context', 'xgb_retrain') else 0)))
            results[arm].append(m_); cost[arm]['fit_seconds'] += fs; cost[arm]['predict_seconds'] += ps
            tails = {r['cls']: round(r['f1'], 3) for r in m_['classes'] if r['support'] > 0 and r['cls'] in [names[i] for i in d['tails']]}
            print(f'period {w} {arm:15} macro(present) {m_["macro_f1_present"]:.4f} benign_fpr {m_["benign_fpr"]:.4f} tails {tails} | fit {fs:.0f}s pred {ps:.0f}s labels {len(acc_idx)}', flush=True)
        dump(out / 'results.json', dict(args=vars(args), dataset=args.dataset, class_names=names,
             periods=[{k: v for k, v in p.items() if k not in ('labels', 'eval')} for p in plan], results=results, cost=cost))
    print('saved', out)


if __name__ == '__main__':
    main()

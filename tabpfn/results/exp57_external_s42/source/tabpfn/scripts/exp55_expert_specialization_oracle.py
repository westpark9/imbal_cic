#!/usr/bin/env python3
"""EXP55: expert-specialization oracles on a frozen EXP47/EXP48-style cache (no training, no GPU).

Why: every oracle reported so far (EXP39/50/51 `correctness_oracle`, 0911 `row_level_oracle_bank`) is
    "any of the K experts is correct -> correct". That is a perfect *row-level picker* over K noisy
    predictors; it says nothing about whether each expert is good on the regime it was built for.

Oracle family (each isolates one component; all computed on the same frozen test predictions):
  global          : reference.
  regime_route    : label-free routing. Row -> expert k = argmin_k eval_distance[i, k] (squared distance
                    of the observable signature (embedding + p0) to expert k's regime centroid, as stored by
                    EXP31). Expert answer used AS-IS. = "scorer is exactly right by design, no verifier".
  regime_route+V  : regime_route, then accept the expert's answer only if it is correct, else global.
                    = "scorer and verifier both perfect" (the oracle the user asked for).
  class_route     : row -> expert designated for the row's TRUE class (block-dominant class mapping from
                    2a_expert_contexts.csv; classes without an expert stay global). Answer AS-IS.
                    = class-specialization oracle, no verifier.
  class_route+V   : class_route + perfect verifier.
  union           : any expert correct -> correct (the old correctness_oracle, for reference).

Specialization readouts:
  2a per regime   : rows, class mix, global acc, own-expert acc, best other expert acc, union acc.
  2b K x K matrix : accuracy of expert j on regime k rows (diagonal = own regime).
  2c K x C matrix : recall of expert k on class c (+ global row).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TAG = 'exp55_expert_specialization'


def load_argmax(path, chunk):
    a = np.load(path, mmap_mode='r')
    out = np.empty(a.shape[0], dtype=np.int16)
    for s in range(0, a.shape[0], chunk):
        out[s:s + chunk] = np.asarray(a[s:s + chunk]).argmax(1)
    return out


def prf(y, pred, n):
    cm = np.bincount(y.astype(np.int64) * n + pred.astype(np.int64), minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pr = cm.sum(0).astype(float)
    f1 = np.divide(2 * tp, sup + pr, out=np.zeros(n), where=(sup + pr) > 0)
    p = np.divide(tp, pr, out=np.zeros(n), where=pr > 0)
    r = np.divide(tp, sup, out=np.zeros(n), where=sup > 0)
    return f1, p, r, sup


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', required=True, help='frozen_cache dir of an EXP47/48/52 run')
    ap.add_argument('--contexts-csv', default=None,
                    help='2a_expert_contexts.csv (default: <COMPLETE.json source_run>/2a_expert_contexts.csv)')
    ap.add_argument('--out-root', default=None, help='default tabpfn/results/exp55_expert_specialization_<dataset>')
    ap.add_argument('--chunk', type=int, default=250_000)
    ap.add_argument('--seed', type=int, default=42, help='unused (no randomness); recorded for the Args line')
    args = ap.parse_args()
    print('Args:', json.dumps(vars(args)), flush=True)
    t0 = time.time()

    cache = Path(args.cache_dir).resolve()
    meta = json.loads((cache / 'COMPLETE.json').read_text())
    names = meta['class_names']
    n = len(names)
    K = int(meta['n_experts'])
    dataset = meta.get('dataset') or meta['source_config']['target_dataset']
    tail = [names.index(c) for c in meta.get('tail_classes', [])]
    ben = names.index('benign')
    ctx_csv = Path(args.contexts_csv) if args.contexts_csv else Path(meta['source_run']) / '2a_expert_contexts.csv'
    out_root = Path(args.out_root) if args.out_root else Path('tabpfn/results') / f'{TAG}_{dataset}'
    out = out_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{TAG}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'experiment_source.py')
    print(f'dataset={dataset} K={K} classes={names} tail={meta.get("tail_classes")} out={out}', flush=True)

    y = np.load(cache / 'eval_y.npy').astype(np.int64)
    N = len(y)
    g = load_argmax(cache / 'eval_p0.npy', args.chunk)
    E = np.stack([load_argmax(cache / f'eval_p{k + 1}.npy', args.chunk) for k in range(K)], axis=1)  # N x K
    correct = (E == y[:, None])  # N x K
    print(f'loaded predictions {N:,} rows x {K} experts in {time.time() - t0:.0f}s', flush=True)

    # ---- routing maps ------------------------------------------------------------------------
    dist = np.load(cache / 'eval_distance.npy', mmap_mode='r')
    assert dist.shape == (N, K), dist.shape
    regime = np.asarray(dist).argmin(1)  # label-free
    ctx = pd.read_csv(ctx_csv)
    ctx = ctx[ctx['expert'].astype(str) != 'anchor(shared)'].copy()
    ctx['expert'] = ctx['expert'].astype(int)
    ctx = ctx.sort_values('expert').reset_index(drop=True)
    assert len(ctx) == K, (len(ctx), K)
    block = ctx[names].to_numpy(dtype=float)  # K x C rows of each class in the expert block (anchor excluded)
    block_share = block / np.maximum(block.sum(1, keepdims=True), 1)
    dominant = block_share.argmax(1)  # expert -> dominant class
    designated = np.full(n, -1, dtype=int)  # class -> expert (block with most rows of that class, only if dominant)
    for c in range(n):
        cand = [k for k in range(K) if dominant[k] == c]
        if cand:
            designated[c] = max(cand, key=lambda k: block[k, c])
    route_map = pd.DataFrame({
        'class': names,
        'designated_expert': [int(designated[c]) + 1 if designated[c] >= 0 else None for c in range(n)],
        'block_rows_of_class': [float(block[designated[c], c]) if designated[c] >= 0 else 0.0 for c in range(n)],
        'block_share': [float(block_share[designated[c], c]) if designated[c] >= 0 else 0.0 for c in range(n)],
        'other_experts_dominated_by_class': [
            ','.join(str(k + 1) for k in range(K) if dominant[k] == c and k != designated[c]) for c in range(n)],
    })
    route_map.to_csv(out / '3a_class_route_map.csv', index=False)
    print('class -> designated expert:\n' + route_map.to_string(index=False), flush=True)

    # ---- policies -------------------------------------------------------------------------------
    idx = np.arange(N)
    pol = {'global': g.astype(np.int64)}
    rr = E[idx, regime].astype(np.int64)
    pol['regime_route'] = rr
    pol['regime_route+V'] = np.where(rr == y, rr, g)
    des = designated[y]
    has = des >= 0
    cr = g.astype(np.int64).copy()
    cr[has] = E[idx[has], des[has]]
    pol['class_route'] = cr
    pol['class_route+V'] = np.where(cr == y, cr, g)
    pol['union'] = np.where(correct.any(1), y, g)

    g_ok = g == y
    summ, per_class = [], []
    for name, pred in pol.items():
        f1, p, r, sup = prf(y, pred, n)
        ok = pred == y
        row = {'policy': name, 'macro_f1': f1.mean(), 'tail_f1': f1[tail].mean() if tail else np.nan,
               'benign_f1': f1[ben], 'accuracy': ok.mean(),
               'attack_called_benign': float(((y != ben) & (pred == ben)).sum() / max((y != ben).sum(), 1)),
               'benign_called_attack': float(((y == ben) & (pred != ben)).sum() / max((y == ben).sum(), 1)),
               'changed': int((pred != g).sum()), 'H': int((~g_ok & ok).sum()), 'D': int((g_ok & ~ok).sum())}
        summ.append(row)
        for c in range(n):
            per_class.append({'policy': name, 'class': names[c], 'support': int(sup[c]), 'f1': f1[c],
                              'precision': p[c], 'recall': r[c],
                              'H': int((~g_ok & ok & (y == c)).sum()), 'D': int((g_ok & ~ok & (y == c)).sum())})
    summ = pd.DataFrame(summ)
    per_class = pd.DataFrame(per_class)
    summ.to_csv(out / '1a_policy_summary.csv', index=False)
    per_class.to_csv(out / '1b_policy_per_class.csv', index=False)
    print('\npolicy summary:\n' + summ.to_string(index=False, float_format=lambda v: f'{v:.4f}'), flush=True)

    # ---- specialization readouts -------------------------------------------------------------
    acc_mat = np.zeros((K, K + 2))  # expert j on regime k rows, then global, union
    reg_rows = []
    for k in range(K):
        m = regime == k
        nk = int(m.sum())
        if nk == 0:
            reg_rows.append({'regime': k + 1, 'rows': 0})
            continue
        acc_j = correct[m].mean(0)
        acc_mat[k, :K] = acc_j
        acc_mat[k, K] = g_ok[m].mean()
        acc_mat[k, K + 1] = correct[m].any(1).mean()
        others = np.delete(acc_j, k)
        best_other = int(np.delete(np.arange(K), k)[others.argmax()])
        comp = np.bincount(y[m], minlength=n) / nk
        top = np.argsort(-comp)[:2]
        reg_rows.append({
            'regime': k + 1, 'rows': nk, 'row_share': nk / N,
            'top_class': names[top[0]], 'top_share': comp[top[0]],
            'second_class': names[top[1]], 'second_share': comp[top[1]],
            'block_dominant_class': names[dominant[k]], 'block_dominant_share': block_share[k, dominant[k]],
            'global_acc': acc_mat[k, K], 'own_acc': acc_j[k],
            'best_other_expert': best_other + 1, 'best_other_acc': acc_j[best_other],
            'union_acc': acc_mat[k, K + 1],
            'own_minus_global': acc_j[k] - acc_mat[k, K],
            'own_minus_best_other': acc_j[k] - acc_j[best_other],
            'own_H': int((~g_ok[m] & correct[m, k]).sum()), 'own_D': int((g_ok[m] & ~correct[m, k]).sum()),
        })
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(out / '2a_regime_specialization.csv', index=False)
    acc_df = pd.DataFrame(acc_mat, index=[f'regime{k + 1}' for k in range(K)],
                          columns=[f'expert{j + 1}' for j in range(K)] + ['global', 'union'])
    acc_df.to_csv(out / '2b_regime_expert_acc_matrix.csv')
    rec = np.zeros((K + 1, n))
    for c in range(n):
        m = y == c
        if m.any():
            rec[:K, c] = correct[m].mean(0)
            rec[K, c] = g_ok[m].mean()
    rec_df = pd.DataFrame(rec, index=[f'expert{k + 1}' for k in range(K)] + ['global'], columns=names)
    rec_df.to_csv(out / '2c_expert_class_recall_matrix.csv')
    print('\nregime specialization:\n' + reg_df.to_string(index=False, float_format=lambda v: f'{v:.3f}'), flush=True)
    print('\nexpert x regime accuracy (rows = regime, cols = expert):\n'
          + acc_df.to_string(float_format=lambda v: f'{v:.3f}'), flush=True)
    print('\nexpert x class recall:\n' + rec_df.to_string(float_format=lambda v: f'{v:.3f}'), flush=True)

    # ---- verdict numbers ----------------------------------------------------------------------
    own_gt_global = int((reg_df['own_minus_global'] > 0).sum()) if 'own_minus_global' in reg_df else 0
    own_gt_others = int((reg_df['own_minus_best_other'] > 0).sum()) if 'own_minus_best_other' in reg_df else 0
    complete = {
        'seconds': time.time() - t0, 'dataset': dataset, 'K': K, 'rows': int(N), 'cache_dir': str(cache),
        'contexts_csv': str(ctx_csv), 'cache_identity': json.loads((cache / 'identity.json').read_text())
        if (cache / 'identity.json').exists() else None,
        'macro_f1': {r['policy']: float(r['macro_f1']) for r in summ.to_dict('records')},
        'regimes_where_own_expert_beats_global': own_gt_global,
        'regimes_where_own_expert_beats_every_other_expert': own_gt_others,
        'classes_with_designated_expert': int((designated >= 0).sum()),
    }
    (out / 'COMPLETE.json').write_text(json.dumps(complete, indent=1))
    print(f'\nown expert beats global on {own_gt_global}/{K} regimes; beats every other expert on {own_gt_others}/{K}.',
          flush=True)
    print(f'DONE {time.time() - t0:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()

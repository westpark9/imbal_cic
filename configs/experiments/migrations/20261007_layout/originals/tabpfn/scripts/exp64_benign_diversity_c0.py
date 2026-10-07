#!/usr/bin/env python3
"""EXP64 (feasibility) — does HOW the 75k benign rows of C0 are chosen change the frozen Global?

Knob: benign selection rule inside the seed-43 C0 recipe (benign 0.75, attack rows unchanged = exact C0 attack rows).
  random      : the recorded C0 benign rows (reference, identical to EXP59/61/62 Global)
  kcenter     : farthest-point (k-center) selection in standardized feature space from a benign candidate subsample
  time        : uniform over time quantiles of the D_global benign rows (window coverage)
  inv_density : inverse-kNN-density weighted sampling (favours sparse benign regions)
  hard        : benign rows nearest to the attack rows of C0 (boundary benign), half random for prior
All variants keep |benign| and all attack rows fixed; the only change is which benign rows.
Evaluation: same stratified test subsample for every variant (screening); --full-test for a chosen variant.
"""
import argparse
import time

import numpy as np

from exp6x_context_common import load_dataset, standardizer, test_subsample, summarize, tabpfn_global, predict, run_dir, dump


def kcenter(Z, k, rng, start=None):
    n = len(Z); chosen = [int(rng.integers(n)) if start is None else start]
    dmin = np.linalg.norm(Z - Z[chosen[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(dmin)); chosen.append(i); dmin = np.minimum(dmin, np.linalg.norm(Z - Z[i], axis=1))
    return np.asarray(chosen)


def select_benign(rule, d, Z, n_benign, rng, cand_size):
    pool_y = d['pool_y']; b = d['benign']; cand_all = np.flatnonzero(pool_y == b)
    if rule == 'random':
        return None  # recorded C0 rows
    cand = rng.choice(cand_all, min(cand_size, len(cand_all)), replace=False)
    if rule == 'kcenter':
        Zc = Z(d['pool_X'][np.sort(cand)]); cand = np.sort(cand)
        return cand[kcenter(Zc, n_benign, rng)]
    if rule == 'time':
        t = d['pool_time'][cand_all]; order = np.argsort(t); bins = np.array_split(order, n_benign)
        return cand_all[np.array([bb[rng.integers(len(bb))] for bb in bins])]
    if rule == 'inv_density':
        from sklearn.neighbors import NearestNeighbors
        cand = np.sort(cand); Zc = Z(d['pool_X'][cand])
        dist = NearestNeighbors(n_neighbors=11).fit(Zc).kneighbors(Zc)[0][:, -1]
        w = dist + 1e-3; w /= w.sum()
        return cand[rng.choice(len(cand), n_benign, replace=False, p=w)]
    if rule == 'hard':
        from sklearn.neighbors import NearestNeighbors
        cand = np.sort(cand); Zc = Z(d['pool_X'][cand])
        attacks = Z(d['c0_X'][d['c0_y'] != b])
        dist = NearestNeighbors(n_neighbors=1).fit(attacks).kneighbors(Zc)[0][:, 0]
        n_hard = n_benign // 2; hard = cand[np.argsort(dist)[:n_hard]]
        rest = np.setdiff1d(cand_all, hard); return np.concatenate([hard, rng.choice(rest, n_benign - n_hard, replace=False)])
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='toniot', choices=['toniot', 'cic2018'])
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--rules', default='random,kcenter,time,inv_density,hard')
    ap.add_argument('--candidate-size', type=int, default=200000, help='benign candidate subsample for k-center / density rules')
    ap.add_argument('--test-per-class', type=int, default=20000)
    ap.add_argument('--full-test', action='store_true')
    ap.add_argument('--n-estimators', type=int, default=4)
    args = ap.parse_args(); print('Args:', vars(args), flush=True)
    rng = np.random.default_rng(args.seed); d = load_dataset(args.dataset); Z = standardizer(d['route_X'][:200000])
    out = run_dir(args.dataset, 'exp64_benign_diversity', args.seed)
    te_idx, tX, ty = test_subsample(d, args.test_per_class, rng, args.full_test)
    b = d['benign']; attack_mask = d['c0_y'] != b; n_benign = int((~attack_mask).sum())
    print(f'C0: {len(d["c0_y"]):,} rows, benign {n_benign:,}; test rows {len(ty):,}', flush=True)
    results = {}
    for rule in args.rules.split(','):
        t0 = time.time(); sel = select_benign(rule, d, Z, n_benign, np.random.default_rng(args.seed + 1), args.candidate_size)
        if sel is None:
            ctx_X, ctx_y = d['c0_X'], d['c0_y']; sel_ids = d['c0_ids'][~attack_mask]
        else:
            sel = np.sort(sel); ctx_X = np.concatenate([d['c0_X'][attack_mask], np.asarray(d['pool_X'][sel])]); ctx_y = np.concatenate([d['c0_y'][attack_mask], d['pool_y'][sel]])
            sel_ids = d['pool_ids'][sel]; assert (d['pool_y'][sel] == b).all()
        sel_s = time.time() - t0
        clf, fit_s = tabpfn_global(ctx_X, ctx_y, args.seed, args.n_estimators)
        t1 = time.time(); pred = predict(clf, tX); pred_s = time.time() - t1
        s = summarize(ty, pred, d); s.update(rule=rule, context_rows=int(len(ctx_y)), select_seconds=sel_s, fit_seconds=fit_s, predict_seconds=pred_s,
                                     benign_overlap_with_c0=int(np.isin(sel_ids, d['c0_ids']).sum()))
        results[rule] = s
        print(f'{rule:12} macro-F1 {s["macro_f1"]:.4f} tail {s["tail_f1"]:.4f} acc {s["accuracy"]:.4f} | ' + ' '.join(f'{r["cls"]}={r["f1"]:.3f}' for r in s['classes']), flush=True)
        np.save(out / f'{rule}_benign_ids.npy', sel_ids)
        dump(out / 'results.json', dict(args=vars(args), dataset=args.dataset, test_rows=int(len(ty)), test_idx_sha=None, results=results))
        del clf
        import torch; torch.cuda.empty_cache()
    print('saved', out)


if __name__ == '__main__':
    main()

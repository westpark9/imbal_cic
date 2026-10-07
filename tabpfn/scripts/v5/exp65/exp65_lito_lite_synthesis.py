#!/usr/bin/env python3
"""EXP65 (feasibility) — LITO-style tail oversampling with a tabular foundation model instead of an LLM.

LITO (Language-Interfaced Tabular Oversampling via Progressive Imputation and Self-Authentication, ICLR 2024;
no code release) = (1) progressive imputation: start from real minority rows, mask part of the features and let the
model fill them in, repeat; (2) self-authentication: keep a synthetic row only if the model itself classifies it as the
intended class with high confidence. Here both roles are played by TabPFN:
  imputer       : the v3 TabPFN classifier over quantile bins of the masked feature (context = real rows of the target class,
                  inputs = the other features); the value is drawn from a real row of the sampled bin (token-like, as in LITO)
  authenticator : the frozen Global classifier on the recorded 100k C0; keep rows with argmax == target and p >= --auth-threshold
Arms on the same test subsample (all rows of the target class are in the subsample):
  c0            : recorded C0 (reference)
  c0+dup        : C0 + the real target rows duplicated to the same count as the synthetic set  (control: prior shift only)
  c0+syn        : C0 + authenticated synthetic rows
  c0+syn_noauth : C0 + synthetic rows without authentication                                     (control: what authentication buys)
Targets: cic2018 web_attacks (224 real rows in C0) by default; any class name via --target.
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
import time

import numpy as np

from exp6x_context_common import load_dataset, test_subsample, summarize, tabpfn_global, predict, run_dir, dump, CKPT


def progressive_impute(seed_rows, n_out, rounds, mask_frac, rng, device, n_bins=10):
    """Return n_out synthetic rows derived from seed_rows by repeated masked imputation.

    Imputer = the v3 TabPFN *classifier* over quantile bins of the masked feature (context = the real class rows,
    inputs = the other features), i.e. a token-like conditional as in LITO; the value is drawn from the predicted bin.
    No regressor checkpoint is needed."""
    from tabpfn import TabPFNClassifier
    X = seed_rows[rng.choice(len(seed_rows), n_out, replace=True)].astype(np.float32).copy()
    F = X.shape[1]
    for r in range(rounds):
        masked = np.array([rng.choice(F, max(1, int(mask_frac * F)), replace=False) for _ in range(n_out)])
        for f in range(F):
            rows = np.flatnonzero((masked == f).any(1))
            if len(rows) == 0:
                continue
            others = [j for j in range(F) if j != f]
            col = seed_rows[:, f]
            edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
            if len(edges) < 3:  # (near-)constant feature: resample the real marginal
                X[rows, f] = rng.choice(col, len(rows), replace=True); continue
            bins = np.clip(np.searchsorted(edges, col, side='right') - 1, 0, len(edges) - 2)
            present = np.unique(bins)
            if len(present) < 2:
                X[rows, f] = rng.choice(col, len(rows), replace=True); continue
            clf = TabPFNClassifier(model_path=str(CKPT), device=device, n_estimators=1, random_state=r * 1000 + f, ignore_pretraining_limits=True)
            clf.fit(seed_rows[:, others], bins)
            proba = clf.predict_proba(X[rows][:, others])
            classes = clf.classes_
            for k, i in enumerate(rows):
                bchoice = classes[rng.choice(len(classes), p=proba[k] / proba[k].sum())]
                members = col[bins == bchoice]
                X[i, f] = rng.choice(members) if len(members) else rng.uniform(edges[bchoice], edges[bchoice + 1])
        print(f'  imputation round {r+1}/{rounds} done', flush=True)
    for f in range(F):
        col = seed_rows[:, f]
        if np.all(col == np.round(col)):
            X[:, f] = np.round(X[:, f])
        if col.min() >= 0:
            X[:, f] = np.clip(X[:, f], 0, None)
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='cic2018', choices=['toniot', 'cic2018'])
    ap.add_argument('--target', default='web_attacks')
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--n-synthetic', type=int, default=2000)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--mask-frac', type=float, default=0.3)
    ap.add_argument('--auth-threshold', type=float, default=0.5)
    ap.add_argument('--test-per-class', type=int, default=20000)
    ap.add_argument('--full-test', action='store_true')
    ap.add_argument('--arms', default='c0,c0+dup,c0+syn,c0+syn_noauth')
    args = ap.parse_args(); print('Args:', vars(args), flush=True)
    rng = np.random.default_rng(args.seed); d = load_dataset(args.dataset); names = d['names']; t = names.index(args.target)
    out = run_dir(args.dataset, f'exp65_lito_lite_{args.target}', args.seed); dev = 'cuda'
    te_idx, tX, ty = test_subsample(d, args.test_per_class, rng, args.full_test)
    real = d['c0_X'][d['c0_y'] == t]; print(f'target {args.target}: {len(real)} real rows in C0; test rows {len(ty):,} (target {(ty==t).sum()})', flush=True)
    t0 = time.time(); syn = progressive_impute(real, args.n_synthetic, args.rounds, args.mask_frac, rng, dev); gen_s = time.time() - t0
    # self-authentication with the frozen Global on the recorded C0
    auth, fit_s = tabpfn_global(d['c0_X'], d['c0_y'], args.seed); p = predict(auth, syn, proba=True)
    keep = (p.argmax(1) == t) & (p[:, t] >= args.auth_threshold)
    print(f'synthetic {len(syn)} rows in {gen_s:.0f}s | authenticated {keep.sum()} ({keep.mean():.1%}) | mean p(target) {p[:, t].mean():.3f}', flush=True)
    np.save(out / 'synthetic_X.npy', syn); np.save(out / 'synthetic_auth_p.npy', p); np.save(out / 'synthetic_keep.npy', keep)
    results = dict(generation_seconds=gen_s, authenticated=int(keep.sum()), n_synthetic=int(len(syn)), arms={})
    # reference arm reuses the authenticator model
    if 'c0' in args.arms.split(','):
        pred = predict(auth, tX); results['arms']['c0'] = summarize(ty, pred, d) | dict(context_rows=int(len(d['c0_y'])))
    del auth
    import torch; torch.cuda.empty_cache()
    extra = {'c0+dup': real[rng.choice(len(real), int(keep.sum()), replace=True)], 'c0+syn': syn[keep], 'c0+syn_noauth': syn[:int(keep.sum())] if keep.sum() else syn}
    for arm in args.arms.split(','):
        if arm == 'c0' or arm not in extra or len(extra[arm]) == 0:
            continue
        cx = np.concatenate([d['c0_X'], extra[arm]]); cy = np.concatenate([d['c0_y'], np.full(len(extra[arm]), t)])
        clf, fit_s = tabpfn_global(cx, cy, args.seed); pred = predict(clf, tX)
        results['arms'][arm] = summarize(ty, pred, d) | dict(context_rows=int(len(cy)), added_rows=int(len(extra[arm])), fit_seconds=fit_s)
        del clf; torch.cuda.empty_cache()
    for arm, s in results['arms'].items():
        r = s['classes'][t]; print(f'{arm:14} macro-F1 {s["macro_f1"]:.4f} | {args.target} P {r["precision"]:.3f} R {r["recall"]:.3f} F1 {r["f1"]:.3f} FP {r["FP"]} | benign F1 {s["classes"][d["benign"]]["f1"]:.4f}', flush=True)
    dump(out / 'results.json', dict(args=vars(args), dataset=args.dataset, target=args.target, test_rows=int(len(ty)), results=results))
    print('saved', out)


if __name__ == '__main__':
    main()

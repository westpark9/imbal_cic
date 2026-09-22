#!/usr/bin/env python3
"""EXP41 (frozen): dataset-level de-duplication BEFORE the chronological split -- does the benchmark get
easier / does XGBoost score higher once identical-vector conflicts are removed from BOTH train and test?

Variants (applied to all target rows, then the same per-scenario chronological 60/20/20 split is redone on
what remains; scenarios with < --min-scenario-rows rows go to train only, as EXP18 did):
  original        no change (reference; test = the original 20%)
  dedup_xy        one row per (46-feature vector, label), earliest by time; conflicts stay as one row per label
  dedup_majority  one row per vector; label = majority label of that vector over the whole dataset
  drop_conflict   one row per vector; vectors carrying > 1 label anywhere in the dataset are removed
XGBoost 300/8/0.05 sub 0.8 col 0.8 hist, seed 42, no weights. Reported per variant: split sizes, macro-F1 and
per-class F1 on the variant's own test, plus the realistic oracle on that test (conflict vectors -> majority).
Outputs results/<ts>_<pid>_nfv3_<target>_exp41_dedup_dataset_xgb/. Lineage: scripts/exp40_dedup_xgb.py.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT / 'tabpfn' / 'scripts')); sys.path.insert(0, str(REPO_ROOT / 'scripts'))
import nfv3_v3_common as core


def chrono_split(idx, scen, ts, min_rows):
    tr, va, te, small = [], [], [], []
    for s in np.unique(scen[idx]):
        sel = idx[scen[idx] == s]; sel = sel[np.argsort(ts[sel], kind='stable')]
        if len(sel) < min_rows: tr.append(sel); small.append((str(s), int(len(sel)))); continue
        k1, k2 = int(len(sel) * .6), int(len(sel) * .8); tr.append(sel[:k1]); va.append(sel[k1:k2]); te.append(sel[k2:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)) if va else np.array([], int), np.sort(np.concatenate(te)) if te else np.array([], int), small


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/nfv3_energy_suite_uncapped_scenarios.pkl'); p.add_argument('--target-dataset', default='cic2018', choices=['cic2018', 'ton_iot', 'bot_iot'])
    p.add_argument('--arms', default='original,dedup_xy,dedup_majority,drop_conflict'); p.add_argument('--min-scenario-rows', type=int, default=5)
    p.add_argument('--n-estimators', type=int, default=300); p.add_argument('--max-depth', type=int, default=8); p.add_argument('--learning-rate', type=float, default=0.05)
    p.add_argument('--subsample', type=float, default=0.8); p.add_argument('--colsample-bytree', type=float, default=0.8); p.add_argument('--threads', type=int, default=32)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--data-dir', default='data')
    args = p.parse_args()
    ts = time.strftime('%Y%m%d_%H%M%S'); out = REPO_ROOT / 'results' / f'{ts}_{os.getpid()}_nfv3_{args.target_dataset}_exp41_dedup_dataset_xgb'; out.mkdir(parents=True, exist_ok=False)
    log = open(out / 'experiment.log', 'w')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    P(f'Args: {vars(args)}')
    import xgboost as xgb
    from sklearn.metrics import precision_recall_fscore_support
    cfg = core.build_dataset_config(args.data_dir)[args.target_dataset]
    X, class_names, train_idx, val_idx, test_idx, _, _, _, label_fn = cfg['loader'](args); C = len(class_names)
    d = core.load_pickle(args.data); scen = np.asarray(d['attack_scenarios']).astype(str); tsm = np.asarray(d['timestamps'], dtype=np.int64); del d
    all_idx = np.sort(np.concatenate([train_idx, val_idx, test_idx])); y_all = label_fn(all_idx)
    t0 = time.time(); h_all = np.concatenate([pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(np.asarray(X[all_idx[i:i + 2_000_000]], dtype=np.float32))), index=False).to_numpy() for i in range(0, len(all_idx), 2_000_000)])
    P(f'rows {len(all_idx):,}; hashing {time.time() - t0:.0f}s; unique vectors {len(np.unique(h_all)):,}')
    df = pd.DataFrame({'h': h_all, 'y': y_all, 't': tsm[all_idx], 'pos': np.arange(len(all_idx))})
    cnt = df.groupby(['h', 'y']).size().reset_index(name='c'); maj = cnt.sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
    nlab = cnt.groupby('h').size(); conflict_h = set(nlab[nlab > 1].index)
    rows, summary = [], []
    for arm in args.arms.split(','):
        if arm == 'original': keep = np.ones(len(all_idx), bool)
        else:
            first = ~df.sort_values('t', kind='stable').duplicated(['h', 'y']).sort_index().to_numpy()   # earliest row per (vector,label)
            if arm == 'dedup_xy': keep = first
            elif arm == 'dedup_majority': keep = first & (df.h.map(maj).to_numpy() == y_all)
            elif arm == 'drop_conflict': keep = first & ~df.h.isin(conflict_h).to_numpy()
            else: raise SystemExit(arm)
        sub = all_idx[keep]; tr, va, te, small = chrono_split(sub, scen, tsm, args.min_scenario_rows)
        ytr, yte = label_fn(tr), label_fn(te); hte = h_all[keep][np.isin(sub, te)] if False else pd.Series(h_all, index=all_idx).loc[te].to_numpy()
        P(f'[{arm}] rows {len(sub):,} -> train {len(tr):,} val {len(va):,} test {len(te):,}; train-only small scenarios {small}; test per class ' + str({class_names[c]: int((yte == c).sum()) for c in range(C)}))
        # realistic oracle on this test: conflict vectors (within test) -> majority test label
        tcnt = pd.DataFrame({'h': hte, 'y': yte}).groupby(['h', 'y']).size().reset_index(name='c'); tmaj = tcnt.sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
        orc = pd.Series(hte).map(tmaj).to_numpy(); po, ro, fo, so = precision_recall_fscore_support(yte, orc, labels=range(C), zero_division=0)
        Xtr = np.nan_to_num(np.asarray(X[tr], dtype=np.float32)); Xte = np.nan_to_num(np.asarray(X[te], dtype=np.float32)); t1 = time.time()
        clf = xgb.XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate, subsample=args.subsample, colsample_bytree=args.colsample_bytree,
                                tree_method='hist', n_jobs=args.threads, random_state=args.seed, objective='multi:softprob'); clf.fit(Xtr, ytr); pred = clf.predict(Xte)
        pr, rc, f1, su = precision_recall_fscore_support(yte, pred, labels=range(C), zero_division=0)
        for c in range(C): rows.append(dict(arm=arm, cls=class_names[c], test_support=int(su[c]), train_rows=int((ytr == c).sum()), precision=pr[c], recall=rc[c], f1=f1[c], realistic_oracle_f1=fo[c]))
        summary.append(dict(arm=arm, rows=int(len(sub)), train=int(len(tr)), val=int(len(va)), test=int(len(te)), macro_f1=float(f1.mean()), accuracy=float((pred == yte).mean()), realistic_oracle_macro=float(fo.mean()),
                            test_conflict_rows=int(pd.Series(hte).isin(set(tcnt.groupby('h').size().pipe(lambda s: s[s > 1]).index)).sum()), fit_seconds=round(time.time() - t1, 1), small_train_only=json.dumps(small)))
        P(f'[{arm}] macro-F1 {f1.mean():.4f} (realistic oracle {fo.mean():.4f}) | ' + ' '.join(f'{class_names[c]}={f1[c]:.3f}' for c in range(C)) + f' | {time.time() - t1:.0f}s')
        np.save(out / f'pred_{arm}.npy', pred.astype(np.int16)); np.save(out / f'test_idx_{arm}.npy', te)
        pd.DataFrame(rows).to_csv(out / '1a_per_class.csv', index=False); pd.DataFrame(summary).to_csv(out / '1b_summary.csv', index=False)
    (out / 'COMPLETE.json').write_text(json.dumps({'arms': args.arms}) + '\n'); P('DONE')


if __name__ == '__main__':
    main()

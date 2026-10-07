#!/usr/bin/env python3
"""EXP40 (frozen): does removing exact-duplicate / label-conflicting training rows help XGBoost?

Dataset: NF-v3 suite target (default cse_cic_ids2018), pipeline scenario-chronological 60/20/20 split
(scripts/exp_utils.scenario_chronological_split via tabpfn/scripts/nfv3_v3_common loaders).
Arms (one knob = the training-pool filter; model, split and test are identical):
  baseline        full train pool, all rows
  dedup_xy        keep one row per (46-feature vector, label)   -> duplicates removed, conflicts kept
  dedup_majority  keep one row per vector, label = majority label of that vector in train
  drop_conflict   keep one row per vector, drop vectors that carry >1 label in train
Test = original 20% rows (unchanged). Reported: per-class P/R/F1 + macro, and the same on the
three test buckets (vector seen in train with same majority label / seen with a different label /
unseen). No sample weights. Seed 42. Outputs to results/<ts>_<pid>_nfv3_<target>_exp40_dedup_xgb/.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'tabpfn' / 'scripts')); sys.path.insert(0, str(REPO_ROOT / 'scripts'))
import nfv3_v3_common as core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/nfv3_energy_suite_uncapped_scenarios.pkl')
    p.add_argument('--target-dataset', default='cic2018', choices=['cic2018', 'ton_iot', 'bot_iot'])
    p.add_argument('--arms', default='baseline,dedup_xy,dedup_majority,drop_conflict')
    p.add_argument('--n-estimators', type=int, default=300); p.add_argument('--max-depth', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=0.05); p.add_argument('--subsample', type=float, default=0.8)
    p.add_argument('--colsample-bytree', type=float, default=0.8); p.add_argument('--threads', type=int, default=32)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--data-dir', default='data')
    args = p.parse_args()
    ts = time.strftime('%Y%m%d_%H%M%S'); out = REPO_ROOT / 'results' / f'{ts}_{os.getpid()}_nfv3_{args.target_dataset}_exp40_dedup_xgb'
    out.mkdir(parents=True, exist_ok=False); log = open(out / 'experiment.log', 'w')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    P(f'Args: {vars(args)}')
    import xgboost as xgb
    from sklearn.metrics import precision_recall_fscore_support
    cfg = core.build_dataset_config(args.data_dir)[args.target_dataset]
    X, class_names, train_idx, val_idx, test_idx, y_train, y_test, split_audit, label_fn = cfg['loader'](args)
    C = len(class_names); P(f'classes {class_names}; train {len(train_idx):,} val {len(val_idx):,} test {len(test_idx):,}')
    Xtr = np.nan_to_num(np.asarray(X[train_idx], dtype=np.float32)); Xte = np.nan_to_num(np.asarray(X[test_idx], dtype=np.float32))
    htr = pd.util.hash_pandas_object(pd.DataFrame(Xtr), index=False).to_numpy(); hte = pd.util.hash_pandas_object(pd.DataFrame(Xte), index=False).to_numpy()
    # train vector -> label counts
    cnt = pd.DataFrame({'h': htr, 'y': y_train}).groupby(['h', 'y']).size().reset_index(name='c')
    maj = cnt.sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
    nlab = cnt.groupby('h').size(); conflict_h = set(nlab[nlab > 1].index)
    # test buckets
    te_maj = pd.Series(hte).map(maj); seen = te_maj.notna().to_numpy(); same = seen & (te_maj.to_numpy() == y_test); diff = seen & ~same; unseen = ~seen
    P(f'test buckets: seen-same-label {same.sum():,} ({same.mean():.4f}) seen-different-label {diff.sum():,} ({diff.mean():.4f}) unseen {unseen.sum():,} ({unseen.mean():.4f})')
    # realistic oracle on test: conflicting test vectors -> majority test label, others -> truth
    tcnt = pd.DataFrame({'h': hte, 'y': y_test}).groupby(['h', 'y']).size().reset_index(name='c'); tmaj = tcnt.sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
    oracle_pred = pd.Series(hte).map(tmaj).to_numpy()
    def metrics(pred, mask=None):
        yy, pp = (y_test, pred) if mask is None else (y_test[mask], pred[mask])
        pr, rc, f1, sup = precision_recall_fscore_support(yy, pp, labels=range(C), zero_division=0)
        return pr, rc, f1, sup
    rows = []; summary = []
    def record(arm, pred, n_train):
        pr, rc, f1, sup = metrics(pred)
        for c in range(C): rows.append(dict(arm=arm, bucket='all', cls=class_names[c], support=int(sup[c]), precision=pr[c], recall=rc[c], f1=f1[c]))
        for bname, m in [('seen_same_label', same), ('seen_diff_label', diff), ('unseen', unseen)]:
            pr2, rc2, f12, sup2 = metrics(pred, m); acc = float((pred[m] == y_test[m]).mean())
            for c in range(C): rows.append(dict(arm=arm, bucket=bname, cls=class_names[c], support=int(sup2[c]), precision=pr2[c], recall=rc2[c], f1=f12[c]))
            summary.append(dict(arm=arm, bucket=bname, rows=int(m.sum()), accuracy=acc, macro_f1=float(f12.mean())))
        summary.append(dict(arm=arm, bucket='all', rows=int(len(y_test)), accuracy=float((pred == y_test).mean()), macro_f1=float(f1.mean()), n_train=int(n_train)))
        P(f'[{arm}] n_train {n_train:,} | macro-F1 {f1.mean():.4f} acc {(pred == y_test).mean():.4f} | per-class F1 ' + ' '.join(f'{class_names[c]}={f1[c]:.3f}' for c in range(C)))
    record('realistic_oracle_majority', oracle_pred, 0)
    for arm in args.arms.split(','):
        t0 = time.time()
        if arm == 'baseline': keep = np.ones(len(htr), bool)
        else:
            first = ~pd.DataFrame({'h': htr, 'y': y_train}).duplicated(['h', 'y']).to_numpy()  # one row per (vector,label)
            if arm == 'dedup_xy': keep = first
            elif arm == 'dedup_majority': keep = first & (pd.Series(htr).map(maj).to_numpy() == y_train)
            elif arm == 'drop_conflict': keep = first & ~pd.Series(htr).isin(conflict_h).to_numpy()
            else: raise SystemExit(f'unknown arm {arm}')
        cls_counts = {class_names[c]: int((y_train[keep] == c).sum()) for c in range(C)}
        P(f'[{arm}] train rows kept {keep.sum():,} / {len(keep):,}; per class {cls_counts}')
        clf = xgb.XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate, subsample=args.subsample,
                                colsample_bytree=args.colsample_bytree, tree_method='hist', n_jobs=args.threads, random_state=args.seed, objective='multi:softprob')
        clf.fit(Xtr[keep], y_train[keep]); pred = clf.predict(Xte); P(f'[{arm}] fit+predict {time.time() - t0:.0f}s')
        record(arm, pred, int(keep.sum())); np.save(out / f'pred_{arm}.npy', pred.astype(np.int16))
        pd.DataFrame(rows).to_csv(out / '1a_per_class_by_bucket.csv', index=False); pd.DataFrame(summary).to_csv(out / '1b_summary.csv', index=False)
    json.dump({'class_names': class_names, 'test_rows': int(len(y_test)), 'train_rows': int(len(y_train))}, open(out / 'meta.json', 'w'), indent=1)
    (out / 'COMPLETE.json').write_text(json.dumps({'arms': args.arms, 'finished': time.strftime('%Y-%m-%d %H:%M:%S')}) + '\n'); P('DONE')


if __name__ == '__main__':
    main()

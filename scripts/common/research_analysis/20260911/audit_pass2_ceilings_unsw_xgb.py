"""Pass 2 (read-only): save hashes; rarest-class ceiling; clean-row shares; XGB baseline on unsw_nb15 with hash ceilings on its test rows."""
import pickle, time, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
SP = sys.argv[1]
d = pickle.load(open('data/nfv3_energy_suite_uncapped_scenarios.pkl', 'rb'))
X = d['X']; ds = np.asarray(d['dataset_names']).astype(str); fam = np.asarray(d['families']).astype(str)
scen = np.asarray(d['attack_scenarios']).astype(str); ts = np.asarray(d['timestamps']).astype(np.int64); del d
def chrono_masks(sc, tt):
    tr = np.zeros(len(sc), bool); va = np.zeros(len(sc), bool); te = np.zeros(len(sc), bool)
    for s_ in np.unique(sc):
        idx = np.where(sc == s_)[0]; idx = idx[np.argsort(tt[idx], kind='stable')]
        k1, k2 = int(len(idx) * .6), int(len(idx) * .8); tr[idx[:k1]] = True; va[idx[k1:k2]] = True; te[idx[k2:]] = True
    return tr, va, te
rows = []
for name in ['bot_iot', 'ton_iot', 'unsw_nb15', 'cse_cic_ids2018']:
    m = np.where(ds == name)[0]; Xs = np.nan_to_num(np.asarray(X[m], dtype=np.float32)); y = fam[m]; sc = scen[m]; tt = ts[m]
    h = pd.util.hash_pandas_object(pd.DataFrame(Xs), index=False).to_numpy()
    np.save(f'{SP}/hash_{name}.npy', h); np.save(f'{SP}/y_{name}.npy', y); np.save(f'{SP}/t_{name}.npy', tt); np.save(f'{SP}/scen_{name}.npy', sc)
    classes = sorted(np.unique(y)); freq = pd.Series(y).value_counts()
    df = pd.DataFrame({'h': h, 'y': y})
    cnt = df.groupby(['h', 'y']).size().reset_index(name='c')
    maj = cnt.sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
    cnt['f'] = cnt.y.map(freq); rare = cnt.sort_values('f').drop_duplicates('h').set_index('h')['y']
    nlab = cnt.groupby('h').size(); clean_h = set(nlab[nlab == 1].index)
    p_maj = df.h.map(maj).to_numpy(); p_rare = df.h.map(rare).to_numpy()
    f_maj = f1_score(y, p_maj, average=None, labels=classes); f_rare = f1_score(y, p_rare, average=None, labels=classes)
    tr, va, te = chrono_masks(sc, tt)
    f_maj_te = f1_score(y[te], p_maj[te], average=None, labels=classes); f_rare_te = f1_score(y[te], p_rare[te], average=None, labels=classes)
    for c, a, b, a2, b2 in zip(classes, f_maj, f_rare, f_maj_te, f_rare_te):
        mc = y == c
        rows.append(dict(dataset=name, cls=c, rows=int(mc.sum()), clean_rows=int(df.h[mc].isin(clean_h).sum()), clean_share=float(df.h[mc].isin(clean_h).mean()),
                         ceil_majority_f1=a, ceil_rarest_f1=b, ceil_majority_f1_test=a2, ceil_rarest_f1_test=b2, test_rows=int((te & mc).sum())))
    print(f'{name}: macro ceiling majority {f_maj.mean():.4f} / rarest-class {f_rare.mean():.4f} | chrono-test majority {f_maj_te.mean():.4f} / rarest {f_rare_te.mean():.4f}', flush=True)
    if name == 'unsw_nb15':
        import xgboost as xgb
        cls_idx = {c: i for i, c in enumerate(classes)}; yi = np.array([cls_idx[v] for v in y])
        for wname, w in [('plain', None), ('sqrt_balanced', np.sqrt(len(yi[tr]) / (len(classes) * np.bincount(yi[tr], minlength=len(classes))))[yi[tr]])]:
            t0 = time.time()
            clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=32, random_state=42, objective='multi:softprob')
            clf.fit(Xs[tr], yi[tr], sample_weight=w)
            pred = clf.predict(Xs[te]); P, R, F, S = precision_recall_fscore_support(yi[te], pred, labels=range(len(classes)), zero_division=0)
            print(f'\n=== unsw_nb15 XGB ({wname}) chrono 60/20/20 by class, test {te.sum():,} rows, {time.time()-t0:.0f}s: macro-F1 {F.mean():.4f}', flush=True)
            clean_te = df.h[te].isin(clean_h).to_numpy()
            Fc = f1_score(yi[te][clean_te], pred[clean_te], average=None, labels=range(len(classes)))
            tab = pd.DataFrame({'class': classes, 'test_n': S, 'P': P, 'R': R, 'F1': F, 'F1_clean_rows_only': Fc, 'ceil_majority': f_maj_te, 'ceil_rarest': f_rare_te})
            print(tab.to_string(index=False, float_format=lambda v: f'{v:.4f}'), flush=True)
            tab.to_csv(f'{SP}/unsw_xgb_{wname}.csv', index=False)
pd.DataFrame(rows).to_csv(f'{SP}/audit_ceilings.csv', index=False); print('DONE', flush=True)

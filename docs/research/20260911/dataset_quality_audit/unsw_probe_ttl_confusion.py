import pickle, time, sys, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import xgboost as xgb
SP = sys.argv[1]
d = pickle.load(open('data/nfv3_energy_suite_uncapped_scenarios.pkl', 'rb'))
m = np.where(np.asarray(d['dataset_names']).astype(str) == 'unsw_nb15')[0]
X = np.nan_to_num(np.asarray(d['X'][m], dtype=np.float32)); y = np.asarray(d['families'])[m].astype(str); tt = np.asarray(d['timestamps'])[m].astype(np.int64)
feat = [str(f) for f in d['feature_names']]; del d
classes = sorted(np.unique(y)); ci = {c: i for i, c in enumerate(classes)}; yi = np.array([ci[v] for v in y])
tr = np.zeros(len(y), bool); te = np.zeros(len(y), bool)
for c in classes:
    idx = np.where(y == c)[0]; idx = idx[np.argsort(tt[idx], kind='stable')]; k1, k2 = int(len(idx)*.6), int(len(idx)*.8); tr[idx[:k1]] = True; te[idx[k2:]] = True
ben = ci['benign']; isatk = (yi != ben).astype(int)
# TTL / protocol shortcut for benign-vs-attack
for cols in [['MIN_TTL', 'MAX_TTL'], ['MIN_TTL', 'MAX_TTL', 'PROTOCOL'], ['PROTOCOL', 'L7_PROTO']]:
    j = [feat.index(c) for c in cols]; t = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[tr][:, j], isatk[tr])
    p = t.predict(X[te][:, j]); acc = (p == isatk[te]).mean(); f = f1_score(isatk[te], p)
    print(f'binary benign-vs-attack with only {cols}: acc {acc:.4f} attack-F1 {f:.4f}', flush=True)
# XGB plain with predictions -> confusion + which classes absorb tail errors; also XGB without TTL features
for tag, drop in [('all46', []), ('noTTL', ['MIN_TTL', 'MAX_TTL'])]:
    keep = [i for i, f in enumerate(feat) if f not in drop]
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=32, random_state=42)
    t0 = time.time(); clf.fit(X[tr][:, keep], yi[tr]); pred = clf.predict(X[te][:, keep])
    P, R, F, S = precision_recall_fscore_support(yi[te], pred, labels=range(len(classes)), zero_division=0)
    print(f'\n=== XGB {tag} ({time.time()-t0:.0f}s): macro-F1 {F.mean():.4f}; per-class F1', dict(zip(classes, np.round(F, 3))), flush=True)
    cm = pd.DataFrame(confusion_matrix(yi[te], pred, labels=range(len(classes))), index=classes, columns=classes)
    print('confusion (rows=true, cols=pred):'); print(cm.to_string())
    np.save(f'{SP}/unsw_xgb_{tag}_pred.npy', pred)
np.save(f'{SP}/unsw_test_mask.npy', te); np.save(f'{SP}/unsw_y.npy', yi)
print('DONE', flush=True)

"""Ceilings v2 (read-only). For each dataset and split basis (all rows / chrono test rows, per-scenario split as the pipeline):
per-class MAX F1 over hash-consistent labelings (purity-sorted group selection; exact for F1), majority-assignment F1 for reference,
train->test label-conflict rate. Plus CIC bank-limited hash-consistent oracle on the EXP39 test set and the twin derivation."""
import json, sys, numpy as np, pandas as pd
SP, OUT = sys.argv[1], sys.argv[2]
def max_f1_per_class(h, y, classes):
    df = pd.DataFrame({'h': h, 'y': y}); cnt = df.groupby(['h', 'y']).size().unstack(fill_value=0); tot = cnt.sum(1)
    res = {}
    for c in classes:
        if c not in cnt.columns or cnt[c].sum() == 0: res[c] = 0.0; continue
        nc = cnt[c]; sel = nc > 0; nc, n = nc[sel].to_numpy(float), tot[sel].to_numpy(float)
        order = np.argsort(-(nc / n), kind='stable'); tp = np.cumsum(nc[order]); fp = np.cumsum((n - nc)[order]); Nc = nc.sum()
        f1 = 2 * tp / (2 * tp + fp + (Nc - tp)); res[c] = float(f1.max())
    return res
def majority_f1(h, y, classes):
    from sklearn.metrics import f1_score
    df = pd.DataFrame({'h': h, 'y': y}); maj = df.groupby(['h', 'y']).size().reset_index(name='c').sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
    pred = df.h.map(maj).to_numpy(); f = f1_score(y, pred, average=None, labels=classes); return dict(zip(classes, map(float, f)))
rows = []
for name in ['cse_cic_ids2018', 'ton_iot', 'bot_iot', 'unsw_nb15']:
    h = np.load(f'{SP}/hash_{name}.npy'); y = np.load(f'{SP}/y_{name}.npy', allow_pickle=True).astype(str); t = np.load(f'{SP}/t_{name}.npy'); sc = np.load(f'{SP}/scen_{name}.npy', allow_pickle=True).astype(str)
    classes = sorted(np.unique(y)); tr = np.zeros(len(y), bool); te = np.zeros(len(y), bool)
    for s_ in np.unique(sc):
        idx = np.where(sc == s_)[0]; idx = idx[np.argsort(t[idx], kind='stable')]; tr[idx[:int(len(idx) * .6)]] = True; te[idx[int(len(idx) * .8):]] = True
    mx_all, mj_all = max_f1_per_class(h, y, classes), majority_f1(h, y, classes)
    mx_te, mj_te = max_f1_per_class(h[te], y[te], classes), majority_f1(h[te], y[te], classes)
    # train->test conflict: test row whose vector exists in train with a different train-majority label; and test-internal mixed rows
    trdf = pd.DataFrame({'h': h[tr], 'y': y[tr]}); trlab = trdf.groupby(['h', 'y']).size().reset_index(name='c').sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
    seen_lab = pd.Series(h[te]).map(trlab); conflict = seen_lab.notna().to_numpy() & (seen_lab.to_numpy() != y[te])
    tedf = pd.DataFrame({'h': h[te], 'y': y[te]}); nl = tedf.groupby('h')['y'].nunique(); mixed_te = tedf.h.isin(set(nl[nl > 1].index)).to_numpy()
    for c in classes:
        mte = y[te] == c
        rows.append(dict(dataset=name, cls=c, rows_all=int((y == c).sum()), rows_test=int(mte.sum()), ceil_maxf1_all=mx_all[c], ceil_majority_all=mj_all[c],
                         ceil_maxf1_test=mx_te[c], ceil_majority_test=mj_te[c], test_rows_in_test_internal_conflict=float(mixed_te[mte].mean()) if mte.any() else np.nan,
                         test_rows_conflicting_with_train_label=float(conflict[mte].mean()) if mte.any() else np.nan))
    print(f'{name}: macro ceiling (mean of per-class max) all {np.mean(list(mx_all.values())):.4f} test {np.mean(list(mx_te.values())):.4f} | majority all {np.mean(list(mj_all.values())):.4f} test {np.mean(list(mj_te.values())):.4f} | test-internal mixed {mixed_te.mean():.3f} | train-conflict {conflict.mean():.3f}', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/ceilings_v2.csv', index=False)
# CIC bank-limited oracle: see cic_bank_oracle.py (separate script)
print('DONE', flush=True)

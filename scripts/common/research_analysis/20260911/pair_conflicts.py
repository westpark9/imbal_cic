"""Class-pair conflict table per dataset (all rows and chrono-test rows): rows of class A whose vector also carries label B."""
import json, sys, numpy as np, pandas as pd
SP, OUT = sys.argv[1], sys.argv[2]; res = {}
for name in ['cse_cic_ids2018', 'ton_iot', 'bot_iot', 'unsw_nb15']:
    h = np.load(f'{SP}/hash_{name}.npy'); y = np.load(f'{SP}/y_{name}.npy', allow_pickle=True).astype(str); t = np.load(f'{SP}/t_{name}.npy'); sc = np.load(f'{SP}/scen_{name}.npy', allow_pickle=True).astype(str)
    te = np.zeros(len(y), bool)
    for s_ in np.unique(sc):
        idx = np.where(sc == s_)[0]; idx = idx[np.argsort(t[idx], kind='stable')]; te[idx[int(len(idx) * .8):]] = True
    out = {}
    for basis, m in [('all', np.ones(len(y), bool)), ('test', te)]:
        df = pd.DataFrame({'h': h[m], 'y': y[m]}); cnt = df.groupby(['h', 'y']).size().unstack(fill_value=0); classes = list(cnt.columns); tot = {c: int((df.y == c).sum()) for c in classes}
        pairs = []
        for a in classes:
            ha = cnt.index[cnt[a] > 0]
            for b in classes:
                if a == b: continue
                shared = cnt.loc[ha]; shared = shared[shared[b] > 0]
                ra = int(shared[a].sum())
                if ra == 0: continue
                pairs.append(dict(cls=a, other=b, rows_of_cls_sharing=ra, frac_of_cls=ra / tot[a], rows_of_other_on_same_vectors=int(shared[b].sum()), vectors=int(len(shared))))
        out[basis] = dict(class_rows=tot, pairs=sorted(pairs, key=lambda d: -d['rows_of_cls_sharing']))
    res[name] = out; print(name, 'pairs(all)', len(out['all']['pairs']), 'pairs(test)', len(out['test']['pairs']), flush=True)
json.dump(res, open(f'{OUT}/pair_conflicts.json', 'w'), indent=1); print('DONE', flush=True)

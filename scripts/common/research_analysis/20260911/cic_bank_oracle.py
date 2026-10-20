"""CIC-IDS2018 EXP39 test set: twin derivation + bank-limited hash-consistent per-class max F1 (forced groups counted)."""
import json, sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
OUT = sys.argv[1]; C = 'tabpfn/resume/exp38_frozen_seed42_20260909_v2'; meta = json.load(open(f'{C}/COMPLETE.json')); names = meta['class_names']
y = np.load(f'{C}/eval_y.npy'); hh = np.load(f'{C}/eval_hash.npy'); preds = [np.asarray(np.load(f'{C}/eval_p{k}.npy', mmap_mode='r')).argmax(1) for k in range(4)]
inf, ben = names.index('infiltration'), names.index('benign'); glob = preds[0]
bank = np.zeros(len(y), bool)
for p in preds[1:]: bank |= p == y
bank &= glob != y
corr_inf = bank & (y == inf); ben_h = np.unique(hh[y == ben]); twin = np.isin(hh[corr_inf], ben_h); twin_h = np.unique(hh[corr_inf][twin])
deriv = dict(bank_correctable_infiltration=int(corr_inf.sum()), with_benign_twin=int(twin.sum()), distinct_twin_vectors=int(len(twin_h)), benign_rows_sharing=int(np.isin(hh[y == ben], twin_h).sum()),
             row_oracle_macro=float(f1_score(y, np.where(bank, y, glob), average='macro')), row_oracle_per_class=dict(zip(names, map(float, f1_score(y, np.where(bank, y, glob), average=None)))))
df = pd.DataFrame({'h': hh, 'y': y}); P = np.stack(preds, 1)
res = {}
for c in range(len(names)):
    anyc = (P == c).any(1); allc = (P == c).all(1)
    g = df.groupby('h'); n = g.size(); nc = g['y'].apply(lambda v: int((v == c).sum()))
    any_g = pd.Series(anyc).groupby(hh).any(); all_g = pd.Series(allc).groupby(hh).all()
    forced = all_g[all_g].index; optional = any_g[any_g & ~all_g].index
    tp0 = float(nc.loc[forced].sum()); fp0 = float((n.loc[forced] - nc.loc[forced]).sum()); Nc = float((y == c).sum())
    no, nco = n.loc[optional].to_numpy(float), nc.loc[optional].to_numpy(float)
    order = np.argsort(-(nco / no), kind='stable'); tp = tp0 + np.concatenate([[0], np.cumsum(nco[order])]); fp = fp0 + np.concatenate([[0], np.cumsum((no - nco)[order])])
    f1 = 2 * tp / np.maximum(2 * tp + fp + (Nc - tp), 1e-9); res[names[c]] = float(f1.max())
    print(names[c], 'forced groups', len(forced), 'optional', len(optional), 'maxF1', round(res[names[c]], 4), flush=True)
deriv['bank_limited_hash_consistent_maxf1_per_class'] = res; deriv['bank_limited_hash_consistent_macro_upper_bound'] = float(np.mean(list(res.values())))
deriv['global_per_class_f1'] = dict(zip(names, map(float, f1_score(y, glob, average=None)))); deriv['global_macro'] = float(f1_score(y, glob, average='macro'))
json.dump(deriv, open(f'{OUT}/cic_bank_oracle_derivation.json', 'w'), indent=1); print(json.dumps({k: v for k, v in deriv.items() if 'per_class' not in k}, indent=1)); print('DONE', flush=True)

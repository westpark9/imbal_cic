import json, sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
OUT = sys.argv[1]; C = 'tabpfn/resume/exp38_frozen_seed42_20260909_v2'; meta = json.load(open(f'{C}/COMPLETE.json')); names = meta['class_names']; K = len(names)
y = np.load(f'{C}/eval_y.npy'); h = np.load(f'{C}/eval_hash.npy'); P = [np.asarray(np.load(f'{C}/eval_p{k}.npy', mmap_mode='r')).argmax(1) for k in range(4)]; glob = P[0]
cnt = pd.DataFrame({'h': h, 'y': y}).groupby(['h', 'y']).size().reset_index(name='c'); maj = cnt.sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
nlab = cnt.groupby('h').size(); conflict = pd.Series(h).isin(set(nlab[nlab > 1].index)).to_numpy()
target = np.where(conflict, pd.Series(h).map(maj).to_numpy(), y)          # what a hash-consistent router can at best aim for
anyc = np.zeros(len(y), bool)
for p in P: anyc |= p == target
row_oracle = np.where(np.any([p == y for p in P], axis=0), y, glob)        # EXP39 correctness oracle (per row)
real_oracle = np.where(anyc, target, glob)                                  # same bank, majority-label constraint on conflict vectors
res = {}
for name, pred in [('global', glob), ('row_level_oracle_bank', row_oracle), ('realistic_oracle_bank', real_oracle), ('realistic_oracle_perfect', np.where(conflict, target, y))]:
    f = f1_score(y, pred, average=None, labels=range(K)); res[name] = dict(macro=float(f.mean()), per_class=dict(zip(names, map(float, f))))
    print(f'{name:26s} macro {f.mean():.4f} | ' + ' '.join(f'{n}={v:.3f}' for n, v in zip(names, f)))
res['conflict_rows_in_test'] = int(conflict.sum()); res['rows_where_target_ne_truth'] = int((target != y).sum())
json.dump(res, open(f'{OUT}/cic_bank_realistic_oracle.json', 'w'), indent=1); print('DONE')

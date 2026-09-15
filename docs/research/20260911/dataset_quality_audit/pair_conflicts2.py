"""Unordered class-pair conflicts per dataset (all rows): rows of A and of B on shared vectors, % of each class, vectors, time spans, top scenarios."""
import json, sys, numpy as np, pandas as pd, datetime as dt
SP, OUT = sys.argv[1], sys.argv[2]; res = {}
def day(ms): return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime('%m-%d')
for name in ['cse_cic_ids2018', 'ton_iot', 'bot_iot', 'unsw_nb15']:
    h = np.load(f'{SP}/hash_{name}.npy'); y = np.load(f'{SP}/y_{name}.npy', allow_pickle=True).astype(str); t = np.load(f'{SP}/t_{name}.npy'); sc = np.load(f'{SP}/scen_{name}.npy', allow_pickle=True).astype(str)
    df = pd.DataFrame({'h': h, 'y': y, 't': t, 's': sc}); cnt = df.groupby(['h', 'y']).size().unstack(fill_value=0); classes = list(cnt.columns); tot = {c: int((y == c).sum()) for c in classes}
    pairs = []
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            shared = cnt[(cnt[a] > 0) & (cnt[b] > 0)]
            if len(shared) == 0: continue
            hs = set(shared.index); m = df.h.isin(hs).to_numpy()
            rec = dict(a=a, b=b, rows_a=int(shared[a].sum()), rows_b=int(shared[b].sum()), frac_a=float(shared[a].sum() / tot[a]), frac_b=float(shared[b].sum() / tot[b]), vectors=int(len(shared)))
            for k, c in (('a', a), ('b', b)):
                mm = m & (y == c); rec[f'span_{k}'] = f'{day(t[mm].min())}–{day(t[mm].max())}'; rec[f'scen_{k}'] = {s_: int(v) for s_, v in pd.Series(sc[mm]).value_counts().head(3).items()}
            pairs.append(rec)
    res[name] = dict(class_rows=tot, pairs=sorted(pairs, key=lambda d: -(d['rows_a'] + d['rows_b']))); print(name, len(pairs), flush=True)
json.dump(res, open(f'{OUT}/pair_conflicts_unordered.json', 'w'), indent=1, ensure_ascii=False); print('DONE')

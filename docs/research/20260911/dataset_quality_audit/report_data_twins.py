"""Top conflicting vectors with their actual feature values (needs X). Output JSON to scratchpad."""
import pickle, json, sys, numpy as np, pandas as pd
SP = sys.argv[1]
d = pickle.load(open('data/nfv3_energy_suite_uncapped_scenarios.pkl', 'rb'))
X = d['X']; ds = np.asarray(d['dataset_names']).astype(str); fam = np.asarray(d['families']).astype(str)
scen = np.asarray(d['attack_scenarios']).astype(str); ts = np.asarray(d['timestamps']).astype(np.int64); feat = [str(f) for f in d['feature_names']]; del d
out = {}
for name in ['cse_cic_ids2018', 'ton_iot', 'bot_iot', 'unsw_nb15']:
    m = np.where(ds == name)[0]; h = np.load(f'{SP}/hash_{name}.npy'); y = fam[m]; sc = scen[m]; tt = ts[m]
    df = pd.DataFrame({'h': h, 'y': y, 's': sc, 't': tt, 'i': np.arange(len(m))})
    cnt = df.groupby(['h', 'y']).size().unstack(fill_value=0)
    mixed = cnt[(cnt > 0).sum(1) > 1].copy(); mixed['total'] = mixed.sum(1)
    # rank: conflicting mass = total minus largest label (rows that necessarily get the wrong label under any single assignment)
    mixed['minority'] = mixed['total'] - mixed.drop(columns='total').max(1)
    top = mixed.sort_values('minority', ascending=False).head(10)
    groups = []
    for hh, row in top.iterrows():
        rows = df[df.h == hh]; i0 = int(rows.i.iloc[0])
        vec = {f: float(v) for f, v in zip(feat, np.nan_to_num(np.asarray(X[m[i0]], dtype=np.float32)))}
        labels = {c: int(row[c]) for c in cnt.columns if row[c] > 0}
        scens = rows.groupby(['y', 's']).size().to_dict(); scens = {f'{k[0]}/{k[1]}': int(v) for k, v in scens.items()}
        tspan = {c: [int(rows.t[rows.y == c].min()), int(rows.t[rows.y == c].max())] for c in labels}
        groups.append(dict(hash=str(hh), total=int(row.total), minority=int(row.minority), labels=labels, scenarios=scens, time_span_ms=tspan, vector=vec))
    # per-class decomposition: clean / twin with benign / twin with other attack only
    ben_h = set(df.h[df.y == 'benign']); nl = (cnt > 0).sum(1); clean_h = set(nl[nl == 1].index)
    dec = {}
    for c in sorted(np.unique(y)):
        hc = df.h[df.y == c]; clean = hc.isin(clean_h); wb = hc.isin(ben_h) & ~clean if c != 'benign' else pd.Series(False, index=hc.index)
        dec[c] = dict(rows=int(len(hc)), clean=int(clean.sum()), twin_benign=int(wb.sum()), twin_other=int((~clean & ~wb).sum()))
    # conflicting mass overall: rows that a single-label-per-vector classifier must get wrong
    out[name] = dict(feature_names=feat, top_conflicts=groups, decomposition=dec, mixed_groups=int(len(mixed)), unavoidable_error_rows=int(mixed.minority.sum()), rows=int(len(m)))
    print(name, 'mixed groups', len(mixed), 'unavoidable rows', int(mixed.minority.sum()), flush=True)
json.dump(out, open(f'{SP}/report_twins.json', 'w'), ensure_ascii=False, indent=1); print('DONE', flush=True)

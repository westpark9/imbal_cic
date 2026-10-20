"""Read-only data-quality audit of the NF-v3 suite (all four datasets). Writes CSVs to the scratchpad only."""
import pickle, time, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
SP = sys.argv[1]; PKL = 'data/nfv3_energy_suite_uncapped_scenarios.pkl'
t0 = time.time(); d = pickle.load(open(PKL, 'rb')); print(f'loaded in {time.time()-t0:.0f}s; keys:', flush=True)
for k, v in d.items():
    try: print('  ', k, type(v).__name__, getattr(v, 'dtype', ''), getattr(v, 'shape', len(v) if hasattr(v, '__len__') else ''))
    except Exception as e: print('  ', k, type(v).__name__, e)
X = None
for k, v in d.items():
    if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] > 10: X = v; xkey = k
if X is None:
    for k, v in d.items():
        if hasattr(v, 'shape') and len(getattr(v, 'shape', ())) == 2: X = np.asarray(v); xkey = k
print('X key', xkey, X.shape, X.dtype, flush=True)
ds = np.asarray(d['dataset_names']).astype(str); fam = np.asarray(d['families']).astype(str)
scen = np.asarray(d['attack_scenarios']).astype(str) if 'attack_scenarios' in d else fam
ts = np.asarray(d['timestamps'] if 'timestamps' in d else d.get('time_proxy')).astype(np.int64)
feat_names = [str(x) for x in d.get('feature_names', [])] if 'feature_names' in d else []
print('feature_names:', feat_names[:46], flush=True)
del d
summary_rows, class_rows, scen_rows = [], [], []
for name in ['bot_iot', 'ton_iot', 'unsw_nb15', 'cse_cic_ids2018']:
    m = np.where(ds == name)[0]
    if len(m) == 0: print('no rows for', name); continue
    t1 = time.time()
    Xs = np.nan_to_num(np.asarray(X[m], dtype=np.float32)); y = fam[m]; sc = scen[m]; tt = ts[m]
    h = pd.util.hash_pandas_object(pd.DataFrame(Xs), index=False).to_numpy()
    n = len(m); nh = len(np.unique(h))
    df = pd.DataFrame({'h': h, 'y': y, 's': sc, 't': tt})
    # mixed-label hash groups
    g = df.groupby('h')['y']
    nlab = g.nunique(); mixed = set(nlab[nlab > 1].index)
    df['mixed'] = df.h.isin(mixed)
    # hash-majority ceiling
    maj = df.groupby(['h', 'y']).size().reset_index(name='c').sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
    pred = df.h.map(maj).to_numpy()
    classes = sorted(np.unique(y)); f1 = f1_score(y, pred, average=None, labels=classes)
    # twin with benign
    ben_h = set(df.h[df.y == 'benign'].unique())
    # timestamps: real?
    tuniq = len(np.unique(tt)); tmin, tmax = int(tt.min()), int(tt.max())
    # chronological split per scenario (60/20/20 by time within scenario): test rows with hash in train
    tr = np.zeros(n, bool); te = np.zeros(n, bool)
    for s_ in np.unique(sc):
        idx = np.where(sc == s_)[0]; idx = idx[np.argsort(tt[idx], kind='stable')]
        k1, k2 = int(len(idx) * .6), int(len(idx) * .8)
        tr[idx[:k1]] = True; te[idx[k2:]] = True
    train_h = df.h[tr]; te_df = df[te]
    train_lab = df[tr].groupby('h')['y'].agg(lambda v: v.mode().iloc[0])
    seen = te_df.h.isin(set(train_h.unique()))
    seen_other = te_df.h.map(train_lab); conflict = seen & (seen_other.to_numpy() != te_df.y.to_numpy())
    summary_rows.append(dict(dataset=name, rows=n, unique_vectors=nh, dup_rate=1 - nh / n, n_classes=len(classes), n_scenarios=len(np.unique(sc)),
        rows_in_mixed_label_groups=int(df.mixed.sum()), frac_mixed=float(df.mixed.mean()), hash_majority_macro_f1=float(f1.mean()),
        ts_unique=tuniq, ts_min=tmin, ts_max=tmax, chrono_test_rows=int(te.sum()), chrono_test_seen_in_train=float(seen.mean()),
        chrono_test_seen_with_other_train_label=float(conflict.mean())))
    for c, f in zip(classes, f1):
        mc = df.y == c; hc = df.h[mc]
        twin_any = df.mixed[mc].mean(); twin_ben = hc.isin(ben_h).mean() if c != 'benign' else np.nan
        tq = np.quantile(tt[mc.to_numpy()], [0, .1, .5, .9, 1]) if mc.any() else [np.nan]*5
        tem = te & mc.to_numpy()
        class_rows.append(dict(dataset=name, cls=c, rows=int(mc.sum()), share=float(mc.mean()), unique_vectors=int(hc.nunique()),
            frac_rows_with_other_label_twin=float(twin_any), frac_rows_with_benign_twin=float(twin_ben) if c != 'benign' else np.nan,
            hash_majority_f1=float(f), n_scenarios=int(df.s[mc].nunique()),
            t_q0=int(tq[0]), t_q10=int(tq[1]), t_q50=int(tq[2]), t_q90=int(tq[3]), t_q100=int(tq[4]),
            chrono_test_rows=int(tem.sum()), chrono_test_seen_in_train=float(seen[tem[te]].mean()) if tem.any() else np.nan,
            chrono_test_conflict=float(conflict[tem[te]].mean()) if tem.any() else np.nan))
    # scenario degeneracy and cross-class sharing
    for s_, gs in df.groupby('s'):
        vc = gs.h.value_counts(); top = int(vc.iloc[0]); nu = len(vc)
        other = df[(df.s != s_) & df.h.isin(set(gs.h.unique()))]
        share_by = other.groupby('s').size().sort_values(ascending=False).head(3).to_dict()
        scen_rows.append(dict(dataset=name, scenario=s_, cls=gs.y.iloc[0], rows=len(gs), unique_vectors=nu, top_vector_share=top / len(gs),
            rows_sharing_vector_with_other_scenario=int(gs.h.isin(set(other.h.unique())).sum()), top_sharing_scenarios=json.dumps(share_by),
            t_min=int(gs.t.min()), t_max=int(gs.t.max())))
    print(f'{name}: rows {n:,} uniq {nh:,} dup {1-nh/n:.3f} mixed {df.mixed.mean():.3f} hash-majority macro {f1.mean():.4f} '
          f'chrono-test seen {seen.mean():.3f} conflict {conflict.mean():.3f} ({time.time()-t1:.0f}s)', flush=True)
    del Xs, df, h
pd.DataFrame(summary_rows).to_csv(f'{SP}/audit_summary.csv', index=False)
pd.DataFrame(class_rows).to_csv(f'{SP}/audit_classes.csv', index=False)
pd.DataFrame(scen_rows).to_csv(f'{SP}/audit_scenarios.csv', index=False)
print('DONE', flush=True)

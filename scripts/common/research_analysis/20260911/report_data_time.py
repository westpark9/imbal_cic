"""Time composition, class timelines, memorization under random vs chronological split (hash lookup). No X needed."""
import json, sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
SP = sys.argv[1]; rng = np.random.default_rng(42); out = {}
for name in ['cse_cic_ids2018', 'ton_iot', 'bot_iot', 'unsw_nb15']:
    h = np.load(f'{SP}/hash_{name}.npy'); y = np.load(f'{SP}/y_{name}.npy', allow_pickle=True).astype(str); t = np.load(f'{SP}/t_{name}.npy'); sc = np.load(f'{SP}/scen_{name}.npy', allow_pickle=True).astype(str)
    order = np.argsort(t, kind='stable'); classes = sorted(np.unique(y)); n = len(y)
    # (1) composition over 60 time-ordered row bins
    nb = 60; edges = np.linspace(0, n, nb + 1).astype(int); bins = []
    for b in range(nb):
        idx = order[edges[b]:edges[b + 1]]; vc = pd.Series(y[idx]).value_counts()
        bins.append(dict(t_start=int(t[idx].min()), t_end=int(t[idx].max()), rows=int(len(idx)), share={c: float(vc.get(c, 0) / len(idx)) for c in classes}))
    # (2) per-class timeline with per-class chronological 60/20/20 boundaries (what the pipeline does per family)
    tl = {}
    for c in classes:
        idx = np.where(y == c)[0]; idx = idx[np.argsort(t[idx], kind='stable')]; k1, k2 = int(len(idx) * .6), int(len(idx) * .8)
        tl[c] = dict(rows=int(len(idx)), t0=int(t[idx[0]]), t_train_end=int(t[idx[max(k1 - 1, 0)]]), t_val_end=int(t[idx[max(k2 - 1, 0)]]), t1=int(t[idx[-1]]),
                     scenarios={s: int(v) for s, v in pd.Series(sc[idx]).value_counts().items()})
    # (3) hash-lookup classifier: random 80/20 vs per-class chronological (train first 60%, test last 20%)
    def lookup_eval(tr, te):
        trdf = pd.DataFrame({'h': h[tr], 'y': y[tr]}); maj = trdf.groupby(['h', 'y']).size().reset_index(name='c').sort_values('c', ascending=False).drop_duplicates('h').set_index('h')['y']
        glob = trdf.y.value_counts().index[0]; pred = pd.Series(h[te]).map(maj).fillna(glob).to_numpy(); seen = pd.Series(h[te]).isin(maj.index).to_numpy()
        f = f1_score(y[te], pred, average=None, labels=classes)
        return dict(acc=float((pred == y[te]).mean()), macro_f1=float(f.mean()), per_class_f1={c: float(v) for c, v in zip(classes, f)}, seen_frac=float(seen.mean()), acc_on_seen=float((pred[seen] == y[te][seen]).mean()) if seen.any() else None, test_rows=int(len(te)))
    perm = rng.permutation(n); k = int(n * .8); rand = lookup_eval(perm[:k], perm[k:])
    tr = np.zeros(n, bool); te = np.zeros(n, bool)
    for c in classes:
        idx = np.where(y == c)[0]; idx = idx[np.argsort(t[idx], kind='stable')]; tr[idx[:int(len(idx) * .6)]] = True; te[idx[int(len(idx) * .8):]] = True
    chrono = lookup_eval(np.where(tr)[0], np.where(te)[0])
    vc = pd.Series(h).value_counts(); dup_rows = float((vc[vc >= 2]).sum() / n)
    out[name] = dict(classes=classes, rows=int(n), bins=bins, timeline=tl, lookup_random=rand, lookup_chrono=chrono, rows_with_repeated_vector=dup_rows)
    print(f"{name}: hash-lookup random acc {rand['acc']:.4f} macro {rand['macro_f1']:.4f} seen {rand['seen_frac']:.3f} | chrono acc {chrono['acc']:.4f} macro {chrono['macro_f1']:.4f} seen {chrono['seen_frac']:.3f}", flush=True)
json.dump(out, open(f'{SP}/report_time.json', 'w'), ensure_ascii=False); print('DONE', flush=True)

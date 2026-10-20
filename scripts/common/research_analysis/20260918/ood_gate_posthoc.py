"""Post-hoc OOD gate on the frozen global predictions: rows predicted as a tail class whose 1-NN
z-distance to the train pool exceeds tau are re-assigned to benign. tau candidates come from the
TRAIN-only LOO 1-NN distance distribution (no val/test labels used). Also: which features are
extreme for the far-away benign rows (data-quality readout)."""
import argparse, sys, json
import numpy as np
sys.path.insert(0, "/home/user/Desktop/imbalcic/tabpfn/scripts")
from nfv3_conflict_clean import install_clean_loader
import nfv3_v3_common as core
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

MAN = "/home/user/Desktop/imbalcic/data/derived/toniot_conflict_free_fixed_split_20260916_021730/manifest.json"
CACHE = "/home/user/Desktop/imbalcic/tabpfn/results/20260916_021730_nfv3_toniot_exp48_clean_revalidation_s42/frozen_cache"
install_clean_loader(core, MAN)
X, names, train_idx, val_idx, test_idx, _, _, _, label_fn = core.load_ton_iot(argparse.Namespace(data="/home/user/Desktop/imbalcic/data/nfv3_energy_suite_uncapped_scenarios.pkl"))
feat = getattr(core, "FEATURE_NAMES", None) or [f"f{i}" for i in range(X.shape[1])]
y_tr = label_fn(train_idx); y_te = np.load(f"{CACHE}/eval_y.npy"); g_pred = np.load(f"{CACHE}/eval_p0.npy", mmap_mode="r").argmax(1).copy()
n = len(names); ben = names.index("benign"); rng = np.random.default_rng(42)
# pool: 300k benign + every non-benign train row of the tail classes (all attack classes, capped 20k each)
parts = [rng.choice(train_idx[y_tr == ben], 300_000, replace=False)]
for c in range(n):
    if c == ben: continue
    r = train_idx[y_tr == c]; parts.append(r if len(r) <= 20_000 else rng.choice(r, 20_000, replace=False))
pool = np.concatenate(parts); ypool = label_fn(pool)
Xp = X[pool].astype(np.float64); mu, sd = Xp.mean(0), Xp.std(0); sd[sd == 0] = 1
Z = (Xp - mu) / sd
with threadpool_limits(8):
    nn = NearestNeighbors(n_neighbors=2, n_jobs=8).fit(Z)
    # train LOO distances (subsample 50k of the pool) -> tau candidates
    q = rng.choice(len(pool), 50_000, replace=False)
    d, i = nn.kneighbors(Z[q]); d_loo = np.where(i[:, 0] == q, d[:, 1], d[:, 0])
    taus = {f"p{p}": float(np.quantile(d_loo, p / 100)) for p in [99, 99.9, 99.99]}
    print("train LOO 1-NN dist quantiles:", json.dumps(taus), "max", float(d_loo.max()), flush=True)
    # distance for every test row predicted as a non-benign class (that's where a gate would act)
    act = np.where(g_pred != ben)[0]
    print("test rows predicted non-benign:", len(act), flush=True)
    dist = np.empty(len(test_idx)); dist[:] = np.nan
    for s in range(0, len(act), 200_000):
        sl = act[s:s + 200_000]
        dd, _ = nn.kneighbors((X[test_idx[sl]].astype(np.float64) - mu) / sd, n_neighbors=1)
        dist[sl] = dd[:, 0]
        print(f"  dist {min(s + 200_000, len(act)):,}/{len(act):,}", flush=True)

def prf(pred):
    cm = np.bincount(y_te.astype(np.int64) * n + pred, minlength=n * n).reshape(n, n)
    tp = np.diag(cm); sup, pr = cm.sum(1), cm.sum(0)
    f1 = np.divide(2 * tp, sup + pr, out=np.zeros(n), where=sup + pr > 0)
    return f1, np.divide(tp, pr, out=np.zeros(n), where=pr > 0), np.divide(tp, sup, out=np.zeros(n), where=sup > 0)

f1g, pg, rg = prf(g_pred)
print(f"\nglobal: macro {f1g.mean():.4f} | " + " ".join(f"{names[c]}={f1g[c]:.3f}" for c in range(n)), flush=True)
rows = []
for tag, tau in list(taus.items()) + [("t20", 20.0), ("t50", 50.0), ("t100", 100.0)]:
    pred = g_pred.copy(); far = (~np.isnan(dist)) & (dist > tau)
    pred[far] = ben
    f1, p, r = prf(pred)
    moved = int(far.sum()); moved_right = int((far & (y_te == ben)).sum())
    print(f"gate {tag:6s} tau={tau:9.3f}: moved {moved:,} (benign among them {moved_right:,}) macro {f1.mean():.4f} | "
          + " ".join(f"{names[c]}={f1[c]:.3f}" for c in range(n)) + f" | mitm P/R {p[names.index('mitm')]:.3f}/{r[names.index('mitm')]:.3f}", flush=True)
    rows.append({"tag": tag, "tau": tau, "moved": moved, "moved_benign": moved_right, "macro": float(f1.mean()), "f1": {names[c]: float(f1[c]) for c in range(n)}})

# which features are extreme for the far-away benign rows predicted mitm?
mid = names.index("mitm"); far_rows = test_idx[(y_te == ben) & (g_pred == mid) & (dist > 50)]
zf = np.abs((X[far_rows].astype(np.float64) - mu) / sd)
med = np.median(zf, 0); top = np.argsort(-med)[:8]
print("\nfar benign->mitm rows:", len(far_rows), "median |z| top features:", [(feat[j], round(float(med[j]), 1)) for j in top], flush=True)
print("raw medians of those features (far rows vs train benign):",
      [(feat[j], float(np.median(X[far_rows][:, j])), float(np.median(Xp[ypool == ben][:, j]))) for j in top[:5]], flush=True)
json.dump({"taus": taus, "gates": rows}, open("/tmp/claude-1000/-home-user-Desktop-imbalcic/304cd430-ed38-41b0-975c-14668583ec8d/scratchpad/ood_gate_posthoc.json", "w"), indent=1)
print("DONE", flush=True)

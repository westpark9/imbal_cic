"""Feature-space separability of weak ToN classes vs benign on the CLEAN train split.
For every mitm / ransomware / scanning / injection train row: leave-one-out 1-NN and 5-NN
label among (benign sample + all rows of the four classes), z-scored features."""
import argparse, sys, time, json
import numpy as np
sys.path.insert(0, "/home/user/Desktop/imbalcic/tabpfn/scripts")
from nfv3_conflict_clean import install_clean_loader
import nfv3_v3_common as core
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

MAN = "/home/user/Desktop/imbalcic/data/derived/toniot_conflict_free_fixed_split_20260916_021730/manifest.json"
install_clean_loader(core, MAN)
X, names, train_idx, val_idx, test_idx, _, _, _, label_fn = core.load_ton_iot(argparse.Namespace(data="/home/user/Desktop/imbalcic/data/nfv3_energy_suite_uncapped_scenarios.pkl"))
y_tr = label_fn(train_idx)
rng = np.random.default_rng(42)
ben = names.index("benign")
weak = ["mitm", "ransomware", "scanning", "injection"]
ben_rows = train_idx[y_tr == ben]; ben_s = rng.choice(ben_rows, 300_000, replace=False)
parts = [ben_s] + [train_idx[y_tr == names.index(w)] for w in weak]
pool = np.concatenate(parts); ypool = label_fn(pool)
Xp = X[pool].astype(np.float64)
mu, sd = Xp.mean(0), Xp.std(0); sd[sd == 0] = 1
Z = (Xp - mu) / sd
print("pool", {n: int((ypool == names.index(n)).sum()) for n in ["benign"] + weak}, flush=True)
t0 = time.time()
with threadpool_limits(8):
    nn = NearestNeighbors(n_neighbors=6, n_jobs=8).fit(Z)
    out = {}
    for w in weak:
        wid = names.index(w); q = np.where(ypool == wid)[0]
        d, i = nn.kneighbors(Z[q])
        # drop self (first column is self unless exact duplicates exist; handle by masking index == query)
        keep = i != q[:, None]
        nb = np.array([row[k][:5] for row, k in zip(i, keep)]); dd = np.array([row[k][:5] for row, k in zip(d, keep)])
        lab = ypool[nb]
        nn1 = lab[:, 0]; maj5 = np.array([np.bincount(r, minlength=len(names)).argmax() for r in lab])
        out[w] = {"n": int(len(q)),
                  "1nn_same": float((nn1 == wid).mean()), "1nn_benign": float((nn1 == ben).mean()),
                  "5nn_maj_same": float((maj5 == wid).mean()), "5nn_maj_benign": float((maj5 == ben).mean()),
                  "1nn_dist_median": float(np.median(dd[:, 0])), "1nn_dist_zero_frac": float((dd[:, 0] < 1e-9).mean()),
                  "1nn_benign_dist_zero_frac": float(((dd[:, 0] < 1e-9) & (nn1 == ben)).mean())}
        print(w, json.dumps(out[w]), f"{time.time()-t0:.0f}s", flush=True)
    # symmetric view: benign rows whose 1-NN is a weak class (how much benign is 'inside' each weak class)
    qb = rng.choice(np.where(ypool == ben)[0], 50_000, replace=False)
    d, i = nn.kneighbors(Z[qb], n_neighbors=2)
    nb1 = np.where(i[:, 0] == qb, i[:, 1], i[:, 0]); lab = ypool[nb1]
    out["benign_1nn_label_frac"] = {n: float((lab == names.index(n)).mean()) for n in ["benign"] + weak}
    print("benign", json.dumps(out["benign_1nn_label_frac"]), flush=True)
json.dump(out, open("/tmp/claude-1000/-home-user-Desktop-imbalcic/304cd430-ed38-41b0-975c-14668583ec8d/scratchpad/knn_separability.json", "w"), indent=1)
print("DONE", flush=True)

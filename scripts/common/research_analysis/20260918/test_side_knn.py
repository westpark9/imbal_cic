"""Where do the benign TEST rows that global calls mitm / ransomware sit relative to the TRAIN pool?
1-NN label + distance in the (benign-300k-sample + all mitm + all ransomware + all scanning) train pool."""
import argparse, sys, time, json
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
y_tr = label_fn(train_idx); y_te = np.load(f"{CACHE}/eval_y.npy"); g_pred = np.load(f"{CACHE}/eval_p0.npy", mmap_mode="r").argmax(1)
ben = names.index("benign"); rng = np.random.default_rng(42)
print("test support", {n: int((y_te == i).sum()) for i, n in enumerate(names)}, flush=True)
print("global predicted-as", {n: int((g_pred == i).sum()) for i, n in enumerate(names)}, flush=True)
weak = ["mitm", "ransomware", "scanning"]
ben_s = rng.choice(train_idx[y_tr == ben], 300_000, replace=False)
pool = np.concatenate([ben_s] + [train_idx[y_tr == names.index(w)] for w in weak]); ypool = label_fn(pool)
Xp = X[pool].astype(np.float64); mu, sd = Xp.mean(0), Xp.std(0); sd[sd == 0] = 1
Z = (Xp - mu) / sd
out = {}
with threadpool_limits(8):
    nn = NearestNeighbors(n_neighbors=1, n_jobs=8).fit(Z)
    def probe(tag, rows):
        if len(rows) == 0: out[tag] = None; print(tag, "empty", flush=True); return
        d, i = nn.kneighbors((X[rows].astype(np.float64) - mu) / sd)
        lab = ypool[i[:, 0]]
        out[tag] = {"n": int(len(rows)), "1nn_label_frac": {n: float((lab == names.index(n)).mean()) for n in ["benign"] + weak},
                    "dist_q": [float(v) for v in np.quantile(d[:, 0], [0.1, 0.5, 0.9])], "dist_zero_frac": float((d[:, 0] < 1e-9).mean())}
        print(tag, json.dumps(out[tag]), flush=True)
    for w in weak:
        wid = names.index(w)
        probe(f"test_benign_predicted_{w}", test_idx[(y_te == ben) & (g_pred == wid)])
        probe(f"test_true_{w}", test_idx[y_te == wid])
        probe(f"test_true_{w}_predicted_benign", test_idx[(y_te == wid) & (g_pred == ben)])
    probe("test_benign_predicted_benign_sample", rng.choice(test_idx[(y_te == ben) & (g_pred == ben)], 30_000, replace=False))
json.dump(out, open("/tmp/claude-1000/-home-user-Desktop-imbalcic/304cd430-ed38-41b0-975c-14668583ec8d/scratchpad/test_side_knn.json", "w"), indent=1)
print("DONE", flush=True)

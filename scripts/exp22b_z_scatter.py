#!/usr/bin/env python3
"""Post-hoc analysis: 2-D scatter of the exp22b embedding z-space.

Reproduces a record run's phi (global TabPFN embedding -> PCA) with the run's
own args.json + the frozen exp22b module's helpers, embeds a class-capped
sample of the SAME D_mine rows, assigns each to the run's saved centroids
(system_dump.npz: mu, tau), and renders one PNG with two panels:
  left  = z[:,0] vs z[:,1] colored by true class
  right = same points colored by assigned residual cluster (+ projected mu)
Output: <run_dir>/7a_z_scatter.png  (analysis artifact; run numbers untouched)

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/exp22b_z_scatter.py \
        tabpfn/results/20260825_174233_nfv3_cic2018_exp22b_racephi
"""

import argparse
import gc
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tabpfn"))
import nfv3_v3_common as core  # noqa: E402

SEED_BAND_SCATTER = 2500  # this script's own draw band (unused elsewhere)

spec = importlib.util.spec_from_file_location(
    "exp22b", os.path.join(REPO_ROOT, "tabpfn", "nfv3_v3_exp22b_racepfn_phi.py"))
e22b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e22b)


def render(args, ra, mu, tau, z, y_s, class_names):
    d2 = e22b.sq_dist_to_centroids(z, mu)
    assign = d2.argmin(axis=1)
    print(f"assignment counts: {np.bincount(assign, minlength=len(mu)).tolist()}")
    med = float(np.median(d2[np.arange(len(z)), assign]))
    print(f"median sq-dist of sample to nearest mu: {med:.1f} (run tau {tau:.1f})")

    n_classes = len(class_names)
    cls_colors = {"benign": "#9AA0A8", "bot": "#7C55AD", "brute_force": "#356FAE",
                  "ddos": "#38875F", "dos": "#B26F1D", "infiltration": "#BC4749",
                  "web_attacks": "#E0219E"}   # magenta — dos(ochre)와 절대 비혼동
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharex=True, sharey=True)
    for ax in axes:
        ax.set_xlabel("z[0]")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("z[1]")
    counts = np.bincount(y_s, minlength=n_classes)
    for cid in np.argsort(-counts):      # big first -> rare classes on top
        m = y_s == cid
        if not m.any():
            continue
        rare = counts[cid] < 1000
        axes[0].scatter(z[m, 0], z[m, 1],
                        s=16 if rare else args.point_size,
                        alpha=.9 if rare else .35,
                        lw=.3 if rare else 0,
                        edgecolors="white" if rare else "none",
                        c=cls_colors.get(class_names[cid], "#333"),
                        label=f"{class_names[cid]} ({int(m.sum()):,})",
                        zorder=5 if rare else 2, rasterized=True)
    axes[0].legend(markerscale=3, fontsize=8, loc="best", framealpha=.9)
    axes[0].set_title("true class")

    # Okabe-Ito: blue / orange / green / vermilion — no blue-vs-purple ambiguity
    cl_cmap = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    a_counts = np.bincount(assign, minlength=len(mu))
    for k in np.argsort(-a_counts):      # big clusters first -> small on top
        m = assign == k
        axes[1].scatter(z[m, 0], z[m, 1], s=args.point_size, alpha=.35, lw=0,
                        c=cl_cmap[k % len(cl_cmap)],
                        label=f"cluster {k} ({int(m.sum()):,})", rasterized=True)
    for k in range(len(mu)):             # centroids: numbered circles
        axes[1].scatter(mu[k, 0], mu[k, 1], s=360, facecolors="white",
                        edgecolors="black", linewidths=1.6, zorder=6)
        axes[1].text(mu[k, 0], mu[k, 1], str(k), ha="center", va="center",
                     fontsize=11, fontweight="bold", color="black", zorder=7)
    hs, ls = axes[1].get_legend_handles_labels()
    idx = np.argsort([int(s.split()[1]) for s in ls])
    axes[1].legend([hs[i] for i in idx], [ls[i] for i in idx],
                   markerscale=3, fontsize=8, loc="best", framealpha=.9)
    axes[1].set_title("assigned residual cluster · numbered circles = centroids μ")
    fig.suptitle(f"exp22b z-space (embed φ → PCA{ra['phi_dim']}, dims 0–1) · "
                 f"{os.path.basename(args.run_dir.rstrip(os.sep))} · τ={tau:.1f} · "
                 f"sample {len(z):,} of D_mine", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.run_dir, "7a_z_scatter.png")
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--sample-per-class", type=int, default=25_000)
    ap.add_argument("--point-size", type=float, default=2.5)
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "args.json")) as f:
        ra = json.load(f)
    dump = np.load(os.path.join(args.run_dir, "system_dump.npz"))
    mu, tau = dump["mu"], float(dump["tau"])
    print(f"run args loaded: target={ra['target_dataset']} phi={ra['phi_mode']} "
          f"K={len(mu)} tau={tau:.1f}")
    if ra["phi_mode"] != "embed_pca":
        raise SystemExit("this scatter reproduces embed_pca runs only")

    # cache: the expensive part (z, labels, r) — restyle without recompute
    cache_path = os.path.join(args.run_dir, "7a_z_scatter_data.npz")
    if os.path.exists(cache_path):
        c = np.load(cache_path, allow_pickle=True)
        if int(c["sample_per_class"]) == args.sample_per_class:
            print(f"cache hit: {cache_path} — skipping GPU recompute")
            render(args, ra, mu, tau, c["z"], c["y_s"],
                   [str(x) for x in c["class_names"]])
            return

    class NS:  # loader wants attribute access
        pass
    la = NS()
    for k, v in ra.items():
        setattr(la, k, v)

    cfg = core.build_dataset_config(la.data_dir)
    X, class_names, train_idx, val_idx, test_idx, _, _, _, label_fn = \
        cfg[la.target_dataset]["loader"](la)
    n_classes = len(class_names)
    y_train = label_fn(train_idx)

    d = core.load_pickle(la.data)
    ts_all = np.asarray(d["timestamps" if "timestamps" in d else "time_proxy"],
                        dtype=np.int64)
    del d
    ts_train = ts_all[train_idx]
    del ts_all

    pools, _ = e22b.class_chrono_partition(
        train_idx, y_train, ts_train, la.context_frac, la.mine_frac, class_names)
    ctx_pool, mine_pool = pools["context"], pools["mine"]
    y_ctx = label_fn(ctx_pool)

    g_idx = e22b.representative_subset(
        ctx_pool, y_ctx, n_classes, la.global_context_size,
        la.seed + e22b.SEED_BAND_GLOBAL)
    phi_fit_idx = core.stratified_subset(
        ctx_pool, y_ctx, n_classes, la.phi_fit_rows,
        la.seed + e22b.SEED_BAND_PHI)
    mine_idx = core.stratified_subset(
        mine_pool, label_fn(mine_pool), n_classes, la.mine_max_rows,
        la.seed + e22b.SEED_BAND_MINE)
    sample_idx = core.cap_per_class(
        mine_idx, label_fn(mine_idx), n_classes, args.sample_per_class,
        la.seed + SEED_BAND_SCATTER)

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    glob_ctx = (feats_of(g_idx), label_fn(g_idx))
    X_phi = feats_of(phi_fit_idx)
    X_s, y_s = feats_of(sample_idx), label_fn(sample_idx)
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()
    print(f"sample rows: {len(sample_idx):,} "
          f"({ {n: int((y_s == i).sum()) for i, n in enumerate(class_names)} })",
          flush=True)

    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(
        device=la.device, model_path=la.model_path,
        ignore_pretraining_limits=la.ignore_pretraining_limits,
        random_state=la.seed, n_estimators=la.n_estimators,
        auto_scale_n_estimators=False, fit_mode=la.fit_mode,
        keep_cache_on_device=la.keep_cache_on_device)
    clf.fit(glob_ctx[0], glob_ctx[1])
    print("global refitted (same C0 rows/seed as the run)", flush=True)

    def embed(Xr):
        Xu, inv = e22b.uniq_rows(Xr)
        outs = []
        for s0 in range(0, len(Xu), la.embed_chunk):
            e = np.asarray(clf.get_embeddings(Xu[s0:s0 + la.embed_chunk], "test"))
            if e.ndim == 3:
                e = e[0]
            outs.append(e.astype(np.float32))
            print(f"    embed {min(s0 + la.embed_chunk, len(Xu)):,}/{len(Xu):,}",
                  flush=True)
        return np.concatenate(outs)[inv]

    phi = e22b.PhiEmbedPCA(X_phi, la.phi_dim,
                           la.seed + e22b.SEED_BAND_PHI + 50, embed)
    z = phi.transform(X_s)
    p0 = clf.predict_proba(X_s)
    p0f = e22b.full_proba(p0, clf.classes_, n_classes).astype(np.float64)
    pool_counts = np.maximum(np.bincount(y_train, minlength=n_classes), 1)
    w = (len(y_train) / (n_classes * pool_counts)) ** la.residual_gamma
    r = e22b.balanced_ce(p0f, y_s, w.astype(np.float64))
    np.savez_compressed(cache_path, z=z, y_s=y_s, r=r,
                        class_names=np.asarray(class_names),
                        sample_per_class=args.sample_per_class)
    print(f"cached sample data -> {cache_path}")
    render(args, ra, mu, tau, z, y_s, class_names)


if __name__ == "__main__":
    main()

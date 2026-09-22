#!/usr/bin/env python3
"""EXP22 -- RACE-PFN v1: Residual Advantage-guided Context Experts.

First implementation of the final research direction (docs/연구방향.pdf,
2026-08-22), Phases 1-5 of its §26: automatic residual experts + counterfactual
advantage routing + sparse top-1 activation with an advantage verifier.
Replaces every exp19/20/21 heuristic it targets: no manual Flood/BF-Bot/Tail
grouping, no benign-return rule, no energy-argmax selection, no dense
all-expert inference, no ownership routing target.

Pipeline (all steps in ONE run; frozen script per CLAUDE.md)
-----------------------------------------------------------
data map (the train pool = the loader's 60% chronological slice, further
partitioned PER CLASS chronologically; val/test slices untouched):

    D_context (--context-frac, default 0.50 of pool)
        -> global context C0 (budget --global-context-size)
        -> common anchor A (--anchor-per-class rows/class)
        -> phi (RobustScaler + PCA --phi-dim) fit rows
        -> affinity reference rows (a0)
    D_mine    (--mine-frac, default 0.25)
        -> class-balanced global residual  r_i = w_y * CE(p0, y),
           w_c = (N/(C*n_c))**--residual-gamma  (counts from the FULL pool)
        -> r-weighted KMeans on z=phi(x) -> K residual regimes (mu_k, tau)
        -> expert residual blocks S_k = TopB( r_i * exp(-d2_ik / tau) )
        -> expert context C_k = A ∪ S_k  (full 7-class posterior; the anchor
           replaces exp19/21's benign-return rule)
    D_router  (remainder, capped --router-cap-per-class; chrono 80/20
               fit/holdout via --router-holdout-frac)
        -> offline advantage  Delta_ik = Lbal_i0 - Lbal_ik - lambda_cost*c_k
        -> proposal router g(h_pre) -> {NO EXPERT, 1..K}
           target k* = argmax_k Delta if max Delta > 0 else NO EXPERT
        -> shared advantage verifier v(h_post) ~ Delta (XGB pseudo-Huber)
    D_val     (loader 20% slice) reserved for K/budget/lambda selection --
              logged, not consumed in v1
    D_test    (loader 20% slice) final eval, single pass:
              global -> router proposal -> top-1 expert only on proposed rows
              -> verifier -> override iff Delta_hat > 0 else global.

Leakage guard: contexts never contain D_router/D_val/D_test rows; router and
verifier are trained only on D_router; phi/centroids/affinity refs are frozen
before any router/eval row is touched.

v1 simplifications (documented knobs, not silent choices)
--------------------------------------------------------
* a0/a_k affinity = z-space mean 16-NN distance to --affinity-ref-rows
  reference rows (global ctx / expert block), negated. The PDF's TabPFN
  attention-affinity statistic is a later knob (exp21 featurize machinery).
* --lambda-cost defaults 0.0 (pure advantage). 3a artifact logs the Delta
  distribution so a later run can set the cost penalty informedly.
* Verifier loss = pseudo-Huber regression on Delta (the PDF's first option);
  pairwise ranking is a later ablation.
* --context-selection defaults 'random' (seeded); 'medoid' = per-class
  MiniBatchKMeans medoids where the per-class target <= --medoid-max.
* h_post additionally carries a K-dim expert-id one-hot (deliberate: the
  verifier stays SHARED -- one score scale -- while calibrating per-expert
  offsets; the PDF's h_post list has no id feature).
* --dense-eval additionally runs EVERY expert on ALL eval rows: oracle
  best-candidate, positive-advantage proposal recall, oracle gap recovery
  (PDF §22 routing metrics). Off by default (K extra full-eval passes).

Judged against (same run, same split): global TabPFN reference and the
full-pool XGBoost baseline. Cross-run reference: exp21 champion macro 0.7411
(different split usage -- exp21's global context drew from the whole pool,
RACE-PFN's from D_context only -- so compare direction, not decimals).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python tabpfn/nfv3_v3_exp22_racepfn.py --target-dataset cic2018 \
        --fit-mode fit_with_cache --test-batch-size 500000

smoke (capped suite, ~tens of minutes):
    ... --target-dataset cic2018_capped --global-context-size 20000 \
        --anchor-per-class 300 --expert-block-rows 4000 --mine-max-rows 60000 \
        --router-cap-per-class 3000 --phi-fit-rows 60000 --kmeans-max-rows 60000 \
        --test-cap-per-class 2000 --dense-eval
"""

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfv3_v3_common as core  # noqa: E402

from tabpfn import TabPFNClassifier  # noqa: E402

# Bands spaced by >=100: per-class offsets (+cid, up to 15 on cic2017_full) and
# cap_per_class's internal +500+i must never land in another band.
SEED_BAND_GLOBAL = 990       # same band as exp21's global context draw
SEED_BAND_ANCHOR = 1100
SEED_BAND_PHI = 1200
SEED_BAND_KMEANS = 1300
SEED_BAND_KMEANS_SUB = 1450  # 1400+i is taken: eval cap = seed+900, +500+i inside
SEED_BAND_ROUTER_CAP = 1500  # +500+i inside cap_per_class -> 2000+i
SEED_BAND_AFFINITY = 1600
SEED_BAND_ROUTER = 1700
SEED_BAND_VERIFIER = 1800
SEED_BAND_MINE = 1900

NO_EXPERT = 0  # router label 0; experts are 1..K


# ---------------- helpers (exp21 lineage; copied, not imported) -------------

def uniq_rows(Xr):
    """(X_unique, inverse) -- distinct-vector dedup for compute only.
    Deterministic model => predictions broadcast back are identical."""
    h = pd.util.hash_pandas_object(pd.DataFrame(Xr), index=False).values
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    return Xr[first], inv


def full_proba(proba, model_classes, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=np.int64)] = proba
    return out


# ---------------- RACE-PFN building blocks ----------------------------------

def entropy_of(p):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1).astype(np.float32)


def margin_of(p):
    if p.shape[1] < 2:
        return np.ones(len(p), dtype=np.float32)
    part = np.partition(p, -2, axis=1)
    return (part[:, -1] - part[:, -2]).astype(np.float32)


def balanced_ce(probs, y, w):
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return (w[y] * (-np.log(p_true))).astype(np.float64)


class PhiMap:
    """z = PCA(RobustScaler(x)) -- the residual representation phi."""

    def __init__(self, feats_fit, dim, seed):
        self.scaler = RobustScaler().fit(feats_fit)
        self.pca = PCA(n_components=dim, random_state=seed).fit(
            self.scaler.transform(feats_fit))

    def transform(self, feats, chunk=500_000):
        outs = []
        for s0 in range(0, len(feats), chunk):
            outs.append(self.pca.transform(
                self.scaler.transform(feats[s0:s0 + chunk])).astype(np.float32))
        return np.concatenate(outs) if outs else \
            np.zeros((0, self.pca.n_components_), dtype=np.float32)


def class_chrono_partition(train_idx, y_train, ts_train, ctx_frac, mine_frac,
                           class_names):
    """Per-class chronological D_context / D_mine / D_router partition of the
    train pool. Mirrors scenario_chronological_split's per-group ordering so
    every class is present in all three pools and time order is preserved."""
    parts = {"context": [], "mine": [], "router": []}
    audit = []
    for cid, cname in enumerate(class_names):
        mask = y_train == cid
        rows = train_idx[mask]
        if len(rows) == 0:
            continue
        order = rows[np.argsort(ts_train[mask], kind="stable")]
        n = len(order)
        n_ctx = int(n * ctx_frac)
        n_mine = int(n * mine_frac)
        n_rt = n - n_ctx - n_mine
        if min(n_ctx, n_mine, n_rt) <= 0:
            raise SystemExit(
                f"class {cname}: {n} train rows cannot fill "
                f"context/mine/router ({n_ctx}/{n_mine}/{n_rt})")
        parts["context"].extend(order[:n_ctx])
        parts["mine"].extend(order[n_ctx:n_ctx + n_mine])
        parts["router"].extend(order[n_ctx + n_mine:])
        audit.append({"class": cname, "train_total": n, "context": n_ctx,
                      "mine": n_mine, "router": n_rt})
    return ({k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in parts.items()},
            pd.DataFrame(audit))


def pick_rows(rows, k, mode, seed, medoid_max, z_of_idx, n_init, batch_size):
    """Pick k representative rows from one class's pool. `seed` must arrive
    ALREADY banded+class-offset by the caller (no second band added here).
    mode='medoid' uses MiniBatchKMeans medoids in z-space when k <= medoid_max
    (else falls back to the seeded draw)."""
    if k >= len(rows):
        return rows, "all"
    if mode == "medoid" and k <= medoid_max:
        z = z_of_idx(rows)
        km = MiniBatchKMeans(n_clusters=k, random_state=seed,
                             n_init=n_init, batch_size=batch_size).fit(z)
        med = np.unique(pairwise_distances_argmin(km.cluster_centers_, z))
        picked = rows[med]
        if len(picked) < k:  # duplicate medoids -> top up with a seeded draw
            spare = np.setdiff1d(rows, picked, assume_unique=False)
            extra = np.random.default_rng(seed + 50) \
                .permutation(spare)[: k - len(picked)]
            picked = np.concatenate([picked, extra])
        return picked, "medoid"
    note = "random" if mode == "random" else f"medoid(fallback>{medoid_max})"
    return np.random.default_rng(seed).permutation(rows)[:k], note


def representative_subset(pool_idx, pool_y, n_classes, budget, mode, seed,
                          medoid_max, z_of_idx, n_init, batch_size):
    """Class-aware representative selection with a POOLED budget: per-class
    targets follow the pool's natural ratios (largest remainder, >=1 per
    present class) -- exp21's global-context recipe, plus the medoid option."""
    counts = np.bincount(pool_y, minlength=n_classes)
    total = int(counts.sum())
    if budget <= 0 or budget >= total:
        return np.sort(pool_idx), "all"
    raw = counts * (budget / total)
    tgt = np.minimum(np.floor(raw).astype(np.int64), counts)
    present = counts > 0
    tgt[present & (tgt < 1)] = 1
    rem = budget - int(tgt.sum())
    for cid in np.argsort(-(raw - np.floor(raw))):
        if rem <= 0:
            break
        if tgt[cid] < counts[cid]:
            tgt[cid] += 1
            rem -= 1
    chosen, notes = [], set()
    for cid in range(n_classes):
        if tgt[cid] <= 0:
            continue
        picked, note = pick_rows(pool_idx[pool_y == cid], int(tgt[cid]), mode,
                                 seed + cid, medoid_max, z_of_idx,
                                 n_init, batch_size)
        chosen.append(picked)
        notes.add(note)
    return np.sort(np.concatenate(chosen)), "+".join(sorted(notes))


def anchor_subset(pool_idx, pool_y, n_classes, per_class, mode, seed,
                  medoid_max, z_of_idx, n_init, batch_size):
    """Anchor A: an EQUAL per-class draw -- min(per_class, n_c) rows for every
    present class (NOT pool-proportional; under IR>1000 a proportional 14k
    anchor would be ~87% benign with ~2 tail rows, defeating A's purpose of
    supporting the full C-class posterior in every expert context)."""
    chosen, notes = [], set()
    for cid in range(n_classes):
        rows = pool_idx[pool_y == cid]
        if len(rows) == 0:
            continue
        picked, note = pick_rows(rows, min(per_class, len(rows)), mode,
                                 seed + cid, medoid_max, z_of_idx,
                                 n_init, batch_size)
        chosen.append(picked)
        notes.add(note)
    return np.sort(np.concatenate(chosen)), "+".join(sorted(notes))


def sq_dist_to_centroids(z, mu, chunk=500_000):
    """(n, K) squared euclidean distances, chunked."""
    outs = []
    for s0 in range(0, len(z), chunk):
        zb = z[s0:s0 + chunk]
        outs.append(((zb[:, None, :] - mu[None, :, :]) ** 2).sum(-1))
    return np.concatenate(outs).astype(np.float32)


class AffinityRef:
    """a(x) = -mean distance to the nn nearest of `ref` rows in z-space.
    Cheap stand-in for TabPFN context-query attention affinity (v1)."""

    def __init__(self, z_ref, nn):
        self.nn = min(nn, len(z_ref))
        self.knn = NearestNeighbors(n_neighbors=self.nn).fit(z_ref)

    def score(self, z, chunk=500_000):
        outs = []
        for s0 in range(0, len(z), chunk):
            d, _ = self.knn.kneighbors(z[s0:s0 + chunk])
            outs.append(-d.mean(axis=1).astype(np.float32))
        return np.concatenate(outs)


def build_hpre(p0, z, dists, a0):
    return np.concatenate(
        [p0.astype(np.float32), entropy_of(p0)[:, None], margin_of(p0)[:, None],
         z, np.sqrt(dists), a0[:, None]], axis=1)


def build_hpost(p0, pk, a0, ak, dk, k_id, K):
    onehot = np.zeros((len(p0), K), dtype=np.float32)
    onehot[:, k_id] = 1.0
    return np.concatenate(
        [p0.astype(np.float32), pk.astype(np.float32),
         (pk - p0).astype(np.float32),
         entropy_of(p0)[:, None], entropy_of(pk)[:, None],
         margin_of(p0)[:, None], margin_of(pk)[:, None],
         a0[:, None], ak[:, None], dk[:, None], onehot], axis=1)


# ---------------------------------------------------------------------------

def run_exp22(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp22_racepfn"
    print(f"Args: {vars(args)}", flush=True)
    if args.subsample_samples:
        raise SystemExit("--subsample-samples must stay 0.")
    if args.context_frac + args.mine_frac >= 1.0:
        raise SystemExit("--context-frac + --mine-frac must be < 1 "
                         "(the remainder is D_router).")
    # base_parser flags that exp22 does NOT implement must not enter the Args
    # record looking active (the Args line IS the config record).
    if args.train_split != "train":
        raise SystemExit("--train-split is not implemented in exp22: the "
                         "D_context/D_mine/D_router partition is carved from "
                         "the 60% slice only; the val slice stays reserved.")
    if args.skip_tabpfn:
        raise SystemExit("--skip-tabpfn is meaningless here -- the pipeline "
                         "IS TabPFN.")

    timings = {}
    t_all = time.time()
    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, _, _, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    K = args.n_clusters
    y_train = label_fn(train_idx)

    # timestamps for the per-class chronological sub-partition
    d = core.load_pickle(args.data)
    ts_all = np.asarray(d["timestamps" if "timestamps" in d else "time_proxy"],
                        dtype=np.int64)
    del d
    ts_train = ts_all[train_idx]
    del ts_all

    pools, pool_audit = class_chrono_partition(
        train_idx, y_train, ts_train, args.context_frac, args.mine_frac,
        class_names)
    ctx_pool, mine_pool, router_pool = \
        pools["context"], pools["mine"], pools["router"]
    print(f"pools: context={len(ctx_pool):,} mine={len(mine_pool):,} "
          f"router={len(router_pool):,} | val(reserved)={len(val_idx):,} "
          f"test={len(test_idx):,}")
    print(pool_audit.to_string(index=False), flush=True)

    # class-balanced weights from the FULL train pool distribution
    pool_counts = np.maximum(np.bincount(y_train, minlength=n_classes), 1)
    w_bal = (len(y_train) / (n_classes * pool_counts)) ** args.residual_gamma
    w_bal = w_bal.astype(np.float64)
    print("balanced weights:", {n: round(float(w_bal[i]), 4)
                                for i, n in enumerate(class_names)})

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    # ---- Step 0: phi (fit on D_context only; frozen for everything after) --
    t0 = time.time()
    y_ctx_pool = label_fn(ctx_pool)
    phi_fit_idx = core.stratified_subset(ctx_pool, y_ctx_pool, n_classes,
                                         args.phi_fit_rows,
                                         args.seed + SEED_BAND_PHI)
    phi = PhiMap(feats_of(phi_fit_idx), args.phi_dim,
                 args.seed + SEED_BAND_PHI + 50)
    timings["phi_fit_s"] = round(time.time() - t0, 1)
    print(f"phi ready (fit rows {len(phi_fit_idx):,}, dim {args.phi_dim}, "
          f"{timings['phi_fit_s']}s)", flush=True)

    def z_of_idx(idx, chunk=500_000):
        outs = []
        for s0 in range(0, len(idx), chunk):
            outs.append(phi.transform(feats_of(idx[s0:s0 + chunk])))
        return np.concatenate(outs)

    # ---- Step 1: global context C0 + common anchor A (from D_context) ------
    g_idx, g_note = representative_subset(
        ctx_pool, y_ctx_pool, n_classes, args.global_context_size,
        args.context_selection, args.seed + SEED_BAND_GLOBAL,
        args.medoid_max, z_of_idx, args.medoid_n_init, args.medoid_batch_size)
    anchor_idx, a_note = anchor_subset(
        ctx_pool, y_ctx_pool, n_classes, args.anchor_per_class,
        args.context_selection, args.seed + SEED_BAND_ANCHOR,
        args.medoid_max, z_of_idx, args.medoid_n_init, args.medoid_batch_size)
    print(f"C0: {len(g_idx):,} rows ({g_note}) | anchor A: {len(anchor_idx):,} "
          f"rows ({a_note})")

    # ---- materialize every row this run touches, then drop the pickle ------
    mine_idx = core.stratified_subset(mine_pool, label_fn(mine_pool), n_classes,
                                      args.mine_max_rows,
                                      args.seed + SEED_BAND_MINE)
    router_idx = core.cap_per_class(router_pool, label_fn(router_pool),
                                    n_classes, args.router_cap_per_class,
                                    args.seed + SEED_BAND_ROUTER_CAP)
    eval_idx = core.cap_per_class(test_idx, label_fn(test_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)

    glob_ctx = (feats_of(g_idx), label_fn(g_idx))
    anchor = (feats_of(anchor_idx), label_fn(anchor_idx))
    X_mine, y_mine = feats_of(mine_idx), label_fn(mine_idx)
    X_router, y_router = feats_of(router_idx), label_fn(router_idx)
    router_ts = ts_train[np.searchsorted(train_idx, router_idx)]
    X_eval, y_eval = feats_of(eval_idx), label_fn(eval_idx)
    xgb_train_idx = train_idx
    if args.max_train_samples > 0 and args.max_train_samples < len(train_idx):
        xgb_train_idx = core.stratified_subset(
            train_idx, y_train, n_classes, args.max_train_samples,
            args.seed + 850)
    X_xgb, y_xgb = feats_of(xgb_train_idx), label_fn(xgb_train_idx)
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()
    print(f"mine rows used: {len(mine_idx):,} | router rows: {len(router_idx):,} "
          f"| eval rows: {len(eval_idx):,} | xgb train: {len(xgb_train_idx):,}",
          flush=True)
    # predicted GPU peak for the binding stage (1M-ctx cache build / predict)
    # + the expandable_segments warning (exp21's global featurize OOM'd without it)
    core.report_memory_plan(len(g_idx),
                            max(len(X_eval), len(X_mine), len(X_router)), args)

    import torch

    def make_clf(Xc, yc):
        clf = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        clf.fit(Xc, yc)
        return clf

    def free_gpu():
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def batched_proba(clf, Xr, tag=""):
        """Dedup + batched predict_proba mapped into the full class space."""
        Xu, inv = uniq_rows(Xr)
        bs = args.test_batch_size or len(Xu)
        outs = []
        for bn, s0 in enumerate(range(0, len(Xu), bs), 1):
            outs.append(clf.predict_proba(Xu[s0:s0 + bs]))
            if bn == 1 or s0 + bs >= len(Xu):
                print(f"    [{tag}] proba rows {min(s0 + bs, len(Xu)):,}"
                      f"/{len(Xu):,} (distinct of {len(Xr):,})", flush=True)
        pr = full_proba(np.concatenate(outs), clf.classes_, n_classes)
        free_gpu()
        return pr[inv].astype(np.float64)

    # ---- XGBoost baseline (same pool, same split) --------------------------
    t0 = time.time()
    booster = xgb.XGBClassifier(
        n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        min_child_weight=args.xgb_min_child_weight,
        reg_lambda=args.xgb_reg_lambda, objective="multi:softprob",
        num_class=n_classes, eval_metric="mlogloss", n_jobs=-1,
        random_state=args.seed)
    y_pred_xgb = None
    if not args.skip_xgboost:
        print(f"XGBoost baseline fitting on {len(X_xgb):,} rows ...", flush=True)
        booster.fit(X_xgb, y_xgb)
        y_pred_xgb = booster.predict(X_eval)
        timings["xgb_baseline_s"] = round(time.time() - t0, 1)
        print(f"XGBoost done ({timings['xgb_baseline_s']}s)", flush=True)
    del X_xgb, y_xgb
    gc.collect()

    # ---- Step 1b: global TabPFN ------------------------------------------
    t0 = time.time()
    glob = make_clf(glob_ctx[0], glob_ctx[1])
    timings["global_fit_s"] = round(time.time() - t0, 1)
    print(f"global fitted: {len(glob_ctx[0]):,} ctx rows "
          f"({timings['global_fit_s']}s)", flush=True)

    # ---- Step 2: residual mining on D_mine --------------------------------
    t0 = time.time()
    p0_mine = batched_proba(glob, X_mine, "global/mine")
    r_mine = balanced_ce(p0_mine, y_mine, w_bal)
    z_mine = phi.transform(X_mine)
    km_rows = np.arange(len(z_mine))
    if args.kmeans_max_rows and len(km_rows) > args.kmeans_max_rows:
        km_rows = np.random.default_rng(args.seed + SEED_BAND_KMEANS_SUB) \
            .permutation(len(z_mine))[: args.kmeans_max_rows]
    km = KMeans(n_clusters=K, random_state=args.seed + SEED_BAND_KMEANS,
                n_init=args.kmeans_n_init).fit(z_mine[km_rows],
                                               sample_weight=r_mine[km_rows])
    mu = km.cluster_centers_.astype(np.float32)
    d2_mine = sq_dist_to_centroids(z_mine, mu)
    assign_mine = d2_mine.argmin(axis=1)
    tau = args.tau_c if args.tau_c > 0 else float(
        np.median(d2_mine[np.arange(len(z_mine)), assign_mine]) + 1e-12)
    timings["mining_s"] = round(time.time() - t0, 1)
    print(f"residual clustering: K={K} tau={tau:.4f} "
          f"({timings['mining_s']}s)", flush=True)

    residual_stats = pd.DataFrame([{
        "class": class_names[c],
        "rows": int((y_mine == c).sum()),
        "weight": float(w_bal[c]),
        "mean_ce": float((r_mine[y_mine == c] / w_bal[c]).mean())
        if (y_mine == c).any() else np.nan,
        "mean_residual": float(r_mine[y_mine == c].mean())
        if (y_mine == c).any() else np.nan,
    } for c in range(n_classes)])
    cluster_rows = []
    for k in range(K):
        m = assign_mine == k
        row = {"cluster": k, "rows": int(m.sum()),
               "weighted_mass": float(r_mine[m].sum() / max(r_mine.sum(), 1e-12)),
               "mean_residual": float(r_mine[m].mean()) if m.any() else np.nan}
        for c in range(n_classes):
            row[class_names[c]] = int((m & (y_mine == c)).sum())
        cluster_rows.append(row)
    cluster_df = pd.DataFrame(cluster_rows)
    print("\n=== residual cluster composition (rows per class) ===")
    print(cluster_df.to_string(index=False), flush=True)

    # ---- Step 3: residual context experts C_k = A ∪ S_k -------------------
    s_all = r_mine[:, None] * np.exp(-d2_mine / tau)          # (n_mine, K)
    experts, expert_ctx_rows = [], []
    anchor_comp = {"expert": "anchor(shared)", "anchor_rows": len(anchor[0]),
                   "block_rows": 0, "fit_s": 0.0}
    for c in range(n_classes):
        anchor_comp[class_names[c]] = int((anchor[1] == c).sum())
    expert_ctx_rows.append(anchor_comp)
    for k in range(K):
        top = np.argsort(-s_all[:, k])[: args.expert_block_rows]
        Xk = np.concatenate([anchor[0], X_mine[top]])
        yk = np.concatenate([anchor[1], y_mine[top]])
        t0 = time.time()
        experts.append({"clf": make_clf(Xk, yk), "block_rows": top})
        comp = {"expert": k + 1, "anchor_rows": len(anchor[0]),
                "block_rows": len(top),
                "fit_s": round(time.time() - t0, 1)}
        for c in range(n_classes):
            comp[class_names[c]] = int((y_mine[top] == c).sum())
        expert_ctx_rows.append(comp)
        print(f"expert {k + 1}: ctx {len(Xk):,} (anchor {len(anchor[0]):,} + "
              f"block {len(top):,}) fit {comp['fit_s']}s", flush=True)
    expert_ctx_df = pd.DataFrame(expert_ctx_rows)

    # ---- affinity references (frozen before router/eval rows) -------------
    rng_a = np.random.default_rng(args.seed + SEED_BAND_AFFINITY)
    z_g = phi.transform(glob_ctx[0])
    aff0 = AffinityRef(z_g[rng_a.permutation(len(z_g))[: args.affinity_ref_rows]],
                       args.affinity_nn)
    aff_k = []
    for k in range(K):
        zb = z_mine[experts[k]["block_rows"]]
        aff_k.append(AffinityRef(
            zb[rng_a.permutation(len(zb))[: args.affinity_ref_rows]],
            args.affinity_nn))
    del z_g
    gc.collect()

    # ---- Step 4: offline advantage on D_router ----------------------------
    t0 = time.time()
    p0_rt = batched_proba(glob, X_router, "global/router")
    L0_rt = balanced_ce(p0_rt, y_router, w_bal)
    pk_rt, delta_rt = [], np.zeros((len(y_router), K), dtype=np.float64)
    for k in range(K):
        pk = batched_proba(experts[k]["clf"], X_router, f"expert{k + 1}/router")
        pk_rt.append(pk.astype(np.float32))
        delta_rt[:, k] = (L0_rt - balanced_ce(pk, y_router, w_bal)
                          - args.lambda_cost * args.expert_cost)
    timings["router_advantage_s"] = round(time.time() - t0, 1)
    best_delta = delta_rt.max(axis=1)
    kstar = np.where(best_delta > 0, delta_rt.argmax(axis=1) + 1, NO_EXPERT)
    print(f"router set: positive-advantage rows "
          f"{(best_delta > 0).sum():,}/{len(kstar):,} "
          f"({timings['router_advantage_s']}s)", flush=True)

    adv_rows = []
    for k in range(K):
        for c in range(n_classes):
            m = y_router == c
            if not m.any():
                continue
            adv_rows.append({
                "expert": k + 1, "class": class_names[c], "rows": int(m.sum()),
                "mean_delta": float(delta_rt[m, k].mean()),
                "frac_positive": float((delta_rt[m, k] > 0).mean())})
    adv_df = pd.DataFrame(adv_rows)
    target_df = pd.DataFrame({
        "target": ["NO_EXPERT"] + [f"expert{k + 1}" for k in range(K)],
        "rows": [int((kstar == 0).sum())] + [int((kstar == k + 1).sum())
                                             for k in range(K)]})
    print(target_df.to_string(index=False), flush=True)

    # ---- Step 5: proposal router on h_pre (chrono 80/20 fit/holdout) ------
    t0 = time.time()
    z_rt = phi.transform(X_router)
    d2_rt = sq_dist_to_centroids(z_rt, mu)
    a0_rt = aff0.score(z_rt)
    hpre_rt = build_hpre(p0_rt, z_rt, d2_rt, a0_rt)
    order = np.argsort(router_ts, kind="stable")
    n_fit = int(len(order) * (1.0 - args.router_holdout_frac))
    fit_rows, hold_rows = order[:n_fit], order[n_fit:]

    router_clf, router_note = None, "trained"
    # xgboost requires y == [0..n-1]: densely encode the present targets
    # (an expert that never wins on fit rows would otherwise crash fit(),
    # reproduced on xgboost 3.2.0) and decode predictions via `present`.
    present = np.unique(kstar[fit_rows])
    if len(present) < 2:
        router_note = f"degenerate targets {present.tolist()} -> all NO_EXPERT"
        print(f"router: {router_note}")
    else:
        cls_counts = np.bincount(kstar[fit_rows], minlength=K + 1)
        sw = np.zeros(len(fit_rows))
        for c in present:
            sw[kstar[fit_rows] == c] = len(fit_rows) / (len(present)
                                                        * cls_counts[c])
        router_clf = xgb.XGBClassifier(
            n_estimators=args.router_n_estimators,
            max_depth=args.router_max_depth,
            learning_rate=args.router_lr, objective="multi:softprob",
            num_class=len(present),  # 2-class softprob needs it explicitly
            eval_metric="mlogloss", n_jobs=-1,
            random_state=args.seed + SEED_BAND_ROUTER)
        router_clf.fit(hpre_rt[fit_rows],
                       np.searchsorted(present, kstar[fit_rows]),
                       sample_weight=sw)
    timings["router_train_s"] = round(time.time() - t0, 1)

    def propose(hpre):
        if router_clf is None:
            return np.zeros(len(hpre), dtype=np.int64)
        # 2-class softprob predict returns float64 {0.,1.} -- cast to index
        return present[router_clf.predict(hpre).astype(np.int64)]

    prop_hold = propose(hpre_rt[hold_rows])
    router_hold_df = pd.crosstab(
        pd.Series([("NO" if v == 0 else f"e{v}") for v in kstar[hold_rows]],
                  name="target"),
        pd.Series([("NO" if v == 0 else f"e{v}") for v in prop_hold],
                  name="proposed")).reset_index()
    pos_hold = best_delta[hold_rows] > 0
    prop_recall_hold = float((prop_hold[pos_hold] != 0).mean()) \
        if pos_hold.any() else np.nan
    print(f"router holdout: proposal-rate "
          f"{float((prop_hold != 0).mean()):.4f}, positive-advantage recall "
          f"{prop_recall_hold:.4f} ({timings['router_train_s']}s)", flush=True)

    # ---- Step 7 (training): shared advantage verifier ---------------------
    t0 = time.time()
    ak_rt = [aff_k[k].score(z_rt) for k in range(K)]
    hpost_parts, tgt_parts = [], []
    for k in range(K):
        hpost_parts.append(build_hpost(
            p0_rt[fit_rows], pk_rt[k][fit_rows], a0_rt[fit_rows],
            ak_rt[k][fit_rows], np.sqrt(d2_rt[fit_rows, k]), k, K))
        tgt_parts.append(delta_rt[fit_rows, k])
    verifier = xgb.XGBRegressor(
        n_estimators=args.verifier_n_estimators,
        max_depth=args.verifier_max_depth, learning_rate=args.verifier_lr,
        objective="reg:pseudohubererror", huber_slope=args.huber_slope,
        n_jobs=-1, random_state=args.seed + SEED_BAND_VERIFIER)
    verifier.fit(np.concatenate(hpost_parts),
                 np.concatenate(tgt_parts).astype(np.float32))
    ver_rows = []
    for k in range(K):
        hp = build_hpost(p0_rt[hold_rows], pk_rt[k][hold_rows],
                         a0_rt[hold_rows], ak_rt[k][hold_rows],
                         np.sqrt(d2_rt[hold_rows, k]), k, K)
        dh = verifier.predict(hp).astype(np.float64)
        dt = delta_rt[hold_rows, k]
        ver_rows.append({
            "expert": k + 1,
            "sign_accuracy": float(((dh > 0) == (dt > 0)).mean()),
            "pearson_r": float(np.corrcoef(dh, dt)[0, 1])
            if dt.std() > 0 else np.nan,
            "mae": float(np.abs(dh - dt).mean())})
    ver_df = pd.DataFrame(ver_rows)
    timings["verifier_train_s"] = round(time.time() - t0, 1)
    print("verifier holdout:\n" + ver_df.to_string(index=False), flush=True)
    del hpost_parts, tgt_parts, z_rt, d2_rt, a0_rt, ak_rt, hpre_rt, pk_rt
    gc.collect()

    # ---- Steps 6+8: sparse inference on D_test ----------------------------
    t0 = time.time()
    p0_eval = batched_proba(glob, X_eval, "global/eval")
    timings["eval_global_s"] = round(time.time() - t0, 1)
    y_glob = p0_eval.argmax(axis=1)

    t0 = time.time()
    z_eval = phi.transform(X_eval)
    d2_eval = sq_dist_to_centroids(z_eval, mu)
    a0_eval = aff0.score(z_eval)
    proposal = propose(build_hpre(p0_eval, z_eval, d2_eval, a0_eval))
    timings["eval_router_s"] = round(time.time() - t0, 1)

    final = y_glob.copy()
    delta_hat = np.zeros(len(y_eval), dtype=np.float32)
    accepted = np.zeros(len(y_eval), dtype=bool)
    decision_rows = []
    t0 = time.time()
    for k in range(K):
        rows = np.flatnonzero(proposal == k + 1)
        if len(rows) == 0:
            decision_rows.append({"expert": k + 1, "proposed": 0,
                                  "accepted": 0, "accept_rate": np.nan})
            continue
        pk = batched_proba(experts[k]["clf"], X_eval[rows], f"expert{k + 1}/eval")
        hp = build_hpost(p0_eval[rows], pk, a0_eval[rows],
                         aff_k[k].score(z_eval[rows]),
                         np.sqrt(d2_eval[rows, k]), k, K)
        dh = verifier.predict(hp).astype(np.float32)
        delta_hat[rows] = dh
        acc = dh > 0
        accepted[rows] = acc
        final[rows[acc]] = pk[acc].argmax(axis=1)
        decision_rows.append({"expert": k + 1, "proposed": len(rows),
                              "accepted": int(acc.sum()),
                              "accept_rate": round(float(acc.mean()), 4)})
        print(f"expert {k + 1}: proposed {len(rows):,} -> accepted "
              f"{int(acc.sum()):,}", flush=True)
    timings["eval_experts_s"] = round(time.time() - t0, 1)
    rho = float((proposal != 0).mean())
    accept_rho = float(accepted.mean())
    decision_df = pd.DataFrame(decision_rows)
    decision_df.loc[len(decision_df)] = {
        "expert": "total", "proposed": int((proposal != 0).sum()),
        "accepted": int(accepted.sum()), "accept_rate": accept_rho}
    print(f"\nactivation rate rho={rho:.4f} (accepted {accept_rho:.4f}) "
          f"=> mean TFM calls/sample = {1 + rho:.3f}", flush=True)

    # ---- routing quality (PDF §22) ----------------------------------------
    glob_ok = y_glob == y_eval
    final_ok = final == y_eval
    n_helpful = int((~glob_ok & final_ok).sum())
    n_harmful = int((glob_ok & ~final_ok).sum())
    quality = {
        "helpful_override": n_helpful, "harmful_override": n_harmful,
        "net_correction": n_helpful - n_harmful,
        "override_precision": round(n_helpful / max(n_helpful + n_harmful, 1), 4),
        "activation_rate": round(rho, 4),
        "accepted_rate": round(accept_rho, 4),
        "router_holdout_pos_adv_recall": round(prop_recall_hold, 4)
        if not np.isnan(prop_recall_hold) else np.nan,
        "router_note": router_note,
    }

    all_rows = list(core.per_class_table("racepfn_system", y_eval, final,
                                         class_names, tail_classes))
    all_rows.extend(core.per_class_table("global_tabpfn", y_eval, y_glob,
                                         class_names, tail_classes))
    if y_pred_xgb is not None:
        all_rows.extend(core.per_class_table("xgboost", y_eval, y_pred_xgb,
                                             class_names, tail_classes))

    # ---- optional dense pass: oracle / recall / gap (PDF §22) -------------
    dense_note = "skipped (--dense-eval off)"
    if args.dense_eval:
        t0 = time.time()
        L_eval = np.zeros((len(y_eval), K + 1), dtype=np.float64)
        L_eval[:, 0] = balanced_ce(p0_eval, y_eval, w_bal)
        dense_preds = [y_glob]
        for k in range(K):
            pk = batched_proba(experts[k]["clf"], X_eval, f"expert{k + 1}/dense")
            L_eval[:, k + 1] = (balanced_ce(pk, y_eval, w_bal)
                                + args.lambda_cost * args.expert_cost)
            dense_preds.append(pk.argmax(axis=1))
        oracle_pick = L_eval.argmin(axis=1)
        y_oracle = np.take_along_axis(
            np.stack(dense_preds, axis=1), oracle_pick[:, None], axis=1)[:, 0]
        all_rows.extend(core.per_class_table("dense_oracle", y_eval, y_oracle,
                                             class_names, tail_classes))
        pos_adv_eval = (L_eval[:, 0:1] - L_eval[:, 1:]).max(axis=1) > 0
        quality["eval_pos_adv_rows"] = int(pos_adv_eval.sum())
        quality["eval_pos_adv_proposal_recall"] = round(
            float((proposal[pos_adv_eval] != 0).mean()), 4) \
            if pos_adv_eval.any() else np.nan
        timings["dense_eval_s"] = round(time.time() - t0, 1)
        dense_note = "done"

    table = pd.DataFrame(all_rows)
    piv = table.pivot(index="class", columns="method", values="f1").round(4)
    print("\n=== per-class F1 ===")
    print(piv.to_string())
    f1_of = lambda m: float(piv.loc["macro_avg", m]) if m in piv.columns else np.nan
    if args.dense_eval and "dense_oracle" in piv.columns:
        gap = f1_of("dense_oracle") - f1_of("global_tabpfn")
        quality["oracle_gap_recovery"] = round(
            (f1_of("racepfn_system") - f1_of("global_tabpfn")) / gap, 4) \
            if abs(gap) > 1e-9 else np.nan
    quality_df = pd.DataFrame([quality])
    print("\n=== routing quality ===")
    print(quality_df.to_string(index=False))
    timings["total_s"] = round(time.time() - t_all, 1)
    timings["dense_eval"] = dense_note
    timings["rho"] = round(rho, 4)

    # ---- artifacts --------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root, f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp22_racepfn")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pool_audit.to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    residual_stats.to_csv(os.path.join(out_dir, "1a_residual_stats.csv"),
                          index=False)
    cluster_df.to_csv(os.path.join(out_dir, "1b_cluster_composition.csv"),
                      index=False)
    expert_ctx_df.to_csv(os.path.join(out_dir, "2a_expert_contexts.csv"),
                         index=False)
    adv_df.to_csv(os.path.join(out_dir, "3a_advantage_stats.csv"), index=False)
    target_df.to_csv(os.path.join(out_dir, "3b_router_targets.csv"), index=False)
    router_hold_df.to_csv(os.path.join(out_dir, "4a_router_holdout_confusion.csv"),
                          index=False)
    ver_df.to_csv(os.path.join(out_dir, "4b_verifier_holdout.csv"), index=False)
    decision_df.to_csv(os.path.join(out_dir, "5a_decision_stats.csv"), index=False)
    quality_df.to_csv(os.path.join(out_dir, "6a_routing_quality.csv"), index=False)
    np.savez_compressed(
        os.path.join(out_dir, "system_dump.npz"),
        proposal=proposal.astype(np.int8), accepted=accepted,
        delta_hat=delta_hat, final=final.astype(np.int64),
        y_glob=y_glob.astype(np.int64), y_true=y_eval.astype(np.int64),
        mu=mu, tau=np.float32(tau), class_names=np.asarray(class_names))
    np.savez_compressed(os.path.join(out_dir, "probs_tabpfn_global.npz"),
                        probs=p0_eval.astype(np.float32),
                        y_true=y_eval.astype(np.int64))
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([timings], f, indent=2)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-frac", type=float, default=0.50,
                   help="fraction of the train pool (per class, chronological "
                        "prefix) forming D_context")
    p.add_argument("--mine-frac", type=float, default=0.25,
                   help="next chronological fraction forming D_mine; the "
                        "remainder is D_router")
    p.add_argument("--global-context-size", type=int, default=1_000_000)
    p.add_argument("--context-selection", default="random",
                   choices=["random", "medoid"],
                   help="C0/anchor row selection. medoid = per-class "
                        "MiniBatchKMeans medoids in z where target <= "
                        "--medoid-max (else seeded random fallback).")
    p.add_argument("--medoid-max", type=int, default=20_000)
    p.add_argument("--medoid-n-init", type=int, default=3)
    p.add_argument("--medoid-batch-size", type=int, default=4_096)
    p.add_argument("--anchor-per-class", type=int, default=2_000,
                   help="EQUAL per-class anchor rows: min(this, n_c) for every "
                        "present class (not pool-proportional)")
    p.add_argument("--expert-block-rows", type=int, default=186_000,
                   help="TopB residual-block rows per expert; context = anchor "
                        "+ block (default ~200k like exp21 experts)")
    p.add_argument("--n-clusters", type=int, default=4,
                   help="K residual regimes / experts")
    p.add_argument("--phi-dim", type=int, default=16)
    p.add_argument("--phi-fit-rows", type=int, default=500_000)
    p.add_argument("--kmeans-max-rows", type=int, default=1_000_000)
    p.add_argument("--kmeans-n-init", type=int, default=5)
    p.add_argument("--tau-c", type=float, default=0.0,
                   help="block-selection temperature; 0 = median squared "
                        "distance to the assigned centroid")
    p.add_argument("--residual-gamma", type=float, default=1.0,
                   help="w_c = (N/(C*n_c))**gamma for residual + balanced loss")
    p.add_argument("--mine-max-rows", type=int, default=2_000_000,
                   help="stratified ratio-preserving cap on D_mine rows mined")
    p.add_argument("--router-cap-per-class", type=int, default=100_000)
    p.add_argument("--router-holdout-frac", type=float, default=0.2)
    p.add_argument("--lambda-cost", type=float, default=0.0,
                   help="advantage cost penalty lambda_cost (v1 default 0; "
                        "NOTE its scale interacts with --residual-gamma)")
    p.add_argument("--expert-cost", type=float, default=1.0)
    p.add_argument("--affinity-ref-rows", type=int, default=4_096)
    p.add_argument("--affinity-nn", type=int, default=16)
    p.add_argument("--router-n-estimators", type=int, default=300)
    p.add_argument("--router-max-depth", type=int, default=6)
    p.add_argument("--router-lr", type=float, default=0.1)
    p.add_argument("--verifier-n-estimators", type=int, default=400)
    p.add_argument("--verifier-max-depth", type=int, default=6)
    p.add_argument("--verifier-lr", type=float, default=0.05)
    p.add_argument("--huber-slope", type=float, default=1.0)
    p.add_argument("--dense-eval", action="store_true",
                   help="also run every expert on ALL eval rows: dense oracle, "
                        "positive-advantage recall, oracle gap recovery")
    p.set_defaults(
        target_dataset="cic2018",
        max_train_samples=-1,
        n_estimators=1,
        test_cap_per_class=0,
        fit_mode="fit_with_cache",
        test_batch_size=500_000,
        subsample_samples=0,
    )
    args = p.parse_args()
    run_exp22(args)


if __name__ == "__main__":
    main()

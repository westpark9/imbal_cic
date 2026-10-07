#!/usr/bin/env python3
"""EXP22C -- RACE-PFN, ONE KNOB vs exp22b: the specialist-context construction
layer (guide Phase-1 package, declared multi-part).

exp22b (run 20260825_174233, macro 0.7084 vs global 0.6700 -- first win) left
a new degeneracy: TopB s_ik = r_i * exp(-d_ik^2/tau) let r (6 decades, benign
0.005 vs web 1527) crush the proximity term (1.7 decades), so all four expert
blocks converged on the same global high-r rows (brute 57,189 in every block;
only the ddos expert differentiated) -> hard-attack clones, router target 94%
on e2, web F1 0.1649->0.0634 via 7,640 wrong->wrong web FP overrides. Lab
log: manuscript/report/0825.md §6. RACE-PFN_RESEARCH_IMPLEMENTATION_GUIDE.md
(§11.2, §18, §19 Phase 1, §26) turns exactly this observation into the next
design: strengthen expert complementarity BEFORE touching the router.

THE KNOB -- one declared package (guide Phase 1) replacing the S_k layer:
  1. residual clip           r_bar = min(r, quantile(r, --residual-clip-q))
  2. full residual signature, per-block standardized then alpha/sqrt(dim)
     scaled so alpha=1 gives each block equal expected mass (guide §9):
       e_i = [ std(z_i) | a_p std(p0) | a_e std(onehot(y)-p0)
               | a_r std(log1p(r_bar)) ]
  3. r_bar-weighted k-means on e_i        (was: on z only, unclipped r)
  4. HARD regime assignment: S_k drawn only from R_k = {i: argmin_k d = k}
     (0825.md §6 knob c-1; kills cross-regime row sharing by construction)
  5. diversity-aware selection inside R_k (weighted k-center approximation,
     guide §11.2): r_bar-weighted MiniBatchKMeans(--diversity-subclusters)
     cells define coverage; block budget split across cells by largest
     remainder on r_bar-mass (>=1 per non-empty cell, capped at cell size);
     top-r_bar rows represent each cell. TopB and --tau-c are gone.
  + mandated diversity diagnostics (guide §11): block row-ID Jaccard (2b),
    expert prediction agreement (3c), gain correlation (3d), positive-gain
    coverage (3e).

  Ablation paths inside the package: --diversity-subclusters 1 degenerates
  to hard assignment + top-r_bar (knob c-1 alone); --sig-alpha-* 0 recovers
  a z-only signature; --residual-clip-q 1.0 disables the clip.

Label-dependent signature blocks (confusion direction, residual severity)
exist only on D_mine, where clustering and selection happen. Router/eval
d_k uses the label-free observable prefix [std(z)|a_p std(p0)] against mu's
matching coordinates (guide §15: pre-call features must be computable before
the expert call); h_pre/h_post shapes are unchanged.

Everything else -- pools, w_bal (gamma=1), K, anchor, phi (embed_pca),
advantage targets, router, verifier, decision rule -- byte-identical to
exp22b, including its declared embed_pca memorized-row confound. Pre-
registered residual cracks NOT addressed here (deferred knobs): verifier
Delta-regression collapse (gamma=1 scale), positive-advantage ~93% anchor
structure, prior correction (guide Phase 2), shared utility scorer (Phase 3),
quantile verifier + calibration (Phase 4), 6-way split manifest (Phase 0).

Judged against (same run, same split): global TabPFN, full-pool XGBoost,
dense oracle. Cross-run reference: exp22b run 20260825_174233
(system 0.7084 / global 0.6700 / XGB 0.7548 / oracle 0.8255).

Guide §20.2/§22/§23/§25 per-run record (measurement-only; system numbers
unaffected): 2c context priors · 5b class activation · 6c per-class 4-way
correctness transitions (wrong->wrong made visible) · 6d predicted-class
FP/TP accounting (the web-FP path net-correction misses, 0825.md §6) ·
6e guardrails (balanced acc, benign FPR, macro-AUPRC, ECE; system posterior
assembled from accepted expert probas) · verifier FA/FR in 4b · realized
gain, rejected/wasted calls, proposal precision and loss-based
oracle-utility-recovery in 6a · npz dumps: system posterior, all context
row IDs, cluster membership, offline gain matrix (+sha256 of the persisted
float32 bytes in timings) · split dup-hash overlap (§25.2) and
env/versions/commit/data-file/peak-GPU in timings.json ·
--feasibility-banks random,proximity = §24 Gate-1 controls (equal-memory
random and feature-only proximity banks, dense-oracle columns only; needs
--dense-eval; each bank adds K expert fits + K dense passes). Out of scope
for one dev run (deferred): multi-seed CI (§23), IF sweep (§20.3), prior
shift (§20.4), temporal refresh (§17/§20.5), batch-latency percentiles
(§20.6), baseline zoo (§21), verifier calibration curves (Phase 4).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python tabpfn/nfv3_v3_exp22c_racepfn_context.py --target-dataset cic2018 \
        --dense-eval

smoke (capped suite):
    ... --target-dataset cic2018_capped --global-context-size 20000 \
        --anchor-per-class 300 --expert-block-rows 4000 --mine-max-rows 60000 \
        --router-cap-per-class 3000 --phi-fit-rows 60000 --kmeans-max-rows 60000 \
        --diversity-subclusters 256 --test-cap-per-class 2000 --dense-eval
"""

import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfv3_v3_common as core  # noqa: E402
from render_run_tables import render_dir  # noqa: E402  (scripts/ on path via core)

from tabpfn import TabPFNClassifier  # noqa: E402

# Seed bands: identical to exp22b, plus DIVERSITY 2100 / FEASIBILITY 2200
# (spaced >=100; cap_per_class adds +500+i -> ROUTER_CAP occupies 2000+i).
SEED_BAND_GLOBAL = 990
SEED_BAND_ANCHOR = 1100
SEED_BAND_PHI = 1200
SEED_BAND_KMEANS = 1300
SEED_BAND_KMEANS_SUB = 1450
SEED_BAND_ROUTER_CAP = 1500
SEED_BAND_AFFINITY = 1600
SEED_BAND_ROUTER = 1700
SEED_BAND_VERIFIER = 1800
SEED_BAND_MINE = 1900
SEED_BAND_DIVERSITY = 2100
SEED_BAND_FEASIBILITY = 2200

NO_EXPERT = 0


# ---------------- helpers (exp22b lineage; copied, not imported) ------------

def uniq_rows(Xr):
    h = pd.util.hash_pandas_object(pd.DataFrame(Xr), index=False).values
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    return Xr[first], inv


def full_proba(proba, model_classes, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=np.int64)] = proba
    return out


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


def pick_rows(rows, k, seed):
    """Seeded per-class draw (exp22's random mode; medoid unsupported here)."""
    if k >= len(rows):
        return rows
    return np.random.default_rng(seed).permutation(rows)[:k]


def representative_subset(pool_idx, pool_y, n_classes, budget, seed):
    """Pooled budget, natural class ratios (largest remainder, >=1/present)."""
    counts = np.bincount(pool_y, minlength=n_classes)
    total = int(counts.sum())
    if budget <= 0 or budget >= total:
        return np.sort(pool_idx)
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
    chosen = [pick_rows(pool_idx[pool_y == cid], int(tgt[cid]), seed + cid)
              for cid in range(n_classes) if tgt[cid] > 0]
    return np.sort(np.concatenate(chosen))


def anchor_subset(pool_idx, pool_y, n_classes, per_class, seed):
    """EQUAL per-class anchor: min(per_class, n_c) rows for every class."""
    chosen = []
    for cid in range(n_classes):
        rows = pool_idx[pool_y == cid]
        if len(rows):
            chosen.append(pick_rows(rows, min(per_class, len(rows)), seed + cid))
    return np.sort(np.concatenate(chosen))


def class_chrono_partition(train_idx, y_train, ts_train, ctx_frac, mine_frac,
                           class_names):
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


# ---------------- phi variants (exp22b's knob, kept frozen) -----------------

class PhiRawPCA:
    """exp22's phi: RobustScaler -> PCA on the raw 46 features."""

    def __init__(self, feats_fit, dim, seed):
        self.scaler = RobustScaler().fit(feats_fit)
        self.pca = PCA(n_components=dim, random_state=seed).fit(
            self.scaler.transform(feats_fit))

    def transform(self, feats, chunk=500_000):
        outs = [self.pca.transform(self.scaler.transform(feats[s0:s0 + chunk]))
                .astype(np.float32) for s0 in range(0, len(feats), chunk)]
        return np.concatenate(outs) if outs else \
            np.zeros((0, self.pca.n_components_), dtype=np.float32)


class PhiQuantilePCA:
    """Rank/quantile transform (kills heavy tails) -> PCA."""

    def __init__(self, feats_fit, dim, seed):
        self.qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, len(feats_fit)),
            subsample=len(feats_fit), random_state=seed).fit(feats_fit)
        self.pca = PCA(n_components=dim, random_state=seed).fit(
            self.qt.transform(feats_fit))

    def transform(self, feats, chunk=500_000):
        outs = [self.pca.transform(self.qt.transform(feats[s0:s0 + chunk]))
                .astype(np.float32) for s0 in range(0, len(feats), chunk)]
        return np.concatenate(outs) if outs else \
            np.zeros((0, self.pca.n_components_), dtype=np.float32)


class PhiEmbedPCA:
    """z = PCA( global TabPFN test-embedding ) -- the foundation-model-native
    geometry. embed_fn(feats) must return the (n, d_emb) embedding matrix."""

    def __init__(self, feats_fit, dim, seed, embed_fn):
        self.embed_fn = embed_fn
        emb = embed_fn(feats_fit)
        self.pca = PCA(n_components=min(dim, emb.shape[1]),
                       random_state=seed).fit(emb)

    def transform(self, feats, chunk=None):
        # dedup + chunking live inside embed_fn; PCA is cheap afterwards
        return self.pca.transform(self.embed_fn(feats)).astype(np.float32)


# ---------------- residual signature + diversity coreset (THE KNOB) ---------

class ResidualSignature:
    """Guide §9 residual failure signature, per-block standardized.

    Block order (the first two are label-free = the observable prefix):
      z  std(z)                       phi coordinates
      p  a_p * std(p0)                global posterior
      e  a_e * std(onehot(y) - p0)    confusion direction   [D_mine only]
      r  a_r * std(log1p(r_bar))      residual severity     [D_mine only]
    Standardization stats fit on D_mine; each block then scaled by
    alpha/sqrt(block_dim) so alpha=1 gives every block equal expected mass.
    Router/eval rows have no labels, so their distances use observable() vs
    the matching centroid prefix mu[:, :obs_dim].
    """

    BLOCKS = ("z", "p", "e", "r")

    def __init__(self, a_p, a_e, a_r):
        self.alpha = {"z": 1.0, "p": a_p, "e": a_e, "r": a_r}
        self.stats = {}

    @staticmethod
    def _blocks(z, p0, y=None, r_bar=None):
        out = {"z": np.asarray(z, dtype=np.float32),
               "p": np.asarray(p0, dtype=np.float32)}
        if y is not None:
            onehot = np.zeros(out["p"].shape, dtype=np.float32)
            onehot[np.arange(len(y)), y] = 1.0
            out["e"] = onehot - out["p"]
            out["r"] = np.log1p(r_bar).astype(np.float32)[:, None]
        return out

    def _std(self, name, arr):
        m, s = self.stats[name]
        return ((arr - m) / s) * np.float32(
            self.alpha[name] / np.sqrt(arr.shape[1]))

    def fit_full(self, z, p0, y, r_bar):
        for name, arr in self._blocks(z, p0, y, r_bar).items():
            m = arr.mean(axis=0).astype(np.float32)
            s = np.maximum(arr.std(axis=0), 1e-6).astype(np.float32)
            self.stats[name] = (m, s)
        return self.full(z, p0, y, r_bar)

    def full(self, z, p0, y, r_bar):
        b = self._blocks(z, p0, y, r_bar)
        return np.concatenate([self._std(n, b[n]) for n in self.BLOCKS], axis=1)

    def observable(self, z, p0):
        b = self._blocks(z, p0)
        return np.concatenate([self._std(n, b[n]) for n in ("z", "p")], axis=1)

    @property
    def obs_dim(self):
        return int(self.stats["z"][0].shape[0] + self.stats["p"][0].shape[0])


def mass_budget_alloc(mass, sizes, budget):
    """Largest-remainder split of `budget` across cells proportional to
    weighted mass; >=1 per non-empty cell when affordable, capped at size."""
    m = np.maximum(np.asarray(mass, dtype=np.float64), 0.0)
    sizes = np.asarray(sizes, dtype=np.int64)
    raw = m / max(m.sum(), 1e-12) * budget
    tgt = np.minimum(np.floor(raw).astype(np.int64), sizes)
    tgt[(sizes > 0) & (tgt < 1)] = 1
    tgt = np.minimum(tgt, sizes)
    over = int(tgt.sum()) - budget
    if over > 0:                       # budget < #non-empty cells: shed lightest
        for cid in np.argsort(m, kind="stable"):
            if over <= 0:
                break
            take = min(int(tgt[cid]), over)
            tgt[cid] -= take
            over -= take
    rem = budget - int(tgt.sum())
    while rem > 0:
        progressed = False
        for cid in np.argsort(-(raw - np.floor(raw)), kind="stable"):
            if rem <= 0:
                break
            if tgt[cid] < sizes[cid]:
                tgt[cid] += 1
                rem -= 1
                progressed = True
        if not progressed:
            break
    return tgt


def diversity_select(e_regime, r_regime, budget, n_sub, n_init, batch_size,
                     seed):
    """Weighted k-center style pick inside ONE hard regime (guide §11.2
    approximation of argmin_S sum_i r_bar_i min_{j in S} ||e_i - e_j||^2):
    r_bar-weighted MiniBatchKMeans cells define coverage, budget follows
    r_bar mass, top-r_bar rows represent each cell. Returns positions into
    e_regime. n_sub=1 degenerates to plain top-r_bar (knob c-1)."""
    n = len(r_regime)
    if n <= budget:
        return np.arange(n)
    m = int(min(n_sub, budget, n))
    if m <= 1:
        return np.sort(np.argsort(-r_regime, kind="stable")[:budget])
    mbk = MiniBatchKMeans(n_clusters=m, random_state=seed, n_init=n_init,
                          batch_size=batch_size).fit(e_regime,
                                                     sample_weight=r_regime)
    lab = mbk.labels_
    mass = np.bincount(lab, weights=r_regime, minlength=m)
    sizes = np.bincount(lab, minlength=m)
    tgt = mass_budget_alloc(mass, sizes, budget)
    cell_order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[cell_order], np.arange(m + 1))
    picked = []
    for cid in range(m):
        members = cell_order[bounds[cid]:bounds[cid + 1]]
        if tgt[cid] <= 0 or len(members) == 0:
            continue
        top = members[np.argsort(-r_regime[members], kind="stable")[:tgt[cid]]]
        picked.append(top)
    return np.sort(np.concatenate(picked)) if picked else \
        np.zeros(0, dtype=np.int64)


def sq_dist_to_centroids(z, mu, chunk=500_000):
    outs = []
    for s0 in range(0, len(z), chunk):
        zb = z[s0:s0 + chunk]
        outs.append(((zb[:, None, :] - mu[None, :, :]) ** 2).sum(-1))
    return np.concatenate(outs).astype(np.float32)


class AffinityRef:
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


# ---------------- guide §20.2 guardrail metrics -----------------------------

def balanced_acc_of(y, pred, n_classes):
    accs = [float((pred[y == c] == c).mean())
            for c in range(n_classes) if (y == c).any()]
    return float(np.mean(accs)) if accs else np.nan


def macro_auprc_of(probs, y, n_classes):
    vals = [average_precision_score(y == c, probs[:, c])
            for c in range(n_classes) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(vals)) if vals else np.nan


def ece_of(probs, y, bins=15):
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == y).astype(np.float64)
    idx = np.clip((conf * bins).astype(np.int64), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------

def run_exp22c(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp22c_racepfn_context"
    print(f"Args: {vars(args)}", flush=True)
    if args.subsample_samples:
        raise SystemExit("--subsample-samples must stay 0.")
    if args.context_frac + args.mine_frac >= 1.0:
        raise SystemExit("--context-frac + --mine-frac must be < 1.")
    if args.train_split != "train":
        raise SystemExit("--train-split is not implemented in exp22c.")
    if args.skip_tabpfn:
        raise SystemExit("--skip-tabpfn is meaningless here.")
    if args.context_selection != "random":
        raise SystemExit("--context-selection medoid is not supported in "
                         "exp22c: embedding phi is built AFTER the global "
                         "fit, so no z exists when C0 is drawn. Use exp22.")
    if not 0.0 < args.residual_clip_q <= 1.0:
        raise SystemExit("--residual-clip-q must be in (0, 1].")
    feas_banks = [] if args.feasibility_banks.strip().lower() in ("", "none") \
        else [b.strip().lower() for b in args.feasibility_banks.split(",")
              if b.strip()]
    for b in feas_banks:
        if b not in ("random", "proximity"):
            raise SystemExit(f"unknown feasibility bank '{b}' "
                             "(choices: random, proximity, none)")

    timings = {"phi_mode": args.phi_mode}
    t_all = time.time()
    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, _, _, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    K = args.n_clusters
    y_train = label_fn(train_idx)

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

    pool_counts = np.maximum(np.bincount(y_train, minlength=n_classes), 1)
    w_bal = (len(y_train) / (n_classes * pool_counts)) ** args.residual_gamma
    w_bal = w_bal.astype(np.float64)
    print("balanced weights:", {n: round(float(w_bal[i]), 4)
                                for i, n in enumerate(class_names)})

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    # ---- Step 1 indices: C0 + anchor (random draws; no z needed yet) -------
    y_ctx_pool = label_fn(ctx_pool)
    g_idx = representative_subset(ctx_pool, y_ctx_pool, n_classes,
                                  args.global_context_size,
                                  args.seed + SEED_BAND_GLOBAL)
    anchor_idx = anchor_subset(ctx_pool, y_ctx_pool, n_classes,
                               args.anchor_per_class,
                               args.seed + SEED_BAND_ANCHOR)
    phi_fit_idx = core.stratified_subset(ctx_pool, y_ctx_pool, n_classes,
                                         args.phi_fit_rows,
                                         args.seed + SEED_BAND_PHI)
    print(f"C0: {len(g_idx):,} rows | anchor A: {len(anchor_idx):,} rows | "
          f"phi fit rows: {len(phi_fit_idx):,}")
    # declared confound bookkeeping: rows that are BOTH phi-fit/anchor and
    # inside C0 get "memorized-row" embeddings in embed_pca mode (docstring)
    phi_in_c0 = int(len(np.intersect1d(phi_fit_idx, g_idx)))
    anchor_in_c0 = int(len(np.intersect1d(anchor_idx, g_idx)))
    timings["phi_fit_rows_in_C0"] = phi_in_c0
    timings["anchor_rows_in_C0"] = anchor_in_c0
    print(f"C0 overlap: phi-fit {phi_in_c0:,}/{len(phi_fit_idx):,} · "
          f"anchor {anchor_in_c0:,}/{len(anchor_idx):,} "
          f"(embed_pca memorized-row confound record)", flush=True)

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
    X_phi = feats_of(phi_fit_idx)
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

    # guide §25.2: duplicate/hash overlap record (row-content hashes; index
    # disjointness does not rule out content duplicates on NetFlow rows)
    t0 = time.time()

    def _rh(Xr):
        return pd.util.hash_pandas_object(pd.DataFrame(Xr),
                                          index=False).values
    h_eval = _rh(X_eval)
    hu_eval = np.unique(h_eval)
    dup = {"eval_rows": len(h_eval), "eval_distinct": int(len(hu_eval))}
    for tag, Xa in (("C0", glob_ctx[0]), ("anchor", anchor[0]),
                    ("phi_fit", X_phi), ("mine", X_mine),
                    ("router", X_router), ("xgb_train", X_xgb)):
        ha = _rh(Xa)
        dup[tag] = {"rows": len(ha),
                    "distinct_shared_with_eval":
                    int(len(np.intersect1d(hu_eval, np.unique(ha)))),
                    "eval_rows_covered": int(np.isin(h_eval, ha).sum())}
    timings["dup_hash_overlap"] = dup
    print(f"dup-hash overlap (§25.2): eval distinct {dup['eval_distinct']:,}"
          f"/{dup['eval_rows']:,} | eval rows content-duplicated in "
          f"C0/mine/router/xgb: {dup['C0']['eval_rows_covered']:,}/"
          f"{dup['mine']['eval_rows_covered']:,}/"
          f"{dup['router']['eval_rows_covered']:,}/"
          f"{dup['xgb_train']['eval_rows_covered']:,} "
          f"({round(time.time() - t0, 1)}s)", flush=True)

    del X
    core._PICKLE_CACHE.clear()
    gc.collect()
    print(f"mine rows used: {len(mine_idx):,} | router rows: {len(router_idx):,} "
          f"| eval rows: {len(eval_idx):,} | xgb train: {len(xgb_train_idx):,}",
          flush=True)
    core.report_memory_plan(len(g_idx),
                            max(len(X_eval), len(X_mine), len(X_router)), args)

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()   # §20.6 peak-memory record

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

    # ---- XGBoost baseline --------------------------------------------------
    t0 = time.time()
    y_pred_xgb, y_proba_xgb = None, None
    if not args.skip_xgboost:
        booster = xgb.XGBClassifier(
            n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda, objective="multi:softprob",
            num_class=n_classes, eval_metric="mlogloss", n_jobs=-1,
            random_state=args.seed)
        print(f"XGBoost baseline fitting on {len(X_xgb):,} rows ...", flush=True)
        booster.fit(X_xgb, y_xgb)
        y_proba_xgb = full_proba(
            booster.predict_proba(X_eval).astype(np.float32),
            booster.classes_, n_classes)
        y_pred_xgb = y_proba_xgb.argmax(axis=1)
        timings["xgb_baseline_s"] = round(time.time() - t0, 1)
        print(f"XGBoost done ({timings['xgb_baseline_s']}s)", flush=True)
    del X_xgb, y_xgb
    gc.collect()

    # ---- global TabPFN (must exist BEFORE phi in embed mode) --------------
    t0 = time.time()
    glob = make_clf(glob_ctx[0], glob_ctx[1])
    timings["global_fit_s"] = round(time.time() - t0, 1)
    print(f"global fitted: {len(glob_ctx[0]):,} ctx rows "
          f"({timings['global_fit_s']}s)", flush=True)

    embed_stage = {"tag": "phi"}   # log label only; set before each pass

    def embed_global(Xr):
        """Dedup + chunked global.get_embeddings; (n, d_emb) float32.
        exp21 precedent: 1M-context embedding OOMs at chunk 250k, OK at 100k."""
        if args.embed_chunk <= 0:
            raise SystemExit("--embed-chunk must be positive")
        Xu, inv = uniq_rows(Xr)
        outs = []
        n_chunks = (len(Xu) + args.embed_chunk - 1) // args.embed_chunk
        for bn, s0 in enumerate(range(0, len(Xu), args.embed_chunk), 1):
            e = np.asarray(glob.get_embeddings(Xu[s0:s0 + args.embed_chunk],
                                               "test"))
            if e.ndim == 3:      # (n_estimators, n, d) -> first estimator
                e = e[0]
            outs.append(e.astype(np.float32))
            if bn == 1 or bn % 5 == 0 or bn == n_chunks:
                print(f"    [embed/{embed_stage['tag']}] chunk {bn}/{n_chunks}",
                      flush=True)
        free_gpu()
        emb = np.concatenate(outs)
        outs.clear()             # drop the chunk copies before broadcasting
        out = emb[inv]
        del emb
        return out

    # ---- phi (exp22b's knob, kept at its record setting) -------------------
    t0 = time.time()
    phi_seed = args.seed + SEED_BAND_PHI + 50
    if args.phi_mode == "raw_pca":
        phi = PhiRawPCA(X_phi, args.phi_dim, phi_seed)
    elif args.phi_mode == "quantile_pca":
        phi = PhiQuantilePCA(X_phi, args.phi_dim, phi_seed)
    else:
        phi = PhiEmbedPCA(X_phi, args.phi_dim, phi_seed, embed_global)
    del X_phi
    gc.collect()
    timings["phi_fit_s"] = round(time.time() - t0, 1)
    z_dim = int(phi.pca.n_components_)
    print(f"phi ready: mode={args.phi_mode} z_dim={z_dim} "
          f"(requested {args.phi_dim}; {timings['phi_fit_s']}s)", flush=True)

    # ---- Step 2 (THE KNOB): residual signature + clustering on D_mine ------
    t0 = time.time()
    p0_mine = batched_proba(glob, X_mine, "global/mine")
    r_mine = balanced_ce(p0_mine, y_mine, w_bal)
    r_max = float(np.quantile(r_mine, args.residual_clip_q)) \
        if args.residual_clip_q < 1.0 else float(r_mine.max())
    r_bar = np.minimum(r_mine, r_max)
    n_clipped = int((r_mine > r_max).sum())
    timings["residual_clip_r_max"] = round(r_max, 6)
    timings["residual_clipped_rows"] = n_clipped
    print(f"residual clip: q={args.residual_clip_q} r_max={r_max:.4f} "
          f"clipped {n_clipped:,}/{len(r_mine):,} rows", flush=True)
    embed_stage["tag"] = "mine"
    z_mine = phi.transform(X_mine)
    sig = ResidualSignature(args.sig_alpha_p, args.sig_alpha_e,
                            args.sig_alpha_r)
    e_mine = sig.fit_full(z_mine, p0_mine, y_mine, r_bar)
    km_rows = np.arange(len(e_mine))
    if args.kmeans_max_rows and len(km_rows) > args.kmeans_max_rows:
        km_rows = np.random.default_rng(args.seed + SEED_BAND_KMEANS_SUB) \
            .permutation(len(e_mine))[: args.kmeans_max_rows]
    km = KMeans(n_clusters=K, random_state=args.seed + SEED_BAND_KMEANS,
                n_init=args.kmeans_n_init).fit(e_mine[km_rows],
                                               sample_weight=r_bar[km_rows])
    mu = km.cluster_centers_.astype(np.float32)
    mu_obs = mu[:, : sig.obs_dim]      # label-free prefix for router/eval d_k
    d2_mine = sq_dist_to_centroids(e_mine, mu)
    assign_mine = d2_mine.argmin(axis=1)
    d2_med = float(np.median(d2_mine[np.arange(len(e_mine)), assign_mine]))
    timings["mining_s"] = round(time.time() - t0, 1)
    print(f"residual clustering: K={K} sig_dim={e_mine.shape[1]} "
          f"(obs {sig.obs_dim}) median-assigned-d2={d2_med:.4f} "
          f"({timings['mining_s']}s)", flush=True)

    residual_stats = pd.DataFrame([{
        "class": class_names[c],
        "rows": int((y_mine == c).sum()),
        "weight": float(w_bal[c]),
        "mean_ce": float((r_mine[y_mine == c] / w_bal[c]).mean())
        if (y_mine == c).any() else np.nan,
        "mean_residual": float(r_mine[y_mine == c].mean())
        if (y_mine == c).any() else np.nan,
        "clipped_rows": int(((r_mine > r_max) & (y_mine == c)).sum()),
    } for c in range(n_classes)])
    cluster_rows = []
    for k in range(K):
        m = assign_mine == k
        row = {"cluster": k, "rows": int(m.sum()),
               "weighted_mass": float(r_bar[m].sum() / max(r_bar.sum(), 1e-12)),
               "mean_residual": float(r_mine[m].mean()) if m.any() else np.nan}
        for c in range(n_classes):
            row[class_names[c]] = int((m & (y_mine == c)).sum())
        cluster_rows.append(row)
    cluster_df = pd.DataFrame(cluster_rows)
    print("\n=== residual cluster composition (rows per class) ===")
    print(cluster_df.to_string(index=False), flush=True)

    # ---- Step 3 (THE KNOB): C_k = A ∪ S_k, hard regime + diversity coreset -
    experts, expert_ctx_rows = [], []
    anchor_comp = {"expert": "anchor(shared)", "regime_rows": 0,
                   "anchor_rows": len(anchor[0]), "block_rows": 0,
                   "select_s": 0.0, "fit_s": 0.0}
    for c in range(n_classes):
        anchor_comp[class_names[c]] = int((anchor[1] == c).sum())
    expert_ctx_rows.append(anchor_comp)
    for k in range(K):
        members = np.flatnonzero(assign_mine == k)
        t0 = time.time()
        sel = diversity_select(e_mine[members], r_bar[members],
                               args.expert_block_rows,
                               args.diversity_subclusters,
                               args.diversity_n_init,
                               args.diversity_batch_size,
                               args.seed + SEED_BAND_DIVERSITY + k)
        t_sel = round(time.time() - t0, 1)
        top = members[sel]
        if len(top) < args.expert_block_rows:
            print(f"expert {k + 1}: regime {len(members):,} rows < budget "
                  f"{args.expert_block_rows:,} -> block truncated", flush=True)
        Xk = np.concatenate([anchor[0], X_mine[top]])
        yk = np.concatenate([anchor[1], y_mine[top]])
        t0 = time.time()
        experts.append({"clf": make_clf(Xk, yk), "block_rows": top})
        comp = {"expert": k + 1, "regime_rows": len(members),
                "anchor_rows": len(anchor[0]), "block_rows": len(top),
                "select_s": t_sel, "fit_s": round(time.time() - t0, 1)}
        for c in range(n_classes):
            comp[class_names[c]] = int((y_mine[top] == c).sum())
        expert_ctx_rows.append(comp)
        print(f"expert {k + 1}: ctx {len(Xk):,} (anchor {len(anchor[0]):,} + "
              f"block {len(top):,} of regime {len(members):,}) "
              f"select {t_sel}s fit {comp['fit_s']}s", flush=True)
    expert_ctx_df = pd.DataFrame(expert_ctx_rows)

    # block row-ID Jaccard (guide §11 diversity metric; hard assignment -> 0)
    jac_rows = []
    for i in range(K):
        si = set(experts[i]["block_rows"].tolist())
        for j in range(i + 1, K):
            sj = set(experts[j]["block_rows"].tolist())
            inter, union = len(si & sj), len(si | sj)
            jac_rows.append({"expert_a": i + 1, "expert_b": j + 1,
                             "intersection": inter,
                             "jaccard": round(inter / max(union, 1), 6)})
    jaccard_df = pd.DataFrame(jac_rows)
    print("block Jaccard (hard assignment -> expected 0):")
    print(jaccard_df.to_string(index=False), flush=True)

    # guide §23 "context class priors" (2c): C0 / anchor / each expert context
    def _prior_row(tag, ys):
        row = {"context": tag, "rows": len(ys)}
        for c in range(n_classes):
            row[class_names[c]] = round(float((ys == c).mean()), 6)
        return row
    prior_df = pd.DataFrame(
        [_prior_row("C0", glob_ctx[1]), _prior_row("anchor", anchor[1])]
        + [_prior_row(f"expert{k + 1}", np.concatenate(
            [anchor[1], y_mine[experts[k]["block_rows"]]]))
           for k in range(K)])

    # ---- affinity references ----------------------------------------------
    rng_a = np.random.default_rng(args.seed + SEED_BAND_AFFINITY)
    embed_stage["tag"] = "ctx-aff"
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

    # ---- guide §11 diversity diagnostics on D_router -----------------------
    preds_all = [p0_rt.argmax(axis=1)] + [p.argmax(axis=1) for p in pk_rt]
    pred_names = ["global"] + [f"e{k + 1}" for k in range(K)]
    agree_rows = []
    for i in range(len(preds_all)):
        for j in range(i + 1, len(preds_all)):
            agree_rows.append({
                "a": pred_names[i], "b": pred_names[j],
                "agreement": round(float((preds_all[i] == preds_all[j]).mean()),
                                   4)})
    agree_df = pd.DataFrame(agree_rows)
    gcorr_rows = []
    for i in range(K):
        for j in range(i + 1, K):
            ok = delta_rt[:, i].std() > 0 and delta_rt[:, j].std() > 0
            gcorr_rows.append({
                "a": f"e{i + 1}", "b": f"e{j + 1}",
                "gain_pearson_r": round(float(np.corrcoef(
                    delta_rt[:, i], delta_rt[:, j])[0, 1]), 4) if ok else np.nan})
    gcorr_df = pd.DataFrame(gcorr_rows)
    pos_rt = delta_rt > 0
    uniq_pos = pos_rt & (pos_rt.sum(axis=1) == 1)[:, None]
    cover_df = pd.DataFrame([{
        "expert": k + 1,
        "frac_positive": round(float(pos_rt[:, k].mean()), 4),
        "frac_unique_positive": round(float(uniq_pos[:, k].mean()), 4),
        "mean_delta_positive": round(float(delta_rt[pos_rt[:, k], k].mean()), 4)
        if pos_rt[:, k].any() else np.nan} for k in range(K)])
    print("expert prediction agreement:\n" + agree_df.to_string(index=False))
    print("gain correlation:\n" + gcorr_df.to_string(index=False))
    print("positive-gain coverage:\n" + cover_df.to_string(index=False),
          flush=True)

    # ---- Step 5: proposal router ------------------------------------------
    t0 = time.time()
    embed_stage["tag"] = "router"
    z_rt = phi.transform(X_router)
    d2_rt = sq_dist_to_centroids(sig.observable(z_rt, p0_rt), mu_obs)
    a0_rt = aff0.score(z_rt)
    hpre_rt = build_hpre(p0_rt, z_rt, d2_rt, a0_rt)
    order = np.argsort(router_ts, kind="stable")
    n_fit = int(len(order) * (1.0 - args.router_holdout_frac))
    fit_rows, hold_rows = order[:n_fit], order[n_fit:]

    router_clf, router_note = None, "trained"
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
            num_class=len(present),
            eval_metric="mlogloss", n_jobs=-1,
            random_state=args.seed + SEED_BAND_ROUTER)
        router_clf.fit(hpre_rt[fit_rows],
                       np.searchsorted(present, kstar[fit_rows]),
                       sample_weight=sw)
    timings["router_train_s"] = round(time.time() - t0, 1)

    def propose(hpre):
        if router_clf is None:
            return np.zeros(len(hpre), dtype=np.int64)
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
            "expert": k + 1, "hold_rows": len(dt),
            "sign_accuracy": float(((dh > 0) == (dt > 0)).mean()),
            "false_accept": int(((dh > 0) & (dt <= 0)).sum()),
            "false_reject": int(((dh <= 0) & (dt > 0)).sum()),
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
    embed_stage["tag"] = "eval"
    z_eval = phi.transform(X_eval)
    d2_eval = sq_dist_to_centroids(sig.observable(z_eval, p0_eval), mu_obs)
    a0_eval = aff0.score(z_eval)
    proposal = propose(build_hpre(p0_eval, z_eval, d2_eval, a0_eval))
    timings["eval_router_s"] = round(time.time() - t0, 1)

    L0_eval_bal = balanced_ce(p0_eval, y_eval, w_bal)
    probs_sys = p0_eval.astype(np.float32)   # system posterior (guide §20.2)
    realized = np.zeros(len(y_eval), dtype=np.float32)  # §22 realized gain
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
        realized[rows] = (L0_eval_bal[rows]
                          - balanced_ce(pk, y_eval[rows], w_bal)
                          ).astype(np.float32)
        probs_sys[rows[acc]] = pk[acc].astype(np.float32)
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

    # ---- routing quality ---------------------------------------------------
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
        "block_jaccard_max": float(jaccard_df["jaccard"].max())
        if len(jaccard_df) else np.nan,
        "gain_corr_max": float(gcorr_df["gain_pearson_r"].max())
        if len(gcorr_df) else np.nan,
    }
    # per-expert override decomposition (0825.md §4 보강 재분석을 상시 기록)
    override_rows = []
    for k in range(K):
        m = (proposal == k + 1) & accepted
        h = int((m & ~glob_ok & final_ok).sum())
        b = int((m & glob_ok & ~final_ok).sum())
        override_rows.append({
            "expert": k + 1, "accepted": int(m.sum()), "helpful": h,
            "harmful": b, "precision": round(h / max(h + b, 1), 4),
            "net": h - b})
    override_df = pd.DataFrame(override_rows)
    print(override_df.to_string(index=False), flush=True)

    # guide §22: correctness transitions / FP accounting / class activation;
    # rates always with absolute counts, realized gain, benign harm
    names_l = [str(n).lower() for n in class_names]
    benign_id = next((names_l.index(n) for n in ("benign", "normal")
                      if n in names_l), None)
    if benign_id is None:
        print("benign guardrails skipped: no benign/normal class", flush=True)
    called = proposal != 0
    n_called, n_acc = int(called.sum()), int(accepted.sum())
    quality.update({
        "eval_rows": len(y_eval),
        "override_among_calls": round(n_acc / max(n_called, 1), 4),
        "rejected_calls": n_called - n_acc,
        "rejected_call_rate": round((n_called - n_acc) / max(n_called, 1), 4),
        "mean_realized_gain_calls": round(float(realized[called].mean()), 4)
        if n_called else np.nan,
        "mean_realized_gain_overrides":
        round(float(realized[accepted].mean()), 4) if n_acc else np.nan,
    })
    if benign_id is not None:
        bmask = y_eval == benign_id
        quality.update({
            "benign_rows": int(bmask.sum()),
            "benign_harmful_override": int((bmask & glob_ok & ~final_ok).sum()),
        })
    trans_rows, fp_rows, act_rows = [], [], []
    for c in range(n_classes):
        m = y_eval == c
        gw = m & ~glob_ok
        ww = gw & ~final_ok
        trans_rows.append({
            "class": class_names[c], "rows": int(m.sum()),
            "g_ok_f_ok": int((m & glob_ok & final_ok).sum()),
            "g_wrong_f_ok_helpful": int((gw & final_ok).sum()),
            "g_ok_f_wrong_harmful": int((m & glob_ok & ~final_ok).sum()),
            "g_wrong_f_wrong": int(ww.sum()),
            "g_wrong_f_wrong_changed": int((ww & (final != y_glob)).sum())})
        fp_rows.append({
            "class": class_names[c],
            "fp_global": int(((y_glob == c) & ~m).sum()),
            "fp_final": int(((final == c) & ~m).sum()),
            "fp_delta": int(((final == c) & ~m).sum())
            - int(((y_glob == c) & ~m).sum()),
            "tp_global": int(((y_glob == c) & m).sum()),
            "tp_final": int(((final == c) & m).sum())})
        act_rows.append({
            "class": class_names[c], "rows": int(m.sum()),
            "proposed": int((m & called).sum()),
            "accepted": int((m & accepted).sum())})
    trans_df, fp_df, act_df = map(pd.DataFrame, (trans_rows, fp_rows, act_rows))
    print("\n=== correctness transitions by class (guide §22) ===")
    print(trans_df.to_string(index=False))
    print("\n=== predicted-class FP/TP accounting (wrong->wrong visible) ===")
    print(fp_df.to_string(index=False), flush=True)

    all_rows = list(core.per_class_table("racepfn_system", y_eval, final,
                                         class_names, tail_classes))
    all_rows.extend(core.per_class_table("global_tabpfn", y_eval, y_glob,
                                         class_names, tail_classes))
    if y_pred_xgb is not None:
        all_rows.extend(core.per_class_table("xgboost", y_eval, y_pred_xgb,
                                             class_names, tail_classes))

    dense_note = "skipped (--dense-eval off)"
    bank_blocks = {}
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
        gains = L_eval[:, 0:1] - L_eval[:, 1:]
        pos_adv_eval = gains.max(axis=1) > 0
        quality["eval_pos_adv_rows"] = int(pos_adv_eval.sum())
        quality["eval_pos_adv_proposal_recall"] = round(
            float((proposal[pos_adv_eval] != 0).mean()), 4) \
            if pos_adv_eval.any() else np.nan
        quality["eval_pos_adv_proposal_precision"] = round(
            float(pos_adv_eval[called].mean()), 4) if n_called else np.nan
        oracle_total = float(np.maximum(gains.max(axis=1), 0.0).sum())
        realized_total = float(realized[accepted].sum())
        quality["oracle_utility_total"] = round(oracle_total, 1)
        quality["realized_utility_total"] = round(realized_total, 1)
        quality["oracle_utility_recovery"] = round(
            realized_total / oracle_total, 4) if oracle_total > 0 else np.nan
        timings["dense_eval_s"] = round(time.time() - t0, 1)
        dense_note = "done"

        # guide §20.1/§24 Gate-1 controls: equal-memory random and
        # feature-only proximity banks -- dense-oracle columns only,
        # system predictions untouched
        for bank in feas_banks:
            t0 = time.time()
            bank_preds, blocks_b = [y_glob], []
            L_bank = np.zeros((len(y_eval), K + 1), dtype=np.float64)
            L_bank[:, 0] = L_eval[:, 0]
            for k in range(K):
                if bank == "random":
                    rows_b = np.sort(np.random.default_rng(
                        args.seed + SEED_BAND_FEASIBILITY + k).choice(
                            len(X_mine),
                            size=min(args.expert_block_rows, len(X_mine)),
                            replace=False))
                else:                        # proximity: z-block space only
                    d2z = ((e_mine[:, :z_dim] - mu[k, :z_dim][None, :]) ** 2
                           ).sum(axis=1)
                    rows_b = np.sort(np.argsort(d2z, kind="stable")
                                     [: args.expert_block_rows])
                blocks_b.append(rows_b)
                clf_b = make_clf(np.concatenate([anchor[0], X_mine[rows_b]]),
                                 np.concatenate([anchor[1], y_mine[rows_b]]))
                pb = batched_proba(clf_b, X_eval, f"{bank}{k + 1}/dense")
                L_bank[:, k + 1] = (balanced_ce(pb, y_eval, w_bal)
                                    + args.lambda_cost * args.expert_cost)
                bank_preds.append(pb.argmax(axis=1))
                del clf_b
                free_gpu()
            pick_b = L_bank.argmin(axis=1)
            y_ob = np.take_along_axis(np.stack(bank_preds, axis=1),
                                      pick_b[:, None], axis=1)[:, 0]
            all_rows.extend(core.per_class_table(
                f"oracle_{bank}_bank", y_eval, y_ob, class_names,
                tail_classes))
            bank_blocks[bank] = blocks_b
            timings[f"feasibility_{bank}_s"] = round(time.time() - t0, 1)
            print(f"feasibility bank '{bank}' done "
                  f"({timings[f'feasibility_{bank}_s']}s)", flush=True)
    if feas_banks and not args.dense_eval:
        print("feasibility banks skipped (--dense-eval off)", flush=True)

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
    for mname in ["dense_oracle"] + [f"oracle_{b}_bank" for b in feas_banks]:
        if mname in piv.columns:
            quality[f"{mname}_macro"] = round(f1_of(mname), 4)

    # guide §20.2 IID guardrails (6e)
    methods_pp = {"racepfn_system": (final, probs_sys),
                  "global_tabpfn": (y_glob, p0_eval)}
    if y_pred_xgb is not None:
        methods_pp["xgboost"] = (y_pred_xgb, y_proba_xgb)
    guard_rows = []
    for mname, (pred, pr) in methods_pp.items():
        grow = {"method": mname, "rows": len(y_eval),
                "balanced_accuracy": round(
                    balanced_acc_of(y_eval, pred, n_classes), 4),
                "macro_auprc": round(macro_auprc_of(pr, y_eval, n_classes), 4),
                "ece_15bin": round(ece_of(pr, y_eval), 4)}
        if benign_id is not None:
            grow["benign_fpr"] = round(
                float((pred[y_eval == benign_id] != benign_id).mean()), 6)
        guard_rows.append(grow)
    guard_df = pd.DataFrame(guard_rows)
    print("\n=== guardrails (guide §20.2) ===")
    print(guard_df.to_string(index=False), flush=True)
    quality_df = pd.DataFrame([quality])
    print("\n=== routing quality ===")
    print(quality_df.to_string(index=False))
    timings["total_s"] = round(time.time() - t_all, 1)
    timings["dense_eval"] = dense_note
    timings["rho"] = round(rho, 4)

    # ---- artifacts ---------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp22c_racectx")
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
    jaccard_df.to_csv(os.path.join(out_dir, "2b_expert_jaccard.csv"),
                      index=False)
    prior_df.to_csv(os.path.join(out_dir, "2c_context_priors.csv"),
                    index=False)
    adv_df.to_csv(os.path.join(out_dir, "3a_advantage_stats.csv"), index=False)
    target_df.to_csv(os.path.join(out_dir, "3b_router_targets.csv"), index=False)
    agree_df.to_csv(os.path.join(out_dir, "3c_expert_agreement.csv"),
                    index=False)
    gcorr_df.to_csv(os.path.join(out_dir, "3d_gain_correlation.csv"),
                    index=False)
    cover_df.to_csv(os.path.join(out_dir, "3e_gain_coverage.csv"), index=False)
    router_hold_df.to_csv(os.path.join(out_dir, "4a_router_holdout_confusion.csv"),
                          index=False)
    ver_df.to_csv(os.path.join(out_dir, "4b_verifier_holdout.csv"), index=False)
    decision_df.to_csv(os.path.join(out_dir, "5a_decision_stats.csv"), index=False)
    act_df.to_csv(os.path.join(out_dir, "5b_activation_by_class.csv"),
                  index=False)
    quality_df.to_csv(os.path.join(out_dir, "6a_routing_quality.csv"), index=False)
    override_df.to_csv(os.path.join(out_dir, "6b_override_by_expert.csv"),
                       index=False)
    trans_df.to_csv(os.path.join(out_dir, "6c_correctness_transition.csv"),
                    index=False)
    fp_df.to_csv(os.path.join(out_dir, "6d_class_fp_accounting.csv"),
                 index=False)
    guard_df.to_csv(os.path.join(out_dir, "6e_guardrails.csv"), index=False)
    np.savez_compressed(
        os.path.join(out_dir, "system_dump.npz"),
        proposal=proposal.astype(np.int8), accepted=accepted,
        delta_hat=delta_hat, final=final.astype(np.int64),
        y_glob=y_glob.astype(np.int64), y_true=y_eval.astype(np.int64),
        mu=mu, obs_dim=np.int64(sig.obs_dim), d2_med=np.float32(d2_med),
        r_max=np.float64(r_max),
        sig_mean=np.concatenate([sig.stats[n][0] for n in sig.BLOCKS]),
        sig_std=np.concatenate([sig.stats[n][1] for n in sig.BLOCKS]),
        sig_alpha=np.asarray([sig.alpha[n] for n in sig.BLOCKS],
                             dtype=np.float64),
        class_names=np.asarray(class_names))
    np.savez_compressed(os.path.join(out_dir, "probs_tabpfn_global.npz"),
                        probs=p0_eval.astype(np.float32),
                        y_true=y_eval.astype(np.int64))
    # guide §23 must-save record: predictions/posterior, all context row IDs,
    # cluster membership, offline gain matrix (+hash), env/versions/commit
    np.savez_compressed(os.path.join(out_dir, "probs_racepfn_system.npz"),
                        probs=probs_sys, y_true=y_eval.astype(np.int64))
    ctx_dump = {"C0": g_idx, "anchor": anchor_idx, "phi_fit": phi_fit_idx,
                "mine": mine_idx, "router": router_idx, "eval": eval_idx}
    for k in range(K):
        ctx_dump[f"expert{k + 1}_block"] = mine_idx[experts[k]["block_rows"]]
    for bank, blocks in bank_blocks.items():
        for k, rows_b in enumerate(blocks):
            ctx_dump[f"{bank}{k + 1}_block"] = mine_idx[rows_b]
    np.savez_compressed(os.path.join(out_dir, "context_rows.npz"), **ctx_dump)
    np.savez_compressed(os.path.join(out_dir, "mining_dump.npz"),
                        mine_idx=mine_idx, assign=assign_mine.astype(np.int8),
                        r=r_mine.astype(np.float32),
                        r_bar=r_bar.astype(np.float32))
    delta_f32 = np.ascontiguousarray(delta_rt.astype(np.float32))
    np.savez_compressed(os.path.join(out_dir, "router_gain.npz"),
                        router_idx=router_idx, delta=delta_f32,
                        kstar=kstar.astype(np.int8),
                        y=y_router.astype(np.int32))
    # hash the persisted float32 bytes so the dump can be verified offline
    timings["gain_matrix_sha256"] = hashlib.sha256(
        delta_f32.tobytes()).hexdigest()
    timings["gain_matrix_dtype"] = "float32"
    if torch.cuda.is_available():
        timings["peak_gpu_mem_gb"] = round(
            torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    st = os.stat(args.data)
    timings["data_file"] = {"path": args.data, "bytes": st.st_size,
                            "mtime": int(st.st_mtime)}

    def _ver(pkg):
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:
            return "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() \
            or "unknown"
    except Exception:
        commit = "unknown"
    timings["env"] = {
        "cmd": " ".join(sys.argv), "git_commit": commit,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": _ver("scikit-learn"),
        "xgboost": _ver("xgboost"), "torch": torch.__version__,
        "tabpfn": _ver("tabpfn"),
        "cuda": getattr(torch.version, "cuda", None),
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else "cpu"}
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([timings], f, indent=2)
    try:
        render_dir(out_dir)   # CSV+PNG pairs (scripts/render_run_tables.py)
    except Exception as exc:
        print(f"PNG rendering failed (CSVs intact): {exc}")
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-frac", type=float, default=0.50)
    p.add_argument("--mine-frac", type=float, default=0.25)
    p.add_argument("--global-context-size", type=int, default=1_000_000)
    p.add_argument("--context-selection", default="random",
                   choices=["random", "medoid"],
                   help="exp22c supports 'random' only (embedding phi exists "
                        "only after the global fit); medoid -> SystemExit")
    p.add_argument("--anchor-per-class", type=int, default=2_000)
    p.add_argument("--expert-block-rows", type=int, default=186_000)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--phi-mode", default="embed_pca",
                   choices=["embed_pca", "quantile_pca", "raw_pca"],
                   help="exp22b's knob, frozen at its record setting "
                        "(embed_pca). raw_pca/quantile_pca kept for ablation.")
    p.add_argument("--embed-chunk", type=int, default=100_000,
                   help="rows per get_embeddings call on the 1M-ctx global "
                        "(exp21: OOM at 250k, OK at 100k on the 24GB card)")
    p.add_argument("--phi-dim", type=int, default=16)
    p.add_argument("--phi-fit-rows", type=int, default=500_000)
    p.add_argument("--residual-clip-q", type=float, default=0.995,
                   help="THE KNOB pt.1: r_bar = min(r, quantile(r, q)); "
                        "1.0 disables the clip (guide §8 r_max)")
    p.add_argument("--sig-alpha-p", type=float, default=1.0,
                   help="THE KNOB pt.2: posterior-block weight in the "
                        "residual signature (0 removes the block)")
    p.add_argument("--sig-alpha-e", type=float, default=1.0,
                   help="THE KNOB pt.2: confusion-direction block weight")
    p.add_argument("--sig-alpha-r", type=float, default=1.0,
                   help="THE KNOB pt.2: residual-severity block weight")
    p.add_argument("--diversity-subclusters", type=int, default=2_048,
                   help="THE KNOB pt.5: coverage cells per regime for the "
                        "weighted k-center approximation; 1 = plain "
                        "hard-assignment + top-r_bar (knob c-1)")
    p.add_argument("--diversity-n-init", type=int, default=3,
                   help="n_init for the per-regime coverage MiniBatchKMeans")
    p.add_argument("--diversity-batch-size", type=int, default=10_000,
                   help="batch_size for the per-regime coverage MiniBatchKMeans")
    p.add_argument("--feasibility-banks", default="random,proximity",
                   help="guide §20.1/§24 Gate-1 controls: comma list of "
                        "{random, proximity} equal-memory context banks "
                        "reported as dense-oracle columns only; 'none' "
                        "disables; needs --dense-eval; measurement-only "
                        "(system predictions unaffected)")
    p.add_argument("--kmeans-max-rows", type=int, default=1_000_000)
    p.add_argument("--kmeans-n-init", type=int, default=5)
    p.add_argument("--residual-gamma", type=float, default=1.0)
    p.add_argument("--mine-max-rows", type=int, default=2_000_000)
    p.add_argument("--router-cap-per-class", type=int, default=100_000)
    p.add_argument("--router-holdout-frac", type=float, default=0.2)
    p.add_argument("--lambda-cost", type=float, default=0.0)
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
    p.add_argument("--dense-eval", action="store_true")
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
    run_exp22c(args)


if __name__ == "__main__":
    main()

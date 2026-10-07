#!/usr/bin/env python3
"""EXP30 -- C0 BENIGN SHARE: global 컨텍스트의 클래스 구성 원노브.

    --c0-benign-share S   C0(1M 또는 --global-context-size) 안에서 benign이
                          차지하는 비율을 S로 고정; 나머지 예산은 공격 클래스에
                          자연 비율(largest remainder)로 배분.  -1(기본) = exp29와
                          동일한 자연 비율(benign 0.8707).

왜 이 knob인가 (0902.md §2a + r3 20260902_021534 진단):
    * r3 global의 ddos 오답 73,287행은 전부 hoic -> benign이고, C0의 ddos 행 수는
      chrono와 층화에서 동일(0.065838)하며 hoic 행은 오히려 늘었다(36,653 ->
      51,184).  같은 C0 행으로 XGB-c0의 hoic recall은 0.99997.  즉 정보는
      C0 안에 있고, 87% benign 컨텍스트에서 TabPFN이 못 쓰는 것이다.
    * 놓친 hoic 행의 global p_benign 중앙값 0.956 -- 정답 benign(0.98)과
      거의 구분되지 않아 어떤 scorer도 라벨 없이 찾지 못한다.  반면 anchor
      (benign 2,000 : ddos 2,000)를 가진 expert는 e1/e3/e4 모두 제안만 되면
      70~96% 채택, 정밀도 0.99+로 ddos로 되돌린다.
    * scorer/verifier는 route 창에서 학습되는데 그 창에서 ddos는 global 오답이
      아니다(1a mean residual 0.02) -- 학습 성분은 test에서만 나타나는 실패를
      배울 수 없다.  컨텍스트 구성은 학습 성분을 우회하는 유일한 축이다.

먼저 읽을 것 (같은 run 안에서):
    1. 6g_test_scenario_recall.csv -- hoic recall (r3 global 0.6557) 이 S에 따라
       열리는가; loic_http/udp 는 유지되는가
    2. per_class_metrics.csv benign recall 소수점 4자리 (r3 0.9981; 0.001 =
       3,500행) 와 inf/web precision -- benign 비중 축소의 대가
    3. 2c_context_priors.csv / 0g_c0_scenario.csv -- 실제로 만들어진 C0 구성
    주의: S를 바꾸면 matched-budget xgb_c0(0902 §1, 0.7283)도 새 C0로 재계산해야
    비교가 성립한다 (scripts/race_budget_diag.py).

탐색 레인 (4090, 약 30분/run; 1M 기록급은 exp29 커맨드 + --c0-benign-share):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python tabpfn/nfv3_v3_exp30_c0share.py --target-dataset cic2018 \
        --prune-mode off --global-context-size 250000 \
        --feasibility-banks none --skip-xgboost --c0-benign-share {-1,0.75,0.60}

smoke (capped suite; 예산을 작게 줘야 share 경로가 실행됨):
    python tabpfn/nfv3_v3_exp30_c0share.py --target-dataset cic2018_capped \
        --prune-mode off --global-context-size 20000 --feasibility-banks none \
        --skip-xgboost --c0-benign-share 0.6

base = exp29 사본 (combo anchor: 층화 분할 + normgain + 등메모리 + det-precision).
추가: representative_subset_share(), 0g_c0_scenario.csv, 6g_test_scenario_recall.csv,
timings.env.git_dirty.  다른 코드 경로는 exp29와 바이트 동일.
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
from sklearn.metrics import average_precision_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfv3_v3_common as core  # noqa: E402
from render_run_tables import render_dir  # noqa: E402  (scripts/ on path via core)

from tabpfn import TabPFNClassifier  # noqa: E402

# Seed bands (spaced >=100; cap_per_class adds +500+i, so the *_CAP bands
# effectively occupy band+500+i -- ROUTE_CAP 2000+i, TUNE_CAP 2800+i,
# CAL_CAP 2900+i; single-value bands sit clear of those ranges).
SEED_BAND_GLOBAL = 990
SEED_BAND_ANCHOR = 1100
SEED_BAND_PHI = 1200
SEED_BAND_KMEANS = 1300
SEED_BAND_KMEANS_SUB = 1450
SEED_BAND_ROUTE_CAP = 1500
SEED_BAND_AFFINITY = 1600
SEED_BAND_SCORER = 1700
SEED_BAND_VERIFIER = 1800
SEED_BAND_EXPERT_POOL = 1900
SEED_BAND_DIVERSITY = 2100
SEED_BAND_FEASIBILITY = 2200
SEED_BAND_TUNE_CAP = 2300
SEED_BAND_CAL_CAP = 2400
SEED_BAND_REGIME_RANDOM = 2500     # exp28: 'regime_random' feasibility bank
SEED_BAND_OOF = 2600

NO_EXPERT = 0


# ---------------- helpers (exp22c lineage; copied, not imported) ------------

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


def gain_norm(y, w, mode):
    """THE KNOB (exp26): per-row divisor taking a w_bal-weighted-CE gain into
    the verifier's target unit.  mode='gain' -> 1 (exp24b, raw weighted-CE);
    mode='normgain' -> w_bal[y], i.e. plain delta-NLL, which removes the
    per-class scale (0.164 .. 1132.8) from the acceptance rule by construction
    and compresses the regression target from ~6 decades to ~2 (0831.md
    cause B).  Applied to the FINAL verifier fit and the D_cal correction
    only; the OOF fold fit stays raw on purpose so b_OOF -- and therefore the
    pre-call scorer and tau_pre -- are byte-identical to exp24b (see the
    module docstring)."""
    if mode == "normgain":
        return w[y].astype(np.float64)
    return np.ones(len(y), dtype=np.float64)


def pick_rows(rows, k, seed):
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


def representative_subset_share(pool_idx, pool_y, n_classes, budget, seed,
                                benign_id, benign_share):
    """THE KNOB (exp30): pooled budget with benign pinned to `benign_share`
    of the budget; the rest is split over the other present classes by their
    natural ratios (largest remainder), capped by availability with the
    shortfall redistributed.  Per-class draw seeds are the same as
    representative_subset, so a class whose target count is unchanged draws
    the identical rows."""
    counts = np.bincount(pool_y, minlength=n_classes)
    total = int(counts.sum())
    if budget <= 0 or budget >= total:
        return np.sort(pool_idx)
    tgt = np.zeros(n_classes, dtype=np.int64)
    tgt[benign_id] = min(int(round(budget * benign_share)),
                         int(counts[benign_id]))
    rem = budget - int(tgt[benign_id])
    for _ in range(n_classes):
        sub = np.asarray([c for c in range(n_classes)
                          if c != benign_id and tgt[c] < counts[c]],
                         dtype=np.int64)
        if rem <= 0 or len(sub) == 0:
            break
        raw = counts[sub] * (rem / counts[sub].sum())
        add = np.floor(raw).astype(np.int64)
        for j in np.argsort(-(raw - np.floor(raw))):
            if int(add.sum()) >= rem:
                break
            add[j] += 1
        add = np.minimum(add, counts[sub] - tgt[sub])
        tgt[sub] += add
        rem = budget - int(tgt.sum())
    present = counts > 0
    tgt[present & (tgt < 1)] = 1
    chosen = [pick_rows(pool_idx[pool_y == cid], int(tgt[cid]), seed + cid)
              for cid in range(n_classes) if tgt[cid] > 0]
    return np.sort(np.concatenate(chosen))


def anchor_subset(pool_idx, pool_y, n_classes, per_class, seed):
    chosen = []
    for cid in range(n_classes):
        rows = pool_idx[pool_y == cid]
        if len(rows):
            chosen.append(pick_rows(rows, min(per_class, len(rows)), seed + cid))
    return np.sort(np.concatenate(chosen))


def scenario_stratified_partition(train_idx, y_train, ts_train, scen_train,
                                  ctx_frac, exp_frac, class_names):
    """THE KNOB (exp27): 50/25/25 chrono split applied WITHIN each
    (class, attack_scenario) group instead of within each class.

    Preserves "context precedes expert/route" inside every scenario; drops only
    the global ordering between scenarios, which is what starved C0 of
    ssh_bruteforce (0831.md cause A).  Scenarios too small to fill three parts
    go wholly to context and are flagged in the audit.
    """
    parts = {"context": [], "expert": [], "route": []}
    scen_rows = []
    for cid, cname in enumerate(class_names):
        cmask = y_train == cid
        if not cmask.any():
            continue
        for sname in sorted(np.unique(scen_train[cmask])):
            mask = cmask & (scen_train == sname)
            rows = train_idx[mask]
            if len(rows) == 0:
                continue
            order = rows[np.argsort(ts_train[mask], kind="stable")]
            n = len(order)
            n_ctx, n_exp = int(n * ctx_frac), int(n * exp_frac)
            n_rt = n - n_ctx - n_exp
            if min(n_ctx, n_exp, n_rt) <= 0:      # too small to split 3 ways
                parts["context"].extend(order)
                scen_rows.append({"class": cname, "scenario": sname,
                                  "train_total": n, "context": n, "expert": 0,
                                  "route": 0, "note": "too_small_all_context"})
                continue
            parts["context"].extend(order[:n_ctx])
            parts["expert"].extend(order[n_ctx:n_ctx + n_exp])
            parts["route"].extend(order[n_ctx + n_exp:])
            scen_rows.append({"class": cname, "scenario": sname,
                              "train_total": n, "context": n_ctx,
                              "expert": n_exp, "route": n_rt, "note": ""})
    scen_audit = pd.DataFrame(scen_rows)
    audit = (scen_audit.groupby("class", as_index=False)
             [["train_total", "context", "expert", "route"]].sum())
    return ({k: np.sort(np.asarray(v, dtype=np.int64))
             for k, v in parts.items()}, audit, scen_audit)


def class_chrono_partition(train_idx, y_train, ts_train, ctx_frac, exp_frac,
                           class_names):
    """Train -> D_global(context) / D_expert / D_route, chrono per class."""
    parts = {"context": [], "expert": [], "route": []}
    audit = []
    for cid, cname in enumerate(class_names):
        mask = y_train == cid
        rows = train_idx[mask]
        if len(rows) == 0:
            continue
        order = rows[np.argsort(ts_train[mask], kind="stable")]
        n = len(order)
        n_ctx = int(n * ctx_frac)
        n_exp = int(n * exp_frac)
        n_rt = n - n_ctx - n_exp
        if min(n_ctx, n_exp, n_rt) <= 0:
            raise SystemExit(
                f"class {cname}: {n} train rows cannot fill "
                f"context/expert/route ({n_ctx}/{n_exp}/{n_rt})")
        parts["context"].extend(order[:n_ctx])
        parts["expert"].extend(order[n_ctx:n_ctx + n_exp])
        parts["route"].extend(order[n_ctx + n_exp:])
        audit.append({"class": cname, "train_total": n, "context": n_ctx,
                      "expert": n_exp, "route": n_rt})
    return ({k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in parts.items()},
            pd.DataFrame(audit))


def scenario_stratified_split2(idx, y, ts, scen, first_frac, class_names):
    """exp27 (same knob, val side): val -> D_tune / D_cal chronologically
    WITHIN each (class, attack_scenario) group.

    Applying the knob only to the train partition would leave C0 containing
    ssh_bruteforce while D_tune stayed ftp-only (val brute = ftp 77,344 +
    ssh 37,694, and the first 40% by class-internal time is all ftp), so
    tuning and calibration would still see a different scenario mix than the
    context. That is the same defect this experiment exists to remove, so the
    knob is applied to both role splits or to neither.
    """
    first, second, rows_out = [], [], []
    for cid, cname in enumerate(class_names):
        cmask = y == cid
        if not cmask.any():
            continue
        for sname in sorted(np.unique(scen[cmask])):
            mask = cmask & (scen == sname)
            rows = idx[mask]
            if len(rows) == 0:
                continue
            order = rows[np.argsort(ts[mask], kind="stable")]
            n1 = int(len(order) * first_frac)
            first.extend(order[:n1])
            second.extend(order[n1:])
            rows_out.append({"class": cname, "scenario": sname,
                             "val_total": len(order), "tune": n1,
                             "cal": len(order) - n1})
    return (np.sort(np.asarray(first, dtype=np.int64)),
            np.sort(np.asarray(second, dtype=np.int64)),
            pd.DataFrame(rows_out))


def class_chrono_split2(idx, y, ts, first_frac, class_names):
    """val -> D_tune (first_frac, earlier) / D_cal (rest, later), per class."""
    first, second = [], []
    for cid in range(len(class_names)):
        mask = y == cid
        rows = idx[mask]
        if len(rows) == 0:
            continue
        order = rows[np.argsort(ts[mask], kind="stable")]
        n1 = int(len(order) * first_frac)
        first.extend(order[:n1])
        second.extend(order[n1:])
    return (np.sort(np.asarray(first, dtype=np.int64)),
            np.sort(np.asarray(second, dtype=np.int64)))


# ---------------- phi variants (exp22b knob, frozen at embed_pca) -----------

class PhiRawPCA:
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
    def __init__(self, feats_fit, dim, seed, embed_fn):
        self.embed_fn = embed_fn
        emb = embed_fn(feats_fit)
        self.pca = PCA(n_components=min(dim, emb.shape[1]),
                       random_state=seed).fit(emb)

    def transform(self, feats, chunk=None):
        return self.pca.transform(self.embed_fn(feats)).astype(np.float32)


# ---------------- Phase-1 machinery (exp22c, unchanged) ---------------------

class ResidualSignature:
    """Guide §9 residual failure signature, per-block standardized (exp22c)."""

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
    m = np.maximum(np.asarray(mass, dtype=np.float64), 0.0)
    sizes = np.asarray(sizes, dtype=np.int64)
    raw = m / max(m.sum(), 1e-12) * budget
    tgt = np.minimum(np.floor(raw).astype(np.int64), sizes)
    tgt[(sizes > 0) & (tgt < 1)] = 1
    tgt = np.minimum(tgt, sizes)
    over = int(tgt.sum()) - budget
    if over > 0:
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


# ---------------- P2: prior correction (guide §7) ---------------------------

class PriorCorrector:
    """z~ = log(p_raw+eps) + beta*(log pi_ref - log pi_hat); softmax(z~/T)."""

    def __init__(self, ctx_labels, n_classes, ref_prior, alpha):
        counts = np.bincount(ctx_labels, minlength=n_classes).astype(np.float64)
        self.pi_hat = (counts + alpha) / (counts.sum() + alpha * n_classes)
        self.shift = (np.log(np.clip(ref_prior, 1e-12, None))
                      - np.log(self.pi_hat))

    def correct(self, p_raw, beta, temp):
        z = np.log(np.clip(p_raw, 1e-12, None)) + beta * self.shift[None, :]
        z = z / max(temp, 1e-6)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def select_prior_hypers(p_raw, y, w, corrector, betas, temps):
    """Grid on D_tune, class-balanced NLL (guide §7: beta,T from D_tune)."""
    rows, best = [], (None, np.inf)
    for b in betas:
        for t in temps:
            nll = float(balanced_ce(corrector.correct(p_raw, b, t), y, w)
                        .mean())
            rows.append({"beta": b, "T": t, "tune_balanced_nll": round(nll, 6)})
            if nll < best[1]:
                best = ((b, t), nll)
    return best[0], pd.DataFrame(rows)


# ---------------- P3/P4: pair features with expert descriptor ---------------

def expert_descriptor(block_r, regime_mass, block_rows, pi_hat, mean_d2,
                      cost):
    """q_k (guide §14): size, mass, mean residual, context prior entropy,
    cluster spread, call cost."""
    ent = float(-(pi_hat * np.log(np.clip(pi_hat, 1e-12, None))).sum())
    return np.asarray(
        [np.log1p(block_rows), float(regime_mass),
         float(np.mean(block_r)) if len(block_r) else 0.0,
         ent, np.log1p(max(mean_d2, 0.0)), float(cost)], dtype=np.float32)


def build_pair_pre(p0, z, d_k, a0, qk):
    n = len(p0)
    return np.concatenate(
        [p0.astype(np.float32), entropy_of(p0)[:, None], margin_of(p0)[:, None],
         z, d_k[:, None].astype(np.float32), a0[:, None],
         np.repeat(qk[None, :], n, axis=0)], axis=1)


def build_pair_post(p0, pk, a0, ak, dk, qk):
    n = len(p0)
    return np.concatenate(
        [p0.astype(np.float32), pk.astype(np.float32),
         (pk - p0).astype(np.float32),
         entropy_of(p0)[:, None], entropy_of(pk)[:, None],
         margin_of(p0)[:, None], margin_of(pk)[:, None],
         a0[:, None], ak[:, None], dk[:, None].astype(np.float32),
         np.repeat(qk[None, :], n, axis=0)], axis=1)


# ---------------- P4: threshold selection on D_cal (guide §14) --------------

def select_thresholds(u_max, g_lower, g_top1, glob_ok_cal, top1_ok_cal,
                      benign_mask, benign_fpr0, tau_pre_grid, tau_post_grid,
                      max_prop, fpr_inc_max, harm_frac_max, min_accepted,
                      min_decided):
    """Grid over (tau_pre, tau_post): maximize net realized gain on D_cal
    subject to the §14 constraint set: benign-FPR increase cap, harmful
    fraction cap, proposal-rate cap, and minimum support (accepted count and
    decided = helpful+harmful count, the harm_frac denominator). The benign
    FPR delta is the TRUE replacement delta: overrides that fix a global FP
    are credited, wrong->wrong overrides add nothing.
    Returns ((tau_pre, tau_post) or None, grid_df)."""
    rows, best = [], (None, -np.inf)
    nb = max(int(benign_mask.sum()), 1)
    for tp in tau_pre_grid:
        called = u_max > tp
        prop = float(called.mean())
        for to in tau_post_grid:
            acc = called & (g_lower > to)
            helpful = int((acc & ~glob_ok_cal & top1_ok_cal).sum())
            harmful = int((acc & glob_ok_cal & ~top1_ok_cal).sum())
            harm_frac = harmful / max(helpful + harmful, 1)
            fpr_added = float(
                (int((acc & benign_mask & ~top1_ok_cal).sum())
                 - int((acc & benign_mask & ~glob_ok_cal).sum())) / nb)
            net = float(g_top1[acc].sum())
            ok = (prop <= max_prop and harm_frac <= harm_frac_max
                  and fpr_added <= fpr_inc_max
                  and int(acc.sum()) >= min_accepted
                  and helpful + harmful >= min_decided)
            rows.append({"tau_pre": float(tp), "tau_post": float(to),
                         "proposal_rate": round(prop, 4),
                         "accepted": int(acc.sum()), "helpful": helpful,
                         "harmful": harmful,
                         "harm_frac": round(harm_frac, 4),
                         "benign_fpr_added": round(fpr_added, 6),
                         "net_gain": round(net, 1), "feasible": ok})
            if ok and net > best[1]:
                best = ((float(tp), float(to)), net)
    return best[0], pd.DataFrame(rows)


# ---------------- guardrail metrics (guide §20.2, exp22c) -------------------

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


def greedy_regime_prune(L_t, y, mask, assign_tune, min_coverage,
                        min_regime_rows, min_regime_gain, mode="regime"):
    """§12 as REGIME-CONDITIONAL greedy elimination (THE KNOB): expert k's
    marginal = mean balanced-loss increase on ITS OWN regime's masked tune
    rows when k is removed from the bank. The shared anchor's generalist
    contribution sits on both sides of the difference and cancels, so only
    the specialist block's unique value is measured. Viability up front:
    global positive-gain coverage >= min_coverage AND regime tune support
    >= min_regime_rows. mode='off' keeps every viable expert (ablation).
    Returns (sorted keep list, per-round record DataFrame)."""
    K = L_t.shape[1] - 1
    G = L_t[:, 0:1] - L_t[:, 1:]
    cov = {k: float((G[mask, k] > 0).mean()) for k in range(K)}
    sup = {k: int((mask & (assign_tune == k)).sum()) for k in range(K)}
    rows, active = [], []
    for k in range(K):
        if cov[k] >= min_coverage and sup[k] >= min_regime_rows:
            active.append(k)
        else:
            why = "coverage" if cov[k] < min_coverage else "min-support"
            rows.append({"round": 0, "expert": k + 1,
                         "coverage": round(cov[k], 4),
                         "regime_tune_rows": sup[k], "marginal": np.nan,
                         "action": f"dropped({why})"})

    def marginal_of(k, act):
        m_k = mask & (assign_tune == k)
        wo = [0] + [j + 1 for j in act if j != k]
        wi = [0] + [j + 1 for j in act]
        return float(L_t[m_k][:, wo].min(axis=1).mean()
                     - L_t[m_k][:, wi].min(axis=1).mean())

    if mode == "off":
        for k in active:
            rows.append({"round": 0, "expert": k + 1,
                         "coverage": round(cov[k], 4),
                         "regime_tune_rows": sup[k],
                         "marginal": round(marginal_of(k, active), 4),
                         "action": "kept(prune off)"})
    else:
        rnd = 1
        while len(active) > 1:
            margs = {k: marginal_of(k, active) for k in active}
            worst = min(active, key=lambda k: (margs[k], cov[k]))
            removing = margs[worst] <= min_regime_gain
            for k in sorted(active):
                rows.append({"round": rnd, "expert": k + 1,
                             "coverage": round(cov[k], 4),
                             "regime_tune_rows": sup[k],
                             "marginal": round(margs[k], 4),
                             "action": "removed" if (removing and k == worst)
                             else "kept"})
            if not removing:
                break
            active.remove(worst)
            rnd += 1
    if not active:
        best_k = max(range(K), key=lambda k: cov[k])
        active = [best_k]
        rows.append({"round": -1, "expert": best_k + 1,
                     "coverage": round(cov[best_k], 4),
                     "regime_tune_rows": sup[best_k], "marginal": np.nan,
                     "action": "kept(fallback)"})
    return sorted(active), pd.DataFrame(rows)


def _parse_floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


# ---------------------------------------------------------------------------

def run_exp29(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp30_c0share"
    print(f"Args: {vars(args)}", flush=True)
    if args.subsample_samples:
        raise SystemExit("--subsample-samples must stay 0.")
    if args.context_frac + args.expert_frac >= 1.0:
        raise SystemExit("--context-frac + --expert-frac must be < 1.")
    if args.train_split != "train":
        raise SystemExit("--train-split is not implemented in exp29.")
    if args.skip_tabpfn:
        raise SystemExit("--skip-tabpfn is meaningless here.")
    if args.context_selection != "random":
        raise SystemExit("--context-selection medoid unsupported (phi is "
                         "built after the global fit). Use exp22.")
    if not 0.0 < args.residual_clip_q <= 1.0:
        raise SystemExit("--residual-clip-q must be in (0, 1].")
    if not 0.0 < args.tune_frac_of_val < 1.0:
        raise SystemExit("--tune-frac-of-val must be in (0, 1).")
    feas_banks = [] if args.feasibility_banks.strip().lower() in ("", "none") \
        else [b.strip().lower() for b in args.feasibility_banks.split(",")
              if b.strip()]
    for b in feas_banks:
        if b not in ("random", "proximity", "regime_random"):
            raise SystemExit(f"unknown feasibility bank '{b}'")
    k_candidates = sorted({int(v) for v in args.k_candidates.split(",")
                           if v.strip()})
    if not k_candidates or min(k_candidates) < 1:
        raise SystemExit("--k-candidates must list positive ints")
    prior_betas = _parse_floats(args.prior_betas)
    prior_temps = _parse_floats(args.prior_temps)
    tau_post_grid = _parse_floats(args.tau_post_grid)
    if any(t < 0 for t in tau_post_grid):
        raise SystemExit("--tau-post-grid values must be >= 0 (guide §14)")

    timings = {"phi_mode": args.phi_mode}
    t_all = time.time()
    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, _, _, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    y_train = label_fn(train_idx)

    d = core.load_pickle(args.data)
    ts_all = np.asarray(d["timestamps" if "timestamps" in d else "time_proxy"],
                        dtype=np.int64)
    scen_all = (np.asarray(d["attack_scenarios"]).astype(str)
                if "attack_scenarios" in d else None)
    del d
    if args.pool_partition == "scenario_stratified" and scen_all is None:
        raise SystemExit("--pool-partition scenario_stratified needs "
                         "'attack_scenarios' in the data pickle"
                         "; use --pool-partition chrono for datasets "
                         "without scenarios (e.g. cic2017_full)")
    ts_train = ts_all[train_idx]
    ts_val = ts_all[val_idx]

    if args.pool_partition == "scenario_stratified":
        pools, pool_audit, scen_audit = scenario_stratified_partition(
            train_idx, y_train, ts_train, scen_all[train_idx],
            args.context_frac, args.expert_frac, class_names)
        print(f"pool partition: scenario-stratified over "
              f"{len(scen_audit)} (class, scenario) groups", flush=True)
    else:
        pools, pool_audit = class_chrono_partition(
            train_idx, y_train, ts_train, args.context_frac, args.expert_frac,
            class_names)
        scen_audit = None
    ctx_pool, expert_pool, route_pool = \
        pools["context"], pools["expert"], pools["route"]
    y_val = label_fn(val_idx)
    if args.pool_partition == "scenario_stratified":
        tune_pool, cal_pool, val_scen_audit = scenario_stratified_split2(
            val_idx, y_val, ts_val, scen_all[val_idx],
            args.tune_frac_of_val, class_names)
    else:
        tune_pool, cal_pool = class_chrono_split2(
            val_idx, y_val, ts_val, args.tune_frac_of_val, class_names)
        val_scen_audit = None
    print(f"pools: D_global={len(ctx_pool):,} D_expert={len(expert_pool):,} "
          f"D_route={len(route_pool):,} D_tune={len(tune_pool):,} "
          f"D_cal={len(cal_pool):,} | test(dev holdout)={len(test_idx):,}")
    print(pool_audit.to_string(index=False), flush=True)

    # 0c split manifest: rows + time range per split*class (guide §5)
    manifest_rows = []
    for tag, rows in (("D_global", ctx_pool), ("D_expert", expert_pool),
                      ("D_route", route_pool), ("D_tune", tune_pool),
                      ("D_cal", cal_pool), ("test_dev_holdout", test_idx)):
        ys, ts_r = label_fn(rows), ts_all[rows]
        for c in range(n_classes):
            m = ys == c
            if not m.any():
                continue
            manifest_rows.append({
                "split": tag, "class": class_names[c], "rows": int(m.sum()),
                "ts_min": int(ts_r[m].min()), "ts_max": int(ts_r[m].max())})
    manifest_df = pd.DataFrame(manifest_rows)
    del ts_val

    pool_counts = np.maximum(np.bincount(y_train, minlength=n_classes), 1)
    w_bal = (len(y_train) / (n_classes * pool_counts)) ** args.residual_gamma
    w_bal = w_bal.astype(np.float64)
    print("balanced weights:", {n: round(float(w_bal[i]), 4)
                                for i, n in enumerate(class_names)})

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    # ---- indices --------------------------------------------------------
    y_ctx_pool = label_fn(ctx_pool)
    ref_prior = (np.bincount(y_ctx_pool, minlength=n_classes)
                 / max(len(y_ctx_pool), 1)).astype(np.float64)
    names_l0 = [str(n).lower() for n in class_names]
    benign_id0 = next((names_l0.index(n) for n in ("benign", "normal")
                       if n in names_l0), None)
    if args.c0_benign_share >= 0:                       # THE KNOB (exp30)
        if benign_id0 is None:
            raise SystemExit("--c0-benign-share needs a benign/normal class")
        if args.c0_benign_share > 1:
            raise SystemExit("--c0-benign-share must be in [0, 1] or -1")
        g_idx = representative_subset_share(
            ctx_pool, y_ctx_pool, n_classes, args.global_context_size,
            args.seed + SEED_BAND_GLOBAL, benign_id0, args.c0_benign_share)
    else:
        g_idx = representative_subset(ctx_pool, y_ctx_pool, n_classes,
                                      args.global_context_size,
                                      args.seed + SEED_BAND_GLOBAL)
    timings["c0_benign_share_arg"] = args.c0_benign_share
    y_g0 = label_fn(g_idx)
    timings["c0_benign_share_realized"] = (
        round(float((y_g0 == benign_id0).mean()), 6)
        if benign_id0 is not None else None)
    c0_scen_df = None
    if scen_all is not None:
        c0_scen_df = (pd.DataFrame({"class": [class_names[c] for c in y_g0],
                                    "scenario": scen_all[g_idx]})
                      .value_counts().rename("c0_rows").reset_index()
                      .sort_values(["class", "scenario"]))
    del y_g0
    anchor_idx = anchor_subset(ctx_pool, y_ctx_pool, n_classes,
                               args.anchor_per_class,
                               args.seed + SEED_BAND_ANCHOR)
    phi_fit_idx = core.stratified_subset(ctx_pool, y_ctx_pool, n_classes,
                                         args.phi_fit_rows,
                                         args.seed + SEED_BAND_PHI)
    phi_in_c0 = int(len(np.intersect1d(phi_fit_idx, g_idx)))
    anchor_in_c0 = int(len(np.intersect1d(anchor_idx, g_idx)))
    timings["phi_fit_rows_in_C0"] = phi_in_c0
    timings["anchor_rows_in_C0"] = anchor_in_c0
    print(f"C0: {len(g_idx):,} | anchor: {len(anchor_idx):,} | phi fit: "
          f"{len(phi_fit_idx):,} | C0 overlap phi {phi_in_c0:,} anchor "
          f"{anchor_in_c0:,} (embed_pca confound record)", flush=True)

    exp_idx = core.stratified_subset(expert_pool, label_fn(expert_pool),
                                     n_classes, args.expert_max_rows,
                                     args.seed + SEED_BAND_EXPERT_POOL)
    route_idx = core.cap_per_class(route_pool, label_fn(route_pool),
                                   n_classes, args.route_cap_per_class,
                                   args.seed + SEED_BAND_ROUTE_CAP)
    tune_idx = core.cap_per_class(tune_pool, label_fn(tune_pool), n_classes,
                                  args.tune_cap_per_class,
                                  args.seed + SEED_BAND_TUNE_CAP)
    cal_idx = core.cap_per_class(cal_pool, label_fn(cal_pool), n_classes,
                                 args.cal_cap_per_class,
                                 args.seed + SEED_BAND_CAL_CAP)
    eval_idx = core.cap_per_class(test_idx, label_fn(test_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)

    glob_ctx = (feats_of(g_idx), label_fn(g_idx))
    anchor = (feats_of(anchor_idx), label_fn(anchor_idx))
    X_phi = feats_of(phi_fit_idx)
    X_exp, y_exp = feats_of(exp_idx), label_fn(exp_idx)
    X_route, y_route = feats_of(route_idx), label_fn(route_idx)
    route_ts = ts_train[np.searchsorted(train_idx, route_idx)]
    X_tune, y_tune = feats_of(tune_idx), label_fn(tune_idx)
    X_cal, y_cal = feats_of(cal_idx), label_fn(cal_idx)
    X_eval, y_eval = feats_of(eval_idx), label_fn(eval_idx)
    del ts_train
    xgb_train_idx = train_idx
    if args.max_train_samples > 0 and args.max_train_samples < len(train_idx):
        xgb_train_idx = core.stratified_subset(
            train_idx, y_train, n_classes, args.max_train_samples,
            args.seed + 850)
    X_xgb, y_xgb = feats_of(xgb_train_idx), label_fn(xgb_train_idx)

    # guide §25.2/§5: duplicate/hash overlap record + ENFORCEMENT masks.
    # NetFlow rows repeat verbatim across time, so index-disjoint splits
    # still share content duplicates; selection/calibration statistics must
    # not be computed on rows the fitted objects contained verbatim.
    t0 = time.time()

    def _rh(Xr):
        return pd.util.hash_pandas_object(pd.DataFrame(Xr),
                                          index=False).values
    hashes = {}
    for tag, Xa in (("C0", glob_ctx[0]), ("anchor", anchor[0]),
                    ("phi_fit", X_phi), ("expert", X_exp),
                    ("route", X_route), ("tune", X_tune), ("cal", X_cal),
                    ("xgb_train", X_xgb), ("eval", X_eval)):
        hashes[tag] = _rh(Xa)
    hu = {t: np.unique(v) for t, v in hashes.items()}
    dup = {"eval_rows": len(hashes["eval"]),
           "eval_distinct": int(len(hu["eval"]))}
    for tag in ("C0", "anchor", "phi_fit", "expert", "route", "tune", "cal",
                "xgb_train"):
        dup[tag] = {"rows": len(hashes[tag]), "distinct": int(len(hu[tag])),
                    "distinct_shared_with_eval":
                    int(len(np.intersect1d(hu["eval"], hu[tag]))),
                    "eval_rows_covered":
                    int(np.isin(hashes["eval"], hu[tag]).sum())}
    tags = list(hu)
    dupmat_df = pd.DataFrame(
        [{"a": a, "b": b,
          "distinct_shared": int(len(np.intersect1d(hu[a], hu[b])))}
         for i, a in enumerate(tags) for b in tags[i + 1:]])
    timings["dup_hash_overlap"] = dup
    # §5 enforcement: tune-side selections (beta/T, K, pruning) exclude tune
    # rows content-duplicated in any used context source (C0/anchor/D_expert);
    # cal-side calibration/thresholds exclude cal rows duplicated in D_route
    # (the scorer/verifier training data).
    tune_ctx_dup = np.isin(hashes["tune"], np.unique(np.concatenate(
        [hu["C0"], hu["anchor"], hu["expert"]])))
    cal_route_dup = np.isin(hashes["cal"], hu["route"])
    timings["tune_rows_ctx_dup"] = int(tune_ctx_dup.sum())
    timings["cal_rows_route_dup"] = int(cal_route_dup.sum())
    print(f"dup-hash overlap (§25.2): eval distinct {dup['eval_distinct']:,}"
          f"/{dup['eval_rows']:,} | §5 masks: tune ctx-dup "
          f"{int(tune_ctx_dup.sum()):,}/{len(tune_ctx_dup):,} · cal "
          f"route-dup {int(cal_route_dup.sum()):,}/{len(cal_route_dup):,} "
          f"({round(time.time() - t0, 1)}s)", flush=True)
    print("§5 note: source_file/session/group IDs are absent from the pkl "
          "schema; group-level separation is recorded as a data limitation",
          flush=True)
    del hashes, hu
    gc.collect()

    def guarded_mask(mask, ys, tag):
        for c in range(n_classes):
            if (ys == c).any() and not (mask & (ys == c)).any():
                print(f"§5 mask '{tag}' would empty class {class_names[c]} "
                      f"-> fallback to unmasked", flush=True)
                timings[f"mask_{tag}_fallback"] = True
                return np.ones(len(mask), dtype=bool)
        return mask
    tune_sel_mask = guarded_mask(~tune_ctx_dup, y_tune, "tune_ctx")
    cal_sel_mask = guarded_mask(~cal_route_dup, y_cal, "cal_route")

    del X
    core._PICKLE_CACHE.clear()
    gc.collect()
    print(f"rows: expert={len(exp_idx):,} route={len(route_idx):,} "
          f"tune={len(tune_idx):,} cal={len(cal_idx):,} eval={len(eval_idx):,} "
          f"xgb={len(xgb_train_idx):,}", flush=True)
    core.report_memory_plan(len(g_idx),
                            max(len(X_eval), len(X_exp), len(X_route)), args)

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ported from exp25 (E-2): opt-in determinism. Env var set in main().
    clf_extra = {}
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.det_precision == "float32":
            clf_extra["inference_precision"] = torch.float32
        print(f"determinism ON: precision={args.det_precision}, "
              "tf32/cudnn-benchmark off", flush=True)
    timings["deterministic"] = bool(args.deterministic)

    def make_clf(Xc, yc):
        clf = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device, **clf_extra)
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

    # ---- XGBoost baseline (§21) ----------------------------------------
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
        print(f"XGBoost fitting on {len(X_xgb):,} rows ...", flush=True)
        booster.fit(X_xgb, y_xgb)
        y_proba_xgb = full_proba(
            booster.predict_proba(X_eval).astype(np.float32),
            booster.classes_, n_classes)
        y_pred_xgb = y_proba_xgb.argmax(axis=1)
        timings["xgb_baseline_s"] = round(time.time() - t0, 1)
        print(f"XGBoost done ({timings['xgb_baseline_s']}s)", flush=True)
    del X_xgb, y_xgb
    gc.collect()

    # ---- global TabPFN + phi -------------------------------------------
    t0 = time.time()
    glob = make_clf(glob_ctx[0], glob_ctx[1])
    timings["global_fit_s"] = round(time.time() - t0, 1)
    print(f"global fitted: {len(glob_ctx[0]):,} rows "
          f"({timings['global_fit_s']}s)", flush=True)

    embed_stage = {"tag": "phi"}

    def embed_global(Xr):
        if args.embed_chunk <= 0:
            raise SystemExit("--embed-chunk must be positive")
        Xu, inv = uniq_rows(Xr)
        outs = []
        n_chunks = (len(Xu) + args.embed_chunk - 1) // args.embed_chunk
        for bn, s0 in enumerate(range(0, len(Xu), args.embed_chunk), 1):
            e = np.asarray(glob.get_embeddings(Xu[s0:s0 + args.embed_chunk],
                                               "test"))
            if e.ndim == 3:
                e = e[0]
            outs.append(e.astype(np.float32))
            if bn == 1 or bn % 5 == 0 or bn == n_chunks:
                print(f"    [embed/{embed_stage['tag']}] chunk {bn}/{n_chunks}",
                      flush=True)
        free_gpu()
        emb = np.concatenate(outs)
        outs.clear()
        out = emb[inv]
        del emb
        return out

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
          f"({timings['phi_fit_s']}s)", flush=True)

    # ---- P2 §7 stage 1: temperature on D_tune (global) ------------------
    # C0 is drawn with natural class ratios, so its prior == pi_ref and the
    # beta shift is ~0 for the GLOBAL: only T is identifiable here. beta is
    # selected in stage 2 on the EXPERT posteriors (whose anchor-balanced
    # contexts are what §7 targets); p~0 uses (beta=0, T) everywhere for
    # cross-stage consistency.
    t0 = time.time()
    corr0 = PriorCorrector(glob_ctx[1], n_classes, ref_prior,
                           args.prior_alpha)
    p0_tune_raw = batched_proba(glob, X_tune, "global/tune")
    (_, temp), prior_grid_df = select_prior_hypers(
        p0_tune_raw[tune_sel_mask], y_tune[tune_sel_mask], w_bal, corr0,
        [0.0], prior_temps)
    timings["prior_select_s"] = round(time.time() - t0, 1)
    timings["prior_T"] = temp
    print(f"prior correction (§7 stage 1): T={temp} on D_tune global NLL "
          f"({timings['prior_select_s']}s)\n"
          + prior_grid_df.to_string(index=False), flush=True)
    p0_tune = corr0.correct(p0_tune_raw, 0.0, temp)

    # ---- Phase 1 (exp22c): residual signature + clustering on D_expert -
    t0 = time.time()
    p0_exp = corr0.correct(batched_proba(glob, X_exp, "global/expert"),
                           0.0, temp)
    r_exp = balanced_ce(p0_exp, y_exp, w_bal)
    r_max = float(np.quantile(r_exp, args.residual_clip_q)) \
        if args.residual_clip_q < 1.0 else float(r_exp.max())
    r_bar = np.minimum(r_exp, r_max)
    timings["residual_clip_r_max"] = round(r_max, 6)
    print(f"residual clip: r_max={r_max:.4f} "
          f"clipped {(r_exp > r_max).sum():,}/{len(r_exp):,}", flush=True)
    embed_stage["tag"] = "expert"
    z_exp = phi.transform(X_exp)
    sig = ResidualSignature(args.sig_alpha_p, args.sig_alpha_e,
                            args.sig_alpha_r)
    e_exp = sig.fit_full(z_exp, p0_exp, y_exp, r_bar)
    km_rows = np.arange(len(e_exp))
    if args.kmeans_max_rows and len(km_rows) > args.kmeans_max_rows:
        km_rows = np.random.default_rng(args.seed + SEED_BAND_KMEANS_SUB) \
            .permutation(len(e_exp))[: args.kmeans_max_rows]
    timings["mining_s"] = round(time.time() - t0, 1)

    embed_stage["tag"] = "tune"
    z_tune = phi.transform(X_tune)

    # ---- P2 §10: K selection on D_tune ---------------------------------
    def build_bank(K, seed_off=0):
        km = KMeans(n_clusters=K,
                    random_state=args.seed + SEED_BAND_KMEANS + seed_off,
                    n_init=args.kmeans_n_init).fit(
                        e_exp[km_rows], sample_weight=r_bar[km_rows])
        mu = km.cluster_centers_.astype(np.float32)
        d2 = sq_dist_to_centroids(e_exp, mu)
        assign = d2.argmin(axis=1)
        experts = []
        for k in range(K):
            members = np.flatnonzero(assign == k)
            sel = diversity_select(e_exp[members], r_bar[members],
                                   args.expert_block_rows,
                                   args.diversity_subclusters,
                                   args.diversity_n_init,
                                   args.diversity_batch_size,
                                   args.seed + SEED_BAND_DIVERSITY + k)
            top = members[sel]
            yk = np.concatenate([anchor[1], y_exp[top]])
            clf = make_clf(np.concatenate([anchor[0], X_exp[top]]), yk)
            corr = PriorCorrector(yk, n_classes, ref_prior, args.prior_alpha)
            mean_d2 = float(d2[members, k].mean()) if len(members) else 0.0
            mass = float(r_bar[members].sum() / max(r_bar.sum(), 1e-12))
            experts.append({
                "clf": clf, "block_rows": top, "corr": corr,
                "regime_rows": len(members),
                "qk": expert_descriptor(r_bar[top], mass, len(top),
                                        corr.pi_hat, mean_d2,
                                        args.expert_cost)})
        return {"K": K, "mu": mu, "assign": assign, "experts": experts}

    t0 = time.time()
    ksel_rows, prior_expert_rows, best = [], [], None
    for K in k_candidates:
        bank = build_bank(K)
        pk_raw = [batched_proba(ex["clf"], X_tune, f"K{K}/e{k + 1}/tune")
                  for k, ex in enumerate(bank["experts"])]
        # §7 stage 2: beta selected per candidate bank on EXPERT tune NLL
        beta_K, best_nll = prior_betas[0], np.inf
        for b in prior_betas:
            nll = float(np.mean([balanced_ce(
                bank["experts"][k]["corr"].correct(pk_raw[k], b, temp),
                y_tune, w_bal)[tune_sel_mask].mean() for k in range(K)]))
            prior_expert_rows.append({
                "K": K, "beta": b,
                "tune_expert_balanced_nll": round(nll, 6)})
            if nll < best_nll:
                beta_K, best_nll = b, nll
        L_t = np.zeros((len(y_tune), K + 1), dtype=np.float64)
        L_t[:, 0] = balanced_ce(p0_tune, y_tune, w_bal)
        preds_t = [p0_tune.argmax(axis=1)]
        pk_tune = []
        for k, ex in enumerate(bank["experts"]):
            pk = ex["corr"].correct(pk_raw[k], beta_K, temp)
            pk_tune.append(pk)
            L_t[:, k + 1] = balanced_ce(pk, y_tune, w_bal)
            preds_t.append(pk.argmax(axis=1))
        del pk_raw
        pick = L_t.argmin(axis=1)
        y_or = np.take_along_axis(np.stack(preds_t, axis=1),
                                  pick[:, None], axis=1)[:, 0]
        mac = float(f1_score(y_tune[tune_sel_mask], y_or[tune_sel_mask],
                             average="macro"))
        ksel_rows.append({"K": K, "beta": beta_K,
                          "tune_oracle_macro": round(mac, 4),
                          "tune_call_frac": round(
                              float((pick[tune_sel_mask] > 0).mean()), 4)})
        print(f"K={K}: beta={beta_K} tune oracle macro {mac:.4f}", flush=True)
        if best is None or mac > best["mac"] + 1e-9:
            best = {"mac": mac, "bank": bank, "L_t": L_t, "beta": beta_K,
                    "pk_tune": pk_tune, "preds_t": preds_t}
        else:
            for ex in bank["experts"]:
                del ex["clf"]
            free_gpu()
    ksel_df = pd.DataFrame(ksel_rows)
    prior_expert_df = pd.DataFrame(prior_expert_rows)
    bank = best["bank"]
    K = bank["K"]
    beta = best["beta"]
    timings["prior_beta"] = beta
    print(f"prior correction (§7 stage 2): beta={beta} "
          f"(expert tune NLL, selected bank)", flush=True)
    mu, assign_exp = bank["mu"], bank["assign"]
    L_t, pk_tune = best["L_t"], best["pk_tune"]
    mu_obs = mu[:, : sig.obs_dim]
    d2_med = float(np.median(
        sq_dist_to_centroids(e_exp, mu)[np.arange(len(e_exp)), assign_exp]))
    timings["k_selection_s"] = round(time.time() - t0, 1)
    print(f"K selected: {K} (§10, tune oracle) "
          f"({timings['k_selection_s']}s)", flush=True)

    # ---- §12 (THE KNOB): regime-conditional greedy pruning -------------
    G_t = L_t[:, 0:1] - L_t[:, 1:]
    r_tune_all = balanced_ce(p0_tune, y_tune, w_bal)
    e_tune_sig = sig.full(z_tune, p0_tune, y_tune,
                          np.minimum(r_tune_all, r_max))
    assign_tune = sq_dist_to_centroids(e_tune_sig, bank["mu"]).argmin(axis=1)
    del e_tune_sig
    gc.collect()
    keep, prune_df = greedy_regime_prune(
        L_t, y_tune, tune_sel_mask, assign_tune,
        args.prune_min_coverage, args.prune_min_regime_rows,
        args.prune_min_regime_gain, args.prune_mode)
    print("regime-conditional greedy pruning (§12):\n"
          + prune_df.to_string(index=False), flush=True)
    experts = [bank["experts"][k] for k in keep]
    mu, mu_obs = mu[keep], mu_obs[keep]
    pk_tune = [pk_tune[k] for k in keep]
    G_t = G_t[:, keep]
    K = len(keep)
    qk_mat = np.stack([ex["qk"] for ex in experts])
    keep_set = set(keep)
    for j, ex in enumerate(bank["experts"]):
        if j not in keep_set:
            ex.pop("clf", None)     # free pruned experts' fit caches
    bank["experts"] = experts
    del best, L_t
    free_gpu()

    # ---- context composition artifacts (2a/2b/2c) ----------------------
    expert_ctx_rows = []
    anchor_comp = {"expert": "anchor(shared)", "regime_rows": 0,
                   "anchor_rows": len(anchor[0]), "block_rows": 0}
    for c in range(n_classes):
        anchor_comp[class_names[c]] = int((anchor[1] == c).sum())
    expert_ctx_rows.append(anchor_comp)
    for k, ex in enumerate(experts):
        comp = {"expert": k + 1, "regime_rows": ex["regime_rows"],
                "anchor_rows": len(anchor[0]),
                "block_rows": len(ex["block_rows"])}
        for c in range(n_classes):
            comp[class_names[c]] = int((y_exp[ex["block_rows"]] == c).sum())
        expert_ctx_rows.append(comp)
    expert_ctx_df = pd.DataFrame(expert_ctx_rows)
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

    def _prior_row(tag, pi):
        row = {"context": tag}
        for c in range(n_classes):
            row[class_names[c]] = round(float(pi[c]), 6)
        return row
    prior_df = pd.DataFrame(
        [_prior_row("ref(D_global)", ref_prior),
         _prior_row("C0", corr0.pi_hat)]
        + [_prior_row(f"expert{k + 1}", experts[k]["corr"].pi_hat)
           for k in range(K)])

    cluster_rows = []
    for k, kk in enumerate(keep):
        m = assign_exp == kk
        row = {"cluster": k, "rows": int(m.sum()),
               "weighted_mass": float(r_bar[m].sum() / max(r_bar.sum(), 1e-12)),
               "mean_residual": float(r_exp[m].mean()) if m.any() else np.nan}
        for c in range(n_classes):
            row[class_names[c]] = int((m & (y_exp == c)).sum())
        cluster_rows.append(row)
    cluster_df = pd.DataFrame(cluster_rows)
    print("\n=== residual cluster composition (kept experts) ===")
    print(cluster_df.to_string(index=False), flush=True)

    residual_stats = pd.DataFrame([{
        "class": class_names[c], "rows": int((y_exp == c).sum()),
        "weight": float(w_bal[c]),
        "mean_residual": float(r_exp[y_exp == c].mean())
        if (y_exp == c).any() else np.nan,
        "clipped_rows": int(((r_exp > r_max) & (y_exp == c)).sum()),
    } for c in range(n_classes)])

    # ---- affinity refs --------------------------------------------------
    rng_a = np.random.default_rng(args.seed + SEED_BAND_AFFINITY)
    embed_stage["tag"] = "ctx-aff"
    z_g = phi.transform(glob_ctx[0])
    aff0 = AffinityRef(z_g[rng_a.permutation(len(z_g))[: args.affinity_ref_rows]],
                       args.affinity_nn)
    aff_k = []
    for k in range(K):
        zb = z_exp[experts[k]["block_rows"]]
        aff_k.append(AffinityRef(
            zb[rng_a.permutation(len(zb))[: args.affinity_ref_rows]],
            args.affinity_nn))
    del z_g
    gc.collect()

    # ---- §13: offline pure-gain matrix on D_route ----------------------
    t0 = time.time()
    p0_rt = corr0.correct(batched_proba(glob, X_route, "global/route"),
                          0.0, temp)
    L0_rt = balanced_ce(p0_rt, y_route, w_bal)
    pk_rt, G_rt = [], np.zeros((len(y_route), K), dtype=np.float64)
    for k in range(K):
        pk = experts[k]["corr"].correct(
            batched_proba(experts[k]["clf"], X_route, f"e{k + 1}/route"),
            beta, temp)
        pk_rt.append(pk.astype(np.float32))
        G_rt[:, k] = L0_rt - balanced_ce(pk, y_route, w_bal)   # PURE gain
    timings["route_gain_s"] = round(time.time() - t0, 1)
    print(f"route pure-gain matrix done "
          f"({timings['route_gain_s']}s)", flush=True)

    adv_rows = []
    for k in range(K):
        for c in range(n_classes):
            m = y_route == c
            if not m.any():
                continue
            adv_rows.append({
                "expert": k + 1, "class": class_names[c], "rows": int(m.sum()),
                "mean_gain": float(G_rt[m, k].mean()),
                "frac_positive": float((G_rt[m, k] > 0).mean())})
    adv_df = pd.DataFrame(adv_rows)

    # ---- §11 diversity diagnostics on D_route --------------------------
    preds_all = [p0_rt.argmax(axis=1)] + [p.argmax(axis=1) for p in pk_rt]
    pred_names = ["global"] + [f"e{k + 1}" for k in range(K)]
    agree_rows = []
    for i in range(len(preds_all)):
        for j in range(i + 1, len(preds_all)):
            agree_rows.append({"a": pred_names[i], "b": pred_names[j],
                               "agreement": round(float(
                                   (preds_all[i] == preds_all[j]).mean()), 4)})
    agree_df = pd.DataFrame(agree_rows)
    gcorr_rows = []
    for i in range(K):
        for j in range(i + 1, K):
            ok = G_rt[:, i].std() > 0 and G_rt[:, j].std() > 0
            gcorr_rows.append({"a": f"e{i + 1}", "b": f"e{j + 1}",
                               "gain_pearson_r": round(float(np.corrcoef(
                                   G_rt[:, i], G_rt[:, j])[0, 1]), 4)
                               if ok else np.nan})
    gcorr_df = pd.DataFrame(gcorr_rows)
    pos_rt = G_rt > 0
    uniq_pos = pos_rt & (pos_rt.sum(axis=1) == 1)[:, None]
    cover_df = pd.DataFrame([{
        "expert": k + 1,
        "frac_positive": round(float(pos_rt[:, k].mean()), 4),
        "frac_unique_positive": round(float(uniq_pos[:, k].mean()), 4),
        "mean_gain_positive": round(float(G_rt[pos_rt[:, k], k].mean()), 4)
        if pos_rt[:, k].any() else np.nan} for k in range(K)])
    print("agreement:\n" + agree_df.to_string(index=False))
    print("gain corr:\n" + gcorr_df.to_string(index=False))
    print("coverage:\n" + cover_df.to_string(index=False), flush=True)

    # ---- P3/P4: pair features on D_route -------------------------------
    embed_stage["tag"] = "route"
    z_rt = phi.transform(X_route)
    d2_rt = sq_dist_to_centroids(sig.observable(z_rt, p0_rt), mu_obs)
    a0_rt = aff0.score(z_rt)
    ak_rt = [aff_k[k].score(z_rt) for k in range(K)]
    hpost_rt = [build_pair_post(p0_rt, pk_rt[k], a0_rt, ak_rt[k],
                                np.sqrt(d2_rt[:, k]), qk_mat[k])
                for k in range(K)]
    hpre_rt = [build_pair_pre(p0_rt, z_rt, np.sqrt(d2_rt[:, k]), a0_rt,
                              qk_mat[k]) for k in range(K)]

    def quantile_verifier(seed):
        return xgb.XGBRegressor(
            n_estimators=args.verifier_n_estimators,
            max_depth=args.verifier_max_depth, learning_rate=args.verifier_lr,
            objective="reg:quantileerror",
            quantile_alpha=args.verifier_quantile,
            n_jobs=-1, random_state=seed)

    # THE KNOB (exp26): the verifier's target unit.  nrm == 1 reproduces
    # exp24b's raw w_bal-weighted-CE gain; nrm == w_bal[y] puts the target --
    # and therefore q_hat, q_corr, g_lower and tau_post -- into plain
    # delta-NLL units (0831.md cause B).  The scorer target below stays raw.
    nrm_rt = gain_norm(y_route, w_bal, args.verifier_target)

    # 2-fold chrono OOF b for the scorer target (guide §15)
    t0 = time.time()
    order_rt = np.argsort(route_ts, kind="stable")
    half = len(order_rt) // 2
    folds = (order_rt[:half], order_rt[half:])
    b_oof = np.zeros((len(y_route), K), dtype=np.float64)
    if min(len(folds[0]), len(folds[1])) >= 100:
        for f_fit, f_pred in ((0, 1), (1, 0)):
            ver_f = quantile_verifier(args.seed + SEED_BAND_OOF + f_fit)
            ver_f.fit(
                np.concatenate([hpost_rt[k][folds[f_fit]]
                                for k in range(K)]),
                np.concatenate([G_rt[folds[f_fit], k]
                                for k in range(K)]).astype(np.float32))
            for k in range(K):
                b_oof[folds[f_pred], k] = (
                    ver_f.predict(hpost_rt[k][folds[f_pred]]) > 0)
    else:
        b_oof[:] = 1.0
        print("route too small for OOF folds -> b_oof=1", flush=True)
    U_rt = b_oof * G_rt - args.lambda_cost * args.expert_cost
    sc_label = np.concatenate([U_rt[:, k] for k in range(K)]) > 0
    if sc_label.min() == sc_label.max():
        raise SystemExit("scorer sign labels degenerate (single class)")
    sc_weight = np.log1p(np.abs(
        np.concatenate([G_rt[:, k] for k in range(K)]))).astype(np.float32)
    scorer = xgb.XGBClassifier(
        n_estimators=args.scorer_n_estimators,
        max_depth=args.scorer_max_depth, learning_rate=args.scorer_lr,
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=-1, random_state=args.seed + SEED_BAND_SCORER)
    scorer.fit(np.concatenate(hpre_rt), sc_label,
               sample_weight=sc_weight + 1e-3)
    print(f"sign-scorer labels: positive {sc_label.mean():.3f}", flush=True)
    verifier = quantile_verifier(args.seed + SEED_BAND_VERIFIER)
    verifier.fit(np.concatenate(hpost_rt),
                 np.concatenate([G_rt[:, k] / nrm_rt for k in range(K)])
                 .astype(np.float32))
    timings["scorer_verifier_s"] = round(time.time() - t0, 1)
    print(f"scorer+verifier trained (§15/§14, verifier target "
          f"'{args.verifier_target}', OOF b positive rate "
          f"{b_oof.mean():.3f}; {timings['scorer_verifier_s']}s)", flush=True)
    del hpost_rt, hpre_rt, z_rt, d2_rt, a0_rt, ak_rt, pk_rt
    gc.collect()

    # ---- diagnostics on D_tune (scorer/verifier quality) ---------------
    # exp26: the verifier's own diagnostics must be read in the verifier's own
    # unit, so gt is divided by the same nrm.  The scorer diagnostics further
    # down keep raw G_t -- the scorer target never left raw units.
    nrm_tu = gain_norm(y_tune, w_bal, args.verifier_target)
    d2_tu = sq_dist_to_centroids(sig.observable(z_tune, p0_tune), mu_obs)
    a0_tu = aff0.score(z_tune)
    U_hat_tu = np.stack(
        [scorer.predict_proba(build_pair_pre(
            p0_tune, z_tune, np.sqrt(d2_tu[:, k]), a0_tu,
            qk_mat[k]))[:, 1] for k in range(K)], axis=1)
    ver_rows = []
    for k in range(K):
        hp = build_pair_post(p0_tune, pk_tune[k], a0_tu,
                             aff_k[k].score(z_tune),
                             np.sqrt(d2_tu[:, k]), qk_mat[k])
        qh = verifier.predict(hp).astype(np.float64)
        gt = G_t[:, k] / nrm_tu
        ver_rows.append({
            "expert": k + 1, "tune_rows": len(gt),
            "target_unit": args.verifier_target,
            "sign_accuracy": round(float(((qh > 0) == (gt > 0)).mean()), 4),
            "false_accept": int(((qh > 0) & (gt <= 0)).sum()),
            "false_reject": int(((qh <= 0) & (gt > 0)).sum()),
            "quantile_coverage": round(float((gt >= qh).mean()), 4),
            "pearson_r": round(float(np.corrcoef(qh, gt)[0, 1]), 4)
            if gt.std() > 0 else np.nan})
    ver_df = pd.DataFrame(ver_rows)
    top1_match = float((U_hat_tu.argmax(axis=1) == G_t.argmax(axis=1)).mean())
    scorer_df = pd.DataFrame([{
        "tune_rows": len(y_tune),
        "top1_matches_best_gain": round(top1_match, 4),
        "u_hat_gain_pearson": round(float(np.corrcoef(
            U_hat_tu.ravel(), G_t.ravel())[0, 1]), 4)
        if G_t.std() > 0 else np.nan}])
    print("verifier tune diag:\n" + ver_df.to_string(index=False))
    print("scorer tune diag:\n" + scorer_df.to_string(index=False), flush=True)
    del z_tune, d2_tu, a0_tu, pk_tune
    gc.collect()

    # ---- P4 §14: one-sided calibration + thresholds on D_cal -----------
    t0 = time.time()
    p0_cal = corr0.correct(batched_proba(glob, X_cal, "global/cal"),
                           0.0, temp)
    embed_stage["tag"] = "cal"
    z_cal = phi.transform(X_cal)
    d2_cal = sq_dist_to_centroids(sig.observable(z_cal, p0_cal), mu_obs)
    a0_cal = aff0.score(z_cal)
    U_hat_cal = np.stack(
        [scorer.predict_proba(build_pair_pre(
            p0_cal, z_cal, np.sqrt(d2_cal[:, k]), a0_cal,
            qk_mat[k]))[:, 1] for k in range(K)], axis=1)
    top1_cal = U_hat_cal.argmax(axis=1)
    u_max_cal = U_hat_cal.max(axis=1)
    L0_cal = balanced_ce(p0_cal, y_cal, w_bal)
    q_hat_cal = np.zeros(len(y_cal), dtype=np.float64)
    g_cal = np.zeros(len(y_cal), dtype=np.float64)
    top1_pred_cal = np.zeros(len(y_cal), dtype=np.int64)
    for k in range(K):
        rows = np.flatnonzero(top1_cal == k)
        if not len(rows):
            continue
        pk = experts[k]["corr"].correct(
            batched_proba(experts[k]["clf"], X_cal[rows], f"e{k + 1}/cal"),
            beta, temp)
        hp = build_pair_post(p0_cal[rows], pk, a0_cal[rows],
                             aff_k[k].score(z_cal[rows]),
                             np.sqrt(d2_cal[rows, k]), qk_mat[k])
        q_hat_cal[rows] = verifier.predict(hp)
        g_cal[rows] = L0_cal[rows] - balanced_ce(pk, y_cal[rows], w_bal)
        top1_pred_cal[rows] = pk.argmax(axis=1)
    # §5 enforcement: correction/thresholds use only cal rows that are NOT
    # content-duplicates of D_route (the scorer/verifier training data)
    msk_c = cal_sel_mask
    # exp26: q_hat now lives in the verifier's target unit, so the one-sided
    # offset has to be fitted against g_cal in that SAME unit or it stops being
    # a valid conservative correction.  Raw g_cal is kept untouched -- it still
    # feeds select_thresholds' net_gain objective and every reported utility.
    nrm_cal = gain_norm(y_cal, w_bal, args.verifier_target)
    g_cal_v = g_cal / nrm_cal
    q_corr = float(np.quantile((q_hat_cal - g_cal_v)[msk_c],
                               1.0 - args.cal_delta))
    g_lower_cal = q_hat_cal - q_corr
    # in-sample by construction ~= 1-delta (q_corr fit on these rows); the
    # informative coverage check is the held-out eval-side record
    lb_coverage = float((g_cal_v[msk_c] >= g_lower_cal[msk_c]).mean())
    names_l = [str(n).lower() for n in class_names]
    benign_id = next((names_l.index(n) for n in ("benign", "normal")
                      if n in names_l), None)
    if benign_id is None:
        print("benign guardrails skipped: no benign/normal class", flush=True)
    y_glob_cal = p0_cal.argmax(axis=1)
    glob_ok_cal = y_glob_cal == y_cal
    top1_ok_cal = top1_pred_cal == y_cal
    bmask_cal = (y_cal == benign_id) if benign_id is not None \
        else np.zeros(len(y_cal), dtype=bool)
    fpr0 = float((y_glob_cal[bmask_cal & msk_c] != benign_id).mean()) \
        if (bmask_cal & msk_c).any() else 0.0
    tau_pre_grid = np.quantile(u_max_cal[msk_c],
                               _parse_floats(args.tau_pre_quantiles))
    chosen, thr_df = select_thresholds(
        u_max_cal[msk_c], g_lower_cal[msk_c], g_cal[msk_c],
        glob_ok_cal[msk_c], top1_ok_cal[msk_c], bmask_cal[msk_c],
        fpr0, tau_pre_grid, tau_post_grid, args.cal_max_proposal,
        args.cal_benign_fpr_increase, args.cal_harmful_frac,
        args.cal_min_accepted, args.cal_min_decided)
    if chosen is None:
        tau_pre, tau_post = np.inf, np.inf
        print("calibration (§14): NO feasible thresholds -> GLOBAL ONLY",
              flush=True)
    else:
        tau_pre, tau_post = chosen
    cal_df = pd.DataFrame([{
        "cal_rows": len(y_cal), "cal_rows_used": int(msk_c.sum()),
        "delta": args.cal_delta,
        "verifier_alpha": args.verifier_quantile,
        "verifier_target": args.verifier_target,
        "q_corr": round(q_corr, 4),
        "lower_bound_coverage_cal_insample": round(lb_coverage, 4),
        "target_coverage": round(1 - args.cal_delta, 4),
        "tau_pre": tau_pre, "tau_post": tau_post,
        "cal_benign_fpr0": round(fpr0, 6)}])
    timings["calibration_s"] = round(time.time() - t0, 1)
    print("calibration (§14):\n" + cal_df.to_string(index=False), flush=True)
    del z_cal, d2_cal, a0_cal, p0_cal
    gc.collect()

    # ---- §16 inference on the development holdout ----------------------
    t0 = time.time()
    p0_eval_raw = batched_proba(glob, X_eval, "global/eval")
    p0_eval = corr0.correct(p0_eval_raw, 0.0, temp)
    y_glob_raw = p0_eval_raw.argmax(axis=1)
    del p0_eval_raw
    timings["eval_global_s"] = round(time.time() - t0, 1)
    y_glob = p0_eval.argmax(axis=1)

    t0 = time.time()
    embed_stage["tag"] = "eval"
    z_eval = phi.transform(X_eval)
    d2_eval = sq_dist_to_centroids(sig.observable(z_eval, p0_eval), mu_obs)
    a0_eval = aff0.score(z_eval)
    U_hat_ev = np.stack(
        [scorer.predict_proba(build_pair_pre(
            p0_eval, z_eval, np.sqrt(d2_eval[:, k]), a0_eval,
            qk_mat[k]))[:, 1] for k in range(K)], axis=1)
    top1_ev = U_hat_ev.argmax(axis=1)
    proposal = np.where(U_hat_ev.max(axis=1) > tau_pre, top1_ev + 1,
                        NO_EXPERT)
    timings["eval_scorer_s"] = round(time.time() - t0, 1)

    L0_eval_bal = balanced_ce(p0_eval, y_eval, w_bal)
    probs_sys = p0_eval.astype(np.float32)
    realized = np.zeros(len(y_eval), dtype=np.float32)
    g_lower_ev = np.zeros(len(y_eval), dtype=np.float32)
    q_hat_ev = np.zeros(len(y_eval), dtype=np.float32)
    final = y_glob.copy()
    accepted = np.zeros(len(y_eval), dtype=bool)
    decision_rows = []
    t0 = time.time()
    for k in range(K):
        rows = np.flatnonzero(proposal == k + 1)
        if len(rows) == 0:
            decision_rows.append({"expert": k + 1, "proposed": 0,
                                  "accepted": 0, "accept_rate": np.nan})
            continue
        pk = experts[k]["corr"].correct(
            batched_proba(experts[k]["clf"], X_eval[rows], f"e{k + 1}/eval"),
            beta, temp)
        hp = build_pair_post(p0_eval[rows], pk, a0_eval[rows],
                             aff_k[k].score(z_eval[rows]),
                             np.sqrt(d2_eval[rows, k]), qk_mat[k])
        qh = verifier.predict(hp).astype(np.float32)
        q_hat_ev[rows] = qh                     # pre-correction verifier output
        gl = (qh - q_corr).astype(np.float32)
        g_lower_ev[rows] = gl
        acc = gl > tau_post
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
    print(f"\nrho={rho:.4f} (accepted {accept_rho:.4f}) => mean TFM "
          f"calls/sample = {1 + rho:.3f}", flush=True)

    # ---- §22 quality / transitions / FP accounting ---------------------
    glob_ok = y_glob == y_eval
    final_ok = final == y_eval
    n_helpful = int((~glob_ok & final_ok).sum())
    n_harmful = int((glob_ok & ~final_ok).sum())
    called = proposal != 0
    n_called, n_acc = int(called.sum()), int(accepted.sum())
    quality = {
        "verifier_target": args.verifier_target,
        "helpful_override": n_helpful, "harmful_override": n_harmful,
        "net_correction": n_helpful - n_harmful,
        "override_precision": round(n_helpful / max(n_helpful + n_harmful, 1),
                                    4),
        "activation_rate": round(rho, 4),
        "accepted_rate": round(accept_rho, 4),
        "eval_rows": len(y_eval),
        "override_among_calls": round(n_acc / max(n_called, 1), 4),
        "rejected_calls": n_called - n_acc,
        "rejected_call_rate": round((n_called - n_acc) / max(n_called, 1), 4),
        "mean_realized_gain_calls": round(float(realized[called].mean()), 4)
        if n_called else np.nan,
        "mean_realized_gain_overrides":
        round(float(realized[accepted].mean()), 4) if n_acc else np.nan,
        "prior_beta": beta, "prior_T": temp, "K_selected": K,
        "tau_pre": tau_pre, "tau_post": tau_post,
        "block_jaccard_max": float(jaccard_df["jaccard"].max())
        if len(jaccard_df) else np.nan,
        "gain_corr_max": float(gcorr_df["gain_pearson_r"].max())
        if len(gcorr_df) else np.nan,
    }
    if benign_id is not None:
        bmask = y_eval == benign_id
        quality.update({
            "benign_rows": int(bmask.sum()),
            "benign_harmful_override": int((bmask & glob_ok & ~final_ok).sum()),
        })
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
    print("\n=== correctness transitions (§22) ===")
    print(trans_df.to_string(index=False))
    print("\n=== FP/TP accounting ===")
    print(fp_df.to_string(index=False), flush=True)

    scen_recall_df = None                       # exp30: 6g test scenario recall
    if scen_all is not None:
        scen_eval = scen_all[eval_idx]
        preds_named = [("global", y_glob), ("system", final)]
        if y_pred_xgb is not None:
            preds_named.append(("xgboost", y_pred_xgb))
        srows = []
        for c in range(n_classes):
            cm = y_eval == c
            for sname in sorted(np.unique(scen_eval[cm])):
                m = cm & (scen_eval == sname)
                row = {"class": class_names[c], "scenario": sname,
                       "rows": int(m.sum())}
                for nm, pr in preds_named:
                    row[f"recall_{nm}"] = round(float((pr[m] == c).mean()), 6)
                    if c != benign_id:
                        row[f"to_benign_{nm}"] = int((pr[m] == benign_id).sum())
                srows.append(row)
        scen_recall_df = pd.DataFrame(srows)
        print("\n=== test per-scenario recall (exp30 6g) ===")
        print(scen_recall_df.to_string(index=False), flush=True)

    all_rows = list(core.per_class_table("racepfn_system", y_eval, final,
                                         class_names, tail_classes))
    all_rows.extend(core.per_class_table("global_tabpfn", y_eval, y_glob,
                                         class_names, tail_classes))
    all_rows.extend(core.per_class_table("global_raw", y_eval, y_glob_raw,
                                         class_names, tail_classes))
    if y_pred_xgb is not None:
        all_rows.extend(core.per_class_table("xgboost", y_eval, y_pred_xgb,
                                             class_names, tail_classes))

    # ---- dense oracle + Gate-1 banks + §21 routing baselines -----------
    dense_note = "skipped (--dense-eval off)"
    bank_blocks = {}
    rb_df = pd.DataFrame()
    if args.dense_eval:
        t0 = time.time()
        L_eval = np.zeros((len(y_eval), K + 1), dtype=np.float64)
        L_eval[:, 0] = L0_eval_bal
        dense_preds = [y_glob]
        for k in range(K):
            pk = experts[k]["corr"].correct(
                batched_proba(experts[k]["clf"], X_eval, f"e{k + 1}/dense"),
                beta, temp)
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

        # §21 routing baselines at zero extra proba cost (6f)
        pred_mat = np.stack(dense_preds[1:], axis=1)
        y_nc = np.take_along_axis(pred_mat,
                                  d2_eval.argmin(axis=1)[:, None],
                                  axis=1)[:, 0]
        y_t1 = np.take_along_axis(pred_mat, top1_ev[:, None], axis=1)[:, 0]
        all_rows.extend(core.per_class_table("route_nearest_centroid",
                                             y_eval, y_nc, class_names,
                                             tail_classes))
        all_rows.extend(core.per_class_table("route_top1_noverify", y_eval,
                                             y_t1, class_names, tail_classes))
        rb_df = pd.DataFrame([
            {"method": name,
             "macro_f1": round(float(f1_score(y_eval, yp, average="macro")),
                               4),
             "agreement_with_system": round(float((yp == final).mean()), 4)}
            for name, yp in (("route_nearest_centroid", y_nc),
                             ("route_top1_noverify", y_t1))])
        timings["dense_eval_s"] = round(time.time() - t0, 1)
        dense_note = "done"

        bank_identical = {b: [] for b in feas_banks}
        for bank_name in feas_banks:
            t0 = time.time()
            b_preds, blocks_b = [y_glob], []
            L_b = np.zeros((len(y_eval), K + 1), dtype=np.float64)
            L_b[:, 0] = L_eval[:, 0]
            for k in range(K):
                # THE KNOB (exp28): 'per_expert' sizes control block k to the
                # residual expert k block it is compared against, instead of a
                # flat args.expert_block_rows for every block (which was
                # 5x186,000 = 930,000 rows against 228,638 residual rows, a
                # 4.07x memory violation of guide 20.1). 'off' + --feasibility-banks random,proximity = exp24b.
                n_b = (len(experts[k]["block_rows"])
                       if args.feas_match_memory == "per_expert"
                       else args.expert_block_rows)
                if bank_name == "random":
                    rows_b = np.sort(np.random.default_rng(
                        args.seed + SEED_BAND_FEASIBILITY + k).choice(
                            len(X_exp),
                            size=min(n_b, len(X_exp)),
                            replace=False))
                elif bank_name == "regime_random":
                    # exp28: uniform draw from WITHIN residual regime k (the
                    # rows hard-assigned to it). Separates "which rows the
                    # diversity selector picked" from "how many rows, and from
                    # which regime" -- random/proximity cannot separate those.
                    memb_b = np.flatnonzero(assign_exp == keep[k])
                    rows_b = np.sort(np.random.default_rng(
                        args.seed + SEED_BAND_REGIME_RANDOM + k).choice(
                            memb_b,
                            size=min(n_b, len(memb_b)),
                            replace=False))
                else:
                    d2z = ((e_exp[:, :z_dim] - mu[k, :z_dim][None, :]) ** 2
                           ).sum(axis=1)
                    rows_b = np.sort(np.argsort(d2z, kind="stable")[: n_b])
                blocks_b.append(rows_b)
                # exp28 guard: diversity_select returns arange(n) whenever the
                # regime is no larger than the block budget, so for those
                # experts regime_random draws the WHOLE regime and is byte-
                # identical to the residual block. Refitting it would burn a
                # full TabPFN fit + 4M-row inference pass to reproduce a curve
                # we already have, and would make the control look informative
                # when it carries no information. Reuse and record instead.
                same_as_residual = np.array_equal(
                    rows_b, np.asarray(experts[k]["block_rows"]))
                bank_identical[bank_name].append(bool(same_as_residual))
                if same_as_residual:
                    L_b[:, k + 1] = L_eval[:, k + 1]
                    b_preds.append(dense_preds[k + 1])
                    print(f"  {bank_name}{k + 1}: identical to residual "
                          f"block ({len(rows_b):,} rows) -- reused, not refit",
                          flush=True)
                    continue
                yb = np.concatenate([anchor[1], y_exp[rows_b]])
                clf_b = make_clf(np.concatenate([anchor[0], X_exp[rows_b]]),
                                 yb)
                corr_b = PriorCorrector(yb, n_classes, ref_prior,
                                        args.prior_alpha)
                pb = corr_b.correct(
                    batched_proba(clf_b, X_eval, f"{bank_name}{k + 1}/dense"),
                    beta, temp)
                L_b[:, k + 1] = (balanced_ce(pb, y_eval, w_bal)
                                 + args.lambda_cost * args.expert_cost)
                b_preds.append(pb.argmax(axis=1))
                del clf_b
                free_gpu()
            pick_b = L_b.argmin(axis=1)
            y_ob = np.take_along_axis(np.stack(b_preds, axis=1),
                                      pick_b[:, None], axis=1)[:, 0]
            all_rows.extend(core.per_class_table(
                f"oracle_{bank_name}_bank", y_eval, y_ob, class_names,
                tail_classes))
            bank_blocks[bank_name] = blocks_b
            timings[f"feasibility_{bank_name}_s"] = round(time.time() - t0, 1)
            print(f"feasibility bank '{bank_name}' done", flush=True)
    if feas_banks and not args.dense_eval:
        print("feasibility banks skipped (--dense-eval off)", flush=True)

    # ---- 2f (exp28): per-bank, per-expert context block sizes ----------
    # The audit that makes the 20.1 equal-memory claim checkable after the
    # fact: every control bank row must match its residual row's block_rows.
    bank_size_rows = [{"bank": "residual", "expert": k + 1,
                       "block_rows": int(len(experts[k]["block_rows"])),
                       "residual_regime_rows": int(experts[k]["regime_rows"]),
                       "identical_to_residual": True,
                       "selector_active": bool(
                           experts[k]["regime_rows"]
                           > len(experts[k]["block_rows"]))}
                      for k in range(K)]
    for bank_name, blocks in bank_blocks.items():
        for k, rows_b in enumerate(blocks):
            bank_size_rows.append({
                "bank": bank_name, "expert": k + 1,
                "block_rows": int(len(rows_b)),
                "residual_regime_rows": int(experts[k]["regime_rows"]),
                "identical_to_residual": bool(
                    bank_identical.get(bank_name, [False] * K)[k]),
                "selector_active": bool(
                    experts[k]["regime_rows"]
                    > len(experts[k]["block_rows"]))})
    bank_size_df = pd.DataFrame(bank_size_rows)
    bank_totals = bank_size_df.groupby("bank")["block_rows"].sum()
    res_total = int(bank_totals.get("residual", 0))
    timings["feas_match_memory"] = args.feas_match_memory
    timings["bank_total_block_rows"] = {b: int(v)
                                        for b, v in bank_totals.items()}
    timings["bank_memory_ratio_vs_residual"] = {
        b: round(float(v) / max(res_total, 1), 4)
        for b, v in bank_totals.items()}
    print("\n=== context block sizes per bank (2f) ===")
    print(bank_size_df.to_string(index=False))
    print(f"total block rows per bank: {timings['bank_total_block_rows']} "
          f"| ratio vs residual: "
          f"{timings['bank_memory_ratio_vs_residual']}", flush=True)

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
    for mname in (["dense_oracle", "route_nearest_centroid",
                   "route_top1_noverify"]
                  + [f"oracle_{b}_bank" for b in feas_banks]):
        if mname in piv.columns:
            quality[f"{mname}_macro"] = round(f1_of(mname), 4)

    # ---- guardrails (§20.2) --------------------------------------------
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
    print("\n=== guardrails (§20.2) ===")
    print(guard_df.to_string(index=False), flush=True)
    quality_df = pd.DataFrame([quality])
    print("\n=== routing quality ===")
    print(quality_df.to_string(index=False))
    timings["total_s"] = round(time.time() - t_all, 1)
    timings["dense_eval"] = dense_note
    timings["rho"] = round(rho, 4)

    # ---- artifacts ------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp30_c0share")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pool_audit.to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    manifest_df.to_csv(os.path.join(out_dir, "0c_split_manifest.csv"),
                       index=False)
    if scen_audit is not None:      # exp27: per-scenario pool coverage
        scen_audit.to_csv(
            os.path.join(out_dir, "0e_pool_partition_scenario.csv"),
            index=False)
    if val_scen_audit is not None:  # exp27: per-scenario tune/cal coverage
        val_scen_audit.to_csv(
            os.path.join(out_dir, "0f_val_split_scenario.csv"), index=False)
    if c0_scen_df is not None:      # exp30: realized C0 (class, scenario)
        c0_scen_df.to_csv(os.path.join(out_dir, "0g_c0_scenario.csv"),
                          index=False)
    if scen_recall_df is not None:  # exp30: test per-scenario recall
        scen_recall_df.to_csv(
            os.path.join(out_dir, "6g_test_scenario_recall.csv"), index=False)
    dupmat_df.to_csv(os.path.join(out_dir, "0d_dup_hash_pairwise.csv"),
                     index=False)
    residual_stats.to_csv(os.path.join(out_dir, "1a_residual_stats.csv"),
                          index=False)
    cluster_df.to_csv(os.path.join(out_dir, "1b_cluster_composition.csv"),
                      index=False)
    prior_grid_df.to_csv(os.path.join(out_dir, "1c_prior_grid.csv"),
                         index=False)
    prior_expert_df.to_csv(os.path.join(out_dir, "1d_prior_expert_grid.csv"),
                           index=False)
    expert_ctx_df.to_csv(os.path.join(out_dir, "2a_expert_contexts.csv"),
                         index=False)
    jaccard_df.to_csv(os.path.join(out_dir, "2b_expert_jaccard.csv"),
                      index=False)
    prior_df.to_csv(os.path.join(out_dir, "2c_context_priors.csv"),
                    index=False)
    ksel_df.to_csv(os.path.join(out_dir, "2d_k_selection.csv"), index=False)
    prune_df.to_csv(os.path.join(out_dir, "2e_pruning.csv"), index=False)
    bank_size_df.to_csv(os.path.join(out_dir, "2f_bank_block_sizes.csv"),
                        index=False)
    adv_df.to_csv(os.path.join(out_dir, "3a_gain_stats.csv"), index=False)
    agree_df.to_csv(os.path.join(out_dir, "3c_expert_agreement.csv"),
                    index=False)
    gcorr_df.to_csv(os.path.join(out_dir, "3d_gain_correlation.csv"),
                    index=False)
    cover_df.to_csv(os.path.join(out_dir, "3e_gain_coverage.csv"), index=False)
    scorer_df.to_csv(os.path.join(out_dir, "4a_scorer_tune_diag.csv"),
                     index=False)
    ver_df.to_csv(os.path.join(out_dir, "4b_verifier_tune_diag.csv"),
                  index=False)
    thr_df["target_unit"] = args.verifier_target   # tau_post unit stamp
    thr_df.to_csv(os.path.join(out_dir, "4c_threshold_grid.csv"), index=False)
    cal_df.to_csv(os.path.join(out_dir, "4d_calibration.csv"), index=False)
    decision_df.to_csv(os.path.join(out_dir, "5a_decision_stats.csv"),
                       index=False)
    act_df.to_csv(os.path.join(out_dir, "5b_activation_by_class.csv"),
                  index=False)
    quality_df.to_csv(os.path.join(out_dir, "6a_routing_quality.csv"),
                      index=False)
    override_df.to_csv(os.path.join(out_dir, "6b_override_by_expert.csv"),
                       index=False)
    trans_df.to_csv(os.path.join(out_dir, "6c_correctness_transition.csv"),
                    index=False)
    fp_df.to_csv(os.path.join(out_dir, "6d_class_fp_accounting.csv"),
                 index=False)
    guard_df.to_csv(os.path.join(out_dir, "6e_guardrails.csv"), index=False)
    if len(rb_df):
        rb_df.to_csv(os.path.join(out_dir, "6f_routing_baselines.csv"),
                     index=False)
    np.savez_compressed(
        os.path.join(out_dir, "system_dump.npz"),
        proposal=proposal.astype(np.int8), accepted=accepted,
        g_lower=g_lower_ev, q_hat=q_hat_ev, final=final.astype(np.int64),
        y_glob=y_glob.astype(np.int64), y_glob_raw=y_glob_raw.astype(np.int64),
        y_true=y_eval.astype(np.int64), mu=mu,
        obs_dim=np.int64(sig.obs_dim), d2_med=np.float32(d2_med),
        r_max=np.float64(r_max), qk=qk_mat,
        prior_beta=np.float64(beta), prior_T=np.float64(temp),
        tau_pre=np.float64(tau_pre), tau_post=np.float64(tau_post),
        q_corr=np.float64(q_corr),
        verifier_target=np.asarray(args.verifier_target),
        sig_mean=np.concatenate([sig.stats[n][0] for n in sig.BLOCKS]),
        sig_std=np.concatenate([sig.stats[n][1] for n in sig.BLOCKS]),
        sig_alpha=np.asarray([sig.alpha[n] for n in sig.BLOCKS],
                             dtype=np.float64),
        class_names=np.asarray(class_names))
    np.savez_compressed(os.path.join(out_dir, "probs_tabpfn_global.npz"),
                        probs=p0_eval.astype(np.float32),
                        y_true=y_eval.astype(np.int64))
    if y_proba_xgb is not None:
        np.savez_compressed(os.path.join(out_dir, "probs_xgboost.npz"),
                            probs=y_proba_xgb.astype(np.float32),
                            y_true=y_eval.astype(np.int64))
    np.savez_compressed(os.path.join(out_dir, "probs_racepfn_system.npz"),
                        probs=probs_sys, y_true=y_eval.astype(np.int64))
    ctx_dump = {"C0": g_idx, "anchor": anchor_idx, "phi_fit": phi_fit_idx,
                "expert": exp_idx, "route": route_idx, "tune": tune_idx,
                "cal": cal_idx, "eval": eval_idx}
    for k in range(K):
        ctx_dump[f"expert{k + 1}_block"] = exp_idx[experts[k]["block_rows"]]
    for bank_name, blocks in bank_blocks.items():
        for k, rows_b in enumerate(blocks):
            ctx_dump[f"{bank_name}{k + 1}_block"] = exp_idx[rows_b]
    np.savez_compressed(os.path.join(out_dir, "context_rows.npz"), **ctx_dump)
    np.savez_compressed(os.path.join(out_dir, "mining_dump.npz"),
                        expert_idx=exp_idx,
                        assign=assign_exp.astype(np.int8),
                        kept=np.asarray(keep, dtype=np.int64),
                        r=r_exp.astype(np.float32),
                        r_bar=r_bar.astype(np.float32))
    gain_f32 = np.ascontiguousarray(G_rt.astype(np.float32))
    np.savez_compressed(os.path.join(out_dir, "route_gain.npz"),
                        route_idx=route_idx, gain=gain_f32,
                        b_oof=b_oof.astype(np.int8),
                        y=y_route.astype(np.int32))
    timings["gain_matrix_sha256"] = hashlib.sha256(
        gain_f32.tobytes()).hexdigest()
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
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        dirty = int(bool(dirty))
    except Exception:
        dirty = -1
    timings["env"] = {
        "cmd": " ".join(sys.argv), "git_commit": commit, "git_dirty": dirty,
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
        render_dir(out_dir)
    except Exception as exc:
        print(f"PNG rendering failed (CSVs intact): {exc}")
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-frac", type=float, default=0.50)
    p.add_argument("--expert-frac", type=float, default=0.25,
                   help="D_expert fraction of train (§18 rename of exp22c's "
                        "--mine-frac)")
    p.add_argument("--tune-frac-of-val", type=float, default=0.40,
                   help="§5: first (earlier) fraction of val -> D_tune; "
                        "rest (later) -> D_cal")
    p.add_argument("--global-context-size", type=int, default=1_000_000)
    p.add_argument("--context-selection", default="random",
                   choices=["random", "medoid"])
    p.add_argument("--anchor-per-class", type=int, default=2_000)
    p.add_argument("--c0-benign-share", type=float, default=-1.0,
                   help="THE KNOB (exp30): benign fraction of the C0 budget; "
                        "-1 = natural class ratios (exp29). Remaining budget "
                        "goes to the other classes by natural ratio.")
    p.add_argument("--expert-block-rows", type=int, default=186_000)
    p.add_argument("--k-candidates", default="2,4,8",
                   help="§10: K grid selected on D_tune by oracle macro")
    p.add_argument("--phi-mode", default="embed_pca",
                   choices=["embed_pca", "quantile_pca", "raw_pca"])
    p.add_argument("--embed-chunk", type=int, default=100_000)
    p.add_argument("--phi-dim", type=int, default=16)
    p.add_argument("--phi-fit-rows", type=int, default=500_000)
    p.add_argument("--residual-clip-q", type=float, default=0.995)
    p.add_argument("--sig-alpha-p", type=float, default=1.0)
    p.add_argument("--sig-alpha-e", type=float, default=1.0)
    p.add_argument("--sig-alpha-r", type=float, default=1.0)
    p.add_argument("--diversity-subclusters", type=int, default=2_048)
    p.add_argument("--diversity-n-init", type=int, default=3)
    p.add_argument("--diversity-batch-size", type=int, default=10_000)
    p.add_argument("--feasibility-banks",
                   default="random,proximity,regime_random",
                   help="20.1 control banks. 'regime_random' (exp28) draws "
                        "uniformly from within residual regime k, so it "
                        "isolates the diversity selector from the regime "
                        "partition itself")
    p.add_argument("--feas-match-memory", choices=["off", "per_expert"],
                   default="per_expert",
                   help="THE KNOB (exp28). 'off' + --feasibility-banks "
                        "random,proximity reproduces exp24b exactly ('off' "
                        "alone still runs the third bank, since the "
                        "--feasibility-banks default also changed): every "
                        "control block takes --expert-block-rows rows. "
                        "'per_expert' makes control block k draw exactly "
                        "len(experts[k]['block_rows']) rows, so each control "
                        "is equal-memory with the residual expert it is "
                        "judged against (guide 20.1), not merely equal in "
                        "total")
    p.add_argument("--prior-alpha", type=float, default=1.0,
                   help="§7 smoothing alpha for context priors")
    p.add_argument("--prior-betas", default="0,0.5,1.0",
                   help="§7 beta grid on D_tune (0 = no-correction ablation)")
    p.add_argument("--prior-temps", default="0.8,1.0,1.25",
                   help="§7 temperature grid on D_tune")
    p.add_argument("--prune-min-coverage", type=float, default=0.005,
                   help="§12: min tune positive-gain coverage to keep")
    p.add_argument("--prune-min-regime-gain", type=float, default=0.01,
                   help="THE KNOB §12: min mean balanced-loss gain on the "
                        "expert's own regime tune rows (anchor cancels)")
    p.add_argument("--prune-min-regime-rows", type=int, default=50,
                   help="THE KNOB §12: min masked tune rows in the regime "
                        "for the marginal to be measurable (min-support)")
    p.add_argument("--prune-mode", default="regime",
                   choices=["regime", "off"],
                   help="'off' keeps every coverage-viable expert "
                        "(ablation control run)")
    p.add_argument("--kmeans-max-rows", type=int, default=1_000_000)
    p.add_argument("--kmeans-n-init", type=int, default=5)
    p.add_argument("--residual-gamma", type=float, default=1.0)
    p.add_argument("--expert-max-rows", type=int, default=2_000_000,
                   help="§18 rename of exp22c's --mine-max-rows")
    p.add_argument("--route-cap-per-class", type=int, default=100_000)
    p.add_argument("--tune-cap-per-class", type=int, default=50_000)
    p.add_argument("--cal-cap-per-class", type=int, default=50_000)
    p.add_argument("--lambda-cost", type=float, default=0.0,
                   help="§15: cost weight in the SCORER target only (the "
                        "verifier target is pure gain, §14/§18)")
    p.add_argument("--expert-cost", type=float, default=1.0)
    p.add_argument("--affinity-ref-rows", type=int, default=4_096)
    p.add_argument("--affinity-nn", type=int, default=16)
    p.add_argument("--scorer-n-estimators", type=int, default=300)
    p.add_argument("--scorer-max-depth", type=int, default=6)
    p.add_argument("--scorer-lr", type=float, default=0.1)
    p.add_argument("--verifier-n-estimators", type=int, default=400)
    p.add_argument("--verifier-max-depth", type=int, default=6)
    p.add_argument("--verifier-lr", type=float, default=0.05)
    p.add_argument("--verifier-quantile", type=float, default=0.25,
                   help="§14: lower conditional quantile alpha")
    p.add_argument("--verifier-target", default="normgain",
                   choices=["gain", "normgain"],
                   help="THE KNOB (exp26). 'gain' reproduces exp24b exactly: "
                        "the verifier predicts a low quantile of the raw "
                        "w_bal-weighted-CE gain G. 'normgain' predicts the "
                        "same quantile of G / w_bal(y) -- plain delta-NLL -- "
                        "so the target, q_corr, g_lower and tau_post are all "
                        "free of the per-class scale (0831.md cause B: "
                        "q_corr=21.145 in weighted-CE units acts as a "
                        "per-class confidence filter that makes ddos and "
                        "benign acceptance structurally impossible)")
    p.add_argument("--cal-delta", type=float, default=0.10,
                   help="§14: one-sided correction level on D_cal")
    p.add_argument("--cal-benign-fpr-increase", type=float, default=5e-4,
                   help="§14 constraint: max benign FPR added on D_cal")
    p.add_argument("--cal-harmful-frac", type=float, default=0.10,
                   help="§14 constraint: max harmful/(helpful+harmful)")
    p.add_argument("--cal-max-proposal", type=float, default=0.35,
                   help="§14/§15 constraint: max proposal rate on D_cal")
    p.add_argument("--cal-min-accepted", type=int, default=200,
                   help="§14 constraint: min accepted rows on D_cal for a "
                        "threshold cell to be feasible")
    p.add_argument("--cal-min-decided", type=int, default=30,
                   help="§14 constraint: min helpful+harmful rows (the "
                        "harm_frac denominator) for feasibility")
    p.add_argument("--tau-pre-quantiles",
                   default="0.0,0.3,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
                   help="tau_pre grid = these quantiles of U_hat_max on D_cal "
                        "(dense near the top: the smoke showed the proposal "
                        "rate cliff sits between q0.8 and q0.95)")
    p.add_argument("--tau-post-grid", default="0,0.25,0.5,1,2,4,8",
                   help="tau_post grid (>=0, guide §14)")
    p.add_argument("--dense-eval", action="store_true")
    p.add_argument("--pool-partition",
                   choices=["chrono", "scenario_stratified"],
                   default="scenario_stratified",
                   help="THE KNOB (exp27). 'chrono' reproduces exp24b's "
                        "class-internal chronological 50/25/25; "
                        "'scenario_stratified' applies the same chrono split "
                        "within each (class, attack_scenario) group so every "
                        "scenario reaches D_global/C0 (0831.md cause A)")
    p.add_argument("--det-precision", choices=["auto", "float32"],
                   default="auto",
                   help="exp25b: precision under --deterministic. float32 "
                        "needs >24GB (run3 OOM); auto keeps bf16 autocast "
                        "with deterministic kernels only")
    p.add_argument("--deterministic", action="store_true",
                   help="opt-in determinism (deterministic kernels, tf32 off); precision governed by --det-precision"
                        "kernels; use once E-2 confirms it collapses the band")
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
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_exp29(args)


if __name__ == "__main__":
    main()

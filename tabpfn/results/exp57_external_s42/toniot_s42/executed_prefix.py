#!/usr/bin/env python3
"""EXP31 -- C0 ALLOCATION: exp30 위에 컨텍스트 구성 knob 둘 추가 (run당 하나만 바꿀 것).

    --c0-attack-alloc {natural,balanced}  benign 외 예산을 공격 클래스에 자연 비율(exp30)
                                          또는 클래스 균등(가용량 상한, 부족분 재배분)으로 배분.
    --c0-dedup {none,exact}               C0 추출 풀에서 동일 (feature 벡터, 라벨) 중복을 제거한
                                          뒤 추출 -- 같은 예산으로 distinct 커버리지를 늘린다.
                                          (NetFlow 행은 시간에 걸쳐 verbatim 반복; 0d 감사표 참조)

기본값(natural, none)이면 exp30과 코드 경로가 동일하다.  --c0-benign-share -1 이면서
balanced 이면 benign 은 (중복 제거 후) 풀의 자연 비율, 나머지만 균등.

레인 (250k, exp30 §8 규칙과 동일; hoic·benign·inf·web 열만 판독):
    python tabpfn/nfv3_v3_exp31_c0alloc.py --target-dataset cic2018 --prune-mode off \
        --global-context-size 250000 --feasibility-banks none --skip-xgboost \
        --c0-benign-share 0.75 --c0-attack-alloc balanced      # knob A
    ... --c0-benign-share 0.75 --c0-dedup exact                 # knob B
    ... --c0-benign-share -1   --c0-dedup exact                 # knob B on natural
smoke: --target-dataset cic2018_capped --global-context-size 20000 + 위 인자.
    --c0-conflict {keep,majority}         (0903 00:20, 첫 cic2018 run 전 추가) 같은 feature 벡터에
                                          여러 라벨이 붙은 행(라벨 충돌)을 풀에서 다수 라벨만 남기고 제거.
                                          근거: 250k S=0.60 run에서 ftp_bruteforce 77,344 + slowhttptest
                                          21,110 = 동일 벡터 98,454행이 전부 infiltration으로 판정(S=0.75는
                                          brute, 250k 자연은 dos) — 한 벡터가 brute F1·dos F1·inf precision을
                                          추첨에 따라 뒤집는다. dedup 과 독립 (dedup 뒤에 적용).

base = exp30 사본. 추가: representative_subset_share(alloc=), 풀 dedup 블록, 라벨 충돌 다수결 블록,
timings.c0_pool_rows / c0_pool_distinct / c0_conflict_*.
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
                                benign_id, benign_share, alloc="natural"):
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
        if alloc == "balanced":                      # THE KNOB (exp31 A)
            raw = np.full(len(sub), rem / len(sub), dtype=np.float64)
        else:
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


def admissible_expert_counts(candidates, n_classes):
    """Only fit banks with fewer experts than modeled classes (global excluded)."""
    candidates = sorted(set(candidates))
    if not candidates or any(k < 1 for k in candidates):
        raise ValueError("--k-candidates must list positive ints")
    allowed = [k for k in candidates if k < n_classes]
    if not allowed:
        raise ValueError(f"No admissible expert count: require 1 <= K < C={n_classes}; "
                         f"requested {candidates}")
    return allowed


# ---------------------------------------------------------------------------

def run_exp29(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp31_c0alloc"
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
    requested_k_candidates = k_candidates
    k_candidates = admissible_expert_counts(k_candidates, n_classes)
    args.k_candidates_requested = ','.join(map(str, requested_k_candidates))
    args.k_candidates = ','.join(map(str, k_candidates))
    args.expert_count_constraint = '1 <= K < number of modeled classes; global excluded'
    print(f"Expert count constraint: K < C={n_classes}; "
          f"requested={requested_k_candidates}, effective={k_candidates}", flush=True)
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
    draw_pool, draw_y = ctx_pool, y_ctx_pool
    timings["c0_pool_rows"] = int(len(ctx_pool))
    if args.c0_dedup == "exact":                        # THE KNOB (exp31 B)
        t0 = time.time()
        Xp = feats_of(ctx_pool)
        hp = pd.util.hash_pandas_object(pd.DataFrame(Xp), index=False).values
        del Xp
        key = (hp.astype(np.uint64)
               ^ (y_ctx_pool.astype(np.uint64)
                  * np.uint64(0x9E3779B97F4A7C15)))
        _, first = np.unique(key, return_index=True)
        keep = np.sort(first)
        draw_pool, draw_y = ctx_pool[keep], y_ctx_pool[keep]
        timings["c0_pool_distinct"] = int(len(keep))
        timings["c0_dedup_s"] = round(time.time() - t0, 1)
        print(f"C0 pool dedup (vector,label): {len(ctx_pool):,} -> "
              f"{len(keep):,} distinct ({timings['c0_dedup_s']}s)",
              flush=True)
        del hp, key, first
    if args.c0_conflict == "majority":                  # THE KNOB (exp31 C)
        t0 = time.time()
        Xp = feats_of(draw_pool)
        hv = pd.util.hash_pandas_object(pd.DataFrame(Xp), index=False).values
        del Xp
        df_c = pd.DataFrame({"h": hv, "y": draw_y})
        cnt = df_c.groupby(["h", "y"]).size().rename("n").reset_index()
        nlab = cnt.groupby("h")["y"].transform("size")
        conf = cnt[nlab > 1]
        # majority label per conflicting vector; ties -> lowest class id
        maj = (conf.sort_values(["h", "n", "y"], ascending=[True, False, True])
               .drop_duplicates("h").set_index("h")["y"])
        is_conf = df_c["h"].isin(maj.index).values
        keep = ~is_conf | (df_c["y"].values == maj.reindex(df_c["h"]).values)
        timings["c0_conflict_vectors"] = int(len(maj))
        timings["c0_conflict_rows_dropped"] = int((~keep).sum())
        timings["c0_conflict_s"] = round(time.time() - t0, 1)
        print(f"C0 pool label-conflict majority: {len(maj):,} vectors with >1 "
              f"label, dropped {int((~keep).sum()):,} minority-label rows "
              f"({timings['c0_conflict_s']}s)", flush=True)
        draw_pool, draw_y = draw_pool[keep], draw_y[keep]
        del hv, df_c, cnt, nlab, conf, maj, is_conf, keep
    share = args.c0_benign_share
    if share < 0 and args.c0_attack_alloc == "balanced":
        share = float((draw_y == benign_id0).mean())     # natural benign share
    if share >= 0:                                      # THE KNOB (exp30)
        if benign_id0 is None:
            raise SystemExit("--c0-benign-share needs a benign/normal class")
        if share > 1:
            raise SystemExit("--c0-benign-share must be in [0, 1] or -1")
        g_idx = representative_subset_share(
            draw_pool, draw_y, n_classes, args.global_context_size,
            args.seed + SEED_BAND_GLOBAL, benign_id0, share,
            alloc=args.c0_attack_alloc)
    else:
        g_idx = representative_subset(draw_pool, draw_y, n_classes,
                                      args.global_context_size,
                                      args.seed + SEED_BAND_GLOBAL)
    del draw_pool, draw_y
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

    return _exp57.run_bank_diagnostics(locals())

#!/usr/bin/env python3
"""EXP6 -- Context Recipe Lab: evidence-changing lightweight contexts.

Pre-registration: manuscript/report/0820.md SS9.  Forked from
nfv3_v3_exp5_energy_view_moe.py (predict/dump/resume machinery copied verbatim
where noted; exp5 stays frozen).  nfv3_v3_common is import-only.

Why this exists (D1's lesson, 0820 SS8c/SS8d)
---------------------------------------------
Mixture-only views are ~equivalent to a prior shift (measured 99.2% argmax
agreement after prior alignment), so tilting alone cannot move the P/R curve.
EXP6 recipes change the EVIDENCE in the context instead:
  * hard negatives  -- benign rows nearest (pool-standardized feature space)
    to a target attack class: boundary examples the FM never sees under
    random sampling.  Mined POOL-ONLY, model-free (F3-safe).
  * confusion pairs -- dos+ddos concentrated in one context.
  * class focus     -- a weak class at high share via a SMALL k (never by
    duplication; label-space invariant: every present class >= 1 row).
Each evidence change ships with a one-knob control (web_hn vs web_ctrl differ
ONLY in hard-neg vs random benign).  One recipe per run; the combination lab
(ensembling with D1's dumped views, gamma correction, val-selected) is
offline post-hoc over the dumps.

Recipes (k=50,000 default; exact realized composition -> composition CSV)
-------------------------------------------------------------------------
  bal50k    alpha=0 powerlaw (miniaturization control vs D1 va000)
  web_hn    web_attacks full pool (1,522) + 20k benign hard-negs (web-kNN)
            + 8k random benign + proportional rest
  web_ctrl  same, hard-negs replaced by 20k random benign (single knob)
  inf_hn    20k infiltration + 15k benign hard-negs (inf-kNN, 5k seeds)
            + 5k random benign + proportional rest
  dos_pair  17.5k dos + 17.5k ddos + 10k benign + proportional rest

    python tabpfn/nfv3_v3_exp6_context_recipes.py --target-dataset cic2018 \\
        --recipe web_hn --test-cap-per-class 0 --fit-mode fit_with_cache \\
        --test-batch-size 500000                      # test eval
    ... --eval-split val                              # near-future eval

Cost (RTX 4090, cic2018 uncapped 4.02M eval rows, k=50k, m=1)
-------------------------------------------------------------
  fit ~1 min; predict ~5 min (test-side attention scales with k); pkl load
  dominates (~8 min).  ALWAYS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

Artifacts: per_class_metrics.csv (tabpfn + xgb columns), composition CSV,
probs_*.npz (fp32 + y_true), resume npz with fp16 raw logits + context prior.
"""

import gc
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfv3_v3_common as core  # noqa: E402

from tabpfn import TabPFNClassifier  # noqa: E402

SEED_BAND_CONTEXT = 950   # shared per-class permutation band (exp4/exp5).
# Mining needs no extra seed band: seeds are the target class's perm prefix and
# benign tie-break is the benign perm order (see mine_hard_negative_order).

# Recipe = ordered component list. Sizes are FRACTIONS of k (so one definition
# serves smoke-scale and full-scale runs); -1 = the class's whole pool. Kinds:
#   ("take", class_name, f)            round(f*k) rows (perm prefix; capped at
#                                      pool with a printed note; -1 = full pool)
#   ("hardneg", target_class, f, fs)   round(f*k) benign rows nearest to the
#                                      target's seed rows (fs*k seeds; -1 = all)
#   ("rest", -1)                       fill to k, proportional over remaining
#                                      pool (>=1 per still-empty present class)
# At k=50,000 these realize the pre-registered compositions (0820.md SS9):
#   web_hn: web 1,522 + hn 20k + benign 8k + rest ~20.5k, etc.
RECIPES = {
    "bal50k":   [("powerlaw", 0.0)],
    "web_hn":   [("take", "web_attacks", -1), ("hardneg", "web_attacks", 0.40, -1),
                 ("take", "benign", 0.16), ("rest", -1)],
    "web_ctrl": [("take", "web_attacks", -1), ("take", "benign", 0.56), ("rest", -1)],
    "inf_hn":   [("take", "infiltration", 0.40), ("hardneg", "infiltration", 0.30, 0.10),
                 ("take", "benign", 0.10), ("rest", -1)],
    "dos_pair": [("take", "dos", 0.35), ("take", "ddos", 0.35),
                 ("take", "benign", 0.20), ("rest", -1)],
}


# ---------------------------------------------------------------------------
# Copied helpers (exp4:143 / exp5 -- both frozen, copy-not-import convention)
# ---------------------------------------------------------------------------

def _largest_remainder_targets(counts, budget):
    """exp4:143 verbatim -- proportional integer targets, >=1 per present class."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    target = np.zeros_like(counts)
    if total == 0:
        return target
    if budget >= total:
        return counts.copy()
    present = counts > 0
    n_present = int(present.sum())
    if budget < n_present:
        raise SystemExit(f"rest budget {budget} < {n_present} present classes.")
    raw = counts * budget / total
    target = np.minimum(np.floor(raw).astype(np.int64), counts)
    target[present & (target < 1)] = 1
    remaining = budget - int(target.sum())
    if remaining > 0:
        headroom = counts - target
        for class_id in np.argsort(-(raw - np.floor(raw))):
            if remaining <= 0:
                break
            if headroom[class_id] <= 0:
                continue
            target[class_id] += 1
            headroom[class_id] -= 1
            remaining -= 1
    else:
        order = np.argsort(raw - np.floor(raw))
        while remaining < 0:
            for class_id in order:
                if remaining >= 0:
                    break
                if target[class_id] > 1:
                    target[class_id] -= 1
                    remaining += 1
    assert int(target.sum()) == min(budget, total)
    return target


def _powerlaw_targets(counts, budget, alpha, class_names):
    """exp5 verbatim -- budget ∝ counts**alpha, capped at pool, waterfill."""
    counts = np.asarray(counts, dtype=np.int64)
    present = counts > 0
    total = int(counts.sum())
    budget = min(int(budget), total)
    if budget < int(present.sum()):
        raise SystemExit(f"--context-size {budget} < present classes.")
    target = np.zeros_like(counts)
    remaining = budget
    for _ in range(len(counts) + 1):
        active = present & (target < counts)
        if remaining <= 0 or not active.any():
            break
        w = np.zeros(len(counts), dtype=np.float64)
        w[active] = counts[active].astype(np.float64) ** alpha
        raw = np.zeros(len(counts), dtype=np.float64)
        raw[active] = w[active] * (remaining / w[active].sum())
        add = np.minimum(np.floor(raw).astype(np.int64), counts - target)
        leftover = remaining - int(add.sum())
        if leftover > 0:
            headroom = counts - target - add
            for cid in np.argsort(-(raw - np.floor(raw))):
                if leftover <= 0:
                    break
                if active[cid] and headroom[cid] > 0:
                    add[cid] += 1
                    headroom[cid] -= 1
                    leftover -= 1
        if int(add.sum()) == 0:
            break
        target += add
        remaining = budget - int(target.sum())
    assert int(target.sum()) == budget
    starved = present & (target < 1)
    if starved.any():
        raise SystemExit(f"powerlaw leaves classes empty: "
                         f"{[class_names[i] for i in np.flatnonzero(starved)]}")
    return target


def full_proba(proba, model_classes, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=np.int64)] = proba
    return out


def tabpfn_predict(clf, X_eval, batch_size):
    """exp5's tabpfn_predict_view minus selector scores: batched raw logits ->
    (probs fp32 == predict_proba, smoke-verified in exp5; raw fp16 for post-hoc)."""
    import torch
    if batch_size <= 0:
        batch_size = len(X_eval)
    n_rows = len(X_eval)
    n_batches = (n_rows + batch_size - 1) // batch_size
    probs_out, raw_out = [], []
    for bi, start in enumerate(range(0, n_rows, batch_size), 1):
        stop = min(start + batch_size, n_rows)
        raw = clf._raw_predict(X_eval[start:stop], return_logits=False,
                               return_raw_logits=True)
        raw = raw.detach().float().cpu()
        probs = clf.logits_to_probabilities(raw).detach().float().cpu().numpy()
        probs_out.append(probs.astype(np.float32))
        raw_out.append(raw.numpy().astype(np.float16))
        if bi == 1 or bi % 5 == 0 or stop == n_rows:
            print(f"  predict batch {bi}/{n_batches}: rows {start:,}:{stop:,}", flush=True)
        del raw
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return np.concatenate(probs_out), np.concatenate(raw_out, axis=1)


# ---------------------------------------------------------------------------
# Hard-negative mining (pool-only, model-free, deterministic)
# ---------------------------------------------------------------------------

def mine_hard_negative_order(X_pool, y_pool, perms, target_cid, benign_cid,
                             seed_rows):
    """Benign pool positions sorted by ascending min distance to the target
    class's seed rows, in pool-standardized feature space.  Uses ONLY the
    train pool (no val/test, no fitted model).  Deterministic: seeds are the
    first `seed_rows` of the target's shared 950-band permutation; candidate
    order before the distance sort is the benign 950-band permutation, and the
    argsort is stable -- ties resolve in that permutation order."""
    mu = X_pool.mean(axis=0)
    sd = X_pool.std(axis=0)
    sd[sd == 0] = 1.0
    tgt = perms[target_cid] if seed_rows <= 0 else perms[target_cid][:seed_rows]
    seeds = ((X_pool[tgt] - mu) / sd).astype(np.float32)
    ben_pos = perms[benign_cid]  # permutation order (deterministic superset)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        S = torch.as_tensor(seeds, device=dev)
        s2 = (S * S).sum(dim=1)
        dmin = np.empty(len(ben_pos), dtype=np.float32)
        chunk = 500_000 if dev == "cuda" else 100_000
        for i in range(0, len(ben_pos), chunk):
            B = torch.as_tensor(
                ((X_pool[ben_pos[i:i + chunk]] - mu) / sd).astype(np.float32),
                device=dev)
            d2 = (B * B).sum(dim=1, keepdim=True) - 2.0 * (B @ S.T) + s2[None, :]
            dmin[i:i + chunk] = d2.min(dim=1).values.float().cpu().numpy()
            del B, d2
        if dev == "cuda":
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        raise SystemExit("torch required for mining")
    order = np.argsort(dmin, kind="stable")
    return ben_pos[order]


# ---------------------------------------------------------------------------
# Recipe builder
# ---------------------------------------------------------------------------

def build_recipe_context(recipe_name, y_pool, X_pool, n_classes, class_names, k,
                         seed, allow_capped_take=False):
    """Returns (positions sorted, targets bincount, composition DataFrame)."""
    counts = np.bincount(y_pool, minlength=n_classes)
    present = counts > 0
    perms = {int(c): np.random.default_rng(seed + SEED_BAND_CONTEXT + int(c))
             .permutation(np.flatnonzero(y_pool == c))
             for c in np.flatnonzero(present)}
    taken = np.zeros(len(y_pool), dtype=bool)
    rows_log, parts = [], []

    def grab(cid, n, source, role):
        avail = perms[cid][~taken[perms[cid]]]
        if len(avail) < n:
            raise SystemExit(f"{recipe_name}: component {role} wants {n} rows of "
                             f"{class_names[cid]}, only {len(avail)} left.")
        sel = avail[:n]
        taken[sel] = True
        parts.append(sel)
        rows_log.append({"component": role, "class": class_names[cid],
                         "rows": int(n), "source": source})

    def frac_to_n(f):
        return int(round(f * k))

    for comp in RECIPES[recipe_name]:
        used = int(taken.sum())
        if comp[0] == "powerlaw":
            targets = _powerlaw_targets(counts, k, comp[1], class_names)
            for c in np.flatnonzero(targets > 0):
                grab(int(c), int(targets[c]), f"powerlaw_a{comp[1]:g}", "powerlaw")
        elif comp[0] == "take":
            cid = class_names.index(comp[1])
            want = int(counts[cid]) if comp[2] == -1 else frac_to_n(comp[2])
            n = min(want, int(counts[cid]))
            if n < want:
                # A capped take silently rewrites the registered composition
                # (shortfall flows into the benign-heavy rest fill) -- fatal
                # unless explicitly allowed for off-registration smoke runs
                # (review finding, 0820 SS9).
                if not allow_capped_take:
                    raise SystemExit(
                        f"{recipe_name}: take_{comp[1]} wants {want:,} rows but the "
                        f"pool holds {n:,}; the realized composition would deviate "
                        "from the registered recipe. Pass --allow-pool-capped-take "
                        "only for off-registration (smoke) runs.")
                print(f"NOTE: take_{comp[1]} capped at pool: {n:,} < {want:,} "
                      "(--allow-pool-capped-take)")
            grab(cid, n, "perm_prefix", f"take_{comp[1]}")
        elif comp[0] == "hardneg":
            tgt_cid = class_names.index(comp[1])
            ben_cid = class_names.index("benign")
            n = frac_to_n(comp[2])
            seed_n = -1 if comp[3] == -1 else frac_to_n(comp[3])
            order = mine_hard_negative_order(X_pool, y_pool, perms, tgt_cid,
                                             ben_cid, seed_n)
            cand = order[~taken[order]][:n]
            if len(cand) < n:
                raise SystemExit(f"{recipe_name}: hardneg wants {n}, got {len(cand)}")
            taken[cand] = True
            parts.append(cand)
            rows_log.append({"component": f"hardneg_{comp[1]}", "class": "benign",
                             "rows": int(n), "source": "knn_pool_only"})
        elif comp[0] == "rest":
            budget = (k - used) if comp[1] == -1 else frac_to_n(comp[1])
            if budget < 0:
                raise SystemExit(f"{recipe_name}: components exceed k before rest.")
            remaining = counts - np.bincount(y_pool[taken], minlength=n_classes)
            targets = _largest_remainder_targets(remaining, budget)
            for c in np.flatnonzero(targets > 0):
                grab(int(c), int(targets[c]), "perm_prefix", "rest_proportional")
        else:
            raise SystemExit(f"unknown component kind {comp[0]}")

    ctx = np.sort(np.concatenate(parts))
    assert len(ctx) == len(np.unique(ctx)), "duplicate rows in context"
    if len(ctx) != k:
        raise SystemExit(f"{recipe_name}: composed {len(ctx):,} rows != k={k:,}.")
    got = np.bincount(y_pool[ctx], minlength=n_classes)
    empty = present & (got == 0)
    if empty.any():
        raise SystemExit(f"{recipe_name}: classes with 0 rows "
                         f"{[class_names[i] for i in np.flatnonzero(empty)]} -- "
                         "label-space invariant violated; adjust the recipe.")
    comp_df = pd.DataFrame(rows_log)
    per_class = pd.DataFrame({"component": "TOTAL", "class": class_names,
                              "rows": got, "source": ""})
    per_class = per_class[per_class["rows"] > 0]
    return ctx, got, pd.concat([comp_df, per_class], ignore_index=True)


# ---------------------------------------------------------------------------
# Resume tags
# ---------------------------------------------------------------------------

def _h8(path):
    return hashlib.sha1(os.path.realpath(path).encode()).hexdigest()[:8]


def exp6_fit_tag(args, ctx_n, ctx_hash):
    """ctx_hash = sha1 of the realized global row ids: a drifted context (e.g.
    device-dependent mining ties) gets fresh checkpoint names automatically, so
    stale dump/xgb siblings can never silently resume (review finding, SS9)."""
    return (f"exp6_{args.target_dataset}_r{args.recipe}_mts{args.max_train_samples}"
            f"_seed{args.seed}_k{args.context_size}_m{args.n_estimators}"
            f"_ipl{int(args.ignore_pretraining_limits)}_fm{args.fit_mode}"
            f"_sp{args.train_split.replace('+', '')}_n{ctx_n}_c{ctx_hash}"
            f"_d{_h8(args.data)}_m{_h8(args.model_path)}")


def xgb_param_sig(args):
    return (f"xne{args.xgb_n_estimators}_xd{args.xgb_max_depth}"
            f"_xlr{args.xgb_learning_rate}_xss{args.xgb_subsample}"
            f"_xcs{args.xgb_colsample_bytree}_xmcw{args.xgb_min_child_weight}"
            f"_xrl{args.xgb_reg_lambda}")


def exp6_pred_tag(fit_tag, args, n_eval, tb):
    # tb is in the tag: batch splitting alone shifts bf16 probs ~4e-4
    # (0820 SS8a), so a dump is only valid for the batching that produced it.
    return (f"{fit_tag}_ev{args.eval_split}_tc{args.test_cap_per_class}"
            f"_tb{tb}_nt{n_eval}")


# ---------------------------------------------------------------------------
# Run body
# ---------------------------------------------------------------------------

def run_exp6(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp6_context_recipes"
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)

    if args.recipe not in RECIPES:
        raise SystemExit(f"--recipe must be one of {sorted(RECIPES)}, got {args.recipe!r}")
    if args.subsample_samples:
        raise SystemExit("exp6 builds its context explicitly; --subsample-samples must stay 0.")
    if args.context_size <= 0:
        raise SystemExit("exp6 needs an explicit --context-size > 0.")
    if args.eval_split == "val" and args.train_split != "train":
        raise SystemExit("--eval-split val is only valid with --train-split train.")
    if args.n_estimators < 1:
        raise SystemExit("--n-estimators (m) must be >= 1.")

    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, \
        split_audit, label_fn = cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    print(f"target={args.target_dataset}  recipe={args.recipe}  "
          f"classes ({n_classes}): {class_names}")
    print(f"split rows: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    if args.train_split == "train+val":
        train_idx = np.sort(np.concatenate([train_idx, val_idx]))

    eval_pool_idx = test_idx if args.eval_split == "test" else val_idx
    eval_idx = core.cap_per_class(eval_pool_idx, label_fn(eval_pool_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_eval = label_fn(eval_idx)
    print(f"eval split={args.eval_split}: {len(eval_idx):,} rows "
          f"(cap_per_class={args.test_cap_per_class})")
    X_eval = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))

    pool_idx = train_idx
    cap_policy = "none"
    if args.max_train_samples > 0 and args.max_train_samples < len(pool_idx):
        pool_idx = core.stratified_subset(pool_idx, label_fn(pool_idx), n_classes,
                                          args.max_train_samples, args.seed + 850)
        cap_policy = "stratified_ratio_preserving"
    y_pool = label_fn(pool_idx)
    print(f"train pool: {len(pool_idx):,} of {len(train_idx):,} (cap_policy={cap_policy})")
    X_pool = np.nan_to_num(np.asarray(X[pool_idx], dtype=np.float32))
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()

    t0 = time.time()
    ctx, targets, composition = build_recipe_context(
        args.recipe, y_pool, X_pool, n_classes, class_names,
        args.context_size, args.seed,
        allow_capped_take=args.allow_pool_capped_take)
    print(f"context built in {time.time()-t0:.1f}s")
    print(composition.to_string(index=False))
    X_fit = X_pool[ctx]
    y_fit = y_pool[ctx]
    del X_pool
    gc.collect()
    core.report_memory_plan(args.context_size, len(eval_idx), args)

    ctx_gids = pool_idx[ctx]
    ctx_hash = hashlib.sha1(np.ascontiguousarray(ctx_gids).tobytes()).hexdigest()[:8]
    fit_tag = exp6_fit_tag(args, len(ctx), ctx_hash)
    ctx_ckpt = os.path.join(args.resume_dir, f"{fit_tag}_ctx.npz")
    if os.path.exists(ctx_ckpt):
        saved = np.load(ctx_ckpt)
        if not (np.array_equal(saved["ctx_pos"], ctx)
                and np.array_equal(saved["ctx_global_ids"], ctx_gids)):
            raise SystemExit(f"{ctx_ckpt} != rebuilt context -- determinism drift; "
                             "delete the checkpoint to refit.")
        print(f"context verified against {ctx_ckpt}")
    else:
        np.savez_compressed(ctx_ckpt, ctx_pos=ctx, ctx_global_ids=ctx_gids)

    all_rows, timings, probs_by_method = [], {}, {}
    tb = args.test_batch_size if args.test_batch_size > 0 else len(eval_idx)

    # ---- TabPFN ----
    if not args.skip_tabpfn:
        pred_ckpt = os.path.join(
            args.resume_dir,
            f"{exp6_pred_tag(fit_tag, args, len(eval_idx), tb)}_dump.npz")
        if not args.force_refit and os.path.exists(pred_ckpt):
            print(f"resuming TabPFN dump from {pred_ckpt}")
            saved = np.load(pred_ckpt)
            probs = saved["probs"].astype(np.float32)
            model_classes = saved["classes"]
            fit_s = pred_s = None
        else:
            clf = TabPFNClassifier(
                device=args.device, model_path=args.model_path,
                ignore_pretraining_limits=args.ignore_pretraining_limits,
                random_state=args.seed, n_estimators=args.n_estimators,
                auto_scale_n_estimators=False, fit_mode=args.fit_mode,
                keep_cache_on_device=args.keep_cache_on_device)
            max_samples = clf.get_inference_config().MAX_NUMBER_OF_SAMPLES
            if not args.ignore_pretraining_limits and len(ctx) > max_samples:
                raise SystemExit(f"context {len(ctx):,} > guard {max_samples:,}")
            print(f"TabPFN fit on {len(ctx):,} rows (m={args.n_estimators}, "
                  f"fit_mode={args.fit_mode}) ...", flush=True)
            t0 = time.time()
            clf.fit(X_fit, y_fit)
            fit_s = time.time() - t0
            print(f"TabPFN fit done in {fit_s:.1f}s")
            t0 = time.time()
            probs, raw16 = tabpfn_predict(clf, X_eval, tb)
            pred_s = time.time() - t0
            model_classes = np.asarray(clf.classes_)
            np.savez_compressed(pred_ckpt, probs=probs.astype(np.float32),
                                raw_logits=raw16, classes=model_classes,
                                context_class_counts=targets,
                                softmax_temperature=clf.softmax_temperature_)
            print(f"TabPFN predict done in {pred_s:.1f}s -> {pred_ckpt}")
            del clf, raw16
            gc.collect()
        probs = full_proba(probs, model_classes, n_classes)
        method = f"tabpfn_{args.recipe}"
        all_rows.extend(core.per_class_table(method, y_eval, np.argmax(probs, axis=1),
                                             class_names, tail_classes))
        probs_by_method[method] = probs
        timings["tabpfn_fit_seconds"] = fit_s
        timings["tabpfn_predict_seconds"] = pred_s

    # ---- XGB mirror ----
    if not args.skip_xgboost:
        ckpt = os.path.join(args.resume_dir, f"{fit_tag}_{xgb_param_sig(args)}_xgb.json")
        t0 = time.time()
        if not args.force_refit and os.path.exists(ckpt):
            print(f"resuming XGB from {ckpt}")
            booster = xgb.XGBClassifier()
            booster.load_model(ckpt)
            fit_s = None
        else:
            booster = xgb.XGBClassifier(
                n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                min_child_weight=args.xgb_min_child_weight,
                reg_lambda=args.xgb_reg_lambda,
                objective="multi:softprob", num_class=n_classes,
                eval_metric="mlogloss", n_jobs=-1, random_state=args.seed)
            print(f"XGB fit on {len(X_fit):,} rows ...", flush=True)
            booster.fit(X_fit, y_fit)
            fit_s = time.time() - t0
            booster.save_model(ckpt)
        t1 = time.time()
        margins = booster.get_booster().inplace_predict(X_eval, predict_type="margin")
        zmax = margins.max(axis=1, keepdims=True)
        probs = np.exp(margins - zmax)
        probs /= probs.sum(axis=1, keepdims=True)
        probs = full_proba(probs.astype(np.float32), booster.classes_, n_classes)
        method = f"xgb_{args.recipe}"
        all_rows.extend(core.per_class_table(method, y_eval, np.argmax(probs, axis=1),
                                             class_names, tail_classes))
        probs_by_method[method] = probs
        timings["xgb_fit_seconds"] = fit_s
        timings["xgb_predict_seconds"] = time.time() - t1

    # ---- artifacts ----
    timings.update({
        "experiment": args.experiment, "target_dataset": args.target_dataset,
        "recipe": args.recipe, "eval_split": args.eval_split,
        "train_pool_rows": int(len(pool_idx)), "train_cap_policy": cap_policy,
        "context_size": int(args.context_size),
        "n_estimators": int(args.n_estimators),
        "context_class_counts": {class_names[i]: int(targets[i])
                                 for i in np.flatnonzero(targets)},
        "eval_rows": int(len(eval_idx)), "test_batch_size": int(tb),
        "test_batches": int(math.ceil(len(eval_idx) / tb)),
    })
    table = pd.DataFrame(all_rows)
    if len(table):
        summary = table[table["class"].isin(["macro_avg", "weighted_avg", "tail_avg"])]
        print("\n=== summary (macro / weighted / tail F1) ===")
        print(summary.pivot(index="class", columns="method", values="f1").to_string())

    ts = time.strftime("%Y%m%d_%H%M%S")
    split_tag = "" if args.train_split == "train" else "_82"
    eval_tag = "" if args.eval_split == "test" else "_evval"
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp6_{args.recipe}{split_tag}{eval_tag}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    composition.to_csv(os.path.join(out_dir, "context_composition.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "split_audit.csv"), index=False)
    for method, probs in probs_by_method.items():
        np.savez_compressed(os.path.join(out_dir, f"probs_{method}.npz"),
                            probs=probs.astype(np.float32),
                            y_true=y_eval.astype(np.int64),
                            class_names=np.asarray(class_names),
                            context_class_counts=targets)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([timings], f, indent=2)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--recipe", required=True,
                   help=f"One of {sorted(RECIPES)} (frozen definitions in-script; "
                        "the realized composition lands in context_composition.csv).")
    p.add_argument("--context-size", type=int, default=50_000,
                   help="Context row budget k (lightweight by design; must be > 0).")
    p.add_argument("--eval-split", default="test", choices=["test", "val"],
                   help="test = far-future 20%% (honest). val = near-future 20%% "
                        "for combination/gamma selection (E2-protocol extension).")
    p.add_argument("--allow-pool-capped-take", action="store_true",
                   help="Permit 'take' components capped at pool availability "
                        "(off-registration compositions; smoke runs only).")
    p.set_defaults(
        max_train_samples=-1,
        n_estimators=1,
        test_cap_per_class=0,
        fit_mode="fit_with_cache",
        test_batch_size=500_000,
        subsample_samples=0,
    )
    args = p.parse_args()
    run_exp6(args)


if __name__ == "__main__":
    main()

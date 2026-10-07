#!/usr/bin/env python3
"""EXP 4 -- Tail-Anchored Context Bagging (TACB) for TabPFN-v3.

The question: exp1/exp3's proportional stratified sampling starves tail classes
(cic2018 `web_attacks`: 126 of 1M context rows; 0813.md measured 13 rows at
100k -> XGB F1 0.000 vs TabPFN 0.560).  0818.md showed the damage is caused by
COMPOSITION, not volume ("sampling erased micro-classes").  This experiment
builds each estimator's in-context training set explicitly:

  ANCHOR  classes whose natural train pool is small (<= --anchor-threshold, or
          named via --anchor-classes) contribute 100%% of their natural pool
          rows to EVERY estimator context.  No duplication, no oversampling,
          no synthetic rows -- just "do not subsample the tail away".
  HEAD    all other classes fill the remaining per-context budget k with the
          same proportional largest-remainder allocation as exp1/exp3
          (natural class ratios preserved), partitioned DISJOINTLY across the
          E contexts (exp3-style coverage).
  FLOOR   optional --context-floor guarantees non-anchored classes a minimum
          row count per context (0818.md:74's standing TODO).

Failure-mode boundary (SRC_HISTORY F8): s41 proved fully class-BALANCED
sampling loses on both legs.  TACB is NOT balancing: head ratios stay natural,
anchored classes are capped at their own natural pool (0.15%% of a 1M cic2018
context), and nothing is duplicated or generated.  No router, no learned
weights (F1/F6): per-estimator probabilities are averaged, then argmax.

Verified library behavior this script depends on (tabpfn 8.2.0)
----------------------------------------------------------------
  * inference_config={"SUBSAMPLE_SAMPLES": [arr0, arr1, ...]} -- a PYTHON list
    of per-estimator positional index arrays into the X passed to fit(), applied
    verbatim BEFORE all preprocessing (preprocessing/transform.py:66-68).
    With len(list) == n_estimators the mapping is identity
    (preprocessing/ensemble.py:525-543; `_balance` degenerates, no warning).
  * auto_scale_n_estimators MUST stay False: if the library raised
    n_estimators_ above len(list), the list would be block-repeated silently.
  * fit() validation sees the matrix passed to fit(), so this script gathers
    the UNION of the E contexts and remaps indices; --ignore-pretraining-limits
    is needed only when the union itself exceeds the 1M checkpoint guard
    (per-estimator memory is one k-row context, not the union).
  * balance_probabilities is applied AFTER estimator averaging + softmax
    (classifier.py:1623-1630 -> utils.py:545-562: divide by train priors,
    renormalize).  It is therefore reproduced POST-HOC here from the dumped
    averaged probabilities -- the `tabpfn_tacb_bp` column costs no extra run.
    Passing --balance-probabilities instead builds it into the model (used once
    in the smoke test to verify the post-hoc arithmetic is identical).

One-knob chains this script is designed to complete
---------------------------------------------------
  exp1 (proportional 1M context, n_estimators=4, uncapped test; recorded in
  tabpfn/results/20260818_*_exp1_1m) is the parent.  CAVEAT: cross-script exp1
  links carry a second knob -- identical per-class counts but a different row
  REALIZATION (exp1 drew rows via stratified_subset at seed+850; exp4's builder
  permutes at seed+950).  R0 below closes that gap; the fully clean one-knob
  chains are the within-exp4 ones (R0/R1/R3c/R3a).  Keep every base arg at the
  exp1 value and move ONE thing per run:

    # R0 (bridge): EXACTLY exp1's context rebuilt in-script. With
    # --max-train-samples 1000000 the pool IS exp1's stratified_subset draw
    # (seed+850), and --context-size -1 makes that whole pool the one context.
    python tabpfn/nfv3_v3_exp4_tail_anchored_bagging.py --target-dataset cic2018 \\
        --max-train-samples 1000000 --context-size -1 --n-contexts 1 \\
        --anchor-threshold 0 --test-cap-per-class 0

    # R1: sampler only (single anchored context, same budget, same 4 members)
    python tabpfn/nfv3_v3_exp4_tail_anchored_bagging.py --target-dataset cic2018 \\
        --n-contexts 1 --context-size 1000000 --anchor-threshold 5000 \\
        --test-cap-per-class 0

    # R3c: context-count only (proportional bagging, E=4 -- the E-only control)
    python tabpfn/nfv3_v3_exp4_tail_anchored_bagging.py --target-dataset cic2018 \\
        --n-contexts 4 --context-size 1000000 --anchor-threshold 0 \\
        --test-cap-per-class 0 --ignore-pretraining-limits

    # R3a: anchor + bagging (the proposed model; differs from R3c by anchor only)
    python tabpfn/nfv3_v3_exp4_tail_anchored_bagging.py --target-dataset cic2018 \\
        --n-contexts 4 --context-size 1000000 --anchor-threshold 5000 \\
        --test-cap-per-class 0 --ignore-pretraining-limits

    # R2: the honest XGB bar (full pool, no TabPFN; xgboost + xgboost_sqrt)
    python tabpfn/nfv3_v3_exp4_tail_anchored_bagging.py --target-dataset cic2018 \\
        --n-contexts 1 --context-size -1 --skip-tabpfn --test-cap-per-class 0

  Every run also trains XGBoost (unweighted AND sqrt-class-weighted, s44's
  formula) on the UNION of the contexts -- the exact row SET the ensemble saw --
  so "is the anchor model-agnostic?" is answered in-run (CLAUDE.md: baseline
  and method in the same split).  NOTE: at E>1 each anchored row appears once in
  the union but in EVERY TabPFN context, so XGB's anchor share is ~E-fold
  diluted relative to each estimator's context; the share comparison is exact
  only at E=1 (R1).

Evaluation protocol
-------------------
  Honest numbers use --test-cap-per-class 0 (0818.md: the 100k cap inflated
  web_attacks F1 0.603 -> 0.162).  --eval-split val scores the near-future 20%%
  slice instead of the far-future test slice (predict-only on a resumed fit) --
  the temporal-drift axis; it is refused under --train-split train+val.

Cost model (RTX 4090 class, measured by exp1/exp3)
--------------------------------------------------
  TabPFN fit ~= 1,250 s at 1M context (scales ~linearly with k).
  predict: each of the n_estimators members re-encodes its own k-row context
  per test batch -- exp1 (4 members, 1M ctx, 4.02M cic2018 test rows) took
  5,290 s.  Memory is ONE context + ONE test batch (estimators sequential):
  k=1M needs ~2 batches on cic2018 uncapped test (see nfv3_v3_common docstring).
  ALWAYS export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

Artifacts (in addition to the standard four)
--------------------------------------------
  context_composition.csv   per (estimator, class): role anchor/floor/head,
                            rows, pool_rows, share, disjointness; plus `union`
                            and `pool` sentinel rows.
  probs_<method>.npz        averaged predicted probabilities on the eval rows
                            (float32), y_true, class_names, union class counts
                            -- enables post-hoc rules without re-predicting.
"""

import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfv3_v3_common as core  # noqa: E402

from tabpfn import TabPFNClassifier  # noqa: E402

SEED_BAND_CONTEXT = 950  # per-class shuffles: seed + 950 + class_id (unused band;
#                          existing code uses +500+i, +800, +850, +900, +class_id)


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def _largest_remainder_targets(counts, budget):
    """Proportional integer targets over `counts` summing to
    min(budget, counts.sum()): floor of the proportional share, >=1 for every
    present class, remainder by largest fractional part where headroom exists.
    Mirrors nfv3_v3_common.stratified_subset:321-334 exactly, but returns the
    per-class targets instead of drawing rows."""
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
        raise SystemExit(
            f"head budget {budget} is smaller than the number of present head "
            f"classes ({n_present}); raise --context-size or anchor fewer classes.")
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
        # The >=1-per-present-class bump can overshoot the budget when many
        # micro-classes round up; walk it back from the weakest claims (multiple
        # passes -- donors must keep >=1) so the context never exceeds
        # --context-size. Terminates: sum(target-1) >= sum-budget because the
        # budget >= n_present guard above holds. (Review finding, 0820.)
        order = np.argsort(raw - np.floor(raw))
        while remaining < 0:
            for class_id in order:
                if remaining >= 0:
                    break
                if target[class_id] > 1:
                    target[class_id] -= 1
                    remaining += 1
    assert int(target.sum()) == min(budget, total), "largest-remainder targets != budget"
    return target


def _resolve_anchor_mask(counts, anchor_threshold, anchor_class_ids):
    """Single source of truth for which classes are anchored (used by both the
    builder and the auto-E derivation, so they can never diverge)."""
    present = counts > 0
    anchored = present & np.isin(np.arange(len(counts)),
                                 np.asarray(list(anchor_class_ids), dtype=np.int64))
    if anchor_threshold > 0:
        anchored |= present & (counts <= anchor_threshold)
    return anchored


def _compute_targets(counts, anchored, k, floor, class_names):
    """Stage 1: per-context class targets (identical for every estimator).
    Returns (targets, floor_pinned mask)."""
    n_classes = len(counts)
    present = counts > 0
    targets = np.zeros(n_classes, dtype=np.int64)
    targets[anchored] = counts[anchored]
    anchor_total = int(targets.sum())
    if anchor_total > k:
        offenders = {class_names[i]: int(counts[i]) for i in np.flatnonzero(anchored)}
        raise SystemExit(
            f"anchored classes alone hold {anchor_total:,} rows > --context-size "
            f"{k:,}: {offenders}. Raise --context-size or anchor fewer classes.")
    head = present & ~anchored
    head_budget = k - anchor_total
    floor_target = np.minimum(counts, floor) if floor > 0 else np.zeros(n_classes, dtype=np.int64)
    fixed = np.zeros(n_classes, dtype=bool)
    if floor > 0 and int(floor_target[head].sum()) > head_budget:
        raise SystemExit(
            f"anchor ({anchor_total:,}) + per-class floors "
            f"({int(floor_target[head].sum()):,}) exceed --context-size {k:,}; "
            "lower --context-floor or raise --context-size.")
    # Waterfilling: proportional allocation over free head classes; any class
    # allocated below its floor is pinned at the floor and the rest reflows.
    # Converges in <= n_classes iterations. With floor=0 this is one pass and
    # the head stays natural-proportional (F8: never fully balanced).
    while True:
        free = head & ~fixed
        remaining_budget = head_budget - int(targets[head & fixed].sum())
        alloc = _largest_remainder_targets(np.where(free, counts, 0), remaining_budget)
        below = free & (alloc < floor_target)
        if floor > 0 and below.any():
            targets[below] = floor_target[below]
            fixed[below] = True
            continue
        targets[free] = alloc[free]
        break
    return targets, fixed


def build_tail_anchored_contexts(labels, n_classes, class_names, k, n_contexts,
                                 anchor_threshold, anchor_class_ids, floor, seed):
    """Build E explicit per-estimator contexts over a train pool.

    Args:
        labels: (P,) int64 labels of the train pool; returned indices are
            POSITIONAL into this pool (0..P-1).
        k: per-context row budget.
        n_contexts: E.
        anchor_threshold: classes with pool count <= this are anchored (0 = off).
        anchor_class_ids: iterable of class ids that are always anchored.
        floor: minimum rows per context for each non-anchored present class
            (0 = off); capped at the class's pool count.
        seed: base seed; per-class RNG is default_rng(seed + 950 + class_id).

    Returns:
        (contexts, targets, anchored, composition)
        contexts: list of E sorted int64 arrays (positions into the pool).
        targets: (n_classes,) per-context class row counts (same for every e).
        anchored: (n_classes,) bool mask of anchored classes.
        composition: DataFrame, one row per (estimator, class) plus `union`
            and `pool` sentinel estimators.

    Anchored classes contribute their whole pool to EVERY context (replicated
    across estimators, never duplicated within one).  Head classes are
    partitioned disjointly; a head class whose pool cannot fill E disjoint
    chunks falls back to a cyclic window (no duplicates within one context,
    flagged in the composition table).
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=n_classes)
    present = counts > 0
    anchored = _resolve_anchor_mask(counts, anchor_threshold, anchor_class_ids)
    targets, fixed = _compute_targets(counts, anchored, k, floor, class_names)

    rng_orders = {}
    contexts = []
    rows = []
    for e in range(n_contexts):
        parts = []
        for c in np.flatnonzero(targets > 0):
            if c not in rng_orders:
                rng = np.random.default_rng(seed + SEED_BAND_CONTEXT + int(c))
                rng_orders[c] = rng.permutation(np.flatnonzero(labels == c))
            order = rng_orders[c]
            t_c, n_c = int(targets[c]), int(counts[c])
            if anchored[c]:
                take = order
                disjoint = "replicated_all_contexts"
            elif n_c >= n_contexts * t_c:
                take = order[e * t_c:(e + 1) * t_c]
                disjoint = "disjoint"
            else:
                take = order[(np.arange(t_c) + e * t_c) % n_c]
                disjoint = "cyclic_wrap"
            parts.append(take)
            role = "anchor" if anchored[c] else (
                "floor" if (floor > 0 and fixed[c]) else "head")
            rows.append({
                "estimator": e, "class": class_names[c], "role": role,
                "rows": len(take), "pool_rows": n_c,
                "share_of_context_pct": round(100.0 * len(take) / max(k, 1), 4),
                "distribution": disjoint,
            })
        contexts.append(np.sort(np.concatenate(parts)))

    union_pos = np.unique(np.concatenate(contexts))
    union_counts = np.bincount(labels[union_pos], minlength=n_classes)
    for label_tag, arr in (("union", union_counts), ("pool", counts)):
        for c in np.flatnonzero(present):
            rows.append({
                "estimator": label_tag, "class": class_names[c],
                "role": "anchor" if anchored[c] else "head",
                "rows": int(arr[c]), "pool_rows": int(counts[c]),
                "share_of_context_pct": round(100.0 * arr[c] / max(int(arr.sum()), 1), 4),
                "distribution": "",
            })
    return contexts, targets, anchored, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring / probability helpers
# ---------------------------------------------------------------------------

def full_proba(proba, model_classes, n_classes):
    """Expand a (n, len(model_classes)) probability matrix to (n, n_classes),
    zero-filling columns for classes absent from the fit rows."""
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=np.int64)] = proba
    return out


def balance_probs_post_hoc(probs, class_counts):
    """Reproduce TabPFN's balance_probabilities from averaged probabilities:
    divide by the fit-set class priors, renormalize (utils.py:545-562; applied
    after estimator averaging, classifier.py:1623-1630, so this is exact)."""
    prior = class_counts / class_counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(prior > 0, probs / prior, 0.0)
    denom = out.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return (out / denom).astype(np.float32)


def sqrt_sample_weights(y, n_classes):
    """s44's class weighting (s44:577 / exp_utils.sqrt_balanced_weights):
    w_c = sqrt(N / (K * n_c)) over the K classes present in y."""
    counts = np.bincount(y, minlength=n_classes)
    k_present = int((counts > 0).sum())
    w_class = np.zeros(n_classes, dtype=np.float64)
    nz = counts > 0
    w_class[nz] = np.sqrt(len(y) / (k_present * counts[nz]))
    return w_class[y]


def predict_proba_in_batches(model, features, batch_size):
    if batch_size <= 0:
        batch_size = len(features)
    n_rows = len(features)
    n_batches = (n_rows + batch_size - 1) // batch_size
    out = []
    for batch_number, start in enumerate(range(0, n_rows, batch_size), 1):
        stop = min(start + batch_size, n_rows)
        out.append(model.predict_proba(features[start:stop]))
        if batch_number == 1 or batch_number % 10 == 0 or stop == n_rows:
            print(f"TabPFN predict_proba batch {batch_number}/{n_batches}: "
                  f"rows {start:,}:{stop:,}", flush=True)
    return np.concatenate(out)


def save_probs(out_dir, method, probs, y_true, class_names, union_counts,
               context_targets):
    np.savez_compressed(
        os.path.join(out_dir, f"probs_{method}.npz"),
        probs=probs.astype(np.float32), y_true=y_true.astype(np.int64),
        class_names=np.asarray(class_names), union_class_counts=union_counts,
        context_class_targets=context_targets)


# ---------------------------------------------------------------------------
# Resume tags
# ---------------------------------------------------------------------------

def _h8(path):
    """8-hex fingerprint of a resolved path, for resume-tag identity."""
    return hashlib.sha1(os.path.realpath(path).encode()).hexdigest()[:8]


def exp4_resume_tag(args, anchor_sig, union_n):
    """Fit-checkpoint stem: every arg that changes the FIT input, including the
    data pickle and TabPFN checkpoint identities (a --data or --model-path
    override must not silently reuse a stale cache -- review finding, 0820).
    Evaluation settings (--eval-split, --test-cap-per-class) are deliberately
    absent so a val-slice re-score reuses the expensive fit."""
    return (f"exp4_{args.target_dataset}_mts{args.max_train_samples}"
            f"_seed{args.seed}_E{args.n_contexts}_ne{args.n_estimators}"
            f"_k{args.context_size}_at{args.anchor_threshold}_ac{anchor_sig}"
            f"_fl{args.context_floor}_bp{int(args.balance_probabilities)}"
            f"_ipl{int(args.ignore_pretraining_limits)}_fm{args.fit_mode}"
            f"_sp{args.train_split.replace('+', '')}_u{union_n}"
            f"_d{_h8(args.data)}_m{_h8(args.model_path)}")


def xgb_param_sig(args):
    """XGB checkpoints are additionally keyed on the seven XGB hyperparameters,
    so a hyperparameter ablation cannot silently reuse a stale booster."""
    return (f"xne{args.xgb_n_estimators}_xd{args.xgb_max_depth}"
            f"_xlr{args.xgb_learning_rate}_xss{args.xgb_subsample}"
            f"_xcs{args.xgb_colsample_bytree}_xmcw{args.xgb_min_child_weight}"
            f"_xrl{args.xgb_reg_lambda}")


def exp4_prediction_tag(fit_tag, args, n_eval):
    return f"{fit_tag}_ev{args.eval_split}_tc{args.test_cap_per_class}_nt{n_eval}"


# ---------------------------------------------------------------------------
# The run body (forked from nfv3_v3_common.run_experiment, which stays frozen:
# exp4 needs explicit per-estimator contexts, three method columns, an eval
# split switch, and probability dumps -- none of which the shared body has)
# ---------------------------------------------------------------------------

def run_exp4(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp4_tail_anchored_bagging"
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)

    if args.subsample_samples:
        raise SystemExit("exp4 builds per-estimator contexts explicitly; "
                         "--subsample-samples must stay 0.")
    if args.eval_split == "val" and args.train_split != "train":
        raise SystemExit("--eval-split val is only valid with --train-split train "
                         "(otherwise the val slice is inside the training pool).")
    if args.n_contexts < 0:
        raise SystemExit("--n-contexts must be >= 0 (0 = auto full head coverage).")

    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, \
        split_audit, label_fn = cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    print(f"target={args.target_dataset}  classes ({n_classes}): {class_names}")
    print(f"split rows: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    if args.train_split == "train+val":
        train_idx = np.sort(np.concatenate([train_idx, val_idx]))
        print(f"--train-split train+val: train pool {len(train_idx):,} (60% + 20%)")

    # ---- resolve anchor classes ----
    if args.anchor_classes == "tail":
        anchor_names = list(tail_classes)
    elif args.anchor_classes in ("none", ""):
        anchor_names = []
    else:
        anchor_names = [s.strip() for s in args.anchor_classes.split(",") if s.strip()]
        unknown = [n for n in anchor_names if n not in class_names]
        if unknown:
            raise SystemExit(f"--anchor-classes contains unknown class(es) {unknown}; "
                             f"valid: {class_names}")
    anchor_class_ids = [class_names.index(n) for n in anchor_names]
    anchor_sig = "none" if not anchor_names and args.anchor_threshold == 0 else (
        "-".join(sorted(anchor_names)) if anchor_names else "thr")

    # ---- evaluation set (test = far-future 20%, val = near-future 20%) ----
    eval_pool_idx = test_idx if args.eval_split == "test" else val_idx
    eval_idx = core.cap_per_class(eval_pool_idx, label_fn(eval_pool_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_eval = label_fn(eval_idx)
    print(f"eval split={args.eval_split}: {len(eval_idx):,} rows "
          f"(cap_per_class={args.test_cap_per_class})")
    X_eval = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))

    print(f"\n--- pre-fit diagnostic: train/{args.eval_split} feature drift per class ---")
    drift_df = core.diagnose_train_test_drift(X, class_names, train_idx, eval_pool_idx, label_fn)
    print(drift_df.to_string(index=False))

    # ---- train pool budget (frozen semantics: stratified proportional pre-cap) ----
    pool_idx = train_idx
    cap_policy = "none"
    if args.max_train_samples > 0 and args.max_train_samples < len(pool_idx):
        pool_idx = core.stratified_subset(pool_idx, label_fn(pool_idx), n_classes,
                                          args.max_train_samples, args.seed + 850)
        cap_policy = "stratified_ratio_preserving"
    y_pool = label_fn(pool_idx)
    print(f"train pool: {len(pool_idx):,} of {len(train_idx):,} (cap_policy={cap_policy})")

    # ---- resolve k and E ----
    k = args.context_size if args.context_size > 0 else len(pool_idx)
    if args.context_size <= 0:
        print(f"[auto] --context-size {args.context_size} -> whole pool k={k:,}")
        args.context_size = k
    if args.n_contexts == 0:
        # Full head coverage: E must satisfy E*t_c >= n_c for EVERY non-anchored
        # present class (then the disjoint/cyclic branches cover all rows). A
        # pool-ratio estimate under-covers when floors replicate or rounding
        # truncates (review finding, 0820), so derive E from the actual targets.
        counts0 = np.bincount(y_pool, minlength=n_classes)
        anch0 = _resolve_anchor_mask(counts0, args.anchor_threshold, anchor_class_ids)
        t0, _ = _compute_targets(counts0, anch0, k, args.context_floor, class_names)
        head0 = (counts0 > 0) & ~anch0 & (t0 > 0)
        args.n_contexts = int(np.max(np.ceil(counts0[head0] / t0[head0]))) if head0.any() else 1
        print(f"[auto] --n-contexts 0 -> {args.n_contexts} (smallest E with full head "
              f"coverage: E*t_c >= pool_c for every non-anchored class)")
    if args.n_estimators % args.n_contexts != 0:
        raise SystemExit(f"--n-estimators ({args.n_estimators}) must be a multiple of "
                         f"--n-contexts ({args.n_contexts}) so every context occupies "
                         "the same number of (interleaved) estimator slots.")

    # ---- build contexts ----
    contexts, targets, anchored, composition = build_tail_anchored_contexts(
        y_pool, n_classes, class_names, k, args.n_contexts,
        args.anchor_threshold, anchor_class_ids, args.context_floor, args.seed)
    union_pos = np.unique(np.concatenate(contexts))
    remapped = [np.searchsorted(union_pos, ctx) for ctx in contexts]
    y_fit = y_pool[union_pos]
    for e, (ctx, rmp) in enumerate(zip(contexts, remapped)):
        assert np.array_equal(union_pos[rmp], ctx), f"context {e}: remap broken"
        got = np.bincount(y_fit[rmp], minlength=n_classes)
        assert np.array_equal(got, targets), f"context {e}: composition mismatch"
    union_counts = np.bincount(y_fit, minlength=n_classes)
    print(f"contexts: E={args.n_contexts} x k={k:,}  union={len(union_pos):,} rows "
          f"({100 * len(union_pos) / len(pool_idx):.1f}% of pool)")
    pool_counts = np.bincount(y_pool, minlength=n_classes)
    shortfall = pool_counts - union_counts
    if (shortfall > 0).any():
        gaps = {class_names[i]: int(shortfall[i]) for i in np.flatnonzero(shortfall > 0)}
        print(f"NOTE: {int(shortfall.sum()):,} pool rows "
              f"({100 * shortfall.sum() / len(pool_idx):.2f}%) are in NO context "
              f"(per-class: {gaps}). Expected whenever E*k < pool; pass "
              "--n-contexts 0 for guaranteed full head coverage.")
    ctx_summary = composition[composition["estimator"] == 0]
    print(ctx_summary.to_string(index=False))

    X_fit = np.nan_to_num(np.asarray(X[pool_idx[union_pos]], dtype=np.float32))
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()
    core.report_memory_plan(max(len(c) for c in contexts), len(eval_idx), args)

    tag = exp4_resume_tag(args, anchor_sig, len(union_pos))
    ctx_ckpt = os.path.join(args.resume_dir, f"{tag}_contexts.npz")
    union_global_ids = pool_idx[union_pos]
    if os.path.exists(ctx_ckpt):
        saved = np.load(ctx_ckpt)
        ok = (np.array_equal(saved["union_pos"], union_pos)
              and np.array_equal(saved["union_global_ids"], union_global_ids)
              and all(np.array_equal(saved[f"ctx_{e}"], contexts[e])
                      for e in range(args.n_contexts)))
        if not ok:
            raise SystemExit(f"{ctx_ckpt} does not match the rebuilt contexts -- "
                             "determinism drift; delete the checkpoint to refit.")
        print(f"contexts verified against {ctx_ckpt}")
    else:
        np.savez_compressed(ctx_ckpt, union_pos=union_pos,
                            union_global_ids=union_global_ids,
                            **{f"ctx_{e}": c for e, c in enumerate(contexts)})

    all_rows, timings, probs_by_method = [], {}, {}

    # ---- XGBoost (unweighted + sqrt-weighted), on the union rows ----
    if not args.skip_xgboost:
        variants = [("xgboost", None)]
        if not args.skip_xgboost_sqrt:
            variants.append(("xgboost_sqrt", sqrt_sample_weights(y_fit, n_classes)))
        for method, sample_weight in variants:
            ckpt = os.path.join(
                args.resume_dir,
                f"{tag}_{xgb_param_sig(args)}_{method.replace('xgboost', 'xgb')}.json")
            t0 = time.time()
            if not args.force_refit and os.path.exists(ckpt):
                print(f"Resuming {method} from {ckpt}")
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
                print(f"{method} fitting on {len(X_fit):,} rows ...", flush=True)
                booster.fit(X_fit, y_fit, sample_weight=sample_weight)
                fit_s = time.time() - t0
                booster.save_model(ckpt)
                print(f"{method} fit done in {fit_s:.1f}s -> {ckpt}")
            t1 = time.time()
            probs = full_proba(booster.predict_proba(X_eval), booster.classes_, n_classes)
            preds = np.argmax(probs, axis=1)
            timings[f"{method}_fit_seconds"] = fit_s
            timings[f"{method}_predict_seconds"] = time.time() - t1
            all_rows.extend(core.per_class_table(method, y_eval, preds,
                                                 class_names, tail_classes))
            probs_by_method[method] = probs

    # ---- TabPFN with explicit per-estimator contexts ----
    if not args.skip_tabpfn:
        # INTERLEAVED context->slot assignment. TabPFN assigns its preprocessor
        # variants (2 in the v3 ckpt) to estimator slots in BLOCKS
        # (ensemble.py:1063-1067 _balance), so block-repeating contexts here would
        # lock each context to a single variant (review finding, 0820). With
        # interleaving, context e occupies slots {e, e+E, ...}, pairing contexts
        # and variant blocks evenly whenever repeats is a multiple of the variant
        # count. At E == n_estimators one variant per context is unavoidable --
        # the R3a-vs-R3c comparison stays clean because both share that structure.
        repeats = args.n_estimators // args.n_contexts
        subsample_list = [np.asarray(remapped[i % args.n_contexts], dtype=np.int64)
                          for i in range(args.n_estimators)]
        method = "tabpfn_tacb_bpfit" if args.balance_probabilities else "tabpfn_tacb"
        clf = TabPFNClassifier(
            device=args.device,
            model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            inference_config={"SUBSAMPLE_SAMPLES": subsample_list},
            random_state=args.seed,
            n_estimators=args.n_estimators,
            auto_scale_n_estimators=False,
            fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device,
            balance_probabilities=args.balance_probabilities,
        )
        max_samples = clf.get_inference_config().MAX_NUMBER_OF_SAMPLES
        print(f"TabPFN: union={len(union_pos):,} rows into fit(), per-estimator "
              f"context <= {max(len(c) for c in contexts):,}, checkpoint_max={max_samples:,}, "
              f"n_estimators={args.n_estimators} ({repeats} per context)")
        if not args.ignore_pretraining_limits and len(union_pos) > max_samples:
            raise SystemExit(
                f"the union ({len(union_pos):,} rows) exceeds the checkpoint guard "
                f"{max_samples:,}; pass --ignore-pretraining-limits (validation sees "
                "the union -- per-estimator memory is still one k-row context).")

        # NOTE: no .tabpfn_fit checkpoint here -- save_fitted_tabpfn_model JSON-dumps
        # the constructor params and cannot serialize the list-form SUBSAMPLE_SAMPLES
        # (verified: TypeError on ndarray). fit() only stores the context (~20 min at
        # 1M rows, deterministic, verified against {tag}_contexts.npz), so the
        # PREDICTIONS npy below is the resume point that matters.
        pred_ckpt = os.path.join(
            args.resume_dir, f"{exp4_prediction_tag(tag, args, len(eval_idx))}_tabpfn_probs.npy")
        if not args.force_refit and os.path.exists(pred_ckpt):
            print(f"Resuming TabPFN probabilities from {pred_ckpt} (fit skipped)")
            probs = np.load(pred_ckpt)
            fit_s = pred_s = None
            model_classes = np.flatnonzero(union_counts > 0)
        else:
            t0 = time.time()
            clf.fit(X_fit, y_fit)
            fit_s = time.time() - t0
            print(f"TabPFN fit done in {fit_s:.1f}s (context stored; not checkpointed)")
            t0 = time.time()
            probs = predict_proba_in_batches(clf, X_eval, args.test_batch_size)
            pred_s = time.time() - t0
            np.save(pred_ckpt, probs.astype(np.float32))
            print(f"TabPFN predict done in {pred_s:.1f}s -> {pred_ckpt}")
            model_classes = clf.classes_
        probs = full_proba(probs, model_classes, n_classes)
        preds = np.argmax(probs, axis=1)
        timings["tabpfn_fit_seconds"] = fit_s
        timings["tabpfn_predict_seconds"] = pred_s
        timings["tabpfn_checkpoint_max_samples"] = int(max_samples)
        all_rows.extend(core.per_class_table(method, y_eval, preds,
                                             class_names, tail_classes))
        probs_by_method[method] = probs

        if not args.balance_probabilities:
            # Free one-knob variants: TabPFN's balance_probabilities is a pure
            # post-hoc transform of the averaged probabilities (see docstring).
            # _bp divides by the UNION prior (the faithful library-flag
            # reproduction) -- note that for anchored classes the union prior is
            # ~E-fold SMALLER than the prior each estimator actually conditioned
            # on. _bpctx divides by the per-context prior (targets), which is
            # what the estimators saw (review finding, 0820).
            probs_bp = balance_probs_post_hoc(probs, union_counts)
            all_rows.extend(core.per_class_table("tabpfn_tacb_bp", y_eval,
                                                 np.argmax(probs_bp, axis=1),
                                                 class_names, tail_classes))
            probs_by_method["tabpfn_tacb_bp"] = probs_bp
            probs_bpctx = balance_probs_post_hoc(probs, targets)
            all_rows.extend(core.per_class_table("tabpfn_tacb_bpctx", y_eval,
                                                 np.argmax(probs_bpctx, axis=1),
                                                 class_names, tail_classes))
            probs_by_method["tabpfn_tacb_bpctx"] = probs_bpctx

    # ---- artifacts ----
    tb = args.test_batch_size if args.test_batch_size > 0 else len(eval_idx)
    timings.update({
        "experiment": args.experiment,
        "target_dataset": args.target_dataset,
        "eval_split": args.eval_split,
        "train_pool_rows": int(len(pool_idx)),
        "train_cap_policy": cap_policy,
        "train_split": args.train_split,
        "n_contexts": int(args.n_contexts),
        "n_estimators": int(args.n_estimators),
        "context_size": int(k),
        "context_rows_actual": [int(len(c)) for c in contexts],
        "anchor_threshold": int(args.anchor_threshold),
        "anchor_classes_resolved": [class_names[i] for i in np.flatnonzero(anchored)],
        "anchor_rows_per_context": int(targets[anchored].sum()),
        "context_floor": int(args.context_floor),
        "union_rows": int(len(union_pos)),
        "union_pool_coverage_pct": round(100 * len(union_pos) / len(pool_idx), 2),
        "balance_probabilities": bool(args.balance_probabilities),
        "eval_rows": int(len(eval_idx)),
        "test_batch_size": int(tb),
        "test_batches": int(math.ceil(len(eval_idx) / tb)),
        "predicted_peak_gib": round(core.estimate_peak_gib(
            max(len(c) for c in contexts), min(tb, len(eval_idx)), args.fit_mode)[0], 2),
        "resume_tag": tag,
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
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp4_tailbag{split_tag}{eval_tag}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    union_rows_audit = pd.DataFrame(
        [{"split": "train_used", "class": class_names[i], "count": int(union_counts[i])}
         for i in range(n_classes) if union_counts[i] > 0])
    pd.concat([split_audit, union_rows_audit], ignore_index=True, sort=False).to_csv(
        os.path.join(out_dir, "split_audit.csv"), index=False)
    composition.to_csv(os.path.join(out_dir, "context_composition.csv"), index=False)
    drift_df.to_csv(os.path.join(out_dir, "train_test_drift_diagnostic.csv"), index=False)
    for method, probs in probs_by_method.items():
        save_probs(out_dir, method, probs, y_eval, class_names, union_counts, targets)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([timings], f, indent=2)
    print(f"\nWrote {out_dir}")

    if args.no_save_models:
        print(f"--no-save-models: fitted models stay in {args.resume_dir} only.")
        return out_dir
    model_tag = os.path.basename(out_dir)
    for suffix in ("xgb.json", "xgb_sqrt.json"):
        src = os.path.join(args.resume_dir, f"{tag}_{xgb_param_sig(args)}_{suffix}")
        if os.path.exists(src):
            dst = os.path.join(args.models_dir, f"{model_tag}_{suffix}")
            shutil.copy(src, dst)
            print(f"Saved {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-size", type=int, default=1_000_000,
                   help="Per-estimator context budget k. -1 = the whole train pool "
                        "as one context (for full-pool XGB bar runs with --skip-tabpfn).")
    p.add_argument("--n-contexts", type=int, default=1,
                   help="E = number of DISTINCT contexts. 0 = auto: the smallest E with "
                        "guaranteed full head coverage (E*t_c >= pool_c for every "
                        "non-anchored class). --n-estimators must be a multiple of E; "
                        "contexts are assigned to estimator slots INTERLEAVED so each "
                        "context meets TabPFN's block-assigned preprocessor variants "
                        "evenly (at E == n-estimators, one variant per context).")
    p.add_argument("--anchor-threshold", type=int, default=0,
                   help="Classes whose train-pool count <= this are ANCHORED: 100%% of "
                        "their natural pool rows in EVERY context. 0 = off. 5000 selects "
                        "exactly the winnable tails (cic2018 web_attacks 1,522; ton_iot "
                        "mitm 3,607 / ransomware 2,382; bot_iot theft 969).")
    p.add_argument("--anchor-classes", default="none",
                   help="'none' (default), 'tail' = the target's tail_classes registry "
                        "(NOTE: for cic2018 that includes bot/infiltration at 100k+ pool "
                        "rows -- usually NOT what you want; prefer --anchor-threshold), "
                        "or a comma-separated class list.")
    p.add_argument("--context-floor", type=int, default=0,
                   help="Per-context minimum rows for each non-anchored present class "
                        "(capped at the class's pool count). 0 = off.")
    p.add_argument("--balance-probabilities", action="store_true",
                   help="Build TabPFN's post-hoc class-prior correction INTO the model. "
                        "Normally unnecessary: the run always derives the identical "
                        "tabpfn_tacb_bp column post-hoc from the dumped probabilities.")
    p.add_argument("--skip-xgboost-sqrt", action="store_true",
                   help="Drop the sqrt-class-weighted XGBoost column (s44's weighting).")
    p.add_argument("--eval-split", default="test", choices=["test", "val"],
                   help="test = far-future 20%% (the honest headline). val = near-future "
                        "20%% (temporal-drift axis; predict-only on a resumed fit; "
                        "refused with --train-split train+val).")
    p.set_defaults(
        max_train_samples=-1,          # the context builder must see the WHOLE pool,
        #                                otherwise the anchor cannot see all tail rows
        n_estimators=4,                # exp1's value, for one-knob comparability
        fit_mode="fit_preprocessors",
        test_batch_size=0,
        subsample_samples=0,           # guarded: exp4 builds contexts itself
    )
    args = p.parse_args()
    run_exp4(args)


if __name__ == "__main__":
    main()

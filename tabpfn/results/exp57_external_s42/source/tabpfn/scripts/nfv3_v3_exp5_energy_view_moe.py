#!/usr/bin/env python3
"""EXP5 -- Energy-weighted View-MoE over distribution-tilted TabPFN contexts.

Pre-registration: manuscript/report/0820.md SS8.  Forked from
nfv3_v3_exp4_tail_anchored_bagging.py (helpers copied, never imported -- exp4 is
frozen with runs).  nfv3_v3_common is reused import-only.

What this measures
------------------
s44's structure (experts + energy selection) with the two recorded killers
removed, and TabPFN as the expert backbone:

  * A "view" is NOT a class partition (s43/s44 ownership).  Every view contains
    every present class; views differ only in the class MIXTURE:
    per-class budget proportional to pool_c**alpha, capped at pool_c (no
    oversampling), largest-remainder + waterfill redistribution.
      alpha=1.0  natural/head view (proportional -- exp1's distribution)
      alpha=0.5  sqrt view (middle tilt)
      alpha=0.0  balanced view (tail-max given no duplication)
    Same label space everywhere -> (a) a wrong selection yields a worse
    prediction, never an impossible one; (b) equal class counts keep the
    energy logsumexp baselines comparable (the s01/s20 padding trap).
    Rows come from ONE shared per-class permutation (seed+950+class_id, the
    same band exp4's builder uses), each view taking a prefix -- so two views
    differ in per-class COUNTS only, not in which rows realize a given count.
  * Selection is thresholdless (failure mode #2): per view a scalar score s_v(x)
    (LOWER = pick), weights w_v(x) = softmax(-s_v(x)/tau).  tau -> inf is the
    uniform average (pure tilted bagging); tau -> 0 is argmin (s44's hard
    selection).  THREE selectors are computed from the same logits (0820 SS8a):
      energy   -T*logsumexp(l/T) -- the pre-registered score.  MEASURED
               DEGENERATE on TabPFN-v3 (its head is log-softmax-normalized:
               logsumexp = +2.5e-4 +/- 7e-5), kept as a control; meaningful
               on XGB margins (the XGB primary).
      nmaxp    -max_c l_c = -max log-prob on TabPFN (MSP family) -- the
               TabPFN primary selector for judgment gate 1.
      nmargin  -(top1 - top2) -- secondary.
    Columns per selector: {backbone}_{sel}_soft_{tau} for tau in --sel-taus and
    {backbone}_{sel}_amin; plus viewavg / viewavg_bpv.  No TTA (s23).
  * TabPFN logits: clf._raw_predict(X, return_logits=False,
    return_raw_logits=True) -> (n_estimators, n, C) raw logits, class-order
    restored, before temperature/averaging (classifier.py:1337/1632).
    Probabilities are rebuilt with clf.logits_to_probabilities(...) -- the
    library's own post-processing -- and smoke-verified == predict_proba.
  * XGB mirror: an unweighted XGBoost per view on the SAME rows; energy from
    inplace_predict(predict_type="margin"), probs = softmax(margins)
    (== predict_proba for multi:softprob, smoke-verified).  If the XGB mirror
    gains the same from selection, the effect is the view composition, not ICL.

Closed-set only: no unseen holdout, no OOD axis (user-fixed scope, 0820).

One-knob chains
---------------
  CAVEAT (review finding, 0820 SS8b): the alpha=1.0 view at k=1M matches
  exp1/R0 in per-class COUNTS only -- the row REALIZATION differs (exp1/R0:
  stratified_subset rng.choice at seed+850 over the 1M pre-cap; exp5: seed+950
  permutation prefix over the FULL pool).  R0 is therefore a counts-matched,
  realization-mismatched external reference; any a100-vs-R0 delta carries the
  row-realization knob (the same recorded gap exp4's R0 bridge closes for
  exp1).  The exact in-band row-set match is exp4's builder context e=0
  (R3c's first slice), which has no standalone recorded run.  The CLEAN
  comparisons are the within-run ones: singles vs viewavg vs soft vs argmin
  share everything but the combiner.

    # D1 (diagnostic, m=1 per view; judgment gates in 0820.md SS8)
    python tabpfn/nfv3_v3_exp5_energy_view_moe.py --target-dataset cic2018 \\
        --context-size 1000000 --view-alphas 1.0,0.5,0.0 --n-estimators 1 \\
        --test-cap-per-class 0 --fit-mode fit_with_cache --test-batch-size 500000

    # D2 (only if D1 passes gate 1: m=4 per view, win bar 0.293 applies)
    python tabpfn/nfv3_v3_exp5_energy_view_moe.py --target-dataset cic2018 \\
        --context-size 1000000 --view-alphas 1.0,0.5,0.0 --n-estimators 4 \\
        --test-cap-per-class 0 --fit-mode fit_with_cache --test-batch-size 500000

Cost model (RTX 4090 24GB, cic2018 uncapped test 4.02M rows, k=1M)
------------------------------------------------------------------
  Per view: cache-mode fit ~= 1,250s * m/4;  predict ~= 1,322s * m + batch
  overhead (KV re-upload ~11.5 GiB per estimator per 500k batch).
  D1 (3 views, m=1) ~= 1.5h.  ALWAYS export
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.  Single-call 4M predict
  OOMs even in cache mode (0820 SS7) -- keep --test-batch-size 500000.

Artifacts
---------
  per_class_metrics.csv       every column (singles / avg / soft / argmin / bpv)
  view_composition.csv        per (view, class): rows, share, pool
  selection_view_diag.csv     s23-analog: per (backbone, selector, class) --
                              per-view mean score, per-view recall, argmin pick
                              shares, argmin/avg/oracle correctness
  probs_<column>.npz          averaged probabilities per column (y_true inside)
  logits_<backbone>_v<atag>.npz  raw per-estimator logits (fp16) + classes --
                              the resume point and the post-hoc raw material
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

SEED_BAND_CONTEXT = 950  # same band as exp4: per-class permutation seed+950+cid


# ---------------------------------------------------------------------------
# View construction
# ---------------------------------------------------------------------------

def _powerlaw_targets(counts, budget, alpha, class_names):
    """Per-class targets summing to min(budget, pool): allocate proportional to
    counts**alpha over classes with headroom, cap at pool counts, redistribute
    (waterfill).  alpha=1 reproduces natural proportions; alpha=0 is uniform
    up to pool caps.  Every present class must end with >= 1 row (same label
    space across views is a design invariant -- see docstring)."""
    counts = np.asarray(counts, dtype=np.int64)
    present = counts > 0
    total = int(counts.sum())
    budget = min(int(budget), total)
    n_present = int(present.sum())
    if budget < n_present:
        raise SystemExit(f"--context-size {budget} < {n_present} present classes.")
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
    assert int(target.sum()) == budget, "waterfill targets != budget"
    starved = present & (target < 1)
    if starved.any():
        raise SystemExit(
            f"alpha={alpha}: budget {budget} leaves classes with 0 rows "
            f"({[class_names[i] for i in np.flatnonzero(starved)]}); raise "
            "--context-size (same-label-space invariant).")
    return target


def build_views(labels, n_classes, class_names, k, alphas, seed):
    """Build one context per alpha over a shared per-class permutation.

    Returns (views, targets_per_view, composition):
      views: list of sorted int64 position arrays into the pool.
      targets_per_view: list of (n_classes,) target vectors.
      composition: DataFrame, one row per (view, class) plus a `pool` sentinel.
    Views take PREFIXES of one permutation -> a class's first t rows are shared
    by every view that wants >= t of them (mixture is the only knob)."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=n_classes)
    present = counts > 0
    perms = {int(c): np.random.default_rng(seed + SEED_BAND_CONTEXT + int(c))
             .permutation(np.flatnonzero(labels == c))
             for c in np.flatnonzero(present)}
    views, targets_all, rows = [], [], []
    for alpha in alphas:
        targets = _powerlaw_targets(counts, k, alpha, class_names)
        parts = [perms[int(c)][: int(targets[c])]
                 for c in np.flatnonzero(targets > 0)]
        view = np.sort(np.concatenate(parts))
        views.append(view)
        targets_all.append(targets)
        for c in np.flatnonzero(present):
            rows.append({
                "view": f"a{int(round(alpha * 100)):03d}", "alpha": alpha,
                "class": class_names[c], "rows": int(targets[c]),
                "pool_rows": int(counts[c]),
                "share_of_context_pct": round(100.0 * targets[c] / max(k, 1), 4),
            })
    for c in np.flatnonzero(present):
        rows.append({"view": "pool", "alpha": np.nan, "class": class_names[c],
                     "rows": int(counts[c]), "pool_rows": int(counts[c]),
                     "share_of_context_pct":
                         round(100.0 * counts[c] / max(int(counts.sum()), 1), 4)})
    return views, targets_all, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probability / energy helpers (copied from exp4 where noted -- exp4 is frozen)
# ---------------------------------------------------------------------------

def full_proba(proba, model_classes, n_classes):
    """exp4:329 -- expand to the global class space, zero-filling."""
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=np.int64)] = proba
    return out


def balance_probs_post_hoc(probs, class_counts):
    """exp4:337 -- divide by fit priors, renormalize."""
    prior = class_counts / class_counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(prior > 0, probs / prior, 0.0)
    denom = out.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return (out / denom).astype(np.float32)


def energy_from_logits(logits, T):
    """Project convention: E = -T * logsumexp(logits / T) over the class dim.
    logits: (..., C) float array.  Lower E = more in-distribution."""
    z = np.asarray(logits, dtype=np.float64) / T
    zmax = z.max(axis=-1, keepdims=True)
    return (-T * (np.log(np.exp(z - zmax).sum(axis=-1)) + zmax[..., 0])).astype(
        np.float32)


SELECTORS = ("energy", "nmaxp", "nmargin")
PRIMARY_SELECTOR = {"tabpfn": "nmaxp", "xgb": "energy"}  # judgment gate 1 (0820 SS8a)


def selector_scores_from_logits(logits, energy_T):
    """Per-row selection scores from a logit/margin vector; LOWER = pick me.

    energy   -T*logsumexp(l/T).  Meaningful on XGB margins; measured DEGENERATE
             on TabPFN-v3 raw logits (the head is log-softmax-normalized:
             logsumexp = +2.5e-4 +/- 7e-5 -- 0820 SS8a), kept as a control.
    nmaxp    -max_c l_c.  On log-prob logits this is -max log-prob (MSP
             family) -- the TabPFN primary selector.
    nmargin  -(top1 - top2).
    """
    logits = np.asarray(logits)
    top2 = np.partition(logits, -2, axis=-1)[..., -2:]
    return {
        "energy": energy_from_logits(logits, energy_T),
        "nmaxp": (-top2[..., 1]).astype(np.float32),
        "nmargin": (-(top2[..., 1] - top2[..., 0])).astype(np.float32),
    }


def softmax_weights(energies, tau):
    """energies: (V, n).  Returns (V, n) weights = softmax(-E/tau) over views."""
    z = -np.asarray(energies, dtype=np.float64) / tau
    z -= z.max(axis=0, keepdims=True)
    w = np.exp(z)
    return (w / w.sum(axis=0, keepdims=True)).astype(np.float32)


def combine_soft(probs_views, energies, tau):
    """probs_views: (V, n, C); energies: (V, n) -> (n, C) weighted average."""
    w = softmax_weights(energies, tau)
    return np.einsum("vn,vnc->nc", w, probs_views).astype(np.float32)


def combine_argmin(probs_views, energies):
    pick = np.argmin(energies, axis=0)                      # (n,)
    return probs_views[pick, np.arange(probs_views.shape[1])], pick


# ---------------------------------------------------------------------------
# Resume tags
# ---------------------------------------------------------------------------

def _h8(path):
    return hashlib.sha1(os.path.realpath(path).encode()).hexdigest()[:8]


def exp5_fit_tag(args, atag, view_n):
    """Per-view fit identity: everything that changes the view's FIT input.
    Eval settings live in the prediction tag; the fit tag keys the XGB booster
    checkpoint (reused across eval splits) and the view-determinism npz.
    TabPFN fits are NOT checkpointed (deterministic; a val re-score refits all
    views, ~1,250s*m/4 each in cache mode -- fit_with_cache state includes the
    ~11.5GiB/estimator KV cache, impractical on disk)."""
    return (f"exp5_{args.target_dataset}_mts{args.max_train_samples}"
            f"_seed{args.seed}_k{args.context_size}_{atag}_m{args.n_estimators}"
            f"_ipl{int(args.ignore_pretraining_limits)}_fm{args.fit_mode}"
            f"_sp{args.train_split.replace('+', '')}_n{view_n}"
            f"_d{_h8(args.data)}_m{_h8(args.model_path)}")


def xgb_param_sig(args):
    """exp4:408 -- XGB checkpoints keyed on the seven hyperparameters."""
    return (f"xne{args.xgb_n_estimators}_xd{args.xgb_max_depth}"
            f"_xlr{args.xgb_learning_rate}_xss{args.xgb_subsample}"
            f"_xcs{args.xgb_colsample_bytree}_xmcw{args.xgb_min_child_weight}"
            f"_xrl{args.xgb_reg_lambda}")


def exp5_pred_tag(fit_tag, args, n_eval):
    # energy_T is in the PREDICTION tag (review finding, 0820 SS8b): the stored
    # score_energy depends on it, and a resumed checkpoint must never silently
    # serve scores computed under a different T.  It is absent from the fit
    # tag on purpose -- fits do not depend on it.
    return (f"{fit_tag}_ev{args.eval_split}_tc{args.test_cap_per_class}"
            f"_eT{args.energy_T:g}_nt{n_eval}")


# ---------------------------------------------------------------------------
# Per-view prediction (TabPFN raw logits, batched)
# ---------------------------------------------------------------------------

def tabpfn_predict_view(clf, X_eval, batch_size, energy_T):
    """Batched raw-logit prediction for one fitted view.

    Returns (probs, scores, raw_fp16):
      probs   (n, C_view) float32 -- library post-processing via
              logits_to_probabilities (== predict_proba up to its final
              16-decimal rounding; smoke-verified).
      scores  dict selector -> (n,) float32, estimator-mean (fp32 source).
      raw_fp16 (m, n, C_view) float16 -- dumped for post-hoc rules.
    """
    import torch
    if batch_size <= 0:
        batch_size = len(X_eval)
    n_rows = len(X_eval)
    n_batches = (n_rows + batch_size - 1) // batch_size
    probs_out, raw_out = [], []
    scores_out = {sel: [] for sel in SELECTORS}
    for bi, start in enumerate(range(0, n_rows, batch_size), 1):
        stop = min(start + batch_size, n_rows)
        raw = clf._raw_predict(X_eval[start:stop], return_logits=False,
                               return_raw_logits=True)      # (m, b, C)
        raw = raw.detach().float().cpu()
        probs = clf.logits_to_probabilities(raw).detach().float().cpu().numpy()
        raw = raw.numpy()
        probs_out.append(probs.astype(np.float32))
        for sel, s in selector_scores_from_logits(raw, energy_T).items():
            scores_out[sel].append(s.mean(axis=0))          # (m,b) -> (b,)
        raw_out.append(raw.astype(np.float16))
        if bi == 1 or bi % 5 == 0 or stop == n_rows:
            print(f"  view predict batch {bi}/{n_batches}: rows {start:,}:{stop:,}",
                  flush=True)
        del raw
        # empty_cache only if this process already holds a CUDA context --
        # calling it first would CREATE one (~300MB) and could evict a
        # concurrent GPU run during its peak phase.
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    scores = {sel: np.concatenate(chunks).astype(np.float32)
              for sel, chunks in scores_out.items()}
    return np.concatenate(probs_out), scores, np.concatenate(raw_out, axis=1)


def save_probs(out_dir, method, probs, y_true, class_names):
    np.savez_compressed(
        os.path.join(out_dir, f"probs_{method}.npz"),
        probs=probs.astype(np.float32), y_true=y_true.astype(np.int64),
        class_names=np.asarray(class_names))


# ---------------------------------------------------------------------------
# Selection columns + s23-analog diagnostics
# ---------------------------------------------------------------------------

def selection_columns(backbone, atags, probs_views, scores_by_sel, taus,
                      targets_per_view, n_classes):
    """All combiner columns from per-view probs (V,n,C) + per-selector scores
    {sel: (V,n)}.  Column names: singles, viewavg, viewavg_bpv, then per
    selector `{backbone}_{sel}_soft_{tau}` and `{backbone}_{sel}_amin`."""
    cols = {}
    for atag, p in zip(atags, probs_views):
        cols[f"{backbone}_v{atag}"] = p
    cols[f"{backbone}_viewavg"] = probs_views.mean(axis=0).astype(np.float32)
    bpv = np.stack([balance_probs_post_hoc(p, t)
                    for p, t in zip(probs_views, targets_per_view)])
    cols[f"{backbone}_viewavg_bpv"] = bpv.mean(axis=0).astype(np.float32)
    picks = {}
    for sel, scores in scores_by_sel.items():
        for tau in taus:
            tag = f"t{tau:g}".replace(".", "p")
            cols[f"{backbone}_{sel}_soft_{tag}"] = combine_soft(
                probs_views, scores, tau)
        argmin_probs, pick = combine_argmin(probs_views, scores)
        cols[f"{backbone}_{sel}_amin"] = argmin_probs
        picks[sel] = pick
    return cols, picks


def selection_diag_rows(backbone, atags, probs_views, scores_by_sel, picks,
                        y_eval, class_names):
    """s23-analog table per selector: does a LOW score identify the view that
    is RIGHT on this row?  Columns: per-view mean score / recall / pick share,
    plus argmin-vs-average-vs-oracle correctness."""
    preds_views = probs_views.argmax(axis=2)                # (V, n)
    correct_views = preds_views == y_eval[None, :]          # (V, n)
    avg_correct = probs_views.mean(axis=0).argmax(axis=1) == y_eval
    oracle = correct_views.any(axis=0)
    rows = []
    groups = [("__all__", np.arange(len(y_eval)))] + [
        (class_names[c], np.flatnonzero(y_eval == c))
        for c in np.unique(y_eval)]
    for sel, scores in scores_by_sel.items():
        pick = picks[sel]
        pick_correct = correct_views[pick, np.arange(len(y_eval))]
        for name, idx in groups:
            if len(idx) == 0:
                continue
            row = {"backbone": backbone, "selector": sel,
                   "primary": sel == PRIMARY_SELECTOR.get(backbone),
                   "class": name, "n": int(len(idx)),
                   "argmin_correct": round(float(pick_correct[idx].mean()), 4),
                   "avg_correct": round(float(avg_correct[idx].mean()), 4),
                   "oracle_view_correct": round(float(oracle[idx].mean()), 4)}
            for v, atag in enumerate(atags):
                row[f"score_mean_{atag}"] = round(float(scores[v, idx].mean()), 4)
                row[f"recall_{atag}"] = round(float(correct_views[v, idx].mean()), 4)
                row[f"pick_share_{atag}"] = round(float((pick[idx] == v).mean()), 4)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Run body
# ---------------------------------------------------------------------------

def run_exp5(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp5_energy_view_moe"
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)

    if args.subsample_samples:
        raise SystemExit("exp5 fits one TabPFN per view; --subsample-samples "
                         "must stay 0 (no SUBSAMPLE_SAMPLES hook here).")
    if args.context_size <= 0:
        raise SystemExit("exp5 needs an explicit --context-size > 0: views share "
                         "one budget so the mixture is the only knob (-1 whole-pool "
                         "is exp4/R2 territory).")
    if args.eval_split == "val" and args.train_split != "train":
        raise SystemExit("--eval-split val is only valid with --train-split train.")
    alphas = []
    for s in str(args.view_alphas).split(","):
        s = s.strip()
        if s:
            a = float(s)
            if not (0.0 <= a <= 2.0):
                raise SystemExit(f"--view-alphas entry {a} outside [0, 2].")
            alphas.append(a)
    if len(alphas) < 2:
        raise SystemExit("--view-alphas needs >= 2 views for any selection to exist.")
    if len(set(alphas)) != len(alphas):
        raise SystemExit(f"--view-alphas has duplicates: {alphas}")
    taus = [float(s) for s in str(args.sel_taus).split(",") if s.strip()]
    if any(t <= 0 for t in taus):
        raise SystemExit(f"--sel-taus must be > 0, got {taus}")
    if args.n_estimators < 1:
        raise SystemExit("--n-estimators (per-view m) must be >= 1.")
    atags = [f"a{int(round(a * 100)):03d}" for a in alphas]
    if len(set(atags)) != len(atags):
        raise SystemExit(
            f"--view-alphas {alphas} collide at the 0.01 tag resolution -> "
            f"{atags}; view identity (checkpoints, columns) is the rounded "
            "tag, so alphas must differ by >= 0.01.")

    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, \
        split_audit, label_fn = cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    print(f"target={args.target_dataset}  classes ({n_classes}): {class_names}")
    print(f"split rows: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    if args.train_split == "train+val":
        train_idx = np.sort(np.concatenate([train_idx, val_idx]))
        print(f"--train-split train+val: train pool {len(train_idx):,} (60% + 20%)")

    # ---- evaluation set ----
    eval_pool_idx = test_idx if args.eval_split == "test" else val_idx
    eval_idx = core.cap_per_class(eval_pool_idx, label_fn(eval_pool_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_eval = label_fn(eval_idx)
    print(f"eval split={args.eval_split}: {len(eval_idx):,} rows "
          f"(cap_per_class={args.test_cap_per_class})")
    X_eval = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))

    print(f"\n--- pre-fit diagnostic: train/{args.eval_split} feature drift per class ---")
    drift_df = core.diagnose_train_test_drift(X, class_names, train_idx,
                                              eval_pool_idx, label_fn)
    print(drift_df.to_string(index=False))

    # ---- train pool (frozen semantics: stratified proportional pre-cap) ----
    pool_idx = train_idx
    cap_policy = "none"
    if args.max_train_samples > 0 and args.max_train_samples < len(pool_idx):
        pool_idx = core.stratified_subset(pool_idx, label_fn(pool_idx), n_classes,
                                          args.max_train_samples, args.seed + 850)
        cap_policy = "stratified_ratio_preserving"
    y_pool = label_fn(pool_idx)
    print(f"train pool: {len(pool_idx):,} of {len(train_idx):,} (cap_policy={cap_policy})")

    # ---- build views ----
    k = args.context_size
    views, targets_per_view, composition = build_views(
        y_pool, n_classes, class_names, k, alphas, args.seed)
    for v, (view, targets) in enumerate(zip(views, targets_per_view)):
        got = np.bincount(y_pool[view], minlength=n_classes)
        assert np.array_equal(got, targets), f"view {atags[v]}: composition mismatch"
    print(f"views: {len(views)} x k={k:,}  alphas={alphas}")
    print(composition[composition["view"] != "pool"].to_string(index=False))

    X_views = [np.nan_to_num(np.asarray(X[pool_idx[view]], dtype=np.float32))
               for view in views]
    y_views = [y_pool[view] for view in views]
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()
    core.report_memory_plan(k, len(eval_idx), args)

    # every view must expose the same class set (energy-comparability invariant)
    class_sets = [tuple(np.flatnonzero(np.bincount(y_v, minlength=n_classes) > 0))
                  for y_v in y_views]
    assert len(set(class_sets)) == 1, (
        f"views expose different class sets {class_sets} -- the >=1-per-class "
        "guard in _powerlaw_targets should make this impossible.")

    # View-determinism checkpoints (exp4's contexts.npz discipline, review
    # finding 0820 SS8b): a resumed logits npz is only valid if today's rebuilt
    # view is byte-identical to the one that produced it.
    for v, atag in enumerate(atags):
        vc = os.path.join(args.resume_dir,
                          f"{exp5_fit_tag(args, atag, len(views[v]))}_view.npz")
        gids = pool_idx[views[v]]
        if os.path.exists(vc):
            saved = np.load(vc)
            if not (np.array_equal(saved["view_pos"], views[v])
                    and np.array_equal(saved["view_global_ids"], gids)):
                raise SystemExit(f"{vc} does not match the rebuilt view -- "
                                 "determinism drift; delete the checkpoint to refit.")
            print(f"[{atag}] view verified against {vc}")
        else:
            np.savez_compressed(vc, view_pos=views[v], view_global_ids=gids)

    all_rows, timings, probs_by_method = [], {}, {}
    tb = args.test_batch_size if args.test_batch_size > 0 else len(eval_idx)

    # ================= TabPFN: one fit per view =================
    if not args.skip_tabpfn:
        probs_views = []
        scores_views = {sel: [] for sel in SELECTORS}
        for v, atag in enumerate(atags):
            fit_tag = exp5_fit_tag(args, atag, len(views[v]))
            logit_ckpt = os.path.join(
                args.resume_dir,
                f"{exp5_pred_tag(fit_tag, args, len(eval_idx))}_logits.npz")
            if not args.force_refit and os.path.exists(logit_ckpt):
                # probs/scores are stored fp32 alongside the fp16 raw logits, so
                # a resumed run is byte-identical to the run that wrote them.
                print(f"[{atag}] resuming probs/scores/logits from {logit_ckpt}")
                saved = np.load(logit_ckpt)
                if float(saved["energy_T"]) != float(args.energy_T):
                    raise SystemExit(
                        f"{logit_ckpt} stores scores for energy_T="
                        f"{float(saved['energy_T'])} but this run asked for "
                        f"{args.energy_T}; tag/npz disagree -- delete the "
                        "checkpoint or pass --force-refit.")
                probs = saved["probs"].astype(np.float32)
                scores = {sel: saved[f"score_{sel}"].astype(np.float32)
                          for sel in SELECTORS}
                model_classes = saved["classes"]
                fit_s = pred_s = None
            else:
                clf = TabPFNClassifier(
                    device=args.device,
                    model_path=args.model_path,
                    ignore_pretraining_limits=args.ignore_pretraining_limits,
                    random_state=args.seed,
                    n_estimators=args.n_estimators,
                    auto_scale_n_estimators=False,
                    fit_mode=args.fit_mode,
                    keep_cache_on_device=args.keep_cache_on_device,
                )
                max_samples = clf.get_inference_config().MAX_NUMBER_OF_SAMPLES
                if not args.ignore_pretraining_limits and len(views[v]) > max_samples:
                    raise SystemExit(f"view {atag} ({len(views[v]):,} rows) exceeds "
                                     f"the checkpoint guard {max_samples:,}.")
                print(f"[{atag}] TabPFN fit on {len(views[v]):,} rows "
                      f"(m={args.n_estimators}, fit_mode={args.fit_mode}) ...",
                      flush=True)
                t0 = time.time()
                clf.fit(X_views[v], y_views[v])
                fit_s = time.time() - t0
                print(f"[{atag}] fit done in {fit_s:.1f}s")
                t0 = time.time()
                probs, scores, raw16 = tabpfn_predict_view(
                    clf, X_eval, tb, args.energy_T)
                pred_s = time.time() - t0
                model_classes = np.asarray(clf.classes_)
                np.savez_compressed(logit_ckpt, raw_logits=raw16,
                                    probs=probs.astype(np.float32),
                                    classes=model_classes,
                                    softmax_temperature=clf.softmax_temperature_,
                                    energy_T=float(args.energy_T),
                                    **{f"score_{sel}": s for sel, s in scores.items()})
                print(f"[{atag}] predict done in {pred_s:.1f}s -> {logit_ckpt}")
                del clf, raw16
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            probs_views.append(full_proba(probs, model_classes, n_classes))
            for sel in SELECTORS:
                scores_views[sel].append(scores[sel])
            timings[f"tabpfn_{atag}_fit_seconds"] = fit_s
            timings[f"tabpfn_{atag}_predict_seconds"] = pred_s
        probs_views = np.stack(probs_views)                 # (V, n, C)
        scores_by_sel = {sel: np.stack(chunks)              # (V, n)
                         for sel, chunks in scores_views.items()}
        cols, picks = selection_columns("tabpfn", atags, probs_views,
                                        scores_by_sel, taus, targets_per_view,
                                        n_classes)
        for method, probs in cols.items():
            all_rows.extend(core.per_class_table(method, y_eval,
                                                 np.argmax(probs, axis=1),
                                                 class_names, tail_classes))
            probs_by_method[method] = probs
        diag_rows = selection_diag_rows("tabpfn", atags, probs_views,
                                        scores_by_sel, picks, y_eval, class_names)
        del probs_views
        gc.collect()
    else:
        diag_rows = []

    # ================= XGB mirror: one booster per view =================
    if not args.skip_xgboost:
        probs_views = []
        scores_views = {sel: [] for sel in SELECTORS}
        for v, atag in enumerate(atags):
            fit_tag = exp5_fit_tag(args, atag, len(views[v]))
            ckpt = os.path.join(args.resume_dir,
                                f"{fit_tag}_{xgb_param_sig(args)}_xgb.json")
            t0 = time.time()
            if not args.force_refit and os.path.exists(ckpt):
                print(f"[{atag}] resuming XGB from {ckpt}")
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
                print(f"[{atag}] XGB fit on {len(X_views[v]):,} rows ...", flush=True)
                booster.fit(X_views[v], y_views[v])
                fit_s = time.time() - t0
                booster.save_model(ckpt)
                print(f"[{atag}] XGB fit done in {fit_s:.1f}s")
            t1 = time.time()
            margins = booster.get_booster().inplace_predict(
                X_eval, predict_type="margin")               # (n, C_view)
            zmax = margins.max(axis=1, keepdims=True)
            probs = (np.exp(margins - zmax)
                     / np.exp(margins - zmax).sum(axis=1, keepdims=True))
            probs_views.append(full_proba(probs.astype(np.float32),
                                          booster.classes_, n_classes))
            for sel, s in selector_scores_from_logits(margins, args.energy_T).items():
                scores_views[sel].append(s)
            timings[f"xgb_{atag}_fit_seconds"] = fit_s
            timings[f"xgb_{atag}_predict_seconds"] = time.time() - t1
        probs_views = np.stack(probs_views)
        scores_by_sel = {sel: np.stack(chunks)
                         for sel, chunks in scores_views.items()}
        cols, picks = selection_columns("xgb", atags, probs_views,
                                        scores_by_sel, taus, targets_per_view,
                                        n_classes)
        for method, probs in cols.items():
            all_rows.extend(core.per_class_table(method, y_eval,
                                                 np.argmax(probs, axis=1),
                                                 class_names, tail_classes))
            probs_by_method[method] = probs
        diag_rows.extend(selection_diag_rows("xgb", atags, probs_views,
                                             scores_by_sel, picks, y_eval,
                                             class_names))
        del probs_views
        gc.collect()

    # ================= artifacts =================
    timings.update({
        "experiment": args.experiment,
        "target_dataset": args.target_dataset,
        "eval_split": args.eval_split,
        "train_pool_rows": int(len(pool_idx)),
        "train_cap_policy": cap_policy,
        "train_split": args.train_split,
        "view_alphas": alphas,
        "n_views": len(alphas),
        "n_estimators_per_view": int(args.n_estimators),
        "context_size": int(k),
        "energy_T": float(args.energy_T),
        "sel_taus": taus,
        "eval_rows": int(len(eval_idx)),
        "test_batch_size": int(tb),
        "test_batches": int(math.ceil(len(eval_idx) / tb)),
        "predicted_peak_gib": round(core.estimate_peak_gib(
            k, min(tb, len(eval_idx)), args.fit_mode)[0], 2),
    })

    table = pd.DataFrame(all_rows)
    if len(table):
        summary = table[table["class"].isin(["macro_avg", "weighted_avg", "tail_avg"])]
        print("\n=== summary (macro / weighted / tail F1) ===")
        print(summary.pivot(index="class", columns="method", values="f1").to_string())
    diag_df = pd.DataFrame(diag_rows)
    if len(diag_df):
        print("\n=== selection view diagnostics (s23-analog) ===")
        print(diag_df.to_string(index=False))

    ts = time.strftime("%Y%m%d_%H%M%S")
    split_tag = "" if args.train_split == "train" else "_82"
    eval_tag = "" if args.eval_split == "test" else "_evval"
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp5_viewmoe{split_tag}{eval_tag}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    composition.to_csv(os.path.join(out_dir, "view_composition.csv"), index=False)
    diag_df.to_csv(os.path.join(out_dir, "selection_view_diag.csv"), index=False)
    drift_df.to_csv(os.path.join(out_dir, "train_test_drift_diagnostic.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "split_audit.csv"), index=False)
    for method, probs in probs_by_method.items():
        save_probs(out_dir, method, probs, y_eval, class_names)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([timings], f, indent=2)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-size", type=int, default=1_000_000,
                   help="Per-view row budget k, shared by every view so the "
                        "mixture is the only knob. Must be > 0 (-1 whole-pool "
                        "is exp2/exp4 territory).")
    p.add_argument("--view-alphas", default="1.0,0.5,0.0",
                   help="Comma list of mixture exponents alpha: per-class budget "
                        "proportional to pool_count**alpha, capped at the pool "
                        "(1.0 natural, 0.5 sqrt, 0.0 balanced-up-to-pool).")
    p.add_argument("--sel-taus", default="0.25,1.0,4.0",
                   help="Softmax-selection temperatures for the post-hoc "
                        "viewsoft columns (tau -> inf = viewavg, -> 0 = argmin).")
    p.add_argument("--energy-T", type=float, default=1.0,
                   help="Energy temperature: E = -T*logsumexp(logits/T).")
    p.add_argument("--eval-split", default="test", choices=["test", "val"],
                   help="test = far-future 20%% (the honest headline). val = "
                        "near-future 20%% (temporal-drift axis; XGB resumes "
                        "from fit-tag checkpoints, TabPFN refits "
                        "deterministically; refused with --train-split "
                        "train+val).")
    p.set_defaults(
        max_train_samples=-1,          # the view builder must see the whole pool
        n_estimators=1,                # per-VIEW m (D1 diagnostic default)
        # Defaults == the registered protocol (0820 SS8/SS8b): honest uncapped
        # test, and the only fit/batch config that survives the 24GB card
        # (fit_preprocessors or a single 4M call both OOM -- 0820 SS7).
        test_cap_per_class=0,
        fit_mode="fit_with_cache",
        test_batch_size=500_000,
        subsample_samples=0,           # guarded: exp5 never uses the hook
    )
    args = p.parse_args()
    run_exp5(args)


if __name__ == "__main__":
    main()

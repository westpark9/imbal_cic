#!/usr/bin/env python3
"""EXP 35 -- BoostPFN (Wang et al., "Prior-Fitted Networks Scale to Larger Datasets
When Treated as Weak Learners", AISTATS 2025; github.com/yxzwang/BoostPFN) on the
NF-v3 IDS chronological split.

Question
--------
The paper's claim is that a PFN whose context is capped at ~1k rows can still use a
large training pool if it is treated as a WEAK LEARNER inside gradient boosting: each
round samples a few hundred rows according to boosting weights, fits TabPFN-v1
in-context on them, and the ensemble is a weighted sum of the per-round logits.
Does that recipe do anything useful on an IDS pool with IR > 1000, and where does it
land relative to (a) XGBoost on the very same rows and (b) this project's record
numbers (full-pool XGBoost 0.7548, TabPFN-v3 100k-context 0.7638)?

What runs
---------
  1. Target (default cic2018 uncapped) -> per-scenario chronological 60/20/20 split,
     identical loaders to every nfv3_v3_exp* script (nfv3_v3_common).
  2. Train budget `--max-train-samples` (default 180,000, the "18만" first try) drawn
     STRATIFIED ratio-preserving from the train pool (core.stratified_subset, seed+850
     -- the same helper and seed the v3 scripts use, so these are the rows exp1 would
     have used at that budget).  Tail classes keep their natural share:
     web_attacks ~ 1,522/12.07M * 180k ~ 23 rows.
  3. Evaluation = test split capped at --test-cap-per-class (default 100k -> 440,276
     rows on cic2018), i.e. the tabpfn-track evaluation set.
  4. XGBoost on the SAME rows (same-run baseline, CLAUDE.md) unless --skip-xgboost.
  5. BoostPFN with the paper's Table-3 large-dataset command as defaults:
        --modelname gboost_tabpfnV2 --updating exphadamard --maxsample 500
        --ensemble-num 10 --loss CE --n-ensemble-configurations 1
     Round t: idx ~ Multinomial(w_t, 500 rows, without replacement) -> TabPFN-v1
     fit(X[idx]) -> logits h_t for train+test -> gamma_t = argmin multinomial
     deviance(F_{t-1} + gamma h_t) -> F_t -> w_{t+1} = w_t * exp(residual of the true
     class), clipped & renormalised.  Prediction = softmax(F_T) on the test rows.

Where the algorithm lives (NOT in this file)
--------------------------------------------
  This script imports BoostPFN's own classes at runtime from --boostpfn-root (default
  <imbalcic>/../SOTA/BoostPFN; see import_boostpfn()): SamplingGradientboost /
  SamplingAdaboost (the boosting loops and get_weak_learner), splitting_predict_proba, and
  upstream's TabPFNClassifier (the TabPFN-v1 weak learner).  Static analysers show those
  imports as unresolved because the clone sits outside this repo.  The clone plus its
  compatibility patches can be rebuilt anywhere from
  tabpfn/third_party/boostpfn_port/setup_boostpfn_port.sh (verified byte-identical rebuild,
  2026-09-08); README.md there maps every symbol to its upstream file.

Port notes (../SOTA/BoostPFN, IMBALCIC_PORT_NOTES.md)
--------------------------------------------------
  BoostPFN targets TabPFN v1 (pip 0.1.9) + sklearn 0.24 + torch 1.9.  It runs in this
  project's env (python 3.13 / torch 2.11 / sklearn 1.8 / numpy 2.4) through a vendored
  `tabpfn_v1` package (no collision with the v3 `tabpfn` editable install) and compat
  patches that do not touch the boosting arithmetic:
    - sklearn>=1.6  force_all_finite -> ensure_all_finite
    - sklearn>=1.3  sklearn.ensemble._gb_losses removed -> gb_losses_compat.py
                    (MultinomialDeviance / LeastSquaresError transcribed from 0.24.2)
    - torch>=2.6    torch.load(weights_only=False) for the (state, None, config) ckpt
    - torch>=2.4    checkpoint() needs use_reentrant; inference path calls predict()
                    directly; torch.cuda.amp.autocast -> torch.autocast('cuda')
    - torch 2.x     layer.py imported typing/nn names via torch.nn.modules.transformer
    - dead URL      v1 loader downloaded epoch_42 from a 404'ing automl/TabPFN link and
                    saved it as epoch_100; the v1.0.0 checkpoint (103,350,223 B, sha256
                    3c9aadae...) is placed at models_diff/..._epoch_100.cpkt instead.

Properties of the ORIGINAL code that matter under IR > 1000 (deliberately not patched)
------------------------------------------------------------------------------------
  * A 500-row weighted sample usually misses the tail classes; the weak learner then
    predicts only the classes it saw and the missing classes receive logit 0
    (gradient_boost_tabpfn.py get_weak_learner, version-2 branch).  Tail rows can only
    be recovered once the boosting weights concentrate on them -- 1b_round_samples.csv
    shows whether that ever happens.
  * With split_test=True (the large-dataset mode) `--sampling-size` is irrelevant:
    sample_num = --maxsample.
  * TabPFN-v1 hard limits: <= 100 features (46 here), <= 10 classes (7 here),
    context <= 1024 unless overwrite_warning (BoostPFN passes it).
  * `seed` in TabPFNClassifier fixes the (class-shift, feature-shift) permutation, so
    with N_ensemble_configurations=1 every weak learner shares ONE config.
  * BoostPFN's weak learner predicts train+test together every round (that is how the
    weights are updated), so the test matrix is touched T times.

Artifacts  results/<ts>_<out_tag>_exp35_boostpfn[/_82]/
------------------------------------------------------
  per_class_metrics.csv/.png   xgboost (same rows) vs boostpfn_<modelname>
  0b_split_audit.csv           per-scenario chronological split
  0c_train_used.csv            per-class rows: pool vs budget
  1a_boost_trajectory.csv/.png cumulative TEST macro / tail / per-class F1 after every
                               round, gamma_t, seconds -- "does boosting help?"
  1b_round_samples.csv/.png    per-round sampled class counts, weight mass per class,
                               effective sample size of w_t
  boost_test_logits.npy        F_T(test) float32 (softmax -> the reported prediction)
  args.json / timings.json / run.log

Cost
----
  Synthetic smoke (RTX 4090, 12k train / 8k test): 3 rounds in 0.3 s, 332 MiB peak.
  Here each round predicts 180k + 440k rows in --boost-test-batch chunks; expect
  seconds per round.  Wall-clock is dominated by loading the 14.7 GB uncapped pickle.

Usage
-----
  cd imbalcic && python tabpfn/scripts/nfv3_v3_exp35_boostpfn.py --target-dataset cic2018
  # smoke on the small pickle:
  python tabpfn/scripts/nfv3_v3_exp35_boostpfn.py --target-dataset cic2018_capped \
      --max-train-samples 20000 --test-cap-per-class 3000 --ensemble-num 3
"""

import argparse
import gc
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support

# Since the 2026-09-04 relocation the v3 scripts live in tabpfn/scripts/, but
# nfv3_v3_common.REPO_ROOT still resolves to tabpfn/ (one level short), so its
# exp_utils import and its data/ results/ defaults point at nothing.  Resolve the
# real repo root here and override the parser defaults; the shared module is not
# edited (CLAUDE.md: never retrofit the shared library).
HERE = os.path.dirname(os.path.abspath(__file__))
TABPFN_DIR = os.path.abspath(os.path.join(HERE, ".."))          # imbalcic/tabpfn
IMBALCIC_ROOT = os.path.abspath(os.path.join(TABPFN_DIR, "..")) # imbalcic
sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))

import nfv3_v3_common as core  # noqa: E402
import nfv3_v3_c0_context as c0ctx  # noqa: E402
from exp_utils import render_table_png  # noqa: E402

# 2026-09-08: the patched clone moved from ~/Desktop/SOTA/BoostPFN into the repo (vendored).
DEFAULT_BOOSTPFN_ROOT = os.path.join(TABPFN_DIR, "third_party", "BoostPFN")
MODELNAMES = ["gboost_tabpfnV2", "gboost_tabpfnV3", "gboost_tabpfnV4", "gboost_tabpfnV5",
              "gboost_tabpfn", "adaboost_tabpfn", "newadaboost_tabpfn", "tabpfn"]


class _Tee:
    """Mirror stdout into a buffer so run.log can be written into the results dir,
    whose name (timestamp) is only known at the end."""

    def __init__(self, stream):
        self.stream = stream
        self.chunks = []

    def write(self, s):
        self.stream.write(s)
        self.chunks.append(s)

    def flush(self):
        self.stream.flush()

    def text(self):
        return "".join(self.chunks)


def parse_args():
    p = core.base_parser(__doc__)
    p.set_defaults(max_train_samples=180_000,
                   data_dir=os.path.join(IMBALCIC_ROOT, "data"),
                   out_root=os.path.join(TABPFN_DIR, "results"),
                   models_dir=os.path.join(TABPFN_DIR, "saved_models"),
                   resume_dir=os.path.join(TABPFN_DIR, "resume"),
                   model_path=os.path.join(TABPFN_DIR, "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"))
    g = p.add_argument_group("BoostPFN (defaults = the paper's large-dataset command)")
    g.add_argument("--boostpfn-root", default=DEFAULT_BOOSTPFN_ROOT,
                   help="Clone of github.com/yxzwang/BoostPFN with the tabpfn_v1 port.")
    g.add_argument("--modelname", default="gboost_tabpfnV2", choices=MODELNAMES,
                   help="gboost_tabpfnV2 = paper Table 3. 'tabpfn' = ONE class-proportional "
                        "500-row TabPFN-v1 context, no boosting (their small baseline).")
    g.add_argument("--ensemble-num", type=int, default=10, help="Boosting rounds T.")
    g.add_argument("--maxsample", type=int, default=500,
                   help="Rows per weak learner (TabPFN-v1 context size).")
    g.add_argument("--sampling-size", type=float, default=0.001,
                   help="Kept for parity; IGNORED in split_test mode (sample_num=--maxsample).")
    g.add_argument("--replacement", action="store_true",
                   help="Sample rows with replacement (paper: without).")
    g.add_argument("--updating", default="exphadamard",
                   choices=["exphadamard", "hadamard", "adaboost", "loghadamard", "none"])
    g.add_argument("--loss", default="CE", choices=["CE"],
                   help="Only CE. BoostPFN's MSE path is not executable against its pinned sklearn 0.24.2 "
                        "(LeastSquaresError rejects (n,K) raw predictions), so it is not offered here.")
    g.add_argument("--wl-num", type=int, default=2, help="Candidates per round (V4 only).")
    g.add_argument("--n-ensemble-configurations", type=int, default=1,
                   help="TabPFN-v1 internal ensemble per weak learner (paper large runs: 1).")
    g.add_argument("--boost-test-batch", type=int, default=50_000,
                   help="Rows per TabPFN-v1 predict call (train+test are predicted every round).")
    g.add_argument("--boost-seed", type=int, default=None,
                   help="Seed for numpy sampling (BoostPFN draws with np.random); defaults to --seed. "
                        "Upstream: set_seed(args.seed) with --seed 5 in the README command.")
    g.add_argument("--tabpfn-v1-seed", type=int, default=None,
                   help="Seed handed to TabPFNClassifier(seed=...), which fixes the v1 (class-shift, "
                        "feature-shift) permutation. UPSTREAM NEVER PASSES IT (default 0). Runs before "
                        "2026-09-08 used --boost-seed here (42); default None keeps that behaviour so "
                        "earlier runs stay reproducible -- pass 0 to match upstream exactly.")
    g.add_argument("--skip-drift-diagnostic", action="store_true",
                   help="Skip the per-class train/test drift table (minutes on 17M benign rows).")
    c0ctx.add_c0_args(p)
    return p.parse_args()


def resolve_device(args):
    if args.device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return args.device


def import_boostpfn(root):
    if not os.path.isdir(os.path.join(root, "tabpfn_v1")):
        raise SystemExit(f"{root} has no tabpfn_v1/ -- run the port first (IMBALCIC_PORT_NOTES.md)")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, root)
    # numpy 2.x warns when np.array() is handed a torch bool tensor (BoostPFN does this
    # for the one-hot residual targets); harmless.
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            message=".*__array__ implementation doesn't accept a copy keyword.*")
    from scripts.transformer_prediction_interface import TabPFNClassifier  # noqa: E402
    from gradient_boost_tabpfn import SamplingGradientboost  # noqa: E402
    from boost_tabpfn import SamplingAdaboost  # noqa: E402
    from utils import splitting_predict_proba  # noqa: E402
    return TabPFNClassifier, SamplingGradientboost, SamplingAdaboost, splitting_predict_proba


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float32_finite(block, name):
    arr = np.asarray(block, dtype=np.float32)
    n_bad = int((~np.isfinite(arr)).sum())
    if n_bad:
        print(f"  {name}: {n_bad:,} non-finite cells -> nan_to_num (same policy as every v3 run)")
    return np.nan_to_num(arr)


def f1_vector(y_true, y_pred, n_classes):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    return f1


def class_proportional_context(y, n_classes, maxsample, rng, replacement=False):
    """largedataset_boostpfn.py 'tabpfn' branch (lines 149-167): ceil(maxsample * class
    share) rows per class, capped at the class size, uniform within class, sampled
    with replace=args.replacement -- the paper's no-boosting reference."""
    n = len(y)
    chosen = []
    for c in range(n_classes):
        rows = np.flatnonzero(y == c)
        if not len(rows):
            continue
        k = min(int(np.ceil(maxsample / n * len(rows))), len(rows))
        chosen.append(rng.choice(rows, size=k, replace=replacement))
    return np.sort(np.concatenate(chosen))


def main():
    args = parse_args()
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    if args.boost_seed is None:
        args.boost_seed = args.seed
    args.experiment = "exp35_boostpfn"
    device = resolve_device(args)
    args.resolved_device = device
    print(f"Args: {vars(args)}", flush=True)
    print("NOTE: --n-estimators/--fit-mode/--subsample-samples/--model-path/--keep-cache-on-device "
          "are TabPFN-v3 args inherited from the shared parser and are NOT used here.")

    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    print(f"target={args.target_dataset}  classes ({n_classes}): {class_names}")
    print(f"split rows: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    if args.train_split == "train+val":
        train_idx = np.sort(np.concatenate([train_idx, val_idx]))
        y_train_all = label_fn(train_idx)
        print(f"--train-split train+val: train pool {len(train_idx):,};  test unchanged")
    else:
        print(f"--train-split train: the {len(val_idx):,}-row val slice is DISCARDED")

    # ---- evaluation set (tabpfn-track convention) ----
    test_eval_idx = core.cap_per_class(test_idx, y_test_all, n_classes,
                                       args.test_cap_per_class, args.seed + 900)
    y_test_eval = label_fn(test_eval_idx)
    print(f"test eval rows: {len(test_eval_idx):,} (cap_per_class={args.test_cap_per_class})")
    print(f"  per class: { {n: int((y_test_eval == i).sum()) for i, n in enumerate(class_names)} }")
    X_test_eval = to_float32_finite(X[test_eval_idx], "X_test_eval")

    drift_df = None
    if not args.skip_drift_diagnostic:
        print("\n--- pre-fit diagnostic: train/test feature drift per class ---")
        drift_df = core.diagnose_train_test_drift(X, class_names, train_idx, test_idx, label_fn)
        print(drift_df.to_string(index=False))

    # ---- train budget (stratified ratio-preserving, same helper/seed as v3 exp1) ----
    train_used_idx, cap_policy, c0_info = train_idx, "none", None
    if args.context_recipe == "c0":
        # the pool BoostPFN samples its 500-row weak-learner contexts from
        c0_info = c0ctx.build_c0(args, train_idx, y_train_all, class_names, label_fn)
        train_used_idx = c0_info["idx"]
        cap_policy = f"c0_share{args.c0_benign_share}_{args.c0_attack_alloc}_{args.c0_pool_partition}"
    elif 0 < args.max_train_samples < len(train_used_idx):
        train_used_idx = core.stratified_subset(train_used_idx, label_fn(train_used_idx),
                                                n_classes, args.max_train_samples, args.seed + 850)
        cap_policy = "stratified_ratio_preserving"
    y_train_used = label_fn(train_used_idx)
    pool_counts = np.bincount(y_train_all, minlength=n_classes)
    used_counts = np.bincount(y_train_used, minlength=n_classes)
    train_used_df = pd.DataFrame({
        "class": class_names, "pool_rows": pool_counts, "used_rows": used_counts,
        "used_share": used_counts / used_counts.sum(),
        "expected_rows_per_weak_learner": args.maxsample * used_counts / used_counts.sum(),
    })
    print(f"\ntrain selection: used={len(train_used_idx):,} of pool {len(train_idx):,} "
          f"({100 * len(train_used_idx) / len(train_idx):.2f}%)  cap_policy={cap_policy}")
    print(train_used_df.to_string(index=False))
    X_train_used = to_float32_finite(X[train_used_idx], "X_train_used")
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()

    all_rows, timings = [], {}

    # ---- XGBoost on the same rows ----
    if not args.skip_xgboost:
        booster = xgb.XGBClassifier(
            n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight, reg_lambda=args.xgb_reg_lambda,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", n_jobs=-1, random_state=args.seed)
        print(f"\nXGBoost fitting on the same {len(X_train_used):,} rows ...", flush=True)
        t0 = time.time()
        booster.fit(X_train_used, y_train_used)
        timings["xgboost_fit_seconds"] = time.time() - t0
        t0 = time.time()
        y_pred_xgb = booster.predict(X_test_eval)
        timings["xgboost_predict_seconds"] = time.time() - t0
        print(f"XGBoost fit {timings['xgboost_fit_seconds']:.1f}s  "
              f"predict {timings['xgboost_predict_seconds']:.1f}s")
        all_rows.extend(core.per_class_table("xgboost", y_test_eval, y_pred_xgb,
                                             class_names, tail_classes))

    # ---- BoostPFN ----
    TabPFNClassifier, SamplingGradientboost, SamplingAdaboost, splitting_predict_proba = \
        import_boostpfn(args.boostpfn_root)
    set_all_seeds(args.boost_seed)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    tabpfn_v1_seed = args.boost_seed if args.tabpfn_v1_seed is None else args.tabpfn_v1_seed
    args.resolved_tabpfn_v1_seed = tabpfn_v1_seed
    print(f"TabPFN-v1 internal seed = {tabpfn_v1_seed} (upstream default 0; earlier runs used boost_seed)")
    base = TabPFNClassifier(device=device, N_ensemble_configurations=args.n_ensemble_configurations,
                            seed=tabpfn_v1_seed)
    timings["tabpfn_v1_load_seconds"] = time.time() - t_load
    print(f"\nTabPFN-v1 loaded in {timings['tabpfn_v1_load_seconds']:.1f}s  "
          f"(max_features={base.max_num_features}, max_classes={base.max_num_classes})")
    if X_train_used.shape[1] > base.max_num_features or n_classes > base.max_num_classes:
        raise SystemExit("dataset exceeds TabPFN-v1 limits (100 features / 10 classes)")

    bargs = argparse.Namespace(replacement=args.replacement, loss=args.loss, updating=args.updating,
                               wl_num=args.wl_num, debug=False)
    Xtr_t = torch.from_numpy(X_train_used)
    ytr_t = torch.from_numpy(y_train_used.astype(np.int64))
    Xte_t = torch.from_numpy(X_test_eval)
    round_times = []
    if args.modelname == "tabpfn":
        print(f"\nBoostPFN reference '{args.modelname}': ONE class-proportional {args.maxsample}-row context, "
              f"no boosting (--ensemble-num/--updating/--loss unused), "
              f"N_ensemble_configurations={args.n_ensemble_configurations}, "
              f"predict batch={args.boost_test_batch:,} over {len(X_test_eval):,} test rows", flush=True)
    else:
        print(f"\nBoostPFN {args.modelname}: T={args.ensemble_num} rounds x {args.maxsample}-row "
              f"weak learners, updating={args.updating}, loss={args.loss}, "
              f"N_ensemble_configurations={args.n_ensemble_configurations}, "
              f"predict batch={args.boost_test_batch:,} over {len(X_train_used) + len(X_test_eval):,} rows/round",
              flush=True)

    t_fit = time.time()
    if args.modelname == "tabpfn":
        rng = np.random.default_rng(args.boost_seed)
        idx = class_proportional_context(y_train_used, n_classes, args.maxsample, rng, args.replacement)
        t0 = time.time()
        base.fit(X_train_used[idx], y_train_used[idx], overwrite_warning=True)
        proba_local = splitting_predict_proba(base, X_test_eval, test_batch=args.boost_test_batch)
        round_times.append(time.time() - t0)
        # map the sampled-class columns back to the full class set (absent -> 0)
        proba_full = np.zeros((len(X_test_eval), n_classes), dtype=np.float64)
        proba_full[:, base.classes_.astype(int)] = proba_local
        alphas, test_probas, sampled_idxs, weights = [1.0], [proba_full], [idx], \
            [np.ones(len(y_train_used)) / len(y_train_used)]
        proba = proba_full
    else:
        if args.modelname.startswith("gboost_tabpfn"):
            version = int(args.modelname[-1]) if args.modelname[-1].isdigit() else 1
            clf = SamplingGradientboost(base, args.ensemble_num, args.sampling_size, bargs,
                                        split_test=True, max_samples=args.maxsample,
                                        batch_test=args.boost_test_batch, version=version)
        else:
            clf = SamplingAdaboost(base, args.ensemble_num, args.sampling_size, bargs,
                                   split_test=True, max_samples=args.maxsample,
                                   batch_test=args.boost_test_batch,
                                   new=(args.modelname == "newadaboost_tabpfn"))

        orig_get_weak_learner = clf.get_weak_learner

        def timed_get_weak_learner(sampler_weights, *a, **k):
            t0 = time.time()
            proba_all, idx = orig_get_weak_learner(sampler_weights, *a, **k)
            round_times.append(time.time() - t0)
            counts = np.bincount(y_train_used[np.asarray(idx, dtype=np.int64)], minlength=n_classes)
            ess = 1.0 / float(np.sum(np.asarray(sampler_weights, dtype=np.float64) ** 2))
            print(f"  weak learner {len(round_times)}: {round_times[-1]:.1f}s  "
                  f"sampled={counts.tolist()}  ESS(w)={ess:,.0f}  "
                  f"gpu_peak={torch.cuda.max_memory_allocated() / 2**30 if device.startswith('cuda') else 0:.2f} GiB",
                  flush=True)
            return proba_all, idx

        clf.get_weak_learner = timed_get_weak_learner
        clf.fit(Xtr_t, ytr_t, Xte_t)
        proba = clf.predict_proba(Xte_t)
        alphas, test_probas, sampled_idxs, weights = clf.alphas, clf.test_probas, clf.sampled_idxs, clf.ws
    timings["boostpfn_fit_predict_seconds"] = time.time() - t_fit
    timings["boostpfn_round_seconds"] = [round(s, 3) for s in round_times]
    if device.startswith("cuda"):
        timings["boostpfn_gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    y_pred_boost = np.asarray(proba).argmax(axis=1)
    method = f"boostpfn_{args.modelname}"
    all_rows.extend(core.per_class_table(method, y_test_eval, y_pred_boost, class_names, tail_classes))
    print(f"BoostPFN done in {timings['boostpfn_fit_predict_seconds']:.1f}s; "
          f"gammas={np.round(np.asarray(alphas, dtype=np.float64), 4).tolist()}")

    # ---- 1a: cumulative test F1 after each boosting round ----
    tail_idx = [i for i, n in enumerate(class_names) if n in tail_classes]
    # V4/V7 call get_weak_learner several times per round (candidate learners), so the
    # per-call timings do not map 1:1 onto rounds; report NaN rather than misaligned numbers.
    per_round_secs = round_times if len(round_times) == len(alphas) else [np.nan] * len(alphas)
    if len(round_times) != len(alphas):
        print(f"NOTE: {len(round_times)} weak-learner calls for {len(alphas)} rounds "
              "(candidate learners); 1a 'seconds' left NaN, timings.boostpfn_round_seconds is per CALL")
    F = np.zeros((len(X_test_eval), n_classes), dtype=np.float64)
    traj_rows = []
    for t, (a, tp) in enumerate(zip(alphas, test_probas)):
        F += float(a) * np.asarray(tp, dtype=np.float64)
        f1 = f1_vector(y_test_eval, F.argmax(axis=1), n_classes)
        row = {"round": t + 1, "gamma": float(a), "seconds": per_round_secs[t],
               "macro_f1": float(f1.mean()),
               "tail_f1": float(f1[tail_idx].mean()) if tail_idx else np.nan}
        row.update({f"f1_{n}": float(f1[i]) for i, n in enumerate(class_names)})
        traj_rows.append(row)
    traj_df = pd.DataFrame(traj_rows)
    final_logits = F.astype(np.float32)
    # gboost: predict_proba = softmax(float32(sum)) -> argmax equals F.argmax except at
    # float32 ties; adaboost normalises per row, same argmax.  Record, never abort:
    # aborting here would discard every artifact of a finished run.
    mismatch = int((F.argmax(axis=1) != y_pred_boost).sum())
    timings["trajectory_vs_predict_proba_argmax_mismatch_rows"] = mismatch
    if mismatch:
        print(f"WARNING: cumulative-sum argmax differs from predict_proba argmax on {mismatch:,} rows; "
              "per_class_metrics uses predict_proba, 1a uses the cumulative sum")

    # ---- 1b: what did each round sample, and where was the weight mass ----
    samp_rows = []
    for t, idx in enumerate(sampled_idxs):
        idx = np.asarray(idx, dtype=np.int64)
        counts = np.bincount(y_train_used[idx], minlength=n_classes)
        w = np.asarray(weights[t], dtype=np.float64) if t < len(weights) else None
        row = {"round": t + 1, "n_sampled": int(len(idx)),
               "n_classes_present": int((counts > 0).sum()),
               "ess_w": float(1.0 / np.sum(w ** 2)) if w is not None else np.nan}
        row.update({f"n_{n}": int(counts[i]) for i, n in enumerate(class_names)})
        if w is not None:
            row.update({f"wmass_{n}": float(w[y_train_used == i].sum()) for i, n in enumerate(class_names)})
        samp_rows.append(row)
    samp_df = pd.DataFrame(samp_rows)

    # ---- summary ----
    table = pd.DataFrame(all_rows)
    summary = table[table["class"].isin(["macro_avg", "weighted_avg", "tail_avg"])]
    print("\n=== summary (macro / weighted / tail F1) ===")
    print(summary.pivot(index="class", columns="method", values="f1").to_string())
    print("\n=== per-class F1 ===")
    print(table[~table["class"].isin(["macro_avg", "weighted_avg", "tail_avg"])]
          .pivot(index="class", columns="method", values="f1").to_string())
    print("\n=== boosting trajectory (test macro / tail F1 per round) ===")
    print(traj_df[["round", "gamma", "seconds", "macro_f1", "tail_f1"]].to_string(index=False))
    print("\n=== per-round sampled class counts ===")
    print(samp_df[["round", "n_classes_present", "ess_w"] + [f"n_{n}" for n in class_names]]
          .to_string(index=False))

    timings.update({
        "experiment": args.experiment, "target_dataset": args.target_dataset,
        "modelname": args.modelname, "ensemble_num": args.ensemble_num, "maxsample": args.maxsample,
        "updating": args.updating, "loss": args.loss,
        "n_ensemble_configurations": args.n_ensemble_configurations,
        "train_pool_rows": int(len(train_idx)), "train_rows_used": int(len(train_used_idx)),
        "train_pool_coverage_pct": round(100 * len(train_used_idx) / len(train_idx), 3),
        "train_cap_policy": cap_policy, "train_split": args.train_split,
        "validation_rows_unused": int(len(val_idx)), "test_pool_rows": int(len(test_idx)),
        "test_evaluation_rows": int(len(test_eval_idx)), "device": device,
        "boostpfn_root": args.boostpfn_root,
        "boostpfn_git": _git_head(args.boostpfn_root),
    })

    ts = time.strftime("%Y%m%d_%H%M%S")
    split_tag = "" if args.train_split == "train" else "_82"
    out_dir = os.path.join(args.out_root,
                           f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp35_boostpfn{split_tag}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    train_used_df.to_csv(os.path.join(out_dir, "0c_train_used.csv"), index=False)
    if c0_info is not None:
        c0_info["pool_audit"].to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
        if c0_info["c0_scenario"] is not None:
            c0_info["c0_scenario"].to_csv(os.path.join(out_dir, "0g_c0_scenario.csv"), index=False)
        timings.update(c0_info["timings"])
    traj_df.to_csv(os.path.join(out_dir, "1a_boost_trajectory.csv"), index=False)
    samp_df.to_csv(os.path.join(out_dir, "1b_round_samples.csv"), index=False)
    np.save(os.path.join(out_dir, "boost_test_logits.npy"), final_logits)
    if drift_df is not None:
        drift_df.to_csv(os.path.join(out_dir, "train_test_drift_diagnostic.csv"), index=False)
    for df, name, title in [
        (table.pivot(index="class", columns="method", values="f1").reset_index(),
         "per_class_metrics", f"exp35 BoostPFN {args.modelname} vs XGBoost (same {len(train_used_idx):,} rows) -- F1"),
        (traj_df[["round", "gamma", "seconds", "macro_f1", "tail_f1"]
                 + [f"f1_{n}" for n in class_names]], "1a_boost_trajectory",
         "exp35 cumulative test F1 per boosting round"),
        (samp_df[["round", "n_classes_present", "ess_w"] + [f"n_{n}" for n in class_names]],
         "1b_round_samples", "exp35 sampled rows per class per round"),
    ]:
        try:
            render_table_png(df, os.path.join(out_dir, f"{name}.png"), title=title,
                             high_good=[c for c in df.columns if c.startswith("f1_") or c in
                                        ("macro_f1", "tail_f1")] + [m for m in df.columns if m.startswith("boostpfn") or m == "xgboost"])
        except Exception as exc:  # PNG is a convenience; never fail the run on it
            print(f"  (png for {name} skipped: {exc})")
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([timings], f, indent=2)
    print(f"\nWrote {out_dir}")
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(tee.text())
    return out_dir


def _git_head(root):
    try:
        import subprocess
        head = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip() != ""
        return f"{head}{'+dirty' if dirty else ''}"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EXP 36 -- TabPFN-kNN and LoCalPFN (Thomas et al., "Retrieval & Fine-Tuning for
In-Context Tabular Models", NeurIPS 2024; github.com/layer6ai-labs/LoCalPFN) on the
NF-v3 IDS chronological split.

What the methods are
--------------------
  knn      TabPFN-kNN: for every test row the in-context "training set" is its
           --context-length nearest TRAIN rows (exact L2 via faiss on the standardised
           features), frozen TabPFN-v1 weights.  No training.
  ft       LoCalPFN: the same retrieval, plus fine-tuning of the TabPFN-v1 weights
           (AdamW, lr 1e-5, --num-epochs x --num-steps steps of --batch-size query
           rows, each with 1000 retrieved context + 1000 retrieved query neighbours),
           early-stopped on a VALIDATION slice by --early-stopping-metric, then
           kNN-evaluated on the test rows with the best weights.
  vanilla  their sub-sampling baseline: ONE fixed random class-proportional context of
           --context-length rows, frozen weights.
  The repo ships its own TabPFN-v1 re-implementation (pfn.py) and the v1.0.0
  checkpoint (sha256 3c9aadae..., byte-identical to the one BoostPFN uses).

What runs
---------
  1. Same loader / chronological split / evaluation set as every nfv3_v3_exp*
     script.  Train budget --max-train-samples (default 180,000 = "18만", stratified
     ratio-preserving, seed+850) is BOTH the retrieval pool and (ft) the fine-tuning
     pool.  Eval = test split capped at --test-cap-per-class (100k -> 440,276 rows).
  2. Preprocessing replicates LoCalPFN.dataset.load_tabzilla_data: StandardScaler fit
     on the train rows, applied to train/valid/test, clipped to +-clipping_val (10).
     All 46 NetFlow features are treated as numeric (no one-hot), as in every run of
     this project.
  3. Validation slice for ft early stopping: the otherwise-discarded chronological 20%
     val split, capped at --val-cap-per-class rows (default 5,000 -> ~35k rows; the
     full slice is 4M rows and is evaluated EVERY epoch).
  4. XGBoost on the same 180k rows (same-run baseline).

Port notes
----------
  faiss-cpu 1.15.0 installed into the project env (only missing dependency).  No
  source change was needed: PFN.load_old + torch.load and the bool attention mask
  through nn.TransformerEncoderLayer work under torch 2.11 (checked in
  scratchpad/smoke_localpfn_pfn.py: 1000-row context, batch 256 -> 0.76 s /
  forward, 5.15 GiB peak; batch 512 -> 10.2 GiB).
  Deviations from `main.py`: results dir naming, per-class metrics (theirs report
  weighted-F1 / ovo-AUC), and the val cap above.  Everything inside the method
  functions (retrieval, context construction, training loop, early stopping) is the
  repo's code, called unmodified.
  One wrapper on the model instance (clone_forward_output): their eval loops retain a
  VIEW of every batch's full decoder output, which leaks ~5 MB/batch on the GPU and
  OOMed the 440k-row eval at batch ~3,336 of 3,440 (2026-09-04).  The wrapper returns a
  contiguous copy; numerics unchanged.  `--ft-workdir` re-runs only the evaluation of
  an earlier fine-tuning workdir (model_best.pth + saved_data.npz).

Artifacts  results/<ts>_<out_tag>_exp36_localpfn_<method>[/_82]/
------------------------------------------------------------------
  per_class_metrics.csv/.png   xgboost vs localpfn_<method>
  0b_split_audit.csv, 0c_train_used.csv
  1a_ft_epochs.csv/.png        (ft) valid loss/acc/weighted-F1/ovo-AUC per epoch
                               (epoch 0 = frozen TabPFN-kNN on the val slice)
  test_logits.npy              (N_test, n_classes) float32
  model_best.pth               (ft) fine-tuned weights (103 MB), unless --no-save-models
  args.json / timings.json / run.log

Cost (RTX 4090, measured forward 0.76 s per 256 test rows at context 1000)
--------------------------------------------------------------------------
  knn on 440k test rows ~ 22 min; ft = 21 epochs x (val eval ~35k rows ~ 2 min +
  30 train steps) + final test eval ~ 1 h.

Usage
-----
  cd imbalcic
  python tabpfn/scripts/nfv3_v3_exp36_localpfn.py --target-dataset cic2018 --method knn
  python tabpfn/scripts/nfv3_v3_exp36_localpfn.py --target-dataset cic2018 --method ft
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
TABPFN_DIR = os.path.abspath(os.path.join(HERE, ".."))
IMBALCIC_ROOT = os.path.abspath(os.path.join(TABPFN_DIR, ".."))
sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))  # nfv3_v3_common.REPO_ROOT is one level short

import nfv3_v3_common as core  # noqa: E402
import nfv3_v3_c0_context as c0ctx  # noqa: E402
from exp_utils import render_table_png  # noqa: E402

# 2026-09-08: the clone moved from ~/Desktop/SOTA/LoCalPFN into the repo (vendored, unmodified).
DEFAULT_LOCALPFN_ROOT = os.path.join(TABPFN_DIR, "third_party", "LoCalPFN")


class _Tee:
    def __init__(self, stream):
        self.stream, self.chunks = stream, []

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
    g = p.add_argument_group("LoCalPFN (defaults = the repo's config.py)")
    g.add_argument("--localpfn-root", default=DEFAULT_LOCALPFN_ROOT)
    g.add_argument("--localpfn-model-path", default=None,
                   help="TabPFN-v1 checkpoint; default <root>/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt")
    g.add_argument("--method", default="knn", choices=["knn", "ft", "vanilla"])
    g.add_argument("--context-length", type=int, default=1000)
    g.add_argument("--class-choice", default="equal", choices=["equal", "balance"],
                   help="knn/ft: only the SUM matters (retrieval is class-agnostic); vanilla: per-class context sizes")
    g.add_argument("--dynamic", action="store_true", help="knn: context = min(10*sqrt(n_train), context_length)")
    g.add_argument("--batch-size-inf", type=int, default=256, help="test/valid rows per forward (repo: 512)")
    g.add_argument("--batch-size", type=int, default=2, help="ft: query rows per training step")
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--opt-weight-decay", type=float, default=0.01)
    g.add_argument("--num-epochs", type=int, default=21)
    g.add_argument("--num-steps", type=int, default=30)
    g.add_argument("--early-stopping-metric", default="auc", choices=["negloss", "acc", "f1", "auc"])
    g.add_argument("--early-stopping-rounds", type=int, default=100)
    g.add_argument("--eval-interval", type=int, default=1)
    g.add_argument("--train-query-length", type=int, default=1000)
    g.add_argument("--scheduler", action="store_true")
    g.add_argument("--better-selection", action="store_true")
    g.add_argument("--exact-knn", action="store_true")
    g.add_argument("--clipping-val", type=float, default=10.0)
    g.add_argument("--disable-normalize-data", action="store_true")
    g.add_argument("--val-cap-per-class", type=int, default=5_000,
                   help="ft: per-class cap on the chronological val slice used for early stopping")
    g.add_argument("--ft-workdir", default=None,
                   help="ft: skip training and evaluate the model_best.pth (+ saved_data.npz) found in this "
                        "earlier ft workdir (<resume-dir>/exp36_localpfn_ft_<ts>_<pid>). Same data args required.")
    g.add_argument("--skip-drift-diagnostic", action="store_true")
    c0ctx.add_c0_args(p)
    return p.parse_args()


def clone_forward_output(module):
    """LoCalPFN's eval loops keep `logits = model(...)[..., :K]` per batch -- a VIEW of
    the whole (context+1, B, 10) decoder output, so the full 1001 x B x 10 tensor of every
    batch stays alive on the GPU: ~5 MB/batch at B=128, 17.6 GB over the 3,440 batches of
    the 440k-row eval set (OOM observed at batch ~3,336, 2026-09-04).  Their TabZilla sets
    have <= 200k rows.  Returning a contiguous copy of the sliced output breaks the view
    chain; numerics and autograd (ft training) are unchanged."""
    orig = module.forward

    def forward(*a, **k):
        return orig(*a, **k).clone()

    module.forward = forward


def resolve_device(args):
    if args.device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return args.device


def to_float32_finite(block, name):
    arr = np.asarray(block, dtype=np.float32)
    n_bad = int((~np.isfinite(arr)).sum())
    if n_bad:
        print(f"  {name}: {n_bad:,} non-finite cells -> nan_to_num (same policy as every v3 run)")
    return np.nan_to_num(arr)


def import_localpfn(root):
    sys.path.insert(0, root)
    import pfn as _pfn  # noqa: E402
    import utils as _utils  # noqa: E402
    from methods.pfknn import eval_pfknn  # noqa: E402
    from methods.vanilla import eval_tabpfn  # noqa: E402
    from methods.ftknn import train_ft_knn, eval_ft_knn  # noqa: E402
    for m in (_pfn, _utils):
        assert os.path.abspath(m.__file__).startswith(os.path.abspath(root)), m.__file__
    return _pfn.PFN, _utils, eval_pfknn, eval_tabpfn, train_ft_knn, eval_ft_knn


def main():
    args = parse_args()
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    if args.localpfn_model_path is None:
        args.localpfn_model_path = os.path.join(args.localpfn_root, "models_diff",
                                                "prior_diff_real_checkpoint_n_0_epoch_42.cpkt")
    args.experiment = f"exp36_localpfn_{args.method}"
    device = resolve_device(args)
    args.resolved_device = device
    os.makedirs(args.resume_dir, exist_ok=True)
    ts_start = time.strftime("%Y%m%d_%H%M%S")
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
        raise SystemExit("exp36 needs the val slice for early stopping; --train-split train+val is not supported")

    test_eval_idx = core.cap_per_class(test_idx, y_test_all, n_classes, args.test_cap_per_class, args.seed + 900)
    y_test_eval = label_fn(test_eval_idx)
    print(f"test eval rows: {len(test_eval_idx):,} (cap_per_class={args.test_cap_per_class})")
    X_test_eval = to_float32_finite(X[test_eval_idx], "X_test_eval")

    val_eval_idx = core.cap_per_class(val_idx, label_fn(val_idx), n_classes, args.val_cap_per_class, args.seed + 901)
    y_val_eval = label_fn(val_eval_idx)
    print(f"val rows for early stopping: {len(val_eval_idx):,} (cap_per_class={args.val_cap_per_class}) "
          f"per class {np.bincount(y_val_eval, minlength=n_classes).tolist()}")
    X_val_eval = to_float32_finite(X[val_eval_idx], "X_val_eval")

    drift_df = None
    if not args.skip_drift_diagnostic:
        print("\n--- pre-fit diagnostic: train/test feature drift per class ---")
        drift_df = core.diagnose_train_test_drift(X, class_names, train_idx, test_idx, label_fn)
        print(drift_df.to_string(index=False))

    train_used_idx, cap_policy, c0_info = train_idx, "none", None
    if args.context_recipe == "c0":
        # the retrieval pool (knn / ft) and the fine-tuning pool (ft)
        c0_info = c0ctx.build_c0(args, train_idx, y_train_all, class_names, label_fn)
        train_used_idx = c0_info["idx"]
        cap_policy = f"c0_share{args.c0_benign_share}_{args.c0_attack_alloc}_{args.c0_pool_partition}"
    elif 0 < args.max_train_samples < len(train_used_idx):
        train_used_idx = core.stratified_subset(train_used_idx, label_fn(train_used_idx), n_classes,
                                                args.max_train_samples, args.seed + 850)
        cap_policy = "stratified_ratio_preserving"
    y_train_used = label_fn(train_used_idx)
    used_counts = np.bincount(y_train_used, minlength=n_classes)
    train_used_df = pd.DataFrame({"class": class_names,
                                  "pool_rows": np.bincount(y_train_all, minlength=n_classes),
                                  "used_rows": used_counts, "used_share": used_counts / used_counts.sum()})
    print(f"\ntrain selection: used={len(train_used_idx):,} of pool {len(train_idx):,} "
          f"({100 * len(train_used_idx) / len(train_idx):.2f}%)  cap_policy={cap_policy}")
    print(train_used_df.to_string(index=False))
    X_train_used = to_float32_finite(X[train_used_idx], "X_train_used")
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()

    all_rows, timings = [], {}

    # ---- XGBoost on the same (raw) rows ----
    if not args.skip_xgboost:
        booster = xgb.XGBClassifier(
            n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate, subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree, min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda, objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", n_jobs=-1, random_state=args.seed)
        print(f"\nXGBoost fitting on the same {len(X_train_used):,} rows ...", flush=True)
        t0 = time.time()
        booster.fit(X_train_used, y_train_used)
        timings["xgboost_fit_seconds"] = time.time() - t0
        t0 = time.time()
        y_pred_xgb = booster.predict(X_test_eval)
        timings["xgboost_predict_seconds"] = time.time() - t0
        print(f"XGBoost fit {timings['xgboost_fit_seconds']:.1f}s predict {timings['xgboost_predict_seconds']:.1f}s")
        all_rows.extend(core.per_class_table("xgboost", y_test_eval, y_pred_xgb, class_names, tail_classes))

    # ---- LoCalPFN preprocessing (dataset.load_tabzilla_data): standardise on train, clip ----
    if args.disable_normalize_data:
        Xtr, Xva, Xte = X_train_used, X_val_eval, X_test_eval
    else:
        scaler = StandardScaler().fit(X_train_used)
        clip = args.clipping_val
        Xtr = np.clip(scaler.transform(X_train_used), -clip, clip).astype(np.float32)
        Xva = np.clip(scaler.transform(X_val_eval), -clip, clip).astype(np.float32)
        Xte = np.clip(scaler.transform(X_test_eval), -clip, clip).astype(np.float32)
    data = {
        "X_train": Xtr, "X_valid": Xva, "X_test": Xte,
        "X_train_one_hot": Xtr, "X_valid_one_hot": Xva, "X_test_one_hot": Xte,  # no categoricals
        "y_train": y_train_used.astype(np.int64), "y_valid": y_val_eval.astype(np.int64),
        "y_test": y_test_eval.astype(np.int64),
        "dataset_info": {"name": cfg[args.target_dataset]["out_tag"], "cat_idx": [], "cat_dims": [],
                         "num_features": int(Xtr.shape[1]), "num_classes": n_classes},
    }

    # ---- the repo's argparse namespace (config.py names) ----
    largs = argparse.Namespace(
        device=device, method=args.method, seed=args.seed, timing=False, inf_temperature=0.8,
        context_length=args.context_length, class_choice=args.class_choice, dynamic=args.dynamic,
        batch_size=args.batch_size, batch_size_inf=args.batch_size_inf,
        use_one_hot_emb=False, onehot_retrieval=False, embedding="raw",
        lr=args.lr, opt_weight_decay=args.opt_weight_decay, num_epochs=args.num_epochs,
        num_steps=args.num_steps, early_stopping_metric=args.early_stopping_metric,
        early_stopping_rounds=args.early_stopping_rounds, eval_interval=args.eval_interval,
        train_query_length=args.train_query_length, scheduler=args.scheduler,
        better_selection=args.better_selection, exact_knn=args.exact_knn, save_data=True,
        splits_evaluated="valid", ensemble_dist=False, integrated=False, ensemble=False,
        ensemble_dist_folder="", clipping_val=args.clipping_val,
    )

    PFN, lutils, eval_pfknn, eval_tabpfn, train_ft_knn, eval_ft_knn = import_localpfn(args.localpfn_root)
    lutils.seed_everything(args.seed)
    lutils.create_dataloaders(largs, data)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, _ = PFN.load_old(device=device, path=args.localpfn_model_path)
    clone_forward_output(model)
    model.eval()
    timings["tabpfn_v1_load_seconds"] = time.time() - t0
    print(f"\nTabPFN-v1 (LoCalPFN pfn.py) loaded in {timings['tabpfn_v1_load_seconds']:.1f}s; "
          f"method={args.method} context_length={args.context_length} batch_size_inf={args.batch_size_inf} "
          f"test rows={len(Xte):,} -> {int(np.ceil(len(Xte) / args.batch_size_inf)):,} forwards", flush=True)

    epochs_df = None
    workdir = None
    t_method = time.time()
    if args.method == "knn":
        if largs.dynamic:
            largs.context_length = min(int(10 * np.sqrt(len(data["X_train"]))), largs.context_length)
            print(f"--dynamic: context_length -> {largs.context_length}")
        loss, logits = eval_pfknn(largs, model, data, data["test_loader"])
    elif args.method == "vanilla":
        loss, logits = eval_tabpfn(largs, model, data, data["test_loader"])
    else:  # ft = LoCalPFN
        if args.ft_workdir:
            workdir = args.ft_workdir
            print(f"--ft-workdir: skipping training, evaluating {workdir}")
            timings["ft_train_seconds"] = None
        else:
            from torch.utils.tensorboard import SummaryWriter  # noqa: E402
            workdir = os.path.join(args.resume_dir, f"exp36_localpfn_ft_{ts_start}_{os.getpid()}")
            os.makedirs(workdir, exist_ok=True)
            writer = SummaryWriter(workdir)
            model.train()
            t1 = time.time()
            train_ft_knn(largs, model, data, writer, workdir)
            timings["ft_train_seconds"] = time.time() - t1
        best = os.path.join(workdir, "data", data["dataset_info"]["name"], "model_best.pth")
        model.load_state_dict(torch.load(best, map_location=device))
        model.eval()
        saved = os.path.join(workdir, "data", data["dataset_info"]["name"], "saved_data.npz")
        if os.path.exists(saved):
            sd = np.load(saved)
            epochs_df = pd.DataFrame({k: sd[k] for k in sd.files})
            epochs_df.insert(0, "epoch", np.arange(len(epochs_df)) * args.eval_interval)
            print("\n=== ft: validation metrics per epoch (epoch 0 = frozen TabPFN-kNN) ===")
            print(epochs_df.to_string(index=False))
        t1 = time.time()
        loss, logits = eval_ft_knn(largs, model, data["test_loader"], data)
        timings["ft_test_eval_seconds"] = time.time() - t1
    timings[f"{args.method}_total_seconds"] = time.time() - t_method
    timings["test_ce_loss_reported_by_repo"] = float(loss)
    if device.startswith("cuda"):
        timings["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    logits = np.asarray(logits, dtype=np.float32)
    assert logits.shape == (len(Xte), n_classes), logits.shape
    y_pred = logits.argmax(axis=1)
    method_name = f"localpfn_{args.method}"
    all_rows.extend(core.per_class_table(method_name, y_test_eval, y_pred, class_names, tail_classes))
    print(f"\n{method_name} done in {timings[f'{args.method}_total_seconds']:.1f}s  "
          f"peak {timings.get('gpu_peak_gib', 0):.2f} GiB")

    table = pd.DataFrame(all_rows)
    piv = table.pivot(index="class", columns="method", values="f1")
    print("\n=== summary (macro / weighted / tail F1) ===")
    print(piv.loc[["macro_avg", "weighted_avg", "tail_avg"]].to_string())
    print("\n=== per-class F1 ===")
    print(piv.drop(index=["macro_avg", "weighted_avg", "tail_avg"]).to_string())

    timings.update({
        "experiment": args.experiment, "target_dataset": args.target_dataset, "method": args.method,
        "context_length": largs.context_length,
        "train_pool_rows": int(len(train_idx)), "train_rows_used": int(len(train_used_idx)),
        "train_cap_policy": cap_policy, "val_rows_used": int(len(val_eval_idx)),
        "test_evaluation_rows": int(len(test_eval_idx)), "device": device,
        "localpfn_root": args.localpfn_root, "localpfn_git": _git_head(args.localpfn_root),
    })
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root,
                           f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp36_localpfn_{args.method}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    train_used_df.to_csv(os.path.join(out_dir, "0c_train_used.csv"), index=False)
    if c0_info is not None:
        c0_info["pool_audit"].to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
        if c0_info["c0_scenario"] is not None:
            c0_info["c0_scenario"].to_csv(os.path.join(out_dir, "0g_c0_scenario.csv"), index=False)
        timings.update(c0_info["timings"])
    np.save(os.path.join(out_dir, "test_logits.npy"), logits)
    if epochs_df is not None:
        epochs_df.to_csv(os.path.join(out_dir, "1a_ft_epochs.csv"), index=False)
    if drift_df is not None:
        drift_df.to_csv(os.path.join(out_dir, "train_test_drift_diagnostic.csv"), index=False)
    pngs = [(piv.reset_index(), "per_class_metrics", f"exp36 {method_name} vs XGBoost (same rows) -- F1")]
    if epochs_df is not None:
        pngs.append((epochs_df, "1a_ft_epochs", "exp36 LoCalPFN fine-tuning: validation metrics per epoch"))
    for df, name, title in pngs:
        try:
            render_table_png(df, os.path.join(out_dir, f"{name}.png"), title=title,
                             high_good=[c for c in df.columns if c in piv.columns or c.startswith(("acc", "f1", "auc"))])
        except Exception as exc:
            print(f"  (png for {name} skipped: {exc})")
    if workdir is not None and not args.no_save_models:
        best = os.path.join(workdir, "data", data["dataset_info"]["name"], "model_best.pth")
        if os.path.exists(best):
            shutil.copy(best, os.path.join(out_dir, "model_best.pth"))
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

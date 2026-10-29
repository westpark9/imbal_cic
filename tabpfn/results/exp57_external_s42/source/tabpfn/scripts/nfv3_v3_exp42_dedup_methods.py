#!/usr/bin/env python3
"""EXP42 (frozen) -- the method comparison ON the dataset-level de-duplicated benchmarks of EXP41.

EXP41 asked "does removing identical-vector/conflicting-label rows raise the score?" with two models
(full-train XGBoost, TabPFN composed-C0). EXP42 keeps EXP41's four benchmarks byte-identical and runs the
whole comparison set on each of them, reporting PER CLASS:

  oracle        realistic oracle  -- perfect routing, conflicting vectors answered with their majority label
  xgb_full      XGBoost on the arm's whole train pool          (reused from the recorded EXP41 XGB run)
  xgb_c0        XGBoost on the same 100k C0 rows TabPFN gets   (matched information budget)
  tabpfn_plain  TabPFN-v3, context = 100k ratio-preserving random rows (the naive context)
  ours_c0       TabPFN-v3, context = exp31 composed C0 (benign 0.75, attacks balanced, n_estimators 4)
  distpfn       ours_c0's probabilities + DistPFN's prior correction (Sec.3 of the DistPFN paper)
  boostpfn      BoostPFN (TabPFN-v1 weak learners, T rounds x --maxsample rows) on the same C0 pool
  localpfn_knn  LoCalPFN's TabPFN-kNN: per-test-row local context retrieved from the same C0 pool

--dataset-dedup selects the benchmark; the dedup + re-split block is a verbatim copy of
nfv3_v3_exp41_dedup_dataset_tabpfn.py, so every arm's train/val/test rows are the same rows EXP41 scored:
  original        no change (test = the original 20%)
  dedup_xy        one row per (46-feature vector, label), earliest by time
  dedup_majority  one row per vector, label = majority label of that vector over the whole dataset
  drop_conflict   one row per vector; vectors carrying >1 label anywhere in the dataset removed

--methods picks which of the above to run in this process. BoostPFN and LoCalPFN vendor their own TabPFN-v1
copies with colliding top-level module names (utils/scripts/methods), so they are imported lazily and are
meant to be run in separate invocations -- never both in one process.

0c_class_counts.csv answers "how many rows per class did train/test become": the pipeline's original split
counts next to this arm's counts, per class, for train/val/test.

Run:
  python tabpfn/scripts/nfv3_v3_exp42_dedup_methods.py --target-dataset cic2018 --dataset-dedup dedup_xy \
      --methods oracle,xgb_full,xgb_c0,tabpfn_plain,ours_c0,distpfn --seed 42
  python tabpfn/scripts/nfv3_v3_exp42_dedup_methods.py --target-dataset cic2018 --dataset-dedup dedup_xy \
      --methods boostpfn --seed 42
  python tabpfn/scripts/nfv3_v3_exp42_dedup_methods.py --target-dataset cic2018 --dataset-dedup dedup_xy \
      --methods localpfn_knn --seed 42
"""
import argparse, gc, json, os, random, sys, time, warnings
import numpy as np, pandas as pd, torch, xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__)); TABPFN_DIR = os.path.abspath(os.path.join(HERE, ".."))
IMBALCIC_ROOT = os.path.abspath(os.path.join(TABPFN_DIR, "..")); sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))
import nfv3_v3_common as core  # noqa: E402
import nfv3_v3_c0_context as c0ctx  # noqa: E402

ALL_METHODS = ["oracle", "xgb_full", "xgb_c0", "tabpfn_plain", "ours_c0", "distpfn", "boostpfn", "localpfn_knn"]
DEFAULT_XGB_FULL_RUN = os.path.join(IMBALCIC_ROOT, "results", "20260911_165635_3341259_nfv3_cic2018_exp41_dedup_dataset_xgb")
DEFAULT_BOOSTPFN_ROOT = os.path.join(TABPFN_DIR, "third_party", "BoostPFN")
DEFAULT_LOCALPFN_ROOT = os.path.join(TABPFN_DIR, "third_party", "LoCalPFN")


class _Tee:
    def __init__(self, stream): self.stream, self.chunks = stream, []
    def write(self, s): self.stream.write(s); self.chunks.append(s)
    def flush(self): self.stream.flush()
    def text(self): return "".join(self.chunks)


def parse_args():
    p = core.base_parser(__doc__)
    p.set_defaults(max_train_samples=180_000, data_dir=os.path.join(IMBALCIC_ROOT, "data"), out_root=os.path.join(TABPFN_DIR, "results"),
                   models_dir=os.path.join(TABPFN_DIR, "saved_models"), resume_dir=os.path.join(TABPFN_DIR, "resume"),
                   model_path=os.path.join(TABPFN_DIR, "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"),
                   n_estimators=4, fit_mode="fit_with_cache", test_batch_size=500_000, test_cap_per_class=0)
    g = p.add_argument_group("EXP42 dataset dedup benchmark + method set")
    g.add_argument("--dataset-dedup", default="original", choices=["original", "dedup_xy", "dedup_majority", "drop_conflict"])
    g.add_argument("--min-scenario-rows", type=int, default=5)
    g.add_argument("--methods", default="oracle,xgb_full,xgb_c0,tabpfn_plain,ours_c0,distpfn",
                   help="comma list from " + ",".join(ALL_METHODS))
    g.add_argument("--xgb-full-run", default=DEFAULT_XGB_FULL_RUN,
                   help="recorded EXP41 XGB run to reuse the full-train-pool predictions from (test_idx is verified)")
    b = p.add_argument_group("BoostPFN (same defaults as exp35's recorded C0 run)")
    b.add_argument("--boostpfn-root", default=DEFAULT_BOOSTPFN_ROOT)
    b.add_argument("--ensemble-num", type=int, default=50, help="boosting rounds T")
    b.add_argument("--maxsample", type=int, default=500, help="rows per weak learner")
    b.add_argument("--sampling-size", type=float, default=0.001)
    b.add_argument("--n-ensemble-configurations", type=int, default=1)
    b.add_argument("--boost-test-batch", type=int, default=50_000)
    l = p.add_argument_group("LoCalPFN (same defaults as exp36's recorded knn run, batch_size_inf raised)")
    l.add_argument("--localpfn-root", default=DEFAULT_LOCALPFN_ROOT)
    l.add_argument("--localpfn-model-path", default=None)
    l.add_argument("--context-length", type=int, default=1000)
    l.add_argument("--class-choice", default="equal", choices=["equal", "balance"])
    l.add_argument("--batch-size-inf", type=int, default=1024,
                   help="test rows per forward; exp36 used 128. Retrieval is per row, so this only batches work.")
    l.add_argument("--clipping-val", type=float, default=10.0)
    l.add_argument("--val-cap-per-class", type=int, default=5_000)
    c0ctx.add_c0_args(p); p.set_defaults(context_recipe="c0")
    return p.parse_args()


def to_float32_finite(block, name):
    arr = np.asarray(block, dtype=np.float32)
    if not np.isfinite(arr).all():
        print(f"  {name}: non-finite values -> nan_to_num"); arr = np.nan_to_num(arr)
    return arr


def clone_forward_output(module):
    """exp36's guard, verbatim: LoCalPFN's eval loop keeps `logits = model(...)[..., :K]` per batch -- a VIEW
    of the whole (context+1, B, 10) decoder output, so every batch's full tensor stays alive on the GPU.
    Returning a contiguous copy breaks the view chain; numerics are unchanged."""
    orig = module.forward

    def forward(*a, **k):
        return orig(*a, **k).clone()

    module.forward = forward


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def distpfn_adjust(y_prob, prior_train, eps=1e-8):
    """DistPFN's prior correction, transcribed in exp37: adjusted = p * P_test_avg / (prior + eps)."""
    p_test_avg = y_prob.mean(axis=0)
    adjusted = (y_prob * p_test_avg) / (prior_train + eps)
    return adjusted / adjusted.sum(axis=1, keepdims=True), p_test_avg


def main():
    args = parse_args(); tee = _Tee(sys.stdout); sys.stdout = tee
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad: raise SystemExit(f"unknown --methods {bad}; choose from {ALL_METHODS}")
    if "distpfn" in methods and "ours_c0" not in methods: raise SystemExit("--methods distpfn needs ours_c0 in the same run (it adjusts its probabilities)")
    if "boostpfn" in methods and "localpfn_knn" in methods: raise SystemExit("boostpfn and localpfn_knn vendor colliding TabPFN-v1 copies -- run them in separate invocations")
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None: args.data = cfg[args.target_dataset]["default_data"]
    args.experiment = f"exp42_dedup_methods_{args.dataset_dedup}"
    device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    args.resolved_device = device; os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = cfg[args.target_dataset]["loader"](args)
    C = len(class_names); print(f"target={args.target_dataset} classes {class_names}; train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
    pipe_counts = {s: np.bincount(label_fn(ix), minlength=C) for s, ix in [("train", train_idx), ("val", val_idx), ("test", test_idx)]}

    # ---- dataset-level dedup BEFORE the split (verbatim from exp41; hash cache shared across arms/seeds)
    d = core.load_pickle(args.data); scen = np.asarray(d["attack_scenarios"]).astype(str); tsm = np.asarray(d["timestamps"], dtype=np.int64); del d
    all_idx = np.sort(np.concatenate([train_idx, val_idx, test_idx])); y_all = label_fn(all_idx)
    hcache = os.path.join(args.resume_dir, f"exp41_hash_{args.target_dataset}.npy"); t0 = time.time()
    if os.path.exists(hcache): h_all = np.load(hcache)
    else:
        h_all = np.concatenate([pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(np.asarray(X[all_idx[i:i + 2_000_000]], dtype=np.float32))), index=False).to_numpy() for i in range(0, len(all_idx), 2_000_000)]); np.save(hcache, h_all)
    df = pd.DataFrame({"h": h_all, "y": y_all, "t": tsm[all_idx]}); cnt = df.groupby(["h", "y"]).size().reset_index(name="c")
    maj = cnt.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]; nlab = cnt.groupby("h").size(); conflict_h = set(nlab[nlab > 1].index)
    if args.dataset_dedup == "original": keep = np.ones(len(all_idx), bool)
    else:
        first = ~df.sort_values("t", kind="stable").duplicated(["h", "y"]).sort_index().to_numpy()
        if args.dataset_dedup == "dedup_xy": keep = first
        elif args.dataset_dedup == "dedup_majority": keep = first & (df.h.map(maj).to_numpy() == y_all)
        else: keep = first & ~df.h.isin(conflict_h).to_numpy()
    sub = all_idx[keep]; tr, va, te, small = [], [], [], []
    for s in np.unique(scen[sub]):
        sel = sub[scen[sub] == s]; sel = sel[np.argsort(tsm[sel], kind="stable")]
        if len(sel) < args.min_scenario_rows: tr.append(sel); small.append((str(s), int(len(sel)))); continue
        k1, k2 = int(len(sel) * .6), int(len(sel) * .8); tr.append(sel[:k1]); va.append(sel[k1:k2]); te.append(sel[k2:])
    train_idx = np.sort(np.concatenate(tr)); val_idx = np.sort(np.concatenate(va)); test_idx = np.sort(np.concatenate(te))
    y_train_all = label_fn(train_idx); y_val_all = label_fn(val_idx); y_test_all = label_fn(test_idx)
    hpos = pd.Series(h_all, index=all_idx); htr = hpos.loc[train_idx].to_numpy(); hte = hpos.loc[test_idx].to_numpy(); X_test = to_float32_finite(X[test_idx], "X_test")
    print(f"dataset-dedup={args.dataset_dedup}: rows {len(all_idx):,} -> {len(sub):,}; train {len(train_idx):,} val {len(val_idx):,} test {len(test_idx):,}; train-only small scenarios {small}; hashing/split {time.time() - t0:.0f}s")

    # ---- 0c: per-class row counts, pipeline split vs this arm's split
    arm_counts = {s: np.bincount(y, minlength=C) for s, y in [("train", y_train_all), ("val", y_val_all), ("test", y_test_all)]}
    cc_rows = [dict(split=s, cls=class_names[c], rows_pipeline=int(pipe_counts[s][c]), rows_arm=int(arm_counts[s][c]),
                    delta=int(arm_counts[s][c] - pipe_counts[s][c]),
                    kept_frac=(float(arm_counts[s][c] / pipe_counts[s][c]) if pipe_counts[s][c] else float("nan")))
               for s in ("train", "val", "test") for c in range(C)]
    class_counts = pd.DataFrame(cc_rows)
    print("\nper-class rows (pipeline split -> this arm's split)")
    print(class_counts.pivot(index="cls", columns="split", values=["rows_pipeline", "rows_arm"]).to_string())

    # buckets on the (new) train pool; realistic oracle on the (new) test
    cnt2 = pd.DataFrame({"h": htr, "y": y_train_all}).groupby(["h", "y"]).size().reset_index(name="c"); maj2 = cnt2.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]
    te_maj = pd.Series(hte).map(maj2); seen = te_maj.notna().to_numpy(); same = seen & (te_maj.to_numpy() == y_test_all); diff = seen & ~same; unseen = ~seen
    tcnt = pd.DataFrame({"h": hte, "y": y_test_all}).groupby(["h", "y"]).size().reset_index(name="c"); tmaj = tcnt.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]

    preds, timings = {}, {"dataset_dedup": args.dataset_dedup, "methods": methods, "rows_after_dedup": int(len(sub)),
                          "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "test_rows": int(len(test_idx)), "small_train_only": small}
    if "oracle" in methods:
        preds["realistic_oracle"] = pd.Series(hte).map(tmaj).to_numpy().astype(np.int16)

    # ---- xgb_full: reuse the recorded EXP41 full-train-pool predictions (test rows verified identical)
    if "xgb_full" in methods:
        pf = os.path.join(args.xgb_full_run, f"pred_{args.dataset_dedup}.npy"); tf = os.path.join(args.xgb_full_run, f"test_idx_{args.dataset_dedup}.npy")
        if not (os.path.exists(pf) and os.path.exists(tf)): raise SystemExit(f"xgb_full needs {pf} and {tf}")
        ti = np.load(tf)
        if not np.array_equal(ti, test_idx): raise SystemExit(f"xgb_full test_idx mismatch: recorded {len(ti):,} rows vs this split {len(test_idx):,}")
        preds["xgboost_full"] = np.load(pf).astype(np.int16); timings["xgb_full_reused_from"] = args.xgb_full_run
        print(f"xgb_full: reused {os.path.basename(args.xgb_full_run)}/pred_{args.dataset_dedup}.npy ({len(ti):,} test rows verified identical)")

    # ---- contexts: composed C0 (ours) and the ratio-preserving plain draw, both from this arm's train pool
    need_c0 = any(m in methods for m in ("xgb_c0", "ours_c0", "boostpfn", "localpfn_knn"))
    c0_info = c0ctx.build_c0(args, train_idx, y_train_all, class_names, label_fn) if need_c0 else None
    used_idx = c0_info["idx"] if need_c0 else None
    plain_idx = core.stratified_subset(train_idx, y_train_all, C, args.c0_size, args.seed + 850) if "tabpfn_plain" in methods else None
    X_used = to_float32_finite(X[used_idx], "X_c0") if need_c0 else None
    y_used = label_fn(used_idx) if need_c0 else None
    X_plain = to_float32_finite(X[plain_idx], "X_plain") if plain_idx is not None else None
    y_plain = label_fn(plain_idx) if plain_idx is not None else None
    val_keep = core.cap_per_class(val_idx, y_val_all, C, args.val_cap_per_class, args.seed + 77) if "localpfn_knn" in methods else None
    X_val = to_float32_finite(X[val_keep], "X_val") if val_keep is not None else None
    y_val = label_fn(val_keep) if val_keep is not None else None
    del X; core._PICKLE_CACHE.clear(); gc.collect()
    if need_c0: print(f"C0 rows {len(used_idx):,}; per class {dict(zip(class_names, np.bincount(y_used, minlength=C).tolist()))}")
    if plain_idx is not None: print(f"plain context rows {len(plain_idx):,}; per class {dict(zip(class_names, np.bincount(y_plain, minlength=C).tolist()))}")

    def fit_xgb(Xc, yc, tag):
        t = time.time(); bst = xgb.XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth, learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample_bytree, min_child_weight=args.xgb_min_child_weight, reg_lambda=args.xgb_reg_lambda,
            objective="multi:softprob", num_class=C, n_jobs=-1, random_state=args.seed)
        bst.fit(Xc, yc); p = bst.predict(X_test); timings[f"{tag}_seconds"] = round(time.time() - t, 1); return p.astype(np.int16)

    if "xgb_c0" in methods: preds["xgboost_c0"] = fit_xgb(X_used, y_used, "xgb_c0")

    # ---- TabPFN-v3 (plain context and composed C0); ours' probabilities feed DistPFN
    if "tabpfn_plain" in methods or "ours_c0" in methods:
        from tabpfn import TabPFNClassifier  # noqa: E402
        def run_tabpfn(Xc, yc, tag, keep_proba):
            clf = TabPFNClassifier(device=device, model_path=args.model_path, ignore_pretraining_limits=args.ignore_pretraining_limits,
                                   inference_config={"SUBSAMPLE_SAMPLES": int(args.subsample_samples) or None}, random_state=args.seed,
                                   n_estimators=args.n_estimators, auto_scale_n_estimators=False, fit_mode=args.fit_mode, keep_cache_on_device=args.keep_cache_on_device)
            if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
            t = time.time(); clf.fit(Xc, yc); timings[f"{tag}_fit_seconds"] = round(time.time() - t, 1)
            t = time.time(); bs = args.test_batch_size if args.test_batch_size > 0 else len(X_test)
            proba = np.empty((len(X_test), C), dtype=np.float32) if keep_proba else None; out = []
            for s in range(0, len(X_test), bs):
                pb = clf.predict_proba(X_test[s:s + bs]).astype(np.float32)
                if keep_proba: proba[s:s + len(pb)] = pb
                out.append(pb.argmax(1).astype(np.int16)); print(f"  [{tag}] predict rows {s:,}:{min(s + bs, len(X_test)):,}", flush=True)
            timings[f"{tag}_predict_seconds"] = round(time.time() - t, 1)
            if device.startswith("cuda"): timings[f"{tag}_gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            assert list(clf.classes_) == list(range(C)), clf.classes_
            del clf; gc.collect(); torch.cuda.empty_cache() if device.startswith("cuda") else None
            return np.concatenate(out), proba
        if "tabpfn_plain" in methods: preds["tabpfn_plain"], _ = run_tabpfn(X_plain, y_plain, "tabpfn_plain", False)
        if "ours_c0" in methods:
            preds["ours_c0"], proba_ours = run_tabpfn(X_used, y_used, "ours_c0", "distpfn" in methods)
            if "distpfn" in methods:
                prior = np.bincount(y_used, minlength=C) / len(y_used)
                adj, p_avg = distpfn_adjust(proba_ours.astype(np.float64), prior)
                preds["distpfn"] = adj.argmax(1).astype(np.int16)
                timings["distpfn_prior_train"] = prior.round(6).tolist(); timings["distpfn_p_test_avg"] = p_avg.round(6).tolist()
                del adj, proba_ours; gc.collect()

    # ---- BoostPFN (vendored TabPFN-v1; lazy import)
    if "boostpfn" in methods:
        if not os.path.isdir(os.path.join(args.boostpfn_root, "tabpfn_v1")): raise SystemExit(f"{args.boostpfn_root} has no tabpfn_v1/")
        os.environ.setdefault("MPLBACKEND", "Agg"); sys.path.insert(0, args.boostpfn_root)
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*__array__ implementation doesn't accept a copy keyword.*")
        from scripts.transformer_prediction_interface import TabPFNClassifier as TabPFNv1  # noqa: E402
        from gradient_boost_tabpfn import SamplingGradientboost  # noqa: E402
        set_all_seeds(args.seed)
        if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
        base = TabPFNv1(device=device, N_ensemble_configurations=args.n_ensemble_configurations, seed=args.seed)
        bargs = argparse.Namespace(replacement=False, loss="CE", updating="exphadamard", wl_num=2, debug=False)
        clf = SamplingGradientboost(base, args.ensemble_num, args.sampling_size, bargs, split_test=True,
                                    max_samples=args.maxsample, batch_test=args.boost_test_batch, version=2)
        t = time.time(); clf.fit(torch.from_numpy(X_used), torch.from_numpy(y_used.astype(np.int64)), torch.from_numpy(X_test))
        proba = np.asarray(clf.predict_proba(torch.from_numpy(X_test)))
        timings["boostpfn_seconds"] = round(time.time() - t, 1)
        if device.startswith("cuda"): timings["boostpfn_gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        timings["boostpfn_gammas"] = np.round(np.asarray(clf.alphas, dtype=np.float64), 4).tolist()
        preds["boostpfn"] = proba.argmax(1).astype(np.int16); del clf, base, proba; gc.collect()

    # ---- LoCalPFN TabPFN-kNN (vendored TabPFN-v1; lazy import)
    if "localpfn_knn" in methods:
        from sklearn.preprocessing import StandardScaler  # noqa: E402
        if args.localpfn_model_path is None:
            args.localpfn_model_path = os.path.join(args.localpfn_root, "models_diff", "prior_diff_real_checkpoint_n_0_epoch_42.cpkt")
        sys.path.insert(0, args.localpfn_root)
        import pfn as _pfn  # noqa: E402
        import utils as _utils  # noqa: E402
        from methods.pfknn import eval_pfknn  # noqa: E402
        for _m in (_pfn, _utils): assert os.path.abspath(_m.__file__).startswith(os.path.abspath(args.localpfn_root)), _m.__file__
        scaler = StandardScaler().fit(X_used); clip = args.clipping_val
        Xtr = np.clip(scaler.transform(X_used), -clip, clip).astype(np.float32)
        Xva = np.clip(scaler.transform(X_val), -clip, clip).astype(np.float32)
        Xte = np.clip(scaler.transform(X_test), -clip, clip).astype(np.float32)
        data = {"X_train": Xtr, "X_valid": Xva, "X_test": Xte, "X_train_one_hot": Xtr, "X_valid_one_hot": Xva, "X_test_one_hot": Xte,
                "y_train": y_used.astype(np.int64), "y_valid": y_val.astype(np.int64), "y_test": y_test_all.astype(np.int64),
                "dataset_info": {"name": cfg[args.target_dataset]["out_tag"], "cat_idx": [], "cat_dims": [], "num_features": int(Xtr.shape[1]), "num_classes": C}}
        largs = argparse.Namespace(device=device, method="knn", seed=args.seed, timing=False, inf_temperature=0.8,
            context_length=args.context_length, class_choice=args.class_choice, dynamic=False, batch_size=2, batch_size_inf=args.batch_size_inf,
            use_one_hot_emb=False, onehot_retrieval=False, embedding="raw", lr=1e-5, opt_weight_decay=0.01, num_epochs=21, num_steps=30,
            early_stopping_metric="auc", early_stopping_rounds=100, eval_interval=1, train_query_length=1000, scheduler=False,
            better_selection=False, exact_knn=False, save_data=True, splits_evaluated="valid", ensemble_dist=False, integrated=False,
            ensemble=False, ensemble_dist_folder="", clipping_val=args.clipping_val)
        _utils.seed_everything(args.seed); _utils.create_dataloaders(largs, data)
        if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
        model, _ = _pfn.PFN.load_old(device=device, path=args.localpfn_model_path)
        clone_forward_output(model)  # without this the per-batch logits stay views of the full decoder output -> GPU OOM (exp36 note)
        model.eval()
        t = time.time(); loss, logits = eval_pfknn(largs, model, data, data["test_loader"])
        timings["localpfn_knn_seconds"] = round(time.time() - t, 1); timings["localpfn_test_ce_loss"] = float(loss)
        if device.startswith("cuda"): timings["localpfn_gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        lg = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
        preds["localpfn_knn"] = lg.argmax(1).astype(np.int16); del model, logits, lg; gc.collect()

    # ---- scoring (all/seen-same/seen-different/unseen buckets, per class)
    rows, summary = [], []
    for method, pred in preds.items():
        if len(pred) != len(y_test_all): raise SystemExit(f"{method}: {len(pred):,} predictions for {len(y_test_all):,} test rows")
        for bname, m in [("all", np.ones(len(y_test_all), bool)), ("seen_same_label", same), ("seen_diff_label", diff), ("unseen", unseen)]:
            if not m.any():
                summary.append(dict(method=method, bucket=bname, rows=0, accuracy=float("nan"), macro_f1=float("nan"), tail_f1=float("nan"))); continue
            pr, rc, f1, sup = precision_recall_fscore_support(y_test_all[m], pred[m], labels=range(C), zero_division=0)
            for c in range(C): rows.append(dict(method=method, bucket=bname, cls=class_names[c], support=int(sup[c]), precision=pr[c], recall=rc[c], f1=f1[c]))
            summary.append(dict(method=method, bucket=bname, rows=int(m.sum()), accuracy=float((pred[m] == y_test_all[m]).mean()), macro_f1=float(f1.mean()),
                                tail_f1=float(np.mean([f1[class_names.index(t)] for t in tail_classes]))))
        f_all = [r for r in rows if r["method"] == method and r["bucket"] == "all"]
        print(f"[{method}] macro-F1 {np.mean([r['f1'] for r in f_all]):.4f} | " + " ".join(f"{r['cls']}={r['f1']:.3f}" for r in f_all))

    ts = time.strftime("%Y%m%d_%H%M%S"); tag = "_".join(sorted(methods))[:40]
    out_dir = os.path.join(args.out_root, f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp42_dedup_methods_{args.dataset_dedup}_{tag}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "1b_summary.csv"), index=False)
    class_counts.to_csv(os.path.join(out_dir, "0c_class_counts.csv"), index=False)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx)
    if need_c0:
        c0_info["pool_audit"].to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
        if c0_info["c0_scenario"] is not None: c0_info["c0_scenario"].to_csv(os.path.join(out_dir, "0g_c0_scenario.csv"), index=False)
        timings.update(c0_info["timings"])
    for method, pred in preds.items(): np.save(os.path.join(out_dir, f"pred_{method}.npy"), pred.astype(np.int16))
    timings.update({"experiment": args.experiment, "seed": args.seed, "device": device, "test_rows": int(len(y_test_all))})
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2, default=str)
    json.dump([timings], open(os.path.join(out_dir, "timings.json"), "w"), indent=2)
    print(f"\nWrote {out_dir}"); open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8").write(tee.text()); open(os.path.join(out_dir, "COMPLETE.json"), "w").write("{}\n")


if __name__ == "__main__":
    main()

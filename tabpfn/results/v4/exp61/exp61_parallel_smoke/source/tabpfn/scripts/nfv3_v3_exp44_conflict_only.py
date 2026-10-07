#!/usr/bin/env python3
"""EXP44 (frozen) -- the missing cell of the 중복 x 모순 grid: KEEP duplicates, REMOVE contradictions.

EXP41/EXP42 filled three of four cells:

                     모순 유지                     모순 제거
    중복 유지        original                      <- THIS SCRIPT
    중복 제거        dedup_xy                      dedup_majority / drop_conflict

dedup_majority already resolves contradictions by majority vote, but it also throws away 56% of the
train rows (12,069,313 -> 5,283,411) because it keeps one row per vector. That confounds two knobs.
Duplicate rows are not an error in NIDS data: a volumetric attack against a static target legitimately
emits millions of near-identical flows, and that repetition is the attack's signature and its base rate
(Flood et al. EuroS&P'24; Chen et al. arXiv 2105.10041). EXP40 measured the cost of removing them:
accuracy on vectors the model had memorised fell 99.99% -> 88.97% while accuracy on unseen vectors did
not move. So this script removes ONLY contradictions and leaves duplicate mass untouched.

THE KNOB --conflict-only:
  relabel_majority  every row whose 46-feature vector carries more than one label in the dataset is
                    RELABELLED to that vector's majority label. No row is deleted, so the re-split is
                    row-for-row identical to the `original` arm and only the labels differ.
  drop_vectors      every row of a contradictory vector is deleted; duplicates of the remaining
                    vectors are kept as-is.

Everything else (scenario-wise chronological 60/20/20 re-split, C0 recipe, scoring, bucket
decomposition) is copied verbatim from nfv3_v3_exp42_dedup_methods.py so the numbers drop straight into
the same per-class tables. XGBoost on the full train pool is fitted here rather than reused, because no
recorded EXP41 run covers these arms.

The majority label is taken over the whole dataset, matching how dedup_majority defines it, so the arms
stay comparable. That makes this a benchmark-construction step, not a deployable preprocessor: deriving
it from train+test is `selective snooping` in the sense of Arp et al. (USENIX Sec'22). Report it as a
diagnostic arm, never as a pipeline stage.

LoCalPFN is deliberately not offered here; it costs ~3.4 h per 4M-row arm and is run separately.

Run:
  python tabpfn/scripts/nfv3_v3_exp44_conflict_only.py --target-dataset cic2018 \
      --conflict-only relabel_majority --seed 42
"""
import argparse, gc, json, os, random, sys, time, warnings
import numpy as np, pandas as pd, torch, xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__)); TABPFN_DIR = os.path.abspath(os.path.join(HERE, ".."))
IMBALCIC_ROOT = os.path.abspath(os.path.join(TABPFN_DIR, "..")); sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))
import nfv3_v3_common as core  # noqa: E402
import nfv3_v3_c0_context as c0ctx  # noqa: E402

ALL_METHODS = ["oracle", "xgb_full", "xgb_c0", "tabpfn_plain", "ours_c0", "distpfn", "boostpfn"]
DEFAULT_BOOSTPFN_ROOT = os.path.join(TABPFN_DIR, "third_party", "BoostPFN")


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
    g = p.add_argument_group("EXP44: keep duplicates, remove contradictions")
    g.add_argument("--conflict-only", default="relabel_majority", choices=["relabel_majority", "drop_vectors"])
    g.add_argument("--min-scenario-rows", type=int, default=5)
    g.add_argument("--methods", default="oracle,xgb_full,xgb_c0,tabpfn_plain,ours_c0,distpfn",
                   help="comma list from " + ",".join(ALL_METHODS))
    b = p.add_argument_group("BoostPFN (same settings as exp35/exp42)")
    b.add_argument("--boostpfn-root", default=DEFAULT_BOOSTPFN_ROOT)
    b.add_argument("--ensemble-num", type=int, default=50)
    b.add_argument("--maxsample", type=int, default=500)
    b.add_argument("--sampling-size", type=float, default=0.001)
    b.add_argument("--n-ensemble-configurations", type=int, default=1)
    b.add_argument("--boost-test-batch", type=int, default=50_000)
    c0ctx.add_c0_args(p); p.set_defaults(context_recipe="c0")
    return p.parse_args()


def to_float32_finite(block, name):
    arr = np.asarray(block, dtype=np.float32)
    if not np.isfinite(arr).all():
        print(f"  {name}: non-finite values -> nan_to_num"); arr = np.nan_to_num(arr)
    return arr


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def distpfn_adjust(y_prob, prior_train, eps=1e-8):
    p_test_avg = y_prob.mean(axis=0)
    adjusted = (y_prob * p_test_avg) / (prior_train + eps)
    return adjusted / adjusted.sum(axis=1, keepdims=True), p_test_avg


def main():
    args = parse_args(); tee = _Tee(sys.stdout); sys.stdout = tee
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad: raise SystemExit(f"unknown --methods {bad}; choose from {ALL_METHODS}")
    if "distpfn" in methods and "ours_c0" not in methods: raise SystemExit("--methods distpfn needs ours_c0")
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None: args.data = cfg[args.target_dataset]["default_data"]
    args.experiment = f"exp44_conflict_only_{args.conflict_only}"
    device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    args.resolved_device = device; os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = cfg[args.target_dataset]["loader"](args)
    C = len(class_names); print(f"target={args.target_dataset} classes {class_names}; train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
    pipe_counts = {s: np.bincount(label_fn(ix), minlength=C) for s, ix in [("train", train_idx), ("val", val_idx), ("test", test_idx)]}

    # ---- contradiction handling on ALL rows, duplicates untouched
    d = core.load_pickle(args.data); scen = np.asarray(d["attack_scenarios"]).astype(str); tsm = np.asarray(d["timestamps"], dtype=np.int64); del d
    all_idx = np.sort(np.concatenate([train_idx, val_idx, test_idx])); y_all = label_fn(all_idx)
    hcache = os.path.join(args.resume_dir, f"exp41_hash_{args.target_dataset}.npy"); t0 = time.time()
    if os.path.exists(hcache): h_all = np.load(hcache)
    else:
        h_all = np.concatenate([pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(np.asarray(X[all_idx[i:i + 2_000_000]], dtype=np.float32))), index=False).to_numpy() for i in range(0, len(all_idx), 2_000_000)]); np.save(hcache, h_all)
    df = pd.DataFrame({"h": h_all, "y": y_all}); cnt = df.groupby(["h", "y"]).size().reset_index(name="c")
    maj = cnt.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]
    nlab = cnt.groupby("h").size(); conflict_h = set(nlab[nlab > 1].index)
    is_conf = df.h.isin(conflict_h).to_numpy()
    if args.conflict_only == "relabel_majority":
        keep = np.ones(len(all_idx), bool)
        y_new = y_all.copy(); y_new[is_conf] = df.h.map(maj).to_numpy()[is_conf]
        n_relabelled = int((y_new != y_all).sum())
    else:
        keep = ~is_conf; y_new = y_all.copy(); n_relabelled = 0
    sub = all_idx[keep]; y_sub = y_new[keep]
    lab = pd.Series(y_new, index=all_idx)          # the arm's label for every surviving row
    label_arm = lambda ix: lab.loc[ix].to_numpy().astype(np.int16)
    tr, va, te, small = [], [], [], []
    for s in np.unique(scen[sub]):
        sel = sub[scen[sub] == s]; sel = sel[np.argsort(tsm[sel], kind="stable")]
        if len(sel) < args.min_scenario_rows: tr.append(sel); small.append((str(s), int(len(sel)))); continue
        k1, k2 = int(len(sel) * .6), int(len(sel) * .8); tr.append(sel[:k1]); va.append(sel[k1:k2]); te.append(sel[k2:])
    train_idx = np.sort(np.concatenate(tr)); val_idx = np.sort(np.concatenate(va)); test_idx = np.sort(np.concatenate(te))
    y_train_all = label_arm(train_idx); y_val_all = label_arm(val_idx); y_test_all = label_arm(test_idx)
    hpos = pd.Series(h_all, index=all_idx); htr = hpos.loc[train_idx].to_numpy(); hte = hpos.loc[test_idx].to_numpy()
    X_test = to_float32_finite(X[test_idx], "X_test")
    print(f"conflict-only={args.conflict_only}: rows {len(all_idx):,} -> {len(sub):,} (모순 벡터 행 {int(is_conf.sum()):,}, 재라벨 {n_relabelled:,}); "
          f"train {len(train_idx):,} val {len(val_idx):,} test {len(test_idx):,}; {time.time() - t0:.0f}s")

    arm_counts = {s: np.bincount(y, minlength=C) for s, y in [("train", y_train_all), ("val", y_val_all), ("test", y_test_all)]}
    class_counts = pd.DataFrame([dict(split=s, cls=class_names[c], rows_pipeline=int(pipe_counts[s][c]), rows_arm=int(arm_counts[s][c]),
                                      delta=int(arm_counts[s][c] - pipe_counts[s][c]),
                                      kept_frac=(float(arm_counts[s][c] / pipe_counts[s][c]) if pipe_counts[s][c] else float("nan")))
                                 for s in ("train", "val", "test") for c in range(C)])
    print("\nper-class rows (pipeline split -> this arm)"); print(class_counts.pivot(index="cls", columns="split", values=["rows_pipeline", "rows_arm"]).to_string())

    cnt2 = pd.DataFrame({"h": htr, "y": y_train_all}).groupby(["h", "y"]).size().reset_index(name="c"); maj2 = cnt2.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]
    te_maj = pd.Series(hte).map(maj2); seen = te_maj.notna().to_numpy(); same = seen & (te_maj.to_numpy() == y_test_all); diff = seen & ~same; unseen = ~seen
    tcnt = pd.DataFrame({"h": hte, "y": y_test_all}).groupby(["h", "y"]).size().reset_index(name="c"); tmaj = tcnt.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]

    preds, timings = {}, {"conflict_only": args.conflict_only, "rows_after": int(len(sub)), "rows_conflicting": int(is_conf.sum()),
                          "rows_relabelled": n_relabelled, "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)),
                          "test_rows": int(len(test_idx)), "small_train_only": small}
    if "oracle" in methods: preds["realistic_oracle"] = pd.Series(hte).map(tmaj).to_numpy().astype(np.int16)

    need_c0 = any(m in methods for m in ("xgb_c0", "ours_c0", "boostpfn"))
    c0_info = c0ctx.build_c0(args, train_idx, y_train_all, class_names, label_arm) if need_c0 else None
    used_idx = c0_info["idx"] if need_c0 else None
    plain_idx = core.stratified_subset(train_idx, y_train_all, C, args.c0_size, args.seed + 850) if "tabpfn_plain" in methods else None
    X_used = to_float32_finite(X[used_idx], "X_c0") if need_c0 else None
    y_used = label_arm(used_idx) if need_c0 else None
    X_plain = to_float32_finite(X[plain_idx], "X_plain") if plain_idx is not None else None
    y_plain = label_arm(plain_idx) if plain_idx is not None else None
    X_full = to_float32_finite(X[train_idx], "X_train_full") if "xgb_full" in methods else None
    del X; core._PICKLE_CACHE.clear(); gc.collect()
    if need_c0: print(f"C0 rows {len(used_idx):,}; per class {dict(zip(class_names, np.bincount(y_used, minlength=C).tolist()))}")

    def fit_xgb(Xc, yc, tag):
        t = time.time(); bst = xgb.XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth, learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample_bytree, min_child_weight=args.xgb_min_child_weight, reg_lambda=args.xgb_reg_lambda,
            objective="multi:softprob", num_class=C, n_jobs=-1, random_state=args.seed)
        print(f"  [{tag}] fitting on {len(Xc):,} rows ...", flush=True); bst.fit(Xc, yc)
        p = bst.predict(X_test); timings[f"{tag}_seconds"] = round(time.time() - t, 1); del bst; gc.collect(); return p.astype(np.int16)

    if "xgb_full" in methods:
        preds["xgboost_full"] = fit_xgb(X_full, y_train_all, "xgb_full"); del X_full; gc.collect()
    if "xgb_c0" in methods: preds["xgboost_c0"] = fit_xgb(X_used, y_used, "xgb_c0")

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
            del clf; gc.collect()
            if device.startswith("cuda"): torch.cuda.empty_cache()
            return np.concatenate(out), proba
        if "tabpfn_plain" in methods: preds["tabpfn_plain"], _ = run_tabpfn(X_plain, y_plain, "tabpfn_plain", False)
        if "ours_c0" in methods:
            preds["ours_c0"], proba_ours = run_tabpfn(X_used, y_used, "ours_c0", "distpfn" in methods)
            if "distpfn" in methods:
                prior = np.bincount(y_used, minlength=C) / len(y_used)
                adj, p_avg = distpfn_adjust(proba_ours.astype(np.float64), prior)
                preds["distpfn"] = adj.argmax(1).astype(np.int16)
                timings["distpfn_prior_train"] = prior.round(6).tolist(); del adj, proba_ours; gc.collect()

    if "boostpfn" in methods:
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
        proba = np.asarray(clf.predict_proba(torch.from_numpy(X_test))); timings["boostpfn_seconds"] = round(time.time() - t, 1)
        preds["boostpfn"] = proba.argmax(1).astype(np.int16); del clf, base, proba; gc.collect()

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
    out_dir = os.path.join(args.out_root, f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp44_conflict_only_{args.conflict_only}_{tag}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "1b_summary.csv"), index=False)
    class_counts.to_csv(os.path.join(out_dir, "0c_class_counts.csv"), index=False)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx)
    if need_c0:
        c0_info["pool_audit"].to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False); timings.update(c0_info["timings"])
    for method, pred in preds.items(): np.save(os.path.join(out_dir, f"pred_{method}.npy"), pred.astype(np.int16))
    timings.update({"experiment": args.experiment, "seed": args.seed, "device": device})
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2, default=str)
    json.dump([timings], open(os.path.join(out_dir, "timings.json"), "w"), indent=2)
    print(f"\nWrote {out_dir}"); open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8").write(tee.text()); open(os.path.join(out_dir, "COMPLETE.json"), "w").write("{}\n")


if __name__ == "__main__":
    main()

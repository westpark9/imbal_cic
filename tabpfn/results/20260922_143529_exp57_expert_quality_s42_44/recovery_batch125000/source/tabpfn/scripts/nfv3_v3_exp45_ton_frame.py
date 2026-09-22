#!/usr/bin/env python3
"""EXP45 (frozen) -- E0: the comparison FRAME on ToN-IoT's original chrono split (no dedup, no routing).

Why (0915.md S4/S6): CIC2018 is exhausted for model-side knobs (row-wise ours/XGB-c0 complementarity oracle
+0.017, composition lens <= +0.01), while ToN-IoT's realistic per-class ceilings leave ~0.17-0.20 macro, almost
all of it the PRECISION of two clean tail classes (full XGB 0818 run 20260818_113117: mitm F1 0.162 with
P 0.09 / R 0.73, ransomware 0.150 with P 0.08 / R 0.94; hash ceilings 0.987 / 0.948; unique vectors 90.6% / 90.8%).
ours (composed C0) has never been scored on ton_iot -- 0821's exp14 global was a natural-proportion context that
collapsed on benign heterogeneity (benign 0.79, ddos 0.20). Before any knob can be read on ToN, the reference
columns must exist in ONE run on ONE split.

Methods (exp42's core set; SOTA arms stay in their own scripts):
  oracle        realistic oracle -- same vector -> its test-majority label = hash-consistent ceiling per class
  xgb_full      XGBoost on the whole train pool (fitted here; later seeds reuse it via --xgb-full-run, test_idx verified)
  xgb_c0        XGBoost on the same C0 rows ours gets (matched information budget)
  tabpfn_plain  TabPFN-v3, context = --c0-size ratio-preserving random rows (the naive context)
  ours_c0       TabPFN-v3, context = exp31 composed C0 (benign share 0.75, attacks balanced, n_estimators 4)

Differences from exp42, all deliberate: (1) NO dataset-level dedup and NO re-split -- the loader's chrono split is
the benchmark (the 0818 XGB record and the 0911 ceilings live on it); (2) hashes only serve the test buckets
(seen_same_label / seen_diff_label / unseen vs the train pool) and the oracle; (3) every model's probabilities are
saved (float16) and so are the context row indices (c0_idx.npy, plain_idx.npy) -- 0915 S2c: EXP43's context
identity could not be verified because no index dump existed; (4) 1c_class_ceiling.csv puts the oracle ceiling,
seen/conflict fractions and every method's F1 side by side per class.

Run (seed 42 fits XGB full; 43/44 reuse it):
  python tabpfn/scripts/nfv3_v3_exp45_ton_frame.py --target-dataset ton_iot --seed 42
  python tabpfn/scripts/nfv3_v3_exp45_ton_frame.py --target-dataset ton_iot --seed 43 --xgb-full-run <seed-42 out dir>
Smoke:
  python tabpfn/scripts/nfv3_v3_exp45_ton_frame.py --target-dataset ton_iot_capped --c0-size 20000 --test-batch-size 100000
"""
import gc, json, os, sys, time
import numpy as np, pandas as pd, torch, xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__)); TABPFN_DIR = os.path.abspath(os.path.join(HERE, ".."))
IMBALCIC_ROOT = os.path.abspath(os.path.join(TABPFN_DIR, "..")); sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))
import nfv3_v3_common as core  # noqa: E402
import nfv3_v3_c0_context as c0ctx  # noqa: E402

ALL_METHODS = ["oracle", "xgb_full", "xgb_c0", "tabpfn_plain", "ours_c0"]


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
                   n_estimators=4, fit_mode="fit_with_cache", test_batch_size=500_000, test_cap_per_class=0, target_dataset="ton_iot")
    g = p.add_argument_group("EXP45 ToN-IoT frame")
    g.add_argument("--methods", default="oracle,xgb_full,xgb_c0,tabpfn_plain,ours_c0", help="comma list from " + ",".join(ALL_METHODS))
    g.add_argument("--xgb-full-run", default=None, help="an earlier exp45 out dir whose pred_xgboost_full.npy / proba_xgboost_full.npy are reused (test_idx verified)")
    g.add_argument("--no-save-proba", action="store_true", help="skip the float16 probability dumps (~C x test_rows x 2 bytes per model)")
    c0ctx.add_c0_args(p); p.set_defaults(context_recipe="c0")
    return p.parse_args()


def to_float32_finite(block, name):
    arr = np.asarray(block, dtype=np.float32)
    if not np.isfinite(arr).all():
        print(f"  {name}: non-finite values -> nan_to_num"); arr = np.nan_to_num(arr)
    return arr


def main():
    args = parse_args(); tee = _Tee(sys.stdout); sys.stdout = tee
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad: raise SystemExit(f"unknown --methods {bad}; choose from {ALL_METHODS}")
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None: args.data = cfg[args.target_dataset]["default_data"]
    args.experiment = "exp45_ton_frame"
    device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    args.resolved_device = device; os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = cfg[args.target_dataset]["loader"](args)
    C = len(class_names); y_val_all = label_fn(val_idx)
    print(f"target={args.target_dataset} classes {class_names}; train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    # ---- hashes over the pipeline split (same definition and cache name as exp41/42: sorted train+val+test)
    all_idx = np.sort(np.concatenate([train_idx, val_idx, test_idx])); t0 = time.time()
    hcache = os.path.join(args.resume_dir, f"exp41_hash_{args.target_dataset}.npy")
    if os.path.exists(hcache):
        h_all = np.load(hcache)
        if len(h_all) != len(all_idx): raise SystemExit(f"{hcache} has {len(h_all):,} hashes for {len(all_idx):,} rows -- stale cache")
    else:
        h_all = np.concatenate([pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(np.asarray(X[all_idx[i:i + 2_000_000]], dtype=np.float32))), index=False).to_numpy()
                                for i in range(0, len(all_idx), 2_000_000)]); np.save(hcache, h_all)
    hpos = pd.Series(h_all, index=all_idx); htr = hpos.loc[train_idx].to_numpy(); hte = hpos.loc[test_idx].to_numpy()
    print(f"hashing {time.time() - t0:.0f}s; unique vectors train {pd.unique(htr).size:,} test {pd.unique(hte).size:,}")
    cnt2 = pd.DataFrame({"h": htr, "y": y_train_all}).groupby(["h", "y"]).size().reset_index(name="c"); maj2 = cnt2.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]
    te_maj = pd.Series(hte).map(maj2); seen = te_maj.notna().to_numpy(); same = seen & (te_maj.to_numpy() == y_test_all); diff = seen & ~same; unseen = ~seen
    tcnt = pd.DataFrame({"h": hte, "y": y_test_all}).groupby(["h", "y"]).size().reset_index(name="c"); tmaj = tcnt.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]

    counts = {s: np.bincount(y, minlength=C) for s, y in [("train", y_train_all), ("val", y_val_all), ("test", y_test_all)]}
    class_counts = pd.DataFrame([dict(split=s, cls=class_names[c], rows=int(counts[s][c])) for s in ("train", "val", "test") for c in range(C)])
    print("\nper-class rows\n" + class_counts.pivot(index="cls", columns="split", values="rows").to_string())

    preds, probas, timings = {}, {}, {"methods": methods, "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "test_rows": int(len(test_idx))}
    X_test = to_float32_finite(X[test_idx], "X_test")
    if "oracle" in methods: preds["realistic_oracle"] = pd.Series(hte).map(tmaj).to_numpy().astype(np.int16)

    # ---- contexts from the train pool: composed C0 (ours / xgb_c0) and the ratio-preserving plain draw
    need_c0 = any(m in methods for m in ("xgb_c0", "ours_c0"))
    c0_info = c0ctx.build_c0(args, train_idx, y_train_all, class_names, label_fn) if need_c0 else None
    used_idx = c0_info["idx"] if need_c0 else None
    plain_idx = core.stratified_subset(train_idx, y_train_all, C, args.c0_size, args.seed + 850) if "tabpfn_plain" in methods else None
    X_used = to_float32_finite(X[used_idx], "X_c0") if need_c0 else None; y_used = label_fn(used_idx) if need_c0 else None
    X_plain = to_float32_finite(X[plain_idx], "X_plain") if plain_idx is not None else None; y_plain = label_fn(plain_idx) if plain_idx is not None else None
    X_train_full = to_float32_finite(X[train_idx], "X_train") if ("xgb_full" in methods and not args.xgb_full_run) else None
    del X; core._PICKLE_CACHE.clear(); gc.collect()
    if need_c0: print(f"C0 rows {len(used_idx):,}; per class {dict(zip(class_names, np.bincount(y_used, minlength=C).tolist()))}")
    if plain_idx is not None: print(f"plain context rows {len(plain_idx):,}; per class {dict(zip(class_names, np.bincount(y_plain, minlength=C).tolist()))}")

    def fit_xgb(Xc, yc, tag):
        t = time.time(); bst = xgb.XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth, learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample_bytree, min_child_weight=args.xgb_min_child_weight, reg_lambda=args.xgb_reg_lambda,
            objective="multi:softprob", num_class=C, n_jobs=-1, random_state=args.seed)
        bst.fit(Xc, yc); timings[f"{tag}_fit_seconds"] = round(time.time() - t, 1); t = time.time()
        pb = np.concatenate([bst.predict_proba(X_test[s:s + 1_000_000]).astype(np.float32) for s in range(0, len(X_test), 1_000_000)])
        timings[f"{tag}_predict_seconds"] = round(time.time() - t, 1); del bst; gc.collect()
        return pb.argmax(1).astype(np.int16), pb

    if "xgb_full" in methods:
        if args.xgb_full_run:
            ti = np.load(os.path.join(args.xgb_full_run, "test_idx.npy"))
            if not np.array_equal(ti, test_idx): raise SystemExit(f"xgb_full test_idx mismatch: {len(ti):,} recorded vs {len(test_idx):,} here")
            preds["xgboost_full"] = np.load(os.path.join(args.xgb_full_run, "pred_xgboost_full.npy")).astype(np.int16)
            pp = os.path.join(args.xgb_full_run, "proba_xgboost_full.npy")
            if os.path.exists(pp): probas["xgboost_full"] = np.load(pp)
            timings["xgb_full_reused_from"] = args.xgb_full_run; print(f"xgb_full: reused {os.path.basename(args.xgb_full_run)} ({len(ti):,} test rows verified identical)")
        else:
            preds["xgboost_full"], probas["xgboost_full"] = fit_xgb(X_train_full, y_train_all, "xgb_full"); del X_train_full; gc.collect()
    if "xgb_c0" in methods: preds["xgboost_c0"], probas["xgboost_c0"] = fit_xgb(X_used, y_used, "xgb_c0")

    # ---- TabPFN-v3 on the plain context and on the composed C0 (fit = prompt loading, no weights change)
    if "tabpfn_plain" in methods or "ours_c0" in methods:
        from tabpfn import TabPFNClassifier  # noqa: E402
        def run_tabpfn(Xc, yc, tag):
            clf = TabPFNClassifier(device=device, model_path=args.model_path, ignore_pretraining_limits=args.ignore_pretraining_limits,
                                   inference_config={"SUBSAMPLE_SAMPLES": int(args.subsample_samples) or None}, random_state=args.seed,
                                   n_estimators=args.n_estimators, auto_scale_n_estimators=False, fit_mode=args.fit_mode, keep_cache_on_device=args.keep_cache_on_device)
            if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
            t = time.time(); clf.fit(Xc, yc); timings[f"{tag}_fit_seconds"] = round(time.time() - t, 1)
            t = time.time(); bs = args.test_batch_size if args.test_batch_size > 0 else len(X_test); proba = np.empty((len(X_test), C), dtype=np.float32)
            for s in range(0, len(X_test), bs):
                pb = clf.predict_proba(X_test[s:s + bs]).astype(np.float32); proba[s:s + len(pb)] = pb
                print(f"  [{tag}] predict rows {s:,}:{min(s + bs, len(X_test)):,}", flush=True)
            timings[f"{tag}_predict_seconds"] = round(time.time() - t, 1)
            if device.startswith("cuda"): timings[f"{tag}_gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            assert list(clf.classes_) == list(range(C)), clf.classes_
            del clf; gc.collect(); torch.cuda.empty_cache() if device.startswith("cuda") else None
            return proba.argmax(1).astype(np.int16), proba
        # ours first, plain second: the natural control of any later context knob must share this process order (0914 S5 RNG-order effect)
        if "ours_c0" in methods: preds["ours_c0"], probas["ours_c0"] = run_tabpfn(X_used, y_used, "ours_c0")
        if "tabpfn_plain" in methods: preds["tabpfn_plain"], probas["tabpfn_plain"] = run_tabpfn(X_plain, y_plain, "tabpfn_plain")

    # ---- scoring: all / seen_same_label / seen_diff_label / unseen, per class; plus the per-class ceiling table
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
    pc = pd.DataFrame(rows); ceil_rows = []
    for c in range(C):
        yc = y_test_all == c; rec = dict(cls=class_names[c], test_rows=int(yc.sum()), train_rows=int(counts["train"][c]),
                                          test_seen_frac=float(seen[yc].mean()) if yc.any() else float("nan"), test_seen_diff_frac=float(diff[yc].mean()) if yc.any() else float("nan"))
        for method in preds:
            x = pc[(pc.method == method) & (pc.bucket == "all") & (pc.cls == class_names[c])]
            rec[f"f1_{method}"] = float(x.f1.iloc[0]) if len(x) else float("nan")
        ceil_rows.append(rec)
    ceiling = pd.DataFrame(ceil_rows); print("\nper-class ceiling table\n" + ceiling.round(4).to_string(index=False))

    ts = time.strftime("%Y%m%d_%H%M%S"); tag = "_".join(sorted(methods))[:40]
    out_dir = os.path.join(args.out_root, f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp45_ton_frame_{tag}_s{args.seed}"); os.makedirs(out_dir, exist_ok=True)
    pc.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False); pd.DataFrame(summary).to_csv(os.path.join(out_dir, "1b_summary.csv"), index=False)
    class_counts.to_csv(os.path.join(out_dir, "0c_class_counts.csv"), index=False); ceiling.to_csv(os.path.join(out_dir, "1c_class_ceiling.csv"), index=False)
    if hasattr(split_audit, "to_csv"): split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx)
    if need_c0:
        np.save(os.path.join(out_dir, "c0_idx.npy"), used_idx); c0_info["pool_audit"].to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
        if c0_info["c0_scenario"] is not None: c0_info["c0_scenario"].to_csv(os.path.join(out_dir, "0g_c0_scenario.csv"), index=False)
        timings.update(c0_info["timings"])
    if plain_idx is not None: np.save(os.path.join(out_dir, "plain_idx.npy"), plain_idx)
    for method, pred in preds.items(): np.save(os.path.join(out_dir, f"pred_{method}.npy"), pred.astype(np.int16))
    if not args.no_save_proba:
        for method, pb in probas.items(): np.save(os.path.join(out_dir, f"proba_{method}.npy"), pb.astype(np.float16))
    timings.update({"experiment": args.experiment, "seed": args.seed, "device": device, "test_buckets": {"seen_same_label": int(same.sum()), "seen_diff_label": int(diff.sum()), "unseen": int(unseen.sum())}})
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2, default=str); json.dump([timings], open(os.path.join(out_dir, "timings.json"), "w"), indent=2)
    print(f"\nWrote {out_dir}"); open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8").write(tee.text()); open(os.path.join(out_dir, "COMPLETE.json"), "w").write("{}\n")


if __name__ == "__main__":
    main()

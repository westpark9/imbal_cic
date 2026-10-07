#!/usr/bin/env python3
"""EXP41 (TabPFN arm, frozen) -- DATASET-level de-duplication BEFORE the chronological split: once identical-vector
conflicts are removed from both train and test, does the composed-context global (exp31 recipe) score higher?

One knob: --dataset-dedup applied to ALL rows of the target dataset, then the per-scenario chronological 60/20/20 split
is redone on what remains (scenarios with < --min-scenario-rows rows go to train only, as EXP18 did):
  original        no change (test = the original 20%)
  dedup_xy        one row per (46-feature vector, label), earliest by time; conflicts stay as one row per label
  dedup_majority  one row per vector, label = majority label of that vector over the whole dataset
  drop_conflict   one row per vector; vectors carrying >1 label anywhere in the dataset removed
Test = the variant's own test rows; the realistic oracle (conflict vectors -> majority) is reported on that test.
Lineage: copy of nfv3_v3_exp40_dedup_tabpfn.py (train-pool-only dedup). XGBoost is fitted on the same C0 rows
for the paired XGB-c0 number. Reported: per-class P/R/F1 + macro (per_class_metrics.csv) and the same
on the three test buckets (vector seen in the full train pool with the same majority label / with a
different label / unseen). Lineage: copy of nfv3_v3_exp37_distpfn.py minus DistPFN, plus the knob.
Run:  python tabpfn/scripts/nfv3_v3_exp41_dedup_dataset_tabpfn.py --target-dataset cic2018 --dataset-dedup original --seed 42
"""
import gc, json, os, sys, time
import numpy as np, pandas as pd, torch, xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__)); TABPFN_DIR = os.path.abspath(os.path.join(HERE, ".."))
IMBALCIC_ROOT = os.path.abspath(os.path.join(TABPFN_DIR, "..")); sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))
import nfv3_v3_common as core  # noqa: E402
import nfv3_v3_c0_context as c0ctx  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402


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
    g = p.add_argument_group("EXP41 dataset dedup")
    g.add_argument("--dataset-dedup", default="original", choices=["original", "dedup_xy", "dedup_majority", "drop_conflict"])
    g.add_argument("--min-scenario-rows", type=int, default=5)
    c0ctx.add_c0_args(p); p.set_defaults(context_recipe="c0")
    return p.parse_args()


def to_float32_finite(block, name):
    arr = np.asarray(block, dtype=np.float32)
    if not np.isfinite(arr).all():
        print(f"  {name}: non-finite values -> nan_to_num"); arr = np.nan_to_num(arr)
    return arr


def main():
    args = parse_args(); tee = _Tee(sys.stdout); sys.stdout = tee
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None: args.data = cfg[args.target_dataset]["default_data"]
    args.experiment = f"exp41_dedup_dataset_tabpfn_{args.dataset_dedup}"
    device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    args.resolved_device = device; os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = cfg[args.target_dataset]["loader"](args)
    C = len(class_names); print(f"target={args.target_dataset} classes {class_names}; train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
    # ---- dataset-level dedup BEFORE the split (hash cache shared across arms/seeds)
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
    train_idx = np.sort(np.concatenate(tr)); val_idx = np.sort(np.concatenate(va)); test_idx = np.sort(np.concatenate(te)); y_train_all = label_fn(train_idx); y_test_all = label_fn(test_idx)
    hpos = pd.Series(h_all, index=all_idx); htr = hpos.loc[train_idx].to_numpy(); hte = hpos.loc[test_idx].to_numpy(); X_test = to_float32_finite(X[test_idx], "X_test")
    print(f"dataset-dedup={args.dataset_dedup}: rows {len(all_idx):,} -> {len(sub):,}; train {len(train_idx):,} val {len(val_idx):,} test {len(test_idx):,}; train-only small scenarios {small}; hashing/split {time.time() - t0:.0f}s")
    print("test per class " + str({class_names[c]: int((y_test_all == c).sum()) for c in range(C)}))
    # buckets on the (new) train pool; realistic oracle on the (new) test
    cnt2 = pd.DataFrame({"h": htr, "y": y_train_all}).groupby(["h", "y"]).size().reset_index(name="c"); maj2 = cnt2.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]
    te_maj = pd.Series(hte).map(maj2); seen = te_maj.notna().to_numpy(); same = seen & (te_maj.to_numpy() == y_test_all); diff = seen & ~same; unseen = ~seen
    tcnt = pd.DataFrame({"h": hte, "y": y_test_all}).groupby(["h", "y"]).size().reset_index(name="c"); tmaj = tcnt.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]
    oracle_pred = pd.Series(hte).map(tmaj).to_numpy().astype(np.int16)
    pool_idx = train_idx; y_pool = y_train_all
    # ---- C0 draw exactly as exp31/37 (on the filtered pool)
    c0_info = c0ctx.build_c0(args, pool_idx, y_pool, class_names, label_fn); used_idx = c0_info["idx"]; y_used = label_fn(used_idx)
    X_used = to_float32_finite(X[used_idx], "X_c0"); del X; core._PICKLE_CACHE.clear(); gc.collect()
    used_counts = np.bincount(y_used, minlength=C); print(f"C0 rows {len(used_idx):,}; per class {dict(zip(class_names, used_counts.tolist()))}")
    preds, timings = {"realistic_oracle": oracle_pred}, {"dataset_dedup": args.dataset_dedup, "rows_after_dedup": int(len(sub)), "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)), "c0_rows": int(len(used_idx)), "small_train_only": small}
    # ---- XGBoost on the same C0 rows (paired)
    if not args.skip_xgboost:
        t0 = time.time(); booster = xgb.XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth, learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample_bytree, min_child_weight=args.xgb_min_child_weight, reg_lambda=args.xgb_reg_lambda,
            objective="multi:softprob", num_class=C, n_jobs=-1, random_state=args.seed)
        booster.fit(X_used, y_used); preds["xgboost_c0"] = booster.predict(X_test); timings["xgb_seconds"] = round(time.time() - t0, 1)
    # ---- TabPFN-v3
    if not args.skip_tabpfn:
        clf = TabPFNClassifier(device=device, model_path=args.model_path, ignore_pretraining_limits=args.ignore_pretraining_limits,
                               inference_config={"SUBSAMPLE_SAMPLES": int(args.subsample_samples) or None}, random_state=args.seed,
                               n_estimators=args.n_estimators, auto_scale_n_estimators=False, fit_mode=args.fit_mode, keep_cache_on_device=args.keep_cache_on_device)
        if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); clf.fit(X_used, y_used); timings["tabpfn_fit_seconds"] = round(time.time() - t0, 1)
        t0 = time.time(); bs = args.test_batch_size if args.test_batch_size > 0 else len(X_test); chunks = []
        for s in range(0, len(X_test), bs):
            chunks.append(clf.predict_proba(X_test[s:s + bs]).argmax(1).astype(np.int16)); print(f"  predict rows {s:,}:{min(s + bs, len(X_test)):,}", flush=True)
        preds["tabpfn_v3"] = np.concatenate(chunks); timings["tabpfn_predict_seconds"] = round(time.time() - t0, 1)
        if device.startswith("cuda"): timings["tabpfn_gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        assert list(clf.classes_) == list(range(C)), clf.classes_
    # ---- scoring
    rows, summary = [], []
    for method, pred in preds.items():
        for bname, m in [("all", np.ones(len(y_test_all), bool)), ("seen_same_label", same), ("seen_diff_label", diff), ("unseen", unseen)]:
            if not m.any():  # empty bucket (seen_diff_label vanishes after dedup_majority/drop_conflict/dedup_xy). Guard added 2026-09-11 after the 3 original-arm runs; no effect on non-empty buckets.
                summary.append(dict(method=method, bucket=bname, rows=0, accuracy=float("nan"), macro_f1=float("nan"), tail_f1=float("nan"))); continue
            pr, rc, f1, sup = precision_recall_fscore_support(y_test_all[m], pred[m], labels=range(C), zero_division=0)
            for c in range(C): rows.append(dict(method=method, bucket=bname, cls=class_names[c], support=int(sup[c]), precision=pr[c], recall=rc[c], f1=f1[c]))
            summary.append(dict(method=method, bucket=bname, rows=int(m.sum()), accuracy=float((pred[m] == y_test_all[m]).mean()), macro_f1=float(f1.mean()),
                                tail_f1=float(np.mean([f1[class_names.index(t)] for t in tail_classes]))))
        f_all = [r for r in rows if r["method"] == method and r["bucket"] == "all"]
        print(f"[{method}] macro-F1 {np.mean([r['f1'] for r in f_all]):.4f} | " + " ".join(f"{r['cls']}={r['f1']:.3f}" for r in f_all))
    ts = time.strftime("%Y%m%d_%H%M%S"); out_dir = os.path.join(args.out_root, f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp41_dedup_dataset_tabpfn_{args.dataset_dedup}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False); pd.DataFrame(summary).to_csv(os.path.join(out_dir, "1b_summary.csv"), index=False)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx); c0_info["pool_audit"].to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
    if c0_info["c0_scenario"] is not None: c0_info["c0_scenario"].to_csv(os.path.join(out_dir, "0g_c0_scenario.csv"), index=False)
    for method, pred in preds.items(): np.save(os.path.join(out_dir, f"pred_{method}.npy"), pred.astype(np.int16))
    timings.update(c0_info["timings"]); timings.update({"experiment": args.experiment, "seed": args.seed, "device": device, "test_rows": int(len(y_test_all))})
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2, default=str); json.dump([timings], open(os.path.join(out_dir, "timings.json"), "w"), indent=2)
    print(f"\nWrote {out_dir}"); open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8").write(tee.text()); open(os.path.join(out_dir, "COMPLETE.json"), "w").write("{}\n")


if __name__ == "__main__":
    main()

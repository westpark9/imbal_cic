#!/usr/bin/env python3
"""EXP43 (frozen) -- ours v2: WHICH rows fill the attack quotas of the composed context.

Context is the one axis of this project that has ever paid off, so this is a composition knob, not a
selection layer: EXP38/39 showed the conditional-routing layer keeps global in 6/6 conditions (delta = 0),
and SRC_HISTORY buries every test-time selection variant (routing error compounds imbalance, val-fitted
gates do not transfer, energy gates go to chance). exp30-34 already burned the other composition knobs:
benign share (exp30), natural vs balanced attack budget (exp31), C0 bagging (exp32, 0.751-0.758 < single
draw 0.7638), copy caps (exp33), draw selection (exp34). What none of them touched is WHICH rows fill a
class's quota.

THE KNOB --attack-fill:
  natural              exp31 behaviour: uniform draw inside each attack class (= ours / exp42's ours_c0)
  majority_consistent  prefer rows whose 46-feature vector has THIS class as its dataset-wide majority
                       label; fall back to the rest only to finish the quota
  clean                prefer rows whose vector carries no other label anywhere; same fallback

Why it should matter (0911 audit): 75.1% of infiltration rows and 69.6% of web rows share their exact
vector with a differently-labelled row, so a uniform draw spends ~3/4 of infiltration's 4,847 context
slots on vectors that also appear as benign. The realistic oracle answers such a vector with its majority
label, so `majority_consistent` fills the context with exactly what that oracle would say -- same budget,
same class balance, same benign rows, only the identity of the attack rows changes.

Benign is deliberately untouched: exp31's pool-wide dedup and exp33's copy cap both destroyed benign
density and lost (0.636-0.70). The per-class TARGET counts here are computed by the same largest-remainder
code as nfv3_v3_c0_context.representative_subset_share(alloc="balanced"), so a run with
--attack-fill natural reproduces exp42's ours_c0 row-for-row.

Benchmarks are EXP41/EXP42's four arms, dedup + re-split copied verbatim, so results drop straight into
the same per-class tables.

Run:
  python tabpfn/scripts/nfv3_v3_exp43_ours_majfill.py --target-dataset cic2018 \
      --dataset-dedup original --attack-fill majority_consistent --seed 42
"""
import gc, json, os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__)); TABPFN_DIR = os.path.abspath(os.path.join(HERE, ".."))
IMBALCIC_ROOT = os.path.abspath(os.path.join(TABPFN_DIR, "..")); sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))
import nfv3_v3_common as core  # noqa: E402
import nfv3_v3_c0_context as c0ctx  # noqa: E402

SEED_BAND_GLOBAL = c0ctx.SEED_BAND_GLOBAL


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
    g = p.add_argument_group("EXP43 ours v2")
    g.add_argument("--dataset-dedup", default="original", choices=["original", "dedup_xy", "dedup_majority", "drop_conflict"])
    g.add_argument("--min-scenario-rows", type=int, default=5)
    g.add_argument("--attack-fill", default="majority_consistent", choices=["natural", "majority_consistent", "clean"],
                   help="THE KNOB: which rows fill each attack class's quota. Benign is never touched.")
    c0ctx.add_c0_args(p); p.set_defaults(context_recipe="c0")
    return p.parse_args()


def to_float32_finite(block, name):
    arr = np.asarray(block, dtype=np.float32)
    if not np.isfinite(arr).all():
        print(f"  {name}: non-finite values -> nan_to_num"); arr = np.nan_to_num(arr)
    return arr


def balanced_targets(pool_y, n_classes, budget, benign_id, benign_share):
    """Per-class target counts, identical to c0_context.representative_subset_share(alloc='balanced')."""
    counts = np.bincount(pool_y, minlength=n_classes); total = int(counts.sum())
    if budget <= 0 or budget >= total: return counts.copy()
    tgt = np.zeros(n_classes, dtype=np.int64)
    tgt[benign_id] = min(int(round(budget * benign_share)), int(counts[benign_id]))
    rem = budget - int(tgt[benign_id])
    for _ in range(n_classes):
        sub = np.asarray([c for c in range(n_classes) if c != benign_id and tgt[c] < counts[c]], dtype=np.int64)
        if rem <= 0 or len(sub) == 0: break
        raw = np.full(len(sub), rem / len(sub), dtype=np.float64)
        add = np.floor(raw).astype(np.int64)
        for j in np.argsort(-(raw - np.floor(raw))):
            if int(add.sum()) >= rem: break
            add[j] += 1
        add = np.minimum(add, counts[sub] - tgt[sub]); tgt[sub] += add; rem = budget - int(tgt.sum())
    present = counts > 0; tgt[present & (tgt < 1)] = 1
    return tgt


def main():
    args = parse_args(); tee = _Tee(sys.stdout); sys.stdout = tee
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None: args.data = cfg[args.target_dataset]["default_data"]
    args.experiment = f"exp43_ours_majfill_{args.dataset_dedup}_{args.attack_fill}"
    device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    args.resolved_device = device; os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, y_train_all, y_test_all, split_audit, label_fn = cfg[args.target_dataset]["loader"](args)
    C = len(class_names); print(f"target={args.target_dataset} classes {class_names}; train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    # ---- dataset-level dedup BEFORE the split (verbatim from exp41/exp42)
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
    y_train_all = label_fn(train_idx); y_test_all = label_fn(test_idx)
    hpos = pd.Series(h_all, index=all_idx); htr = hpos.loc[train_idx].to_numpy(); hte = hpos.loc[test_idx].to_numpy(); X_test = to_float32_finite(X[test_idx], "X_test")
    print(f"dataset-dedup={args.dataset_dedup}: rows {len(all_idx):,} -> {len(sub):,}; train {len(train_idx):,} val {len(val_idx):,} test {len(test_idx):,}; hashing/split {time.time() - t0:.0f}s")

    # buckets on the (new) train pool
    cnt2 = pd.DataFrame({"h": htr, "y": y_train_all}).groupby(["h", "y"]).size().reset_index(name="c"); maj2 = cnt2.sort_values("c", ascending=False).drop_duplicates("h").set_index("h")["y"]
    te_maj = pd.Series(hte).map(maj2); seen = te_maj.notna().to_numpy(); same = seen & (te_maj.to_numpy() == y_test_all); diff = seen & ~same; unseen = ~seen

    # ---- the same context-pool partition exp31/exp42 use, then THE KNOB on the attack quotas
    t0 = time.time()
    pools, pool_audit, _ = c0ctx.scenario_stratified_partition(train_idx, y_train_all, tsm[train_idx], scen[train_idx],
                                                              args.c0_context_frac, args.c0_expert_frac, class_names)
    ctx_pool = pools["context"]; y_ctx = label_fn(ctx_pool)
    names_l0 = [str(n).lower() for n in class_names]
    benign_id = next((names_l0.index(n) for n in ("benign", "normal") if n in names_l0), None)
    share = args.c0_benign_share if args.c0_benign_share >= 0 else float((y_ctx == benign_id).mean())
    tgt = balanced_targets(y_ctx, C, args.c0_size, benign_id, share)
    c0_seed = int(args.seed + SEED_BAND_GLOBAL)
    h_ctx = hpos.loc[ctx_pool].to_numpy()
    maj_ctx = pd.Series(h_ctx).map(maj).to_numpy()          # dataset-wide majority label of each pool row's vector
    conflicted = pd.Series(h_ctx).isin(conflict_h).to_numpy()
    chosen, fill_audit = [], []
    for cid in range(C):
        k = int(tgt[cid])
        if k <= 0: continue
        rows = ctx_pool[y_ctx == cid]
        if cid == benign_id or args.attack_fill == "natural":
            take = c0ctx.pick_rows(rows, k, c0_seed + cid); pref_used = -1
        else:
            m = (y_ctx == cid) & ((maj_ctx == cid) if args.attack_fill == "majority_consistent" else ~conflicted)
            pref, rest = ctx_pool[m], ctx_pool[(y_ctx == cid) & ~m]
            if len(pref) >= k:
                take = c0ctx.pick_rows(pref, k, c0_seed + cid); pref_used = k
            else:
                take = np.concatenate([pref, c0ctx.pick_rows(rest, k - len(pref), c0_seed + cid)]); pref_used = len(pref)
        chosen.append(np.sort(take))
        fill_audit.append(dict(cls=class_names[cid], target=k, pool_rows=int(len(rows)),
                               preferred_available=int(((y_ctx == cid) & ((maj_ctx == cid) if args.attack_fill == "majority_consistent" else ~conflicted)).sum()) if cid != benign_id else -1,
                               preferred_used=int(pref_used), unique_vectors_used=int(pd.unique(hpos.loc[np.sort(take)].to_numpy()).size)))
    used_idx = np.sort(np.concatenate(chosen)); y_used = label_fn(used_idx)
    fa = pd.DataFrame(fill_audit); print(f"\nC0 ({args.attack_fill}) {len(used_idx):,} rows in {time.time() - t0:.0f}s"); print(fa.to_string(index=False))
    X_used = to_float32_finite(X[used_idx], "X_c0"); del X; core._PICKLE_CACHE.clear(); gc.collect()

    # ---- frozen TabPFN-v3 on that context
    from tabpfn import TabPFNClassifier  # noqa: E402
    timings = {"dataset_dedup": args.dataset_dedup, "attack_fill": args.attack_fill, "train_rows": int(len(train_idx)),
               "test_rows": int(len(test_idx)), "c0_rows": int(len(used_idx)), "c0_seed": c0_seed,
               "c0_unique_vectors": int(pd.unique(hpos.loc[used_idx].to_numpy()).size)}
    clf = TabPFNClassifier(device=device, model_path=args.model_path, ignore_pretraining_limits=args.ignore_pretraining_limits,
                           inference_config={"SUBSAMPLE_SAMPLES": int(args.subsample_samples) or None}, random_state=args.seed,
                           n_estimators=args.n_estimators, auto_scale_n_estimators=False, fit_mode=args.fit_mode, keep_cache_on_device=args.keep_cache_on_device)
    if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); clf.fit(X_used, y_used); timings["fit_seconds"] = round(time.time() - t0, 1)
    t0 = time.time(); bs = args.test_batch_size if args.test_batch_size > 0 else len(X_test); out = []
    for s in range(0, len(X_test), bs):
        out.append(clf.predict_proba(X_test[s:s + bs]).argmax(1).astype(np.int16)); print(f"  predict rows {s:,}:{min(s + bs, len(X_test)):,}", flush=True)
    pred = np.concatenate(out); timings["predict_seconds"] = round(time.time() - t0, 1)
    if device.startswith("cuda"): timings["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    assert list(clf.classes_) == list(range(C)), clf.classes_

    method = f"ours_v2_{args.attack_fill}" if args.attack_fill != "natural" else "ours_c0_replica"
    rows, summary = [], []
    for bname, m in [("all", np.ones(len(y_test_all), bool)), ("seen_same_label", same), ("seen_diff_label", diff), ("unseen", unseen)]:
        if not m.any():
            summary.append(dict(method=method, bucket=bname, rows=0, accuracy=float("nan"), macro_f1=float("nan"), tail_f1=float("nan"))); continue
        pr, rc, f1, sup = precision_recall_fscore_support(y_test_all[m], pred[m], labels=range(C), zero_division=0)
        for c in range(C): rows.append(dict(method=method, bucket=bname, cls=class_names[c], support=int(sup[c]), precision=pr[c], recall=rc[c], f1=f1[c]))
        summary.append(dict(method=method, bucket=bname, rows=int(m.sum()), accuracy=float((pred[m] == y_test_all[m]).mean()), macro_f1=float(f1.mean()),
                            tail_f1=float(np.mean([f1[class_names.index(t)] for t in tail_classes]))))
    f_all = [r for r in rows if r["bucket"] == "all"]
    print(f"[{method}] macro-F1 {np.mean([r['f1'] for r in f_all]):.4f} | " + " ".join(f"{r['cls']}={r['f1']:.3f}" for r in f_all))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp43_ours_{args.attack_fill}_{args.dataset_dedup}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "1b_summary.csv"), index=False)
    fa.to_csv(os.path.join(out_dir, "0d_fill_audit.csv"), index=False)
    pool_audit.to_csv(os.path.join(out_dir, "0a_pool_partition.csv"), index=False)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx); np.save(os.path.join(out_dir, f"pred_{method}.npy"), pred)
    timings.update({"experiment": args.experiment, "seed": args.seed, "device": device})
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2, default=str)
    json.dump([timings], open(os.path.join(out_dir, "timings.json"), "w"), indent=2)
    print(f"\nWrote {out_dir}"); open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8").write(tee.text()); open(os.path.join(out_dir, "COMPLETE.json"), "w").write("{}\n")


if __name__ == "__main__":
    main()

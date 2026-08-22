#!/usr/bin/env python3
"""EXP20 -- ENERGY-FREE variant: assignment by the plain global classifier.

Pre-registration: 0821.md SS19.  User direction: remove the energy-score
concept from the WHOLE model.  One knob vs exp19: the assignment stage.

  exp19: energy-score gate argmax (similarity-LSE features -> learned heads)
  exp20: plain global TabPFN (natural-proportion 1M) predicts 7-way argmax;
         the predicted class's OWNER expert is activated:
             g_pred = benign        -> label benign (no conditional)
             g_pred in owned(k)     -> expert k's conditional
         conditional (unchanged from exp19): context = owned(k) + benign
         200,000, label = argmax over owned(k) U {benign} (return exit kept).

  NO gates, NO featurize/QK, NO aux, NO learned heads -- energy appears
  nowhere.  The global column of the judgment comes for free (same fit).

Recorded expectation (before the run): routing by global argmax inherits
global's tail recall (inf ~0.05, web ~0.44 on the recorded draw; exp10's
hard_route measured ~= global), so tail activation should collapse vs
exp19's energy gate (inf 0.324, web 0.908).  This run is the CONTROL that
prices what the energy gate buys; within-group corrections (e.g. flood
conditional fixing global's ddos/dos mixups) and the return exit are the
only levers it keeps.

    python tabpfn/nfv3_v3_exp20_global_route.py \\
        --fit-mode fit_with_cache --test-batch-size 500000
"""

import gc
import json
import os
import time

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfv3_v3_common as core  # noqa: E402

from tabpfn import TabPFNClassifier  # noqa: E402

SEED_BAND_CONTEXT = 950
SEED_BAND_GLOBAL = 990
EXPERT_GROUPS = {
    "flood": ["ddos", "dos"],
    "bf_bot": ["brute_force", "bot"],
    "tail": ["infiltration", "web_attacks"],
}


def uniq_rows(Xr):
    h = pd.util.hash_pandas_object(pd.DataFrame(Xr), index=False).values
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    return Xr[first], inv


def natural_targets(counts, k):
    total = int(counts.sum())
    k = min(k, total)
    raw = counts * (k / max(total, 1))
    tgt = np.minimum(np.floor(raw).astype(np.int64), counts)
    present = counts > 0
    tgt[present & (tgt < 1)] = 1
    rem = k - int(tgt.sum())
    for cid in np.argsort(-(raw - np.floor(raw))):
        if rem <= 0:
            break
        if tgt[cid] < counts[cid]:
            tgt[cid] += 1; rem -= 1
    return tgt


def full_proba(proba, model_classes, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=np.int64)] = proba
    return out


def run_exp20(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp20_global_route"
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    if args.subsample_samples:
        raise SystemExit("--subsample-samples must stay 0.")

    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, _, _, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    benign_id = class_names.index("benign")
    experts = {}
    for ename, fams in EXPERT_GROUPS.items():
        ids = [class_names.index(f) for f in fams if f in class_names]
        if ids:
            experts[ename] = ids
    enames = list(experts)
    owner_of = np.full(n_classes, -1, dtype=np.int64)   # -1 = benign/self
    for j, e in enumerate(enames):
        for c in experts[e]:
            owner_of[c] = j
    print("route: benign→benign, " +
          ", ".join(f"{class_names[c]}→{enames[owner_of[c]]}"
                    for c in range(n_classes) if owner_of[c] >= 0))

    eval_idx = core.cap_per_class(test_idx, label_fn(test_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_eval = label_fn(eval_idx)
    X_eval = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))
    Xu_eval, inv_eval = uniq_rows(X_eval)
    print(f"eval rows: {len(eval_idx):,} -> distinct {len(Xu_eval):,}")
    y_train = label_fn(train_idx)

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    def draw_class(c, n, off=0, skip=0):
        perm = np.random.default_rng(args.seed + SEED_BAND_CONTEXT + c + off) \
            .permutation(train_idx[y_train == c])
        return perm[skip: skip + n]

    # conditional contexts: identical construction to exp19 (owned + benign)
    cond_ctx = {}
    for ename, owned in experts.items():
        counts = np.zeros(n_classes, dtype=np.int64)
        for c in owned:
            counts[c] = int((y_train == c).sum())
        tgt = natural_targets(counts, args.context_size)
        parts = []
        for c in owned:
            if tgt[c] == 0:
                continue
            reserve = min(args.holdout_rows, max(1, counts[c] // 5))
            ctx_take = min(int(tgt[c]), int(counts[c]) - reserve)
            if ctx_take < 1:
                raise SystemExit(f"[{ename}] class {class_names[c]} too small.")
            parts.append(draw_class(c, ctx_take))
        cond_ctx[ename] = np.sort(np.concatenate(parts))
        print(f"cond context[{ename}]: owned {len(cond_ctx[ename]):,}")
    benign_ret_idx = np.sort(draw_class(benign_id, args.return_benign_rows,
                                        off=50))
    print(f"conditional benign block: {len(benign_ret_idx):,} rows")
    benign_ret = (feats_of(benign_ret_idx), label_fn(benign_ret_idx))
    cond_data = {e: (feats_of(idxs), label_fn(idxs))
                 for e, idxs in cond_ctx.items()}

    g_idx = core.stratified_subset(train_idx, y_train, n_classes,
                                   args.global_context_size,
                                   args.seed + SEED_BAND_GLOBAL)
    glob_ctx = (feats_of(g_idx), label_fn(g_idx))
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()

    import torch

    def make_clf(Xc, yc):
        clf = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        clf.fit(Xc, yc)
        return clf

    # ---- stage 1: plain global classifier routes ----
    t0 = time.time()
    clf = make_clf(glob_ctx[0], glob_ctx[1])
    bs = args.test_batch_size or len(Xu_eval)
    pr = []
    for s0 in range(0, len(Xu_eval), bs):
        pr.append(clf.predict_proba(Xu_eval[s0:s0 + bs]))
        print(f"[global] predict distinct {min(s0+bs, len(Xu_eval)):,}"
              f"/{len(Xu_eval):,}", flush=True)
    p_glob = full_proba(np.concatenate(pr), clf.classes_,
                        n_classes)[inv_eval].astype(np.float64)
    print(f"[global] fitted+scored in {time.time()-t0:.0f}s", flush=True)
    del clf
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()

    g_pred = np.argmax(p_glob, axis=1)
    route = np.where(g_pred == benign_id, -1, owner_of[g_pred])
    route_names = np.where(route < 0, "benign(직결)",
                           np.asarray(enames + ["?"])[np.clip(route, 0, len(enames)-1)])
    act = pd.DataFrame({
        "class": [class_names[c] for c in y_eval],
        "routed": route_names}).value_counts().reset_index(name="rows")
    print("\n=== routing counts (true class x routed) ===")
    print(act.sort_values(["class", "rows"], ascending=[True, False])
          .to_string(index=False))

    # ---- stage 2: conditionals with benign return ----
    final = np.where(g_pred == benign_id, benign_id, -1)
    ret_stats = []
    for j, ename in enumerate(enames):
        rows = np.flatnonzero(route == j)
        if len(rows) == 0:
            continue
        Xc = np.concatenate([cond_data[ename][0], benign_ret[0]])
        yc = np.concatenate([cond_data[ename][1], benign_ret[1]])
        clf = make_clf(Xc, yc)
        Xr = X_eval[rows]
        Xu, invu = uniq_rows(Xr)
        pr = []
        for s0 in range(0, len(Xu), 500_000):
            pr.append(clf.predict_proba(Xu[s0:s0 + 500_000]))
        pr = full_proba(np.concatenate(pr), clf.classes_, n_classes)[invu]
        allowed = np.asarray(sorted(experts[ename] + [benign_id]))
        final[rows] = allowed[pr[:, allowed].argmax(axis=1)]
        n_ret = int((final[rows] == benign_id).sum())
        ret_stats.append({"expert": ename, "routed": len(rows),
                          "returned_benign": n_ret,
                          "ret_frac": round(n_ret / len(rows), 4)})
        print(f"[{ename}] classified {len(rows):,} routed rows "
              f"— benign 반환 {n_ret:,}", flush=True)
        del clf
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    assert (final >= 0).all()
    ret_df = pd.DataFrame(ret_stats)
    print("\n=== return stats ===")
    print(ret_df.to_string(index=False))

    all_rows = list(core.per_class_table("system_globroute", y_eval, final,
                                         class_names, tail_classes))
    all_rows.extend(core.per_class_table("global_reference", y_eval, g_pred,
                                         class_names, tail_classes))
    table = pd.DataFrame(all_rows)
    print("\n=== per-class F1 ===")
    print(table.pivot(index="class", columns="method", values="f1").round(4)
          .to_string())

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp20_globroute")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    act.to_csv(os.path.join(out_dir, "4a_routing_counts.csv"), index=False)
    ret_df.to_csv(os.path.join(out_dir, "5a_return_stats.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "system_dump.npz"),
                        g_pred=g_pred.astype(np.int8),
                        final=final, y_true=y_eval.astype(np.int64),
                        p_global=p_glob.astype(np.float32),
                        expert_names=np.asarray(enames),
                        class_names=np.asarray(class_names))
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-size", type=int, default=200_000)
    p.add_argument("--holdout-rows", type=int, default=15_000,
                   help="reserve kept identical to exp19's context draw so "
                        "the conditional contexts match exp19 bit-for-bit")
    p.add_argument("--return-benign-rows", type=int, default=200_000)
    p.add_argument("--global-context-size", type=int, default=1_000_000)
    p.set_defaults(
        target_dataset="cic2018",
        max_train_samples=-1,
        n_estimators=1,
        test_cap_per_class=0,
        fit_mode="fit_with_cache",
        test_batch_size=500_000,
        subsample_samples=0,
    )
    args = p.parse_args()
    run_exp20(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EXP16 -- exp12c system on a per-class DEDUPLICATED train pool.

Pre-registration: 0821.md SS11.  One knob vs exp12c: within each class of the
TARGET train pool, exact-duplicate 46-feature rows are collapsed to their
chronologically first occurrence (user direction: "slowhttptest-style
duplicated rows -- delete them all").  Representation, method, grouping,
gates, aux: unchanged.  Test (4.02M) and aux are NOT deduplicated, so the
judgment tuple stays comparable.  The global reference is refit IN THIS RUN
on the same deduplicated pool (0820.md SS12n: same-run comparison columns).

Expected mechanics (recorded before the run): degenerate masses collapse
(dos_attacks-slowhttptest 63,330 train rows -> ~1), natural-proportion
contexts shift from row-proportional to distinct-vector-proportional, and the
duplicate-memorisation / majority-betting signal (0821.md SS5 twin regions)
is removed from the training side.  dos->bf_bot 21,110 on TEST is expected
unchanged (the mislabel block lives in test).

    python tabpfn/nfv3_v3_exp16_dedup_system.py --target-dataset cic2018 \\
        --fit-mode fit_with_cache --test-batch-size 500000 --fit-global
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
SEED_BAND_AUX = 970
SEED_BAND_GLOBAL = 990
AUX_DATASETS = ["bot_iot", "ton_iot"]
AUX_EXCLUDE = {("ton_iot", "scanning"), ("bot_iot", "theft")}
EXP12_GROUPS = {
    "benign": ["benign"],
    "flood": ["ddos", "dos"],
    "bf_bot": ["brute_force", "bot"],
    "tail": ["infiltration", "web_attacks"],
}


# ---------------- helpers (exp12c lineage; frozen scripts copied) ----------

def find_class_attention_module(clf):
    seen = []
    for root in (getattr(clf, "models_", None) or []):
        mod = getattr(root, "model", root)
        try:
            for _, m in mod.named_modules():
                if hasattr(m, "_project_qk") and hasattr(m, "attention_weights"):
                    seen.append(m)
        except AttributeError:
            continue
    return seen[0] if seen else None


def streaming_scores_perhead(q, k, T, chunk_m=2048, chunk_n=65536, device="cpu"):
    import torch
    if q.ndim == 2:
        q = q.unsqueeze(1); k = k.unsqueeze(1)
    M, H, D = q.shape
    scale = 1.0 / np.sqrt(D)
    lse = torch.full((M, H), -torch.inf)
    mx = torch.full((M, H), -torch.inf)
    q = q.to(device)
    for n0 in range(0, k.shape[0], chunk_n):
        kb = k[n0:n0 + chunk_n].to(device)
        for m0 in range(0, M, chunk_m):
            s = torch.einsum("mhd,nhd->mhn", q[m0:m0 + chunk_m], kb).float() * scale
            s = s / T
            lse[m0:m0 + chunk_m] = torch.logaddexp(
                lse[m0:m0 + chunk_m], torch.logsumexp(s, dim=-1).cpu())
            mx[m0:m0 + chunk_m] = torch.maximum(
                mx[m0:m0 + chunk_m], s.max(dim=-1).values.cpu())
            del s
    return (T * lse).numpy(), mx.numpy()


def load_aux_rows(args, cfg, target_families, cap_per_family, seed):
    import copy
    rows, meta = [], []
    suffix = "_capped" if args.target_dataset.endswith("_capped") else ""
    for ds in [d + suffix for d in AUX_DATASETS]:
        a = copy.copy(args)
        a.target_dataset = ds
        a.data = cfg[ds]["default_data"]
        X, class_names, train_idx, val_idx, _, _, _, _, label_fn = cfg[ds]["loader"](a)
        for cid, cname in enumerate(class_names):
            if cname == "benign" or cname in target_families:
                continue
            if (ds.replace("_capped", ""), cname) in AUX_EXCLUDE:
                continue
            pool = np.concatenate([train_idx, val_idx])
            pos = pool[label_fn(pool) == cid]
            if len(pos) == 0:
                continue
            take = np.random.default_rng(seed + SEED_BAND_AUX + cid) \
                .permutation(pos)[:cap_per_family]
            rows.append(np.nan_to_num(np.asarray(X[take], dtype=np.float32)))
            meta.append({"dataset": ds, "family": cname, "rows": len(take)})
        del X
        gc.collect()
    if not rows:
        raise SystemExit("no aux families left")
    return np.concatenate(rows), pd.DataFrame(meta)


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


def train_score_head(feat_pos, feat_neg, seed, pos_labels=None):
    """energy_ood-faithful head training (0820 SS12d): 1:2 pos:neg batches,
    positives class-uniform with replacement, loss = sum of per-portion
    means."""
    import torch
    torch.manual_seed(seed)
    Xall = np.concatenate([feat_pos, feat_neg]).astype(np.float32)
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd == 0] = 1.0
    Xp = torch.tensor((feat_pos - mu) / sd)
    Xn = torch.tensor((feat_neg - mu) / sd)
    if pos_labels is None:
        pos_groups = [np.arange(len(feat_pos))]
    else:
        pos_groups = [np.flatnonzero(pos_labels == c)
                      for c in np.unique(pos_labels)]
    net = torch.nn.Sequential(torch.nn.Linear(Xp.shape[1], 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    n_total = len(Xp) + len(Xn)
    steps = 60 * max(1, n_total // 8192)
    b_pos, b_neg = 2731, 5461                     # 1:2, batch ~8192
    for _ in range(steps):
        cls_pick = torch.randint(len(pos_groups), (b_pos,), generator=g)
        pos_idx = torch.empty(b_pos, dtype=torch.long)
        for ci, grp in enumerate(pos_groups):
            m = cls_pick == ci
            k = int(m.sum())
            if k:
                pos_idx[m] = torch.as_tensor(
                    grp[torch.randint(len(grp), (k,), generator=g).numpy()])
        neg_idx = torch.randint(len(Xn), (b_neg,), generator=g)
        opt.zero_grad()
        lp = lossf(net(Xp[pos_idx]).squeeze(1), torch.ones(b_pos))
        ln = lossf(net(Xn[neg_idx]).squeeze(1), torch.zeros(b_neg))
        (lp + ln).backward()
        opt.step()

    def predict(feats):
        with torch.no_grad():
            z = torch.tensor(((feats - mu) / sd).astype(np.float32))
            return net(z).squeeze(1).numpy()
    return predict


def full_proba(proba, model_classes, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=np.int64)] = proba
    return out


def dedup_train_pool(X, train_idx, y_train, class_names):
    """Per-class exact-duplicate removal: within each class, keep the
    chronologically first row of every distinct 46-feature vector."""
    keep_parts, report = [], []
    for c, cname in enumerate(class_names):
        rows = train_idx[y_train == c]          # loader order = chronological
        if len(rows) == 0:
            continue
        h_parts = []
        for s0 in range(0, len(rows), 2_000_000):
            blk = np.nan_to_num(np.asarray(X[rows[s0:s0 + 2_000_000]],
                                           dtype=np.float32))
            h_parts.append(pd.util.hash_pandas_object(
                pd.DataFrame(blk), index=False).values)
        h = np.concatenate(h_parts)
        _, first = np.unique(h, return_index=True)
        kept = rows[np.sort(first)]
        keep_parts.append(kept)
        report.append({"class": cname, "before": len(rows), "after": len(kept),
                       "kept_frac": round(len(kept) / len(rows), 6)})
        print(f"  dedup[{cname}]: {len(rows):,} -> {len(kept):,} "
              f"({len(kept)/len(rows):.2%})", flush=True)
    return np.sort(np.concatenate(keep_parts)), pd.DataFrame(report)


def build_expert_data(args, experts, class_names, train_idx, y_train, benign_id):
    def draw_class(c, n, off=0, skip=0):
        perm = np.random.default_rng(args.seed + SEED_BAND_CONTEXT + c + off) \
            .permutation(train_idx[y_train == c])
        return perm[skip: skip + n]

    n_classes = len(class_names)
    contexts, holdouts = {}, {}
    for ename, owned in experts.items():
        counts = np.zeros(n_classes, dtype=np.int64)
        for c in owned:
            counts[c] = int((y_train == c).sum())
        tgt = natural_targets(counts, args.context_size)
        parts, hold = [], []
        for c in owned:
            if tgt[c] == 0:
                continue
            reserve = min(args.holdout_rows, max(1, counts[c] // 5))
            ctx_take = min(int(tgt[c]), int(counts[c]) - reserve)
            if ctx_take < 1:
                raise SystemExit(f"[{ename}] class {class_names[c]} too small.")
            parts.append(draw_class(c, ctx_take))
            n_hold = min(args.holdout_rows, int(counts[c]) - ctx_take)
            hold.append(draw_class(c, n_hold, skip=ctx_take))
        owned_idxs = np.sort(np.concatenate(parts))
        filler_idxs = np.array([], dtype=np.int64)
        # single-class filler (benign expert only in this grouping)
        if ename == "benign":
            fill = [c for c in range(n_classes)
                    if c != benign_id and (y_train == c).any()]
            per = max(1, args.filler_rows // len(fill))
            filler_idxs = np.sort(np.concatenate(
                [draw_class(c, min(per, int((y_train == c).sum())), off=25)
                 for c in fill]))
        idxs = np.concatenate([owned_idxs, filler_idxs])
        contexts[ename] = (idxs, len(owned_idxs))
        holdouts[ename] = np.sort(np.concatenate(hold))
        print(f"context[{ename}]: owned {len(owned_idxs):,} + filler "
              f"{len(filler_idxs):,}; holdout {len(holdouts[ename]):,}")
    return contexts, holdouts


def run_exp16(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp16_dedup_system"
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
    for ename, fams in EXP12_GROUPS.items():
        ids = [class_names.index(f) for f in fams if f in class_names]
        if ids:
            experts[ename] = ids
    print("experts:", {e: [class_names[c] for c in v] for e, v in experts.items()})

    # ---- THE KNOB: per-class exact-duplicate removal on the train pool ----
    print("dedup (per-class, keep chronologically first):", flush=True)
    train_idx, dedup_report = dedup_train_pool(X, train_idx,
                                               label_fn(train_idx), class_names)
    y_train = label_fn(train_idx)

    eval_idx = core.cap_per_class(test_idx, label_fn(test_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_eval = label_fn(eval_idx)
    X_eval = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))
    print(f"eval rows: {len(eval_idx):,}")

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    contexts, holdouts = build_expert_data(args, experts, class_names,
                                           train_idx, y_train, benign_id)
    ctx_data = {e: (feats_of(idxs), label_fn(idxs), n_owned)
                for e, (idxs, n_owned) in contexts.items()}
    hold_data = {e: feats_of(h) for e, h in holdouts.items()}
    hold_labels = {e: label_fn(h) for e, h in holdouts.items()}

    glob_ctx = None
    if args.fit_global:
        g_idx = core.stratified_subset(train_idx, y_train, n_classes,
                                       args.global_context_size,
                                       args.seed + SEED_BAND_GLOBAL)
        glob_ctx = (feats_of(g_idx), label_fn(g_idx))
        comp = {class_names[c]: int(n) for c, n in
                zip(*np.unique(glob_ctx[1], return_counts=True))}
        print(f"global context: {len(g_idx):,} rows {comp}")
    del X
    gc.collect()

    X_aux, aux_meta = load_aux_rows(args, cfg, set(class_names),
                                    args.aux_cap, args.seed)
    aux_train = X_aux[np.random.default_rng(args.seed + SEED_BAND_AUX)
                      .permutation(len(X_aux))][: args.aux_train_rows]
    core._PICKLE_CACHE.clear()
    gc.collect()

    import torch
    device = "cuda" if (args.device in ("auto", "cuda")
                        and torch.cuda.is_available()) else "cpu"

    def make_expert(ename):
        Xc, yc, n_owned = ctx_data[ename]
        clf = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        clf.fit(Xc, yc)
        dec = find_class_attention_module(clf)
        if dec is None:
            raise SystemExit(f"[{ename}] decoder not found")
        p0 = next(dec.parameters())
        emb_ref = np.asarray(clf.get_embeddings(
            Xc[: min(n_owned, args.emb_context_rows)], "test"))
        if emb_ref.ndim == 3:
            emb_ref = emb_ref[0]
        return clf, dec, p0, emb_ref

    def featurize(clf, dec, p0, emb_ref, Xr, chunk=250_000):
        outs = []
        for s0 in range(0, len(Xr), chunk):
            xb = Xr[s0:s0 + chunk]
            emb = np.asarray(clf.get_embeddings(xb, "test"))
            if emb.ndim == 3:
                emb = emb[0]
            with torch.inference_mode():
                q, kk = dec._project_qk(
                    torch.as_tensor(emb_ref).to(p0.device, p0.dtype).unsqueeze(0),
                    torch.as_tensor(emb).to(p0.device, p0.dtype).unsqueeze(0))
            lse_h, max_h = streaming_scores_perhead(
                q.squeeze(0).float().cpu(), kk.squeeze(0).float().cpu(),
                args.energy_T, device=device)
            pr = np.clip(clf.predict_proba(xb).astype(np.float64), 1e-12, 1)
            outs.append(np.concatenate(
                [lse_h, max_h, pr.max(1, keepdims=True),
                 (pr * np.log(pr)).sum(1, keepdims=True)],
                axis=1).astype(np.float32))
            if len(Xr) > chunk:
                print(f"    featurize chunk {s0//chunk+1}/"
                      f"{(len(Xr)+chunk-1)//chunk}", flush=True)
        return np.concatenate(outs)

    # ---- pass 1: gate scores on the full eval set ----
    enames = list(experts)
    gate_scores = np.zeros((len(enames), len(y_eval)), dtype=np.float32)
    for j, ename in enumerate(enames):
        t0 = time.time()
        clf, dec, p0, emb_ref = make_expert(ename)
        f_pos = featurize(clf, dec, p0, emb_ref, hold_data[ename])
        opp = (np.concatenate([hold_data[e] for e in enames if e != "benign"])
               if ename == "benign" else hold_data["benign"])
        rng_n = np.random.default_rng(args.seed + 4242)
        opp = opp[rng_n.permutation(len(opp))[: args.aux_train_rows]]
        f_neg = featurize(clf, dec, p0, emb_ref, np.concatenate([aux_train, opp]))
        head = train_score_head(f_pos, f_neg, args.seed,
                                pos_labels=hold_labels[ename])
        print(f"[{ename}] gate head ready ({time.time()-t0:.0f}s); scoring "
              f"{len(y_eval):,} rows", flush=True)
        gate_scores[j] = head(featurize(clf, dec, p0, emb_ref, X_eval))
        del clf, dec
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        print(f"[{ename}] pass-1 done in {time.time()-t0:.0f}s", flush=True)

    winner = gate_scores.argmax(axis=0)
    act = pd.DataFrame({
        "class": [class_names[c] for c in y_eval], "winner":
        [enames[w] for w in winner]}).value_counts().reset_index(name="rows")
    print("\n=== full-eval activation counts ===")
    print(act.sort_values(["class", "rows"], ascending=[True, False])
          .to_string(index=False))

    # ---- pass 2: conditionals on won rows ----
    final = np.full(len(y_eval), benign_id, dtype=np.int64)
    for j, ename in enumerate(enames):
        if ename == "benign":
            continue
        rows = np.flatnonzero(winner == j)
        if len(rows) == 0:
            continue
        clf, dec, p0, emb_ref = make_expert(ename)
        pr = []
        for s0 in range(0, len(rows), 500_000):
            pr.append(clf.predict_proba(X_eval[rows[s0:s0 + 500_000]]))
        pr = full_proba(np.concatenate(pr), clf.classes_, n_classes)
        owned = experts[ename]
        sub = pr[:, owned]
        final[rows] = np.asarray(owned)[sub.argmax(axis=1)]
        print(f"[{ename}] classified {len(rows):,} won rows", flush=True)
        del clf, dec
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    # ---- global reference fitted in the same run, same deduplicated pool ----
    p_glob, out_dir_probs = None, None
    if glob_ctx is not None:
        t0 = time.time()
        Xg, yg = glob_ctx
        clf = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        clf.fit(Xg, yg)
        bs = args.test_batch_size or len(X_eval)
        pr = []
        for s0 in range(0, len(X_eval), bs):
            pr.append(clf.predict_proba(X_eval[s0:s0 + bs]))
            print(f"[global] predict rows {s0:,}:{min(s0+bs, len(X_eval)):,}",
                  flush=True)
        p_glob = full_proba(np.concatenate(pr), clf.classes_,
                            n_classes).astype(np.float64)
        out_dir_probs = p_glob.astype(np.float32)
        print(f"[global] fitted+scored in {time.time()-t0:.0f}s", flush=True)
        del clf
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    all_rows = list(core.per_class_table("system_dedup", y_eval, final,
                                         class_names, tail_classes))
    if p_glob is not None:
        all_rows.extend(core.per_class_table("global_dedup_reference", y_eval,
                                             np.argmax(p_glob, axis=1),
                                             class_names, tail_classes))
    table = pd.DataFrame(all_rows)
    print("\n=== per-class F1 ===")
    print(table.pivot(index="class", columns="method", values="f1").round(4)
          .to_string())

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp16_dedup")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    act.to_csv(os.path.join(out_dir, "4a_full_activation_counts.csv"), index=False)
    dedup_report.to_csv(os.path.join(out_dir, "0a_dedup_report.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "system_dump.npz"),
                        gate_scores=gate_scores, winner=winner.astype(np.int8),
                        final=final, y_true=y_eval.astype(np.int64),
                        expert_names=np.asarray(enames),
                        class_names=np.asarray(class_names))
    if out_dir_probs is not None:
        np.savez_compressed(os.path.join(out_dir, "probs_tabpfn_global.npz"),
                            probs=out_dir_probs,
                            y_true=y_eval.astype(np.int64))
    aux_meta.to_csv(os.path.join(out_dir, "0b_aux_composition.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-size", type=int, default=200_000)
    p.add_argument("--holdout-rows", type=int, default=15_000)
    p.add_argument("--aux-cap", type=int, default=20_000)
    p.add_argument("--aux-train-rows", type=int, default=30_000)
    p.add_argument("--emb-context-rows", type=int, default=200_000)
    p.add_argument("--filler-rows", type=int, default=20_000)
    p.add_argument("--energy-T", type=float, default=1.0)
    p.add_argument("--fit-global", action="store_true",
                   help="fit the global reference on the same deduplicated "
                        "pool in this run")
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
    run_exp16(args)


if __name__ == "__main__":
    main()

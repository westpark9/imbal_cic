#!/usr/bin/env python3
"""EXP21 -- ENERGY-UNIFIED selection, no CE: global joins the gate race.

Pre-registration: 0821.md SS21.  One knob vs exp19: the BENIGN-EXPERT gate
candidate is replaced by a GLOBAL candidate --

  candidates = { global (natural 1M context), flood, bf_bot, tail }
  each candidate has an ENERGY-SCORE gate (same featurize + head recipe);
  the global's gate head: pos = benign holdout, neg = aux + attack holdouts
  (the exp12c benign-expert recipe, but measured on the 1M full-distribution
  context geometry).  Selection = plain 4-way argmax of gate scores -- the
  exp20 CE selector is GONE (nothing learned from predictions for routing).

  winner = global      -> label = global's 7-way argmax (it can also name
                          attacks -- unlike exp19's benign-fixed label)
  winner = attack k    -> exp19 conditional over owned(k) U {benign}
                          (context = owned + benign 200,000; return kept)

Judged against exp19 (benign expert candidate, 0.7331) and exp20 (CE prob
selector, 0.7235).

Engineering (results-identical): eval rows are deduplicated per distinct
46-feature vector before featurize/predict and predictions are broadcast
back (deterministic model => identical scores; measured 4.02M -> 2.10M
distinct, ~x1.9 compute saving; also removes SS12n batch-boundary jitter).

    python tabpfn/nfv3_v3_exp21_energy_global.py --target-dataset cic2018 \\
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
    "flood": ["ddos", "dos"],
    "bf_bot": ["brute_force", "bot"],
    "tail": ["infiltration", "web_attacks"],
}


def uniq_rows(Xr):
    """(X_unique, inverse) -- distinct-vector dedup for compute only.
    Deterministic model => predictions broadcast back are identical."""
    h = pd.util.hash_pandas_object(pd.DataFrame(Xr), index=False).values
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    return Xr[first], inv


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


# ---------------------------------------------------------------------------

def run_exp21(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp21_energy_global"
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

    eval_idx = core.cap_per_class(test_idx, label_fn(test_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_eval = label_fn(eval_idx)
    X_eval = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))
    Xu_eval, inv_eval = uniq_rows(X_eval)
    print(f"eval rows: {len(eval_idx):,} -> distinct {len(Xu_eval):,} "
          f"({len(Xu_eval)/len(eval_idx):.1%}) — broadcast 최적화", flush=True)
    y_train = label_fn(train_idx)

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    def draw_class(c, n, off=0, skip=0):
        perm = np.random.default_rng(args.seed + SEED_BAND_CONTEXT + c + off) \
            .permutation(train_idx[y_train == c])
        return perm[skip: skip + n]

    # gate contexts/holdouts: IDENTICAL to exp12c
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
        if ename == "benign":
            fill = [c for c in range(n_classes)
                    if c != benign_id and (y_train == c).any()]
            per = max(1, args.filler_rows // len(fill))
            filler_idxs = np.sort(np.concatenate(
                [draw_class(c, per, off=25) for c in fill]))
        idxs = np.concatenate([owned_idxs, filler_idxs])
        contexts[ename] = (idxs, len(owned_idxs))
        holdouts[ename] = np.sort(np.concatenate(hold))
        print(f"context[{ename}]: owned {len(owned_idxs):,} + filler "
              f"{len(filler_idxs):,}; holdout {len(holdouts[ename]):,}")

    benign_ret_idx = np.sort(draw_class(benign_id, args.return_benign_rows,
                                        off=50))
    benign_hold = np.sort(draw_class(benign_id, args.holdout_rows))
    print(f"conditional benign block: {len(benign_ret_idx):,} rows · "
          f"global-gate benign holdout: {len(benign_hold):,}")

    ctx_data = {e: (feats_of(idxs), label_fn(idxs), n_owned)
                for e, (idxs, n_owned) in contexts.items()}
    hold_data = {e: feats_of(h) for e, h in holdouts.items()}
    hold_labels = {e: label_fn(h) for e, h in holdouts.items()}
    benign_ret = (feats_of(benign_ret_idx), label_fn(benign_ret_idx))
    benign_hold_X = feats_of(benign_hold)

    g_idx = core.stratified_subset(train_idx, y_train, n_classes,
                                   args.global_context_size,
                                   args.seed + SEED_BAND_GLOBAL)
    glob_ctx = (feats_of(g_idx), label_fn(g_idx))
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

    def make_clf(Xc, yc):
        clf = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        clf.fit(Xc, yc)
        return clf

    def make_expert(ename):
        Xc, yc, n_owned = ctx_data[ename]
        clf = make_clf(Xc, yc)
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

    # ---- pass 1: energy gates for [global, flood, bf_bot, tail] ----
    enames = ["global"] + list(experts)
    gate_scores = np.zeros((len(enames), len(y_eval)), dtype=np.float32)

    # candidate 0: GLOBAL — 1M context, gate pos=benign holdout, + 7-way probs
    t0 = time.time()
    clf = make_clf(glob_ctx[0], glob_ctx[1])
    dec = find_class_attention_module(clf)
    if dec is None:
        raise SystemExit("[global] decoder not found")
    p0 = next(dec.parameters())
    emb_ref = np.asarray(clf.get_embeddings(
        glob_ctx[0][: args.emb_context_rows], "test"))
    if emb_ref.ndim == 3:
        emb_ref = emb_ref[0]
    f_pos = featurize(clf, dec, p0, emb_ref, benign_hold_X)
    opp = np.concatenate([hold_data[e] for e in experts])
    rng_n = np.random.default_rng(args.seed + 4242)
    opp = opp[rng_n.permutation(len(opp))[: args.aux_train_rows]]
    f_neg = featurize(clf, dec, p0, emb_ref, np.concatenate([aux_train, opp]))
    head = train_score_head(f_pos, f_neg, args.seed)
    print(f"[global] gate head ready ({time.time()-t0:.0f}s)", flush=True)
    gate_scores[0] = head(featurize(clf, dec, p0, emb_ref, Xu_eval))[inv_eval]
    pr = []
    for s0 in range(0, len(Xu_eval), args.test_batch_size or len(Xu_eval)):
        pr.append(clf.predict_proba(
            Xu_eval[s0:s0 + (args.test_batch_size or len(Xu_eval))]))
    p_glob = full_proba(np.concatenate(pr), clf.classes_,
                        n_classes)[inv_eval].astype(np.float64)
    del clf, dec
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    print(f"[global] pass-1 done in {time.time()-t0:.0f}s", flush=True)

    for j, ename in enumerate(experts, start=1):
        t0 = time.time()
        clf, dec, p0, emb_ref = make_expert(ename)
        f_pos = featurize(clf, dec, p0, emb_ref, hold_data[ename])
        opp = benign_hold_X
        f_neg = featurize(clf, dec, p0, emb_ref, np.concatenate([aux_train, opp]))
        head = train_score_head(f_pos, f_neg, args.seed,
                                pos_labels=hold_labels[ename])
        print(f"[{ename}] gate head ready ({time.time()-t0:.0f}s); scoring "
              f"{len(Xu_eval):,} distinct rows", flush=True)
        gate_scores[j] = head(featurize(clf, dec, p0, emb_ref,
                                        Xu_eval))[inv_eval]
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

    # ---- pass 2: global-won rows = global 7-way argmax; attack = return-cond ----
    final = np.empty(len(y_eval), dtype=np.int64)
    g_rows = np.flatnonzero(winner == 0)
    final[g_rows] = np.argmax(p_glob[g_rows], axis=1)
    print(f"[global] labeled {len(g_rows):,} won rows (7-way argmax)")
    ret_stats = []
    for j, ename in enumerate(enames):
        if ename == "global":
            continue
        rows = np.flatnonzero(winner == j)
        if len(rows) == 0:
            continue
        Xc, yc, n_owned = ctx_data[ename]
        Xc_owned = Xc[:n_owned]
        yc_owned = yc[:n_owned]
        Xc2 = np.concatenate([Xc_owned, benign_ret[0]])
        yc2 = np.concatenate([yc_owned, benign_ret[1]])
        clf = make_clf(Xc2, yc2)
        Xr = X_eval[rows]
        Xu, invu = uniq_rows(Xr)
        pr = []
        for s0 in range(0, len(Xu), 500_000):
            pr.append(clf.predict_proba(Xu[s0:s0 + 500_000]))
        pr = full_proba(np.concatenate(pr), clf.classes_, n_classes)[invu]
        allowed = np.asarray(sorted(experts[ename] + [benign_id]))
        sub = pr[:, allowed]
        final[rows] = allowed[sub.argmax(axis=1)]
        n_ret = int((final[rows] == benign_id).sum())
        ret_stats.append({"expert": ename, "won": len(rows),
                          "returned_benign": n_ret,
                          "ret_frac": round(n_ret / len(rows), 4)})
        print(f"[{ename}] classified {len(rows):,} won rows "
              f"(distinct {len(Xu):,}) — benign 반환 {n_ret:,}", flush=True)
        del clf
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    ret_df = pd.DataFrame(ret_stats)
    print("\n=== return stats ===")
    print(ret_df.to_string(index=False))

    out_dir_probs = p_glob.astype(np.float32)

    all_rows = list(core.per_class_table("system_energy_global", y_eval, final,
                                         class_names, tail_classes))
    if p_glob is not None:
        all_rows.extend(core.per_class_table("global_reference", y_eval,
                                             np.argmax(p_glob, axis=1),
                                             class_names, tail_classes))
    table = pd.DataFrame(all_rows)
    print("\n=== per-class F1 ===")
    print(table.pivot(index="class", columns="method", values="f1").round(4)
          .to_string())

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp21_eglobal")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    act.to_csv(os.path.join(out_dir, "4a_full_activation_counts.csv"), index=False)
    ret_df.to_csv(os.path.join(out_dir, "5a_return_stats.csv"), index=False)
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
    p.add_argument("--return-benign-rows", type=int, default=200_000,
                   help="benign block added to each attack conditional's "
                        "context (the knob)")
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
    run_exp21(args)


if __name__ == "__main__":
    main()

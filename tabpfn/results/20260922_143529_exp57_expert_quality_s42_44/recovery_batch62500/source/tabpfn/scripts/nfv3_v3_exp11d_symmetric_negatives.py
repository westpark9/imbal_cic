#!/usr/bin/env python3
"""EXP11d -- exp11c + SYMMETRIC negatives (user design).

One knob vs exp11c: every expert's negative pool = aux + benign + the OTHER
experts' attack holdouts (the benign expert already had this form).  To keep
the benign pressure that fixed the 45.9%->4.1% false-activation, the negative
side of each batch is sampled GROUP-UNIFORM (aux : benign : non-own attacks =
1:1:1), symmetric to the class-uniform positive side.  Target: the dos->bf_bot
34.7% leak (bf_bot's gate has never seen dos as a negative).

One knob vs exp11b: the score-head loss becomes per-portion means
(mean over positives + mean over negatives, positives additionally averaged
per class) -- the original energy_ood loss structure (0.1*(relu(..)^2 ID-mean
+ relu(..)^2 aux-mean)) is ratio-invariant the same way, while our previous
global-mean BCE baked each expert's pos:neg ratio (1:2..1:4) into its logit
offset and biased the cross-expert contest (~log-ratio, comparable to the
observed p10 margins 0.2-0.5).  Also balances web (304 holdout rows) against
inf inside tail's positives.  (user regrouping, closed-set)

Pre-registration: manuscript/report/0820.md SS12.  Forked from exp9 (frozen;
helpers copied).  Structure under test (user-specified):

  benign expert   context = benign (natural cap) + attack filler (real labels)
                  -- an explicit benign-vs-attack judge; retrieval reference =
                  benign rows only
  flood expert    ddos + dos
  bf_bot expert   brute_force + bot
  tail expert     infiltration + web_attacks

Every test row is scored by every expert's GATE (exp9's fine-tuned score head:
positives = owned TRAIN-holdout rows, negatives = a COMMON aux-OOD pool so the
four logits share the "log-odds vs aux" scale; the s01 cross-expert
comparability trap is the reason for the common negative set).  The expert with
the max gate score ACTIVATES.  This run measures activation only -- no final
classification, no unseen.

Artifacts:
  4a_activation_confusion.csv/png   true group x activated expert (row-normed)
  4b_per_class_activation.csv       per true class: activation share per expert
                                    (raw-logit argmax + percentile-calibrated)
  4c_margin_quantiles.csv           winner-vs-runnerup margins (delta material)
  scores_dump.npz                   per-class per-expert gate scores

    python tabpfn/nfv3_v3_exp11d_symmetric_negatives.py --target-dataset cic2018 \\
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
SEED_BAND_AUX = 970
AUX_DATASETS = ["bot_iot", "ton_iot"]
AUX_EXCLUDE = {("ton_iot", "scanning"), ("bot_iot", "theft")}

EXP11_GROUPS = {          # user regrouping (differs from nfv3_experts.json)
    "benign": ["benign"],
    "flood": ["ddos", "dos"],
    "bf_bot": ["brute_force", "bot"],
    "tail": ["infiltration", "web_attacks"],
}


# ---------------- helpers copied from exp9 (frozen) ----------------

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


def train_score_head(feat_pos, neg_groups, seed, pos_labels=None):
    """1:2 pos:neg batches; positives class-uniform; negatives GROUP-uniform
    over neg_groups (list of arrays: [aux, benign, non-own attacks...]);
    loss = per-portion means."""
    import torch
    torch.manual_seed(seed)
    neg_groups = [g for g in neg_groups if len(g)]
    Xall = np.concatenate([feat_pos] + neg_groups).astype(np.float32)
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd == 0] = 1.0
    Xp = torch.tensor((feat_pos - mu) / sd)
    Ng = [torch.tensor((g - mu) / sd) for g in neg_groups]
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
    n_total = len(Xp) + sum(len(x) for x in Ng)
    steps = 60 * max(1, n_total // 8192)
    b_pos, b_neg = 2731, 5461
    per_ng = max(1, b_neg // len(Ng))
    for _ in range(steps):
        cls_pick = torch.randint(len(pos_groups), (b_pos,), generator=g)
        pos_idx = torch.empty(b_pos, dtype=torch.long)
        for ci, grp in enumerate(pos_groups):
            m = cls_pick == ci
            k = int(m.sum())
            if k:
                pos_idx[m] = torch.as_tensor(
                    grp[torch.randint(len(grp), (k,), generator=g).numpy()])
        negs = [Xn[torch.randint(len(Xn), (per_ng,), generator=g)] for Xn in Ng]
        Xneg = torch.cat(negs)
        opt.zero_grad()
        lp = lossf(net(Xp[pos_idx]).squeeze(1), torch.ones(b_pos))
        ln = lossf(net(Xneg).squeeze(1), torch.zeros(len(Xneg)))
        (lp + ln).backward()
        opt.step()

    def predict(feats):
        with torch.no_grad():
            z = torch.tensor(((feats - mu) / sd).astype(np.float32))
            return net(z).squeeze(1).numpy()
    return predict


def run_exp11(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp11d_symmetric_negatives"
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    if args.subsample_samples:
        raise SystemExit("--subsample-samples must stay 0.")

    X, class_names, train_idx, val_idx, test_idx, _, _, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    experts = {}
    for ename, fams in EXP11_GROUPS.items():
        ids = [class_names.index(f) for f in fams if f in class_names]
        if not ids:
            raise SystemExit(f"group {ename}: no classes present")
        experts[ename] = ids
    group_of_class = {}
    for e, ids in experts.items():
        for c in ids:
            group_of_class[c] = e
    missing = [class_names[c] for c in range(n_classes) if c not in group_of_class]
    if missing:
        raise SystemExit(f"classes without an expert: {missing}")
    print("experts:", {e: [class_names[c] for c in v] for e, v in experts.items()})

    rng = np.random.default_rng(args.seed + 900)

    def sample(idx, n):
        return np.sort(rng.permutation(idx)[:n]) if len(idx) > n else np.sort(idx)

    y_test = label_fn(test_idx)
    y_train = label_fn(train_idx)

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    class_rows = {c: feats_of(sample(test_idx[y_test == c], args.id_sample))
                  for c in range(n_classes) if (y_test == c).any()}

    def draw_class(c, n, off=0, skip=0):
        perm = np.random.default_rng(args.seed + SEED_BAND_CONTEXT + c + off) \
            .permutation(train_idx[y_train == c])
        return perm[skip: skip + n]

    benign_id = class_names.index("benign")
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
            # reserve score-head positives FIRST when the pool is smaller than
            # the context would take (tail's pool < k): per class up to
            # --holdout-rows but never more than 20% of the class pool.
            reserve = min(args.holdout_rows, max(1, counts[c] // 5))
            ctx_take = min(int(tgt[c]), int(counts[c]) - reserve)
            if ctx_take < 1:
                raise SystemExit(f"[{ename}] class {class_names[c]} pool too "
                                 f"small ({counts[c]}) for context+holdout.")
            parts.append(draw_class(c, ctx_take))
            n_hold = min(args.holdout_rows, int(counts[c]) - ctx_take)
            hold.append(draw_class(c, n_hold, skip=ctx_take))
        owned_idxs = np.sort(np.concatenate(parts))
        filler_idxs = np.array([], dtype=np.int64)
        if len(np.unique(label_fn(owned_idxs))) < 2:
            fill = [c for c in range(n_classes)
                    if c != benign_id and (y_train == c).any()] \
                if ename == "benign" else [benign_id]
            per = max(1, args.filler_rows // len(fill))
            filler_idxs = np.sort(np.concatenate(
                [draw_class(c, per, off=25) for c in fill]))
        idxs = np.concatenate([owned_idxs, filler_idxs])
        contexts[ename] = (feats_of(idxs), label_fn(idxs), len(owned_idxs))
        if not hold:
            raise SystemExit(f"[{ename}] no train-holdout; lower --context-size.")
        hold_idx = np.sort(np.concatenate(hold))
        holdouts[ename] = (feats_of(hold_idx), label_fn(hold_idx))
        print(f"context[{ename}]: owned {len(owned_idxs):,} + filler "
              f"{len(filler_idxs):,}; holdout {len(holdouts[ename][0]):,}")
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

    # per expert: fit -> featurize class rows -> score head -> gate scores
    gate = {}          # gate[e][class_id] = scores (n,)
    id_holdout_scores = {}
    for ename, owned in experts.items():
        Xc, yc, n_owned = contexts[ename]
        clf = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        t0 = time.time()
        clf.fit(Xc, yc)
        dec = find_class_attention_module(clf)
        if dec is None:
            raise SystemExit(f"[{ename}] decoder q/k not found")
        p0 = next(dec.parameters())
        emb_ref = np.asarray(clf.get_embeddings(
            Xc[: min(n_owned, args.emb_context_rows)], "test"))
        if emb_ref.ndim == 3:
            emb_ref = emb_ref[0]
        print(f"[{ename}] fit+ref {time.time()-t0:.1f}s", flush=True)

        def featurize(Xr):
            emb = np.asarray(clf.get_embeddings(Xr, "test"))
            if emb.ndim == 3:
                emb = emb[0]
            with torch.inference_mode():
                q, kk = dec._project_qk(
                    torch.as_tensor(emb_ref).to(p0.device, p0.dtype).unsqueeze(0),
                    torch.as_tensor(emb).to(p0.device, p0.dtype).unsqueeze(0))
            lse_h, max_h = streaming_scores_perhead(
                q.squeeze(0).float().cpu(), kk.squeeze(0).float().cpu(),
                args.energy_T, device=device)
            pr = np.clip(clf.predict_proba(Xr).astype(np.float64), 1e-12, 1)
            return np.concatenate(
                [lse_h, max_h, pr.max(1, keepdims=True),
                 (pr * np.log(pr)).sum(1, keepdims=True)], axis=1).astype(np.float32)

        # exp11b one-knob change (0820 SS12b): negatives = aux + the
        # opposite side's TRAIN-holdout rows, so the gate learns the closed-set
        # contest boundary while keeping aux (the unseen-generalization
        # ingredient, exp9). All rows are train-slice holdouts -- F3-safe.
        # symmetric negatives (11d): aux / benign / non-own attacks as GROUPS
        groups_raw = [aux_train]
        if ename != "benign":
            groups_raw.append(holdouts["benign"][0])
        non_own = [holdouts[e][0] for e in experts
                   if e not in (ename, "benign")]
        if non_own:
            groups_raw.append(np.concatenate(non_own))
        rng_n = np.random.default_rng(args.seed + 4242)
        groups_raw = [gr[rng_n.permutation(len(gr))[: args.aux_train_rows]]
                      for gr in groups_raw]
        f_hold = featurize(holdouts[ename][0])
        neg_feats = [featurize(gr) for gr in groups_raw]
        predict_head = train_score_head(f_hold, neg_feats, args.seed,
                                        pos_labels=holdouts[ename][1])
        id_holdout_scores[ename] = predict_head(f_hold)
        gate[ename] = {}
        for c, Xr in class_rows.items():
            gate[ename][c] = predict_head(featurize(Xr))
            print(f"  gate[{ename}/{class_names[c]}] n={len(Xr):,}", flush=True)
        del clf
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    # ---------------- activation analysis (offline) ----------------
    enames = list(experts)
    rows4b, conf_raw, conf_cal = [], {}, {}
    margins = []
    for c, Xr in class_rows.items():
        S = np.stack([gate[e][c] for e in enames])            # (E, n)
        # percentile calibration: score -> quantile within own ID-holdout dist
        Scal = np.stack([
            np.searchsorted(np.sort(id_holdout_scores[e]), gate[e][c]) /
            max(len(id_holdout_scores[e]), 1) for e in enames])
        for tag, M, conf in (("raw", S, conf_raw), ("cal", Scal, conf_cal)):
            win = M.argmax(axis=0)
            shares = {enames[j]: float((win == j).mean()) for j in range(len(enames))}
            row = {"class": class_names[c], "calibration": tag,
                   "true_expert": group_of_class[c],
                   "correct_activation": shares.get(group_of_class[c], 0.0)}
            row.update({f"act_{e}": round(shares.get(e, 0.0), 4) for e in enames})
            rows4b.append(row)
            conf.setdefault(group_of_class[c], np.zeros(len(enames)))
            conf[group_of_class[c]] += np.bincount(win, minlength=len(enames)) \
                * (len(Xr) / len(Xr))
        top2 = np.sort(S, axis=0)[-2:]
        margins.append({"class": class_names[c],
                        "margin_p10": round(float(np.quantile(top2[1]-top2[0], .1)), 3),
                        "margin_p50": round(float(np.quantile(top2[1]-top2[0], .5)), 3),
                        "margin_p90": round(float(np.quantile(top2[1]-top2[0], .9)), 3)})

    df4b = pd.DataFrame(rows4b)
    print("\n=== 4b activation shares (raw argmax) ===")
    print(df4b[df4b.calibration == "raw"]
          .drop(columns=["calibration"]).to_string(index=False))
    print("\n=== 4b activation shares (percentile-calibrated) ===")
    print(df4b[df4b.calibration == "cal"]
          .drop(columns=["calibration"]).to_string(index=False))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp11_activation")
    os.makedirs(out_dir, exist_ok=True)
    df4b.to_csv(os.path.join(out_dir, "4b_per_class_activation.csv"), index=False)
    pd.DataFrame(margins).to_csv(
        os.path.join(out_dir, "4c_margin_quantiles.csv"), index=False)
    # group-level confusion (raw), row-normalized
    conf_rows = []
    for g in enames:
        classes_g = [c for c in class_rows if group_of_class[c] == g]
        if not classes_g:
            continue
        agg = np.zeros(len(enames))
        tot = 0
        for c in classes_g:
            S = np.stack([gate[e][c] for e in enames])
            agg += np.bincount(S.argmax(axis=0), minlength=len(enames))
            tot += S.shape[1]
        conf_rows.append({"true_group": g,
                          **{f"act_{e}": round(float(agg[j] / tot), 4)
                             for j, e in enumerate(enames)}})
    pd.DataFrame(conf_rows).to_csv(
        os.path.join(out_dir, "4a_activation_confusion.csv"), index=False)
    print("\n=== 4a group confusion (raw) ===")
    print(pd.DataFrame(conf_rows).to_string(index=False))

    np.savez_compressed(
        os.path.join(out_dir, "scores_dump.npz"),
        **{f"{e}__{class_names[c]}": gate[e][c].astype(np.float32)
           for e in gate for c in gate[e]},
        **{f"idhold__{e}": id_holdout_scores[e].astype(np.float32)
           for e in id_holdout_scores})
    aux_meta.to_csv(os.path.join(out_dir, "0b_aux_composition.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-size", type=int, default=200_000)
    p.add_argument("--id-sample", type=int, default=50_000)
    p.add_argument("--holdout-rows", type=int, default=15_000)
    p.add_argument("--aux-cap", type=int, default=20_000)
    p.add_argument("--aux-train-rows", type=int, default=30_000)
    p.add_argument("--emb-context-rows", type=int, default=200_000)
    p.add_argument("--filler-rows", type=int, default=20_000)
    p.add_argument("--energy-T", type=float, default=1.0)
    p.set_defaults(
        max_train_samples=-1,
        n_estimators=1,
        test_cap_per_class=0,
        fit_mode="fit_with_cache",
        test_batch_size=500_000,
        subsample_samples=0,
    )
    args = p.parse_args()
    run_exp11(args)


if __name__ == "__main__":
    main()

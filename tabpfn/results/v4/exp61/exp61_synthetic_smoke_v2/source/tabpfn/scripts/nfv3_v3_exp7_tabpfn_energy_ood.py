#!/usr/bin/env python3
"""EXP7 -- Can TabPFN do energy-score OOD?  (s43/s44 unseen protocol)

Pre-registration: manuscript/report/0820.md SS10.  Scope change recorded there:
the campaign's closed-set restriction is lifted; this script follows the
established unseen protocol and the 3a/3b/3c artifact format of the
resolved-expert track.

Protocol
--------
  ID        cic2018 WITHOUT the --unseen class.  Context = natural-proportional
            rows of the ID classes (shared permutation band, seed+950+cid).
  unseen    the held-out cic2018 class's TEST-window rows (the evaluation OOD).
  aux OOD   attack families from the OTHER suite datasets whose family name
            does not collide with any cic2018 family, minus the 0819 SS3
            polluted families (ton_iot scanning, bot_iot theft).  Role:
            CALIBRATION AID and reported side-role -- never the headline OOD.

Why TabPFN's *output* energy cannot work, and what can (measured + source)
--------------------------------------------------------------------------
  The v3 classification head is RETRIEVAL ATTENTION over the context rows
  (architectures/tabpfn_v3.py:547-586): test queries attend to train-row keys,
  values are one-hot train labels, probabilities are the attention-weighted
  label average, and the returned "logits" are log(probs) -- logsumexp == 0 by
  construction (measured +2.5e-4, 0820 SS8a).  The magnitude information lives
  one step earlier, in the pre-softmax similarity scores between the test row
  and the context rows.  This script recovers that quantity WITHOUT modifying
  the library:
    * trunk embeddings via the public clf.get_embeddings(X, "train"/"test");
    * the decoder's own q/k projections located by duck-typing (_project_qk)
      over model.named_modules() -- true head scores when found;
    * fallback (recorded in the log) = cosine-similarity scores in embedding
      space, the standard deep-kNN surrogate.
  Scores are reduced STREAMING over the context axis (never materializing
  M x N), producing per-test-row:
    energy_head   -T*logsumexp(scores/T)   (lower = more ID by convention
                                            "accept if score >= tau" after
                                            sign flip: we report -energy)
    maxsim        max similarity to any context row
    knn_dist      mean distance to the K nearest context embeddings (negated)
  plus probability-based baselines from predict_proba (nmaxp, entropy).

Variant B -- TabPFN-native outlier exposure ("other" class in-context)
----------------------------------------------------------------------
  A second fit whose context appends aux-OOD rows labeled as an extra "other"
  class.  P(other) is then a threshold-free unknown score: the attention mass
  retrieved from aux rows.  Aux rows are split disjointly: context-aux vs
  eval-aux (never scored on rows that sit in the context).

Judgment (pre-registered): headline = energy_head AUROC on the unseen role,
compared against the recorded global MLP energy AUROC 0.954 (s37, same suite).
>=0.90 -> "TabPFN energy OOD holds", extend to per-expert contexts.
<=0.70 -> head scores are not an OOD signal either; only embedding / Variant B
results carry forward.

    python tabpfn/nfv3_v3_exp7_tabpfn_energy_ood.py --target-dataset cic2018 \\
        --unseen bot --context-size 1000000 --fit-mode fit_with_cache \\
        --test-batch-size 500000

Artifacts (3a/3b/3c format of the resolved-expert track)
--------------------------------------------------------
  3a_tabpfn_ood_scores.csv        score x role: auroc, aupr_id, fpr95, supports
  3b_tabpfn_ood_thresholds.csv    tau at the ID-val quantile (accept if
                                  score >= tau), id_retain_at_tau,
                                  ood_detect_at_tau per score x role
  3c_energy_hist_2x2.png          per-score ID-val / ID-test / unseen / aux
                                  histograms with the tau line
  0a args.json / split audit / context composition, probs+score dumps (npz)
"""

import gc
import hashlib
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
SEED_BAND_AUX = 970          # aux draws; distinct from 950+cid (cic2018 has 7 classes)

AUX_DATASETS = ["bot_iot", "ton_iot"]      # unsw not wired in this loader config
AUX_EXCLUDE = {("ton_iot", "scanning"), ("bot_iot", "theft")}  # 0819 SS3
ROLES = ("id_val", "id_test", "unseen", "aux")
SCORES = ("energy_head", "maxsim", "knn_dist", "nmaxp", "entropy")


# ---------------------------------------------------------------------------
# Context (natural proportional over ID classes; copied conventions, not code)
# ---------------------------------------------------------------------------

def build_natural_context(y_pool, id_class_ids, k, seed, n_classes):
    counts = np.bincount(y_pool, minlength=n_classes)
    mask = np.zeros(n_classes, dtype=bool)
    mask[id_class_ids] = True
    counts = np.where(mask, counts, 0)
    total = int(counts.sum())
    k = min(k, total)
    raw = counts * (k / total)
    tgt = np.minimum(np.floor(raw).astype(np.int64), counts)
    present = counts > 0
    tgt[present & (tgt < 1)] = 1
    rem = k - int(tgt.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    for cid in order:
        if rem <= 0:
            break
        if tgt[cid] < counts[cid]:
            tgt[cid] += 1
            rem -= 1
    parts = []
    for c in np.flatnonzero(tgt > 0):
        perm = np.random.default_rng(seed + SEED_BAND_CONTEXT + int(c)) \
            .permutation(np.flatnonzero(y_pool == c))
        parts.append(perm[: int(tgt[c])])
    ctx = np.sort(np.concatenate(parts))
    assert len(ctx) == int(tgt.sum())
    return ctx, tgt


# ---------------------------------------------------------------------------
# Score machinery
# ---------------------------------------------------------------------------

def find_class_attention_module(clf):
    """Locate the retrieval-attention decoder by duck-typing (_project_qk +
    attention_weights).  Returns the module or None (fallback path)."""
    roots = []
    for attr in ("models_",):
        ms = getattr(clf, attr, None)
        if ms:
            roots.extend(ms if isinstance(ms, (list, tuple)) else [ms])
    if not roots:
        ex = getattr(clf, "executor_", None)
        ms = getattr(ex, "models", None) if ex is not None else None
        if ms:
            roots.extend(ms if isinstance(ms, (list, tuple)) else [ms])
    seen = []
    for root in roots:
        mod = getattr(root, "model", root)
        try:
            for _, m in mod.named_modules():
                if hasattr(m, "_project_qk") and hasattr(m, "attention_weights"):
                    seen.append(m)
        except AttributeError:
            continue
    return seen[0] if seen else None


def streaming_scores(q, k, T, chunk_m=2048, chunk_n=65536, device="cpu"):
    """q: (M,H,D) or (M,D); k: (N,H,D) or (N,D) torch tensors.
    Returns (logsumexp_over_N, max_over_N) as (M,) numpy arrays, computed in
    blocks so M x N is never materialized.  Head dim, if present, is averaged
    AFTER the per-head reduction (mirrors the decoder's head averaging)."""
    import torch
    if q.ndim == 2:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
    M, H, D = q.shape
    N = k.shape[0]
    scale = 1.0 / np.sqrt(D)
    lse = torch.full((M, H), -torch.inf)
    mx = torch.full((M, H), -torch.inf)
    q = q.to(device)
    for n0 in range(0, N, chunk_n):
        kb = k[n0:n0 + chunk_n].to(device)                    # (n,H,D)
        for m0 in range(0, M, chunk_m):
            qb = q[m0:m0 + chunk_m]                           # (m,H,D)
            s = torch.einsum("mhd,nhd->mhn", qb, kb).float() * scale
            s = s / T
            b_lse = torch.logsumexp(s, dim=-1).cpu()          # (m,H)
            b_max = s.max(dim=-1).values.cpu()
            cur = lse[m0:m0 + chunk_m]
            lse[m0:m0 + chunk_m] = torch.logaddexp(cur, b_lse)
            mx[m0:m0 + chunk_m] = torch.maximum(mx[m0:m0 + chunk_m], b_max)
            del s
    return (T * lse.mean(dim=1)).numpy(), mx.mean(dim=1).numpy()


def knn_mean_dist(test_emb, train_emb, kk=10, chunk_m=2048, chunk_n=65536,
                  device="cpu"):
    """Mean euclidean distance to the k nearest train embeddings, streaming."""
    import torch
    te = torch.as_tensor(test_emb)
    tr = torch.as_tensor(train_emb)
    M = te.shape[0]
    out = np.empty(M, dtype=np.float32)
    tr_dev_chunks = None
    for m0 in range(0, M, chunk_m):
        qb = te[m0:m0 + chunk_m].to(device)
        best = None
        for n0 in range(0, tr.shape[0], chunk_n):
            kb = tr[n0:n0 + chunk_n].to(device)
            d = torch.cdist(qb.float(), kb.float())           # (m,n)
            top = torch.topk(d, k=min(kk, d.shape[1]), dim=1, largest=False).values
            best = top if best is None else torch.topk(
                torch.cat([best, top], dim=1), k=kk, dim=1, largest=False).values
            del d, kb
        out[m0:m0 + chunk_m] = best.mean(dim=1).cpu().numpy()
        del qb, best
    return out


def prob_scores(probs):
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return {"nmaxp": p.max(axis=1),                      # higher = more ID
            "entropy": (p * np.log(p)).sum(axis=1)}      # = -H; higher = more ID


def ood_metrics(id_scores, ood_scores):
    """Convention: higher score = more ID.  AUROC of ID vs OOD, AUPR(ID
    positive), FPR at 95% ID retention (fraction of OOD ABOVE the 5%-ID
    threshold)."""
    s = np.concatenate([id_scores, ood_scores])
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # tie-average ranks
    sr = pd.Series(s); ranks = sr.rank(method="average").to_numpy()
    n1, n0 = len(id_scores), len(ood_scores)
    auroc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    # AUPR via precision at each recall step (ID positive, descending score)
    desc = np.argsort(-s, kind="stable")
    yl = y[desc]
    tp = np.cumsum(yl); fpc = np.cumsum(1 - yl)
    prec = tp / np.maximum(tp + fpc, 1)
    aupr = float((prec[yl == 1]).mean()) if n1 else float("nan")
    tau95 = np.quantile(id_scores, 0.05)
    fpr95 = float((ood_scores >= tau95).mean())
    return float(auroc), aupr, fpr95


# ---------------------------------------------------------------------------
# Aux assembly
# ---------------------------------------------------------------------------

def load_aux_rows(args, cfg, target_families, cap_per_family, seed):
    """Non-overlapping attack rows from the other suite datasets.  Uses each
    dataset's own loader (same 46-feature schema).  Returns (X_aux, labels_df)."""
    import copy
    rows, meta = [], []
    suffix = "_capped" if args.target_dataset.endswith("_capped") else ""
    for ds in [d + suffix for d in AUX_DATASETS]:
        a = copy.copy(args)
        a.target_dataset = ds
        a.data = cfg[ds]["default_data"]
        X, class_names, train_idx, val_idx, test_idx, _, _, _, label_fn = \
            cfg[ds]["loader"](a)
        for cid, cname in enumerate(class_names):
            if cname == "benign" or cname in target_families:
                continue
            if (ds.replace("_capped", ""), cname) in AUX_EXCLUDE:
                continue
            pool = np.concatenate([train_idx, val_idx])
            lab = label_fn(pool)
            pos = pool[lab == cid]
            if len(pos) == 0:
                continue
            take = np.random.default_rng(seed + SEED_BAND_AUX + cid) \
                .permutation(pos)[:cap_per_family]
            rows.append(np.nan_to_num(np.asarray(X[take], dtype=np.float32)))
            meta.append({"dataset": ds, "family": cname, "rows": len(take)})
        del X
        gc.collect()
    # clear the 15GB suite cache once, after all aux datasets are sliced
    core._PICKLE_CACHE.clear()
    gc.collect()
    if not rows:
        raise SystemExit("no aux families left after overlap/exclusion filters")
    return np.concatenate(rows), pd.DataFrame(meta)


# ---------------------------------------------------------------------------
# Run body
# ---------------------------------------------------------------------------

def run_exp7(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp7_tabpfn_energy_ood"
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)

    if args.subsample_samples:
        raise SystemExit("--subsample-samples must stay 0.")
    if args.context_size <= 0:
        raise SystemExit("--context-size must be > 0.")

    X, class_names, train_idx, val_idx, test_idx, _, _, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    if args.unseen not in class_names:
        raise SystemExit(f"--unseen must be one of {class_names}")
    if args.unseen == "benign":
        raise SystemExit("benign cannot be the unseen class.")
    unseen_id = class_names.index(args.unseen)
    id_class_ids = [c for c in range(n_classes) if c != unseen_id]
    print(f"ID classes: {[class_names[c] for c in id_class_ids]}  "
          f"unseen: {args.unseen}")

    rng = np.random.default_rng(args.seed + 900)

    def sample(idx, n):
        return np.sort(rng.permutation(idx)[:n]) if len(idx) > n else np.sort(idx)

    y_val_all = label_fn(val_idx)
    y_test_all = label_fn(test_idx)
    id_val_idx = sample(val_idx[np.isin(y_val_all, id_class_ids)], args.id_sample)
    id_test_idx = sample(test_idx[np.isin(y_test_all, id_class_ids)], args.id_sample)
    unseen_idx = sample(test_idx[y_test_all == unseen_id], args.id_sample)
    if len(unseen_idx) < 50:
        raise SystemExit(f"unseen class {args.unseen} has only {len(unseen_idx)} "
                         "test rows; pick another class.")
    print(f"role rows: id_val {len(id_val_idx):,} / id_test {len(id_test_idx):,} "
          f"/ unseen {len(unseen_idx):,}")

    # train pool = train slice, unseen class rows EXCLUDED
    y_train_all = label_fn(train_idx)
    pool_idx = train_idx[y_train_all != unseen_id]
    y_pool = label_fn(pool_idx)
    ctx, tgt = build_natural_context(y_pool, id_class_ids, args.context_size,
                                     args.seed, n_classes)
    print("context per class:",
          {class_names[c]: int(tgt[c]) for c in np.flatnonzero(tgt)})

    X_fit = np.nan_to_num(np.asarray(X[pool_idx[ctx]], dtype=np.float32))
    y_fit = y_pool[ctx]
    X_roles = {
        "id_val": np.nan_to_num(np.asarray(X[id_val_idx], dtype=np.float32)),
        "id_test": np.nan_to_num(np.asarray(X[id_test_idx], dtype=np.float32)),
        "unseen": np.nan_to_num(np.asarray(X[unseen_idx], dtype=np.float32)),
    }
    del X
    gc.collect()

    # aux while the suite pickle is still cached (one load total)
    target_families = set(class_names)
    X_aux_all, aux_meta = load_aux_rows(args, cfg, target_families,
                                        args.aux_cap, args.seed)
    print(f"aux OOD: {len(X_aux_all):,} rows from\n{aux_meta.to_string(index=False)}")
    # disjoint split: eval-aux vs context-aux (variant B)
    perm = np.random.default_rng(args.seed + SEED_BAND_AUX).permutation(len(X_aux_all))
    n_eval_aux = len(perm) // 2
    X_roles["aux"] = X_aux_all[perm[:n_eval_aux]]
    X_aux_ctx = X_aux_all[perm[n_eval_aux:]][: args.aux_context_rows]
    del X_aux_all

    import torch
    device = "cuda" if (args.device in ("auto", "cuda")
                        and torch.cuda.is_available()) else "cpu"

    # ---------------- Variant A: ID-only context ----------------
    clf = TabPFNClassifier(
        device=args.device, model_path=args.model_path,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        random_state=args.seed, n_estimators=args.n_estimators,
        auto_scale_n_estimators=False, fit_mode=args.fit_mode,
        keep_cache_on_device=args.keep_cache_on_device)
    t0 = time.time()
    clf.fit(X_fit, y_fit)
    print(f"fit A done in {time.time()-t0:.1f}s", flush=True)

    scores = {r: {} for r in ROLES}
    probs_dump = {}
    for role in ROLES:
        Xr = X_roles[role]
        probs = clf.predict_proba(Xr)
        probs_dump[role] = probs.astype(np.float32)
        for k2, v in prob_scores(probs).items():
            scores[role][k2] = v
        print(f"probs[{role}] done ({len(Xr):,})", flush=True)

    # trunk embeddings (public API) -- train once, then each role
    t0 = time.time()
    emb_train = clf.get_embeddings(X_fit[: args.emb_context_rows], "test")
    emb_train = np.asarray(emb_train)
    if emb_train.ndim == 3:
        emb_train = emb_train[0]
    dec = find_class_attention_module(clf)
    print(f"decoder module: {'FOUND' if dec is not None else 'fallback (cosine on trunk embeddings)'}")
    for role in ROLES:
        emb = np.asarray(clf.get_embeddings(X_roles[role], "test"))
        if emb.ndim == 3:
            emb = emb[0]
        if dec is not None:
            p0 = next(dec.parameters())
            with torch.inference_mode():
                q, kk = dec._project_qk(
                    torch.as_tensor(emb_train).to(p0.device, p0.dtype).unsqueeze(0),
                    torch.as_tensor(emb).to(p0.device, p0.dtype).unsqueeze(0))
            q = q.squeeze(0).float().cpu()    # (M,H,D) test queries
            kk = kk.squeeze(0).float().cpu()  # (N,H,D) context keys
            lse, mx = streaming_scores(q, kk, args.energy_T, device=device)
        else:
            q = torch.as_tensor(emb, dtype=torch.float32)
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
            kt = torch.as_tensor(emb_train, dtype=torch.float32)
            kt = kt / kt.norm(dim=1, keepdim=True).clamp_min(1e-8)
            lse, mx = streaming_scores(q, kt, args.energy_T, device=device)
        scores[role]["energy_head"] = lse       # higher = more ID (-energy)
        scores[role]["maxsim"] = mx
        scores[role]["knn_dist"] = -knn_mean_dist(emb, emb_train, device=device)
        print(f"scores[{role}] done in {time.time()-t0:.1f}s cum", flush=True)
    del clf
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()

    # ---------------- Variant B: aux as in-context "other" ----------------
    p_other = {}
    if not args.skip_variant_b:
        other_id = n_classes  # new label id
        Xb = np.concatenate([X_fit, X_aux_ctx])
        yb = np.concatenate([y_fit, np.full(len(X_aux_ctx), other_id)])
        clf_b = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=True,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        t0 = time.time()
        clf_b.fit(Xb, yb)
        print(f"fit B done in {time.time()-t0:.1f}s (context +{len(X_aux_ctx):,} "
              "aux rows as 'other')", flush=True)
        oc = list(clf_b.classes_).index(other_id)
        for role in ROLES:
            pb = clf_b.predict_proba(X_roles[role])
            p_other[role] = pb[:, oc].astype(np.float64)
        del clf_b
        gc.collect()

    # ---------------- 3a / 3b / 3c ----------------
    score_names = list(SCORES) + (["p_other_neg"] if p_other else [])
    if p_other:
        for role in ROLES:
            scores[role]["p_other_neg"] = -p_other[role]  # higher = more ID
    rows3a, rows3b = [], []
    for sn in score_names:
        idv = scores["id_val"][sn]
        tau = float(np.quantile(idv, args.threshold_quantile))
        for role in ("unseen", "aux", "id_test"):
            au, ap, f95 = ood_metrics(idv, scores[role][sn]) if role != "id_test" \
                else ood_metrics(idv, scores[role][sn])
            rows3a.append({"score": sn, "ood_role": role,
                           "id_support": len(idv),
                           "ood_support": len(scores[role][sn]),
                           "auroc": round(au, 4), "aupr_id": round(ap, 4),
                           "fpr95": round(f95, 4)})
            rows3b.append({"score": sn, "ood_role": role,
                           "threshold": tau,
                           "id_retain_at_tau": round(float((idv >= tau).mean()), 4),
                           "ood_detect_at_tau": round(
                               float((scores[role][sn] < tau).mean()), 4)})
    df3a, df3b = pd.DataFrame(rows3a), pd.DataFrame(rows3b)
    print("\n=== 3a (score x role) ===")
    print(df3a.to_string(index=False))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp7_ood_unseen_{args.unseen}")
    os.makedirs(out_dir, exist_ok=True)
    df3a.to_csv(os.path.join(out_dir, "3a_tabpfn_ood_scores.csv"), index=False)
    df3b.to_csv(os.path.join(out_dir, "3b_tabpfn_ood_thresholds.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panel = [s for s in ("energy_head", "maxsim", "knn_dist",
                         "p_other_neg" if p_other else "nmaxp")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, sn in zip(axes.ravel(), panel):
        for role, color in (("id_val", "#2a78d6"), ("id_test", "#1baf7a"),
                            ("unseen", "#e34948"), ("aux", "#eda100")):
            ax.hist(scores[role][sn], bins=60, density=True, alpha=0.45,
                    label=role, color=color)
        tau = float(np.quantile(scores["id_val"][sn], args.threshold_quantile))
        ax.axvline(tau, color="k", ls="--", lw=1, label=f"tau(q{args.threshold_quantile})")
        ax.set_title(sn)
        ax.legend(fontsize=8)
    fig.suptitle(f"exp7 TabPFN OOD scores -- unseen={args.unseen} "
                 f"(higher = more ID; accept if >= tau)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3c_energy_hist_2x2.png"), dpi=130)

    np.savez_compressed(
        os.path.join(out_dir, "scores_dump.npz"),
        **{f"{r}__{sn}": np.asarray(scores[r][sn], dtype=np.float32)
           for r in ROLES for sn in score_names},
        **{f"probs__{r}": probs_dump[r] for r in probs_dump})
    aux_meta.to_csv(os.path.join(out_dir, "0b_aux_composition.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--unseen", required=True,
                   help="cic2018 class held out of training and evaluated as OOD.")
    p.add_argument("--context-size", type=int, default=1_000_000)
    p.add_argument("--id-sample", type=int, default=100_000,
                   help="Rows per role (id_val / id_test / unseen cap) for the "
                        "score panels.")
    p.add_argument("--aux-cap", type=int, default=20_000,
                   help="Rows per aux family before the eval/context split.")
    p.add_argument("--aux-context-rows", type=int, default=50_000,
                   help="Aux rows labeled 'other' in Variant B's context.")
    p.add_argument("--emb-context-rows", type=int, default=200_000,
                   help="Context rows used as the reference set for head-score "
                        "and kNN reductions (memory bound).")
    p.add_argument("--energy-T", type=float, default=1.0)
    p.add_argument("--threshold-quantile", type=float, default=0.05,
                   help="tau = this quantile of the ID-val score (s24 convention).")
    p.add_argument("--skip-variant-b", action="store_true")
    p.set_defaults(
        max_train_samples=-1,
        n_estimators=1,
        test_cap_per_class=0,
        fit_mode="fit_with_cache",
        test_batch_size=500_000,
        subsample_samples=0,
    )
    args = p.parse_args()
    run_exp7(args)


if __name__ == "__main__":
    main()

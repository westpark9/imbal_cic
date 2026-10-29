#!/usr/bin/env python3
"""EXP9 -- Energy-score fine-tuning on frozen TabPFN experts (old pre/post format).

Pre-registration: manuscript/report/0820.md SS10d.  Forked from exp8 (frozen).
User-fixed terms: ENERGY for TabPFN = -T*logsumexp of the retrieval head's
pre-softmax similarities (the only place the classic formula is non-degenerate,
0820 SS8a/SS10a).  Fine-tuning does NOT touch the backbone: it trains a small
score head ON TOP of the frozen model's similarity statistics -- the old
pipeline's structure (energy from logits -> score tuned with aux OOD).

Per expert (taxonomy families, configs/nfv3_experts.json; --unseen excluded
from every context):
  pre  score  = energy = T*logsumexp(sim/T) (head-mean; higher = more owned)
  post score  = MLP( [per-head logsumexp, per-head max, max output prob,
                      output entropy] ) trained with BCE:
      positives = owned families' TRAIN-slice held-out rows (drawn AFTER the
                  context prefix from the same permutation -- disjoint from
                  the context, val never touched: F3-safe)
      negatives = the aux-OOD TRAIN half (disjoint from the aux EVAL half)
Roles and artifacts follow the old 3a/3b/3c vocabulary; both stages reported:
  3a  expert x role with pre_/post_ auroc, aupr_id, fpr95, id_retain,
      ood_detect and deltas
  3c  3c_energy_hist_2x2_pretrained.png / _finetuned.png

Pre-registered judgment: does post improve only aux (OE-overfit; Variant B's
failure mode, aux 0.993 / unseen 0.390) or also unseen?  post_unseen >=
pre_unseen -> OE generalizes; the reverse repeats the s31-37 record
("energy fine-tuning worsens/inverts OOD").

    python tabpfn/nfv3_v3_exp9_energy_score_finetune.py --target-dataset cic2018 \\
        --unseen bot --fit-mode fit_with_cache --test-batch-size 500000
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
CONFIG_KEY = {"cic2018": "cse_cic_ids2018", "cic2018_capped": "cse_cic_ids2018"}


# ---------------- helpers copied from exp8 (frozen; copy-not-import) --------

def find_class_attention_module(clf):
    roots = []
    ms = getattr(clf, "models_", None)
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


def streaming_scores_perhead(q, k, T, chunk_m=2048, chunk_n=65536, device="cpu"):
    """Returns per-head (lse, mx) as (M,H) numpy arrays (no head averaging)."""
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


def ood_metrics(id_scores, ood_scores):
    s = np.concatenate([id_scores, ood_scores])
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    ranks = pd.Series(s).rank(method="average").to_numpy()
    n1, n0 = len(id_scores), len(ood_scores)
    auroc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    desc = np.argsort(-s, kind="stable")
    yl = y[desc]
    tp = np.cumsum(yl); fpc = np.cumsum(1 - yl)
    prec = tp / np.maximum(tp + fpc, 1)
    aupr = float(prec[yl == 1].mean()) if n1 else float("nan")
    tau95 = np.quantile(id_scores, 0.05)
    return float(auroc), aupr, float((ood_scores >= tau95).mean())


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
        raise SystemExit("no aux families left after filters")
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


def train_score_head(feat_pos, feat_neg, seed):
    """Small MLP on standardized features, BCE, deterministic CPU training.
    Returns (predict_fn, norm_stats)."""
    import torch
    torch.manual_seed(seed)
    Xp = np.concatenate([feat_pos, feat_neg]).astype(np.float32)
    yp = np.concatenate([np.ones(len(feat_pos)), np.zeros(len(feat_neg))]) \
        .astype(np.float32)
    mu, sd = Xp.mean(0), Xp.std(0)
    sd[sd == 0] = 1.0
    Xn = torch.tensor((Xp - mu) / sd)
    yt = torch.tensor(yp)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xn.shape[1], 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(Xn)
    g = torch.Generator().manual_seed(seed)
    for epoch in range(60):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 8192):
            b = perm[i:i + 8192]
            opt.zero_grad()
            loss = lossf(net(Xn[b]).squeeze(1), yt[b])
            loss.backward()
            opt.step()

    def predict(feats):
        with torch.no_grad():
            z = torch.tensor(((feats - mu) / sd).astype(np.float32))
            return net(z).squeeze(1).numpy()
    return predict, (mu, sd)


# ---------------------------------------------------------------------------

def run_exp9(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp9_energy_score_finetune"
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    if args.subsample_samples:
        raise SystemExit("--subsample-samples must stay 0.")
    groups = json.load(open(args.expert_config))[CONFIG_KEY[args.target_dataset]]

    X, class_names, train_idx, val_idx, test_idx, _, _, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)
    if args.unseen not in class_names or args.unseen == "benign":
        raise SystemExit(f"--unseen must be an attack class of {class_names}")
    unseen_id = class_names.index(args.unseen)
    benign_id = class_names.index("benign")

    experts = {}
    for ename, fams in groups.items():
        owned = [class_names.index(f) for f in fams
                 if f in class_names and f != args.unseen]
        if owned:
            experts[ename] = owned
    experts["benign"] = [benign_id]
    print("experts:", {e: [class_names[c] for c in v] for e, v in experts.items()})

    rng = np.random.default_rng(args.seed + 900)

    def sample(idx, n):
        return np.sort(rng.permutation(idx)[:n]) if len(idx) > n else np.sort(idx)

    y_val = label_fn(val_idx)
    y_test = label_fn(test_idx)
    y_train = label_fn(train_idx)

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    unseen_rows = feats_of(sample(test_idx[y_test == unseen_id], args.id_sample))
    benign_test_rows = feats_of(sample(test_idx[y_test == benign_id], args.id_sample))
    attack_test_rows = {c: feats_of(sample(test_idx[y_test == c], args.id_sample))
                        for c in range(n_classes)
                        if c not in (benign_id, unseen_id) and (y_test == c).any()}
    val_rows = {c: feats_of(sample(val_idx[y_val == c], args.id_sample))
                for c in range(n_classes) if (y_val == c).any()}

    # contexts + TRAIN-holdout positives (next slice of the same permutation)
    def draw_class(c, n, off=0, skip=0):
        pos = train_idx[y_train == c]
        perm = np.random.default_rng(args.seed + SEED_BAND_CONTEXT + c + off) \
            .permutation(pos)
        return perm[skip: skip + n]

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
            parts.append(draw_class(c, int(tgt[c])))
            n_hold = min(args.holdout_rows,
                         max(0, counts[c] - int(tgt[c])))
            if n_hold > 0:
                hold.append(draw_class(c, n_hold, skip=int(tgt[c])))
        owned_idxs = np.sort(np.concatenate(parts))
        filler_idxs = np.array([], dtype=np.int64)
        if len(np.unique(label_fn(owned_idxs))) < 2:
            fill = ([c for c in range(n_classes)
                     if c not in (benign_id, unseen_id) and (y_train == c).any()]
                    if ename == "benign" else [benign_id])
            per = max(1, args.filler_rows // len(fill))
            filler_idxs = np.sort(np.concatenate(
                [draw_class(c, per, off=25) for c in fill]))
        idxs = np.concatenate([owned_idxs, filler_idxs])
        contexts[ename] = (feats_of(idxs), label_fn(idxs), len(owned_idxs))
        if not hold:
            raise SystemExit(f"[{ename}] no train-holdout rows (context ate the "
                             "pool); lower --context-size.")
        holdouts[ename] = feats_of(np.sort(np.concatenate(hold)))
        print(f"context[{ename}]: owned {len(owned_idxs):,} + filler "
              f"{len(filler_idxs):,}; holdout {len(holdouts[ename]):,}")
    del X
    gc.collect()

    X_aux, aux_meta = load_aux_rows(args, cfg, set(class_names),
                                    args.aux_cap, args.seed)
    perm = np.random.default_rng(args.seed + SEED_BAND_AUX).permutation(len(X_aux))
    half = len(perm) // 2
    aux_eval = X_aux[perm[:half]][: args.id_sample]
    aux_train = X_aux[perm[half:]][: args.aux_train_rows]
    core._PICKLE_CACHE.clear()
    gc.collect()
    print(f"aux: eval {len(aux_eval):,} / score-train {len(aux_train):,}")

    import torch
    device = "cuda" if (args.device in ("auto", "cuda")
                        and torch.cuda.is_available()) else "cpu"

    rows3a, rows3b, hists = [], [], {}
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
        print(f"[{ename}] fit+ref in {time.time()-t0:.1f}s", flush=True)

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
            feats = np.concatenate(
                [lse_h, max_h, pr.max(1, keepdims=True),
                 (pr * np.log(pr)).sum(1, keepdims=True)], axis=1)
            energy_pre = lse_h.mean(1)          # head-mean logsumexp = ENERGY
            return feats.astype(np.float32), energy_pre.astype(np.float32)

        own_val = np.concatenate([val_rows[c] for c in owned if c in val_rows])
        roles = {"id_val": own_val, "unseen": unseen_rows, "aux": aux_eval}
        if ename == "benign":
            roles["known_attack"] = np.concatenate(list(attack_test_rows.values()))
        else:
            non_owned = [v for c, v in attack_test_rows.items() if c not in owned]
            if non_owned:
                roles["non_owned_known_attack"] = np.concatenate(non_owned)
            roles["benign"] = benign_test_rows

        f_pos, _ = featurize(holdouts[ename])
        f_neg, _ = featurize(aux_train)
        predict_head, _ = train_score_head(f_pos, f_neg, args.seed)
        print(f"[{ename}] score head trained (pos {len(f_pos):,} / neg {len(f_neg):,})",
              flush=True)

        sc = {}
        for role, Xr in roles.items():
            feats, pre = featurize(Xr)
            sc[role] = {"pre": pre, "post": predict_head(feats)}
            print(f"  [{ename}/{role}] n={len(Xr):,}", flush=True)

        owned_names = ",".join(class_names[c] for c in owned)
        idv = sc["id_val"]
        tau_pre = float(np.quantile(idv["pre"], args.threshold_quantile))
        tau_post = float(np.quantile(idv["post"], args.threshold_quantile))
        for role in sc:
            if role == "id_val":
                continue
            pa, pap, pf = ood_metrics(idv["pre"], sc[role]["pre"])
            qa, qap, qf = ood_metrics(idv["post"], sc[role]["post"])
            rows3a.append({
                "expert": ename, "active_owned_classes": owned_names,
                "ood_role": role,
                "pre_auroc": round(pa, 4), "pre_aupr_id": round(pap, 4),
                "pre_fpr95": round(pf, 4),
                "pre_id_retain": round(float((idv["pre"] >= tau_pre).mean()), 4),
                "pre_ood_detect": round(float((sc[role]["pre"] < tau_pre).mean()), 4),
                "post_auroc": round(qa, 4), "post_aupr_id": round(qap, 4),
                "post_fpr95": round(qf, 4),
                "post_id_retain": round(float((idv["post"] >= tau_post).mean()), 4),
                "post_ood_detect": round(float((sc[role]["post"] < tau_post).mean()), 4),
                "auroc_delta": round(qa - pa, 4),
                "fpr95_delta": round(qf - pf, 4),
                "ood_detect_delta": round(
                    float((sc[role]["post"] < tau_post).mean())
                    - float((sc[role]["pre"] < tau_pre).mean()), 4)})
            rows3b.append({"expert": ename, "ood_role": role,
                           "threshold_pre": tau_pre, "threshold_post": tau_post})
        hists[ename] = sc
        del clf
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    df3a = pd.DataFrame(rows3a)
    print("\n=== 3a (pre/post) ===")
    print(df3a[["expert", "ood_role", "pre_auroc", "post_auroc",
                "auroc_delta", "pre_ood_detect", "post_ood_detect"]]
          .to_string(index=False))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp9_eft_unseen_{args.unseen}")
    os.makedirs(out_dir, exist_ok=True)
    df3a.to_csv(os.path.join(out_dir, "3a_expert_ood_before_after.csv"), index=False)
    pd.DataFrame(rows3b).to_csv(
        os.path.join(out_dir, "3b_expert_ood_thresholds.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"id_val": "#2a78d6", "unseen": "#e34948", "aux": "#eda100",
              "non_owned_known_attack": "#4a3aa7", "known_attack": "#4a3aa7",
              "benign": "#1baf7a"}
    for stage, fname in (("pre", "3c_energy_hist_2x2_pretrained.png"),
                         ("post", "3c_energy_hist_2x2_finetuned.png")):
        enames = list(hists)
        n = len(enames)
        fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, 3.4 * ((n + 1) // 2)),
                                 squeeze=False)
        for i, ename in enumerate(enames):
            ax = axes[i // 2][i % 2]
            for role, s in hists[ename].items():
                ax.hist(s[stage], bins=60, density=True, alpha=0.45,
                        label=role, color=colors.get(role, "#888"))
            tau = float(np.quantile(hists[ename]["id_val"][stage],
                                    args.threshold_quantile))
            ax.axvline(tau, color="k", ls="--", lw=1)
            ax.set_title(f"{ename} ({'energy' if stage == 'pre' else 'finetuned score'})")
            ax.legend(fontsize=7)
        fig.suptitle(f"exp9 {stage} -- unseen={args.unseen} "
                     "(higher = owned; accept if >= tau)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)

    np.savez_compressed(
        os.path.join(out_dir, "scores_dump.npz"),
        **{f"{e}__{r}__{st}": np.asarray(hists[e][r][st], dtype=np.float32)
           for e in hists for r in hists[e] for st in ("pre", "post")})
    aux_meta.to_csv(os.path.join(out_dir, "0b_aux_composition.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--unseen", required=True)
    p.add_argument("--expert-config",
                   default=os.path.join(core.REPO_ROOT, "configs", "nfv3_experts.json"))
    p.add_argument("--context-size", type=int, default=200_000)
    p.add_argument("--id-sample", type=int, default=50_000)
    p.add_argument("--holdout-rows", type=int, default=15_000,
                   help="Per owned class: TRAIN rows after the context prefix, "
                        "used as score-head positives.")
    p.add_argument("--aux-cap", type=int, default=20_000)
    p.add_argument("--aux-train-rows", type=int, default=30_000,
                   help="Aux rows (train half) used as score-head negatives.")
    p.add_argument("--emb-context-rows", type=int, default=200_000)
    p.add_argument("--filler-rows", type=int, default=20_000)
    p.add_argument("--energy-T", type=float, default=1.0)
    p.add_argument("--threshold-quantile", type=float, default=0.05)
    p.set_defaults(
        max_train_samples=-1,
        n_estimators=1,
        test_cap_per_class=0,
        fit_mode="fit_with_cache",
        test_batch_size=500_000,
        subsample_samples=0,
    )
    args = p.parse_args()
    run_exp9(args)


if __name__ == "__main__":
    main()

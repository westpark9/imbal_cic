#!/usr/bin/env python3
"""EXP8 -- Per-expert TabPFN OOD gates over the taxonomy family partition.

The s43/s44 structure reproduced with TabPFN experts: each expert OWNS a
disjoint set of taxonomy families (configs/nfv3_experts.json), its context
holds ONLY its owned families' rows (the --unseen family excluded everywhere),
plus a benign expert (context = benign rows; ID = benign, OOD = every attack).
No fine-tuning exists for TabPFN, so all artifacts are single-stage (the
"pretrained" column of the old 3a/3b/3c format).

Scores per expert (from the retrieval head, exp7's finding):
  maxsim       similarity of the test row to its closest context row
               (the model's own q/k metric; softmax-precursor)
  energy_head  T*logsumexp of similarities over the context (total support
               mass -- the classic energy analog)
Both: higher = more "this expert's row".  Convention: accept if score >= tau,
tau = the --threshold-quantile of the expert's OWN ID-val scores.

Roles per attack expert (old 3a vocabulary):
  id_val                   owner families' val rows (defines tau)
  non_owned_known_attack   other attack families' test rows (known, not owned)
  unseen                   the held-out family's test rows
  benign                   benign test rows
  aux                      non-overlapping other-dataset attacks (side role)
Benign expert: id_val = benign val rows; OOD roles = known attacks / unseen / aux.

    python tabpfn/nfv3_v3_exp8_expert_family_ood.py --target-dataset cic2018 \\
        --unseen bot --fit-mode fit_with_cache --test-batch-size 500000

Artifacts: 3a_expert_ood_scores.csv (expert x score x role), 3b thresholds
with id_retain/ood_detect at tau, 3c hist grid PNG, 0a expert map, dumps.
Pre-registration: 0820.md SS10 (per-expert extension branch, taken because
exp7's global gate passed: unseen-bot maxsim AUROC 0.9839 / energy 0.9443).
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
CONFIG_KEY = {"cic2018": "cse_cic_ids2018", "cic2018_capped": "cse_cic_ids2018",
              "bot_iot": "bot_iot", "bot_iot_capped": "bot_iot",
              "ton_iot": "ton_iot", "ton_iot_capped": "ton_iot"}
SCORES = ("maxsim", "energy_head")


# --------------- helpers copied from exp7 (frozen; copy-not-import) ---------

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


def streaming_scores(q, k, T, chunk_m=2048, chunk_n=65536, device="cpu"):
    import torch
    if q.ndim == 2:
        q = q.unsqueeze(1); k = k.unsqueeze(1)
    M, H, D = q.shape
    N = k.shape[0]
    scale = 1.0 / np.sqrt(D)
    lse = torch.full((M, H), -torch.inf)
    mx = torch.full((M, H), -torch.inf)
    q = q.to(device)
    for n0 in range(0, N, chunk_n):
        kb = k[n0:n0 + chunk_n].to(device)
        for m0 in range(0, M, chunk_m):
            s = torch.einsum("mhd,nhd->mhn", q[m0:m0 + chunk_m], kb).float() * scale
            s = s / T
            lse[m0:m0 + chunk_m] = torch.logaddexp(
                lse[m0:m0 + chunk_m], torch.logsumexp(s, dim=-1).cpu())
            mx[m0:m0 + chunk_m] = torch.maximum(
                mx[m0:m0 + chunk_m], s.max(dim=-1).values.cpu())
            del s
    return (T * lse.mean(dim=1)).numpy(), mx.mean(dim=1).numpy()


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


# ---------------------------------------------------------------------------

def run_exp8(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp8_expert_family_ood"
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

    # expert -> owned class ids (unseen removed everywhere)
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

    # shared role pools (feature slices)
    def feats(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    unseen_rows = feats(sample(test_idx[y_test == unseen_id], args.id_sample))
    benign_test_rows = feats(sample(test_idx[y_test == benign_id], args.id_sample))
    attack_test = {c: sample(test_idx[y_test == c], args.id_sample)
                   for c in range(n_classes)
                   if c not in (benign_id, unseen_id) and (y_test == c).any()}
    attack_test_rows = {c: feats(i) for c, i in attack_test.items()}
    val_rows = {c: feats(sample(val_idx[y_val == c], args.id_sample))
                for c in range(n_classes) if (y_val == c).any()}

    # expert contexts from the train pool (unseen excluded globally).
    # TabPFN needs >= 2 classes to fit: single-family experts get a small
    # FILLER draw from the opposite side (attack expert -> benign filler;
    # benign expert -> known-attack filler) with REAL labels.  The retrieval
    # reference for scoring is always the OWNED rows only (filler excluded),
    # so the gate semantics stay "similarity to my families' support".
    def draw_class(c, n, band_off=0):
        pos = train_idx[y_train == c]
        perm = np.random.default_rng(args.seed + SEED_BAND_CONTEXT + c + band_off) \
            .permutation(pos)
        return perm[:n]

    contexts = {}
    for ename, owned in experts.items():
        counts = np.zeros(n_classes, dtype=np.int64)
        for c in owned:
            counts[c] = int((y_train == c).sum())
        tgt = natural_targets(counts, args.context_size)
        parts = [draw_class(c, int(tgt[c])) for c in owned if tgt[c] > 0]
        owned_idxs = np.sort(np.concatenate(parts))
        filler_idxs = np.array([], dtype=np.int64)
        if len(np.unique(label_fn(owned_idxs))) < 2:
            if ename == "benign":
                fill_classes = [c for c in range(n_classes)
                                if c not in (benign_id, unseen_id)
                                and (y_train == c).any()]
            else:
                fill_classes = [benign_id]
            per = max(1, args.filler_rows // len(fill_classes))
            filler_idxs = np.sort(np.concatenate(
                [draw_class(c, per, band_off=25) for c in fill_classes]))
        idxs = np.concatenate([owned_idxs, filler_idxs])
        Xc = feats(idxs)
        contexts[ename] = (Xc, label_fn(idxs), len(owned_idxs))
        print(f"context[{ename}]: owned {len(owned_idxs):,} + filler "
              f"{len(filler_idxs):,} rows "
              f"({ {class_names[c]: int(tgt[c]) for c in owned} })")
    del X
    gc.collect()

    target_families = set(class_names)
    X_aux, aux_meta = load_aux_rows(args, cfg, target_families,
                                    args.aux_cap, args.seed)
    aux_rows = X_aux[np.random.default_rng(args.seed + SEED_BAND_AUX)
                     .permutation(len(X_aux))[: args.id_sample]]
    core._PICKLE_CACHE.clear()
    gc.collect()

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
        print(f"[{ename}] fit {len(Xc):,} rows in {time.time()-t0:.1f}s", flush=True)

        dec = find_class_attention_module(clf)
        if dec is None:
            raise SystemExit(f"[{ename}] decoder q/k module not found")
        p0 = next(dec.parameters())
        # retrieval reference = OWNED rows only (filler excluded)
        emb_ref = np.asarray(clf.get_embeddings(
            Xc[: min(n_owned, args.emb_context_rows)], "test"))
        if emb_ref.ndim == 3:
            emb_ref = emb_ref[0]

        # roles for this expert
        own_val = np.concatenate([val_rows[c] for c in owned if c in val_rows])
        roles = {"id_val": own_val, "unseen": unseen_rows, "aux": aux_rows}
        if ename == "benign":
            roles["known_attack"] = np.concatenate(
                list(attack_test_rows.values())) if attack_test_rows else None
        else:
            non_owned = [v for c, v in attack_test_rows.items() if c not in owned]
            roles["non_owned_known_attack"] = (
                np.concatenate(non_owned) if non_owned else None)
            roles["benign"] = benign_test_rows

        sc = {}
        for role, Xr in roles.items():
            if Xr is None or len(Xr) == 0:
                continue
            emb = np.asarray(clf.get_embeddings(Xr, "test"))
            if emb.ndim == 3:
                emb = emb[0]
            with torch.inference_mode():
                q, kk = dec._project_qk(
                    torch.as_tensor(emb_ref).to(p0.device, p0.dtype).unsqueeze(0),
                    torch.as_tensor(emb).to(p0.device, p0.dtype).unsqueeze(0))
            lse, mx = streaming_scores(q.squeeze(0).float().cpu(),
                                       kk.squeeze(0).float().cpu(),
                                       args.energy_T, device=device)
            sc[role] = {"energy_head": lse, "maxsim": mx}
            print(f"  scores[{ename}/{role}] n={len(Xr):,}", flush=True)

        owned_names = ",".join(class_names[c] for c in owned)
        for sn in SCORES:
            idv = sc["id_val"][sn]
            tau = float(np.quantile(idv, args.threshold_quantile))
            for role in sc:
                if role == "id_val":
                    continue
                au, ap, f95 = ood_metrics(idv, sc[role][sn])
                rows3a.append({"expert": ename, "active_owned_classes": owned_names,
                               "score": sn, "ood_role": role,
                               "id_support": len(idv),
                               "ood_support": len(sc[role][sn]),
                               "auroc": round(au, 4), "aupr_id": round(ap, 4),
                               "fpr95": round(f95, 4)})
                rows3b.append({"expert": ename, "score": sn, "ood_role": role,
                               "threshold": tau,
                               "id_retain_at_tau": round(float((idv >= tau).mean()), 4),
                               "ood_detect_at_tau": round(
                                   float((sc[role][sn] < tau).mean()), 4)})
        hists[ename] = sc
        del clf
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    df3a, df3b = pd.DataFrame(rows3a), pd.DataFrame(rows3b)
    print("\n=== 3a (expert x score x role) ===")
    print(df3a.to_string(index=False))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp8_expertood_unseen_{args.unseen}")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"expert": e,
                   "owned": ",".join(class_names[c] for c in v)}
                  for e, v in experts.items()]).to_csv(
        os.path.join(out_dir, "0a_expert_map.csv"), index=False)
    df3a.to_csv(os.path.join(out_dir, "3a_expert_ood_scores.csv"), index=False)
    df3b.to_csv(os.path.join(out_dir, "3b_expert_ood_thresholds.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    enames = list(hists)
    fig, axes = plt.subplots(len(enames), 2, figsize=(12, 3.2 * len(enames)),
                             squeeze=False)
    colors = {"id_val": "#2a78d6", "unseen": "#e34948", "aux": "#eda100",
              "non_owned_known_attack": "#4a3aa7", "known_attack": "#4a3aa7",
              "benign": "#1baf7a"}
    for i, ename in enumerate(enames):
        for j, sn in enumerate(SCORES):
            ax = axes[i][j]
            for role, s in hists[ename].items():
                ax.hist(s[sn], bins=60, density=True, alpha=0.45,
                        label=role, color=colors.get(role, "#888"))
            tau = float(np.quantile(hists[ename]["id_val"][sn],
                                    args.threshold_quantile))
            ax.axvline(tau, color="k", ls="--", lw=1)
            ax.set_title(f"{ename} -- {sn}")
            ax.legend(fontsize=7)
    fig.suptitle(f"exp8 per-expert TabPFN OOD -- unseen={args.unseen} "
                 "(higher = owned; accept if >= tau)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3c_energy_hist_grid.png"), dpi=130)

    np.savez_compressed(
        os.path.join(out_dir, "scores_dump.npz"),
        **{f"{e}__{r}__{sn}": np.asarray(hists[e][r][sn], dtype=np.float32)
           for e in hists for r in hists[e] for sn in SCORES})
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
    p.add_argument("--context-size", type=int, default=200_000,
                   help="Per-expert context cap (natural proportion within "
                        "owned families).")
    p.add_argument("--id-sample", type=int, default=50_000)
    p.add_argument("--aux-cap", type=int, default=20_000)
    p.add_argument("--emb-context-rows", type=int, default=200_000)
    p.add_argument("--filler-rows", type=int, default=20_000,
                   help="Second-class filler rows for single-family experts "
                        "(real labels; excluded from the retrieval reference).")
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
    run_exp8(args)


if __name__ == "__main__":
    main()

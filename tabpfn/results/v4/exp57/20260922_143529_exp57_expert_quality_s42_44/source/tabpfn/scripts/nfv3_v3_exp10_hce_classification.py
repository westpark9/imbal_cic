#!/usr/bin/env python3
"""EXP10 -- HCE: taxonomy-disjoint expert CLASSIFICATION via chain rule.

Pre-registration: manuscript/report/0820.md SS11.  Closed-set only (no unseen).

Structure (diagram: docs/tabpfn_tacb_report.html SS8):
  global expert   natural-proportional context over ALL classes (same
                  construction as D1's va100: powerlaw alpha=1, shared
                  per-class permutation band seed+950).
  family experts  ONLY the multi-family taxonomy groups (user fix: a
                  single-family group's conditional is trivially 1, so
                  brute_force/benign need no expert -- their probability flows
                  through the global group marginal):
                    flood = ddos + dos      (natural within group, cap)
                    tail  = bot + infiltration + web_attacks (FULL pool)
  combiner        chain rule, no selection:
                    p(c|x) = p(g(c)|x) * p_expert(c|x, g)
                  where p(g|x) = sum of the global's class probs within g.

Columns (all offline from the dumped probabilities):
  global          the global expert alone (baseline)
  factorized      chain rule as above
  blend_l{v}      (1-l)*global + l*factorized; the single scalar l is selected
                  on VAL (tuple-constrained) and confirmed on TEST once
  oracle_group    perfect assignment upper bound: within the TRUE group, the
                  expert conditional's argmax (single-class groups: the class)
  hard_route      argmax-group assignment (the failure-mode-1 control)

Both --eval-split runs are needed (val for lambda selection, test for the
confirmation); fits are deterministic and resumed via prob dumps.

    python tabpfn/nfv3_v3_exp10_hce_classification.py --target-dataset cic2018 \\
        --fit-mode fit_with_cache --test-batch-size 500000                # test
    ... --eval-split val                                                  # val
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
CONFIG_KEY = {"cic2018": "cse_cic_ids2018", "cic2018_capped": "cse_cic_ids2018"}


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


def predict_batched(clf, X_eval, batch_size):
    if batch_size <= 0:
        batch_size = len(X_eval)
    out = []
    n_batches = (len(X_eval) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(X_eval), batch_size), 1):
        out.append(clf.predict_proba(X_eval[start:start + batch_size]))
        if bi == 1 or bi % 3 == 0 or start + batch_size >= len(X_eval):
            print(f"  predict batch {bi}/{n_batches}", flush=True)
    return np.concatenate(out)


def _h8(path):
    return hashlib.sha1(os.path.realpath(path).encode()).hexdigest()[:8]


def run_exp10(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp10_hce_classification"
    os.makedirs(args.resume_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    if args.subsample_samples:
        raise SystemExit("--subsample-samples must stay 0.")
    if args.eval_split == "val" and args.train_split != "train":
        raise SystemExit("--eval-split val is only valid with --train-split train.")

    groups_cfg = json.load(open(args.expert_config))[CONFIG_KEY[args.target_dataset]]

    tail_classes = cfg[args.target_dataset]["tail_classes"]
    X, class_names, train_idx, val_idx, test_idx, _, _, split_audit, label_fn = \
        cfg[args.target_dataset]["loader"](args)
    n_classes = len(class_names)

    # group map: every class -> group name (benign its own group)
    group_of = {class_names.index("benign"): "benign"}
    groups = {"benign": [class_names.index("benign")]}
    for gname, fams in groups_cfg.items():
        ids = [class_names.index(f) for f in fams if f in class_names]
        if ids:
            groups[gname] = ids
            for c in ids:
                group_of[c] = gname
    missing = [class_names[c] for c in range(n_classes) if c not in group_of]
    if missing:
        raise SystemExit(f"classes without a group: {missing}")
    multi_groups = {g: ids for g, ids in groups.items() if len(ids) >= 2}
    print("groups:", {g: [class_names[c] for c in v] for g, v in groups.items()})
    print("experts (multi-family only):", list(multi_groups))

    eval_pool_idx = test_idx if args.eval_split == "test" else val_idx
    eval_idx = core.cap_per_class(eval_pool_idx, label_fn(eval_pool_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_eval = label_fn(eval_idx)
    X_eval = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))
    print(f"eval split={args.eval_split}: {len(eval_idx):,} rows")

    y_train = label_fn(train_idx)

    def build_context(class_ids, k):
        counts = np.zeros(n_classes, dtype=np.int64)
        for c in class_ids:
            counts[c] = int((y_train == c).sum())
        tgt = natural_targets(counts, k)
        parts = []
        for c in class_ids:
            if tgt[c] == 0:
                continue
            perm = np.random.default_rng(args.seed + SEED_BAND_CONTEXT + c) \
                .permutation(train_idx[y_train == c])
            parts.append(perm[: int(tgt[c])])
        idxs = np.sort(np.concatenate(parts))
        return idxs, tgt

    # experts to fit: global + multi-family groups.  The global is D1's va100
    # (natural 1M, m=1) -- if its recorded dump is supplied, reuse it and skip
    # the ~50min global fit+predict (user decision; provenance in Args).
    probs_of = {}
    comp_rows = []
    fit_plan = {}
    if args.global_probs_npz:
        z = np.load(args.global_probs_npz)
        if not np.array_equal(z["y_true"], y_eval):
            raise SystemExit(
                f"--global-probs-npz y_true does not match this run's eval rows "
                f"({args.global_probs_npz}); wrong split or stale dump.")
        if list(z["class_names"]) != class_names:
            raise SystemExit("--global-probs-npz class_names mismatch.")
        probs_of["global"] = z["probs"].astype(np.float64)
        comp_rows.append({"expert": "global", "class": "(reused dump)",
                          "rows": 0})
        print(f"[global] reused recorded probs: {args.global_probs_npz}")
    else:
        fit_plan["global"] = (list(range(n_classes)), args.global_context_size)
    for g, ids in multi_groups.items():
        fit_plan[g] = (ids, args.expert_context_size)
    for ename, (class_ids, k) in fit_plan.items():
        idxs, tgt = build_context(class_ids, k)
        for c in np.flatnonzero(tgt):
            comp_rows.append({"expert": ename, "class": class_names[c],
                              "rows": int(tgt[c])})
        tagk = (f"exp10_{args.target_dataset}_{ename}_k{len(idxs)}_seed{args.seed}"
                f"_m{args.n_estimators}_fm{args.fit_mode}"
                f"_sp{args.train_split.replace('+', '')}"
                f"_d{_h8(args.data)}_m{_h8(args.model_path)}"
                f"_ev{args.eval_split}_tc{args.test_cap_per_class}"
                f"_tb{args.test_batch_size}_nt{len(eval_idx)}")
        ckpt = os.path.join(args.resume_dir, f"{tagk}_probs.npz")
        if not args.force_refit and os.path.exists(ckpt):
            print(f"[{ename}] resuming probs from {ckpt}")
            saved = np.load(ckpt)
            probs_of[ename] = saved["probs"].astype(np.float64)
        else:
            Xc = np.nan_to_num(np.asarray(X[idxs], dtype=np.float32))
            yc = label_fn(idxs)
            clf = TabPFNClassifier(
                device=args.device, model_path=args.model_path,
                ignore_pretraining_limits=args.ignore_pretraining_limits,
                random_state=args.seed, n_estimators=args.n_estimators,
                auto_scale_n_estimators=False, fit_mode=args.fit_mode,
                keep_cache_on_device=args.keep_cache_on_device)
            t0 = time.time()
            clf.fit(Xc, yc)
            print(f"[{ename}] fit {len(Xc):,} rows in {time.time()-t0:.1f}s",
                  flush=True)
            t0 = time.time()
            pr = predict_batched(clf, X_eval, args.test_batch_size)
            probs_of[ename] = full_proba(pr, clf.classes_, n_classes).astype(np.float64)
            np.savez_compressed(ckpt, probs=probs_of[ename].astype(np.float32))
            print(f"[{ename}] predict in {time.time()-t0:.1f}s -> {ckpt}", flush=True)
            del clf, Xc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()

    # ---------------- offline columns ----------------
    p_glob = probs_of["global"]
    gnames = list(groups)
    gidx = {g: np.asarray(ids) for g, ids in groups.items()}
    p_group = np.stack([p_glob[:, gidx[g]].sum(axis=1) for g in gnames], axis=1)
    p_group = p_group / p_group.sum(axis=1, keepdims=True)

    # within-group conditionals
    cond = np.zeros_like(p_glob)
    for g in gnames:
        ids = gidx[g]
        if g in probs_of:                       # multi-family expert
            e = probs_of[g][:, ids]
            e = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
            cond[:, ids] = e
        else:                                   # single-class group
            cond[:, ids] = 1.0

    factor = np.zeros_like(p_glob)
    for j, g in enumerate(gnames):
        factor[:, gidx[g]] = p_group[:, [j]] * cond[:, gidx[g]]

    true_group = np.array([gnames.index(group_of[c]) for c in range(n_classes)])
    columns = {"global": p_glob, "factorized": factor}
    for lam in args.blend_grid:
        columns[f"blend_l{lam:g}".replace(".", "p")] = \
            (1 - lam) * p_glob + lam * factor
    # oracle: within the true group of y, expert conditional argmax
    oracle_pred = np.empty(len(y_eval), dtype=np.int64)
    hard_pred = np.empty(len(y_eval), dtype=np.int64)
    ghat = p_group.argmax(axis=1)
    for j, g in enumerate(gnames):
        ids = gidx[g]
        sub = cond[:, ids]
        arg = ids[sub.argmax(axis=1)]
        mask_o = np.asarray([gnames.index(group_of[c]) for c in y_eval]) == j
        oracle_pred[mask_o] = arg[mask_o]
        hard_pred[ghat == j] = arg[ghat == j]

    all_rows = []
    for method, probs in columns.items():
        all_rows.extend(core.per_class_table(f"hce_{method}", y_eval,
                                             np.argmax(probs, axis=1),
                                             class_names, tail_classes))
    all_rows.extend(core.per_class_table("hce_oracle_group", y_eval, oracle_pred,
                                         class_names, tail_classes))
    all_rows.extend(core.per_class_table("hce_hard_route", y_eval, hard_pred,
                                         class_names, tail_classes))
    table = pd.DataFrame(all_rows)
    summary = table[table["class"].isin(["macro_avg", "tail_avg"])]
    print("\n=== summary (macro / tail F1) ===")
    print(summary.pivot(index="class", columns="method", values="f1").round(4)
          .to_string())

    ts = time.strftime("%Y%m%d_%H%M%S")
    eval_tag = "" if args.eval_split == "test" else "_evval"
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp10_hce{eval_tag}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(out_dir, "context_composition.csv"), index=False)
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "probs_dump.npz"),
                        y_true=y_eval.astype(np.int64),
                        class_names=np.asarray(class_names),
                        group_names=np.asarray(gnames),
                        p_group=p_group.astype(np.float32),
                        **{f"probs_{e}": p.astype(np.float32)
                           for e, p in probs_of.items()})
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump([{"experiment": args.experiment,
                    "eval_split": args.eval_split,
                    "experts": list(fit_plan),
                    "eval_rows": int(len(eval_idx))}], f, indent=2)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--expert-config",
                   default=os.path.join(core.REPO_ROOT, "configs", "nfv3_experts.json"))
    p.add_argument("--global-context-size", type=int, default=1_000_000)
    p.add_argument("--expert-context-size", type=int, default=250_000,
                   help="Cap per multi-family expert (tail's pool ~239k fits "
                        "whole; flood gets a natural-proportional 250k).")
    p.add_argument("--blend-grid", type=float, nargs="+",
                   default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--global-probs-npz", default=None,
                   help="Recorded global-expert probs npz (D1 va100 / E2 evval "
                        "probs_tabpfn_va100.npz). Reuses the record and skips "
                        "the global fit; y_true/class_names are verified.")
    p.add_argument("--eval-split", default="test", choices=["test", "val"])
    p.set_defaults(
        max_train_samples=-1,
        n_estimators=1,
        test_cap_per_class=0,
        fit_mode="fit_with_cache",
        test_batch_size=500_000,
        subsample_samples=0,
    )
    args = p.parse_args()
    run_exp10(args)


if __name__ == "__main__":
    main()

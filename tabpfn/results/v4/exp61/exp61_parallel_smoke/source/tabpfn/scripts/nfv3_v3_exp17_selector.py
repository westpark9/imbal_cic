#!/usr/bin/env python3
"""EXP17 -- learned global-vs-expert SELECTOR (user structure change).

Pre-registration: 0821.md SS12.  The user's diagnosis: the exp12c structure
has no return path once a row is mis-assigned.  New structure:

  candidates = { global (natural-proportion 1M context),
                 flood{ddos,dos}, bf_bot{brute,bot}, inf{infiltration},
                 web{web_attacks} }                     (user-corrected set)
  (the benign expert is REMOVED -- global plays the benign/default exit)

  single-class experts (inf, web): TabPFN fit needs >=2 classes, so their
  context is own-class rows + a disjoint benign filler (drawn from the same
  benign permutation AFTER the selector-holdout slice -- no leakage).  The
  filler exists to make the fit legal and to give the selector a meaningful
  P(own) feature; when a single-class expert is CHOSEN the label is its
  owned class (deterministic).

  selector input  = [ p_global(7) | p_flood(2) | p_bf_bot(2) | p_inf(2) |
                      p_web(2) ]  = 15 dims (candidates' distributions on x)
  selector target = global      if y == benign
                    expert k    if y in owned(k)         (label-derived)
  selector        = 2-layer MLP(32), 5-way cross-entropy, target-balanced
                    batches, trained on TRAIN-side holdout rows (same data
                    regime as the exp11c gate heads -- NOT validation
                    predictions)
  inference       = argmax choice; label = global's 7-way argmax if global
                    chosen, else the chosen expert's owned-class argmax
                    (single-class experts: the owned class)

DISCLOSED RISK (required by SRC_HISTORY): this is nearest to recorded failure
mode #2 (competence models fitted on predictions do not transfer under chrono
splits) and the exp5 D1 view-selector death.  Escape arguments recorded in
0821.md SS12; if the selector lands at or below exp12c argmax gating or the
global column, it is recorded as a #2 reproduction.

    python tabpfn/nfv3_v3_exp17_selector.py --target-dataset cic2018 \\
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
    "inf": ["infiltration"],
    "web": ["web_attacks"],
}


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


def build_expert_data(args, experts, class_names, train_idx, y_train,
                      benign_id):
    """Contexts + holdouts for the attack experts (no benign expert here).

    Single-class experts get a benign filler so the TabPFN fit is legal;
    the filler is drawn from the benign permutation AFTER the slice reserved
    for the selector holdout (skip bookkeeping below), so the selector's
    benign training rows never sit inside any expert context."""
    def draw_class(c, n, off=0, skip=0):
        perm = np.random.default_rng(args.seed + SEED_BAND_CONTEXT + c + off) \
            .permutation(train_idx[y_train == c])
        return perm[skip: skip + n]

    n_classes = len(class_names)
    contexts, holdouts = {}, {}
    benign_skip = args.holdout_rows              # slice 0 = selector holdout
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
        filler = np.array([], dtype=np.int64)
        if len(owned) == 1:                       # inf / web: fit needs 2 cls
            filler = draw_class(benign_id, args.filler_rows, skip=benign_skip)
            benign_skip += args.filler_rows
        contexts[ename] = np.sort(np.concatenate(parts + [filler]))
        holdouts[ename] = np.sort(np.concatenate(hold))
        print(f"context[{ename}]: owned "
              f"{len(contexts[ename]) - len(filler):,} + benign filler "
              f"{len(filler):,}; holdout {len(holdouts[ename]):,}")
    return contexts, holdouts, draw_class


def train_selector(feats, targets, n_choices, seed):
    """Target-balanced 4-way CE on standardized candidate-probability
    features (mirrors the gate heads' batch regime)."""
    import torch
    torch.manual_seed(seed)
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd == 0] = 1.0
    Xt = torch.tensor((feats - mu) / sd)
    groups = [np.flatnonzero(targets == t) for t in range(n_choices)]
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, n_choices))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    per = 2048
    steps = 60 * max(1, len(Xt) // (per * n_choices))
    for _ in range(steps):
        idx_parts, y_parts = [], []
        for t, grp in enumerate(groups):
            if len(grp) == 0:
                continue
            pick = grp[torch.randint(len(grp), (per,), generator=g).numpy()]
            idx_parts.append(torch.as_tensor(pick))
            y_parts.append(torch.full((per,), t, dtype=torch.long))
        idx = torch.cat(idx_parts); yb = torch.cat(y_parts)
        opt.zero_grad()
        lossf(net(Xt[idx]), yb).backward()
        opt.step()

    def predict(f):
        with torch.no_grad():
            z = torch.tensor(((f - mu) / sd).astype(np.float32))
            return net(z).numpy()
    return predict


def run_exp17(args):
    cfg = core.build_dataset_config(args.data_dir)
    if args.data is None:
        args.data = cfg[args.target_dataset]["default_data"]
    args.auto_scale_n_estimators = False
    args.experiment = "exp17_selector"
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
    owner_of = {}
    for j, e in enumerate(enames):
        for c in experts[e]:
            owner_of[c] = j + 1                    # choice 0 = global
    n_choices = 1 + len(enames)
    print("candidates: ['global'] +", enames)

    y_train = label_fn(train_idx)
    eval_idx = core.cap_per_class(test_idx, label_fn(test_idx), n_classes,
                                  args.test_cap_per_class, args.seed + 900)
    y_eval = label_fn(eval_idx)
    X_eval = np.nan_to_num(np.asarray(X[eval_idx], dtype=np.float32))
    print(f"eval rows: {len(eval_idx):,}")

    def feats_of(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    contexts, holdouts, draw_class = build_expert_data(
        args, experts, class_names, train_idx, y_train, benign_id)
    benign_hold = np.sort(draw_class(benign_id, args.holdout_rows))
    print(f"benign holdout (selector train, target=global): "
          f"{len(benign_hold):,}")
    sel_idx = np.concatenate([benign_hold] + [holdouts[e] for e in enames])
    y_sel = label_fn(sel_idx)
    sel_target = np.array([0 if c == benign_id else owner_of[c]
                           for c in y_sel], dtype=np.int64)
    X_sel = feats_of(sel_idx)
    ctx_data = {e: (feats_of(idxs), label_fn(idxs)) for e, idxs
                in contexts.items()}

    g_idx = core.stratified_subset(train_idx, y_train, n_classes,
                                   args.global_context_size,
                                   args.seed + SEED_BAND_GLOBAL)
    glob_ctx = (feats_of(g_idx), label_fn(g_idx))
    comp = {class_names[c]: int(n) for c, n in
            zip(*np.unique(glob_ctx[1], return_counts=True))}
    print(f"global context: {len(g_idx):,} rows {comp}")
    del X
    core._PICKLE_CACHE.clear()
    gc.collect()

    import torch

    def fit_predict(name, Xc, yc, row_sets):
        t0 = time.time()
        clf = TabPFNClassifier(
            device=args.device, model_path=args.model_path,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            random_state=args.seed, n_estimators=args.n_estimators,
            auto_scale_n_estimators=False, fit_mode=args.fit_mode,
            keep_cache_on_device=args.keep_cache_on_device)
        clf.fit(Xc, yc)
        outs = []
        bs = args.test_batch_size or None
        for Xr in row_sets:
            pr = []
            step = bs or len(Xr)
            for s0 in range(0, len(Xr), step):
                pr.append(clf.predict_proba(Xr[s0:s0 + step]))
                if len(Xr) > step:
                    print(f"  [{name}] predict {min(s0+step, len(Xr)):,}"
                          f"/{len(Xr):,}", flush=True)
            outs.append((np.concatenate(pr), np.asarray(clf.classes_,
                                                        dtype=np.int64)))
        del clf
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        print(f"[{name}] fit+predict done in {time.time()-t0:.0f}s", flush=True)
        return outs

    # ---- candidate probabilities on selector-train rows and eval ----
    (glob_sel, _), (glob_eval, glob_classes) = fit_predict(
        "global", glob_ctx[0], glob_ctx[1], [X_sel, X_eval])
    p_glob_sel = full_proba(glob_sel, glob_classes, n_classes)
    p_glob_eval = full_proba(glob_eval, glob_classes, n_classes)

    exp_sel, exp_eval = {}, {}
    for ename in enames:
        Xc, yc = ctx_data[ename]
        (ps, cls_s), (pe, _) = fit_predict(ename, Xc, yc, [X_sel, X_eval])
        order = np.argsort(cls_s)                  # owned ids ascending
        exp_sel[ename] = ps[:, order]
        exp_eval[ename] = pe[:, order]

    def stack(pg, pex):
        return np.concatenate([pg] + [pex[e] for e in enames],
                              axis=1).astype(np.float32)

    F_sel = stack(p_glob_sel, exp_sel)
    F_eval = stack(p_glob_eval, exp_eval)
    print(f"selector features: {F_sel.shape[1]} dims, "
          f"train {len(F_sel):,} rows", flush=True)

    predict_logits = train_selector(F_sel, sel_target, n_choices, args.seed)
    choice = predict_logits(F_eval).argmax(axis=1)

    sel_names = ["global"] + enames
    sel_tab = pd.DataFrame({
        "class": [class_names[c] for c in y_eval],
        "chosen": [sel_names[c] for c in choice]}) \
        .value_counts().reset_index(name="rows")
    print("\n=== selection counts (true class x chosen source) ===")
    print(sel_tab.sort_values(["class", "rows"], ascending=[True, False])
          .to_string(index=False))

    # ---- final labels ----
    final = np.empty(len(y_eval), dtype=np.int64)
    m0 = choice == 0
    final[m0] = np.argmax(p_glob_eval[m0], axis=1)
    for j, ename in enumerate(enames):
        m = choice == j + 1
        if not m.any():
            continue
        owned_sorted = np.sort(np.asarray(experts[ename]))
        if len(owned_sorted) == 1:
            final[m] = owned_sorted[0]            # single-class: deterministic
        else:
            final[m] = owned_sorted[np.argmax(exp_eval[ename][m], axis=1)]

    all_rows = list(core.per_class_table("system_selector", y_eval, final,
                                         class_names, tail_classes))
    all_rows.extend(core.per_class_table("global_reference", y_eval,
                                         np.argmax(p_glob_eval, axis=1),
                                         class_names, tail_classes))
    table = pd.DataFrame(all_rows)
    print("\n=== per-class F1 ===")
    print(table.pivot(index="class", columns="method", values="f1").round(4)
          .to_string())

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"{ts}_{cfg[args.target_dataset]['out_tag']}_exp17_selector")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    sel_tab.to_csv(os.path.join(out_dir, "4a_selection_counts.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "system_dump.npz"),
                        choice=choice.astype(np.int8), final=final,
                        y_true=y_eval.astype(np.int64),
                        p_global=p_glob_eval.astype(np.float32),
                        candidate_names=np.asarray(sel_names),
                        class_names=np.asarray(class_names))
    split_audit.to_csv(os.path.join(out_dir, "0b_split_audit.csv"), index=False)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"\nWrote {out_dir}")
    return out_dir


def main():
    p = core.base_parser(__doc__)
    p.add_argument("--context-size", type=int, default=200_000)
    p.add_argument("--holdout-rows", type=int, default=15_000)
    p.add_argument("--filler-rows", type=int, default=20_000,
                   help="benign filler for single-class experts (fit needs "
                        ">=2 classes); drawn AFTER the selector-holdout slice")
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
    run_exp17(args)


if __name__ == "__main__":
    main()

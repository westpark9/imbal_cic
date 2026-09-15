#!/usr/bin/env python3
"""C0 context construction shared by the SOTA adapters (exp35/36/37) -- a VERBATIM
copy of the functions nfv3_v3_exp31_c0alloc.py uses to draw the record-grade
"100k, benign share 0.75, balanced attacks" context (0902.md §8ai, chain 13), so
that every external method can be handed EXACTLY the rows the project's TabPFN-v3
result was built on.

Pipeline (exp31 run_exp29 lines 736-870, defaults of the record runs):
  train split -> scenario_stratified_partition(context_frac 0.5, expert_frac 0.25)
              -> D_global (= ctx_pool: the chronologically FIRST 50% of every
                 (class, attack_scenario) group of the train split)
              -> representative_subset_share(D_global, budget, seed + 990,
                 benign_share, alloc)  = C0
  c0_dedup none / c0_conflict keep (the record defaults) are the only options here.

Copied functions (unchanged): pick_rows, representative_subset,
representative_subset_share, scenario_stratified_partition, class_chrono_partition,
SEED_BAND_GLOBAL.  Adding a helper alongside instead of importing exp31 keeps that
2,300-line script frozen (CLAUDE.md) and avoids executing its module-level code.

Check against the record: run 20260904_010654 (seed 42) 0g_c0_scenario.csv =
benign 75,000 / bot 4,848 / ftp 3,270 + ssh 1,578 / hoic 3,742 + loic_udp 20 +
loic_http 1,086 / goldeneye 981 + hulk 1,643 + slowhttptest 1,673 + slowloris 551 /
infiltration 4,847 / web 485 + 144 + 132.  build_c0() returns the same per-scenario
table for a byte-identical draw.
"""

import time

import numpy as np
import pandas as pd

import nfv3_v3_common as core

SEED_BAND_GLOBAL = 990          # exp31 line 57


# ---------------------------------------------------------------------------
# verbatim from nfv3_v3_exp31_c0alloc.py
# ---------------------------------------------------------------------------

def pick_rows(rows, k, seed):
    if k >= len(rows):
        return rows
    return np.random.default_rng(seed).permutation(rows)[:k]


def representative_subset(pool_idx, pool_y, n_classes, budget, seed):
    """Pooled budget, natural class ratios (largest remainder, >=1/present)."""
    counts = np.bincount(pool_y, minlength=n_classes)
    total = int(counts.sum())
    if budget <= 0 or budget >= total:
        return np.sort(pool_idx)
    raw = counts * (budget / total)
    tgt = np.minimum(np.floor(raw).astype(np.int64), counts)
    present = counts > 0
    tgt[present & (tgt < 1)] = 1
    rem = budget - int(tgt.sum())
    for cid in np.argsort(-(raw - np.floor(raw))):
        if rem <= 0:
            break
        if tgt[cid] < counts[cid]:
            tgt[cid] += 1
            rem -= 1
    chosen = [pick_rows(pool_idx[pool_y == cid], int(tgt[cid]), seed + cid)
              for cid in range(n_classes) if tgt[cid] > 0]
    return np.sort(np.concatenate(chosen))


def representative_subset_share(pool_idx, pool_y, n_classes, budget, seed,
                                benign_id, benign_share, alloc="natural"):
    """THE KNOB (exp30): pooled budget with benign pinned to `benign_share`
    of the budget; the rest is split over the other present classes by their
    natural ratios (largest remainder), capped by availability with the
    shortfall redistributed.  Per-class draw seeds are the same as
    representative_subset, so a class whose target count is unchanged draws
    the identical rows."""
    counts = np.bincount(pool_y, minlength=n_classes)
    total = int(counts.sum())
    if budget <= 0 or budget >= total:
        return np.sort(pool_idx)
    tgt = np.zeros(n_classes, dtype=np.int64)
    tgt[benign_id] = min(int(round(budget * benign_share)),
                         int(counts[benign_id]))
    rem = budget - int(tgt[benign_id])
    for _ in range(n_classes):
        sub = np.asarray([c for c in range(n_classes)
                          if c != benign_id and tgt[c] < counts[c]],
                         dtype=np.int64)
        if rem <= 0 or len(sub) == 0:
            break
        if alloc == "balanced":                      # THE KNOB (exp31 A)
            raw = np.full(len(sub), rem / len(sub), dtype=np.float64)
        else:
            raw = counts[sub] * (rem / counts[sub].sum())
        add = np.floor(raw).astype(np.int64)
        for j in np.argsort(-(raw - np.floor(raw))):
            if int(add.sum()) >= rem:
                break
            add[j] += 1
        add = np.minimum(add, counts[sub] - tgt[sub])
        tgt[sub] += add
        rem = budget - int(tgt.sum())
    present = counts > 0
    tgt[present & (tgt < 1)] = 1
    chosen = [pick_rows(pool_idx[pool_y == cid], int(tgt[cid]), seed + cid)
              for cid in range(n_classes) if tgt[cid] > 0]
    return np.sort(np.concatenate(chosen))


def scenario_stratified_partition(train_idx, y_train, ts_train, scen_train,
                                  ctx_frac, exp_frac, class_names):
    """THE KNOB (exp27): 50/25/25 chrono split applied WITHIN each
    (class, attack_scenario) group instead of within each class.

    Preserves "context precedes expert/route" inside every scenario; drops only
    the global ordering between scenarios, which is what starved C0 of
    ssh_bruteforce (0831.md cause A).  Scenarios too small to fill three parts
    go wholly to context and are flagged in the audit.
    """
    parts = {"context": [], "expert": [], "route": []}
    scen_rows = []
    for cid, cname in enumerate(class_names):
        cmask = y_train == cid
        if not cmask.any():
            continue
        for sname in sorted(np.unique(scen_train[cmask])):
            mask = cmask & (scen_train == sname)
            rows = train_idx[mask]
            if len(rows) == 0:
                continue
            order = rows[np.argsort(ts_train[mask], kind="stable")]
            n = len(order)
            n_ctx, n_exp = int(n * ctx_frac), int(n * exp_frac)
            n_rt = n - n_ctx - n_exp
            if min(n_ctx, n_exp, n_rt) <= 0:      # too small to split 3 ways
                parts["context"].extend(order)
                scen_rows.append({"class": cname, "scenario": sname,
                                  "train_total": n, "context": n, "expert": 0,
                                  "route": 0, "note": "too_small_all_context"})
                continue
            parts["context"].extend(order[:n_ctx])
            parts["expert"].extend(order[n_ctx:n_ctx + n_exp])
            parts["route"].extend(order[n_ctx + n_exp:])
            scen_rows.append({"class": cname, "scenario": sname,
                              "train_total": n, "context": n_ctx,
                              "expert": n_exp, "route": n_rt, "note": ""})
    scen_audit = pd.DataFrame(scen_rows)
    audit = (scen_audit.groupby("class", as_index=False)
             [["train_total", "context", "expert", "route"]].sum())
    return ({k: np.sort(np.asarray(v, dtype=np.int64))
             for k, v in parts.items()}, audit, scen_audit)


def class_chrono_partition(train_idx, y_train, ts_train, ctx_frac, exp_frac,
                           class_names):
    """Train -> D_global(context) / D_expert / D_route, chrono per class."""
    parts = {"context": [], "expert": [], "route": []}
    audit = []
    for cid, cname in enumerate(class_names):
        mask = y_train == cid
        rows = train_idx[mask]
        if len(rows) == 0:
            continue
        order = rows[np.argsort(ts_train[mask], kind="stable")]
        n = len(order)
        n_ctx = int(n * ctx_frac)
        n_exp = int(n * exp_frac)
        n_rt = n - n_ctx - n_exp
        if min(n_ctx, n_exp, n_rt) <= 0:
            raise SystemExit(
                f"class {cname}: {n} train rows cannot fill "
                f"context/expert/route ({n_ctx}/{n_exp}/{n_rt})")
        parts["context"].extend(order[:n_ctx])
        parts["expert"].extend(order[n_ctx:n_ctx + n_exp])
        parts["route"].extend(order[n_ctx + n_exp:])
        audit.append({"class": cname, "train_total": n, "context": n_ctx,
                      "expert": n_exp, "route": n_rt})
    return ({k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in parts.items()},
            pd.DataFrame(audit))


# ---------------------------------------------------------------------------
# adapter-facing API
# ---------------------------------------------------------------------------

def add_c0_args(parser):
    g = parser.add_argument_group(
        "context recipe (c0 = the record-grade composed context of exp30/31, chain 13)")
    g.add_argument("--context-recipe", default="stratified", choices=["stratified", "c0"],
                   help="stratified: --max-train-samples ratio-preserving rows from the whole train "
                        "split (exp1 style). c0: exp31's composed context -- see nfv3_v3_c0_context.py.")
    g.add_argument("--c0-size", type=int, default=100_000, help="exp31 --global-context-size")
    g.add_argument("--c0-benign-share", type=float, default=0.75,
                   help="benign share of the C0 budget; -1 = natural share (exp31 default -1)")
    g.add_argument("--c0-attack-alloc", default="balanced", choices=["natural", "balanced"])
    g.add_argument("--c0-context-frac", type=float, default=0.5, help="exp31 --context-frac")
    g.add_argument("--c0-expert-frac", type=float, default=0.25, help="exp31 --expert-frac")
    g.add_argument("--c0-pool-partition", default="scenario_stratified",
                   choices=["scenario_stratified", "chrono"])
    return g


def build_c0(args, train_idx, y_train, class_names, label_fn):
    """Return dict(idx, pool_audit, scen_audit, c0_scenario, timings) for the C0
    drawn exactly as exp31 draws it (defaults c0_dedup=none, c0_conflict=keep)."""
    t0 = time.time()
    n_classes = len(class_names)
    d = core.load_pickle(args.data)
    ts_all = np.asarray(d["timestamps" if "timestamps" in d else "time_proxy"], dtype=np.int64)
    scen_all = (np.asarray(d["attack_scenarios"]).astype(str)
                if "attack_scenarios" in d else None)
    del d
    ts_train = ts_all[train_idx]
    if args.c0_pool_partition == "scenario_stratified":
        if scen_all is None:
            raise SystemExit("--c0-pool-partition scenario_stratified needs 'attack_scenarios'")
        pools, pool_audit, scen_audit = scenario_stratified_partition(
            train_idx, y_train, ts_train, scen_all[train_idx],
            args.c0_context_frac, args.c0_expert_frac, class_names)
    else:
        pools, pool_audit = class_chrono_partition(
            train_idx, y_train, ts_train, args.c0_context_frac, args.c0_expert_frac, class_names)
        scen_audit = None
    ctx_pool = pools["context"]
    y_ctx_pool = label_fn(ctx_pool)
    names_l0 = [str(n).lower() for n in class_names]
    benign_id0 = next((names_l0.index(n) for n in ("benign", "normal") if n in names_l0), None)
    share = args.c0_benign_share
    if share < 0 and args.c0_attack_alloc == "balanced":
        share = float((y_ctx_pool == benign_id0).mean())
    if share >= 0:
        if benign_id0 is None:
            raise SystemExit("--c0-benign-share needs a benign/normal class")
        if share > 1:
            raise SystemExit("--c0-benign-share must be in [0, 1] or -1")
        g_idx = representative_subset_share(ctx_pool, y_ctx_pool, n_classes, args.c0_size,
                                            args.seed + SEED_BAND_GLOBAL, benign_id0, share,
                                            alloc=args.c0_attack_alloc)
    else:
        g_idx = representative_subset(ctx_pool, y_ctx_pool, n_classes, args.c0_size,
                                      args.seed + SEED_BAND_GLOBAL)
    y_g0 = label_fn(g_idx)
    c0_scen = None
    if scen_all is not None:
        c0_scen = (pd.DataFrame({"class": [class_names[c] for c in y_g0],
                                 "scenario": scen_all[g_idx]})
                   .value_counts().rename("c0_rows").reset_index()
                   .sort_values(["class", "scenario"]))
    timings = {
        "c0_pool_rows": int(len(ctx_pool)), "c0_rows": int(len(g_idx)),
        "c0_benign_share_arg": args.c0_benign_share,
        "c0_benign_share_realized": (round(float((y_g0 == benign_id0).mean()), 6)
                                     if benign_id0 is not None else None),
        "c0_attack_alloc": args.c0_attack_alloc, "c0_pool_partition": args.c0_pool_partition,
        "c0_seed": int(args.seed + SEED_BAND_GLOBAL), "c0_build_seconds": round(time.time() - t0, 1),
    }
    print(f"C0 ({args.c0_pool_partition}, ctx_frac {args.c0_context_frac}): pool {len(ctx_pool):,} -> "
          f"{len(g_idx):,} rows, benign share {timings['c0_benign_share_realized']}, "
          f"alloc {args.c0_attack_alloc}, seed {timings['c0_seed']}  ({timings['c0_build_seconds']}s)")
    if c0_scen is not None:
        print(c0_scen.to_string(index=False))
    return {"idx": g_idx, "pool_audit": pool_audit, "scen_audit": scen_audit,
            "c0_scenario": c0_scen, "timings": timings}

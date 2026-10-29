#!/usr/bin/env python3
"""EXP54 round 2 (fork of exp54_expert_context_design; helpers imported, not edited).
Round-1 readout: sqrt-prior context (share ~ sqrt(n_c)) was the only arm that lifted a weak
class (scanning 0.138 -> 0.614) and macro-F1 (0.684 -> 0.730); mitm/ransomware precision was
flat across every composition. Round 2 asks four follow-ups, one fit each:

    pow0.25 / pow0.75      is sqrt the right exponent? (share ~ n_c^p; p=0.5 = round-1 sqrt)
    sqrt_natural_s43/_s44  draw variance of the same sqrt rule (context rows re-drawn; TabPFN
                           random_state fixed at --seed so only the draw changes)
    sqrt_natural_40k       does doubling the context (same rule) help?
    hardneg:<target>       target 2,000 + 8,000 benign HARD negatives (benign train rows nearest
                           to the target class in z-scored feature space) + 6,000 random benign
                           + 500 per remaining class -- the boundary-sharpening alternative to
                           round-1 'contrastive' (random benign).
Same clean ToN split, same global reference (EXP48 frozen_cache), same metrics as round 1.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nfv3_conflict_clean import install_clean_loader, write_json
from nfv3_v3_exp31_c0alloc import pick_rows, PriorCorrector
from exp54_expert_context_design import make_clf, batched_proba, confusion, prf, alloc, build


def hard_negatives(X, target_rows, benign_rows, k, seed, sample=300_000, threads=8):
    rng = np.random.default_rng(seed)
    ben_s = rng.choice(benign_rows, min(sample, len(benign_rows)), replace=False)
    ref = np.concatenate([X[ben_s], X[target_rows]]).astype(np.float64)
    mu, sd = ref.mean(0), ref.std(0); sd[sd == 0] = 1
    with threadpool_limits(threads):
        nn = NearestNeighbors(n_neighbors=1, n_jobs=threads).fit((X[target_rows] - mu) / sd)
        d, _ = nn.kneighbors((X[ben_s] - mu) / sd)
    order = np.argsort(d[:, 0], kind="stable")[:k]
    return ben_s[order], ben_s[order[-1]:order[-1] + 1], float(d[order[-1], 0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-manifest", required=True)
    p.add_argument("--global-cache", required=True)
    p.add_argument("--data", default="/home/user/Desktop/imbalcic/data/nfv3_energy_suite_uncapped_scenarios.pkl")
    p.add_argument("--targets", default="mitm,ransomware,scanning,injection")
    p.add_argument("--hardneg-targets", default="scanning,mitm,ransomware")
    p.add_argument("--context-size", type=int, default=20_000)
    p.add_argument("--powers", default="0.25,0.75")
    p.add_argument("--draw-seeds", default="43,44")
    p.add_argument("--big-context-size", type=int, default=40_000)
    p.add_argument("--hardneg-target-rows", type=int, default=2000)
    p.add_argument("--hardneg-rows", type=int, default=8000)
    p.add_argument("--hardneg-random-benign", type=int, default=6000)
    p.add_argument("--hardneg-other-rows", type=int, default=500)
    p.add_argument("--prior-alpha", type=float, default=1.0)
    p.add_argument("--betas", default="0.5,1.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--fit-mode", default="fit_with_cache")
    p.add_argument("--device", default="auto")
    p.add_argument("--model-path", default="/home/user/Desktop/imbalcic/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt")
    p.add_argument("--ignore-pretraining-limits", action="store_true")
    p.add_argument("--test-batch-size", type=int, default=500_000)
    p.add_argument("--out-root", default="/home/user/Desktop/imbalcic/tabpfn/results")
    args = p.parse_args()
    print("Args: " + json.dumps(vars(args), sort_keys=True), flush=True)
    started = time.time()

    import nfv3_v3_common as core
    install_clean_loader(core, args.clean_manifest)
    X, names, train_idx, val_idx, test_idx, _, _, _, label_fn = core.load_ton_iot(argparse.Namespace(data=args.data))
    n = len(names); ben = names.index("benign")
    targets = [t for t in args.targets.split(",") if t]
    y_train = label_fn(train_idx)
    rows_by_class = [train_idx[y_train == c] for c in range(n)]
    avail = np.array([len(r) for r in rows_by_class]); train_prior = avail / avail.sum()

    cache = Path(args.global_cache)
    assert np.array_equal(np.load(cache / "eval_ids.npy"), np.asarray(test_idx))
    y_test = np.load(cache / "eval_y.npy")
    g_pred = np.load(cache / "eval_p0.npy", mmap_mode="r").argmax(1)
    g_right = g_pred == y_test
    g_f1, g_prec, g_rec, support = prf(confusion(y_test, g_pred, n))
    print(f"global macro_f1={g_f1.mean():.6f}", flush=True)

    out = Path(args.out_root) / (time.strftime("%Y%m%d_%H%M%S") + "_nfv3_toniot_exp54r2_expert_context_design")
    out.mkdir(parents=True, exist_ok=False)
    betas = [float(b) for b in args.betas.split(",")]
    B = args.context_size

    # ---- arms: (key, arm, target, context row indices, note) ----
    arms = []
    for pw in [float(v) for v in args.powers.split(",")]:
        sizes = alloc(avail ** pw, B, avail)
        arms.append((f"pow{pw}", f"pow{pw}", None, build(rows_by_class, sizes, args.seed + 1000), {}))
    for ds in [int(v) for v in args.draw_seeds.split(",")]:
        sizes = alloc(np.sqrt(avail), B, avail)
        arms.append((f"sqrt_natural_s{ds}", "sqrt_natural_draw", None, build(rows_by_class, sizes, ds + 1000), {"draw_seed": ds}))
    sizes = alloc(np.sqrt(avail), args.big_context_size, avail)
    arms.append(("sqrt_natural_40k", "sqrt_natural_big", None, build(rows_by_class, sizes, args.seed + 1000), {"context_size": args.big_context_size}))
    for t in [v for v in args.hardneg_targets.split(",") if v]:
        tid = names.index(t)
        hard, _, radius = hard_negatives(X, rows_by_class[tid], rows_by_class[ben], args.hardneg_rows, args.seed)
        rest_ben = np.setdiff1d(rows_by_class[ben], hard)
        parts = [pick_rows(rows_by_class[tid], args.hardneg_target_rows, args.seed + tid), hard,
                 pick_rows(rest_ben, args.hardneg_random_benign, args.seed + 77)]
        parts += [pick_rows(rows_by_class[c], args.hardneg_other_rows, args.seed + c) for c in range(n) if c not in (tid, ben)]
        arms.append((f"hardneg:{t}", "hardneg", t, np.sort(np.concatenate(parts)), {"hardneg_radius_z": radius}))
        print(f"[hardneg:{t}] hard-negative radius (z-dist of 8000th nearest benign) = {radius:.4f}", flush=True)

    metric_rows, quality_rows, compositions = [], [], {}
    for key, arm, target, ctx, note in arms:
        yc = label_fn(ctx); counts = np.bincount(yc, minlength=n)
        compositions[key] = {"note": note, "rows": {nm: int(c) for nm, c in zip(names, counts)}}
        print(f"[{key}] rows={len(ctx):,} " + ", ".join(f"{nm}={c}" for nm, c in zip(names, counts) if c), flush=True)
        t0 = time.time()
        clf = make_clf(args, X[ctx], yc)
        praw = batched_proba(args, clf, X[test_idx], n, tag=key)
        del clf
        print(f"[{key}] fit+predict {time.time() - t0:.1f}s", flush=True)
        corr = PriorCorrector(yc, n, train_prior, args.prior_alpha)
        variants = [("raw", praw)] + [(f"prior_b{b}", corr.correct(praw, b, 1.0)) for b in betas]
        eval_targets = targets if target is None else [target]
        for vname, prob in variants:
            pred = prob.argmax(1)
            f1, prec, rec, _ = prf(confusion(y_test, pred, n))
            e_right = pred == y_test
            H_all = int((~g_right & e_right).sum()); D_all = int((g_right & ~e_right).sum())
            for c in range(n):
                metric_rows.append({"arm": arm, "target": target or "", "key": key, "variant": vname, "class": names[c],
                                    "ctx_rows": int(counts[c]), "ctx_frac": float(counts[c] / counts.sum()),
                                    "support": int(support[c]), "f1": float(f1[c]), "precision": float(prec[c]),
                                    "recall": float(rec[c]), "global_f1": float(g_f1[c]), "delta_f1": float(f1[c] - g_f1[c]),
                                    "macro_f1": float(f1.mean())})
            for t in eval_targets:
                tid = names.index(t)
                region = (y_test == tid) | (pred == tid) | (g_pred == tid)
                H_t = int((region & ~g_right & e_right).sum()); D_t = int((region & g_right & ~e_right).sum())
                quality_rows.append({"arm": arm, "target": target or "", "key": key, "variant": vname, "eval_target": t,
                                     "target_f1": float(f1[tid]), "target_precision": float(prec[tid]), "target_recall": float(rec[tid]),
                                     "global_target_f1": float(g_f1[tid]), "delta_target_f1": float(f1[tid] - g_f1[tid]),
                                     "H_all": H_all, "D_all": D_all, "H_ratio_all": H_all / max(H_all + D_all, 1),
                                     "H_target": H_t, "D_target": D_t, "H_ratio_target": H_t / max(H_t + D_t, 1),
                                     "macro_f1": float(f1.mean())})
        del praw, variants
        pd.DataFrame(metric_rows).to_csv(out / "per_class_metrics.csv", index=False)
        pd.DataFrame(quality_rows).to_csv(out / "candidate_quality.csv", index=False)
        write_json(out / "contexts.json", compositions)

    write_json(out / "COMPLETE.json", {"seconds": time.time() - started, "targets": targets,
               "global_macro_f1": float(g_f1.mean()), "arms": [k for k in compositions]})
    print("WROTE: " + str(out), flush=True)


if __name__ == "__main__":
    main()

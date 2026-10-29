#!/usr/bin/env python3
"""EXP54: which context composition makes a good class specialist that is also
a good scorer candidate? Standalone harness (same as EXP53: no clustering, no
shared anchor, no K-selection) on the clean ToN split. Every arm is one TabPFN
fit on a hand-built 20k context, predicted on the full test set, and compared
against the frozen EXP48 global predictions (same rows, same order -- verified).

Target-agnostic arms (one fit each, evaluated for every target class):
    natural       context ~ clean TRAIN class prevalence
    sqrt_natural  context ~ sqrt(prevalence)   (imbalanced-learning dampening)
    balanced      context-size / n_classes per class
Target-specific arms (one fit per target class):
    enriched30    target 30% of context, remainder split evenly (capped by pool)
    contrastive   target balanced-share + the dominant confuser (benign) at 70%,
                  remaining classes 2.5% each -- teaches the boundary that the
                  09-18 confusion matrices showed actually fails
Post-hoc variants for every arm (no refit): the pipeline's PriorCorrector
(imported) with ref prior = clean TRAIN prior, beta in {0.5, 1.0}, T=1.

Candidate-quality proxy for a specialist: on rows where the expert disagrees
with global, H = global wrong & expert right, D = global right & expert wrong.
Reported over all rows and restricted to the target-relevant region
(true==target or expert==target or global==target).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nfv3_conflict_clean import install_clean_loader, write_json
from nfv3_v3_exp31_c0alloc import uniq_rows, full_proba, pick_rows, PriorCorrector
from tabpfn import TabPFNClassifier


def make_clf(args, Xc, yc):
    clf = TabPFNClassifier(
        device=args.device, model_path=args.model_path,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        random_state=args.seed, n_estimators=args.n_estimators,
        auto_scale_n_estimators=False, fit_mode=args.fit_mode,
        keep_cache_on_device=False)
    clf.fit(Xc, yc)
    return clf


def batched_proba(args, clf, Xr, n_classes, tag=""):
    Xu, inv = uniq_rows(Xr)
    bs = args.test_batch_size or len(Xu)
    outs = []
    for s0 in range(0, len(Xu), bs):
        outs.append(clf.predict_proba(Xu[s0:s0 + bs]))
    print(f"    [{tag}] proba {len(Xu):,} distinct rows", flush=True)
    pr = full_proba(np.concatenate(outs), clf.classes_, n_classes)
    return pr[inv].astype(np.float32)


def confusion(y, pred, n):
    return np.bincount(y.astype(np.int64) * n + pred, minlength=n * n).reshape(n, n)


def prf(cm):
    support, predicted = cm.sum(1), cm.sum(0)
    tp = np.diag(cm)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(tp)), where=support + predicted > 0)
    rec = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    prec = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    return f1, prec, rec, support


def alloc(weights, budget, avail):
    """Largest-remainder allocation of budget by weights, capped by availability."""
    w = np.asarray(weights, float); w = w / w.sum()
    raw = w * budget
    tgt = np.minimum(np.floor(raw).astype(int), avail)
    rem = budget - tgt.sum()
    for c in np.argsort(-(raw - np.floor(raw))):
        if rem <= 0: break
        if tgt[c] < avail[c]:
            tgt[c] += 1; rem -= 1
    return tgt


def build(rows_by_class, sizes, seed):
    parts = [pick_rows(rows_by_class[c], int(k), seed + c) for c, k in enumerate(sizes) if k > 0]
    return np.sort(np.concatenate(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-manifest", required=True)
    p.add_argument("--global-cache", required=True, help="EXP48 frozen_cache with eval_p0.npy / eval_y.npy")
    p.add_argument("--data", default="/home/user/Desktop/imbalcic/data/nfv3_energy_suite_uncapped_scenarios.pkl")
    p.add_argument("--targets", default="mitm,ransomware,scanning,injection")
    p.add_argument("--context-size", type=int, default=20_000)
    p.add_argument("--enriched-frac", type=float, default=0.30)
    p.add_argument("--contrastive-confuser", default="benign")
    p.add_argument("--contrastive-confuser-frac", type=float, default=0.70)
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
    n = len(names)
    targets = [t for t in args.targets.split(",") if t]
    for t in targets: assert t in names, t
    conf_id = names.index(args.contrastive_confuser)

    y_train = label_fn(train_idx)
    rows_by_class = [train_idx[y_train == c] for c in range(n)]
    avail = np.array([len(r) for r in rows_by_class])
    train_prior = avail / avail.sum()
    print("train pool: " + ", ".join(f"{nm}={a:,}" for nm, a in zip(names, avail)), flush=True)

    cache = Path(args.global_cache)
    g_ids = np.load(cache / "eval_ids.npy")
    assert np.array_equal(g_ids, np.asarray(test_idx)), "global cache eval rows != clean test rows"
    y_test = np.load(cache / "eval_y.npy")
    g_pred = np.load(cache / "eval_p0.npy", mmap_mode="r").argmax(1)
    g_right = g_pred == y_test
    g_cm = confusion(y_test, g_pred, n)
    g_f1, g_prec, g_rec, support = prf(g_cm)
    print(f"global macro_f1={g_f1.mean():.6f}", flush=True)

    out = Path(args.out_root) / (time.strftime("%Y%m%d_%H%M%S") + "_nfv3_toniot_exp54_expert_context_design")
    out.mkdir(parents=True, exist_ok=False)
    betas = [float(b) for b in args.betas.split(",")]
    B = args.context_size

    # ---- arm definitions: (arm, target or None, sizes) ----
    arms = []
    arms.append(("natural", None, alloc(avail, B, avail)))
    arms.append(("sqrt_natural", None, alloc(np.sqrt(avail), B, avail)))
    arms.append(("balanced", None, alloc(np.ones(n), B, avail)))
    for t in targets:
        tid = names.index(t)
        w = np.full(n, (1 - args.enriched_frac) / (n - 1)); w[tid] = args.enriched_frac
        arms.append(("enriched30", t, alloc(w, B, avail)))
        w = np.full(n, (1 - args.contrastive_confuser_frac - 1.0 / n) / (n - 2))
        w[tid] = 1.0 / n; w[conf_id] = args.contrastive_confuser_frac
        arms.append(("contrastive", t, alloc(w, B, avail)))

    metric_rows, quality_rows, compositions = [], [], {}
    for ai, (arm, target, sizes) in enumerate(arms):
        key = arm if target is None else f"{arm}:{target}"
        ctx = build(rows_by_class, sizes, args.seed + 1000 * ai)
        yc = label_fn(ctx)
        counts = np.bincount(yc, minlength=n)
        compositions[key] = {nm: int(c) for nm, c in zip(names, counts)}
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
            cm = confusion(y_test, pred, n)
            f1, prec, rec, _ = prf(cm)
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

    write_json(out / "COMPLETE.json", {"seconds": time.time() - started, "targets": targets, "context_size": B,
               "global_macro_f1": float(g_f1.mean()), "arms": [k for k in compositions]})
    print("WROTE: " + str(out), flush=True)


if __name__ == "__main__":
    main()

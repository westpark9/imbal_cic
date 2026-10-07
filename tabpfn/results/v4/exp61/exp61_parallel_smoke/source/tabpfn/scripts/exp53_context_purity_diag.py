#!/usr/bin/env python3
"""EXP53: does a majority-target-class context help or hurt that class's F1,
vs a class-balanced context of the same size? Bypasses the whole EXP31
pipeline (no clustering, no shared anchor, no PriorCorrector, no K-selection)
to isolate exactly one variable: context class composition. Fits two TabPFN
classifiers directly on hand-built contexts of equal total size and evaluates
both on the full clean ToN test set.

    balanced: --context-size rows split evenly across all classes
              (context-size / n_classes each).
    majority: --majority-frac of --context-size is the target class; the
              remainder is split evenly across the other classes.

Both draw from the same TRAIN split of the already-cleaned ToN data (same
conflict-free manifest as EXP48/EXP52 -- no re-cleaning). No PriorCorrector,
no anchor, no residual/embedding clustering: raw predict_proba argmax.
Frozen from EXP31's make_clf/batched_proba/full_proba/uniq_rows/pick_rows
(imported, not reimplemented) to match its exact TabPFN fit settings.
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
from nfv3_v3_exp31_c0alloc import uniq_rows, full_proba, pick_rows
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
    for bn, s0 in enumerate(range(0, len(Xu), bs), 1):
        outs.append(clf.predict_proba(Xu[s0:s0 + bs]))
        print(f"    [{tag}] proba rows {min(s0 + bs, len(Xu)):,}/{len(Xu):,}", flush=True)
    pr = full_proba(np.concatenate(outs), clf.classes_, n_classes)
    return pr[inv].astype(np.float64)


def confusion(y, pred, n_classes):
    return np.bincount(y.astype(np.int64) * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def per_class_metrics(cm):
    support, predicted = cm.sum(1), cm.sum(0)
    tp = np.diag(cm)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(tp)), where=support + predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    return f1, precision, recall, support


def build_context(rows_by_class, sizes_per_class, seed):
    chosen = []
    for c, k in enumerate(sizes_per_class):
        if k <= 0:
            continue
        chosen.append(pick_rows(rows_by_class[c], k, seed + c))
    return np.sort(np.concatenate(chosen))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-manifest", required=True)
    p.add_argument("--data", default="/home/user/Desktop/imbalcic/data/nfv3_energy_suite_uncapped_scenarios.pkl")
    p.add_argument("--target-class", default="injection")
    p.add_argument("--context-size", type=int, default=20_000)
    p.add_argument("--majority-frac", type=float, default=0.9)
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
    load_args = argparse.Namespace(data=args.data)
    X, class_names, train_idx, val_idx, test_idx, y_train, y_test, audit, label_fn = core.load_ton_iot(load_args)
    n_classes = len(class_names)
    target_id = class_names.index(args.target_class)
    print(f"classes: {class_names}, target={args.target_class} (id {target_id})", flush=True)

    y_all_train = label_fn(train_idx)
    rows_by_class = [train_idx[y_all_train == c] for c in range(n_classes)]
    counts = [len(r) for r in rows_by_class]
    print("train pool per class: " + ", ".join(f"{n}={c}" for n, c in zip(class_names, counts)), flush=True)

    balanced_each = args.context_size // n_classes
    balanced_sizes = [balanced_each] * n_classes

    majority_target = int(round(args.context_size * args.majority_frac))
    majority_rest = args.context_size - majority_target
    other = [c for c in range(n_classes) if c != target_id]
    rest_each = majority_rest // len(other)
    majority_sizes = [0] * n_classes
    majority_sizes[target_id] = majority_target
    for c in other:
        majority_sizes[c] = rest_each

    arms = {"balanced": balanced_sizes, "majority": majority_sizes}
    out = Path(args.out_root) / (time.strftime("%Y%m%d_%H%M%S") + "_nfv3_toniot_exp53_context_purity")
    out.mkdir(parents=True, exist_ok=False)

    y_true_test = label_fn(test_idx)
    rows = []
    for arm, sizes in arms.items():
        ctx_idx = build_context(rows_by_class, sizes, args.seed + (0 if arm == "balanced" else 1000))
        actual = np.bincount(label_fn(ctx_idx), minlength=n_classes)
        frac_target = actual[target_id] / actual.sum()
        print(f"[{arm}] context rows={len(ctx_idx):,}, target_class frac={frac_target:.3f}, "
              f"composition={dict(zip(class_names, actual.tolist()))}", flush=True)
        t0 = time.time()
        clf = make_clf(args, X[ctx_idx], label_fn(ctx_idx))
        fit_s = time.time() - t0
        print(f"[{arm}] fit done ({fit_s:.1f}s)", flush=True)
        t0 = time.time()
        proba = batched_proba(args, clf, X[test_idx], n_classes, tag=arm)
        pred = proba.argmax(1)
        pred_s = time.time() - t0
        cm = confusion(y_true_test, pred, n_classes)
        f1, precision, recall, support = per_class_metrics(cm)
        macro_f1 = float(f1.mean())
        print(f"[{arm}] predict done ({pred_s:.1f}s), macro_f1={macro_f1:.6f}, "
              f"target({args.target_class})_f1={f1[target_id]:.4f}", flush=True)
        for c, name in enumerate(class_names):
            rows.append({"arm": arm, "class": name, "is_target": name == args.target_class,
                        "context_rows": int(actual[c]), "context_frac": float(actual[c] / actual.sum()),
                        "support": int(support[c]), "f1": float(f1[c]), "precision": float(precision[c]),
                        "recall": float(recall[c]), "macro_f1_all_classes": macro_f1})
        np.save(out / f"{arm}_test_pred.npy", pred.astype(np.int32))
        np.save(out / f"{arm}_context_idx.npy", ctx_idx)

    table = pd.DataFrame(rows)
    table.to_csv(out / "per_class_metrics.csv", index=False)
    write_json(out / "COMPLETE.json", {"seconds": time.time() - started, "target_class": args.target_class,
               "context_size": args.context_size, "majority_frac": args.majority_frac, "seed": args.seed})
    print("WROTE: " + str(out), flush=True)


if __name__ == "__main__":
    main()

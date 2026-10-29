#!/usr/bin/env python3
"""EXP46: paired 2x2 ToN global-context comparison (no routing).

Factors: early/full train window x legacy/natural class quotas. The realized
legacy quotas are derived ONCE from the early pool, then reused in both windows;
natural quotas are derived ONCE from the full train. No relabeling/deduplication.
Thus the window contrast does not silently change the number of rare examples.

One controller prepares all four contexts and both complete evaluation splits
BEFORE reading any model result. Each arm runs in a fresh worker with identical
Python/NumPy/Torch seeds, estimator count, fit/predict order and batch sizes.
Both TabPFN and XGB-on-the-same-context are evaluated, with float32 posteriors,
original row indices and per-class metrics saved. No test-dependent selection.

Run from repository root:
  python -u tabpfn/scripts/nfv3_v3_exp46_ton_global_context.py --seed 42
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

import nfv3_v3_common as core
import nfv3_v3_c0_context as c0ctx

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("early_natural", "full_natural", "early_legacy", "full_legacy")


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temp.replace(path)


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def draw_quota(pool, labels, quota, seed):
    """Identical per-class RNG for both recipes; no sampling with replacement."""
    pool = np.asarray(pool, np.int64)
    labels = np.asarray(labels)
    quota = np.asarray(quota, np.int64)
    counts = np.bincount(labels, minlength=len(quota))
    if np.any(quota < 0) or np.any(quota > counts):
        raise ValueError(f"Quota outside pool capacity: {quota} vs {counts}")
    rows = [c0ctx.pick_rows(pool[labels == c], int(k), seed + c)
            for c, k in enumerate(quota) if k]
    selected = np.sort(np.concatenate(rows))
    if len(selected) != int(quota.sum()) or len(np.unique(selected)) != len(selected):
        raise ValueError("Context budget or row uniqueness mismatch")
    return selected


def make_contexts(train, ytrain, early, yearly, classes, budget, seed):
    C = len(classes)
    if not C <= budget <= len(early):
        raise ValueError("Budget must fit early pool and include all classes")
    draw_seed = seed + c0ctx.SEED_BAND_GLOBAL
    legacy_reference = c0ctx.representative_subset_share(
        early, yearly, C, budget, draw_seed, classes.index("benign"), .75, "balanced")
    natural_reference = c0ctx.representative_subset(train, ytrain, C, budget, draw_seed)
    lookup = pd.Series(ytrain, index=train)
    quotas = {"legacy": np.bincount(lookup.loc[legacy_reference], minlength=C),
              "natural": np.bincount(lookup.loc[natural_reference], minlength=C)}
    if any(q.sum() != budget or np.any(q == 0) for q in quotas.values()):
        raise ValueError("Requested budget cannot preserve all classes exactly")
    contexts = {}
    for arm in ARMS:
        window, recipe = arm.split("_")
        pool, labels = (early, yearly) if window == "early" else (train, ytrain)
        contexts[arm] = draw_quota(pool, labels, quotas[recipe], draw_seed)
        np.testing.assert_array_equal(np.bincount(lookup.loc[contexts[arm]], minlength=C),
                                      quotas[recipe])
    np.testing.assert_array_equal(contexts["early_legacy"], legacy_reference)
    np.testing.assert_array_equal(contexts["full_natural"], natural_reference)
    return contexts, quotas


def save_features(path, X, indices, chunk=500_000):
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                  shape=(len(indices), X.shape[1]))
    nonfinite = 0
    for s in range(0, len(indices), chunk):
        block = np.asarray(X[indices[s:s + chunk]], dtype=np.float32)
        bad = ~np.isfinite(block)
        nonfinite += int(bad.sum())
        if bad.any():
            block = np.nan_to_num(block)
        arr[s:s + len(block)] = block
    arr.flush()
    del arr
    return nonfinite


def prepare(args, run):
    cfg = core.build_dataset_config(str(ROOT / "data"))
    args.data = args.data or cfg[args.target_dataset]["default_data"]
    # This loader uses the unchanged per-scenario chronological outer split.
    X, names, train, val, test, ytrain, ytest, audit, label_fn = cfg[args.target_dataset]["loader"](args)
    suite = core.load_pickle(args.data)
    ts = np.asarray(suite["timestamps"], dtype=np.int64)
    scenarios = np.asarray(suite["attack_scenarios"])
    pools, pool_audit, scenario_audit = c0ctx.scenario_stratified_partition(
        train, ytrain, ts[train], scenarios[train], .5, .25, names)
    early = pools["context"]
    contexts, quotas = make_contexts(train, ytrain, early, label_fn(early), names,
                                     args.context_size, args.seed)
    prepared = run / "prepared"
    prepared.mkdir()
    audit.to_csv(run / "outer_split_audit.csv", index=False)
    pool_audit.to_csv(run / "pool_audit.csv", index=False)
    scenario_audit.to_csv(run / "scenario_audit.csv", index=False)
    np.save(prepared / "train_idx.npy", train)
    counts, context_rows, time_rows = [], [], []
    for split, ids, labels in [("train", train, ytrain), ("val", val, label_fn(val)),
                               ("test", test, ytest)]:
        np.save(prepared / f"{split}_idx.npy", ids)
        np.save(prepared / f"{split}_y.npy", np.asarray(labels, np.int16))
        counts.extend(dict(split=split, cls=name, rows=int((labels == c).sum()))
                      for c, name in enumerate(names))
    # Every context is fully constructed before any model is fitted/evaluated.
    time_edges = {c: np.unique(np.quantile(ts[train[ytrain == c]], np.linspace(0, 1, 11)))
                  for c in range(len(names))}
    for arm, ids in contexts.items():
        np.save(prepared / f"{arm}_idx.npy", ids)
        yc = label_fn(ids)
        np.save(prepared / f"{arm}_y.npy", yc.astype(np.int16))
        save_features(prepared / f"{arm}_X.npy", X, ids)
        from_early = np.isin(ids, early, assume_unique=True)
        for c, name in enumerate(names):
            subset = ids[yc == c]
            times = ts[subset]
            context_rows.append(dict(arm=arm, cls=name, rows=len(subset),
                from_early=int(from_early[yc == c].sum()),
                time_min=int(times.min()), time_median=float(np.median(times)),
                time_max=int(times.max())))
            bins = np.searchsorted(time_edges[c][1:-1], times, side="right")
            for b, n in zip(*np.unique(bins, return_counts=True)):
                time_rows.append(dict(arm=arm, cls=name, train_time_bin=int(b), rows=int(n)))
        print(f"PREPARED {arm}: {len(ids):,} rows hash={array_hash(ids)}", flush=True)
    pd.DataFrame(counts).to_csv(run / "class_counts.csv", index=False)
    pd.DataFrame(context_rows).to_csv(run / "context_composition.csv", index=False)
    pd.DataFrame(time_rows).to_csv(run / "context_time_bins.csv", index=False)
    nonfinite = {}
    for split, ids in [("val", val), ("test", test)]:
        nonfinite[split] = save_features(prepared / f"{split}_X.npy", X, ids)
        print(f"PREPARED {split}: {len(ids):,} rows", flush=True)
    source_path = Path(args.data)
    meta = dict(classes=names, tail_classes=cfg[args.target_dataset]["tail_classes"],
        train_rows=len(train), val_rows=len(val), test_rows=len(test),
        quotas={k: v.tolist() for k, v in quotas.items()},
        context_sha256={k: array_hash(v) for k, v in contexts.items()},
        split_sha256={"train": array_hash(train), "val": array_hash(val), "test": array_hash(test)},
        nonfinite_cells=nonfinite, data_path=str(source_path.resolve()),
        data_bytes=source_path.stat().st_size, data_mtime_ns=source_path.stat().st_mtime_ns)
    write_json(prepared / "COMPLETE.json", meta)
    del X, suite, label_fn, ts, scenarios
    core._PICKLE_CACHE.clear()
    gc.collect()
    return meta


def metrics(y, pred, names, tails):
    C = len(names)
    cm = np.bincount(y.astype(np.int64) * C + pred, minlength=C*C).reshape(C, C)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(C), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(C), where=support > 0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(C), where=support+predicted > 0)
    rows = [dict(cls=n, support=int(support[c]), predicted=int(predicted[c]),
        tp=int(tp[c]), fp=int(predicted[c]-tp[c]), fn=int(support[c]-tp[c]),
        precision=float(precision[c]), recall=float(recall[c]), f1=float(f1[c]))
        for c, n in enumerate(names)]
    summary = dict(rows=len(y), accuracy=float(tp.sum()/len(y)), macro_f1=float(f1.mean()),
        tail_f1=float(np.mean([f1[names.index(t)] for t in tails])),
        benign_fpr=float(1-recall[names.index("benign")]))
    return summary, rows, cm


def predict_save(model, X, y, names, tails, batch, out, model_name, split):
    proba = np.lib.format.open_memmap(out / f"proba_{model_name}_{split}.npy", mode="w+",
                                    dtype=np.float32, shape=(len(y), len(names)))
    pred = np.empty(len(y), dtype=np.int16)
    t0 = time.time()
    for s in range(0, len(y), batch):
        stop = min(s + batch, len(y))
        p = np.asarray(model.predict_proba(np.asarray(X[s:stop])), dtype=np.float32)
        if p.shape != (stop-s, len(names)) or not np.isfinite(p).all():
            raise ValueError("Invalid probability shape or non-finite output")
        if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1, atol=1e-4):
            raise ValueError("Invalid probability range/normalization")
        proba[s:stop] = p
        pred[s:stop] = p.argmax(1)
        proba.flush()
        write_json(out / "progress.json", dict(model=model_name, split=split, rows_done=stop,
                    rows_total=len(y), predict_seconds=time.time()-t0))
        print(f"[{model_name}/{split}] {stop:,}/{len(y):,} rows {time.time()-t0:.1f}s", flush=True)
    del proba
    np.save(out / f"pred_{model_name}_{split}.npy", pred)
    summary, per_class, cm = metrics(y, pred, names, tails)
    np.save(out / f"confusion_{model_name}_{split}.npy", cm)
    return {**summary, "predict_seconds": time.time()-t0}, per_class


def worker(run, arm):
    conf = json.loads((run / "args.json").read_text())
    expected = json.loads((run / "source_hashes.json").read_text())
    for filename, digest in expected.items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed after preregistration: {filename}")
    prepared = run / "prepared"
    meta = json.loads((prepared / "COMPLETE.json").read_text())
    names, tails = meta["classes"], meta["tail_classes"]
    out = run / arm
    out.mkdir(exist_ok=False)
    ids = np.load(prepared / f"{arm}_idx.npy")
    if array_hash(ids) != meta["context_sha256"][arm]:
        raise ValueError("Prepared context identity mismatch")
    np.save(out / "context_idx.npy", ids)
    Xc = np.load(prepared / f"{arm}_X.npy", mmap_mode="r")
    yc = np.load(prepared / f"{arm}_y.npy")
    seed = conf["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if conf["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing an unintended CPU full run")
    summaries, per_class_rows = [], []
    write_json(out / "RUNNING.json", dict(pid=os.getpid(), arm=arm, started=time.time()))
    for model_name in ["xgboost_c0", "tabpfn"]:
        t0 = time.time()
        if model_name == "xgboost_c0":
            model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.05,
                subsample=.8, colsample_bytree=.8, min_child_weight=1., reg_lambda=1.,
                objective="multi:softprob", num_class=len(names), n_jobs=conf["threads"], random_state=seed)
        else:
            from tabpfn import TabPFNClassifier
            # Reset all RNGs immediately before the PFN, independent of the XGB path.
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
            model = TabPFNClassifier(device=conf["device"], model_path=conf["model_path"],
                n_estimators=conf["n_estimators"], auto_scale_n_estimators=False,
                fit_mode="fit_with_cache", keep_cache_on_device=False,
                inference_config={"SUBSAMPLE_SAMPLES": None}, random_state=seed)
        model.fit(np.asarray(Xc), yc)
        if list(model.classes_) != list(range(len(names))):
            raise ValueError("Class output order differs from canonical order")
        fit_seconds = time.time()-t0
        print(f"{arm}/{model_name} fit {fit_seconds:.1f}s", flush=True)
        for split in ["val", "test"]:
            X = np.load(prepared / f"{split}_X.npy", mmap_mode="r")
            y = np.load(prepared / f"{split}_y.npy")
            summary, rows = predict_save(model, X, y, names, tails, conf["batch_size"],
                                         out, model_name, split)
            if model_name == "tabpfn" and conf["device"].startswith("cuda"):
                summary["gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
            tags = dict(arm=arm, window=arm.split("_")[0], quota=arm.split("_")[1],
                        model=model_name, split=split, seed=seed)
            summaries.append(dict(**tags, **summary, fit_seconds=fit_seconds))
            per_class_rows.extend(dict(**tags, **r) for r in rows)
            pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
            pd.DataFrame(per_class_rows).to_csv(out / "per_class_metrics.csv", index=False)
            print(f"RESULT {arm}/{model_name}/{split} macro={summary['macro_f1']:.6f} "
                  f"tail={summary['tail_f1']:.6f} benign_fpr={summary['benign_fpr']:.6f}", flush=True)
            del X, y
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(out / "COMPLETE.json", dict(completed=time.time(), arm=arm, context_sha256=array_hash(ids)))
    (out / "RUNNING.json").unlink()


def aggregate(run):
    for filename in ["summary.csv", "per_class_metrics.csv"]:
        frames = [pd.read_csv(run / arm / filename) for arm in ARMS
                  if (run / arm / "COMPLETE.json").exists()]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(run / filename, index=False)


def controller(args):
    run = Path(args.run_dir).resolve() if args.run_dir else ROOT / "tabpfn/results" / (
        time.strftime("%Y%m%d_%H%M%S") + f"_nfv3_toniot_exp46_global_context_s{args.seed}")
    run.mkdir(parents=True, exist_ok=False)
    print(f"RUN_DIR: {run}", flush=True)
    config = vars(args).copy()
    config.update(run_dir=str(run), started=time.time(), arms=list(ARMS),
                  protocol="fixed 2x2, full validation/test; no policy tuning or routing",
                  torch_version=torch.__version__, xgboost_version=xgb.__version__,
                  python=sys.version, python_executable=sys.executable)
    write_json(run / "args.json", config)
    filenames = [str(Path(__file__).resolve().relative_to(ROOT)),
                 "tabpfn/scripts/nfv3_v3_common.py", "tabpfn/scripts/nfv3_v3_c0_context.py"]
    hashes = {}
    (run / "source").mkdir()
    for filename in filenames:
        source = ROOT / filename
        hashes[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, run / "source" / source.name)
    write_json(run / "source_hashes.json", hashes)
    write_json(run / "RUNNING.json", dict(pid=os.getpid(), phase="preparing", started=time.time()))
    try:
        prepare(args, run)
        config["data"] = args.data
        write_json(run / "args.json", config)
        for arm in ARMS:
            with (run / f"{arm}.log").open("w", buffering=1) as log:
                env = os.environ.copy()
                env["PYTHONHASHSEED"] = str(args.seed)
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                    "--worker-run", str(run), "--worker-arm", arm], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT)
                write_json(run / "RUNNING.json", dict(pid=os.getpid(), worker_pid=child.pid,
                           arm=arm, phase="evaluating", started=config["started"]))
                print(f"STARTED {arm} PID={child.pid} log={run / (arm + '.log')}", flush=True)
                code = child.wait()
            if code != 0:
                raise RuntimeError(f"{arm} failed with exit {code}; see {arm}.log")
            aggregate(run)
            print(f"COMPLETED {arm}", flush=True)
        write_json(run / "COMPLETE.json", dict(completed=time.time(), arms=list(ARMS), seed=args.seed))
        (run / "RUNNING.json").unlink()
        print(f"ALL COMPLETE: {run}", flush=True)
    except BaseException:
        write_json(run / "ERROR.json", dict(time=time.time(), traceback=traceback.format_exc()))
        (run / "RUNNING.json").unlink(missing_ok=True)
        raise


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-size", type=int, default=100_000)
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=500_000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-dataset", default="ton_iot", choices=["ton_iot", "ton_iot_capped"])
    p.add_argument("--data", default=None)
    p.add_argument("--model-path", default=str(ROOT / "tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"))
    p.add_argument("--run-dir", default=None)
    p.add_argument("--worker-run", default=None)
    p.add_argument("--worker-arm", choices=ARMS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker_run:
        if not args.worker_arm:
            raise SystemExit("--worker-arm is required")
        worker(Path(args.worker_run), args.worker_arm)
    else:
        if args.batch_size <= 0:
            raise SystemExit("--batch-size must be positive")
        lockpath = ROOT / "tabpfn/logs/exp46_gpu.lock"
        lockpath.parent.mkdir(parents=True, exist_ok=True)
        with lockpath.open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("Another EXP46 controller is already running")
            controller(args)

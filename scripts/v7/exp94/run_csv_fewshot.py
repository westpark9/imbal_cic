"""EXP94 CSV side: CIC2017 CICFlowMeter CSV few-shot curves, audits and oracle ceiling.

Companion of ``scripts/v7/exp93/run_pcap_benchmark.py``.  The CSV is the
official MachineLearningCVE release (78 statistics + Label, no IDs or
timestamps), split chronologically within each file; the complete test split
is scored.  XGBoost and frozen TabPFN receive exactly the same rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv as csvmod
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(ROOT / 'scripts/common'))
from experiment_paths import resolve_path

import numpy as np
import pandas as pd


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def map_label(raw: str, label_map: dict[str, str]) -> str | None:
    raw = raw.strip()
    if raw in label_map:
        return label_map[raw]
    for key, family in label_map.items():
        if '*' in key:
            head, tail = key.split('*', 1)
            if raw.startswith(head.strip()) and raw.endswith(tail.strip()):
                return family
    return None


def load_table(config: dict[str, Any], out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Return X float32, y family index (-1 unmapped), split, names and a manifest."""
    cache = out / 'dataset.npz'
    manifest_path = out / 'dataset_manifest.json'
    names = list(config['class_order'])
    if cache.exists() and manifest_path.exists():
        with np.load(cache, allow_pickle=False) as z:
            return z['X'], z['y'], z['split'], names, read(manifest_path)
    csv_dir = resolve_path(config['csv_dir'], ROOT)
    frames, per_file = [], []
    feature_columns: list[str] | None = None
    fractions = config['split']['fractions']
    for index, name in enumerate(config['file_order']):
        path = csv_dir / name
        df = pd.read_csv(path, encoding='latin-1', skipinitialspace=True, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        # The release repeats 'Fwd Header Length'; keep the first copy.
        df = df.loc[:, ~df.columns.duplicated()]
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != 'Label']
        if [c for c in df.columns if c != 'Label'] != feature_columns:
            raise ValueError(f'Column mismatch in {name}')
        raw_labels = df['Label'].astype(str)
        fam = raw_labels.map(lambda v: map_label(v, config['label_map']))
        if fam.isna().any():
            raise ValueError(f'Unmapped labels in {name}: {sorted(set(raw_labels[fam.isna()]))[:10]}')
        X = df[feature_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
        finite = np.isfinite(X).all(axis=1)
        dropped = Counter(fam[~finite].tolist())
        X, fam, raw_labels = X[finite], fam[finite].to_numpy(), raw_labels[finite].to_numpy()
        n = len(X)
        y = np.asarray([names.index(f) for f in fam], dtype=np.int16)
        # Chronological within each (file, family): rows keep file order, so
        # the first 70% of a family's rows precede its validation and test
        # rows in time.  Splitting the whole file by position would leave
        # families whose attack window ends early (web, Hulk) without test rows.
        split = np.empty(n, dtype=object)
        # 'chronological_within_file_and_label' orders by the raw CIC label so
        # that sub-types sharing a family (FTP- then SSH-Patator) each keep
        # train and test rows; the family variant can leave a sub-type entirely
        # in train when its window precedes the other sub-type's window.
        by_label = config['split'].get('method') == 'chronological_within_file_and_label'
        key = raw_labels if by_label else y
        for c in np.unique(key):
            idx = np.flatnonzero(key == c)
            a, b = int(len(idx) * fractions[0]), int(len(idx) * (fractions[0] + fractions[1]))
            split[idx[:a]] = 'train'; split[idx[a:b]] = 'validation'; split[idx[b:]] = 'test'
        split = split.astype(str)
        a, b = int((split == 'train').sum()), int((split != 'test').sum())
        frames.append((X.astype(np.float32), y, split, np.full(n, index, dtype=np.int8)))
        per_file.append(dict(file=name, rows_read=int(len(finite)), rows_kept=n, dropped_nonfinite=dict(dropped),
                             split_rows=dict(train=a, validation=b - a, test=n - b),
                             labels=dict(Counter(raw_labels.tolist())), families=dict(Counter(fam.tolist()))))
        print(f'loaded {name}: {n:,} rows kept, {int((~finite).sum()):,} non-finite dropped', flush=True)
    X = np.concatenate([f[0] for f in frames]); y = np.concatenate([f[1] for f in frames])
    split = np.concatenate([f[2] for f in frames]); file_index = np.concatenate([f[3] for f in frames])
    np.savez_compressed(cache, X=X, y=y, split=split, file_index=file_index)
    manifest = dict(schema=1, dataset=config['dataset'], feature_columns=feature_columns, feature_count=len(feature_columns),
                    rows=int(len(X)), class_order=names, per_file=per_file,
                    split_counts={part: {name: int(((split == part) & (y == i)).sum()) for i, name in enumerate(names)}
                                  for part in ('train', 'validation', 'test')},
                    dataset_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                    split=config['split'])
    write(manifest_path, manifest)
    return X, y, split, names, manifest


def coverage(config: dict[str, Any], y: np.ndarray, split: np.ndarray, names: list[str], out: Path) -> list[int]:
    max_budget = max(int(v) for v in config['fewshot']['budgets'])
    rows, included, excluded = {}, [], []
    for i, name in enumerate(names):
        tr, te = int(((split == 'train') & (y == i)).sum()), int(((split == 'test') & (y == i)).sum())
        rows[name] = dict(train=tr, test=te)
        (included if tr >= max_budget and te >= 1 else excluded).append(i)
    write(out / 'label_coverage.json', dict(max_fewshot_budget=max_budget, coverage=rows,
                                            included=[names[i] for i in included], excluded=[names[i] for i in excluded],
                                            rule=config['fewshot']['coverage_rule']))
    return included


def _f1(y: np.ndarray, pred: np.ndarray, classes: list[int], names: list[str]) -> dict[str, Any]:
    values = []
    for c in classes:
        tp = int(np.sum((y == c) & (pred == c))); fp = int(np.sum((y != c) & (pred == c))); fn = int(np.sum((y == c) & (pred != c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        values.append(dict(name=names[c], support=int(np.sum(y == c)), precision=precision, recall=recall, f1=f, tp=tp, fp=fp, fn=fn))
    return dict(macro_f1=float(sum(v['f1'] for v in values) / len(values)), classes=values, rows=int(len(y)))


def audits(config: dict[str, Any], X: np.ndarray, y: np.ndarray, split: np.ndarray, names: list[str],
           included: list[int], out: Path) -> dict[str, Any]:
    digests = np.array([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in X])
    def cross_label(mask: np.ndarray) -> dict[str, Any]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i in np.flatnonzero(mask):
            groups[digests[i]].append(i)
        conflicts = [(d, m) for d, m in groups.items() if len({int(y[i]) for i in m}) > 1]
        by_family: Counter = Counter()
        for _, m in conflicts:
            for i in m:
                by_family[names[int(y[i])]] += 1
        examples = sorted(conflicts, key=lambda item: -len(item[1]))[:50]
        return dict(rows=int(mask.sum()), identical_groups=sum(len(m) > 1 for m in groups.values()),
                    cross_label_groups=len(conflicts), cross_label_rows=sum(len(m) for _, m in conflicts),
                    cross_label_rows_by_family={k: int(v) for k, v in by_family.items()},
                    examples=[dict(sha256=d, rows=len(m), labels=dict(Counter(names[int(y[i])] for i in m))) for d, m in examples])
    train_set = set(digests[split == 'train'].tolist())
    test_mask = split == 'test'
    dup = np.array([d in train_set for d in digests[test_mask]])
    dup_by_family = {names[c]: float(dup[y[test_mask] == c].mean()) if (y[test_mask] == c).any() else None for c in range(len(names))}
    # Oracle ceiling: majority family within each identical-feature group of the test split.
    test_idx = np.flatnonzero(test_mask)
    groups: dict[str, list[int]] = defaultdict(list)
    for i in test_idx:
        groups[digests[i]].append(i)
    oracle = np.empty(len(test_idx), dtype=np.int16)
    pos = {int(i): k for k, i in enumerate(test_idx)}
    for members in groups.values():
        maj = Counter(int(y[i]) for i in members).most_common(1)[0][0]
        for i in members:
            oracle[pos[int(i)]] = maj
    ceilings = {}
    for regime in config['fewshot']['regimes']:
        classes = included if regime == 'with_benign' else [c for c in included if names[c] != 'benign']
        m = np.isin(y[test_idx], classes)
        ceilings[regime] = _f1(y[test_idx][m], oracle[m], classes, names)
    result = dict(schema=1, full_table=cross_label(np.ones(len(X), dtype=bool)), test_split=cross_label(test_mask),
                  train_to_test_exact_duplicate_fraction=float(dup.mean()), train_to_test_exact_duplicate_by_family=dup_by_family,
                  oracle_ceiling=dict(definition=config['audits']['oracle_ceiling'],
                                      note='Only classes included by the coverage rule are scored; macro over those classes.',
                                      **{k: dict(macro_f1=v['macro_f1'], classes=[(c['name'], c['f1']) for c in v['classes']]) for k, v in ceilings.items()}))
    write(out / 'audit.json', result)
    return result


def _timed(call):
    import torch
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.perf_counter(); result = call()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return result, time.perf_counter() - start


def _predict_proba_batched(model: Any, X: np.ndarray, rows_per_batch: int) -> np.ndarray:
    if len(X) <= rows_per_batch:
        return model.predict_proba(X)
    return np.concatenate([model.predict_proba(X[s:s + rows_per_batch]) for s in range(0, len(X), rows_per_batch)], axis=0)


def _weights_digest(model: Any) -> str:
    import torch
    h = hashlib.sha256()
    for base in model.models_:
        for name, value in sorted(base.state_dict().items()):
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _make_model(method: str, seed: int, n_classes: int, config: dict[str, Any]) -> Any:
    if method == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                             min_child_weight=1, reg_lambda=1, objective='multi:softprob', num_class=n_classes,
                             eval_metric='mlogloss', tree_method='hist', device='cuda:0', n_jobs=1, random_state=seed)
    if method == 'tabpfn':
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device='cuda:0', model_path=str(resolve_path(config['tabpfn_checkpoint'], ROOT)),
                                random_state=seed, n_estimators=4, auto_scale_n_estimators=False,
                                fit_mode='fit_with_cache', keep_cache_on_device=False, n_preprocessing_jobs=1,
                                inference_config={'SUBSAMPLE_SAMPLES': None})
    raise ValueError(method)


def fewshot(config: dict[str, Any], X: np.ndarray, y: np.ndarray, split: np.ndarray, names: list[str],
            included: list[int], out: Path) -> dict[str, Any]:
    eval_dir = out / 'fewshot'
    eval_dir.mkdir(parents=True, exist_ok=True)
    fs = config['fewshot']
    budgets, seeds, max_budget = fs['budgets'], fs['seeds'], max(fs['budgets'])
    train, test = split == 'train', split == 'test'
    batch_rows = int(fs.get('inference_batch_rows', int(test.sum())))
    results = []
    for regime in fs['regimes']:
        classes = included if regime == 'with_benign' else [c for c in included if names[c] != 'benign']
        test_mask = test & np.isin(y, classes)
        x_test, y_test = X[test_mask], y[test_mask]
        for seed in seeds:
            selected = {c: np.random.default_rng(seed + 1009 * c).choice(np.flatnonzero(train & (y == c)), max_budget, replace=False)
                        for c in classes}
            for budget in budgets:
                chosen = np.concatenate([selected[c][:budget] for c in classes])
                local = {c: i for i, c in enumerate(classes)}
                y_train = np.asarray([local[int(v)] for v in y[chosen]], dtype=np.int64)
                for method in fs['methods']:
                    tag = f'{regime}_s{seed}_n{budget}_{method}'
                    job = eval_dir / 'jobs' / tag
                    done = job / 'result.json'
                    if done.exists():
                        results.append(read(done)); continue
                    job.mkdir(parents=True, exist_ok=True)
                    model, construct = _timed(lambda: _make_model(method, seed, len(classes), fs))
                    _, fit = _timed(lambda: model.fit(X[chosen], y_train))
                    frozen = _weights_digest(model) if method == 'tabpfn' else None
                    proba, infer = _timed(lambda: _predict_proba_batched(model, x_test, batch_rows))
                    if proba.shape != (len(x_test), len(classes)) or not np.isfinite(proba).all():
                        raise RuntimeError('Invalid probability matrix')
                    if frozen and frozen != _weights_digest(model):
                        raise RuntimeError('TabPFN weights changed')
                    pred = np.asarray(classes, dtype=np.int16)[proba.argmax(1)]
                    row = dict(regime=regime, seed=seed, budget=budget, method=method, train_rows=int(len(chosen)),
                               train_row_ids=chosen.tolist(), test_rows=int(len(x_test)),
                               metrics=_f1(y_test, pred, classes, names), construct_seconds=construct, fit_seconds=fit,
                               inference_seconds=infer, frozen_weights_sha256=frozen, feature_count=int(X.shape[1]),
                               inference_batch_rows=batch_rows, finetuning=False)
                    np.save(job / 'predictions.npy', pred)
                    write(done, row); results.append(row)
                    print(f'{tag}: macro-F1 {row["metrics"]["macro_f1"]:.4f} (fit {fit:.1f}s, infer {infer:.1f}s)', flush=True)
                    del model; gc.collect()
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
    rows = [dict(regime=r['regime'], seed=r['seed'], budget=r['budget'], method=r['method'], macro_f1=r['metrics']['macro_f1'],
                 train_rows=r['train_rows'], test_rows=r['test_rows'], fit_seconds=r['fit_seconds'], inference_seconds=r['inference_seconds'])
            for r in results]
    with (eval_dir / 'summary.csv').open('w', newline='') as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['regime'], r['budget'], r['seed'], r['method'])))
    cells: dict[tuple, list[float]] = defaultdict(list)
    class_cells: dict[tuple, list[float]] = defaultdict(list)
    for r in results:
        cells[(r['regime'], r['method'], r['budget'])].append(r['metrics']['macro_f1'])
        for c in r['metrics']['classes']:
            class_cells[(r['regime'], r['method'], r['budget'], c['name'])].append(c['f1'])
    aggregate = [dict(regime=k[0], method=k[1], budget=k[2], seeds=len(v), macro_f1_mean=float(np.mean(v)),
                      macro_f1_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in sorted(cells.items())]
    class_aggregate = [dict(regime=k[0], method=k[1], budget=k[2], name=k[3], f1_mean=float(np.mean(v))) for k, v in sorted(class_cells.items())]
    manifest = dict(completed_jobs=len(results), expected_jobs=len(fs['regimes']) * len(seeds) * len(budgets) * len(fs['methods']),
                    classes_scored=[names[c] for c in included], aggregate=aggregate, class_aggregate=class_aggregate)
    write(eval_dir / 'manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['data', 'audit', 'fewshot', 'all'], default='all')
    args = parser.parse_args()
    config = read(resolve_path(args.protocol, ROOT))
    out = resolve_path(args.output, ROOT); out.mkdir(parents=True, exist_ok=True)
    write(out / 'protocol_snapshot.json', config)
    X, y, split, names, _ = load_table(config, out)
    included = coverage(config, y, split, names, out)
    if args.stage in ('audit', 'all'):
        audits(config, X, y, split, names, included, out)
    if args.stage in ('fewshot', 'all'):
        fewshot(config, X, y, split, names, included, out)
    write(out / 'status.json', dict(status='complete', stage=args.stage))


if __name__ == '__main__':
    main()

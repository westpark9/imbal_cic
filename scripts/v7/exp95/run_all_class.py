"""EXP95: nine-class few-shot and the two side columns, with residual flows deleted."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
for rel in ('scripts/common', 'scripts/v7/exp92', 'scripts/v7/exp93', 'scripts/v7/exp94', 'scripts/v7/exp95'):
    sys.path.insert(0, str(ROOT / rel))
from experiment_paths import resolve_path
from run_boundary_context import base_run, context_features, snapshot, stage_base
from run_drop_paired import paired_rows
from run_pcap_benchmark import _f1, _make_model, _predict_proba_batched, _timed, _weights_digest, read, write

import dpkt
import numpy as np


def scan_capture(path: Path, wanted: np.ndarray, out_file: Path) -> None:
    """Raw seq and ack of the given 1-based record numbers."""
    want = set(wanted.tolist()); top = int(wanted.max()); found = []
    with open(path, 'rb') as f:
        magic = f.read(4); f.seek(0)
        reader = dpkt.pcapng.Reader(f) if magic == b'\x0a\x0d\x0d\x0a' else dpkt.pcap.Reader(f)
        for i, (_, raw) in enumerate(reader, 1):
            if i in want:
                o = 14; et = (raw[12] << 8) | raw[13]
                if et == 0x8100:
                    o = 18
                ihl = (raw[o] & 15) * 4; p = o + ihl
                if raw[o + 9] == 6 and len(raw) >= p + 12:
                    found.append((i, int.from_bytes(raw[p + 4:p + 8], 'big'), int.from_bytes(raw[p + 8:p + 12], 'big')))
            if i >= top:
                break
    np.save(out_file, np.asarray(found, dtype=np.int64))


def prepare(cfg: dict[str, Any], out: Path, config_path: str) -> None:
    done = out / 'prepared.json'
    if done.exists():
        return
    stage_base(cfg, out)
    a = dict(np.load(out / 'base/arrays.npz')); tail = np.load(out / 'base/residual.npz')['tail']
    meta = read(out / 'base/meta.json'); caps = meta['captures']; snap = snapshot(cfg)
    N = len(tail); classes = cfg['classes']
    index = {c: i for i, c in enumerate(classes)}
    y9 = np.full(N, -1, np.int16)
    pattern = re.compile(r'"label":"?([a-z]+)"?,"annotation_label"')
    with (base_run(cfg) / 'corpus/flows.jsonl').open() as f:
        for i, line in enumerate(f):
            m = pattern.search(line)
            if m and m.group(1) in index:
                y9[i] = index[m.group(1)]
    base = a['y']
    if not (y9[base >= 0] == base[base >= 0]).all():
        raise RuntimeError('class order differs from EXP94')
    (out / 'scan').mkdir(exist_ok=True)
    procs = []
    for cap in snap['captures']:
        c = caps[cap['id']]
        np.save(out / 'scan' / f"{cap['id']}.wanted.npy", np.unique(a['first_idx'][(a['cap'] == c) & (a['proto'] == 6)]))
        procs.append(subprocess.Popen([sys.executable, __file__, '--config', config_path, '--output', str(out), '--stage', 'scan', '--capture', cap['id']]))
    if any(p.wait() != 0 for p in procs):
        raise RuntimeError('a capture scan failed')
    seq = np.full(N, -1, np.int64); ack = np.full(N, -1, np.int64)
    for cap in snap['captures']:
        rows = np.flatnonzero((a['cap'] == caps[cap['id']]) & (a['proto'] == 6))
        got = np.load(out / 'scan' / f"{cap['id']}.fields.npy")
        order = np.argsort(got[:, 0]); pos = np.searchsorted(got[order, 0], a['first_idx'][rows])
        if (pos >= len(got)).any() or (got[order, 0][pos] != a['first_idx'][rows]).any():
            raise RuntimeError(f"missing raw fields in {cap['id']}")
        seq[rows] = got[order, 1][pos]; ack[rows] = got[order, 2][pos]
    idx = np.flatnonzero(~tail)
    fail = ((a['frst'] + a['brst']) > 0) | ((a['fpay'] + a['bpay']) == 0)
    t0 = time.time()
    ctx = context_features(a, idx, a['en'][idx], fail[idx], a['L2'][idx], 60.0)
    from run_boundary_context import CTX_BASE
    ports = ctx[:, CTX_BASE.index('pair_ports')]
    side = np.column_stack([((a['proto'] == 6) & (seq == 0))[idx], ((a['proto'] == 6) & (ack == 0))[idx], ports]).astype(np.float32)
    np.savez(out / 'side.npz', X=side, names=np.asarray(['first_seq_zero', 'first_ack_zero', 'pair_ports_60']), row=idx)
    np.save(out / 'y9.npy', y9)
    write(done, dict(rows=len(idx), class_counts={c: int((y9[idx] == i).sum()) for c, i in index.items()},
                     test_counts={c: int(((y9[idx] == i) & (a['split'][idx] == 'test')).sum()) for c, i in index.items()},
                     train_counts={c: int(((y9[idx] == i) & (a['split'][idx] == 'train')).sum()) for c, i in index.items()},
                     first_seq_zero_rows=int(side[:, 0].sum()), first_ack_zero_rows=int(side[:, 1].sum()), context_seconds=round(time.time() - t0)))
    print('prepare: done', flush=True)


def arm_data(cfg: dict[str, Any], out: Path, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    if arm == 'csv':
        with np.load(resolve_path(cfg['csv_run'], ROOT) / 'dataset.npz') as z:
            return z['X'], z['y'].astype(np.int16), z['split'], None
    a = np.load(out / 'base/arrays.npz'); tail = np.load(out / 'base/residual.npz')['tail']
    idx = np.flatnonzero(~tail)
    with np.load(base_run(cfg) / 'encoders/netfound-v2-small/dataset.npz') as z:
        X = z['X'][idx]
    y9 = np.load(out / 'y9.npy')[idx]; split = a['split'][idx]; ids = a['flow_id'][idx]
    with np.load(out / 'side.npz') as z:
        side, names = z['X'], [str(v) for v in z['names']]
    cols = {'netfound': [], 'netfound+seq': ['first_seq_zero', 'first_ack_zero'], 'netfound+ports': ['pair_ports_60']}[arm]
    if cols:
        X = np.column_stack([X, side[:, [names.index(c) for c in cols]]])
    return X.astype(np.float32), y9, split, dict(ids=ids)


def worker(cfg: dict[str, Any], out: Path, arm: str) -> None:
    few = snapshot(cfg)['fewshot']; src = base_run(cfg); names = cfg['classes']
    X, y, split, extra = arm_data(cfg, out, arm)
    classes = list(range(len(names))); max_budget = max(few['budgets'])
    test = np.flatnonzero((split == 'test') & (y >= 0)); val = np.flatnonzero((split == 'validation') & (y >= 0))
    where = {str(v): i for i, v in enumerate(extra['ids'])} if extra else None
    for seed in few['seeds']:
        selected: dict[int, np.ndarray] = {}
        if extra:
            old = read(src / 'evaluations/netfound-v2-small/fewshot/jobs' / f'with_benign_s{seed}_n{max_budget}_xgb' / 'result.json')['train_flow_ids']
            selected, _ = paired_rows(old, classes[:7], max_budget, seed, y, split, where)
        for c in classes:
            if c not in selected:
                pool = np.flatnonzero((split == 'train') & (y == c))
                selected[c] = np.random.default_rng(seed + 1009 * c).choice(pool, min(max_budget, len(pool)), replace=False)
        for budget in few['budgets']:
            chosen = np.concatenate([selected[c][:budget] for c in classes])
            y_train = y[chosen].astype(np.int64)
            for method in few['methods']:
                job = out / 'evaluations' / arm / 'jobs' / f's{seed}_n{budget}_{method}'
                if (job / 'result.json').exists():
                    continue
                job.mkdir(parents=True, exist_ok=True)
                model, construct = _timed(lambda: _make_model(method, seed, classes, few))
                _, fit = _timed(lambda: model.fit(X[chosen], y_train))
                frozen = _weights_digest(model) if method == 'tabpfn' else None
                batch = int(few['inference_batch_rows'])
                proba, infer = _timed(lambda: _predict_proba_batched(model, X[test], batch))
                if proba.shape != (len(test), len(classes)) or not np.isfinite(proba).all():
                    raise RuntimeError('invalid probability matrix')
                pred = proba.argmax(1).astype(np.int16)
                row = dict(arm=arm, seed=seed, budget=budget, method=method, train_rows=len(chosen), train_per_class={names[c]: int((y_train == c).sum()) for c in classes},
                           test_rows=len(test), construct_seconds=construct, fit_seconds=fit, inference_seconds=infer, frozen_weights_sha256=frozen)
                if budget in cfg['validation_budgets']:
                    pv = np.asarray(_predict_proba_batched(model, X[val], batch)).argmax(1).astype(np.int16)
                    row['validation_macro9'] = _f1(y[val], pv, classes, names)['macro_f1']
                if frozen and frozen != _weights_digest(model):
                    raise RuntimeError('TabPFN backbone weights changed during inference')
                r = _f1(y[test], pred, classes, names)
                row.update(macro9=r['macro_f1'], macro7=float(np.mean([c['f1'] for c in r['classes'][:7]])), classes={c['name']: {k: c[k] for k in ('f1', 'precision', 'recall', 'support', 'tp', 'fp', 'fn')} for c in r['classes']})
                ben = y[test] == 0
                row['benign_alarm'] = float((pred[ben] != 0).mean())
                np.save(job / 'predictions.npy', pred); write(job / 'result.json', row)
                del model; gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print('worker: done', arm, flush=True)


def report(cfg: dict[str, Any], out: Path) -> None:
    few = snapshot(cfg)['fewshot']; names = cfg['classes']
    rows = [read(p) for p in (out / 'evaluations').glob('*/jobs/*/result.json')]
    expected = len(cfg['arms']) * len(few['seeds']) * len(few['budgets']) * len(few['methods'])
    if len(rows) != expected:
        raise RuntimeError(f'incomplete jobs: {len(rows)}/{expected}')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['arm'], r['method'], r['budget'])].append(r)
    base = {(r['method'], r['budget'], r['seed']): r for r in rows if r['arm'] == 'netfound'}
    table = []
    for (arm, method, budget), g in sorted(groups.items()):
        item = dict(arm=arm, method=method, budget=budget, seeds=len(g), macro9=float(np.mean([x['macro9'] for x in g])), macro9_sd=float(np.std([x['macro9'] for x in g], ddof=1)),
                    macro7=float(np.mean([x['macro7'] for x in g])), benign_alarm=float(np.mean([x['benign_alarm'] for x in g])), test_rows=g[0]['test_rows'], train_per_class=g[0]['train_per_class'],
                    class_f1={c: float(np.mean([x['classes'][c]['f1'] for x in g])) for c in names}, class_recall={c: float(np.mean([x['classes'][c]['recall'] for x in g])) for c in names},
                    class_precision={c: float(np.mean([x['classes'][c]['precision'] for x in g])) for c in names})
        if 'validation_macro9' in g[0]:
            item['validation_macro9'] = float(np.mean([x['validation_macro9'] for x in g]))
        if arm.startswith('netfound') and arm != 'netfound':
            d = np.asarray([x['macro9'] - base[(method, budget, x['seed'])]['macro9'] for x in g])
            item.update(gain_vs_netfound=float(d.mean()), gain_sd=float(d.std(ddof=1)), seeds_improved=int((d > 0).sum()))
            if 'validation_macro9' in g[0]:
                item['validation_gain'] = float(np.mean([x['validation_macro9'] - base[(method, budget, x['seed'])]['validation_macro9'] for x in g]))
        table.append(item)
    write(out / 'comparison.json', dict(jobs=len(rows), prepared=read(out / 'prepared.json'), table=table))
    print('report: jobs', len(rows), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--stage', default='all', choices=['all', 'prepare', 'scan', 'worker', 'report']); ap.add_argument('--arm'); ap.add_argument('--capture')
    args = ap.parse_args()
    cfg = read(resolve_path(args.config, ROOT)); out = resolve_path(args.output, ROOT); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'scan':
        cap = next(c for c in snapshot(cfg)['captures'] if c['id'] == args.capture)
        return scan_capture(resolve_path(cap['path'], ROOT), np.load(out / 'scan' / f'{args.capture}.wanted.npy'), out / 'scan' / f'{args.capture}.fields.npy')
    if args.stage == 'worker':
        return worker(cfg, out, args.arm)
    if args.stage in ('all', 'prepare'):
        write(out / 'protocol_snapshot.json', cfg)
        prepare(cfg, out, args.config)
    if args.stage == 'prepare':
        return
    if args.stage == 'all':
        procs = []
        for arm in cfg['arms']:
            log = (out / f"worker_{arm.replace('+', '_')}.log").open('a')
            procs.append(subprocess.Popen([sys.executable, __file__, '--config', args.config, '--output', str(out), '--stage', 'worker', '--arm', arm], stdout=log, stderr=subprocess.STDOUT))
        if any(p.wait() != 0 for p in procs):
            raise RuntimeError('a worker failed')
    report(cfg, out)
    write(out / 'status.json', dict(status='complete', stage=args.stage))


if __name__ == '__main__':
    main()

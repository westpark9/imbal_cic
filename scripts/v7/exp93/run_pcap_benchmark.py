"""Reusable CSE-CIC-IDS2018 PCAP corpus, encoder views, and paired few-shot tests.

The corpus stage is intentionally independent of an encoder.  It preserves the
capture checksum, flow/group IDs, packet indices retained per flow and the
bounded packet-header sequence.  A later adapter can reopen the verified PCAP
from those packet indices and write another encoder view without altering the
labelled corpus, split, or few-shot selections.

Implemented adapter: ``netfound-v2-small``.  A different adapter must first
write a compatible view, then this runner can project, audit and evaluate it;
it is never silently substituted for netFound.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(ROOT / 'scripts/common'))
sys.path.insert(0, str(ROOT / 'scripts/v7/exp92'))
from experiment_paths import resolve_path
from fetch_assets import download_member, sha256
from flows import iter_flows
from netfound_encoder import NetFoundEncoder


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def split_group(group: str, seed: int, fractions: list[float]) -> str:
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if value < fractions[0] else 'validation' if value < sum(fractions[:2]) else 'test'


class CorrectedRuleSet:
    """CNS2022-derived, explicit rules adapted to the preserved PCAP flow fields.

    The upstream notebook labelled corrected CICFlowMeter rows.  This class
    records the same time/end-point/payload intent for our independently
    reconstructed sessions; it never claims row-for-row identity with that
    different flow extractor.
    """
    def __init__(self, capture: dict[str, Any], default_source: str):
        self.capture = capture
        self.rules = sorted(capture.get('rules', []), key=lambda r: (-int(r.get('priority', 0)), r['id']))
        self.default_source = default_source
        self.background = capture.get('unmatched_policy')

    @staticmethod
    def _endpoint_match(row: dict[str, Any], rule: dict[str, Any]) -> tuple[bool, bool]:
        """Return (matched, reverse), where reverse locates the rule source."""
        any_ips = set(rule.get('any_endpoint_ips', []))
        if any_ips:
            return bool({row['src_ip'], row['dst_ip']} & any_ips), False

        srcs, dsts = set(rule.get('src_ips', [])), set(rule.get('dst_ips', []))
        sports, dports = set(rule.get('src_ports', [])), set(rule.get('dst_ports', []))
        def matches(reverse: bool) -> bool:
            src_ip = row['dst_ip'] if reverse else row['src_ip']
            dst_ip = row['src_ip'] if reverse else row['dst_ip']
            src_port = row['dst_port'] if reverse else row['src_port']
            dst_port = row['src_port'] if reverse else row['dst_port']
            return ((not srcs or src_ip in srcs) and (not dsts or dst_ip in dsts)
                    and (not sports or src_port in sports) and (not dports or dst_port in dports))
        if matches(False):
            return True, False
        if rule.get('bidirectional', True) and matches(True):
            return True, True
        return False, False

    @staticmethod
    def _condition(row: dict[str, Any], rule: dict[str, Any], reverse: bool) -> bool:
        direction = 1 if reverse else 0
        prefix = 'bwd_' if direction else 'fwd_'
        attack_payload = row[prefix + 'payload_bytes']
        peer_payload = row[('fwd_' if direction else 'bwd_') + 'payload_bytes']
        attack_packets = row[prefix + 'packets']
        attack_rst = row.get(prefix + 'rst_packets', 0)
        peer_rst = row.get(('fwd_' if direction else 'bwd_') + 'rst_packets', 0)
        attack_src_port = row['dst_port'] if reverse else row['src_port']
        c = rule
        if 'protocol' in c and row['protocol'] != c['protocol']:
            return False
        if c.get('min_attack_payload_bytes') is not None and attack_payload < c['min_attack_payload_bytes']:
            return False
        if c.get('max_attack_payload_bytes') is not None and attack_payload > c['max_attack_payload_bytes']:
            return False
        if c.get('min_peer_payload_bytes') is not None and peer_payload < c['min_peer_payload_bytes']:
            return False
        total_payload = row['fwd_payload_bytes'] + row['bwd_payload_bytes']
        if c.get('min_total_payload_bytes') is not None and total_payload < c['min_total_payload_bytes']:
            return False
        if c.get('max_total_payload_bytes') is not None and total_payload > c['max_total_payload_bytes']:
            return False
        if c.get('min_attack_packets') is not None and attack_packets < c['min_attack_packets']:
            return False
        if c.get('max_attack_packets') is not None and attack_packets > c['max_attack_packets']:
            return False
        if c.get('min_duration_us') is not None and row['flow_duration'] < c['min_duration_us']:
            return False
        if c.get('attack_rst_eq') is not None and attack_rst != c['attack_rst_eq']:
            return False
        if c.get('peer_rst_eq') is not None and peer_rst != c['peer_rst_eq']:
            return False
        if attack_src_port in set(c.get('exclude_attack_src_ports', [])):
            return False
        alternatives = c.get('any_of', [])
        if alternatives:
            def one(part: dict[str, Any]) -> bool:
                return CorrectedRuleSet._condition(row, part, reverse)
            if not any(one(part) for part in alternatives):
                return False
        return True

    def apply(self, row: dict[str, Any]) -> dict[str, Any]:
        fits: list[dict[str, Any]] = []
        boundary = []
        for rule in self.rules:
            ok, reverse = self._endpoint_match(row, rule)
            if not ok:
                continue
            start, end = int(rule['start_ns']), int(rule['end_ns'])
            if row['end_ns'] < start or row['start_ns'] > end:
                continue
            if row['start_ns'] < start or row['end_ns'] > end:
                boundary.append(rule['id'])
                continue
            if self._condition(row, rule, reverse):
                fits.append(rule)
        labels = {r.get('label') for r in fits if r.get('label')}
        if len(labels) > 1:
            return dict(label=None, annotation_label=None, label_status='conflict',
                        label_rule_ids=[r['id'] for r in fits], label_sources=sorted({r.get('source', self.default_source) for r in fits}))
        if fits:
            rule = fits[0]
            annotation = rule.get('annotation', rule.get('label'))
            if rule.get('attempted', False):
                label = self.background if self.background else None
                status = 'attempted_to_' + str(label or 'unresolved')
            else:
                label, status = rule['label'], 'confirmed_attack'
            return dict(label=label, annotation_label=annotation, label_status=status,
                        label_rule_ids=[r['id'] for r in fits], label_sources=sorted({r.get('source', self.default_source) for r in fits}))
        if boundary:
            return dict(label=None, annotation_label=None, label_status='boundary', label_rule_ids=boundary, label_sources=[self.default_source])
        if self.background:
            return dict(label=self.background, annotation_label='background', label_status='background_benign',
                        label_rule_ids=[], label_sources=[self.default_source])
        return dict(label=None, annotation_label=None, label_status='unmatched', label_rule_ids=[], label_sources=[self.default_source])


def fetch(config: dict[str, Any], out: Path) -> dict[str, Any]:
    receipts = []
    total = len(config['captures'])
    progress_path = out / 'fetch_progress.json'
    for index, cap in enumerate(config['captures'], 1):
        dest = resolve_path(cap['path'], ROOT)
        write(progress_path, dict(status='downloading', completed=len(receipts), total=total,
                                  current_index=index, capture_id=cap['id'], destination=str(dest),
                                  started_at=time.time()))
        print(f'fetch {index}/{total}: {cap["id"]} → {dest}', flush=True)
        receipt = download_member(cap['url'], cap['member'], dest, max_bytes=int(cap['max_uncompressed_bytes']))
        # ``download_member`` already records the selected archive member.
        # Keep that single, checksum-bound value instead of passing ``member``
        # twice when an existing capture is reused.
        receipts.append(dict(id=cap['id'], path=cap['path'], **receipt))
        write(progress_path, dict(status='downloading', completed=len(receipts), total=total,
                                  current_index=index, capture_id=cap['id'], destination=str(dest),
                                  last_receipt=receipts[-1], updated_at=time.time()))
        print(f'fetch {index}/{total}: complete {cap["id"]} ({receipt["bytes"]:,} bytes)', flush=True)
    result = dict(captures=receipts, downloaded_at=time.time())
    write(out / 'fetch.json', result)
    write(progress_path, dict(status='complete', completed=total, total=total, updated_at=time.time()))
    return result


def _reservoir_add(bucket: list[dict[str, Any]], seen: int, row: dict[str, Any], cap: int, rng: random.Random) -> None:
    if len(bucket) < cap:
        bucket.append(row)
        return
    candidate = rng.randrange(seen)
    if candidate < cap:
        bucket[candidate] = row


def build_corpus(config: dict[str, Any], out: Path) -> dict[str, Any]:
    corpus = out / 'corpus'
    manifest_path = corpus / 'manifest.json'
    if manifest_path.exists():
        return read(manifest_path)
    corpus.mkdir(parents=True, exist_ok=False)
    cap = int(config['corpus']['reservoir_per_label_split'])
    unknown_cap = int(config['corpus'].get('reservoir_unresolved_per_split', 0))
    fractions = config['split_fractions']
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError('split fractions must sum to one')
    rng = random.Random(config['seed'])
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[tuple[str, str]] = Counter()
    capture_meta, raw_stats, all_status = [], {}, Counter()
    source = config['label_source']
    for cap_cfg in config['captures']:
        pcap = resolve_path(cap_cfg['path'], ROOT)
        if not pcap.exists():
            raise FileNotFoundError(f'Missing verified PCAP: {pcap}')
        engine = CorrectedRuleSet(cap_cfg, source)
        stats: Counter[str] = Counter()
        for row in iter_flows(pcap, cap_cfg['id'], cap_cfg['group_namespace'], stats,
                              **config['flow']):
            labelled = engine.apply(row)
            if row.get('capture_truncated'):
                labelled = dict(label=None, annotation_label=None, label_status='incomplete_capture',
                                label_rule_ids=[], label_sources=[source])
            row.update(labelled)
            row['split'] = split_group(row['group_id'], config['seed'], fractions)
            label_key = row['label'] if row['label'] is not None else '__UNRESOLVED__'
            key = (label_key, row['split'])
            seen[key] += 1
            take = cap if row['label'] is not None else unknown_cap
            if take:
                _reservoir_add(buckets[key], seen[key], row, take, rng)
            all_status[row['label_status']] += 1
        raw_stats[cap_cfg['id']] = dict(stats)
        capture_meta.append(dict(id=cap_cfg['id'], path=cap_cfg['path'], bytes=pcap.stat().st_size,
                                 sha256=sha256(pcap), group_namespace=cap_cfg['group_namespace']))
    flows = corpus / 'flows.jsonl'
    with flows.open('x') as f:
        for key in sorted(buckets):
            for row in sorted(buckets[key], key=lambda x: x['flow_id']):
                f.write(json.dumps(row, separators=(',', ':')) + '\n')
    result = dict(schema=1, dataset=config['dataset'], seed=config['seed'], captures=capture_meta,
                  raw_stats=raw_stats, label_source=source,
                  label_rule_note=config['label_rule_note'], label_status_counts=dict(all_status),
                  source_counts=[dict(label=k[0], split=k[1], rows=v) for k, v in sorted(seen.items())],
                  sampled_counts=[dict(label=k[0], split=k[1], rows=len(v)) for k, v in sorted(buckets.items())],
                  reservoir_per_label_split=cap, split='SHA256(capture group namespace + bidirectional 5-tuple), before reservoir',
                  class_order=config['class_order'], flows_sha256=sha256(flows),
                  encoder_reuse_contract=dict(
                      artifact='flows.jsonl',
                      raw_source='capture path plus sha256 above',
                      packet_locator=('capture_id + bidirectional 5-tuple + start/end timestamps + first/last packet index; '
                                      'future adapters reopen the verified PCAP to reconstruct packets beyond the bounded prefix'),
                      common_fields=['flow_id', 'group_id', 'split', 'label', 'annotation_label', 'start_ns', 'end_ns',
                                     'src_ip', 'dst_ip', 'src_port', 'dst_port', 'first_packet_index', 'last_packet_index',
                                     'retained_packet_indices'],
                      netfound_view='burst_tokens/directions/bytes/iats/counts/protocol',
                      guarantee='A new encoder writes a separate view and never changes corpus labels, split, or flow IDs'))
    write(manifest_path, result)
    return result


def load_rows(corpus: Path) -> list[dict[str, Any]]:
    meta = read(corpus / 'manifest.json')
    flows = corpus / 'flows.jsonl'
    if sha256(flows) != meta['flows_sha256']:
        raise RuntimeError('Corpus flow artifact changed after its manifest was written')
    return [json.loads(line) for line in flows.open()]


def encode_netfound(config: dict[str, Any], corpus: Path, out: Path, device: str, batch_size: int) -> Path:
    adapter = 'netfound-v2-small'
    target = out / 'encoders' / adapter
    target.mkdir(parents=True, exist_ok=True)
    done = target / 'encoder.json'
    if done.exists():
        return target
    import numpy as np
    rows = load_rows(corpus)
    checkpoint = resolve_path(config['encoders'][adapter]['checkpoint'], ROOT)
    encoder = NetFoundEncoder(ROOT, checkpoint, device, config['encoders'][adapter].get('dtype', 'float32'))
    emb = np.lib.format.open_memmap(target / 'embeddings.npy', mode='w+', dtype='float32',
                                    shape=(len(rows), encoder.metadata['hidden_size']))
    offset, batch, parity = 0, [], None
    def flush() -> None:
        nonlocal offset, batch, parity
        if not batch:
            return
        vector = encoder.encode(batch)
        if parity is None:
            one = encoder.encode([batch[0]])[0]
            parity = float(np.max(np.abs(one - vector[0])))
            if not np.allclose(one, vector[0], atol=2e-4, rtol=2e-4):
                raise AssertionError(f'netFound batch/padding dependence: {parity}')
        emb[offset:offset + len(batch)] = vector
        offset += len(batch)
        batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            flush()
    flush(); emb.flush()
    if offset != len(rows):
        raise AssertionError('Embedding count mismatch')
    write(done, dict(adapter=adapter, corpus_sha256=read(corpus / 'manifest.json')['flows_sha256'], rows=len(rows),
                     batch_size=batch_size, batch_parity_max_abs_error=parity,
                     embeddings_sha256=sha256(target / 'embeddings.npy'), **encoder.metadata))
    return target


def project(config: dict[str, Any], corpus: Path, encoder_dir: Path) -> dict[str, Any]:
    import numpy as np
    from sklearn.decomposition import PCA
    result_file = encoder_dir / 'projection.json'
    if result_file.exists():
        return read(result_file)
    rows = load_rows(corpus)
    info = read(encoder_dir / 'encoder.json')
    x = np.load(encoder_dir / 'embeddings.npy', mmap_mode='r')
    if sha256(encoder_dir / 'embeddings.npy') != info['embeddings_sha256']:
        raise RuntimeError('Embedding artifact changed after encoding')
    ids = np.asarray([r['flow_id'] for r in rows])
    split = np.asarray([r['split'] for r in rows])
    names = list(config['class_order'])
    y = np.asarray([names.index(r['label']) if r['label'] in names else -1 for r in rows], dtype=np.int16)
    train = split == 'train'
    if train.sum() < 2:
        raise ValueError('No sufficient train rows for PCA')
    dim = min(int(config['projection']['components']), int(train.sum()) - 1, x.shape[1])
    pca = PCA(n_components=dim, svd_solver='full', random_state=config['seed']).fit(x[train])
    z = pca.transform(x).astype('float32')
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite projection')
    np.savez_compressed(encoder_dir / 'pca.npz', mean=pca.mean_, components=pca.components_,
                        explained_variance_ratio=pca.explained_variance_ratio_, fit_flow_ids=ids[train])
    np.savez_compressed(encoder_dir / 'dataset.npz', X=z, y=y, split=split, flow_id=ids,
                        class_names=np.asarray(names))
    for part in ['train', 'validation', 'test']:
        mask = (split == part) & (y >= 0)
        np.savez_compressed(encoder_dir / (part + '.npz'), X=z[mask], y=y[mask], flow_id=ids[mask],
                            class_names=np.asarray(names))
    groups = {part: {r['group_id'] for r in rows if r['split'] == part} for part in ['train', 'validation', 'test']}
    if groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test']:
        raise AssertionError('Flow group split overlap')
    result = dict(adapter=info['adapter'], rows=len(rows), input_dimensions=x.shape[1], pca_dimensions=dim,
                  pca_fit_rows=int(train.sum()), pca_fit_scope='all corpus train rows, including unresolved rows; no validation/test',
                  pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
                  class_order=names, split_counts={k: int((split == k).sum()) for k in groups},
                  labelled_counts={part: {name: int(((split == part) & (y == i)).sum()) for i, name in enumerate(names)}
                                  for part in groups}, group_overlap=0, classifier_evaluated=False)
    write(result_file, result)
    return result


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call):
    _sync(); start = time.perf_counter(); result = call(); _sync()
    return result, time.perf_counter() - start


def _f1(y: Any, pred: Any, classes: list[int], names: list[str]) -> dict[str, Any]:
    import numpy as np
    y, pred = np.asarray(y), np.asarray(pred)
    values = []
    for c in classes:
        tp = int(np.sum((y == c) & (pred == c)))
        fp = int(np.sum((y != c) & (pred == c)))
        fn = int(np.sum((y == c) & (pred != c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        values.append(dict(name=names[c], support=int(np.sum(y == c)), precision=precision, recall=recall, f1=f,
                           tp=tp, fp=fp, fn=fn))
    return dict(macro_f1=float(sum(v['f1'] for v in values) / len(values)), classes=values, rows=len(y))


def _weights_digest(model: Any) -> str:
    import torch
    h = hashlib.sha256()
    for base in model.models_:
        for name, value in sorted(base.state_dict().items()):
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _make_model(method: str, seed: int, classes: list[int], config: dict[str, Any]) -> Any:
    if method == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.05, subsample=.8,
                             colsample_bytree=.8, min_child_weight=1, reg_lambda=1,
                             objective='multi:softprob', num_class=len(classes), eval_metric='mlogloss',
                             tree_method='hist', device='cuda:0', n_jobs=1, random_state=seed)
    if method == 'tabpfn':
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device='cuda:0', model_path=str(resolve_path(config['tabpfn_checkpoint'], ROOT)),
                                random_state=seed, n_estimators=4, auto_scale_n_estimators=False,
                                fit_mode='fit_with_cache', keep_cache_on_device=False,
                                n_preprocessing_jobs=1, inference_config={'SUBSAMPLE_SAMPLES': None})
    raise ValueError(method)


def ceiling_audit(results: list[dict[str, Any]], names: list[str], out: Path) -> dict[str, Any]:
    """Summarise, but do not automatically diagnose, early high-score plateaus."""
    import numpy as np
    cells: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        cells[(item['regime'], item['method'], int(item['budget']))].append(item)
    aggregate, class_aggregate = [], []
    for (regime, method, budget), items in sorted(cells.items()):
        values = np.asarray([item['metrics']['macro_f1'] for item in items], dtype=float)
        aggregate.append(dict(regime=regime, method=method, budget=budget, seeds=len(items),
                              macro_f1_mean=float(values.mean()), macro_f1_std=float(values.std(ddof=0)),
                              macro_f1_min=float(values.min()), macro_f1_max=float(values.max())))
        for name in names:
            f1 = [record['f1'] for item in items for record in item['metrics']['classes']
                  if record['name'] == name]
            if f1:
                class_aggregate.append(dict(regime=regime, method=method, budget=budget, name=name,
                                            seeds=len(f1), f1_mean=float(np.mean(f1)), f1_std=float(np.std(f1, ddof=0))))
    by_key = {(r['regime'], r['method'], r['budget']): r for r in aggregate}
    saturation_candidates = []
    for regime in sorted({r['regime'] for r in aggregate}):
        for method in sorted({r['method'] for r in aggregate}):
            at20, at50 = by_key.get((regime, method, 20)), by_key.get((regime, method, 50))
            if at20 and at50:
                delta = at50['macro_f1_mean'] - at20['macro_f1_mean']
                saturation_candidates.append(dict(regime=regime, method=method, budget_a=20, budget_b=50,
                                                  macro_f1_a=at20['macro_f1_mean'], macro_f1_b=at50['macro_f1_mean'],
                                                  delta=delta,
                                                  high_and_flat_candidate=bool(at20['macro_f1_mean'] >= .90 and abs(delta) <= .01)))
    result = dict(schema=1, aggregate=aggregate, class_aggregate=class_aggregate,
                  saturation_candidates=saturation_candidates,
                  interpretation=("A high-and-flat flag is a descriptive trigger for data/label review, not proof that a "
                                  "dataset has an intrinsic ceiling. Compare it with class support, cross-label identical-input "
                                  "audit, normal-included versus attack-only results, and a new encoder view."))
    write(out / 'ceiling_audit.json', result)
    return result


def fewshot(config: dict[str, Any], encoder_dir: Path, out: Path, adapter: str) -> dict[str, Any]:
    import numpy as np
    eval_dir = out / 'evaluations' / adapter / 'fewshot'
    eval_dir.mkdir(parents=True, exist_ok=True)
    with np.load(encoder_dir / 'dataset.npz') as data:
        x, y, split, ids, names = data['X'], data['y'], data['split'], data['flow_id'], data['class_names'].tolist()
    expected = config['class_order']
    if names != expected:
        raise ValueError(f'Unexpected class order: {names}')
    train = split == 'train'
    test = split == 'test'
    budgets, seeds = config['fewshot']['budgets'], config['fewshot']['seeds']
    max_budget = max(budgets)
    all_results = []
    for regime in config['fewshot']['regimes']:
        classes = list(range(len(names))) if regime == 'with_benign' else list(range(1, len(names)))
        test_mask = test & np.isin(y, classes)
        if not test_mask.any():
            raise ValueError(f'No held-out rows for {regime}')
        for seed in seeds:
            selected = {}
            for c in classes:
                pool = np.flatnonzero(train & (y == c))
                if len(pool) < max_budget:
                    raise ValueError(f'{names[c]} has only {len(pool)} train rows, below n={max_budget}')
                selected[c] = np.random.default_rng(seed + 1009 * c).choice(pool, max_budget, replace=False)
            for budget in budgets:
                chosen = np.concatenate([selected[c][:budget] for c in classes])
                local = {c: i for i, c in enumerate(classes)}
                y_train = np.asarray([local[int(v)] for v in y[chosen]], dtype=np.int64)
                for method in config['fewshot']['methods']:
                    tag = f'{regime}_s{seed}_n{budget}_{method}'
                    job = eval_dir / 'jobs' / tag
                    done = job / 'result.json'
                    if done.exists():
                        all_results.append(read(done)); continue
                    job.mkdir(parents=True, exist_ok=False)
                    model, construct = _timed(lambda: _make_model(method, seed, classes, config['fewshot']))
                    _, fit = _timed(lambda: model.fit(x[chosen], y_train))
                    frozen_before = _weights_digest(model) if method == 'tabpfn' else None
                    proba, infer = _timed(lambda: model.predict_proba(x[test_mask]))
                    if proba.shape != (int(test_mask.sum()), len(classes)) or not np.isfinite(proba).all():
                        raise RuntimeError('Invalid probability matrix')
                    if frozen_before and frozen_before != _weights_digest(model):
                        raise RuntimeError('TabPFN backbone weights changed during inference')
                    pred = np.asarray(classes, dtype=np.int16)[proba.argmax(1)]
                    row = dict(regime=regime, seed=seed, budget=budget, method=method,
                               train_rows=len(chosen), train_flow_ids=ids[chosen].tolist(),
                               test_rows=int(test_mask.sum()), test_flow_ids_sha256=hashlib.sha256(ids[test_mask].tobytes()).hexdigest(),
                               metrics=_f1(y[test_mask], pred, classes, names), construct_seconds=construct,
                               fit_seconds=fit, inference_seconds=infer, frozen_weights_sha256=frozen_before,
                               adapter=adapter, pca_dimensions=x.shape[1], finetuning=False)
                    np.save(job / 'predictions.npy', pred)
                    write(done, row); all_results.append(row)
                    del model
                    gc.collect()
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
    rows = []
    for item in all_results:
        rows.append(dict(regime=item['regime'], seed=item['seed'], budget=item['budget'], method=item['method'],
                         macro_f1=item['metrics']['macro_f1'], train_rows=item['train_rows'], test_rows=item['test_rows'],
                         preparation_seconds=item['construct_seconds'] + item['fit_seconds'], inference_seconds=item['inference_seconds']))
    with (eval_dir / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: (r['regime'], r['budget'], r['seed'], r['method'])))
    result = dict(completed_jobs=len(all_results), expected_jobs=len(config['fewshot']['regimes']) * len(seeds) * len(budgets) * len(config['fewshot']['methods']),
                  adapter=adapter, budgets=budgets, seeds=seeds, regimes=config['fewshot']['regimes'],
                  comparison='XGB and frozen TabPFN receive exactly the same selected rows and PCA vectors in every job')
    write(eval_dir / 'manifest.json', result)
    result['ceiling_audit'] = ceiling_audit(all_results, names, eval_dir)
    write(eval_dir / 'manifest.json', result)
    return result


def audit(corpus: Path, encoder_dir: Path) -> dict[str, Any]:
    import numpy as np
    rows = load_rows(corpus)
    with np.load(encoder_dir / 'dataset.npz') as z:
        x, y, split, ids = z['X'], z['y'], z['split'], z['flow_id']
    assert list(ids) == [r['flow_id'] for r in rows]
    groups = {part: {r['group_id'] for r in rows if r['split'] == part} for part in ['train', 'validation', 'test']}
    if groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test']:
        raise AssertionError('Group overlap')
    token_groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        # These are exactly the model-input fields before tokenization.  The
        # signature deliberately excludes flow ID, addresses, ports, time,
        # label and split, none of which are encoder features.
        payload = {k: row[k] for k in ['flow_duration', 'burst_tokens', 'directions', 'bytes', 'iats', 'counts', 'protocol']}
        token_groups[hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()].append(i)
    feature_conflicts = []
    for digest, members in token_groups.items():
        labels = sorted({rows[i]['label'] for i in members if rows[i]['label'] is not None})
        if len(labels) > 1:
            feature_conflicts.append(dict(input_feature_sha256=digest, rows=len(members), labels=labels,
                                          flow_ids=[rows[i]['flow_id'] for i in members[:20]]))
    vector_groups: dict[str, list[int]] = defaultdict(list)
    for i, vector in enumerate(x):
        vector_groups[hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest()].append(i)
    vector_conflicts = []
    for digest, members in vector_groups.items():
        labels = sorted({rows[i]['label'] for i in members if rows[i]['label'] is not None})
        if len(labels) > 1:
            vector_conflicts.append(dict(projected_feature_sha256=digest, rows=len(members), labels=labels,
                                         flow_ids=[rows[i]['flow_id'] for i in members[:20]]))
    result = dict(passed=True, rows=len(rows), labelled_rows=int((y >= 0).sum()), unresolved_rows=int((y < 0).sum()),
                  group_overlap=0, finite_vectors=bool(np.isfinite(x).all()),
                  corpus_flow_sha256=read(corpus / 'manifest.json')['flows_sha256'],
                  identical_input_feature_groups=sum(len(v) > 1 for v in token_groups.values()),
                  identical_input_feature_cross_label_groups=len(feature_conflicts),
                  identical_input_feature_cross_label_examples=feature_conflicts[:100],
                  identical_projected_feature_groups=sum(len(v) > 1 for v in vector_groups.values()),
                  identical_projected_feature_cross_label_groups=len(vector_conflicts),
                  identical_projected_feature_cross_label_examples=vector_conflicts[:100],
                  interpretation='Checks corpus/vector alignment and group separation; it is not a classifier result.')
    write(encoder_dir / 'audit.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['fetch', 'corpus', 'encode', 'project', 'audit', 'fewshot', 'all'], default='all')
    parser.add_argument('--corpus', help='Existing encoder-independent corpus directory for a new encoder view')
    parser.add_argument('--adapter', default='netfound-v2-small')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    config = read(resolve_path(args.config, ROOT))
    out = resolve_path(args.output, ROOT); out.mkdir(parents=True, exist_ok=True)
    write(out / 'protocol_snapshot.json', config)
    if args.adapter != 'netfound-v2-small' and args.stage in ('fetch', 'corpus', 'encode', 'all'):
        raise ValueError(f'No in-run encoder implementation for {args.adapter}. Write its compatible encoder view first, then use --stage project, audit or fewshot.')
    if args.stage in ('fetch', 'all'):
        fetch(config, out)
    if args.stage == 'fetch':
        write(out / 'status.json', dict(status='complete', stage=args.stage, adapter=args.adapter,
                                        classifier_evaluated=False))
        return
    corpus = resolve_path(args.corpus, ROOT) if args.corpus else out / 'corpus'
    if args.stage in ('corpus', 'all'):
        build_corpus(config, out)
    if args.stage == 'corpus':
        write(out / 'status.json', dict(status='complete', stage=args.stage, adapter=args.adapter,
                                        corpus=str(corpus), classifier_evaluated=False))
        return
    if not (corpus / 'manifest.json').exists():
        raise FileNotFoundError(f'Corpus missing: {corpus}')
    if args.stage in ('encode', 'all'):
        encoder_dir = encode_netfound(config, corpus, out, args.device, args.batch_size)
    else:
        encoder_dir = out / 'encoders' / args.adapter
    if args.stage in ('project', 'all'):
        project(config, corpus, encoder_dir)
    if args.stage in ('audit', 'all'):
        audit(corpus, encoder_dir)
    if args.stage in ('fewshot', 'all'):
        fewshot(config, encoder_dir, out, args.adapter)
    write(out / 'status.json', dict(status='complete', stage=args.stage, adapter=args.adapter,
                                    corpus=str(corpus), classifier_evaluated=args.stage in ('fewshot', 'all')))


if __name__ == '__main__':
    main()

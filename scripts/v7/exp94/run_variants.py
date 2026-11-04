"""EXP94 follow-up variants on a completed PCAP run (encodings reused).

(a) label policy for *unresolved* flows: ``exclude`` (base), ``attack`` (parent
    family, the original-CSV convention) and ``benign`` (CNS2022/EXP93
    convention).  Every policy re-derives coverage, the three-layer cross-label
    audit with oracle ceilings and the paired few-shot curves for every view.
(b) netFound projection size: PCA 128 and the raw 512-dimensional embedding
    (TabPFN runs with ``ignore_pretraining_limits``) instead of PCA 64.

Outputs go to ``<base_run>/variants/<name>/`` so the base run stays untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
for rel in ('scripts/common', 'scripts/v7/exp92', 'scripts/v7/exp93', 'scripts/v7/exp94'):
    sys.path.insert(0, str(ROOT / rel))
from experiment_paths import resolve_path
import run_pcap_benchmark as exp93
from run_pcap_benchmark import fewshot, read, write
from run_pcap_cic2017 import _layer, _oracle

import numpy as np

LIGHT = ('flow_id', 'group_id', 'capture_id', 'split', 'label', 'label_status', 'annotation_label', 'label_rule_ids')


def load_light_rows(corpus: Path, need_keys: bool) -> tuple[list[dict], dict[str, list[str]]]:
    """Stream flows.jsonl; keep the small label fields and (optionally) the L1/L2 hash keys."""
    rows, keys = [], {'L1_flow_signature': [], 'L2_tokenizer_input': [], 'L3_flowstats_vector': []}
    t0 = time.time()
    with (corpus / 'flows.jsonl').open() as f:
        for n, line in enumerate(f, 1):
            r = json.loads(line)
            rows.append({k: r.get(k) for k in LIGHT})
            if need_keys:
                sig = {k: r[k] for k in ('stats', 'burst_tokens', 'directions', 'bytes', 'iats', 'counts', 'protocol')}
                keys['L1_flow_signature'].append(hashlib.sha256(json.dumps(sig, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
                tok = dict(b=[b[:6 * 19] for b in r['burst_tokens'][:12]], d=r['directions'][:12], y=r['bytes'][:12],
                           i=[int(v * 1e-3) for v in r['iats'][:12]], c=r['counts'][:12], p=r['protocol'])
                keys['L2_tokenizer_input'].append(hashlib.sha256(json.dumps(tok, separators=(',', ':')).encode()).hexdigest())
                keys['L3_flowstats_vector'].append(hashlib.sha256(np.asarray(r['stats'], dtype=np.float32).tobytes()).hexdigest())
            if n % 500000 == 0:
                print(f'rows {n:,} ({time.time() - t0:.0f}s)', flush=True)
    return rows, keys


def parent_family_map(config: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """annotation -> family and rule_id -> family, for relabelling unresolved flows as attacks."""
    ann, rid = {}, {}
    fam_map = config.get('family_mapping') or {}
    for cap in config['captures']:
        for r in cap.get('rules', []):
            label = r.get('label')
            family = label if label in config['class_order'] else fam_map.get(label) or fam_map.get(r.get('annotation'))
            if family and family != 'benign':
                rid[r['id']] = family
                if r.get('annotation'):
                    ann[r['annotation']] = family
    for a in config.get('attempted_categories', []) or []:
        if a.get('parent_family'):
            ann[a['annotation']] = a['parent_family']
    return ann, rid


def relabel(rows: list[dict], policy: str, ann_map: dict[str, str], rid_map: dict[str, str]) -> tuple[list[str | None], Counter]:
    labels, moved = [], Counter()
    for r in rows:
        if not str(r['label_status']).startswith('unresolved') or policy == 'exclude':
            labels.append(r['label']); continue
        if policy == 'benign':
            labels.append('benign'); moved['benign'] += 1; continue
        fam = ann_map.get(r['annotation_label'])
        if fam is None:
            parent = (r['annotation_label'] or '').split('-Attempted')[0]
            fam = ann_map.get(parent)
        if fam is None:
            for rule_id in r.get('label_rule_ids') or []:
                if rule_id in rid_map:
                    fam = rid_map[rule_id]; break
        labels.append(fam); moved[fam or '__unmapped__'] += 1
    return labels, moved


def coverage(config: dict[str, Any], labels: list[str | None], rows: list[dict], out: Path) -> list[str]:
    max_budget = max(int(v) for v in config['fewshot']['budgets'])
    cov, inc, exc = {}, [], []
    for fam in config['class_order']:
        c = Counter(r['split'] for r, l in zip(rows, labels) if l == fam)
        cov[fam] = dict(train=int(c['train']), validation=int(c['validation']), test=int(c['test']))
        (inc if c['train'] >= max_budget and c['test'] >= 1 else exc).append(fam)
    write(out / 'label_coverage.json', dict(max_fewshot_budget=max_budget, coverage=cov, included=inc, excluded=exc))
    return inc


def audit_layers(config: dict[str, Any], rows: list[dict], labels: list[str | None], names: list[str],
                 layer_keys: dict[str, list[str]], out: Path) -> dict[str, Any]:
    lrows = [dict(label=l, flow_id=r['flow_id'], split=r['split']) for r, l in zip(rows, labels)]
    labelled = np.asarray([i for i, r in enumerate(lrows) if r['label'] in names])
    test = np.asarray([i for i in labelled if lrows[i]['split'] == 'test'])
    classes = list(range(len(names)))
    result = dict(schema=1, rows=len(lrows), labelled_rows=int(len(labelled)), test_rows=int(len(test)), layers={})
    for name, keys in layer_keys.items():
        train_set = {keys[i] for i in labelled if lrows[i]['split'] == 'train'}
        dup = np.asarray([keys[i] in train_set for i in test])
        result['layers'][name] = dict(
            full_labelled=_layer(keys, lrows, names, labelled), test=_layer(keys, lrows, names, test),
            train_to_test_exact_duplicate_fraction=float(dup.mean()) if len(dup) else None,
            oracle_ceiling={regime: _oracle(keys, lrows, names, test, classes if regime == 'with_benign' else [c for c in classes if names[c] != 'benign'])
                            for regime in config['fewshot']['regimes']})
    write(out / 'audit3.json', result)
    return result


def write_dataset(src_view: Path, dst_view: Path, labels: list[str | None], names: list[str], X: np.ndarray | None = None) -> None:
    dst_view.mkdir(parents=True, exist_ok=True)
    with np.load(src_view / 'dataset.npz') as z:
        base_X, split, ids = z['X'], z['split'], z['flow_id']
    y = np.asarray([names.index(l) if l in names else -1 for l in labels], dtype=np.int16)
    np.savez_compressed(dst_view / 'dataset.npz', X=base_X if X is None else X, y=y, split=split, flow_id=ids, class_names=np.asarray(names))


def patched_make_model():
    """Allow >500 features for TabPFN in the raw-512 variant without touching EXP93 code."""
    original = exp93._make_model

    def make(method, seed, classes, config):
        model = original(method, seed, classes, config)
        if method == 'tabpfn' and config.get('_ignore_pretraining_limits'):
            model.set_params(ignore_pretraining_limits=True)
        return model
    exp93._make_model = make


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-run', required=True)
    ap.add_argument('--output', required=True, help='run directory for the variant records (registry run_dir)')
    ap.add_argument('--policies', default='attack,benign')
    ap.add_argument('--pca-dims', default='128,512')
    ap.add_argument('--views', default='flowstats-v1,netfound-v2-small')
    ap.add_argument('--what', choices=['labels', 'pca', 'both'], default='both')
    a = ap.parse_args()
    base = resolve_path(a.base_run, ROOT); out = resolve_path(a.output, ROOT); out.mkdir(parents=True, exist_ok=True)
    config = read(base / 'protocol_snapshot.json')
    full_names = list(config['class_order'])
    corpus = base / 'corpus'
    print('loading corpus rows + layer keys', flush=True)
    rows, keys = load_light_rows(corpus, need_keys=True)
    with np.load(base / 'encoders/netfound-v2-small/dataset.npz') as z:
        keys['L3_netfound_pca64_vector'] = [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in z['X']]
    write(out / 'variants_manifest.json', dict(base_run=(str(base.relative_to(ROOT)) if base.is_relative_to(ROOT) else str(base)), rows=len(rows), policies=a.policies, pca_dims=a.pca_dims,
                                               base_flows_sha256=read(corpus / 'manifest.json')['flows_sha256']))
    patched_make_model()
    ann_map, rid_map = parent_family_map(config)
    if a.what in ('labels', 'both'):
        for policy in a.policies.split(','):
            vdir = out / f'label_{policy}'; vdir.mkdir(parents=True, exist_ok=True)
            labels, moved = relabel(rows, policy, ann_map, rid_map)
            cfg = json.loads(json.dumps(config)); cfg['class_order'] = full_names
            names = coverage(cfg, labels, rows, vdir); cfg['class_order'] = names
            write(vdir / 'policy.json', dict(policy=policy, unresolved_relabelled=dict(moved), classes=names,
                                             note='attack = parent family of the attempted/near-miss rule (original CSV convention); benign = CNS2022/EXP93 convention'))
            print(f'[{policy}] moved {dict(moved)}; classes {names}', flush=True)
            audit_layers(cfg, rows, labels, names, keys, vdir)
            for view in a.views.split(','):
                write_dataset(base / 'encoders' / view, vdir / 'encoders' / view, labels, names)
                fewshot(cfg, vdir / 'encoders' / view, vdir, view)
            write(vdir / 'status.json', dict(status='complete'))
    if a.what in ('pca', 'both'):
        from sklearn.decomposition import PCA
        emb = np.load(base / 'encoders/netfound-v2-small/embeddings.npy', mmap_mode='r')
        split = np.asarray([r['split'] for r in rows]); train = split == 'train'
        labels = [r['label'] for r in rows]
        for dim in [int(d) for d in a.pca_dims.split(',')]:
            vdir = out / f'pca_{dim}'; vdir.mkdir(parents=True, exist_ok=True)
            cfg = json.loads(json.dumps(config)); cfg['class_order'] = full_names
            names = coverage(cfg, labels, rows, vdir); cfg['class_order'] = names
            if dim >= emb.shape[1]:
                X = np.ascontiguousarray(emb, dtype=np.float32); info = dict(pca=None, dimensions=int(emb.shape[1]))
                cfg['fewshot']['_ignore_pretraining_limits'] = True
            else:
                pca = PCA(n_components=dim, svd_solver='full', random_state=cfg['seed']).fit(emb[train])
                X = pca.transform(emb).astype(np.float32); info = dict(pca=dim, explained_variance=float(pca.explained_variance_ratio_.sum()))
            view = vdir / 'encoders' / 'netfound-v2-small'
            write_dataset(base / 'encoders/netfound-v2-small', view, labels, names, X=X)
            lk = {'L3_netfound_vector': [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in X]}
            audit_layers(cfg, rows, labels, names, lk, vdir)
            write(vdir / 'projection.json', dict(info, classes=names))
            print(f'[pca_{dim}] {info}', flush=True)
            fewshot(cfg, view, vdir, 'netfound-v2-small')
            write(vdir / 'status.json', dict(status='complete'))
    write(out / 'status.json', dict(status='complete', what=a.what))


if __name__ == '__main__':
    main()

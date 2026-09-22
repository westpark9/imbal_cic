"""Versioned conflict-only row exclusion, preserving original outer split IDs."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(X):
    a = np.nan_to_num(np.asarray(X, dtype=np.float32), copy=True)
    a[a == 0] = 0.0  # Numeric equality includes signed zero.
    return np.ascontiguousarray(a)


def conflict_mask(X, y, hashes=None, chunk=100_000, progress=None):
    """Hash candidate groups, then verify exact vectors; handle hash collisions."""
    X = canonical(X)
    y = np.asarray(y)
    if len(X) != len(y) or len(y) == 0:
        raise ValueError('Nonempty, aligned X and y required')
    if hashes is None:
        hashes = np.empty(len(X), dtype=np.uint64)
        for start in range(0, len(X), chunk):
            stop = min(start + chunk, len(X))
            hashes[start:stop] = pd.util.hash_pandas_object(
                pd.DataFrame(X[start:stop]), index=False).to_numpy()
            if progress and (start == 0 or stop == len(X) or stop % 2_000_000 == 0):
                progress('hashing', stop, len(X))
    hashes = np.asarray(hashes, dtype=np.uint64)
    if len(hashes) != len(y):
        raise ValueError('Hash length mismatch')
    unique, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    low = np.full(len(unique), np.iinfo(np.int64).max, dtype=np.int64)
    high = np.full(len(unique), -1, dtype=np.int64)
    np.minimum.at(low, inverse, y)
    np.maximum.at(high, inverse, y)
    group_conflict = low != high
    drop = group_conflict[inverse]
    positions = np.flatnonzero(drop)
    collision_groups = set()
    for start in range(0, len(positions), chunk):
        pos = positions[start:start + chunk]
        equal = (X[pos] == X[first[inverse[pos]]]).all(axis=1)
        collision_groups.update(inverse[pos[~equal]].tolist())
        if progress and (start == 0 or start + chunk >= len(positions)):
            progress('exact_verification', min(start + chunk, len(positions)), len(positions))
    actual_groups = int(group_conflict.sum()) - len(collision_groups)
    for group in sorted(collision_groups):
        pos = np.flatnonzero(inverse == group)
        vectors = np.ascontiguousarray(X[pos]).view(np.dtype((np.void, X.shape[1] * X.dtype.itemsize))).ravel()
        _, exact_inv = np.unique(vectors, return_inverse=True)
        lo = np.full(int(exact_inv.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.full(len(lo), -1, dtype=np.int64)
        np.minimum.at(lo, exact_inv, y[pos]); np.maximum.at(hi, exact_inv, y[pos])
        conflicts = lo != hi
        drop[pos] = conflicts[exact_inv]
        actual_groups += int(conflicts.sum())
        # No actual multi-label vector may survive even when the hash collides.
        assert not conflicts[exact_inv[~drop[pos]]].any()
    normal = ~np.isin(inverse, list(collision_groups)) if collision_groups else np.ones(len(X), bool)
    assert not (group_conflict[inverse[normal & ~drop]]).any()
    meta = {'hash_candidate_groups': int(group_conflict.sum()),
            'hash_collision_groups_verified': len(collision_groups),
            'conflicting_vector_groups': actual_groups,
            'remaining_conflicting_vector_groups': 0,
            'canonical_feature_sha256': hashlib.sha256(memoryview(X).cast('B')).hexdigest()}
    return drop, hashes, meta


def filter_splits(splits, keep_idx):
    return {key: rows[np.isin(rows, keep_idx, assume_unique=True)] for key, rows in splits.items()}


def prepare(data, out):
    import nfv3_v3_common as core
    started = time.monotonic()
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    data = Path(data).resolve()
    def progress(phase, done=0, total=0):
        status = {'phase': phase, 'rows_done': done, 'rows_total': total,
                  'elapsed_seconds': round(time.monotonic() - started, 1)}
        write_json(out / 'progress.json', status)
        print(json.dumps(status), flush=True)
    progress('loading_source')
    args = SimpleNamespace(data=str(data))
    X, names, tr, va, te, _, _, audit, label = core.load_cic2018(args)
    splits = {'train': tr, 'val': va, 'test': te}
    all_idx = np.sort(np.concatenate(list(splits.values())))
    assert len(np.unique(all_idx)) == len(all_idx)
    y = label(all_idx)
    progress('canonicalize', 0, len(all_idx))
    drop, hashes, meta = conflict_mask(X[all_idx], y, progress=progress)
    keep_idx, drop_idx = all_idx[~drop], all_idx[drop]
    cleaned = filter_splits(splits, keep_idx)
    assert sum(map(len, cleaned.values())) == len(keep_idx)
    assert len(keep_idx) + len(drop_idx) == len(all_idx)
    suite = core.load_pickle(str(data))
    scenarios = np.asarray(suite['attack_scenarios'])
    timestamps = np.asarray(suite['timestamps'])
    rows, scenario_rows = [], []
    for key, original in splits.items():
        survived = cleaned[key]
        assert np.isin(survived, original, assume_unique=True).all()
        np.save(out / f'original_{key}_idx.npy', original, allow_pickle=False)
        np.save(out / f'{key}_idx.npy', survived, allow_pickle=False)
        for c, name in enumerate(names):
            before = original[label(original) == c]
            after = survived[label(survived) == c]
            rows.append({'split': key, 'class': name, 'before': len(before), 'removed': len(before) - len(after),
                         'after': len(after), 'removed_fraction': (len(before) - len(after)) / max(len(before), 1),
                         'time_min': int(timestamps[after].min()) if len(after) else None,
                         'time_max': int(timestamps[after].max()) if len(after) else None})
        for scenario in np.unique(scenarios[original]):
            before = int((scenarios[original] == scenario).sum())
            after = int((scenarios[survived] == scenario).sum())
            scenario_rows.append({'split': key, 'scenario': str(scenario), 'before': before,
                                  'removed': before - after, 'after': after})
    pd.DataFrame(rows).to_csv(out / 'class_counts.csv', index=False)
    pd.DataFrame(scenario_rows).to_csv(out / 'scenario_counts.csv', index=False)
    audit.to_csv(out / 'original_split_audit.csv', index=False)
    pd.DataFrame({'vector_hash': hashes[drop], 'label': np.asarray(names)[y[drop]]}).value_counts().rename('removed_rows').reset_index().to_csv(out / 'removed_groups.csv', index=False)
    np.save(out / 'keep_idx.npy', keep_idx, allow_pickle=False)
    np.save(out / 'drop_idx.npy', drop_idx, allow_pickle=False)
    np.save(out / 'source_idx.npy', all_idx, allow_pickle=False)
    np.save(out / 'vector_hash.npy', hashes, allow_pickle=False)
    progress('fingerprinting_source')
    st = data.stat()
    source = {'path': str(data), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': sha256(data)}
    artifacts = {p.name: sha256(p) for p in sorted(out.glob('*.npy'))}
    manifest = {'schema': 1, 'dataset': 'cic2018', 'source': source, 'class_names': names,
                'feature_count': X.shape[1], 'feature_names': list(map(str, suite.get('feature_names', []))),
                'preprocessing': 'float32; numpy.nan_to_num defaults; canonical signed zero',
                'rule': 'drop every row of exact vectors with multiple model-family labels; preserve all other duplicates',
                'split_rule': 'intersect original outer split memberships with keep_idx; no resplit',
                'rows_before': len(all_idx), 'rows_removed': len(drop_idx), 'rows_after': len(keep_idx),
                'split_rows': {key: len(value) for key, value in cleaned.items()},
                'same_label_duplicates_preserved': True, 'original_row_ids_preserved': True,
                'artifacts_sha256': artifacts, 'verification': meta,
                'seconds': round(time.monotonic() - started, 1)}
    missing_train = [r['class'] for r in rows if r['split'] == 'train' and r['after'] == 0]
    manifest['missing_train_classes'] = missing_train
    write_json(out / 'manifest.json', manifest)
    if missing_train:
        raise ValueError(f'Clean train has no support for {missing_train}; inspect saved manifest')
    write_json(out / 'COMPLETE.json', {'manifest_sha256': sha256(out / 'manifest.json'), 'seconds': manifest['seconds']})
    progress('complete', len(keep_idx), len(all_idx))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    core._PICKLE_CACHE.clear(); gc.collect()


def install_clean_loader(core, manifest_path):
    """Install only in the isolated EXP47 worker; historical loader stays intact."""
    manifest_path = Path(manifest_path).resolve()
    directory = manifest_path.parent
    meta = json.loads(manifest_path.read_text())
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    if sha256(manifest_path) != complete['manifest_sha256']:
        raise ValueError('Clean manifest identity mismatch')
    original_loader = core.load_cic2018
    def clean_loader(args):
        path = Path(args.data).resolve(); st = path.stat()
        src = meta['source']
        if str(path) != src['path'] or (st.st_size, st.st_mtime_ns) != (src['bytes'], src['mtime_ns']):
            raise ValueError('Source dataset identity mismatch')
        X, names, tr, va, te, _, _, audit, label = original_loader(args)
        if names != meta['class_names']:
            raise ValueError('Class vocabulary changed')
        indices = []
        for key, orig in [('train', tr), ('val', va), ('test', te)]:
            p = directory / f'{key}_idx.npy'
            if sha256(p) != meta['artifacts_sha256'][p.name]:
                raise ValueError(f'Clean index identity mismatch: {key}')
            idx = np.load(p, allow_pickle=False)
            if not np.isin(idx, orig, assume_unique=True).all():
                raise ValueError(f'Clean {key} moved across outer split')
            indices.append(idx)
        clean_audit = pd.read_csv(directory / 'scenario_counts.csv')
        return X, names, *indices, label(indices[0]), label(indices[2]), clean_audit, label
    core.load_cic2018 = clean_loader
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    prepare(args.data, args.out)

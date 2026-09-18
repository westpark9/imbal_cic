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


from cic2018_conflict_clean import write_json, sha256, conflict_mask, filter_splits


def prepare(data, out, dataset='ton_iot'):
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
    X, names, tr, va, te, _, _, audit, label = getattr(core, {'cic2018': 'load_cic2018', 'ton_iot': 'load_ton_iot', 'bot_iot': 'load_bot_iot'}[dataset])(args)
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
    manifest = {'schema': 1, 'dataset': dataset, 'source': source, 'class_names': names,
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
    """Install only in the isolated cleaned-data worker; historical loader stays intact."""
    manifest_path = Path(manifest_path).resolve()
    directory = manifest_path.parent
    meta = json.loads(manifest_path.read_text())
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    if sha256(manifest_path) != complete['manifest_sha256']:
        raise ValueError('Clean manifest identity mismatch')
    loader_name = {'cic2018': 'load_cic2018', 'ton_iot': 'load_ton_iot', 'bot_iot': 'load_bot_iot'}[meta['dataset']]
    original_loader = getattr(core, loader_name)
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
    setattr(core, loader_name, clean_loader)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cic2018', 'ton_iot', 'bot_iot'], default='ton_iot')
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    prepare(args.data, args.out, args.dataset)

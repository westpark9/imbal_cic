#!/usr/bin/env python3
"""Rebuild/reuse the exact EXP47/48 clean splits from an existing suite PKL."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = repo_root(__file__)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def verify_clean_split(directory, reference):
    """Compare against git-pinned benchmark identities, not just a local manifest."""
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = read_record(manifest_path)
    complete = read_record(directory / 'COMPLETE.json')
    if complete['manifest_sha256'] != sha256(manifest_path):
        raise ValueError(f'Clean manifest integrity mismatch: {directory}')
    for key in ['bytes', 'sha256']:
        if manifest['source'][key] != reference['source'][key]:
            raise ValueError(f'Clean source {key} differs from the EXP57 benchmark')
    expected = reference['dataset']
    for key in ['dataset', 'class_names', 'feature_count', 'rule', 'split_rule',
                'split_rows', 'rows_before', 'rows_removed', 'rows_after']:
        if manifest[key] != expected[key]:
            raise ValueError(f'Clean {key} differs from the EXP57 benchmark')
    if manifest['verification']['remaining_conflicting_vector_groups'] != 0:
        raise ValueError('Conflicting vectors remain')
    hashes = {}
    for name, expected_hash in expected['artifacts_sha256'].items():
        actual = sha256(directory / name)
        if actual != expected_hash or manifest['artifacts_sha256'][name] != expected_hash:
            raise ValueError(f'Clean split identity mismatch: {name}')
        hashes[name] = actual
    if not (directory / 'scenario_counts.csv').is_file():
        raise ValueError('Missing clean scenario audit')
    return {'directory': str(directory.resolve()), 'split_rows': manifest['split_rows'],
            'split_sha256': hashes, 'manifest_sha256': complete['manifest_sha256']}


def ensure_clean_split(data, directory, reference, prepare_script=None):
    """Serialize publishers; only expose a verified, completed cache directory."""
    data, directory = Path(data).resolve(), Path(directory).resolve()
    if data.stat().st_size != reference['source']['bytes']:
        raise ValueError('Source PKL size differs from the EXP57 benchmark')
    directory.parent.mkdir(parents=True, exist_ok=True)
    script = Path(prepare_script) if prepare_script else script_path('nfv3_conflict_clean.py')
    with (directory.parent / ('.' + directory.name + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if directory.exists():
            result = verify_clean_split(directory, reference)
            return {**result, 'action': 'reused'}
        print(f'Preparing conflict-clean split: {reference["dataset"]["dataset"]}', flush=True)
        temp = Path(tempfile.mkdtemp(prefix='.' + directory.name + '.', dir=directory.parent))
        output = temp / 'prepared'
        command = [sys.executable, '-u', str(script), '--dataset', reference['dataset']['dataset'],
                   '--data', str(data), '--out', str(output)]
        # Run preprocessing separately so its full source matrices are released
        # before the GPU worker builds model contexts.
        subprocess.run(command, check=True)
        result = verify_clean_split(output, reference)
        output.rename(directory)
        temp.rmdir()
        return {**result, 'directory': str(directory), 'action': 'generated'}

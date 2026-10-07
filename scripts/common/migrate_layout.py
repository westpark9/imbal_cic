#!/usr/bin/env python3
"""Manifest-based, reversible migration of the existing experiment layout.

plan inventories files without following symlinks. apply uses same-filesystem
renames and repairs recorded symlinks. verify checks every regular file's inode,
size and mtime, plus hashes of small result files. No model is executed.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / 'configs/experiments/migrations/20261007_layout'
MANIFEST = HOME / 'manifest.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def experiment(name):
    match = re.search(r'exp(\d+)', name)
    if match:
        return 'exp' + match[1]
    if 'resolved_expert' in name or name.startswith('s44'):
        return 's44'
    if 'independent_expert' in name or name.startswith('s43'):
        return 's43'
    if 'multiclass' in name:
        return 'multiclass'
    return None


def code_version(name):
    exp = experiment(name)
    if not exp or not exp.startswith('exp'):
        return 'v1'
    n = int(exp[3:])
    if n <= 21 or (n == 22 and '22c' not in name):
        return 'v1'
    if n <= 39:
        return 'v2'
    # Preserve the existing registry's date grouping for the 15 September EXP47.
    if n <= 47:
        return 'v3'
    if n <= 63:
        return 'v4'
    if n <= 68:
        return 'v5'
    return 'v7'


def translated(path, moves):
    value = str(path)
    for item in sorted(moves, key=lambda x: len(x['old']), reverse=True):
        old = item['old']
        if value == old or value.startswith(old + '/'):
            return item['new'] + value[len(old):]
    return value


def plan():
    if MANIFEST.exists():
        raise ValueError('Manifest already exists; inspect it instead of overwriting history')
    spec = importlib.util.spec_from_file_location('layout_old_versions', HOME / 'results_versions.before.py')
    rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)
    registry = rv.Registry(ROOT); versions = registry.load()
    prior = {r['path']: registry.version_of(r, versions) for r in registry.runs(versions)}
    moves = []
    keep = {'results_versions.py', 'sync_html_reports.py'}
    common = {'exp_utils.py', 'exp61_utils.py', 'exp57_prepare_inputs.py',
              'setup_exp57_env.py', 'setup_exp61_env.py', 'report_order.py',
              'report_typography.py', 'build_latex_results.py', 'render_run_tables.py',
              'analyze_nfv3_embedding.py', 'analyze_nfv3_temporal.py',
              'nfv3_expert_data_composition.py', 'materialize_nfv3_pools_s43.py',
              'race_budget_diag.py', 'race_offline_eval.py', 'race_scenario_label_diag.py',
              'build_merged_expert_report.py', 'check_merged_expert_report.py',
              'nfv3_v3_common.py', 'nfv3_v3_c0_context.py', 'cic2018_conflict_clean.py',
              'nfv3_conflict_clean.py', 'exp6x_context_common.py'}
    for base in ['scripts', 'tabpfn/scripts']:
        for p in sorted((ROOT/base).iterdir()):
            if not p.is_file() or p.name in keep:
                continue
            exp = experiment(p.name)
            is_common = p.name in common or p.name.startswith('preprocess_')
            if not is_common and not exp:
                raise ValueError(f'Unclassified script: {p}')
            version = None if is_common else code_version(p.name)
            new = f'{base}/common/{p.name}' if is_common else f'{base}/{version}/{exp}/{p.name}'
            moves.append(dict(old=str(p.relative_to(ROOT)), new=new, kind='code',
                              version=version, experiment=None if is_common else exp))
    for base in ['results', 'tabpfn/results', 'tabpfn/results_past']:
        for p in sorted((ROOT/base).iterdir()):
            if p.name in {'versions', 'VERSIONS.json', 'INDEX.md', '.versions.lock'}:
                continue
            if p.name == 'scripts_analysis':
                # This directory mixes cross-experiment aggregates, not raw runs.
                moves.append(dict(old=str(p.relative_to(ROOT)), new='results/common/analysis',
                                  kind='analysis', version=None, experiment=None))
                continue
            exp = experiment(p.name)
            if not exp:
                raise ValueError(f'Unclassified result: {p}')
            old = str(p.relative_to(ROOT))
            # Keep already-indexed research assignments, even if the experiment
            # crosses an historical version boundary. Past runs predate v2.
            version = prior.get(old, code_version(p.name))
            target_base = 'results' if base == 'results' else 'tabpfn/results'
            suffix = p.name if p.is_dir() else 'analysis/' + p.name
            moves.append(dict(old=old, new=f'{target_base}/{version}/{exp}/{suffix}',
                              kind='run' if p.is_dir() else 'analysis',
                              version=version, experiment=exp,
                              assignment='preserved_registry' if old in prior else 'historical_experiment'))
    targets = [x['new'] for x in moves]
    if len(targets) != len(set(targets)):
        raise ValueError('Duplicate destinations')
    files = []; links = []; dirs = []
    for item in moves:
        p = ROOT/item['old']
        if (ROOT/item['new']).exists():
            raise ValueError(f'Destination exists: {item["new"]}')
        paths = [p]
        if p.is_dir():
            paths = []
            for current, subdirs, names in os.walk(p, followlinks=False):
                dirs.append(str(Path(current).relative_to(ROOT)))
                for name in subdirs + names:
                    child = Path(current)/name
                    if child.is_symlink() or child.is_file(): paths.append(child)
        for child in paths:
            old = str(child.relative_to(ROOT)); new = translated(old, moves)
            if child.is_symlink():
                resolved = child.resolve()
                target = str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)
                links.append(dict(old=old, new=new, target=os.readlink(child),
                                  resolved=target, new_target=translated(target,moves), existed=child.exists()))
                continue
            s = child.stat()
            info = dict(old=old,new=new,kind=item['kind'],bytes=s.st_size,
                        inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns)
            if s.st_size <= 1_000_000 or item['kind'] == 'code':
                info['sha256'] = hashlib.sha256(child.read_bytes()).hexdigest()
            files.append(info)
            if item['kind'] == 'code':
                target = HOME/'originals'/old; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(child,target)
    data = dict(schema=1,state='planned',date='2026-10-07',versions=versions,
                moves=moves,files=files,symlinks=links,directories=dirs)
    write(MANIFEST,data)
    print(json.dumps(dict(moves=len(moves),files=len(files),bytes=sum(x['bytes'] for x in files),
                          symlinks=len(links),kinds=dict(Counter(x['kind'] for x in moves))),indent=2))


def apply():
    data=json.loads(MANIFEST.read_text())
    if data['state'] not in {'planned','moving'}:raise ValueError('Migration already applied')
    data['state']='moving';write(MANIFEST,data)
    for item in data['moves']:
        src=ROOT/item['old']; dst=ROOT/item['new']
        if not src.exists() and dst.exists():continue
        if dst.exists():raise ValueError(f'Refusing overwrite: {dst}')
        dst.parent.mkdir(parents=True,exist_ok=True)
        src.rename(dst)
    for link in data['symlinks']:
        p=ROOT/link['new']; target=Path(link['new_target'])
        if not target.is_absolute():target=ROOT/target
        if p.is_symlink():p.unlink()
        p.symlink_to(os.path.relpath(target,p.parent))
    data['state']='moved';write(MANIFEST,data)
    print('Moved directories/files and repaired recorded symlinks.')


def verify():
    data=json.loads(MANIFEST.read_text()); errors=[]; total=0
    expected={f['new'] for f in data['files'] if f['kind']!='code'} | {s['new'] for s in data['symlinks']}
    actual=set()
    for item in data['moves']:
        if item['kind']=='code':continue
        p=ROOT/item['new']
        if p.is_file():actual.add(item['new']);continue
        for current,dirs,names in os.walk(p,followlinks=False):
            for name in dirs+names:
                child=Path(current)/name
                if child.is_file() or child.is_symlink():actual.add(str(child.relative_to(ROOT)))
    errors.extend(f'Unexpected result file: {name}' for name in sorted(actual-expected))
    for f in data['files']:
        p=ROOT/f['new']
        if not p.is_file():errors.append(f'Missing: {p}');continue
        if f['kind']=='code':continue  # Active code is separately patched/tested.
        s=p.stat();total+=s.st_size
        if (s.st_size,s.st_ino,s.st_dev,s.st_mtime_ns)!=(f['bytes'],f['inode'],f['device'],f['mtime_ns']):
            errors.append(f'File changed: {p}')
        if 'sha256' in f and hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:
            errors.append(f'Hash changed: {p}')
    for link in data['symlinks']:
        p=ROOT/link['new']; target=Path(link['new_target'])
        if not target.is_absolute():target=ROOT/target
        if not p.is_symlink() or p.resolve()!=target.resolve() or (link['existed'] and not p.exists()):
            errors.append(f'Broken/wrong symlink: {p}')
    result=dict(ok=not errors,original_regular_files=len(data['files']),
                preserved_result_bytes=total,symlinks=len(data['symlinks']),errors=errors)
    write(HOME/'verification.json',result)
    print(json.dumps(result,indent=2))
    if errors:raise ValueError('Migration verification failed')


def seal():
    """Record post-migration code hashes so rollback cannot discard later edits."""
    data=json.loads(MANIFEST.read_text())
    files={x['new'] for x in data['moves'] if x['kind']=='code'}
    files.update(str(p.relative_to(HOME/'consumer_originals')) for p in (HOME/'consumer_originals').rglob('*') if p.is_file())
    files.update(['scripts/results_versions.py','configs/experiments/registry.json',
                  'scripts/common/experiment_paths.py','scripts/tests/test_layout_paths.py'])
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(files)}
    write(HOME/'completed_file_hashes.json',hashes)
    data['state']='complete';write(MANIFEST,data)
    print(f'Sealed {len(hashes)} active code/document/registry files for guarded rollback.')


def rollback():
    data=json.loads(MANIFEST.read_text())
    if data['state']!='complete':raise ValueError('Only a sealed migration supports automated rollback')
    verify()
    hashes=json.loads((HOME/'completed_file_hashes.json').read_text())
    changed=[name for name,digest in hashes.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changed:raise ValueError(f'Refusing to discard post-migration edits: {changed}')
    # Refuse to overwrite new work created at original paths.
    for item in data['moves']:
        if (ROOT/item['old']).exists():raise ValueError(f'Original path occupied: {item["old"]}')
    for link in data['symlinks']:
        p=ROOT/link['new'];p.unlink();p.symlink_to(link['target'])
    for item in reversed(data['moves']):
        src=ROOT/item['new'];dst=ROOT/item['old'];dst.parent.mkdir(parents=True,exist_ok=True)
        src.rename(dst)
        if item['kind']=='code':shutil.copy2(HOME/'originals'/item['old'],dst)
    for p in (HOME/'consumer_originals').rglob('*'):
        if p.is_file():shutil.copy2(p,ROOT/p.relative_to(HOME/'consumer_originals'))
    shutil.copy2(HOME/'results_versions.before.py',ROOT/'scripts/results_versions.py')
    versions=ROOT/'tabpfn/results/VERSIONS.json'
    if versions.is_symlink():versions.unlink()
    shutil.copy2(HOME/'versions.before.json',versions)
    shutil.copy2(HOME/'index.before.md',ROOT/'tabpfn/results/INDEX.md')
    for name,target in json.loads((HOME/'legacy_view_links.json').read_text()).items():
        p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(target)
    (ROOT/'configs/experiments/registry.json').unlink()
    (ROOT/'scripts/tests/test_layout_paths.py').unlink()
    data['state']='rolled_back';write(MANIFEST,data)
    print('Original paths, code, consumer documents and registry restored. Migration audit records retained.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','apply','verify','seal','rollback'])
    args=p.parse_args();globals()[args.action]()

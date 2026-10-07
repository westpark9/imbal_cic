"""Portable code/result paths for the versioned experiment layout.

No filesystem or third-party API is monkey-patched. Old paths resolve through
the migration aliases; historical result files and source snapshots stay intact.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import sys


def repo_root(start=None):
    start = Path(start or __file__).resolve()
    for parent in [start.parent, *start.parents]:
        if (parent/'configs/experiments/registry.json').is_file():
            return parent
        if (parent/'scripts/common/experiment_paths.py').is_file():
            return parent
    raise FileNotFoundError(f'Cannot locate experiment repository from {start}')


@lru_cache(maxsize=4)
def registry(root=None):
    root=Path(root) if root else repo_root()
    path=root/'configs/experiments/registry.json'
    return json.loads(path.read_text()) if path.exists() else {}


def bootstrap(root=None):
    """Expose existing bare module names after moving their files.

    The local tabpfn directory remains a workspace, not a Python package, so it
    cannot shadow the installed upstream TabPFN package.
    """
    root=Path(root) if root else repo_root()
    directories=[]
    for base in [root/'scripts',root/'tabpfn/scripts']:
        directories.extend([base,base/'common'])
        directories.extend(sorted(p for p in base.glob('v*/*') if p.is_dir()))
    for p in reversed(directories):
        if p.is_dir() and str(p) not in sys.path:sys.path.insert(0,str(p))
    return root


def resolve_path(value, root=None):
    root=Path(root) if root else repo_root()
    p=Path(value)
    if p.is_absolute():
        if not p.is_relative_to(root):return p
        key=str(p.relative_to(root))
    else:key=str(p)
    aliases=registry(str(root)).get('path_aliases',{})
    for old in sorted(aliases,key=len,reverse=True):
        if key==old or key.startswith(old+'/'):
            return root/(aliases[old]+key[len(old):])
    return root/key


def result_glob(pattern, root=None):
    """Match historical flat result globs against the registered nested runs."""
    import glob
    root=Path(root) if root else repo_root()
    mapped=resolve_path(pattern,root)
    direct=glob.glob(str(mapped))
    if direct:return sorted(direct)
    value=str(pattern)
    for prefix in ['tabpfn/results_past/','tabpfn/results/','results/']:
        if value.startswith(prefix):
            base='results' if prefix=='results/' else 'tabpfn/results'
            return sorted(glob.glob(str(root/base/'v*'/'*'/value[len(prefix):])))
    return []


def script_path(name, root=None):
    root=Path(root) if root else repo_root()
    p=resolve_path(name,root)
    if p.is_file():return p
    found=[root/x['path'] for x in registry(str(root)).get('scripts',[])
           if Path(x['path']).name==Path(name).name]
    if len(found)==1:return found[0]
    # Allow portable snapshots containing only a subset of the repository.
    found=[p for base in ['scripts','tabpfn/scripts']
           for p in (root/base).rglob(Path(name).name) if p.is_file()]
    if len(found)==1:return found[0]
    raise FileNotFoundError(f'Expected one script for {name}, found {len(found)}')


def result_root(script=None, scope=None):
    root=repo_root()
    path=Path(script or getattr(sys.modules.get('__main__'),'__file__','')).resolve()
    code=registry(str(root)).get('scripts',[])
    entry=next((x for x in code if root/x['path']==path),None)
    if entry and entry.get('version'):
        base='results' if (scope or entry.get('result_scope'))=='root' else 'tabpfn/results'
        return root/base/entry['version']/entry['experiment']
    # Common modules may be imported by notebook/test callers. A generic result
    # root is only a search base; managed runs require explicit version/experiment.
    return root/('results' if scope=='root' else 'tabpfn/results')


def snapshot_path(source_root, old_path):
    """Find a script in either an immutable flat snapshot or a new nested one."""
    source_root=Path(source_root)
    original=source_root/old_path
    if original.exists():return original
    project=repo_root()
    relative=resolve_path(old_path,project).relative_to(project)
    return source_root/relative


def read_record(path, root=None):
    """Resolve path-valued strings in a legacy JSON record, only in memory."""
    root=Path(root) if root else repo_root()
    aliases=registry(str(root)).get('path_aliases',{})
    def convert(value):
        if isinstance(value,list):return [convert(x) for x in value]
        if isinstance(value,dict):return {convert(k):convert(v) for k,v in value.items()}
        if isinstance(value,str):
            key=value[len(str(root))+1:] if value.startswith(str(root)+'/') else value
            if any(key==old or key.startswith(old+'/') for old in aliases):
                p=resolve_path(value,root)
                return str(p) if Path(value).is_absolute() else str(p.relative_to(root))
        return value
    return convert(json.loads(resolve_path(path,root).read_text()))

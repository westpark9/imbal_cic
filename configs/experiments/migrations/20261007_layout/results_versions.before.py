#!/usr/bin/env python3
"""Research-version views and automatic run records; raw result folders stay in place.

index | new --label LABEL --note NOTE | assign RUN vN
mark RUN --evidence selected|exploratory|not_adopted|invalid --note NOTE
run --version vN --label LABEL -- COMMAND ... {run_dir} ...

The run wrapper creates metadata/logs before execution and updates the index on exit.
Use {run_dir} as the worker's output argument, or read EXPERIMENT_RUN_DIR in new workers.
A zero exit is 'finished', not scientific validation. No experiment is launched by index.
"""
import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = {'versions': [
 {'id':'v1','label':'tabpfn_intro_and_tailguard','start':'2026-08-01','note':'TabPFN introduction and tailguard'},
 {'id':'v2','label':'racepfn_structure_c0_sota_smoke','start':'2026-08-26','note':'Residual experts, scorer/verifier, C0 and SOTA'},
 {'id':'v3','label':'data_quality_audit','start':'2026-09-11','note':'Contradiction and duplicate audit'},
 {'id':'v4','label':'conflict_free_expert_sv','start':'2026-09-16','note':'Clean-data SOTA, expert capability and S/V'},
 {'id':'v5','label':'context_optimisation_temporal','start':'2026-10-01','note':'Separate exploratory context/scenario runs; not automatically adopted'},
], 'pins':{}, 'run_annotations':{}}


def now(): return datetime.now(timezone.utc).isoformat()
def atomic(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n' if not isinstance(value,str) else value)
    os.replace(temp,path)

def safe_label(label):
    s=re.sub(r'[^A-Za-z0-9가-힣]+','_',label).strip('_')
    if not s: raise ValueError('Label must contain letters or numbers')
    return s

class Registry:
    def __init__(self, root):
        self.root=Path(root).resolve(); self.base=self.root/'tabpfn/results'
        self.base.mkdir(parents=True,exist_ok=True)
        self.manifest=self.base/'VERSIONS.json'; self.views=self.base/'versions'
    @contextmanager
    def lock(self):
        with (self.base/'.versions.lock').open('a') as fh:
            fcntl.flock(fh,fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(fh,fcntl.LOCK_UN)
    def load(self):
        m=json.loads(self.manifest.read_text()) if self.manifest.exists() else json.loads(json.dumps(DEFAULT))
        m.setdefault('pins',{});m.setdefault('run_annotations',{})
        self.validate(m); return m
    def validate(self,m):
        ids=[v['id'] for v in m['versions']]
        if len(ids)!=len(set(ids)) or not ids:raise ValueError('Duplicate or missing version IDs')
        for v in m['versions']:
            date.fromisoformat(v['start'])
            if safe_label(v['label'])!=v['label'] or not re.fullmatch(r'v\d+',v['id']):raise ValueError('Unsafe version identifier')
        if any(v not in ids for v in m['pins'].values()):raise ValueError('Pin references unknown version')
    def save(self,m):self.validate(m);atomic(self.manifest,m)
    def run_path(self,value):
        p=(self.root/value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        if not p.is_dir() or p.parent not in [self.base,self.root/'results']:raise ValueError('Run must be a direct result directory under this repository')
        return p
    def status(self,p):
        for name in ['RUN_META.json','status.json','COMPLETE.json']:
            f=p/name
            if not f.exists():continue
            try:d=json.loads(f.read_text())
            except (OSError,json.JSONDecodeError):return 'invalid_metadata'
            if not isinstance(d,dict):return 'unverified'
            status=str(d.get('status',d.get('state',''))).lower()
            if status in ('running','failed','interrupted','finished','complete','completed','success','invalid'):
                return {'completed':'complete','success':'complete'}.get(status,status)
            if d.get('complete') is True or name=='COMPLETE.json':return 'complete_marker'
            # Existence of status.json must not imply completion.
        if any((p/n).exists() for n in ('results.json','summary.csv')):return 'results_present'
        return 'partial_or_legacy'
    def runs(self,m):
        out=[]
        for base in [self.base,self.root/'results']:
            if not base.exists():continue
            for p in sorted(base.iterdir()):
                if not p.is_dir() or p.is_symlink() or p.name in ('versions','archive') or p.name.startswith('.'):continue
                match=re.search(r'(20\d{6})',p.name)
                d=match.group(1) if match else None
                try: day=date.fromisoformat(f'{d[:4]}-{d[4:6]}-{d[6:]}').isoformat() if d else None
                except ValueError:day=None
                key=str(p.relative_to(self.root));a=m['run_annotations'].get(key,{})
                tag=re.search(r'exp\d+[a-z0-9]*',p.name)
                out.append(dict(path=key,date=day,tag=tag.group() if tag else 'run',status=a.get('status',self.status(p)),evidence=a.get('evidence','unreviewed'),note=a.get('note',''),mtime=datetime.fromtimestamp(p.stat().st_mtime).date().isoformat()))
        return out
    def version_of(self,r,m):
        if r['path'] in m['pins']:return m['pins'][r['path']]
        versions=sorted(m['versions'],key=lambda v:(v['start'],int(v['id'][1:])))
        choices=[v for v in versions if v['start'] <= (r['date'] or r['mtime'])]
        return (choices[-1] if choices else versions[0])['id']
    def index(self,m):
        self.validate(m);rs=self.runs(m); by={v['id']:[] for v in m['versions']}
        for r in rs:by[self.version_of(r,m)].append(r)
        self.views.mkdir(exist_ok=True)
        desired={}
        lines=['# Experiment run index','','Generated view; raw directories are never moved. `results_present` is not validated completion.','Date-based grouping is a fallback; explicit pins identify research intent. New directions may be registered before experiments exist.','']
        for v in m['versions']:
            directory=self.views/f"{v['id']}_{v['label']}";directory.mkdir(exist_ok=True)
            lines += [f"## {v['id']} · {v['label']} ({v['start']})",v.get('note',''),'','| Date | Experiment | Execution status | Evidence | Run | Note |','|---|---|---|---|---|---|']
            for r in by[v['id']]:
                p=self.root/r['path']
                # Include source-root prefix to avoid two legacy directories sharing a name.
                name=('tabpfn__' if r['path'].startswith('tabpfn/') else 'root__')+p.name
                desired[directory/name]=os.path.relpath(p,directory)
                note=r['note'].replace('|','/').replace('\n',' ')
                lines.append(f"| {r['date'] or '?'} | {r['tag']} | {r['status']} | {r['evidence']} | `{r['path']}` | {note} |")
            lines.append('')
        # Clean only generated symlinks; never delete files/directories or raw results.
        for link in self.views.glob('*/*'):
            if link.is_symlink() and (link not in desired or os.readlink(link)!=desired[link]):link.unlink()
        for link,target in desired.items():
            if link.is_symlink():continue
            if link.exists():raise FileExistsError(f'Refusing to overwrite non-link {link}')
            link.symlink_to(target)
        atomic(self.base/'INDEX.md','\n'.join(lines)+'\n')
        print(f'Indexed {len(rs)} runs across {len(by)} research versions')
        return rs
    def execute(self,args):
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        if not command:raise ValueError('Missing command after --')
        with self.lock():
            m=self.load()
            if args.version not in {v['id'] for v in m['versions']}:raise ValueError('Unknown version')
            run=self.base/f"{datetime.now():%Y%m%d_%H%M%S}_{safe_label(args.label)}_{uuid.uuid4().hex[:8]}"
            run.mkdir();key=str(run.relative_to(self.root))
            command=[v.replace('{run_dir}',str(run)) for v in command]
            def git(*opts):
                return subprocess.run(['git',*opts],cwd=self.root,text=True,capture_output=True).stdout.strip()
            dirty=git('diff','--binary','HEAD'); git_status=git('status','--short')
            script_hashes={}
            for arg in command:
                candidate=Path(arg) if Path(arg).is_absolute() else self.root/arg
                if candidate.suffix in ('.py','.sh','.json','.yaml','.toml') and candidate.is_file():
                    script_hashes[arg]=hashlib.sha256(candidate.read_bytes()).hexdigest()
            meta=dict(schema=1,status='running',started=now(),version=args.version,command=command,cwd=str(self.root),protocol=args.protocol or None,git_head=git('rev-parse','HEAD'),git_status=git_status,git_diff_sha256=hashlib.sha256(dirty.encode()).hexdigest(),input_file_sha256=script_hashes)
            atomic(run/'RUN_META.json',meta)
            m['pins'][key]=args.version; m['run_annotations'][key]={'evidence':'unreviewed','note':'Managed command; zero exit does not validate results'}
            self.save(m);self.index(m)
        print(f'Run record: {key}',flush=True)
        env=os.environ.copy();env['EXPERIMENT_RUN_DIR']=str(run)
        rc=1
        try:
            with (run/'stdout.log').open('w') as log:
                proc=subprocess.Popen(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,env=env)
                try:rc=proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    rc=130
            meta['status']='finished' if rc==0 else ('interrupted' if rc==130 else 'failed')
        except OSError as exc:
            meta['status']='failed';meta['error']=str(exc)
        finally:
            meta.update(finished=now(),returncode=rc)
            atomic(run/'RUN_META.json',meta)
            with self.lock():self.index(self.load())
        return rc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT,help='Repository root; also permits isolated fixture testing')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('index')
    n=sub.add_parser('new');n.add_argument('--label',required=True);n.add_argument('--start',default=date.today().isoformat());n.add_argument('--note',default='')
    a=sub.add_parser('assign');a.add_argument('run');a.add_argument('version')
    k=sub.add_parser('mark');k.add_argument('run');k.add_argument('--evidence',choices=['selected','exploratory','not_adopted','invalid'],required=True);k.add_argument('--note',required=True);k.add_argument('--status',choices=['complete','failed','invalid'])
    x=sub.add_parser('run');x.add_argument('--version',required=True);x.add_argument('--label',required=True);x.add_argument('--protocol');x.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args(argv);r=Registry(args.root)
    if args.cmd=='run':return r.execute(args)
    with r.lock():
        m=r.load()
        if args.cmd=='new':
            date.fromisoformat(args.start);vid=f"v{max(int(v['id'][1:]) for v in m['versions'])+1}"
            m['versions'].append(dict(id=vid,label=safe_label(args.label),start=args.start,note=args.note));print('Opened',vid)
        elif args.cmd=='assign':
            if args.version not in {v['id'] for v in m['versions']}:raise ValueError('Unknown version')
            m['pins'][str(r.run_path(args.run).relative_to(r.root))]=args.version
        elif args.cmd=='mark':
            a=m['run_annotations'].setdefault(str(r.run_path(args.run).relative_to(r.root)),{})
            a.update(evidence=args.evidence,note=args.note)
            if args.status:a['status']=args.status
        r.save(m);r.index(m)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError) as e:print(f'Error: {e}',file=sys.stderr);sys.exit(2)

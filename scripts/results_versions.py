#!/usr/bin/env python3
"""Register, find and run experiments in results/vN/experiment/run_id.

Research versions change only after a user request/confirmation. The recorded
approval note documents that decision; it is not supplied automatically.
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
sys.path.insert(0, str(ROOT/'scripts/common'))
from experiment_paths import resolve_path


def now(): return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n' if not isinstance(value,str) else value)
    os.replace(temp,path)


def safe_label(label):
    result=re.sub(r'[^A-Za-z0-9가-힣]+','_',label).strip('_')
    if not result:raise ValueError('Label must contain letters or numbers')
    return result


class Registry:
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.base=self.root/'tabpfn/results'
        self.manifest=self.root/'configs/experiments/registry.json'
        self.manifest.parent.mkdir(parents=True,exist_ok=True)
        self.views=self.base/'versions'  # Legacy attribute; no duplicate symlink view is generated.

    @contextmanager
    def lock(self):
        with (self.manifest.parent/'.registry.lock').open('a') as fh:
            fcntl.flock(fh,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(fh,fcntl.LOCK_UN)

    def load(self):
        if self.manifest.exists():m=json.loads(self.manifest.read_text())
        elif (self.base/'VERSIONS.json').is_file():m=json.loads((self.base/'VERSIONS.json').read_text())
        else:m={'versions':[], 'schema':2}
        for key,default in [('pins',{}),('run_annotations',{}),('experiments',{}),('scripts',[]),('path_aliases',{})]:
            m.setdefault(key,default)
        self.validate(m);return m

    def validate(self,m):
        ids=[v['id'] for v in m['versions']]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate version IDs')
        for v in m['versions']:
            date.fromisoformat(v['start'])
            if safe_label(v['label'])!=v['label'] or not re.fullmatch(r'v\d+',v['id']):raise ValueError('Unsafe version identifier')
        if any(v not in ids for v in m['pins'].values()):raise ValueError('Pin references unknown version')

    def save(self,m):self.validate(m);atomic(self.manifest,m)

    def run_path(self,value):
        p=resolve_path(value,self.root).resolve()
        roots=[self.base,self.root/'results']
        if not p.is_dir() or not any(p.is_relative_to(b) and p!=b for b in roots):
            raise ValueError('Run must be an existing result directory in this repository')
        return p

    def status(self,p):
        for name in ['RUN_META.json','status.json','COMPLETE.json']:
            f=p/name
            if not f.exists():continue
            try:d=json.loads(f.read_text())
            except (OSError,json.JSONDecodeError):return 'invalid_metadata'
            if not isinstance(d,dict):return 'unverified'
            status=str(d.get('status',d.get('state',''))).lower()
            if status in ('running','failed','interrupted','finished','complete','completed','success','invalid',
                          'completed_with_failures','complete_with_errors','prepared'):
                return {'completed':'complete','success':'complete'}.get(status,status)
            if d.get('complete') is True or name=='COMPLETE.json':return 'complete_marker'
        if any((p/n).exists() for n in ('results.json','summary.csv')):return 'results_present'
        return 'partial_or_legacy'

    def runs(self,m):
        paths=set(m['pins'])
        ids={v['id'] for v in m['versions']}
        for base in [self.base,self.root/'results']:
            if not base.exists():continue
            for p in base.iterdir():
                if not p.is_dir() or p.is_symlink() or p.name in {'common','versions','archive'} or p.name.startswith('.'):continue
                if p.name in ids:
                    for exp in p.iterdir():
                        if not exp.is_dir():continue
                        for run in exp.iterdir():
                            if run.is_dir() and not run.is_symlink() and run.name not in {'analysis','source','__pycache__'}:
                                paths.add(str(run.relative_to(self.root)))
                else:paths.add(str(p.relative_to(self.root)))
        out=[]
        for key in sorted(paths):
            p=self.root/key
            if not p.is_dir():continue
            a=m['run_annotations'].get(key,{})
            parts=p.relative_to(self.base if p.is_relative_to(self.base) else self.root/'results').parts
            nested=len(parts)>=3 and parts[0] in ids
            match=re.search(r'(20\d{6})',p.name)
            day=None
            if match:
                try:day=date.fromisoformat(match.group()).isoformat()
                except ValueError:pass
            out.append(dict(path=key,date=day,tag=parts[1] if nested else 'unassigned',
                            version=m['pins'].get(key,parts[0] if nested else None),
                            status=a.get('status',self.status(p)),evidence=a.get('evidence','unreviewed'),note=a.get('note','')))
        return out

    def version_of(self,r,m):return r.get('version') or m['pins'].get(r['path'])

    def index(self,m):
        self.validate(m);runs=self.runs(m)
        lines=['# Experiment index','','Research version changes require user confirmation. Execution success and evidence adoption are separate.','']
        for v in m['versions']:
            lines += [f"## {v['id']} · {v['label']}",v.get('note',''),'','| Experiment | Date | Status | Evidence | Run |','|---|---|---|---|---|']
            for r in runs:
                if self.version_of(r,m)!=v['id']:continue
                lines.append(f"| {r['tag']} | {r['date'] or '?'} | {r['status']} | {r['evidence']} | `{r['path']}` |")
            lines.append('')
        unassigned=[r['path'] for r in runs if not self.version_of(r,m)]
        if unassigned:lines+=['## Unassigned legacy paths','',*['- `'+x+'`' for x in unassigned],'']
        text='\n'.join(lines)+'\n'
        atomic(self.manifest.parent/'INDEX.md',text)
        for base in [self.base,self.root/'results']:
            base.mkdir(parents=True,exist_ok=True);atomic(base/'INDEX.md',text)
        for base in [self.root/'scripts',self.root/'tabpfn/scripts']:
            entries=[x for x in m['scripts'] if (self.root/x['path']).is_relative_to(base)]
            if base==self.root/'scripts':entries=[x for x in entries if not x['path'].startswith('tabpfn/')]
            lines=['# Code index','','| Version | Experiment | Code |','|---|---|---|']
            lines += [f"| {x.get('version') or 'common'} | {x.get('experiment') or 'shared'} | [{Path(x['path']).name}]({os.path.relpath(self.root/x['path'],base)}) |" for x in entries]
            atomic(base/'INDEX.md','\n'.join(lines)+'\n')
        print(f'Indexed {len(runs)} runs across {len(m["versions"])} research versions; {len(unassigned)} unassigned')
        return runs

    def register(self,m,version,experiment,scope,entrypoint=None,protocol=None,note=''):
        if version not in {v['id'] for v in m['versions']}:raise ValueError('Unknown research version')
        if safe_label(experiment)!=experiment:raise ValueError('Unsafe experiment identifier')
        key=f'{version}/{experiment}'
        record=m['experiments'].setdefault(key,dict(version=version,id=experiment,scopes=[],entrypoints=[]))
        if scope not in record['scopes']:record['scopes'].append(scope)
        if entrypoint:
            p=resolve_path(entrypoint,self.root)
            if not p.is_file():raise ValueError(f'Missing entrypoint: {entrypoint}')
            value=str(p.relative_to(self.root))
            if value not in record['entrypoints']:record['entrypoints'].append(value)
        if protocol:record['protocol']=protocol
        if note:record['note']=note
        return record

    def execute(self,args):
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        if not command:raise ValueError('Missing command after --')
        with self.lock():
            m=self.load();self.register(m,args.version,args.experiment,args.scope,protocol=args.protocol)
            base=self.root/('results' if args.scope=='root' else 'tabpfn/results')
            run=base/args.version/args.experiment/f'{datetime.now():%Y%m%d_%H%M%S}_{safe_label(args.label)}_{uuid.uuid4().hex[:8]}'
            run.mkdir(parents=True);key=str(run.relative_to(self.root))
            command=[v.replace('{run_dir}',str(run)) for v in command]
            command=[str(resolve_path(v,self.root)) if v.endswith('.py') and resolve_path(v,self.root).is_file() else v for v in command]
            def git(*opts):return subprocess.run(['git',*opts],cwd=self.root,text=True,capture_output=True).stdout.strip()
            dirty=git('diff','--binary','HEAD');files={}
            for arg in command:
                candidate=Path(arg) if Path(arg).is_absolute() else self.root/arg
                if candidate.suffix in ('.py','.sh','.json','.yaml','.toml') and candidate.is_file():
                    files[arg]=hashlib.sha256(candidate.read_bytes()).hexdigest()
            meta=dict(schema=2,status='running',started=now(),version=args.version,experiment=args.experiment,
                      scope=args.scope,command=command,cwd=str(self.root),protocol=args.protocol or None,
                      git_head=git('rev-parse','HEAD'),git_status=git('status','--short'),
                      git_diff_sha256=hashlib.sha256(dirty.encode()).hexdigest(),input_file_sha256=files)
            atomic(run/'RUN_META.json',meta)
            m['pins'][key]=args.version;m['run_annotations'][key]={'evidence':'unreviewed','note':'Managed execution; results require scientific review'}
            self.save(m);self.index(m)
        print(f'Run record: {key}',flush=True)
        env=dict(os.environ,EXPERIMENT_RUN_DIR=str(run));rc=1
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
        except OSError as exc:meta.update(status='failed',error=str(exc))
        finally:
            meta.update(finished=now(),returncode=rc);atomic(run/'RUN_META.json',meta)
            with self.lock():self.index(self.load())
        return rc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    sub=p.add_subparsers(dest='cmd',required=True);sub.add_parser('index')
    n=sub.add_parser('new');n.add_argument('--label',required=True);n.add_argument('--start',default=date.today().isoformat());n.add_argument('--note',default='');n.add_argument('--approval-note',required=True,help='Actual user request/confirmation; never infer from dates or run count')
    reg=sub.add_parser('register');reg.add_argument('--version',required=True);reg.add_argument('--experiment',required=True);reg.add_argument('--scope',choices=['root','tabpfn'],default='tabpfn');reg.add_argument('--entrypoint');reg.add_argument('--protocol');reg.add_argument('--note',default='')
    a=sub.add_parser('assign');a.add_argument('run');a.add_argument('version');a.add_argument('--approval-note',required=True)
    k=sub.add_parser('mark');k.add_argument('run');k.add_argument('--evidence',choices=['selected','exploratory','not_adopted','invalid'],required=True);k.add_argument('--note',required=True);k.add_argument('--status',choices=['complete','failed','invalid'])
    x=sub.add_parser('run');x.add_argument('--version',required=True);x.add_argument('--experiment',required=True);x.add_argument('--scope',choices=['root','tabpfn'],default='tabpfn');x.add_argument('--label',required=True);x.add_argument('--protocol');x.add_argument('command',nargs=argparse.REMAINDER)
    q=sub.add_parser('find');q.add_argument('query')
    w=sub.add_parser('where');w.add_argument('path')
    args=p.parse_args(argv);r=Registry(args.root)
    if args.cmd=='where':print(resolve_path(args.path,r.root));return 0
    if args.cmd=='find':
        m=r.load();q=args.query
        print(json.dumps(dict(experiments={k:v for k,v in m['experiments'].items() if q in k},
                              scripts=[s for s in m['scripts'] if q in (s.get('experiment') or '') or q in s['path']],
                              runs=[s for s in r.runs(m) if q in s['tag'] or q in s['path']]),ensure_ascii=False,indent=2));return 0
    if args.cmd=='run':return r.execute(args)
    with r.lock():
        m=r.load()
        if args.cmd=='new':
            if not args.approval_note.strip():raise ValueError('User approval note cannot be empty')
            date.fromisoformat(args.start);vid=f'v{max([int(v["id"][1:]) for v in m["versions"]],default=0)+1}'
            m['versions'].append(dict(id=vid,label=safe_label(args.label),start=args.start,note=args.note,approval_note=args.approval_note,registered=now()));print('Opened',vid)
        elif args.cmd=='register':r.register(m,args.version,args.experiment,args.scope,args.entrypoint,args.protocol,args.note)
        elif args.cmd=='assign':
            if args.version not in {v['id'] for v in m['versions']}:raise ValueError('Unknown version')
            path=r.run_path(args.run);key=str(path.relative_to(r.root))
            parent_version=path.parent.parent.name
            if re.fullmatch(r'v\d+',parent_version) and parent_version!=args.version:
                raise ValueError('Physical research-version reassignment requires a reviewed migration manifest')
            if not args.approval_note.strip():raise ValueError('User approval note cannot be empty')
            m['pins'][key]=args.version;m['run_annotations'].setdefault(key,{})['assignment_approval']=args.approval_note
        elif args.cmd=='mark':
            note=m['run_annotations'].setdefault(str(r.run_path(args.run).relative_to(r.root)),{})
            note.update(evidence=args.evidence,note=args.note)
            if args.status:note['status']=args.status
        r.save(m);r.index(m)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError) as e:print(f'Error: {e}',file=sys.stderr);sys.exit(2)

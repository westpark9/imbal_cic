"""Fixture-only integration tests for the nested registry; never runs a model."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'results_versions.py'
spec=importlib.util.spec_from_file_location('rv',SCRIPT);rv=importlib.util.module_from_spec(spec);spec.loader.exec_module(rv)


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.r=rv.Registry(self.root)
        m=self.r.load();m['versions']=[dict(id='v1',label='fixture',start='2026-01-01',approval_note='fixture approval')];self.r.save(m)
    def tearDown(self):self.tmp.cleanup()
    def folder(self,name,scope='tabpfn/results'):
        p=self.root/scope/name;p.mkdir(parents=True);return p
    def cli(self,*args):return subprocess.run([sys.executable,str(SCRIPT),'--root',str(self.root),*args],text=True,capture_output=True)
    def run_args(self,label,scope='tabpfn'):
        return ['run','--version','v1','--experiment','exp1','--scope',scope,'--label',label,'--']
    def test_status_does_not_infer_completion_from_presence(self):
        p=self.folder('v1/exp1/20261007_test');(p/'status.json').write_text('{"status":"failed"}')
        self.assertEqual(self.r.status(p),'failed')
        (p/'status.json').write_text('{"progress":2}');(p/'results.json').write_text('{"partial":true}')
        self.assertEqual(self.r.status(p),'results_present')
    def test_nested_scan_does_not_count_children_or_analyses(self):
        p=self.folder('v1/exp1/20261007_run')
        (p/'source/scripts').mkdir(parents=True);(p/'dataset/job').mkdir(parents=True)
        self.folder('v1/exp1/analysis');self.folder('common/analysis')
        rs=self.r.index(self.r.load())
        self.assertEqual([r['path'] for r in rs],[str(p.relative_to(self.root))])
        self.assertFalse(self.r.views.exists())
    def test_no_date_based_version_assignment(self):
        self.folder('20990101_legacy')
        runs=self.r.runs(self.r.load());self.assertIsNone(runs[0]['version'])
    def test_unknown_version_does_not_change_manifest(self):
        before=self.r.manifest.read_bytes()
        c=self.cli('run','--version','v999','--experiment','exp1','--label','invalid','--',sys.executable,'-c','print(1)')
        self.assertNotEqual(c.returncode,0);self.assertEqual(self.r.manifest.read_bytes(),before)
    def test_new_research_version_requires_approval_note(self):
        before=self.r.manifest.read_bytes()
        c=self.cli('new','--label','unapproved');self.assertNotEqual(c.returncode,0)
        self.assertEqual(self.r.manifest.read_bytes(),before)
        c=self.cli('new','--label','approved','--approval-note','User explicitly requested the fixture version')
        self.assertEqual(c.returncode,0,c.stderr);self.assertEqual(len(self.r.load()['versions']),2)
    def test_wrapper_success_failure_and_scope(self):
        code="import os,sys;from pathlib import Path;p=Path(sys.argv[1]);assert str(p)==os.environ['EXPERIMENT_RUN_DIR'];(p/'result.txt').write_text('ok');print('captured')"
        a=self.cli(*self.run_args('ok'),sys.executable,'-c',code,'{run_dir}')
        self.assertEqual(a.returncode,0,a.stderr)
        b=self.cli(*self.run_args('failure','root'),sys.executable,'-c','import sys;sys.exit(7)')
        self.assertEqual(b.returncode,7,b.stderr)
        metas=[json.loads(f.read_text()) for base in ['tabpfn/results','results'] for f in (self.root/base).glob('v1/exp1/*/RUN_META.json')]
        self.assertEqual({m['status'] for m in metas},{'finished','failed'})
        self.assertEqual({m['returncode'] for m in metas},{0,7})
        self.assertEqual({m['scope'] for m in metas},{'root','tabpfn'})
        self.assertEqual({a['evidence'] for a in self.r.load()['run_annotations'].values()},{'unreviewed'})
    def test_concurrent_registration_is_atomic(self):
        cmd=[sys.executable,str(SCRIPT),'--root',str(self.root),*self.run_args('parallel'),sys.executable,'-c','print("ok")']
        children=[subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
        for child in children:
            out,err=child.communicate(timeout=20);self.assertEqual(child.returncode,0,err)
        self.assertEqual(len(self.r.load()['pins']),2)
        self.assertEqual(len(self.r.runs(self.r.load())),2)
    def test_old_path_resolves_without_compatibility_directories(self):
        p=self.folder('v1/exp1/20261007_saved');(p/'raw.txt').write_text('original')
        m=self.r.load();m['path_aliases']['tabpfn/results/old']=str(p.relative_to(self.root));self.r.save(m)
        c=self.cli('where','tabpfn/results/old/raw.txt')
        self.assertEqual(c.stdout.strip(),str(p/'raw.txt'))
        c=self.cli('mark','tabpfn/results/old','--evidence','exploratory','--note','fixture')
        self.assertEqual(c.returncode,0,c.stderr)
        self.assertEqual((p/'raw.txt').read_text(),'original');self.assertFalse((self.r.base/'old').exists())


if __name__=='__main__':unittest.main()

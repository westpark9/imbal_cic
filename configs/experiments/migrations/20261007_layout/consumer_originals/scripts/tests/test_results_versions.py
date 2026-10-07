"""Fixture-only registry integration checks; no model/data/GPU dependencies."""
import importlib.util
import json
import os
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
 def tearDown(self):self.tmp.cleanup()
 def folder(self,name,root='tabpfn/results'):
  p=self.root/root/name;p.mkdir(parents=True);return p
 def cli(self,*args):return subprocess.run([sys.executable,str(SCRIPT),'--root',str(self.root),*args],text=True,capture_output=True)
 def test_status_does_not_infer_completion_from_file_presence(self):
  p=self.folder('20261001_test');(p/'status.json').write_text('{"status":"failed"}')
  self.assertEqual(self.r.status(p),'failed')
  (p/'status.json').write_text('{"progress":2}');(p/'results.json').write_text('{"partial":true}')
  self.assertEqual(self.r.status(p),'results_present')
 def test_reassignment_is_portable_and_keeps_raw_files(self):
  a=self.folder('20261001_same');b=self.folder('20261001_same','results');(a/'raw.txt').write_text('original')
  m=self.r.load();self.r.index(m)
  links=list(self.r.views.glob('*/*'));self.assertEqual(len(links),2)
  self.assertTrue(all(x.is_symlink() and not os.path.isabs(os.readlink(x)) for x in links))
  m['pins'][str(a.relative_to(self.root))]='v4';self.r.index(m);self.r.index(m)
  links=list(self.r.views.glob('*/*'));self.assertEqual(len(links),2)
  self.assertEqual(sum(x.resolve()==a for x in links),1)
  self.assertEqual((a/'raw.txt').read_text(),'original');self.assertTrue(b.exists())
 def test_unknown_pin_rejected_without_manifest_change(self):
  p=self.folder('20261001_a');self.r.save(self.r.load());before=self.r.manifest.read_bytes()
  c=self.cli('assign',str(p),'v999');self.assertNotEqual(c.returncode,0)
  self.assertEqual(self.r.manifest.read_bytes(),before)
 def test_wrapper_success_and_failure(self):
  code="import os,sys;from pathlib import Path;p=Path(sys.argv[1]);assert str(p)==os.environ['EXPERIMENT_RUN_DIR'];(p/'result.txt').write_text('ok');print('captured')"
  p=self.cli('run','--version','v5','--label','fixture','--',sys.executable,'-c',code,'{run_dir}')
  self.assertEqual(p.returncode,0,p.stderr)
  q=self.cli('run','--version','v5','--label','failure','--',sys.executable,'-c','import sys;print("failure");sys.exit(7)')
  self.assertEqual(q.returncode,7,q.stderr)
  metas=[json.loads(f.read_text()) for f in self.r.base.glob('*/RUN_META.json')]
  self.assertEqual({m['status'] for m in metas},{'finished','failed'})
  self.assertEqual({m['returncode'] for m in metas},{0,7})
  self.assertEqual(len(self.r.load()['pins']),2)
 def test_parallel_wrappers_register_both(self):
  cmd=[sys.executable,str(SCRIPT),'--root',str(self.root),'run','--version','v5','--label','parallel','--',sys.executable,'-c','print("ok")']
  children=[subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
  for child in children:
   out,err=child.communicate(timeout=20);self.assertEqual(child.returncode,0,err)
  self.assertEqual(len(self.r.load()['pins']),2)
  self.assertEqual(len(list(self.r.views.glob('*/*'))),2)

if __name__=='__main__':unittest.main()

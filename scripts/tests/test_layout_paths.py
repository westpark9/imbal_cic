"""Migration-specific path and portable-snapshot checks; no GPU inference."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/common'))
from experiment_paths import bootstrap,read_record,resolve_path,result_glob,script_path,snapshot_path
bootstrap(ROOT)


class LayoutPathsTests(unittest.TestCase):
    def test_current_experiment_aliases_and_cache_links_exist(self):
        for exp,day in [('exp59','20260928'),('exp61','20260929'),('exp62','20260930'),('exp63','20260930'),('exp69','20261002'),('exp70','20261002')]:
            registry=json.loads((ROOT/'configs/experiments/registry.json').read_text())
            matches=[old for old in registry['path_aliases'] if old.startswith(f'tabpfn/results/{day}_{exp}_')]
            self.assertEqual(len(matches),1)
            self.assertTrue(resolve_path(matches[0]).is_dir())
        old='tabpfn/results/20260930_exp63_k_sweep_s43/toniot_k2/cache/eval_y.npy'
        self.assertTrue(resolve_path(old).is_file())
    def test_read_record_relocates_paths_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json'
            old='tabpfn/results/20261002_exp70_class_arrival_100k_s43/summary.csv'
            payload=dict(relative=old,absolute=str(ROOT/old),score=0.25,note='a scientific statement')
            p.write_text(json.dumps(payload));before=p.read_bytes()
            loaded=read_record(p)
            self.assertEqual(loaded['relative'],str(resolve_path(old).relative_to(ROOT)))
            self.assertEqual(loaded['absolute'],str(resolve_path(old)))
            self.assertEqual(loaded['score'],0.25);self.assertEqual(p.read_bytes(),before)
    def test_old_glob_finds_nested_runs(self):
        paths=result_glob('tabpfn/results/20260930_exp63*')
        self.assertEqual(len(paths),1);self.assertIn('/v4/exp63/',paths[0])
    def test_snapshot_path_supports_both_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old='scripts/run_exp63_local.py'
            current=script_path('run_exp63_local.py').relative_to(ROOT)
            (root/current).parent.mkdir(parents=True);(root/current).touch()
            self.assertEqual(snapshot_path(root,old),root/current)
            (root/old).parent.mkdir(parents=True,exist_ok=True);(root/old).touch()
            self.assertEqual(snapshot_path(root,old),root/old)
    def test_portable_snapshot_starts_outside_repository(self):
        from run_exp61_sota import snapshot
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'source';snapshot(out)
            for name in ['exp69_class_arrival.py','run_exp63_local.py']:
                relative=script_path(name).relative_to(ROOT)
                c=subprocess.run([sys.executable,str(out/relative),'--help'],cwd=tmp,capture_output=True,text=True,timeout=40)
                self.assertEqual(c.returncode,0,c.stderr)
            self.assertTrue((out/'configs/experiments/registry.json').is_file())


if __name__=='__main__':unittest.main()

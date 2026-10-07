"""Small filesystem fixtures for migration integrity and guarded rollback."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('migration',ROOT/'scripts/common/migrate_layout.py')
migration=importlib.util.module_from_spec(spec);spec.loader.exec_module(migration)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        migration.ROOT=self.root;migration.HOME=self.root/'configs/experiments/migrations/fixture';migration.MANIFEST=migration.HOME/'manifest.json'
        self.home=migration.HOME;self.home.mkdir(parents=True)
        self.put('scripts/old.py','original source')
        self.put('tabpfn/results/old/raw.bin','original raw bytes')
        (self.root/'tabpfn/results/old/link.bin').symlink_to('raw.bin')
        self.put('scripts/results_versions.py','new registry implementation')
        self.put('configs/experiments/registry.json','{}')
        self.put('scripts/common/experiment_paths.py','helper')
        self.put('scripts/tests/test_layout_paths.py','test')
        self.put('tabpfn/results/VERSIONS.json','{}')
        for name,value in [('results_versions.before.py','old registry'),('versions.before.json','{}'),('index.before.md','old index'),('legacy_view_links.json','{}')]:
            (self.home/name).write_text(value)
        backup=self.home/'originals/scripts/old.py';backup.parent.mkdir(parents=True);shutil.copy2(self.root/'scripts/old.py',backup)
        (self.home/'consumer_originals').mkdir()
        moves=[dict(old='scripts/old.py',new='scripts/v1/exp1/old.py',kind='code'),dict(old='tabpfn/results/old',new='tabpfn/results/v1/exp1/old',kind='run')]
        files=[]
        for old,new,kind in [('scripts/old.py','scripts/v1/exp1/old.py','code'),('tabpfn/results/old/raw.bin','tabpfn/results/v1/exp1/old/raw.bin','run')]:
            p=self.root/old;s=p.stat();files.append(dict(old=old,new=new,kind=kind,bytes=s.st_size,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        link=dict(old='tabpfn/results/old/link.bin',new='tabpfn/results/v1/exp1/old/link.bin',target='raw.bin',new_target='tabpfn/results/v1/exp1/old/raw.bin',existed=True)
        migration.write(migration.MANIFEST,dict(state='planned',moves=moves,files=files,symlinks=[link]))
    def tearDown(self):self.temp.cleanup()
    def put(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    def test_move_verify_and_full_rollback_preserve_original_bytes(self):
        migration.apply();migration.verify()
        self.put('scripts/v1/exp1/old.py','patched source')
        migration.seal();migration.rollback()
        self.assertEqual((self.root/'scripts/old.py').read_text(),'original source')
        self.assertEqual((self.root/'tabpfn/results/old/link.bin').read_text(),'original raw bytes')
        self.assertEqual((self.root/'scripts/results_versions.py').read_text(),'old registry')
    def test_rollback_refuses_later_code_or_result_edits(self):
        migration.apply();migration.seal()
        self.put('scripts/v1/exp1/old.py','later user edit')
        with self.assertRaisesRegex(ValueError,'post-migration edits'):migration.rollback()
        self.assertFalse((self.root/'scripts/old.py').exists())
        self.put('tabpfn/results/v1/exp1/old/new.bin','later run artifact')
        with self.assertRaisesRegex(ValueError,'verification failed'):migration.verify()


if __name__=='__main__':unittest.main()

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from exp57_prepare_inputs import ensure_clean_split,verify_clean_split,sha256


def fixture(directory,source):
    directory.mkdir(parents=True,exist_ok=True)
    for split,ids in [('train',[0,1]),('val',[2]),('test',[3])]:
        np.save(directory/(split+'_idx.npy'),np.array(ids))
    expected=dict(dataset='cic2018',class_names=['benign','attack'],feature_count=2,
                  rule='drop conflict groups',split_rule='preserve original IDs',
                  split_rows=dict(train=2,val=1,test=1),rows_before=6,rows_removed=2,rows_after=4,
                  artifacts_sha256={p.name:sha256(p) for p in directory.glob('*_idx.npy')})
    source_info=dict(bytes=len(source),sha256=hashlib.sha256(source).hexdigest())
    manifest={**expected,'source':source_info,'verification':{'remaining_conflicting_vector_groups':0}}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    (directory/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':sha256(directory/'manifest.json')}))
    (directory/'scenario_counts.csv').write_text('split,scenario,after\ntest,a,1\n')
    return dict(source=source_info,dataset=expected)


class AutoCleanInputsTests(unittest.TestCase):
    def test_existing_cache_is_reused_without_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'existing.pkl';data.write_bytes(b'raw-data')
            target=root/'clean';ref=fixture(target,data.read_bytes())
            with patch('exp57_prepare_inputs.subprocess.run') as run:
                result=ensure_clean_split(data,target,ref)
            run.assert_not_called();self.assertEqual(result['action'],'reused')

    def test_rejects_self_consistent_but_different_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref=fixture(root,b'raw-data')
            np.save(root/'val_idx.npy',np.array([0]))
            manifest=json.loads((root/'manifest.json').read_text())
            manifest['artifacts_sha256']['val_idx.npy']=sha256(root/'val_idx.npy')
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':sha256(root/'manifest.json')}))
            with self.assertRaisesRegex(ValueError,'split identity'):verify_clean_split(root,ref)

    def test_missing_cache_is_generated_and_published_after_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'existing.pkl';data.write_bytes(b'raw-data')
            ref=fixture(root/'reference',data.read_bytes());target=root/'generated'
            def prepare(command,check):
                self.assertFalse(target.exists());self.assertTrue(check)
                self.assertEqual(command[command.index('--data')+1],str(data))
                fixture(Path(command[command.index('--out')+1]),data.read_bytes())
            with patch('exp57_prepare_inputs.subprocess.run',side_effect=prepare) as run:
                result=ensure_clean_split(data,target,ref)
            run.assert_called_once();self.assertEqual(result['action'],'generated')
            self.assertTrue((target/'COMPLETE.json').is_file())
            verify_clean_split(target,ref)

    def test_failed_preparation_does_not_publish_partial_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'existing.pkl';data.write_bytes(b'raw-data')
            ref=fixture(root/'reference',data.read_bytes());target=root/'generated'
            with patch('exp57_prepare_inputs.subprocess.run',side_effect=subprocess.CalledProcessError(1,'prepare')):
                with self.assertRaises(subprocess.CalledProcessError):ensure_clean_split(data,target,ref)
            self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nfv3_conflict_clean import install_clean_loader


class PortableInputsTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'moved.pkl';data.write_bytes(b'fixture-data')
        names=['benign','attack'];splits=[np.array([0,1]),np.array([2]),np.array([3])]
        label=lambda idx:np.asarray(idx)%2
        core=types.SimpleNamespace(load_cic2018=lambda args:(np.zeros((4,2)),names,*splits,label(splits[0]),label(splits[2]),pd.DataFrame(),label))
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        for key,ids in zip(['train','val','test'],splits):np.save(root/(key+'_idx.npy'),ids)
        pd.DataFrame([dict(split='test',scenario='test',after=1)]).to_csv(root/'scenario_counts.csv',index=False)
        manifest=dict(dataset='cic2018',class_names=names,source=dict(path='/original/host/source.pkl',bytes=data.stat().st_size,mtime_ns=1,sha256=sha(data)),
                      artifacts_sha256={p.name:sha(p) for p in root.glob('*_idx.npy')})
        m=root/'manifest.json';m.write_text(json.dumps(manifest));(root/'COMPLETE.json').write_text(json.dumps(dict(manifest_sha256=sha(m))))
        return core,data,m

    def test_relocation_accepts_same_bytes_and_preserves_split_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            core,data,m=self.fixture(Path(temp));install_clean_loader(core,m,source_override=data)
            actual=core.load_cic2018(types.SimpleNamespace(data=str(data)))
            np.testing.assert_array_equal(actual[2],[0,1]);np.testing.assert_array_equal(actual[4],[3])

    def test_relocation_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            core,data,m=self.fixture(Path(temp));data.write_bytes(b'changed-data')
            with self.assertRaisesRegex(ValueError,'content hash'):install_clean_loader(core,m,source_override=data)

    def test_default_strict_path_check_and_split_hash_remain_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);core,data,m=self.fixture(root);install_clean_loader(core,m)
            with self.assertRaisesRegex(ValueError,'Source dataset identity'):core.load_cic2018(types.SimpleNamespace(data=str(data)))
            core,data,m=self.fixture(root);install_clean_loader(core,m,source_override=data)
            np.save(root/'test_idx.npy',np.array([2]))
            with self.assertRaisesRegex(ValueError,'Clean index identity'):core.load_cic2018(types.SimpleNamespace(data=str(data)))


if __name__=='__main__':unittest.main()

"""Check frozen identities, posterior adjustment, and the cost-safe BoostPFN port."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tabpfn/scripts'))
from exp61_utils import context_ids
from exp61_sota_worker import dist_adjust


class ProtocolTests(unittest.TestCase):
    def test_frozen_contexts(self):
        for dataset in ['cic2018','toniot']:
            ids=context_ids(ROOT,dataset)
            self.assertEqual(len(ids),100000)
            self.assertEqual(len(np.unique(ids)),100000)

    def test_dist_adjust_uses_context_prior_and_unlabelled_posteriors(self):
        probabilities=np.array([[.8,.2],[.4,.6]],dtype='float32')
        adjusted,prior,target=dist_adjust(probabilities,np.array([0,0,0,1]))
        np.testing.assert_allclose(prior,[.75,.25])
        np.testing.assert_allclose(target,[.6,.4],atol=1e-7)
        np.testing.assert_allclose(adjusted,[[2/3,1/3],[.25,.75]],atol=1e-7)
        np.testing.assert_allclose(adjusted.sum(1),1)


@unittest.skipUnless(os.environ.get('EXP61_GPU_TESTS')=='1','opt-in GPU parity tests')
class GPUParityTests(unittest.TestCase):
    def test_localpfn_query_labels_do_not_enter_prediction(self):
        import importlib.util
        import torch
        path=ROOT/'tabpfn/third_party/LoCalPFN/pfn.py'
        spec=importlib.util.spec_from_file_location('exp61_test_local_pfn',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model,_=module.PFN.load_old(path=path.parent/'models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt',device='cuda:0')
        model.eval();torch.manual_seed(3)
        X=torch.randn(20,1,100,device='cuda:0');y=torch.arange(20,device='cuda:0',dtype=torch.float32).reshape(20,1)%3
        changed=y.clone();changed[17:]=(changed[17:]+1)%3
        with torch.no_grad():
            kwargs=dict(x_src=X,eval_pos=17,normalization=True,outlier_clipping=False,nan_replacement=False,used_features=8)
            a=model(y_src=y,**kwargs);b=model(y_src=changed,**kwargs)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_boost_training_without_test_and_actual_replay_match_original(self):
        import torch
        from exp61_sota_worker import boost_factory,replay_boost
        torch.set_num_threads(4)
        rng=np.random.default_rng(17)
        X=rng.normal(size=(215,8)).astype('float32')
        y=np.array([0]*180+[1]*30+[2]*5,dtype='int64')
        test=rng.normal(size=(17,8)).astype('float32')
        request=dict(boost_weights_root=str(ROOT/'tabpfn/third_party/BoostPFN'),boost_rounds=3,boost_samples=32,boost_batch=1000)
        base,original,split=boost_factory(request)
        original.fit(torch.from_numpy(X),torch.from_numpy(y),torch.from_numpy(test))
        expected=original.predict_proba(torch.from_numpy(test))
        actual=replay_boost(base,original,split,X,y,test,1000)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        base2,separate,split2=boost_factory(request)
        separate.fit(torch.from_numpy(X),torch.from_numpy(y),torch.empty((0,8)))
        np.testing.assert_array_equal(separate.sampled_idxs,original.sampled_idxs)
        np.testing.assert_allclose(separate.alphas,original.alphas,rtol=2e-5,atol=2e-6)
        actual2=replay_boost(base2,separate,split2,X,y,test,1000)
        np.testing.assert_allclose(actual2,expected,rtol=2e-5,atol=2e-6)
        # A new query set must trigger new inference, unlike upstream's cached API.
        different=replay_boost(base2,separate,split2,X,y,test[:3],1000)
        self.assertEqual(different.shape,(3,3))
        self.assertTrue(any(len(np.unique(y[i]))<3 for i in original.sampled_idxs))


if __name__=='__main__':unittest.main()

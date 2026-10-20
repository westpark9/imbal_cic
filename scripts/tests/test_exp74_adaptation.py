import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'v7/exp74'))
from run_adaptation import defaults
from distribution_heads import prior_offset, flatten_embeddings
from context_distillation import context_indices


class DistributionTargets(unittest.TestCase):
    def test_target_priors_cancel_in_log_product(self):
        # At the population optimum, logits+offset recover the source posterior,
        # even when head-training prior is different from the context prior.
        q = np.array([.4, .3, .3]); pi = np.array([.9, .09, .01])
        likelihood = np.array([[.8, .3, .1], [.1, .4, .9]])
        scores = []
        for exponent in (1., 0., -1.):
            f = np.log(likelihood)+exponent*np.log(pi)
            offset = prior_offset(q, pi, exponent)
            np.testing.assert_allclose(f+offset, np.log(likelihood)+np.log(q), atol=1e-6)
            scores.append(f)
        np.testing.assert_allclose(np.mean(scores, axis=0), np.log(likelihood), atol=1e-12)

    def test_embedding_estimators_remain_separate(self):
        x = np.arange(24).reshape(2, 3, 4)
        z = flatten_embeddings(x, 3)
        np.testing.assert_array_equal(z[1], np.r_[x[0, 1], x[1, 1]])
        with self.assertRaises(ValueError): flatten_embeddings(x, 2)

    def test_distillation_has_queries_for_rare_classes(self):
        y = np.repeat(np.arange(4), [100, 31, 9, 2])
        indices = context_indices(y, 30, 43)
        self.assertEqual(len(indices), 30)
        self.assertEqual(len(indices), len(np.unique(indices)))
        self.assertEqual(set(y[indices]), set(range(4)))
        left = np.setdiff1d(np.arange(len(y)), indices)
        self.assertTrue((np.bincount(y[left], minlength=4) >= 1).all())
        np.testing.assert_array_equal(indices, context_indices(y, 30, 43))

    def test_singleton_is_not_silently_used_as_query_and_context(self):
        with self.assertRaises(ValueError): context_indices(np.array([0, 0, 1]), 2, 43)

    def test_nan_and_constant_scaler_forward_parity_and_gradients(self):
        import torch
        from gradient_preprocessing import StableStandardScaler, StableDifferentiableZNormStep
        from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler
        x = torch.tensor([[1., 0., float('nan')], [2., 0., 3.], [5., 0., 6.]], requires_grad=True)
        old = TorchStandardScaler().fit(x)
        new = StableStandardScaler().fit(x)
        for k in ('mean', 'std'):
            torch.testing.assert_close(old[k], new[k])
        sum(t.sum() for t in new.values()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        z = torch.tensor([[1., 0.], [3., 0.]], requires_grad=True)
        step = StableDifferentiableZNormStep(); step._fit(z, None)
        transformed = step._transform(z)[0]
        self.assertTrue(torch.isfinite(transformed).all())
        transformed.sum().backward(); self.assertTrue(torch.isfinite(z.grad).all())

    def test_outlier_mask_does_not_backpropagate_nan(self):
        import torch
        from gradient_preprocessing import enable_context_preprocessor
        from tabpfn.preprocessing.torch.torch_soft_clip_outliers import TorchSoftClipOutliers
        x = torch.tensor([[0.], [0.], [0.], [0.], [0.], [10000.]], requires_grad=True)
        op = TorchSoftClipOutliers(n_sigma=1.)
        expected = op(x.detach())
        enable_context_preprocessor()
        actual = op(x)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__': unittest.main()

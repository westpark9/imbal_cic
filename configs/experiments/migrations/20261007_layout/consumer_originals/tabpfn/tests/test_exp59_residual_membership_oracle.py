import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from exp59_residual_membership_oracle import assigned_prediction, best_region_constants, confusion, scores


class ResidualOracleTests(unittest.TestCase):
    def test_membership_keeps_expert_errors_even_when_another_expert_is_correct(self):
        experts=np.array([[0,1],[0,1],[1,0]])
        regions=np.array([0,1,0])
        np.testing.assert_array_equal(assigned_prediction(experts,regions),[0,1,1])
        # The hypothetical truth [1,0,0] must not repair any of these mistakes.
        self.assertFalse((assigned_prediction(experts,regions)==[1,0,0]).any())

    def test_constant_reference_matches_independent_row_level_enumeration(self):
        rng=np.random.default_rng(59)
        for _ in range(35):
            C=int(rng.integers(2,5));K=int(rng.integers(1,4))
            y=rng.integers(0,C,17);regions=rng.integers(0,K,17)
            counts=np.bincount(regions*C+y,minlength=K*C).reshape(K,C)
            mapping,best=best_region_constants(counts)
            expected=max(f1_score(y,np.asarray(m)[regions],labels=np.arange(C),average='macro',zero_division=0)
                         for m in itertools.product(range(C),repeat=K))
            self.assertAlmostEqual(best,expected,places=13)
            self.assertAlmostEqual(scores(confusion(y,mapping[regions],C))['macro_f1'],expected,places=13)

    def test_class_constant_experts_cannot_receive_positive_excess_credit(self):
        # Even perfect class-revealing membership has zero excess over its null.
        y=np.array([0,0,1,1,2,2]);regions=y.copy()
        E=np.tile(np.arange(3),(len(y),1))
        pred=assigned_prediction(E,regions)
        mapping,null=best_region_constants(np.diag([2,2,2]))
        self.assertEqual(scores(confusion(y,pred,3))['macro_f1'],1.)
        self.assertEqual(null,1.)
        self.assertEqual(scores(confusion(y,pred,3))['macro_f1']-null,0.)

    def test_mixed_regions_can_show_real_within_region_discrimination(self):
        counts=np.array([[3,2,0],[0,2,3]])
        mapping,null=best_region_constants(counts)
        self.assertLess(null,1.)
        y=np.array([0,0,0,1,1,1,1,2,2,2]);regions=np.array([0]*5+[1]*5)
        E=np.stack([y,y],axis=1)
        self.assertGreater(scores(confusion(y,assigned_prediction(E,regions),3))['macro_f1']-null,0.)


if __name__=='__main__':unittest.main()

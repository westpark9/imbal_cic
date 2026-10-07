import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from exp58_bank_oracle_upper_bound import candidate_masks,exact_macro_f1_selector


def macro(y,p,C):
    cm=np.bincount(np.asarray(y)*C+np.asarray(p),minlength=C*C).reshape(C,C)
    den=cm.sum(0)+cm.sum(1)
    return np.divide(2*np.diag(cm),den,out=np.zeros(C),where=den>0).mean()


class BankOracleTests(unittest.TestCase):
    def test_matches_exhaustive_row_selection_with_zeros_ties_and_missing_classes(self):
        rng=np.random.default_rng(923)
        for _ in range(120):
            C=int(rng.integers(2,5));N=6;y=rng.integers(0,C,N)
            masks=rng.integers(1,1<<C,N,dtype=np.uint32)
            pred,cert=exact_macro_f1_selector(y,masks,C)
            options=[[c for c in range(C) if m&(1<<c)] for m in masks]
            best=max(macro(y,p,C) for p in itertools.product(*options))
            self.assertAlmostEqual(cert['macro_f1_upper_bound'],best,places=12)
            self.assertAlmostEqual(macro(y,pred,C),best,places=12)

    def test_frozen_calls_exclude_experts_outside_the_call_set(self):
        y=np.array([1,1,2]);g=np.array([0,0,0]);E=np.array([[1,2],[1,2],[1,2]])
        masks=candidate_masks(g,E,np.array([True,False,False]))
        pred,cert=exact_macro_f1_selector(y,masks,3)
        np.testing.assert_array_equal(pred,[1,0,0]);self.assertEqual(cert['accuracy_upper_bound'],1/3)

    def test_constant_class_bank_exposes_oracle_quality_confound(self):
        y=np.array([0,1,2,0,1,2]);g=np.zeros(6,int);E=np.tile([1,2],(6,1))
        pred,cert=exact_macro_f1_selector(y,candidate_masks(g,E),3)
        self.assertEqual(cert['macro_f1_upper_bound'],1.)
        np.testing.assert_array_equal(pred,y)

    def test_global_fallback_is_not_macro_f1_ceiling(self):
        # All correctable rows are already right; choosing the destination of
        # the unavoidable false positive still changes macro-F1.
        y=np.array([0,0,0,1,2]);g=np.array([0,0,0,1,1]);E=np.array([[0],[0],[0],[1],[0]])
        pred,cert=exact_macro_f1_selector(y,candidate_masks(g,E),3)
        self.assertGreater(cert['macro_f1_upper_bound'],macro(y,g,3))
        self.assertEqual((pred==y).sum(),(g==y).sum())


if __name__=='__main__':unittest.main()

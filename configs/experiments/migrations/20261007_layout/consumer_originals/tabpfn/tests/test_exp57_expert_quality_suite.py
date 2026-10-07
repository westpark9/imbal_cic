import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from exp57_expert_quality_suite import allocate_balanced,sample_histogram,validation_region_choice,paired_bootstrap

class QualitySuiteTests(unittest.TestCase):
    def test_balancing_preserves_budget_under_rare_class_caps(self):
        result=allocate_balanced(12,np.array([2,20,20]));self.assertEqual(result.sum(),12)
        self.assertEqual(result[0],2);self.assertEqual(result.tolist(),[2,5,5])
        with self.assertRaises(ValueError):allocate_balanced(5,[1,1])

    def test_histogram_control_preserves_every_class_without_duplicates(self):
        y=np.repeat(np.arange(3),[20,5,30]);counts=np.array([7,3,9])
        a=sample_histogram(y,counts,42);b=sample_histogram(y,counts,43)
        np.testing.assert_array_equal(np.bincount(y[a]),counts)
        self.assertEqual(len(a),len(np.unique(a)));self.assertFalse(np.array_equal(a,b))

    def test_reference_selects_on_validation_and_keeps_global_for_ties_and_low_support(self):
        y=np.array([0,1,0,1]);regions=np.array([0,0,1,1]);g=np.array([0,0,0,1]);e=np.array([0,1,0,1])
        np.testing.assert_array_equal(validation_region_choice(y,[g,e],regions,2,2,minimum=2),[1,0])
        np.testing.assert_array_equal(validation_region_choice(y,[g,e],regions,2,2,minimum=3),[0,0])

    def test_paired_bootstrap_identical_models_have_zero_difference(self):
        y=np.array([0,0,1,1]);pred=np.array([0,0,0,0]);h=np.array([9,9,7,7]);t=np.array([0,1,1,2])
        rows=paired_bootstrap(y,pred,pred,h,t,2,42,repeats=20)
        for r in rows:self.assertEqual(r['delta_macro_f1_low'],0);self.assertEqual(r['delta_macro_f1_high'],0)

if __name__=='__main__':unittest.main()

"""Label-free selection, historical-oracle counterexample, and matched-call invariants."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from exp56_five_condition_evaluation import (
    assignment_oracle, designated_mapping, fixed_region_assignment, fpr_threshold, scorer_policies,
)


class EvaluationTests(unittest.TestCase):
    def test_fixed_regions_do_not_gain_truth_from_constant_class_experts(self):
        # Every expert emits a constant class; identical observations cannot route by y.
        experts = np.tile([0, 1], (4, 1))
        pred, owner = fixed_region_assignment(experts, np.tile([.1, .9], (4, 1)))
        np.testing.assert_array_equal(pred, [0, 0, 0, 0])
        np.testing.assert_array_equal(owner, [0, 0, 0, 0])
        y = np.array([0, 1, 0, 1])
        self.assertEqual(float((pred == y).mean()), .5)
        historical, _ = assignment_oracle(y, pred, experts, np.array([0, 1]))
        self.assertTrue((historical == y).all())

    def test_fixed_regions_keep_wrong_outputs_and_break_distance_ties_by_id(self):
        pred, owner = fixed_region_assignment(np.array([[1, 0], [0, 1]]),
                                               np.array([[1., 1.], [3., 2.]]))
        np.testing.assert_array_equal(pred, [1, 1])
        np.testing.assert_array_equal(owner, [0, 1])

    def test_assignment_keeps_wrong_owner_even_if_global_or_other_expert_is_right(self):
        y = np.array([0, 1, 2, 1])
        global_pred = np.array([0, 1, 0, 2])
        experts = np.array([[1, 0], [1, 0], [2, 2], [2, 1]])
        pred, called = assignment_oracle(y, global_pred, experts, np.array([0, 1, -1]))
        np.testing.assert_array_equal(pred, [1, 0, 0, 1])
        np.testing.assert_array_equal(called, [True, True, False, True])

    def test_mapping_is_context_only_with_declared_tie_break(self):
        dominant, mapping = designated_mapping(np.array([[10, 2, 0], [20, 3, 0], [1, 30, 0], [1, 30, 0]]))
        np.testing.assert_array_equal(dominant, [0, 0, 1, 1])
        np.testing.assert_array_equal(mapping, [1, 2, -1])

    def test_removing_verifier_preserves_candidate_and_calls(self):
        scores = {'candidate': np.array([1, 1, 2]), 'pre': np.array([.9, .9, .1]),
                  'post': np.array([-.5, .5, .5])}
        s, sv, calls, accepted = scorer_policies(np.zeros(3, int), scores, {'tau_pre':.5, 'tau_post':0})
        np.testing.assert_array_equal(s, [1, 1, 0])
        np.testing.assert_array_equal(sv, [0, 1, 0])
        np.testing.assert_array_equal(calls, [True, True, False])
        np.testing.assert_array_equal(accepted, [False, True, False])

    def test_global_fallback_disables_both_policies(self):
        g = np.array([0,1])
        scores = {'candidate': 1-g, 'pre': np.ones(2), 'post': np.ones(2)}
        s, sv, calls, accepted = scorer_policies(g, scores, {'tau_pre':None, 'tau_post':None})
        np.testing.assert_array_equal(s, g)
        np.testing.assert_array_equal(sv, g)
        self.assertFalse(calls.any() or accepted.any())

    def test_tied_negative_scores_cannot_break_weighted_fpr_budget(self):
        scores = np.array([.1,.2,.2,.9,.99])
        positive = np.array([False,False,False,False,True])
        weights = np.array([1.,1.,7.,1.,1.])
        t = fpr_threshold(scores, positive, weights, .2)
        pred = scores > t
        self.assertEqual(t, .2)
        self.assertLessEqual(weights[pred & ~positive].sum()/weights[~positive].sum(), .2)
        self.assertTrue(pred[-1])


if __name__ == '__main__':
    unittest.main()

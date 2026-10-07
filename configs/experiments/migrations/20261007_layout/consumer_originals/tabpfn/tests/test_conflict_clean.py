import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cic2018_conflict_clean import conflict_mask, filter_splits


class ConflictCleanTests(unittest.TestCase):
    def test_entire_conflict_group_removed_other_duplicates_preserved(self):
        X = np.array([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [5, 6]], float)
        mask, _, meta = conflict_mask(X, [0, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(mask, [True, True, True, False, False, False])
        self.assertEqual(meta['conflicting_vector_groups'], 1)

    def test_hash_collision_does_not_delete_different_vectors(self):
        X = np.array([[1], [1], [2], [2], [3]])
        mask, _, meta = conflict_mask(X, [0, 1, 0, 0, 1], hashes=np.zeros(5, dtype=np.uint64))
        np.testing.assert_array_equal(mask, [True, True, False, False, False])
        self.assertEqual(meta['conflicting_vector_groups'], 1)

    def test_model_input_conversion_and_signed_zero(self):
        X = np.array([[np.nan, -0.0], [0., 0.], [1., 2.], [1. + 1e-9, 2.]])
        mask, _, _ = conflict_mask(X, [0, 1, 0, 1])
        self.assertTrue(mask.all())

    def test_split_membership_preserved_with_cross_split_conflict(self):
        X = np.array([[1], [2], [1], [3], [2], [3]])
        mask, _, _ = conflict_mask(X, [0, 0, 1, 1, 0, 1])
        splits = {'train': np.array([0, 1]), 'val': np.array([2, 3]), 'test': np.array([4, 5])}
        result = filter_splits(splits, np.flatnonzero(~mask))
        for key, expected in [('train', [1]), ('val', [3]), ('test', [4, 5])]:
            np.testing.assert_array_equal(result[key], expected)


if __name__ == '__main__':
    unittest.main()

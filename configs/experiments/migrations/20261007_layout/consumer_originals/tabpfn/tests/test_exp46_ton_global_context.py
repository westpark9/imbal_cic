"""Scientific controls: window contrasts preserve quotas and metric definitions."""
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nfv3_v3_exp46_ton_global_context as exp


class GlobalContextTests(unittest.TestCase):
    def test_window_factor_preserves_realized_rare_counts(self):
        names = ["benign", "mitm", "ransomware"]
        y = np.repeat(np.arange(3), [200, 20, 8])
        train = np.arange(len(y))
        early = np.concatenate([np.flatnonzero(y == c)[:n] for c, n in enumerate([100, 10, 4])])
        contexts, quota = exp.make_contexts(train, y, early, y[early], names, 40, 42)
        # A window change must NOT expand rare support merely because it is available.
        np.testing.assert_array_equal(quota["legacy"], [30, 6, 4])
        np.testing.assert_array_equal(quota["natural"], [35, 4, 1])
        for recipe in ["natural", "legacy"]:
            a, b = contexts["early_" + recipe], contexts["full_" + recipe]
            np.testing.assert_array_equal(np.bincount(y[a]), np.bincount(y[b]))
            self.assertTrue(np.isin(a, early).all())
            self.assertTrue(np.isin(b, train).all())
            self.assertTrue(np.any(~np.isin(b, early)))
            self.assertEqual(len(np.unique(a)), 40)
            self.assertEqual(len(np.unique(b)), 40)

    def test_draws_are_nested_when_only_class_quota_changes(self):
        ids = np.arange(100)
        y = np.repeat([0, 1], 50)
        a = exp.draw_quota(ids, y, [15, 5], 1032)
        b = exp.draw_quota(ids, y, [10, 10], 1032)
        self.assertTrue(np.isin(b[y[b] == 0], a).all())
        self.assertTrue(np.isin(a[y[a] == 1], b).all())

    def test_infeasible_quota_is_rejected_instead_of_replaced(self):
        with self.assertRaises(ValueError):
            exp.draw_quota(np.arange(5), np.array([0, 0, 0, 1, 1]), [2, 3], 1)

    def test_metrics_match_sklearn_including_absent_predictions(self):
        names = ["benign", "mitm", "ransomware"]
        y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
        p = np.array([0, 0, 1, 0, 1, 0, 1, 0])
        summary, rows, cm = exp.metrics(y, p, names, names[1:])
        self.assertAlmostEqual(summary["macro_f1"], f1_score(y, p, average="macro"))
        self.assertAlmostEqual(summary["benign_fpr"], .25)
        self.assertEqual(rows[1]["fp"], 2)
        self.assertEqual(rows[2]["precision"], 0.)
        self.assertEqual(int(cm.sum()), len(y))


if __name__ == "__main__":
    unittest.main()

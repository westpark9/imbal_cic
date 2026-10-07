"""Scientific invariants for EXP39, including an end-to-end synthetic cache run."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import nfv3_v3_exp39_decision_routing as exp


class RoutingTests(unittest.TestCase):
    def test_gain_is_correctness_change_not_confidence(self):
        y = np.array([1, 1, 1, 1])
        global_pred = np.array([0, 1, 1, 0])
        expert_pred = np.array([1, 0, 1, 2])
        np.testing.assert_array_equal(exp.decision_gain(y, global_pred, expert_pred),
                                      [1, -1, 0, 0])

    def test_weighted_confusion_f1_matches_sklearn(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        pred = np.array([0, 1, 2, 1, 0, 2, 1, 2])
        w = np.array([10, 10, 10, 1, 1, .5, .5, .5])
        m = exp.metrics_from_cm(exp.confusion(y, pred, 3, w), [2], 0)
        self.assertAlmostEqual(m['macro_f1'], f1_score(y, pred, average='macro', sample_weight=w))
        self.assertAlmostEqual(m['benign_fpr'], 2 / 3)

    def test_fpr_guard_blocks_tail_gain_from_benign_harm(self):
        y = np.repeat(np.arange(3), 20)
        glob = y.copy(); glob[40:50] = 0
        candidate = y.copy(); candidate[:10] = 2
        best, grid = exp.select_thresholds(
            y, glob, candidate, np.ones(60), np.ones(60), np.ones(60),
            3, [2], 0, [1], [0., .5], [0., .5], .0005, 0., 1., 1)
        self.assertIsNone(best['tau_pre'])
        self.assertTrue(grid.reason.str.contains('benign_fpr').any())

    def test_protected_class_guard_blocks_average_gain(self):
        y = np.repeat(np.arange(3), 100)
        glob = y.copy(); glob[200:260] = 0
        candidate = y.copy(); candidate[100:105] = 0
        best, grid = exp.select_thresholds(
            y, glob, candidate, np.ones(300), np.ones(300), np.ones(300),
            3, [2], 0, [1], [0.], [0.], .0005, 0., 1., 1)
        self.assertIsNone(best['tau_pre'])
        violating = grid[grid.reason.str.contains('protected_class_f1')]
        self.assertTrue((violating.delta_macro_f1 > 0).any())

    def test_calibration_forward_and_hash_disjoint(self):
        y = np.repeat([0, 1], 10)
        ts = np.tile(np.arange(10), 2)
        hashes = np.arange(20); hashes[8] = hashes[1]
        a, b = exp.chronological_cal_split(y, np.repeat('s', 20), ts,
                                           hashes, np.ones(20, bool), .3)
        self.assertNotIn(8, b)
        self.assertFalse(np.intersect1d(hashes[a], hashes[b]).size)
        for c in [0, 1]: self.assertLess(ts[a[y[a] == c]].max(), ts[b[y[b] == c]].min())

    def test_end_to_end_cache_and_all_arms(self):
        rng = np.random.default_rng(13)
        with tempfile.TemporaryDirectory(prefix='exp39_test_') as tmp:
            cache = Path(tmp) / 'cache'; cache.mkdir()
            names = ['benign', 'bot', 'brute_force', 'ddos', 'dos', 'infiltration', 'web_attacks']
            C, K = len(names), 2
            meta = {'class_names': names, 'n_experts': K,
                    'train_counts': [1000, 100, 100, 100, 100, 50, 20],
                    'tail_classes': ['bot', 'infiltration', 'web_attacks'],
                    'source_config': {'residual_gamma': 1.},
                    'test_role': 'synthetic unit test'}
            (cache / 'COMPLETE.json').write_text(json.dumps(meta))
            (cache / 'identity.json').write_text(json.dumps({'synthetic': True}))
            np.save(cache / 'qk.npy', np.ones((K, 6), np.float32))
            for split, reps in [('route', 30), ('cal', 20), ('eval', 15)]:
                y = np.tile(np.arange(C), reps); n = len(y)
                x = rng.normal(size=(n, 46)).astype(np.float32)
                data = {'X': x, 'z': x[:, :16], 'y': y, 'ids': np.arange(n),
                        'hash': np.arange(n, dtype=np.uint64),
                        'scenario': np.repeat('known', n), 'time': np.arange(n),
                        'distance': rng.random((n, K)).astype(np.float32)}
                for k in range(K + 1):
                    pred = y.copy()
                    wrong = (np.arange(n) + k) % 5 == 0
                    pred[wrong] = (pred[wrong] + 1) % C
                    p = np.full((n, C), .05, np.float32)
                    p[np.arange(n), pred] = .7
                    data[f'p{k}'] = p
                    data[f'affinity_{k}'] = -rng.random(n).astype(np.float32)
                if split == 'cal': data['mask'] = np.ones(n, bool)
                for name, value in data.items(): np.save(cache / f'{split}_{name}.npy', value)
            args = SimpleNamespace(cache_dir=str(cache), out_root=str(Path(tmp) / 'results'),
                arms=','.join(exp.SPECS), trees=2, depth=2, learning_rate=.1,
                threads=2, seed=42, predict_chunk=100, confirm_fraction=.3,
                benign_fpr_increase=.0005, protected_classes='brute_force,ddos,dos',
                protected_f1_drop=0., max_proposal=1., min_decided=1)
            exp.run(args)
            out = next((Path(tmp) / 'results').iterdir())
            self.assertTrue((out / 'COMPLETE.json').exists())
            self.assertTrue((out / '6a_full_test_deltas.png').exists())
            summary = pd.read_csv(out / 'summary.csv')
            test = summary[summary.split == 'full_test']
            self.assertEqual(set(exp.SPECS) - set(test.arm), set())
            self.assertTrue((test.rows == C * 15).all())
            for arm in exp.SPECS:
                final = np.load(out / f'{arm}_final.npy')
                row = test[test.arm == arm].iloc[0]
                self.assertAlmostEqual(row.macro_f1, f1_score(np.load(cache / 'eval_y.npy'), final, average='macro'))


if __name__ == '__main__':
    unittest.main()

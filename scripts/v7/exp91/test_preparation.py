"""Sampling invariants needed before portable IDS preparation."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import types
import numpy as np
from prepare_inputs import choose_labels, choose_unlabelled, filename
from common import row_groups


class PreparationTests(unittest.TestCase):
    def test_sampling_matches_exp89_and_fixed750_and_is_nested(self):
        names = ['backdoor', 'benign', 'ddos']
        families = np.repeat(names, [50, 1200, 80])
        pool = np.arange(len(families))
        draws = choose_labels(pool, families, names, 43)
        self.assertEqual(list(map(len, draws)), [20, 750, 20])
        for c, name in enumerate(names):
            candidates = pool[families == name]
            expected = np.random.default_rng(43+c*1009).choice(candidates, 20, replace=False)
            np.testing.assert_array_equal(draws[c][:20], expected)
        self.assertEqual(len(np.unique(np.concatenate(draws))), 790)
        self.assertTrue(set(draws[0][:2]) < set(draws[0][:20]))
        np.testing.assert_array_equal(choose_labels(pool, families, names, 43)[1], draws[1])

    def test_unlabelled_excludes_feature_copies_and_preserves_requested_count(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(100, 3)).astype('float32')
        X[10:20] = X[0]
        reserved = np.array([0, 1, 2])
        pool = np.arange(80)  # outside rows represent other splits and must never be sampled
        ids, _ = choose_unlabelled(X, pool, reserved, 40, 43)
        self.assertEqual(len(ids), 40)
        self.assertEqual(len(np.unique(ids)), 40)
        self.assertTrue(np.isin(ids, pool).all())
        self.assertFalse(np.intersect1d(row_groups(X[ids]), row_groups(X[reserved])).size)
        np.testing.assert_array_equal(ids, choose_unlabelled(X, pool, reserved, 40, 43)[0])
        self.assertEqual(filename('toniot', 20, 43, 10000),
                         'toniot_original_fixed750_n20_s43_u10000_full.npz')

    def test_insufficient_unlabelled_pool_is_not_silently_shortened(self):
        X = np.zeros((8, 2), dtype='float32')
        with self.assertRaisesRegex(ValueError, 'eligible'):
            choose_unlabelled(X, np.arange(8), np.array([0]), 2, 43)

    def test_launcher_uses_manifest_benign_and_categorical_and_resumes(self):
        import run_night
        from prepare_inputs import SOURCE_HASH, sha, write
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root/'inputs'
            inputs.mkdir()
            source = inputs/filename('toniot', 20, 43, 10000)
            source.write_bytes(b'fixture; worker is replaced by a test double')
            write(source.with_suffix('.json'), dict(source_sha256=SOURCE_HASH, dataset='toniot', n=20,
                seed=43, unlabelled_rows=10000, bytes=source.stat().st_size, npz_sha256=sha(source),
                benign_class=1, categorical_columns=[0, 1], historical_test_feature_overlap_allowed=True))
            checkpoint = root/'model.ckpt'
            checkpoint.write_bytes(b'fixture')
            output = root/'run'
            calls = []
            def worker(command, **kwargs):
                calls.append(command)
                self.assertEqual(command[command.index('--benign-class')+1], '1')
                self.assertIn('--allow-test-feature-overlap', command)
                self.assertEqual(command[command.index('--categorical-columns')+1:][:2], ['0', '1'])
                dest = Path(command[command.index('--output')+1])
                write(dest/'summary.json', {'a': {'fit_seconds': 1, 'inference_seconds': 2}})
                write(dest/'COMPLETE.json', {'status': 'complete'})
                return types.SimpleNamespace(returncode=0)
            args = ['run_night', '--datasets', 'toniot', '--budgets', '20', '--input-dir', str(inputs),
                    '--checkpoint', str(checkpoint), '--output', str(output), '--model', 'a']
            with patch('sys.argv', args), patch.object(run_night, 'ROOT', root), \
                    patch.object(run_night.subprocess, 'run', side_effect=worker), \
                    patch.object(run_night, 'report_run'):
                run_night.main()
                run_night.main()
            self.assertEqual(len(calls), 1)
            self.assertTrue((output/'COMPLETE.json').is_file())


if __name__ == '__main__':
    unittest.main()

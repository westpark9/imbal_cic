"""Contract/invariant tests. Stub-backend tests are not IDS performance evidence."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from common import FlowPreprocessor, aligned, row_groups
from teacher_student import StudentConfig, TeacherStudentClassifier, soft_assignment
from pair_residual import PairConfig, PairResidualClassifier, correct_pair


class CentroidEstimator:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centres = np.stack([X[y == c].mean(0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        from scipy.special import softmax
        return softmax(-((X[:, None]-self.centres[None])**2).sum(2), axis=1)


class ScriptedEstimator:
    """A deliberately confused Global and informative binary expert, for plumbing."""
    records = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.train_groups = set(row_groups(X))
        return self

    def predict_proba(self, X):
        self.records.append((self.train_groups, set(row_groups(X))))
        key = X[:, 0].astype(int)
        p = np.full((len(X), len(self.classes_)), .01)
        if len(self.classes_) == 3:
            for i, c in enumerate(key):
                p[i] = [.90, .05, .05] if c == 0 else [.55, .40, .05] if c == 1 else [.01, .01, .98]
        else:
            for i, c in enumerate(key):
                p[i] = .5
                if c in self.classes_:
                    p[i] = .01
                    p[i, np.flatnonzero(self.classes_ == c)[0]] = .99
        return p/p.sum(1, keepdims=True)


def toy(n=6):
    rng = np.random.default_rng(15)
    y = np.repeat(np.arange(3), n)
    X = np.column_stack([y, rng.normal(size=len(y)), np.arange(len(y))*.001]).astype('float32')
    return X, y


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_assignment_satisfies_stationarity_and_nonuniform_marginal(self):
        cost = torch.tensor([[0., .2], [.7, 0.], [.5, .4]], dtype=torch.float64)
        prior = torch.tensor([.8, .2], dtype=torch.float64)
        q = soft_assignment(cost, prior, .3, .7, 500)
        np.testing.assert_allclose(q.sum(1), 1, atol=1e-7)
        # Convex objective's row-constrained first-order condition.
        derivative = cost+.3*(q.log()+1)+.7*((q.mean(0)/prior).log()+1)
        np.testing.assert_allclose(derivative[:, 0], derivative[:, 1], atol=2e-6)
        uniform_cost = torch.zeros((12, 2), dtype=torch.float64)
        result = soft_assignment(uniform_cost, prior, .1, 1., 500)
        self.assertGreater(float(result[:, 0].mean()), .7)

    def test_assignment_without_prior_equals_entropy_softmax(self):
        cost = torch.tensor([[.2, .8, 2.], [.5, .3, .1]])
        q = soft_assignment(cost, torch.ones(3)/3, .2, 0)
        torch.testing.assert_close(q, torch.softmax(-cost/.2, 1))

    def test_category_unknown_missing_and_train_only_scaling(self):
        p = FlowPreprocessor([1]).fit(np.array([[1., 3.], [3., 8.], [np.nan, np.nan]]))
        before = dict(p.medians)
        values, observed = p.transform(np.array([[1000., 99.], [np.nan, np.nan]]))
        self.assertEqual(before, p.medians)
        _, start, stop, _ = p.blocks[1]
        self.assertEqual(values[0, start:stop].argmax(), 3)  # unknown
        self.assertEqual(values[1, start:stop].argmax(), 2)  # missing
        self.assertFalse(observed[1].any())
        self.assertTrue(np.isfinite(values).all())

    def test_student_trains_without_changing_inputs_and_roundtrips(self):
        rng = np.random.default_rng(2)
        y = np.repeat([2, 7, 11], 5)
        X = rng.normal(size=(15, 4)).astype('float32')+np.repeat([0., 2., 4.], 5)[:, None]
        U = rng.normal(size=(24, 4)).astype('float32')
        xcopy, ycopy, ucopy = X.copy(), y.copy(), U.copy()
        cfg = StudentConfig(hidden=16, embedding=8, warmup_epochs=1, rounds=1,
                            epochs_per_round=1, batch_size=12, device='cpu')
        model = TeacherStudentClassifier(CentroidEstimator, cfg).fit(X, y, U)
        p = model.predict_proba(U[:7])
        np.testing.assert_allclose(p.sum(1), 1)
        np.testing.assert_array_equal(model.classes_, [2, 7, 11])
        for a, b in [(X, xcopy), (y, ycopy), (U, ucopy)]: np.testing.assert_array_equal(a, b)
        self.assertNotEqual(model.audit_['encoder_initial_sha256'], model.audit_['encoder_final_sha256'])
        self.assertFalse(hasattr(model, 'decoder_'))
        self.assertFalse(hasattr(model, 'teacher_'))
        np.testing.assert_allclose(model.pseudo_labels_.sum(1), 1, atol=1e-6)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'student.joblib'
            model.save(path)
            loaded = TeacherStudentClassifier.load(path)
            np.testing.assert_allclose(loaded.predict_proba(U[:7]), p, atol=1e-7)

    def test_student_rejects_unlabelled_copy_of_labelled_feature(self):
        X, y = toy()
        model = TeacherStudentClassifier(CentroidEstimator, StudentConfig(device='cpu'))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            model.fit(X, y, X[[0]])

    def test_pair_score_locality_bound_and_identity(self):
        s = np.array([[3., -1., 2.], [-4., 2., 7.]])
        out = correct_pair(s, [.99, .01], 0, 1, .5, 1.)
        np.testing.assert_array_equal(out[:, 2], s[:, 2])
        np.testing.assert_allclose(out[:, :2].sum(1), s[:, :2].sum(1))
        self.assertLessEqual(np.max(np.abs(out-s)), .25+1e-12)
        np.testing.assert_array_equal(correct_pair(s, [.5, .8], 0, 1, 0, 1.), s)

    def test_n2_falls_back_exactly_to_global(self):
        X, y = toy(2)
        m = PairResidualClassifier(CentroidEstimator).fit(X, y)
        self.assertEqual(m.steps_, [])
        np.testing.assert_array_equal(m.predict_proba(X), aligned(m.global_, X, m.classes_))
        self.assertEqual(m.audit_['global_context_rows'], len(X))

    def test_selected_pair_uses_disjoint_oof_groups_and_all_final_labels(self):
        X, y = toy(9)
        ScriptedEstimator.records = []
        m = PairResidualClassifier(ScriptedEstimator, config=PairConfig(max_experts=2)).fit(X, y)
        self.assertGreater(len(m.steps_), 0)
        self.assertEqual(m.steps_[0]['pair'], [0, 1])
        for train, query in ScriptedEstimator.records:
            self.assertFalse(train & query)
        self.assertEqual(len(m.global_.train_groups), len(X))
        self.assertEqual(m.audit_['expert_context_rows'][0], 18)
        self.assertEqual(len(set(tuple(s['pair']) for s in m.steps_)), len(m.steps_))
        p = m.predict_proba(X)
        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(np.mean(p.argmax(1) == y), 2/3)

    def test_conflicting_vectors_stay_in_one_fold(self):
        X, y = toy(8)
        X[8] = X[0]  # same vector, different true labels
        m = PairResidualClassifier(CentroidEstimator).fit(X, y)
        self.assertEqual(m.fold_ids_[0], m.fold_ids_[8])

    def test_missing_benign_rejected_and_invalid_configs_rejected(self):
        X, y = toy(4)
        with self.assertRaisesRegex(ValueError, 'benign'):
            PairResidualClassifier(CentroidEstimator, benign_class=99).fit(X, y)
        with self.assertRaises(ValueError): StudentConfig(prototype_mix=.9).validate()
        with self.assertRaises(ValueError): PairConfig(caps=(0.,)).validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)

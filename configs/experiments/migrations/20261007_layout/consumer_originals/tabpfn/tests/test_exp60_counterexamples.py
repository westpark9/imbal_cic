import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from exp60_counterexamples import core_subset, nearest_pairs, matched_random


def test_core_preserves_rare_class_and_exact_memory_budget():
    y = np.array([0] * 19 + [1])
    core, budget = core_subset(np.arange(20), y, .2, 43)
    assert len(core) == 16 and budget == 4
    assert 19 in core
    assert len(set(core)) == len(core)


def test_nearest_search_crosses_labels_instead_of_returning_same_class():
    obs = np.array([[0., 0.], [.01, 0.], [.1, 0.], [10., 0.], [10.1, 0.]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0])
    pairs = nearest_pairs(obs, y, np.array([1, 2, 4]), np.array([0, 3]), 2, 43, lambda *a, **k: None)
    assert {(q, c) for q, c, _ in pairs} == {(0, 2), (3, 4)}


def test_random_control_matches_each_counterpart_class_without_replacement():
    y = np.array([0, 1, 2, 1, 2, 0, 1, 2, 0])
    pairs = [(0, 3, .1), (0, 4, .2), (1, 5, .3), (2, 6, .4)]
    candidates = np.arange(3, 9)
    random = matched_random(pairs, candidates, y, 43)
    assert np.array_equal(y[random], y[[3, 4, 5, 6]])
    assert len(np.unique(random)) == 4
    assert np.all(y[random] != y[[0, 0, 1, 2]])
    assert set(random).issubset(set(candidates))


if __name__ == '__main__':
    tests = [value for key, value in globals().copy().items() if key.startswith('test_')]
    for test in tests:
        test()
    print(f'{len(tests)} selection checks passed')

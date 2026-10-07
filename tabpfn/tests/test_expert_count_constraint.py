"""Exercise the production count guard without importing GPU/model dependencies."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import ast
from pathlib import Path
import unittest

source = script_path('nfv3_v3_exp31_c0alloc.py')
tree = ast.parse(source.read_text())
definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'admissible_expert_counts')
scope = {}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), scope)
allowed = scope['admissible_expert_counts']


class ExpertCountTests(unittest.TestCase):
    def test_cic_rejects_equal_and_larger_counts(self):
        self.assertEqual(allowed([2, 4, 7, 8], 7), [2, 4])

    def test_ton_default_grid_remains_admissible(self):
        self.assertEqual(allowed([8, 2, 4, 4], 10), [2, 4, 8])

    def test_empty_or_invalid_grid_fails_before_fitting(self):
        for grid, classes in [([7, 8], 7), ([0, 2], 7), ([], 7), ([1], 1)]:
            with self.subTest(grid=grid, classes=classes), self.assertRaises(ValueError):
                allowed(grid, classes)


if __name__ == '__main__':
    unittest.main()

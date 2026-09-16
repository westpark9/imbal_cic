import ast
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from exp47_trace import instrument
from exp47_audit import counts


class TraceTests(unittest.TestCase):
    def test_only_observation_callbacks_are_inserted(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/nfv3_v3_exp31_c0alloc.py'
        original = path.read_text()
        modified = instrument(original)
        compile(modified, '<instrumented>', 'exec')
        class RemoveTrace(ast.NodeTransformer):
            def visit_Expr(self, node):
                v = node.value
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and isinstance(v.func.value, ast.Name) and v.func.value.id == '_exp47_trace':
                    return None
                return self.generic_visit(node)
        self.assertEqual(ast.dump(RemoveTrace().visit(ast.parse(modified))), ast.dump(ast.parse(original)))

    def test_recovery_deduplicates_bank_opportunities_and_counts_harm(self):
        y = np.array([0, 0, 1, 1])
        glob = np.array([0, 1, 0, 1])
        candidate = np.array([1, 0, 1, 1])
        bank = np.array([True, True, True, True])
        masks = {'all': np.ones(4, bool), 'none': np.zeros(4, bool)}
        rows = counts(y, glob, candidate, bank, masks, ['a', 'b'])
        self.assertEqual(sum(r['helpful'] for r in rows if r['stage'] == 'all'), 2)
        self.assertEqual(sum(r['harmful'] for r in rows if r['stage'] == 'all'), 1)
        self.assertEqual(sum(r['bank_correctable'] for r in rows if r['stage'] == 'all'), 2)
        self.assertEqual(sum(r['helpful'] for r in rows if r['stage'] == 'none'), 0)


if __name__ == '__main__': unittest.main()

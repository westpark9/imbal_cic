"""Protocol invariants for class arrival, integer allocation, and bounded history."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('exp71', ROOT/'scripts/v7/exp71/exp71_bounded_updates.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ProtocolTests(unittest.TestCase):
    def test_capped_proportions_and_minimum(self):
        np.testing.assert_array_equal(mod.apportion(10, [9, 1], [5, 20]), [5, 5])
        np.testing.assert_array_equal(mod.apportion(10, [999, 1], [100, 100]), [10, 0])
        np.testing.assert_array_equal(mod.apportion(10, [999, 1], [100, 100], 1), [9, 1])
        with self.assertRaises(ValueError): mod.apportion(1, [1, 1], [5, 5], 1)

    def test_allocation_totals_and_natural_ratios(self):
        available={'benign':1000000,'a':9000,'b':900,'c':100}
        alloc=mod.allocations(available,100)
        self.assertEqual(alloc['uniform'],{'a':100,'b':100,'c':100,'benign':900})
        self.assertEqual(alloc['natural'],{'a':270,'b':27,'c':3,'benign':900})

    def test_memory_no_resurrection_and_coverage(self):
        names=['benign','a','b'];y=np.array([0]*800+[1]*190+[2]*10)
        available={'benign':800,'a':190,'b':10}
        first,_=mod.select_history(np.arange(1000),y,names,'natural',available,100,43)
        second,_=mod.select_history(first,y,names,'natural',available,50,44)
        self.assertTrue(set(second)<=set(first));self.assertEqual(len(second),50)
        self.assertEqual(set(y[second]),{0,1,2})

    def test_stream_equality_and_future_exclusion(self):
        ds='cic2018';names=sorted(c for group in mod.STAGES[ds] for c in group)
        available={n:10000+(i*2000) for i,n in enumerate(names)}
        alloc=mod.allocations(available,30)
        y=np.concatenate([np.full(max(v[c] for v in alloc.values()),i) for i,c in enumerate(names)])
        plans=mod.make_plans(y,names,ds,alloc,available,43,128)
        for case in mod.CASES:
            delivered=set();previous=set();discarded=set()
            for stage,p in enumerate(plans[case]):
                incoming=set(p['incoming']);fit=set(p['fit']);history=set(p['history'])
                self.assertFalse(incoming&delivered);self.assertTrue(history<=previous)
                self.assertFalse(fit&discarded);self.assertLessEqual(len(history),128)
                self.assertEqual({names[i] for i in y[list(fit)]},set(mod.seen_classes(ds,stage)))
                discarded|=previous-history;delivered|=incoming;previous=fit
                other=plans['natural' if case=='uniform' else 'uniform'][stage]
                b=names.index('benign')
                self.assertEqual({i for i in incoming if y[i]==b},{i for i in other['incoming'] if y[i]==b})
            self.assertEqual(len(delivered),sum(alloc[case].values()))

    def test_old_population_counts_predictions_into_new_class(self):
        # Correct old prediction -> a new class must count as damage, not disappear.
        out=mod.metrics(np.array([0,1]),np.array([0,2]),['benign','old','new'],['new'],['benign','old'])
        self.assertEqual(out['classes'][1]['fn'],1)
        self.assertEqual(out['previous_class_macro_f1'],.5)


if __name__=='__main__':unittest.main()

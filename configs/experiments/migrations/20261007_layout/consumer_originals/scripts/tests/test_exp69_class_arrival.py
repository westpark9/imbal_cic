import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exp69_class_arrival import STAGES, seen_classes, select_context, metrics, encode_labels, validate_allocation
import json


class ProtocolTests(unittest.TestCase):
    def test_fixed_100k_approved_stage_counts(self):
        path=Path(__file__).resolve().parents[2]/'tabpfn/configs/exp70_allocation.json'
        allocation=validate_allocation(json.loads(path.read_text()))
        expected={'cic2018':[84820,89730,90180,95090,100000],
                  'toniot':[80712,86424,89280,92136,97144,100000]}
        for ds,groups in STAGES.items():
            names=sorted(allocation[ds])
            y=np.concatenate([np.full(allocation[ds][name],i) for i,name in enumerate(names)])
            previous=set()
            for stage in range(len(groups)):
                indices=select_context(y,names,ds,stage,100000,allocation)
                self.assertEqual(len(indices),expected[ds][stage])
                self.assertTrue(previous <= set(indices))
                self.assertEqual({names[i] for i in y[indices]},set(seen_classes(ds,stage)))
                for name in seen_classes(ds,stage):
                    self.assertEqual(int((y[indices]==names.index(name)).sum()),allocation[ds][name])
                previous=set(indices)
        allocation['cic2018']['web_attacks']+=1
        with self.assertRaises(ValueError):validate_allocation(allocation)

    def test_no_future_classes_and_nested_budget(self):
        for ds, groups in STAGES.items():
            names=sorted({n for g in groups for n in g})
            y=np.repeat(np.arange(len(names)),300)
            previous=set()
            for stage in range(len(groups)):
                small=select_context(y,names,ds,stage,16)
                large=select_context(y,names,ds,stage,64)
                self.assertTrue(set(small)<=set(large))
                self.assertTrue(previous<=set(large))
                self.assertEqual({names[i] for i in y[large]},set(seen_classes(ds,stage)))
                previous=set(large)

    def test_global_class_ids_are_not_local_output_ids(self):
        names=['benign','scanning','dos']
        np.testing.assert_array_equal(encode_labels(['dos','benign','scanning'],names),[2,0,1])
        with self.assertRaises(KeyError):encode_labels(['xss'],names)

    def test_benign_false_alarm_is_not_attack_miss_rate(self):
        y=np.array([0,0,0,0,1,1]);p=np.array([0,0,0,1,0,0])
        m=metrics(y,p,['benign','attack'],['attack'],['benign'])
        self.assertEqual(m['benign_false_alarm_rate'],.25)
        self.assertEqual(m['classes'][1]['recall'],0)
        self.assertEqual(m['classes'][1]['fp'],1)

    def test_precision_counts_cross_class_false_positives(self):
        y=np.array([0,0,1,2]);p=np.array([1,0,1,1])
        m=metrics(y,p,['benign','new','old'],['new'],['benign','old'])
        self.assertAlmostEqual(m['classes'][1]['precision'],1/3)
        self.assertEqual(m['new_class_macro_f1'],.5)
        self.assertEqual(m['classes'][2]['f1'],0)


if __name__=='__main__':unittest.main()

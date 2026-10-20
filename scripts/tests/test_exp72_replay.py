"""Scientific invariants: causality, no repeated test rows, full-denominator FP."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v7/exp72'))
from replay_protocol import (partition_test, update_schedule, dispatch_versions,
                             fifo_response_times, dist_adjust_current_batch, confusion_metrics)


class ReplayTests(unittest.TestCase):
    def test_each_test_row_once_and_no_future_class(self):
        from replay_protocol import STAGES
        names=['benign','brute_force','dos','ddos','web_attacks','infiltration','bot']
        y=np.repeat(np.arange(7), [101,53,17,20,7,4,9])
        order,batches,bounds=partition_test(y,names,'cic2018',batch_size=13)
        self.assertEqual(sorted(order.tolist()),list(range(len(y))))
        for stage,(lo,hi) in enumerate(bounds):
            permitted=sum(STAGES['cic2018'][:stage+1],[])
            self.assertTrue(all(names[c] in permitted for c in y[order[lo:hi]]))
        self.assertEqual(batches[-1,2],len(y))

    def test_update_queue_and_model_selection(self):
        bounds=np.array([[0,100],[100,200],[200,300]])
        ready=update_schedule(bounds,[99,15,3],10)
        np.testing.assert_equal(ready,[0,25,28])
        batches=np.array([[0,0,100],[1,100,200],[2,200,250],[2,250,280],[2,280,300]])
        np.testing.assert_equal(dispatch_versions(batches,ready,10),[0,0,1,2,2])

    def test_collection_queue_and_service_latency(self):
        b=np.array([[0,0,10],[0,10,20]])
        t=fifo_response_times(b,[2.,.1],10)
        np.testing.assert_allclose(t['finish'],[3,3.1])
        np.testing.assert_allclose(t['queue_wait'],[0,1])
        np.testing.assert_allclose(t['mean_row_latency'],[2.45,1.55])

    def test_distpfn_never_reads_future_batch(self):
        a=np.array([[.9,.1],[.8,.2]])
        before=dist_adjust_current_batch(a,[.5,.5])
        future=np.array([[.01,.99]])
        self.assertFalse(np.allclose(before,dist_adjust_current_batch(np.vstack([a,future]),[.5,.5])[:2]))
        np.testing.assert_allclose(before.sum(axis=1),1)

    def test_other_class_false_positives_lower_tail_precision(self):
        m=confusion_metrics(np.array([0,0,1]),np.array([1,1,1]),['benign','tail'])
        self.assertAlmostEqual(m['classes'][1]['precision'],1/3)
        self.assertEqual(m['classes'][1]['recall'],1)
        self.assertAlmostEqual(m['classes'][1]['f1'],.5)

if __name__=='__main__': unittest.main()

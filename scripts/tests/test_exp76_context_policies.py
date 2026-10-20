"""Guard causal context selection and the unchanged class support population."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(ROOT)
from run_context_policies import replace_context,metric,select_initial

class ContextPolicyTests(unittest.TestCase):
    def test_buffer_replacement_protects_classes_and_never_resurrects(self):
        y=np.array([0,0,0,0,1,1,2,3,3,4,4]);X=np.arange(len(y),dtype=float)[:,None]
        for policy in ['fifo','similar','random']:
            old=np.arange(7);new=np.array([7,8])
            ctx,removed=replace_context(old,new,y,X,policy,43)
            self.assertEqual(len(ctx),7)
            self.assertEqual(set(y[ctx]),{0,1,2,3})
            self.assertEqual(set(removed)|set(ctx),set(old)|set(new))
            nextctx,_=replace_context(ctx,np.array([9,10]),y,X,policy,44)
            self.assertFalse(set(removed)&set(nextctx))
    def test_similarity_removes_nearest_and_fifo_oldest(self):
        y=np.array([0,0,1,1,2]);X=np.array([[0.],[1.],[10.],[11.],[10.1]])
        _,r=replace_context(np.arange(4),np.array([4]),y,X,'similar',43)
        self.assertEqual(r.tolist(),[2])
        _,r=replace_context(np.arange(4),np.array([4]),y,X,'fifo',43)
        self.assertEqual(r.tolist(),[0])
    def test_fixed_model_cannot_hide_new_class_errors(self):
        r=metric(np.array([0,1,2]),np.array([0,1,0]),['benign','old','new'])
        self.assertEqual(r['rows'],3)
        self.assertEqual(r['classes'][2]['recall'],0.)
        self.assertLess(r['macro_f1'],1.)
    def test_initial_context_excludes_future_classes(self):
        names=['benign','brute_force','dos','ddos'];y=np.repeat(np.arange(4),1000)
        ids=select_initial(y,names,'cic2018',1000,43)
        self.assertEqual(len(np.unique(ids)),1000)
        self.assertEqual(np.bincount(y[ids]).tolist(),[750,125,125])

if __name__=='__main__':unittest.main()

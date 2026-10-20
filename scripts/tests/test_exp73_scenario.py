import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v7/exp73'))
from scenario_data import assign_new_roles
from replay_protocol import dist_adjust_current_batch

class Roles(unittest.TestCase):
 def test_persistent_duplicate_groups_and_disjoint_roles(self):
  y=np.repeat(np.arange(3),30);h=np.arange(90)//2;known={}
  a=assign_new_roles(np.arange(60),y,h,known,43)
  saved=known.copy();b=assign_new_roles(np.arange(60,90),y,h,known,44)
  self.assertTrue(all(known[k]==v for k,v in saved.items()))
  self.assertEqual(sorted(np.concatenate(list(a.values())).tolist()),list(range(60)))
  for c in [0,1]:
   self.assertTrue(all((y[ids]==c).any() for ids in a.values()))
  values=list(a.values())
  for i,aa in enumerate(values):
   for bb in values[i+1:]:self.assertEqual(len(np.intersect1d(h[aa],h[bb])),0)
 def test_future_rows_cannot_change_previous_assignment(self):
  y=np.repeat(np.arange(3),30);h=np.arange(90);known={}
  first=assign_new_roles(np.arange(30),y,h,known,43)
  changed=y.copy();changed[30:]=0
  repeat=assign_new_roles(np.arange(30),changed,h,{},43)
  for r in first:np.testing.assert_array_equal(first[r],repeat[r])
 def test_distpfn_uses_current_batch_only(self):
  p=np.array([[.9,.1],[.8,.2]]);prior=np.array([.5,.5])
  a=dist_adjust_current_batch(p,prior)
  b=dist_adjust_current_batch(np.r_[p,[[.01,.99]]],prior)[:2]
  self.assertFalse(np.allclose(a,b));np.testing.assert_allclose(a.sum(1),1)
if __name__=='__main__':unittest.main()

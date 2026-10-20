"""Synthetic execution/serialization checks; never benchmark evidence."""
from scenario_data import ROOT,ROLES
import argparse,json,time
from pathlib import Path
import numpy as np
from replay_models import fit_stage,Predictor
from current_model import fit_current,CurrentPredictor
from exp61_utils import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(43);data=dict(names=['benign','scanning','dos'],synthetic=True)
 offset=0
 for role,n in zip(ROLES,[90,60,60,30,60]):
  y=np.arange(n)%3;X=rng.normal(size=(n,46)).astype('float32');X[:,0]+=y*2
  data['X_'+role]=X;data['y_'+role]=y.astype('int16');data['i_'+role]=np.arange(offset,offset+n);offset+=n
 data.update(Xtr=np.concatenate([data['X_'+r] for r in ROLES[:3]]),ytr=np.concatenate([data['y_'+r] for r in ROLES[:3]]),
             Xva=np.concatenate([data['X_tune'],data['X_cal']]),yva=np.concatenate([data['y_tune'],data['y_cal']]))
 data.update(Xte=data['Xva'][:32],yte=data['yva'][:32])
 config=dict(smoke=True,seed=43,threads=4,batch=32,prepare_batch=256,cost_repeats=1,experts=4,
             checkpoint=str(ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'))
 start=time.time()
 if a.method=='ours':
  info=fit_current(data,config,'toniot',a.out);pred=CurrentPredictor(a.out,config)
 else:
  _,info=fit_stage(a.method,config,'toniot',data,a.out);pred=Predictor(a.method,config,'toniot',data,a.out)
 p1=pred.predict(data['Xte']);p2=pred.predict(data['Xte'])
 assert p1.shape==(32,3) and np.isfinite(p1).all() and np.allclose(p1.sum(1),1,atol=1e-5)
 np.testing.assert_allclose(p1,p2,atol=1e-5)
 if a.method=='ours':
  # Force the call path only in this synthetic test, without changing saved
  # policy or actual-run thresholds; compare to dense selected-expert output.
  from tabpfn import TabPFNClassifier
  if len(pred.models)==1:pred.models += [TabPFNClassifier.load_from_fit_state(a.out/f'model_{k}.tabpfn_fit',device='cuda:0') for k in range(1,5)]
  pred.state['thresholds']=dict(tau_pre='-inf',tau_post=-1e20)
  p3=pred.predict(data['Xte']);assert pred.aux['calls'].all() and pred.aux['accepted'].all()
  for k in range(4):
   ids=np.flatnonzero(pred.aux['expert']==k)
   if len(ids):np.testing.assert_allclose(p3[ids],pred.models[k+1].predict_proba(data['Xte'][ids]),atol=1e-5)
 pred.close();write(a.out/'SMOKE_PASSED.json',dict(method=a.method,seconds=time.time()-start,synthetic=True,update=info['update_seconds']))
 print('SMOKE PASSED',a.method,time.time()-start,flush=True)
if __name__=='__main__':main()

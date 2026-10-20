#!/usr/bin/env python3
"""Independent metrics and context-budget checks against saved predictions."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,json
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score
from exp61_utils import write

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();out=a.run
 cfg=json.loads((out/'config.json').read_text());r=json.loads((out/'results.json').read_text());checks=[]
 for j in r['jobs']:
  ds=j['dataset'];method=j['method'];base=Path(cfg['source'])/'data'/ds;data=out/'data'/ds
  meta=json.loads((data/'manifest.json').read_text());names=meta['class_names'];C=len(names)
  y=np.load(base/'test_y.npy',mmap_mode='r');order=np.load(data/'order.npy');bounds=np.load(data/'bounds.npy')
  assert len(order)==len(y) and len(np.unique(order))==len(y)
  folder=out/'jobs'/f'{ds}_{"pfn_accumulate" if method=="distpfn" else method}'
  pred=np.load(folder/('dist_predictions.npy' if method=='distpfn' else 'predictions.npy'))
  cm=confusion_matrix(y[order],pred,labels=np.arange(C));np.testing.assert_array_equal(cm,j['metrics']['confusion'])
  assert abs(f1_score(y[order],pred,average='macro')-j['metrics']['macro_f1'])<1e-12
  b=names.index('benign');assert abs(1-cm[b,b]/cm[b].sum()-j['metrics']['benign_false_positive_rate'])<1e-12
  ty=np.load(base/'train_y.npy');plans=json.loads((data/'plans.json').read_text());previous=None;discarded=set()
  for i,(fit,plan) in enumerate(zip(j['fits'],plans)):
   ctx=set(fit['context_indices']);release=set(plan['released']);inc=set(plan['incoming']);removed=set(fit['removed_indices'])
   assert len(ctx)==fit['context_rows'] and ctx<=release and not(ctx&discarded)
   if method in ['pfn_fifo','pfn_similar','pfn_random']:
    assert len(ctx)==cfg['context_budget']
    if i:assert inc<=ctx and removed<=previous and len(removed)==len(inc) and ctx==(previous-removed)|inc
   elif method=='pfn_fixed':assert ctx==set(plans[0]['released']) and (not i or fit['update_seconds']==0)
   else:assert ctx==release
   if method!='pfn_fixed':assert set(ty[list(ctx)])==set(ty[list(release)])
   lo,hi=bounds[i];sm=confusion_matrix(y[order[lo:hi]],pred[lo:hi],labels=np.arange(C))
   np.testing.assert_array_equal(sm,j['stages'][i]['confusion'])
   assert set(pred[lo:hi])<=set(names.index(n) for n in fit['classes'])
   discarded|=removed;previous=ctx
  assert abs(sum(f['update_seconds'] for f in j['fits'][1:])-j['update_seconds'])<1e-9
  checks.append(dict(dataset=ds,method=method,rows=len(order),classes=C,complete=True))
 state=dict(complete=r['state']=='complete',passed=True,verified_jobs=len(checks),checks=checks,
            independent_checks=['all test IDs once','sklearn confusion and macro F1','Benign FPR denominator','stage metrics','released-label causality','buffer size and class protection','no discarded-row resurrection','cost sum'])
 write(out/'verification.json',state);print(json.dumps(state,indent=2))
if __name__=='__main__':main()

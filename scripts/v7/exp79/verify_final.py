#!/usr/bin/env python3
"""Independently verify saved full-test predictions and immutable model sources."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,json
import numpy as np
from sklearn.metrics import confusion_matrix
from exp61_utils import write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
    cfg=json.loads((run/'config.json').read_text());r=json.loads((run/'results.json').read_text())
    assert r['state']=='complete' and len(r['jobs'])==20
    assert {(j['dataset'],j['case'],j['method']) for j in r['jobs']}=={(d,c,m) for d in cfg['datasets'] for c in cfg['cases'] for m in ['xgb','tabpfn','ours','boostpfn','distpfn']}
    checked=[];hashed=set()
    additional=json.loads((run/'additional_model_provenance.json').read_text())
    for path,digest in additional['source_hashes'].items():
        assert sha(_root/path)==digest,path;hashed.add(path)
    for j in r['jobs']:
        base=Path(cfg['source'])/'data'/j['dataset'];y=np.load(base/'test_y.npy',mmap_mode='r')
        names=json.loads((base/'manifest.json').read_text())['class_names'];pred=np.load(j['predictions_path'],mmap_mode='r')
        assert len(pred)==len(y)==j['test_rows'] and (pred>=0).all()
        assert j['evaluation']=='final_full_test' and j['batch']==8192 and not j['training_rerun'] and j['restoration_mismatches']==0
        cm=confusion_matrix(y,pred,labels=np.arange(len(names)));np.testing.assert_array_equal(cm,j['metrics']['confusion'])
        denom=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denom,out=np.zeros(len(names)),where=denom!=0)
        assert abs(f1.mean()-j['metrics']['macro_f1'])<1e-12
        for path,digest in j['source_hashes'].items():
            if path in hashed:continue
            assert sha(_root/path)==digest,path;hashed.add(path)
        if j['method']=='ours':
            d=Path(j['predictions_path']).parent;g=np.load(d/'global_pred.npy');calls=np.load(d/'calls.npy');accepted=np.load(d/'accepted.npy')
            assert np.all(accepted<=calls)
            assert j['policy']['called']==int(calls.sum()) and j['policy']['accepted']==int(accepted.sum())
            assert j['policy']['helpful']-j['policy']['harmful']==int((pred==y).sum()-(g==y).sum())
        checked.append(dict(dataset=j['dataset'],case=j['case'],method=j['method'],macro_f1=float(f1.mean()),predictions_sha256=sha(j['predictions_path'])))
    write(run/'verification.json',dict(passed=True,jobs=checked,source_files_unchanged=len(hashed),training_rerun=False,seed_repetitions=1,
        checks=['complete original-order test','independent confusion matrix and macro F1','saved final-stage model restoration','immutable source model hashes','proposed policy identities']))
    print(json.dumps(dict(passed=True,jobs=len(checked))))


if __name__=='__main__':main()

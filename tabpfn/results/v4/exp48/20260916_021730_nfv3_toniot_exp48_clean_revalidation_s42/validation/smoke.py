import json,sys,tempfile,subprocess,pickle
from pathlib import Path
import numpy as np
repo=Path('/home/user/Desktop/imbalcic'); sys.path.insert(0,str(repo/'tabpfn/scripts'))
from nfv3_conflict_clean import prepare, write_json
from nfv3_v3_exp48_ton_clean_revalidation import prepare_run
root=Path(tempfile.mkdtemp(prefix='exp48_smoke_')); run=root/'run'; run.mkdir()
names=['backdoor','benign','ddos','dos','injection','mitm','password','ransomware','scanning','xss']
y=np.repeat(np.arange(10),700); rng=np.random.default_rng(42); X=rng.normal(size=(len(y),46)).astype('float32'); X[:,0]+=y*2
# Two cross-label contradictory rows are excluded; a same-label duplicate survives.
X[700]=X[0]; X[2]=X[1]
d={'X':X,'families':np.array(names)[y],'dataset_names':np.repeat('ton_iot',len(y)),'attack_scenarios':np.array(names)[y], 'timestamps':np.arange(len(y)), 'feature_names':[f'f{i}' for i in range(46)]}
with (root/'synthetic.pkl').open('wb') as f:pickle.dump(d,f)
prepare(root/'synthetic.pkl',root/'clean','ton_iot')
manifest=json.loads((root/'clean/manifest.json').read_text()); assert manifest['rows_removed']==2; assert manifest['class_names'].index('benign')==1
launch={'run_dir':str(run),'clean_dir':str(root/'clean'),'baseline_source_run':str(repo/'tabpfn/results/20260904_010654_nfv3_cic2018_exp31_c0alloc')}
write_json(run/'launch.json',launch); prepare_run(run)
args=json.loads(Path('/tmp/exp47_smoke_1j4w85yz/run/baseline_args.json').read_text())
args.update(verifier_quantile=0.75, scorer_n_estimators=50, verifier_n_estimators=50, target_dataset='ton_iot',data=str(root/'synthetic.pkl'),clean_manifest=str(root/'clean/manifest.json'))
for k,sub in [('out_root','baseline'),('resume_dir','resume'),('models_dir','models')]:args[k]=str(run/sub)
write_json(run/'baseline_args.json',args)
print('SMOKE_ROOT:',root,flush=True)
subprocess.run([sys.executable,'-u',str(repo/'tabpfn/scripts/nfv3_v3_exp48_ton_clean_revalidation.py'),'--run-dir',str(run)],check=True)
meta=json.loads((run/'frozen_cache/COMPLETE.json').read_text())
assert meta['dataset']=='ton_iot' and meta['tail_classes']==['mitm','ransomware'] and meta['protected_classes']==['ddos','dos']
assert (run/'diagnostics/COMPLETE.json').exists()
Path('/tmp/exp48_smoke_result.json').write_text(json.dumps({'root':str(root),'run':str(run),'passed':True}))
print('SMOKE PASSED',root)

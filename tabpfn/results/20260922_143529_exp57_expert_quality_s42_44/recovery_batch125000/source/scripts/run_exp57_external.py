#!/usr/bin/env python3
"""Portable, shardable EXP57 launch. Default external assignment: both datasets, seed 42."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA='nfv3_energy_suite_uncapped_scenarios.pkl'
CHECKPOINT='tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'
CLEAN={'cic2018':'cic2018_conflict_free_fixed_split_20260915_180000','toniot':'toniot_conflict_free_fixed_split_20260916_021730'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True,help='Contains data/ and tabpfn/ from the input bundle; repository root also works')
    p.add_argument('--datasets',nargs='+',choices=list(CLEAN),default=list(CLEAN));p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--root',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--batch-size',type=int,default=125000)
    p.add_argument('--cpu-threads',type=int,default=1);p.add_argument('--detach',action='store_true');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.batch_size<1 or a.cpu_threads<1:p.error('batch size and CPU threads must be positive')
    inputs=a.inputs.resolve();data=inputs/'data'/DATA;checkpoint=inputs/'tabpfn'/CHECKPOINT
    required=[data,checkpoint]
    for ds in a.datasets:
        required += [inputs/'data/derived'/CLEAN[ds]/f for f in ['manifest.json','COMPLETE.json','train_idx.npy','val_idx.npy','test_idx.npy','scenario_counts.csv']]
    missing=[str(x) for x in required if not x.is_file()]
    if missing:p.error('Missing required inputs:\n'+'\n'.join(missing))
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS=str(a.cpu_threads),MKL_NUM_THREADS=str(a.cpu_threads),
                      OPENBLAS_NUM_THREADS=str(a.cpu_threads),NUMEXPR_NUM_THREADS=str(a.cpu_threads),MALLOC_ARENA_MAX='2',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import torch
    if not torch.cuda.is_available():p.error('CUDA is unavailable; CPU fallback is not allowed for this experiment')
    capability=torch.cuda.get_device_capability(0)
    if capability[0]<8:p.error('Requires compute capability >=8.0 for the long-context attention backend')
    sys.path.insert(0,str(ROOT/'tabpfn/scripts'))
    from exp57_expert_quality_suite import prepare,write
    root=(a.root or ROOT/'tabpfn/results'/('exp57_external_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    prepare(root,ROOT,datasets=list(dict.fromkeys(a.datasets)),seeds=list(dict.fromkeys(a.seeds)),data=data,checkpoint=checkpoint,
            clean_root=inputs/'data/derived',test_batch_size=a.batch_size,cpu_threads=a.cpu_threads,portable_inputs=True)
    versions={k:importlib.metadata.version(k) for k in ['torch','tabpfn','numpy','pandas','scikit-learn','scipy','xgboost','matplotlib','threadpoolctl']}
    write(root/'environment.json',dict(python=sys.version,packages=versions,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,compute_capability=capability))
    if a.prepare_only:print('Prepared only; no training started.');return
    command=[sys.executable,'-u',str(root/'source/tabpfn/scripts/exp57_expert_quality_suite.py'),'--root',str(root),'--stage','controller']
    if a.detach:
        with (root/'controller.log').open('a') as log:
            process=subprocess.Popen(command,cwd=root/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'external_launch.json',dict(pid=process.pid,command=command));print(json.dumps(dict(pid=process.pid,root=str(root),status=str(root/'status.json')),indent=2))
    else:raise SystemExit(subprocess.call(command,cwd=root/'source'))


if __name__=='__main__':main()

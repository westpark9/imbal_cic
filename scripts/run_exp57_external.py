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
import traceback

ROOT=Path(__file__).resolve().parents[1]
DATA='nfv3_energy_suite_uncapped_scenarios.pkl'
CHECKPOINT='tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'
CLEAN={'cic2018':'cic2018_conflict_free_fixed_split_20260915_180000','toniot':'toniot_conflict_free_fixed_split_20260916_021730'}


def run_prepared(root):
    """The detached process owns both preprocessing and the training controller."""
    root=Path(root).resolve()
    request=json.loads((root/'external_request.json').read_text())
    sys.path.insert(0,str(ROOT/'tabpfn/scripts'))
    from exp57_expert_quality_suite import write
    from exp57_prepare_inputs import ensure_clean_split,sha256
    verification={}
    try:
        write(root/'status.json',dict(state='verifying_source',launcher_pid=os.getpid(),completed=[]))
        # Validate the user's existing PKL before spending time regenerating splits.
        if sha256(request['data']) != request['reference']['source']['sha256']:
            raise ValueError('Source PKL SHA-256 differs from the EXP57 benchmark')
        for ds in request['datasets']:
            write(root/'status.json',dict(state='preparing_clean',dataset=ds,launcher_pid=os.getpid(),completed=[]))
            ref=dict(source=request['reference']['source'],dataset=request['reference']['datasets'][ds])
            directory=Path(request['clean_root'])/CLEAN[ds]
            verification[ds]=ensure_clean_split(request['data'],directory,ref)
            write(root/'clean_input_verification.json',verification)
        if request['prepare_only']:
            write(root/'status.json',dict(state='prepared',jobs=json.loads((root/'protocol.json').read_text())['jobs'],completed=[]))
            print('Inputs verified and jobs prepared; no training started.',flush=True)
            return 0
        command=[sys.executable,'-u',str(root/'source/tabpfn/scripts/exp57_expert_quality_suite.py'),'--root',str(root),'--stage','controller']
        return subprocess.call(command,cwd=root/'source')
    except Exception:
        write(root/'status.json',dict(state='input_preparation_failed',launcher_pid=os.getpid(),traceback=traceback.format_exc(),completed=[]))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs_group=p.add_mutually_exclusive_group()
    inputs_group.add_argument('--data',type=Path,help='Existing nfv3_energy_suite PKL; no input archive required')
    inputs_group.add_argument('--inputs',type=Path,help='Optional legacy bundle root containing data/ and tabpfn/')
    p.add_argument('--model-path','--checkpoint',dest='checkpoint',type=Path,help='Existing TabPFN-v3 checkpoint; default: repository tabpfn/ checkpoint')
    p.add_argument('--clean-root',type=Path,help='Reuse or automatically generate clean splits here; default: tabpfn/results/exp57_clean_splits')
    p.add_argument('--datasets',nargs='+',choices=list(CLEAN),default=list(CLEAN));p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--root',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--batch-size',type=int,default=125000)
    p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--detach',action='store_true',help='Run source verification, split preparation and training in the background')
    p.add_argument('--prepare-only',action='store_true',help='Prepare and verify inputs and jobs without fitting models')
    p.add_argument('--run-prepared',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.run_prepared:raise SystemExit(run_prepared(a.run_prepared))
    if a.data is None and a.inputs is None:p.error('Provide --data /path/to/existing.pkl (or legacy --inputs)')
    if a.batch_size<1 or a.cpu_threads<1:p.error('batch size and CPU threads must be positive')
    inputs=a.inputs.resolve() if a.inputs else ROOT
    data=(a.data if a.data is not None else inputs/'data'/DATA).resolve()
    checkpoint=(a.checkpoint or inputs/'tabpfn'/CHECKPOINT).resolve()
    clean_root=(a.clean_root or (inputs/'data/derived' if a.inputs else ROOT/'tabpfn/results/exp57_clean_splits')).resolve()
    required=[data,checkpoint]
    missing=[str(x) for x in required if not x.is_file()]
    if missing:p.error('Missing required inputs:\n'+'\n'.join(missing))
    reference=json.loads((ROOT/'tabpfn/configs/exp57_clean_split_reference.json').read_text())
    if data.stat().st_size != reference['source']['bytes']:p.error('Source PKL size differs from the EXP57 benchmark')
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS=str(a.cpu_threads),MKL_NUM_THREADS=str(a.cpu_threads),
                      OPENBLAS_NUM_THREADS=str(a.cpu_threads),NUMEXPR_NUM_THREADS=str(a.cpu_threads),MALLOC_ARENA_MAX='2',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import torch
    if not torch.cuda.is_available():
        p.error(f'CUDA is unavailable (python={sys.executable}, torch={torch.__version__}, '
                f'wheel CUDA={torch.version.cuda}, selected GPU={a.gpu}). '
                'Create a driver-compatible environment with: '
                'python scripts/setup_exp57_env.py --prefix /YOUR_PERSISTENT_MOUNT/envs/exp57 '
                f'--gpu {a.gpu}; then launch with that prefix/bin/python. CPU fallback is not allowed.')
    capability=torch.cuda.get_device_capability(0)
    if capability[0]<8:p.error('Requires compute capability >=8.0 for the long-context attention backend')
    sys.path.insert(0,str(ROOT/'tabpfn/scripts'))
    from exp57_expert_quality_suite import prepare,write
    root=(a.root or ROOT/'tabpfn/results'/('exp57_external_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    prepare(root,ROOT,datasets=list(dict.fromkeys(a.datasets)),seeds=list(dict.fromkeys(a.seeds)),data=data,checkpoint=checkpoint,
            clean_root=clean_root,test_batch_size=a.batch_size,cpu_threads=a.cpu_threads,portable_inputs=True)
    write(root/'external_request.json',dict(data=str(data),checkpoint=str(checkpoint),clean_root=str(clean_root),datasets=list(dict.fromkeys(a.datasets)),reference=reference,prepare_only=a.prepare_only))
    versions={k:importlib.metadata.version(k) for k in ['torch','tabpfn','numpy','pandas','scikit-learn','scipy','xgboost','matplotlib','threadpoolctl']}
    write(root/'environment.json',dict(python=sys.version,packages=versions,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,compute_capability=capability))
    command=[sys.executable,'-u',str(root/'source/scripts/run_exp57_external.py'),'--run-prepared',str(root)]
    if a.detach:
        with (root/'controller.log').open('a') as log:
            process=subprocess.Popen(command,cwd=root/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'external_launch.json',dict(pid=process.pid,command=command));print(json.dumps(dict(pid=process.pid,root=str(root),status=str(root/'status.json')),indent=2))
    else:raise SystemExit(subprocess.call(command,cwd=root/'source'))


if __name__=='__main__':main()

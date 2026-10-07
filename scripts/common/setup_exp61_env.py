#!/usr/bin/env python3
"""Reuse the driver-compatible EXP57 prefix and install SOTA-specific dependencies."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import argparse
from pathlib import Path
import subprocess
import sys

ROOT=repo_root(__file__)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--gpu',default='0')
    a=p.parse_args();prefix=a.prefix.resolve()
    subprocess.run([sys.executable,str(script_path('setup_exp57_env.py')),'--prefix',str(prefix),'--gpu',a.gpu],check=True)
    python=prefix/'bin/python'
    # Keep the working CUDA torch build while resolving the added packages.
    version=subprocess.check_output([str(python),'-c','import torch;print(torch.__version__)'],text=True).strip()
    constraints=prefix/'exp61_torch_constraints.txt';constraints.write_text('torch=='+version+'\n')
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements-sota.txt'),'-c',str(constraints)],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    subprocess.run([str(python),'-c','import faiss,psutil,tensorboard,torch,xgboost,tabpfn;print("EXP61 dependencies ready")'],check=True)
    subprocess.run(['bash',str(ROOT/'tabpfn/third_party/fetch_checkpoints.sh')],check=True)
    print(f'Ready: {python}')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Snapshot only fully evaluated context arms without touching the GPU worker."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tabpfn/scripts'))
from exp59_residual_membership_oracle import ARMS,analyze,write
RESULTS=ROOT/'tabpfn/results/20260928_exp59_residual_oracle_s43'


def snapshot(dataset):
    run=RESULTS/f'{dataset}_s43';source=run/'fresh_bank'
    design=json.loads((source/'design.json').read_text())
    # Read each small live table once; the saved snapshot stays immutable later.
    tables={name:pd.read_csv(io.StringIO((source/(name+'.csv')).read_text()))
            for name in ['summary','per_class','probability_metrics','context_compositions']}
    models=set(tables['summary'].model)
    arms=[a for a in ARMS if all(f'{a}_e{k}_raw' in models for k in range(1,5))]
    if not arms:raise RuntimeError('No fully evaluated expert context arm yet')
    wanted=['global_raw']+[f'{a}_e{k}_raw' for a in arms for k in range(1,5)]
    for name in ['per_class','probability_metrics']:
        for model in wanted:
            assert len(tables[name].query('model == @model'))==design['C'],(name,model)
    out=run/'partial';bundle=out/'fresh_bank';bundle.mkdir(parents=True,exist_ok=True)
    for name,frame in tables.items():
        frame=frame[frame.arm.isin(arms)] if name=='context_compositions' else frame[frame.model.isin(wanted)]
        frame.to_csv(bundle/(name+'.csv'),index=False)
    for name in ['design.json','evaluation_identity.npz','shared_context_ids.npz','job.json']:
        shutil.copy2(source/name,bundle/name)
    # Only completed prediction files are linked; the worker never rewrites these.
    for model in wanted:
        name=model.removesuffix('_raw')
        files=[f'predictions/{model}.npy',f'probabilities/{name}_test.npy']
        if name!='global':files.append(f'contexts/{name}.npy')
        for f in files:
            target=bundle/f;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():os.link(source/f,target)
    for name in ['residual_test_regions.npy','residual_state.npz','RECONSTRUCTION_COMPLETE.json',
                 'reconstruction_audit.json','split_manifest.csv','train_pool_partition.csv']:
        shutil.copy2(run/name,out/name)
    write(out/'snapshot.json',dict(source=str(source),created_epoch=time.time(),completed_arms=arms,
        purpose='partial report; official experiment completion remains at dataset root'))
    analyze(bundle,out,arms=arms)
    print(out)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['cic2018','toniot'],required=True)
    snapshot(p.parse_args().dataset)

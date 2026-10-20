#!/usr/bin/env python3
"""Independently audit saved predictions, causal schedules, and paired IDs."""
from replay_protocol import ROOT, STAGES
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from exp61_utils import write
from exp69_class_arrival import utc


def verify(out,require_all=False):
    config=json.loads((out/'config.json').read_text())
    checks=[]
    for ds in config['datasets']:
        prep=out/'data'/ds;base=Path(config['source'])/'data'/ds
        meta=json.loads((prep/'manifest.json').read_text());names=meta['class_names']
        y_original=np.load(base/'test_y.npy',mmap_mode='r')
        order=np.load(prep/'order.npy');batches=np.load(prep/'batches.npy');bounds=np.load(prep/'stage_bounds.npy')
        np.testing.assert_array_equal(np.sort(order),np.arange(len(y_original)))
        assert batches[0,1]==0 and batches[-1,2]==len(order)
        np.testing.assert_array_equal(batches[1:,1],batches[:-1,2])
        y=y_original[order]
        for case in config['cases']:
            plans=json.loads((base/'plans.json').read_text())[case]
            for method in config['methods']:
                job=out/'jobs'/f'{ds}_{case}_{method}'
                if not (job/'COMPLETE.json').exists():
                    if require_all:raise ValueError('Missing completed job: '+str(job))
                    continue
                for stage,plan in enumerate(plans):
                    actual=np.load(job/f'stage_{stage}'/'train_ids.npy')
                    expected=np.load(base/'train_ids.npy',mmap_mode='r')[plan['fit']]
                    np.testing.assert_array_equal(actual,expected)
                schedule=json.loads((job/'schedule.json').read_text())
                result=json.loads((job/'results.json').read_text())['results']
                for key,reported in result.items():
                    pred=np.load(job/f'pred_{key}.npy')
                    assert len(pred)==len(y)
                    cm=confusion_matrix(y,pred,labels=np.arange(len(names)))
                    np.testing.assert_array_equal(cm,reported['confusion'])
                    independent=f1_score(y,pred,labels=np.arange(len(names)),average='macro',zero_division=0)
                    assert abs(independent-reported['macro_f1'])<1e-12
                    versions=np.load(job/f'versions_{key}.npy')
                    if key=='ideal':np.testing.assert_array_equal(versions,batches[:,0])
                    else:
                        rate=int(key.split('_')[1]);r=schedule['rates'][key];ready=np.asarray(r['ready_seconds'])
                        for stage in range(1,len(bounds)):
                            expected=max(bounds[stage,0]/rate,ready[stage-1])+schedule['updates_seconds'][stage]
                            assert abs(expected-ready[stage])<1e-8
                        close=batches[:,2]/rate
                        assert np.all(ready[versions]<=close+1e-9)
                        assert np.all(versions<=batches[:,0])
                        timeline=np.load(job/f'timeline_{key}.npz')
                        finish=timeline['finish'];service=timeline['service_seconds']
                        start=finish-service
                        assert np.all(start>=close-1e-8)
                        assert np.all(start[1:]>=finish[:-1]-1e-8)
                        assert np.all(np.diff(finish)>=0)
                    # The written stream must use the very predictions of its
                    # scheduled version, never ideal predictions relabelled.
                    saved={}
                    for j,(_,lo,hi) in enumerate(batches):
                        v=int(versions[j])
                        if v not in saved:saved[v]=np.load(job/f'stage_{v}'/'replay_pred.npy',mmap_mode='r')
                        np.testing.assert_array_equal(pred[lo:hi],saved[v][lo:hi])
                    checks.append(dict(dataset=ds,case=case,method=method,policy=key,rows=len(y),passed=True))
    write(out/'verification.json',dict(state='passed' if require_all else 'partial_passed',
          checks=checks,expected_policies=24,verified_policies=len(checks),verified_at=utc()))
    print('Verified',len(checks),'policies; full=',require_all,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args();out=a.out.resolve()
    try:
        if a.wait:
            while True:
                status=json.loads((out/'status.json').read_text())
                if status['state']=='complete':break
                if status['state'] in ['completed_with_failures','failed','paused']:
                    write(out/'verification.json',dict(state='waiting_for_successful_resume',status=status,at=utc()))
                    return 1
                time.sleep(15)
        verify(out,require_all=a.wait)
    except Exception:
        write(out/'verification.json',dict(state='failed',error=traceback.format_exc(),at=utc()))
        raise

if __name__=='__main__':raise SystemExit(main())

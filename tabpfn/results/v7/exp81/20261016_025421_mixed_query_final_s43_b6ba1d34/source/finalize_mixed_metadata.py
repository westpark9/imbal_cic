#!/usr/bin/env python3
"""Replace inherited prediction diagnostics; retain unchanged update costs."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,json,shutil
import numpy as np
from exp61_utils import write
from replay_protocol import confusion_metrics
from mixed_query_eval import SMALL


def finalize(run):
    marker=run/'metadata_finalization.json'
    if marker.exists():
        assert json.loads(marker.read_text())['complete'];return
    result=json.loads((run/'results.json').read_text());assert result['state']=='complete'
    data={}
    for ds in ['cic2018','toniot']:
        meta=json.loads((SMALL/'data'/ds/'manifest.json').read_text());base=Path(meta['source'])
        plans=json.loads((SMALL/'data'/ds/'plans.json').read_text());ids=plans[-1]['released']
        train_hash=np.load(base/'train_hash.npy',mmap_mode='r');test_hash=np.load(base/'test_hash.npy',mmap_mode='r')
        mask=~np.isin(test_hash,train_hash[ids]);y=np.load(base/'test_y.npy',mmap_mode='r')
        data[ds]=(y,meta['class_names'],mask)
    changed=[]
    for path in sorted((run/'jobs').glob('*/*results.json')):
        j=json.loads(path.read_text());inherited={}
        if 'note' in j:inherited['note']=j.pop('note')
        if j['full_reinference']:
            if 'serving_seconds' in j:inherited['serving_seconds']=j.pop('serving_seconds')
            if j['method'] in ['pfn_pretrained','tabpfn','distpfn_pretrained','distpfn']:
                times=np.load(path.parent/'service_times.npy',mmap_mode='r')
                j['base_inference_seconds']=float(times[:,2].sum())
                if 'distpfn' in j['method']:
                    j['serving_seconds_scope']='Total not measured; base_inference_seconds covers PFN forward only; posterior correction was not separately timed'
                else:
                    j['serving_seconds']=j['base_inference_seconds']
                    j['serving_seconds_scope']='GPU-synchronized PFN forward on mixed query batches; disk I/O excluded'
            else:
                j['serving_seconds_scope']='Fallback mixed inference was not timed; inherited original timing retained separately'
            if 'feature_disjoint_diagnostic' in j:
                inherited['feature_disjoint_diagnostic']=j.pop('feature_disjoint_diagnostic')
                y,names,mask=data[j['dataset']];pred=np.load(j['predictions_path'],mmap_mode='r')
                j['feature_disjoint_diagnostic']=confusion_metrics(y[mask],np.asarray(pred)[mask],names)
        else:
            j['serving_seconds_scope']='Original full-test inference measurement; full predictions reused after mixed-query audit'
        j['inherited_metadata_from_source_result']=inherited
        j['note']='Saved model, unchanged training/context; seed43 mixed query evaluation or explicitly audited prediction reuse; update costs retain their original preparation measurements'
        write(path,j);changed.append(str(path.relative_to(run)))
    jobs=[json.loads(p.read_text()) for p in sorted((run/'jobs').glob('*/*results.json'))]
    write(run/'results.json',dict(state='complete',jobs=jobs))
    shutil.copy2(__file__,run/'source'/Path(__file__).name)
    write(marker,dict(complete=True,predictions_changed=False,primary_metrics_changed=False,update_costs_changed=False,
        reason='Recompute feature-disjoint diagnostics for the new predictions and distinguish newly measured forward time from inherited timing',files=changed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);finalize(p.parse_args().run)

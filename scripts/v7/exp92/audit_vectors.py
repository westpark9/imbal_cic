"""Independent export, split, checkpoint pooling, and collision audit."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from run_pipeline import ROOT,resolve_path,save_json

def audit(out,check_encoder=False):
    with (out/'flows.jsonl').open() as f:rows=[json.loads(line) for line in f]
    x=np.load(out/'embeddings.npy')
    with np.load(out/'dataset.npz') as z:
        ids,zx,y,split=z['flow_id'],z['X'],z['y'],z['split']
    assert list(ids)==[r['flow_id'] for r in rows]
    assert len(set(ids))==len(ids)
    assert len(x)==len(zx)==len(y)==len(rows)
    assert np.isfinite(x).all() and np.isfinite(zx).all()
    with np.load(out/'pca.npz') as p:
        train_ids=set(ids[split=='train']);assert set(p['fit_flow_ids'])==train_ids
        np.testing.assert_allclose(p['mean'],x[split=='train'].mean(0),rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(zx,(x-p['mean'])@p['components'].T,rtol=1e-4,atol=1e-5)
    groups={s:{r['group_id'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    assert not(groups['train']&groups['test'] or groups['train']&groups['validation'] or groups['validation']&groups['test'])
    for s in groups:
        with np.load(out/(s+'.npz')) as part:
            np.testing.assert_array_equal(part['flow_id'],ids[(split==s)&(y>=0)])
            assert np.all(part['y']>=0)
    by_vector=defaultdict(list)
    for i,v in enumerate(x):by_vector[hashlib.sha256(v.tobytes()).hexdigest()].append(i)
    conflicts=[];overlap=0;annotation_conflicts=0
    for key,ii in by_vector.items():
        if len({rows[i]['split'] for i in ii})>1:overlap+=1
        labels=Counter(rows[i]['label'] for i in ii if rows[i]['label'])
        annotations={rows[i].get('annotation_label',rows[i]['label']) for i in ii if rows[i].get('annotation_label',rows[i]['label'])}
        if len(annotations)>1:annotation_conflicts+=1
        if len(labels)<2:continue
        examples=[]
        for label in sorted(labels):
            i=next(i for i in ii if rows[i]['label']==label);r=rows[i]
            examples.append({k:r[k] for k in ['flow_id','label','split','src_ip','dst_ip','src_port','dst_port','start_ns','end_ns','packets','fwd_payload_bytes','bwd_payload_bytes','first_packet_index','last_packet_index']})
        conflicts.append(dict(embedding_sha256=key,rows=len(ii),label_counts=dict(labels),examples=examples))
    parity=None
    if check_encoder:
        from netfound_encoder import NetFoundEncoder
        config=json.loads((out/'pipeline_config.json').read_text());enc=NetFoundEncoder(ROOT,resolve_path(config['checkpoint']))
        lo=min(range(len(rows)),key=lambda i:sum(map(len,rows[i]['burst_tokens'])))
        hi=max(range(len(rows)),key=lambda i:sum(map(len,rows[i]['burst_tokens'])))
        a,b=enc.encode([rows[lo]]),enc.encode([rows[hi]])
        together=enc.encode([rows[lo],rows[hi]])
        np.testing.assert_allclose(together,np.concatenate([a,b]),rtol=2e-4,atol=2e-4)
        parity=float(np.max(np.abs(together-np.concatenate([a,b]))))
    result=dict(passed=True,rows=len(rows),labeled_rows=int((y>=0).sum()),unresolved_rows=int((y<0).sum()),
                group_overlap=0,exact_embedding_groups_across_splits=overlap,
                embedding_conflict_groups=len(conflicts),embedding_conflicts=conflicts,
                annotation_conflict_groups=annotation_conflicts,
                varied_length_batch_parity_max_abs_error=parity,
                interpretation='Group-disjoint is not feature-disjoint. No collision rows removed. This is a representation/label audit, not classifier evaluation.')
    save_json(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='embedding_conflicts'},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--check-encoder',action='store_true')
    args=p.parse_args();audit(resolve_path(args.run),args.check_encoder)

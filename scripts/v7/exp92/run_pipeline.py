"""PCAP -> auditable flows/labels -> frozen netFound -> train-only PCA -> NPZ/CSV.

Run from the repository root with the experiment registry. All paths in a
configuration are repo-relative or absolute; there are no machine-specific
historical result paths. No online replay or classifier training occurs here.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import random
import sys
import time

_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').exists())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import repo_root, resolve_path
from fetch_assets import sha256
from flows import iter_flows
from labels import LabelRules

ROOT = repo_root(__file__)

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def split_group(group, seed, fractions):
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16],16)/2**64
    return 'train' if value < fractions[0] else 'validation' if value < sum(fractions[:2]) else 'test'

def prepare(config, out):
    rng = random.Random(config['seed']); counts, sampled, raw_stats = Counter(), defaultdict(list), {}
    label_sources, capture_sources = [], []
    cap = int(config.get('reservoir_per_label_split',0))
    if cap < 0: raise ValueError('Negative reservoir size')
    fractions = config.get('split_fractions',[.7,.15,.15])
    if len(fractions)!=3 or min(fractions)<0 or abs(sum(fractions)-1)>1e-9: raise ValueError('Invalid split fractions')
    ids = [c['id'] for c in config['captures']]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate capture IDs')
    # cap=0 is exhaustive streaming; pilot uses a deterministic per-label/split reservoir.
    path = out/'flows.jsonl'
    with path.open('x') as writer:
        for capture in config['captures']:
            pcap, rules_path = resolve_path(capture['path']), resolve_path(capture['labels'])
            if not pcap.exists(): raise FileNotFoundError(f'PCAP missing: {pcap}; CSV files cannot replace a PCAP')
            rules = LabelRules(rules_path)
            label_sources.append(dict(path=capture['labels'],sha256=sha256(rules_path)))
            capture_sources.append(dict(id=capture['id'],path=capture['path'],bytes=pcap.stat().st_size,sha256=sha256(pcap)))
            stats = Counter()
            for row in iter_flows(pcap,capture['id'],capture.get('group_namespace',config['dataset']),stats,**config.get('flow',{})):
                row.update(rules.apply(row))
                if row.get('capture_truncated'):
                    row['label']=None
                    row['label_status']='incomplete_capture'
                row['split'] = split_group(row['group_id'],config['seed'],fractions)
                key = (row['annotation_label'] or '__UNRESOLVED__',row['split'])
                counts[key] += 1
                stats['flows'] += 1; stats['label_status_'+row['label_status']] += 1
                if cap:
                    bucket = sampled[key]
                    if len(bucket) < cap: bucket.append(row)
                    else:
                        j = rng.randrange(counts[key])
                        if j < cap: bucket[j] = row
                else: writer.write(json.dumps(row,separators=(',',':'))+'\n')
                if stats['flows'] % 100000 == 0:
                    print(json.dumps(dict(capture=capture['id'],flows=stats['flows'],packets=stats['raw_packets'])),flush=True)
            raw_stats[capture['id']] = dict(stats)
        if cap:
            for key in sorted(sampled):
                for row in sorted(sampled[key],key=lambda r:r['flow_id']):
                    writer.write(json.dumps(row,separators=(',',':'))+'\n')
    manifest = dict(dataset=config['dataset'],seed=config['seed'],raw_stats=raw_stats,captures=capture_sources,
                    label_rules=label_sources,source_counts=[dict(label=k[0],split=k[1],count=v) for k,v in sorted(counts.items())],
                    sampled_rows=sum(len(v) for v in sampled.values()) if cap else sum(counts.values()),
                    reservoir_per_label_split=cap, sampling_strata='annotation label x split (Attempted categories are metadata, not target classes)',
                    split='SHA256 of capture group namespace + bidirectional 5-tuple; before sampling/PCA',
                    flows_sha256=sha256(path), unknown_policy='retained with label=-1; excluded from supervised exports',
                    short_flow_policy='one-packet traffic retained; closed-session pure ACK tails counted separately')
    save_json(out/'prepare.json',manifest)
    return manifest

def embed(config,out,device,batch_size):
    import numpy as np
    from netfound_encoder import NetFoundEncoder
    manifest = json.loads((out/'prepare.json').read_text())
    if sha256(out/'flows.jsonl') != manifest['flows_sha256']: raise RuntimeError('Prepared flow file was modified')
    checkpoint = resolve_path(config['checkpoint'])
    encoder = NetFoundEncoder(ROOT,checkpoint,device,config.get('dtype','float32'))
    embeddings = np.lib.format.open_memmap(out/'embeddings.npy',mode='w+',dtype='float32',shape=(manifest['sampled_rows'],encoder.metadata['hidden_size']))
    offset, batch, parity = 0, [], None
    def flush():
        nonlocal offset, batch, parity
        if not batch: return
        result = encoder.encode(batch)
        if parity is None:
            single = encoder.encode([batch[0]])[0]
            parity = float(np.max(np.abs(single-result[0])))
            if not np.allclose(single,result[0],atol=2e-4,rtol=2e-4):
                raise AssertionError(f'Batch-dependent embedding (padding error): {parity}')
        embeddings[offset:offset+len(batch)] = result
        offset += len(batch); batch = []
        if offset % (batch_size*10)==0: print(f'Encoded {offset}/{len(embeddings)} flows',flush=True)
    with (out/'flows.jsonl').open() as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= batch_size: flush()
    flush(); embeddings.flush()
    if offset != len(embeddings): raise AssertionError('Embedding row count mismatch')
    save_json(out/'encoder.json',{**encoder.metadata,'rows':offset,'batch_size':batch_size,'batch_parity_max_abs_error':parity,
                                  'embeddings_sha256':sha256(out/'embeddings.npy')})

def project(config,out):
    import csv
    import numpy as np
    from sklearn.decomposition import PCA
    with (out/'flows.jsonl').open() as f:
        # Projection needs identity/labels only, not the potentially large token arrays.
        rows = [{k:r[k] for k in ['flow_id','group_id','split','label','label_status']}
                for r in (json.loads(line) for line in f)]
    x = np.load(out/'embeddings.npy',mmap_mode='r')
    enc = json.loads((out/'encoder.json').read_text())
    if sha256(out/'embeddings.npy')!=enc['embeddings_sha256']: raise RuntimeError('Embedding file was modified')
    split = np.array([r['split'] for r in rows])
    train = split=='train'
    if not train.any() or train.sum()<2: raise ValueError('PCA needs at least two train flows')
    for s in ['validation','test']:
        if not (split==s).any(): raise ValueError(f'Empty {s} split; add captures/flows')
    groups = {s:{r['group_id'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    if groups['train'] & (groups['test'] | groups['validation']) or groups['test'] & groups['validation']:
        raise AssertionError('A flow group leaked across splits')
    dimension = min(int(config['pca_components']),int(train.sum())-1,x.shape[1])
    if dimension < 1: raise ValueError('Invalid PCA dimension')
    pca = PCA(n_components=dimension,svd_solver='full',random_state=config['seed']).fit(x[train])
    z = pca.transform(x).astype(np.float32)
    if not np.isfinite(z).all(): raise FloatingPointError('Non-finite PCA output')
    classes = sorted({r['label'] for r in rows if r['label'] is not None})
    y = np.array([classes.index(r['label']) if r['label'] is not None else -1 for r in rows],dtype=np.int64)
    ids = np.array([r['flow_id'] for r in rows])
    np.savez_compressed(out/'pca.npz',mean=pca.mean_,components=pca.components_,explained_variance_ratio=pca.explained_variance_ratio_,fit_flow_ids=ids[train])
    np.savez_compressed(out/'dataset.npz',X=z,y=y,split=split,flow_id=ids,class_names=np.array(classes))
    for s in groups:
        mask = (split==s)&(y>=0)
        np.savez_compressed(out/(s+'.npz'),X=z[mask],y=y[mask],flow_id=ids[mask],class_names=np.array(classes))
    with (out/'vectors.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['flow_id','split','label','label_status']+[f'z{i:03d}' for i in range(dimension)])
        for r,v in zip(rows,z):w.writerow([r['flow_id'],r['split'],r['label'],r['label_status'],*v])
    # Exact collisions are an audit, never a test-label-dependent removal rule.
    collision_labels=defaultdict(set)
    for r,v in zip(rows,x):
        if r['label'] is not None:collision_labels[hashlib.sha256(v.tobytes()).hexdigest()].add(r['label'])
    result = dict(rows=len(rows),embedding_dimensions=x.shape[1],pca_dimensions=dimension,pca_requested=config['pca_components'],
                  pca_fit_rows=int(train.sum()),pca_fit_scope='all train features, including unresolved unlabeled train flows; no validation/test',
                  train_pool_is_not_fewshot_context=True,explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
                  split_counts={s:int((split==s).sum()) for s in groups},
                  label_counts={s:dict(Counter(r['label'] or '__UNRESOLVED__' for r in rows if r['split']==s)) for s in groups},
                  exact_embedding_conflict_groups=sum(len(v)>1 for v in collision_labels.values()),
                  classes=classes,group_overlap=0,classification_evaluated=False)
    save_json(out/'projection.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['all','prepare','embed','project'],default='all')
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args();config_path=resolve_path(args.config);config=json.loads(config_path.read_text())
    out=resolve_path(args.output);out.mkdir(parents=True,exist_ok=True)
    snapshot=out/'pipeline_config.json'
    if snapshot.exists() and json.loads(snapshot.read_text())!=config:raise ValueError('Output directory belongs to a different configuration')
    save_json(snapshot,config)
    start=time.time()
    try:
        for stage,fn,sentinel in [('prepare',lambda:prepare(config,out),'prepare.json'),
                                  ('embed',lambda:embed(config,out,args.device,args.batch_size),'encoder.json'),
                                  ('project',lambda:project(config,out),'projection.json')]:
            if args.stage not in ('all',stage):continue
            if (out/sentinel).exists():
                print(f'Reusing completed {stage} stage',flush=True);continue
            if stage=='prepare' and (out/'flows.jsonl').exists():raise RuntimeError('Incomplete prepare output; retain it and use a new output directory')
            print(f'Starting {stage}',flush=True);t=time.time();fn()
            save_json(out/(stage+'_timing.json'),dict(seconds=time.time()-t))
        save_json(out/'pipeline_status.json',dict(status='complete',requested_stage=args.stage,seconds=time.time()-start,
                                                classification_evaluated=False))
    except Exception as e:
        save_json(out/'pipeline_status.json',dict(status='failed',requested_stage=args.stage,error=repr(e),seconds=time.time()-start))
        raise
    print(json.dumps(dict(output=str(out),status='complete',stage=args.stage)),flush=True)

if __name__=='__main__':main()

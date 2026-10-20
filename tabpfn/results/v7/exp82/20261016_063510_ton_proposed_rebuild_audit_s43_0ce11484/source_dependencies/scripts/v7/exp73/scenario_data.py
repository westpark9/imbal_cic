"""Causal, persistent role assignments within 100k training rows plus separate validation."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import json
import numpy as np
from exp61_utils import sha,write
from replay_protocol import partition_test,STAGES
from exp69_class_arrival import seen_classes

ROLES=('global','expert','route','tune','cal')
FRACTIONS=np.array([.5,.25,.25])
SOURCE=ROOT/'tabpfn/results/v7/exp71/20261014_103108_cumulative_100k_uniform_natural_s43_43a766d9'


def assign_new_roles(ids,y,hashes,known,seed,roles=ROLES[:3],fractions=FRACTIONS):
    """Only newly revealed rows; an existing feature group never changes role."""
    assigned={role:[] for role in roles}
    for c in sorted(np.unique(y[ids])):
        members=ids[y[ids]==c]
        unique,inv,counts=np.unique(hashes[members],return_inverse=True,return_counts=True)
        ordered=members[np.argsort(inv,kind="stable")]
        groups=np.split(ordered,np.cumsum(counts)[:-1])
        order=np.random.default_rng(seed+1009*int(c)).permutation(len(unique))
        filled=np.zeros(len(roles),int)
        for i in order:
            key=int(unique[i])
            if key in known:filled[known[key]]+=len(groups[i])
        remaining=[int(i) for i in order if int(unique[i]) not in known]
        # Reserve each role when enough distinct groups are currently available.
        # A large duplicate group is not used for a one-row minimum reservation.
        small=sorted(remaining,key=lambda i:len(groups[i]))
        for role in np.flatnonzero(filled==0):
            if not small:break
            i=small.pop(0);known[int(unique[i])]=int(role);filled[role]+=len(groups[i]);remaining.remove(i)
        target=np.asarray(fractions)*len(members)
        for i in remaining:
            role=int(np.argmax(target-filled));known[int(unique[i])]=role;filled[role]+=len(groups[i])
        for i,group in enumerate(groups):assigned[roles[known[int(unique[i])]]].extend(group.tolist())
    return {role:np.array(sorted(values),dtype=np.int64) for role,values in assigned.items()}


def prepare(out,config):
    if (out/'PREPARED.json').exists():return
    source=Path(config['source'])
    for ds in config['datasets']:
        base=source/'data'/ds;dest=out/'data'/ds;dest.mkdir(parents=True,exist_ok=True)
        meta=json.loads((base/'manifest.json').read_text())
        for name,digest in meta['files'].items():
            if sha(base/name)!=digest:raise ValueError('Source changed: '+name)
        if sha(base/'plans.json')!=meta['plans_sha256']:raise ValueError('Source plans changed')
        names=meta['class_names'];y=np.load(base/'train_y.npy');hashes=np.load(base/'train_hash.npy')
        val=ROOT/f'tabpfn/results/v4/exp62/20260930_exp62_sv_current_bank_s43/{ds}/cache'
        vm=json.loads((val/'COMPLETE.json').read_text());vy0=np.load(val/'cal_y.npy');vh=np.load(val/'cal_hash.npy')
        vy=np.array([names.index(vm['class_names'][c]) for c in vy0],dtype='int16')
        old=json.loads((base/'plans.json').read_text());new={}
        for case in config['cases']:
            known={};roles={role:[] for role in ROLES[:3]};prior=set();stages=[];vroles={role:[] for role in ROLES[3:]};seen_val=set()
            cap=int(np.ceil(config['validation_cap']/len(names)))
            for stage,plan in enumerate(old[case]):
                fit=np.array(plan['fit'],dtype=np.int64)
                incoming=np.array([i for i in fit if int(i) not in prior],dtype=np.int64)
                split=assign_new_roles(incoming,y,hashes,known,config['seed']+stage*104729)
                for role,values in split.items():roles[role].extend(values.tolist())
                saved={role:sorted(values) for role,values in roles.items()}
                allowed=[names.index(n) for n in seen_classes(ds,stage)]
                forbidden=np.unique(hashes[fit])
                for c in allowed:
                    if c in seen_val:continue
                    eligible=np.flatnonzero((vy==c)&~np.isin(vh,forbidden))
                    chosen=np.random.default_rng(config['seed']+9001+c*1009).permutation(eligible)[:cap]
                    vs=assign_new_roles(chosen,vy,vh,{},config['seed']+9101+c*1009,roles=ROLES[3:],fractions=(.4,.6))
                    for role,values in vs.items():vroles[role].extend(values.tolist())
                    seen_val.add(c)
                # Previously selected validation stays fixed. When later
                # training reveals a duplicate feature, exclude it from current
                # validation instead of replenishing with future labels.
                for role,values in vroles.items():
                    ids=np.array(sorted(values),dtype=np.int64)
                    saved[role]=ids[~np.isin(vh[ids],forbidden)].tolist()
                saved['fit']=fit.tolist();saved['validation']=sorted(saved['tune']+saved['cal'])
                saved['released']=fit.tolist();saved['stage']=stage
                assert sorted(saved['global']+saved['expert']+saved['route'])==sorted(saved['fit'])
                for role in ROLES:
                    yy=vy if role in ROLES[3:] else y
                    assert set(np.unique(yy[saved[role]]))==set(allowed),(ds,case,stage,role)
                for i,a in enumerate(ROLES):
                    for b in ROLES[i+1:]:
                        ha=vh if a in ROLES[3:] else hashes;hb=vh if b in ROLES[3:] else hashes
                        assert not np.intersect1d(ha[saved[a]],hb[saved[b]]).size
                stages.append(saved);prior=set(fit.tolist())
            new[case]=stages
        write(dest/'plans.json',new)
        np.save(dest/'validation_y.npy',vy)
        for key in ('X','ids','hash','time','scenario'):
            target=dest/('validation_'+key+'.npy')
            if not target.exists():target.symlink_to(val/('cal_'+key+'.npy'))
        order,batches,bounds=partition_test(np.load(base/'test_y.npy'),names,ds,config['seed'],config['batch'])
        for name,value in [('order',order),('batches',batches),('stage_bounds',bounds)]:np.save(dest/(name+'.npy'),value)
        write(dest/'manifest.json',dict(source=str(base),source_manifest_sha256=sha(base/'manifest.json'),class_names=names,rows=len(order),
            training_roles=ROLES[:3],role_fractions=FRACTIONS.tolist(),roles_feature_disjoint=True,roles_persistent=True,future_labels_used=False,
            validation_source=str(val),validation_original_split=vm['clean_manifest'],
            validation_files={key:sha(dest/('validation_'+key+'.npy')) for key in ('X','y','ids','hash','time','scenario')},
            plans_sha256=sha(dest/'plans.json'),order_sha256=sha(dest/'order.npy')))
    write(out/'PREPARED.json',dict(state='complete',source=str(source),training_budget=100000,validation='separate existing clean validation'))


def stage_data(out,config,ds,case,stage):
    base=Path(config['source'])/'data'/ds;prep=out/'data'/ds
    plan=json.loads((prep/'plans.json').read_text())[case][stage]
    names=json.loads((base/'manifest.json').read_text())['class_names'];seen=seen_classes(ds,stage)
    X=np.load(base/'train_X.npy',mmap_mode='r');y=np.load(base/'train_y.npy',mmap_mode='r')
    VX=np.load(prep/'validation_X.npy',mmap_mode='r');VY=np.load(prep/'validation_y.npy',mmap_mode='r')
    lookup=np.full(len(names),-1,dtype=np.int16)
    for i,n in enumerate(seen):lookup[names.index(n)]=i
    data={'names':seen,'synthetic':config.get('smoke',False)}
    for role in (*ROLES,'fit','validation'):
        ids=np.array(plan[role],dtype=np.int64);external=role in ('tune','cal','validation')
        data['X_'+role]=np.asarray((VX if external else X)[ids]);data['y_'+role]=lookup[(VY if external else y)[ids]];data['i_'+role]=ids
        assert (data['y_'+role]>=0).all()
    data.update(Xtr=data['X_fit'],ytr=data['y_fit'],Xva=data['X_validation'],yva=data['y_validation'])
    data.update(Xte=data['Xtr'][:config['batch']],yte=data['ytr'][:config['batch']])
    train_ids=np.load(base/'train_ids.npy',mmap_mode='r');val_ids=np.load(prep/'validation_ids.npy',mmap_mode='r')
    identity=dict(global_class_indices=[names.index(n) for n in seen],released_ids=train_ids[plan['released']],
                  train_ids=train_ids[plan['fit']],validation_ids=val_ids[plan['validation']],
                  role_ids={role:(val_ids if role in ROLES[3:] else train_ids)[plan[role]] for role in ROLES})
    assert not np.intersect1d(identity['train_ids'],identity['validation_ids']).size
    return data,identity

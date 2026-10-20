"""Current anchor/residual K=4 bank and S1V0, adapted to causal stage pools.

Policy functions and features are reused unchanged from EXP62/EXP51. Test-time
expert inference is actually sparse; no test labels are accepted by predict.
"""
from scenario_data import ROOT
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nfv3_v3_exp38_frozen_cache import ResidualSignature,AffinityRef,sq_dist_to_centroids,build_pair_pre,build_pair_post
from nfv3_v3_exp51_scorer_target import fit_model,decision_gain,select_thresholds,population_weights,apply_thresholds
from replay_models import make_base,timed
from exp61_utils import write


def temperature(p,t):
    z=np.log(np.clip(p,1e-12,1))/t;z-=z.max(1,keepdims=True);z=np.exp(z);return z/z.sum(1,keepdims=True)


def infer(model,X,batch=4096,embed=False):
    parts=[]
    for start in range(0,len(X),batch):
        value=np.asarray(model.get_embeddings(X[start:start+batch],'test') if embed else model.predict_proba(X[start:start+batch]))
        if embed and value.ndim==3:value=value[0]
        parts.append(value.astype('float32'))
    return np.concatenate(parts)


def features(state,p0,z):
    dist=np.sqrt(sq_dist_to_centroids(state['sig'].observable(z,temperature(p0,state['temperature'])),state['centroids'][:,:state['sig'].obs_dim]))
    aff=state['affinity'][0].score(z)
    return dist,aff


def candidates(state,p0,z,ps):
    dist,a0=features(state,p0,z)
    pre=np.column_stack([state['scorer'].predict(build_pair_pre(p0,z,dist[:,k],a0,state['qk'][k])) for k in range(state['K'])])
    winner=pre.argmax(1);best=pre.max(1);post=np.zeros(len(p0),np.float32);candidate=np.zeros(len(p0),np.int16)
    for k in range(state['K']):
        rows=np.flatnonzero(winner==k)
        if not len(rows):continue
        pk=ps[k][rows]
        post[rows]=state['verifier'].predict(build_pair_post(p0[rows],pk,a0[rows],state['affinity'][k+1].score(z[rows]),dist[rows,k],state['qk'][k]))
        candidate[rows]=pk.argmax(1)
    return best,post,candidate,winner


def fit_current(data,config,ds,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    C=len(data['names']);K=config['experts'];phases=[];models=[]
    def phase(name,fn):
        print('OURS',ds,name,flush=True)
        write(out/'current_progress.json',dict(phase=name,updated_epoch=time.time()))
        value,seconds=timed(fn);phases.append(dict(phase=name,seconds=seconds));return value
    global_model=phase('global_construct',lambda:make_base('tabpfn',config,C));models.append(global_model)
    phase('global_fit',lambda:global_model.fit(data['X_global'],data['y_global']))
    emb=phase('global_embedding',lambda:infer(global_model,data['X_global'],config['prepare_batch'],True))
    pca=phase('embedding_pca',lambda:PCA(n_components=min(16,emb.shape[1],len(emb)),random_state=config['seed']+1250).fit(emb))
    zglobal=pca.transform(emb).astype('float32');del emb
    counts=np.bincount(data['ytr'],minlength=C);weights=len(data['ytr'])/(C*counts)
    pt=phase('temperature_predictions',lambda:infer(global_model,data['X_tune'],config['prepare_batch']))
    temperatures=(.8,1.,1.25)
    losses=[float((-np.log(np.clip(temperature(pt,t)[np.arange(len(pt)),data['y_tune']],1e-12,1))*weights[data['y_tune']]).mean()) for t in temperatures]
    temp=temperatures[int(np.argmin(losses))]
    pe=phase('residual_predictions',lambda:infer(global_model,data['X_expert'],config['prepare_batch']))
    ze=phase('residual_embedding',lambda:pca.transform(infer(global_model,data['X_expert'],config['prepare_batch'],True)).astype('float32'))
    corrected=temperature(pe,temp);res=-np.log(np.clip(corrected[np.arange(len(pe)),data['y_expert']],1e-12,1))*weights[data['y_expert']]
    res=np.minimum(res,np.quantile(res,.995)).astype('float32');sig=ResidualSignature(1.,1.,1.)
    full=sig.fit_full(ze,corrected,data['y_expert'],res)
    km=phase('residual_clustering',lambda:KMeans(n_clusters=K,random_state=config['seed']+1300,n_init=5).fit(full,sample_weight=res))
    dist=sq_dist_to_centroids(full,km.cluster_centers_);assignment=dist.argmin(1)
    rng=np.random.default_rng(config['seed']+1100)
    anchor=np.concatenate([rng.permutation(np.flatnonzero(data['y_global']==c))[:2000] for c in range(C)])
    refs_rng=np.random.default_rng(config['seed']+1600)
    refs=[zglobal[refs_rng.permutation(len(zglobal))[:4096]]]
    ps={};zs={};ep={};qk=[];contexts=[]
    for role in ('route','cal'):
        ps[role]=phase('global_'+role,lambda role=role:infer(global_model,data['X_'+role],config['prepare_batch']))
        zs[role]=phase('embedding_'+role,lambda role=role:pca.transform(infer(global_model,data['X_'+role],config['prepare_batch'],True)).astype('float32'))
        ep[role]=[]
    for k in range(K):
        ids=np.flatnonzero(assignment==k)
        if not len(ids):raise ValueError('Empty residual cluster')
        # At a 100k training budget plus separate validation, every cluster fits under the original
        # 186k specialty-block cap. Original diversity_select returns all rows.
        if len(ids)>186000:raise ValueError('Original block cap exceeded: diversity selection required')
        X=np.concatenate([data['X_global'][anchor],data['X_expert'][ids]])
        y=np.concatenate([data['y_global'][anchor],data['y_expert'][ids]])
        model=phase(f'expert{k+1}_construct',lambda:make_base('tabpfn',config,C));models.append(model)
        phase(f'expert{k+1}_fit',lambda:model.fit(X,y))
        for role in ('route','cal'):
            ep[role].append(phase(f'expert{k+1}_{role}',lambda role=role:infer(model,data['X_'+role],config['prepare_batch'])))
        prior=(np.bincount(y,minlength=C)+1)/(len(y)+C)
        qk.append([np.log1p(len(ids)),float(res[ids].sum()/max(res.sum(),1e-12)),float(res[ids].mean()),
                   float(-(prior*np.log(prior)).sum()),float(np.log1p(dist[ids,k].mean())),1.])
        refs.append(ze[ids[refs_rng.permutation(len(ids))[:4096]]])
        contexts.append(dict(expert=k+1,anchor_indices=data['i_global'][anchor].tolist(),block_indices=data['i_expert'][ids].tolist(),
                             anchor_counts=np.bincount(data['y_global'][anchor],minlength=C).tolist(),block_counts=np.bincount(data['y_expert'][ids],minlength=C).tolist()))
    state=dict(K=K,pca=pca,sig=sig,temperature=temp,centroids=km.cluster_centers_.astype('float32'),qk=np.array(qk,dtype='float32'),names=data['names'])
    state['affinity']=phase('affinity_indices',lambda:[AffinityRef(z,16) for z in refs])
    args=argparse.Namespace(trees=3 if config.get('smoke') else 300,depth=6,learning_rate=.05,threads=config['threads'],seed=config['seed'])
    y=data['y_route'];p0=ps['route'];z=zs['route'];g=p0.argmax(1);rr=np.arange(len(y))
    def train_policy():
        d,a=features(state,p0,z)
        route_counts=np.bincount(y,minlength=C);balance=len(y)/(C*route_counts)
        X=np.concatenate([build_pair_pre(p0,z,d[:,k],a,state['qk'][k]) for k in range(K)])
        target=np.concatenate([decision_gain(y,g,ep['route'][k].argmax(1)) for k in range(K)])
        state['scorer']=fit_model(args,X,target,'mean',config['seed']+1700,np.tile(balance[y],K))
        X=np.concatenate([build_pair_post(p0,ep['route'][k],a,state['affinity'][k+1].score(z),d[:,k],state['qk'][k]) for k in range(K)])
        target=np.concatenate([np.log(np.clip(ep['route'][k][rr,y],1e-12,1))-np.log(np.clip(p0[rr,y],1e-12,1)) for k in range(K)])
        state['verifier']=fit_model(args,X,target,'quantile',config['seed']+1800)
    phase('S1V0_fit',train_policy)
    def calibrate():
        y=data['y_cal'];p0=ps['cal'];pre,post,candidate,winner=candidates(state,p0,zs['cal'],ep['cal'])
        # D_cal is already disjoint by feature group from every training/tuning
        # role. Use all of it for threshold selection; test is the assessment.
        tail_names=['bot','infiltration','web_attacks'] if ds=='cic2018' else ['mitm','ransomware']
        tail=[i for i,n in enumerate(data['names']) if n in tail_names]
        # Early stages have no originally designated tail yet. This neutral
        # constraint avoids an empty mean while retaining macro-F1 selection.
        tail_for_selection=tail or list(range(C))
        protected=[i for i,n in enumerate(data['names']) if n in (['brute_force','ddos','dos'] if ds=='cic2018' else ['ddos','dos'])]
        prior=counts/counts.sum();pop=population_weights(y,prior,C)
        best,grid=select_thresholds(y,p0.argmax(1),candidate,pre,post,pop,C,tail_for_selection,data['names'].index('benign'),protected,
            np.array([0.,.3,.5,.65,.75,.8,.85,.9,.95,.98,.99]),np.array([0.,.1,.25,.5,.75,.9,.95,.99]),.0005,0.,1.,30)
        state['thresholds']=best
        grid.to_csv(out/'threshold_grid.csv',index=False)
        final,calls,accepted=apply_thresholds(p0.argmax(1),candidate,pre,post,best)
        write(out/'selection.json',dict(thresholds=best,calibration_rows=len(y),class_counts=np.bincount(y,minlength=C).tolist(),
            calls=int(calls.sum()),accepted=int(accepted.sum()),helpful=int(((final==y)&(p0.argmax(1)!=y)).sum()),harmful=int(((final!=y)&(p0.argmax(1)==y)).sum()),
            tail_classes_present=[data['names'][i] for i in tail],test_access=False,all_calibration_used=True))
    phase('threshold_selection',calibrate)
    # Serialization is excluded from readiness timing, as for the baselines.
    for i,model in enumerate(models):model.save_fit_state(out/f'model_{i}.tabpfn_fit')
    joblib.dump(state,out/'state.joblib');write(out/'contexts.json',contexts)
    write(out/'current_phases.json',phases)
    models.clear();global_model=None;model=None;gc.collect()
    import torch
    torch.cuda.empty_cache()
    return dict(update_seconds=sum(p['seconds'] for p in phases),phases=phases,train_rows=len(data['ytr']),
                validation_rows=len(data['yva']),global_rows=len(data['y_global']),experts=K,
                context_recipe='common anchor + residual cluster block (all cluster rows below original 186000 cap)',
                policy='EXP62 S1V0; original features, targets and threshold constraints',class_names=data['names'],
                protocol_adaptations=['100k training budget plus separate validation; causal feature-group-disjoint role allocation',
                    'all disjoint D_cal selects thresholds; no chronological cal-confirm split',
                    'tail constraint neutral until original tail classes appear'],selection=state['thresholds'])


class CurrentPredictor:
    def __init__(self,out,config):
        from tabpfn import TabPFNClassifier
        self.out=Path(out);self.config=config;self.state=joblib.load(self.out/'state.joblib')
        self.models=[TabPFNClassifier.load_from_fit_state(self.out/'model_0.tabpfn_fit',device='cuda:0')]
        if self.state['thresholds']['tau_pre'] is not None:
            self.models += [TabPFNClassifier.load_from_fit_state(self.out/f'model_{k+1}.tabpfn_fit',device='cuda:0') for k in range(self.state['K'])]

    def predict(self,X):
        state=self.state;p0=np.asarray(self.models[0].predict_proba(X));global_pred=p0.argmax(1);final=p0.copy()
        self.aux=dict(global_pred=global_pred.astype('int16'),calls=np.zeros(len(X),bool),accepted=np.zeros(len(X),bool),expert=np.full(len(X),-1,dtype='int16'))
        if state['thresholds']['tau_pre'] is None:return final
        emb=np.asarray(self.models[0].get_embeddings(X,'test'))
        if emb.ndim==3:emb=emb[0]
        z=state['pca'].transform(emb).astype('float32');d,a=features(state,p0,z)
        pre=np.column_stack([state['scorer'].predict(build_pair_pre(p0,z,d[:,k],a,state['qk'][k])) for k in range(state['K'])])
        winner=pre.argmax(1);calls=pre.max(1)>float(state['thresholds']['tau_pre']);self.aux['calls']=calls
        self.aux['expert'][calls]=winner[calls]
        for k in range(state['K']):
            rows=np.flatnonzero(calls&(winner==k))
            if not len(rows):continue
            pk=np.asarray(self.models[k+1].predict_proba(X[rows]))
            post=state['verifier'].predict(build_pair_post(p0[rows],pk,a[rows],state['affinity'][k+1].score(z[rows]),d[rows,k],state['qk'][k]))
            chosen=post>state['thresholds']['tau_post'];self.aux['accepted'][rows[chosen]]=True
            final[rows[chosen]]=pk[chosen]
        return final

    def close(self):
        self.models.clear();gc.collect()
        import torch
        torch.cuda.empty_cache()

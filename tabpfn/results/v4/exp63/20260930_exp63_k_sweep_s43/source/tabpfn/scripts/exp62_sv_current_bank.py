#!/usr/bin/env python3
"""Fit S1V0 with frozen EXP59 contexts and consistently refitted experts."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

import numpy as np
import pandas as pd
import joblib

CODE_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE_ROOT/'scripts'))
from exp61_utils import Cost,sha,write


def feature_hash(X):
    return pd.util.hash_pandas_object(pd.DataFrame(X),index=False).to_numpy()


class Worker:
    def __init__(self,args):
        self.args=args;self.ds=args.dataset;self.project=args.project.resolve()
        self.out=args.root.resolve()/self.ds;self.out.mkdir(parents=True,exist_ok=True)
        self.cache=self.out/'cache';self.cache.mkdir(exist_ok=True)
        self.base=self.project/f'tabpfn/results/20260928_exp59_residual_oracle_s43/{self.ds}_s43'
        self.source=self.base/'fresh_bank';self.config=json.loads((self.source/'job.json').read_text())['args']
        self.design=json.loads((self.source/'design.json').read_text());self.names=self.design['class_names'];self.C=len(self.names)
        self.cost=Cost(self.out);self.started=time.time()

    def progress(self,phase,**kw):
        value=dict(dataset=self.ds,pid=os.getpid(),phase=phase,elapsed_seconds=time.time()-self.started,updated_epoch=time.time(),**kw)
        write(self.out/'progress.json',value);print('EXP62 '+json.dumps(value),flush=True)

    def cached(self,name,producer):
        from nfv3_v3_exp38_frozen_cache import save_array
        path=self.cache/(name+'.npy')
        if not path.exists():
            self.progress('cache_'+name)
            save_array(path,producer())
        return np.load(path,mmap_mode='r')

    def infer(self,model,X,tag,embed=False):
        hashes=feature_hash(X);_,first,inverse=np.unique(hashes,return_index=True,return_inverse=True)
        outputs=[];start=time.time()
        for i in range(0,len(first),self.args.batch):
            ix=first[i:i+self.args.batch]
            a=np.asarray(model.get_embeddings(X[ix],'test') if embed else model.predict_proba(X[ix]))
            if embed and a.ndim==3:a=a[0]
            outputs.append(a.astype('float32'))
            self.progress(tag,rows_done=min(i+self.args.batch,len(first)),unique_rows=len(first),full_rows=len(X),seconds=time.time()-start)
        return np.concatenate(outputs)[inverse]

    def prepare(self):
        import torch
        from tabpfn import TabPFNClassifier
        from nfv3_v3_exp38_frozen_cache import ResidualSignature,sq_dist_to_centroids,AffinityRef
        if (self.cache/'COMPLETE.json').exists():return
        frozen=self.project/'tabpfn/results/20260922_143529_exp57_expert_quality_s42_44/source/tabpfn/scripts'
        sys.path.insert(0,str(frozen))
        name='exp62_frozen_helpers_'+self.ds;h=types.ModuleType(name);h.__file__=str(frozen/'nfv3_v3_exp31_c0alloc.py');sys.modules[name]=h
        exec(compile((self.base/'executed_fresh_prefix.py').read_text(),h.__file__,'exec'),h.__dict__)
        self.h=h
        cfg=self.config;clean=Path(cfg['clean_manifest']).parent
        shared=np.load(self.source/'shared_context_ids.npz');gids=shared['global_context'];anchor=shared['anchor'];expids=shared['expert_pool']
        blocks=[np.load(self.source/f'contexts/designed_e{k}.npy') for k in range(1,5)]
        state=dict(np.load(self.base/'residual_state.npz'))
        np.testing.assert_array_equal(state['train_ids'],expids)
        self.progress('loading_frozen_inputs')
        with self.cost.phase('input_loading'):
            suite=h.core.load_pickle(cfg['data']);X=suite['X'];families=np.asarray(suite['families']);times=np.asarray(suite['timestamps']);scenarios=np.asarray(suite['attack_scenarios'])
            class_index={n:i for i,n in enumerate(self.names)}
            def labels(ids):return np.asarray([class_index[n] for n in families[ids]],dtype='int16')
            def feats(ids):return np.nan_to_num(np.asarray(X[ids],dtype='float32'))
            tr=np.load(clean/'train_idx.npy');va=np.load(clean/'val_idx.npy');te=np.load(clean/'test_idx.npy')
            pools,_,_=h.scenario_stratified_partition(tr,labels(tr),times[tr],scenarios[tr],cfg['context_frac'],cfg['expert_frac'],self.names)
            tune,cal,_=h.scenario_stratified_split2(va,labels(va),times[va],scenarios[va],cfg['tune_frac_of_val'],self.names)
            route=h.core.cap_per_class(pools['route'],labels(pools['route']),self.C,cfg['route_cap_per_class'],43+h.SEED_BAND_ROUTE_CAP)
            cal=h.core.cap_per_class(cal,labels(cal),self.C,cfg['cal_cap_per_class'],43+h.SEED_BAND_CAL_CAP)
            tune=h.core.cap_per_class(tune,labels(tune),self.C,cfg['tune_cap_per_class'],43+h.SEED_BAND_TUNE_CAP)
            reference=np.load(self.source/'evaluation_identity.npz')
            np.testing.assert_array_equal(te,reference['ids']);np.testing.assert_array_equal(labels(te),reference['y'])
            assert not np.intersect1d(route,np.concatenate([gids,anchor,expids])).size
            assert not np.intersect1d(cal,route).size and not np.intersect1d(cal,te).size
            np.testing.assert_array_equal(labels(expids),state['train_labels'])
            contexts=[gids]+[np.concatenate([anchor,b]) for b in blocks]
            self.ctx_X=[feats(ids) for ids in contexts];self.ctx_y=[labels(ids) for ids in contexts]
            self.X_tune=feats(tune)
            train_counts=np.bincount(labels(tr),minlength=self.C)
            for split,ids in [('route',route),('cal',cal),('eval',te)]:
                for key,producer in [('ids',lambda ids=ids:ids),('X',lambda ids=ids:feats(ids)),('y',lambda ids=ids:labels(ids)),('time',lambda ids=ids:times[ids]),('scenario',lambda ids=ids:scenarios[ids].astype(str))]:
                    self.cached(split+'_'+key,producer)
            del X,suite,families,times,scenarios;h.core._PICKLE_CACHE.clear();gc.collect()
        for split in ['route','cal','eval']:self.cached(split+'_hash',lambda split=split:feature_hash(self.array(split,'X')))
        mask=self.cached('cal_mask',lambda:~np.isin(self.array('cal','hash'),np.unique(self.array('route','hash'))))
        assert int(mask.sum())==self.design['calibration_mask_rows'],(int(mask.sum()),self.design['calibration_mask_rows'])
        # Keep raw probability outputs for classification and the fitted residual
        # temperature only for the original observable geometry.
        temp=float(state['global_temperature'])
        self.cached('eval_p0',lambda:np.load(self.source/'probabilities/global_test.npy',mmap_mode='r'))
        self.cached('eval_z',lambda:np.load(self.base/'test_z.npy',mmap_mode='r'))
        self.progress('restore_frozen_global')
        global_model=TabPFNClassifier.load_from_fit_state(self.source/'global_fit_state.tabpfn_fit',device='cuda')
        pca=joblib.load(self.base/'feature_pca.joblib')
        for split in ['route','cal']:
            self.cached(split+'_p0',lambda split=split:self.infer(global_model,self.array(split,'X'),'global_'+split))
            self.cached(split+'_z',lambda split=split:pca.transform(self.infer(global_model,self.array(split,'X'),'embedding_'+split,True)).astype('float32'))
        sig=ResidualSignature(cfg['sig_alpha_p'],cfg['sig_alpha_e'],cfg['sig_alpha_r'])
        for key in ['z','p','e','r']:sig.stats[key]=(state[key+'_mean'],state[key+'_std'])
        def temperature(p):
            v=np.log(np.clip(p,1e-12,None))/temp;v-=v.max(1,keepdims=True);v=np.exp(v);return v/v.sum(1,keepdims=True)
        for split in ['route','cal','eval']:
            self.cached(split+'_distance',lambda split=split:np.sqrt(sq_dist_to_centroids(sig.observable(self.array(split,'z'),temperature(self.array(split,'p0'))),state['full_centroids'][:,:sig.obs_dim])))
        # A cached, unchanged Global representation of D_expert is reusable; no
        # outputs or contexts of the discarded intervention experiment are used.
        rep=self.project/f'tabpfn/results/20260929_exp60_counterexamples_s43/{self.ds}_s43/cache'
        audit=json.loads((rep/'TRAIN_REPRESENTATION_COMPLETE.json').read_text())
        assert audit['global_fit_sha256']==sha(self.source/'global_fit_state.tabpfn_fit') and audit['train_ids_sha256']==sha(self.base/'residual_state.npz')
        obs=np.load(rep/'train_observable.npy',mmap_mode='r');p_exp=np.load(rep/'train_global_corrected.npy',mmap_mode='r')
        z_exp=(obs[:,:cfg['phi_dim']]*np.sqrt(cfg['phi_dim'])*state['z_std']+state['z_mean']).astype('float32')
        residual=np.minimum(h.balanced_ce(p_exp,state['train_labels'],state['w_bal']),float(state['train_residual_clip']))
        full=sig.full(z_exp,p_exp,state['train_labels'],residual);q=[]
        for k,b in enumerate(blocks):
            loc=np.searchsorted(expids,b);np.testing.assert_array_equal(expids[loc],b)
            members=state['train_regions']==k
            spread=float(((full[members]-state['full_centroids'][k])**2).sum(1).mean())
            counts=np.bincount(self.ctx_y[k+1],minlength=self.C).astype(float);prior=(counts+cfg['prior_alpha'])/(counts.sum()+cfg['prior_alpha']*self.C)
            q.append(h.expert_descriptor(residual[loc],float(residual[members].sum()/residual.sum()),len(b),prior,spread,cfg['expert_cost']))
        self.cached('qk',lambda:np.stack(q));del full,residual,p_exp,obs,z_exp;gc.collect()
        rng=np.random.default_rng(43+h.SEED_BAND_AFFINITY)
        for k in range(5):
            # EXP31 affinity references use C0 or the expert block without anchor.
            Xref=self.ctx_X[0] if k==0 else self.ctx_X[k][len(anchor):]
            pos=rng.permutation(len(Xref))[:cfg['affinity_ref_rows']]
            zr=self.cached(f'affinity_ref_{k}',lambda Xref=Xref,pos=pos:pca.transform(self.infer(global_model,Xref[pos],f'affinity_ref_{k}',True)).astype('float32'))
            affinity=AffinityRef(zr,cfg['affinity_nn'])
            for split in ['route','cal','eval']:
                def affinity_values(split=split):
                    _,first,inv=np.unique(self.array(split,'hash'),return_index=True,return_inverse=True)
                    return affinity.score(self.array(split,'z')[first],chunk=65536)[inv]
                self.cached(f'{split}_affinity_{k}',affinity_values)
        del global_model;gc.collect();torch.cuda.empty_cache()
        for k in range(1,5):
            if all((self.cache/f'{s}_p{k}.npy').exists() for s in ['route','cal','eval']):continue
            self.progress('fit_frozen_expert',expert=k,context_rows=len(self.ctx_y[k]))
            fit_path=self.out/f'expert{k}.tabpfn_fit'
            if fit_path.exists():
                model=TabPFNClassifier.load_from_fit_state(fit_path,device='cuda')
            else:
                extra={'inference_precision':torch.float32} if cfg['det_precision']=='float32' else {}
                model=TabPFNClassifier(device=cfg['device'],model_path=cfg['model_path'],random_state=43,n_estimators=cfg['n_estimators'],auto_scale_n_estimators=False,fit_mode=cfg['fit_mode'],keep_cache_on_device=cfg['keep_cache_on_device'],ignore_pretraining_limits=cfg['ignore_pretraining_limits'],**extra)
                with self.cost.phase(f'expert{k}_fit'):model.fit(self.ctx_X[k],self.ctx_y[k])
                model.save_fit_state(fit_path)
            # Original fitted expert states were unavailable and even an original-
            # shape replay differed. Use this one saved model on EVERY split;
            # never mix old test probabilities with newly fitted route/cal ones.
            np.testing.assert_array_equal(model.classes_,np.arange(self.C))
            write(self.out/f'expert{k}_model_identity.json',dict(seed=43,fit_sha256=sha(fit_path),context_ids_sha256=sha(self.source/f'contexts/designed_e{k}.npy'),anchor_ids_sha256=sha(self.source/'shared_context_ids.npz'),original_test_predictions_reused=False))
            for split in ['route','cal','eval']:
                with self.cost.phase(f'expert{k}_{split}'):
                    self.cached(f'{split}_p{k}',lambda split=split:self.infer(model,self.array(split,'X'),f'expert{k}_{split}'))
            del model;gc.collect();torch.cuda.empty_cache()
        identity=dict(schema='exp62_exp59_frozen_contexts_refit_experts',seed=43,K=4,source=str(self.source),context_ids_sha256=sha(self.source/'shared_context_ids.npz'),residual_state_sha256=sha(self.base/'residual_state.npz'),clean_manifest_sha256=sha(cfg['clean_manifest']),probabilities='raw; beta=0,T=1; frozen residual temperature used only for centroid distance',global_test_predictions_reused=True,expert_test_predictions_reused=False,bank_note='identical context IDs; all expert route/cal/test predictions come from newly saved fitted states')
        write(self.cache/'identity.json',identity)
        meta=dict(class_names=self.names,n_experts=4,tail_classes=['bot','infiltration','web_attacks'] if self.ds=='cic2018' else ['mitm','ransomware'],train_counts=train_counts.tolist(),source_config=cfg,test_role='previously observed development holdout',source_run=str(self.source),clean_manifest=cfg['clean_manifest'],full_test_rows=len(te))
        for split in ['route','cal','eval']:
            n=len(self.array(split,'y'))
            for key in ['X','z','distance']+[f'p{k}' for k in range(5)]+[f'affinity_{k}' for k in range(5)]:assert len(self.array(split,key))==n
        write(self.cache/'COMPLETE.json',meta);self.progress('cache_complete')
        del self.ctx_X,self.ctx_y,self.X_tune;gc.collect()

    def array(self,split,key):return np.load(self.cache/f'{split}_{key}.npy',mmap_mode='r')

    def policy(self):
        from nfv3_v3_exp51_scorer_target import FrozenCache,fit_model,decision_gain,policy_scores,chronological_cal_split,population_weights,select_thresholds,apply_thresholds,evaluation_rows
        from nfv3_v3_exp38_frozen_cache import save_array
        out=self.out/'policy';out.mkdir(exist_ok=True)
        cache=FrozenCache(self.cache)
        args=argparse.Namespace(trees=300,depth=6,learning_rate=.05,threads=self.args.threads,seed=43,predict_chunk=100000)
        y=cache.array('route','y');p0=cache.array('route','p0');g=p0.argmax(1);rr=np.arange(len(y));counts=np.bincount(y,minlength=cache.C);assert (counts>0).all();balance=len(y)/(cache.C*counts)
        models=[]
        for kind in ['scorer','verifier']:
            path=out/(kind+'.joblib')
            if path.exists():models.append(joblib.load(path));continue
            self.progress('fit_'+kind,route_rows=len(y),experts=cache.K)
            with self.cost.phase('fit_'+kind):
                if kind=='scorer':
                    X=np.concatenate([cache.pre('route',k,rr) for k in range(cache.K)])
                    target=np.concatenate([decision_gain(y,g,cache.array('route',f'p{k+1}').argmax(1)) for k in range(cache.K)])
                    model=fit_model(args,X,target,'mean',43+1700,np.tile(balance[y],cache.K))
                else:
                    X=np.concatenate([cache.post('route',k,rr,'base') for k in range(cache.K)])
                    target=np.concatenate([np.log(np.clip(cache.array('route',f'p{k+1}')[rr,y],1e-12,1))-np.log(np.clip(p0[rr,y],1e-12,1)) for k in range(cache.K)])
                    model=fit_model(args,X,target,'quantile',43+1800)
                joblib.dump(model,path);models.append(model);del X,target;gc.collect()
        ycal=cache.array('cal','y');select,confirm=chronological_cal_split(ycal,cache.array('cal','scenario'),cache.array('cal','time'),cache.array('cal','hash'),cache.array('cal','mask'),.3)
        save_array(out/'cal_select_positions.npy',select);save_array(out/'cal_confirm_positions.npy',confirm)
        write(out/'selection_counts.json',{s:dict(zip(self.names,np.bincount(ycal[ix],minlength=self.C).tolist())) for s,ix in [('select',select),('confirm',confirm)]})
        assert all((ycal[select]==c).any() for c in range(cache.C))
        prior=np.asarray(cache.meta['train_counts'],float);prior/=prior.sum();weights=population_weights(ycal[select],prior,cache.C)
        tail=[self.names.index(n) for n in cache.meta['tail_classes']];ben=self.names.index('benign');protected=[self.names.index(n) for n in (['brute_force','ddos','dos'] if self.ds=='cic2018' else ['ddos','dos'])]
        self.progress('validation_policy_selection')
        with self.cost.phase('policy_calibration'):
            score=policy_scores(cache,'cal','s1v0',models,args,out);gcal=cache.array('cal','p0').argmax(1)
            best,grid=select_thresholds(ycal[select],gcal[select],score['candidate'][select],score['pre'][select],score['post'][select],weights,cache.C,tail,ben,protected,np.array([0.,.3,.5,.65,.75,.8,.85,.9,.95,.98,.99]),np.array([0.,.1,.25,.5,.75,.9,.95,.99]),.0005,0.,1.,30)
            grid.to_csv(out/'threshold_grid.csv',index=False);write(out/'selected_thresholds.json',best)
            final,calls,accepted=apply_thresholds(gcal,score['candidate'],score['pre'],score['post'],best)
            pc,row=evaluation_rows(cache,'cal_confirm','s1v0',ycal,gcal,final,calls,accepted,tail,ben,confirm)
        summaries=[row];classes=pc
        # Candidate and thresholds are frozen before opening test labels here.
        self.progress('evaluate_frozen_policy')
        with self.cost.phase('test_policy_scoring'):
            score=policy_scores(cache,'eval','s1v0',models,args,out);gtest=cache.array('eval','p0').argmax(1)
            final,calls,accepted=apply_thresholds(gtest,score['candidate'],score['pre'],score['post'],best)
            save_array(out/'s1v0_final.npy',final)
        ytest=cache.array('eval','y');off=np.zeros(len(ytest),bool)
        # Re-evaluate the simple fixed route on this same fitted bank, so its
        # comparison with S/V does not depend on exact reproduction of EXP59.
        fixed_regions=self.array('eval','distance').argmin(1)
        fixed=gtest.copy()
        for k in range(cache.K):
            mask=fixed_regions==k;fixed[mask]=cache.array('eval',f'p{k+1}')[mask].argmax(1)
        save_array(out/'fixed_region_final.npy',fixed)
        for name,pred,cl,ac in [('global',gtest,off,off),('fixed_region',fixed,~off,~off),('s1v0',final,calls,accepted)]:
            pc,row=evaluation_rows(cache,'full_test',name,ytest,gtest,pred,cl,ac,tail,ben);summaries.append(row);classes+=pc
        pd.DataFrame(summaries).to_csv(self.out/'summary.csv',index=False);pd.DataFrame(classes).to_csv(self.out/'class_metrics.csv',index=False)
        write(out/'protocol.json',dict(seed=43,K=cache.K,arm='s1v0',scorer='class-balanced direct decision-gain regression',verifier='NLL-gain quantile regression, alpha=.25',thresholds=best,selected_on_test=False,source_cache=str(self.cache),global_test_inference_reused=True,expert_test_inference_reused=False,contexts_unchanged=cache.K==4,expert_fit_states_saved=True,cost_note='Offline cache preparation and policy scoring; not measured online sparse inference latency',protected_classes=[self.names[i] for i in protected],benign_fpr_increase=.0005,min_decided=30))
        print(pd.DataFrame(summaries).to_string(index=False),flush=True)

    def run(self):
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(self.args.threads);torch.set_num_interop_threads(4);torch.manual_seed(43);np.random.seed(43)
        torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:
            with threadpool_limits(self.args.threads):self.prepare();self.policy()
            write(self.out/'COMPLETE.json',dict(seed=43,K=4,dataset=self.ds,seconds=time.time()-self.started));self.progress('complete')
        except BaseException:
            write(self.out/'ERROR.json',dict(traceback=traceback.format_exc(),failed_epoch=time.time()));raise
        finally:self.cost.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',choices=['cic2018','toniot'],required=True);p.add_argument('--batch',type=int,default=65536);p.add_argument('--threads',type=int,default=6)
    Worker(p.parse_args()).run()

#!/usr/bin/env python3
"""K=2/4/6/8, frozen Global/anchor/splits; original residual context recipe.

Reuses EXP62 S1V0 unchanged. Expert capability is measured directly on the
same full evaluation population, separately from label-assisted routing.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import types

import joblib
import numpy as np
import pandas as pd

from exp62_sv_current_bank import Worker, Cost, sha, write


def link_file(source, target):
    source=Path(source).resolve();target=Path(target)
    if not source.is_file():raise FileNotFoundError(source)
    if target.is_symlink():
        assert target.resolve()==source,(target,source)
    elif target.exists():
        raise FileExistsError(target)
    else:target.symlink_to(source)


def class_records(y, pred, global_pred, names, **tags):
    C=len(names)
    cm=np.bincount(y.astype(np.int64)*C+pred,minlength=C*C).reshape(C,C)
    tp=np.diag(cm);support=cm.sum(1);fp=cm.sum(0)-tp;fn=support-tp
    precision=np.divide(tp,tp+fp,out=np.zeros(C),where=tp+fp>0)
    recall=np.divide(tp,support,out=np.zeros(C),where=support>0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(C),where=2*tp+fp+fn>0)
    helpful=(pred==y)&(global_pred!=y);harmful=(pred!=y)&(global_pred==y)
    rows=[dict(**tags,**{'class':name},support=int(support[c]),TP=int(tp[c]),FP=int(fp[c]),FN=int(fn[c]),precision=float(precision[c]),recall=float(recall[c]),f1=float(f1[c]),helpful=int(helpful[y==c].sum()),harmful=int(harmful[y==c].sum())) for c,name in enumerate(names)]
    summary=dict(**tags,rows=len(y),macro_f1=float(f1.mean()),accuracy=float(tp.sum()/len(y)),helpful=int(helpful.sum()),harmful=int(harmful.sum()),predicted_class_count=int((cm.sum(0)>0).sum()),dominant_prediction_share=float(cm.sum(0).max()/len(y)))
    return rows,summary,cm


class SweepWorker(Worker):
    def __init__(self,args):
        self.args=args;self.ds=args.dataset;self.K=args.k;self.project=args.project.resolve()
        self.out=args.root.resolve()/f'{self.ds}_k{self.K}';self.out.mkdir(parents=True,exist_ok=True)
        self.cache=self.out/'cache';self.cache.mkdir(exist_ok=True)
        self.base=self.project/f'tabpfn/results/20260928_exp59_residual_oracle_s43/{self.ds}_s43'
        self.source=self.base/'fresh_bank'
        self.ref=self.project/f'tabpfn/results/20260930_exp62_sv_current_bank_s43/{self.ds}'
        assert (self.ref/'COMPLETE.json').is_file()
        self.config=json.loads((self.source/'job.json').read_text())['args']
        self.design=json.loads((self.source/'design.json').read_text())
        self.names=self.design['class_names'];self.C=len(self.names)
        self.started=time.time();self.cost=Cost(self.out)

    def progress(self,phase,**kw):
        value=dict(dataset=self.ds,K=self.K,pid=os.getpid(),phase=phase,elapsed_seconds=time.time()-self.started,updated_epoch=time.time(),**kw)
        write(self.out/'progress.json',value);print('EXP63 '+json.dumps(value),flush=True)

    def load_helpers(self):
        # Execute only definitions from the exact original bank builder.
        # Its run_exp29 is never called: Global and all splits remain frozen.
        frozen=self.project/'tabpfn/results/20260922_143529_exp57_expert_quality_s42_44/source/tabpfn/scripts'
        name=f'exp63_frozen_{self.ds}_{self.K}';h=types.ModuleType(name)
        h.__file__=str(frozen/'nfv3_v3_exp31_c0alloc.py');sys.modules[name]=h
        prefix=self.out/'original_bank_helpers.py'
        if not prefix.exists():shutil.copy2(self.base/'executed_fresh_prefix.py',prefix)
        exec(compile(prefix.read_text(),h.__file__,'exec'),h.__dict__)
        return h

    def signature(self,state):
        from nfv3_v3_exp38_frozen_cache import ResidualSignature
        cfg=self.config;sig=ResidualSignature(cfg['sig_alpha_p'],cfg['sig_alpha_e'],cfg['sig_alpha_r'])
        for key in ['z','p','e','r']:sig.stats[key]=(state[key+'_mean'],state[key+'_std'])
        return sig

    def temperature(self,p):
        state=np.load(self.base/'residual_state.npz');temp=float(state['global_temperature'])
        v=np.log(np.clip(p,1e-12,None))/temp;v-=v.max(1,keepdims=True);v=np.exp(v)
        return v/v.sum(1,keepdims=True)

    def build_contexts(self,h):
        from sklearn.cluster import KMeans
        from nfv3_v3_exp38_frozen_cache import save_array,sq_dist_to_centroids
        ctx=self.out/'contexts';ctx.mkdir(exist_ok=True)
        if (ctx/'COMPLETE.json').exists():return
        cfg=self.config;state=dict(np.load(self.base/'residual_state.npz'));sig=self.signature(state)
        shared=np.load(self.source/'shared_context_ids.npz');ids=shared['expert_pool']
        np.testing.assert_array_equal(ids,state['train_ids'])
        rep=self.project/f'tabpfn/results/20260929_exp60_counterexamples_s43/{self.ds}_s43/cache'
        audit=json.loads((rep/'TRAIN_REPRESENTATION_COMPLETE.json').read_text())
        assert audit['global_fit_sha256']==sha(self.source/'global_fit_state.tabpfn_fit')
        assert audit['train_ids_sha256']==sha(self.base/'residual_state.npz')
        # Only the unchanged Global representations are reused from this cache.
        # No intervention contexts or expert predictions are used.
        obs=np.load(rep/'train_observable.npy',mmap_mode='r')
        p=np.load(rep/'train_global_corrected.npy',mmap_mode='r')
        y=state['train_labels'];r=np.minimum(h.balanced_ce(p,y,state['w_bal']),float(state['train_residual_clip']))
        err=-np.array(p);err[np.arange(len(y)),y]+=1
        full=np.concatenate([obs,sig._std('e',err),sig._std('r',np.log1p(r).astype('float32')[:,None])],axis=1)
        assert full.shape==(len(ids),len(state['full_centroids'][0])) and np.isfinite(full).all()
        rows=np.arange(len(ids))
        if cfg['kmeans_max_rows'] and len(rows)>cfg['kmeans_max_rows']:
            rows=np.random.default_rng(43+h.SEED_BAND_KMEANS_SUB).permutation(len(rows))[:cfg['kmeans_max_rows']]
        self.progress('residual_clustering',train_rows=len(ids),fit_rows=len(rows))
        with self.cost.phase('residual_clustering'):
            km=KMeans(n_clusters=self.K,random_state=43+h.SEED_BAND_KMEANS,n_init=cfg['kmeans_n_init']).fit(full[rows],sample_weight=r[rows])
            mu=km.cluster_centers_.astype('float32');d2=sq_dist_to_centroids(full,mu);assign=d2.argmin(1)
        save_array(ctx/'centroids.npy',mu);save_array(ctx/'train_regions.npy',assign.astype('int16'))
        compositions=[];descriptors=[];anchor=shared['anchor']
        # Recover anchor labels from the original composition, without raw data.
        raw=h.core.load_pickle(cfg['data']);families=np.asarray(raw['families'])
        ix={n:i for i,n in enumerate(self.names)};anchor_y=np.asarray([ix[n] for n in families[anchor]])
        del raw,families;h.core._PICKLE_CACHE.clear();gc.collect()
        for k in range(self.K):
            members=np.flatnonzero(assign==k)
            if not len(members):raise ValueError(f'Empty residual cluster {k}')
            self.progress('select_specialty_block',expert=k+1,cluster_rows=len(members))
            with self.cost.phase(f'block{k+1}_selection'):
                sel=h.diversity_select(full[members],r[members],cfg['expert_block_rows'],cfg['diversity_subclusters'],cfg['diversity_n_init'],cfg['diversity_batch_size'],43+h.SEED_BAND_DIVERSITY+k)
            top=members[sel];block=ids[top];allids=np.concatenate([anchor,block])
            assert len(np.unique(block))==len(block) and not np.intersect1d(anchor,block).size
            assert len(block)==min(len(members),cfg['expert_block_rows'])
            save_array(ctx/f'expert{k+1}_block.npy',block);save_array(ctx/f'expert{k+1}_all.npy',allids)
            counts=np.bincount(np.r_[anchor_y,y[top]],minlength=self.C).astype(float)
            assert (counts>0).all()
            prior=(counts+cfg['prior_alpha'])/(counts.sum()+cfg['prior_alpha']*self.C)
            descriptors.append(h.expert_descriptor(r[top],float(r[members].sum()/r.sum()),len(top),prior,float(d2[members,k].mean()),cfg['expert_cost']))
            for role,labels in [('anchor',anchor_y),('block',y[top]),('context',np.r_[anchor_y,y[top]])]:
                hist=np.bincount(labels,minlength=self.C)
                compositions.append(dict(expert=k+1,role=role,rows=len(labels),classes_present=int((hist>0).sum()),global_correct_block_rows=int((p[top].argmax(1)==y[top]).sum()) if role=='block' else None,**dict(zip(self.names,map(int,hist)))))
        save_array(self.cache/'qk.npy',np.stack(descriptors))
        pd.DataFrame(compositions).to_csv(ctx/'composition.csv',index=False)
        write(ctx/'COMPLETE.json',dict(K=self.K,seed=43,anchor_rows=len(anchor),block_cap=cfg['expert_block_rows'],recipe='unchanged residual-weighted KMeans + diversity selection + common anchor',source_global_sha256=audit['global_fit_sha256'],source_state_sha256=audit['train_ids_sha256'],test_labels_used=False,selection_helper_sha256=sha(self.out/'original_bank_helpers.py'),total_context_rows=sum(v['rows'] for v in compositions if v['role']=='context')))

    def prepare(self):
        import torch
        from tabpfn import TabPFNClassifier
        from nfv3_v3_exp38_frozen_cache import sq_dist_to_centroids,AffinityRef
        if (self.cache/'COMPLETE.json').exists():return
        h=self.load_helpers();self.build_contexts(h);cfg=self.config
        meta=json.loads((self.ref/'cache/COMPLETE.json').read_text())
        # Immutable symlinks reuse precisely the same populations and Global.
        for split in ['route','cal','eval']:
            for key in ['ids','X','y','time','scenario','hash','p0','z','affinity_0']:
                link_file(self.ref/f'cache/{split}_{key}.npy',self.cache/f'{split}_{key}.npy')
        link_file(self.ref/'cache/cal_mask.npy',self.cache/'cal_mask.npy')
        state=dict(np.load(self.base/'residual_state.npz'));sig=self.signature(state)
        mu=np.load(self.out/'contexts/centroids.npy')
        for split in ['route','cal','eval']:
            self.cached(split+'_distance',lambda split=split:np.sqrt(sq_dist_to_centroids(sig.observable(self.array(split,'z'),self.temperature(self.array(split,'p0'))),mu[:,:sig.obs_dim])))
        self.progress('load_context_features')
        with self.cost.phase('input_loading'):
            suite=h.core.load_pickle(cfg['data']);X=suite['X'];families=np.asarray(suite['families']);ix={n:i for i,n in enumerate(self.names)}
            contexts=[np.load(self.out/f'contexts/expert{k}_all.npy') for k in range(1,self.K+1)]
            ctx_X=[np.nan_to_num(np.asarray(X[ids],dtype='float32')) for ids in contexts]
            ctx_y=[np.asarray([ix[n] for n in families[ids]],dtype='int16') for ids in contexts]
            del suite,X,families;h.core._PICKLE_CACHE.clear();gc.collect()
        shared=np.load(self.source/'shared_context_ids.npz');anchor=shared['anchor']
        for ids in contexts:
            np.testing.assert_array_equal(ids[:len(anchor)],anchor)
            assert not np.intersect1d(ids,np.r_[self.array('route','ids'),self.array('cal','ids'),self.array('eval','ids')]).size
        pca=joblib.load(self.base/'feature_pca.joblib')
        if not all((self.cache/f'eval_affinity_{k}.npy').exists() and (self.cache/f'route_affinity_{k}.npy').exists() and (self.cache/f'cal_affinity_{k}.npy').exists() for k in range(1,self.K+1)):
            global_model=TabPFNClassifier.load_from_fit_state(self.source/'global_fit_state.tabpfn_fit',device='cuda')
            rng=np.random.default_rng(43+h.SEED_BAND_AFFINITY)
            rng.permutation(len(shared['global_context']))  # same RNG sequence as EXP62
            for k in range(1,self.K+1):
                block_X=ctx_X[k-1][len(anchor):];pos=rng.permutation(len(block_X))[:cfg['affinity_ref_rows']]
                zr=self.cached(f'affinity_ref_{k}',lambda block_X=block_X,pos=pos:pca.transform(self.infer(global_model,block_X[pos],f'affinity_ref_{k}',True)).astype('float32'))
                affinity=AffinityRef(zr,cfg['affinity_nn'])
                for split in ['route','cal','eval']:
                    def produce(split=split):
                        _,first,inv=np.unique(self.array(split,'hash'),return_index=True,return_inverse=True)
                        return affinity.score(self.array(split,'z')[first],chunk=65536)[inv]
                    with self.cost.phase(f'affinity{k}_{split}'):self.cached(f'{split}_affinity_{k}',produce)
            del global_model;gc.collect();torch.cuda.empty_cache()
        for k in range(1,self.K+1):
            if all((self.cache/f'{s}_p{k}.npy').exists() for s in ['route','cal','eval']):continue
            self.progress('fit_expert',expert=k,context_rows=len(ctx_y[k-1]))
            fit=self.out/f'expert{k}.tabpfn_fit'
            if fit.exists():model=TabPFNClassifier.load_from_fit_state(fit,device='cuda')
            else:
                extra={'inference_precision':torch.float32} if cfg['det_precision']=='float32' else {}
                model=TabPFNClassifier(device=cfg['device'],model_path=cfg['model_path'],random_state=43,n_estimators=cfg['n_estimators'],auto_scale_n_estimators=False,fit_mode=cfg['fit_mode'],keep_cache_on_device=cfg['keep_cache_on_device'],ignore_pretraining_limits=cfg['ignore_pretraining_limits'],**extra)
                with self.cost.phase(f'expert{k}_fit'):model.fit(ctx_X[k-1],ctx_y[k-1])
                model.save_fit_state(fit)
            np.testing.assert_array_equal(model.classes_,np.arange(self.C))
            write(self.out/f'expert{k}_identity.json',dict(seed=43,K=self.K,fit_sha256=sha(fit),context_ids_sha256=sha(self.out/f'contexts/expert{k}_all.npy'),shared_fitted_model_for_all_splits=True))
            for split in ['route','cal','eval']:
                with self.cost.phase(f'expert{k}_{split}'):self.cached(f'{split}_p{k}',lambda split=split:self.infer(model,self.array(split,'X'),f'expert{k}_{split}'))
            del model;gc.collect();torch.cuda.empty_cache()
        for split in ['route','cal','eval']:
            n=len(self.array(split,'y'))
            for key in ['distance']+[f'p{k}' for k in range(self.K+1)]+[f'affinity_{k}' for k in range(self.K+1)]:
                a=self.array(split,key);assert len(a)==n and np.isfinite(a).all()
        meta.update(n_experts=self.K,source_run=str(self.out),context_recipe='original anchor + residual diversity-selected block',shared_global_and_splits=str(self.ref))
        write(self.cache/'COMPLETE.json',meta)
        self.progress('cache_complete')

    def reuse_k4(self):
        for path in (self.ref/'cache').glob('*'):
            if path.is_file():link_file(path,self.cache/path.name)
        for name in ['summary.csv','class_metrics.csv','cost.json']:
            # Keep the new diagnostic cost separate from original K=4 run cost.
            link_file(self.ref/name,self.out/('source_cost.json' if name=='cost.json' else name))
        policy=self.out/'policy'
        if not policy.exists():policy.symlink_to(self.ref/'policy',target_is_directory=True)
        write(self.out/'REUSED.json',dict(source=str(self.ref),K=4,seed=43,source_complete_sha256=sha(self.ref/'COMPLETE.json'),new_tabpfn_inference=False))

    def diagnostics(self):
        from nfv3_v3_exp38_frozen_cache import sq_dist_to_centroids,save_array
        from nfv3_v3_exp51_scorer_target import chronological_cal_split
        out=self.out/'diagnostics';out.mkdir(exist_ok=True)
        ycal=self.array('cal','y')
        select,confirm=chronological_cal_split(ycal,self.array('cal','scenario'),self.array('cal','time'),self.array('cal','hash'),self.array('cal','mask'),.3)
        np.testing.assert_array_equal(select,np.load(self.ref/'policy/cal_select_positions.npy'))
        np.testing.assert_array_equal(confirm,np.load(self.ref/'policy/cal_confirm_positions.npy'))
        rows=[];summaries=[]
        for split,positions,label in [('cal',select,'cal_select'),('cal',confirm,'cal_confirm'),('eval',slice(None),'full_test')]:
            y=self.array(split,'y')[positions];g=self.array(split,'p0')[positions].argmax(1)
            for k in range(self.K+1):
                pred=self.array(split,f'p{k}')[positions].argmax(1)
                tags=dict(dataset=self.ds,K=self.K,seed=43,split=label,model='global' if k==0 else f'expert{k}')
                pc,summary,cm=class_records(y,pred,g,self.names,**tags)
                rows.extend(pc);summaries.append(summary);save_array(out/f'{label}_{tags["model"]}_confusion.npy',cm)
        frame=pd.DataFrame(rows)
        baseline=frame[frame.model=='global'][['split','class','precision','recall','f1']].rename(columns={k:'global_'+k for k in ['precision','recall','f1']})
        frame=frame.merge(baseline,on=['split','class'],validate='many_to_one')
        for metric in ['precision','recall','f1']:frame['delta_'+metric]=frame[metric]-frame['global_'+metric]
        frame.to_csv(out/'expert_class_metrics.csv',index=False);pd.DataFrame(summaries).to_csv(out/'expert_summary.csv',index=False)
        # Truth-assisted residual membership is explicitly a routing diagnostic,
        # separate from direct expert classification above and deployment policy.
        h=self.load_helpers();state=dict(np.load(self.base/'residual_state.npz'));sig=self.signature(state)
        mu=state['full_centroids'] if self.K==4 else np.load(self.out/'contexts/centroids.npy')
        y=self.array('eval','y');p=self.temperature(self.array('eval','p0'))
        r=np.minimum(h.balanced_ce(p,y,state['w_bal']),float(state['train_residual_clip']))
        full=sig.full(self.array('eval','z'),p,y,r);region=sq_dist_to_centroids(full,mu).argmin(1)
        g=self.array('eval','p0').argmax(1);pred=g.copy()
        for k in range(self.K):
            mask=region==k;pred[mask]=self.array('eval',f'p{k+1}')[mask].argmax(1)
        pc,summary,cm=class_records(y,pred,g,self.names,dataset=self.ds,K=self.K,seed=43,split='full_test',model='label_assisted_residual_oracle')
        pd.DataFrame(pc).to_csv(out/'residual_oracle_class_metrics.csv',index=False);write(out/'residual_oracle_summary.json',summary)
        save_array(out/'residual_oracle_confusion.npy',cm)
        write(out/'protocol.json',dict(expert_metrics='same full population, raw multiclass argmax, includes all other-class FP',validation='same frozen choose/confirm populations across K; raw metrics, not prevalence-matched to test',oracle='test truth used only for residual membership; never expert input or policy training',test_role='previously observed development holdout',K_selected_on_test=False))

    def run(self):
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(self.args.threads);torch.set_num_interop_threads(4);torch.manual_seed(43);np.random.seed(43)
        torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:
            with threadpool_limits(self.args.threads):
                if self.K==4:self.reuse_k4()
                else:self.prepare();self.policy()
                self.progress('direct_expert_diagnostics');self.diagnostics()
            self.cost.finish()
            write(self.out/'COMPLETE.json',dict(seed=43,K=self.K,dataset=self.ds,seconds=time.time()-self.started,K4_reused=self.K==4));self.progress('complete')
            (self.out/'ERROR.json').unlink(missing_ok=True)
        except BaseException:
            write(self.out/'ERROR.json',dict(traceback=traceback.format_exc(),failed_epoch=time.time()));self.cost.finish();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',choices=['cic2018','toniot'],required=True);p.add_argument('--k',type=int,choices=[2,4,6,8],required=True);p.add_argument('--batch',type=int,default=65536);p.add_argument('--threads',type=int,default=6)
    SweepWorker(p.parse_args()).run()

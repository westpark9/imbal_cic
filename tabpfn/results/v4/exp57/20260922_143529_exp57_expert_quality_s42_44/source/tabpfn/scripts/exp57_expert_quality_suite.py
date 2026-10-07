#!/usr/bin/env python3
"""EXP57: paired K<C context controls, actual seed repeats, and fixed operating points.

Six workers (two datasets x three seeds), one GPU worker at a time. Each worker
shares one fitted global, one anchor, one residual partition and K=4 across three
banks: designed, class-histogram-matched random, and size-matched balanced random.
No HTML publishing. Existing observed holdout/time slices are explicitly diagnostic.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from exp56_five_condition_evaluation import fpr_threshold


def write(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def allocate_balanced(budget,available):
    available=np.asarray(available,int);out=np.zeros_like(available)
    if budget<0 or budget>available.sum():raise ValueError('Infeasible context budget')
    while out.sum()<budget:
        active=np.flatnonzero(out<available);remaining=budget-int(out.sum())
        addition=np.full(len(active),remaining//len(active),int);addition[:remaining%len(active)]+=1
        out[active]+=np.minimum(addition,available[active]-out[active])
    return out


def sample_histogram(y,counts,seed):
    rng=np.random.default_rng(seed);parts=[]
    for c,n in enumerate(counts):
        pool=np.flatnonzero(y==c)
        if n>len(pool):raise ValueError('Insufficient class support')
        if n:parts.append(rng.choice(pool,int(n),replace=False))
    out=np.sort(np.concatenate(parts)) if parts else np.empty(0,int)
    assert len(np.unique(out))==len(out)
    np.testing.assert_array_equal(np.bincount(y[out],minlength=len(counts)),counts)
    return out


def class_metrics(y,pred,C,weights=None):
    cm=np.bincount(y.astype(np.int64)*C+pred,minlength=C*C,weights=weights).reshape(C,C)
    tp=np.diag(cm);support=cm.sum(1);fp=cm.sum(0)-tp;fn=support-tp
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(C),where=2*tp+fp+fn>0)
    return cm,tp,fp,fn,f1


def validation_region_choice(y,predictions,regions,K,C,minimum=50):
    """One fixed predictor per observable region; global wins ties/low support."""
    choices=np.zeros(K,int)
    for k in range(K):
        mask=regions==k
        if mask.sum()<minimum:continue
        present=np.unique(y[mask]);best=-np.inf
        for j,pred in enumerate(predictions):
            score=class_metrics(y[mask],pred[mask],C)[-1][present].mean()
            if score>best+1e-12:choices[k]=j;best=score
    return choices


def paired_bootstrap(y,pred,g,hashes,times,C,seed,repeats=200):
    """Poisson cluster bootstrap of paired macro-F1 differences; two dependence checks."""
    rows=[];rng=np.random.default_rng(seed)
    for kind,groups in [('exact_vector',hashes),('time_block',times)]:
        _,inv=np.unique(groups,return_inverse=True);G=int(inv.max())+1
        if kind=='exact_vector':
            # Conflicting vectors were removed; identical inputs have identical predictions.
            _,first,counts=np.unique(groups,return_index=True,return_counts=True)
            if not (np.array_equal(y,y[first][inv]) and np.array_equal(pred,pred[first][inv]) and np.array_equal(g,g[first][inv])):
                raise ValueError('Identical-vector labels/predictions disagree')
            ey=y[first]*C+pred[first];gy=y[first]*C+g[first]
        else:
            em=np.bincount(inv*C*C+y*C+pred,minlength=G*C*C).reshape(G,C*C)
            gm=np.bincount(inv*C*C+y*C+g,minlength=G*C*C).reshape(G,C*C)
        deltas=[]
        for _ in range(repeats):
            w=rng.poisson(1.,G)
            if kind=='exact_vector':
                a=np.bincount(ey,weights=w*counts,minlength=C*C).reshape(C,C)
                b=np.bincount(gy,weights=w*counts,minlength=C*C).reshape(C,C)
            else:a=(w@em).reshape(C,C);b=(w@gm).reshape(C,C)
            def score(cm):
                den=cm.sum(0)+cm.sum(1)
                return np.divide(2*np.diag(cm),den,out=np.zeros(C),where=den>0).mean()
            deltas.append(score(a)-score(b))
        rows.append(dict(grouping=kind,groups=G,replicates=repeats,delta_macro_f1_low=float(np.quantile(deltas,.025)),delta_macro_f1_high=float(np.quantile(deltas,.975))))
    return rows


class Experiment:
    def __init__(self,out,job):
        self.out=Path(out);self.job=job;self.started=time.time();self.summaries=[];self.perclass=[];self.probability=[];self.operating=[];self.uncertainty=[];self.region_metrics=[]

    def progress(self,phase,**extra):
        write(self.out/'progress.json',dict(phase=phase,pid=os.getpid(),seconds=time.time()-self.started,**extra));print('EXP57 '+phase+' '+json.dumps(extra),flush=True)

    def save_tables(self):
        for name,rows in [('summary',self.summaries),('per_class',self.perclass),('probability_metrics',self.probability),('fixed_operating_points',self.operating),('paired_bootstrap',self.uncertainty),('region_metrics',self.region_metrics)]:
            if rows:pd.DataFrame(rows).to_csv(self.out/(name+'.csv'),index=False)

    def evaluate(self,name,cal,test,thresholds=None):
        v=self.v;names=v['class_names'];C=len(names);y=v['y_eval'];yc=v['y_cal'];pred=test.argmax(1)
        np.save(self.out/'predictions'/(name+'.npy'),pred.astype(np.int16))
        cm,tp,fp,fn,f1=class_metrics(y,pred,C)
        self.summaries.append(dict(model=name,macro_f1=float(f1.mean()),accuracy=float((pred==y).mean()),rows=len(y)))
        for c,cl in enumerate(names):
            self.perclass.append(dict(model=name,**{'class':cl},support=int(tp[c]+fn[c]),TP=int(tp[c]),FP=int(fp[c]),FN=int(fn[c]),f1=float(f1[c])))
            positive=y==c;negative=~positive
            if not positive.any() or not negative.any():continue
            score=np.asarray(test[:,c]);fpr,recall,_=roc_curve(positive,score,drop_intermediate=False)
            self.probability.append(dict(model=name,**{'class':cl},AP=float(average_precision_score(positive,score)),R_at_FPR_001=float(recall[fpr<=.001].max()),R_at_FPR_01=float(recall[fpr<=.01].max())))
            q=self.cal_mask;neg=q&(yc!=c)
            if not neg.any():continue
            for target in [.001,.01]:
                threshold=thresholds[(c,target)]
                for window,mask in [('full_test',np.ones(len(y),bool))]+[(f'time_quartile_{j+1}',self.time_window==j) for j in range(4)]:
                    pos=mask&positive;non=mask&negative;decision=score>threshold
                    self.operating.append(dict(model=name,**{'class':cl},window=window,validation_FPR_target=target,threshold=threshold,
                                               positives=int(pos.sum()),negatives=int(non.sum()),TP=int((decision&pos).sum()),FP=int((decision&non).sum()),
                                               recall=float(decision[pos].mean()) if pos.any() else np.nan,FPR=float(decision[non].mean()) if non.any() else np.nan))
        if name!='global_raw':
            for r in paired_bootstrap(y,pred,self.g,self.hashes,self.time_blocks,C,self.job['seed'],self.job['bootstrap_repeats']):self.uncertainty.append(dict(model=name,**r))
        self.save_tables();return pred

    def model(self,name,clf,corr,tune=None):
        v=self.v;C=v['n_classes'];base=self.out/'probabilities';self.progress('predicting',model=name)
        if tune is None:tune=v['batched_proba'](clf,v['X_tune'],name+'/tune')
        from nfv3_v3_exp31_c0alloc import select_prior_hypers
        (beta,temp),grid=select_prior_hypers(tune[v['tune_sel_mask']],v['y_tune'][v['tune_sel_mask']],v['w_bal'],corr,[0.,.5,1.],[.8,1.,1.25])
        grid.to_csv(self.out/(name+'_tune_grid.csv'),index=False)
        write(self.out/(name+'_selection.json'),dict(beta=beta,temperature=temp,selection='tune class-balanced NLL; no test selection'))
        cal=v['batched_proba'](clf,v['X_cal'],name+'/cal').astype(np.float32)
        # Freeze thresholds/selected hyperparameters before computing test probabilities.
        selected_cal=corr.correct(cal,beta,temp).astype(np.float32)
        thresholds=[]
        for variant,pr in [('raw',cal),('calibrated',selected_cal)]:
            for c in range(C):
                for target in [.001,.01]:
                    q=self.cal_mask
                    thresholds.append(dict(variant=variant,class_index=c,target=target,threshold=fpr_threshold(pr[q,c],v['y_cal'][q]==c,self.cal_weights[q],target)))
        write(self.out/(name+'_thresholds.json'),thresholds)
        test=v['batched_proba'](clf,v['X_eval'],name+'/test').astype(np.float32)
        for split,p in [('tune',tune),('cal',cal),('test',test)]:np.save(base/(name+'_'+split+'.npy'),p.astype(np.float32))
        if name=='global':self.g=test.argmax(1)
        frozen={variant:{(r['class_index'],r['target']):r['threshold'] for r in thresholds if r['variant']==variant} for variant in ['raw','calibrated']}
        raw_pred=self.evaluate(name+'_raw',cal,test,frozen['raw'])
        selected_test=corr.correct(test,beta,temp).astype(np.float32)
        self.evaluate(name+'_calibrated',selected_cal,selected_test,frozen['calibrated'])
        self.progress('model_complete',model=name)
        return tune.argmax(1),raw_pred

    def run_bank_diagnostics(self,v):
        self.v=v;K=self.job['K'];C=v['n_classes'];assert 1<=K<C
        self.progress('preparing_reference_bank',K=K,classes=C)
        names=v['class_names'];y=v['y_eval'];yc=v['y_cal']
        self.hashes=pd.util.hash_pandas_object(pd.DataFrame(v['X_eval']),index=False).to_numpy()
        t=v['ts_all'][v['eval_idx']];bounds=np.quantile(t,[.25,.5,.75]);self.time_window=np.searchsorted(bounds,t,side='right')
        # Time units vary in source; 100 fixed equal-duration blocks avoid assuming seconds.
        edges=np.linspace(float(t.min()),float(t.max())+1,101);self.time_blocks=np.searchsorted(edges[1:-1],t,side='right')
        np.savez(self.out/'evaluation_identity.npz',ids=v['eval_idx'],y=y,hashes=self.hashes,time=t,time_quartile=self.time_window)
        self.cal_mask=v['cal_sel_mask'];counts=np.bincount(yc[self.cal_mask],minlength=C)
        if (counts==0).any():raise ValueError('Calibration class missing after train-overlap mask')
        self.cal_weights=v['ref_prior'][yc]/np.maximum(counts[yc],1)
        write(self.out/'design.json',dict(dataset=self.job['dataset'],seed=self.job['seed'],K=K,C=C,arms=['designed','matched_random','balanced_random'],
             anchor_rows=len(v['anchor'][1]),global_rows=len(v['glob_ctx'][1]),class_names=names,calibration_mask_rows=int(self.cal_mask.sum()),
             test_rows=len(y),time_quartile_boundaries=bounds.tolist(),test_role='previously observed development holdout; temporal slices are not new independent data'))
        np.savez(self.out/'shared_context_ids.npz',global_context=v['g_idx'],anchor=v['anchor_idx'],expert_pool=v['exp_idx'])
        g_tune,_=self.model('global',v['glob'],v['corr0'],v['p0_tune_raw'])
        bank=v['build_bank'](K);mu=bank['mu'][:,:v['sig'].obs_dim]
        v['embed_stage']['tag']='exp57/eval_regions';z_eval=v['phi'].transform(v['X_eval'])
        from nfv3_v3_exp31_c0alloc import sq_dist_to_centroids,PriorCorrector
        g_test_raw=np.load(self.out/'probabilities/global_test.npy',mmap_mode='r')
        p0_eval=v['corr0'].correct(g_test_raw,0.,v['temp'])
        regions=sq_dist_to_centroids(v['sig'].observable(z_eval,p0_eval),mu).argmin(1)
        tune_regions=sq_dist_to_centroids(v['sig'].observable(v['z_tune'],v['p0_tune']),mu).argmin(1)
        del z_eval,p0_eval;v['free_gpu']()
        np.savez(self.out/'fixed_regions.npz',centroids=mu,test=regions,tune=tune_regions)
        constant_labels=np.full(K,np.bincount(v['y_tune'][v['tune_sel_mask']],minlength=C).argmax(),int)
        for k in range(K):
            mask=v['tune_sel_mask']&(tune_regions==k)
            if mask.any():constant_labels[k]=np.bincount(v['y_tune'][mask],minlength=C).argmax()
        constant_pred=constant_labels[regions]
        write(self.out/'region_constant_reference.json',dict(class_per_region=constant_labels.tolist(),selection='tune-only majority label per observable region'))
        references=[ex['block_rows'].copy() for ex in bank['experts']];composition=[]
        available=np.bincount(v['y_exp'],minlength=C)
        for arm_id,arm in enumerate(['designed','matched_random','balanced_random']):
            tune_preds=[g_tune];test_preds=[self.g]
            for k in range(K):
                reference=references[k];counts=np.bincount(v['y_exp'][reference],minlength=C)
                if arm=='designed':top=reference;ex=bank['experts'][k];clf=ex['clf'];corr=ex['corr']
                else:
                    target=counts if arm=='matched_random' else allocate_balanced(len(reference),available)
                    top=sample_histogram(v['y_exp'],target,self.job['seed']+10000*arm_id+k)
                    yc_ctx=np.concatenate([v['anchor'][1],v['y_exp'][top]])
                    self.progress('fitting',arm=arm,expert=k+1,context_rows=len(yc_ctx))
                    clf=v['make_clf'](np.concatenate([v['anchor'][0],v['X_exp'][top]]),yc_ctx)
                    corr=PriorCorrector(yc_ctx,C,v['ref_prior'],v['args'].prior_alpha)
                name=f'{arm}_e{k+1}';np.save(self.out/'contexts'/(name+'.npy'),v['exp_idx'][top])
                actual=np.bincount(v['y_exp'][top],minlength=C)
                if arm=='matched_random':np.testing.assert_array_equal(actual,counts)
                assert len(top)==len(reference)
                composition.append(dict(arm=arm,expert=k+1,block_rows=len(top),**{c:int(actual[j]) for j,c in enumerate(names)}))
                pd.DataFrame(composition).to_csv(self.out/'context_compositions.csv',index=False)
                tp,ep=self.model(name,clf,corr);tune_preds.append(tp);test_preds.append(ep)
                if arm=='designed':bank['experts'][k].pop('clf',None)
                del clf;v['free_gpu']()
            choices=validation_region_choice(v['y_tune'][v['tune_sel_mask']],[p[v['tune_sel_mask']] for p in tune_preds],tune_regions[v['tune_sel_mask']],K,C)
            write(self.out/(arm+'_validation_region_reference.json'),dict(choice_global0_expert1toK=choices.tolist(),selection='fixed predictor per input region on tune; global wins ties; not an oracle'))
            matrix=np.stack(test_preds,axis=1);raw_region=matrix[np.arange(len(y)),regions+1];best_region=matrix[np.arange(len(y)),choices[regions]]
            for k in range(K):
                mask=regions==k
                if not mask.any():continue
                for model,pred in [('global',self.g)]+[(f'e{j+1}',p) for j,p in enumerate(test_preds[1:])]+[('region_constant',constant_pred)]:
                    cm,tp,fp,fn,f1=class_metrics(y[mask],pred[mask],C)
                    self.region_metrics.append(dict(arm=arm,region=k+1,model=model,rows=int(mask.sum()),classes_present=int((cm.sum(1)>0).sum()),macro_f1=float(f1.mean()),present_class_macro_f1=float(f1[cm.sum(1)>0].mean()),accuracy=float((pred[mask]==y[mask]).mean())))
            for label,pred in [('fixed_region',raw_region),('validation_region_reference',best_region),('region_constant',constant_pred)]:
                cm,tp,fp,fn,f1=class_metrics(y,pred,C);name=arm+'_'+label
                self.summaries.append(dict(model=name,macro_f1=float(f1.mean()),accuracy=float((y==pred).mean()),rows=len(y)))
                np.save(self.out/'predictions'/(name+'.npy'),pred.astype(np.int16))
                for c,cl in enumerate(names):self.perclass.append(dict(model=name,**{'class':cl},support=int(tp[c]+fn[c]),TP=int(tp[c]),FP=int(fp[c]),FN=int(fn[c]),f1=float(f1[c])))
            self.save_tables()
        self.progress('complete');write(self.out/'COMPLETE.json',dict(seconds=time.time()-self.started,dataset=self.job['dataset'],seed=self.job['seed'],K=K,models=len(self.summaries)))
        return str(self.out)


def prepare(root,workspace):
    root.mkdir(parents=True,exist_ok=False);(root/'source/tabpfn/scripts').mkdir(parents=True);(root/'source/scripts').mkdir(parents=True)
    for directory in ['tabpfn/scripts','scripts']:
        for p in (workspace/directory).glob('*.py'):shutil.copy2(p,root/'source'/directory/p.name)
    bases={'cic2018':workspace/'tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42',
           'toniot':workspace/'tabpfn/results/20260916_021730_nfv3_toniot_exp48_clean_revalidation_s42'}
    jobs=[]
    for seed in [42,43,44]:
        for ds,base in bases.items():
            args=json.loads((base/'baseline_args.json').read_text());job_dir=root/f'{ds}_s{seed}';job_dir.mkdir()
            for directory in ['probabilities','predictions','contexts']:(job_dir/directory).mkdir()
            args.update(seed=seed,k_candidates='4',deterministic=True,prior_betas='0,0.5,1.0',prune_mode='off',
                        feasibility_banks='none',out_root=str(job_dir),resume_dir=str(job_dir/'resume'),models_dir=str(job_dir/'models'))
            job=dict(id=f'{ds}_s{seed}',dataset=ds,seed=seed,K=4,bootstrap_repeats=200,args=args)
            write(job_dir/'job.json',job);jobs.append(job['id'])
    write(root/'protocol.json',dict(experiment='EXP57',jobs=jobs,K=4,seeds=[42,43,44],arms=['designed','matched_random','balanced_random'],
          scope='paired expert quality, global calibration controls, fixed-FPR temporal transfer, actual independent refits',
          test_role='existing development holdout and temporal slices; no new independent dataset available',
          html_updates=False,created_epoch=time.time(),source_sha256={str(p.relative_to(root/'source')):digest(p) for p in (root/'source').rglob('*.py')}))
    write(root/'status.json',dict(state='prepared',jobs=jobs,completed=[]));print(root,flush=True)


def worker(root,job_id):
    from threadpoolctl import threadpool_limits
    from nfv3_conflict_clean import install_clean_loader
    import nfv3_v3_common as core
    out=root/job_id;job=json.loads((out/'job.json').read_text());experiment=Experiment(out,job)
    install_clean_loader(core,job['args']['clean_manifest'])
    source=(HERE/'nfv3_v3_exp31_c0alloc.py').read_text()
    anchor='    t0 = time.time()\n    ksel_rows, prior_expert_rows, best = [], [], None'
    if source.count(anchor)!=1:raise ValueError('EXP31 checkpoint anchor changed')
    source=source[:source.index(anchor)]+'    return _exp57.run_bank_diagnostics(locals())\n'
    (out/'executed_prefix.py').write_text(source)
    module=types.ModuleType('exp57_base');module.__file__=str(HERE/'nfv3_v3_exp31_c0alloc.py');module._exp57=experiment
    sys.modules[module.__name__]=module
    exec(compile(source,str(out/'executed_prefix.py'),'exec'),module.__dict__)
    experiment.progress('loading_and_fitting_global')
    try:
        with threadpool_limits(16):module.run_exp29(types.SimpleNamespace(**job['args']))
    except BaseException:
        write(out/'ERROR.json',dict(traceback=traceback.format_exc(),seconds=time.time()-experiment.started));raise


def controller(root):
    protocol=json.loads((root/'protocol.json').read_text());started=time.time();completed=[];failed=[];child=None
    def stop(signum,frame):
        if child and child.poll() is None:child.terminate()
        write(root/'status.json',dict(state='stopped',completed=completed,failed=failed,signal=signum));raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for job in protocol['jobs']:
        if (root/job/'COMPLETE.json').exists():completed.append(job);continue
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--stage','worker','--job',job]
        with (root/job/'worker.log').open('a',buffering=1) as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=root/'source')
            write(root/'status.json',dict(state='running',controller_pid=os.getpid(),worker_pid=child.pid,active_job=job,completed=completed,failed=failed,
                                          remaining=[j for j in protocol['jobs'] if j not in completed+failed+[job]],started_epoch=started,log=str(root/job/'worker.log')))
            code=child.wait()
        if code:failed.append(job)
        else:completed.append(job)
        print(json.dumps(dict(job=job,exit_code=code,completed=completed,failed=failed)),flush=True)
    # Seed-level paired summaries; do not pool row counts as independent replications.
    frames=[]
    for job in completed:
        d=pd.read_csv(root/job/'summary.csv');d['job']=job;d['dataset']=job.rsplit('_s',1)[0];d['seed']=int(job.rsplit('_s',1)[1]);frames.append(d)
    if frames:
        all_results=pd.concat(frames);all_results.to_csv(root/'all_seed_results.csv',index=False)
        all_results.groupby(['dataset','model']).macro_f1.agg(['count','mean','std','min','max']).to_csv(root/'seed_summary.csv')
    write(root/'status.json',dict(state='complete' if not failed else 'complete_with_errors',completed=completed,failed=failed,seconds=time.time()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['prepare','controller','worker'],required=True);p.add_argument('--workspace',type=Path,default=HERE.parents[1]);p.add_argument('--job');a=p.parse_args()
    if a.stage=='prepare':prepare(a.root.resolve(),a.workspace.resolve())
    elif a.stage=='controller':controller(a.root.resolve())
    else:worker(a.root.resolve(),a.job)

if __name__=='__main__':main()

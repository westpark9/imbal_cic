#!/usr/bin/env python3
"""One isolated SOTA job, exact cleaned split and frozen seed-43 Global context."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from exp61_utils import Cost, context_ids, sha, write


def load_data(request, dataset, method, out):
    import numpy as np
    if request.get('synthetic_smoke'):
        rng=np.random.default_rng(43)
        def samples(n):
            y=np.arange(n)%3;X=rng.normal(size=(n,8)).astype('float32');X[:,0]+=y*2
            return X,y.astype('int16')
        tr,ytr=samples(300);te,yte=samples(18);va,yva=samples(18)
        return dict(Xtr=tr,ytr=ytr,Xte=te,yte=yte,Xva=va,yva=yva,names=['a','b','c'],train_ids=np.arange(300),test_ids=np.arange(300,318),validation_ids=np.arange(318,336),train_pool_rows=300,synthetic=True)
    from nfv3_conflict_clean import install_clean_loader
    import nfv3_v3_common as core
    ref=json.loads((ROOT/'tabpfn/configs/exp57_clean_split_reference.json').read_text())['datasets'][dataset]
    directory=Path(request['clean_root'])/ref['directory']
    install_clean_loader(core,directory/'manifest.json',source_override=request['data'])
    args=argparse.Namespace(data=request['data'])
    loader_key={'cic2018':'cic2018','toniot':'ton_iot'}[dataset]
    X,names,tr,va,te,_,yte,_,label=core.build_dataset_config('.')[loader_key]['loader'](args)
    ids=tr if method=='xgb_full' else context_ids(ROOT,dataset)
    if not np.isin(ids,tr,assume_unique=True).all():raise ValueError('Context contains rows outside clean train')
    if len(te)!=ref['split_rows']['test']:raise ValueError('Full test required')
    ytr=label(ids)
    def finite(rows):return np.nan_to_num(np.asarray(X[rows],dtype=np.float32))
    data=dict(Xtr=finite(ids),ytr=ytr,Xte=finite(te),yte=yte,names=names,train_ids=ids,test_ids=te,train_pool_rows=len(tr),synthetic=False)
    if method=='localpfn':
        val=core.cap_per_class(va,label(va),len(names),5000,43+901)
        data.update(Xva=finite(val),yva=label(val),validation_ids=val)
    del X,label
    core._PICKLE_CACHE.clear();gc.collect()
    np.savez_compressed(out/'evaluation_identity.npz',test_ids=te,y=yte,train_ids=ids,validation_ids=data.get('validation_ids',np.array([],dtype='int64')))
    write(out/'data_identity.json',dict(dataset=dataset,class_names=names,train_pool_rows=len(tr),train_rows=len(ids),test_rows=len(te),validation_rows=len(data.get('validation_ids',[])),test_index_sha256=sha(directory/'test_idx.npy'),context='all clean train' if method=='xgb_full' else 'exact EXP59 Global C0; 100000 rows',seed=43))
    return data


def score(tag, proba, data, out):
    import numpy as np
    C=len(data['names']);p=np.asarray(proba);y=data['yte']
    assert p.shape==(len(y),C) and np.isfinite(p).all()
    pred=p.argmax(1).astype('int16');cm=np.bincount(y*C+pred,minlength=C*C).reshape(C,C)
    tp=np.diag(cm);support=cm.sum(1);fp=cm.sum(0)-tp;fn=support-tp
    div=lambda a,b:np.divide(a,b,out=np.zeros(C,dtype=float),where=b!=0)
    precision=div(tp,tp+fp);recall=div(tp,support);f1=div(2*tp,2*tp+fp+fn)
    value=dict(method=tag,macro_f1=float(f1.mean()),accuracy=float(tp.sum()/len(y)),rows=len(y),train_rows=len(data['ytr']),synthetic=data['synthetic'],classes=[dict(name=n,support=int(support[i]),TP=int(tp[i]),FP=int(fp[i]),FN=int(fn[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i,n in enumerate(data['names'])])
    np.save(out/f'{tag}_pred.npy',pred);np.save(out/f'{tag}_proba.npy',p.astype('float32'))
    write(out/f'{tag}_metrics.json',value)
    return value


def xgboost(data, request, out, cost):
    import xgboost as xgb
    model=xgb.XGBClassifier(n_estimators=3 if data['synthetic'] else 300,max_depth=8,learning_rate=.05,subsample=.8,colsample_bytree=.8,min_child_weight=1,reg_lambda=1,
        objective='multi:softprob',num_class=len(data['names']),eval_metric='mlogloss',tree_method='hist',device='cuda:0',n_jobs=request['cpu_threads'],random_state=43)
    with cost.phase('fit'):model.fit(data['Xtr'],data['ytr'])
    # The transfer to GPU is part of inference, rather than hidden from its cost.
    with cost.phase('predict'):p=model.predict_proba(data['Xte'])
    model.save_model(out/'model.ubj')
    return {request['method']:p}


def dist_adjust(proba, ytrain):
    import numpy as np
    prior=np.bincount(ytrain,minlength=proba.shape[1])/len(ytrain)
    target=np.asarray(proba,dtype=np.float64).mean(0)
    adjusted=proba.astype('float64')*target/(prior+1e-8)
    adjusted/=adjusted.sum(1,keepdims=True)
    return adjusted.astype('float32'),prior,target


def distpfn(data, request, out, cost):
    import numpy as np
    from tabpfn import TabPFNClassifier
    model=TabPFNClassifier(device='cuda:0',model_path=request['checkpoint'],random_state=43,n_estimators=1 if data['synthetic'] else 4,
        auto_scale_n_estimators=False,fit_mode='fit_with_cache',keep_cache_on_device=False,inference_config={'SUBSAMPLE_SAMPLES':None})
    with cost.phase('fit'):model.fit(data['Xtr'],data['ytr'])
    result=np.lib.format.open_memmap(out/'global_raw_proba.npy',mode='w+',dtype='float32',shape=(len(data['yte']),len(data['names'])))
    with cost.phase('predict'):
        bs=request['pfn_batch']
        for start in range(0,len(result),bs):
            end=min(start+bs,len(result));result[start:end]=model.predict_proba(data['Xte'][start:end])
            write(out/'progress.json',dict(state='running',phase='predict',rows_done=end,rows_total=len(result),updated_epoch=time.time()))
    with cost.phase('test_prior_adjustment'):adjusted,prior,target=dist_adjust(result,data['ytr'])
    write(out/'distpfn_prior.json',dict(context_prior=prior.tolist(),mean_test_posterior=target.tolist(),test_labels_used=False,variant='DistPFN; not DistPFN-T'))
    return {'global_raw':np.array(result),'distpfn':adjusted}


def boost_factory(request):
    from nfv3_v3_exp35_boostpfn import import_boostpfn, set_all_seeds
    Base,Boost,_,split=import_boostpfn(str(ROOT/'tabpfn/third_party/BoostPFN'))
    set_all_seeds(43)
    base=Base(device='cuda:0',base_path=request['boost_weights_root'],N_ensemble_configurations=1,seed=43)
    opts=argparse.Namespace(replacement=False,loss='CE',updating='exphadamard',wl_num=2,debug=False)
    model=Boost(base,request['boost_rounds'],.001,opts,split_test=True,max_samples=request['boost_samples'],batch_test=request['boost_batch'],version=2)
    return base,model,split


def replay_boost(base, model, split, Xtr, ytr, Xte, batch, callback=None):
    """Evaluate stored contexts and coefficients on new rows, without labels.

    Original predict_proba ignores X and combines cached test logits. This replay
    uses the same v2 class mapping (unseen classes get logit zero) and softmax.
    """
    import numpy as np
    import torch
    C=int(ytr.max())+1;total=np.zeros((len(Xte),C),dtype=np.float64)
    for number,(idx,alpha) in enumerate(zip(model.sampled_idxs,model.alphas),1):
        idx=np.asarray(idx,dtype=np.int64)
        base.fit(Xtr[idx],ytr[idx],overwrite_warning=True)
        local=split(base,Xte,test_batch=batch,return_logits=True)
        total[:,base.classes_.astype(int)]+=float(alpha)*local
        if callback:callback(number,len(model.alphas))
    return torch.softmax(torch.tensor(total,dtype=torch.float32),dim=-1).numpy()


def boostpfn(data, request, out, cost):
    import numpy as np
    import torch
    with cost.phase('model_load'):base,model,split=boost_factory(request)
    # No test observations are needed to learn boosting weights. Empty test is
    # supported by the original fit; parity is covered by the GPU smoke test.
    original=model.get_weak_learner
    def progress(*a,**kw):
        value=original(*a,**kw)
        print(f'boost training learner {len(model.sampled_idxs)+1}/{request["boost_rounds"]}',flush=True)
        return value
    model.get_weak_learner=progress
    with cost.phase('fit'):
        model.fit(torch.from_numpy(data['Xtr']),torch.from_numpy(data['ytr'].astype('int64')),torch.empty((0,data['Xtr'].shape[1])))
    np.savez_compressed(out/'boost_state.npz',sampled_idxs=np.asarray(model.sampled_idxs),alphas=np.asarray(model.alphas))
    def update(done,total):write(out/'progress.json',dict(state='running',phase='predict',learners_done=done,learners_total=total,updated_epoch=time.time()))
    with cost.phase('predict'):p=replay_boost(base,model,split,data['Xtr'],data['ytr'],data['Xte'],request['boost_batch'],update)
    return {'boostpfn':p}


def localpfn(data, request, out, cost):
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    from nfv3_v3_exp36_localpfn import import_localpfn,clone_forward_output
    from torch.utils.tensorboard import SummaryWriter
    with cost.phase('preprocess'):
        scaler=StandardScaler().fit(data['Xtr'])
        import joblib
        joblib.dump(scaler,out/'scaler.joblib')
        d={}
        for split,short in [('train','tr'),('valid','va'),('test','te')]:
            d['X_'+split]=np.clip(scaler.transform(data['X'+short]),-10,10).astype('float32')
            d['X_'+split+'_one_hot']=d['X_'+split]
            d['y_'+split]=data['y'+short].astype('int64')
        d['dataset_info']=dict(name=request['dataset'],cat_idx=[],cat_dims=[],num_features=data['Xtr'].shape[1],num_classes=len(data['names']))
    small=data['synthetic']
    args=argparse.Namespace(device='cuda:0',method='ft',seed=43,timing=False,inf_temperature=.8,context_length=32 if small else 1000,class_choice='equal',dynamic=False,
        batch_size=2,batch_size_inf=4 if small else request['local_batch'],use_one_hot_emb=False,onehot_retrieval=False,embedding='raw',lr=1e-5,opt_weight_decay=.01,
        num_epochs=2 if small else 21,num_steps=1 if small else 30,early_stopping_metric='auc',early_stopping_rounds=100,eval_interval=1,train_query_length=32 if small else 1000,
        scheduler=False,better_selection=False,exact_knn=False,save_data=True,splits_evaluated='valid',ensemble_dist=False,integrated=False,ensemble=False,ensemble_dist_folder='',clipping_val=10)
    with cost.phase('model_load'):
        PFN,utils,_,_,train,eval_ft=import_localpfn(str(ROOT/'tabpfn/third_party/LoCalPFN'))
        import faiss
        faiss.omp_set_num_threads(request['cpu_threads'])
        utils.seed_everything(43);utils.create_dataloaders(args,d)
        model,_=PFN.load_old(device='cuda:0',path=request['local_checkpoint']);clone_forward_output(model)
    import methods.ftknn as ft
    def valid_eval(*a,**kw):
        with cost.phase('validation'):return eval_ft(*a,**kw)
    ft.eval_ft_knn=valid_eval
    best=out/'fit/data'/request['dataset']/'model_best.pth'
    fit_complete=out/'FIT_COMPLETE.json'
    if fit_complete.exists():
        saved=json.loads(fit_complete.read_text())
        if sha(best)!=saved['best_sha256']:raise ValueError('Fine-tuning checkpoint changed')
        cost.phases.extend(saved['phases'])
        print('Reusing completed fine-tuning; cost retains original fit phases.',flush=True)
    else:
        with SummaryWriter(out/'fit') as writer:
            with cost.phase('fit'):model.train();train(args,model,d,writer,str(out/'fit'))
        write(fit_complete,dict(best_sha256=sha(best),phases=[p for p in cost.phases if p['phase'] in ['fit','validation']]))
    model.load_state_dict(torch.load(best,map_location='cuda:0',weights_only=True));model.eval()
    # Chunked evaluation checkpoints prediction progress. Each chunk has every
    # class represented in its dummy labels so the upstream output slice uses C;
    # real test labels are not passed to the model or used in retrieval.
    from torch.utils.data import DataLoader,TensorDataset
    C=len(data['names']);N=len(data['yte']);path=out/'test_logits.npy'
    resume=out/'PREDICT_PROGRESS.json';start=0;old_seconds=0.
    if resume.exists():
        saved=json.loads(resume.read_text());start=saved['rows_done'];old_seconds=saved['predict_seconds']
    logits=np.lib.format.open_memmap(path,mode='r+' if start else 'w+',dtype='float32',shape=(N,C))
    step=max(C,request['local_checkpoint_rows']);step=max(C,step//C*C)
    elapsed=old_seconds
    for s in range(start,N,step):
        end=min(s+step,N);X=d['X_test'][s:end];dummy=np.arange(len(X))%C
        # The final tiny remainder is padded and discarded after prediction.
        original_rows=len(X)
        if len(X)<C:X=np.concatenate([X,np.repeat(X[-1:],C-len(X),axis=0)]);dummy=np.arange(C)
        loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.tensor(dummy),torch.from_numpy(X)),batch_size=args.batch_size_inf,shuffle=False)
        with cost.phase('predict'):
            _,part=eval_ft(args,model,loader,d)
        elapsed=old_seconds+cost.seconds('predict')
        logits[s:end]=part[:original_rows];logits.flush()
        write(resume,dict(rows_done=end,rows_total=N,predict_seconds=elapsed))
        write(out/'progress.json',dict(state='running',phase='predict',rows_done=end,rows_total=N,predict_seconds=elapsed,
            estimated_predict_remaining_seconds=elapsed*(N-end)/end,updated_epoch=time.time()))
        print(f'LoCalPFN test {end:,}/{N:,}; cumulative prediction {elapsed:.1f}s',flush=True)
    if old_seconds:cost.phases.append(dict(phase='predict',seconds=old_seconds,complete=True,reused_from_prior_attempt=True))
    from scipy.special import softmax
    return {'localpfn':softmax(np.asarray(logits),axis=1)}


def run(request, dataset, method):
    import numpy as np
    import torch
    torch.set_num_threads(request['cpu_threads']);torch.set_num_interop_threads(min(4,request['cpu_threads']))
    torch.manual_seed(43);np.random.seed(43)
    out=Path(request['root'])/f'{dataset}_{method}';out.mkdir(parents=True,exist_ok=True)
    request={**request,'dataset':dataset,'method':method}
    cost=Cost(out);success=False
    try:
        with cost.phase('input_loading'):data=load_data(request,dataset,method,out)
        fn={'xgb':xgboost,'xgb_full':xgboost,'distpfn':distpfn,'boostpfn':boostpfn,'localpfn':localpfn}[method]
        probabilities=fn(data,request,out,cost)
        with cost.phase('scoring'):
            results=[score(tag,p,data,out) for tag,p in probabilities.items()]
        measured=cost.finish();success=True
        for r in results:
            adjustment=cost.seconds('test_prior_adjustment') if r['method']=='distpfn' else 0
            predict=cost.seconds('predict')+adjustment
            r.update(fit_seconds=cost.seconds('fit'),validation_seconds_included_in_fit=cost.seconds('validation'),predict_seconds=predict,
                postprocess_seconds=adjustment,rows_per_second=len(data['yte'])/predict if predict else None,
                seconds_per_1000_rows=1000*predict/len(data['yte']),preprocess_seconds=cost.seconds('preprocess'),input_loading_seconds=cost.seconds('input_loading'),
                peak_gpu_sampled_gib=measured['peak_gpu_sampled_gib'],peak_torch_allocated_gib=measured['peak_torch_allocated_gib'],peak_rss_gib=measured['peak_rss_gib'],
                peak_model_rss_sampled_gib=max((v for k,v in measured['phase_peak_rss_gib'].items() if k in ['preprocess','model_load','fit','validation','predict','test_prior_adjustment']),default=None),
                cost_note='Global backbone fit/predict included; computation shared with global_raw row' if r['method']=='distpfn' else 'Full test inference; no test prediction deduplication')
        write(out/'results.json',results);write(out/'COMPLETE.json',dict(dataset=dataset,method=method,completed_epoch=time.time(),synthetic=data['synthetic']))
        write(out/'progress.json',dict(state='complete',updated_epoch=time.time()))
        print(json.dumps([{k:v for k,v in r.items() if k!='classes'} for r in results]),flush=True)
    except BaseException:
        write(out/'ERROR.json',dict(traceback=traceback.format_exc(),failed_epoch=time.time()))
        raise
    finally:
        if not success:cost.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--dataset',required=True);p.add_argument('--method',required=True)
    a=p.parse_args();run(json.loads(a.request.read_text()),a.dataset,a.method)

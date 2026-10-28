"""Stage-wise adapters using the project's already validated SOTA ports."""
from replay_protocol import ROOT
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from exp61_utils import Cost, write
import exp61_sota_worker as sota


def sync():
    import torch
    if torch.cuda.is_initialized(): torch.cuda.synchronize()


def timed(call):
    sync(); start=time.perf_counter(); value=call(); sync()
    return value,time.perf_counter()-start


def local_request(config, dataset):
    return dict(dataset=dataset, checkpoint=config['checkpoint'], seed=config['seed'],
                cpu_threads=config['threads'], boost_weights_root=str(ROOT/'tabpfn/third_party/BoostPFN'),
                local_checkpoint=str(ROOT/'tabpfn/third_party/LoCalPFN/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'),
                boost_rounds=2 if config.get('smoke') else 50,
                boost_samples=32 if config.get('smoke') else 500,
                boost_batch=config['batch'], local_batch=4 if config.get('smoke') else 256,
                local_checkpoint_rows=config['batch'])


def make_base(method,config,C):
    from exp71_bounded_updates import get_model
    args=argparse.Namespace(threads=config['threads'],seed=config['seed'],checkpoint=Path(config['checkpoint']))
    model=get_model(method,args,C)
    if config.get('smoke'):
        if method=='xgb': model.set_params(n_estimators=3)
        else: model.set_params(n_estimators=1)
    return model


def fit_stage(method,config,dataset,data,out,prior_model=None):
    """Save a version; disk checkpoint writes are experimental overhead, not update."""
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cost=Cost(out);model=None; samples=[]
    C=len(data['names'])
    try:
        if method in ('xgb','tabpfn'):
            reps=1 if config.get('smoke') else config['cost_repeats']
            # PFN checkpoint loading happens on its first fit. Separate cold
            # construction from steady updates; keep the same object thereafter.
            model=prior_model if method=='tabpfn' else None
            for rep in range(reps):
                with cost.phase('construct'):
                    if method=='xgb' or model is None: model=make_base(method,config,C)
                construct=cost.phases[-1]['seconds']
                with cost.phase('fit'):
                    model.fit(data['Xtr'],data['ytr'])
                fit=cost.phases[-1]['seconds']
                p,first=timed(lambda:model.predict_proba(data['Xte']))
                if p.shape!=(len(data['Xte']),C):raise ValueError('Prediction dimension mismatch')
                if not np.array_equal(model.classes_,np.arange(C)):raise ValueError('Class map mismatch')
                samples.append(dict(construct_seconds=construct,fit_seconds=fit,first_batch_seconds=first,
                                    first_batch_rows=len(data['Xte']),update_seconds=construct+fit))
            if method=='xgb': model.save_model(out/'model.ubj')
            else: model.save_fit_state(out/'model.tabpfn_fit')
            update=samples[0]['update_seconds']
        elif method in ('boostpfn','localpfn'):
            request=local_request(config,dataset)
            fn=sota.boostpfn if method=='boostpfn' else sota.localpfn
            fn(data,request,out,cost)
            # Validation is nested within fit, never added twice.
            update=sum(cost.seconds(k) for k in ['preprocess','model_load','fit'])
        else: raise ValueError(method)
    finally:
        resources=cost.finish()
    info=dict(method=method,dataset=dataset,update_seconds=update,cost_repetitions=samples,
              update_median_seconds=float(np.median([s['update_seconds'] for s in samples])) if samples else update,
              train_rows=len(data['ytr']),validation_rows=len(data['yva']),class_names=data['names'],
              train_class_counts=np.bincount(data['ytr'],minlength=C).tolist(),
              phases=resources['phases'],peak_gpu_gib=resources['peak_gpu_sampled_gib'])
    write(out/'fit_metrics.json',info)
    return model,info


class Predictor:
    """Predict on observed inputs only. No query ground truth is accepted."""
    def __init__(self,method,config,dataset,data,out):
        import torch
        self.method=method;self.config=config;self.dataset=dataset;self.data=data;self.out=Path(out)
        self.C=len(data['names']);self.request=local_request(config,dataset)
        if method=='xgb':
            from xgboost import XGBClassifier
            self.model=XGBClassifier();self.model.load_model(self.out/'model.ubj')
            # UBJ stores the trees but not execution-device/thread settings.
            # Restoring defaults silently falls back to CPU inference.
            self.model.set_params(device='cuda:0',n_jobs=config['threads'])
            device=json.loads(self.model.get_booster().save_config())['learner']['generic_param']['device']
            if device!='cuda:0':raise RuntimeError('Restored XGBoost is not configured for CUDA')
        elif method=='tabpfn':
            from tabpfn import TabPFNClassifier
            self.model=TabPFNClassifier.load_from_fit_state(self.out/'model.tabpfn_fit',device='cuda:0')
        elif method=='boostpfn':
            self.base,self.model,self.split=sota.boost_factory(self.request)
            z=np.load(self.out/'boost_state.npz')
            self.model.sampled_idxs=list(z['sampled_idxs']);self.model.alphas=list(z['alphas'])
        elif method=='localpfn':
            import joblib,faiss
            from nfv3_v3_exp36_localpfn import import_localpfn,clone_forward_output
            PFN,utils,_,_,_,self.eval_ft=import_localpfn(str(ROOT/'tabpfn/third_party/LoCalPFN'))
            faiss.omp_set_num_threads(config['threads']);utils.seed_everything(config['seed'])
            self.scaler=joblib.load(self.out/'scaler.joblib')
            self.Xtr=np.clip(self.scaler.transform(data['Xtr']),-10,10).astype('float32')
            self.model,_=PFN.load_old(device='cuda:0',path=self.request['local_checkpoint'])
            clone_forward_output(self.model)
            state=torch.load(self.out/'fit/data'/dataset/'model_best.pth',map_location='cuda:0',weights_only=True)
            self.model.load_state_dict(state);self.model.eval()
            self.args=argparse.Namespace(device='cuda:0',onehot_retrieval=False,
                context_length=32 if config.get('smoke') else 1000)
            self.local_data=dict(X_train=self.Xtr,X_train_one_hot=self.Xtr,y_train=data['ytr'])
            # Identical exact L2 retrieval; avoid rebuilding an unchanged index
            # on every input microbatch. Charge its construction to readiness.
            import methods.ftknn as ft
            original=utils.SingleclassFaiss
            self.index,self.index_seconds=timed(lambda:original(self.Xtr,data['ytr']))
            self.ft=ft;self.original_index=ft.SingleclassFaiss
            ft.SingleclassFaiss=lambda *a,**kw:self.index
        else: raise ValueError(method)

    def predict(self,X):
        if self.method in ('xgb','tabpfn'):return self.model.predict_proba(X)
        if self.method=='boostpfn':
            return sota.replay_boost(self.base,self.model,self.split,
                    self.data['Xtr'],self.data['ytr'],X,self.config['batch'])
        import torch
        from scipy.special import softmax
        from torch.utils.data import TensorDataset,DataLoader
        X=np.clip(self.scaler.transform(X),-10,10).astype('float32')
        rows=len(X)
        if rows<self.C:X=np.concatenate([X,np.repeat(X[-1:],self.C-rows,axis=0)])
        dummy=(np.arange(len(X))%self.C).astype('int64')
        loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(dummy),torch.from_numpy(X)),
                          batch_size=self.request['local_batch'],shuffle=False)
        _,logits=self.eval_ft(self.args,self.model,loader,self.local_data)
        return softmax(logits[:rows],axis=1).astype('float32')

    def close(self):
        import torch
        if self.method=='localpfn': self.ft.SingleclassFaiss=self.original_index
        self.model=None
        if hasattr(self,'base'):self.base=None
        gc.collect();torch.cuda.empty_cache()

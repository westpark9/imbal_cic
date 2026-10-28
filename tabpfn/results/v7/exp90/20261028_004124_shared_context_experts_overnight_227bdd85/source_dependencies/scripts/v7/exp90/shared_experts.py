"""Label-budget-preserving shared-context expert models. No evaluation labels."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)

import gc
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
from replay_models import make_base, timed
from finetune_final import weights_digest

BANK = ['xgb_tuned', 'pfn', 'pfn_log', 'hier_pfn']
COMPONENTS = ['xgb0','xgb1','xgb2','xgb3','pfn','pfn_log','pfn_gate','pfn_attack','xgb_gate','xgb_attack']
METHODS = ['xgb','xgb_tuned','pfn','pfn16','pfn_log','mean_xgb_pfn','mean_views',
           'hier_pfn','hier_pfn_xgb','hier_xgb','mean_hierarchy','mean_bank','convex_oof',
           'gate_oof','anchored_gate','pfn_bank_mean']
NO_LOG = {'PROTOCOL','L7_PROTO','TCP_FLAGS','CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS',
          'MIN_TTL','MAX_TTL','ICMP_TYPE','ICMP_IPV4_TYPE','DNS_QUERY_TYPE','FTP_COMMAND_RET_CODE'}


def transform(X, kind, names):
    X = np.asarray(X, dtype=np.float32)
    if kind != 'pfn_log':
        return X
    result = X.copy()
    cols = np.array([name not in NO_LOG for name in names])
    result[:, cols] = np.sign(result[:, cols])*np.log1p(np.abs(result[:, cols]))
    assert np.isfinite(result).all()
    return result


def folds(X, y, n, seed):
    _, groups = np.unique(X, axis=0, return_inverse=True)
    K = 2 if n <= 3 else 3
    split = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
    fold = np.full(len(y), -1, dtype=np.int16)
    audits = []
    for k, (tr, va) in enumerate(split.split(X,y,groups)):
        assert not np.intersect1d(groups[tr],groups[va]).size
        assert len(tr) and len(va)
        fold[va] = k
        audits.append(dict(fold=k,train_rows=len(tr),validation_rows=len(va),
            train_counts=np.bincount(y[tr],minlength=int(y.max())+1).tolist(),
            validation_counts=np.bincount(y[va],minlength=int(y.max())+1).tolist(),
            absent_train_classes=sorted(set(map(int,y))-set(map(int,y[tr])))))
    assert np.all(fold>=0)
    return fold, groups, audits


class Component:
    def __init__(self, kind, cfg, names, benign, feature_names):
        self.kind,self.cfg,self.names,self.benign,self.feature_names = kind,cfg,names,benign,feature_names
        self.model = None
        self.output_classes = np.arange(len(names))
        if kind.endswith('_gate'):
            self.output_classes = np.arange(2)
        elif kind.endswith('_attack'):
            self.output_classes = np.array([i for i in range(len(names)) if i!=benign])

    def fit(self, X, y):
        if self.kind.endswith('_gate'):
            y = (y!=self.benign).astype(np.int64)
        elif self.kind.endswith('_attack'):
            rows = y!=self.benign
            X,y = X[rows],y[rows]
        classes = np.unique(y)
        self.classes = classes
        self.model = None
        self.before = None
        if len(classes)<2:
            # Entire exact-vector groups may contain the only examples of a class.
            self.constant = int(classes[0]) if len(classes) else int(self.output_classes[0])
            self.fit_seconds = 0.
            return self
        method = 'xgb' if self.kind.startswith('xgb') else 'tabpfn'
        self.model, construct = timed(lambda:make_base(method,self.cfg,len(classes)))
        if self.kind.startswith('xgb') and self.kind[3:].isdigit():
            self.model.set_params(**self.cfg['design']['xgb_grid'][int(self.kind[3:])])
        if self.kind=='pfn16':
            self.model.set_params(n_estimators=16)
        if method=='xgb' and len(classes)==2:
            self.model.set_params(objective='binary:logistic',num_class=None,eval_metric='logloss')
        Xt = transform(X,self.kind,self.feature_names)
        _, elapsed = timed(lambda:self.model.fit(Xt,np.searchsorted(classes,y)))
        self.fit_seconds = construct+elapsed
        if method=='tabpfn':
            self.before = weights_digest(self.model.models_[0].state_dict())
        return self

    def predict(self, X):
        out = np.zeros((len(X),len(self.output_classes)),dtype=np.float32)
        if self.model is None:
            out[:,int(np.flatnonzero(self.output_classes==self.constant)[0])] = 1.
        else:
            raw = np.asarray(self.model.predict_proba(transform(X,self.kind,self.feature_names)),dtype=np.float32)
            out[:,np.searchsorted(self.output_classes,self.classes)] = raw
        assert np.isfinite(out).all() and np.allclose(out.sum(1),1,atol=1e-3)
        return out

    def close(self):
        if self.before:
            assert self.before == weights_digest(self.model.models_[0].state_dict())
        self.model = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def normalize(p):
    p = np.maximum(p,1e-12)
    return (p/p.sum(axis=1,keepdims=True)).astype(np.float32)


def hierarchy(gate, attacks, benign, C):
    result = np.empty((len(gate),C),dtype=np.float32)
    result[:,benign] = gate[:,0]
    result[:,np.arange(C)!=benign] = gate[:,1,None]*attacks
    return normalize(result)


def class_weights(y, C):
    counts = np.bincount(y,minlength=C)
    weights = 1/np.maximum(counts[y],1)
    return weights/weights.sum()


def nll(p,y,w):
    return float(-np.dot(w,np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))))


def bank_arrays(component, chosen_xgb, benign, C):
    return np.stack([component[f'xgb{chosen_xgb}'], component['pfn'],component['pfn_log'],
                     hierarchy(component['pfn_gate'],component['pfn_attack'],benign,C)],axis=1)


def gate_features(bank, benign):
    N,M,C = bank.shape
    blocks = [np.ones((N,1))]
    pred = bank.argmax(2)
    for m in range(M):
        p = bank[:,m]
        top = np.partition(p,-2,axis=1)[:,-2:]
        entropy = -(p*np.log(np.clip(p,1e-12,1))).sum(1)/np.log(C)
        blocks.append(np.column_stack([p.max(1),entropy,top.max(1)-top.min(1),p[:,benign],
                                       pred[:,m]!=pred[:,1]]))
    blocks.append(np.eye(C)[pred[:,1]])
    return np.column_stack(blocks).astype(np.float64)


def fit_combiner(component, y, benign, C):
    w = class_weights(y,C)
    losses = [nll(component[f'xgb{i}'],y,w) for i in range(4)]
    chosen = int(np.argmin(losses))
    bank = bank_arrays(component,chosen,benign,C)
    # Optimize non-negative mixture coefficients, not test-derived thresholds.
    # A fold may have no examples of a class. Equal numerical floors prevent
    # the hierarchy's normalization epsilon from masquerading as evidence.
    ptrue = np.maximum(bank[np.arange(len(y)),:,y].astype(np.float64),1e-9)
    def convex_loss(logits):
        a = softmax(logits)
        q = ptrue@a
        value = -np.dot(w,np.log(q))+.01*np.sum((a-.25)**2)
        da = -(w[:,None]*ptrue/q[:,None]).sum(0)+.02*(a-.25)
        return value,a*(da-a@da)
    started=time.perf_counter()
    result = minimize(convex_loss,np.zeros(4),jac=True,method='L-BFGS-B',
        options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
    if not result.success:
        raise RuntimeError('Convex combiner did not converge: '+result.message)
    convex_seconds=time.perf_counter()-started
    F = gate_features(bank,benign)
    # Features already have bounded natural scales. Intercept is regularized too.
    def loss(theta):
        A = theta.reshape(F.shape[1],4)
        g = softmax(F@A,axis=1)
        q = np.clip((g*ptrue).sum(1),1e-12,1)
        value = -np.dot(w,np.log(q))+.1*np.sum(A*A)
        dl = w[:,None]*g*(1-ptrue/q[:,None])
        grad = F.T@dl+.2*A
        return value,grad.ravel()
    started=time.perf_counter()
    gate = minimize(loss,np.zeros(F.shape[1]*4),jac=True,method='L-BFGS-B',
                    options={'maxiter':300,'ftol':1e-12,'gtol':1e-7})
    if not gate.success:
        raise RuntimeError('Dynamic combiner did not converge: '+gate.message)
    return dict(chosen_xgb=chosen,xgb_oof_nll=losses,convex_weights=softmax(result.x).tolist(),
                gate_coefficients=gate.x.reshape(F.shape[1],4).tolist(),
                gate_objective=float(gate.fun),convex_objective=float(result.fun),
                gate_iterations=int(gate.nit),label_rows=len(y),test_access=False,
                convex_fit_seconds=convex_seconds,gate_fit_seconds=time.perf_counter()-started,
                bank=BANK,gate_l2=.1,convex_l2=.01,anchored_global_fraction=.5)


def combine(component,state,benign,C,include_reference=True):
    bank = bank_arrays(component,state['chosen_xgb'],benign,C)
    p = bank[:,1]
    dynamic = softmax(gate_features(bank,benign)@np.asarray(state['gate_coefficients']),axis=1)
    gated = np.einsum('nm,nmc->nc',dynamic,bank).astype(np.float32)
    result = dict(xgb=component['xgb0'],xgb_tuned=bank[:,0],pfn=p,pfn_log=bank[:,2],
        mean_xgb_pfn=(bank[:,0]+p)/2,mean_views=(p+bank[:,2])/2,hier_pfn=bank[:,3],
        hier_pfn_xgb=hierarchy(component['pfn_gate'],component['xgb_attack'],benign,C),
        hier_xgb=hierarchy(component['xgb_gate'],component['xgb_attack'],benign,C),
        mean_hierarchy=(p+bank[:,3])/2,mean_bank=bank.mean(1),
        convex_oof=np.einsum('m,nmc->nc',state['convex_weights'],bank).astype(np.float32),
        gate_oof=gated,anchored_gate=(p+gated)/2,pfn_bank_mean=bank[:,1:].mean(1))
    if include_reference:
        result['pfn16'] = component['pfn16']
    return result


def dependencies(method,state):
    tuned=f"xgb{state['chosen_xgb']}"
    mapping={'xgb':['xgb0'],'xgb_tuned':[tuned],'pfn':['pfn'],'pfn16':['pfn16'],
        'pfn_log':['pfn_log'],'mean_xgb_pfn':[tuned,'pfn'],'mean_views':['pfn','pfn_log'],
        'hier_pfn':['pfn_gate','pfn_attack'],'hier_pfn_xgb':['pfn_gate','xgb_attack'],
        'hier_xgb':['xgb_gate','xgb_attack'],'mean_hierarchy':['pfn','pfn_gate','pfn_attack'],
        'pfn_bank_mean':['pfn','pfn_log','pfn_gate','pfn_attack']}
    return mapping.get(method,[tuned,'pfn','pfn_log','pfn_gate','pfn_attack'])
